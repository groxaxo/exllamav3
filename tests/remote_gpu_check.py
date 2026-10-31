#!/usr/bin/env python3
"""
GPU validation for exllamav3.remote, on the 3090 box. Use a model that ALSO fits locally, so the remote run
has a reference:

  1. 3060 box:  EXL3_REMOTE_TOKEN=$T python -m exllamav3.remote.worker --model $M --layers 0-3 --bind <ip>
  2. 3090 box:  EXL3_REMOTE=<ip>:50070 EXL3_REMOTE_TOKEN=$T python tests/remote_gpu_check.py --model $M

Each mode runs in its own process (local only, then with the remote span):
  - raw prefill + greedy decode: logits and tokens must match, and prefill/decode speed is printed
  - Generator workload with shared prompt prefixes and a forced defrag, which exercises the page copy and
    page rotation mirroring: completions must match
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_one(a):
    sys.path.insert(0, ROOT)
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.generator.sampler import ArgmaxSampler
    config = Config.from_directory(a.model)
    model = Model.from_config(config)
    tok = Tokenizer.from_config(config)
    max_len = 8192
    if a.cache == "fp16":
        cache = Cache(model, max_num_tokens = max_len)
    else:
        k, v = map(int, a.cache.split(","))
        cache = Cache(model, max_num_tokens = max_len, layer_type = CacheLayer_quant, k_bits = k, v_bits = v)
    model.load(use_per_device = [float(x) for x in a.gpu_split.split(",")], progressbar = True)
    out = {"mode": a.mode, "remote_stages": [s.key for s in model.remote_stages()]}

    # 1) Raw path: prefill() stops at the last K/V block, then one forward per token
    text = "Explain how pipeline parallelism splits a transformer across machines, and why latency matters. " * 60
    ids = tok.encode(text)[:, : a.prompt]
    n = ids.shape[-1]
    logits_all, toks = [], []
    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.prefill(ids[:, :-1], params = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0,
                                             "batch_shape": (1, max_len)})
        logits = model.forward(ids[:, -1:], params = {"attn_mode": "flash_attn", "cache": cache, "past_len": n - 1,
                                                      "batch_shape": (1, max_len)}).float()
        torch.cuda.synchronize()
        t_pp = time.perf_counter() - t0
        logits_all.append(logits.cpu())
        pos = n
        t0 = time.perf_counter()
        for _ in range(a.gen):
            nxt = torch.argmax(logits[0, -1]).view(1, 1)
            toks.append(int(nxt))
            logits = model.forward(nxt, params = {"attn_mode": "flash_attn", "cache": cache, "past_len": pos,
                                                  "batch_shape": (1, max_len)}).float()
            logits_all.append(logits.cpu())
            pos += 1
        torch.cuda.synchronize()
        t_tg = time.perf_counter() - t0
    out.update({"prompt": n, "pp_tok_s": n / t_pp, "tg_tok_s": a.gen / t_tg, "tokens": toks})

    # 2) Generator: shared prefixes reuse cached pages (copy_page on partial pages), then a forced defrag
    #    rotates pages, then the same prompts are served again from the rotated cache
    gen = Generator(model = model, cache = cache, tokenizer = tok, max_chunk_size = 2048)
    base = "The 3060 holds the first blocks and their cache; the 3090s hold the rest. " * 20
    prompts = [base + f"Question {i}: list {i + 2} reasons this layout is slower than one GPU." for i in range(6)]
    kw = {"max_new_tokens": 48, "sampler": ArgmaxSampler(), "completion_only": True, "add_bos": True}
    first = gen.generate(prompt = prompts[:3], **kw) + [gen.generate(prompt = p, **kw) for p in prompts[3:]]
    gen.pagetable.last_defrag_serial = -(1 << 40)
    gen.generate(prompt = "Short filler prompt to retire the queue.", **kw)
    second = [gen.generate(prompt = p, **kw) for p in prompts]
    out["gen_first"] = first
    out["gen_second"] = second
    out["remote_calls"] = sum(s.stats["calls"] for s in model.remote_stages())
    torch.save({"logits": torch.cat(logits_all, dim = 1)}, a.out)
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--gpu-split", default = "22,22,22")
    ap.add_argument("--cache", default = "fp16")
    ap.add_argument("--prompt", type = int, default = 2048)
    ap.add_argument("--gen", type = int, default = 64)
    ap.add_argument("--mode", choices = ["local", "remote"])
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.mode:
        return run_one(a)
    if not os.environ.get("EXL3_REMOTE"):
        sys.exit("set EXL3_REMOTE / EXL3_REMOTE_TOKEN for the remote run")
    import torch
    res = {}
    for mode in ("local", "remote"):
        path = f"/tmp/exl3r_{mode}.pt"
        env = dict(os.environ)
        if mode == "local":
            env.pop("EXL3_REMOTE", None)
        cmd = [sys.executable, __file__, "--model", a.model, "--gpu-split", a.gpu_split, "--cache", a.cache,
               "--prompt", str(a.prompt), "--gen", str(a.gen), "--mode", mode, "--out", path]
        r = subprocess.run(cmd, env = env, capture_output = True, text = True)
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        if r.returncode or not line:
            sys.exit(f"{mode} run failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
        res[mode] = json.loads(line[-1])
        res[mode]["logits"] = torch.load(path)["logits"]
    if not res["remote"]["remote_stages"]:
        sys.exit("remote run attached no stage: check EXL3_REMOTE and that the worker serves this model")
    la, lb = res["local"]["logits"], res["remote"]["logits"]
    pa, pb = torch.log_softmax(la, -1), torch.log_softmax(lb, -1)
    kl = (pa.exp() * (pa - pb)).sum(-1)
    for m in ("local", "remote"):
        r = res[m]
        print(f"{m:6s}: pp {r['pp_tok_s']:8.1f} tok/s  tg {r['tg_tok_s']:6.2f} tok/s  remote calls {r['remote_calls']}")
    def prefix_ratio(a, b):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n / max(len(a), len(b), 1)

    # The 3060 and the 3090s run the same kernels, but SM count can change reduction order, so allow
    # near-tie greedy flips late in a completion. A mirroring bug shows up as early, wholesale divergence.
    raw = prefix_ratio(res["local"]["tokens"], res["remote"]["tokens"])
    g1 = [prefix_ratio(x, y) for x, y in zip(res["local"]["gen_first"], res["remote"]["gen_first"])]
    g2 = [prefix_ratio(x, y) for x, y in zip(res["local"]["gen_second"], res["remote"]["gen_second"])]
    print(f"raw: max |dlogit| {float((la - lb).abs().max()):.4g}  mean KL {float(kl.mean()):.3g}  "
          f"max KL {float(kl.max()):.3g}  greedy prefix match {raw:.0%}")
    print(f"generator, matching prefix per prompt: before defrag {[f'{r:.0%}' for r in g1]}, "
          f"after page reuse + defrag {[f'{r:.0%}' for r in g2]}")
    bad = float(kl.mean()) > 1e-4 or float(kl.max()) > 1e-2 or raw < 0.5 or \
        sorted(g1)[len(g1) // 2] < 0.5 or sorted(g2)[len(g2) // 2] < 0.5
    if bad:
        sys.exit("MISMATCH: the remote span does not reproduce the local run")
    print("OK")

if __name__ == "__main__":
    main()
