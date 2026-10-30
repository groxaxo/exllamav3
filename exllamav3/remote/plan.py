#!/usr/bin/env python3
"""
Plan a remote span plus local layer split for an EXL3 checkpoint, from actual tensor byte counts.
Port of llama.cpp-cluster's tools/cluster/plan_layers.py (-ts / -ot) to EXL3 (remote stage, gpu_split, RAM experts).

Order of preference, like the llama.cpp setup: 3090 VRAM, then the remote 3060, then (with --ram-experts) system RAM.
The remote span is always the first k blocks; the last block stays on the head.

    python -m exllamav3.remote.plan --model /models/MiniMax-M2.5-exl3-2.7bpw --remote-gib 12 --local-gib 24,24,24 \
        --cache-tokens 65536 --cache 6,6
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import struct
import sys

GiB = 1024**3
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")


def read_headers(model_dir: str) -> dict:
    out = {}
    for name in sorted(os.listdir(model_dir)):
        if name.endswith(".safetensors"):
            with open(os.path.join(model_dir, name), "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                hdr = json.loads(f.read(n))
            for k, v in hdr.items():
                if k != "__metadata__":
                    s, e = v["data_offsets"]
                    out[k] = (name, e - s)
    if not out:
        raise SystemExit(f"no .safetensors in {model_dir}")
    return out


def kv_bytes_per_token_layer(cfg: dict, cache: str) -> float:
    if "kv_lora_rank" in cfg:
        td = cfg["kv_lora_rank"] + cfg.get("qk_rope_head_dim", 64)
        if cache == "fp16":
            return td * 2
        k, v = map(int, cache.split(","))
        return td * (max(k, v) / 8 + 2 / 32)
    heads = cfg["num_attention_heads"]
    hd = cfg.get("head_dim") or cfg["hidden_size"] // heads
    td = cfg.get("num_key_value_heads", heads) * hd
    if cache == "fp16":
        return 4 * td
    k, v = map(int, cache.split(","))
    return td * (k + v) / 8 + td / 8


def scan(model_dir: str):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    tcfg = cfg.get("text_config", cfg)
    L = tcfg["num_hidden_layers"]
    headers = read_headers(model_dir)
    dense = [0] * L
    experts = [0] * L
    n_exp = [0] * L
    files = [set() for _ in range(L)]
    embed_gpu = embed_cpu = head = other = 0
    for k, (f, nb) in headers.items():
        m = LAYER_RE.search(k)
        if m and int(m.group(1)) < L:
            i = int(m.group(1))
            files[i].add(f)
            em = EXPERT_RE.search(k)
            if em:
                experts[i] += nb
                n_exp[i] = max(n_exp[i], int(em.group(1)) + 1)
            elif ".experts." in k or k.endswith(".experts"):
                experts[i] += nb
            else:
                dense[i] += nb
        elif m:
            pass  # MTP / nextn layers: not loaded for plain serving
        elif "embed_tokens" in k:
            # exl3 keeps an unquantized embedding table in system RAM (Embedding caps prefer_cpu)
            if k.endswith(".weight"):
                embed_cpu += nb
            else:
                embed_gpu += nb
        elif k.startswith("lm_head"):
            head += nb
        else:
            other += nb
    if not head and tcfg.get("tie_word_embeddings"):
        head = embed_cpu + embed_gpu
    for i in range(L):
        if experts[i] and not n_exp[i]:
            n_exp[i] = tcfg.get("num_local_experts") or tcfg.get("n_routed_experts") or tcfg.get("num_experts") or 1
    return tcfg, L, dense, experts, n_exp, files, embed_gpu, embed_cpu, head + other


def pack_local(lo, L, dense, experts, kv_layer, budgets, head_bytes, embed_gpu, cpu_layers = 0, split = 0, n_exp = None):
    """
    Sequential fill in load order, as exl3 autosplit does: a module goes to the current device or the next one.
    cpu_layers: first N local MoE layers keep their experts in RAM (EXL3_MOE_CPU_OFFLOAD).
    split: tail N experts of every MoE layer in RAM (EXL3_MOE_CPU_SPLIT). Returns (per-device bytes, ram bytes) or None.
    """
    used = [0] * len(budgets)
    dev = 0
    ram = 0
    moe_seen = 0
    items = [embed_gpu] if embed_gpu else []
    for i in range(lo, L):
        e = experts[i]
        if e and cpu_layers and moe_seen < cpu_layers:
            ram += e
            e = 0
        elif e and split:
            part = e * min(split, n_exp[i] - 1) / n_exp[i]
            ram += part
            e -= part
        if experts[i]:
            moe_seen += 1
        items.append(dense[i] + e + kv_layer)
    items.append(head_bytes)
    for size in items:
        while dev < len(budgets) and used[dev] + size > budgets[dev]:
            dev += 1
        if dev == len(budgets):
            return None
        used[dev] += size
    return used, ram


def plan(a):
    tcfg, L, dense, experts, n_exp, files, embed_gpu, embed_cpu, head_bytes = scan(a.model)
    kv = kv_bytes_per_token_layer(tcfg, a.cache) * a.cache_tokens
    local = [x * GiB - a.reserve_gib * GiB for x in a.local_gib]
    remote_budget = a.remote_gib * GiB - a.reserve_gib * GiB if a.remote_gib else 0
    layer = [dense[i] + experts[i] + kv for i in range(L)]

    def remote_fits(k):
        return sum(layer[:k]) <= remote_budget

    k_max = 0
    while k_max < L - 1 and remote_fits(k_max + 1):
        k_max += 1

    choice = None
    candidates = range(0, k_max + 1) if a.policy == "min-remote" else [k_max]
    for k in candidates:
        r = pack_local(k, L, dense, experts, kv, local, head_bytes, embed_gpu)
        if r:
            choice = (k, r, 0, 0)
            break
    if choice is None and not a.ram_experts:
        k = k_max
        short = sum(layer[k:]) + head_bytes + embed_gpu - sum(local)
        raise SystemExit(f"DOES NOT FIT with layers 0-{k - 1} on the remote GPU: short ~{max(short, 0) / GiB:.1f} GiB "
                         f"locally (more if per-device packing strands space). Lower bpw, cache tokens or cache bits. "
                         f"--ram-experts plans system-RAM experts, which needs an exllamav3 with EXL3_MOE_CPU_OFFLOAD "
                         f"(upstream 1.x; this fork's base does not have it)")
    if choice is None:
        k = k_max
        for n in range(1, L - k + 1):
            r = pack_local(k, L, dense, experts, kv, local, head_bytes, embed_gpu, cpu_layers = n)
            if r:
                choice = (k, r, n, 0)
                break
        split_n = 0
        max_e = max(n_exp[k:]) if any(experts[k:]) else 0
        for m in range(1, max_e):
            r2 = pack_local(k, L, dense, experts, kv, local, head_bytes, embed_gpu, split = m, n_exp = n_exp)
            if r2:
                split_n = m
                break
        if choice is None and not split_n:
            short = sum(layer[k:]) + head_bytes + embed_gpu - sum(local)
            raise SystemExit(f"DOES NOT FIT even with all local experts in RAM: short ~{short / GiB:.1f} GiB. "
                             f"Lower bpw, cache tokens or cache bits.")
        if choice is None:
            choice = (k, pack_local(k, L, dense, experts, kv, local, head_bytes, embed_gpu, split = split_n, n_exp = n_exp), 0, split_n)
        else:
            choice = (*choice[:3], split_n)

    k, (used, ram), cpu_layers, split_n = choice
    if a.ram_gib is not None and (ram + embed_cpu) / GiB > a.ram_gib:
        raise SystemExit(f"DOES NOT FIT: needs {(ram + embed_cpu) / GiB:.1f} GiB of system RAM for experts + embedding, "
                         f"--ram-gib allows {a.ram_gib}")
    hidden = tcfg["hidden_size"]
    remote_files = sorted(set().union(*files[:k])) if k else []
    return {
        "layers": L,
        "remote_layers": f"0-{k - 1}" if k else None,
        "remote_gib": round(sum(layer[:k]) / GiB, 2),
        "remote_kv_gib": round(k * kv / GiB, 2),
        "local_layers": f"{k}-{L - 1}",
        "local_gib_per_device": [round(u / GiB, 2) for u in used],
        "gpu_split": [round(u / GiB + a.reserve_gib, 1) for u in used],
        "moe_cpu_offload": cpu_layers,
        "moe_cpu_split": split_n,
        "ram_experts_gib": round(ram / GiB, 2),
        "ram_embedding_gib": round(embed_cpu / GiB, 2),
        "kv_tokens": a.cache_tokens,
        "kv_gib_total": round(L * kv / GiB, 2),
        "wire_bytes_per_decode_token": 2 * hidden * 2,  # fp16 residual, both directions
        "wire_mib_per_prefill_chunk": round(2 * hidden * 2 * a.chunk / 2**20, 1),
        "remote_shards": remote_files,
        "note": "reserve_gib covers CUDA context and the prefill transient; check the worker's load report and the "
                "autosplit log, then rerun with the measured reserve",
    }


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required = True)
    ap.add_argument("--remote-gib", type = float, default = 12.0, help = "VRAM of the remote GPU, 0 for none")
    ap.add_argument("--local-gib", required = True, type = lambda s: [float(x) for x in s.split(",")])
    ap.add_argument("--reserve-gib", type = float, default = 1.25, help = "per device: CUDA context + prefill transient")
    ap.add_argument("--cache-tokens", type = int, default = 32768)
    ap.add_argument("--cache", default = "fp16", help = "fp16 or k_bits,v_bits (e.g. 6,6)")
    ap.add_argument("--chunk", type = int, default = 2048)
    ap.add_argument("--ram-experts", action = "store_true",
                    help = "allow experts in system RAM (EXL3_MOE_CPU_OFFLOAD/SPLIT, upstream exllamav3 1.x only)")
    ap.add_argument("--ram-gib", type = float, default = None, help = "system RAM available for offloaded experts")
    ap.add_argument("--policy", choices = ["min-remote", "fill-remote"], default = "min-remote",
                    help = "min-remote: fewest layers on the slower remote GPU that avoid RAM experts")
    ap.add_argument("--json", action = "store_true")
    a = ap.parse_args()
    if a.cache_tokens % 256:
        sys.exit("--cache-tokens must be a multiple of 256")
    p = plan(a)
    if a.json:
        print(json.dumps(p, indent = 2))
        return
    print(f"blocks: {p['layers']}, cache {a.cache_tokens} tokens ({a.cache}) = {p['kv_gib_total']} GiB total")
    if p["remote_layers"]:
        print(f"remote 3060 : layers {p['remote_layers']}  weights+KV {p['remote_gib']} GiB (KV {p['remote_kv_gib']})")
    else:
        print("remote      : not needed, the model fits locally")
    print(f"local       : layers {p['local_layers']}  per GPU {p['local_gib_per_device']} GiB")
    if p["moe_cpu_offload"]:
        print(f"RAM experts : EXL3_MOE_CPU_OFFLOAD={p['moe_cpu_offload']} (or EXL3_MOE_CPU_SPLIT={p['moe_cpu_split']})"
              f", {p['ram_experts_gib']} GiB")
    elif p["moe_cpu_split"]:
        print(f"RAM experts : EXL3_MOE_CPU_SPLIT={p['moe_cpu_split']}, {p['ram_experts_gib']} GiB")
    print(f"wire        : {p['wire_bytes_per_decode_token']} B per decode token, "
          f"{p['wire_mib_per_prefill_chunk']} MiB per {a.chunk}-token prefill chunk (round trip, fp16 residual)")
    print()
    if p["remote_layers"]:
        print("# on the 3060 box: copy the *.json files + these shards (no other weights are read):")
        print("#   " + " ".join(p["remote_shards"]))
        print(f"EXL3_REMOTE_TOKEN=$TOKEN python -m exllamav3.remote.worker --model {a.model} --layers {p['remote_layers']} "
              f"--bind <3060-box-ip>")
        print("# on the 3090 box (TabbyAPI):")
        env = "EXL3_REMOTE=<3060-box-ip>:50070 EXL3_REMOTE_TOKEN=$TOKEN"
        if p["moe_cpu_offload"]:
            env += f" EXL3_MOE_CPU_OFFLOAD={p['moe_cpu_offload']}"
        elif p["moe_cpu_split"]:
            env += f" EXL3_MOE_CPU_SPLIT={p['moe_cpu_split']}"
        print(f"{env} python main.py   # TabbyAPI, or any script that calls Model.from_config")
    print(f"# TabbyAPI model: gpu_split_auto: false, gpu_split: {p['gpu_split']}, tensor_parallel: false, "
          f"cache_size: {a.cache_tokens}" + (f", cache_mode: \"{a.cache}\"" if a.cache != "fp16" else ""))


if __name__ == "__main__":
    main()
