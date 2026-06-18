#!/usr/bin/env python3
"""
Benchmark script for Qwen3.5-27B-exl3 decode throughput.
Tests single-GPU and tensor-parallel configurations on RTX 3090 GPUs.

Usage:
    # Run inside the exl3-dev conda env
    cd /home/op/exllamav3_ampere
    python bench_tp.py --tp 1   # 1×RTX 3090
    python bench_tp.py --tp 2   # 2×RTX 3090
    python bench_tp.py --tp 3   # 3×RTX 3090
"""

import sys
import os
import time
import argparse
import json

site_packages = "/home/op/miniconda3/envs/exl3-dev/lib/python3.11/site-packages"
lib_paths = [
    f"{site_packages}/nvidia/cuda_runtime/lib",
    f"{site_packages}/nvidia/nccl/lib",
    f"{site_packages}/torch/lib",
]
lib_paths = [path for path in lib_paths if os.path.isdir(path)]
if lib_paths:
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    prepend = ":".join(lib_paths)
    os.environ["LD_LIBRARY_PATH"] = prepend + (":" + current_ld if current_ld else "")

# Prevent CPU thread over-subscription next to multi-GPU workloads.
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

# Ampere optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from exllamav3 import Config, Model, Tokenizer, Cache, Generator, Job
from exllamav3.cache import CacheLayer_fp16, CacheLayer_quant
from exllamav3.generator.sampler.presets import ArgmaxSampler
from exllamav3.constants import PAGE_SIZE

DEFAULT_MODEL_DIR = "/home/op/exllamav3_ampere/models/Qwen3.5-27B-exl3"

PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "Write a detailed, comprehensive essay about the history of computing, "
    "covering mechanical calculators, vacuum tubes, transistors, integrated circuits, "
    "microprocessors, personal computers, the internet, and modern AI. "
    "Be thorough and detailed, covering at least 10 major milestones.\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def align_to_page(n: int) -> int:
    return (n // PAGE_SIZE) * PAGE_SIZE


def parse_cache_quant(spec: str | None) -> tuple[int, int] | None:
    if spec is None:
        return None
    normalized = spec.strip().lower()
    if normalized in ("", "none", "false", "0", "fp16"):
        return None
    split = [int(bits.strip()) for bits in spec.split(",") if bits.strip()]
    if len(split) == 1:
        return split[0], split[0]
    if len(split) == 2:
        return split[0], split[1]
    raise ValueError("Specify cache quantization as 'bits' or 'k_bits,v_bits'")


def format_cache_quant(cache_quant: tuple[int, int] | None) -> str:
    if cache_quant is None:
        return "fp16"
    k_bits, v_bits = cache_quant
    if k_bits == v_bits:
        return f"quantized {k_bits}-bit"
    return f"quantized k={k_bits}, v={v_bits}"


def create_cache(model: Model, cache_tokens: int, cache_quant: tuple[int, int] | None) -> Cache:
    if cache_quant is None:
        return Cache(model, max_num_tokens=cache_tokens, layer_type=CacheLayer_fp16)
    k_bits, v_bits = cache_quant
    return Cache(
        model,
        max_num_tokens=cache_tokens,
        layer_type=CacheLayer_quant,
        k_bits=k_bits,
        v_bits=v_bits,
    )


def run_benchmark(tp_size: int, gpu_split: list[float], cache_tokens: int,
                  max_new_tokens: int, num_runs: int, tp_backend: str,
                  layer_split: bool = False, model_dir: str = DEFAULT_MODEL_DIR,
                  cache_quant: tuple[int, int] | None = None):
    print(f"\n{'='*70}")
    if layer_split:
        mode_label = f"layer-split-{tp_size}GPU"
    elif tp_size == 1:
        mode_label = "single-GPU"
    else:
        mode_label = f"TP{tp_size}"
    gpu_names = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpu_names.append(p.name)
    print(f"  BENCHMARK: {mode_label}")
    print(f"{'='*70}")
    print(f"  GPU split    : {gpu_split}")
    print(f"  Mode         : {'layer-split (pipeline)' if layer_split else ('tensor-parallel' if tp_size > 1 else 'single-GPU')}")
    print(f"  Cache tokens : {cache_tokens:,}")
    print(f"  KV cache     : {format_cache_quant(cache_quant)}")
    print(f"  Max new tok  : {max_new_tokens}")
    print(f"  TP backend   : {tp_backend if not layer_split else 'n/a (layer-split)'}")
    print(f"  Runs         : {num_runs}")
    print(f"  OMP threads  : {os.environ.get('OMP_NUM_THREADS', 'unset')}")

    num_visible = torch.cuda.device_count()
    print(f"\n  Visible GPUs : {num_visible}")
    for i in range(num_visible):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"    GPU {i}: {p.name}  ({total/1024**3:.1f} GB total, {free/1024**3:.1f} GB free)")

    assert num_visible >= tp_size, (
        f"Need {tp_size} GPUs but only {num_visible} visible. "
        f"Set CUDA_VISIBLE_DEVICES correctly."
    )
    assert len(gpu_split) == tp_size, (
        f"gpu_split has {len(gpu_split)} entries but tp_size is {tp_size}"
    )

    cache_tokens = align_to_page(cache_tokens)

    print("\nLoading model config...")
    cfg = Config.from_directory(model_dir)
    model = Model.from_config(cfg)

    print(f"Creating {format_cache_quant(cache_quant)} KV cache ({cache_tokens:,} tokens)...")
    cache = create_cache(model, cache_tokens, cache_quant)

    use_tensor_parallel = tp_size > 1 and not layer_split
    load_mode_str = "layer-split" if layer_split else ("tensor parallel" if use_tensor_parallel else "single GPU")
    print(f"Loading weights ({load_mode_str})...")
    t0 = time.time()
    if layer_split or not use_tensor_parallel:
        # Layer-split: loads each layer onto the device that has the most free VRAM.
        # EXLLAMA_EMBED_PREFER_CPU=0 keeps embeddings on GPU for fully-GPU inference.
        import os as _os
        _os.environ.setdefault("EXLLAMA_EMBED_PREFER_CPU", "0")
        model.load(
            use_per_device=gpu_split,
            progressbar=True,
        )
    else:
        model.load(
            tensor_p=use_tensor_parallel,
            use_per_device=gpu_split,
            tp_output_device=0,
            tp_backend=tp_backend,
            max_chunk_size=2048,
            max_output_size=1,
            progressbar=True,
        )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    for i in range(num_visible):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        res = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: alloc={alloc:.2f}GB  reserved={res:.2f}GB")

    print("\nLoading tokenizer...")
    tokenizer = Tokenizer.from_config(cfg)

    print("Creating Generator...")
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    input_ids = tokenizer.encode(PROMPT, add_bos=True, encode_special_tokens=True)
    if isinstance(input_ids, tuple):
        input_ids = input_ids[0]
    prompt_len = input_ids.shape[-1]
    print(f"\nPrompt tokens: {prompt_len}")

    eos_token_id = tokenizer.eos_token_id
    im_end_id = tokenizer.single_id("<|im_end|>")
    stop_conditions = {t for t in [eos_token_id, im_end_id] if t is not None}

    results = []
    for run_idx in range(num_runs):
        print(f"\n--- Run {run_idx+1}/{num_runs} ---")
        sampler = ArgmaxSampler()
        job = Job(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_conditions=stop_conditions,
            sampler=sampler,
        )
        generator.enqueue(job)

        gen_tokens = 0
        first_token_time = None
        t_start = time.time()

        while generator.num_remaining_jobs():
            for r in generator.iterate():
                stage = r.get("stage", "")
                if stage == "streaming":
                    if first_token_time is None:
                        first_token_time = time.time()
                    text = r.get("text", "")
                    if r.get("new_tokens") is not None:
                        gen_tokens = r.get("new_tokens")
                    if r.get("eos"):
                        break

        t_end = time.time()
        total_time = t_end - t_start
        decode_time = t_end - (first_token_time or t_start)
        prefill_time = (first_token_time or t_end) - t_start

        tps = gen_tokens / decode_time if decode_time > 0 else 0
        prefill_tps = prompt_len / prefill_time if prefill_time > 0 else 0

        result = {
            "run": run_idx + 1,
            "gen_tokens": gen_tokens,
            "total_time": round(total_time, 3),
            "prefill_time": round(prefill_time, 3),
            "decode_time": round(decode_time, 3),
            "decode_tps": round(tps, 2),
            "prefill_tps": round(prefill_tps, 2),
        }
        results.append(result)
        print(f"  Generated: {gen_tokens} tokens")
        print(f"  Prefill:   {prefill_time:.3f}s ({prefill_tps:.1f} t/s)")
        print(f"  Decode:    {decode_time:.3f}s ({tps:.1f} t/s)")
        print(f"  Total:     {total_time:.3f}s")

    # Summary
    avg_tps = sum(r["decode_tps"] for r in results) / len(results)
    avg_prefill = sum(r["prefill_tps"] for r in results) / len(results)
    print(f"\n{'='*70}")
    print(f"  RESULTS: {mode_label}")
    print(f"{'='*70}")
    print(f"  Avg decode:  {avg_tps:.2f} t/s")
    print(f"  Avg prefill: {avg_prefill:.2f} t/s")
    print(f"  Runs:        {[r['decode_tps'] for r in results]}")
    print(f"{'='*70}\n")

    summary = {
        "tp_size": tp_size,
        "mode": mode_label,
        "gpu_split": gpu_split,
        "gpu_names": gpu_names[:tp_size],
        "layer_split": layer_split,
        "cache_tokens": cache_tokens,
        "cache_quant": list(cache_quant) if cache_quant is not None else None,
        "max_new_tokens": max_new_tokens,
        "tp_backend": tp_backend if not layer_split else "n/a",
        "nccl_algo": os.environ.get("NCCL_ALGO") if use_tensor_parallel and tp_backend == "nccl" else None,
        "prompt_tokens": prompt_len,
        "avg_decode_tps": round(avg_tps, 2),
        "avg_prefill_tps": round(avg_prefill, 2),
        "runs": results,
    }

    # Clean up
    model.unload()
    del generator, cache, model, tokenizer
    torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Decode benchmark for Qwen3.5-27B-exl3")
    parser.add_argument("--tp", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="GPU count: 1-5 visible GPUs for tensor-parallel or layer-split")
    parser.add_argument("--mode", default="tp", choices=["tp", "layer_split"],
                        help="Inference mode: 'tp' for tensor-parallel (default) or 'layer_split' for pipeline")
    parser.add_argument("--cache-tokens", type=int, default=32768)
    parser.add_argument("--cache-quant", default=None,
                        help="KV cache quantization: '4' or 'k_bits,v_bits'. Default: fp16")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--tp-backend", default="nccl")
    parser.add_argument("--gpu-split", default=None,
                        help="Comma-separated GB per GPU (default: 22.0 per GPU)")
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR,
                        help="Path to model directory (default: Qwen3.5-27B-exl3)")
    parser.add_argument("--output", default=None,
                        help="Write JSON results to file (default: auto-named with timestamp)")
    args = parser.parse_args()

    if args.gpu_split:
        gpu_split = [float(x) for x in args.gpu_split.split(",")]
    else:
        gpu_split = [22.0] * args.tp

    layer_split = args.mode == "layer_split"

    summary = run_benchmark(
        tp_size=args.tp,
        gpu_split=gpu_split,
        cache_tokens=args.cache_tokens,
        max_new_tokens=args.max_new_tokens,
        num_runs=args.num_runs,
        tp_backend=args.tp_backend,
        layer_split=layer_split,
        model_dir=args.model,
        cache_quant=parse_cache_quant(args.cache_quant),
    )

    ts = time.strftime("%Y%m%dT%H%M%S")
    mode_tag = "ls" if layer_split else "tp"
    default_out = f"bench_{mode_tag}{args.tp}_{ts}.json"
    out_file = args.output or default_out
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
