#!/usr/bin/env python3
"""
Benchmark script for ExLlamaV3 with FA2 + paged attention on Ampere GPUs.
Supports explicit single- and multi-GPU split configs and can verify that a
requested multi-GPU run actually spans all visible GPUs.
"""

import os
import sys
import time
import torch
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch_lib = "/home/op/miniconda3/envs/exl3-dev/lib/python3.11/site-packages/torch/lib"
if os.path.exists(torch_lib):
    os.environ["LD_LIBRARY_PATH"] = (
        torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.constants import PAGE_SIZE
from exllamav3.modules.attn import _has_flash_attn, _has_flash_attn_with_paged


def check_flash_attn():
    print("\n=== Flash Attention Status ===")
    print(f"Flash Attention available: {_has_flash_attn}")
    print(f"Paged Flash Attention available: {_has_flash_attn_with_paged}")

    if _has_flash_attn:
        import flash_attn

        print(f"flash_attn version: {flash_attn.__version__}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} (CC {props.major}.{props.minor})")
    return _has_flash_attn and _has_flash_attn_with_paged


def validate_cache_tokens(cache_tokens):
    if cache_tokens % PAGE_SIZE == 0:
        return

    lower = cache_tokens - (cache_tokens % PAGE_SIZE)
    upper = lower + PAGE_SIZE
    raise ValueError(
        f"--cache-tokens must be a multiple of {PAGE_SIZE}. "
        f"Nearest valid values are {lower} and {upper}."
    )


def parse_gpu_split(value, expected_len, flag_name):
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != expected_len:
        raise ValueError(
            f"{flag_name} must contain exactly {expected_len} comma-separated values."
        )

    split = [float(part) for part in parts]
    if any(amount <= 0 for amount in split):
        raise ValueError(f"{flag_name} values must be positive.")
    return split


def default_gpu_split(num_gpus):
    if num_gpus == 1:
        return [24.0]
    return [12.0] * num_gpus


def resolve_config_gpu_count(value):
    normalized = value.strip().lower()
    aliases = {
        "single": 1,
        "dual": 2,
        "triple": 3,
        "quad": 4,
        "penta": 5,
        "quint": 5,
    }
    if normalized in aliases:
        return aliases[normalized]

    if normalized.endswith("-gpu"):
        normalized = normalized[:-4]
    elif normalized.endswith("gpu"):
        normalized = normalized[:-3]

    if normalized.isdigit():
        count = int(normalized)
        if count > 0:
            return count

    raise ValueError(
        "Config names must be a positive GPU count like '3' or '5-gpu', "
        "or an alias like 'single', 'dual', 'triple', 'quad', or 'penta'."
    )


def config_label(num_gpus):
    if num_gpus == 1:
        return "single"
    if num_gpus == 2:
        return "dual"
    return f"{num_gpus}-gpu"


def config_title(num_gpus):
    if num_gpus == 1:
        return "TEST: Single GPU"
    return f"TEST: {num_gpus}-GPU"


def parse_named_gpu_splits(values):
    overrides = {}
    for value in values or []:
        config_name, separator, split_values = value.partition("=")
        if not separator:
            raise ValueError(
                "--gpu-split entries must use the form <config>=<gb0,gb1,...>."
            )

        num_gpus = resolve_config_gpu_count(config_name)
        if num_gpus in overrides:
            raise ValueError(
                f"Duplicate --gpu-split override for {config_label(num_gpus)}."
            )

        overrides[num_gpus] = parse_gpu_split(
            split_values,
            num_gpus,
            f"--gpu-split {config_name.strip()}",
        )

    return overrides


def normalize_device_label(device):
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        return f"cuda:{index}"
    return str(device)


def summarize_module_devices(model):
    summary = {}
    for module in model.modules:
        device = getattr(module, "device", None)
        if device is None:
            continue
        label = normalize_device_label(device)
        summary[label] = summary.get(label, 0) + 1
    return summary


def print_gpu_memory(gpu_ids):
    for i, gpu_id in enumerate(gpu_ids):
        mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
        mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
        print(
            f"GPU {gpu_id} (logical {i}): "
            f"Allocated={mem_allocated:.2f}GB, Reserved={mem_reserved:.2f}GB"
        )


def benchmark_model(
    model_dir,
    gpu_ids,
    cache_tokens=32768,
    test_tokens=512,
    warmup=True,
    num_runs=3,
    gpu_split=None,
    tensor_parallel=False,
    require_all_gpus=False,
):
    gpu_split = gpu_split or default_gpu_split(len(gpu_ids))
    if len(gpu_split) != len(gpu_ids):
        raise ValueError("gpu_split length must match gpu_ids length.")

    print(f"\n{'=' * 60}")
    print(f"Testing: {os.path.basename(model_dir)}")
    print(f"Logical GPUs: {gpu_ids}")
    print(f"Load mode: {'tensor-parallel' if tensor_parallel else 'layer-split'}")
    print(f"GPU split (GB): {gpu_split}")
    print(f"Cache tokens: {cache_tokens}")
    print(f"Test output tokens: {test_tokens}")
    print(f"{'=' * 60}")

    torch.cuda.empty_cache()

    config = Config.from_directory(model_dir)
    model = Model.from_config(config)

    cache = Cache(
        model,
        max_num_tokens=cache_tokens,
        layer_type=CacheLayer_quant,
        k_bits=4,
        v_bits=4,
    )

    if tensor_parallel and not model.caps.get("supports_tp"):
        raise NotImplementedError(
            f"Tensor-parallel is not currently implemented for {config.architecture}"
        )

    model.load(use_per_device=gpu_split, progressbar=True, tensor_p=tensor_parallel)

    module_devices = summarize_module_devices(model)
    print("Module placement:")
    for device_label, module_count in sorted(module_devices.items()):
        print(f"  {device_label}: {module_count} modules")

    if require_all_gpus:
        required_devices = {f"cuda:{i}" for i in range(len(gpu_ids))}
        active_devices = {label for label in module_devices if label.startswith("cuda:")}
        missing_devices = sorted(required_devices - active_devices)
        if missing_devices:
            del model, cache
            torch.cuda.empty_cache()
            raise RuntimeError(
                "Model was not distributed across all requested GPUs. "
                f"Missing modules on: {', '.join(missing_devices)}. "
                "Lower the first device split further or increase later-device headroom."
            )

    print("GPU memory after load:")
    print_gpu_memory(gpu_ids)

    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    prompt = "<|im_start|>user\nWrite a detailed explanation of how transformer attention mechanisms work, including the mathematical formulations.<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(prompt, add_bos=True, encode_special_tokens=True)
    if isinstance(input_ids, tuple):
        input_ids = input_ids[0]

    prompt_tokens = input_ids.shape[-1]
    print(f"Prompt tokens: {prompt_tokens}")

    from exllamav3.generator.sampler import ComboSampler

    sampler = ComboSampler(temperature=0.7, top_p=0.9, min_p=0.0, top_k=0)

    if warmup:
        print("Warming up...")
        warmup_job = Job(
            input_ids=input_ids,
            max_new_tokens=16,
            sampler=sampler,
        )
        generator.enqueue(warmup_job)
        while generator.num_remaining_jobs():
            for _ in generator.iterate():
                pass
        torch.cuda.synchronize()
        time.sleep(0.5)

    all_results = []
    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")
        print(f"Generating {test_tokens} tokens...")

        job = Job(
            input_ids=input_ids,
            max_new_tokens=test_tokens,
            sampler=sampler,
        )

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        generator.enqueue(job)
        total_tokens = 0
        while generator.num_remaining_jobs():
            for r in generator.iterate():
                if r.get("new_tokens") is not None:
                    total_tokens = r.get("new_tokens")

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        tps = total_tokens / elapsed
        all_results.append({"tokens": total_tokens, "time": elapsed, "tps": tps})
        print(f"Run {run + 1}: {total_tokens} tokens in {elapsed:.2f}s = {tps:.2f} t/s")

    avg_tps = sum(r["tps"] for r in all_results) / len(all_results)
    avg_time = sum(r["time"] for r in all_results) / len(all_results)
    total_tokens = all_results[-1]["tokens"]

    print(f"\n--- Results (avg of {num_runs} run{'s' if num_runs != 1 else ''}) ---")
    print(f"Generated tokens: {total_tokens}")
    print(f"Avg Time: {avg_time:.2f}s")
    print(f"Avg Tokens/second: {avg_tps:.2f}")

    print("GPU memory after benchmark:")
    print_gpu_memory(gpu_ids)

    del model, cache, generator, tokenizer
    torch.cuda.empty_cache()

    return {
        "tokens": total_tokens,
        "time": avg_time,
        "tps": avg_tps,
        "runs": all_results,
        "gpu_split": gpu_split,
        "module_devices": module_devices,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ExLlamaV3 with FA2 paged attention"
    )
    parser.add_argument(
        "--model", type=str, default="/home/op/exllamav3_ampere/models/Qwen3.5-9B-exl3"
    )
    parser.add_argument("--cache-tokens", type=int, default=32768)
    parser.add_argument("--test-tokens", type=int, default=4000)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["single", "dual"],
        help=(
            "Which benchmark configs to run. Each value may be a positive GPU "
            "count like '3' or '5-gpu', or an alias like 'single', 'dual', "
            "'triple', 'quad', or 'penta'."
        ),
    )
    parser.add_argument(
        "--single-gpu-split",
        type=str,
        default=None,
        help="Override single-GPU use_per_device in GB, e.g. 22.0",
    )
    parser.add_argument(
        "--dual-gpu-split",
        type=str,
        default=None,
        help="Override dual-GPU use_per_device in GB, e.g. 5.0,19.0",
    )
    parser.add_argument(
        "--gpu-split",
        action="append",
        default=[],
        help=(
            "Override use_per_device for a given config. Format: "
            "<config>=<gb0,gb1,...>. Example: --gpu-split 3=1.5,1.5,19.0 "
            "--gpu-split 5-gpu=1.0,1.0,1.0,1.0,18.0"
        ),
    )
    parser.add_argument(
        "--tensor-parallel",
        action="store_true",
        help="Use tensor-parallel loading. Unsupported for Qwen3.5 in this fork.",
    )
    parser.add_argument(
        "--require-all-gpus",
        action="store_true",
        help="Fail if loaded modules do not span every requested logical GPU.",
    )
    args = parser.parse_args()

    validate_cache_tokens(args.cache_tokens)
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count == 0:
        raise RuntimeError("No visible CUDA devices were found.")

    requested_gpu_counts = []
    seen_gpu_counts = set()
    for config_name in args.configs:
        num_gpus = resolve_config_gpu_count(config_name)
        if num_gpus > visible_gpu_count:
            raise ValueError(
                f"Requested config '{config_name}' needs {num_gpus} visible GPUs, "
                f"but only {visible_gpu_count} are available."
            )
        if num_gpus in seen_gpu_counts:
            raise ValueError(
                f"Duplicate config requested for {config_label(num_gpus)}."
            )
        requested_gpu_counts.append(num_gpus)
        seen_gpu_counts.add(num_gpus)

    gpu_split_overrides = parse_named_gpu_splits(args.gpu_split)
    if args.single_gpu_split:
        if 1 in gpu_split_overrides:
            raise ValueError(
                "Do not specify both --single-gpu-split and --gpu-split for 1 GPU."
            )
        gpu_split_overrides[1] = parse_gpu_split(
            args.single_gpu_split,
            1,
            "--single-gpu-split",
        )
    if args.dual_gpu_split:
        if 2 in gpu_split_overrides:
            raise ValueError(
                "Do not specify both --dual-gpu-split and --gpu-split for 2 GPUs."
            )
        gpu_split_overrides[2] = parse_gpu_split(
            args.dual_gpu_split,
            2,
            "--dual-gpu-split",
        )

    print("ExLlamaV3 FA2 + Paged Attention Benchmark")
    print("=" * 60)

    fa_ok = check_flash_attn()
    if not fa_ok:
        print("\nWARNING: Flash Attention 2 with paged attention not fully available!")

    results = {}

    configs = [
        {
            "label": config_label(num_gpus),
            "title": config_title(num_gpus),
            "gpu_ids": list(range(num_gpus)),
            "gpu_split": gpu_split_overrides.get(
                num_gpus,
                default_gpu_split(num_gpus),
            ),
        }
        for num_gpus in requested_gpu_counts
    ]

    for idx, config in enumerate(configs):
        print("\n\n" + "=" * 60)
        print(config["title"])
        print("=" * 60)
        results[config["label"]] = benchmark_model(
            args.model,
            gpu_ids=config["gpu_ids"],
            cache_tokens=args.cache_tokens,
            test_tokens=args.test_tokens,
            num_runs=args.num_runs,
            gpu_split=config["gpu_split"],
            tensor_parallel=args.tensor_parallel,
            require_all_gpus=args.require_all_gpus,
        )
        if idx < len(configs) - 1:
            time.sleep(2)

    print("\n\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"\nModel: {os.path.basename(args.model)}")
    print(f"Cache tokens: {args.cache_tokens}")
    print(f"\n| Config | Tokens | Time (s) | Tokens/sec |")
    print("|--------|--------|----------|------------|")
    for name, r in results.items():
        print(f"| {name:6} | {r['tokens']:6} | {r['time']:8.2f} | {r['tps']:10.2f} |")
        for i, run in enumerate(r.get("runs", [])):
            print(
                f"|   Run {i + 1}  | {run['tokens']:6} | {run['time']:8.2f} | {run['tps']:10.2f} |"
            )

    print(f"\nFlash Attention 2: {'✓' if _has_flash_attn else '✗'}")
    print(f"Paged Attention:   {'✓' if _has_flash_attn_with_paged else '✗'}")


if __name__ == "__main__":
    main()
