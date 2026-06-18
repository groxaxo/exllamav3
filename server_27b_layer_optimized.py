#!/usr/bin/env python3
"""
OpenAI-compatible, strictly GPU-only ExLlamaV3 layer-split server optimized for
Qwen3.5-122B-A10B-abliterated-exl3-4bpw on three Ampere GPUs.

Key properties:
- Explicit contiguous stage placement across visible GPUs.
- No CPU embedding fallback.
- No host recurrent checkpoint cache; recurrent state remains GPU-resident only.
- KV cache is allocated once at startup and reused across requests.
- Single active generation at a time for deterministic VRAM use.
- Startup planning retries safer cache modes / cache sizes instead of allowing CPU spill.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, AsyncIterator, Iterable, Optional, Sequence, Union

# ------------------------------------------------------------------------------
# Environment hardening must happen before importing torch / exllamav3.
# ------------------------------------------------------------------------------
TORCH_LIB = "/home/op/miniconda3/envs/exl3-dev/lib/python3.11/site-packages/torch/lib"
if os.path.exists(TORCH_LIB):
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        TORCH_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    )

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.80",
)
os.environ.setdefault("EXLLAMA_EMBED_PREFER_CPU", "0")
os.environ.setdefault("EXLLAMA_GDN_RECURRENT_BACKEND", "ext")
os.environ.setdefault("EXLLAMA_STRICT_GPU_ONLY", "1")
os.environ.setdefault("EXLLAMA_DISABLE_HOST_RECURRENT_CACHE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from exllamav3 import Cache, Config, Model, Tokenizer
from exllamav3.cache import CacheLayer_fp16, CacheLayer_quant
from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.sampler.custom import (
    CustomSampler,
    SS_MinP,
    SS_Sample,
    SS_Temperature,
    SS_TopK,
    SS_TopP,
)
from exllamav3.generator.sampler.presets import ArgmaxSampler
from exllamav3.util.memory import free_mem, set_memory_fraction_use, unset_memory_fraction
from exllamav3.util.tensor import g_tensor_cache

GiB = 1024 ** 3
MiB = 1024 ** 2

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ------------------------------------------------------------------------------
# Defaults tuned for the local 122B deployment.
# ------------------------------------------------------------------------------
DEFAULT_MODEL_DIR = os.getenv(
    "MODEL_DIR",
    "/home/op/exllamav3_ampere/models/Qwen3.5-122B-A10B-abliterated-exl3-4bpw",
)
DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("PORT", "1234"))
DEFAULT_GPU_SPLIT = os.getenv("GPU_SPLIT", "auto")
DEFAULT_CACHE_TOKENS = int(os.getenv("CACHE_TOKENS", "150016"))
DEFAULT_CACHE_MODE = os.getenv("CACHE_QUANT", "auto")
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "4096"))
MIN_MAX_TOKENS = int(os.getenv("MIN_MAX_TOKENS", "12"))
MAX_MAX_TOKENS = int(os.getenv("MAX_MAX_TOKENS", "16000"))
DEFAULT_RESERVE_PER_GPU_GB = float(os.getenv("RESERVE_PER_GPU_GB", "1.00"))
DEFAULT_MAX_VRAM_UTILIZATION = float(os.getenv("MAX_VRAM_UTILIZATION", "0.965"))
DEFAULT_RUNTIME_HEADROOM_GB = float(os.getenv("RUNTIME_HEADROOM_GB", "1.20"))
DEFAULT_PREFILL_CHUNK_TOKENS = int(os.getenv("PREFILL_CHUNK_TOKENS", "1536"))
DEFAULT_WARMUP_PROMPT_TOKENS = int(os.getenv("WARMUP_PROMPT_TOKENS", "160"))
DEFAULT_WARMUP_NEW_TOKENS = int(os.getenv("WARMUP_NEW_TOKENS", "24"))
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "false").lower() in ("1", "true", "yes")
PRESERVE_THINK_OUTPUT = os.getenv("PRESERVE_THINK_OUTPUT", "false").lower() in (
    "1",
    "true",
    "yes",
)
ENABLE_STARTUP_WARMUP = os.getenv("EXLLAMA_STARTUP_WARMUP", "1").lower() in (
    "1",
    "true",
    "yes",
)
MAX_THINKING_TOKENS = int(os.getenv("MAX_THINKING_TOKENS", "1024"))
DEFAULT_DISABLE_HOST_RECURRENT_CACHE = os.getenv(
    "EXLLAMA_DISABLE_HOST_RECURRENT_CACHE",
    "1",
).lower() in ("1", "true", "yes")

# ------------------------------------------------------------------------------
# Global runtime state.
# ------------------------------------------------------------------------------
_model: Optional[Model] = None
_tokenizer: Optional[Tokenizer] = None
_cache: Optional[Cache] = None
_cache_tokens: int = 0
_cache_quant: Optional[tuple[int, int]] = None
_model_name: str = os.path.basename(DEFAULT_MODEL_DIR.rstrip("/"))
_model_max_context: Optional[int] = None
_gen_lock: asyncio.Lock
_runtime_state: "RuntimeState | None" = None
_startup_args: "StartupArgs | None" = None
_startup_metrics: dict[str, Any] = {}

app = FastAPI(title="Qwen3.5-122B ExLlamaV3 strictly GPU-only server")


# ------------------------------------------------------------------------------
# Pydantic models.
# ------------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: Union[str, list[dict]]


class ChatCompletionRequest(BaseModel):
    model: str = "Qwen3.5-122B-A10B-abliterated-exl3-4bpw"
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    min_p: Optional[float] = 0.08
    top_k: Optional[int] = 0
    stream: Optional[bool] = False
    enable_thinking: Optional[bool] = None
    thinking_budget: Optional[int] = None


class InternalBenchmarkRequest(BaseModel):
    prompt_tokens: int = Field(default=512, ge=32, le=8192)
    max_new_tokens: int = Field(default=128, ge=8, le=2048)
    num_runs: int = Field(default=2, ge=1, le=10)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    enable_thinking: bool = False


# ------------------------------------------------------------------------------
# Data classes.
# ------------------------------------------------------------------------------
@dataclass(slots=True)
class StartupArgs:
    model: str
    host: str
    port: int
    gpu_split: str
    cache_tokens: int
    cache_quant: str
    reserve_per_gpu_gb: float
    max_vram_utilization: float
    runtime_headroom_gb: float
    prefill_chunk_tokens: int
    plan_only: bool
    disable_host_recurrent_cache: bool


@dataclass(slots=True)
class GPUInfo:
    logical_index: int
    physical_id: str
    name: str
    total_bytes: int
    free_bytes: int


@dataclass(slots=True)
class ModuleEstimate:
    index: int
    key: str
    class_name: str
    weight_bytes: int
    cache_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.cache_bytes


@dataclass(slots=True)
class CachePlan:
    cache_tokens: int
    cache_quant: Optional[tuple[int, int]]
    cache_mode_name: str
    per_token_bytes: int
    total_cache_bytes: int


@dataclass(slots=True)
class StagePlan:
    stage_index: int
    device_index: int
    device_name: str
    physical_id: str
    module_start: int
    module_end: int
    weight_bytes: int
    cache_bytes: int
    total_bytes: int
    budget_bytes: int


@dataclass(slots=True)
class LoadPlan:
    cache_plan: CachePlan
    stages: list[StagePlan]
    module_estimates: list[ModuleEstimate]
    requested_gpu_split_gb: list[float]
    recommended_gpu_split_gb: list[float]
    planning_strategy: str


@dataclass(slots=True)
class DecodeMetrics:
    prompt_tokens: int
    prefill_tokens: int
    completion_tokens: int
    total_time_s: float
    prefill_time_s: float
    decode_time_s: float

    @property
    def prefill_tps(self) -> float:
        if self.prefill_tokens <= 0 or self.prefill_time_s <= 0:
            return 0.0
        return self.prefill_tokens / self.prefill_time_s

    @property
    def decode_tps(self) -> float:
        if self.completion_tokens <= 0 or self.decode_time_s <= 0:
            return 0.0
        return self.completion_tokens / self.decode_time_s


@dataclass(slots=True)
class RuntimeState:
    args: StartupArgs
    load_plan: LoadPlan
    block_table_cpu: torch.Tensor
    prefill_chunk_tokens: int
    cpu_modules: list[str]
    module_device_summary: dict[str, int]
    warmup: Optional[dict[str, Any]] = None


@dataclass(slots=True)
class DirectGenerationResult:
    token_ids: list[int]
    text: str
    metrics: DecodeMetrics
    stop_token_id: Optional[int] = None


@dataclass(slots=True)
class DirectGenerationChunk:
    text: str
    eos: bool
    completion_tokens: int


# ------------------------------------------------------------------------------
# Helpers.
# ------------------------------------------------------------------------------
def align_to_page(n: int) -> int:
    return max(PAGE_SIZE, (n // PAGE_SIZE) * PAGE_SIZE)


def pages_for_tokens(n: int) -> int:
    return max(1, (max(1, n) + PAGE_SIZE - 1) // PAGE_SIZE)


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc)
    return exc.__class__.__name__ == "OutOfMemoryError" or "CUDA out of memory" in msg


def parse_cache_quant(spec: Optional[str]) -> Optional[tuple[int, int]]:
    if spec is None:
        return None
    normalized = spec.strip().lower()
    if normalized in ("", "none", "false", "0", "fp16", "auto"):
        return None
    if normalized == "q8":
        return (8, 8)
    if normalized == "q4":
        return (4, 4)
    parts = [int(part.strip()) for part in normalized.split(",") if part.strip()]
    if len(parts) == 1:
        return (parts[0], parts[0])
    if len(parts) == 2:
        return (parts[0], parts[1])
    raise ValueError("Specify cache quantization as 'fp16', 'q8', 'q4', 'bits' or 'k_bits,v_bits'.")


def format_cache_quant(cache_quant: Optional[tuple[int, int]]) -> str:
    if cache_quant is None:
        return "fp16"
    k_bits, v_bits = cache_quant
    if k_bits == v_bits:
        return f"q{k_bits}"
    return f"q{k_bits},v{v_bits}"


def cache_mode_candidates(spec: str) -> list[Optional[tuple[int, int]]]:
    normalized = spec.strip().lower()
    if normalized == "auto":
        return [None, (8, 8), (4, 4)]
    return [parse_cache_quant(normalized)]


def unique_descending(values: Iterable[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value > 0 and value not in seen:
            seen.add(value)
            out.append(value)
    out.sort(reverse=True)
    return out


def cache_token_candidates(requested_tokens: int, model_max_context: Optional[int]) -> list[int]:
    cap = model_max_context or requested_tokens
    requested = min(cap, requested_tokens)
    base = align_to_page(requested)
    candidates = [base]
    for target in (150016, 131072, 98304, 65536, 49152, 32768):
        if target < base:
            candidates.append(align_to_page(min(cap, target)))
    return unique_descending(candidates)


def get_process_rss_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except FileNotFoundError:
        pass
    return 0


def extract_content(content: Union[str, list[dict]]) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return " ".join(parts)
    return str(content)


def format_chatml(messages: list[ChatMessage], add_assistant: bool = True) -> str:
    prompt = ""
    for msg in messages:
        role = msg.role.strip().lower()
        content = extract_content(msg.content)
        if role == "system":
            prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    if add_assistant:
        prompt += "<|im_start|>assistant\n"
    return prompt


def get_input_ids(encoded: Any) -> torch.Tensor:
    if isinstance(encoded, tuple):
        return encoded[0]
    return encoded


def should_add_bos() -> bool:
    assert _tokenizer is not None
    bos_token_id = getattr(_tokenizer, "bos_token_id", None)
    return bos_token_id is not None and bos_token_id >= 0


def encode_messages(messages: list[ChatMessage], enable_thinking: Optional[bool] = None) -> torch.Tensor:
    assert _tokenizer is not None
    effective_thinking = ENABLE_THINKING if enable_thinking is None else enable_thinking
    hf_messages = [{"role": m.role, "content": extract_content(m.content)} for m in messages]
    try:
        if hasattr(_tokenizer, "hf_render_chat_template"):
            rendered = _tokenizer.hf_render_chat_template(
                hf_messages,
                add_generation_prompt=True,
                enable_thinking=effective_thinking,
            )
            return get_input_ids(
                _tokenizer.encode(
                    rendered,
                    add_bos=should_add_bos(),
                    encode_special_tokens=True,
                )
            )
        if hasattr(_tokenizer, "hf_chat_template"):
            return get_input_ids(
                _tokenizer.hf_chat_template(
                    hf_messages,
                    add_generation_prompt=True,
                    enable_thinking=effective_thinking,
                )
            )
    except Exception as exc:
        print(f"HF chat template failed: {exc}, falling back to ChatML")
    prompt = format_chatml(messages)
    return get_input_ids(
        _tokenizer.encode(
            prompt,
            add_bos=should_add_bos(),
            encode_special_tokens=True,
        )
    )


def strip_thinking(text: str) -> str:
    while True:
        start = text.find("<think>")
        if start == -1:
            break
        end = text.find("</think>", start)
        if end == -1:
            text = text[:start].strip()
            break
        text = (text[:start] + text[end + len("</think>") :]).strip()
    return text


def normalize_max_tokens(requested: Optional[int]) -> int:
    if requested is None or requested <= 0:
        return DEFAULT_MAX_TOKENS
    return max(MIN_MAX_TOKENS, min(requested, MAX_MAX_TOKENS))


def read_model_max_context(model_dir: str) -> Optional[int]:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as handle:
        config_data = json.load(handle)
    text_config = config_data.get("text_config")
    if isinstance(text_config, dict):
        return text_config.get("max_position_embeddings")
    return config_data.get("max_position_embeddings")


def create_cache(model: Model, cache_tokens: int, cache_quant: Optional[tuple[int, int]]) -> Cache:
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


def cache_bytes_per_token_for_attention(attn: Any, cache_quant: Optional[tuple[int, int]]) -> int:
    token_dim = int(attn.num_kv_heads) * int(attn.head_dim)
    if cache_quant is None:
        return 4 * token_dim
    k_bits, v_bits = cache_quant
    blocks = token_dim // 32
    return blocks * (4 * k_bits + 4 * v_bits + 4)


def cache_bytes_for_module(module: Any, cache_tokens: int, cache_quant: Optional[tuple[int, int]]) -> int:
    total = 0
    for submodule in module:
        if getattr(submodule, "caps", {}).get("kv_cache"):
            total += cache_bytes_per_token_for_attention(submodule, cache_quant) * cache_tokens
    return total


def stage_partition_dp(weights: list[int], num_parts: int) -> list[tuple[int, int]]:
    n = len(weights)
    prefix = [0]
    for weight in weights:
        prefix.append(prefix[-1] + weight)

    dp = [[0] * (n + 1) for _ in range(num_parts + 1)]
    split = [[0] * (n + 1) for _ in range(num_parts + 1)]

    for i in range(1, n + 1):
        dp[1][i] = prefix[i]
    for p in range(2, num_parts + 1):
        for i in range(1, n + 1):
            best = None
            best_j = 0
            start = p - 1
            for j in range(start, i):
                cost = max(dp[p - 1][j], prefix[i] - prefix[j])
                if best is None or cost < best:
                    best = cost
                    best_j = j
            dp[p][i] = best if best is not None else 0
            split[p][i] = best_j

    parts: list[tuple[int, int]] = []
    p = num_parts
    i = n
    while p > 0:
        j = split[p][i] if p > 1 else 0
        parts.append((j, i))
        i = j
        p -= 1
    parts.reverse()
    return parts


def visible_physical_ids() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not raw:
        return [str(i) for i in range(torch.cuda.device_count())]
    return [part.strip() for part in raw.split(",") if part.strip()]


def collect_gpu_info() -> list[GPUInfo]:
    physical = visible_physical_ids()
    infos: list[GPUInfo] = []
    for logical_idx in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(logical_idx)
        props = torch.cuda.get_device_properties(logical_idx)
        physical_id = physical[logical_idx] if logical_idx < len(physical) else str(logical_idx)
        infos.append(
            GPUInfo(
                logical_index=logical_idx,
                physical_id=physical_id,
                name=props.name,
                total_bytes=int(total_bytes),
                free_bytes=int(free_bytes),
            )
        )
    return infos


def compute_device_budgets(gpus: list[GPUInfo], reserve_per_gpu_gb: float, max_vram_utilization: float) -> list[int]:
    reserve_bytes = int(reserve_per_gpu_gb * GiB)
    budgets: list[int] = []
    for gpu in gpus:
        cap_from_fraction = int(gpu.total_bytes * max_vram_utilization)
        budget = min(gpu.free_bytes, cap_from_fraction) - reserve_bytes
        budgets.append(max(budget, int(2 * GiB)))
    return budgets


def parse_requested_gpu_split(spec: str, num_devices: int, budgets: list[int]) -> list[float]:
    normalized = spec.strip().lower()
    if normalized == "auto":
        return [round(b / GiB, 3) for b in budgets]
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if len(parts) != num_devices:
        raise ValueError("--gpu-split must match the number of visible GPUs.")
    values = [float(part) for part in parts]
    if any(value <= 0 for value in values):
        raise ValueError("--gpu-split values must be positive.")
    return values


def estimate_modules(model: Model, cache_tokens: int, cache_quant: Optional[tuple[int, int]]) -> list[ModuleEstimate]:
    estimates: list[ModuleEstimate] = []
    for index, module in enumerate(model.modules):
        weight_bytes = sum(model.config.stc.get_tensor_sizes(module.key)) if module.key else 0
        cache_bytes = cache_bytes_for_module(module, cache_tokens, cache_quant)
        estimates.append(
            ModuleEstimate(
                index=index,
                key=module.key or f"module_{index}",
                class_name=type(module).__name__,
                weight_bytes=int(weight_bytes),
                cache_bytes=int(cache_bytes),
            )
        )
    return estimates


def build_load_plan(
    model: Model,
    gpus: list[GPUInfo],
    requested_gpu_split_gb: list[float],
    cache_tokens: int,
    cache_quant: Optional[tuple[int, int]],
    runtime_headroom_gb: float,
) -> LoadPlan:
    budgets = [int(gb * GiB) for gb in requested_gpu_split_gb]
    estimates = estimate_modules(model, cache_tokens, cache_quant)
    weights = [estimate.total_bytes for estimate in estimates]
    partitions = stage_partition_dp(weights, len(gpus))
    runtime_headroom_bytes = int(runtime_headroom_gb * GiB)
    stages: list[StagePlan] = []
    for stage_index, ((start, end), gpu, budget_bytes) in enumerate(zip(partitions, gpus, budgets)):
        stage_modules = estimates[start:end]
        weight_bytes = sum(module.weight_bytes for module in stage_modules)
        cache_bytes = sum(module.cache_bytes for module in stage_modules)
        total_bytes = weight_bytes + cache_bytes
        if total_bytes + runtime_headroom_bytes > budget_bytes:
            raise RuntimeError(
                "Planned stage exceeds requested GPU budget: "
                f"stage={stage_index} total={total_bytes / GiB:.2f}GiB "
                f"headroom={runtime_headroom_bytes / GiB:.2f}GiB budget={budget_bytes / GiB:.2f}GiB"
            )
        stages.append(
            StagePlan(
                stage_index=stage_index,
                device_index=gpu.logical_index,
                device_name=gpu.name,
                physical_id=gpu.physical_id,
                module_start=start,
                module_end=end,
                weight_bytes=weight_bytes,
                cache_bytes=cache_bytes,
                total_bytes=total_bytes,
                budget_bytes=budget_bytes,
            )
        )

    recommended_gpu_split_gb = [
        round(min(stage.budget_bytes, stage.total_bytes + runtime_headroom_bytes) / GiB, 3)
        for stage in stages
    ]
    cache_plan = CachePlan(
        cache_tokens=cache_tokens,
        cache_quant=cache_quant,
        cache_mode_name=format_cache_quant(cache_quant),
        per_token_bytes=sum(
            cache_bytes_per_token_for_attention(submodule, cache_quant)
            for submodule in model.get_cache_layers()
        ),
        total_cache_bytes=sum(estimate.cache_bytes for estimate in estimates),
    )
    return LoadPlan(
        cache_plan=cache_plan,
        stages=stages,
        module_estimates=estimates,
        requested_gpu_split_gb=requested_gpu_split_gb,
        recommended_gpu_split_gb=recommended_gpu_split_gb,
        planning_strategy="explicit_contiguous_minimax",
    )


def iter_plan_candidates(args: StartupArgs):
    global _model_max_context, _model_name

    gpus = collect_gpu_info()
    if not gpus:
        raise RuntimeError("No CUDA devices are visible. Refusing to start a CPU fallback server.")

    _model_max_context = read_model_max_context(args.model)
    _model_name = os.getenv("MODEL_ID", os.path.basename(args.model.rstrip("/")))

    budgets = compute_device_budgets(gpus, args.reserve_per_gpu_gb, args.max_vram_utilization)
    requested_split_gb = parse_requested_gpu_split(args.gpu_split, len(gpus), budgets)
    cache_modes = cache_mode_candidates(args.cache_quant)
    cache_tokens_candidates = cache_token_candidates(args.cache_tokens, _model_max_context)

    last_error: Optional[BaseException] = None
    yielded = False
    for cache_tokens in cache_tokens_candidates:
        for cache_quant in cache_modes:
            cfg = Config.from_directory(args.model)
            model = Model.from_config(cfg)
            tokenizer = Tokenizer.from_config(cfg)
            cache = create_cache(model, cache_tokens, cache_quant)
            try:
                plan = build_load_plan(
                    model=model,
                    gpus=gpus,
                    requested_gpu_split_gb=requested_split_gb,
                    cache_tokens=cache_tokens,
                    cache_quant=cache_quant,
                    runtime_headroom_gb=args.runtime_headroom_gb,
                )
                yielded = True
                yield plan, cache, model, tokenizer
            except Exception as exc:
                last_error = exc
                model.unload()
                free_mem()
                del cache, model, tokenizer, cfg
                gc.collect()
                continue
    if not yielded:
        if last_error is not None:
            raise RuntimeError(f"Unable to produce a GPU-only load plan: {last_error}") from last_error
        raise RuntimeError("Unable to produce a GPU-only load plan.")


def summarize_module_devices(model: Model) -> tuple[dict[str, int], list[str]]:
    summary: dict[str, int] = {}
    cpu_modules: list[str] = []
    for module in model:
        device = getattr(module, "device", None)
        if device is None:
            continue
        label = str(device)
        summary[label] = summary.get(label, 0) + 1
        if torch.device(device).type != "cuda":
            cpu_modules.append(module.key)
    return summary, cpu_modules


def validate_no_cpu_modules(model: Model) -> tuple[dict[str, int], list[str]]:
    summary, cpu_modules = summarize_module_devices(model)
    if cpu_modules:
        raise RuntimeError(
            "Strict GPU mode refused to start because modules were loaded on CPU: "
            + ", ".join(cpu_modules[:10])
        )
    return summary, cpu_modules


def explicit_load_model(model: Model, plan: LoadPlan) -> None:
    modules = model.modules
    touched_devices: list[int] = []
    try:
        total_modules = sum(stage.module_end - stage.module_start for stage in plan.stages)
        loaded_modules = 0
        print(f"Loading {total_modules} top-level modules with explicit contiguous stage placement...")
        for stage in plan.stages:
            set_memory_fraction_use(stage.budget_bytes, stage.device_index)
            touched_devices.append(stage.device_index)
            device = torch.device(f"cuda:{stage.device_index}")
            print(
                f"  Stage {stage.stage_index}: cuda:{stage.device_index} physical={stage.physical_id} "
                f"modules={stage.module_start}:{stage.module_end} target={stage.total_bytes / GiB:.2f}GiB"
            )
            for module in modules[stage.module_start : stage.module_end]:
                defer = module.can_defer_load()
                try:
                    if defer:
                        model.config.stc.begin_deferred_load()
                    module.load(device)
                    if defer:
                        model.config.stc.end_deferred_load()
                except Exception:
                    model.config.stc.abort_deferred_load()
                    raise
                loaded_modules += 1
                if loaded_modules % 4 == 0 or loaded_modules == total_modules:
                    print(f"    loaded {loaded_modules}/{total_modules} top-level modules")

        model.active_devices = [stage.device_index for stage in plan.stages]
        model.output_device = torch.device(f"cuda:{plan.stages[-1].device_index}")
        model.config.stc.close()
        free_mem()
        g_tensor_cache.drop_all()
    except Exception:
        model.unload()
        free_mem()
        raise
    finally:
        if touched_devices:
            unset_memory_fraction(touched_devices)


def make_sampler(request: Union[ChatCompletionRequest, InternalBenchmarkRequest]):
    temp = request.temperature if request.temperature is not None else 0.7
    if temp <= 0:
        return ArgmaxSampler()
    steps: list[Any] = []
    min_p = getattr(request, "min_p", 0.0) or 0.0
    if min_p > 0:
        steps.append(SS_MinP(min_p))
    top_k = getattr(request, "top_k", 0) or 0
    if top_k > 0:
        steps.append(SS_TopK(top_k))
    top_p = getattr(request, "top_p", 1.0)
    if top_p is None:
        top_p = 1.0
    if 0 < top_p < 1.0:
        steps.append(SS_TopP(top_p))
    steps.append(SS_Temperature(temp))
    steps.append(SS_Sample())
    return CustomSampler(steps)


def synchronize_active_devices() -> None:
    if _model is not None and _model.active_devices:
        for device_index in _model.active_devices:
            torch.cuda.synchronize(device_index)
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


class PendingTextBuffer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.pending_ids: list[int] = []

    def push(self, token_id: int) -> str:
        self.pending_ids.append(token_id)
        token_tensor = torch.tensor(self.pending_ids, dtype=torch.long)
        text = self.tokenizer.decode(token_tensor, decode_special_tokens=False)
        if isinstance(text, list):
            text = text[0]
        if "�" in text and len(text) < 24:
            return ""
        self.pending_ids.clear()
        return text

    def flush(self) -> str:
        if not self.pending_ids:
            return ""
        token_tensor = torch.tensor(self.pending_ids, dtype=torch.long)
        text = self.tokenizer.decode(token_tensor, decode_special_tokens=False)
        if isinstance(text, list):
            text = text[0]
        self.pending_ids.clear()
        return text


class DirectGenerationRecorder:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.buffer = PendingTextBuffer(tokenizer)
        self.tokens: list[int] = []

    def add_token(self, token_id: int) -> str:
        self.tokens.append(token_id)
        return self.buffer.push(token_id)

    def finalize_text(self) -> str:
        self.buffer.flush()
        if self.tokens:
            token_tensor = torch.tensor(self.tokens, dtype=torch.long)
            final_text = self.tokenizer.decode(token_tensor, decode_special_tokens=False)
            if isinstance(final_text, list):
                final_text = final_text[0]
            return final_text
        return ""


def build_inference_params(
    past_len: int,
    input_len: int,
    recurrent_states: Optional[dict[int, Any]],
) -> dict[str, Any]:
    assert _runtime_state is not None
    assert _cache is not None
    used_pages = pages_for_tokens(past_len + input_len)
    params: dict[str, Any] = {
        "attn_mode": "flash_attn",
        "cache": _cache,
        "block_table": _runtime_state.block_table_cpu[:, :used_pages],
        "cache_seqlens": torch.tensor([past_len], dtype=torch.int32),
    }
    if recurrent_states is not None:
        params["recurrent_states"] = recurrent_states
    return params


@torch.inference_mode()
def direct_generate(
    input_ids: torch.Tensor,
    sampler: Any,
    max_new_tokens: int,
    stop_token_ids: set[int],
    stream_recorder: Optional[DirectGenerationRecorder] = None,
) -> tuple[DirectGenerationResult, list[DirectGenerationChunk]]:
    assert _model is not None and _tokenizer is not None and _cache is not None and _runtime_state is not None
    if input_ids.device.type != "cpu":
        input_ids = input_ids.cpu()
    prompt_len = int(input_ids.shape[-1])
    if prompt_len <= 0:
        raise ValueError("Prompt must not be empty.")
    if prompt_len + max_new_tokens >= _cache_tokens:
        raise ValueError(
            f"Prompt + max_new_tokens ({prompt_len + max_new_tokens}) exceeds cache size {_cache_tokens}."
        )

    recurrent_states = _cache.new_recurrent_state() if _model.caps.get("recurrent_states") else None
    current_input = input_ids[:, -1:]
    prefill_tokens = max(0, prompt_len - 1)
    chunks: list[DirectGenerationChunk] = []
    stop_token_id: Optional[int] = None

    t0 = time.perf_counter()
    prefill_start = t0
    past_len = 0
    if prompt_len > 1:
        prefix_ids = input_ids[:, :-1]
        chunk_size = _runtime_state.prefill_chunk_tokens
        for start in range(0, prefix_ids.shape[-1], chunk_size):
            end = min(prefix_ids.shape[-1], start + chunk_size)
            prefill_chunk = prefix_ids[:, start:end]
            _model.prefill(
                prefill_chunk,
                build_inference_params(
                    past_len=past_len,
                    input_len=int(prefill_chunk.shape[-1]),
                    recurrent_states=recurrent_states,
                ),
            )
            past_len += int(prefill_chunk.shape[-1])
    synchronize_active_devices()
    prefill_end = time.perf_counter()

    recorder = stream_recorder or DirectGenerationRecorder(_tokenizer)
    generated = 0
    decode_start = prefill_end
    for _ in range(max_new_tokens):
        logits = _model.forward(
            current_input,
            build_inference_params(
                past_len=past_len,
                input_len=int(current_input.shape[-1]),
                recurrent_states=recurrent_states,
            ),
        )
        token_tensor = sampler.forward(
            logits[:, -1:, :],
            None,
            random.randint(0, (1 << 32) - 1),
            _tokenizer,
        )
        token_id = int(token_tensor.view(-1)[0].item())
        past_len += int(current_input.shape[-1])

        if token_id in stop_token_ids:
            stop_token_id = token_id
            break

        generated += 1
        emitted_text = recorder.add_token(token_id)
        if stream_recorder is not None and emitted_text:
            chunks.append(
                DirectGenerationChunk(
                    text=emitted_text,
                    eos=False,
                    completion_tokens=generated,
                )
            )

        current_input = torch.tensor([[token_id]], dtype=torch.long)

    synchronize_active_devices()
    t1 = time.perf_counter()
    if stream_recorder is not None:
        tail = recorder.buffer.flush()
        if tail:
            chunks.append(
                DirectGenerationChunk(
                    text=tail,
                    eos=False,
                    completion_tokens=generated,
                )
            )
    text = recorder.finalize_text()
    metrics = DecodeMetrics(
        prompt_tokens=prompt_len,
        prefill_tokens=prefill_tokens,
        completion_tokens=generated,
        total_time_s=max(t1 - t0, 0.0),
        prefill_time_s=max(prefill_end - prefill_start, 0.0),
        decode_time_s=max(t1 - decode_start, 0.0),
    )
    return (
        DirectGenerationResult(
            token_ids=recorder.tokens,
            text=text,
            metrics=metrics,
            stop_token_id=stop_token_id,
        ),
        chunks,
    )


@torch.inference_mode()
def generate_with_optional_thinking(
    input_ids: torch.Tensor,
    request: ChatCompletionRequest,
    max_new_tokens: int,
    stop_token_ids: set[int],
) -> tuple[DirectGenerationResult, list[DirectGenerationChunk]]:
    assert _tokenizer is not None
    sampler = make_sampler(request)
    use_thinking = ENABLE_THINKING if request.enable_thinking is None else request.enable_thinking
    if not use_thinking:
        return direct_generate(input_ids, sampler, max_new_tokens, stop_token_ids)

    effective_budget = request.thinking_budget if request.thinking_budget is not None else MAX_THINKING_TOKENS
    if effective_budget <= 0:
        return direct_generate(input_ids, sampler, max_new_tokens, stop_token_ids)

    think_end_id = _tokenizer.single_id("</think>")
    phase1_stop = set(stop_token_ids)
    if think_end_id is not None:
        phase1_stop.add(think_end_id)

    phase1, _ = direct_generate(input_ids, sampler, min(max_new_tokens, effective_budget), phase1_stop)
    phase1_tokens = list(phase1.token_ids)
    phase1_context_tokens = list(phase1_tokens)
    visible_phase1_text = phase1.text
    hit_think_end = think_end_id is not None and phase1.stop_token_id == think_end_id
    extra_context_tokens = 0

    if hit_think_end and think_end_id is not None:
        phase1_context_tokens.append(think_end_id)
        extra_context_tokens += 1
    elif not hit_think_end:
        injected = get_input_ids(
            _tokenizer.encode("</think>\n", add_bos=False, encode_special_tokens=True)
        ).view(-1).tolist()
        phase1_context_tokens.extend(injected)
        extra_context_tokens += len(injected)
        if PRESERVE_THINK_OUTPUT:
            visible_phase1_text += "</think>\n"

    phase2_input = torch.cat(
        [input_ids, torch.tensor(phase1_context_tokens, dtype=torch.long).view(1, -1)],
        dim=-1,
    )
    phase2_max_new = max_new_tokens - len(phase1.token_ids) - extra_context_tokens
    if phase2_max_new <= 0:
        return (
            DirectGenerationResult(
                token_ids=phase1.token_ids,
                text=visible_phase1_text if PRESERVE_THINK_OUTPUT else strip_thinking(visible_phase1_text),
                metrics=phase1.metrics,
                stop_token_id=phase1.stop_token_id,
            ),
            [],
        )
    phase2, chunks = direct_generate(phase2_input, sampler, phase2_max_new, stop_token_ids)

    if PRESERVE_THINK_OUTPUT:
        merged_text = visible_phase1_text + phase2.text
        merged_tokens = list(phase1.token_ids) + list(phase2.token_ids)
        merged_metrics = DecodeMetrics(
            prompt_tokens=input_ids.shape[-1],
            prefill_tokens=phase1.metrics.prefill_tokens + phase2.metrics.prefill_tokens,
            completion_tokens=len(merged_tokens),
            total_time_s=phase1.metrics.total_time_s + phase2.metrics.total_time_s,
            prefill_time_s=phase1.metrics.prefill_time_s + phase2.metrics.prefill_time_s,
            decode_time_s=phase1.metrics.decode_time_s + phase2.metrics.decode_time_s,
        )
        return (
            DirectGenerationResult(
                token_ids=merged_tokens,
                text=merged_text,
                metrics=merged_metrics,
                stop_token_id=phase2.stop_token_id,
            ),
            chunks,
        )

    phase2.text = strip_thinking(phase2.text)
    return phase2, chunks


def chat_stream_payload(request_id: str, created: int, chunk: str) -> str:
    data = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": _model_name,
        "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
    }
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def chat_stream_done_payload(request_id: str, created: int) -> str:
    data = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": _model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def current_cuda_snapshot() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    physical = visible_physical_ids()
    for idx in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        rows.append(
            {
                "logical_index": idx,
                "physical_id": physical[idx] if idx < len(physical) else str(idx),
                "name": torch.cuda.get_device_properties(idx).name,
                "allocated_bytes": int(torch.cuda.memory_allocated(idx)),
                "reserved_bytes": int(torch.cuda.memory_reserved(idx)),
                "free_bytes": int(free_bytes),
                "total_bytes": int(total_bytes),
            }
        )
    return rows


def build_synthetic_chat_input(target_prompt_tokens: int, enable_thinking: bool = False) -> torch.Tensor:
    seed = (
        "Explain, with exact technical detail, how to keep a long-context multi-GPU MoE inference stack "
        "strictly GPU-bound while maximizing decode throughput, cache locality, and memory safety. "
    )
    body = seed
    ids = encode_messages([ChatMessage(role="user", content=body)], enable_thinking=enable_thinking)
    while ids.shape[-1] < target_prompt_tokens:
        body += seed
        ids = encode_messages([ChatMessage(role="user", content=body)], enable_thinking=enable_thinking)
    return ids[:, :target_prompt_tokens].contiguous()


def run_internal_benchmark(request: InternalBenchmarkRequest) -> dict[str, Any]:
    assert _tokenizer is not None
    stop_tokens = {
        token
        for token in [_tokenizer.eos_token_id, _tokenizer.single_id("<|im_end|>")]
        if token is not None
    }
    sampler = make_sampler(request)
    rss_before = get_process_rss_bytes()
    runs: list[dict[str, Any]] = []
    for run_index in range(request.num_runs):
        input_ids = build_synthetic_chat_input(request.prompt_tokens, enable_thinking=request.enable_thinking)
        result, _ = direct_generate(
            input_ids=input_ids,
            sampler=sampler,
            max_new_tokens=request.max_new_tokens,
            stop_token_ids=stop_tokens,
        )
        runs.append(
            {
                "run": run_index + 1,
                "prompt_tokens": result.metrics.prompt_tokens,
                "prefill_tokens": result.metrics.prefill_tokens,
                "completion_tokens": result.metrics.completion_tokens,
                "prefill_time_s": round(result.metrics.prefill_time_s, 6),
                "decode_time_s": round(result.metrics.decode_time_s, 6),
                "total_time_s": round(result.metrics.total_time_s, 6),
                "prefill_tps": round(result.metrics.prefill_tps, 4),
                "decode_tps": round(result.metrics.decode_tps, 4),
            }
        )
    gc.collect()
    rss_after = get_process_rss_bytes()
    avg_prefill_tps = sum(run["prefill_tps"] for run in runs) / len(runs)
    avg_decode_tps = sum(run["decode_tps"] for run in runs) / len(runs)
    return {
        "model": _model_name,
        "cache_tokens": _cache_tokens,
        "cache_quant": list(_cache_quant) if _cache_quant is not None else None,
        "avg_prefill_tps": round(avg_prefill_tps, 4),
        "avg_decode_tps": round(avg_decode_tps, 4),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "rss_delta_bytes": rss_after - rss_before,
        "gpu": current_cuda_snapshot(),
        "runs": runs,
    }


def choose_prefill_chunk_tokens(args: StartupArgs, load_plan: LoadPlan) -> int:
    min_budget_gb = min(stage.budget_bytes / GiB for stage in load_plan.stages)
    max_stage_gb = max(stage.total_bytes / GiB for stage in load_plan.stages)
    free_margin_gb = max(0.25, min_budget_gb - max_stage_gb)
    requested = align_to_page(args.prefill_chunk_tokens)
    if free_margin_gb >= 2.5:
        return max(requested, 2048)
    if free_margin_gb >= 1.75:
        return min(requested, 1536)
    if free_margin_gb >= 1.0:
        return min(requested, 1024)
    return min(requested, 768)


def build_startup_args(argv: Optional[Sequence[str]] = None) -> StartupArgs:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--gpu-split", default=DEFAULT_GPU_SPLIT)
    parser.add_argument("--cache-tokens", type=int, default=DEFAULT_CACHE_TOKENS)
    parser.add_argument("--cache-quant", default=DEFAULT_CACHE_MODE)
    parser.add_argument("--reserve-per-gpu-gb", type=float, default=DEFAULT_RESERVE_PER_GPU_GB)
    parser.add_argument("--max-vram-utilization", type=float, default=DEFAULT_MAX_VRAM_UTILIZATION)
    parser.add_argument("--runtime-headroom-gb", type=float, default=DEFAULT_RUNTIME_HEADROOM_GB)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=DEFAULT_PREFILL_CHUNK_TOKENS)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--disable-host-recurrent-cache",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DISABLE_HOST_RECURRENT_CACHE,
    )
    args, _ = parser.parse_known_args(argv)
    return StartupArgs(
        model=args.model,
        host=args.host,
        port=args.port,
        gpu_split=args.gpu_split,
        cache_tokens=align_to_page(args.cache_tokens),
        cache_quant=args.cache_quant,
        reserve_per_gpu_gb=args.reserve_per_gpu_gb,
        max_vram_utilization=args.max_vram_utilization,
        runtime_headroom_gb=args.runtime_headroom_gb,
        prefill_chunk_tokens=align_to_page(args.prefill_chunk_tokens),
        plan_only=args.plan_only,
        disable_host_recurrent_cache=args.disable_host_recurrent_cache,
    )


def startup_plan_json(args: StartupArgs) -> dict[str, Any]:
    for plan, cache, model, tokenizer in iter_plan_candidates(args):
        try:
            return {
                "model": args.model,
                "visible_devices": visible_physical_ids(),
                "load_plan": {
                    "planning_strategy": plan.planning_strategy,
                    "cache_plan": {
                        "cache_tokens": plan.cache_plan.cache_tokens,
                        "cache_quant": list(plan.cache_plan.cache_quant) if plan.cache_plan.cache_quant is not None else None,
                        "cache_mode_name": plan.cache_plan.cache_mode_name,
                        "per_token_bytes": plan.cache_plan.per_token_bytes,
                        "total_cache_bytes": plan.cache_plan.total_cache_bytes,
                    },
                    "requested_gpu_split_gb": plan.requested_gpu_split_gb,
                    "recommended_gpu_split_gb": plan.recommended_gpu_split_gb,
                    "stages": [asdict(stage) for stage in plan.stages],
                },
            }
        finally:
            model.unload()
            free_mem()
            del cache, model, tokenizer
            gc.collect()
    raise RuntimeError("No viable GPU-only plan found.")


# ------------------------------------------------------------------------------
# Startup.
# ------------------------------------------------------------------------------
@app.on_event("startup")
async def load_model_on_startup() -> None:
    global _model, _tokenizer, _cache, _cache_tokens, _cache_quant, _runtime_state, _gen_lock, _startup_args, _startup_metrics

    _startup_args = build_startup_args()
    print("\n" + "=" * 72)
    print("Strictly GPU-only ExLlamaV3 layer-split server")
    print("=" * 72)
    print(f"Model               : {_startup_args.model}")
    print(f"Visible GPUs        : {visible_physical_ids()}")
    print(f"Requested split     : {_startup_args.gpu_split}")
    print(f"Requested cache     : {_startup_args.cache_tokens} tokens ({_startup_args.cache_quant})")
    print(f"Host recurrent cache: {'disabled' if _startup_args.disable_host_recurrent_cache else 'enabled'}")
    print(f"Endpoint            : http://{_startup_args.host}:{_startup_args.port}/v1")
    print()

    if _startup_args.disable_host_recurrent_cache:
        os.environ["EXLLAMA_DISABLE_HOST_RECURRENT_CACHE"] = "1"
    os.environ["EXLLAMA_EMBED_PREFER_CPU"] = "0"

    last_error: Optional[BaseException] = None
    for plan, cache, model, tokenizer in iter_plan_candidates(_startup_args):
        try:
            explicit_load_model(model, plan)
            module_summary, cpu_modules = validate_no_cpu_modules(model)

            _model = model
            _tokenizer = tokenizer
            _cache = cache
            _cache_tokens = plan.cache_plan.cache_tokens
            _cache_quant = plan.cache_plan.cache_quant
            _gen_lock = asyncio.Lock()
            _runtime_state = RuntimeState(
                args=_startup_args,
                load_plan=plan,
                block_table_cpu=torch.arange(_cache_tokens // PAGE_SIZE, dtype=torch.int32).view(1, -1),
                prefill_chunk_tokens=choose_prefill_chunk_tokens(_startup_args, plan),
                cpu_modules=cpu_modules,
                module_device_summary=module_summary,
            )
            break
        except Exception as exc:
            last_error = exc
            print(f"Load attempt failed: {exc}")
            model.unload()
            free_mem()
            del cache, model, tokenizer
            gc.collect()
    else:
        raise RuntimeError(f"All GPU-only load attempts failed: {last_error}")

    assert _runtime_state is not None
    assert _model is not None

    print("Chosen load plan:")
    print(f"  Cache mode        : {_runtime_state.load_plan.cache_plan.cache_mode_name}")
    print(f"  Cache tokens      : {_runtime_state.load_plan.cache_plan.cache_tokens:,}")
    print(f"  Requested split   : {_runtime_state.load_plan.requested_gpu_split_gb}")
    print(f"  Recommended split : {_runtime_state.load_plan.recommended_gpu_split_gb}")
    print(f"  Prefill chunk     : {_runtime_state.prefill_chunk_tokens}")
    for stage in _runtime_state.load_plan.stages:
        print(
            f"  Stage {stage.stage_index}: cuda:{stage.device_index} physical={stage.physical_id} "
            f"modules={stage.module_start}:{stage.module_end} total={stage.total_bytes / GiB:.2f}GiB"
        )
    print("Module placement:")
    for device_label, count in sorted(_runtime_state.module_device_summary.items()):
        print(f"  {device_label}: {count} modules")
    print("GPU memory after load:")
    for row in current_cuda_snapshot():
        print(
            f"  cuda:{row['logical_index']} physical={row['physical_id']} "
            f"alloc={row['allocated_bytes'] / GiB:.2f}GiB reserved={row['reserved_bytes'] / GiB:.2f}GiB "
            f"free={row['free_bytes'] / GiB:.2f}GiB"
        )

    warmup = None
    if ENABLE_STARTUP_WARMUP:
        bench_request = InternalBenchmarkRequest(
            prompt_tokens=DEFAULT_WARMUP_PROMPT_TOKENS,
            max_new_tokens=DEFAULT_WARMUP_NEW_TOKENS,
            num_runs=1,
            temperature=0.0,
            top_p=1.0,
            min_p=0.0,
            top_k=0,
            enable_thinking=False,
        )
        print("Running startup warmup benchmark...")
        warmup = run_internal_benchmark(bench_request)
        print(
            f"  warmup prefill={warmup['avg_prefill_tps']:.2f} tok/s, "
            f"decode={warmup['avg_decode_tps']:.2f} tok/s, "
            f"rss_delta={warmup['rss_delta_bytes'] / MiB:.2f} MiB"
        )
    _runtime_state.warmup = warmup
    _startup_metrics = {
        "strict_gpu_only": True,
        "disable_host_recurrent_cache": _startup_args.disable_host_recurrent_cache,
        "module_device_summary": _runtime_state.module_device_summary,
        "warmup": warmup,
    }
    print("\nServer ready.\n")


# ------------------------------------------------------------------------------
# Routes.
# ------------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "strict_gpu_only": True,
        "disable_host_recurrent_cache": _startup_args.disable_host_recurrent_cache if _startup_args else True,
        "model": _model_name,
        "cache_tokens": _cache_tokens,
        "cache_quant": list(_cache_quant) if _cache_quant is not None else None,
        "max_position_embeddings": _model_max_context,
        "thinking_enabled": ENABLE_THINKING,
        "preserve_think_output": PRESERVE_THINK_OUTPUT,
        "max_thinking_tokens": MAX_THINKING_TOKENS,
        "visible_devices": visible_physical_ids(),
        "gpu_snapshot": current_cuda_snapshot(),
        "module_device_summary": _runtime_state.module_device_summary if _runtime_state else {},
        "cpu_modules": _runtime_state.cpu_modules if _runtime_state else [],
        "load_plan": {
            "planning_strategy": _runtime_state.load_plan.planning_strategy if _runtime_state else None,
            "requested_gpu_split_gb": _runtime_state.load_plan.requested_gpu_split_gb if _runtime_state else None,
            "recommended_gpu_split_gb": _runtime_state.load_plan.recommended_gpu_split_gb if _runtime_state else None,
            "stages": [asdict(stage) for stage in _runtime_state.load_plan.stages] if _runtime_state else [],
            "cache_mode_name": _runtime_state.load_plan.cache_plan.cache_mode_name if _runtime_state else None,
        },
        "prefill_chunk_tokens": _runtime_state.prefill_chunk_tokens if _runtime_state else None,
        "process_rss_bytes": get_process_rss_bytes(),
        "startup_benchmark": _runtime_state.warmup if _runtime_state else None,
    }


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": _model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "context_length": _cache_tokens,
                "max_position_embeddings": _model_max_context,
            }
        ],
    }


@app.post("/v1/internal/benchmark")
async def internal_benchmark(request: InternalBenchmarkRequest) -> dict[str, Any]:
    if _model is None or _tokenizer is None or _cache is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    async with _gen_lock:
        return run_internal_benchmark(request)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if _model is None or _tokenizer is None or _cache is None or _runtime_state is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    input_ids = encode_messages(request.messages, enable_thinking=request.enable_thinking)
    prompt_len = int(input_ids.shape[-1])
    if prompt_len >= _cache_tokens:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt length {prompt_len} exceeds cache size {_cache_tokens}",
        )

    request.max_tokens = normalize_max_tokens(request.max_tokens)
    max_new = min(request.max_tokens, _cache_tokens - prompt_len - 1)
    eos_token_id = _tokenizer.eos_token_id
    im_end_id = _tokenizer.single_id("<|im_end|>")
    stop_token_ids = {token for token in [eos_token_id, im_end_id] if token is not None}
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    use_thinking = ENABLE_THINKING if request.enable_thinking is None else request.enable_thinking

    if request.stream:
        async def generate_stream() -> AsyncIterator[str]:
            async with _gen_lock:
                if use_thinking:
                    result, _ = generate_with_optional_thinking(input_ids, request, max_new, stop_token_ids)
                    visible_text = result.text if PRESERVE_THINK_OUTPUT else strip_thinking(result.text)
                    if visible_text:
                        yield chat_stream_payload(request_id, created, visible_text)
                        await asyncio.sleep(0)
                else:
                    sampler = make_sampler(request)
                    recorder = DirectGenerationRecorder(_tokenizer)
                    _, chunks = direct_generate(input_ids, sampler, max_new, stop_token_ids, stream_recorder=recorder)
                    for chunk in chunks:
                        if chunk.text:
                            yield chat_stream_payload(request_id, created, chunk.text)
                            await asyncio.sleep(0)
                yield chat_stream_done_payload(request_id, created)
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    async with _gen_lock:
        result, _ = generate_with_optional_thinking(input_ids, request, max_new, stop_token_ids)
        response_text = result.text if PRESERVE_THINK_OUTPUT else strip_thinking(result.text)

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": _model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": result.metrics.completion_tokens,
            "total_tokens": prompt_len + result.metrics.completion_tokens,
        },
        "timings": {
            "prefill_time_s": round(result.metrics.prefill_time_s, 6),
            "decode_time_s": round(result.metrics.decode_time_s, 6),
            "total_time_s": round(result.metrics.total_time_s, 6),
            "prefill_tps": round(result.metrics.prefill_tps, 4),
            "decode_tps": round(result.metrics.decode_tps, 4),
        },
    }


# ------------------------------------------------------------------------------
# Entry point.
# ------------------------------------------------------------------------------
def main() -> None:
    args = build_startup_args(sys.argv[1:])
    if args.plan_only:
        print(json.dumps(startup_plan_json(args), indent=2))
        return
    uvicorn.run(
        "server_27b_layer_optimized:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
 
