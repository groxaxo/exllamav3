#!/usr/bin/env python3
"""
OpenAI-compatible layer-split server for Qwen3.5-27B-exl3 (6bpw heretic variant).

Uses layer-split (pipeline) mode across multiple GPUs — each GPU holds a subset of
layers and activations pass sequentially. Avoids tensor-parallel cross-GPU comms,
which fixes the GDN recurrent-state corruption seen with native/NCCL TP backends.

Works on RTX 3090 pairs (PCIe) and any multi-GPU setup where NCCL is unavailable.

Usage:
  # 2-GPU layer-split (2×RTX 3090)
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \\
      python server_27b_layer.py --gpu-split 11.0,11.0

  # Single GPU
  CUDA_VISIBLE_DEVICES=0 python server_27b_layer.py --gpu-split 22.0

Environment variables (override CLI defaults):
  MODEL_DIR, GPU_SPLIT, CACHE_TOKENS, HOST, PORT,
  DEFAULT_MAX_TOKENS (16000), MIN_MAX_TOKENS (12), MAX_MAX_TOKENS (16000),
  DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_TOP_K, DEFAULT_MIN_P,
  DEFAULT_REPETITION_PENALTY, DEFAULT_PRESENCE_PENALTY,
  DEFAULT_FREQUENCY_PENALTY, DEFAULT_PENALTY_RANGE
"""

import sys
import os
import argparse
import asyncio
import time
import json
import uuid

# ------------------------------------------------------------------
# Torch library path (required on this machine before importing torch)
# ------------------------------------------------------------------
torch_lib = "/home/op/miniconda3/envs/exl3-dev/lib/python3.11/site-packages/torch/lib"
if os.path.exists(torch_lib):
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    )

# Prevent CPU thread over-subscription next to GPU workloads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Ampere fast-path: use native ext kernel for GDN recurrent state, keep all
# tensors GPU-resident, disable host-side recurrent cache, lazy CUDA module
# loading, and optimised CUDA memory allocator settings.
os.environ.setdefault("EXLLAMA_GDN_RECURRENT_BACKEND", "ext")
os.environ.setdefault("EXLLAMA_STRICT_GPU_ONLY", "1")
os.environ.setdefault("EXLLAMA_DISABLE_HOST_RECURRENT_CACHE", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.80",
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Union
import uvicorn

from exllamav3 import Config, Model, Tokenizer, Cache, Generator, Job
from exllamav3.cache import CacheLayer_fp16, CacheLayer_quant
from exllamav3.generator.sampler.presets import ComboSampler, ArgmaxSampler
from exllamav3.constants import PAGE_SIZE

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
DEFAULT_MODEL_DIR = os.getenv(
    "MODEL_DIR", "/home/op/exllamav3_ampere/models/Qwen3.5-27B-exl3"
)
DEFAULT_GPU_SPLIT = [float(x) for x in os.getenv("GPU_SPLIT", "22.0,22.0").split(",")]
DEFAULT_CACHE_TOKENS = int(os.getenv("CACHE_TOKENS", "32768"))
DEFAULT_CACHE_QUANT = os.getenv("CACHE_QUANT")
DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("PORT", "1234"))
DEFAULT_TP_BACKEND = os.getenv(
    "TP_BACKEND", "nccl"
)  # unused in layer-split, kept for compat
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "16000"))
MIN_MAX_TOKENS = int(os.getenv("MIN_MAX_TOKENS", "12"))
MAX_MAX_TOKENS = int(os.getenv("MAX_MAX_TOKENS", "16000"))
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "true").lower() in ("1", "true", "yes")
PRESERVE_THINK_OUTPUT = os.getenv("PRESERVE_THINK_OUTPUT", "true").lower() in (
    "1",
    "true",
    "yes",
)
ENABLE_STARTUP_WARMUP = os.getenv("EXLLAMA_STARTUP_WARMUP", "1").lower() in (
    "1",
    "true",
    "yes",
)
# Cap on tokens spent inside the <think> block before forcing </think> and moving to the response.
# Prevents the model from "overthinking" and using the entire token budget on reasoning.
MAX_THINKING_TOKENS = int(os.getenv("MAX_THINKING_TOKENS", "1024"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.9"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "0"))
DEFAULT_MIN_P = float(os.getenv("DEFAULT_MIN_P", "0.08"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("DEFAULT_REPETITION_PENALTY", "1.0"))
DEFAULT_PRESENCE_PENALTY = float(os.getenv("DEFAULT_PRESENCE_PENALTY", "0.0"))
DEFAULT_FREQUENCY_PENALTY = float(os.getenv("DEFAULT_FREQUENCY_PENALTY", "0.0"))
DEFAULT_PENALTY_RANGE = int(os.getenv("DEFAULT_PENALTY_RANGE", "1024"))

# ------------------------------------------------------------------
# Globals (populated during startup)
# ------------------------------------------------------------------
_model: Optional[Model] = None
_tokenizer: Optional[Tokenizer] = None
_generator: Optional[Generator] = None
_model_name: str = os.getenv(
    "MODEL_ID", os.path.basename(DEFAULT_MODEL_DIR.rstrip("/"))
)
_cache_tokens: int = DEFAULT_CACHE_TOKENS
_cache_quant: Optional[tuple[int, int]] = None
_model_max_context: Optional[int] = None
_gen_lock: asyncio.Lock

app = FastAPI(title="Qwen3.5-27B heretic exl3 - layer-split multi-GPU server")


# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: Union[
        str, List[dict]
    ]  # Accept both "text" and [{"type": "text", "text": "..."}] formats


class ChatCompletionRequest(BaseModel):
    model: str = "Qwen3.5-27B-exl3-heretic-6bpw-tp2"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    penalty_range: Optional[int] = None
    stream: Optional[bool] = False
    # Per-request thinking controls (None = use server defaults)
    enable_thinking: Optional[bool] = None
    thinking_budget: Optional[int] = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def align_to_page(n: int) -> int:
    return (n // PAGE_SIZE) * PAGE_SIZE


def _parse_cache_quant(spec: Optional[str]) -> Optional[tuple[int, int]]:
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


def _format_cache_quant(cache_quant: Optional[tuple[int, int]]) -> str:
    if cache_quant is None:
        return "fp16"
    k_bits, v_bits = cache_quant
    if k_bits == v_bits:
        return f"quantized {k_bits}-bit"
    return f"quantized k={k_bits}, v={v_bits}"


def _create_cache(
    model: Model, cache_tokens: int, cache_quant: Optional[tuple[int, int]]
) -> Cache:
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


def _normalize_max_tokens(requested: Optional[int]) -> int:
    if requested is None or requested <= 0:
        return DEFAULT_MAX_TOKENS
    return max(MIN_MAX_TOKENS, min(requested, MAX_MAX_TOKENS))


def _get_input_ids(encoded):
    if isinstance(encoded, tuple):
        return encoded[0]
    return encoded


def _should_add_bos() -> bool:
    assert _tokenizer is not None
    bos_token_id = getattr(_tokenizer, "bos_token_id", None)
    return bos_token_id is not None and bos_token_id >= 0


def _read_model_max_context(model_dir: str) -> Optional[int]:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path) as f:
        config_data = json.load(f)
    text_config = config_data.get("text_config")
    if isinstance(text_config, dict):
        return text_config.get("max_position_embeddings")
    return config_data.get("max_position_embeddings")


def format_chatml(messages: List[ChatMessage], add_assistant: bool = True) -> str:
    prompt = ""
    for msg in messages:
        role = msg.role.strip().lower()
        content = _extract_content(msg.content)
        if role == "system":
            prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    if add_assistant:
        prompt += "<|im_start|>assistant\n"
    return prompt


def _extract_content(content: Union[str, List[dict]]) -> str:
    """Extract text from either string or array content format."""
    if isinstance(content, str):
        return content
    # Array format: [{"type": "text", "text": "..."}]
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return " ".join(text_parts)
    return str(content)


def _encode_messages(
    messages: List[ChatMessage], enable_thinking: Optional[bool] = None
):
    assert _tokenizer is not None
    effective_thinking = ENABLE_THINKING if enable_thinking is None else enable_thinking
    hf_messages = [
        {"role": m.role, "content": _extract_content(m.content)} for m in messages
    ]

    # Try HF chat template first, fall back to ChatML if it fails
    try:
        if hasattr(_tokenizer, "hf_render_chat_template"):
            rendered = _tokenizer.hf_render_chat_template(
                hf_messages,
                add_generation_prompt=True,
                enable_thinking=effective_thinking,
            )
            return _get_input_ids(
                _tokenizer.encode(
                    rendered,
                    add_bos=_should_add_bos(),
                    encode_special_tokens=True,
                )
            )
        if hasattr(_tokenizer, "hf_chat_template"):
            return _get_input_ids(
                _tokenizer.hf_chat_template(
                    hf_messages,
                    add_generation_prompt=True,
                    enable_thinking=effective_thinking,
                )
            )
    except Exception as e:
        # Fall back to ChatML format if HF template fails
        print(f"HF chat template failed: {e}, falling back to ChatML")

    prompt = format_chatml(messages)
    return _get_input_ids(
        _tokenizer.encode(
            prompt,
            add_bos=_should_add_bos(),
            encode_special_tokens=True,
        )
    )


def _strip_thinking(text: str) -> str:
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


def make_sampler(request: ChatCompletionRequest):
    temp = DEFAULT_TEMPERATURE if request.temperature is None else request.temperature
    if temp <= 0:
        return ArgmaxSampler()
    return ComboSampler(
        rep_p=(
            DEFAULT_REPETITION_PENALTY
            if request.repetition_penalty is None
            else request.repetition_penalty
        ),
        pres_p=(
            DEFAULT_PRESENCE_PENALTY
            if request.presence_penalty is None
            else request.presence_penalty
        ),
        freq_p=(
            DEFAULT_FREQUENCY_PENALTY
            if request.frequency_penalty is None
            else request.frequency_penalty
        ),
        rep_sustain_range=(
            DEFAULT_PENALTY_RANGE
            if request.penalty_range is None
            else request.penalty_range
        ),
        rep_decay_range=(
            DEFAULT_PENALTY_RANGE
            if request.penalty_range is None
            else request.penalty_range
        ),
        temperature=temp,
        min_p=DEFAULT_MIN_P if request.min_p is None else request.min_p,
        top_k=DEFAULT_TOP_K if request.top_k is None else request.top_k,
        top_p=DEFAULT_TOP_P if request.top_p is None else request.top_p,
    )


def generate_with_generator(
    input_ids: torch.Tensor,
    request: ChatCompletionRequest,
    max_new: int,
    stop_conditions: set,
):
    """Generate tokens using the cached Generator pipeline."""
    sampler = make_sampler(request)
    job = Job(
        input_ids=input_ids,
        max_new_tokens=max_new,
        stop_conditions=stop_conditions,
        sampler=sampler,
    )
    _generator.enqueue(job)

    total_new_tokens = 0
    while _generator.num_remaining_jobs():
        for r in _generator.iterate():
            stage = r.get("stage", "")
            if stage == "streaming":
                text = r.get("text", "")
                eos = r.get("eos", False)
                if r.get("new_tokens") is not None:
                    total_new_tokens = r.get("new_tokens")
                if text or eos:
                    yield text, eos, total_new_tokens


def generate_with_thinking_budget(
    input_ids: torch.Tensor,
    request: ChatCompletionRequest,
    max_new: int,
    stop_conditions: set,
    max_thinking: int,
):
    """Two-phase generation that caps the <think> block to max_thinking tokens.

    Phase 1: generate up to max_thinking tokens, stopping early at </think> if the
             model finishes its reasoning on its own.  If the budget is exhausted
             before </think> appears we forcibly inject </think> so the model is
             compelled to produce an actual response.
    Phase 2: concatenate the original prompt with the (capped) thinking block and
             generate the real answer with the remaining token budget.

    Token IDs from phase 1 are collected directly from the streaming results and
    reused verbatim in phase 2, avoiding a lossy text→token roundtrip that can
    cause token count drift and context mismatches.
    """
    assert _generator is not None and _tokenizer is not None

    think_end_id = _tokenizer.single_id("</think>")
    # Phase-1 stop set: normal EOS tokens PLUS the </think> token so we stop as
    # soon as thinking ends naturally.
    phase1_stop = set(stop_conditions)
    if think_end_id is not None:
        phase1_stop.add(think_end_id)

    # ------------------------------------------------------------------ Phase 1
    sampler = make_sampler(request)
    job = Job(
        input_ids=input_ids,
        max_new_tokens=max_thinking,
        stop_conditions=phase1_stop,
        sampler=sampler,
    )
    _generator.enqueue(job)

    think_pieces: list[str] = []
    think_id_chunks: list[torch.Tensor] = []  # raw token IDs from the generator
    think_tokens_used = 0
    hit_think_end = False

    while _generator.num_remaining_jobs():
        for r in _generator.iterate():
            if r.get("stage") == "streaming":
                text = r.get("text", "")
                tids = r.get("token_ids")
                eos = r.get("eos", False)
                if r.get("new_tokens") is not None:
                    think_tokens_used = r["new_tokens"]
                if text:
                    think_pieces.append(text)
                if tids is not None:
                    think_id_chunks.append(tids.cpu().view(-1))
                if text and PRESERVE_THINK_OUTPUT:
                    yield text, False, think_tokens_used
                if eos:
                    hit_think_end = "".join(think_pieces).rstrip().endswith("</think>")

    # If the budget ran out without a closing tag, inject one and log it.
    injected_close_ids: Optional[torch.Tensor] = None
    if not hit_think_end:
        print(
            f"  [ThinkBudget] budget of {max_thinking} tokens exhausted — "
            f"injecting </think> after {think_tokens_used} thinking tokens"
        )
        close_tag = "</think>\n"
        think_pieces.append(close_tag)
        if PRESERVE_THINK_OUTPUT:
            yield close_tag, False, think_tokens_used
        # Encode only the short injected tag to append to the token stream.
        injected_close_ids = (
            _get_input_ids(
                _tokenizer.encode(close_tag, add_bos=False, encode_special_tokens=True)
            )
            .cpu()
            .view(-1)
        )

    # ------------------------------------------------------------------ Phase 2
    # Build phase-2 input from actual token IDs (no lossy text→token roundtrip).
    # Fall back to re-encoding only if no IDs were collected (shouldn't happen).
    if think_id_chunks or injected_close_ids is not None:
        all_chunks = think_id_chunks
        if injected_close_ids is not None:
            all_chunks = all_chunks + [injected_close_ids]
        if all_chunks:
            think_tensor = torch.cat(all_chunks, dim=0).unsqueeze(0)
        else:
            think_tensor = torch.zeros((1, 0), dtype=torch.long)
        phase2_input = torch.cat([input_ids.cpu(), think_tensor], dim=-1)
    else:
        # Fallback: re-encode the collected text (may diverge from actual tokens).
        full_thinking = "".join(think_pieces)
        think_encoded = _get_input_ids(
            _tokenizer.encode(full_thinking, add_bos=False, encode_special_tokens=True)
        )
        phase2_input = torch.cat([input_ids, think_encoded], dim=-1)

    phase2_max = max(1, max_new - think_tokens_used)
    if phase2_max <= 1:
        yield "", True, think_tokens_used
        return

    sampler2 = make_sampler(request)
    job2 = Job(
        input_ids=phase2_input,
        max_new_tokens=phase2_max,
        stop_conditions=stop_conditions,
        sampler=sampler2,
    )
    _generator.enqueue(job2)

    while _generator.num_remaining_jobs():
        for r in _generator.iterate():
            if r.get("stage") == "streaming":
                text = r.get("text", "")
                eos = r.get("eos", False)
                total_new = think_tokens_used + (r.get("new_tokens") or 0)
                if text or eos:
                    yield text, eos, total_new


# ------------------------------------------------------------------
# Startup
# ------------------------------------------------------------------
@app.on_event("startup")
async def load_model():
    global \
        _model, \
        _tokenizer, \
        _generator, \
        _gen_lock, \
        _cache_tokens, \
        _cache_quant, \
        _model_max_context, \
        _model_name

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--gpu-split", default=",".join(str(x) for x in DEFAULT_GPU_SPLIT)
    )
    parser.add_argument("--cache-tokens", type=int, default=DEFAULT_CACHE_TOKENS)
    parser.add_argument("--cache-quant", default=DEFAULT_CACHE_QUANT)
    parser.add_argument("--tp-backend", default=DEFAULT_TP_BACKEND)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args, _ = parser.parse_known_args()

    gpu_split = [float(x) for x in args.gpu_split.split(",")]
    _cache_tokens = align_to_page(args.cache_tokens)
    _cache_quant = _parse_cache_quant(args.cache_quant)
    _model_max_context = _read_model_max_context(args.model)
    _model_name = os.getenv("MODEL_ID", os.path.basename(args.model.rstrip("/")))

    # On multi-GPU layer-split, keep embeddings on GPU by default to avoid
    # the CPU→GPU transfer on every prefill.  The caller can still override
    # by exporting EXLLAMA_EMBED_PREFER_CPU=1 before launching.
    if len(gpu_split) > 1:
        os.environ.setdefault("EXLLAMA_EMBED_PREFER_CPU", "0")

    embed_on_cpu = os.environ.get("EXLLAMA_EMBED_PREFER_CPU", "1") in (
        "1",
        "true",
        "yes",
    )

    print(f"\n{'=' * 60}")
    num_gpus = len(gpu_split)
    print(f"  Qwen3.5-27B-exl3 heretic - layer-split {num_gpus}-GPU server")
    print(f"{'=' * 60}")
    print(f"  Model dir    : {args.model}")
    print(f"  GPU split    : {gpu_split}")
    print(f"  Mode         : layer-split (pipeline, no TP comms)")
    print(f"  Cache tokens : {_cache_tokens:,}")
    if _model_max_context is not None:
        print(f"  Model limit  : {_model_max_context:,}")
    print(f"  KV cache     : {_format_cache_quant(_cache_quant)}")
    print(f"  Decode path  : Generator + paged flash_attn (cached)")
    print(f"  Max tokens   : {DEFAULT_MAX_TOKENS} (cap {MAX_MAX_TOKENS})")
    print(
        f"  Sampling     : temp={DEFAULT_TEMPERATURE}, top_p={DEFAULT_TOP_P}, top_k={DEFAULT_TOP_K}, min_p={DEFAULT_MIN_P}"
    )
    print(
        f"  Penalties    : presence={DEFAULT_PRESENCE_PENALTY}, repetition={DEFAULT_REPETITION_PENALTY}, frequency={DEFAULT_FREQUENCY_PENALTY}, range={DEFAULT_PENALTY_RANGE}"
    )
    print(
        f"  Thinking     : {ENABLE_THINKING} (preserve={PRESERVE_THINK_OUTPUT}, budget={MAX_THINKING_TOKENS})"
    )
    print(
        f"  Embed device : {'CPU' if embed_on_cpu else 'GPU'} (EXLLAMA_EMBED_PREFER_CPU={os.environ.get('EXLLAMA_EMBED_PREFER_CPU', '1')})"
    )
    print(f"  OMP threads  : {os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print(f"  Endpoint     : http://{args.host}:{args.port}/v1")
    print()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    print(f"  CUDA_VISIBLE_DEVICES : {visible}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  ({p.total_memory / 1024**3:.1f} GB)")
    print()

    # Ampere (SM 8.x) optimizations: TF32 for matmul, cudnn
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Loading model config...")
    cfg = Config.from_directory(args.model)
    _model = Model.from_config(cfg)

    print(
        f"Creating {_format_cache_quant(_cache_quant)} KV cache ({_cache_tokens:,} tokens)..."
    )
    cache = _create_cache(_model, _cache_tokens, _cache_quant)

    print("Loading weights (layer-split across GPUs)...")
    _model.load(
        use_per_device=gpu_split,
        progressbar=True,
    )

    for i in range(torch.cuda.device_count()):
        mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
        mem_res = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: alloc={mem_alloc:.2f}GB  reserved={mem_res:.2f}GB")

    print("\nLoading tokenizer...")
    _tokenizer = Tokenizer.from_config(cfg)

    print("Creating Generator...")
    _generator = Generator(
        model=_model,
        cache=cache,
        tokenizer=_tokenizer,
    )

    if ENABLE_STARTUP_WARMUP:
        print("Running startup warmup...")
        warmup_prompt = (
            "<|im_start|>user\nWhat is 2+2? Answer in one word only.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        warmup_ids = _tokenizer.encode(
            warmup_prompt,
            add_bos=_should_add_bos(),
            encode_special_tokens=True,
        )
        if isinstance(warmup_ids, tuple):
            warmup_ids = warmup_ids[0]
        warmup_job = Job(
            input_ids=warmup_ids,
            max_new_tokens=8,
            stop_conditions={
                t
                for t in [_tokenizer.eos_token_id, _tokenizer.single_id("<|im_end|>")]
                if t is not None
            },
            sampler=ArgmaxSampler(),
        )
        t0 = time.time()
        _generator.enqueue(warmup_job)
        while _generator.num_remaining_jobs():
            _generator.iterate()
        print(f"  Warmup complete in {time.time() - t0:.3f}s")

    _gen_lock = asyncio.Lock()
    print(f"\nServer ready.  Cache = {_cache_tokens:,} tokens.\n")


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": _model_name,
        "cache_tokens": _cache_tokens,
        "cache_quant": list(_cache_quant) if _cache_quant is not None else None,
        "max_position_embeddings": _model_max_context,
        "thinking_enabled": ENABLE_THINKING,
        "preserve_think_output": PRESERVE_THINK_OUTPUT,
        "max_thinking_tokens": MAX_THINKING_TOKENS,
        "default_sampling": {
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "top_k": DEFAULT_TOP_K,
            "min_p": DEFAULT_MIN_P,
            "presence_penalty": DEFAULT_PRESENCE_PENALTY,
            "repetition_penalty": DEFAULT_REPETITION_PENALTY,
            "frequency_penalty": DEFAULT_FREQUENCY_PENALTY,
            "penalty_range": DEFAULT_PENALTY_RANGE,
        },
    }


@app.get("/v1/models")
async def list_models():
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


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if _model is None or _tokenizer is None or _generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Resolve per-request thinking flag (None → use server default)
    use_thinking = (
        ENABLE_THINKING if request.enable_thinking is None else request.enable_thinking
    )
    effective_budget = (
        (
            request.thinking_budget
            if request.thinking_budget is not None
            else MAX_THINKING_TOKENS
        )
        if use_thinking
        else 0
    )

    input_ids = _encode_messages(request.messages, enable_thinking=use_thinking)

    prompt_len = input_ids.shape[-1]
    if prompt_len >= _cache_tokens:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt length {prompt_len} exceeds cache size {_cache_tokens}",
        )

    request.max_tokens = _normalize_max_tokens(request.max_tokens)
    max_new = min(request.max_tokens, _cache_tokens - prompt_len - 1)

    eos_token_id = _tokenizer.eos_token_id
    im_end_id = _tokenizer.single_id("<|im_end|>")
    stop_conditions = {t for t in [eos_token_id, im_end_id] if t is not None}

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # Choose generator: two-phase with thinking budget, or plain single-phase.
    # NOTE: the chat template bakes `<think>\n` into the generation prompt prefix
    # (it is part of input_ids, not generated by the model).  We must emit it
    # ourselves as the very first chunk so clients see the complete think block.
    def _gen():
        if use_thinking and PRESERVE_THINK_OUTPUT:
            yield "<think>\n", False, 0
        if use_thinking and effective_budget > 0:
            yield from generate_with_thinking_budget(
                input_ids, request, max_new, stop_conditions, effective_budget
            )
        else:
            yield from generate_with_generator(
                input_ids, request, max_new, stop_conditions
            )

    if request.stream:

        async def generate_stream():
            async with _gen_lock:
                for chunk, eos, _ntok in _gen():
                    if chunk:
                        data = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": _model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        await asyncio.sleep(0)
                data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": _model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    async with _gen_lock:
        pieces = []
        completion_tokens = 0
        for piece, eos, ntok in _gen():
            pieces.append(piece)
            completion_tokens = ntok
        response_text = "".join(pieces)
        if not PRESERVE_THINK_OUTPUT:
            response_text = _strip_thinking(response_text)

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
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_len + completion_tokens,
        },
    }


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3.5-27B-exl3 heretic TP N-GPU server"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--gpu-split", default=",".join(str(x) for x in DEFAULT_GPU_SPLIT)
    )
    parser.add_argument("--cache-tokens", type=int, default=DEFAULT_CACHE_TOKENS)
    parser.add_argument("--cache-quant", default=DEFAULT_CACHE_QUANT)
    parser.add_argument("--tp-backend", default=DEFAULT_TP_BACKEND)
    args = parser.parse_args()

    uvicorn.run(
        "server_27b_layer:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )
