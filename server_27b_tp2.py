#!/usr/bin/env python3
"""
OpenAI-compatible TP server for local ExLlamaV3 models.

Uses real tensor-parallel (NCCL) with paged KV cache and the Generator pipeline
for cached, high-throughput generation.

Supports TP2, TP3, or any GPU count — set CUDA_VISIBLE_DEVICES accordingly.

Usage:
  # TP2 (2×RTX 3090)
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \\
      python server_27b_tp2.py --gpu-split 22.0,22.0

  # TP3 (3×RTX 3090)
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,4 \\
      python server_27b_tp2.py --gpu-split 22.0,22.0,22.0

Environment variables (override CLI defaults):
  MODEL_DIR, GPU_SPLIT, CACHE_TOKENS, CACHE_QUANT, HOST, PORT, TP_BACKEND,
  DEFAULT_MAX_TOKENS (16000), MIN_MAX_TOKENS (12), MAX_MAX_TOKENS (16000)
"""

import sys
import os
import argparse
import asyncio
import time
import json
import uuid

# ------------------------------------------------------------------
# CUDA / NCCL library paths — must be set before importing torch or
# spawning worker processes.
#
# NCCL 2.28.9 requires libcudart.so.12 from CUDA 12.8.  The conda
# environment has three candidates:
#   nvidia/cuda_runtime/lib/libcudart.so.12  → CUDA 12.8  ✓
#   ${conda_env}/lib/libcudart.so.12         → CUDA 12.1  ✗
#   /lib/x86_64-linux-gnu/libcudart.so.12   → CUDA 12.0  ✗
# Prepend the correct path so dlopen always finds CUDA 12.8.
# ------------------------------------------------------------------
_env_root = "/home/op/miniconda3/envs/exl3-dev/lib/python3.11/site-packages"
_cuda_rt_lib = f"{_env_root}/nvidia/cuda_runtime/lib"
_nccl_lib    = f"{_env_root}/nvidia/nccl/lib"
_torch_lib   = f"{_env_root}/torch/lib"
_prepend = ":".join(p for p in [_cuda_rt_lib, _nccl_lib, _torch_lib] if os.path.isdir(p))
if _prepend:
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = _prepend + (":" + existing if existing else "")

# Prevent CPU thread over-subscription next to multi-GPU workloads.
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Any
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
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", os.path.basename(DEFAULT_MODEL_DIR.rstrip("/")))
DEFAULT_GPU_SPLIT = [
    float(x) for x in os.getenv("GPU_SPLIT", "22.0,22.0").split(",")
]
DEFAULT_CACHE_TOKENS = int(os.getenv("CACHE_TOKENS", "32768"))
DEFAULT_CACHE_QUANT = os.getenv("CACHE_QUANT", "fp16")
DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("PORT", "1234"))
DEFAULT_TP_BACKEND = os.getenv("TP_BACKEND", "nccl")
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "16000"))
MIN_MAX_TOKENS = int(os.getenv("MIN_MAX_TOKENS", "12"))
MAX_MAX_TOKENS = int(os.getenv("MAX_MAX_TOKENS", "16000"))
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "true").lower() in ("1", "true", "yes")
PRESERVE_THINK_OUTPUT = os.getenv("PRESERVE_THINK_OUTPUT", "true").lower() in ("1", "true", "yes")
ENABLE_STARTUP_WARMUP = os.getenv("EXLLAMA_STARTUP_WARMUP", "1").lower() in ("1", "true", "yes")
# Cap on tokens spent inside the <think> block before forcing </think>.
MAX_THINKING_TOKENS = int(os.getenv("MAX_THINKING_TOKENS", "1024"))

# ------------------------------------------------------------------
# Globals (populated during startup)
# ------------------------------------------------------------------
_model: Optional[Model] = None
_tokenizer: Optional[Tokenizer] = None
_generator: Optional[Generator] = None
_model_name: str = DEFAULT_MODEL_ID
_cache_tokens: int = DEFAULT_CACHE_TOKENS
_effective_context_tokens: int = DEFAULT_CACHE_TOKENS
_cache_quant: Optional[tuple[int, int]] = None
_model_max_context: Optional[int] = None
_gen_lock: asyncio.Lock

app = FastAPI(title="ExLlamaV3 TP multi-GPU server")


# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL_ID
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    min_p: Optional[float] = 0.08
    top_k: Optional[int] = 0
    stream: Optional[bool] = False
    enable_thinking: Optional[bool] = None
    thinking_budget: Optional[int] = None
    extra_body: Optional[dict[str, Any]] = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def align_to_page(n: int) -> int:
    return (n // PAGE_SIZE) * PAGE_SIZE


def _normalize_max_tokens(requested: Optional[int]) -> int:
    if requested is None or requested <= 0:
        return DEFAULT_MAX_TOKENS
    return max(MIN_MAX_TOKENS, min(requested, MAX_MAX_TOKENS))


def parse_cache_quant(spec: Optional[str]) -> Optional[tuple[int, int]]:
    if spec is None:
        return None
    normalized = spec.strip().lower()
    if normalized in ("", "none", "false", "0", "fp16"):
        return None
    bits = [int(part.strip()) for part in spec.split(",") if part.strip()]
    if len(bits) == 1:
        return bits[0], bits[0]
    if len(bits) == 2:
        return bits[0], bits[1]
    raise ValueError("Specify cache quantization as 'bits' or 'k_bits,v_bits'")


def format_cache_quant(cache_quant: Optional[tuple[int, int]]) -> str:
    if cache_quant is None:
        return "fp16"
    k_bits, v_bits = cache_quant
    if k_bits == v_bits:
        return f"quantized {k_bits}-bit"
    return f"quantized k={k_bits}, v={v_bits}"


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


def _resolve_thinking(request: ChatCompletionRequest) -> tuple[bool, int]:
    extra_body = request.extra_body or {}
    extra_thinking = extra_body.get("thinking")

    use_thinking = ENABLE_THINKING
    if isinstance(extra_thinking, bool):
        use_thinking = extra_thinking
    elif request.enable_thinking is not None:
        use_thinking = request.enable_thinking

    if not use_thinking:
        return False, 0

    effective_budget = MAX_THINKING_TOKENS
    if request.thinking_budget is not None:
        effective_budget = request.thinking_budget
    elif isinstance(extra_thinking, dict):
        budget = extra_thinking.get("budget")
        if isinstance(budget, int) and budget > 0:
            effective_budget = budget

    return True, effective_budget


def format_chatml(messages: List[ChatMessage], add_assistant: bool = True) -> str:
    prompt = ""
    for msg in messages:
        role = msg.role.strip().lower()
        if role == "system":
            prompt += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
        elif role == "user":
            prompt += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
        elif role == "assistant":
            prompt += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
    if add_assistant:
        prompt += "<|im_start|>assistant\n"
    return prompt


def _encode_messages(messages: List[ChatMessage], enable_thinking: Optional[bool] = None):
    assert _tokenizer is not None
    effective_thinking = ENABLE_THINKING if enable_thinking is None else enable_thinking
    hf_messages = [{"role": m.role, "content": m.content} for m in messages]
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
        text = (text[:start] + text[end + len("</think>"):]).strip()
    return text


def _thinking_output_prefix(use_thinking: bool) -> str:
    # The Qwen3.5 chat template already appends "<think>\n" to the generation
    # prompt when enable_thinking=True, so the model's first streamed token IS
    # that "<think>\n".  Do NOT inject it a second time here.
    return ""


def make_sampler(request: ChatCompletionRequest):
    from exllamav3.generator.sampler.custom import SS_Temperature, SS_TopK, SS_TopP, SS_MinP, SS_Sample, SS_Argmax
    temp = request.temperature if request.temperature is not None else 0.7
    if temp <= 0:
        return ArgmaxSampler()
    steps = []
    min_p = request.min_p or 0.0
    if min_p > 0:
        steps.append(SS_MinP(min_p))
    top_k = request.top_k or 0
    if top_k > 0:
        steps.append(SS_TopK(top_k))
    top_p = request.top_p if request.top_p is not None else 1.0
    if 0 < top_p < 1.0:
        steps.append(SS_TopP(top_p))
    steps.append(SS_Temperature(temp))
    steps.append(SS_Sample())
    from exllamav3.generator.sampler.custom import CustomSampler
    return CustomSampler(steps)


def generate_with_generator(input_ids: torch.Tensor, request: ChatCompletionRequest, max_new: int, stop_conditions: set):
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
    reasoning_collector: Optional[list[str]] = None,
):
    """Two-phase generation that caps the <think> block to max_thinking tokens."""
    assert _generator is not None and _tokenizer is not None

    think_end_id = _tokenizer.single_id("</think>")
    phase1_stop = set(stop_conditions)
    if think_end_id is not None:
        phase1_stop.add(think_end_id)

    # Phase 1: generate reasoning, stopping at </think> or the thinking budget.
    sampler = make_sampler(request)
    job = Job(
        input_ids=input_ids,
        max_new_tokens=max_thinking,
        stop_conditions=phase1_stop,
        sampler=sampler,
    )
    _generator.enqueue(job)

    think_pieces: list[str] = []
    think_tokens_used = 0
    hit_think_end = False

    while _generator.num_remaining_jobs():
        for r in _generator.iterate():
            if r.get("stage") == "streaming":
                text = r.get("text", "")
                eos = r.get("eos", False)
                if r.get("new_tokens") is not None:
                    think_tokens_used = r["new_tokens"]
                if text:
                    think_pieces.append(text)
                    if reasoning_collector is not None:
                        reasoning_collector.append(text)
                    if PRESERVE_THINK_OUTPUT:
                        yield text, False, think_tokens_used
                if eos:
                    hit_think_end = "".join(think_pieces).rstrip().endswith("</think>")

    if not hit_think_end:
        print(
            f"  [ThinkBudget] budget of {max_thinking} tokens exhausted — "
            f"injecting </think> after {think_tokens_used} thinking tokens"
        )
        close_tag = "</think>\n"
        think_pieces.append(close_tag)
        if reasoning_collector is not None:
            reasoning_collector.append(close_tag)
        if PRESERVE_THINK_OUTPUT:
            yield close_tag, False, think_tokens_used

    # Phase 2: append the capped think block and generate the final answer.
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
    global _model, _tokenizer, _generator, _gen_lock, _cache_tokens, _effective_context_tokens, _cache_quant, _model_max_context, _model_name

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--gpu-split", default=",".join(str(x) for x in DEFAULT_GPU_SPLIT))
    parser.add_argument("--cache-tokens", type=int, default=DEFAULT_CACHE_TOKENS)
    parser.add_argument("--cache-quant", default=DEFAULT_CACHE_QUANT)
    parser.add_argument("--tp-backend", default=DEFAULT_TP_BACKEND)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args, _ = parser.parse_known_args()

    gpu_split = [float(x) for x in args.gpu_split.split(",")]
    _cache_tokens = align_to_page(args.cache_tokens)
    _cache_quant = parse_cache_quant(args.cache_quant)
    _model_max_context = _read_model_max_context(args.model)
    _effective_context_tokens = (
        min(_cache_tokens, _model_max_context)
        if _model_max_context is not None
        else _cache_tokens
    )
    _model_name = os.getenv("MODEL_ID", os.path.basename(args.model.rstrip("/")))

    print(f"\n{'='*60}")
    print(f"  ExLlamaV3 TP {len(gpu_split)}-GPU server")
    print(f"{'='*60}")
    print(f"  Model dir    : {args.model}")
    print(f"  GPU split    : {gpu_split}")
    print(f"  TP backend   : {args.tp_backend}")
    print(f"  Cache tokens : {_cache_tokens:,}")
    if _model_max_context is not None:
        print(f"  Model limit  : {_model_max_context:,}")
        print(f"  Usable ctx   : {_effective_context_tokens:,}")
    print(f"  KV cache     : {format_cache_quant(_cache_quant)}")
    print(f"  Decode path  : Generator + paged flash_attn (cached)")
    print(f"  Max tokens   : {DEFAULT_MAX_TOKENS} (cap {MAX_MAX_TOKENS})")
    print(f"  Thinking     : {ENABLE_THINKING} (preserve={PRESERVE_THINK_OUTPUT}, budget={MAX_THINKING_TOKENS})")
    print(f"  Endpoint     : http://{args.host}:{args.port}/v1")
    print()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    print(f"  CUDA_VISIBLE_DEVICES : {visible}")
    print(f"  OMP threads          : {os.environ.get('OMP_NUM_THREADS', 'unset')}")
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

    print(f"Creating {format_cache_quant(_cache_quant)} KV cache ({_cache_tokens:,} tokens)...")
    cache = create_cache(_model, _cache_tokens, _cache_quant)

    print("Loading weights (tensor parallel)...")
    _model.load(
        tensor_p=True,
        use_per_device=gpu_split,
        tp_output_device=0,
        tp_backend=args.tp_backend,
        max_chunk_size=2048,
        max_output_size=1,
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
    print(f"\nServer ready.  Cache = {_cache_tokens:,} tokens, usable context = {_effective_context_tokens:,}.\n")


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": _model_name,
        "cache_tokens": _cache_tokens,
        "usable_context_tokens": _effective_context_tokens,
        "cache_quant": list(_cache_quant) if _cache_quant is not None else None,
        "max_position_embeddings": _model_max_context,
        "thinking_enabled": ENABLE_THINKING,
        "preserve_think_output": PRESERVE_THINK_OUTPUT,
        "max_thinking_tokens": MAX_THINKING_TOKENS,
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
                "context_length": _effective_context_tokens,
                "cache_tokens": _cache_tokens,
                "cache_quant": list(_cache_quant) if _cache_quant is not None else None,
                "max_position_embeddings": _model_max_context,
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if _model is None or _tokenizer is None or _generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    use_thinking, effective_budget = _resolve_thinking(request)

    input_ids = _encode_messages(request.messages, enable_thinking=use_thinking)

    prompt_len = input_ids.shape[-1]
    if prompt_len >= _effective_context_tokens:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt length {prompt_len} exceeds usable context {_effective_context_tokens}",
        )

    request.max_tokens = _normalize_max_tokens(request.max_tokens)
    max_new = min(request.max_tokens, _effective_context_tokens - prompt_len - 1)

    eos_token_id = _tokenizer.eos_token_id
    im_end_id = _tokenizer.single_id("<|im_end|>")
    stop_conditions = {t for t in [eos_token_id, im_end_id] if t is not None}

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    think_prefix = _thinking_output_prefix(use_thinking)

    reasoning_chunks: list[str] = []

    def _gen():
        if use_thinking and effective_budget > 0:
            return generate_with_thinking_budget(
                input_ids,
                request,
                max_new,
                stop_conditions,
                effective_budget,
                reasoning_collector=reasoning_chunks,
            )
        return generate_with_generator(input_ids, request, max_new, stop_conditions)

    if request.stream:
        async def generate_stream():
            async with _gen_lock:
                if think_prefix:
                    data = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": _model_name,
                        "choices": [{"index": 0, "delta": {"content": think_prefix}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                for chunk, eos, _ntok in _gen():
                    if chunk:
                        data = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": _model_name,
                            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
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
        pieces = [think_prefix] if think_prefix else []
        completion_tokens = 0
        for piece, eos, ntok in _gen():
            pieces.append(piece)
            completion_tokens = ntok
        response_text = "".join(pieces)
        if not PRESERVE_THINK_OUTPUT:
            response_text = _strip_thinking(response_text)

    response_message = {"role": "assistant", "content": response_text}
    reasoning_text = "".join(reasoning_chunks).strip()
    if reasoning_text:
        response_message["reasoning"] = reasoning_text

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": _model_name,
        "choices": [
            {
                "index": 0,
                "message": response_message,
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
    parser = argparse.ArgumentParser(description="ExLlamaV3 TP N-GPU server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--gpu-split", default=",".join(str(x) for x in DEFAULT_GPU_SPLIT))
    parser.add_argument("--cache-tokens", type=int, default=DEFAULT_CACHE_TOKENS)
    parser.add_argument("--cache-quant", default=DEFAULT_CACHE_QUANT)
    parser.add_argument("--tp-backend", default=DEFAULT_TP_BACKEND)
    args = parser.parse_args()

    uvicorn.run(
        "server_27b_tp2:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )
