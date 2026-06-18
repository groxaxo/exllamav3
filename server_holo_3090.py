#!/usr/bin/env python3
"""
OpenAI-compatible, strictly GPU-only server for Holo3-35B-A3B-exl3-6bpw on dual RTX 3090s.

Uses layer-split (pipeline) mode across GPU 1 and GPU 4.
150k context with FP16 KV cache.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,4 \\
      python server_holo_3090.py
"""

import sys
import os

torch_lib = "/home/op/miniconda3/envs/exl3-dev/lib/python3.11/site-packages/torch/lib"
if os.path.exists(torch_lib):
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
    )

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("EXLLAMA_GDN_RECURRENT_BACKEND", "ext")
os.environ.setdefault("EXLLAMA_STRICT_GPU_ONLY", "1")
os.environ.setdefault("EXLLAMA_DISABLE_HOST_RECURRENT_CACHE", "1")
os.environ.setdefault("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.80",
)
os.environ.setdefault("EXLLAMA_EMBED_PREFER_CPU", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import json
import time
import uuid
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_fp16
from exllamav3.generator.sampler.presets import ComboSampler, ArgmaxSampler
from exllamav3.constants import PAGE_SIZE

MODEL_DIR = "/home/op/exllamav3_ampere/models/Holo3-35B-A3B-exl3-6bpw"
MODEL_ID = "Holo3-35B-A3B-exl3-6bpw"
GENERATION_CONFIG_PATH = os.path.join(MODEL_DIR, "generation_config.json")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "1235"))
GPU_SPLIT = [
    float(x.strip())
    for x in os.getenv("GPU_SPLIT", "22.0,22.0").split(",")
    if x.strip()
]
CACHE_TOKENS = int(os.getenv("CACHE_TOKENS", "150000"))
CACHE_QUANT = None
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "16000"))
MAX_THINKING_TOKENS = int(os.getenv("MAX_THINKING_TOKENS", "6000"))

app = FastAPI(title="Holo3-35B ExLlamaV3 Server")
_model = None
_tokenizer = None
_generator = None
_gen_lock = asyncio.Lock()
_stop_token_ids = set()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: List[ChatMessage]
    max_tokens: Optional[int] = MAX_TOKENS
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 20
    min_p: Optional[float] = 0.0
    stream: Optional[bool] = False
    enable_thinking: bool = True


def _align(n):
    return ((n + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE


def _should_add_bos():
    if _tokenizer is None:
        return True
    return getattr(_tokenizer, "add_bos_token", True)


def _extract_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif "text" in item:
                    parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return str(content)


def _encode_messages(messages, enable_thinking=True):
    assert _tokenizer is not None
    hf_messages = [
        {"role": m.role, "content": _extract_content(m.content)} for m in messages
    ]
    try:
        if hasattr(_tokenizer, "hf_render_chat_template"):
            rendered = _tokenizer.hf_render_chat_template(
                hf_messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            ids = _tokenizer.encode(
                rendered, add_bos=_should_add_bos(), encode_special_tokens=True
            )
            if isinstance(ids, tuple):
                ids = ids[0]
            return ids
        if hasattr(_tokenizer, "hf_chat_template"):
            ids = _tokenizer.hf_chat_template(
                hf_messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            if isinstance(ids, tuple):
                ids = ids[0]
            return ids
    except Exception as e:
        print(f"HF chat template failed: {e}, falling back to ChatML")

    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
        elif msg.role == "user":
            prompt += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
        elif msg.role == "assistant":
            prompt += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    ids = _tokenizer.encode(
        prompt, add_bos=_should_add_bos(), encode_special_tokens=True
    )
    if isinstance(ids, tuple):
        ids = ids[0]
    return ids


def _strip_thinking(text):
    while True:
        start = text.find("\U0001f9e0")
        if start == -1:
            break
        end = text.find("\U0001f4a4", start)
        if end == -1:
            text = text[:start].strip()
            break
        text = (text[:start] + text[end + len("\U0001f4a4") :]).strip()
    return text


def _load_stop_token_ids():
    stop_token_ids = set()
    try:
        with open(GENERATION_CONFIG_PATH, "r", encoding="utf-8") as f:
            generation_config = json.load(f)
        eos_token_id = generation_config.get("eos_token_id")
        if isinstance(eos_token_id, list):
            stop_token_ids.update(int(token_id) for token_id in eos_token_id)
        elif eos_token_id is not None:
            stop_token_ids.add(int(eos_token_id))
    except Exception as e:
        print(f"Could not load generation_config stop tokens: {e}")

    if _tokenizer is not None and getattr(_tokenizer, "eos_token_id", None) is not None:
        stop_token_ids.add(int(_tokenizer.eos_token_id))

    if _tokenizer is not None and hasattr(_tokenizer, "single_id"):
        try:
            stop_token_ids.add(int(_tokenizer.single_id("<|im_end|>")))
        except Exception:
            pass

    return stop_token_ids


def _maybe_prepend_think_prefix(text, enable_thinking):
    if not enable_thinking:
        return text
    if text.lstrip().startswith("<think>"):
        return text
    return "<think>\n" + text


def _finish_reason_from_eos_reason(eos_reason):
    if eos_reason == "max_new_tokens":
        return "length"
    return "stop"


def _fit_token_budget(input_ids, max_new):
    prompt_tokens = input_ids.shape[-1]
    cache_limit = _align(CACHE_TOKENS)
    available_completion_tokens = cache_limit - prompt_tokens
    if available_completion_tokens > 0:
        return min(max_new, available_completion_tokens)

    raise HTTPException(
        status_code=400,
        detail={
            "error": "token_budget_exceeded",
            "prompt_tokens": int(prompt_tokens),
            "max_tokens": int(max_new),
            "required_tokens": int(prompt_tokens + max_new),
            "cache_limit_tokens": int(cache_limit),
            "max_prompt_tokens_for_request": int(cache_limit - 1),
        },
    )


def make_sampler(request):
    from exllamav3.generator.sampler.custom import (
        SS_Temperature,
        SS_TopK,
        SS_TopP,
        SS_MinP,
        SS_Sample,
        SS_Argmax,
    )

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


def generate_with_generator(input_ids, request, max_new, stop_conditions):
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
            text = r.get("text", "")
            eos = r.get("eos", False)
            eos_reason = r.get("eos_reason")
            if r.get("new_tokens") is not None:
                total_new_tokens = r.get("new_tokens")
            if text or eos:
                yield text, eos, total_new_tokens, eos_reason


def generate_with_thinking_budget(input_ids, request, max_new, stop_conditions):
    think_end_id = _tokenizer.single_id("</think>")
    phase1_stop = set(stop_conditions)
    if think_end_id is not None:
        phase1_stop.add(think_end_id)

    phase1_max = min(max_new, MAX_THINKING_TOKENS)

    sampler = make_sampler(request)
    job = Job(
        input_ids=input_ids,
        max_new_tokens=phase1_max,
        stop_conditions=phase1_stop,
        sampler=sampler,
    )
    _generator.enqueue(job)

    think_pieces = []
    think_id_chunks = []
    think_tokens_used = 0
    hit_think_end = False
    phase1_eos_reason = None

    while _generator.num_remaining_jobs():
        for r in _generator.iterate():
            text = r.get("text", "")
            token_ids = r.get("token_ids")
            eos = r.get("eos", False)
            eos_reason = r.get("eos_reason")
            if r.get("new_tokens") is not None:
                think_tokens_used = r["new_tokens"]
            if text:
                think_pieces.append(text)
                yield text, False, think_tokens_used, None
            if token_ids is not None:
                think_id_chunks.append(token_ids.cpu().view(-1))
            if eos:
                phase1_eos_reason = eos_reason
                hit_think_end = "".join(think_pieces).rstrip().endswith("</think>")

    injected_close_ids = None
    if not hit_think_end:
        close_tag = "</think>\n\n"
        think_pieces.append(close_tag)
        injected_close_ids = _tokenizer.encode(
            close_tag,
            add_bos=False,
            encode_special_tokens=True,
        )
        if isinstance(injected_close_ids, tuple):
            injected_close_ids = injected_close_ids[0]
        injected_close_ids = injected_close_ids.cpu().view(-1)
        think_tokens_used += injected_close_ids.numel()
        yield close_tag, False, think_tokens_used, None

    if think_id_chunks or injected_close_ids is not None:
        all_chunks = list(think_id_chunks)
        if injected_close_ids is not None:
            all_chunks.append(injected_close_ids)
        if all_chunks:
            think_tensor = torch.cat(all_chunks, dim=0).unsqueeze(0)
        else:
            think_tensor = torch.zeros((1, 0), dtype=torch.long)
        phase2_input = torch.cat([input_ids.cpu(), think_tensor], dim=-1)
    else:
        full_thinking = "".join(think_pieces)
        think_encoded = _tokenizer.encode(
            full_thinking,
            add_bos=False,
            encode_special_tokens=True,
        )
        if isinstance(think_encoded, tuple):
            think_encoded = think_encoded[0]
        phase2_input = torch.cat([input_ids, think_encoded], dim=-1)

    phase2_max = max_new - think_tokens_used
    if phase2_max <= 0:
        yield "", True, think_tokens_used, phase1_eos_reason or "max_new_tokens"
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
            text = r.get("text", "")
            eos = r.get("eos", False)
            eos_reason = r.get("eos_reason")
            total_new = think_tokens_used + (r.get("new_tokens") or 0)
            if text or eos:
                yield text, eos, total_new, eos_reason


@app.on_event("startup")
async def load_model():
    global _model, _tokenizer, _generator, _gen_lock, _stop_token_ids

    print(f"\n{'=' * 60}")
    print(f"  Holo3-35B-A3B-exl3 - dual 3090 server")
    print(f"{'=' * 60}")
    print(f"  Model dir    : {MODEL_DIR}")
    print(f"  GPU split    : {GPU_SPLIT}")
    print(f"  Cache tokens : {CACHE_TOKENS:,}")
    print(f"  KV cache     : FP16")
    print(f"  GPU only     : strict (no CPU offload)")
    print(f"  Max tokens   : {MAX_TOKENS}")
    print(f"  Think budget : {MAX_THINKING_TOKENS}")
    print(f"  Endpoint     : http://{HOST}:{PORT}/v1")
    print()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    print(f"  CUDA_VISIBLE_DEVICES : {visible}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free = torch.cuda.mem_get_info(i)[0] / 1024**3
        print(
            f"  GPU {i}: {p.name}  ({p.total_memory / 1024**3:.1f} GB, {free:.1f} GB free)"
        )
    print()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Loading model config...")
    cfg = Config.from_directory(MODEL_DIR)
    _model = Model.from_config(cfg)

    aligned_cache = _align(CACHE_TOKENS)
    print(f"Creating FP16 KV cache ({aligned_cache:,} tokens)...")
    cache = Cache(
        _model,
        max_num_tokens=aligned_cache,
        layer_type=CacheLayer_fp16,
    )

    print("Loading weights (layer-split across GPUs)...")
    _model.load(use_per_device=GPU_SPLIT, progressbar=True)

    for i in range(torch.cuda.device_count()):
        mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
        mem_res = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: alloc={mem_alloc:.2f}GB  reserved={mem_res:.2f}GB")

    print("\nLoading tokenizer...")
    _tokenizer = Tokenizer.from_config(cfg)
    _stop_token_ids = _load_stop_token_ids()
    print(f"Stop tokens   : {sorted(_stop_token_ids)}")

    print("Creating Generator...")
    _generator = Generator(model=_model, cache=cache, tokenizer=_tokenizer)

    print("\nRunning startup warmup...")
    warmup_ids = _encode_messages(
        [ChatMessage(role="user", content="What is 2+2? Answer in one word.")],
        enable_thinking=False,
    )

    warmup_job = Job(
        input_ids=warmup_ids,
        max_new_tokens=32,
        stop_conditions=_stop_token_ids,
        sampler=ArgmaxSampler(),
    )
    _generator.enqueue(warmup_job)
    warmup_text = ""
    while _generator.num_remaining_jobs():
        for r in _generator.iterate():
            warmup_text += r.get("text", "")
    print(f"  Warmup response: {warmup_text.strip()[:80]}")
    print(f"\n  Server ready at http://{HOST}:{PORT}/v1\n")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global _model, _tokenizer, _generator

    async with _gen_lock:
        input_ids = _encode_messages(
            request.messages,
            enable_thinking=request.enable_thinking,
        )

        stop_conditions = set(_stop_token_ids)

        sampler = make_sampler(request)
        max_new = min(request.max_tokens or MAX_TOKENS, MAX_TOKENS)
        max_new = _fit_token_budget(input_ids, max_new)

        if request.stream:

            async def generate_stream():
                sent_think_prefix = False
                token_iter = (
                    generate_with_thinking_budget(
                        input_ids, request, max_new, stop_conditions
                    )
                    if request.enable_thinking
                    else generate_with_generator(
                        input_ids, request, max_new, stop_conditions
                    )
                )
                for chunk, eos, _total_new_tokens, eos_reason in token_iter:
                    if request.enable_thinking and not sent_think_prefix:
                        if chunk.lstrip().startswith("<think>"):
                            sent_think_prefix = True
                        elif chunk or eos:
                            chunk = "<think>\n" + chunk
                            sent_think_prefix = True
                    if chunk:
                        data = {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    if eos:
                        data = {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": _finish_reason_from_eos_reason(
                                        eos_reason
                                    ),
                                }
                            ],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        yield "data: [DONE]\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        else:
            response_text = ""
            total_new_tokens = 0
            finish_reason = "stop"
            token_iter = (
                generate_with_thinking_budget(
                    input_ids, request, max_new, stop_conditions
                )
                if request.enable_thinking
                else generate_with_generator(
                    input_ids, request, max_new, stop_conditions
                )
            )
            for chunk, eos, total_new_tokens, eos_reason in token_iter:
                response_text += chunk
                if eos:
                    finish_reason = _finish_reason_from_eos_reason(eos_reason)

            response_text = _maybe_prepend_think_prefix(
                response_text,
                bool(request.enable_thinking),
            )

            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": input_ids.shape[-1],
                    "completion_tokens": total_new_tokens,
                    "total_tokens": input_ids.shape[-1] + total_new_tokens,
                },
            }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "Holo3-35B-A3B-exl3-6bpw",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
