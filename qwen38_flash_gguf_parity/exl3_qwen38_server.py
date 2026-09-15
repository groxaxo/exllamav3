#!/usr/bin/env python3
"""OpenAI-compatible server for Qwen3.8-Flash-Next-Uncensored exl3-4bpw.

Backend for the qwen36 multi-variant proxy (:12434). Uses exllamav3 with
tensor-parallel inference across 3x RTX 3090 and an fp16 KV cache.

Endpoints (what the proxy needs):
  GET  /health, /v1/models
  POST /v1/chat/completions   (streaming + non-streaming)

Config via environment (see run_qwen38_flash_next_unc_exl3.sh):
  MODEL_DIR, HOST, PORT, SERVED_MODEL_NAME, CACHE_SIZE, GPU_SPLIT,
  TP_BACKEND, CHUNK_SIZE, MAX_BATCH_SIZE, NGRAM_RAM, LOGFILE,
  CHAT_TEMPLATE_FILE

Template / tool-call parity with the GGUF sibling
(qwen38-flash-next-unc-q4km-agentic-v2 on :12702, llama-server
--chat-template-file .../qwen-sharp-chat-templates/chat_template.jinja):
this server renders the SAME sharp template file and mirrors the
llama-server response behavior for the Qwen3-Coder output parser that the
template selects (default deepseek reasoning format):

* a leading <think>...</think> block becomes `reasoning_content`
  (reasoning may also end at <tool_call>; unclosed at EOF still counts);
* content runs until the first <tool_call> (leading whitespace stripped);
* XML <tool_call> blocks (<function=NAME> / <parameter=ARG> matching the
  request tools/schemas) become OpenAI `tool_calls` with finish_reason
  "tool_calls"; anything else in a tool region fails it (content
  truncates at <tool_call>, no calls, no error) unless tool_choice is
  "required", which raises like llama-server;
* streaming re-parses the accumulated text per update and emits ordered
  diffs (reasoning, content, then tool calls) with stable per-index ids,
  a {"role", "content": null} first chunk, and a separate usage chunk.

Deliberate scope: response_format/JSON-schema constrained generation is
not implemented (llama-server grammar has no exllamav3 equivalent here),
and this backend stays text-only (no mmproj).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.cache import CacheLayer_fp16, CacheLayer_quant
from exllamav3.generator.sampler import ComboSampler

from exl3_qwen38_parser import diff_parsed, parse_completion

# ---------------------------------------------------------------- config

MODEL_DIR = os.environ.get("MODEL_DIR", "/home/op/models/Qwen3.8-Flash-Next-Unc-exl3-4bpw")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "12703"))
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "qwen38-flash-next-unc-exl3-4bpw")
CACHE_KV_BITS = int(os.environ.get("CACHE_KV_BITS", "0"))  # 0=fp16 else quant k/v bits
CACHE_SIZE = int(os.environ.get("CACHE_SIZE", "65536"))
GPU_SPLIT = os.environ.get("GPU_SPLIT", "22.5,22.5,22.5")
TP_BACKEND = os.environ.get("TP_BACKEND", "native")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "4096"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "4"))
NGRAM_RAM = os.environ.get("NGRAM_RAM", "1") == "1"
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", "4096"))
REQUEST_TIMEOUT_S = float(os.environ.get("REQUEST_TIMEOUT_S", "1800"))
# Same file the GGUF sibling passes via llama-server --chat-template-file.
CHAT_TEMPLATE_FILE = os.environ.get(
    "CHAT_TEMPLATE_FILE", "/home/op/models/qwen-sharp-chat-templates/chat_template.jinja"
)

LOGFILE = os.environ.get(
    "LOGFILE", "/home/op/ULTIMATE/NAMEOFMODEL/logs/qwen38-flash-next-unc-exl3.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("exl3-server")

app = FastAPI(title="exl3-qwen38-unc")

# ---------------------------------------------------------------- globals

config: Config | None = None
model: Model | None = None
cache: Cache | None = None
tokenizer: Tokenizer | None = None
generator: Generator | None = None
chat_template = None
EOS_TOKEN_ID: int | None = None

# Single-dispatch worker: the exl3 Generator is driven by exactly one thread,
# so requests are serialized without any lock in the request path (a lock held
# across a sync streaming generator can be orphaned when the client
# disconnects, wedging the server). Worker protocol on out_q (unbounded, so
# the worker never blocks even if the client went away):
#   ("text", str)     streamed delta (streaming requests only)
#   ("final", dict)   eos result dict (always last on success)
#   ("error", exc)    failure (always last on error)
import queue as _queue_module

_dispatch_queue: _queue_module.Queue = _queue_module.Queue()


def _dispatch_worker() -> None:
    while True:
        body, out_q, want_stream = _dispatch_queue.get()
        try:
            if want_stream:
                final = _run_job(body, lambda t: out_q.put(("text", t)))
            else:
                chunks: list[str] = []
                final = _run_job(body, chunks.append)
                out_q.put(("result", "".join(chunks)))
            out_q.put(("final", final))
        except Exception as exc:  # noqa: BLE001 - forwarded to the request
            out_q.put(("error", exc))
        finally:
            _dispatch_queue.task_done()


threading.Thread(target=_dispatch_worker, daemon=True, name="exl3-dispatch").start()


def _raise_exception(msg: str):
    raise ValueError(f"chat_template: {msg}")


def load_model() -> None:
    global config, model, cache, tokenizer, generator, chat_template, EOS_TOKEN_ID
    t0 = time.time()
    log.info("loading config from %s", MODEL_DIR)
    cfg = Config.from_directory(MODEL_DIR)
    if NGRAM_RAM:
        # PLE n-gram table fully in RAM (tens of GB) instead of per-token disk reads
        cfg.infer_params.ngram_stream_from_disk = False
        log.info("ngram_stream_from_disk=False (NGRAM_RAM=1)")
    mdl = Model.from_config(cfg)
    cache_desc = "fp16"
    if CACHE_KV_BITS in (2, 3, 4, 6, 8):
        cch = Cache(mdl, max_num_tokens=CACHE_SIZE, layer_type=CacheLayer_quant, k_bits=CACHE_KV_BITS, v_bits=CACHE_KV_BITS)
        cache_desc = "q" + str(CACHE_KV_BITS) + "-kv"
    else:
        cch = Cache(mdl, max_num_tokens=CACHE_SIZE, layer_type=CacheLayer_fp16)
    split = [float(x) for x in GPU_SPLIT.split(",")] if GPU_SPLIT else None
    log.info(
        "loading weights: tensor_parallel TP-backend=%s gpu_split=%s cache=%d %s",
        TP_BACKEND, split, CACHE_SIZE, cache_desc,
    )
    # Qwen4Exp: supports_tp=False in exl3 1.5.x -> layer-split (pipeline)
    # autosplit across all visible GPUs. TP is used automatically if a future
    # exl3/arch revision enables it.
    use_tp = bool(mdl.caps.get("supports_tp")) and TP_BACKEND != "layer-split"
    log.info("multi-gpu mode: %s", "tensor-parallel" if use_tp else "layer-split autosplit")
    mdl.load(
        use_per_device=split,
        tensor_p=use_tp,
        tp_backend=TP_BACKEND,
        progressbar=False,
        verbose=False,
        max_chunk_size=CHUNK_SIZE,
        max_batch_size=1,
    )
    tok = Tokenizer.from_config(cfg)
    gen = Generator(
        model=mdl,
        cache=cch,
        tokenizer=tok,
        max_batch_size=MAX_BATCH_SIZE,
        max_chunk_size=CHUNK_SIZE,
    )
    from jinja2 import BaseLoader, Environment

    template_path = Path(CHAT_TEMPLATE_FILE)
    if not template_path.is_file():
        log.warning(
            "CHAT_TEMPLATE_FILE %s missing; falling back to stock %s/chat_template.jinja",
            template_path, MODEL_DIR,
        )
        template_path = Path(MODEL_DIR) / "chat_template.jinja"
    env = Environment(loader=BaseLoader(), autoescape=False)
    env.globals["raise_exception"] = _raise_exception
    chat_template = env.from_string(template_path.read_text())
    log.info("chat template: %s", template_path)

    config, model, cache, tokenizer, generator = cfg, mdl, cch, tok, gen
    EOS_TOKEN_ID = tok.eos_token_id
    log.info(
        "ready in %.1fs: model=%s cache=%d ctx_devices=%s",
        time.time() - t0, SERVED_MODEL_NAME, CACHE_SIZE,
        [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    )


# ---------------------------------------------------------------- prompt

def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten content blocks to text; reject vision parts (text-only server)."""
    out: list[dict[str, Any]] = []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, list):
            texts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = str(part.get("type", "")).lower()
                if ptype in ("text",):
                    texts.append(str(part.get("text", "")))
                elif "text" in part and ptype not in ("image_url", "image", "video", "input_audio"):
                    texts.append(str(part.get("text", "")))
                elif ptype in ("image_url", "image", "video") or "image_url" in part or "image" in part:
                    raise ValueError("vision content is not supported by this backend (text-only)")
            m["content"] = "\n".join(t for t in texts if t)
        # OpenAI tool_calls carry JSON-string arguments; the jinja template
        # iterates them as a mapping, so parse defensively.
        tcs = m.get("tool_calls")
        if isinstance(tcs, list):
            fixed = []
            for tc in tcs:
                tc = dict(tc)
                fn = dict(tc.get("function", {}) or {})
                args = fn.get("arguments")
                if isinstance(args, str) and args.strip():
                    try:
                        fn["arguments"] = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        fn["arguments"] = {}
                tc["function"] = fn
                fixed.append(tc)
            m["tool_calls"] = fixed
        out.append(m)
    return out


def _render_prompt(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    messages = _normalize_messages(body.get("messages", []))
    if not messages:
        raise ValueError("no messages provided")
    tools = body.get("tools")
    # llama-server parity: tool_choice "none" suppresses the tools block.
    if body.get("tool_choice") == "none":
        tools = None
    ctk = dict(body.get("chat_template_kwargs") or {})
    ctk.setdefault("add_generation_prompt", True)
    # top-level reasoning_effort / reasoning.effort aliases (proxy may send)
    if "reasoning_effort" not in ctk:
        if isinstance(body.get("reasoning_effort"), str):
            ctk["reasoning_effort"] = body["reasoning_effort"]
        elif isinstance(body.get("reasoning"), dict) and body["reasoning"].get("effort"):
            ctk["reasoning_effort"] = body["reasoning"]["effort"]
        elif isinstance(body.get("extra_body"), dict) and body["extra_body"].get("reasoning_effort"):
            ctk["reasoning_effort"] = body["extra_body"]["reasoning_effort"]
    prompt = chat_template.render(messages=messages, tools=tools, **ctk)
    return prompt, ctk


def _assistant_generation_prompt(body: dict[str, Any]) -> str:
    prompt, _ctk = _render_prompt(body)
    marker = "<|im_start|>assistant\n"
    idx = prompt.rfind(marker)
    if idx == -1:
        return ""
    return prompt[idx:]


# ---------------------------------------------------------------- sampling

def _build_sampler(body: dict[str, Any]) -> ComboSampler:
    rep = body.get("repetition_penalty", body.get("repeat_penalty", 1.0))
    try:
        rep = float(rep)
    except (TypeError, ValueError):
        rep = 1.0
    return ComboSampler(
        rep_p=rep,
        pres_p=float(body.get("presence_penalty", 0.0) or 0.0),
        freq_p=float(body.get("frequency_penalty", 0.0) or 0.0),
        temperature=float(body.get("temperature", 0.7)),
        min_p=float(body.get("min_p", 0.0) or 0.0),
        top_k=int(body.get("top_k", 20) or 0),
        top_p=float(body.get("top_p", 0.8)),
        temp_last=False,
    )


def _stop_conditions(body: dict[str, Any]) -> list:
    stops: list = [EOS_TOKEN_ID, "<|im_end|>"]
    req = body.get("stop")
    if isinstance(req, str) and req:
        stops.append(req)
    elif isinstance(req, list):
        stops.extend(s for s in req if isinstance(s, str) and s)
    return stops


def _max_new_tokens(body: dict[str, Any], prompt_len: int) -> int:
    want = body.get("max_tokens", body.get("max_completion_tokens"))
    try:
        want = int(want) if want else DEFAULT_MAX_TOKENS
    except (TypeError, ValueError):
        want = DEFAULT_MAX_TOKENS
    room = CACHE_SIZE - prompt_len - 1
    if room <= 0:
        raise ValueError(f"prompt too long ({prompt_len} tokens) for cache size {CACHE_SIZE}")
    return max(1, min(want, room))


# ---------------------------------------------------------------- generation

def _parse_response(
    text: str,
    body: dict[str, Any],
    *,
    partial: bool,
    generation_prompt: str = "",
) -> dict[str, Any]:
    """Parse generated text with the request tools (llama-server parity)."""
    # llama-server parity: tool_choice "none" suppresses the tools block in the
    # prompt, so the parser must not validate a tool region against tools either.
    tools = None if body.get("tool_choice") == "none" else body.get("tools")
    return parse_completion(
        text,
        tools,
        body.get("tool_choice"),
        partial=partial,
        generation_prompt=generation_prompt,
    )


def _finish_reason(eos_reason: str | None, has_tool_calls: bool = False) -> str:
    if has_tool_calls:
        return "tool_calls"
    return "length" if eos_reason == "max_new_tokens" else "stop"


def _run_job(body: dict[str, Any], on_text) -> dict[str, Any]:
    """Enqueue one job and iterate until eos. Calls on_text(delta) per chunk.

    Runs only on the dispatch worker thread. Returns the final eos result dict.
    """
    prompt, _ctk = _render_prompt(body)
    input_ids = tokenizer.encode(prompt, add_bos=True, encode_special_tokens=True)
    prompt_len = input_ids.shape[1] if hasattr(input_ids, "shape") else len(input_ids[0])
    max_new = _max_new_tokens(body, int(prompt_len))
    seed = body.get("seed")
    try:
        seed = int(seed) if seed is not None else None
    except (TypeError, ValueError):
        seed = None
    job = Job(
        input_ids=input_ids,
        max_new_tokens=max_new,
        sampler=_build_sampler(body),
        seed=seed,
        stop_conditions=_stop_conditions(body),
        identifier="req",
    )
    generator.enqueue(job)
    final: dict[str, Any] = {}
    deadline = time.time() + REQUEST_TIMEOUT_S
    while generator.num_remaining_jobs():
        if time.time() > deadline:
            raise TimeoutError(f"generation exceeded {REQUEST_TIMEOUT_S:.0f}s")
        for result in generator.iterate():
            if result.get("stage") != "streaming":
                continue
            text = result.get("text") or ""
            if text:
                on_text(text)
            if result.get("eos"):
                final = result
    if not final:
        raise RuntimeError("generator finished without eos result")
    final["prompt_tokens"] = int(prompt_len)
    return final


def _sse(data: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(data, ensure_ascii=False).encode("utf-8") + b"\n\n"


# ---------------------------------------------------------------- routes

@app.get("/health")
def health():
    return {"status": "ok" if generator is not None else "loading", "model": SERVED_MODEL_NAME}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": SERVED_MODEL_NAME, "object": "model", "owned_by": "exl3"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(body: dict[str, Any]):
    stream = bool(body.get("stream"))
    if stream:
        return StreamingResponse(
            _stream_async(body), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    t0 = time.time()
    out_q: _queue_module.Queue = _queue_module.Queue()
    _dispatch_queue.put((body, out_q, False))
    completion = ""
    final: dict[str, Any] | None = None
    while True:
        kind, payload = out_q.get()
        if kind == "result":
            completion = payload
        elif kind == "final":
            final = payload
            break
        elif kind == "error":
            if isinstance(payload, ValueError):
                return JSONResponse(status_code=400, content={"error": str(payload)})
            if isinstance(payload, TimeoutError):
                return JSONResponse(status_code=504, content={"error": str(payload)})
            log.exception("chat completion failed")
            return JSONResponse(status_code=500, content={"error": "generation failed"})
    assert final is not None
    dt = time.time() - t0
    new_tokens = int(final.get("new_tokens", 0) or 0)
    log.info(
        "non-stream done: prompt=%d new=%d eos=%s %.1fs %.1f tok/s",
        final.get("prompt_tokens"), new_tokens,
        final.get("eos_reason"), dt, (new_tokens / dt) if dt > 0 else 0,
    )
    rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    generation_prompt = _assistant_generation_prompt(body)
    try:
        parsed = _parse_response(
            completion, body, partial=False, generation_prompt=generation_prompt
        )
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    message: dict[str, Any] = {"role": "assistant", "content": parsed["content"]}
    if parsed["reasoning_content"]:
        message["reasoning_content"] = parsed["reasoning_content"]
    if parsed["tool_calls"]:
        message["tool_calls"] = parsed["tool_calls"]
    return {
        "id": rid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": SERVED_MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": _finish_reason(
                final.get("eos_reason"), bool(parsed["tool_calls"])
            ),
        }],
        "usage": {
            "prompt_tokens": final.get("prompt_tokens", 0),
            "completion_tokens": new_tokens,
            "total_tokens": final.get("prompt_tokens", 0) + new_tokens,
        },
    }


async def _stream_async(body: dict[str, Any]):
    """Async SSE iterator; generation runs on the dispatch worker.

    Client disconnects cancel this coroutine only; the dispatch worker always
    runs the job to completion (its queue is unbounded) and stays healthy.
    """
    import anyio.to_thread

    rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def chunk(delta: dict[str, Any], finish: str | None = None) -> bytes:
        return _sse({
            "id": rid, "object": "chat.completion.chunk", "created": created,
            "model": SERVED_MODEL_NAME,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        })

    t0 = time.time()
    yield chunk({"role": "assistant", "content": None})
    out_q: _queue_module.Queue = _queue_module.Queue()
    _dispatch_queue.put((body, out_q, True))
    generation_prompt = _assistant_generation_prompt(body)
    acc = ""
    prev = _parse_response("", body, partial=True, generation_prompt=generation_prompt)
    try:
        while True:
            kind, payload = await anyio.to_thread.run_sync(out_q.get)
            if kind == "text":
                acc += payload
                cur = _parse_response(
                    acc, body, partial=True, generation_prompt=generation_prompt
                )
                for delta in diff_parsed(prev, cur):
                    yield chunk(delta)
                prev = cur
            elif kind == "final":
                final = payload
                break
            elif kind == "error":
                if isinstance(payload, (ValueError, TimeoutError)):
                    yield _sse({"error": str(payload)})
                else:
                    log.exception("stream failed")
                    yield _sse({"error": "generation failed"})
                return
    except (GeneratorExit, anyio.get_cancelled_exc_class()):
        # Client went away; the dispatch worker finishes harmlessly.
        return
    dt = time.time() - t0
    new_tokens = int(final.get("new_tokens", 0) or 0)
    log.info(
        "stream done: prompt=%d new=%d eos=%s %.1fs %.1f tok/s",
        final.get("prompt_tokens"), new_tokens,
        final.get("eos_reason"), dt, (new_tokens / dt) if dt > 0 else 0,
    )
    try:
        cur = _parse_response(
            acc, body, partial=False, generation_prompt=generation_prompt
        )
    except ValueError as exc:
        yield _sse({"error": str(exc)})
        return
    for delta in diff_parsed(prev, cur):
        yield chunk(delta)
    finish = _finish_reason(final.get("eos_reason"), bool(cur["tool_calls"]))
    tail: dict[str, Any] = {
        "id": rid, "object": "chat.completion.chunk", "created": created,
        "model": SERVED_MODEL_NAME,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
    }
    yield _sse(tail)
    if isinstance(body.get("stream_options"), dict) and body["stream_options"].get("include_usage"):
        pt = int(final.get("prompt_tokens", 0) or 0)
        yield _sse({
            "id": rid, "object": "chat.completion.chunk", "created": created,
            "model": SERVED_MODEL_NAME, "choices": [],
            "usage": {
                "prompt_tokens": pt, "completion_tokens": new_tokens,
                "total_tokens": pt + new_tokens,
            },
        })
    yield b"data: [DONE]\n\n"


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    import uvicorn

    load_model()
    uvicorn.run(app, host=HOST, port=PORT, workers=1, log_level="info", timeout_keep_alive=75)

