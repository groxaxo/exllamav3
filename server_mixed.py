import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch_lib = "/home/op/miniconda3/envs/exl3-dev/lib/python3.11/site-packages/torch/lib"
if os.path.exists(torch_lib):
    os.environ["LD_LIBRARY_PATH"] = (
        torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )

import json
import logging
import time
import traceback
import uuid
import torch
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, Optional, List
import uvicorn

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant

MODEL_DIR = os.getenv("MODEL_DIR", "/home/op/exllamav3_ampere/models/Qwen3.5-27B-exl3")
MODEL_ID = os.getenv("MODEL_ID", os.path.basename(MODEL_DIR.rstrip("/")))
PORT = int(os.getenv("PORT", "8003"))
CACHE_TOKENS = int(os.getenv("CACHE_TOKENS", "150016"))
GPU_SPLIT = [float(x) for x in os.getenv("GPU_SPLIT", "22.0,10.5").split(",")]
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "16000"))
MIN_MAX_TOKENS = int(os.getenv("MIN_MAX_TOKENS", "12"))
MAX_MAX_TOKENS = int(os.getenv("MAX_MAX_TOKENS", "16000"))
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "false").lower() == "true"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("server_mixed")

app = FastAPI(title=f"ExLlamaV3 OpenAI Server - {MODEL_ID} Mixed")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
config = None
cache = None
tokenizer = None
generator = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False


def format_chatml(messages: List[ChatMessage], add_assistant: bool = True):
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
        elif msg.role == "user":
            prompt += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
        elif msg.role == "assistant":
            prompt += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
    if add_assistant:
        prompt += "<|im_start|>assistant\n"
    return prompt


def test_max_context():
    global model, config, cache, tokenizer
    assert model is not None

    logger.info("=== Testing Maximum Context Length ===")

    for gpu_id in range(torch.cuda.device_count()):
        torch.cuda.set_device(gpu_id)
        free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
        logger.info(f"Logical GPU {gpu_id} free memory: {free_mem:.2f} GB")

    test_sizes = [65536, 98304, 131072, CACHE_TOKENS]

    for size in test_sizes:
        try:
            test_cache = Cache(
                model,
                max_num_tokens=size,
                layer_type=CacheLayer_quant,
                k_bits=4,
                v_bits=4,
            )
            test_cache.detach_from_model(model)
            del test_cache
            torch.cuda.empty_cache()
            logger.info(f"  {size} tokens: OK")
        except Exception as e:
            logger.exception(f"  {size} tokens: FAILED - {e}")
            break


@app.on_event("startup")
async def load_model():
    global model, config, cache, tokenizer, generator

    logger.info(f"Loading model from {MODEL_DIR}...")
    logger.info(f"Using split {GPU_SPLIT} across visible GPUs")

    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)

    cache = Cache(
        model,
        max_num_tokens=CACHE_TOKENS,
        layer_type=CacheLayer_quant,
        k_bits=4,
        v_bits=4,
    )

    model.load(use_per_device=GPU_SPLIT, progressbar=True)
    tokenizer = Tokenizer.from_config(config)

    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    logger.info("Model loaded!")
    test_max_context()


def _log_request_start(
    request_id: str, request: ChatCompletionRequest, prompt_tokens: int
):
    logger.info(
        "request_id=%s stream=%s prompt_tokens=%s max_tokens=%s temperature=%s top_p=%s messages=%s",
        request_id,
        request.stream,
        prompt_tokens,
        request.max_tokens,
        request.temperature,
        request.top_p,
        len(request.messages),
    )


def _log_request_end(
    request_id: str, finish_reason: str, prompt_tokens: int, completion_tokens: int
):
    logger.info(
        "request_id=%s finish_reason=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
        request_id,
        finish_reason,
        prompt_tokens,
        completion_tokens,
        prompt_tokens + completion_tokens,
    )


def _get_input_ids(encoded: Any):
    if isinstance(encoded, tuple):
        return encoded[0]
    return encoded


def _normalize_max_tokens(requested: Optional[int]) -> int:
    if requested is None or requested <= 0:
        return DEFAULT_MAX_TOKENS
    return min(requested, MAX_MAX_TOKENS)


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


def _render_response_text(text: str) -> str:
    if ENABLE_THINKING:
        return text
    return _strip_thinking(text)


def _encode_messages(messages: List[ChatMessage]):
    assert tokenizer is not None
    hf_messages = [{"role": m.role, "content": m.content} for m in messages]
    try:
        return tokenizer.hf_chat_template(
            hf_messages,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
    except Exception:
        logger.warning(
            "hf_chat_template_failed_falling_back_to_manual\n%s", traceback.format_exc()
        )
        prompt = format_chatml(messages)
        return _get_input_ids(
            tokenizer.encode(prompt, add_bos=True, encode_special_tokens=True)
        )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global model, config, cache, tokenizer, generator
    assert tokenizer is not None
    assert generator is not None
    gen: Any = generator

    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    request.max_tokens = _normalize_max_tokens(request.max_tokens)

    input_ids = _encode_messages(request.messages)
    prompt_tokens = input_ids.shape[-1]
    _log_request_start(request_id, request, prompt_tokens)

    prompt_capacity = CACHE_TOKENS - prompt_tokens - 1
    if prompt_capacity < 1:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt length {prompt_tokens} exceeds cache size {CACHE_TOKENS}",
        )
    effective_max_tokens = min(request.max_tokens, prompt_capacity)

    stop_tokens = [tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")]

    from exllamav3.generator.sampler import ComboSampler

    sampler = ComboSampler(
        temperature=request.temperature if request.temperature is not None else 0.7,
        top_p=request.top_p if request.top_p is not None else 0.9,
        min_p=0.0,
        top_k=0,
    )

    job = Job(
        input_ids=input_ids,
        max_new_tokens=effective_max_tokens,
        stop_conditions=stop_tokens,
        sampler=sampler,
    )

    if request.stream:

        async def generate_stream():
            completion_tokens = 0
            finish_reason = "length"
            try:
                gen.enqueue(job)
                while gen.num_remaining_jobs():
                    for r in gen.iterate():
                        chunk = r.get("text", "")
                        if r.get("new_tokens") is not None:
                            completion_tokens = r.get("new_tokens")
                        if chunk:
                            data = {
                                "id": request_id,
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

                        if r.get("eos"):
                            finish_reason = "stop"

                data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": finish_reason}
                    ],
                }
                _log_request_end(
                    request_id, finish_reason, prompt_tokens, completion_tokens
                )
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception:
                logger.error(
                    "request_id=%s stream_generation_failed\n%s",
                    request_id,
                    traceback.format_exc(),
                )
                raise

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    else:
        try:
            gen.enqueue(job)
            response_text = ""
            total_new_tokens = 0
            finish_reason = "length"
            while gen.num_remaining_jobs():
                for r in gen.iterate():
                    response_text += r.get("text", "")
                    if r.get("new_tokens") is not None:
                        total_new_tokens = r.get("new_tokens")
                    if r.get("eos"):
                        finish_reason = "stop"

            cleaned_response_text = _render_response_text(response_text)
            _log_request_end(request_id, finish_reason, prompt_tokens, total_new_tokens)
            return {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": cleaned_response_text,
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": total_new_tokens,
                    "total_tokens": prompt_tokens + total_new_tokens,
                },
            }
        except Exception:
            logger.error(
                "request_id=%s generation_failed\n%s",
                request_id,
                traceback.format_exc(),
            )
            raise HTTPException(
                status_code=500,
                detail="Generation failed; see server log for traceback",
            )


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
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
    uvicorn.run(app, host="0.0.0.0", port=PORT, access_log=True)
