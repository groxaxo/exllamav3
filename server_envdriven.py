"""
server_envdriven.py
~~~~~~~~~~~~~~~~~~~
OpenAI-compatible single-GPU EXL3 server, fully driven by environment variables.

Environment variables:
  MODEL_DIR        Path to quantized EXL3 model directory (required)
  MODEL_ID         Model ID returned in /v1/models  (default: local-exl3)
  GPU_ID           GPU index to use                 (default: 0)
  PORT             HTTP port                        (default: 8001)
  CACHE_TOKENS     KV-cache size in tokens          (default: 32768)
  ENABLE_THINKING  Set to 'true' to strip <think> blocks (default: false)
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Prepend torch lib so EXL3 native extension can find CUDA libraries.
torch_lib = "/home/op/miniconda3/envs/exl3-dev/lib/python3.11/site-packages/torch/lib"
if os.path.exists(torch_lib):
    os.environ["LD_LIBRARY_PATH"] = torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import re
import torch
import json
import time
import uuid
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_fp16

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_ID = os.getenv("MODEL_ID", "local-exl3")
GPU_ID = int(os.getenv("GPU_ID", "0"))
PORT = int(os.getenv("PORT", "8001"))
CACHE_TOKENS = int(os.getenv("CACHE_TOKENS", "32768"))
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "false").lower() == "true"

app = FastAPI(title="ExLlamaV3 OpenAI Server")

model = None
config = None
cache = None
tokenizer = None
generator = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False


def format_chatml(messages: List[ChatMessage], add_assistant: bool = True) -> str:
    prompt = ""
    for msg in messages:
        prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
    if add_assistant:
        if ENABLE_THINKING:
            prompt += "<|im_start|>assistant\n<think>\n"
        else:
            prompt += "<|im_start|>assistant\n"
    return prompt


def strip_think(text: str) -> str:
    """Remove <think>…</think> blocks when thinking mode is off."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


@app.on_event("startup")
async def load_model():
    global model, config, cache, tokenizer, generator
    print(f"[startup] Loading {MODEL_ID} on GPU {GPU_ID} ...")
    torch.cuda.set_device(GPU_ID)
    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS, layer_type=CacheLayer_fp16)
    model.load(use_per_device=[23.0], progressbar=True)
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
    print(f"[startup] {MODEL_ID} ready — cache={CACHE_TOKENS} tokens")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    prompt = format_chatml(request.messages)
    input_ids = tokenizer.encode(prompt, add_bos=True, encode_special_tokens=True)
    stop_tokens = [tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")]

    from exllamav3.generator.sampler import ComboSampler
    sampler = ComboSampler(
        temperature=request.temperature or 0.7,
        top_p=request.top_p or 0.9,
        min_p=0.0,
        top_k=0,
    )
    job = Job(
        input_ids=input_ids,
        max_new_tokens=request.max_tokens or 512,
        stop_conditions=stop_tokens,
        sampler=sampler,
    )

    if request.stream:
        async def generate_stream():
            generator.enqueue(job)
            while generator.num_remaining_jobs():
                for r in generator.iterate():
                    chunk = r.get("text", "")
                    if chunk:
                        data = {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": MODEL_ID,
                            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    if r.get("eos"):
                        data = {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": MODEL_ID,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        yield "data: [DONE]\n\n"
        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    generator.enqueue(job)
    response_text = ""
    total_new_tokens = 0
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            response_text += r.get("text", "")
            if r.get("new_tokens"):
                total_new_tokens = r.get("new_tokens")

    if not ENABLE_THINKING:
        response_text = strip_think(response_text)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
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
        "data": [{"id": MODEL_ID, "object": "model", "created": int(time.time()), "owned_by": "local"}],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
