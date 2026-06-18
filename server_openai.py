import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch_lib = "/home/op/miniconda3/envs/exl3-dev/lib/python3.11/site-packages/torch/lib"
if os.path.exists(torch_lib):
    os.environ["LD_LIBRARY_PATH"] = (
        torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )

import torch
import json
import time
import uuid
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import uvicorn

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_fp16

MODEL_DIR = "/home/op/exllamav3_ampere/models/Qwen3.5-9B-exl3"
GPU_ID = int(os.getenv("GPU_ID", "0"))
PORT = int(os.getenv("PORT", "8001"))

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
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
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

    print("\n=== Testing Maximum Context Length ===")

    torch.cuda.set_device(GPU_ID)
    free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
    print(f"GPU {GPU_ID} free memory: {free_mem:.2f} GB")

    model_size = 8.4
    overhead = 0.5
    available_for_cache = free_mem - model_size - overhead

    bytes_per_token = 2 * 4 * 4096 * 4 / 1024 / 1024 / 1024
    max_tokens_est = int(available_for_cache / bytes_per_token)

    print(f"Estimated max context (fp16 KV): ~{max_tokens_est} tokens")

    test_sizes = [4096, 8192, 16384, 24576, 32768]

    for size in test_sizes:
        try:
            test_cache = Cache(model, max_num_tokens=size, layer_type=CacheLayer_fp16)
            test_cache.detach_from_model(model)
            del test_cache
            torch.cuda.empty_cache()
            print(f"  {size} tokens: OK")
        except Exception as e:
            print(f"  {size} tokens: FAILED - {e}")
            break

    return max_tokens_est


@app.on_event("startup")
async def load_model():
    global model, config, cache, tokenizer, generator

    print(f"Loading model on GPU {GPU_ID}...")
    torch.cuda.set_device(GPU_ID)

    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)

    cache = Cache(model, max_num_tokens=16384, layer_type=CacheLayer_fp16)

    model.load(use_per_device=[12.0], progressbar=True)
    tokenizer = Tokenizer.from_config(config)

    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    print("Model loaded!")
    test_max_context()


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global model, config, cache, tokenizer, generator

    prompt = format_chatml(request.messages)

    input_ids = tokenizer.encode(prompt, add_bos=True, encode_special_tokens=True)

    stop_tokens = [tokenizer.eos_token_id, tokenizer.single_id("<|im_end|>")]

    from exllamav3.generator.sampler import ComboSampler

    sampler = ComboSampler(
        temperature=request.temperature,
        top_p=request.top_p,
        min_p=0.0,
        top_k=0,
    )

    job = Job(
        input_ids=input_ids,
        max_new_tokens=request.max_tokens,
        stop_conditions=stop_tokens,
        sampler=sampler,
    )

    if request.stream:

        async def generate_stream():
            generator.enqueue(job)
            response_text = ""
            while generator.num_remaining_jobs():
                for r in generator.iterate():
                    chunk = r.get("text", "")
                    if chunk:
                        response_text += chunk
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

                    if r.get("eos"):
                        data = {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": "stop"}
                            ],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    else:
        generator.enqueue(job)
        response_text = ""
        total_new_tokens = 0
        while generator.num_remaining_jobs():
            for r in generator.iterate():
                response_text += r.get("text", "")
                if r.get("new_tokens"):
                    total_new_tokens = r.get("new_tokens")

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
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
                "id": "Qwen3.5-9B-exl3",
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
    uvicorn.run(app, host="0.0.0.0", port=PORT)
