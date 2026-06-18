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
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_fp16

MODEL_DIR = "/home/op/exllamav3_ampere/models/Qwen3.5-9B-exl3"
PORT = 8004

app = FastAPI(title="ExLlamaV3 - 3090+3060 @ 32K ctx")

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


@app.on_event("startup")
async def load_model():
    global model, config, cache, tokenizer, generator

    print("Loading model on RTX 3090 (GPU 1) + RTX 3060 (GPU 3) @ 32K context...")

    for i in range(torch.cuda.device_count()):
        free = torch.cuda.mem_get_info(i)[0] / (1024**3)
        print(f"  Logical GPU {i}: {free:.2f} GB free")

    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)

    # 32K context
    cache = Cache(model, max_num_tokens=32768, layer_type=CacheLayer_fp16)

    # 3060 gets ~8GB, 3090 gets ~16GB
    model.load(use_per_device=[16.0, 8.0], progressbar=True)
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    print("Model loaded with 32K context!")


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
