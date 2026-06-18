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

MODEL_DIR = os.getenv(
    "MODEL_DIR", "/home/op/models/gemma4/groxaxo_gemma4-prometheus-exl3-6bpw"
)
MODEL_ID = os.getenv("MODEL_ID", "gemma4-prometheus-exl3-6bpw")
PORT = int(os.getenv("PORT", "1234"))
CACHE_TOKENS = int(os.getenv("CACHE_TOKENS", "65536"))
# GPU_SPLIT for CUDA_VISIBLE_DEVICES=0,1,2,4 with CUDA_DEVICE_ORDER=PCI_BUS_ID
# Logical 0=RTX3090(24GB), 1=RTX3090(24GB), 2=RTX3060(12GB), 3=RTX3090(24GB)
GPU_SPLIT = [float(x) for x in os.getenv("GPU_SPLIT", "21.0,21.0,11.0,21.0").split(",")]
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "8192"))
MIN_MAX_TOKENS = int(os.getenv("MIN_MAX_TOKENS", "12"))
MAX_MAX_TOKENS = int(os.getenv("MAX_MAX_TOKENS", "16384"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("server_gemma4")

app = FastAPI(title=f"ExLlamaV3 OpenAI Server - {MODEL_ID}")

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
stop_token_ids = None


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


class CompletionRequest(BaseModel):
    model: str
    prompt: Any  # str or List[int] (token ids)
    max_tokens: Optional[int] = 1
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    temperature: Optional[float] = 0.0


def test_max_context():
    global model, config, cache, tokenizer
    assert model is not None

    logger.info("=== Testing Maximum Context Length ===")

    for gpu_id in range(torch.cuda.device_count()):
        torch.cuda.set_device(gpu_id)
        free_mem = torch.cuda.mem_get_info()[0] / (1024**3)
        logger.info(f"Logical GPU {gpu_id} free memory: {free_mem:.2f} GB")

    test_sizes = [32768, CACHE_TOKENS]

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
    global model, config, cache, tokenizer, generator, stop_token_ids

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

    # Gemma 4 uses eos_token_id [1, 106] — NOT <|im_end|>
    # Token 1 = </s>, token 106 = <end_of_turn>
    eos = tokenizer.eos_token_id
    stop_token_ids = [eos] if isinstance(eos, int) else list(eos)
    # Also add token 106 (<end_of_turn>) if not already present
    if 106 not in stop_token_ids:
        stop_token_ids.append(106)
    logger.info(f"Stop token IDs: {stop_token_ids}")

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


def _encode_messages(messages: List[ChatMessage]):
    assert tokenizer is not None
    hf_messages = [{"role": m.role, "content": m.content} for m in messages]
    try:
        return tokenizer.hf_chat_template(
            hf_messages,
            add_generation_prompt=True,
        )
    except Exception:
        logger.warning(
            "hf_chat_template failed, falling back to manual Gemma format\n%s",
            traceback.format_exc(),
        )
        # Gemma format fallback
        prompt = "<bos>"
        for msg in messages:
            role = msg.role
            content = msg.content
            if role == "system":
                # Gemma 4 wraps system as user turn
                prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif role == "user":
                prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif role == "assistant":
                prompt += f"<start_of_turn>model\n{content}<end_of_turn>\n"
        prompt += "<start_of_turn>model\n"
        return _get_input_ids(
            tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True)
        )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    global model, config, cache, tokenizer, generator, stop_token_ids
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
        stop_conditions=stop_token_ids,
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
                            "content": response_text,
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


@app.post("/tokenize")
async def tokenize_endpoint(request: dict):
    """Tokenize a prompt string, returns token ids."""
    assert tokenizer is not None
    prompt = request.get("prompt", "")
    ids = tokenizer.encode(str(prompt), add_bos=True, encode_special_tokens=True)
    if isinstance(ids, tuple):
        ids = ids[0]
    return {"tokens": ids.squeeze(0).tolist()}


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """
    OpenAI-compatible /v1/completions with echo+logprobs support.
    Uses model.forward() for a single prefill pass to get per-token logprobs.
    Works for PPL benchmarking (echo=True, logprobs=1).
    """
    global model, tokenizer
    assert model is not None
    assert tokenizer is not None

    # Encode prompt
    if isinstance(request.prompt, list):
        # Already token ids
        import torch as _torch

        input_ids = _torch.tensor([request.prompt], dtype=_torch.long)
    else:
        input_ids = tokenizer.encode(
            str(request.prompt), add_bos=True, encode_special_tokens=True
        )
        if isinstance(input_ids, tuple):
            input_ids = input_ids[0]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

    seq_len = input_ids.shape[-1]

    # Run full forward pass to get logits for all positions
    import torch as _torch
    import torch.nn.functional as _F

    try:
        with _torch.inference_mode():
            # model.forward expects (batch, seq) and returns (batch, seq, vocab)
            logits = model.forward(input_ids)  # (1, seq, vocab)

        # Compute log-probs: softmax over vocab dim
        log_probs = _F.log_softmax(logits.float(), dim=-1)  # (1, seq, vocab)

        # For each position i, the logprob of the token at position i+1
        # (i.e., the "next token" given context up to position i)
        token_logprobs = []
        token_ids_list = input_ids.squeeze(0).tolist()
        tokens_list = []
        for i in range(seq_len):
            tok_id = token_ids_list[i]
            # Decode token piece
            try:
                piece = tokenizer.get_id_to_piece_list()[tok_id]
            except Exception:
                piece = f"<{tok_id}>"
            tokens_list.append(piece)
            if i == 0:
                # First token has no previous context
                token_logprobs.append(None)
            else:
                lp = log_probs[0, i - 1, tok_id].item()
                token_logprobs.append(lp)

    except Exception:
        logger.error("completions_forward_failed\n%s", traceback.format_exc())
        raise HTTPException(
            status_code=500, detail="Forward pass failed; see server log"
        )

    # Build response
    request_id = f"cmpl-{uuid.uuid4().hex[:8]}"

    if request.echo:
        text_out = "".join(tokens_list)
        logprobs_out = (
            {
                "tokens": tokens_list,
                "token_logprobs": token_logprobs,
                "token_ids": token_ids_list,
            }
            if request.logprobs
            else None
        )
    else:
        text_out = ""
        logprobs_out = None

    return {
        "id": request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "text": text_out,
                "index": 0,
                "logprobs": logprobs_out,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": seq_len,
            "completion_tokens": 0,
            "total_tokens": seq_len,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, access_log=True)
