# Holo3-35B-A3B-exl3-6bpw Server - Session Notes

## Deployment Summary

Deployed Holo3-35B-A3B-exl3-6bpw (Qwen3.5 MoE, 35B params, 256 experts, 8 active) as an OpenAI-compatible API server across two RTX 3090 GPUs with layer-split (pipeline) mode.

### Server Files
- `server_holo_3090.py` - Main server script
- `start_holo_3090.sh` - Launch wrapper (kills old process, nohup, records PID)
- `logs/server_holo_3090_port1235.log` - Current server log
- `logs/server_holo_3090_port1235.pid` - Current PID file

### Configuration
| Setting | Value | Source |
|---------|-------|--------|
| Model dir | `/home/op/exllamav3_ampere/models/Holo3-35B-A3B-exl3-6bpw/` | Hardcoded |
| GPUs | CUDA 1, 4 (two RTX 3090s) | `CUDA_VISIBLE_DEVICES=1,4` |
| GPU split | 22.0, 22.0 GB | `GPU_SPLIT` env |
| Cache tokens | 80,000 (80,128 aligned) | `CACHE_TOKENS` env |
| KV cache quant | 8-bit (k=8, v=8) | Hardcoded `CACHE_QUANT = (8,8)` |
| Max tokens | 16,000 | `MAX_TOKENS` env, also API default |
| Thinking budget | 6,000 | `MAX_THINKING_TOKENS` env |
| Endpoint | `http://0.0.0.0:1235/v1` | `HOST`/`PORT` env |
| Python | `/home/op/miniconda3/envs/exl3-dev/bin/python` | Hardcoded in script |
| Weight distribution | GPU 0: ~21.75 GB, GPU 1: ~4.20 GB | Very uneven layer split |

### Generation Defaults (from `generation_config.json`)
- temperature: 1.0
- top_p: 0.95
- top_k: 20
- eos_token_id: [248044, 248046]

### Special Token IDs
| Token | ID |
|-------|----|
| `<|im_end|>` | 248046 |
| `<|im_start|>` | 248045 |
| `💭` (think open) | 248068 |
| `💭` (think close) | 248069 |
| eos | 248044 |

### Thinking Mode Implementation
- Jinja chat template appends `💭\n` to prompt when `enable_thinking=True`
- Model generates AFTER the tag, never outputs it
- Server prepends `💭\n` to responses so clients see it
- Two-phase generation strategy:
  - **Phase 1**: Thinking up to `min(max_tokens, MAX_THINKING_TOKENS)`, stops at `💭` token or budget cap
  - **Phase 2**: Answer with remaining token budget (`max_tokens - thinking_tokens_used`)
  - If thinking exhausts the budget, `💭\n\n` is injected and phase 2 is skipped
- Thinking budget is part of the same token pool as the answer

### API Parameters
- `max_tokens` (default: 16000) - Total token budget for thinking + answer
- `enable_thinking` (default: true) - Enable two-phase thinking mode
- `temperature`, `top_p`, `top_k`, `min_p`, `stream` - Standard OpenAI params

### Finish Reasons
- `"stop"` - Model produced EOS or `💭` naturally
- `"length"` - Token budget exhausted (both phases)

## Bugs Fixed

### 1. CacheLayer_quant parameter names
- **Issue**: Constructor uses `k_bits`/`v_bits`, not `key_bits`/`value_bits`
- **Fix**: Changed to `k_bits=8, v_bits=8` in Cache constructor

### 2. flash-attn version
- **Issue**: System Python's flash-attn too old for Paged FlashAttention
- **Fix**: Switched to `exl3-dev` conda env (flash-attn 2.8.3)

### 3. Answers getting capped/truncated (this session)
- **Issue**: Phase 1 thinking used server-wide `MAX_THINKING_TOKENS` (6000) regardless of request `max_tokens`. With `max_tokens=300`, thinking consumed thousands of tokens, leaving zero for the answer. Phase 2 early-exit condition (`phase2_max <= 1`) dropped the response entirely.
- **Fix**: Three changes in `server_holo_3090.py`:
  1. Phase 1 capped with `min(max_new, MAX_THINKING_TOKENS)` instead of `MAX_THINKING_TOKENS` alone
  2. Phase 2 early-exit changed from `<= 1` to `<= 0`
  3. `finish_reason` now maps `eos_reason="max_new_tokens"` to `"length"` instead of always returning `"stop"`
- **Refs**: Lines 281, 349-351, 217-220, 513-515

### 4. API default max_tokens too low
- **Issue**: Pydantic default was 512, fallback was 512
- **Fix**: Changed both to `MAX_TOKENS` (16000) so omitted `max_tokens` gets the full budget
- **Refs**: Lines 88, 466

## Reference Server
- `server_27b_layer.py` - Qwen3.5-27B layer-split server, used as pattern source for two-phase thinking budget, CustomSampler, env-driven config, and Ampere optimizations

## Quick Commands
```bash
# Restart server
/home/op/exllamav3_ampere/start_holo_3090.sh

# Health check
curl -s http://127.0.0.1:1235/health

# Non-streaming request (thinking)
curl -s http://127.0.0.1:1235/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Holo3-35B-A3B-exl3-6bpw","messages":[{"role":"user","content":"Hello"}]}'

# Non-streaming request (no thinking)
curl -s http://127.0.0.1:1235/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Holo3-35B-A3B-exl3-6bpw","messages":[{"role":"user","content":"Hello"}],"enable_thinking":false}'
```
