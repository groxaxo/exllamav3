# Remote stage: blocks on a GPU in another machine

`exllamav3.remote` runs the first k transformer blocks of a model, together with their paged K/V cache, on a
GPU in a second machine over Ethernet. The rest of the model loads in layer-split mode on the local GPUs as
usual. This is the ExLlamaV3 version of the llama.cpp RPC layer split from `groxaxo/llama.cpp-cluster`, for
the 3x RTX 3090 + remote RTX 3060 setup.

## GPU validation status

On 31 October 2026, the unchanged PR #4 GPU check passed on one RTX 3090 head and an RTX 3060 worker (layers 0–3), using an explicitly adapted MOSS EXL3-4 Qwen3 text backbone. Real page copies and defrag were verified. Decode measured 69.24 tok/s local versus 59.62 tok/s remote. One Generator control token differed across GPUs; a same-GPU loopback control was bit-identical. This is **PASS_WITH_LIMITS**, not general production or language-quality validation.

See the [GPU validation report, metrics and scope](validation/remote-stage-2026-10-31.md) before deploying another model, runtime or context size.

## How it works

`Model.from_config` checks `EXL3_REMOTE` (`host:port[,host:port]`). When a worker there serves the same model
(same `config.json` hash, same tensor sizes for its layers), that worker's span of blocks is replaced by one
`RemoteStage` module. In `forward_ls`/`prefill_ls`, the stage sends the residual stream and the serializable
forward params (block table, cache seqlens, positions, ...) to the worker. The worker runs the same loop over
its blocks with its own cache and sends the residual back. Each forward pass makes one round trip per stage.

The generator changes K/V pages outside the forward pass, so the worker's cache is kept in step through
hooks in the core classes:

| Hook | Where |
|---|---|
| attach stages, opt-in by env | `Model.from_config` |
| refuse TP and single-device loads | `Model.load_gen` |
| skip the network during the autosplit measuring forward (`params["autosplit_measure"]`) | `Model_LSMixin._load_autosplit` |
| send the cache config to the worker | `Cache.__init__` |
| repeat partial page copies (prompt reuse) | `Cache.copy_page` |
| repeat defrag page rotations | `PageTable.defrag` |

Without `EXL3_REMOTE`, `remote_stages()` is empty and every hook is a no-op.

Limits:
- No recurrent (linear-attention/SSM) layers in the remote span. Qwen3.5/Qwen3-Next-style hybrids need their
  recurrent states and checkpoints mirrored too; that is not done yet.
- One cache per model.
- No tensor-parallel load.
- The last block stays on the head, because `prefill()` stops there.

The wire format has no pickle: a fixed header, JSON metadata and raw tensor bytes. The worker requires an
HMAC token. Bind it to a LAN address only.

## Use

```bash
# 1) plan (any box, no GPU): byte-accurate split, K/V per stage, shards the 3060 box needs
python -m exllamav3.remote.plan --model /models/X-exl3 --remote-gib 12 --local-gib 24,24,24 \
    --cache-tokens 65536 --cache 6,6

# 2) 3060 box: copy *.json + the listed shards (no other weights are read)
export EXL3_REMOTE_TOKEN=$(openssl rand -hex 16)          # same value on both boxes
python -m exllamav3.remote.worker --model /models/X-exl3 --layers 0-7 --bind 10.0.0.2

# 3) 3090 box: any server that calls Model.from_config, then Cache, then model.load(use_per_device=...)
EXL3_REMOTE=10.0.0.2:50070 EXL3_REMOTE_TOKEN=$EXL3_REMOTE_TOKEN python server_openai.py ...
```

Use `use_per_device` (or the server's GPU split) from the plan. Head and worker compare `config.json` and the
effective RoPE settings before the worker accepts a head. If the head applies a runtime override such as
`override_dynamic_seq_len` (Phi-3 LongRoPE), start the worker with the same `--override-dynamic-seq-len`. The worker keeps its weights loaded across
head restarts. A new head replaces the old connection, and a changed cache config rebuilds the worker's cache.

The planner ports `plan_layers.py`: 3090 VRAM first, then the remote 3060 (`min-remote` policy: fewest
remote layers that make the model fit). This fork's base has no MoE CPU offload, so a model that overflows
both is reported as not fitting. `--ram-experts` plans for upstream's `EXL3_MOE_CPU_OFFLOAD`/`EXL3_MOE_CPU_SPLIT`
instead.

## Validate

```bash
sh tests/remote_stubext/build.sh && PYTHONPATH=tests/remote_stubext python -m pytest tests/test_remote.py   # CPU only
EXL3_REMOTE=10.0.0.2:50070 EXL3_REMOTE_TOKEN=... python tests/remote_gpu_check.py --model /models/small-exl3 --gpu-split 22
```

`test_remote.py` runs on CPU. It covers the codec, framing, auth, a loopback worker over TCP, head takeover,
the splice into this repo's `Model`/`Cache` classes, the hooks, a worker starting from only its own shards,
and the planner. It uses a no-op stub of the CUDA extension when no toolchain is present.

`remote_gpu_check.py` needs a model that also fits on the local GPUs. It runs everything twice in separate
processes, local only and then with the remote span:
- raw prefill + greedy decode, with logits KL and speed;
- a `Generator` workload with shared prompt prefixes (page copies), a forced defrag (page rotations) and a
  second round served from the rotated cache.

Both GPUs are sm_86, so expect KL near zero. A mirroring bug shows up as early, wholesale divergence
in the second round.

## Cost

Per decode token, per stage, there is one network round trip (~0.1-0.3 ms on wired 1/2.5 GbE) plus ~0.3-0.5 ms
of serialization; the loopback round trip measured ~0.6 ms in a CPU sandbox. Prefill sends `chunk x hidden x 2 B` each
way, which is 12 MiB per 2048-token chunk at hidden 3072, so 2.5/10 GbE helps prefill, not decode. The 3060
reads weights at ~360 GB/s against ~936 GB/s on a 3090, so keep its span as small as the fit allows.
`EXL3_REMOTE_WIRE_DTYPE=float16` halves traffic for fp32 residual streams, at a small rounding cost.
