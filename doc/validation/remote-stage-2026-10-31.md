# Remote-stage GPU validation — 31 October 2026

**PASS_WITH_LIMITS.** PR #4 at `971b0a9fd0a2f5e73fbd503f36c38e8e22240b88` passed the unchanged GPU check on an RTX 3090 head plus an RTX 3060 worker. This documents a completed hardware run, not a new test of subsequent changes.

## Configuration and model provenance

- One RTX 3090 24 GiB head; one RTX 3060 12 GiB worker serving layers 0–3.
- 36-layer Qwen3 text-backbone view adapted from existing MOSS-TTS-v1.5 EXL3-4 weights.
- Original checkpoint SHA-256: `e0df9f724ea33af5d10be89a19b612f5ba63c1552061d8fe74363ee041a523c6`.
- Original MOSS revision: `cdd3b911b1585e3f2dbc7775ef10f9926f58850a`.
- EXL3 projection tensors retained byte-for-byte with renamed keys. Existing affine W8 text embedding and text output head expanded using original BF16 rounding and stored FP16.
- Audio components and custom per-linear range guards omitted. This is not full MOSS speech validation or an ordinary English language-quality evaluation.
- Head view: 5.615 GiB including CPU embedding. Required worker shard, config and tokenizer hashes matched across hosts. Worker regenerated its four shards from an existing identical checkpoint; no large weights were transferred.
- `--gpu-split 22`, FP16 cache with 8,192-token capacity, default 2,048-token loading/prefill chunk, no wire-dtype override.
- Torch 2.9.1+cu128, runtime CUDA 12.8; Python 3.12.15 on head and 3.11.15 on worker. Drivers: 595.58.03 and 580.178.04 respectively.
- Both exact-source EXL3 extensions built for sm_86. Temporary FlashAttention 2.8.3 packaging restricted dispatch to head dimension 128 and disabled unused backward compilation. All 12 selected CUDA kernel sources matched upstream byte-for-byte.
- A GPU queue admitted every head workload. The other two 3090s were unused.

The default embedding remained in CPU RAM. This run does not establish all-weight GPU residency.

## Original unchanged GPU check

The requested prompt cap was 1,024, but the supplied text contained **962 actual input tokens**. Raw decode generated 32 tokens. Six shared-prefix Generator prompts ran before and after cache reuse/defrag, with 47 actual output tokens each (48 requested).

| Placement | Prefill tok/s | Decode tok/s | Remote forwards |
|---|---:|---:|---:|
| Local 3090 | 3185.14 | 69.24 | 0 |
| 3090 + remote 3060 | 1524.97 | 59.62 | 558 |

The check exited 0 and printed `OK`. Raw greedy tokens matched 100%; logits were finite. Mean KL was 1.75428483e-9 and maximum KL 2.8133794e-8. Maximum absolute logit difference was 6.125: the cross-device logits were not bit-identical.

Five of six Generator completions matched exactly. Prompt index 3 differed at one control token, output index 4: local 151656 (`<|audio_assistant_gen_slot|>`) versus remote 151662 (`<|audio_assistant_delay_slot|>`). That prompt matched 46/47 tokens; all six prompts together matched 281/282 tokens per round. The same difference occurred in both rounds, and each mode reproduced all its own outputs exactly after reuse/defrag.

The original check accepts mean KL <= 1e-4, maximum KL <= 1e-2, raw greedy prefix >= 50%, and median Generator prefix >= 50%. Its median criterion passed despite the affected prompt's 9.86% character-prefix score. Saturated TTS slot predictions limit how much tiny KL and repeated control-token outputs establish about general model correctness.

## Additional real-GPU evidence

| Additional raw prompt | Decode tokens | Mean KL | Max KL | Greedy agreement |
|---|---:|---:|---:|---:|
| 512 tokens | 24 | 3.07826431e-10 | 7.61498953e-9 | 100% |
| 384 tokens | 24 | 1.79739764e-5 | 4.49349405e-4 | 100% |

Both modes performed 11 actual partial page copies and a defrag moving 11 page indices. Across the original and successful supplemental runs, the worker recorded **1,135 FORWARD, 22 COPY_PAGE and 2 ROTATE operations**. The supplemental stage transferred 70,549,504 activation bytes, excluding protocol/parameter overhead.

A same-physical-3090 CUDA loopback control ran the unchanged original check: maximum absolute logit difference and mean/maximum KL were all **zero**; all raw tokens and all six Generator completions matched exactly before and after reuse/defrag. Local/loopback decode was 69.68/67.59 tok/s. This supports a device/runtime arithmetic explanation for the cross-host control-token difference, without isolating hardware from driver/toolchain differences.

The first supplemental exporter failed to serialize a Python Job object after local GPU work. Only that task-owned logging helper was corrected; the full paired rerun exited 0. No PR source, assertion or tolerance was changed. CPU-only protocol regressions passed: **16 tests in 13.34 s**, using the documented stub. All GPU runs used genuine compiled CUDA extensions.

## Memory and performance interpretation

| Measurement | Local head | Remote head | 3060 worker |
|---|---:|---:|---:|
| CUDA peak allocation during load, MiB | 6926.23 | 6429.57 | — |
| CUDA peak allocation, first raw prefill, MiB | 5846.15 | 5349.48 | — |
| CUDA peak allocation, first raw decode, MiB | 5695.71 | 5199.06 | — |
| Driver-sampled memory during load, MiB | 7634 | 7152 | — |
| Worker process peak, driver-reported MiB | — | — | 1050 |
| Worker CUDA peak allocation, MiB | — | — | 808.96 |

Two equally warmed supplemental prompts combined measured 2993.15/1448.89 prefill tok/s and 70.27/60.14 decode tok/s local/remote. These are two-prompt aggregates, not repeated-run medians. Decode was about 14% slower, prefill about 52% slower. Steady head allocation decreased by 496.66 MiB.

The stage provides extra capacity in a serial layer split; it did not improve speed for this small model. No maximum-context capacity sweep or benchmark confidence interval was measured. Driver sampling was every 0.5 s; CUDA allocator peaks supplement fast phases missed between samples. Unrelated process memory was excluded from the worker peak.

## Reproduction and release scope

Use a model that fits on the head alone as the numerical reference. Install matching real CUDA extensions and a compatible FlashAttention build; do not use the CPU stub for GPU validation. On the worker, privately provision the same fresh `EXL3_REMOTE_TOKEN` as the head, then run:

~~~bash
python -m exllamav3.remote.worker --model "$MODEL" --layers 0-3 --bind "$WORKER_IP"
~~~

On an admitted head GPU, with the same privately provisioned token:

~~~bash
EXL3_REMOTE="$WORKER_IP:50070" python tests/remote_gpu_check.py \
  --model "$MODEL" --gpu-split 22 --prompt 1024 --gen 32
~~~

The cited adapted weights require their documented extraction/renaming and head/embedding expansion; the original MOSS TTS checkpoint cannot be passed directly as a standard EXL3 directory. A different model or runtime needs its own GPU validation. These generic commands do not reproduce the supplementary observational harness by themselves.

[Recorded metrics](remote-stage-2026-10-31.json), [original GPU output](remote-stage-2026-10-31-gpu.log), and [same-GPU control output](remote-stage-2026-10-31-control.log) are committed alongside this report. Full local evidence, adaptation/build/measurement scripts and tensor dumps remain under `/home/op/experiments/exl3-remote-stage-gpu-20261031` on the test head; the exported evidence archive is `EXL3_PR4_GPU_validation_2026-10-31.zip`. No credentials, model weights or large tensor dumps are committed here.

Owned workers/samplers stopped, both token files removed, ports 50070/50071 closed, GPU reservations released; unrelated workloads and live installations preserved.

**Validated scope:** adapted dense Qwen3 backbone, four remote layers, FP16 cache, short prompts and one active head. Full MOSS audio, other head dimensions, remote recurrent layers, TP, quantized caches, MTP, large MoE models, three-local-GPU loading, long context and sustained multi-client use remain unvalidated. Cross-3090/3060 bit-exactness and general production readiness are not claimed.
