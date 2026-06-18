# Qwen3.5-9B dual RTX 3060 fastest viable deployment

This is the fastest working dual-3060 deployment found in the current
`exllamav3_ampere` tree while meeting these constraints:

- use only the two RTX 3060 GPUs (`CUDA_VISIBLE_DEVICES=2,3`)
- no CPU placement for embeddings or model blocks
- fp16 KV cache
- longest possible context for this model

## Result

- model: `/home/op/exllamav3_ampere/models/Qwen3.5-9B-exl3`
- server: `server_27b_layer.py`
- mode: layer-split across the two 3060s
- GPU split: `9.5,11.0`
- cache tokens: `262144`
- env: `EXLLAMA_EMBED_PREFER_CPU=0`
- verified placement: all model modules on `cuda:0` / `cuda:1`, no CPU embedding
- measured short decode speed: about `37.8 t/s`
- longest usable configured context: `262144` tokens

## Why this is the recommended path

- Native TP2 on the 3060 pair did not hold up in this environment. It hit
  synchronization and CPU-reduce timeouts during warmup / prefill.
- NCCL TP2 also failed on this machine due to CUDA driver/runtime mismatch.
- Layer-split avoids TP communication problems and was the fastest stable path
  that stayed fully on GPU.
- By default the embedding module prefers CPU. Setting
  `EXLLAMA_EMBED_PREFER_CPU=0` keeps embeddings on GPU and removes the last CPU
  placement.

## Launch

Use:

```bash
/home/op/exllamav3_ampere/run_qwen9b_3060_pair.sh
```

Equivalent command:

```bash
source /home/op/miniconda3/etc/profile.d/conda.sh
conda activate exl3-dev
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=2,3 \
EXLLAMA_EMBED_PREFER_CPU=0 \
python /home/op/exllamav3_ampere/server_27b_layer.py \
  --host 0.0.0.0 \
  --port 1235 \
  --model /home/op/exllamav3_ampere/models/Qwen3.5-9B-exl3 \
  --gpu-split 9.5,11.0 \
  --cache-tokens 262144
```

## Health checks

```bash
curl http://127.0.0.1:1235/health
curl http://127.0.0.1:1235/v1/models
```

## Notes

- The script name still points at `server_27b_layer.py`, but the model path is
  the 9B Qwen EXL3 model.
- The model's own max position limit is `262144`, so there is no benefit in
  allocating a larger cache for this deployment.
