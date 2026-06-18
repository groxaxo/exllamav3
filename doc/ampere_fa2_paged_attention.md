# ExLlamaV3 Ampere: FA2 hardening + quant-cache local-page staging

## Included in this patch

This patch combines two changes:

1. Harden the existing FlashAttention-2 integration so it only enables when the local FA2 build and GPU support it.
2. Add a quant-cache local-page staging path so paged decode no longer dequantizes the entire cache tensor shape for every FA2 call.

## Why the second part matters

`flash_attn_with_kvcache` already exists in the fork, but `CacheLayer_quant.get_kv()` currently allocates fp16 staging buffers shaped like the full cache and dequantizes the whole paged cache view into them.

This patch changes that path to:

- collect the unique pages referenced by `block_table`
- build a `global_page -> local_page` remap
- stage only those quantized pages into local quant buffers
- dequantize only the local pages into fp16 buffers
- call `flash_attn_with_kvcache` with the remapped local block table
- requantize only the local pages and scatter them back into the global quant cache

## Important caveat

This version is intentionally conservative:

- it still allocates local staging buffers per call
- it does not add a fused C++/CUDA scatter kernel
- it does not yet pool the local staging buffers

So this should cut the full-cache staging waste, but the next step for maximum speed is still buffer pooling and/or a dedicated ext kernel for local-page quant-cache writeback.
