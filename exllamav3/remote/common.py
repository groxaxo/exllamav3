from __future__ import annotations


def block_span(model, lo: int, hi: int) -> tuple[int, int]:
    """Module indices [s, e) of the top-level blocks for layers lo..hi-1"""
    fb = model.first_block_idx
    if fb is None:
        raise NotImplementedError(f"{type(model).__name__} does not mark its first block")
    n = model.config.num_hidden_layers
    if not (0 <= lo < hi <= n):
        raise ValueError(f"layer range {lo}-{hi - 1} outside model with {n} blocks")
    s, e = fb + lo, fb + hi
    for i, m in enumerate(model.modules[s:e]):
        if m.caps.get("remote_stage"):
            raise ValueError(f"layer {lo + i} is already remote")
        for sub in m:
            idx = getattr(sub, "layer_idx", None)
            if idx is not None and idx != lo + i:
                raise ValueError(f"module {m.key} holds layer {idx}, expected {lo + i}")
    return s, e


def check_blocks_supported(blocks: list):
    """The remote stage mirrors paged K/V state only. Reject layers with other cross-call state"""
    for b in blocks:
        for m in b:
            if m.caps.get("recurrent_cache"):
                raise NotImplementedError(f"{m.key}: recurrent (linear-attn/SSM) layers cannot be remote yet")


def drop_cached_props(model):
    for name in ("_get_cache_layers", "_get_recurrent_layers"):
        model.__dict__.pop(name, None)
    for m in model.modules:
        for sub in m:
            sub.__dict__.pop("_all_cache_modules", None)
    model.modules_dict = None


def run_blocks(mods: list, x, params: dict):
    """Same loop as Model_LSMixin.forward_ls, restricted to a span of blocks"""
    for m in mods:
        x = m.prepare_for_device(x, params)
        x = m.forward(x, params)
    return x
