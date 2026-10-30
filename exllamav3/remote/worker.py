"""
Remote stage worker. Runs on the box with the extra GPU and owns a contiguous span of transformer blocks
plus their paged K/V cache. Only tensors for those blocks are read from disk, so this box needs the model's
*.json files and the shards the planner lists (python -m exllamav3.remote.plan ...).

    EXL3_REMOTE_TOKEN=... python -m exllamav3.remote.worker --model /models/X-exl3 --layers 0-7 --bind 10.0.0.2
"""
from __future__ import annotations
import argparse
import hmac
import os
import socket
import sys
import threading
import time
import torch

from . import wire
from .common import block_span, check_blocks_supported, drop_cached_props, run_blocks


class Worker:

    def __init__(self, model_dir: str, lo: int, hi: int, device: str):
        # The worker's own model must never get remote stages spliced in
        os.environ.pop("EXL3_REMOTE", None)
        from exllamav3 import Config, Model
        self.model_dir = model_dir
        self.lo, self.hi = lo, hi
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        self.config = Config.from_directory(model_dir)
        self.model = Model.from_config(self.config)
        s, e = block_span(self.model, lo, hi)
        blocks = self.model.modules[s:e]
        check_blocks_supported(blocks)
        self.model.modules = blocks
        self.model.last_kv_module_idx = len(blocks) - 1
        self.model.logit_layer_idx = None
        self.model.first_block_idx = 0
        self.model.caps["supports_tp"] = False
        drop_cached_props(self.model)
        self.block_keys = [b.key for b in blocks]
        headers = wire.read_st_headers(model_dir)
        missing = [k for k in self.block_keys if not any(t.startswith(k + ".") for t in headers)]
        if missing:
            raise FileNotFoundError(f"no tensors for {missing[:3]}... in {model_dir}: copy the shards listed by the planner")
        self.digest = wire.layer_digest(headers, self.block_keys)
        self.fingerprint = wire.config_fingerprint(model_dir)
        self.cache = None
        self.cache_cfg = None
        self.loaded = False
        self.stats = {"forwards": 0, "tokens": 0, "busy_s": 0.0}

    def info(self) -> dict:
        p = torch.cuda.get_device_properties(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        from exllamav3.version import __version__ as exl3_version
        return {
            "layers": [self.lo, self.hi], "block_keys": self.block_keys, "digest": self.digest,
            "fingerprint": self.fingerprint, "exl3": exl3_version, "torch": torch.__version__,
            "gpu": p.name, "vram_free": free, "vram_total": total, "loaded": self.loaded,
            "cache_cfg": self.cache_cfg, "host": socket.gethostname(),
        }

    def configure_cache(self, cfg: dict):
        if cfg == self.cache_cfg and self.cache is not None:
            return
        from exllamav3 import Cache
        from exllamav3.cache import CacheLayer_fp16, CacheLayer_quant
        if self.loaded:
            self.unload()
        if self.cache is not None:
            self.cache.detach_from_model()
            self.cache = None
        layer_type = {"fp16": CacheLayer_fp16, "quant": CacheLayer_quant}[cfg["layer_type"]]
        self.cache = Cache(
            self.model,
            max_num_tokens = cfg["max_num_tokens"],
            layer_type = layer_type,
            **cfg.get("layer_kwargs", {}),
        )
        self.cache_cfg = cfg

    def load(self, max_chunk_size: int) -> dict:
        if not self.loaded:
            t0 = time.time()
            self.model.load(device = self.device, progressbar = True)
            self.loaded = True
            self.load_s = time.time() - t0
            # Measuring forward at the head's chunk size, as the autosplit loader does on every local device:
            # a prefill transient that does not fit fails here at load time instead of on the first request
            if self.cache is not None and max_chunk_size:
                self._measure(max_chunk_size)
        free, total = torch.cuda.mem_get_info(self.device)
        return {"load_s": round(getattr(self, "load_s", 0.0), 2), "vram_free": free, "vram_total": total,
                "allocated": torch.cuda.memory_allocated(self.device)}

    @torch.inference_mode()
    def _measure(self, chunk: int):
        from exllamav3.modules.attn import prepare_for_attn
        from exllamav3.constants import PAGE_SIZE
        chunk = min(chunk, self.cache.max_num_tokens)
        span = -(-chunk // PAGE_SIZE) * PAGE_SIZE
        x = torch.zeros((1, chunk, self.config.hidden_size), dtype = torch.half, device = self.device)
        params = {"attn_mode": "flash_attn", "cache": self.cache, "past_len": 0, "batch_shape": (1, span)}
        prepare_for_attn(torch.zeros((1, chunk), dtype = torch.long), params)
        run_blocks(self.model.modules, x, params)
        torch.cuda.synchronize(self.device)

    def unload(self):
        if self.loaded:
            self.model.unload()
            self.loaded = False
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, params: dict, x_dtype: str | None, has_cache: bool) -> torch.Tensor:
        if not self.loaded:
            raise RuntimeError("forward before load")
        t0 = time.perf_counter()
        if x_dtype:
            x = x.to(wire._DTYPES[x_dtype])
        if has_cache:
            if self.cache is None:
                raise RuntimeError("head runs a cached forward but no cache was configured on this worker")
            params["cache"] = self.cache
        y = run_blocks(self.model.modules, x.to(self.device), params)
        y = y.cpu()
        self.stats["forwards"] += 1
        self.stats["tokens"] += int(x.shape[-2]) if x.dim() >= 2 else 1
        self.stats["busy_s"] += time.perf_counter() - t0
        return y

    def copy_page(self, from_page: int, to_page: int, num_tokens: int):
        self.cache.copy_page(self.cache, from_page, to_page, num_tokens)

    def rotate(self, rotations: torch.Tensor):
        # Same per-tensor call PageTable.defrag makes on local caches
        from exllamav3.ext import exllamav3_ext as ext
        rot = rotations.to(self.device)
        for t in self.cache.get_all_tensors():
            if t is None:
                continue
            buf = torch.empty(t[0].shape, device = t.device, dtype = t.dtype)
            ext.cache_rotate(t, rot, buf)


def serve(worker: Worker, bind: str, port: int, token: str | None):
    """
    One active head at a time. A newly authenticated head replaces the previous connection, so a head that
    restarts (or reloads the model) never queues behind its own stale socket
    """
    srv = socket.create_server((bind, port))
    print(f" -- exl3 remote worker: layers {worker.lo}-{worker.hi - 1} ({len(worker.block_keys)} blocks) "
          f"on {worker.device}, listening on {bind}:{port}", flush = True)
    state = {"conn": None}
    state_lock = threading.Lock()
    op_lock = threading.Lock()

    def takeover(conn):
        with state_lock:
            old, state["conn"] = state["conn"], conn
        if old is not None:
            try:
                old.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def run(conn, addr):
        wire.tune_socket(conn)
        try:
            handle(worker, conn, token, takeover, op_lock)
        except (ConnectionError, OSError) as e:
            print(f" -- client {addr[0]}:{addr[1]} gone: {e}", flush = True)
        finally:
            conn.close()
            s = worker.stats
            if s["forwards"]:
                print(f" -- {s['forwards']} forwards, {s['tokens']} tokens, {s['busy_s']:.2f} s busy", flush = True)

    while True:
        conn, addr = srv.accept()
        print(f" -- client {addr[0]}:{addr[1]} connected", flush = True)
        threading.Thread(target = run, args = (conn, addr), daemon = True).start()


def handle(worker, conn: socket.socket, token: str | None, takeover = None, op_lock = None):
    nonce = wire.new_nonce()
    conn.settimeout(30)
    wire.send_frame(conn, wire.HELLO, {"nonce": nonce.hex(), "auth": bool(token)})
    op, meta, _ = wire.recv_frame(conn)
    if op != wire.AUTH or (token and not hmac.compare_digest(meta.get("mac", ""), wire.auth_mac(token, nonce))):
        wire.send_frame(conn, wire.ERROR, {"error": "authentication failed"})
        return
    conn.settimeout(None)
    if takeover:
        takeover(conn)
    op_lock = op_lock or threading.Lock()
    with op_lock:
        wire.send_frame(conn, wire.RESULT, worker.info())
    while True:
        op, meta, tensors = wire.recv_frame(conn, device = worker.device)
        with op_lock:
            _dispatch(worker, conn, op, meta, tensors)


def _dispatch(worker, conn, op, meta, tensors):
    try:
        if op == wire.FORWARD:
            params = wire.decode_params(meta["params"], tensors)
            y = worker.forward(tensors[meta["x"]], params, meta.get("x_dtype"), meta.get("has_cache", False))
            out_dtype = meta.get("wire_dtype")
            if out_dtype and y.dtype != wire._DTYPES[out_dtype]:
                wire.send_frame(conn, wire.RESULT, {"y_dtype": wire._DNAMES[y.dtype]}, [y.to(wire._DTYPES[out_dtype])])
            else:
                wire.send_frame(conn, wire.RESULT, {}, [y])
        elif op == wire.COPY_PAGE:
            worker.copy_page(meta["from_page"], meta["to_page"], meta["num_tokens"])
            wire.send_frame(conn, wire.RESULT)
        elif op == wire.ROTATE:
            worker.rotate(tensors[0])
            wire.send_frame(conn, wire.RESULT)
        elif op == wire.CONFIG_CACHE:
            worker.configure_cache(meta["cache"])
            wire.send_frame(conn, wire.RESULT)
        elif op == wire.LOAD:
            wire.send_frame(conn, wire.RESULT, worker.load(meta.get("max_chunk_size", 0)))
        elif op == wire.UNLOAD:
            worker.unload()
            wire.send_frame(conn, wire.RESULT)
        elif op == wire.PING:
            wire.send_frame(conn, wire.RESULT, {"t": time.time()})
        else:
            wire.send_frame(conn, wire.ERROR, {"error": f"unknown op {op}"})
    except (ConnectionError, OSError):
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        wire.send_frame(conn, wire.ERROR, {"error": f"{type(e).__name__}: {e}"})


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required = True)
    ap.add_argument("--layers", required = True, help = "inclusive block range, e.g. 0-7")
    ap.add_argument("--device", default = "cuda:0")
    ap.add_argument("--bind", required = True, help = "LAN address to listen on (never 0.0.0.0 on an open network)")
    ap.add_argument("--port", type = int, default = 50070)
    ap.add_argument("--insecure-no-token", action = "store_true", help = "allow running without EXL3_REMOTE_TOKEN")
    a = ap.parse_args()
    token = os.environ.get("EXL3_REMOTE_TOKEN")
    if not token and not a.insecure_no_token:
        sys.exit("set EXL3_REMOTE_TOKEN (same value on both boxes) or pass --insecure-no-token")
    lo, hi = wire.parse_layers(a.layers)
    serve(Worker(a.model, lo, hi, a.device), a.bind, a.port, token)


if __name__ == "__main__":
    main()
