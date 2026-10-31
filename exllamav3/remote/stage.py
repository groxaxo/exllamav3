"""
Head side of the remote stage: a module that stands in for a span of transformer blocks living on another
machine. It forwards the residual stream to the worker over TCP. Cache and PageTable call into it so the
worker's K/V pages follow every page copy and defrag rotation the generator makes locally.

Opt-in only: Model.from_config attaches stages when EXL3_REMOTE is set (host:port[,host:port]), or call
attach_remote() yourself before creating the Cache.
"""
from __future__ import annotations
import os
import socket
import threading
import time
import weakref
import torch

from ..modules.module import Module
from . import wire
from .common import block_span, check_blocks_supported, drop_cached_props


class RemoteClient:

    def __init__(
        self,
        host: str,
        port: int,
        token: str | None,
        fingerprint: str,
        rope: str | None = None,
        timeout: float | None = None,
    ):
        """
        fingerprint, rope: config_fingerprint() and rope_fingerprint() of the model to attach. The worker checks
        them before it hands its connection over, so probing with a different model (e.g. a draft model) leaves
        the active head alone. Raises wire.ModelMismatch when they differ
        """
        self.addr = (host, port)
        self.lock = threading.Lock()
        self.sock = socket.create_connection(self.addr, timeout = 10)
        self.sock.settimeout(30)
        wire.tune_socket(self.sock)
        op, meta, _ = wire.recv_frame(self.sock, max_meta = wire.PRE_AUTH_META, max_blob = 0)
        if op != wire.HELLO:
            raise ConnectionError("worker did not greet")
        if meta.get("auth") and not token:
            raise PermissionError("worker requires EXL3_REMOTE_TOKEN")
        mac = wire.auth_mac(token or "", bytes.fromhex(meta["nonce"]))
        wire.send_frame(self.sock, wire.AUTH, {"mac": mac, "fingerprint": fingerprint, "rope": rope})
        op, self.info, _ = wire.recv_frame(self.sock)
        if op == wire.ERROR:
            self.close()
            if self.info.get("mismatch"):
                e = wire.ModelMismatch(f"{host}:{port}: {self.info.get('error')}")
                e.kind = self.info["mismatch"]
                raise e
            raise PermissionError(self.info.get("error"))
        # Loads take minutes; a dead peer still surfaces as a reset or EOF
        self.sock.settimeout(timeout if timeout is not None else float(os.environ.get("EXL3_REMOTE_TIMEOUT", 3600)))
        self.rtt_us = self.ping()

    def call(self, op: int, meta: dict | None = None, tensors: list | None = None):
        with self.lock:
            wire.send_frame(self.sock, op, meta, tensors)
            rop, rmeta, rtensors = wire.recv_frame(self.sock)
        if rop == wire.ERROR:
            raise wire.RemoteError(f"worker {self.addr[0]}: {rmeta.get('error')}")
        return rmeta, rtensors

    def ping(self) -> float:
        t = time.perf_counter()
        for _ in range(8):
            self.call(wire.PING)
        return (time.perf_counter() - t) / 8 * 1e6

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def __del__(self):
        self.close()


class RemoteStage(Module):

    def __init__(self, config, client: RemoteClient, lo: int, hi: int, replaced_keys: list[str]):
        super().__init__(config, f"remote[{lo}-{hi - 1}]@{client.addr[0]}", None)
        self.caps.update({"remote_stage": True})
        self.client = client
        self.lo, self.hi = lo, hi
        self.replaced_keys = replaced_keys
        self.wire_dtype = os.environ.get("EXL3_REMOTE_WIRE_DTYPE")  # e.g. float16 to halve fp32 traffic
        self.remote_loaded = False
        self.max_chunk_size = 2048  # set by Model.load_gen before loading
        self.cache_ref = None
        self.skipped_warned = set()
        self.stats = {"calls": 0, "s": 0.0, "bytes": 0}

    def load(self, device, **kwargs):
        self.device = device
        if not self.remote_loaded:
            t0 = time.time()
            info, _ = self.client.call(wire.LOAD, {"max_chunk_size": self.max_chunk_size})
            self.remote_loaded = True
            print(f" -- {self.key}: worker loaded in {time.time() - t0:.1f} s, "
                  f"{info['allocated'] / 1024**3:.2f} GiB allocated, {info['vram_free'] / 1024**3:.2f} GiB free")

    def unload(self):
        if self.remote_loaded:
            self.client.call(wire.UNLOAD)
            self.remote_loaded = False
        self.device = None

    def prepare_for_device(self, x, params):
        # The state leaves this process anyway: skip the upload when the previous module left it on the host
        return x

    def forward(self, x, params, out_dtype = None):
        if params.get("autosplit_measure"):
            return x.to(self.device) if self.device is not None else x
        t0 = time.perf_counter()
        send = x
        meta = {"x": 0, "has_cache": "cache" in params}
        if self.wire_dtype and x.dtype != wire._DTYPES[self.wire_dtype]:
            meta["x_dtype"] = wire._DNAMES[x.dtype]
            meta["wire_dtype"] = self.wire_dtype
            send = x.to(wire._DTYPES[self.wire_dtype])
        tensors = [send]
        meta["params"], skipped = wire.encode_params(params, tensors)
        for k in skipped:
            if k not in self.skipped_warned:
                self.skipped_warned.add(k)
                print(f" !! {self.key}: param '{k}' ({type(params[k]).__name__}) is not sent to the worker")
        rmeta, (y,) = self.client.call(wire.FORWARD, meta, tensors)
        if "y_dtype" in rmeta:
            y = y.to(wire._DTYPES[rmeta["y_dtype"]])
        y = y.to(self.device)
        self.stats["calls"] += 1
        self.stats["s"] += time.perf_counter() - t0
        self.stats["bytes"] += send.numel() * send.element_size() + y.numel() * y.element_size()
        return y

    # Called by Cache and PageTable

    def configure_cache(self, cache, layer_kwargs: dict):
        from ..cache import CacheLayer_fp16, CacheLayer_quant
        if self.cache_ref is not None and self.cache_ref() is not None and self.cache_ref() is not cache:
            raise RuntimeError("only one cache per model is supported with a remote stage")
        lt = cache.layer_type
        if lt is CacheLayer_fp16:
            name, lkw = "fp16", {}
        elif lt is CacheLayer_quant:
            name, lkw = "quant", {k: layer_kwargs[k] for k in ("k_bits", "v_bits") if k in layer_kwargs}
        else:
            raise NotImplementedError(f"cache layer type {lt.__name__} with a remote stage")
        self.client.call(wire.CONFIG_CACHE, {"cache": {
            "layer_type": name, "layer_kwargs": lkw, "max_num_tokens": cache.max_num_tokens,
        }})
        self.cache_ref = weakref.ref(cache)

    def copy_page(self, from_page: int, to_page: int, num_tokens: int):
        self.client.call(wire.COPY_PAGE, {"from_page": from_page, "to_page": to_page, "num_tokens": num_tokens})

    def rotate_cache_pages(self, all_rotations_cpu: torch.Tensor):
        self.client.call(wire.ROTATE, {}, [all_rotations_cpu])

    def optimizer_targets(self):
        return []

    def weights_numel(self):
        return 0

    def get_name(self):
        return "RemoteStage"


def attach_remote(model, endpoints: list[tuple[str, int]], token: str | None) -> list:
    """
    Connect to workers and splice their spans into model.modules. Call before creating the Cache.
    Returns the stages (empty if no worker serves this model)
    """
    fp = wire.config_fingerprint(model.config.directory)
    rope = wire.rope_fingerprint(model.config)
    from ..version import __version__ as exl3_version
    clients = []
    for host, port in endpoints:
        try:
            c = RemoteClient(host, port, token, fp, rope)
        except wire.ModelMismatch as e:
            # Same model with incompatible runtime settings is a setup error; another model (a draft) is not
            if getattr(e, "kind", "model") != "model":
                raise
            print(f" -- exl3 remote: {host}:{port} serves a different model, not attaching to {model.config.directory}")
            continue
        if c.info.get("exl3") != exl3_version:
            print(f" !! exl3 remote: worker {host} runs exllamav3 {c.info.get('exl3')}, head runs {exl3_version}: "
                  f"pin the same version on both boxes")
        clients.append(c)
    if not clients:
        return []
    headers = wire.read_st_headers(model.config.directory)
    # Splice from the highest span down so earlier module indices stay valid
    clients.sort(key = lambda c: -c.info["layers"][0])
    taken = set()
    for c in clients:
        lo, hi = c.info["layers"]
        if taken & set(range(lo, hi)):
            raise ValueError(f"overlapping remote layer ranges at {lo}-{hi - 1}")
        taken |= set(range(lo, hi))
        s, e = block_span(model, lo, hi)
        blocks = model.modules[s:e]
        check_blocks_supported(blocks)
        if model.last_kv_module_idx is not None and s <= model.last_kv_module_idx < e:
            raise ValueError("the last block must stay on the head (prefill stops there); move the remote span earlier")
        keys = [b.key for b in blocks]
        if keys != c.info["block_keys"]:
            raise ValueError("worker and head disagree on block keys")
        if wire.layer_digest(headers, keys) != c.info["digest"]:
            raise ValueError(f"worker {c.addr[0]} has a different quant of layers {lo}-{hi - 1}")
        model.modules[s:e] = [RemoteStage(model.config, c, lo, hi, keys)]
        shift = (e - s) - 1
        for attr in ("last_kv_module_idx", "logit_layer_idx"):
            v = getattr(model, attr, None)
            if v is not None and v >= e:
                setattr(model, attr, v - shift)
        print(f" -- exl3 remote: layers {lo}-{hi - 1} -> {c.addr[0]}:{c.addr[1]} ({c.info['gpu']}, "
              f"rtt {c.rtt_us:.0f} us)")
    model.caps["supports_tp"] = False
    drop_cached_props(model)
    return model.remote_stages()


def parse_endpoints(spec: str) -> list[tuple[str, int]]:
    out = []
    for part in spec.split(","):
        host, _, port = part.strip().rpartition(":")
        out.append((host, int(port)))
    return out


def attach_from_env(model) -> list:
    spec = os.environ.get("EXL3_REMOTE")
    if not spec:
        return []
    return attach_remote(model, parse_endpoints(spec), os.environ.get("EXL3_REMOTE_TOKEN"))
