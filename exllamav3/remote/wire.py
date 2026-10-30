"""
Framed TCP transport and codec for the remote stage.

No pickle anywhere: a frame is a fixed header, a JSON metadata block and one blob of raw tensor bytes.
A peer that is not authenticated can only make the server parse JSON and fail.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import os
import socket
import struct
import torch

MAGIC = b"EX3R"
PROTO = 1
HDR = struct.Struct("<4sBBHIQ")  # magic, proto, op, reserved, meta_len, blob_len
MAX_META = 16 << 20
MAX_BLOB = 2 << 30

PRE_AUTH_META = 4096

HELLO, AUTH, CONFIG_CACHE, LOAD, UNLOAD, FORWARD, COPY_PAGE, ROTATE, PING, RESULT, ERROR = \
    0x01, 0x02, 0x10, 0x11, 0x12, 0x20, 0x21, 0x22, 0x30, 0x7E, 0x7F

# Params that are process-local on the head, or rebuilt by the worker
SKIP_PARAMS = {
    "cache", "layer_instance", "resid_defer", "resid_pending", "residual", "recurrent_states",
    "backend", "autosplit_measure", "indexed_embeddings", "export_states", "draft_conf",
    "draft_embedding",
}

_DTYPES = {
    "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32, "float64": torch.float64,
    "int8": torch.int8, "uint8": torch.uint8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64,
    "bool": torch.bool,
}
_DNAMES = {v: k for k, v in _DTYPES.items()}


class RemoteError(RuntimeError):
    pass


class ModelMismatch(RemoteError):
    """The worker serves a different model or quant; it kept its current head"""


def _pack(obj, tensors: list):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, torch.Tensor):
        tensors.append(obj)
        return {"__x": len(tensors) - 1}
    if isinstance(obj, torch.dtype):
        return {"__dt": _DNAMES[obj]}
    if isinstance(obj, tuple):
        return {"__t": [_pack(v, tensors) for v in obj]}
    if isinstance(obj, list):
        return [_pack(v, tensors) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return {"__s": [_pack(v, tensors) for v in obj]}
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj):
        return {"__d": {k: _pack(v, tensors) for k, v in obj.items()}}
    raise TypeError(type(obj).__name__)


def _unpack(obj, tensors: list):
    if isinstance(obj, list):
        return [_unpack(v, tensors) for v in obj]
    if not isinstance(obj, dict):
        return obj
    if "__x" in obj:
        return tensors[obj["__x"]]
    if "__dt" in obj:
        return _DTYPES[obj["__dt"]]
    if "__t" in obj:
        return tuple(_unpack(v, tensors) for v in obj["__t"])
    if "__s" in obj:
        return set(_unpack(v, tensors) for v in obj["__s"])
    if "__d" in obj:
        return {k: _unpack(v, tensors) for k, v in obj["__d"].items()}
    raise ValueError("bad packed object")


def encode_params(params: dict, tensors: list) -> tuple[dict, list[str]]:
    """Pack the serializable part of a forward params dict. Returns (packed, skipped_keys)"""
    packed, skipped = {}, []
    for k, v in params.items():
        if k in SKIP_PARAMS:
            continue
        try:
            packed[k] = _pack(v, tensors)
        except (TypeError, KeyError):
            skipped.append(k)
    return packed, skipped


def decode_params(packed: dict, tensors: list) -> dict:
    return {k: _unpack(v, tensors) for k, v in packed.items()}


def _tensor_bytes(t: torch.Tensor) -> memoryview:
    t = t.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    t = t.contiguous().reshape(-1)
    if t.numel() == 0:
        return memoryview(b"")
    return memoryview(t.view(torch.uint8).numpy())


def _sendall_vec(sock: socket.socket, bufs: list):
    bufs = [memoryview(b).cast("B") for b in bufs if len(b)]
    while bufs:
        sent = sock.sendmsg(bufs)
        while sent and bufs:
            n = len(bufs[0])
            if sent >= n:
                sent -= n
                bufs.pop(0)
            else:
                bufs[0] = bufs[0][sent:]
                sent = 0


def _recv_into(sock: socket.socket, buf) -> None:
    view = memoryview(buf)
    while len(view):
        n = sock.recv_into(view)
        if n == 0:
            raise ConnectionError("peer closed connection")
        view = view[n:]


def send_frame(sock: socket.socket, op: int, meta: dict | None = None, tensors: list | None = None):
    tensors = tensors or []
    descs, bufs, off = [], [], 0
    for t in tensors:
        mv = _tensor_bytes(t)
        descs.append({
            "dt": _DNAMES[t.dtype], "shape": list(t.shape),
            "dev": t.device.type, "off": off, "n": len(mv),
        })
        bufs.append(mv)
        off += len(mv)
    body = dict(meta or {})
    body["__tensors"] = descs
    mb = json.dumps(body, separators = (",", ":")).encode()
    _sendall_vec(sock, [HDR.pack(MAGIC, PROTO, op, 0, len(mb), off), mb, *bufs])


_ITEMSIZE = {name: torch.empty(0, dtype = dt).element_size() for name, dt in _DTYPES.items()}
MAX_DIMS = 8


def _check_desc(d, blen: int) -> int:
    """Validate one tensor descriptor against the blob before anything is allocated. Returns its byte count"""
    if not isinstance(d, dict) or d.get("dt") not in _DTYPES:
        raise ConnectionError("bad tensor descriptor")
    shape, off, n = d.get("shape"), d.get("off"), d.get("n")
    if not isinstance(shape, list) or len(shape) > MAX_DIMS or \
            not all(type(x) is int and x >= 0 for x in shape) or type(off) is not int or type(n) is not int:
        raise ConnectionError("bad tensor descriptor")
    numel = 1
    for x in shape:
        numel *= x
    if numel * _ITEMSIZE[d["dt"]] != n or off < 0 or off + n > blen:
        raise ConnectionError("tensor descriptor does not match the frame's data")
    return n


def recv_frame(
    sock: socket.socket,
    device: torch.device | None = None,
    max_meta: int = MAX_META,
    max_blob: int = MAX_BLOB,
) -> tuple[int, dict, list]:
    """
    Read one frame. Tensors that were on a GPU at the sender land on device (if given), CPU tensors stay
    on the CPU (attention reads host-side cache_seqlens on purpose). Every descriptor must account for
    exactly its bytes of the blob, so a frame cannot make the reader allocate more than it sent.
    Pre-authentication reads pass small limits and max_blob = 0.
    """
    hb = bytearray(HDR.size)
    _recv_into(sock, hb)
    magic, proto, op, _, mlen, blen = HDR.unpack(hb)
    if magic != MAGIC or proto != PROTO:
        raise ConnectionError(f"bad frame header {magic!r} v{proto}")
    if mlen > max_meta or blen > max_blob:
        raise ConnectionError(f"frame too large ({mlen}, {blen})")
    mb = bytearray(mlen)
    _recv_into(sock, mb)
    meta = json.loads(mb)
    if not isinstance(meta, dict):
        raise ConnectionError("bad frame metadata")
    descs = meta.pop("__tensors", [])
    if not isinstance(descs, list):
        raise ConnectionError("bad tensor list")
    sizes = [_check_desc(d, blen) for d in descs]
    blob = bytearray(blen)
    if blen:
        _recv_into(sock, blob)
    tensors = []
    for d, n in zip(descs, sizes):
        dt = _DTYPES[d["dt"]]
        if n == 0:
            t = torch.empty(d["shape"], dtype = dt)
        else:
            t = torch.frombuffer(blob, dtype = torch.uint8, offset = d["off"], count = n).view(dt).reshape(d["shape"])
        if device is not None and d.get("dev") == "cuda":
            t = t.to(device)
        tensors.append(t)
    return op, meta, tensors


def tune_socket(sock: socket.socket):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        try:
            sock.setsockopt(socket.SOL_SOCKET, opt, 8 << 20)
        except OSError:
            pass


def auth_mac(token: str, nonce: bytes) -> str:
    return hmac.new(token.encode(), b"exl3r1" + nonce, hashlib.sha256).hexdigest()


def new_nonce() -> bytes:
    return os.urandom(32)


def config_fingerprint(model_dir: str) -> str:
    """Hash of config.json (exl3 writes its quantization_config there)"""
    h = hashlib.sha256()
    with open(os.path.join(model_dir, "config.json"), "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def read_st_headers(model_dir: str) -> dict[str, tuple[str, int]]:
    """tensor key -> (file name, nbytes), from safetensors headers only"""
    out = {}
    for name in sorted(os.listdir(model_dir)):
        if not name.endswith(".safetensors"):
            continue
        with open(os.path.join(model_dir, name), "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                s, e = v["data_offsets"]
                out[k] = (name, e - s)
    return out


def layer_digest(headers: dict[str, tuple[str, int]], block_keys: list[str]) -> str:
    """Hash of (key, nbytes) for every tensor under the given block keys: catches a different quant of the same model"""
    prefixes = tuple(k + "." for k in block_keys)
    h = hashlib.sha256()
    for k in sorted(headers):
        if k.startswith(prefixes):
            h.update(f"{k}:{headers[k][1]};".encode())
    return h.hexdigest()


def files_for_blocks(headers: dict[str, tuple[str, int]], block_keys: list[str]) -> list[str]:
    prefixes = tuple(k + "." for k in block_keys)
    return sorted({f for k, (f, _) in headers.items() if k.startswith(prefixes)})


def parse_layers(s: str) -> tuple[int, int]:
    """'0-7' (inclusive) -> (0, 8)"""
    a, _, b = s.partition("-")
    lo, hi = int(a), int(b if b else a) + 1
    if lo < 0 or hi <= lo:
        raise ValueError(f"bad layer range {s!r}")
    return lo, hi
