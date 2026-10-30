"""
CPU-only tests for exllamav3.remote: codec, framing, auth, a loopback worker over TCP, the planner, and the
splice into this repo's own Model/Cache classes. No GPU needed. Without a CUDA toolchain, build the stub
extension first so exllamav3 imports:

    sh tests/remote_stubext/build.sh
    PYTHONPATH=tests/remote_stubext python -m pytest tests/test_remote.py
"""
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from exllamav3.remote import wire, worker as wmod, stage as smod
from exllamav3.remote.common import block_span

GiB = 1024**3
L = 6


# Codec and framing

def test_codec_roundtrip():
    params = {
        "attn_mode": "flash_attn",
        "block_table": torch.arange(12, dtype = torch.int32).view(3, 4),
        "cache_seqlens": torch.tensor([5, 0, 17], dtype = torch.int32),
        "positions": None,
        "position": 7,
        "batch_shape": (3, 1024),
        "prefill": False,
        "layers": {3, 9},
        "nested": {"a": [1, 2.5, "x", (torch.ones(2, dtype = torch.bfloat16),)]},
        "dt": torch.float16,
        "cache": object(),
        "weird": object(),
        "empty": torch.empty(0, 4),
        "flag": torch.tensor(True),
    }
    tensors = []
    packed, skipped = wire.encode_params(params, tensors)
    assert skipped == ["weird"], skipped
    assert "cache" not in packed
    out = wire.decode_params(packed, tensors)
    assert out["batch_shape"] == (3, 1024) and isinstance(out["batch_shape"], tuple)
    assert out["layers"] == {3, 9}
    assert torch.equal(out["block_table"], params["block_table"])
    assert out["nested"]["a"][3][0].dtype == torch.bfloat16
    assert out["dt"] is torch.float16
    assert out["positions"] is None and out["position"] == 7


def test_frames_over_socketpair():
    a, b = socket.socketpair()
    x = torch.randn(1, 2048, 3072).half()
    bt = torch.arange(8, dtype = torch.int32).view(1, 8)
    big = torch.randn(4096, 4096)  # 64 MiB, forces partial sendmsg
    t = threading.Thread(target = lambda: wire.send_frame(a, wire.FORWARD, {"k": 1}, [x, bt, big]))
    t.start()
    op, meta, ts = wire.recv_frame(b)
    t.join()
    assert op == wire.FORWARD and meta["k"] == 1
    assert torch.equal(ts[0], x) and torch.equal(ts[1], bt) and torch.equal(ts[2], big)
    a.close()
    b.close()


# Loopback worker

class FakeWorker:
    """Stands in for exl3 blocks: y = 2x + cache_seqlens.sum(), and records cache ops"""

    def __init__(self):
        self.device = torch.device("cpu")
        self.lo, self.hi = 0, 4
        self.block_keys = [f"model.layers.{i}" for i in range(4)]
        self.ops = []
        self.stats = {"forwards": 0, "tokens": 0, "busy_s": 0.0}

    def info(self):
        return {"layers": [0, 4], "block_keys": self.block_keys, "digest": "d", "fingerprint": "f", "gpu": "fake"}

    def forward(self, x, params, x_dtype, has_cache):
        if x_dtype:
            x = x.to(wire._DTYPES[x_dtype])
        self.ops.append(("forward", tuple(x.shape), x.dtype, has_cache, sorted(params)))
        return x * 2 + params["cache_seqlens"].sum()

    def copy_page(self, f, t, n): self.ops.append(("copy", f, t, n))
    def rotate(self, r): self.ops.append(("rotate", r.tolist()))
    def configure_cache(self, cfg): self.ops.append(("cache", cfg))
    def load(self, chunk): self.ops.append(("load", chunk)); return {"load_s": 0, "vram_free": 1, "vram_total": 2, "allocated": 0}
    def unload(self): self.ops.append(("unload",))


def _serve(w, srv, token, n = 1):
    state, lock, op_lock = {"conn": None}, threading.Lock(), threading.Lock()

    def takeover(conn):
        with lock:
            old, state["conn"] = state["conn"], conn
        if old is not None:
            old.shutdown(socket.SHUT_RDWR)

    def run(conn):
        try:
            wmod.handle(w, conn, token, takeover, op_lock)
        except (ConnectionError, OSError):
            pass
        finally:
            conn.close()

    for _ in range(n):
        conn, _ = srv.accept()
        wire.tune_socket(conn)
        threading.Thread(target = run, args = (conn,), daemon = True).start()


def _worker_server(w, token, n = 1):
    srv = socket.create_server(("127.0.0.1", 0))
    threading.Thread(target = _serve, args = (w, srv, token, n), daemon = True).start()
    return srv.getsockname()[1]


def test_loopback_worker():
    w = FakeWorker()
    port = _worker_server(w, "s3cret")
    c = smod.RemoteClient("127.0.0.1", port, "s3cret")
    assert c.info["gpu"] == "fake"
    x = torch.randn(2, 1, 3072, dtype = torch.float32)
    params = {"attn_mode": "flash_attn", "cache_seqlens": torch.tensor([3, 4], dtype = torch.int32),
              "block_table": torch.zeros(2, 4, dtype = torch.int32), "cache": object()}
    tensors = [x]
    meta = {"x": 0, "has_cache": True}
    meta["params"], _ = wire.encode_params(params, tensors)
    _, (y,) = c.call(wire.FORWARD, meta, tensors)
    assert torch.allclose(y, x * 2 + 7)
    c.call(wire.COPY_PAGE, {"from_page": 3, "to_page": 9, "num_tokens": 100})
    c.call(wire.ROTATE, {}, [torch.tensor([2, 0, 1], dtype = torch.int32)])
    try:
        c.call(wire.FORWARD, {"x": 0, "params": {}}, [x])   # missing cache_seqlens -> remote KeyError
        raise AssertionError("expected RemoteError")
    except wire.RemoteError as e:
        assert "KeyError" in str(e)
    assert [o[0] for o in w.ops] == ["forward", "copy", "rotate", "forward"]
    assert w.ops[0][3] is True and "cache" not in w.ops[0][4]
    c.close()


def test_bad_token_rejected():
    w = FakeWorker()
    port = _worker_server(w, "right")
    try:
        smod.RemoteClient("127.0.0.1", port, "wrong")
        raise AssertionError("expected PermissionError")
    except PermissionError:
        pass
    assert not w.ops


def test_new_head_takes_over():
    """A second head (restart, model reload) replaces the first instead of queueing behind it"""
    w = FakeWorker()
    port = _worker_server(w, "t", n = 2)
    c1 = smod.RemoteClient("127.0.0.1", port, "t")
    c2 = smod.RemoteClient("127.0.0.1", port, "t")
    c2.call(wire.PING)
    try:
        c1.call(wire.PING)
        raise AssertionError("old head should be cut off")
    except (ConnectionError, OSError):
        pass
    c2.close()


def test_decode_round_trip_overhead():
    w = FakeWorker()
    port = _worker_server(w, None)
    c = smod.RemoteClient("127.0.0.1", port, None)
    x = torch.randn(1, 1, 3072).half()
    params = {"attn_mode": "flash_attn", "cache_seqlens": torch.tensor([100], dtype = torch.int32),
              "block_table": torch.zeros(1, 64, dtype = torch.int32), "positions": None, "prefill": False}
    n = 1000
    t0 = time.perf_counter()
    for _ in range(n):
        tensors = [x]
        meta = {"x": 0, "has_cache": True}
        meta["params"], _ = wire.encode_params(params, tensors)
        c.call(wire.FORWARD, meta, tensors)
    us = (time.perf_counter() - t0) / n * 1e6
    print(f"loopback decode round trip: {us:.0f} us (ping {c.rtt_us:.0f} us)")
    c.close()
    assert us < 5000


# Model splice against this repo's classes

def make_checkpoint(d: str):
    cfg = {
        "architectures": ["MiniMaxM2ForCausalLM"], "hidden_size": 256, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 64, "intermediate_size": 128, "num_local_experts": 4,
        "num_experts_per_tok": 2, "rms_norm_eps": 1e-6, "num_hidden_layers": L, "vocab_size": 512,
        "rope_theta": 10000.0, "max_position_embeddings": 4096, "qk_norm_type": "per_layer",
        "hidden_act": "silu", "torch_dtype": "bfloat16",
    }
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    keys = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for i in range(L):
        p = f"model.layers.{i}"
        keys += [f"{p}.input_layernorm.weight", f"{p}.post_attention_layernorm.weight"]
        keys += [f"{p}.self_attn.{n}.weight" for n in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")]
        keys += [f"{p}.block_sparse_moe.gate.weight"]
        keys += [f"{p}.block_sparse_moe.experts.{e}.{w}.weight" for e in range(4) for w in ("w1", "w2", "w3")]
    for shard, part in enumerate((keys[: len(keys) // 2], keys[len(keys) // 2:])):
        off, hdr = 0, {}
        for k in part:
            hdr[k] = {"dtype": "BF16", "shape": [4], "data_offsets": [off, off + 8]}
            off += 8
        h = json.dumps(hdr).encode()
        with open(os.path.join(d, f"model-{shard:05d}-of-00002.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", len(h)) + h + b"\0" * off)


class _FakeClient:
    """Answers like a worker serving layers 0-1 of the checkpoint in d"""

    def __init__(self, d):
        headers = wire.read_st_headers(d)
        keys = [f"model.layers.{i}" for i in range(2)]
        self.addr = ("10.0.0.2", 50070)
        self.rtt_us = 0.0
        self.info = {"layers": [0, 2], "block_keys": keys, "digest": wire.layer_digest(headers, keys),
                     "fingerprint": wire.config_fingerprint(d), "gpu": "RTX 3060", "exl3": None}
        self.calls = []

    def call(self, op, meta = None, tensors = None):
        self.calls.append((op, meta, tensors))
        return {"allocated": 0, "vram_free": 0}, []

    def close(self):
        pass


def test_splice_and_hooks():
    from exllamav3 import Config, Model, Cache
    from exllamav3.cache import CacheLayer_quant
    with tempfile.TemporaryDirectory() as d:
        make_checkpoint(d)
        plain = Model.from_config(Config.from_directory(d))
        n_mod = len(plain.modules)
        assert block_span(plain, 0, 2) == (plain.first_block_idx, plain.first_block_idx + 2)
        assert not plain.remote_stages()

        fake = _FakeClient(d)
        orig = smod.RemoteClient
        smod.RemoteClient = lambda host, port, token: fake
        os.environ["EXL3_REMOTE"] = "10.0.0.2:50070"
        try:
            model = Model.from_config(Config.from_directory(d))
        finally:
            smod.RemoteClient = orig
            del os.environ["EXL3_REMOTE"]
        stages = model.remote_stages()
        assert len(stages) == 1 and model.modules[plain.first_block_idx] is stages[0]
        assert len(model.modules) == n_mod - 1
        assert model.last_kv_module_idx == plain.last_kv_module_idx - 1
        assert model.logit_layer_idx == plain.logit_layer_idx - 1
        assert type(model.modules[model.last_kv_module_idx]).__name__ == "TransformerBlock"
        assert sorted(m.layer_idx for m in model.get_cache_layers()) == list(range(2, L))

        cache = Cache(model, max_num_tokens = 1024, layer_type = CacheLayer_quant, k_bits = 6, v_bits = 6)
        assert sorted(cache.layers) == list(range(2, L))
        op, meta, _ = fake.calls[-1]
        assert op == wire.CONFIG_CACHE and meta["cache"]["layer_kwargs"] == {"k_bits": 6, "v_bits": 6}
        assert meta["cache"]["max_num_tokens"] == 1024
        try:
            Cache(model, max_num_tokens = 1024)
            raise AssertionError("second cache should be refused")
        except RuntimeError:
            pass

        for layer in cache.layers.values():
            layer.copy_page = lambda *a, **k: None
        cache.copy_page(cache, 3, 7, 100)
        assert fake.calls[-1][:2] == (wire.COPY_PAGE, {"from_page": 3, "to_page": 7, "num_tokens": 100})

        rot = torch.tensor([1, 0, 2], dtype = torch.int)
        stages[0].rotate_cache_pages(rot)
        assert fake.calls[-1][0] == wire.ROTATE and torch.equal(fake.calls[-1][2][0], rot)

        try:
            next(model.load_gen(tensor_p = True))
            raise AssertionError("tensor_p should be refused")
        except AssertionError as e:
            assert "layer-split" in str(e)

        st = stages[0]
        st.device = torch.device("cpu")
        n_calls = len(fake.calls)
        x = torch.zeros(1, 8, 256, dtype = torch.half)
        assert st.forward(x, {"autosplit_measure": True}) is x
        assert len(fake.calls) == n_calls


def test_defrag_mirrors_rotation():
    """PageTable.defrag hands the same page permutation it applies locally to every remote stage"""
    import inspect
    from exllamav3.generator import pagetable
    src = inspect.getsource(pagetable.PageTable.defrag)
    local = src.index("ext.cache_rotate(cache, all_rotations, buffer)")
    remote = src.index("stage.rotate_cache_pages(all_rotations_cpu)")
    assert local < remote


def test_worker_needs_only_its_shards():
    torch.cuda.set_device = lambda d: None
    with tempfile.TemporaryDirectory() as d:
        make_checkpoint(d)
        full = wire.read_st_headers(d)
        keys = [f"model.layers.{i}" for i in range(2)]
        need = wire.files_for_blocks(full, keys)
        assert need == ["model-00000-of-00002.safetensors"], need
        sub = os.path.join(d, "worker_copy")
        os.mkdir(sub)
        shutil.copy(os.path.join(d, "config.json"), sub)
        for f in need:
            shutil.copy(os.path.join(d, f), sub)
        os.environ["EXL3_REMOTE"] = "10.0.0.9:1"   # the worker must ignore it
        w = wmod.Worker(sub, 0, 2, "cuda:0")
        assert "EXL3_REMOTE" not in os.environ
        assert w.block_keys == keys
        assert w.digest == wire.layer_digest(full, keys)
        assert w.fingerprint == wire.config_fingerprint(d)
        assert not w.model.remote_stages()
        assert sorted(m.layer_idx for m in w.model.get_cache_layers()) == [0, 1]
        try:
            wmod.Worker(sub, 0, L, "cuda:0")
            raise AssertionError("missing shards should be reported")
        except FileNotFoundError:
            pass


# Planner

def _minimax_like(d, expert_gib_per_layer = 1.13, layers = 62, experts = 256):
    def write(path, tensors):
        off, hdr = 0, {}
        for n, b in tensors:
            hdr[n] = {"dtype": "U8", "shape": [b], "data_offsets": [off, off + b]}
            off += b
        h = json.dumps(hdr).encode()
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(h)) + h)
    bpe = int(expert_gib_per_layer * GiB / experts)
    shards = [[] for _ in range(8)]
    shards[0].append(("model.embed_tokens.weight", int(1.15 * GiB)))
    shards[7] += [("lm_head.trellis", int(0.45 * GiB)), ("model.norm.weight", 6144), (f"model.layers.{layers}.mtp.weight", 5 * GiB)]
    for i in range(layers):
        s = shards[i * 8 // layers]
        s += [(f"model.layers.{i}.self_attn.q_proj.trellis", 20 << 20), (f"model.layers.{i}.block_sparse_moe.gate.weight", 3 << 20)]
        s += [(f"model.layers.{i}.block_sparse_moe.experts.{e}.w1.trellis", bpe) for e in range(experts)]
    for j, s in enumerate(shards):
        write(os.path.join(d, f"model-{j:05d}-of-00008.safetensors"), s)
    json.dump({"num_hidden_layers": layers, "hidden_size": 3072, "num_attention_heads": 48, "num_key_value_heads": 8,
               "head_dim": 128, "num_local_experts": experts}, open(os.path.join(d, "config.json"), "w"))


def _plan(*args):
    r = subprocess.run([sys.executable, "-m", "exllamav3.remote.plan", *args, "--json"], cwd = ROOT,
                       capture_output = True, text = True)
    if r.returncode:
        return None, r.stderr + r.stdout
    return json.loads(r.stdout), None


def test_plan():
    with tempfile.TemporaryDirectory() as d:
        whale, mid, small = (os.path.join(d, n) for n in ("whale", "mid", "small"))
        for p, g in ((whale, 1.13), (mid, 1.02), (small, 0.8)):
            os.mkdir(p)
            _minimax_like(p, expert_gib_per_layer = g)
        common = ["--local-gib", "24,24,24", "--cache-tokens", "65536", "--cache", "6,6", "--remote-gib", "12"]
        s, err = _plan("--model", small, *common)
        assert s and s["remote_layers"] is None, (s, err)
        m, err = _plan("--model", mid, *common)
        assert m and m["remote_layers"] and not m["moe_cpu_offload"], (m, err)
        assert int(m["remote_layers"].split("-")[1]) + 1 < 8
        assert m["remote_shards"] == ["model-00000-of-00008.safetensors"]
        assert all(g <= 24 for g in m["gpu_split"])
        _, err = _plan("--model", whale, *common)
        assert err and "DOES NOT FIT" in err and "--ram-experts" in err, err
        w, err = _plan("--model", whale, *common, "--ram-experts")
        assert w and w["remote_layers"] == "0-7" and w["moe_cpu_offload"] > 0, (w, err)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
