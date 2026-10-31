"""Test stub for the exllamav3 CUDA extension: every symbol is a no-op callable returning 0"""
class _Fn:
    def __init__(self, name): self.name = name
    def __call__(self, *a, **k): return 0
    def __getattr__(self, n): return _Fn(self.name + "." + n)
    def __int__(self): return 0
    def __bool__(self): return False
def __getattr__(name):
    if name.startswith("__"): raise AttributeError(name)
    return _Fn(name)
