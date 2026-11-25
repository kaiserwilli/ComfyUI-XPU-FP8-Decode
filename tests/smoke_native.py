#!/usr/bin/env python3
"""Standalone native-op smoke test, no model files or ComfyUI import required.
This is not a full decoder correctness test. Run with the ComfyUI XPU Python.
"""
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def check(name, got, expected, tolerance=0.01):
    torch.xpu.synchronize()
    a, b = got.float(), expected.float()
    error = (torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(b).clamp_min(1e-12)).item()
    print(f"{name}: relative_L2={error:.8f}, finite={torch.isfinite(a).all().item()}", flush=True)
    if not torch.isfinite(a).all().item() or error >= tolerance:
        raise RuntimeError(f"{name} failed")


def main():
    if not torch.xpu.is_available():
        raise SystemExit("XPU unavailable")
    print(torch.__version__, torch.xpu.get_device_name(0), flush=True)
    for name in ("qwen_sycl_test", "qwen_sycl_fused", "qwen_sycl_lut"):
        torch.ops.load_library(str(ROOT / "lib" / (name + ".so")))
    dev = torch.device("xpu:0")
    torch.manual_seed(42)
    def weight(n, k):
        # Quantize on CPU, then transfer raw bytes; no checkpoint required.
        return (torch.randn(n, k) * 0.1).to(torch.float8_e4m3fn).view(torch.uint8).to(dev)
    x = torch.randn(1, 2560).to(torch.bfloat16).to(dev)
    scale = torch.tensor(0.5, dtype=torch.float32, device=dev)
    base = torch.ops.qwen_sycl_test.gemv
    def gemv(w, wg=128):
        out = torch.empty((1, w.shape[0]), dtype=torch.bfloat16, device=dev)
        base(x, w, scale, out, wg)
        return out
    w = weight(128, 2560)
    expected = (x.cpu().float() @ w.cpu().view(torch.float8_e4m3fn).float().T * 0.5).to(torch.bfloat16).to(dev)
    check("base WG128", gemv(w), expected)
    check("base WG256", gemv(w, 256), expected)
    qw, kw, vw = (weight(n, 2560) for n in (4096, 1024, 1024))
    outs = [torch.empty((1, n), dtype=torch.bfloat16, device=dev) for n in (4096, 1024, 1024)]
    torch.ops.qwen_sycl_fused.qkv(x, qw, kw, vw, scale, scale, scale, *outs, 128)
    for label, out, w in zip(("Q", "K", "V"), outs, (qw, kw, vw)):
        check("fused " + label, out, gemv(w))
    g, u = weight(9728, 2560), weight(9728, 2560)
    bits = torch.arange(65536, device=dev, dtype=torch.int32).to(torch.int16)
    lut = F.silu(bits.view(torch.bfloat16)).contiguous()
    y = torch.empty((1, 9728), dtype=torch.bfloat16, device=dev)
    torch.ops.qwen_sycl_lut.mlp(x, g, u, scale, scale, lut, y, 128)
    check("LUT MLP", y, F.silu(gemv(g)) * gemv(u))
    print("All native smoke tests passed.", flush=True)


if __name__ == "__main__":
    main()
