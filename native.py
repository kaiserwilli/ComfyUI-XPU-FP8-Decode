"""Load the three verified native libraries; each uses its existing registered namespace."""
import torch
from .settings import PACKAGE_DIR


def load_native_ops():
    libraries = (
        ("qwen_sycl_test", "gemv", "qwen_sycl_test.so"),
        ("qwen_sycl_fused", "qkv", "qwen_sycl_fused.so"),
        ("qwen_sycl_lut", "mlp", "qwen_sycl_lut.so"),
    )
    for namespace, operation, filename in libraries:
        if not hasattr(getattr(torch.ops, namespace), operation):
            torch.ops.load_library(str(PACKAGE_DIR / "lib" / filename))
    return torch.ops.qwen_sycl_test.gemv, torch.ops.qwen_sycl_fused, torch.ops.qwen_sycl_lut
