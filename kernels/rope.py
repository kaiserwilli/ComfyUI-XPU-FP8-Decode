"""Mixed FMA ordering matches the installed eager split-half RoPE's diagonal multiply/addcmul_."""
import torch
import triton
import triton.language as tl

@triton.jit
def _qwen_mixed_rope_kernel(Q, K, R, QO, KO, FORM: tl.constexpr):
    head = tl.program_id(0)
    cols = tl.arange(0, 64)
    a = tl.load(R + cols * 4)
    b = tl.load(R + cols * 4 + 1)
    c = tl.load(R + cols * 4 + 2)
    d = tl.load(R + cols * 4 + 3)
    if head < 32:
        src = Q
        dst = QO
        offset = head * 128
    else:
        src = K
        dst = KO
        offset = (head - 32) * 128
    x = tl.load(src + offset + cols).to(tl.float32)
    y = tl.load(src + offset + cols + 64).to(tl.float32)
    ax = a * x
    by = b * y
    cx = c * x
    dy = d * y
    first = tl.fma(b, y, ax)
    second = tl.fma(c, x, dy)
    tl.store(dst + offset + cols, first)
    tl.store(dst + offset + cols + 64, second)

def _qwen_mixed_rope(q, k, rotation, form=3):
    assert q.shape == (1, 32, 1, 128)
    assert k.shape == (1, 8, 1, 128)
    assert q.dtype == torch.bfloat16
    assert k.dtype == torch.bfloat16
    assert rotation.dtype == torch.float32
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert rotation.is_contiguous()
    assert rotation.numel() == 256
    qo = torch.empty_like(q)
    ko = torch.empty_like(k)
    _qwen_mixed_rope_kernel[40,](q, k, rotation, qo, ko, FORM=form, num_warps=4, enable_fp_fusion=False)
    return (qo, ko)
