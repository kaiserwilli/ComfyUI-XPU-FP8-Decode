import torch
import triton
import triton.language as tl

@triton.jit
def fused_gemv(
    X, W, S, Y,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BK)

    x = tl.load(
        X + cols,
        cols < K,
        other=0
    ).to(tl.float32)

    raw = tl.load(
        W + rows[:, None] * K + cols[None, :],
        (rows[:, None] < N) & (cols[None, :] < K),
        other=0
    ).to(tl.uint16)

    # Decode E4M3FN directly from its byte representation.
    magnitude = raw & 127

    half_bits = (
        (magnitude << 7) + 0x2000
    ).to(tl.uint16)

    normal = half_bits.to(
        tl.float16, bitcast=True
    ).to(tl.float32)

    subnormal = (
        magnitude.to(tl.float32) * 0.001953125
    )

    w = tl.where(
        magnitude < 8,
        subnormal,
        normal
    )

    w = tl.where((raw & 128) != 0, -w, w)

    w = tl.where(
        magnitude == 127,
        float("nan"),
        w
    )

    acc = tl.sum(w * x[None, :], axis=1)
    scale = tl.load(S)

    tl.store(
        Y + rows,
        acc * scale,
        rows < N
    )


