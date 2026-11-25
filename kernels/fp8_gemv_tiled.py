import torch
import triton
import triton.language as tl

@triton.jit
def decode_fp8(raw):
    raw = raw.to(tl.uint16)
    mag = raw & 127

    bits = ((mag << 7) + 0x2000).to(tl.uint16)
    normal = bits.to(tl.float16, bitcast=True).to(tl.float32)

    subnormal = mag.to(tl.float32) * 0.001953125

    w = tl.where(mag < 8, subnormal, normal)
    w = tl.where((raw & 128) != 0, -w, w)

    return tl.where(mag == 127, float("nan"), w)


@triton.jit
def tiled_gemv(
    X, W, S, Y,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)

    acc = tl.full((BM, BK), 0, tl.float32)

    for tile in tl.range(0, tl.cdiv(K, BK)):
        cols = tile * BK + tl.arange(0, BK)

        x = tl.load(
            X + cols, cols < K, other=0
        ).to(tl.float32)

        raw = tl.load(
            W + rows[:, None] * K + cols[None, :],
            (rows[:, None] < N) & (cols[None, :] < K),
            other=0
        )

        acc += decode_fp8(raw) * x[None, :]

    result = tl.sum(acc, axis=1) * tl.load(S)

    tl.store(Y + rows, result, rows < N)


@triton.jit
def split_gemv(
    X, W, P,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    tile = tl.program_id(1)
    cols = tile * BK + tl.arange(0, BK)

    x = tl.load(
        X + cols, cols < K, other=0
    ).to(tl.float32)

    raw = tl.load(
        W + rows[:, None] * K + cols[None, :],
        (rows[:, None] < N) & (cols[None, :] < K),
        other=0
    )

    result = tl.sum(
        decode_fp8(raw) * x[None, :],
        axis=1
    )

    tl.store(
        P + tile * N + rows,
        result,
        rows < N
    )


@triton.jit
def reduce_partials(
    P, S, Y,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
    BN: tl.constexpr
):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    splits = tl.arange(0, BS)

    partial = tl.load(
        P + splits[:, None] * N + rows[None, :],
        (splits[:, None] < SPLITS) &
        (rows[None, :] < N),
        other=0
    )

    result = tl.sum(partial, axis=0) * tl.load(S)

    tl.store(Y + rows, result, rows < N)


