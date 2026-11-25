# v0.1.0 — Experimental

Initial source-only release of native Qwen3-VL-4B FP8 single-token decoding acceleration for ComfyUI on Intel XPU.

## Highlights

- Direct FP8 GEMV, fused SYCL QKV and MLP projections, Triton kernels, and XPU graph replay.
- Native text and image-conditioned decoding without a second model or external inference server.
- Approximately 17x overall warm text-generation improvement on the tested Intel Arc Pro B70; approximately 92 tokens/s steady-state greedy multimodal decoding.
- Native vision prefill, conditioning, sampling and EOS behavior retained.
- Local build helper with mandatory PIC device linking and ELF TEXTREL rejection.
- Standalone native numerical tests, multimodal benchmark workflow, opt-in modes, and GPL-3.0 licensing.

## Validation

Validated on Linux / Arc Pro B70 / ComfyUI 0.42.0 (8f04fd6ecf4a548937971e840af2a985529fab76) / PyTorch 2.14.1+xpu / Python 3.12.15 / Triton 3.8.0 / oneAPI 2026.2, using scalar-scaled Qwen3-VL-4B FP8 weights. The corrected PIC libraries passed native smoke tests, full ComfyUI integration, cancellation and idle unload/reload recovery. See [integration validation](INTEGRATION_VALIDATION.md).

## Limitations

Only the tested configuration is validated. Output is not guaranteed identical to unoptimized native execution. Active-generation low-VRAM offload, concurrent generation, speculative decoding, altered model weights, other GPUs, longer contexts, and multi-image use are outside the validated scope. The extension ships disabled; enable it only after building and testing locally. No binaries or weights are distributed.

See the [README](../README.md) for build, installation, validation, and rollback instructions, and [NOTICE](../NOTICE.md) for attribution.
