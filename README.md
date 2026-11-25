# ComfyUI-XPU-FP8-Decode

**Experimental native Qwen3-VL-4B FP8 decoding acceleration for Intel XPU in ComfyUI.**

On the tested Intel Arc Pro B70, warm text generation improved from approximately **5.3 to 92 tokens/s overall**, with steady greedy decoding around 96 tokens/s. Image-conditioned decoding reaches approximately 92 tokens/s steady-state. No separate llama.cpp server or additional model checkpoint is needed. Native vision encoding, prefill and diffusion conditioning remain in place.

> Initial release scope: one validated Linux/B70 software configuration, opt-in acceleration, source-only distribution. Other hardware, models, and ComfyUI revisions are unverified.

## Results

| Test | Generated tokens | Overall tokens/s | Steady decode tokens/s |
|---|---:|---:|---:|
| Original native text, warm | 1024 | 5.31 | Not measured |
| Optimized text, greedy | 1024 | 92.10 | 96.84 |
| Optimized multimodal, greedy | 512 | 83.67 | 92.21 |
| Optimized multimodal, sampled | 512 | 59.87 | 74.37 |
| Corrected PIC build, text greedy | 1024 | 90.95 | 95.71 |
| Corrected PIC build, multimodal greedy | 512 | 83.65 | 92.10 |

The text results represent about a 17x overall improvement on the tested workload, not a general image-generation speedup. Native multimodal trace: 64 tokens in 17.565 s, with native forward calls around 0.18–0.23 s/token; not a matched 512-token comparison. Historical warm full regression workflow: 102.31 s with text-only acceleration, 53.77 s with text and multimodal acceleration, 54.62 s after the PIC rebuild. Token counts, seeds, and loading histories differ, so these are practical workflow observations rather than controlled comparisons. Peak sampled VRAM in the historical multimodal regression was 21.20 GiB.

See [integration results](docs/INTEGRATION_VALIDATION.md) and [build validation](docs/BUILD_VALIDATION.md) for details.

## Tested configuration

| Component | Version / configuration |
|---|---|
| GPU | Intel Arc Pro B70 |
| OS | Linux |
| ComfyUI | 0.42.0, commit `8f04fd6ecf4a548937971e840af2a985529fab76` |
| PyTorch | 2.14.1+xpu |
| Python | 3.12.15 |
| Triton | 3.8.0 |
| comfy-kitchen | 0.2.39 |
| Intel oneAPI compiler | 2026.2 |
| Text encoder | Huihui Qwen3-VL-4B Instruct FP8 scaled, CLIP type `krea2` |

The accelerator targets `Qwen3VL_4BConfig`, scalar-scaled E4M3FN FP8 weights, BF16 activations, single-token decode, batch size 1 and compatible tuple KV storage. Models and dependencies must be acquired separately under their own terms.

## How it works

- Direct FP8 GEMV avoids repeatedly materializing dequantized projection weights.
- Fused SYCL QKV and gate/up projections, a BF16 SiLU lookup table, and Triton projection/RoPE kernels reduce memory and launch overhead.
- Fixed-storage KV and XPU graph replay reduce host submission overhead. Native generation, sampling and EOS handling remain in use; greedy argmax can be captured.
- Multimodal prefill stays native. Graph capture begins only when the supported decode contract is established.

## Installation (Linux, experimental)

Prerequisites: an already functioning ComfyUI XPU environment matching the configuration above, the Intel oneAPI C++/SYCL compiler, a host C++ compiler, Python development headers available for your ComfyUI interpreter, and `readelf` (binutils). Use the existing PyTorch/XPU environment; do not blindly upgrade dependencies or source oneAPI `setvars.sh` into it. The latter caused a runtime mismatch during development.

### 1. Build and smoke-test outside ComfyUI

Activate the same Python environment used by ComfyUI before running these commands. Adjust the compiler path for your installation.

```sh
git clone https://github.com/kaiserwilli/ComfyUI-XPU-FP8-Decode.git
cd ComfyUI-XPU-FP8-Decode
python scripts/build_native.py --sycl-home /opt/intel/oneapi/compiler/2026.2
python tests/smoke_native.py
```

The build derives its include/library paths from the running Python interpreter and local compiler. It uses PIC device linking and rejects a new library set if any library contains ELF TEXTREL. All six smoke comparisons passed on B70. No model download or ComfyUI import is needed for these tests. The build writes three local libraries to `lib/` only after all checks pass.

### 2. Install while ComfyUI is stopped

From the built checkout, replace `/path/to/ComfyUI` with your installation directory. Ensure no other copy of this extension is installed. Stop ComfyUI when idle before copying.

```sh
COMFYUI=/path/to/ComfyUI
DEST="$COMFYUI/custom_nodes/qwen_xpu_fp8_test"
test ! -e "$DEST" && mkdir "$DEST" && cp -a . "$DEST/"
```

Restart ComfyUI using its usual launch command. The packaged `mode` is `off`: the original generator runs until explicitly enabled. Import still patches the native generation method even in off mode. This is not a separate graph node and should not be combined with another extension that patches the same generator or registers the same native operators.

### 3. Enable, validate, or disable

Run from the installed extension directory:

```sh
printf 'mm_on\n' > mode        # accelerated supported text and multimodal decoding
printf 'mm_validate\n' > mode  # slow reference validation
printf 'off\n' > mode          # original native generation
```

Use one command at a time. Other modes: `on` (text only), `validate` (text reference comparison), `mm_trace` (native multimodal trace, accelerated text). Modes apply on the next uncached generation without restarting. To remove the extension entirely, stop ComfyUI while idle, move its directory outside `custom_nodes`, and restart.

## Numerical and reliability limits

- In 127 multimodal decode comparisons, graph replay matched the optimized eager reference exactly. Original native argmax matched 125/127 times on the same incoming token sequence (relative L2 0.552–3.472%). These are not independent autoregressive sequences or a comprehensive quality benchmark. Identical native output is not guaranteed.
- The corrected PIC libraries passed standalone smoke tests, full integration, cancellation recovery, and idle model unload/reload on the tested configuration. Two warm greedy runs reproduced the earlier optimized token digest. The first cold run had a different digest.
- Do not use with concurrent generation, speculative decoding, model modifications, active-generation low-VRAM/partial offload or untested cache layouts. The text path has less extensive residency/cache checks than the multimodal path. Other GPUs, software versions, multiple images, long contexts and prolonged stress testing remain unverified.
- This is an experimental source release, not a general XPU compatibility guarantee. Keep a working backup.

## Benchmark and report results

Load [the multimodal benchmark workflow](workflows/multimodal_benchmark.json), select a non-private image and the tested checkpoint, and leave MTP off. Compare `off` and `mm_on` with the same inputs, greedy sampling and 512 maximum tokens. Warm both paths and force actual execution without changing the benchmark inputs (clear/disable execution caching as appropriate). Cached results are not measurements. Record generated token count, generation time, end-to-end workflow time, steady decode throughput when logged, and GPU memory. For deeper comparison, use `mm_validate`.

When reporting a problem or another successful configuration, include the GPU, OS, exact ComfyUI commit, Python/PyTorch/Triton/oneAPI versions, model format, mode, relevant sanitized logs, and reproduction steps. Do not include private prompts, images, credentials or model weights.

## License and attribution

[GPL-3.0-only](LICENSE), to the extent of the rights held by contributors. Adapted upstream code retains its applicable terms. See [NOTICE.md](NOTICE.md) for upstream attribution and dependency licenses. No model weights, dependency packages, or prebuilt native libraries are distributed. Independent project; not an official ComfyUI or Intel release.
