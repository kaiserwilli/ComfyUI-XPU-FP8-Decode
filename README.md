# ComfyUI-XPU-FP8-Decode

**Experimental release candidate, validated on the specific Intel Arc Pro B70 configuration below.** Experimental acceleration of native ComfyUI Qwen3-VL-4B FP8 autoregressive decoding on Intel XPU. Uses existing model weights, native vision prefill and conditioning, direct FP8 GEMV, fused projections, and XPU graph replay. No external inference server or additional model checkpoint.

## Measured results from the original implementation

Intel Arc Pro B70, Linux, ComfyUI commit `8f04fd6ecf4a548937971e840af2a985529fab76` (0.42.0), PyTorch 2.14.1+xpu, Python 3.12.15, Triton 3.8.0, comfy-kitchen 0.2.39, oneAPI compiler 2026.2. Tested checkpoint: Huihui Qwen3-VL-4B Instruct FP8 scaled, CLIP type `krea2`.

| Test | Generated tokens | Overall tokens/s | Steady decode tokens/s |
|---|---:|---:|---:|
| Original native text, warm | 1024 | 5.31 | not measured |
| Optimized text, greedy | 1024 | 92.10 | 96.84 |
| Optimized multimodal, greedy | 512 | 83.67 | 92.21 |
| Optimized multimodal, sampled | 512 | 59.87 | 74.37 |

Original native multimodal trace: 64 generated tokens in 17.565 s (3.64 overall), with native forward time roughly 0.18–0.23 s per decode step. This is not a matched 512-token benchmark. Full warm regression workflow: 102.31 s text-only optimization versus 53.77 s text + multimodal optimization, with changed seeds/token counts. Observed peak VRAM 21.20 GiB in the latter run. These are historical measurements of the original working installation. The corrected PIC build subsequently achieved 92.103 steady tokens/s (83.650 overall) for warm 512-token greedy multimodal generation, 95.711 steady tokens/s for 1024-token text generation, and 54.62 s for the warm full regression workflow. The warm multimodal token digest matched the earlier optimized implementation. See `docs/INTEGRATION_VALIDATION.md`.

## Scope and limitations

- Only the Qwen3VL_4BConfig architecture and eligible XPU single-token decoding. The tested model uses scalar-scaled E4M3FN FP8 weights, BF16 activations, batch 1, and tuple KV storage. Other GPUs, models and software versions are unverified.
- Native prefill, vision encoding, DeepStack and diffusion conditioning are not replaced. Original text-only path has less extensive residency/cache checks than the multimodal path. Do not use with concurrent generation, speculative decoding, model modifications, forced offload or untested cache layouts.
- Graph/eager validation was exact on 127 multimodal decode positions. Original native logits differed: 125/127 matching greedy argmax on the same incoming token sequence, relative L2 0.552–3.472%. Identical native output is not guaranteed. Graph validation is not an independent autoregressive quality test.
- Cancellation and idle model unloading/reloading passed on the tested configuration. Active-generation low-VRAM/partial offload, multiple images, long contexts and broader hardware compatibility remain unverified. Do not combine with another extension that patches the same generation method or registers the same native operator namespaces.

## Build candidate (Linux)

Use a **standalone checkout outside `custom_nodes`** for the first build and smoke test. Use the exact Python environment used by ComfyUI and an existing compatible Intel oneAPI compiler. Do not install or upgrade packages merely to try this candidate. Do not source oneAPI `setvars.sh` into the ComfyUI environment; an earlier runtime conflict was observed.

```sh
python scripts/build_native.py --sycl-home /opt/intel/oneapi/compiler/2026.2
python tests/smoke_native.py
```

Replace the compiler path if needed. The build helper derives the PyTorch and Python include paths from the running interpreter, builds into a temporary directory and copies the three resulting libraries to `lib/`. The helper includes `-fPIC` during SYCL device linking and rejects all newly built libraries if any contain ELF TEXTREL. This flag eliminated TEXTREL from all three libraries on the tested Arc Pro B70 / PyTorch 2.14.1+xpu environment; all six native smoke comparisons passed. Both the earlier non-PIC rebuild and the corrected PIC rebuild passed full ComfyUI integration. The corrected libraries were byte-compared to the tested PIC candidate, verified free of TEXTREL, and confirmed mapped into the running process. Cancellation and idle model unloading/reloading passed. See `docs/INTEGRATION_VALIDATION.md`. Source-only distribution remains the intended first release. See `docs/BUILD_VALIDATION.md` and `docs/RELEASE_REVIEW.md`.

## Install only after clean-build and smoke validation

Copy the checkout (with locally built `lib/`) into ComfyUI's `custom_nodes`, with no other copy of this accelerator present. Restart ComfyUI when idle. The `mode` file is set to `off` in this candidate. Import patches the native generation method even in off mode; use only the tested ComfyUI version. A missing `mode` file also defaults to off.

From the extension directory:

```sh
printf 'mm_validate\n' > mode  # slow comparison against eager references
printf 'mm_on\n' > mode        # supported text and multimodal acceleration
printf 'off\n' > mode          # native execution
```

Other modes: `on` (text only), `validate` (text reference comparison), `mm_trace` (native multimodal trace, accelerated text). Mode changes apply on the next uncached generation without a restart. To fully remove the extension, stop ComfyUI when idle, move the extension outside `custom_nodes`, and restart.

## Benchmarking

Load `workflows/multimodal_benchmark.json`, choose a non-private image, select the tested checkpoint, leave MTP off, and use greedy 512-token generation. For a matched comparison, hold prompt, image, sampling settings and maximum token count fixed. Force actual node execution without altering inputs (e.g. disable/clear execution cache as supported by the tested version); do not count cached results. Warm both paths, record actual token counts, complete generation and workflow durations, and steady decode speed when logged. Compare `off` and `mm_on`, then validate with `mm_validate`. Record GPU memory throughout, not only after execution.

## Licensing

This project is distributed under the GNU General Public License, version 3 (GPL-3.0-only), to the extent of the rights held by its contributors. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md) for upstream attribution and third-party dependencies. Adapted upstream code retains its original applicable licensing terms. No model weights, dependency packages, or prebuilt native libraries are distributed. This is an independent experimental project, not an official ComfyUI or Intel release.
