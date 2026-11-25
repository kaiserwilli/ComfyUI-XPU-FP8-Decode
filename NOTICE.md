# Licensing and attribution

ComfyUI-XPU-FP8-Decode is an independent experimental extension. The project's contributions are made available under GNU GPL version 3 (GPL-3.0-only), to the extent of rights held by the contributors. See [LICENSE](LICENSE). Upstream contributions retain their original applicable license and copyright notices; no exclusive ownership of third-party code or uncopyrightable material is asserted.

## Adapted upstream code

**ComfyUI — respective upstream contributors.** The attention and MLP integration in `decoder.py` adapts the upstream native implementation, including projection/normalization structure, forward interfaces, and cache behavior, from [`comfy/text_encoders/llama.py`](https://github.com/Comfy-Org/ComfyUI/blob/8f04fd6ecf4a548937971e840af2a985529fab76/comfy/text_encoders/llama.py). Modified during November 2026 for Intel XPU fixed-storage graph replay, fused projections, RoPE and decode validation. `engine.py`, `multimodal.py` and the graph integration are coupled to upstream native generation and quantization interfaces. Upstream source and license: [ComfyUI](https://github.com/Comfy-Org/ComfyUI/tree/8f04fd6ecf4a548937971e840af2a985529fab76) and [GPL v3](https://github.com/Comfy-Org/ComfyUI/blob/8f04fd6ecf4a548937971e840af2a985529fab76/LICENSE). Upstream copyright remains with the original rights holders.

## External dependencies (not bundled)

| Dependency | Upstream license information |
|---|---|
| ComfyUI | [GPL-3.0](https://github.com/Comfy-Org/ComfyUI/blob/8f04fd6ecf4a548937971e840af2a985529fab76/LICENSE) |
| PyTorch | [BSD-style and third-party notices](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| Triton | [MIT and applicable third-party notices](https://github.com/triton-lang/triton/blob/main/LICENSE) |
| comfy-kitchen | [Apache-2.0](https://github.com/Comfy-Org/comfy-kitchen/blob/main/LICENSE) |
| Intel LLVM/SYCL | [Apache-2.0 with LLVM exceptions and applicable component notices](https://github.com/intel/llvm/blob/sycl/LICENSE.TXT) |

The project uses these libraries/toolchains through their interfaces and local build inputs. Their licenses do not change as a result of this project. Refer to the notices for the actual installed versions and Intel oneAPI distribution; the open-source LLVM/SYCL license does not establish the license of every oneAPI component. This source-only repository does not redistribute compiler/runtime packages, compiled libraries, or model checkpoints. Any independently acquired model is subject to its own license and terms. The benchmark workflow only identifies a checkpoint; it does not distribute it.

The implementation was developed iteratively with AI-assisted coding, testing and human-directed engineering. Attribution is based on inspected source and identified upstream adaptations, not an assertion that all generated portions are independently copyrightable. If additional third-party material is identified, preserve its applicable notices and resolve its distribution rights before publication.
