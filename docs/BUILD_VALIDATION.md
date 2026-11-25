# Native build validation

2026-11-25 UTC — Linux, Intel Arc Pro B70, PyTorch 2.14.1+xpu. The source-only candidate was built with the existing Intel oneAPI 2026.2 compiler and its standalone smoke test was run in a separate Python process. All three libraries built successfully.

| Test | Relative L2 error | Finite |
|---|---:|---|
| Base GEMV WG128 | 0.00000000 | Yes |
| Base GEMV WG256 | 0.00000000 | Yes |
| Fused Q | 0.00000000 | Yes |
| Fused K | 0.00000000 | Yes |
| Fused V | 0.00000000 | Yes |
| LUT MLP | 0.00000000 | Yes |

All native smoke tests passed. The reported errors are rounded to eight decimal places. All three rebuilt libraries still report ELF TEXTREL. No compiled libraries are distributed. This is a single-environment native smoke test, not a full ComfyUI integration test or a guarantee of portability to other systems.

Pending: test rebuilt extension in ComfyUI, cancellation/offload recovery, matched-workload benchmarks, source provenance and license review, investigate text relocations before any binary distribution.

## PIC device-link follow-up (2026-11-25 UTC)

Rebuilt all three libraries with `--pic-device-link` (adds `-fPIC` to the SYCL device-link command). `readelf -d` no longer showed TEXTREL for any library. All six native smoke tests again returned relative L2=0.00000000 and finite=True. The ignored subgroup-size attribute warnings from host compilation remain distinct from the corrected ELF relocations. PIC device linking and strict rejection of TEXTREL are now mandatory in the build helper; the earlier optional CLI flags remain accepted for compatibility. Full ComfyUI integration with PIC-built libraries is not yet tested. The earlier non-PIC build passed integration; see `INTEGRATION_VALIDATION.md`.
