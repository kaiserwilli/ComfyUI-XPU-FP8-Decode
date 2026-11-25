# Private release review

- Runtime Python and SYCL source copied byte-for-byte from the working archive. All 22 entries in the original manifest matched, and all 10 original Python files parsed.
- Original `mode` changed from `mm_on` to `off` in the new candidate. README replaced; original machine-specific Ninja configurations, shared objects and old checksum files excluded. New build script, smoke test and sanitized benchmark workflow added. No kernel logic changes.
- Original binaries all contain DT_TEXTREL and absolute build paths. Rebuild locally; investigate whether device-link PIC removes the relocations. Do not assume the optional flag fixes this. Check new ELF dynamic sections with readelf.
- Source review: no obvious hardcoded secrets found in source files; this is a static review, not a comprehensive security or provenance audit. ComfyUI imports and modified attention/graph behavior should be reviewed against the pinned upstream revision for copyright/attribution requirements. Calibration derived from the tested checkpoint, not independently validated on other fine-tunes. Confirm rights and license before publication.
- Graph safety: multimodal route checks cache pointers and positions each step and initially verifies model residency; text path has weaker contract checks. Restrict release to tested stack, explicit opt-in, no concurrency/offload/MTP. Add conservative hardware, version, residency, cache and concurrency guards before general distribution.
- Build helper and native smoke tests passed on the known GPU. Next, validate the rebuilt extension in a separate or reversible ComfyUI installation. Confirm cancellation, offload, and repeatability. Compare matched workloads. Investigate DT_TEXTREL before binary distribution.
- No public release has been made. Rebuilt native libraries passed standalone smoke tests; full integration remains unverified.

## Follow-up status

The earlier non-PIC rebuild passed full integration. The PIC device-link experiment removed TEXTREL from all three shared libraries and passed all six smoke tests. The build helper now always applies `-fPIC` at device link and rejects TEXTREL. PIC integration, cancellation/offload tests, source provenance and licensing remain pending. Historical statements above describe the initial candidate; no public release yet.

## Technical validation complete

Final PIC integration and cancellation recovery passed on the tested B70 stack; the operator confirmed idle model unload/reload and correct resulting images. Broader compatibility, active-generation low-VRAM offload and extended stress testing remain outside the validated scope. The remaining initial-release gates are provenance/license review and public publication. No binary distribution is planned.
