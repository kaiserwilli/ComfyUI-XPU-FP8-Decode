# Rebuilt-library integration validation

Tested 2026-11-25 UTC on Intel Arc Pro B70, Linux, ComfyUI 0.42.0 (8f04fd6ecf4a548937971e840af2a985529fab76), PyTorch 2.14.1+xpu, oneAPI compiler 2026.2, Qwen3-VL-4B FP8 scaled. Source candidate built locally; all six native smoke tests reported finite outputs and relative L2=0.00000000. All three native libraries were subsequently confirmed mapped into the ComfyUI process.

| Test | Tokens | Generation time (s) | Overall tokens/s | Steady decode tokens/s |
|---|---:|---:|---:|---:|
| Multimodal greedy | 512 | 6.147 | 83.299 | 91.937 |
| Multimodal sampled | 512 | 8.387 | 61.046 | 75.921 |
| Text greedy standalone | 1024 | 11.559 | 88.589 | 95.399 |
| Text greedy within regression | 1024 | 11.171 | 91.666 | 96.834 |

Multimodal prefill: 1055 tokens, approximately 0.216–0.217 s warm. Full regression containing native text generation, diffusion and six multimodal calls: 161.28 s with loading, 53.47 s warm. Earlier optimized implementation: 53.77 s warm; different generated token counts and loading histories, not a controlled same-workload speed comparison. Final sampled idle VRAM 19633 MiB; peak VRAM was not measured during this test.

Numerical evidence: 128-token mm_validate with 127 replays reproduced the earlier optimized token SHA-256 digest `69dd2a5f04bdd387bef0059f69c4a7f6f9781b9f8919ceb8d9f5e61ee14e975b`. No graph logit mismatches or nonfinite values were reported in the filtered journal. The two native-reference argmax disagreements were reproduced at cache indices 1127 and 1131 (relative L2 0.01260277 and 0.00793625 respectively); the reference follows the same incoming tokens, so this does not establish identical native autoregressive sequences. The 512-token greedy benchmark also reproduced digest `0d94945735fc4e4a10411535f71209773a4e6a64954522d97c21b7ee15cac495`. No rejected multimodal projection calls were reported; text prefill rejects expected multi-token calls. An unrelated legacy external-inference workflow reported a connection failure; native accelerated regression completed.

**Status:** source rebuild and actual integration passed on the tested configuration. ELF DT_TEXTREL warning persists in all three libraries; source-only distribution pending license/provenance review. Cancellation, forced offload/reload, broad hardware/model coverage and extended repeated-run testing remain outstanding. Visual quality of the rebuilt full-workflow output was not independently inspected. This is an experimental release candidate, not a general compatibility claim.
