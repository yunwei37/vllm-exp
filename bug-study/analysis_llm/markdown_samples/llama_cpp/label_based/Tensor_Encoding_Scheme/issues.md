# Tensor_Encoding_Scheme - issues

**Total Issues**: 1
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 1

### Label Distribution

- enhancement: 1 issues
- stale: 1 issues
- Tensor Encoding Scheme: 1 issues

---

## Issue #N/A: Support BitNet b1.58 ternary models

**Link**: https://github.com/ggml-org/llama.cpp/issues/5761
**State**: closed
**Created**: 2024-02-28T09:41:38+00:00
**Closed**: 2024-09-18T01:07:17+00:00
**Comments**: 90
**Labels**: enhancement, stale, Tensor Encoding Scheme

### Description

New paper just dropped on Arxiv describing a way to train models in 1.58 bits (with ternary values: 1,0,-1). Paper shows performance increases from equivalently-sized fp16 models, and perplexity nearly equal to fp16 models. Authors state that their test model is built on LLaMA architecture and can be easily adapted to llama.cpp.

[Edited to add: Further reading into it by fellow Redditors shows that we can't use this to quantize existing models trained to fp16. They'd have to be trained in this ternary mode from the start. But I think it would still be something that we should implement, because models of that flavor will be coming soon.]

This is all over Reddit /LocalLLaMA right now:

https://www.reddit.com/r/LocalLLaMA/comments/1b21bbx/this_is_pretty_revolutionary_for_the_local_llm/

I think, if my napkin math is right, it would let us run something like 120B models in 24 GB VRAM, or 30B in... 8 GB?

Please implement @ggerganov and friends!

https://arxiv.org/abs/2402.17

[... truncated for brevity ...]

---

