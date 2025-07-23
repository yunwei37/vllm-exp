# microsoft - issues

**Total Issues**: 1
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 0

### Label Distribution

- help wanted: 1 issues
- high priority: 1 issues
- microsoft: 1 issues

---

## Issue #N/A: [Feature] Phi-4-MM support

**Link**: https://github.com/sgl-project/sglang/issues/6544
**State**: open
**Created**: 2025-05-23T04:17:59+00:00
**Comments**: 0
**Labels**: help wanted, high priority, microsoft

### Description

### Update

Currently we have added text & vision support. 

Repeated MMMU benchmark runs range between 53.6 - 55.5, consistent with the the benchmark reported in the original paper (55).

**Known limitations:** (See *Execution Plan* before for full list):

1. Audio capabilities: currently we do not support audio at all. 
2. ~~LoRA / Image quality: Phi4MM depends on LoRA for full image capability, but there is some compatibility issues with the native SGL LORA solution. We are working on solving it by refactoring / generalizing SGL LoRA capabilities.~~ Fixed with #6585, #6734, #6861)
3. Token: Phi4MM supports two types of image token conventions (`<|image1|>` and `<|endoftext10|>`), currently we only support  the latter. If you use the default chat template, it will automatically pick up the supported one.

### Motivation

Supporting the Phi4 Multimodal model (https://[huggingface.co/microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) in SGL

[... truncated for brevity ...]

---

