# low-priority - issues

**Total Issues**: 2
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 1

### Label Distribution

- low-priority: 2 issues
- lora: 1 issues

---

## Issue #N/A: [Feature] Customized mapping for LoRA weight names

**Link**: https://github.com/sgl-project/sglang/issues/6608
**State**: open
**Created**: 2025-05-26T04:08:39+00:00
**Comments**: 0
**Labels**: low-priority, lora

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

The current LoRA impl in SGL maps LoRA weight to modules by (layer index, op_type) tuple, where op_type operation looks like `qkv_proj`, `o_proj`, `gate_up`, etc. This works fine for most standard cases, however, there are some limitations:
1. For models where there are more than one attention stacks (e.g., VLM), there could be multiple modules with the same (layer index, op_type), e.g., one from vision tower, the other from the language model. Currently SGL cannot handle such cases correctly and would usually fail during loading due to incorrect mapping.
2. Users cannot enable/disable application of LoRA at module-level, e.g., if user only wants to apply LoRA at language model but not vision (common); or when user

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Google TPU Support

**Link**: https://github.com/sgl-project/sglang/issues/919
**State**: closed
**Created**: 2024-08-04T20:55:26+00:00
**Closed**: 2024-09-22T14:21:39+00:00
**Comments**: 4
**Labels**: low-priority

### Description

### Motivation

TPUs potentially provide a cheap serving option

### Related resources

vLLM does support TPUs: https://docs.vllm.ai/en/latest/getting_started/tpu-installation.html#installation-with-tpu

---

