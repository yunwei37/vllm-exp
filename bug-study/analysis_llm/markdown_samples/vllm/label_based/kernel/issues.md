# kernel - issues

**Total Issues**: 1
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 1

### Label Distribution

- good first issue: 1 issues
- feature request: 1 issues
- kernel: 1 issues

---

## Issue #N/A: [Feature]: Vectorize `scaled_int8_quant`

**Link**: https://github.com/vllm-project/vllm/issues/18866
**State**: closed
**Created**: 2025-05-28T23:47:33+00:00
**Closed**: 2025-06-15T11:08:02+00:00
**Comments**: 3
**Labels**: good first issue, feature request, kernel

### Description

### 🚀 The feature, motivation and pitch

Similar to the recent discoveries in https://github.com/vllm-project/vllm/pull/18844, vectorizing our quantization methods can have a huge impact on e2e performance.

Currently we only use `vectorization.h` in `csrc/quantization/fp8/common.cuh` and `csrc/quantization/fused_kernels/layernorm_utils.cuh`, so we should expand this to more implementations like `csrc/quantization/compressed_tensors/int8_quant_kernels.cu` for faster INT8 activation quantization.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

