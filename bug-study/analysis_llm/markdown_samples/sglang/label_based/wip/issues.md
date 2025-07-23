# wip - issues

**Total Issues**: 2
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 2

### Label Distribution

- wip: 2 issues
- good first issue: 1 issues
- high priority: 1 issues
- performance: 1 issues

---

## Issue #N/A: [Feature] optimize moe_align_block_size_kernel

**Link**: https://github.com/sgl-project/sglang/issues/2732
**State**: closed
**Created**: 2025-01-05T05:56:21+00:00
**Closed**: 2025-03-25T04:11:57+00:00
**Comments**: 7
**Labels**: good first issue, high priority, wip, performance

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

The original version performs poorly and needs optimization. I suggest rewriting a new implementation.

https://github.com/sgl-project/sglang/blob/main/sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu

### Related resources

_No response_

---

## Issue #N/A: [Feature] Make vLLM optional in model code

**Link**: https://github.com/sgl-project/sglang/issues/1673
**State**: closed
**Created**: 2024-10-15T06:49:05+00:00
**Closed**: 2025-03-03T23:17:23+00:00
**Comments**: 3
**Labels**: wip

### Description

### UPDATE(11/23/2024)

Currently, @james-p-xu  is removing rope, @yizhang2077  is removing distributed, @HandH1998 is removing weight loader. Optimistically, we can remove these dependencies by the end of the month and make quant optional (try import). cc @merrymercy @Ying1123 

### Motivation

This is a tracker of removing vLLM dependencies in general model code (not considering quantization). This is our current  import from vLLM, and we want to remove all them.

```python
from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
   ParallelLMHead,
   VocabParallelEmbedding,
)
```

### Tracker

- [x] Remove `CacheConfig`: https://github.com/sgl-project/sglang/pull/1658
- [x] Remove RoPE: https://github.com/flashinfer-ai/flashinfer/issues/530
- [x] Remove `get_tensor_model_parallel_world_size`
- [x] Remove `Para

[... truncated for brevity ...]

---

