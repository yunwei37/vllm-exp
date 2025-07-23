# sgl-kernel - issues

**Total Issues**: 2
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 2
- Closed Issues: 0

### Label Distribution

- sgl-kernel: 2 issues
- MLLM: 1 issues
- good first issue: 1 issues

---

## Issue #N/A: [Perf] improve the hash kernel for mm

**Link**: https://github.com/sgl-project/sglang/issues/8054
**State**: open
**Created**: 2025-07-15T09:08:36+00:00
**Comments**: 3
**Labels**: MLLM, sgl-kernel

### Description

The current `gpu_tensor_hash` implementated in #5974  has following drawbacks:
1. `add` itself is not a very decent reduction method
2. will perform a torch tensor reduction, which is not very performant for large tensors

## TODO

1. Rewrite a performant and robust tensor hash function
2. Test the performance, consistency and correctness of the hash function against real data


## Reference

You can reference [here](https://github.com/sgl-project/sglang/pull/5974#issuecomment-3017284280) for inspirations


---

## Issue #N/A: [Feature] Support PDL on norm in sgl-kernel

**Link**: https://github.com/sgl-project/sglang/issues/5946
**State**: open
**Created**: 2025-05-01T07:41:57+00:00
**Comments**: 4
**Labels**: good first issue, sgl-kernel

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

In previous versions, we updated flashinfer. Flashinfer 0.2.5 supports norm's PDL, but currently, norm's PDL is disabled by default. We would like to modify the code to enable it.

### Related resources

We need change code at `sgl-kernel/python/sgl_kernel`, those who have enable_pdl parameter.

For example:
```python
def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    r"""Root mean square normalization.

    ``out[i] = (input[i] / RMS(input)) * weight[i]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor

[... truncated for brevity ...]

---

