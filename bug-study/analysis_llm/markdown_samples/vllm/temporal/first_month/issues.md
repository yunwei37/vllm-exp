# first_month - issues

**Total Issues**: 9
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 9

### Label Distribution

- performance: 2 issues
- stale: 1 issues

---

## Issue #N/A: Improve Weight Loading

**Link**: https://github.com/vllm-project/vllm/issues/48
**State**: closed
**Created**: 2023-04-22T04:16:22+00:00
**Closed**: 2023-05-03T07:32:05+00:00
**Comments**: 0

### Description

Just use Huggingface's weights. Don't do another copy!

---

## Issue #N/A: Frontend Improvements

**Link**: https://github.com/vllm-project/vllm/issues/47
**State**: closed
**Created**: 2023-04-22T03:57:50+00:00
**Closed**: 2023-05-24T04:39:52+00:00
**Comments**: 3

### Description

1. Current implementation of the FastAPI+asyncio+ray combination seems slow
2. Merge Hao’s throughput profiling code.
3. Make the frontend looks like OpenAI’s API.


---

## Issue #N/A: Debug the optimal upper-bound performance for swapping (0-cost swapping).

**Link**: https://github.com/vllm-project/vllm/issues/46
**State**: closed
**Created**: 2023-04-22T03:57:07+00:00
**Closed**: 2024-11-30T02:03:37+00:00
**Comments**: 4
**Labels**: performance, stale

### Description

Rerun the experiment comparing 0-cost swapping and recomputation. Recomputation should not be faster in any case. If recomputation is consistently faster, we should debug into this.

---

## Issue #N/A: Turn shareGPT data into a standard benchmark

**Link**: https://github.com/vllm-project/vllm/issues/45
**State**: closed
**Created**: 2023-04-22T03:52:50+00:00
**Closed**: 2023-06-15T02:55:39+00:00
**Comments**: 0

### Description

1. Extract out the lengths of the conversation rounds, and maybe have that data directly available from github.
2. The current L-shape evaluation with binary search for throughput is hard to run and not scalable. We should find an easier way to benchmark the performance.

---

## Issue #N/A: Fix the rushed out multi-query kernel

**Link**: https://github.com/vllm-project/vllm/issues/44
**State**: closed
**Created**: 2023-04-22T03:49:18+00:00
**Closed**: 2024-03-08T10:19:19+00:00
**Comments**: 2

### Description

1. Fix the correctness issue in the current FlashAttention-copy-based kernel. Make sure we call the FlashAttention kernel correctly. Evaluate the performance of this kernel.
2. Reduce the memory usage of the current kernel by limiting the buffer size and calling the kernel multiple times.

---

## Issue #N/A: Add support for Stable-LM and OpenAssistant

**Link**: https://github.com/vllm-project/vllm/issues/43
**State**: closed
**Created**: 2023-04-22T03:48:25+00:00
**Closed**: 2023-04-28T07:32:12+00:00
**Comments**: 0

### Description

The two models are popularly used. As we support LLaMA, it'll not be difficult to support these models.

---

## Issue #N/A: Modify the current PyTorch model to C++

**Link**: https://github.com/vllm-project/vllm/issues/42
**State**: closed
**Created**: 2023-04-22T03:36:03+00:00
**Closed**: 2024-09-20T20:59:21+00:00
**Comments**: 5
**Labels**: performance

### Description

Expected gain: For 13B models, we should see a 20%-30% latency gain on a single GPU and 2-3x on 4 GPUs. For smaller models, the gain should be even higher.

Having a single iteration's computation being completely C++ should be enough for high performance. In this way, we can keep most complicated scheduling logics in Python, including weight loading.

Potential sources of overheads:
1. Python v.s. C++.
2. PyTorch (even in C++) v.s. FasterTransformer.

How to implement a C++ version:
1. (Fake C++) Torch compiler (torch.jit).
2. Libtorch, C++ version of PyTorch (easier to implement and extend, but can only solve overhead 1).
3. Prune out the useful single model code from FasterTransformer to CacheFlow. This solves both overheads but is harder to implement.


---

## Issue #N/A: Add an option to disable Ray when using a single GPU

**Link**: https://github.com/vllm-project/vllm/issues/23
**State**: closed
**Created**: 2023-04-02T07:32:38+00:00
**Closed**: 2023-04-30T07:42:19+00:00
**Comments**: 0

### Description

When working with a single GPU, Ray is not useful. Therefore, it would be beneficial to have an option to disable Ray in such scenarios.

---

## Issue #N/A: Tensor Parallel profiling result

**Link**: https://github.com/vllm-project/vllm/issues/22
**State**: closed
**Created**: 2023-04-02T06:50:04+00:00
**Closed**: 2023-06-16T02:38:15+00:00
**Comments**: 0

### Description

Will update the profiling results in this PR.

## BS=8, input_len=32, output_len=128

```
OPT-13B
TP 1: 3.5404738585154214 seconds
TP 2: 4.742188215255737 seconds
TP 4: 4.907034238179524 seconds

OPT-30B
TP 1: OOM
TP 2: 5.9848620891571045 seconds
TP 4: 5.943212985992432 seconds
```

---

