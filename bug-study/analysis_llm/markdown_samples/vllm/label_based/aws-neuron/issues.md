# aws-neuron - issues

**Total Issues**: 3
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 3

### Label Distribution

- aws-neuron: 3 issues
- bug: 2 issues
- stale: 2 issues

---

## Issue #N/A: [Bug]: vLLM with Neuron performance degrades dramatically if request concurrency is >= max_num_seqs

**Link**: https://github.com/vllm-project/vllm/issues/8007
**State**: closed
**Created**: 2024-08-29T18:44:52+00:00
**Closed**: 2024-12-29T02:05:15+00:00
**Comments**: 2
**Labels**: bug, aws-neuron, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
collecting environment information...
WARNING 08-29 18:36:46 _custom_ops.py:18] Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
/usr/local/lib/python3.10/dist-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
No module named 'vllm.commit_id'
  from vllm.version import __version__ as VLLM_VERSION
PyTorch version: 2.1.2+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.27.7
Libc version: glibc-2.35

Python version: 3.10.12 (main, Jul 29 2024, 16:56:48) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.2.0-1017-aws-x86_64-with-glibc2.35
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: AssertionError in neuron_model_runner.py assert len(block_table) == 1

**Link**: https://github.com/vllm-project/vllm/issues/4553
**State**: closed
**Created**: 2024-05-02T11:11:34+00:00
**Closed**: 2024-12-19T02:05:09+00:00
**Comments**: 4
**Labels**: bug, aws-neuron, stale

### Description

### Your current environment

PyTorch version: 2.1.2+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.29.2
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-1031-aws-x86_64-with-glibc2.35
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: No CUDA
Nvidia driver version: No CUDA
cuDNN version: No CUDA
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Address sizes:                   48 bits physical, 48 bits virtual
Byte Order:                      Little Endian
CPU(s):                          192
On-

[... truncated for brevity ...]

---

## Issue #N/A: [RFC] Initial Support for AWS Inferentia

**Link**: https://github.com/vllm-project/vllm/issues/1866
**State**: closed
**Created**: 2023-11-30T15:02:11+00:00
**Closed**: 2024-03-02T00:59:00+00:00
**Comments**: 9
**Labels**: aws-neuron

### Description

## Proposal

We propose to integrate transformers-neuronx to be the execution engine in vLLM for supporting LLM inference on Inferentia. This would require changes on both transformers-neuronx and vLLM.

### Changes to transformers-neuronx

1. Support batch size 1 prompt encoding, while share same cache space with max batch size decoding.
2. Support batch-dependent KV cache update. Each sequence will have a specified position_id to update cache.
3. Support virtual dynamic batching. This would enable multi-batch prompt encoding virtually agnostic to vLLM.

### Changes to vLLM

- [x] Make CUDA kernel compilation optional, so that when we are trying to perform LLM inference on inf2 instances we don’t necessarily compile the CUDA kernels. Meanwhile, we would still keep CUDA kernel compilation enabled by default. https://github.com/vllm-project/vllm/pull/2065
- [x] Add transformers-neuronx package as a (optional) thirdparty dependency of vllm.  Note that transformers-neuronx wo

[... truncated for brevity ...]

---

