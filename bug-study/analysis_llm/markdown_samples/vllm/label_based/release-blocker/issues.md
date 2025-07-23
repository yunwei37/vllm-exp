# release-blocker - issues

**Total Issues**: 2
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 2

### Label Distribution

- release-blocker: 2 issues
- performance: 1 issues
- bug: 1 issues

---

## Issue #N/A: Performance Regression between v0.4.0 and v0.4.1

**Link**: https://github.com/vllm-project/vllm/issues/4210
**State**: closed
**Created**: 2024-04-19T17:13:42+00:00
**Closed**: 2024-04-23T20:12:42+00:00
**Comments**: 2
**Labels**: performance, release-blocker

### Description

### Anything you want to discuss about vllm.

#3550 seems to reduce throughput of vLLM

Before: Throughput: 20.13 requests/s, 10308.29 tokens/s
After: Throughput: 17.67 requests/s, 9048.03 tokens/s

(reported by @esmeetu and @youkaichao)

---

## Issue #N/A: [Bug]: OpenAI API Server always reports 0 tokens/s

**Link**: https://github.com/vllm-project/vllm/issues/4209
**State**: closed
**Created**: 2024-04-19T15:49:06+00:00
**Closed**: 2024-04-20T03:48:02+00:00
**Comments**: 3
**Labels**: bug, release-blocker

### Description

### Your current environment

```text
Collecting environment information...
PyTorch version: 2.2.1+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A
 
OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.29.0
Libc version: glibc-2.35
 
Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-97-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.3.103
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA RTX A6000
GPU 1: NVIDIA RTX A6000
GPU 2: NVIDIA RTX A6000
GPU 3: NVIDIA RTX A6000
GPU 4: NVIDIA RTX A6000
GPU 5: NVIDIA RTX A6000
GPU 6: NVIDIA RTX A6000
GPU 7: NVIDIA RTX A6000
 
Nvidia driver version: 545.23.08
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True
 
CPU:
Architecture: x86_64


[... truncated for brevity ...]

---

