# x86-cpu - issues

**Total Issues**: 4
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 4

### Label Distribution

- x86-cpu: 4 issues
- bug: 2 issues
- stale: 2 issues
- usage: 1 issues
- RFC: 1 issues
- unstale: 1 issues

---

## Issue #N/A: [Bug]:  benchmark_throughput gets TypeError: XFormersMetadata.__init__() got an unexpected keyword argument 'is_prompt' wit CPU 

**Link**: https://github.com/vllm-project/vllm/issues/6225
**State**: closed
**Created**: 2024-07-08T21:58:11+00:00
**Closed**: 2025-03-14T02:02:55+00:00
**Comments**: 18
**Labels**: bug, x86-cpu, stale

### Description

### Your current environment

```text
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.0
Libc version: glibc-2.35

Python version: 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-91-generic-x86_64-with-glibc2.35

...

[pip3] numpy==1.26.4
[pip3] nvidia-nccl-cu12==2.20.5
[pip3] torch==2.3.0
[pip3] torchvision==0.18.0
[pip3] transformers==4.42.3
[pip3] triton==2.3.0
[conda] numpy                     1.26.4                   pypi_0    pypi
[conda] nvidia-nccl-cu12          2.20.5                   pypi_0    pypi
[conda] torch                     2.3.0                    pypi_0    pypi
[conda] torchvision               0.18.0                   pypi_0    pypi
[conda] transformers              4.42.3       

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Recommended setting for running vLLM for CPU 

**Link**: https://github.com/vllm-project/vllm/issues/5682
**State**: closed
**Created**: 2024-06-19T09:01:53+00:00
**Closed**: 2025-01-09T02:14:23+00:00
**Comments**: 4
**Labels**: usage, x86-cpu, stale

### Description

### How would you like to use vllm

What are the recommended settings for running vLLM on a CPU to achieve high performance? For instance, if I have a dual-socket server with 96 cores per socket, how many cores (--cpuset-cpus) should be allocated to run multiple replicas of vLLM?

---

## Issue #N/A: [Bug]: Runtime Error: GET was unable to find an engine to execute this computation for LLaVa-NEXT

**Link**: https://github.com/vllm-project/vllm/issues/5465
**State**: closed
**Created**: 2024-06-12T17:48:41+00:00
**Closed**: 2024-06-14T01:34:58+00:00
**Comments**: 4
**Labels**: bug, x86-cpu

### Description

### Your current environment

```text
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 18.04.6 LTS (x86_64)
GCC version: (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
Clang version: Could not collect
CMake version: version 3.29.5
Libc version: glibc-2.27

Python version: 3.10.12 (main, Jul 19 2023, 10:44:52) [GCC 7.5.0] (64-bit runtime)
Python platform: Linux-4.15.0-213-generic-x86_64-with-glibc2.27
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
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              12
On-line CPU(s) list: 0-11
Thread(s) per core:  1
Core(s) per socket:  12
Socket(s):           1
NUMA node(s

[... truncated for brevity ...]

---

## Issue #N/A: [RFC] Initial Support for CPUs

**Link**: https://github.com/vllm-project/vllm/issues/3654
**State**: closed
**Created**: 2024-03-27T07:45:25+00:00
**Closed**: 2025-01-14T16:19:23+00:00
**Comments**: 11
**Labels**: RFC, x86-cpu, unstale

### Description

## Progress

- [ ] Integrate CPU executor to support the basic model inference (BF16/FP32) without TP. 
  - #3634 
  - #3824 
  - #4113 
  - #4971 
  - #5452 
  - #5446 
- [ ] Support FP16 model inference.
- [x] Support TP inference for multiple CPU sockets inside the same node. 
  - #6008 
  - #6125 
- [ ] Support model and KV cache quantization.
  - #5492 
  - #7257 

## Features

The CPU executor plans to support the following features:

- Basic models of vLLM with FP16/BF16/FP32, except MoE models
- Tensor-parallel model inference based on Ray
- AWQ quantization, 8-bit KVCache Quantization
- Others

## Design

Our target is seamless porting vLLM to CPU devices and sharing most of vLLM core components (e.g., **schedular**, **cache management**, **model definitions**, **Megatron-style model partitioning**, ...). 

The CPU executor will depend on Pytorch CPU and leverage optimized kernels and features from [intel-extension-for-pytorch](https://github.com/

[... truncated for brevity ...]

---

