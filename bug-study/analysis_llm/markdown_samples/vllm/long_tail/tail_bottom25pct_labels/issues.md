# tail_bottom25pct_labels - issues

**Total Issues**: 19
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 3
- Closed Issues: 16

### Label Distribution

- bug: 11 issues
- x86-cpu: 4 issues
- stale: 4 issues
- security: 3 issues
- aws-neuron: 3 issues
- feature request: 3 issues
- llama: 2 issues
- release-blocker: 2 issues
- good first issue: 2 issues
- usage: 1 issues

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

## Issue #N/A: [Bug]: Merge security updates for 0.9.0

**Link**: https://github.com/vllm-project/vllm/issues/17667
**State**: closed
**Created**: 2025-05-05T16:08:43+00:00
**Closed**: 2025-05-09T14:07:58+00:00
**Comments**: 1
**Labels**: security

### Description

This is a placeholder to ensure any pending security patches have been merged prior to release.

---

## Issue #N/A: [Bug]: clients can crash the openai server with invalid regex

**Link**: https://github.com/vllm-project/vllm/issues/17313
**State**: closed
**Created**: 2025-04-28T15:27:44+00:00
**Closed**: 2025-05-12T01:06:11+00:00
**Comments**: 2
**Labels**: bug, security

### Description

### Your current environment

```
root@3bea15cf4c9f:/# uv run --with vllm python collect_env.py
INFO 04-28 15:38:49 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
/usr/local/lib/python3.11/dist-packages/_distutils_hack/__init__.py:31: UserWarning: Setuptools is replacing distutils. Support for replacing an already imported distutils is deprecated. In the future, this condition will fail. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.11.10 (main, Sep  7 2024, 18:35:41) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.8.0-52-generic-x86_64-with-glibc2.35
Is CU

[... truncated for brevity ...]

---

## Issue #N/A: [Tracker] Merge security fixes for v0.8.5

**Link**: https://github.com/vllm-project/vllm/issues/17128
**State**: closed
**Created**: 2025-04-24T17:19:49+00:00
**Closed**: 2025-04-25T16:23:36+00:00
**Comments**: 1
**Labels**: bug, security

### Description

This issue is for tracking that pending security fixes are merged prior to releasing v0.8.5

- [x] GHSA-hj4w-hm2g-p6w5 - https://github.com/vllm-project/vllm/pull/17192
- [x] GHSA-9f8f-2vmf-885j - https://github.com/vllm-project/vllm/pull/17197

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

## Issue #N/A: [Bug]: Illegal memory access on llama4 maverick

**Link**: https://github.com/vllm-project/vllm/issues/19631
**State**: closed
**Created**: 2025-06-13T22:33:29+00:00
**Closed**: 2025-07-07T17:10:56+00:00
**Comments**: 9
**Labels**: bug, torch.compile, llama

### Description

### Your current environment

PyTorch 2.7.0, vLLM main branch built from source.

### 🐛 Describe the bug

Repro:
```py
vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 --tensor-parallel-size 8 --max-num-batched-tokens 40000 --max-model-len 8192 --max-num-seqs 128 --gpu-memory-utilization 0.8
```
gives a CUDA Illegal Memory Access, as well as some errors:
```
ERROR 06-13 15:32:09 [core.py:515] EngineCore failed to start.
ERROR 06-13 15:32:09 [core.py:515] Traceback (most recent call last):
ERROR 06-13 15:32:09 [core.py:515]   File "/home/rzou/dev/stable0/vllm-stable0/vllm/v1/engine/core.py", line 506, in run_engine_core
ERROR 06-13 15:32:09 [core.py:515]     engine_core = EngineCoreProc(*args, **kwargs)
ERROR 06-13 15:32:09 [core.py:515]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 06-13 15:32:09 [core.py:515]   File "/home/rzou/dev/stable0/vllm-stable0/vllm/v1/engine/core.py", line 390, in __init__
ERROR 06-13 15:32:09 [core.py:515]     super().__init__(vllm_conf

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vLLM does not serve text-only version of Llama4

**Link**: https://github.com/vllm-project/vllm/issues/18022
**State**: open
**Created**: 2025-05-12T20:23:48+00:00
**Comments**: 1
**Labels**: feature request, llama

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
Not related
```

</details>


### 🐛 Describe the bug

Hi all! 
I am trying to serve a text-only version of Llama 4 Scout (17B-16E) using vLLM. This model requires the Llama4ForCausalLM architecture. However, it seems that vLLM currently expects only the multimodal Llama 4.

Although the Llama4ForCausalLM class is implemented in vllm/model_executor/models/llama4.py, it is not registered in the _TEXT_GENERATION_MODELS dictionary in vllm/model_executor/models/registry.py. After manually adding an entry for Llama4ForCausalLM, I was able to serve the model successfully.

This looks like an oversight or a missing feature, and might be considered a bug.

For the reference, the text-only version of Llama4 was loaded and saved with AutoModelForCausalLM with the model config updated accordingly. 
```
model_config = AutoConfig.from_pretrained(config["model"]["path"], trust_remote_c

[... truncated for brevity ...]

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

## Issue #N/A: [Bug]: Assertion error when serving "deepseek-ai/DeepSeek-V2-Lite" with PP in 0.9.2

**Link**: https://github.com/vllm-project/vllm/issues/20647
**State**: closed
**Created**: 2025-07-08T22:48:28+00:00
**Closed**: 2025-07-10T03:34:42+00:00
**Comments**: 0
**Labels**: bug, v1, deepseek

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
==============================
        System Info
==============================
OS                           : Ubuntu 22.04.5 LTS (x86_64)
GCC version                  : (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version                : Could not collect
CMake version                : Could not collect
Libc version                 : glibc-2.35

==============================
       PyTorch Info
==============================
PyTorch version              : 2.7.0+cu128
Is debug build               : False
CUDA used to build PyTorch   : 12.8
ROCM used to build PyTorch   : N/A

==============================
      Python Environment
==============================
Python version               : 3.11.11 | packaged by conda-forge | (main, Mar  3 2025, 20:43:55) [GCC 13.3.0] (64-bit runtime)
Python platform              : Linux-6.5.0-1024-aws-x86_64-with-glibc2.35

=========

[... truncated for brevity ...]

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

## Issue #N/A: [Feature]: OpenAI Response API

**Link**: https://github.com/vllm-project/vllm/issues/15237
**State**: closed
**Created**: 2025-03-20T17:13:04+00:00
**Closed**: 2025-03-20T22:52:05+00:00
**Comments**: 5
**Labels**: good first issue, feature request, frontend

### Description

### 🚀 The feature, motivation and pitch

I come across this https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses&example=chain-of-thought#streaming wrt to their new Response API, so we probably also want to add support in vLLM.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug][PERF]: Qwen2.5 performance degradation 0.8.4 -> 0.8.5

**Link**: https://github.com/vllm-project/vllm/issues/18619
**State**: open
**Created**: 2025-05-23T14:48:43+00:00
**Comments**: 34
**Labels**: bug, qwen

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code> for v0.8.4</summary>

```text
INFO 05-23 14:36:10 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
==============================
        System Info
==============================
OS                           : Ubuntu 22.04.5 LTS (x86_64)
GCC version                  : (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version                : Could not collect
CMake version                : version 4.0.2
Libc version                 : glibc-2.35

==============================
       PyTorch Info
==============================
PyTorch version              : 2.6.0+cu124
Is debug build               : False
CUDA used to build PyTorch   : 12.4
ROCM used to build PyTorch   : N/A

==============================
      Python Environment
==============================
Python version               : 3.10.12 (main, Feb  4 2025, 14:57:36) [GCC 11.4.0] (64-bit 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Bug in LRUEvictor: priority_queue and free_table desynchronization cause error

**Link**: https://github.com/vllm-project/vllm/issues/16825
**State**: open
**Created**: 2025-04-18T08:19:48+00:00
**Comments**: 6
**Labels**: bug, v0

### Description

### Your current environment

vllm 0.7.3

### 🐛 Describe the bug

### Your current environment
vllm 0.7.3
### 🐛 Describe the bug
We encountered a bug in the LRUEvictor implementation when running VLLM (version 0.7.3) with the --preemption-mode swap flag.
The issue arises due to desynchronization between self.priority_queue and self.free_table in the remove method.
<img width="1561" alt="Image" src="https://github.com/user-attachments/assets/4d048b91-f914-43b9-89e2-5b6daf0c2012" />
Add logging to evictor.py and prefix_caching_block.py to track block additions and removals.
<img width="1259" alt="Image" src="https://github.com/user-attachments/assets/e2b876a9-bff6-4004-b591-636d77eaa64d" />
<img width="589" alt="Image" src="https://github.com/user-attachments/assets/1a72b5d7-867c-49d9-a0af-cc3229a3ed47" />
<img width="644" alt="Image" src="https://github.com/user-attachments/assets/996e42c3-5899-4d33-8532-9193eb91c3f0" />
<img width="761" alt="Image" src="https://github.com/user-attachme

[... truncated for brevity ...]

---

