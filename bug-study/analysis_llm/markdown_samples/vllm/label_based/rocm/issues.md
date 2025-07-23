# rocm - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 4
- Closed Issues: 26

### Label Distribution

- rocm: 30 issues
- bug: 21 issues
- stale: 9 issues
- feature request: 4 issues
- usage: 1 issues
- performance: 1 issues

---

## Issue #N/A: [Bug]: Error when running Llama-4-Maverick-17B-128E-Instruct-FP8 on mi300x

**Link**: https://github.com/vllm-project/vllm/issues/16474
**State**: closed
**Created**: 2025-04-11T10:27:28+00:00
**Closed**: 2025-04-23T12:07:16+00:00
**Comments**: 8
**Labels**: bug, rocm

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 04-11 09:49:33 [__init__.py:239] Automatically detected platform rocm.
Collecting environment information...
PyTorch version: 2.7.0a0+git295f2ed
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.3.42133-1b9c17779

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.1 24491 1e0fda770a2079fbd71e4b70974d74f62fd3af10)
CMake version: version 3.31.6
Libc version: glibc-2.35

Python version: 3.12.10 (main, Apr  9 2025, 08:55:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-128-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Instinct MI300X (gfx942:sramecc+:xnack-)
Nvidia driver version: Could not collect
cuDNN v

[... truncated for brevity ...]

---

## Issue #N/A: [Bug][Rocm] Garbage Response from vLLM When Using Tensor Parallelism on AMD CPX/NPS4 Partitioned GPUs

**Link**: https://github.com/vllm-project/vllm/issues/20125
**State**: open
**Created**: 2025-06-26T13:18:47+00:00
**Comments**: 0
**Labels**: bug, rocm

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
I have attached output file.
```

[vllm_collect_env_output.txt](https://github.com/user-attachments/files/20926332/vllm_collect_env_output.txt)

</details>


### 🐛 Describe the bug

**Steps to reproduce:**
We referred to doc:  [Steps to Run a vLLM Workload on AMD partition](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/gpu-partitioning/mi300x/run-vllm.html).
- [ ] **Do CPS/NPS4 Partition**
`sudo amd-smi set --memory-partition NPS4`


- [ ] **Launch container**
`docker run -it --network=host --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device /dev/kfd --device /dev/dri rocm/vllm:latest /bin/bash`


 - [ ] **Set Env**
```
export HF_TOKEN=<token>
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```


- [ ] `vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 8`


- [ ] **Query the model**
```
curl http://local

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: ROCM with AWQ

**Link**: https://github.com/vllm-project/vllm/issues/11249
**State**: closed
**Created**: 2024-12-17T03:37:54+00:00
**Closed**: 2024-12-18T02:57:04+00:00
**Comments**: 8
**Labels**: bug, rocm

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.6.0.dev20241113+rocm6.2
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.2.41133-dd7f95766

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.2.0 24292 26466ce804ac523b398608f17388eb6d605a3f09)
CMake version: version 3.26.4
Libc version: glibc-2.31

Python version: 3.9.19 (main, May  6 2024, 19:43:03)  [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.8.0-50-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: Radeon RX 7900 XTX (gfx1100)
Nvidia driver version: Could not collect
cuDNN version: Could not collect
HIP runtime version: 6.2.41133
MIOpen runtime version: 3.2.0
Is XNNPACK 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: AiterFlashAttentionImpl.__init__() got multiple values for argument 'use_irope' for llama4 model

**Link**: https://github.com/vllm-project/vllm/issues/19867
**State**: closed
**Created**: 2025-06-19T14:36:59+00:00
**Closed**: 2025-07-14T17:39:11+00:00
**Comments**: 1
**Labels**: bug, rocm

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
Your output of `python collect_env.py` here
```

</details>


### 🐛 Describe the bug

We hit an exception on running llama4 models with latest code on ROCm V1:

```
(VllmWorker rank=2 pid=267) ERROR 06-19 01:00:39 [multiproc_executor.py:488] TypeError: AiterFlashAttentionImpl.__init__() got multiple values for argument 'use_irope'
```
Current work-around:
To turn off AITER_MHA, with VLLM_ROCM_USE_AITER_MHA=0


Proposal:

- [ ] Fix the bug (the team is working on it)
- [ ] Add a end-to-end test for one of the small llama4 models
- [ ] 

The motivation for adding an end to end test for a small version of llama4 models, is that we have seen issues of breaking llama4 models in the past because of lacking such tests.


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug][ROCm] The embedding layer does not support long inputs

**Link**: https://github.com/vllm-project/vllm/issues/6807
**State**: closed
**Created**: 2024-07-25T23:58:46+00:00
**Closed**: 2024-07-27T03:16:15+00:00
**Comments**: 1
**Labels**: bug, rocm

### Description

### Your current environment

8xMI300x machine using the docker image built with `Dockerfile.rocm`.

Versions of relevant libraries:
[pip3] mypy==1.7.0
[pip3] mypy-extensions==1.0.0
[pip3] numpy==1.26.4
[pip3] optree==0.9.1
[pip3] pytorch-triton-rocm==3.0.0+21eae954ef
[pip3] torch==2.5.0.dev20240710+rocm6.1
[pip3] torchaudio==2.4.0.dev20240710+rocm6.1
[pip3] torchvision==0.20.0.dev20240710+rocm6.1
[pip3] transformers==4.43.2
[pip3] triton==3.0.0
[conda] No relevant packages
ROCM Version: 6.1.40093-bd86f1708
Neuron SDK Version: N/A
vLLM Version: 0.5.3.post1
vLLM Build Flags:
CUDA Archs: Not Set; ROCm: Disabled; Neuron: Disabled
GPU Topology:
Could not collect

### 🐛 Describe the bug

```python
import torch
import torch.nn as nn

with torch.inference_mode():
    NUM_TOKENS = 128 * 1024
    HIDDEN_SIZE = 16 * 1024
    VOCAB_SIZE = 128 * 1024
    DTYPE = torch.bfloat16
    x = torch.randint(VOCAB_SIZE, (NUM_TOKENS,), dtype=torch.int64, device="cuda")
    embed

[... truncated for brevity ...]

---

## Issue #N/A: Error when Running HIPGraph with TP 8

**Link**: https://github.com/vllm-project/vllm/issues/2217
**State**: closed
**Created**: 2023-12-20T09:27:29+00:00
**Closed**: 2024-08-30T09:44:57+00:00
**Comments**: 6
**Labels**: rocm

### Description

Command that was run:
```
python benchmark_throughput.py -tp 8 --model meta-llama_Llama-2-70b-chat-hf --dataset ShareGPT_V3_unfiltered_cleaned_split.json 
```
Error Logs:
```
...
ensor_parallel_size=8, quantization=None, enforce_eager=False, seed=0)                                                                                                                                      
(RayWorkerVllm pid=343834) WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions. xFormers was built for:                                                                                               
(RayWorkerVllm pid=343834)     PyTorch 2.1.1+cu121 with CUDA 1201 (you have 2.1.1+rocm5.6)                                                                                                                   
(RayWorkerVllm pid=343834)     Python  3.10.13 (you have 3.10.13)                                                                                                                                 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Critical distributed executor bug

**Link**: https://github.com/vllm-project/vllm/issues/7791
**State**: closed
**Created**: 2024-08-22T18:18:36+00:00
**Closed**: 2025-05-29T21:37:16+00:00
**Comments**: 11
**Labels**: bug, rocm

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.5.0.dev20240726+rocm6.1
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.1.40091-a8dbc0c19

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.1.2 24193 669db884972e769450470020c06a6f132a8a065b)
CMake version: version 3.26.4
Libc version: glibc-2.31

Python version: 3.9.19 (main, May  6 2024, 19:43:03)  [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-72-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Instinct MI300X (gfx942:sramecc+:xnack-)
Nvidia driver version: Could not collect
cuDNN version: Could not collect
HIP runtime version: 6.1.40093
MIOpen runtime version: 3.1.0

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: RuntimeError: HIP Error on vLLM ROCm Image in Kubernetes Cluster with AMD GPUs

**Link**: https://github.com/vllm-project/vllm/issues/10855
**State**: closed
**Created**: 2024-12-03T09:04:58+00:00
**Closed**: 2025-06-17T02:14:30+00:00
**Comments**: 6
**Labels**: bug, rocm, stale

### Description

### Your current environment

<details>
Hi,

I am attempting to run the [vLLM ROCm image](https://hub.docker.com/r/rocm/vllm-ci/tags) on a Kubernetes cluster. The AMD GPU is successfully detected, and the AMD GPU operator is installed and functioning correctly. However, when initializing the vLLM engine, I encounter the following error:

```
RuntimeError: HIP error: invalid argument
HIP kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing AMD_SERIALIZE_KERNEL=3
Compile with `TORCH_USE_HIP_DSA` to enable device-side assertions.
```

###Steps to Reproduce:
- Run the rocm/vllm-ci Docker image on a Kubernetes cluster.
- Ensure the AMD GPU operator is installed and that GPUs are detected.
- Attempt to initialize the vLLM server.

###Observations:
- AMD GPU detection confirms the presence of an AMD Instinct MI210 (gfx90a).
- The environment appears to meet the prerequisites:
     

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vllm 0.9 image gives me gibberish

**Link**: https://github.com/vllm-project/vllm/issues/19052
**State**: open
**Created**: 2025-06-03T05:03:20+00:00
**Comments**: 2
**Labels**: bug, rocm

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
Your output of `python collect_env.py` here
==============================
        System Info
==============================
OS                           : Ubuntu 22.04.5 LTS (x86_64)
GCC version                  : (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version                : 19.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.4.1 25184 c87081df219c42dc27c5b6d86c0525bc7d01f727)
CMake version                : version 3.31.6
Libc version                 : glibc-2.35

==============================
       PyTorch Info
==============================
PyTorch version              : 2.7.0+gitf717b2a
Is debug build               : False
CUDA used to build PyTorch   : N/A
ROCM used to build PyTorch   : 6.4.43483-a187df25c

==============================
      Python Environment
==============================
Python version               : 3.12.10 (main, Apr

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: openai whisper model response is not accurate on AMD-based(MI300x) systems.

**Link**: https://github.com/vllm-project/vllm/issues/20069
**State**: open
**Created**: 2025-06-25T10:17:20+00:00
**Comments**: 7
**Labels**: bug, rocm

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
DEBUG 06-25 10:10:02 [__init__.py:28] No plugins for group vllm.platform_plugins found.
DEBUG 06-25 10:10:02 [__init__.py:34] Checking if TPU platform is available.
DEBUG 06-25 10:10:02 [__init__.py:44] TPU platform is not available because: No module named 'libtpu'
DEBUG 06-25 10:10:02 [__init__.py:51] Checking if CUDA platform is available.
DEBUG 06-25 10:10:02 [__init__.py:75] Exception happens when checking CUDA platform: NVML Shared Library Not Found
DEBUG 06-25 10:10:02 [__init__.py:92] CUDA platform is not available because: NVML Shared Library Not Found
DEBUG 06-25 10:10:02 [__init__.py:99] Checking if ROCm platform is available.
DEBUG 06-25 10:10:02 [__init__.py:106] Confirmed ROCm platform is available.
DEBUG 06-25 10:10:02 [__init__.py:120] Checking if HPU platform is available.
DEBUG 06-25 10:10:02 [__init__.py:127] HPU platform is not available because haban

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Llama 3.1 405 B FP8 model is not support by vLLM (v0.5.3.post1) on AMD GPU

**Link**: https://github.com/vllm-project/vllm/issues/7031
**State**: closed
**Created**: 2024-08-01T11:55:49+00:00
**Closed**: 2024-08-06T09:24:40+00:00
**Comments**: 2
**Labels**: bug, rocm

### Description

### Your current environment

vLLM version: 0.5.3.post1 (For ROCm)
Model:  meta-llama/Meta-Llama-3.1-405B-Instruct-FP8
AMD MI300x GPU


### 🐛 Describe the bug

![Screenshot 2024-07-31 131408](https://github.com/user-attachments/assets/5b0771b4-8b4b-4303-9eff-df8b425aaf60)

vLLM is throwing value error when loading meta-llama/Meta-Llama-3.1-405B-Instruct-FP8 on AMD MI300x GPU.
Value Erorr: fbgemm_fp8 quantization is currently not supported in ROCm. Refer screenshot for reference. 

---

## Issue #N/A: [Bug]: error: triton_flash_attention.py

**Link**: https://github.com/vllm-project/vllm/issues/5696
**State**: closed
**Created**: 2024-06-20T01:01:18+00:00
**Closed**: 2024-09-27T13:58:03+00:00
**Comments**: 4
**Labels**: bug, rocm

### Description

### Your current environment

Collecting environment information...
/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/cuda/__init__.py:611: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
PyTorch version: 2.1.1+git011de5c
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.0.32830-d62f6a171

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.0.0 23483 7208e8d15fbf218deb74483ea8c549c67ca4985e)
CMake version: version 3.29.5
Libc version: glibc-2.31

Python version: 3.9.18 (main, Sep 11 2023, 13:41:44)  [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-75-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 10.1.243
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Radeon PRO W6800NoGCNArchNameOnOldPyTorch
Nvidia driver version: Could not collec

[... truncated for brevity ...]

---

## Issue #N/A: [Bug][ROCm]: Process killed during tensor-parallel inference

**Link**: https://github.com/vllm-project/vllm/issues/4019
**State**: closed
**Created**: 2024-04-11T20:58:32+00:00
**Closed**: 2024-09-04T14:07:11+00:00
**Comments**: 3
**Labels**: bug, rocm

### Description

### Your current environment

I'm using a docker container built from `Dockerfile.rocm` with MI250x GPUs.

```text
PyTorch version: 2.1.1+git011de5c
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.0.32830-d62f6a171

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.0.0 23483 7208e8d15fbf218deb74483ea8c549c67ca4985e)
CMake version: version 3.29.1
Libc version: glibc-2.31

Python version: 3.9.18 (main, Sep 11 2023, 13:41:44)  [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.19.0-45-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 10.1.243
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Instinct MI250X/MI250NoGCNArchNameOnOldPyTorch
Nvidia driver version: Could not collect
cuDNN version: Could not collect
HIP runtime version: 6.0.32830
MIOpen runtime version: 3.0.0
Is 

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Benchmarking Issues: Low Success Rate and Tensor Parallel Size Constraints on 8x AMD MI300x GPUs

**Link**: https://github.com/vllm-project/vllm/issues/9070
**State**: closed
**Created**: 2024-10-04T10:53:10+00:00
**Closed**: 2025-05-20T02:11:42+00:00
**Comments**: 6
**Labels**: rocm, usage, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
Collecting environment information...
WARNING 10-04 10:39:09 rocm.py:13] `fork` method is not supported by ROCm. VLLM_WORKER_MULTIPROC_METHOD is overridden to `spawn` instead.
WARNING 10-04 10:39:09 _custom_ops.py:18] Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
WARNING 10-04 10:39:09 _custom_ops.py:18] Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
PyTorch version: 2.4.1+rocm6.1
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.1.40091-a8dbc0c19

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.1.0 24103 7db7f5e49612030319346f900c08f474b1f9023a)
CMake version: version 3.26.4
Libc version: glibc-2.35

Python version: 3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0] (64-bit 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Sparsity in LLMs

**Link**: https://github.com/vllm-project/vllm/issues/11563
**State**: closed
**Created**: 2024-12-27T09:18:52+00:00
**Closed**: 2025-04-27T02:11:28+00:00
**Comments**: 3
**Labels**: feature request, rocm, stale

### Description

### 🚀 The feature, motivation and pitch

Great action for support 2:4 sparsity (with quantization) in vllm for nvidia Ampere+ architectures!

I wonder 2:4 sparsity support in AMD MI300/MI300X+ Accelerators, Will this be the roadmap of the future?

Will you provide unstructured sparsity support in the future? Flash-LLM (https://github.com/AlibabaResearch/flash-llm) currently provides 1.3x end-to-end acceleration of 70% unstructured sparse LLM on NVIDIA GPUs.

Thanks~

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: Running Llama-3.1-405B on AMD MI300X with FP8 quantization fails

**Link**: https://github.com/vllm-project/vllm/issues/8538
**State**: closed
**Created**: 2024-09-17T16:26:51+00:00
**Closed**: 2024-09-20T13:23:30+00:00
**Comments**: 6
**Labels**: bug, rocm

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
WARNING 09-17 16:08:06 rocm.py:14] `fork` method is not supported by ROCm. VLLM_WORKER_MULTIPROC_METHOD is overridden to `spawn` instead.
PyTorch version: 2.5.0.dev20240726+rocm6.1
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.1.40091-a8dbc0c19

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.1.2 24193 669db884972e769450470020c06a6f132a8a065b)
CMake version: version 3.26.4
Libc version: glibc-2.31

Python version: 3.9.19 (main, May  6 2024, 19:43:03)  [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-121-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Instinct MI300X (gfx942:sramecc+:xnack

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] [ROCm]: RuntimeError: Calling `torch.linalg.cholesky` on a CUDA tensor requires compiling PyTorch with MAGMA. Please use PyTorch built with MAGMA support.

**Link**: https://github.com/vllm-project/vllm/issues/14914
**State**: closed
**Created**: 2025-03-17T02:49:48+00:00
**Closed**: 2025-07-18T02:28:29+00:00
**Comments**: 3
**Labels**: bug, rocm, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here

Your output of `python collect_env.py` here
Collecting environment information...                                                                                            
PyTorch version: 2.5.1+cu124                                                                                                     
Is debug build: False                                                                                                            
CUDA used to build PyTorch: 12.4                                                                                                 
ROCM used to build PyTorch: N/A                                                                                                  
                                                                                                                                 
OS: Ubuntu 24.04.1 LTS (x

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Llama 3.1 405 B  FP16 model failed to load on AMD GPU 

**Link**: https://github.com/vllm-project/vllm/issues/7032
**State**: closed
**Created**: 2024-08-01T12:02:09+00:00
**Closed**: 2024-08-02T13:53:56+00:00
**Comments**: 4
**Labels**: bug, rocm

### Description

### Your current environment

vLLM version: 0.5.3.post1 (For ROCm)
Model: meta-llama/Meta-Llama-3.1-405B-Instruct
8 x AMD MI300x GPU

### 🐛 Describe the bug

```
services:
  vllm-serving:
    container_name: vllm-serving
    image: vllm-rocm:v0.5.3.post1
    environment:
      - LLM_MODEL=$LLM_MODEL
      - HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
    command: >
      sh -c "
      python3 -m vllm.entrypoints.openai.api_server \
      --model $LLM_MODEL --dtype float16 \
      --tensor-parallel-size 8
      "
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
    volumes:
      - /mnt/model/:/models/
    shm_size: 16G
    ports:
      - 8000:8000
```

[rank0]: huggingface_hub.utils._errors.HfHubHTTPError: 416 Client Error: Requested Range Not Satisfiable for url: https://cdn-lfs-us- .....
ERROR 08-01 07:45:19 multiproc_worker_utils.py:120] Worker VllmWorkerProcess pid 76 died, exit code: -15
INFO 08-01 07:45:19 multiproc_worker_utils

[... truncated for brevity ...]

---

## Issue #N/A: Limited Request Handling for AMD Instinct MI300 X GPUs with Tensor Parallelism > 1

**Link**: https://github.com/vllm-project/vllm/issues/2988
**State**: closed
**Created**: 2024-02-22T12:34:02+00:00
**Closed**: 2024-07-15T06:34:20+00:00
**Comments**: 9
**Labels**: rocm

### Description

Reproducing steps:

1. Clone the vllm repo and switch to [tag v0.3.1](https://github.com/vllm-project/vllm/tree/v0.3.1)
2. Build the Dockerfile.rocm dockerfile with instructions from [Option 3: Build from source with docker -Installation with ROCm](https://docs.vllm.ai/en/latest/getting_started/amd-installation.html#build-from-source-docker-rocm)

    build command:
    ```sh
    docker build  -f Dockerfile.rocm -t vllm-rocm .
    ```

3. The vLLM serving command used:
    ```sh
    python3 -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-70b-chat-hf --dtype float16 --tensor-parallel-size 8
    ```
4.  Used Apache Bench for testing with 256 concurrent requests

The error below:
```sh
INFO 02-21 10:31:34 metrics.py:161] Avg prompt throughput: 352.5 tokens/s, Avg generation throughput: 55.2 tokens/s, Running: 67 reqs, Swapped: 0 reqs, Pending: 130 reqs, GPU KV cache usage: 0.2%, CPU KV cache usage: 0.0%
Memory access fault by GPU node-2 (Agent handle: 0

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: V1 engine runs slower than V0 on the MI300X

**Link**: https://github.com/vllm-project/vllm/issues/19692
**State**: open
**Created**: 2025-06-16T14:20:16+00:00
**Comments**: 1
**Labels**: performance, rocm

### Description

### Proposal to improve performance

I run a Llama3 8B inference benchmark on the MI300X with both V0 and V1 engines. I noticed that V1 is quite slower at decoding compared to V0. Normally, V1 is much faster than V0 on Nvidia. 

One thing I noticed though is that, with V1, it doesn't print the Triton autotune output of the flash attn kernel, could be related to the attn implementation with V1.

### Report of performance regression

![Image](https://github.com/user-attachments/assets/84bdebe9-ff9d-486e-b60c-79c38588aa3e)

### Misc discussion on performance

_No response_

### Your current environment (if you think it is necessary)

```text
==============================
       PyTorch Info  
==============================
PyTorch version              : 2.8.0.dev20250615+rocm6.4
Is debug build               : False
CUDA used to build PyTorch   : N/A
ROCM used to build PyTorch   : 6.4.43482-0f2d60242

==============================
       CUDA / GPU Info
==============================
Is 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: FlashInfer + Gemma 2 for AMD GPU

**Link**: https://github.com/vllm-project/vllm/issues/6218
**State**: closed
**Created**: 2024-07-08T17:17:30+00:00
**Closed**: 2024-12-14T02:04:29+00:00
**Comments**: 9
**Labels**: feature request, rocm, stale

### Description

This could be question rather than a feature request.

flashinfer is not supported for AMD GPUs and it's not currently planned until a [later version](https://github.com/flashinfer-ai/flashinfer/issues/19), 

Is there a way to run Gemma 2 models on AMD (I'm getting `ValueError: Please use Flashinfer backend for models withlogits_soft_cap (i.e., Gemma-2). Otherwise, the output might be wrong. Set Flashinfer backend by export VLLM_ATTENTION_BACKEND=FLASHINFER.` even though I set the env var. I wanted to give it a try and remove the [validation](https://github.com/vllm-project/vllm/blob/v0.5.1/vllm/attention/selector.py#L137-L147) but it fails with None type because of [import error](https://github.com/vllm-project/vllm/blob/v0.5.1/vllm/attention/backends/flashinfer.py#L4-L11)) ?

Or is there an alternative that can be used?

### Alternatives

_No response_

### Additional context

_No response_

---

## Issue #N/A: [Bug][CI/Build]: Missing attribute 'nvmlDeviceGetHandleByIndex' in AMD tests

**Link**: https://github.com/vllm-project/vllm/issues/6059
**State**: closed
**Created**: 2024-07-02T07:54:44+00:00
**Closed**: 2024-07-03T03:12:23+00:00
**Comments**: 1
**Labels**: bug, rocm

### Description

### Your current environment

AMD CI

### 🐛 Describe the bug

Most AMD CI runs are failing. Example: https://buildkite.com/vllm/ci-aws/builds/3706#0190716d-09a9-49d5-a9d3-f61dc45ae12c

---

## Issue #N/A: [Feature]: gfx1100安装的flash_attn分支不支持，切了howiejay/navi_support分支后，flash_attn_varlen_func() got an unexpected keyword argument 'window_size' 如何解决

**Link**: https://github.com/vllm-project/vllm/issues/10014
**State**: closed
**Created**: 2024-11-05T01:45:53+00:00
**Closed**: 2025-03-11T02:03:41+00:00
**Comments**: 2
**Labels**: feature request, rocm, stale

### Description

### 🚀 The feature, motivation and pitch

gfx1100安装的flash_attn分支不支持，切了howiejay/navi_support分支后，flash_attn_varlen_func() got an unexpected keyword argument 'window_size' 如何解决

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: For RDNA3 (navi31; gfx1100) VLLM_USE_TRITON_FLASH_ATTN=0 currently must be forced

**Link**: https://github.com/vllm-project/vllm/issues/4514
**State**: closed
**Created**: 2024-05-01T05:44:22+00:00
**Closed**: 2024-12-08T10:04:32+00:00
**Comments**: 17
**Labels**: bug, rocm, stale

### Description

### Your current environment

```text
Collecting environment information...
/opt/conda/envs/py_3.9/lib/python3.9/site-packages/torch/cuda/__init__.py:611: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
PyTorch version: 2.1.1+git011de5c
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.0.32830-d62f6a171

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.0.0 23483 7208e8d15fbf218deb74483ea8c549c67ca4985e)
CMake version: version 3.29.2
Libc version: glibc-2.31

Python version: 3.9.18 (main, Sep 11 2023, 13:41:44)  [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.5.0-28-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 10.1.243
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Radeon PRO W7900NoGCNArchNameOnOldPyTorch
Nvidia driver version: Could no

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Assertion `idx < size()' failed (vllm on AMD)

**Link**: https://github.com/vllm-project/vllm/issues/4268
**State**: closed
**Created**: 2024-04-22T14:00:53+00:00
**Closed**: 2024-11-29T02:06:35+00:00
**Comments**: 2
**Labels**: bug, rocm, stale

### Description

### Your current environment

```text
vllm-0.4.1+rocm573-py3.9-linux-x86_64.egg
compiled from source on an AMD cluster
conda/22.9.0 virtual env.
run with MI250
```


### 🐛 Describe the bug

Hi. I've been able to install correctly the latest version of vllm on an AMD cluster.

Yet, just after loading the model I have a low-level bug from llvm:

/root/.triton/llvm/llvm-5e5a22ca-centos-x64/include/llvm/ADT/SmallVector.h:298:
const T& llvm::SmallVectorTemplateCommon<T, <template-parameter-1-2>
>::operator[](llvm::SmallVectorTemplateCommon<T, <template-parameter-1-2>
>::size_type) const [with T = long int; <template-parameter-1-2> = void;
llvm::SmallVectorTemplateCommon<T, <template-parameter-1-2> >::const_reference =
const long int&; llvm::SmallVectorTemplateCommon<T, <template-parameter-1-2>
>::size_type = long unsigned int]: Assertion `idx < size()' failed.

I haven't been able to get a detailed traceback. Within vllm the bug comes just after loading the hidden st

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: ROCm 6.2 support & FP8 Support

**Link**: https://github.com/vllm-project/vllm/issues/7469
**State**: closed
**Created**: 2024-08-13T11:43:49+00:00
**Closed**: 2025-03-12T06:17:15+00:00
**Comments**: 5
**Labels**: feature request, rocm

### Description

### 🚀 The feature, motivation and pitch


Last week AMD announced  rocm 6.2 (https://rocm.docs.amd.com/en/latest/about/release-notes.html) also announcing expanded support for VLLM & FP8. 

Actually I was able to run it following the guides ( Rocm branch ) and executing it like this:

python -m vllm.entrypoints.openai.api_server  --model /work/work2/Meta-Llama-3.1-70B-Instruct --tensor-parallel-size 1 --port 8010 --host 0.0.0.0 --quantization fp8 --quantized-weights-path /work/work2/Meta-Llama-3.1-70B-Instruct-fp8/llama.safetensors --kv-cache-dtype fp8_e4m3 --quantization-param-path /work/work2/Meta-Llama-3.1-70B-Instruct-fp8-scales/kv_cache_scales.json

But the performance is like 3/4 times slower than using the model withouth quantitization. 

I don't know if ROCm 6.2 can solve thsi issues ... actually the performance we got with mi300x(half) is similar than running the a A100(FP8) on our tests.




### Alternatives

_No response_

### Additional context

_No response_

---

## Issue #N/A: [Bug]: AMD GPU docker image build No matching distribution found for torch==2.6.0.dev20241113+rocm6.2

**Link**: https://github.com/vllm-project/vllm/issues/12178
**State**: closed
**Created**: 2025-01-17T23:36:10+00:00
**Closed**: 2025-03-12T05:50:14+00:00
**Comments**: 2
**Labels**: bug, rocm

### Description

### Your current environment

Archlinux 13th Gen Intel(R) Core(TM) i9-13900HX environment to build the docker image

### Model Input Dumps

_No response_

### 🐛 Describe the bug

Trying to build the AMD GPU docker image:
```
git checkout v0.6.6.post1
DOCKER_BUILDKIT=1 docker build -f Dockerfile.rocm -t substratusai/vllm-rocm:v0.6.6.post1 .
```

Results in following error:

```
1.147 Looking in indexes: https://pypi.org/simple, https://download.pytorch.org/whl/nightly/rocm6.2
1.717 ERROR: Could not find a version that satisfies the requirement torch==2.6.0.dev20241113+rocm6.2 (from versions: 1.7.1, 1.8.0, 1.8.1, 1.9.0, 1.9.1, 1.10.0, 1.10.1, 1.10.2, 1.11.0, 1.12.0, 1.12.1, 1.13.0, 1.13.1, 2.0.0, 2.0.1, 2.1.0, 2.1.1, 2.1.2, 2.2.0, 2.2.1, 2.2.2, 2.3.0, 2.3.1, 2.4.0, 2.4.1, 2.5.0, 2.5.1, 2.6.0.dev20241119+rocm6.2, 2.6.0.dev20241120+rocm6.2, 2.6.0.dev20241121+rocm6.2, 2.6.0.dev20241122+rocm6.2)
2.135 ERROR: No matching distribution found for torch==2.6.0.dev20241113+rocm6.2
------
Dockerfil

[... truncated for brevity ...]

---

## Issue #N/A: Incorrect completions with tensor parallel size of 8 on MI300X GPUs

**Link**: https://github.com/vllm-project/vllm/issues/2817
**State**: closed
**Created**: 2024-02-08T17:32:07+00:00
**Closed**: 2024-09-04T14:03:45+00:00
**Comments**: 3
**Labels**: rocm

### Description

I'm encountering an issue where vLLM fails to generate complete or sensible responses when the tensor parallel size is set to 8 on MI300X GPUs.  Completions work as expected with tensor parallel sizes of 1 and 4.

**Expected behavior:**

vLLM should generate a correct and meaningful completion for the given prompt, similar to its behavior with tensor parallel sizes of 1 and 4.

**Actual behavior:**

vLLM provides an incomplete or nonsensical response, often similar to the following:

```json
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": " <"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 96,
        "total_tokens": 99,
        "completion_tokens": 3
    }
```

**System information:**

* **OS:** `Linux test 5.15.0-94-generic #104-Ubuntu SMP Tue Jan 9 15:25:40 UTC 2024 x86_64 x86_64 x86_64 GNU/Linux`
* **

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Invalid Device Ordinal on ROCm

**Link**: https://github.com/vllm-project/vllm/issues/4131
**State**: closed
**Created**: 2024-04-16T23:58:58+00:00
**Closed**: 2024-09-04T14:08:36+00:00
**Comments**: 13
**Labels**: bug, rocm

### Description

### Your current environment

```text
PyTorch version: 2.4.0.dev20240415+rocm6.0
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.0.32830-d62f6a171

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.29.2
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-102-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Instinct MI300X (gfx942:sramecc+:xnack-)
Nvidia driver version: Could not collect
cuDNN version: Could not collect
HIP runtime version: 6.0.32830
MIOpen runtime version: 3.0.0
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:                     

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Tensor Parallel > 1 causes desc_act=True GPTQ models to give bad output on ROCm

**Link**: https://github.com/vllm-project/vllm/issues/7374
**State**: closed
**Created**: 2024-08-09T18:01:13+00:00
**Closed**: 2024-12-15T02:10:33+00:00
**Comments**: 2
**Labels**: bug, rocm, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.5.0.dev20240710+rocm6.1
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.1.40091-a8dbc0c19

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.1.2 24193 669db884972e769450470020c06a6f132a8a065b)
CMake version: version 3.30.1
Libc version: glibc-2.31

Python version: 3.9.19 (main, May  6 2024, 19:43:03)  [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.5.0-44-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Instinct MI100 (gfx908:sramecc+:xnack-)
Nvidia driver version: Could not collect
cuDNN version: Could not collect
HIP runtime version: 6.

[... truncated for brevity ...]

---

