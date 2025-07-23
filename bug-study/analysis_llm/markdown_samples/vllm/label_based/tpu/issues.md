# tpu - issues

**Total Issues**: 18
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 4
- Closed Issues: 14

### Label Distribution

- tpu: 18 issues
- bug: 7 issues
- feature request: 6 issues
- stale: 5 issues
- ray: 4 issues
- RFC: 2 issues
- usage: 1 issues
- installation: 1 issues

---

## Issue #N/A: [RFC]: How to handle the compilation of PyTorch/XLA in vLLM

**Link**: https://github.com/vllm-project/vllm/issues/16282
**State**: closed
**Created**: 2025-04-08T19:37:11+00:00
**Closed**: 2025-04-16T01:29:59+00:00
**Comments**: 4
**Labels**: RFC, tpu

### Description

### Motivation.

vLLM currently utilizes PyTorch/XLA to provide TPU backend support. However, PyTorch/XLA differs significantly from native PyTorch in terms of usage. PyTorch/XLA is a compilation only framework, it doesn't have a real eager mode. In particular, for LLM serving services, recompilation should be avoided once the server is running. 
When compiling, it's important to consider which code might create PyTorch operations (e.g., tensor.copy(), tensor[:index], torch.ones(...)) and when graph capture and compilation is triggered (e.g., xm.mark_step(), xla_tensor.cpu(), if xla_tensor:, torch.compile(backend="openxla")). Due to the complexity of PyTorch/XLA, this document will only provide basic rules to simplify vLLM development on TPU.

### Ways to avoid recompilation
The model executor has two primary components:
- preparing the model and sampler inputs
- executing the model and sampler.
#### Step 1
It is recommended to avoid TPU operations when preparing the model and sampler 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature][Hardware][TPU]:Reduce the compile time

**Link**: https://github.com/vllm-project/vllm/issues/14582
**State**: closed
**Created**: 2025-03-10T22:36:38+00:00
**Closed**: 2025-04-16T05:31:48+00:00
**Comments**: 1
**Labels**: feature request, tpu

### Description

### 🚀 The feature, motivation and pitch

After the fix of https://github.com/vllm-project/vllm/pull/14310,

We have num_token_bucket compilations for the main model and num_token_bucket x num_reqs_bucket for the logits processor.

We can make some improvement on this, as the num_token_bucket x num_reqs_bucket only happens on hidden_states[logits_indices], where we select part of the hidden states. Therefore, we can partition the graph to 3 parts:

main model: num_token_bucket
hidden_states[logits_indices]: num_token_bucket x num_reqs_bucket
logits: num_reqs_bucket

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature][Hardware][TPU]: Improve the token_num padding logic

**Link**: https://github.com/vllm-project/vllm/issues/14581
**State**: closed
**Created**: 2025-03-10T22:34:10+00:00
**Closed**: 2025-03-25T21:27:24+00:00
**Comments**: 1
**Labels**: feature request, tpu

### Description

### 🚀 The feature, motivation and pitch

Currently the token_num is padded to power of 2. It is quite a waste of computation when token_num is large. In the meantime, in one of our benchmarking, the best max-num-batched-tokens is 512, also according to https://jax-ml.github.io/scaling-book/roofline/, we don't really need max-num-batched-tokens to be very large.

Also cudagraph precompiles for [512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1], we can use a similar padding strategy for TPU.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer l

[... truncated for brevity ...]

---

## Issue #N/A: [Feature][Hardware][TPU]: Add Recompilation Check for vLLM on TPU

**Link**: https://github.com/vllm-project/vllm/issues/14580
**State**: closed
**Created**: 2025-03-10T22:31:05+00:00
**Closed**: 2025-03-25T16:59:34+00:00
**Comments**: 1
**Labels**: feature request, tpu

### Description

### 🚀 The feature, motivation and pitch

Ideally, post-warmup, no further compilation should occur. However, PyTorch/XLA's implicit compilation can lead to excessive recompilation during LLM serving, impacting performance. We can add an option to detect recompilation after warmup, requiring a PyTorch/XLA method like xm.num_graph_hash() to track the number of captured graphs. This number should remain constant post-warmup if no recompilation occurs.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug][TPU]: Non-deterministic behaviour

**Link**: https://github.com/vllm-project/vllm/issues/12580
**State**: closed
**Created**: 2025-01-30T17:32:24+00:00
**Closed**: 2025-02-07T16:29:46+00:00
**Comments**: 6
**Labels**: bug, tpu

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 01-30 17:31:38 __init__.py:183] Automatically detected platform tpu.
Collecting environment information...
PyTorch version: 2.6.0.dev20241126+cpu
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.31.4
Libc version: glibc-2.35

Python version: 3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.8.0-1015-gcp-x86_64-with-glibc2.35
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
Architecture:                         x86_64
CPU op-mode(s):              

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Running Tensor Parallel on TPUs on Ray Cluster

**Link**: https://github.com/vllm-project/vllm/issues/12058
**State**: closed
**Created**: 2025-01-14T21:32:12+00:00
**Closed**: 2025-01-24T05:41:50+00:00
**Comments**: 9
**Labels**: usage, tpu, ray

### Description

### Your current environment

```text
The output of `python collect_env.py`
The output of `python collect_env.py`
(test_hf_qwen pid=17527, ip=10.130.4.26) Environment Information:
(test_hf_qwen pid=17527, ip=10.130.4.26) Collecting environment information...
(test_hf_qwen pid=17527, ip=10.130.4.26) PyTorch version: 2.6.0.dev20241126+cpu
(test_hf_qwen pid=17527, ip=10.130.4.26) Is debug build: False
(test_hf_qwen pid=17527, ip=10.130.4.26) CUDA used to build PyTorch: None
(test_hf_qwen pid=17527, ip=10.130.4.26) ROCM used to build PyTorch: N/A
(test_hf_qwen pid=17527, ip=10.130.4.26) 
(test_hf_qwen pid=17527, ip=10.130.4.26) OS: Ubuntu 22.04.4 LTS (x86_64)
(test_hf_qwen pid=17527, ip=10.130.4.26) GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
(test_hf_qwen pid=17527, ip=10.130.4.26) Clang version: 14.0.0-1ubuntu1.1
(test_hf_qwen pid=17527, ip=10.130.4.26) CMake version: version 3.31.2
(test_hf_qwen pid=17527, ip=10.130.4.26) Libc version: glibc-2.35
(test_hf_qwen pid=17527, ip=10.13

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Allow head_size smaller than 128 on TPU with Pallas backend

**Link**: https://github.com/vllm-project/vllm/issues/10343
**State**: closed
**Created**: 2024-11-14T21:51:45+00:00
**Closed**: 2025-07-10T05:02:05+00:00
**Comments**: 14
**Labels**: feature request, tpu

### Description

### 🚀 The feature, motivation and pitch

I would like to serve smaller models (e.g facebook/opt-125m) using VLLM on TPU. I can't do this currently because the Pallas backend has the limitation `NotImplementedError: Head size must be a multiple of 128`. I can't find a reason why this limitation is in place, and it would be great to be able to remove it with a flag or entirely. If my understanding is incorrect and there is a reason to have this limitation in place, please let me know! Thanks for your work on VLLM.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Installation]: VLLM does not support TPU v5p-16 (Multi-Host) with Ray Cluster

**Link**: https://github.com/vllm-project/vllm/issues/10155
**State**: closed
**Created**: 2024-11-08T11:36:17+00:00
**Closed**: 2025-02-07T02:16:05+00:00
**Comments**: 12
**Labels**: installation, tpu, ray

### Description

### Your current environment

```text
The output of `python collect_env.py`

Collecting environment information...
WARNING:root:libtpu.so and TPU device found. Setting PJRT_DEVICE=TPU.
INFO 11-04 16:11:44 importing.py:15] Triton not installed or not compatible; certain GPU-related functions will not be available.
PyTorch version: 2.6.0
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Debian GNU/Linux 11 (bullseye) (x86_64)
GCC version: (Debian 10.2.1-6) 10.2.1 20210110
Clang version: Could not collect
CMake version: version 3.30.5
Libc version: glibc-2.31

Python version: 3.10.15 (main, Oct 17 2024, 02:58:23) [GCC 10.2.1 20210110] (64-bit runtime)
Python platform: Linux-5.19.0-1022-gcp-x86_64-with-glibc2.31
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: No CUDA
Nvidia driver version: No CUDA
cuDNN version: No CUDA
HIP runtime version: N/A
MIOpen run

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vLLM crashes with larger context sizes on TPUs

**Link**: https://github.com/vllm-project/vllm/issues/8318
**State**: closed
**Created**: 2024-09-10T05:06:36+00:00
**Closed**: 2024-10-16T20:53:42+00:00
**Comments**: 9
**Labels**: bug, tpu

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
INFO 09-10 05:05:28 importing.py:10] Triton not installed; certain GPU-related functions will not be available.
PyTorch version: 2.5.0
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Debian GNU/Linux 11 (bullseye) (x86_64)
GCC version: (Debian 10.2.1-6) 10.2.1 20210110
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.31

Python version: 3.10.14 (main, Aug 13 2024, 02:16:06) [GCC 10.2.1 20210110] (64-bit runtime)
Python platform: Linux-6.1.85+-x86_64-with-glibc2.31
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
Architect

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: TPU InternVL2 Model Error Graph break due to unsupported builtin _XLAC.PyCapsule._xla_get_replication_devices_count

**Link**: https://github.com/vllm-project/vllm/issues/8066
**State**: closed
**Created**: 2024-09-01T20:04:57+00:00
**Closed**: 2025-01-02T01:59:36+00:00
**Comments**: 5
**Labels**: bug, tpu, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```
Collecting environment information...
Traceback (most recent call last):
  File "/home/kojoe/EasyAnimate/easyanimate/image_caption/collect_env.py", line 735, in <module>
    main()
  File "/home/kojoe/EasyAnimate/easyanimate/image_caption/collect_env.py", line 714, in main
    output = get_pretty_env_info()
  File "/home/kojoe/EasyAnimate/easyanimate/image_caption/collect_env.py", line 709, in get_pretty_env_info
    return pretty_str(get_env_info())
  File "/home/kojoe/EasyAnimate/easyanimate/image_caption/collect_env.py", line 510, in get_env_info
    pip_version, pip_list_output = get_pip_packages(run_lambda)
  File "/home/kojoe/EasyAnimate/easyanimate/image_caption/collect_env.py", line 480, in get_pip_packages
    out = run_with_pip([sys.executable, '-mpip'])
  File "/home/kojoe/EasyAnimate/easyanimate/image_caption/collect_env.py", line 476, in run_with_pip
   

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: TPU 'TYPE' property not found in Pallas backend

**Link**: https://github.com/vllm-project/vllm/issues/7989
**State**: closed
**Created**: 2024-08-29T08:51:07+00:00
**Closed**: 2024-08-30T07:31:48+00:00
**Comments**: 2
**Labels**: bug, tpu

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
INFO 08-29 08:52:05 importing.py:10] Triton not installed; certain GPU-related functions will not be available.
PyTorch version: 2.5.0
Is debug build: False
CUDA used to build PyTorch: Could not collect
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: 14.0.0-1ubuntu1.1
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.1.85+-x86_64-with-glibc2.35
Is CUDA available: False
CUDA runtime version: 12.5.40
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: Could not collect
Nvidia driver version: Could not collect
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.1.0
/usr/lib/x86_6

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Enable Prefix caching kernel on Pallas for TPU backend

**Link**: https://github.com/vllm-project/vllm/issues/7607
**State**: closed
**Created**: 2024-08-16T19:30:19+00:00
**Closed**: 2024-12-16T02:08:36+00:00
**Comments**: 2
**Labels**: feature request, tpu, stale

### Description

### 🚀 The feature, motivation and pitch

Enable Prefix caching kernel on Pallas for TPU backend

According to @WoosukKwon, we have a Triton and CUDA kernel implementations.

* https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/prefix_prefill.py
* https://github.com/vllm-project/vllm/issues/2614

---

## Issue #N/A: [TPU] Make sure worker index aligns with node boundary

**Link**: https://github.com/vllm-project/vllm/issues/7485
**State**: closed
**Created**: 2024-08-13T23:13:35+00:00
**Closed**: 2024-09-02T06:09:47+00:00
**Comments**: 0
**Labels**: tpu

### Description

In ray gpu executor, there are these lines:

https://github.com/vllm-project/vllm/blob/7025b11d949b4efeb2584690c35f919c77027368/vllm/executor/ray_gpu_executor.py#L175-L191

to make sure the worker index aligns with machine boundary. you might need it in TPU, too. Otherwise local ranks can be wrong. for example, rank 0, 1, 2, 4 in one node, and 3, 5, 6, 7 in another node.

_Originally posted by @youkaichao in https://github.com/vllm-project/vllm/issues/7457#issuecomment-2285381534_

---

## Issue #N/A: [RFC] Initial Support for Cloud TPUs

**Link**: https://github.com/vllm-project/vllm/issues/3620
**State**: closed
**Created**: 2024-03-25T17:08:43+00:00
**Closed**: 2025-03-11T14:04:01+00:00
**Comments**: 17
**Labels**: RFC, tpu, stale

### Description

# Progress

- [x] Implement TPU executor that works on a single TPU chip (without tensor parallelism) #5292 
- [x] Support single-host tensor parallel inference #5871 
- [x] Support multi-host tensor parallel inference #7457 
- [ ] Support INT8 quantization
- [x] Support MoE models such as Mixtral #6457
- [ ] Benchmark and optimize the TPU backend performance

# Project Scope

This project focuses on making vLLM compatible with Google cloud TPUs. Our goal is seamless integration so users can easily run vLLM on TPUs for both online and offline inference. We will target common setups, like popular models such as Gemma, using the bfloat16 data type.

## Target TPUs and Models

We will focus on the most recent generations of TPUs, namely **TPU v4, v5e, and v5p**, considering their superior performance to previous generations. We will start by making sure vLLM works with dense models such as Gemma. After that, we will expand support to Mixture-of-Experts (MoE) models such as 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: TPU Embedding models support?

**Link**: https://github.com/vllm-project/vllm/issues/20869
**State**: open
**Created**: 2025-07-13T05:32:13+00:00
**Comments**: 0
**Labels**: feature request, tpu

### Description

### 🚀 The feature, motivation and pitch

Hi I want to run latest Embedding models, eg `Qwen/Qwen3-Embedding-0.6B`, on TPU nodes. I found that although vLLM has support on TPU it does not really support embedding models since the only available attention implementation on TPU is `PALLAS` which is DECODER only. https://github.com/vllm-project/vllm/blob/99b4f080d83ae284941b01922d7fe3b9a39034fd/vllm/v1/attention/backends/pallas.py#L164-L168

Meanwhile, Qwen3 Embedding is ENCODER-only so it can't run on TPU. https://github.com/vllm-project/vllm/blob/99b4f080d83ae284941b01922d7fe3b9a39034fd/vllm/model_executor/models/qwen3.py#L166-L173

It will be nice if we can support Qwen3 Embedding on TPU,

### Alternatives

I am trying to use Qwen3 Embedding via `transformers` but it's not as performant as vLLM.

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] [TPU]: OOMing on Llama-8B on new vllm nightly docker

**Link**: https://github.com/vllm-project/vllm/issues/19490
**State**: open
**Created**: 2025-06-11T13:58:55+00:00
**Comments**: 5
**Labels**: bug, tpu

### Description

### Your current environment
<details>
<summary>The output of <code>python collect_env.py</code></summary>

```
root@t1v-n-82109f0e-w-0:/opt# python collect_env.py 
WARNING:root:libtpu.so and TPU device found. Setting PJRT_DEVICE=TPU.
INFO 06-11 13:54:20 [__init__.py:244] Automatically detected platform tpu.
INFO 06-11 13:54:20 [tpu.py:215] tpu_commons not found, using vLLM's TpuPlatform
Collecting environment information...
==============================
        System Info
==============================
OS                           : Debian GNU/Linux 11 (bullseye) (x86_64)
GCC version                  : (Debian 10.2.1-6) 10.2.1 20210110
Clang version                : Could not collect
CMake version                : version 4.0.2
Libc version                 : glibc-2.31

==============================
       PyTorch Info
==============================
PyTorch version              : 2.8.0.dev20250605+cpu
Is debug build               : False
CUDA used to build PyTorch   : None
ROCM use

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Multi-Node Online Inference on TPUs Failing

**Link**: https://github.com/vllm-project/vllm/issues/12179
**State**: open
**Created**: 2025-01-17T23:38:35+00:00
**Comments**: 5
**Labels**: bug, tpu, ray, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
root@t1v-n-4d36f9a1-w-0:/workspace/vllm# python collect_env.py
INFO 01-17 23:21:42 __init__.py:179] Automatically detected platform tpu.
Collecting environment information...
PyTorch version: 2.6.0.dev20241126+cpu
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Debian GNU/Linux 11 (bullseye) (x86_64)
GCC version: (Debian 10.2.1-6) 10.2.1 20210110
Clang version: Could not collect
CMake version: version 3.31.4
Libc version: glibc-2.31

Python version: 3.10.15 (main, Oct 17 2024, 02:58:23) [GCC 10.2.1 20210110] (64-bit runtime)
Python platform: Linux-5.19.0-1022-gcp-x86_64-with-glibc2.31
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: No CUDA
Nvidia driver version: No CUDA
cuDNN version: No CUDA
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vLLM on TPU does not support --pipeline-parallel-size with Ray

**Link**: https://github.com/vllm-project/vllm/issues/11260
**State**: open
**Created**: 2024-12-17T13:04:46+00:00
**Comments**: 4
**Labels**: bug, tpu, ray, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.6.0.dev20241126+cpu
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.31.2
Libc version: glibc-2.35

Python version: 3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.5.0-1013-gcp-x86_64-with-glibc2.35
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
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:         

[... truncated for brevity ...]

---

