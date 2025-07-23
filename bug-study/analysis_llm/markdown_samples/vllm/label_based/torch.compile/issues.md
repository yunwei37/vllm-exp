# torch.compile - issues

**Total Issues**: 28
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 18
- Closed Issues: 10

### Label Distribution

- torch.compile: 28 issues
- bug: 20 issues
- RFC: 5 issues
- stale: 3 issues
- feature request: 2 issues
- startup-ux: 2 issues
- llama: 1 issues
- performance: 1 issues
- unstale: 1 issues

---

## Issue #N/A: [Bug]: Inductor codegen: fatal error: stddef.h: No such file or directory

**Link**: https://github.com/vllm-project/vllm/issues/19656
**State**: closed
**Created**: 2025-06-15T05:05:13+00:00
**Closed**: 2025-06-20T05:00:54+00:00
**Comments**: 3
**Labels**: bug, torch.compile

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
Collecting environment information...
==============================
        System Info
==============================
OS                           : CentOS Stream 9 (x86_64)
GCC version                  : (GCC) 11.5.0 20240719 (Red Hat 11.5.0-5)
Clang version                : 20.1.1 (CentOS 20.1.1-3.el9)
CMake version                : version 3.26.5
Libc version                 : glibc-2.34

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
Python version               : 3.12.10 (main, Apr  9 2025, 00:00:00) [GCC 11.5.0 20240719 (Red Hat 11.5.0-5)] (64-bit runtime)
Python platform              : Linux-6

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

## Issue #N/A: [Bug]:  h unknown: block: [487,0,0], thread: [31,0,0] Assertion `index out of bounds: 0 <= tl.broadcast_to(tmp34, [XBLOCK]) < 131072` failed.

**Link**: https://github.com/vllm-project/vllm/issues/17348
**State**: closed
**Created**: 2025-04-29T03:54:46+00:00
**Closed**: 2025-04-29T05:45:53+00:00
**Comments**: 3
**Labels**: bug, torch.compile

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: CentOS Linux 8 (x86_64)
GCC version: (GCC) 10.5.0
Clang version: Could not collect
CMake version: version 3.20.2
Libc version: glibc-2.29

Python version: 3.10.0 (default, Mar  3 2022, 09:58:08) [GCC 7.5.0] (64-bit runtime)
Python platform: Linux-4.18.0-348.7.1.el8_5.x86_64-x86_64-with-glibc2.29
Is CUDA available: True
CUDA runtime version: Could not collect
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

Nvidia driver version: 550.135
cuDNN version: Probably one of the following:
/usr/local/cuda-12.2/targets/x86_64-linux

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Running `vllm serve Qwen2.5-VL-72B-Instruct-AWQ` results in an error when upgrading the vLLM version to 0.8.5.

**Link**: https://github.com/vllm-project/vllm/issues/17344
**State**: closed
**Created**: 2025-04-29T02:51:02+00:00
**Closed**: 2025-05-13T04:18:28+00:00
**Comments**: 15
**Labels**: bug, torch.compile

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.1 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-91-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A100-SXM4-80GB
Nvidia driver version: 535.161.08
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:                      46 bits physical, 5

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: triton placeholder is conflicting with pytorch's triton checks

**Link**: https://github.com/vllm-project/vllm/issues/17309
**State**: closed
**Created**: 2025-04-28T14:18:25+00:00
**Closed**: 2025-05-02T07:45:02+00:00
**Comments**: 2
**Labels**: bug, torch.compile

### Description

### Your current environment

Addition of a PlaceholderModule for triton [PR:15099](https://github.com/vllm-project/vllm/pull/15099) has broken pytorch's internal checks for triton. This is breaking vllm's model serving (tested for arch: ppc64le).

Pytorch has  conditional checks for triton [_is_triton_available()](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_inductor/runtime/hints.py#L34)
Once vllm is imported, the above referenced function returns `True` and the control wrongly flows to importing triton functions which causes `ModuleNotFoundError` [here](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/_inductor/runtime/hints.py#L67)


Suggestions:

1. We can try bumping up torch version to 2.7.0
    v2.7.0 slightly different imports to check for triton - [has_triton_package()](https://github.com/pytorch/pytorch/blob/v2.7.0/torch/_inductor/runtime/hints.py#L38)
    Implementation details for has_triton_package [here](https://github.com/pytorch/pytorch/blob/v2.7.0/torch/u

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Error When Launching Llama-4-Scout-17B-16E-Instruct Without `--kv-cache-dtype fp8`

**Link**: https://github.com/vllm-project/vllm/issues/16150
**State**: closed
**Created**: 2025-04-07T03:33:28+00:00
**Closed**: 2025-04-15T06:11:13+00:00
**Comments**: 6
**Labels**: bug, torch.compile

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 04-07 11:13:31 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
/usr/local/lib/python3.10/dist-packages/_distutils_hack/__init__.py:33: UserWarning: Setuptools is replacing distutils.
  warnings.warn("Setuptools is replacing distutils.")
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.28.1
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-94-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.3.107
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H100 80GB HBM3
GPU 1: NVIDIA H100 80G

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Enable CUDA Graph without turn on torch.compile / Inductor for V1

**Link**: https://github.com/vllm-project/vllm/issues/15896
**State**: closed
**Created**: 2025-04-01T17:19:26+00:00
**Closed**: 2025-05-29T02:16:53+00:00
**Comments**: 12
**Labels**: feature request, torch.compile

### Description

### 🚀 The feature, motivation and pitch

For simple models, we may not need fusion from torch.compile. And piecewise approach may be slow. So we would like to enable this feature.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]:ModuleNotFoundError: No module named 'vllm._C' 

**Link**: https://github.com/vllm-project/vllm/issues/15592
**State**: closed
**Created**: 2025-03-27T02:52:34+00:00
**Closed**: 2025-05-28T17:19:27+00:00
**Comments**: 18
**Labels**: bug, torch.compile

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
(vllm3) [root@hygpu-002 envs]# python vllm/collect_env.py 
/mnt/qy-test/envs/vllm/vllm/__init__.py:5: RuntimeWarning: Failed to read commit hash:
No module named 'vllm._version'
  from .version import __version__, __version_tuple__  # isort:skip
INFO 03-27 10:36:47 [__init__.py:239] Automatically detected platform cuda.
Traceback (most recent call last):
  File "/mnt/qy-test/envs/vllm/collect_env.py", line 17, in <module>
    from vllm.envs import environment_variables
  File "/mnt/qy-test/envs/vllm/vllm/__init__.py", line 11, in <module>
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
  File "/mnt/qy-test/envs/vllm/vllm/engine/arg_utils.py", line 22, in <module>
    from vllm.executor.executor_base import ExecutorBase
  File "/mnt/qy-test/envs/vllm/vllm/executor/executor_base.py", line 16, in <module>
    from vllm.model_executor.layers.sampler import SamplerOutp

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: V1 CudaGrpah

**Link**: https://github.com/vllm-project/vllm/issues/10945
**State**: closed
**Created**: 2024-12-06T06:33:54+00:00
**Closed**: 2025-04-15T02:08:08+00:00
**Comments**: 4
**Labels**: performance, torch.compile, stale

### Description

### Proposal to improve performance

_No response_

### Report of performance regression

_No response_

### Misc discussion on performance

I have ported the vllm code to my TTS model, using llama for autoregressive token generation, and I am using version v0.2.7. I noticed that during the decode step, using the `torch.cuda.CUDAGraph().replay()` method, my inference speed has increased to 6 times the original. I observed that version V1 does not use `torch.cuda.CUDAGraph()`, and upon testing, I found that setting `VLLM_TORCH_COMPILE_LEVEL=3` not only fails to achieve a 6-fold increase in inference speed but also slows down the process, with a significant increase in the time taken by RMSNorm. Are there any suggestions to help me modify the code?


### Your current environment (if you think it is necessary)

```text
The output of `python collect_env.py`
```


### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Support attention backend with FlexAttention

**Link**: https://github.com/vllm-project/vllm/issues/7315
**State**: closed
**Created**: 2024-08-08T19:52:26+00:00
**Closed**: 2025-02-14T01:59:26+00:00
**Comments**: 10
**Labels**: feature request, torch.compile, stale

### Description

### 🚀 The feature, motivation and pitch

FlexAttention was proposed as a performant attention implementation leveraging `torch.compile` with easy APIs for adding support for complex attention variants such as Causal, [Relative Positional Embeddings](https://paperswithcode.com/method/relative-position-encodings), [Alibi](https://paperswithcode.com/method/alibi), [Sliding Window Attention](https://mistral.ai/news/announcing-mistral-7b/), [PrefixLM](https://twitter.com/andersonbcdefg/status/1800907703688339569), [Document Masking/Sample Packing/Jagged Tensors](https://github.com/pytorch/torchtune/pull/875), [Tanh Soft-Capping](https://twitter.com/LysandreJik/status/1807779471891538199), [PagedAttention](https://arxiv.org/abs/2309.06180), etc.

https://pytorch.org/blog/flexattention/

While it is not the fastest attention backend (yet!) it is clearly performant enough while enabling much more flexibility than current compiled backends to easily implement attention features we need fo

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: vLLM-compile low-hanging fruit cold start improvements

**Link**: https://github.com/vllm-project/vllm/issues/20451
**State**: open
**Created**: 2025-07-03T19:22:18+00:00
**Comments**: 0
**Labels**: RFC, torch.compile, startup-ux

### Description

### Motivation.

This issue tracks potential low-hanging fruit for improving vLLM-compile cold start time. @anijain2305, @BoyuanFeng, and I sat down to look at some traces and noticed some things we can improve.

There are more longer-term projects for improving torch.compile cold start time, but those will probably take a bit to hit.

### Proposed Change.

- [ ] vLLM's [custom bytecode hook](https://github.com/vllm-project/vllm/blob/536fd330036b0406786c847f68e4f67cba06f421/vllm/compilation/wrapper.py#L77-L121) seems to take a long time (~7 seconds on llama-3.1-70b model). I'm not sure how much of this is actually needed for runtime execution. We should guard the decompilation step behind an envvar. If VLLM_COMPILE_DEPYF=0 (default), we write out a `transformed_code.py` that has a comment that says "Please set VLLM_COMPILE_DEPYF=1 to populate this file".
- [ ] In llama-3.1-70b, with piecewise cudagraphs, we split a module into 80 different subgraphs. A lot of these subgraphs are litera

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: vLLM-compile (minus cudagraphs) warm-start time should be close to zero

**Link**: https://github.com/vllm-project/vllm/issues/20402
**State**: open
**Created**: 2025-07-02T22:04:20+00:00
**Comments**: 6
**Labels**: RFC, torch.compile, startup-ux

### Description

### Motivation.

@BoyuanFeng did some benchmarks of vLLM cold vs warm start of a 70B model. In the warm start, compilation (ignoring cudagraphs) took 25 out of 132 seconds, almost 20% of the time. On warm start, all of the hard work (compiling artifacts) should have been already done.

The theoretical minimum amount of time that vLLM-compile needs to spend in warm start is the amount of time it takes to load all the compiled code.

![Image](https://github.com/user-attachments/assets/b34204f8-5ad5-49d4-bdc6-6805610ac6be)

### Proposed Change.

The following categories correspond to what is in the chart above.

Dynamo:
- On warm start, vLLM always re-runs Dynamo. We don't need to do this: instead, we can directly serialize the bytecode that Dynamo produces and re-load it.
- Originally I was planning on waiting until torch.compile implemented "precompilation", which will skip Dynamo on warm start. It might be worth figuring out how to get a simpler version of this into vLLM, especially be

[... truncated for brevity ...]

---

## Issue #N/A: [RFC][UX]: debug mode for vLLM-compile

**Link**: https://github.com/vllm-project/vllm/issues/20394
**State**: open
**Created**: 2025-07-02T17:56:56+00:00
**Comments**: 1
**Labels**: RFC, torch.compile

### Description

### Motivation.

vLLM-compile (CompilationLevel.PIECEWISE) makes a lot of assumptions about the models that allow it to make them run really fast. There are two main assumptions that commonly lead to silent incorrectness if the models violate them. I've spent countless hours debugging user issues for it to turn out to be one of these assumptions. We should add a debug mode option for vLLM-compile that, when turned on, adds some safety checks for these assumptions at the tradeoff of some additional overhead. This will let users self-diagnose the issues without me in the loop.

This is one of the items mentioned in https://github.com/vllm-project/vllm/issues/20283, I'm expanding it to include some more details.

### Proposed Change.

The two assumptions that bite us are:
1) the [vLLM Dynamic Shapes Issue](https://docs.google.com/document/d/1R3XvVEpJeVi3whyxf4xpyZufGplbrfw628oXLZ6fqG0/edit?tab=t.0#heading=h.59xosv6nz9lg). vLLM performs one single graph capture with dynamic batch size and 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: V1 pre-compiled graph loading much slower than V0

**Link**: https://github.com/vllm-project/vllm/issues/20342
**State**: open
**Created**: 2025-07-01T23:47:53+00:00
**Comments**: 4
**Labels**: bug, torch.compile

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
Collecting environment information...
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
Python version               : 3.12.11 (main, Jun  4 20

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: V1 piecewise cudagraph capture size on ROCm is much higher than on cuda

**Link**: https://github.com/vllm-project/vllm/issues/19579
**State**: open
**Created**: 2025-06-12T20:55:01+00:00
**Comments**: 0
**Labels**: bug, torch.compile

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
ROCM Version                 : 6.3.42133-1b9c17779
vLLM Version                 : 0.9.1.dev325+g9d880f594 (git sha: 9d880f594)
PYTORCH_TUNABLEOP_TUNING=0
PYTORCH_TUNABLEOP_ENABLED=1
PYTORCH_ROCM_ARCH=gfx942
LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib:
PYTORCH_TUNABLEOP_FILENAME=/app/afo_tune_device_%d_full.csv
NCCL_CUMEM_ENABLE=0
PYTORCH_NVML_BASED_CUDA_CHECK=1
TORCHINDUCTOR_COMPILE_THREADS=1
CUDA_MODULE_LOADING=LAZY

```
</details>





### 🐛 Describe the bug

The size of piecewise cudagraph is much higher on rocm (mi300) than on cuda (h100). See table below. Also, this issue seems to be specific to piecewise capture; when doing a fullgraph capture on rocm, the graph size is fine.

**Note**: The issue is Not related to rccl/all_reduce etc. because the captured sizes below use TP=1

#### Instructions to reproduce the issue:
Engine init logs contain the graph captured si

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Qwen3-GPTQ | Error in inspecting model architecture 'Qwen3MoeForCausalLM'

**Link**: https://github.com/vllm-project/vllm/issues/19504
**State**: open
**Created**: 2025-06-11T18:11:20+00:00
**Comments**: 4
**Labels**: bug, torch.compile

### Description

### Your current environment

 **VLLM v 0.9.0.1**


### 🐛 Describe the bug

I am using docker image with **VLLM v 0.9.0.1**
I have download the model [`Qwen/Qwen3-235B-A22B-GPTQ-Int4`] at this directory `qwen3-gptq`: 

I have a node with 8 H100 GPUs
`VLLM_USE_V1=0 vllm serve qwen3-gptq   --tensor-parallel-size 8  --max-model-len 32000   --gpu-memory-utilization 0.9   --distributed-executor-backend mp `


I have this error 
```
INFO 06-01 11:19:03 [__init__.py:243] Automatically detected platform cuda.
INFO 06-01 11:19:24 [__init__.py:31] Available plugins for group vllm.general_plugins:
INFO 06-01 11:19:24 [__init__.py:33] - lora_filesystem_resolver -> vllm.plugins.lora_resolvers.filesystem_resolver:register_filesystem_resolver
INFO 06-01 11:19:24 [__init__.py:36] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.

init_-py:36] All plugins in this group will be loaded. Set "VLLM_PLUGINS' to control which plugins to load.
[registry-py: 363] Er

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vLLM outputs are not reproducible

**Link**: https://github.com/vllm-project/vllm/issues/19491
**State**: open
**Created**: 2025-06-11T14:40:27+00:00
**Comments**: 16
**Labels**: bug, torch.compile

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
Collecting environment information...
==============================
        System Info
==============================
OS                           : Ubuntu 22.04.5 LTS (x86_64)
GCC version                  : (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version                : Could not collect
CMake version                : version 3.22.1
Libc version                 : glibc-2.35

==============================
       PyTorch Info
==============================
PyTorch version              : 2.7.0+cu126
Is debug build               : False
CUDA used to build PyTorch   : 12.6
ROCM used to build PyTorch   : N/A

==============================
      Python Environment
==============================
Python version               : 3.11.11 (main, Dec 11 2024, 16:28:39) [GCC 11.2.0] (64-bit runtime)
Python platform              : Linux-5.15.0-135-generic-x86_64-with-glibc2.35

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Compile inductor / CUDA Graph build before the memory profiling

**Link**: https://github.com/vllm-project/vllm/issues/19480
**State**: open
**Created**: 2025-06-11T08:42:44+00:00
**Comments**: 3
**Labels**: bug, torch.compile

### Description

### Your current environment

Running Llama4 Maverick on H100x8

### 🐛 Describe the bug

Otherwise, it's easy to get OOM. Inductor and CUDA graph themselves may consume a lot of memory, especially, inductor may leverage some profiling to search the best config for the kernels.

```
export LLAMA_DIR=meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8; export PORT=8081 VLLM_LOGGING_LEVEL=DEBUG VLLM_DISABLE_COMPILE_CACHE=1 SAFETENSORS_FAST_GPU=1 vllm serve $LLAMA_DIR --disable-log-requests -tp 8 --host :: --port $PORT --served-model-name default --no-enable-prefix-caching --max-model-len 4096 --gpu-memory-utilization 0.8 2>&1 | tee marverik_fp8_no_compile.log
```

If we use 0.9 or 0.95, it's easy to reproduce the issue on H100x8 machines.
0.8 may be okay.

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Issue of Unstable Output for Identical Queries

**Link**: https://github.com/vllm-project/vllm/issues/19403
**State**: open
**Created**: 2025-06-10T07:07:59+00:00
**Comments**: 23
**Labels**: bug, torch.compile

### Description

### Your current environment

INFO 06-10 00:07:35 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.31.4
Libc version: glibc-2.35

Python version: 3.12.8 (main, Dec  4 2024, 08:54:12) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.4.54-1.0.0.std7c.el7.2.x86_64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA GeForce RTX 4090
GPU 1: NVIDIA GeForce RTX 4090
GPU 2: NVIDIA GeForce RTX 4090
GPU 3: NVIDIA GeForce RTX 4090
GPU 4: NVIDIA GeForce RTX 4090
GPU 5: NVIDIA GeForce RTX 4090
GPU 6: NVIDIA GeForce RTX 4090
GPU 7: NVIDIA GeForce RTX 4090

Nvidia driver version: 535.104.05
cuDNN

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Strange error `AssertionError: failed to get the hash of the compiled graph` when running `Qwen/Qwen3-8B` via `LLM` class

**Link**: https://github.com/vllm-project/vllm/issues/18851
**State**: open
**Created**: 2025-05-28T19:04:20+00:00
**Comments**: 20
**Labels**: bug, torch.compile

### Description

### Your current environment

```
>>> import vllm; vllm.__version__
INFO 05-28 19:02:30 [__init__.py:248] Automatically detected platform cuda.
'0.9.1.dev59+gb6a6e7a52'
>>>
>>> import torch; torch.__version__
'2.7.0+cu126'
>>> import transformers; transformers.__version__
'4.52.2'
```

### 🐛 Describe the bug

``` 
(VllmWorker rank=1 pid=191128) ERROR 05-28 18:58:32 [multiproc_executor.py:522] Traceback (most recent call last):                                                                                                                                                                                                      (VllmWorker rank=1 pid=191128) ERROR 05-28 18:58:32 [multiproc_executor.py:522]   File "/mnt/fs/venv_cu126_py312/lib/python3.12/site-packages/vllm/v1/executor/multiproc_executor.py", line 517, in worker_busy_loop                                                                                                    (VllmWorker rank=1 pid=191128) ERROR 05-28 18:58:32 [multipr

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: torch._inductor.exc.InductorError: TypeError: cannot pickle 'torch._C.DispatchKeySet' object

**Link**: https://github.com/vllm-project/vllm/issues/17593
**State**: open
**Created**: 2025-05-02T15:20:59+00:00
**Comments**: 1
**Labels**: bug, torch.compile

### Description

### Your current environment

vLLM main branch, PyTorch main branch

### 🐛 Describe the bug

Repro:
`pytest -v -s tests/compile/piecewise/test_toy_llama.py`

Gives:
```
>                           rv = reductor(4)
E                           torch._inductor.exc.InductorError: TypeError: cannot pickle 'torch._C.DispatchKeySet' object
E
E                           Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even m
ore developer context, set TORCH_LOGS="+dynamo"

../env/lib/python3.12/copy.py:151: InductorError
====================================================================== warnings summary =======================================================================
```

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of f

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: [V1][Spec Dec] EAGLE TP > 1 leads to errors when using --enforce_eager

**Link**: https://github.com/vllm-project/vllm/issues/17513
**State**: open
**Created**: 2025-05-01T01:42:30+00:00
**Comments**: 9
**Labels**: bug, torch.compile

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: Could not collect
CMake version: version 3.16.3
Libc version: glibc-2.31

Python version: 3.12.9 | packaged by conda-forge | (main, Mar  4 2025, 22:48:41) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-5.15.0-1064-aws-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A100-SXM4-40GB
GPU 1: NVIDIA A100-SXM4-40GB
GPU 2: NVIDIA A100-SXM4-40GB
GPU 3: NVIDIA A100-SXM4-40GB
GPU 4: NVIDIA A100-SXM4-40GB
GPU 5: NVIDIA A100-SXM4-40GB
GPU 6: NVIDIA A100-SXM4-40GB
GPU 7: NVIDIA A100-SXM4-40GB

Nvidia driver version: 535.183.01
cuDNN version: Could not co

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: vLLM x torch.compile caching should be opt-out by default

**Link**: https://github.com/vllm-project/vllm/issues/16501
**State**: open
**Created**: 2025-04-11T16:58:19+00:00
**Comments**: 5
**Labels**: RFC, torch.compile

### Description

### Motivation.

How vLLM decides to cache torch.compile compilations is brittle. There's [a list of configs](https://github.com/vllm-project/vllm/blob/70de35a8816e224663aede45b7f54eef250a5cfe/vllm/compilation/backends.py#L360-L394) that it takes into account and hashes, if any of these configs change then vLLM decides that it needs to do a fresh torch.compile run.

As we saw in https://github.com/vllm-project/vllm/pull/16491, it's very easy to add a new feature to one of the configs and forget to update the hash function. In that PR, the problem was that ModelConfig's hash function did not take into account [everything that could change the compilation](https://github.com/vllm-project/vllm/blob/70de35a8816e224663aede45b7f54eef250a5cfe/vllm/config.py#L279-L303).





### Proposed Change.

The hash functions are currently opt-in: when someone adds a new feature or does a refactor they may need to add something to the hash functions. After discussion with the PyTorch Compiler team (cc @o

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: FP8 Quantization with enforce_eager=False Causes Gibberish Output on Llama-4-Scout Model (VLLM_USE_V1=1)

**Link**: https://github.com/vllm-project/vllm/issues/16337
**State**: open
**Created**: 2025-04-09T11:22:10+00:00
**Comments**: 8
**Labels**: bug, torch.compile

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.7.0a0+git295f2ed
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.3.42133-1b9c17779

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.1 24491 1e0fda770a2079fbd71e4b70974d74f62fd3af10)
CMake version: version 3.31.6
Libc version: glibc-2.35

Python version: 3.12.9 (main, Feb  5 2025, 08:49:00) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-116-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Instinct MI300X (gfx942:sramecc+:xnack-)
Nvidia driver version: Could not collect
cuDNN version: Could not collect
HIP runtime version: 6.3.42133
MIOpen runtime version: 3.3.0
Is XNNPACK available: True



[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: TypeError: __init__() missing 1 required positional argument: 'inner_exception'

**Link**: https://github.com/vllm-project/vllm/issues/16009
**State**: open
**Created**: 2025-04-03T10:58:22+00:00
**Comments**: 5
**Labels**: bug, torch.compile, stale

### Description

### Your current environment

<details>

```text
PyTorch version: 2.6.0+cu118
Is debug build: False
CUDA used to build PyTorch: 11.8
ROCM used to build PyTorch: N/A

OS: CentOS Linux 7 (Core) (x86_64)
GCC version: (GCC) 9.3.1 20200408 (Red Hat 9.3.1-2)
Clang version: Could not collect
CMake version: version 3.27.6
Libc version: glibc-2.17

Python version: 3.9.19 | packaged by conda-forge | (main, Mar 20 2024, 12:50:21)  [GCC 12.3.0] (64-bit runtime)
Python platform: Linux-4.18.0-147.mt20200626.413.el8_1.x86_64-x86_64-with-glibc2.17
Is CUDA available: True
CUDA runtime version: 11.7.99
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A100-SXM4-80GB
GPU 1: NVIDIA A100-SXM4-80GB

Nvidia driver version: Could not collect
cuDNN version: Probably one of the following:
/usr/lib64/libcudnn.so.8.5.0
/usr/lib64/libcudnn_adv_infer.so.8.5.0
/usr/lib64/libcudnn_adv_train.so.8.5.0
/usr/lib64/libcudnn_cnn_infer.so.8.5.0
/usr/lib64/libcudnn_cnn_train.so.8.5.0
/usr/lib64/li

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: crash during debug, works ok running cli

**Link**: https://github.com/vllm-project/vllm/issues/16006
**State**: open
**Created**: 2025-04-03T10:13:52+00:00
**Comments**: 4
**Labels**: bug, torch.compile

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...

PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: 14.0.0-1ubuntu1.1
CMake version: version 3.31.4
Libc version: glibc-2.35

Python version: 3.12.9 (main, Feb  5 2025, 08:49:00) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.8.0-50-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 4090
Nvidia driver version: 570.86.15
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:        

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vLLM v1 hanging during Torch compilation

**Link**: https://github.com/vllm-project/vllm/issues/15360
**State**: open
**Created**: 2025-03-23T16:41:41+00:00
**Comments**: 0
**Labels**: bug, torch.compile

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.11.0rc1 (main, Aug 12 2022, 10:02:14) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-1077-aws-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A10G
Nvidia driver version: 535.161.07
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8.9.7
/usr/lib/x86_64-linux-gnu/

[... truncated for brevity ...]

---

## Issue #N/A: [WIP][RFC]: Use auto-functionalization V2 in PyTorch 2.7+

**Link**: https://github.com/vllm-project/vllm/issues/14703
**State**: open
**Created**: 2025-03-12T21:19:52+00:00
**Comments**: 6
**Labels**: RFC, torch.compile, unstale

### Description

### Motivation

In PyTorch 2.6, `auto_functionalized_v2` was introduced as a replacement for the `auto_functionalized` higher-order, partially to address the issues with redundant tensor copies in vLLM. However, certain custom fusion passes rely on pattern matching and don't currently work with `auto_functionalized_v2`.

Due to this as well as a separate issue with V2 ([PyTorch#147924](https://github.com/pytorch/pytorch/issues/147924)), we are currently disabling V2 in PyTorch 2.6+. We have also circumvented the copy issues using a `FixFunctionalizationPass`, reducing the urgency for enabling V2.

I am creating this RFC to centralize the discussion about when to upgrade to V2 and how to mitigate it in custom fusion passes.

#### Motivation for custom passes

Our graph-level optimization system performs graph transformations that would break abstractions or be intrusive to model code in some other way. For example, `RMSNormFusionPass` performs manual fusion of RMSNorm and quantization c

[... truncated for brevity ...]

---

