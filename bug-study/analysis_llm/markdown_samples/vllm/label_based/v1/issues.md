# v1 - issues

**Total Issues**: 28
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 7
- Closed Issues: 21

### Label Distribution

- v1: 28 issues
- bug: 18 issues
- stale: 10 issues
- RFC: 4 issues
- structured-output: 3 issues
- ci/build: 3 issues
- speculative-decoding: 2 issues
- deepseek: 1 issues
- help wanted: 1 issues
- multi-modality: 1 issues

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

## Issue #N/A: [Bug][V1] 'PixtralVisionConfig' object has no attribute 'spatial_merge_size' in 0.8.5

**Link**: https://github.com/vllm-project/vllm/issues/17565
**State**: closed
**Created**: 2025-05-01T23:49:41+00:00
**Closed**: 2025-05-02T05:14:10+00:00
**Comments**: 6
**Labels**: bug, v1

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.11.11 | packaged by conda-forge | (main, Mar  3 2025, 20:43:55) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-6.5.0-1024-aws-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA L4
GPU 1: NVIDIA L4
GPU 2: NVIDIA L4
GPU 3: NVIDIA L4

Nvidia driver version: 550.163.01
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.1.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.1.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9.1.0
/usr/lib/x8

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: SpecDecoding metrics showing with disabled spec decoding

**Link**: https://github.com/vllm-project/vllm/issues/15958
**State**: closed
**Created**: 2025-04-02T18:23:33+00:00
**Closed**: 2025-04-04T15:52:42+00:00
**Comments**: 1
**Labels**: bug, speculative-decoding, v1

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
WARNING:root:libtpu.so and TPU device found. Setting PJRT_DEVICE=TPU.
INFO 04-02 18:23:13 [__init__.py:239] Automatically detected platform tpu.
Collecting environment information...
PyTorch version: 2.8.0
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.31.6
Libc version: glibc-2.35

Python version: 3.11.11 (main, Feb 12 2025, 14:51:05) [Clang 19.1.6 ] (64-bit runtime)
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
Architecture:   

[... truncated for brevity ...]

---

## Issue #N/A: [Bug][V1]: ngram + guided decoding

**Link**: https://github.com/vllm-project/vllm/issues/15554
**State**: closed
**Created**: 2025-03-26T14:44:47+00:00
**Closed**: 2025-06-27T04:35:33+00:00
**Comments**: 3
**Labels**: bug, stale, v1

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.12.8 | packaged by Anaconda, Inc. | (main, Dec 11 2024, 16:31:09) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-130-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.3.103
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A10
GPU 1: NVIDIA A10
GPU 2: NVIDIA A10
GPU 3: NVIDIA A10

Nvidia driver version: 535.183.01
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                      

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Mismatch between `get_multimodal_embedding` output and `PlaceholderRange`

**Link**: https://github.com/vllm-project/vllm/issues/15144
**State**: closed
**Created**: 2025-03-19T16:53:23+00:00
**Closed**: 2025-03-30T10:47:54+00:00
**Comments**: 4
**Labels**: bug, help wanted, v1, multi-modality

### Description

In V1, we expect the output of `get_multimodal_embedding` to correspond to the `PlaceholderRange`, which is in turn constructed based on `PromptUpdateDetails.features`. However, the current V1 code doesn't validate this, causing the model to crash during inference when under high load (e.g. #14897, #14963).

From a quick look at the code, these models output embedding sizes which are inconsistent with the placeholder range:

- [x] Fuyu (fixed by #15731)
- [x] Gemma3 (fixed by #14980)
- [x] Idefics3 (fixed by #15696)
- [x] InternVL-based models (fixed by #15086)
- [x] MiniCPM-V (fixed by #15487)

(Basically, any model that has image newline/column tokens after applying HF processor needs a mask to map image patch features to image embeddings, as described below.)

To fix this, we can follow these steps:

1. Update the multi-modal processor to output a mask to indicate which positions in the `PlaceholderRange`-aligned embeddings should the patch features (outputted by vision encoder) be 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: UserWarning on skipping serialisation of PostGradPassManager

**Link**: https://github.com/vllm-project/vllm/issues/14911
**State**: closed
**Created**: 2025-03-17T01:16:18+00:00
**Closed**: 2025-07-16T02:42:28+00:00
**Comments**: 3
**Labels**: bug, stale, v1

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
Clang version: 19.1.7
CMake version: version 3.31.4
Libc version: glibc-2.35

Python version: 3.11.10 (main, Oct 16 2024, 04:38:48) [Clang 18.1.8 ] (64-bit runtime)
Python platform: Linux-5.15.0-134-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A100-SXM4-80GB
Nvidia driver version: 565.57.01
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                        46 bits physical, 48 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Intermittent CUDA IMA in V1 CI tests

**Link**: https://github.com/vllm-project/vllm/issues/14777
**State**: closed
**Created**: 2025-03-13T18:55:58+00:00
**Closed**: 2025-03-14T16:29:00+00:00
**Comments**: 2
**Labels**: bug, v1

### Description

### Your current environment

CI

### 🐛 Describe the bug


```
Processed prompts:   0% 0/100 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]ERROR 03-13 08:00:58 [core.py:337] EngineCore hit an exception: Traceback (most recent call last):
[2025-03-13T08:00:58Z] ERROR 03-13 08:00:58 [core.py:337]   File "/opt/venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 330, in run_engine_core
[2025-03-13T08:00:58Z] ERROR 03-13 08:00:58 [core.py:337]     engine_core.run_busy_loop()
[2025-03-13T08:00:58Z] ERROR 03-13 08:00:58 [core.py:337]   File "/opt/venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 364, in run_busy_loop
[2025-03-13T08:00:58Z] ERROR 03-13 08:00:58 [core.py:337]     outputs = step_fn()
[2025-03-13T08:00:58Z] ERROR 03-13 08:00:58 [core.py:337]               ^^^^^^^^^
[2025-03-13T08:00:58Z] ERROR 03-13 08:00:58 [core.py:337]   File "/opt/venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 192, in step
[2025-03-13T08:00:58Z]

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: [V1] Support `MistralTokenizer` on V1

**Link**: https://github.com/vllm-project/vllm/issues/14522
**State**: closed
**Created**: 2025-03-09T19:38:30+00:00
**Closed**: 2025-03-12T02:40:10+00:00
**Comments**: 1
**Labels**: structured-output, usage, v1

### Description

### Your current environment

Currently, Xgrammar does not work with `MistralTokenizer`. This means we cannot use `MistralTokenizer` with V1

### How would you like to use vllm

I want to run inference of a [specific model](put link here). I don't know how to integrate it with vllm.


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [V1][Bug]: TP with Ray does not terminate gracefully

**Link**: https://github.com/vllm-project/vllm/issues/13437
**State**: closed
**Created**: 2025-02-17T23:37:17+00:00
**Closed**: 2025-02-19T17:40:51+00:00
**Comments**: 3
**Labels**: bug, ray, v1

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```

</details>


### 🐛 Describe the bug

When using Ray as the distributed executor backend and using the `LLM` Python API , the main process does not terminate gracefully:

```
*** SIGTERM received at time=1739834838 on cpu 88 ***
PC: @     0x7fe108d1f117  (unknown)  (unknown)
    @     0x7fe108cd0520  (unknown)  (unknown)
[2025-02-17 15:27:18,341 E 2669821 2669821] logging.cc:460: *** SIGTERM received at time=1739834838 on cpu 88 ***
[2025-02-17 15:27:18,341 E 2669821 2669821] logging.cc:460: PC: @     0x7fe108d1f117  (unknown)  (unknown)
[2025-02-17 15:27:18,341 E 2669821 2669821] logging.cc:460:     @     0x7fe108cd0520  (unknown)  (unknown)
2025-02-17 15:27:18,342 INFO compiled_dag_node.py:1867 -- Tearing down compiled DAG
2025-02-17 15:27:18,342 INFO compiled_dag_node.py:1872 -- Cancelling compiled worker on actor: Actor(RayWorkerW

[... truncated for brevity ...]

---

## Issue #N/A: [V1][Help Wanted] Porting missing sampling parameters to V1

**Link**: https://github.com/vllm-project/vllm/issues/13058
**State**: closed
**Created**: 2025-02-10T23:13:42+00:00
**Closed**: 2025-03-20T14:15:16+00:00
**Comments**: 10
**Labels**: good first issue, misc, v1

### Description

### Anything you want to discuss about vllm.

To switch the engine from V0 to V1, we need to comprehensively support the sampling parameters in https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py

While most of the key parameters are already supported, some of them are missing:

TODO (help wanted):
- [x] `n` (parallel sampling) #10980  @afeldman-nm 
- [x] `guided_decoding` (structured decoding) #12388  @aarnphm 
- [x] `logit_bias` #13079 @houseroad 
- [x] `min_p` #13191 @AoyuQC
- [ ] `bad_words` (originally implemented via logits processor) #13376 @22quinn 
- [x] `allowed_token_ids` (originally implemented via logits processor) #13210 @houseroad 

Parameters that will not be supported in V1:
* best_of
* logits_processors


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequentl

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vllm v1: RuntimeError: Cannot re-initialize CUDA in forked subprocess

**Link**: https://github.com/vllm-project/vllm/issues/12754
**State**: closed
**Created**: 2025-02-04T23:27:14+00:00
**Closed**: 2025-02-13T18:30:01+00:00
**Comments**: 3
**Labels**: bug, v1

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 02-04 23:23:46 __init__.py:186] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.31.4
Libc version: glibc-2.35

Python version: 3.12.8 (main, Jan 14 2025, 22:49:14) [Clang 19.1.6 ] (64-bit runtime)
Python platform: Linux-5.15.0-122-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA H200
Nvidia driver version: 550.90.12
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.9.7
/usr/lib/x86_64-linux-gnu/li

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: V1 Engine Non-Coherent output

**Link**: https://github.com/vllm-project/vllm/issues/12741
**State**: closed
**Created**: 2025-02-04T16:29:10+00:00
**Closed**: 2025-02-22T16:45:38+00:00
**Comments**: 11
**Labels**: bug, v1

### Description

### Your current environment

Vllm 0.7.1
CUDA 12.6
Driver Version 560.94
torch 2.5.1
transformers 4.46.0


### 🐛 Describe the bug

When using V1 engine, the output is noncoherent. 
For examples:
".SetString that a I. to . v the . the v the on the this . p et is Jansw to A 2000 v and . tochemically the I in we ,  (self. . v or on we Fnew, international lawrence of for000,   __in U, do — and it - arse not  C is images that super()000 the use a . v the . v to000 an A for to000 with from in g be for . "
I have tried various changes including changing temp, frequency_penalty, etc. I have tried various models, and nothing improved.
Here is what I use
vllm serve nm-testing/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic  --host 0.0.0.0   --port 8000   --tensor-parallel-size 8  --seed 1234  --max-model-len 16000  --enable-auto-tool-choice --tool-call-parser llama3_json --chat-template /home/nd600/miniconda3/envs/vllm/lib/python3.12/site-packages/vllm/examples/tool_chat_template_llama3.1_json.jinja
T

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: V1 support Xformers

**Link**: https://github.com/vllm-project/vllm/issues/12724
**State**: closed
**Created**: 2025-02-04T09:05:37+00:00
**Closed**: 2025-06-06T02:18:19+00:00
**Comments**: 10
**Labels**: feature request, stale, v1

### Description

### 🚀 The feature, motivation and pitch

I tried to use V1 engine to load Qwen2.5-72B-Instruct-GPTQ-int4 on 4*2080ti 22g, and it raised an AssertionError. The error message shows assert is_fa_version_supported(self.fa_version) caused this AssertionError. It seems like the 2080ti does not support any version of FA. In the V0 engine, it uses the Xformers backend and works fine. However, in V1, it raises an error and stops working. So, I would like to request Xformers support for the V1 engine. I know the 2080ti is a bit outdated, but it is the only choice for getting a large GPU memory at an acceptable price. I really appreciate your help with this. It would mean a lot.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: V1 engine ignores guided json

**Link**: https://github.com/vllm-project/vllm/issues/12692
**State**: closed
**Created**: 2025-02-03T14:01:44+00:00
**Closed**: 2025-06-06T02:18:22+00:00
**Comments**: 5
**Labels**: bug, stale, v1

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 02-03 05:55:13 __init__.py:183] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.31.4
Libc version: glibc-2.35

Python version: 3.12.8 (main, Dec  4 2024, 08:54:12) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.8.0-52-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA H100 80GB HBM3
Nvidia driver version: 560.35.05
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: V1 cannot be run in Triton Inference Server Backend

**Link**: https://github.com/vllm-project/vllm/issues/12690
**State**: closed
**Created**: 2025-02-03T13:28:06+00:00
**Closed**: 2025-03-25T12:16:53+00:00
**Comments**: 3
**Labels**: bug, v1

### Description

### Your current environment

. NA

### Model Input Dumps

_No response_

### 🐛 Describe the bug

When attempting to use the `VLLM_USE_V1=1` feature in triton inference server backend the models fail to start up due to signal handling being attempted outside of the main thread.

The following error occurs in startup.

```text
model.py:244] "[vllm] Failed to start engine: signal only works in main thread of the main interpreter"
pb_stub.cc:366] "Failed to initialize Python stub: ValueError: signal only works in main thread of the main interpreter

At:
  /usr/lib/python3.12/signal.py(58): signal
  /app/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py(160): __init__
  /app/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py(252): __init__
  /app/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py(53): make_client
  /app/.venv/lib/python3.12/site-packages/vllm/v1/engine/async_llm.py(79): __init__
  /app/.venv/lib/python3.12/site-packages/vllm/v1/en

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: V1 engine ignores logits processors and min-p sampling

**Link**: https://github.com/vllm-project/vllm/issues/12678
**State**: closed
**Created**: 2025-02-03T06:43:57+00:00
**Closed**: 2025-07-04T02:18:21+00:00
**Comments**: 5
**Labels**: bug, stale, v1

### Description

### Your current environment

**vLLM Version**: 0.7.0

### Model Input Dumps

_No response_

### 🐛 Describe the bug

# Issue: V1 engine ignores custom logits processors **and** does not implement min-p sampling

**Problem**  
1. **Custom logits processors**: In the new V1 engine, specifying a `logits_processor` in `SamplingParams` for `LLM.generate()` has no effect. The code in [`gpu_model_runner.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py) never passes any sampling metadata into `self.model.compute_logits(...)`, so the logits processor is silently ignored.

2. **Min-p**: Similarly, `min_p` (a sampling parameter supported in V0 akin to `top_k` and `top_p`) is not applied at all in V1. The [`sampler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/sampler.py) for the new engine appears to skip it entirely, so it never factors into the final token selection.

If those features are not yet supported, consider at least raising a 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: [V1] New v1 engine does not support n>1?

**Link**: https://github.com/vllm-project/vllm/issues/12584
**State**: closed
**Created**: 2025-01-30T18:24:17+00:00
**Closed**: 2025-04-22T03:40:20+00:00
**Comments**: 2
**Labels**: bug, v1

### Description

### Your current environment

VLLM version 0.7.0

### Model Input Dumps

_No response_

### 🐛 Describe the bug

When using v1 engine, `LLM.generate()` only returns 1 `CompletionOutput` even when `SamplingParams` sets `n>1`

Is this expected to work or is `n>1` not yet supported for v1? If so, are there plans to support it? 

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Performance]: V1 higher memory usage

**Link**: https://github.com/vllm-project/vllm/issues/12529
**State**: closed
**Created**: 2025-01-28T22:23:50+00:00
**Closed**: 2025-03-14T03:40:24+00:00
**Comments**: 22
**Labels**: performance, v1

### Description

### Proposal to improve performance

_No response_

### Report of performance regression

*Hardware:* 4x RTX 3070 = 32GB VRAM

*Issue:* I was able to run `Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4` with 12K context length with 0.6.x, now with `0.7.0 + VLLM_USE_V1=1` I cannot push the context length higher than 3K or encountering a CUDA OOM error. 
Of course, I can reconfigure it to avoid OOM, my question is: *Is V1 expected to consume more memory?*


Some of the libraries:
```
flashinfer==0.1.6+cu124torch2.4
torch==2.5.1
transformers==4.48.1
vllm==0.7.0
```

*VLLM command*
```
        - vllm
        - serve
        - Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4
        - --gpu-memory-utilization=1
        - --tensor-parallel-size=4
        - --load-format=auto
        - --enforce-eager
        - --swap-space=0
        - --max-model-len=12K
        - --max-num-batched-tokens=12K
        - --disable-fastapi-docs
        - --trust-remote-code
        - --enable-auto-tool-choice
        - --t

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: [V1] TPU support and multiple architecture support

**Link**: https://github.com/vllm-project/vllm/issues/12480
**State**: closed
**Created**: 2025-01-27T18:31:51+00:00
**Closed**: 2025-06-05T02:12:54+00:00
**Comments**: 7
**Labels**: RFC, stale, v1

### Description

### Motivation.

We are in process of adding Google TPU support to the vLLM V1. 

Here is the WIP PR [https://github.com/vllm-project/vllm/pull/11936](https://github.com/vllm-project/vllm/pull/11936). 

Since this is the first time we add another hardware backend to V1, the PR has some refactor to avoid code duplications, which requires discussion and feedback.



### Proposed Change.

Here is the summary of changes this PR introduces:

1. Refactors the common logic of model_runner to **model_runner_base.py** in the folllowing way (Virtual functions in italic):
       \_\_init\_\_() => Has common config init
       get_model() => Just simply returns model
       get_kv_cache_spec() => Common logic for KV cache management
       _initialize_kv_cache()_ => Virtual API
       _execute_model()_ => Virtual API
       _load_model()_ => Virtual API
       _dummy_run()_ => Virtual API
       _profile_run()_ => Virtual API
       _capture_model()_ => Virtual API

2. Refactors common logic of wo

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Pipeline-Parallelism for vLLM V1

**Link**: https://github.com/vllm-project/vllm/issues/11945
**State**: closed
**Created**: 2025-01-10T23:01:10+00:00
**Closed**: 2025-05-09T02:14:11+00:00
**Comments**: 19
**Labels**: RFC, stale, v1

### Description

### Motivation.

This RFC describes the approach for supporting pipeline parallelism in [vLLM V1 architecture](https://github.com/vllm-project/vllm/issues/8779).

Pipeline parallelism was [supported in V0 with the virtual-engine approach](https://github.com/vllm-project/vllm/issues/4461). In short, we create multiple virtual engines to match the number of pipeline stages, and each virtual engine has its own scheduler, block manager and cache engine, so that they can schedule multiple batches simultaneously to the same executor with pipeline parallelism, saturating all pipeline stages to improve the efficiency. However, virtual engine introduces the following drawbacks:

1. The lack of a centralized scheduler prevents global optimization from being applied.
2. It introduces complexity to the engine architecture and implementation.

In this RFC, we aim to support pipeline parallelism in the V1 LLMEngineCore, with the following properties: 

- Good performance: throughput and TTF

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Implement Structured Output support for V1 engine

**Link**: https://github.com/vllm-project/vllm/issues/11908
**State**: closed
**Created**: 2025-01-09T23:41:04+00:00
**Closed**: 2025-03-10T13:28:30+00:00
**Comments**: 2
**Labels**: structured-output, RFC, v1

### Description

### Motivation.

Structured Output is supported in v0, but not yet in v1. One reason for the delay is there have been performance challenges with the integration in v0, and we'd like to rethink the integration approach. We would also like to account for supporting additional techniques, jump decoding in particular, in the future.

The document below covers the proposed integration of the Structured Output functionality in V1 of the vLLM engine.


### Proposed Change.

A draft proposal can be found in this google doc: https://docs.google.com/document/d/1H6m_Y3FLJ1FYGCmjXdZzoJv-JCDSxnKuSY2XiAj-c6c/edit?tab=t.0

This content will eventually be moved into a PR as an addition to the design docs section of the vllm docs.

Related issue for closing xgrammar feature gaps: https://github.com/vllm-project/vllm/issues/12131

### Feedback Period.

_No response_

### CC List.

@mgoin @aarnphm @markmc @simon-mo @xuechendi @WoosukKwon 

### Any Other Things.

_No response_

### Before submitting a ne

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Response format extensions for structured outputs

**Link**: https://github.com/vllm-project/vllm/issues/19097
**State**: open
**Created**: 2025-06-03T17:05:05+00:00
**Comments**: 11
**Labels**: structured-output, RFC, v1

### Description

### Motivation.

Currently, users can provide additional constraints format via `extra_body` in OpenAI client:

```python
from enum import Enum
from pydantic import BaseModel
from openai import OpenAI

simplified_sql_grammar = """
        root ::= select_statement

        select_statement ::= "SELECT " column " from " table " where " condition

        column ::= "col_1 " | "col_2 "

        table ::= "table_1 " | "table_2 "

        condition ::= column "= " number

        number ::= "1 " | "2 "
    """

prompt = (
        "Generate an SQL query to show the 'username' and 'email'"
        "from the 'users' table."
    )

completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        extra_body={"guided_grammar": simplified_sql_grammar},
```

This also applies with `guided_json`, `structural_tag`, `guided_regex`.

While this is pretty convenient for 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Missing metrics  in V1

**Link**: https://github.com/vllm-project/vllm/issues/16348
**State**: open
**Created**: 2025-04-09T14:38:15+00:00
**Comments**: 13
**Labels**: bug, v1

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Debian GNU/Linux 11 (bullseye) (x86_64)
GCC version: (Debian 10.2.1-6) 10.2.1 20210110
Clang version: Could not collect
CMake version: version 3.18.4
Libc version: glibc-2.31

Python version: 3.10.16 | packaged by conda-forge | (main, Dec  5 2024, 14:16:10) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-5.10.0-34-cloud-amd64-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA L4
GPU 1: NVIDIA L4

Nvidia driver version: 550.90.07
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Byte Order:   

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: CI flake - v1/engine/test_async_llm.py::test_abort - assert has_unfinished_requests()

**Link**: https://github.com/vllm-project/vllm/issues/16054
**State**: open
**Created**: 2025-04-04T09:48:13+00:00
**Comments**: 1
**Labels**: bug, ci/build, stale, v1

### Description

### Your current environment

...

### 🐛 Describe the bug

main commit 51d7c6a2b23e100cd9e7d85b8e7c0eea656b331e

Seen in https://github.com/vllm-project/vllm/pull/15894

https://buildkite.com/organizations/vllm/pipelines/ci/builds/16742/jobs/0195f24d-e81a-46a3-ad08-6a51983d65d6/log


```
=================================== FAILURES ===================================
[2025-04-01T17:38:12Z] _ test_abort[engine_args0-Hello my name is Robert and-RequestOutputKind.DELTA] _
[2025-04-01T17:38:12Z]
[2025-04-01T17:38:12Z] monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7fd1fa052e70>
[2025-04-01T17:38:12Z] output_kind = <RequestOutputKind.DELTA: 1>
[2025-04-01T17:38:12Z] engine_args = AsyncEngineArgs(model='meta-llama/Llama-3.2-1B-Instruct', served_model_name=None, tokenizer='meta-llama/Llama-3.2-1B-I...additional_config=None, enable_reasoning=None, reasoning_parser=None, use_tqdm_on_load=True, disable_log_requests=True)
[2025-04-01T17:38:12Z] prompt = 'Hello my name is Robert and'
[

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: CI flake - v1/entrypoints/llm/test_struct_output_generate.py::test_structured_output - JSONDecodeError: Expecting value: line 1 column 1 (char 0)

**Link**: https://github.com/vllm-project/vllm/issues/16053
**State**: open
**Created**: 2025-04-04T09:46:16+00:00
**Comments**: 2
**Labels**: bug, ci/build, stale, v1

### Description

### Your current environment

...


### 🐛 Describe the bug

main commit 51d7c6a2b23e100cd9e7d85b8e7c0eea656b331e

Seen in https://github.com/vllm-project/vllm/pull/15894

https://buildkite.com/organizations/vllm/pipelines/ci/builds/16742/jobs/0195fc58-3d11-45b5-b76f-8e962cbda765/log

```
FAILED v1/entrypoints/llm/test_struct_output_generate.py::test_structured_output[Qwen/Qwen2.5-1.5B-Instruct-guidance:disable-any-whitespace-auto] - json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)

[2025-04-03T16:08:35Z] _ test_structured_output[Qwen/Qwen2.5-1.5B-Instruct-guidance:disable-any-whitespace-auto] _
[2025-04-03T16:08:35Z]
[2025-04-03T16:08:35Z] monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f318d89eb40>
[2025-04-03T16:08:35Z] sample_json_schema = {'properties': {'age': {'type': 'integer'}, 'name': {'type': 'string'}, 'skills': {'items': {'type': 'string'}, 'type'...ition'], 'type': 'object'}, 'type': 'array'}}, 'required': ['name', 'age', 'skills', 'work_

[... truncated for brevity ...]

---

## Issue #N/A: [SpecDecode] Support EAGLE in V1

**Link**: https://github.com/vllm-project/vllm/issues/15901
**State**: open
**Created**: 2025-04-01T19:45:13+00:00
**Comments**: 21
**Labels**: speculative-decoding, v1

### Description

- [x] 1. Correctly initializing and loading the EAGLE draft model
- [x] 2. Consider the lookahead slots in the KV cache manager
- [x] 3. Cache `draft_probs` inside the model runner and correctly feed it to the rejection sampler in the next step (temporarily workaround: #16899)
- [x] 4. Handle the edge cases like when the draft model generates beyond `max_pos_embeddings`
- [ ] 5. Handle the seeds correctly
- [ ] 6. Do E2E correctness and performance tests
- [x] 7. Support prefix caching. Eagle requires special handling because Eagle's i-th KV cache is coupled with the i+1-th token ID. (@LiuXiaoxuanPKU)
- [ ] 8. Properly handle the sampling parameters that are not (currently) compatible with spec decoding (e.g., min_p).
- [x] 9. Use CUDA graphs for draft model. (@luyuzhe111)
- [x] 10. Support Eagle 3 (https://github.com/vllm-project/vllm/pull/16937)

_Originally posted by @WoosukKwon in https://github.com/vllm-project/vllm/issues/15729#issuecomment-2765192455_
            

---

## Issue #N/A: [Bug]: CI flake - v1/engine/test_llm_engine.py::test_parallel_sampling[True]

**Link**: https://github.com/vllm-project/vllm/issues/15855
**State**: open
**Created**: 2025-04-01T06:55:01+00:00
**Comments**: 4
**Labels**: bug, ci/build, stale, v1

### Description

### Your current environment

...


### 🐛 Describe the bug

Saw V1 test failing with this yesterday, went away with recheck:

```
[2025-03-31T17:33:47Z] _________________________ test_parallel_sampling[True] _________________________
[2025-03-31T17:33:47Z]
[2025-03-31T17:33:47Z] vllm_model = <tests.conftest.VllmRunner object at 0x7f0d875e06e0>
[2025-03-31T17:33:47Z] example_prompts = ['vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.\n', 'Briefly describe the majo...me.\n', 'Analyze the impact of the COVID-19 pandemic on global economic structures and future business models.\n', ...]
[2025-03-31T17:33:47Z]
[2025-03-31T17:33:47Z]     def test_parallel_sampling(vllm_model, example_prompts) -> None:
[2025-03-31T17:33:47Z]         """Test passes if parallel sampling `n>1` yields `n` unique completions.
[2025-03-31T17:33:47Z]
[2025-03-31T17:33:47Z]         Args:
[2025-03-31T17:33:47Z]           vllm_model: VllmRunner instance under test.
[2025-03-31T17:3

[... truncated for brevity ...]

---

## Issue #N/A: [V1] Feedback Thread

**Link**: https://github.com/vllm-project/vllm/issues/12568
**State**: open
**Created**: 2025-01-30T02:46:45+00:00
**Comments**: 92
**Labels**: v1

### Description

Please leave comments here about your usage of V1, does it work? does it not work? which feature do you need in order to adopt it? any bugs? 

For bug report, please file it separately and link the issue here. 

For in depth discussion, please feel free to join #sig-v1 in the vLLM Slack workspace. 


---

