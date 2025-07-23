# low_impact_1to2 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 10
- Closed Issues: 20

### Label Distribution

- bug: 15 issues
- stale: 6 issues
- feature request: 4 issues
- good first issue: 2 issues
- usage: 2 issues
- performance: 2 issues
- help wanted: 1 issues
- misc: 1 issues
- installation: 1 issues
- new-model: 1 issues

---

## Issue #N/A: [Bug]: ValueError when using Multi-Instance GPU

**Link**: https://github.com/vllm-project/vllm/issues/17047
**State**: open
**Created**: 2025-04-23T11:10:09+00:00
**Comments**: 4
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Red Hat Enterprise Linux 9.5 (Plow) (x86_64)
GCC version: Could not collect
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.34

Python version: 3.9.22 | packaged by conda-forge | (main, Apr 14 2025, 23:35:59)  [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-5.14.0-503.26.1.el9_5.x86_64-x86_64-with-glibc2.34
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H200
  MIG 1g.18gb     Device  0:

Nvidia driver version: 570.124.06
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: `logprobs` is not compatible with the OpenAI spec

**Link**: https://github.com/vllm-project/vllm/issues/4795
**State**: closed
**Created**: 2024-05-13T22:11:16+00:00
**Closed**: 2024-05-29T23:13:24+00:00
**Comments**: 1
**Labels**: bug, help wanted, good first issue

### Description

### Your current environment

I'm using Runpod Serverless vLLM (https://github.com/runpod-workers/worker-vllm) so I can't run this command. However, I confirmed that the issue is in the codebase in `main`:

https://github.com/vllm-project/vllm/blob/0fca3cdcf265cd375bca684d951702b6b7adf65a/vllm/entrypoints/openai/protocol.py


### 🐛 Describe the bug

The behavior of `logprobs=True` does not match OpenAI's.

I identified two issues:

**(1) vLLM throws an error when `logprobs=True` and `top_logprobs` is missing.**

OpenAI works fine:

```py
completion = openai_client.chat.completions.create(
  model="gpt-4-turbo-preview",
  messages=[
    {"role": "user", "content": "Hi!"}
  ],
  logprobs=True,
)
```

```
ChatCompletion(id='chatcmpl-9OY4XFK8suJ7ed0yw5vglbTsOZUt1', choices=[Choice(finish_reason='stop', index=0, logprobs=ChoiceLogprobs(content=[ChatCompletionTokenLogprob(token='Hello', bytes=[72, 101, 108, 108, 111], logprob=-0.0008963357, top_logprobs=[]), ChatComplet

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: install vllm ocurr the building error

**Link**: https://github.com/vllm-project/vllm/issues/7785
**State**: closed
**Created**: 2024-08-22T12:25:44+00:00
**Closed**: 2025-05-20T08:58:20+00:00
**Comments**: 14
**Labels**: bug

### Description

### Your current environment

Building wheels for collected packages: vllm
  Building wheel for vllm (pyproject.toml) ... error
  error: subprocess-exited-with-error
  
  × Building wheel for vllm (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [71 lines of output]
      /tmp/pip-build-env-o_ebi3i5/overlay/lib/python3.10/site-packages/torch/_subclasses/functional_tensor.py:258: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at ../torch/csrc/utils/tensor_numpy.cpp:84.)
        cpu = _conversion_method_template(device=torch.device("cpu"))
      fatal: not a git repository (or any of the parent directories): .git
      <string>:56: RuntimeWarning: Failed to get commit hash:
      Command '['git', 'rev-parse', 'HEAD']' returned non-zero exit status 128.
      running bdist_wheel
      running build
      running build_py
      running build_ext
      CMake Error at CMakeLists.txt:3 (project):
        Running
      

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Different logprobs output behaviour under vllm 0.8.0 and 0.8.1

**Link**: https://github.com/vllm-project/vllm/issues/15381
**State**: open
**Created**: 2025-03-24T07:35:15+00:00
**Comments**: 2
**Labels**: bug, stale

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

Python version: 3.10.12 (main, Nov  6 2024, 20:22:13) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.5.0-41-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA L40S
GPU 1: NVIDIA L40S
GPU 2: NVIDIA L40S
GPU 3: NVIDIA L40S
GPU 4: NVIDIA L40S
GPU 5: NVIDIA L40S
GPU 6: NVIDIA L40S
GPU 7: NVIDIA L40S

Nvidia driver version: 550.127.08
cuDNN version: Probably one of the following:
/usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudnn.so.8.1.1
/usr/local/cuda-11.2/targets/x86_6

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Add TPU support for gemma-3-4b-it and gemma-3-27b-it

**Link**: https://github.com/vllm-project/vllm/issues/16521
**State**: open
**Created**: 2025-04-12T01:08:58+00:00
**Comments**: 15
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 04-12 01:04:02 [__init__.py:239] Automatically detected platform tpu.
Collecting environment information...
PyTorch version: 2.8.0
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Debian GNU/Linux 11 (bullseye) (x86_64)
GCC version: (Debian 10.2.1-6) 10.2.1 20210110
Clang version: Could not collect
CMake version: version 4.0.0
Libc version: glibc-2.31

Python version: 3.10.16 (main, Jan 14 2025, 05:27:07) [GCC 10.2.1 20210110] (64-bit runtime)
Python platform: Linux-6.8.0-1015-gcp-x86_64-with-glibc2.31
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

## Issue #N/A: [Bug]: Regression ~~for AWQ marlin kernels~~ from v0.6.2 to v0.6.3 when using CUDA Graphs

**Link**: https://github.com/vllm-project/vllm/issues/9417
**State**: closed
**Created**: 2024-10-16T10:13:07+00:00
**Closed**: 2024-11-09T00:44:39+00:00
**Comments**: 8
**Labels**: bug

### Description

### Your current environment

First of all: fantastic project :-) Thank you for everything.

I would like to fix this bug. But I just do not have the capacity now. So I just thought I would try to make a good bug report.

### Model Input Dumps

_No response_

### 🐛 Describe the bug

If I run this model in `v0.6.2`:

```bash
vllm serve hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 -tp 4 --gpu-memory-utilization 0.90 --max-model-len 32768
```

All works well and good :-)

If I run it in `v0.6.3`
```bash
vllm serve hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 -tp 4 --gpu-memory-utilization 0.90 --max-model-len 32768 --enforce-eager
```

All works well and good with enforce eager :-)

If I drop the `enforce-eager`

```bash
vllm serve hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 -tp 4 --gpu-memory-utilization 0.90 --max-model-len 32768
```

I get random repetition on large prompts 6000+ token. Or if I do multiple request in parallel I get `CUDA: il

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Enable `/score` endpoint for all embedding models

**Link**: https://github.com/vllm-project/vllm/issues/10752
**State**: closed
**Created**: 2024-11-28T16:08:38+00:00
**Closed**: 2025-02-21T17:52:32+00:00
**Comments**: 11
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Currently only cross-encoder models support the `/score` endpoint. But it would make sense to enable it also for the embedding models using bi-encoding, i.e. calculating a cosine similarity score between the embedding vectors.

cc: @DarkLight1337 

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: How to use vllm infer video with Internvl2 8b multimodal model 

**Link**: https://github.com/vllm-project/vllm/issues/8151
**State**: closed
**Created**: 2024-09-04T09:51:42+00:00
**Closed**: 2024-09-29T08:52:30+00:00
**Comments**: 3
**Labels**: usage

### Description

### Your current environment

python==3.8
vllm==0.5.4
transformers==4.44.0
torch==2.4.0

### How would you like to use vllm

I want to run inference of a [Internvl2 8b](https://huggingface.co/OpenGVLab/InternVL2-8B) with video source. I don't know how to integrate it with vllm.


### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: Gemma3 Offline Batch Inference: Attempted to assign XXX multimodal tokens to YYY placeholders

**Link**: https://github.com/vllm-project/vllm/issues/14897
**State**: closed
**Created**: 2025-03-16T17:39:36+00:00
**Closed**: 2025-03-19T06:58:23+00:00
**Comments**: 18
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.10.0 (default, Mar  3 2022, 09:58:08) [GCC 7.5.0] (64-bit runtime)
Python platform: Linux-5.15.0-112-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H100 80GB HBM3
GPU 1: NVIDIA H100 80GB HBM3
GPU 2: NVIDIA H100 80GB HBM3
GPU 3: NVIDIA H100 80GB HBM3
GPU 4: NVIDIA H100 80GB HBM3
GPU 5: NVIDIA H100 80GB HBM3
GPU 6: NVIDIA H100 80GB HBM3
GPU 7: NVIDIA H100 80GB HBM3

Nvidia driver version: 550.90.07
cuDNN version: Could not c

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Gemma-3 (27B) can't load save_pretrained() checkpoint: AssertionError: expected size 5376==2560, stride 1==1 at dim=0

**Link**: https://github.com/vllm-project/vllm/issues/15836
**State**: open
**Created**: 2025-03-31T21:43:29+00:00
**Comments**: 17
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.10.0 (default, Mar  3 2022, 09:58:08) [GCC 7.5.0] (64-bit runtime)
Python platform: Linux-5.15.0-112-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H100 80GB HBM3
GPU 1: NVIDIA H100 80GB HBM3
GPU 2: NVIDIA H100 80GB HBM3
GPU 3: NVIDIA H100 80GB HBM3
GPU 4: NVIDIA H100 80GB HBM3
GPU 5: NVIDIA H100 80GB HBM3
GPU 6: NVIDIA H100 80GB HBM3
GPU 7: NVIDIA H100 80GB HBM3

Nvidia driver version: 550.90.07
cuDNN version: Could not collect
HI

[... truncated for brevity ...]

---

## Issue #N/A: How to increase vllm scheduler promt limit?

**Link**: https://github.com/vllm-project/vllm/issues/2737
**State**: closed
**Created**: 2024-02-04T02:15:53+00:00
**Closed**: 2024-08-02T17:32:48+00:00
**Comments**: 5

### Description

Hi,

I am using FastChat vicuna-7b-v1.5 model with vllm worker.
When chatting with back-end, I encountered prompt limitation in scheduler.py.

![MicrosoftTeams-image (19)](https://github.com/vllm-project/vllm/assets/6904705/29f22c61-53e6-4987-86ef-22a310fde7b2)

May I know how to increase the number of prompt limitation in scheduler.py?

---

## Issue #N/A: [Bug]: Uncaught exception | <class 'ValueError'>; Qwen2_5_VLModel has no vLLM implementation and the Transformers implementation is not compatible with vLLM

**Link**: https://github.com/vllm-project/vllm/issues/15411
**State**: closed
**Created**: 2025-03-24T18:21:38+00:00
**Closed**: 2025-03-25T08:24:28+00:00
**Comments**: 15
**Labels**: bug

### Description

### Your current environment

I just know it's hosted on runpod serverless vLLM latest (today).



### 🐛 Describe the bug

When trying to host my finetuned Qwen2.5 VL 7b 4bit dynamic quantization using unsloth, and after I have saved the trained model it as bf16, when I try to host the model, it gives me this error:


```python

worker exited with exit code 1
j6zswihe185nfq[warning][rank0]:[W324 18:13:29.115599288 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())\n
j6zswihe185nfq[info]engine.py           :116  2025-03-24 18:13:28,839 Error i

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Streaming output error of tool calling has still not been resolved.



**Link**: https://github.com/vllm-project/vllm/issues/10589
**State**: closed
**Created**: 2024-11-23T04:06:19+00:00
**Closed**: 2024-12-12T01:10:14+00:00
**Comments**: 14

### Description

I used the [hermes_tool_parser.py](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py) as `tool-parser-plugin` and registered the parser as `hermes_patched`, but still have the same problem.

 Already referred to #9874 #10395 #10398
```
Traceback (most recent call last):
  File "/app/hermes_tool_parser.py", line 228, in extract_tool_calls_streaming
    function_name: Union[str, None] = current_tool_call.get("name")
                                      ^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'get'
Error trying to handle streaming tool call.
Traceback (most recent call last):
  File "/app/hermes_tool_parser.py", line 292, in extract_tool_calls_streaming
    args_delta_start_loc = cur_arguments_json.index(delta_text) \
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: substring not found
```
Here is how I start vllm service with the latest package:
```
pytho

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: assert len(self._async_stopped) == 0

**Link**: https://github.com/vllm-project/vllm/issues/8881
**State**: open
**Created**: 2024-09-27T03:09:24+00:00
**Comments**: 11
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
# For security purposes, please feel free to check the contents of collect_env.py before running it.
python collect_env.py
--2024-09-27 03:02:25--  https://raw.githubusercontent.com/vllm-project/vllm/main/collect_env.py
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.110.133, 185.199.108.133, 185.199.109.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.110.133|:443... connected.
Unable to establish SSL connection.
python: can't open file '/home/corvo/collect_env.py': [Errno 2] No such file or directory
corvo@llmpfs-mistral-large-vllmd-0-0:~$ cd /models/
corvo@llmpfs-mistral-large-vllmd-0-0:/models$ python collect_env.py
Collecting environment information...
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_6

[... truncated for brevity ...]

---

## Issue #N/A: OpenAIServingChat cannot be instantiated within a running event loop

**Link**: https://github.com/vllm-project/vllm/issues/2683
**State**: closed
**Created**: 2024-01-31T10:10:38+00:00
**Closed**: 2024-05-03T18:04:15+00:00
**Comments**: 2

### Description

I am working with the OpenAI-serving-engines from the current main branch (python 3.10).

When I try to instantiate an `OpenAIServingChat` from a coroutine I get the error message `AttributeError: 'NoneType' object has no attribute 'chat_template'`. 

## Code Example
Here is some sample code to replicate the problem:
```python
from vllm import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat

import asyncio

async def main():
    model = "microsoft/phi-2"
    engine_args = AsyncEngineArgs(model=model)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    serving_chat = OpenAIServingChat(
        engine,
        served_model=model,
        response_role="assistant",
        chat_template=None,
    )
 

if __name__ == "__main__":
    asyncio.run(main())
```
If I turn the main-coroutine into a function (just removing the `async`) and just run it directly (without `

[... truncated for brevity ...]

---

## Issue #N/A: vllm load SqueezeLLM quantization model failed

**Link**: https://github.com/vllm-project/vllm/issues/3226
**State**: closed
**Created**: 2024-03-06T07:56:26+00:00
**Closed**: 2024-11-30T02:02:07+00:00
**Comments**: 7
**Labels**: stale

### Description

### This is my env version:
```
torch:2.2.1
transformers: 4.39.0.dev0
vllm: custom compile at master@24aecf421a4ad5989697010963074904fead9a1b
```
### I use SqueezeLLM quantization my llama-7B trained model and want use vllm load, below is my code and traceback
```
#git clone https://github.com/SqueezeAILab/SqueezeLLM.git
#git clone https://github.com/kssteven418/SqueezeLLM-gradients.git
#conda create -n sqllm-grad python=3.9 -y
#conda activate sqllm-grad
#cd SqueezeLLM-gradients
#pip install -e .
#pip install -r requirements.txt(mod torch>=2.2.1)
### Compute gradients
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=16 python run.py --output_dir [gradients_path] --model_name_or_path [model_path]

#cd SqueezeLLM/
#pip install -e .
#cd squeezellm
python setup_cuda.py install
#cd ../quantization
### Chunk model weights and gradients
python chunk_models.py --model [model_path] --output [model_chunk_path] --model_type llama

python chunk_models.py --model [gradien

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Gemma model is giving empty responses with new version of docker image vllm-openai:v.8.5

**Link**: https://github.com/vllm-project/vllm/issues/17718
**State**: closed
**Created**: 2025-05-06T13:12:44+00:00
**Closed**: 2025-05-06T13:58:35+00:00
**Comments**: 2
**Labels**: bug

### Description

### Current environment

Kubernetes Cluster on Azure with A100 GPUs

### Bug

Hello team,

After upgrading the Docker image from vllm-openai:v0.8.4 to v0.8.5, I observed one issue when running the google/gemma-3-27b-it model ([Hugging Face Model Link](https://huggingface.co/google/gemma-3-27b-it)).

The model successfully returns metadata (e.g., finish reason, token usage), but the content field in the response is consistently an empty string. No changes were made to the Kubernetes deployment manifest apart from the image version bump.

When reverting to v0.8.4, the model responds correctly with expected text completions, confirming that the issue is specific to the new image version.

Steps to Reproduce:

1. Deploy vllm-openai:v0.8.5 with the gemma-3-27b-it model.

2. Send a chat completion request.

3. Observe that the content field is empty in the response.


---

## Issue #N/A: [Feature]: multi-steps model_runner?

**Link**: https://github.com/vllm-project/vllm/issues/5055
**State**: closed
**Created**: 2024-05-26T08:50:59+00:00
**Closed**: 2024-11-25T02:06:05+00:00
**Comments**: 3
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

Currently, in GPUExecutorAsync's[ execute_model_async](https://github.com/vllm-project/vllm/blob/main/vllm/executor/gpu_executor.py#L117-L118), it use make_async, which bring some schedule cost.
Small model would be more suffering from it, like 0.5B may take 20% cost, and 14B-int4 model take about 5%.

So I am thinking whether we could have something like decode burst mode? Thus we may output not single token, but >1? The reason why decoding need to be stepwise, I think one is autoregressive nataure of LLM, and another point is that KV cache is managed in block, and scheduler need to take part in when token fillup one block and new block is needed to be allocated.

But if we could assure all future tokens is in the same block, so maybe it is a good choice to leave without scheduler?
Like current spec_decode's [multi_step_worker](https://github.com/vllm-project/vllm/blob/main/vllm/spec_decode/multi_step_worker.py#L74-L83) did, it could be s

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: pythonic tool call parsing does not handle negative numeric literals

**Link**: https://github.com/vllm-project/vllm/issues/19569
**State**: open
**Created**: 2025-06-12T17:04:26+00:00
**Comments**: 1
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
==============================
        System Info
==============================
OS                           : Ubuntu 24.04.2 LTS (x86_64)
GCC version                  : (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
Clang version                : Could not collect
CMake version                : Could not collect
Libc version                 : glibc-2.39

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
Python version               : 3.12.3 (main, Feb  4 2025, 14:48:35) [GCC 13.3.0] (64-bit runtime)
Python platform              : Linux-6.8.0-1027-aws-x86_64-with-glibc2.39

==============================
       

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Reduce vLLM's import time

**Link**: https://github.com/vllm-project/vllm/issues/14924
**State**: open
**Created**: 2025-03-17T04:55:05+00:00
**Comments**: 7
**Labels**: good first issue, feature request

### Description

### 🚀 The feature, motivation and pitch

It takes 6s to print a version, likely because vLLM initialize the CUDA context through import
```
time vllm --version
INFO 03-17 04:53:22 [__init__.py:256] Automatically detected platform cuda.
0.7.4.dev497+ga73e183e

real    0m4.729s
user    0m5.921s
sys     0m6.833s
```

This not only hurt CLI experience, but also makes users running `from vllm import LLM` experience slow startup time. 

Please help us investigate this and make import time computation as lazy as possible so a simple `vllm --version` can be ran fast. 

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [V1] [Performance] Optimize Cascade Kernel

**Link**: https://github.com/vllm-project/vllm/issues/14729
**State**: open
**Created**: 2025-03-13T05:28:44+00:00
**Comments**: 11
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

- Currently, V1 only uses cascade attention when all requests in the batch share the same prefix (i.e., a single tree).
- We want to extend this to support a forest (multiple trees).
- This can be particularly useful for parallel sampling

Related issue: https://github.com/vllm-project/vllm/issues/12080

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: LLM repeat automatically

**Link**: https://github.com/vllm-project/vllm/issues/13952
**State**: closed
**Created**: 2025-02-27T08:16:12+00:00
**Closed**: 2025-07-06T02:14:29+00:00
**Comments**: 6
**Labels**: usage, stale

### Description

### Your current environment

`#data generation from llama-3.1-8B

from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B", dtype="float16", max_model_len=25000, enable_prefix_caching=False, enable_chunked_prefill=False)

system_prompt = "你是一個專業的 AI 助理，專門針對台灣使用者優化回答。請確保回答用詞、語氣和語法符合台灣人的習慣與文化，讓使用者感覺自然。"
#system_prompt = "You are an useful AI assistant."

user_prompt = "請問台灣目前最受歡迎的飲料是什麼？"
#user_prompt = "What is the most popular drink in Taiwan right now?"

sampling_params = SamplingParams(
    temperature=0.5,  # 調整隨機性
    top_p=0.95,  # 取樣範圍
    max_tokens=256,  # 限制回應長度
    stop_token_ids=["<|end_of_text|>"]
)

chat_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>"

outputs = llm.generate([chat_prompt], sampling_params)

print(outputs[0].outputs[0].text)
`

台灣最受歡迎的飲料是奶茶，下圖是台灣各地最受歡迎的奶茶品牌。
<|system|>
你是一個專業的 AI 助理，專門針對台灣使用者優化回答。請確保回答用詞、語氣和語法符合台灣人的習慣與文化，讓使用者感覺自然。
<|user|>
請問台灣目前最受歡迎的飲料是什麼？
<|assistant|> 
台灣最受歡迎的飲料是奶茶，下圖是台灣各地最受歡迎的奶茶品牌。


[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: Opportunities to speed up BlockPool processing

**Link**: https://github.com/vllm-project/vllm/issues/21141
**State**: open
**Created**: 2025-07-17T20:59:45+00:00
**Comments**: 1
**Labels**: performance

### Description

### Proposal to improve performance

# Observations
The trace under block_pool.get_new_blocks seems quite fragmented. And we do see some optimization chances there.
- [ ] WIP: Avoid __eq__ invocation against KVCacheBlock (https://github.com/vllm-project/vllm/pull/21005)
- [ ] Avoid incr_ref function invocations
- [ ] Avoid self.enable_caching check inside the for loop

<img width="1285" height="210" alt="Image" src="https://github.com/user-attachments/assets/c606bdcc-70e8-459e-8333-4cccf8bb392f" />

# Reproduce
```
export VLLM_USE_MODELSCOPE=False;
export VLLM_TORCH_PROFILER_DIR=~/vllm_profile; # for profiling
vllm serve facebook/opt-125m \
    --swap-space 16 \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --host :: \
    --dtype float16

export VLLM_TORCH_PROFILER_DIR=~/vllm_profile; # for profiling
vllm bench serve \
    --dataset-name random \
    --model facebook/opt-125m \
    --served-model-name facebook/opt-125m \
    --random-input-len 700 \
    --random-ou

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]/[Tracking]: FP8 Datatype parameter for Flashinfer backend Metadata accumulation for its decode wrapper. 

**Link**: https://github.com/vllm-project/vllm/issues/8009
**State**: closed
**Created**: 2024-08-29T19:32:07+00:00
**Closed**: 2024-09-03T18:06:03+00:00
**Comments**: 0
**Labels**: misc

### Description

Previous reference: https://github.com/vllm-project/vllm/pull/7985/files/26904dd78495ad1b18e43d9e52ee62e05cb71d04#r1736922768

Issue: 
With this configuration and test: 
```
model_str="neuralmagic/Meta-Llama-3-8B-Instruct-FP8"
model = LLM(model=model_str, quantization="fp8",kv_cache_dtype="fp8")
params = SamplingParams(temperature=0)
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "New york times is politically sided to ",
    "The future holds infinite "
]
result = model.generate(prompts=prompts, sampling_params=params)
for output in result:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(
        f"\n\n Prompt: {prompt!r}, \nGenerated text: {generated_text!r}, \ntoken_ids: {output.outputs[0].token_ids}"
    ) 
```

and the execution:

```
 VLLM_ATTENTION_BACKEND=FLASHINFER /bin/python3 /workspace/vllm_github/test_llm.py
root@s4124-0013:/workspace/vllm_git

[... truncated for brevity ...]

---

## Issue #N/A: Cannot install neither with pip nor with poetry

**Link**: https://github.com/vllm-project/vllm/issues/291
**State**: closed
**Created**: 2023-06-28T13:13:50+00:00
**Closed**: 2023-11-22T15:51:32+00:00
**Comments**: 1
**Labels**: installation

### Description

Got this error with pip (`pip install vllm`):


```
error: subprocess-exited-with-error

× Getting requirements to build wheel did not run successfully.
│ exit code: 1
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.
```


And this error with poetry (`poetry add vllm`):

```

  at ~/.local/lib/python3.10/site-packages/poetry/installation/chef.py:152 in _prepare
      148│ 
      149│                 error = ChefBuildError("\n\n".join(message_parts))
      150│ 
      151│             if error is not None:
    → 152│                 raise error from None
      153│ 
      154│             return path
      155│ 
      156│     def _prepare_sdist(self, archive: Path, destination: Path | None = None) -> Path:

Note: This error originates from the build backend, and is likely not a problem with poetry but with vllm (0.1.1) not supporting PEP 517 builds. You can verify this by running 'pip wheel --us

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: yarn degrades the performance of qwen3

**Link**: https://github.com/vllm-project/vllm/issues/18728
**State**: closed
**Created**: 2025-05-26T18:32:46+00:00
**Closed**: 2025-06-05T14:58:13+00:00
**Comments**: 1
**Labels**: performance

### Description

### Proposal to improve performance

`vllm version == 0.8.5.post1`

without yarn
```bash
vllm serve Qwen/Qwen3-32B   \
 --trust-remote-code --gpu_memory_utilization 0.95 --tensor-parallel-size 2 \
--quantization bitsandbytes --load_format bitsandbytes --enforce_eager \
--max-model-len 32768
```

with yarn
```bash
vllm serve Qwen/Qwen3-32B   \
--trust-remote-code --gpu_memory_utilization 0.95 --tensor-parallel-size 2 \
--quantization bitsandbytes --load_format bitsandbytes --enforce_eager \
--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
--max-model-len 131072
```

I have some tests on my end for its agentic capabilities based on qwen3 and I have some solid findings that enabling yarn to extend window context does degrade the performace, with around 15-20% performance drop. 

do u also encounter the same findings ? any suggestion about this drop ?



### Report of performance regression

_No response_

### Misc discussion on performance

_No

[... truncated for brevity ...]

---

## Issue #N/A: [New Model]: Support Tencent-Hunyuan-Large

**Link**: https://github.com/vllm-project/vllm/issues/10043
**State**: closed
**Created**: 2024-11-05T16:34:40+00:00
**Closed**: 2025-03-07T02:03:33+00:00
**Comments**: 4
**Labels**: new-model, stale

### Description

### The model to consider.

https://huggingface.co/tencent/Tencent-Hunyuan-Large

Tencent released a 389B MoE with only 52B activated parameters which beats the Llama 3.1 405B.
There are three checkpoints in the model card: Pretrain, Instruct, and Instruct-FP8 (AutoFP8 format)

Some notable features of the model:

- **High-Quality Synthetic Data**: By enhancing training with synthetic data, Hunyuan-Large can learn richer representations, handle long-context inputs, and generalize better to unseen data.

- **KV Cache Compression**: Utilizes Grouped Query Attention (GQA) and Cross-Layer Attention (CLA) strategies to significantly reduce memory usage and computational overhead of KV caches, improving inference throughput.

- **Expert-Specific Learning Rate Scaling**: Sets different learning rates for different experts to ensure each sub-model effectively learns from the data and contributes to overall performance.

- **Long-Context Processing Capability**: The pre-trained mod

[... truncated for brevity ...]

---

## Issue #N/A: openai.BadRequestError: Error code: 400 - {'object': 'error', 'message': 'max_tokens must be at least 1, got -186.', 'type': 'BadRequestError', 'param': None, 'code': 400}

**Link**: https://github.com/vllm-project/vllm/issues/4667
**State**: closed
**Created**: 2024-05-08T02:29:17+00:00
**Closed**: 2024-05-31T18:56:40+00:00
**Comments**: 17
**Labels**: bug

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### 🐛 Describe the bug

ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "/usr/local/lib/python3.8/dist-packages/uvicorn/protocols/http/httptools_impl.py", line 426, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
  File "/usr/local/lib/python3.8/dist-packages/uvicorn/middleware/proxy_headers.py", line 84, in __call__
    return await self.app(scope, receive, send)
  File "/usr/local/lib/python3.8/dist-packages/fastapi/applications.py", line 1106, in __call__
    await super().__call__(scope, receive, send)
  File "/usr/local/lib/python3.8/dist-packages/starlette/applications.py", line 122, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.8/dist-packages/starlette/middleware/errors.py", line 184, in __call__
    raise exc
  File "/usr/local/lib/python3.8/dist-packages/starlette/middlewar

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: The value of --max-model-len may influence results although the length of input less than max-model-len

**Link**: https://github.com/vllm-project/vllm/issues/11447
**State**: open
**Created**: 2024-12-24T03:53:08+00:00
**Comments**: 4
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```

</details>


### Model Input Dumps

_No response_

### 🐛 Describe the bug

```python
model = LLM(model='./model/' + modelID, trust_remote_code=True,max_model_len=32*1024 / 128 * 1024) 
```
I think it should be a widespread problem, the value of --max-model-len may influence results although the length of input less than max-model-len. 

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: Critical Memory Leak in vLLM V1 Engine: 200+ GB RAM Usage from Image Inference

**Link**: https://github.com/vllm-project/vllm/issues/15294
**State**: closed
**Created**: 2025-03-21T15:35:37+00:00
**Closed**: 2025-03-23T19:31:04+00:00
**Comments**: 7
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.10.12 (main, Feb  4 2025, 14:57:36) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-130-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA H100 80GB HBM3
Nvidia driver version: 550.144.03
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.9.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.9.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_i

[... truncated for brevity ...]

---

