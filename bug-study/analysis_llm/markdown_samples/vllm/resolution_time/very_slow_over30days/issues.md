# very_slow_over30days - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- stale: 20 issues
- bug: 12 issues
- feature request: 6 issues
- performance: 3 issues
- usage: 2 issues
- help wanted: 1 issues

---

## Issue #N/A:  [Feature Request] Mixtral Offloading

**Link**: https://github.com/vllm-project/vllm/issues/2394
**State**: closed
**Created**: 2024-01-09T17:26:55+00:00
**Closed**: 2024-11-30T02:03:04+00:00
**Comments**: 3
**Labels**: feature request, stale

### Description

There's a new cache technique mentioned in the paper https://arxiv.org/abs/2312.17238. (github: https://github.com/dvmazur/mixtral-offloading)
They introduced LRU cache to cache experts based on patterns they found, and also took speculative guess to pre-load experts before the computation of the next layer. The result looks quite promising. Can we support it for Mixtral? This helps a lot to run on smaller GPUs.

---

## Issue #N/A: [Bug]: Vllm automatically restarts while using cortecs/phi-4-FP8-Dynamic

**Link**: https://github.com/vllm-project/vllm/issues/14675
**State**: closed
**Created**: 2025-03-12T10:37:21+00:00
**Closed**: 2025-07-11T02:15:41+00:00
**Comments**: 2
**Labels**: bug, stale

### Description

### Your current environment

Vllm version used : vllm/vllm-openai:v0.7.3

### 🐛 Describe the bug

Vllm restarts randomly while running multiple subsequent request with **cortecs/phi-4-FP8-Dynamic.**

DEBUG 03-12 03:19:03 launcher.py:59] python3 -m vllm.entrypoints.openai.api_server --model cortecs/phi-4-FP8-Dynamic --dtype auto --max-model-len 14336 --tensor-parallel-size 1 --host=0.0.0.0 --port=9000 --gpu-memory-utilization=0.9 --trust-remote-code --api-key mits-d326429bf1aa6c4c7f6f0c910fd0aa04c8976498df5e06ab --enable-prefix-caching
INFO 03-12 03:19:03 launcher.py:62] Shutting down FastAPI HTTP server.
INFO:     Shutting down
DEBUG 03-12 03:19:07 client.py:174] Shutting down MQLLMEngineClient check health loop.
DEBUG 03-12 03:19:07 client.py:257] Shutting down MQLLMEngineClient output handler.


Here the last few logs after restarted unexpectedly. 

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bot

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Timeout Error When Deploying Llamafied InternLM2-5-7B-Chat-1M Model via vLLM OpenAI API Server

**Link**: https://github.com/vllm-project/vllm/issues/6414
**State**: closed
**Created**: 2024-07-13T15:17:03+00:00
**Closed**: 2024-11-24T02:08:20+00:00
**Comments**: 2
**Labels**: bug, stale

### Description

### Your current environment

```text
Collecting environment information...
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.5 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: Could not collect
CMake version: version 3.29.6
Libc version: glibc-2.31

Python version: 3.10.14 (main, Apr  6 2024, 18:45:05) [GCC 9.4.0] (64-bit runtime)
Python platform: Linux-5.4.0-172-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.2.91
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A100 80GB PCIe
  MIG 7g.80gb     Device  0:

Nvidia driver version: 535.129.03
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Byte Order:                       

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: bind python and c++ through tools other than pybind11

**Link**: https://github.com/vllm-project/vllm/issues/4694
**State**: closed
**Created**: 2024-05-08T23:18:56+00:00
**Closed**: 2024-10-27T22:53:29+00:00
**Comments**: 3
**Labels**: help wanted, feature request, stale

### Description

### 🚀 The feature, motivation and pitch

As vLLM goes into a fast release schedule (currently one release every two weeks), we will quickly hit the project-wide limit of pypi (around 5GB per project). One solution, as pointed out in https://github.com/pypi/support/issues/3792#issuecomment-2099941677 , is to build one wheel for all python versions (Python 3.8+).

I have figured out the procedure https://github.com/pypi/support/issues/3792#issuecomment-2101360740 , but pybind11 does not support this Python Limited API protocol.

One possible solution is to replace pybind11 with some other tools, so that the binding procedure can be used with Python Limited API.

Possible solutions:

- Nanobind (seems to support it starting from Python 3.12 only: https://github.com/wjakob/nanobind/pull/561 )
- register ops through pytorch directly https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html

### Alternatives

_No response_

### Additional context

_No response_

---

## Issue #N/A: The Effect of Chinese Alpaca

**Link**: https://github.com/vllm-project/vllm/issues/2264
**State**: closed
**Created**: 2023-12-26T03:25:46+00:00
**Closed**: 2024-03-28T12:03:09+00:00
**Comments**: 0

### Description

Why does the effect of the Chinese alphaca chat model after starting the model with VLLM look like the effect of the Chinese Lama generated model.
<img width="773" alt="Snipaste_2023-12-26_11-25-23" src="https://github.com/vllm-project/vllm/assets/136042459/1bed4c73-998a-434e-bfa8-a4764a94020f">


---

## Issue #N/A: [Feature]: Access to user information in scheduler

**Link**: https://github.com/vllm-project/vllm/issues/5605
**State**: closed
**Created**: 2024-06-17T17:59:37+00:00
**Closed**: 2024-11-25T02:05:51+00:00
**Comments**: 2
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

To my knowledge, there is no user awareness in the core implementation of vLLM. However, in order to perform optimizations having the final user in mind, it would be very useful to be able to receive and use this information.

I see that there is a parameter called _user_ in the openAI API, i.e. (line 353 of file vllm/entrypoints/openai/protocol.py), but this information is not transferred further to the core of vLLM.
```python
class CompletionRequest(OpenAIBaseModel):
    [...]
    user: Optional[str] = None
``` 

Specifically, I am interested in creating scheduling policies based on the users, i.e.,  a scheduler that divides service fairly among multiple users. For that, it would be necessary to receive the user identifier in the scheduler. The scheduler only receives _SequenceGroup_ objects without user information in method _add_seq_group_ (line 320 of file vllm/core/scheduler.py).

Is there any way to access this information th

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: Why the avg. througput generation is low?

**Link**: https://github.com/vllm-project/vllm/issues/4760
**State**: closed
**Created**: 2024-05-11T08:22:52+00:00
**Closed**: 2025-05-01T02:13:38+00:00
**Comments**: 7
**Labels**: performance, stale

### Description

### Report of performance regression

Hi I use this:
```
server_vllm.py \
  --model "/data/models_temp/functionary-small-v2.4/" \
  --served-model-name "functionary" \
  --dtype=bfloat16 \
  --max-model-len 2048 \
  --host 0.0.0.0 \
  --port 8000 \
  --enforce-eager \
  --gpu-memory-utilization 0.94
```
on rtx 3090 24gb

Why I've got low speed?:
`Avg prompt throughput: 102.2 tokens/s, Avg generation throughput: 2.2 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.8%, CPU KV cache usage: 0.0%`


This is my config:
```
| INFO 05-11 08:17:48 server_vllm.py:473] args: Namespace(host='0.0.0.0', port=8000, allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], served_model_name='functionary', grammar_sampling=False, model='/data/models_temp/functionary-small-v2.4/', tokenizer=None, skip_tokenizer_init=False, revision=None, code_revision=None, tokenizer_revision=None, tokenizer_mode='auto', trust_re

[... truncated for brevity ...]

---

## Issue #N/A: 求问 qwen-14b微调后的模型用vllm推理后结果都为空 

**Link**: https://github.com/vllm-project/vllm/issues/2981
**State**: closed
**Created**: 2024-02-22T07:49:32+00:00
**Closed**: 2024-11-29T02:08:15+00:00
**Comments**: 3
**Labels**: stale

### Description

No description provided.

---

## Issue #N/A: [Feature]: Add argument terminators "eos_token_id" to serving models api_server.py

**Link**: https://github.com/vllm-project/vllm/issues/4260
**State**: closed
**Created**: 2024-04-22T09:19:28+00:00
**Closed**: 2024-06-28T06:11:12+00:00
**Comments**: 2
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

New models as LLama-3 use different end terminator, that are need to be specified. 
For example when using the API the client response return "me know if this is correct!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThat\'s correct! The output is", thats seems the roles are not well parsed. 

### Alternatives

_No response_

### Additional context

_No response_

---

## Issue #N/A: Update documentation for OpenAI API > 1.0.0

**Link**: https://github.com/vllm-project/vllm/issues/1875
**State**: closed
**Created**: 2023-12-01T02:43:36+00:00
**Closed**: 2024-04-04T08:12:57+00:00
**Comments**: 4

### Description

Hi, I'd like to use vllm with the openAI python API, so I can switch between VLLM and OpenAI just by changing the URL. I have `openai` version 1.3.5.

[Here are the docs ](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#using-openai-chat-api-with-vllm):
```python
import openai
# Set OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"
chat_response = openai.ChatCompletion.create(
    model="facebook/opt-125m",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]
)
print("Chat response:", chat_response)
```

This results in the following error: 
```bash
openai.lib._old_api.APIRemovedInV1: 

You tried to access openai.ChatCompletion, but this is no longer supported in openai>=1.0.0 - see the README at https://github.com/openai/openai-python for the API.

You can run `openai migrate` t

[... truncated for brevity ...]

---

## Issue #N/A: Deployment stuck when using kuberay to scale Multi-GPU LLM on Kubernetes

**Link**: https://github.com/vllm-project/vllm/issues/973
**State**: closed
**Created**: 2023-09-07T07:24:02+00:00
**Closed**: 2024-04-04T08:12:03+00:00
**Comments**: 5

### Description

I want to use kuberay to serve and horizontaly-scale my LLM on Kubernetes.

The python code i want to deploy looks somewhat like this:
```
import json
import logging
from typing import AsyncGenerator

import ray
from fastapi import BackgroundTasks
from huggingface_hub import login
from ray import serve
from starlette.requests import Request
from starlette.responses import StreamingResponse, Response
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

logger = logging.getLogger("ray.serve")
@serve.deployment()
class VLLMPredictDeployment:
    def __init__(self, **kwargs):
        """
        Construct a VLLM deployment.

        Refer to https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
        for the full list of arguments.

        Args:
            model: name or path of the huggingface model to 

[... truncated for brevity ...]

---

## Issue #N/A: How can I deploy vllm model with multi-replicas

**Link**: https://github.com/vllm-project/vllm/issues/1995
**State**: closed
**Created**: 2023-12-09T03:39:26+00:00
**Closed**: 2024-08-28T19:04:12+00:00
**Comments**: 6

### Description

I want to deploy a LLM model on 8 A100 gpus. 
To support the higher concurrency, I want to deploy 8 replicas (one replica on one gpu), and I want to expose one service to handle user requests, how can I do it?

---

## Issue #N/A: [Bug]: ValueError: There is no module or parameter named 'lm_head.qweight_type' in Qwen2ForCausalLM.When use GGUF and draft model

**Link**: https://github.com/vllm-project/vllm/issues/11839
**State**: closed
**Created**: 2025-01-08T10:22:59+00:00
**Closed**: 2025-05-09T02:10:03+00:00
**Comments**: 5
**Labels**: bug, stale

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

OS: Ubuntu 24.04.1 LTS (x86_64)
GCC version: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.39

Python version: 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 13:27:36) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.153.1-microsoft-standard-WSL2-x86_64-with-glibc2.39
Is CUDA available: True
CUDA runtime version: 12.6.77
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 4090
Nvidia driver version: 560.94
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.5.1
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.5.1
/usr/lib/x86_64-linux-gnu/

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Crash with Qwen2-Audio Model in vLLM During Audio Processing

**Link**: https://github.com/vllm-project/vllm/issues/10627
**State**: closed
**Created**: 2024-11-25T08:49:42+00:00
**Closed**: 2025-04-24T02:08:13+00:00
**Comments**: 7
**Labels**: bug, stale

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
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 13:27:36) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-122-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA RTX A6000
Nvidia driver version: 560.35.03
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):  

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: FP8 performance worse than FP16 for Qwen2-VL-2B-Instruct

**Link**: https://github.com/vllm-project/vllm/issues/9992
**State**: closed
**Created**: 2024-11-04T13:29:35+00:00
**Closed**: 2025-03-06T02:02:32+00:00
**Comments**: 6
**Labels**: performance, stale

### Description

### Proposal to improve performance

_No response_

### Report of performance regression

_No response_

### Misc discussion on performance

estimated QPS is as follows:
bs=1：11.402357925880366 for FP16 and 10.642891382295932 for FP8
bs=8：51.62193861376064 for FP16 and 49.57986576846022 for FP8
bs=16：61.87048607358999 for FP16 and 57.58566218192532 for FP8
bs=32:
For FP8:
Processed prompts: 100%|████████████████████| 32/32 [00:00<00:00, 67.85it/s, est. speed input: 11468.33 toks/s, output: 271.44 toks/s]

For FP16:
Processed prompts: 100%|████████████████████| 32/32 [00:00<00:00, 74.14it/s, est. speed input: 12531.11 toks/s, output: 296.59 toks/s]

The FP8 model convert script is as follow:
```
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot, wrap_hf_model_class
MODEL_ID = "/home/hadoop-platcv/qwen2-vl-2b-instruct/00-sr

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: failed when run Qwen2-54B-A14B-GPTQ-Int4(MOE)

**Link**: https://github.com/vllm-project/vllm/issues/6465
**State**: closed
**Created**: 2024-07-16T07:35:14+00:00
**Closed**: 2025-03-01T02:05:58+00:00
**Comments**: 13
**Labels**: bug, stale

### Description

### Your current environment

```text
Collecting environment information...
PyTorch version: 2.3.1+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.0
Libc version: glibc-2.35

Python version: 3.10.12 (main, Mar 22 2024, 16:50:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-91-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA RTX 6000 Ada Generation
Nvidia driver version: 545.23.08
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:                      52 bits physi

[... truncated for brevity ...]

---

## Issue #N/A: Inference server not working with models tuned on <|system|>,<|prompter|>,<|assistant|> or <|im_start|>,<|im_end|> format

**Link**: https://github.com/vllm-project/vllm/issues/1000
**State**: closed
**Created**: 2023-09-09T14:51:10+00:00
**Closed**: 2024-03-20T12:43:35+00:00
**Comments**: 5

### Description

Trying to run the vLLM server with https://huggingface.co/Open-Orca/LlongOrca-13B-16k but it returns just white space.

It uses messages formatted as:

```
<|im_start|>system
You are LlongOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!
<|im_end|>
```

Also tried https://huggingface.co/OpenAssistant/llama2-13b-orca-8k-3319 but it returns empty.
Message format:
```
<|system|>system message</s><|prompter|>user prompt</s><|assistant|>
```

Is it possible to use models that require such different formatting? The vLLM request is abstracted away and only sends messages list. I tried wrapping the content with the special tokens.

The only prompt format that works for me on vLLM server is
```
### Instruction:
<prompt>
### Response:
```
from Open-Orca/OpenOrca-Platypus2-13B

Maybe I am missing an argument when running the server?

---

## Issue #N/A: [Bug]: vllm-0.7.3. gptq-int3 model cannot run.

**Link**: https://github.com/vllm-project/vllm/issues/14394
**State**: closed
**Created**: 2025-03-07T01:51:21+00:00
**Closed**: 2025-07-05T02:11:58+00:00
**Comments**: 2
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>Error when running gptq-int3 model</summary>

- python3.10
- vllm==0.7.3
- transformers==4.49.0
- torch==2.5.1

</details>




### 🐛 Describe the bug

infer code:
```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_name = "Xu-Ouyang/Qwen2-1.5B-int3-GPTQ-wikitext2"
max_model_len, tp_size = 1024, 1

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, tensor_parallel_size=tp_size, max_model_len=max_model_len, trust_remote_code=True, enforce_eager=True)
sampling_params = SamplingParams(temperature=0.3, max_tokens=128, stop_token_ids=[tokenizer.eos_token_id])

messages_list = [[{"role": "user", "content": "Who are you?"}],]

prompt_token_ids = [tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in messages_list]

outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

generated_text = [output.outputs[0].text

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vLLM is erroneously sending some information outputs into the error stream

**Link**: https://github.com/vllm-project/vllm/issues/11686
**State**: closed
**Created**: 2025-01-02T11:42:49+00:00
**Closed**: 2025-05-09T02:45:25+00:00
**Comments**: 8
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
--2025-01-02 11:38:20--  https://raw.githubusercontent.com/vllm-project/vllm/main/collect_env.py
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.111.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 26218 (26K) [text/plain]
Saving to: 'collect_env.py'

collect_env.py               100%[=============================================>]  25.60K  --.-KB/s    in 0.001s  

2025-01-02 11:38:20 (34.2 MB/s) - 'collect_env.py' saved [26218/26218]

Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version:

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: how to shutdown vllm server

**Link**: https://github.com/vllm-project/vllm/issues/8356
**State**: closed
**Created**: 2024-09-11T06:34:38+00:00
**Closed**: 2025-01-11T01:59:46+00:00
**Comments**: 3
**Labels**: usage, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How would you like to use vllm

if i use 
> vllm serve llm/qwen/Qwen2-0.5B-Instruct

how to shutdown vllm server use command?

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Performance]: Qwen 7b chat model, under 128 concurrency, the CPU utilization rate is 100%, and the GPU SM utilization rate is only about 60%-75%. Is it a CPU bottleneck?

**Link**: https://github.com/vllm-project/vllm/issues/4806
**State**: closed
**Created**: 2024-05-14T07:48:00+00:00
**Closed**: 2024-11-27T02:08:34+00:00
**Comments**: 3
**Labels**: performance, stale

### Description

### Proposal to improve performance

_No response_

### Report of performance regression

_No response_

### Misc discussion on performance

I am using vllm to deploy the qwen 7b chat model service. In a very high concurrency scenario, such as 128 concurrency, I found that the CPU utilization reached 100%, but I saw the GPU utilization rate is less than 60%

My question is, because a lot of vllm's scheduling and calculation logic is implemented by Python coroutines, it can only use the computing power of a single CPU. In a scenario like this with 128 concurrency, is the CPU becoming a computing bottleneck, causing GPU CUDA to be unable to achieve higher performance?

Model download address：https://huggingface.co/Qwen/Qwen-7B-Chat/tree/main

1. For sever scenario
![image](https://github.com/vllm-project/vllm/assets/94596925/28d326a1-9d7a-437b-b503-6db4dc559a70)
![image](https://github.com/vllm-project/vllm/assets/94596925/ccda7dcc-b793-4d18-a7bf-096e05be10ff)
2. For o

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]:  torch.OutOfMemoryError: CUDA out of memory.

**Link**: https://github.com/vllm-project/vllm/issues/11560
**State**: closed
**Created**: 2024-12-27T08:57:15+00:00
**Closed**: 2025-03-07T13:48:29+00:00
**Comments**: 18
**Labels**: usage

### Description

### Your current environment

```
I  get this error when load "Qwen2-VL-72B-Instruct"

Here is the detail of error :
```

```
INFO 12-27 08:52:17 config.py:350] This model supports multiple tasks: {'embedding', 'generate'}. Defaulting to 'generate'.
INFO 12-27 08:52:17 llm_engine.py:249] Initializing an LLM engine (v0.6.4) with config: model='/data/fffan/model/Qwen2-VL-72B-Instruct', speculative_config=None, tokenizer='/data/fffan/model/Qwen2-VL-72B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=Observabili

[... truncated for brevity ...]

---

## Issue #N/A: [Roadmap] vLLM Roadmap Q2 2024

**Link**: https://github.com/vllm-project/vllm/issues/3861
**State**: closed
**Created**: 2024-04-04T22:38:01+00:00
**Closed**: 2024-06-25T00:08:31+00:00
**Comments**: 39

### Description

This document includes the features in vLLM's roadmap for Q2 2024. Please feel free to discuss and contribute to the specific features at related RFC/Issues/PRs and add anything else you'd like to talk about in this issue.

You can see our historical roadmap at #2681, #244. This roadmap contains work committed by the vLLM team from UC Berkeley, as well as the broader vLLM contributor groups including but not limited to Anyscale, IBM, NeuralMagic, Roblox, Oracle Cloud. You can also find help wanted items in this roadmap as well! Additionally, this roadmap is shaped by you, our user community!

### Themes. 

We categorized our roadmap into 6 broad themes:

* **Broad model support**: vLLM should support a wide range of transformer based models. It should be kept up to date as much as possible. This includes new auto-regressive decoder models, encoder-decoder models, hybrid architectures, and models supporting multi-modal inputs. 
* **Excellent hardware coverage**: vLLM should run

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: How to run the int4 quantized version of the gemma2-27b model

**Link**: https://github.com/vllm-project/vllm/issues/7125
**State**: closed
**Created**: 2024-08-04T13:15:30+00:00
**Closed**: 2024-12-08T02:10:51+00:00
**Comments**: 7
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

How to run the int4 quantized version of the gemma2-27b model

### Alternatives

_No response_

### Additional context

_No response_

---

## Issue #N/A: [Bug]: docker 启动vllm,配置了host_IP ，还是 [W socket.cpp:663] [c10d] The client socket has failed to connect to [::ffff:172.16.8.232]:39623 (errno: 110 - Connection timed out)

**Link**: https://github.com/vllm-project/vllm/issues/3771
**State**: closed
**Created**: 2024-04-01T08:57:27+00:00
**Closed**: 2024-11-28T02:07:11+00:00
**Comments**: 5
**Labels**: bug, stale

### Description

### Your current environment

Collecting environment information...
PyTorch version: 1.12.1+cu113
Is debug build: False
CUDA used to build PyTorch: 11.3
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.31

Python version: 3.8.16 (default, Mar  2 2023, 03:21:46)  [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-101-generic-x86_64-with-glibc2.17
Is CUDA available: True
CUDA runtime version: 12.3.107
CUDA_MODULE_LOADING set to: 
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 4090
Nvidia driver version: 550.54.14
cuDNN version: Probably one of the following:
/usr/local/cuda-12.0/targets/x86_64-linux/lib/libcudnn.so.8.8.0
/usr/local/cuda-12.0/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.8.0
/usr/local/cuda-12.0/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.8.0
/usr/local/cuda-12.0/targets/

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: enable_prefix_caching cause a triron crash

**Link**: https://github.com/vllm-project/vllm/issues/6099
**State**: closed
**Created**: 2024-07-03T09:10:43+00:00
**Closed**: 2024-08-12T09:47:09+00:00
**Comments**: 8
**Labels**: bug

### Description

### Your current environment

`Collecting environment information...
PyTorch version: 2.3.0+cu118
Is debug build: False
CUDA used to build PyTorch: 11.8
ROCM used to build PyTorch: N/A

OS: CentOS Linux 7 (Core) (x86_64)
GCC version: (GCC) 11.2.1 20220127 (Red Hat 11.2.1-9)
Clang version: Could not collect
CMake version: version 3.29.3
Libc version: glibc-2.17

Python version: 3.9.16 (main, Jul 10 2023, 11:13:07)  [GCC 8.3.1 20190311 (Red Hat 8.3.1-3)] (64-bit runtime)
Python platform: Linux-4.18.0-147.20200626.413.el8_1.x86_64-x86_64-with-glibc2.17
Is CUDA available: True
CUDA runtime version: 11.8.89
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA A100-SXM4-80GB
GPU 1: NVIDIA A100-SXM4-80GB

Nvidia driver version: 470.103.01
cuDNN version: Probably one of the following:
/usr/lib64/libcudnn.so.8.9.2
/usr/lib64/libcudnn_adv_infer.so.8.9.2
/usr/lib64/libcudnn_adv_train.so.8.9.2
/usr/lib64/libcudnn_cnn_infer.so.8.9.2
/usr/lib64/li

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Chunked prefill + lora

**Link**: https://github.com/vllm-project/vllm/issues/4995
**State**: closed
**Created**: 2024-05-23T01:12:17+00:00
**Closed**: 2025-04-02T02:06:40+00:00
**Comments**: 13
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

Currently lora doesn't work with chunked prefill because some of lora index logic doesn't cover the case where sampling is not required. This also means lora is not working with sampling_params do_sample=True. 

We need to add test cases for these. WIP https://github.com/vllm-project/vllm/pull/4994

### Alternatives

_No response_

### Additional context

_No response_

---

## Issue #N/A: ValueError: The number of GPUs per node is not divisible by the number of tensor parallelism.

**Link**: https://github.com/vllm-project/vllm/issues/596
**State**: closed
**Created**: 2023-07-27T07:11:53+00:00
**Closed**: 2024-03-25T10:57:02+00:00
**Comments**: 10
**Labels**: bug

### Description

I have 3 GPUs (3x3090). When I try to load `LLaMA-2-13B` and set the `tensor_parallel_size` to 2 it gives me this error. When I set it to 3 error follows like `ValueError: Total number of attention heads (40) must be divisible by tensor parallel size (3).`

---

## Issue #N/A: [Bug]: Extremely slow inference speed when deploying with vLLM on 16 H100 GPUs according to instructions on DeepSeekV3

**Link**: https://github.com/vllm-project/vllm/issues/11705
**State**: closed
**Created**: 2025-01-03T05:25:02+00:00
**Closed**: 2025-06-11T02:14:13+00:00
**Comments**: 8
**Labels**: bug, stale

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
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.11.10 (main, Oct  3 2024, 07:29:13) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.10.134-008.7.kangaroo.al8.x86_64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA L20Z
GPU 1: NVIDIA L20Z
GPU 2: NVIDIA L20Z
GPU 3: NVIDIA L20Z
GPU 4: NVIDIA L20Z
GPU 5: NVIDIA L20Z
GPU 6: NVIDIA L20Z
GPU 7: NVIDIA L20Z

Nvidia driver version: 535.161.08
cuDNN version: Could not collect
HIP runtime version: N

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Ray on multi machine cluster fails to detect all nodes.

**Link**: https://github.com/vllm-project/vllm/issues/4655
**State**: closed
**Created**: 2024-05-07T14:19:54+00:00
**Closed**: 2024-11-28T02:05:19+00:00
**Comments**: 3
**Labels**: bug, stale

### Description

### Your current environment

```text
python collect_env.py
--2024-05-07 16:14:33--  https://raw.githubusercontent.com/vllm-project/vllm/main/collect_env.py
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.109.133, 185.199.110.133, 185.199.111.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.109.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 24877 (24K) [text/plain]
Saving to: ‘collect_env.py’

collect_env.py                                                                     100%[================================================================================================================================================================================================================>]  24.29K  --.-KB/s    in 0.003s  

2024-05-07 16:14:33 (9.38 MB/s) - ‘collect_env.py’ saved [24877/24877]

Collecting environment information...
PyTorch version: 2.3.0+cu121
Is debug build: False

[... truncated for brevity ...]

---

