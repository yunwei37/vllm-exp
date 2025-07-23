# structured-output - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 17
- Closed Issues: 13

### Label Distribution

- structured-output: 30 issues
- bug: 20 issues
- unstale: 6 issues
- feature request: 5 issues
- tool-calling: 3 issues
- RFC: 2 issues
- installation: 1 issues
- v1: 1 issues
- performance: 1 issues
- usage: 1 issues

---

## Issue #N/A: [Bug]: Corrupted output when using JSON structured response (v0.9.1)

**Link**: https://github.com/vllm-project/vllm/issues/19493
**State**: closed
**Created**: 2025-06-11T15:23:27+00:00
**Closed**: 2025-06-12T23:30:10+00:00
**Comments**: 7
**Labels**: bug, structured-output

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
==============================
        System Info
==============================
OS                           : Ubuntu 22.04.4 LTS (x86_64)
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
Python version               : 3.10.12 (main, Feb  4 2025, 14:57:36) [GCC 11.4.0] (64-bit runtime)
Python platform              : Linux-5.15.0-141-generic-x86_64-with-glibc2.35

==============================
     

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: guided_json 请求报错 在 v0.7.2

**Link**: https://github.com/vllm-project/vllm/issues/15073
**State**: closed
**Created**: 2025-03-19T02:24:13+00:00
**Closed**: 2025-04-24T17:39:42+00:00
**Comments**: 2
**Labels**: bug, structured-output

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```
最近把vllm从0.6.3升级到了0.7.2 发现之前的guided_json调用报错了。
之前在vllm0.6.3版本是没问题的

代码如下

from pydantic import BaseModel
from enum import Enum

from openai import OpenAI

openai_api_key = "none"
openai_api_base = "http://10.12.167.20:8888/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"


class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType


json_schema = CarDescription.model_json_schema()

completion = client.chat.completions.create(
    model="QWen",
    messages=[
        {
            "role": "user",
            "content": "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
        }
    ],
    extra_body={"guided_json": json_schema},
)
print(

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: guided_json not working correctly with (quantized) mistral-small model

**Link**: https://github.com/vllm-project/vllm/issues/15577
**State**: open
**Created**: 2025-03-26T21:10:04+00:00
**Comments**: 4
**Labels**: bug, structured-output

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text

INFO 03-26 14:07:54 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.31.6
Libc version: glibc-2.35

Python version: 3.12.9 (main, Feb  5 2025, 08:49:00) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.5.0-45-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA GeForce RTX 4090
GPU 1: NVIDIA GeForce RTX 4090

Nvidia driver version: 550.120
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]:  Could not find a version that satisfies the requirement xgrammar>=0.1.6; platform_machine == "x86_64" (from vllm) (from versions: none)

**Link**: https://github.com/vllm-project/vllm/issues/11886
**State**: closed
**Created**: 2025-01-09T07:25:11+00:00
**Closed**: 2025-03-11T15:27:39+00:00
**Comments**: 38
**Labels**: installation, structured-output

### Description

### Your current environment

```text
 # 创建环境
 conda create -n online_model_v4  python=3.10.13
 
 # 激活环境
 conda activate online_model_v4


pip install vllm==0.6.6.post1


```


### How you are installing vllm

```sh
pip install vllm==0.6.6.post1




问题
Collecting prometheus-fastapi-instrumentator>=7.0.0 (from vllm==0.6.6.post1)
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/59/66/2e93a8f56adb51ede41d0ef5f4f0277522acc4adc87937f5457b7b5692a8/prometheus_fastapi_instrumentator-7.0.0-py3-none-any.whl (19 kB)
Collecting tiktoken>=0.6.0 (from vllm==0.6.6.post1)
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/2e/28/cf3633018cbcc6deb7805b700ccd6085c9a5a7f72b38974ee0bffd56d311/tiktoken-0.8.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.2 MB)
Collecting lm-format-enforcer<0.11,>=0.10.9 (from vllm==0.6.6.post1)
  Using cached https://pypi.tuna.tsinghua.edu.cn/packages/c1/01/e78fdf09de2b4e7750a402eaa4f6783c7215ededd4bc6fe4a3f6d69c49da/lm

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Compiling FSM index high memory && subprocess OOM

**Link**: https://github.com/vllm-project/vllm/issues/7332
**State**: closed
**Created**: 2024-08-09T03:10:36+00:00
**Closed**: 2025-06-27T19:59:20+00:00
**Comments**: 7
**Labels**: bug, structured-output

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.1
Libc version: glibc-2.35

Python version: 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-3.10.0-1160.71.1.el7.x86_64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA GeForce RTX 3090
GPU 1: NVIDIA GeForce RTX 3090

Nvidia driver version: 535.129.03
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                    x86_64
CPU

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Guided Decoding Schema Cache Store

**Link**: https://github.com/vllm-project/vllm/issues/8902
**State**: open
**Created**: 2024-09-27T12:07:20+00:00
**Comments**: 3
**Labels**: feature request, structured-output, unstale

### Description

### 🚀 The feature, motivation and pitch

# Problem

I am currently working with structured outputs and experimented a little with VLLM + Outlines. Since our JSON Schemas can get quite complex the generation of the FSM can take around 2 Minutes per Schema. It would be great to have a feature where you can provide a Schema-Store to save your generated schemas over time in a local file and reload them when you restart your deployment. Ideally this would be implemented as flag in the vllm serve arguments:

https://docs.vllm.ai/en/latest/models/engine_args.html

# Current Implementation
I assume that this is currently not supported and the code to not recompute the schema is handled with the @cache() decorator here:
![Screenshot 2024-09-27 134948](https://github.com/user-attachments/assets/4d6480a8-5a79-40ab-8b5c-6023b0551233)


### Alternatives

Alternative solution would probably be to create custom python code to handle this for my use-case and use the VLLM python function

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Distilled DeepSeek Models do not work with guided_json

**Link**: https://github.com/vllm-project/vllm/issues/12548
**State**: open
**Created**: 2025-01-29T11:47:24+00:00
**Comments**: 5
**Labels**: bug, structured-output

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

When using **DeepSeek distilled models** with **guided JSON output**, the response does not always adhere to the expected schema. Unlike the standard versions of the models (e.g., Llama 3), which complete the JSON properly within a given `max_tokens` limit, the distilled models often fail to do so.  

For example, when setting `max_tokens = x`, **Llama 3** correctly generates a full JSON response. However, with **DeepSeek's distilled versions**, the output is sometimes **incomplete**, often stopping at an **open bracket (`{`)** or other partial structures. This suggests that the distilled models may require a **higher `max_tokens` setting** than their non-distilled counterparts to function correctly.  

## 🔍 Expected Behavior  
- The model should generate a **

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: [v0.8.4][Critical] Tools calling broken: xgrammar rejects minItems in JSON Schema, blocking agent functionality

**Link**: https://github.com/vllm-project/vllm/issues/16880
**State**: open
**Created**: 2025-04-19T19:09:55+00:00
**Comments**: 5
**Labels**: bug, structured-output, tool-calling

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

OS: Red Hat Enterprise Linux 8.10 (Ootpa) (x86_64)
GCC version: (GCC) 9.2.1 20191120 (Red Hat 9.2.1-2)
Clang version: Could not collect
CMake version: version 3.27.7
Libc version: glibc-2.28

Python version: 3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-4.18.0-553.40.1.el8_10.x86_64-x86_64-with-glibc2.28
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A100 80GB PCIe
GPU 1: NVIDIA A100 80GB PCIe
GPU 2: NVIDIA A100 80GB PCIe
GPU 3: NVIDIA A100 80GB PCIe

Nvidia driver version: 550.54.15
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:


[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Guided Decoding Broken in Streaming mode

**Link**: https://github.com/vllm-project/vllm/issues/10376
**State**: open
**Created**: 2024-11-15T21:56:33+00:00
**Comments**: 1
**Labels**: bug, structured-output, unstale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.0
Libc version: glibc-2.35

Python version: 3.11.10 (main, Oct  3 2024, 07:29:13) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.8.0-1017-azure-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A100 80GB PCIe
GPU 1: NVIDIA A100 80GB PCIe

Nvidia driver version: 550.127.05
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-b

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

## Issue #N/A: [Feature]: Align the API with OAI's structured output

**Link**: https://github.com/vllm-project/vllm/issues/7220
**State**: open
**Created**: 2024-08-06T20:30:45+00:00
**Comments**: 5
**Labels**: feature request, structured-output, unstale

### Description

### 🚀 The feature, motivation and pitch

OpenAI API introduced a feature that supports structured output, this is basically the same as our `guided_json` feature.

1. We should simply alias it to support this feature 🌟
2. we might want to consider implementing this also for tools
3. Implement `refusal`

### Alternatives

_No response_

### Additional context

https://openai.com/index/introducing-structured-outputs-in-the-api/

---

## Issue #N/A: [Bug]: Guided Decoding Backend options with the OpenAI server recently broken

**Link**: https://github.com/vllm-project/vllm/issues/17002
**State**: closed
**Created**: 2025-04-22T18:54:27+00:00
**Closed**: 2025-04-29T19:02:24+00:00
**Comments**: 7
**Labels**: bug, structured-output

### Description

### Your current environment

vLLM installed with:
```
pip install https://wheels.vllm.ai/5536b30a4c7877d75758d21bdaf39b3a59aa2dc2/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```


### 🐛 Describe the bug

After merging https://github.com/vllm-project/vllm/pull/16789, using "options" for guided decoding backends no longer works. Attempting to include a backend option results in:
```
$ vllm serve meta-llama/Llama-3.2-3B-Instruct --guided-decoding-backend xgrammar:disable-any-whitespace
INFO 04-22 18:45:12 [__init__.py:239] Automatically detected platform cuda.
usage: vllm serve [model_tag] [options]
vllm serve: error: argument --guided-decoding-backend: invalid choice: 'xgrammar:disable-any-whitespace' (choose from 'auto', 'outlines', 'lm-format-enforcer', 'xgrammar')
```
The new type checking of the args checks against a Literal type for the backend name, disallowing any options. For reference, backend options are briefly documented [REF](https://docs.vllm.ai/en/latest/features/struc

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Extra Characters in `content` When Using `enable_reasoning` with `stop` Parameter

**Link**: https://github.com/vllm-project/vllm/issues/15188
**State**: open
**Created**: 2025-03-20T05:33:28+00:00
**Comments**: 4
**Labels**: bug, structured-output, tool-calling

### Description

![Image](https://github.com/user-attachments/assets/59d64b2b-986e-46e1-8ff1-d66588bd431e)

### Your current environment

#### Environment  
- vLLM version: 0.7.3  
- Model: DeepSeek R1  
- Running on: H20 

### 🐛 Describe the bug

#### Description  
When running the **DeepSeek R1** model with the `vllm` framework and enabling the `enable_reasoning` parameter, the model’s response is structured into two fields:  
- **`reasoning_content`**: Represents the reasoning process.  
- **`content`**: Represents the final output.  

However, when specifying the `stop` parameter with any stop sequence, the `content` field in the response contains extra unintended characters. This issue does not occur when `enable_reasoning` is disabled.  

#### Steps to Reproduce  
1. Start `vllm` with `--enable-reasoning`.  
2. Query the model with a `stop` parameter (e.g., `stop=["\nObservation"]`).  
3. Observe that the `content` field includes additional characters beyond the expected stop sequence.  

#### Ex

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: gemma 3 structured output api occurs assertion error

**Link**: https://github.com/vllm-project/vllm/issues/15766
**State**: open
**Created**: 2025-03-30T09:26:45+00:00
**Comments**: 7
**Labels**: bug, structured-output

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```

</details>


### 🐛 Describe the bug

I use vllm v0.8.2 with docker compose, to test structured output api on gemma-3-27b-it, and get this errror.

```
vllm-llm-1  | ERROR 03-30 02:15:01 [engine.py:160]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/guided_decoding/xgrammar_decoding.py", line 355, in __call__
vllm-llm-1  | ERROR 03-30 02:15:01 [engine.py:160]     assert self.matchers[i].accept_token(sampled_token)
vllm-llm-1  | ERROR 03-30 02:15:01 [engine.py:160]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
vllm-llm-1  | ERROR 03-30 02:15:01 [engine.py:160] AssertionError
vllm-llm-1  | CRITICAL 03-30 02:15:01 [launcher.py:116] MQLLMEngine is already dead, terminating server process
```
<details>
<summary>there is docker-compose.yaml</summary>

```yaml
services:
  vllm-llm:
    image: vllm/vllm-openai:v

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Mistral-Small-24B-Instruct-2501 on V1 fails to start with Mistral tokenizer since V1 enabled guided decoding

**Link**: https://github.com/vllm-project/vllm/issues/14465
**State**: closed
**Created**: 2025-03-07T23:47:08+00:00
**Closed**: 2025-03-12T08:01:35+00:00
**Comments**: 3
**Labels**: bug, structured-output

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 03-07 16:45:32 [__init__.py:256] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Arch Linux (x86_64)
GCC version: (GCC) 14.2.1 20250207
Clang version: 19.1.7
CMake version: version 3.30.0
Libc version: glibc-2.41

Python version: 3.12.9 (main, Feb  9 2025, 04:01:11) [GCC 14.2.1 20250128] (64-bit runtime)
Python platform: Linux-6.12.9-arch1-1-kvm-local-x86_64-with-glibc2.41
Is CUDA available: True
CUDA runtime version: 12.8.61
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 3090 Ti
Nvidia driver version: 570.124.04
cuDNN version: Probably one of the following:
/usr/lib/libcudnn.so.9.7.0
/usr/lib/libcudnn_adv.so.9.7.0
/usr/lib/libcudnn_cnn.so.9.7.0
/usr/lib/libcudnn_engines_precompiled.so.9.7

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: guided decoding on TPU

**Link**: https://github.com/vllm-project/vllm/issues/11104
**State**: closed
**Created**: 2024-12-11T14:26:43+00:00
**Closed**: 2025-04-23T18:32:19+00:00
**Comments**: 9
**Labels**: feature request, structured-output

### Description

### 🚀 The feature, motivation and pitch

I’m not sure if this is possible, but right now the `execute_model` function on the `TPUModelRunner` is only outputting the predicted token_ids, rather than the distribution of tokens that we can sample from with some guidance (e.g., using outlines). I believe structured output is becoming more common, and most projects that require LLMs need this structured output feature.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: [V1] Molmo/Aria not supported on V1 due to xgrammar

**Link**: https://github.com/vllm-project/vllm/issues/14534
**State**: closed
**Created**: 2025-03-10T03:12:01+00:00
**Closed**: 2025-03-14T14:51:50+00:00
**Comments**: 7
**Labels**: bug, structured-output

### Description

### Your current environment

Cannot use these models on V1 due to Xgrammar assert

### 🐛 Describe the bug

- run the following
```bash
VLLM_USE_V1=1 pytest -s -x models/decoder_only/vision_language/test_models.py -k molmo
VLLM_USE_V1=1 pytest -s -x models/decoder_only/vision_language/test_models.py -k aria
```

- get the following back
```bash
ERROR 03-10 03:06:35 [core.py:324] EngineCore hit an exception: Traceback (most recent call last):
ERROR 03-10 03:06:35 [core.py:324]   File "/home/rshaw/vllm/vllm/v1/engine/core.py", line 316, in run_engine_core
ERROR 03-10 03:06:35 [core.py:324]     engine_core = EngineCoreProc(*args, **kwargs)
ERROR 03-10 03:06:35 [core.py:324]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 03-10 03:06:35 [core.py:324]   File "/home/rshaw/vllm/vllm/v1/engine/core.py", line 271, in __init__
ERROR 03-10 03:06:35 [core.py:324]     super().__init__(vllm_config, executor_class, log_stats)
ERROR 03-10 03:06:35 [core.py:324]   File "/home/rshaw/vllm/vllm/v1

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: xgrammar missing file crashes the server

**Link**: https://github.com/vllm-project/vllm/issues/16030
**State**: closed
**Created**: 2025-04-03T18:09:44+00:00
**Closed**: 2025-04-24T17:39:51+00:00
**Comments**: 2
**Labels**: bug, structured-output

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```

</details>


### 🐛 Describe the bug

I have the following xgrammar version

```
[root@~]# pip show xgrammar
Name: xgrammar
Version: 0.1.17
Summary: Efficient, Flexible and Portable Structured Generation
Home-page: https://xgrammar.mlc.ai/
Author: MLC Team
Author-email:
License: Apache 2.0
Location: /opt/pytorch/lib/python3.12/site-packages
Requires: nanobind, ninja, pydantic, sentencepiece, tiktoken, torch, transformers
Required-by: vllm
```
and this is the command I use to initialize the model

```
   /opt/pytorch/bin/vllm serve Mistral/ --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --limit_mm_per_prompt 'image=1' --tensor-parallel-size 4 --max-model-len 120000 --quantization fp8 --port 8006 --host localhost --guided-decoding-backend xgrammar
```
The model

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: xgrammar crashes with speculative decoding

**Link**: https://github.com/vllm-project/vllm/issues/11484
**State**: open
**Created**: 2024-12-25T07:09:45+00:00
**Comments**: 10
**Labels**: bug, structured-output

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
$python collect_env.py 
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Alibaba Group Enterprise Linux Server 7.2 (Paladin) (x86_64)
GCC version: (GCC) 10.2.1 20200825 (Alibaba 10.2.1-3 2.17)
Clang version: Could not collect
CMake version: version 3.26.4
Libc version: glibc-2.32

Python version: 3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-4.9.151-015.ali3000.alios7.x86_64-x86_64-with-glibc2.32
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA L20
GPU 1: NVIDIA L20

Nvidia driver version: 535.161.08
cuDNN version: Probably one of the following:
/usr/local/cuda/targets/x86_64-linux/lib/libcudnn.so.8.9.3
/usr/local/cuda

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Disable unicode characters in structured decoding

**Link**: https://github.com/vllm-project/vllm/issues/16363
**State**: open
**Created**: 2025-04-09T21:58:07+00:00
**Comments**: 4
**Labels**: feature request, structured-output

### Description

### 🚀 The feature, motivation and pitch

Currently, the xgrammar backend will often return lots of messy unicode characters that are hard to parse and deal with. It requires a lot of custom code to parse these out (with best efforts, as some are not even valid). 



### Alternatives

Opening this issue in the `xgrammar` repo, or create a custom unicode parser.

### Additional context

From the [documentation](https://xgrammar.mlc.ai/docs/api/python/index.html#xgrammar.VocabType) it appears that this issue only arises for certain tokenizer types. It would be nice if it offered consistent behavior across models.

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: Error with structured output inference after upgrade 0.6.2->0.6.3

**Link**: https://github.com/vllm-project/vllm/issues/9462
**State**: open
**Created**: 2024-10-17T12:35:39+00:00
**Comments**: 2
**Labels**: bug, structured-output, unstale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
ollecting environment information...
/opt/conda/lib/python3.11/site-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
No module named 'vllm._version'
  from vllm.version import __version__ as VLLM_VERSION
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.11.8 | packaged by conda-forge | (main, Feb 16 2024, 20:53:32) [GCC 12.3.0] (64-bit runtime)
Python platform: Linux-6.1.109-118.189.amzn2023.x86_64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A10G
Nvidia drive

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Support guided decoding with multistep decoding

**Link**: https://github.com/vllm-project/vllm/issues/9893
**State**: open
**Created**: 2024-10-31T22:29:47+00:00
**Comments**: 1
**Labels**: feature request, structured-output

### Description

### 🚀 The feature, motivation and pitch

See https://github.com/vllm-project/vllm/issues/8985. It would be great if we could get the speedup from multi-step decoding without having to disallow users from using guided decoding.

I have no idea how feasible that is to do, but if anybody has a sketch of how it would be done I could be up for learning and helping to implement. I'm mostly opening this issue so it's documented and I can link it from the feature compatibility matrix in the docs.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Performance]: guided generation is very slow in offline mode

**Link**: https://github.com/vllm-project/vllm/issues/8313
**State**: open
**Created**: 2024-09-10T02:21:33+00:00
**Comments**: 21
**Labels**: performance, structured-output, unstale

### Description

### Proposal to improve performance

With a single request / online mode I'm getting:

- no guided 300 tok/sec
- `outlines` 150 tok/sec (2x slower)
- `lm-format-enforcer` 90 tok/sec (~3x slower)

with offline mode I get:
- `outlines` **is about 10-20x slower than no guided generation**
- `lm-format-enforcer` is about 4x faster than `outlines` (note that it is slower than `outlines` for online)

for online I was using this schema:

```
json_template = {
    "type": "object",
    "properties": {
        "criteria": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "response": { "type": "string" }
    },
    "required": ["criteria", "response"]
}
```

for offline I was using an even simpler schema:
```

{
   "type":"object",
   "properties":{
      "name":{
         "type":"string", "minLength":2, "maxLength":5
      },
      "age":{
         "type":"integer"
      }
   },
   "required":[ "name", "age"]
}
```
the huge performan

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Chat with n>1 breaks xgrammar

**Link**: https://github.com/vllm-project/vllm/issues/11312
**State**: closed
**Created**: 2024-12-18T23:08:00+00:00
**Closed**: 2025-01-22T21:27:54+00:00
**Comments**: 4
**Labels**: bug, structured-output

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>


$ python collect_env.py
/workspace/my-vllm/lib64/python3.12/site-packages/transformers/utils/hub.py:128: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Red Hat Enterprise Linux 9.5 (Plow) (x86_64)
GCC version: (GCC) 11.5.0 20240719 (Red Hat 11.5.0-2)
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.34

Python version: 3.12.5 (main, Sep 11 2024, 00:00:00) [GCC 11.5.0 20240719 (Red Hat 11.5.0-2)] (64-bit runtime)
Python platform: Linux-5.14.0-284.88.1.el9_2.x86_64-x86_64-with-glibc2.34
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Structured output requests can hang the server

**Link**: https://github.com/vllm-project/vllm/issues/14151
**State**: closed
**Created**: 2025-03-03T18:16:57+00:00
**Closed**: 2025-03-20T04:33:53+00:00
**Comments**: 0
**Labels**: bug, structured-output

### Description

### Your current environment

This isn't version specific, the use of a `ThreadPoolExecutor` to build grammars for structured output has been around since the original `outlines` integration


### 🐛 Describe the bug

To build structured output (guided decoding) processors in vLLM, we currently either:
- Execute the non-async grammar creation right in the event loop, or
- Use a ThreadPoolExectuor to run the grammar creation in a separate thread

However, there are cases where a user may pass in a json schema for structured output that will cause grammar compilation to take a really long time. One such case reported with outlines is here:  https://github.com/dottxt-ai/outlines-core/issues/180, and we've had many reports from products with >1k line json schemas input as guided decoding parameters that exhibit this behavior.

The problem is that we don't have a way to cancel the construction of these grammars when the api request times out or is cancelled. Specifically when using the threa

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Support for Specifying ```extra_body``` Parameters in vLLM Terminal Commands for structuring the JSON output 

**Link**: https://github.com/vllm-project/vllm/issues/11153
**State**: closed
**Created**: 2024-12-12T23:55:33+00:00
**Closed**: 2025-05-18T02:13:52+00:00
**Comments**: 3
**Labels**: structured-output, usage, stale

### Description


Hi there,

I understand that vLLM currently supports [outlines-dev/outlines](https://github.com/outlines-dev/outlines), [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar), and [noamgat/lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer) for guided decoding. I would like to directly configure guided decoding via the terminal using the following command:


`vllm serve meta-llama/Llama-3.1-8B-Instruct --device neuron --tensor-parallel-size 2 --block-size 8 --max-model-len 4096 --max-num-seqs 32 --guided-decoding-backend lm-format-enforcer
`


The challenge I’m facing is how to specify ```extra_body``` parameters such as:

`extra_body={"guided_regex": "\w+@\w+\.com\n", "stop": ["\n"]}
`

directly from the terminal, so I don’t need to modify any other code. Is there a way to pass these parameters via the CLI, or do I need to rely exclusively on the Python API for such configurations?

Any advice or pointers on this would be greatly appreciated!

### How

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Unification of frontend parser

**Link**: https://github.com/vllm-project/vllm/issues/17817
**State**: open
**Created**: 2025-05-07T21:46:18+00:00
**Comments**: 0
**Labels**: structured-output, RFC, tool-calling

### Description

## motivation

https://github.com/vllm-project/vllm/issues/11522 (with draft implementation at https://github.com/vllm-project/vllm/pull/11554)
aims to simplify the logics of the tool parser interface. However, this doesn't cover the cases for reasoning models (where we want to parse
tokens generated within the thinking budgets, etc. Our current solutions involves a reasoning parser, which will soon be running into the same
issue mentioned in #11522 when dealing with very long thinking budget). Additionally, the current implementations of tool calling are relatively
fragile, and not scalable when adding more tool format.

This RFC aims to build on top of some similar ideas from the RFC and unify both tool calling and reasoning parser logic for a more robust
way for us to move forward, especially with v0.10.x.

## proposed change


The workflow can be seen as follows:

- function/tool calling format for supported models (defined by the LLMEngine)
- Construct structural tags <- said tool

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: xgrammar doesn't support enums, but vllm isn't falling back to outlines

**Link**: https://github.com/vllm-project/vllm/issues/15762
**State**: open
**Created**: 2025-03-30T05:44:20+00:00
**Comments**: 1
**Labels**: bug, structured-output

### Description

### Your current environment

PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.5 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: Could not collect
CMake version: version 3.25.2
Libc version: glibc-2.31

Python version: 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-113-generic-x86_64-with-glibc2.31
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

Nvidia driver version: 555.42.06
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Guided decoding is broken because tokenizers can't be pickled

**Link**: https://github.com/vllm-project/vllm/issues/7557
**State**: open
**Created**: 2024-08-15T14:16:17+00:00
**Comments**: 3
**Labels**: bug, structured-output, unstale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.3.1+cpu
Is debug build: False
CUDA used to build PyTorch: Could not collect
ROCM used to build PyTorch: N/A

OS: Fedora release 39 (Thirty Nine) (x86_64)
GCC version: (GCC) 13.3.1 20240522 (Red Hat 13.3.1-1)
Clang version: 17.0.6 (Fedora 17.0.6-2.fc39)
CMake version: version 3.29.6
Libc version: glibc-2.38

Python version: 3.11.8 (main, Mar 27 2024, 15:03:48) [GCC 13.2.1 20231205 (Red Hat 13.2.1-6)] (64-bit runtime)
Python platform: Linux-6.7.11-200.fc39.x86_64-x86_64-with-glibc2.38
Is CUDA available: False
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: GPU 0: NVIDIA GeForce MX330
Nvidia driver version: 545.23.08
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Speculative decoding breaks guided decoding.

**Link**: https://github.com/vllm-project/vllm/issues/9423
**State**: open
**Created**: 2024-10-16T13:51:18+00:00
**Comments**: 13
**Labels**: bug, structured-output

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: N/A
Is debug build: N/A
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.10.12 (main, Jul 29 2024, 16:56:48) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-1063-azure-x86_64-with-glibc2.35
Is CUDA available: N/A
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: 
GPU 0: NVIDIA A100 80GB PCIe
GPU 1: NVIDIA A100 80GB PCIe

Nvidia driver version: 535.161.08
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: N/A

CPU:
Architecture:                       x86_64
CPU op-mode(s):            

[... truncated for brevity ...]

---

