# low_discussion_1to5 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 4
- Closed Issues: 26

### Label Distribution

- bug: 11 issues
- stale: 9 issues
- usage: 6 issues
- feature request: 5 issues
- help wanted: 1 issues
- performance: 1 issues
- speculative-decoding: 1 issues
- installation: 1 issues
- misc: 1 issues
- structured-output: 1 issues

---

## Issue #N/A: [Usage]: How to use tensor-parallel-size argument when deploy Llama3-8b with AsyncLLMEngine

**Link**: https://github.com/vllm-project/vllm/issues/4825
**State**: closed
**Created**: 2024-05-15T07:08:55+00:00
**Closed**: 2024-06-27T17:19:53+00:00
**Comments**: 5
**Labels**: usage

### Description

### Your current environment

```text
 My model is Llama3-8B which takes about 14GB GPU-memory.
And the machine have 2 * 40GB GPUs. （NVIDIA L40S）
```


### How would you like to use vllm


Hey, 
Recently I tried to use AsyncLLMEngine to speed up my LLM inference server. My model is Llama3-8B which takes about 14GB GPU-memory. And the machine have 2 * 40GB GPUs（NVIDIA L40S）. I want to know that:
1. What is the meaning of the "tensor-parallel-size" when init the AsyncLLMEngine? if I set it as 2, how is the parallelism been executed when a inference request comes,  It parallel the input tensors to the 2 different GPUs?  or it paralle-distribute the model's weight?
 2. When I test the time-comsuming of the server with "tensor-parallel-size" = 1or2, I didn't see an obvious difference on them, so I was wondering maybe I didn't use AsyncLLMEngine  and  "tensor-parallel-size" in a correct-way.


Thanks!


---

## Issue #N/A: [Bug]: llama3.2-11B-Vision-Instruct not working

**Link**: https://github.com/vllm-project/vllm/issues/9356
**State**: closed
**Created**: 2024-10-15T02:51:07+00:00
**Closed**: 2024-10-15T04:25:06+00:00
**Comments**: 3
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.11.10 (main, Oct  3 2024, 07:29:13) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-113-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.5.82
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: Tesla V100-SXM2-32GB
GPU 1: Tesla V100-SXM2-32GB
GPU 2: Tesla V100-SXM2-32GB
GPU 3: Tesla V100-SXM2-32GB
GPU 4: Tesla V100-SXM2-32GB
GPU 5: Tesla V100-SXM2-32GB
GPU 6: Tesla V100-SXM2-32GB
GPU 7: Tesla V100-SXM2-32GB

Nvidia driver version: 555.42.06
cuDNN version: Probably one of the following:
/us

[... truncated for brevity ...]

---

## Issue #N/A: [Performance] [Speculative decoding]: Support draft model on different tensor-parallel size than target model

**Link**: https://github.com/vllm-project/vllm/issues/4632
**State**: closed
**Created**: 2024-05-06T18:14:40+00:00
**Closed**: 2024-06-25T09:56:08+00:00
**Comments**: 5
**Labels**: help wanted, performance, speculative-decoding

### Description

## Overview
Speculative decoding allows a speedup for memory-bound LLMs by using a fast proposal method to propose tokens that are verified in a single forward pass by the larger LLM. Papers report 2-3x speedup for bs=1, in Anyscale's fork we see up to 2x speedup with a small draft model for bs=8 (30% for bs=16) (we can improve this! see https://github.com/vllm-project/vllm/issues/4630 if you want to help).

A key optimization for small models (68m/160m domain) is to use tensor-parallel degree 1, even if the target model is using tensor-parallel degree 4 or 8. In our fork, this reduces proposal time from 5ms/tok to 1.5ms/tok. This will allow a well-aligned 68m draft model to get 2x per-user throughput improvement on 70B target model.

Furthermore, a 1B/7B proposer model may ideally be placed on TP=2 or TP=4, while the larger model is placed on TP=8. vLLM should support these configuration so the community can use the configuration best for their draft model.

## Design suggestio

[... truncated for brevity ...]

---

## Issue #N/A: missing latest tag in vllm-cpu image

**Link**: https://github.com/vllm-project/vllm/issues/15142
**State**: closed
**Created**: 2025-03-19T16:27:37+00:00
**Closed**: 2025-03-20T08:19:12+00:00
**Comments**: 1
**Labels**: installation

### Description

### Your current environment

current CI/CD builds a new vllm-cpu image on new releases which is great. but ‘latest’ tag is missing which requires having to keep changing the image manually on every release. 

we expect the CI/CD not only to create a versioned tag, but also to tag latest once a new release is out.

### How would you like to use vllm

I want to run inference

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: Model serving failed with these arguments --tensor-parallel-size 2 --pipeline-parallel-size 2

**Link**: https://github.com/vllm-project/vllm/issues/7474
**State**: closed
**Created**: 2024-08-13T15:47:01+00:00
**Closed**: 2024-12-12T02:06:42+00:00
**Comments**: 3
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: Could not collect
CMake version: version 3.30.2
Libc version: glibc-2.31

Python version: 3.11.9 | packaged by conda-forge | (main, Apr 19 2024, 18:36:13) [GCC 12.3.0] (64-bit runtime)
Python platform: Linux-5.15.0-1066-aws-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA L4
GPU 1: NVIDIA L4
GPU 2: NVIDIA L4
GPU 3: NVIDIA L4

Nvidia driver version: 535.183.01
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.2
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.9.2
/usr/lib/x86_64-linux-gnu/libcudnn_ad

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: NVIDIA多型号的GPU如何利用到？

**Link**: https://github.com/vllm-project/vllm/issues/5704
**State**: closed
**Created**: 2024-06-20T06:56:23+00:00
**Closed**: 2024-06-21T05:11:23+00:00
**Comments**: 1
**Labels**: usage

### Description

### Your current environment


![gpu](https://github.com/vllm-project/vllm/assets/47100639/772cf312-d447-472c-918f-b3e495c04a9b)
vllm本地部署模型加载的时候只用到了RTX3060显存，跑9B的模型无法跑起来，有没有办法可以实现在多个卡上面实现并行？


### How would you like to use vllm

I want to run inference of a [specific model](put link here). I don't know how to integrate it with vllm.


---

## Issue #N/A: [Bug]: qwen2.5vl internal server error when processing videos from split_video_ffmpeg after realease 0.8.3

**Link**: https://github.com/vllm-project/vllm/issues/17775
**State**: closed
**Created**: 2025-05-07T09:03:18+00:00
**Closed**: 2025-05-07T15:36:08+00:00
**Comments**: 3
**Labels**: bug

### Description

### Your current environment

<details>
vllm 0.8.3 and vllm 0.7.3
</details>


### 🐛 Describe the bug

I tried to use scenedetect to split video into small slices e.g. :split_video_ffmpeg(video_path, scene_list, output_dir=output_dir, show_progress=True, show_output=True)

And I found out vllm can deal with original video but not those slice of videos. 
The log report says: vllm/multimodal/video.py", line 174, in load_bytes | assert i == num_frames | | AssertionError

I went to the source code and found out that code was added in realease 0.8.3

![Image](https://github.com/user-attachments/assets/461b57f6-cdeb-4316-897e-70a6d5386d41)

So I install vllm==0.7.3 instead and it solves that problem, that is the code of 0.7.3

![Image](https://github.com/user-attachments/assets/254a4b6c-82b7-43ea-84dc-709f524c1502)

Please check that case, thankyou very much!


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the 

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]:  Question about Grouped-query attention (GQA)

**Link**: https://github.com/vllm-project/vllm/issues/13222
**State**: closed
**Created**: 2025-02-13T13:14:37+00:00
**Closed**: 2025-02-14T16:24:47+00:00
**Comments**: 1
**Labels**: misc

### Description

### Implementation of Grouped-query attention (GQA)

Hello:) I was wondering whether [Grouped-query attention](https://arxiv.org/pdf/2305.13245#:~:text=Multi%2Dquery%20attention%20shares%20single,head%20and%20multi%2Dquery%20attention) (GQA) is implemented in vLLM. I see that Llama3 models come with this feature in their [architecture](https://arxiv.org/pdf/2407.21783), and they are available through vLLM. Are they using GQA in the backend?

Thanks a lot and sorry for the inconveniences

### Before submitting a new issue...

- [ ] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: Only apply Guided/Structured grammar after reasoning steps in Reasoning models

**Link**: https://github.com/vllm-project/vllm/issues/12619
**State**: closed
**Created**: 2025-01-31T16:49:13+00:00
**Closed**: 2025-03-05T05:01:40+00:00
**Comments**: 5
**Labels**: feature request, structured-output

### Description

### 🚀 The feature, motivation and pitch

Only apply Guided/Structured grammar only in the answer for reasoning model. i.e. for DeepSeek R1 only enforce grammar inside `<answer></answer>` or after `</think>`
This would make Reasoning models more useful in agent workflow expecting structured output.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: classification with BartForSequenceClassification?

**Link**: https://github.com/vllm-project/vllm/issues/1187
**State**: closed
**Created**: 2023-09-27T00:53:52+00:00
**Closed**: 2024-03-13T11:14:11+00:00
**Comments**: 2

### Description

another huge problem requiring speed is BartForSequenceClassification ala "facebook/bart-large-mnli"

---

## Issue #N/A: [Bug]: vllm async engine can not use adag

**Link**: https://github.com/vllm-project/vllm/issues/8158
**State**: closed
**Created**: 2024-09-04T16:30:14+00:00
**Closed**: 2024-09-24T20:13:10+00:00
**Comments**: 1
**Labels**: bug

### Description

### Your current environment

PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.2 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.2
Libc version: glibc-2.35

Python version: 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.2.0-37-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.2.128
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA RTX A6000
GPU 1: NVIDIA RTX A6000
GPU 2: NVIDIA RTX A6000
GPU 3: NVIDIA RTX A6000

Nvidia driver version: 530.30.02
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:                      48 bits 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vllm 0.6.3.post1 does not work with `response_format`

**Link**: https://github.com/vllm-project/vllm/issues/9900
**State**: closed
**Created**: 2024-11-01T02:18:13+00:00
**Closed**: 2024-11-05T22:48:53+00:00
**Comments**: 2
**Labels**: bug

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

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.12.7 (main, Oct  1 2024, 08:52:12) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.8.0-1017-aws-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A10G
Nvidia driver version: 550.120
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: What's the relationship between KV cache and MAX_SEQUENCE_LENGTH.

**Link**: https://github.com/vllm-project/vllm/issues/10517
**State**: closed
**Created**: 2024-11-21T05:49:40+00:00
**Closed**: 2025-04-02T02:06:20+00:00
**Comments**: 4
**Labels**: usage, stale

### Description

### Your current environment

GPU : H100 80G *2
Model : Llama 3.1 70B

Model Params:
~~~
      env:
        - name: MODEL_NAME
          value: /mnt/models/models--meta-llama--llama-3-1-70b-instruct
        - name: DTYPE_STR
          value: float16
        - name: MAX_SEQUENCE_LENGTH
          value: '20000'
        - name: MAX_BATCH_SIZE
          value: '4'
        - name: MAX_NEW_TOKENS
          value: '4096'
        - name: MAX_LOG_LEN
          value: '100'
        - name: DEFAULT_INCLUDE_STOP_SEQS
          value: 'false'
        - name: NUM_GPUS
          value: '2'
        - name: CUDA_VISIBLE_DEVICES
          value: '0,1'
        - name: HUGGINGFACE_HUB_CACHE
          value: /mnt/models/
        - name: HF_MODULES_CACHE
          value: /tmp/huggingface/modules
        - name: PORT
          value: '3000'
~~~

Initializing an LLM engine (v0.5.4) with config
~~~
model='/mnt/models/models--meta-llama--llama-3-1-70b-instruct', speculative_c

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: GLM4 function call is supported ?

**Link**: https://github.com/vllm-project/vllm/issues/6721
**State**: closed
**Created**: 2024-07-24T03:12:49+00:00
**Closed**: 2024-11-25T02:04:33+00:00
**Comments**: 2
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

GLM4 function call is supported ?

### Alternatives

GLM4 function call is supported ?

### Additional context

GLM4 function call is supported ?

---

## Issue #N/A: [Bug]: unhandled system error with NCCL on v0.5.0.post1

**Link**: https://github.com/vllm-project/vllm/issues/5828
**State**: closed
**Created**: 2024-06-25T16:20:41+00:00
**Closed**: 2024-06-25T20:43:36+00:00
**Comments**: 3
**Labels**: bug

### Description

### Your current environment

```text
Collecting environment information...
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.29.2
Libc version: glibc-2.35

Python version: 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.5.0-1022-azure-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: Tesla T4
Nvidia driver version: 535.171.04
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:                      48 bits physical, 48 bits virtual
B

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: any plan to support V1 engine fp8 kvcache with flashinfer?

**Link**: https://github.com/vllm-project/vllm/issues/20360
**State**: open
**Created**: 2025-07-02T07:56:17+00:00
**Comments**: 1
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

vllm already supports fp8 kvcache of v0 engine + flashinfer, and fp8 kvcache of v1 engine + flashattn. Is there any plan to support fp8 kvcache of v1 engine + flashinfer?

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: gptq model fails on pascal gpu with long prompt

**Link**: https://github.com/vllm-project/vllm/issues/6567
**State**: closed
**Created**: 2024-07-19T08:19:47+00:00
**Closed**: 2024-08-27T03:52:11+00:00
**Comments**: 1
**Labels**: bug

### Description

### Your current environment

```text
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.2 LTS (x86_64)
GCC version: (Ubuntu 11.3.0-1ubuntu1~22.04.1) 11.3.0
Clang version: Could not collect
CMake version: version 3.30.0
Libc version: glibc-2.35

Python version: 3.10.6 (main, May 29 2023, 11:10:38) [GCC 11.3.0] (64-bit runtime)
Python platform: Linux-4.14.83-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: Tesla P40
GPU 1: Tesla P40

Nvidia driver version: 535.129.03
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.3
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.9.3
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.9.3
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8.9.3
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_train.so.8.9.3
/usr/lib/x86_6

[... truncated for brevity ...]

---

## Issue #N/A: awq CUDA error: an illegal memory access was encountered

**Link**: https://github.com/vllm-project/vllm/issues/1830
**State**: closed
**Created**: 2023-11-29T08:57:46+00:00
**Closed**: 2023-11-30T09:14:47+00:00
**Comments**: 3

### Description

hi, 

I get an "an illegal memory access was encountered" error when inference [deepseek-coder-33B-base-AWQ](https://huggingface.co/TheBloke/deepseek-coder-33B-base-AWQ),  which is a Llama2 (GQA) architecture model, but the smaller model is fine([deepseek-coder-6.7B-base-AWQ](https://huggingface.co/TheBloke/deepseek-coder-6.7B-base-AWQ)), the relevant information as follows:

## Environment
python==3.8
torch==2.0.1+cu118
transformers==4.34.1
vllm==0.2.2

## Code
````
from vllm import LLM, SamplingParams
import torch

model_path = "deepseek-coder-33b-base-awq"

sampling_params = SamplingParams(temperature=0.0, 
                                      n=1,
                                      use_beam_search=False,
                                      top_p=1, top_k=-1, max_tokens=200, 
                                      skip_special_tokens=False, 
                                      stop_token_ids=stop_token_ids)

llm = LLM(model=model_path, quantization="aw

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]:  vllm infer QWQ32B can‘t enable sliding window

**Link**: https://github.com/vllm-project/vllm/issues/17306
**State**: closed
**Created**: 2025-04-28T12:11:27+00:00
**Closed**: 2025-05-15T05:29:39+00:00
**Comments**: 1
**Labels**: usage

### Description

### Your current environment

工具： Vllm=0.8.4
模型：Qwq32B
配置：
{
"architectures": [
"Qwen2ForCausalLM"
],
"attention_dropout": 0.0,
"bos_token_id": 151643,
"eos_token_id": 151645,
"hidden_act": "silu",
"hidden_size": 5120,
"initializer_range": 0.02,
"intermediate_size": 27648,
"max_position_embeddings": 40960,
"max_window_layers": 64,
"model_type": "qwen2",
"num_attention_heads": 40,
"num_hidden_layers": 64,
"num_key_value_heads": 8,
"rms_norm_eps": 1e-05,
"rope_theta": 1000000.0,
"sliding_window": 40960,
"tie_word_embeddings": false,
"torch_dtype": "bfloat16",
"transformers_version": "4.43.1",
"use_cache": true,
"use_sliding_window": true,
"vocab_size": 152064
}

启动命令：
python -m vllm.entrypoints.openai.api_server --model /home/user/Models/QwQ-32B --host "::" --port 8600 --tensor-parallel-size 8 --gpu-memory-utilization 0.95 --max-model-len 40960 --dtype bfloat16 --max-num-seqs 16 --served-model-name qwq32b --swap-space 10 --enable_prefix_caching --enable-chunked-prefill --use-v2-block-man

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Offline multi-node inference

**Link**: https://github.com/vllm-project/vllm/issues/17711
**State**: open
**Created**: 2025-05-06T10:47:00+00:00
**Comments**: 5
**Labels**: usage

### Description

### Your current environment

Hello everybody
According to the vLLM documentation, it seems that in order to performe multi-node inference, one has to do this in an online setting.
I am working with access to a GPU cluster, where the compute nodes do not have internet access. My goal is to run inference with llama 3.3 70B Instruct on a file using 4 nodes (4 gpus per node), however, if I try to use the LLM class, I get an error saying that data parallelism isn't possible and I should use AsyncEngine instead.
However, asyncEngine cannot be used with the chat() method, thus I am currently unable to perform inference on this file containing samples.
I hereby wanted to ask if it's possible to perform offline multi-node inference and if so whether there are guides or further documentation on it, thank you

### How would you like to use vllm

I want to run inference of a [specific model](put link here). I don't know how to integrate it with vllm.


### Before submitting a new issue...

- [x] 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Qwen2-72B-Instruct-gptq-int4 Repetitive issues

**Link**: https://github.com/vllm-project/vllm/issues/5663
**State**: closed
**Created**: 2024-06-19T02:29:03+00:00
**Closed**: 2024-11-25T02:05:43+00:00
**Comments**: 3
**Labels**: bug, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### 🐛 Describe the bug

Machine A800, VLLM 0.5.0, PROMPT=开始, output max tokens = 2048, Temperature sets 0.7

VLLM loads Qwen2-72b-InStruct-GPTQ-IT4, and uses the Benchmark script of VLLM to do concurrent testing. Whether it is a concurrent limit or 10 concurrency restrictions, the output will be repeated.
https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py
![image](https://github.com/vllm-project/vllm/assets/57557769/f3440a14-71a1-4b8c-b3e4-6a66aaba4aa8)


---

## Issue #N/A: [Bug]: Performance regression when use PyTorch regional compilation

**Link**: https://github.com/vllm-project/vllm/issues/12410
**State**: closed
**Created**: 2025-01-24T15:10:46+00:00
**Closed**: 2025-02-05T21:24:27+00:00
**Comments**: 1
**Labels**: bug

### Description

### Your current environment

<details>
<summary>Run on hpu backend  on version from  https://github.com/HabanaAI/vllm-fork </summary>

```text
INFO 01-24 15:38:37 __init__.py:188] Automatically detected platform hpu.
Collecting environment information...
PyTorch version: 2.5.1a0+git354fc07
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov  6 2024, 20:22:13) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-113-generic-x86_64-with-glibc2.35
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

## Issue #N/A: [Bug]: Wrong lora mapping during prompt logprobs computing

**Link**: https://github.com/vllm-project/vllm/issues/16668
**State**: open
**Created**: 2025-04-15T15:13:52+00:00
**Comments**: 3
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 04-15 08:06:07 [__init__.py:239] Automatically detected platform cuda.
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
Python platform: Linux-5.4.203-1-tlinux4-0011.spr.0001.2-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H20
GPU 1: NVIDIA H20
GPU 2: NVIDIA H20
GPU 3: NVIDIA H20
GPU 4: NVIDIA H20
GPU 5: NVIDIA H20
GPU 6: NVIDIA H20
GPU 7: NVIDIA H20

Nvidia driver version: 535.161.08
cuDNN version: Could not collec

[... truncated for brevity ...]

---

## Issue #N/A: `stream` should only accept type Boolean when using OpenAI API Server spec

**Link**: https://github.com/vllm-project/vllm/issues/1273
**State**: closed
**Created**: 2023-10-06T06:52:39+00:00
**Closed**: 2024-03-13T09:44:07+00:00
**Comments**: 1

### Description

The current behaviour of vLLM does not match the behaviour of OpenAI and Azure OpenAI when it comes to the `stream` parameter in the request body.

Current behaviour of OpenAI and Azure OpenAI:
- Only `"stream": true` or `"stream": false` are accepted. Setting `"stream": "true"` or `"stream": "false"` (or any other non-Boolean values) will raise the following error:
```
{
  "error": {
    "message": "'false' is not of type 'boolean' - 'stream'",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

Current behaviour of vLLM:
- The following values for the `stream` request body parameter are accepted by vLLM: `true`, `"true"`, `false`, `"false"`
- Any other values will raise the following error:
```
{
    "object": "error",
    "message": "[{'loc': ('body', 'stream'), 'msg': 'value could not be parsed to a boolean', 'type': 'type_error.bool'}]",
    "type": "invalid_request_error",
    "param": null,
    "code": null
}
```
- It se

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: With ROCm and certain HF models that require 'trust-remote-code', you get VLLM_RPC_TIMEOUT and failure to finish loading.

**Link**: https://github.com/vllm-project/vllm/issues/11232
**State**: closed
**Created**: 2024-12-16T11:28:31+00:00
**Closed**: 2025-04-18T02:06:36+00:00
**Comments**: 4
**Labels**: bug, rocm, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
user1@dev1:~/vllm$ python collect_env.py
WARNING 12-16 05:24:35 rocm.py:31] `fork` method is not supported by ROCm. VLLM_WORKER_MULTIPROC_METHOD is overridden to `spawn` instead.
Collecting environment information...
PyTorch version: 2.6.0.dev20241211+rocm6.1
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.1.40091-a8dbc0c19

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (conda-forge gcc 12.1.0-17) 12.1.0
Clang version: Could not collect
CMake version: version 3.31.1
Libc version: glibc-2.35

Python version: 3.10.16 | packaged by conda-forge | (main, Dec  5 2024, 14:16:10) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-5.15.0-97-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 11.7.64
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Instinct MI250X/MI250 (gfx90a:sramecc+:xnac

[... truncated for brevity ...]

---

## Issue #N/A: Answers generated by vllm 0.2.2+cu121 not align with vllm 0.2.2+cu118, the former is more similar to hf

**Link**: https://github.com/vllm-project/vllm/issues/1721
**State**: closed
**Created**: 2023-11-20T08:02:24+00:00
**Closed**: 2024-03-20T12:50:53+00:00
**Comments**: 4

### Description

the latter is same as vllm 0.2.1.post1

python3.9

---

## Issue #N/A: [Usage]: torchrun data parallel and tensor parallel at the same time

**Link**: https://github.com/vllm-project/vllm/issues/15672
**State**: open
**Created**: 2025-03-28T03:38:33+00:00
**Comments**: 4
**Labels**: usage, stale

### Description

### Your current environment

```
Versions of relevant libraries:
[pip3] numpy==1.26.4
[pip3] nvidia-cublas-cu12==12.4.5.8
[pip3] nvidia-cuda-cupti-cu12==12.4.127
[pip3] nvidia-cuda-nvrtc-cu12==12.4.127
[pip3] nvidia-cuda-runtime-cu12==12.4.127
[pip3] nvidia-cudnn-cu12==9.1.0.70
[pip3] nvidia-cufft-cu12==11.2.1.3
[pip3] nvidia-curand-cu12==10.3.5.147
[pip3] nvidia-cusolver-cu12==11.6.1.9
[pip3] nvidia-cusparse-cu12==12.3.1.170
[pip3] nvidia-cusparselt-cu12==0.6.2
[pip3] nvidia-nccl-cu12==2.21.5
[pip3] nvidia-nvjitlink-cu12==12.4.127
[pip3] nvidia-nvtx-cu12==12.4.127
[pip3] torch==2.6.0
[pip3] torchaudio==2.6.0
[pip3] torchvision==0.21.0
[pip3] triton==3.2.0

Name: vllm
Version: 0.8.3.dev77+gb4245a48

[conda] cuda-cudart               12.4.127             h99ab3db_0  
[conda] cuda-cudart-dev           12.4.127             h99ab3db_0  
[conda] cuda-cudart-dev_linux-64  12.4.127             hd681fbe_0  
[conda] cuda-cudart-static        12.4.127             h99ab3db_0  
[conda] cuda-cudar

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: performance optimization by nanoflow

**Link**: https://github.com/vllm-project/vllm/issues/8150
**State**: closed
**Created**: 2024-09-04T09:06:03+00:00
**Closed**: 2025-01-04T01:58:21+00:00
**Comments**: 2
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

optimize pipeline design
https://github.com/efeslab/Nanoflow
https://arxiv.org/abs/2408.12757

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: Evaluate multiple ngram speculations in speculative decoding

**Link**: https://github.com/vllm-project/vllm/issues/6785
**State**: closed
**Created**: 2024-07-25T12:35:32+00:00
**Closed**: 2024-11-24T02:07:16+00:00
**Comments**: 3
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

 During the ngram-spec-decode stage, I've always had a question: In RAG, there isn't just one document relevant to the answer; why don't we first let the large model generate 3 tokens, and then take all possible results in the N-gram?

In simpler terms, imagine you're looking for an object in several rooms but can only carry three things at once. You might want to pick up some important items now so you won't forget them when carrying more stuff later. This way, you make sure your search is efficient and effective.

### Alternatives

_No response_

### Additional context

_No response_

---

## Issue #N/A: Large `max_tokens` causes vLLM server to throw AsyncEngineDeadError and the server doesn't recover from the error

**Link**: https://github.com/vllm-project/vllm/issues/1543
**State**: closed
**Created**: 2023-11-01T22:55:18+00:00
**Closed**: 2024-03-13T12:42:02+00:00
**Comments**: 1

### Description

Hi vLLM team, it appears that a large sequence length (for instance, `max_tokens` being set to 10000) can cause the vLLM server to throw an `AsyncEngineDeadError` error. After the error, the server doesn't recover and becomes unable to handle future requests. Could you help take a look at this issue? Thank you so much.

Specifications for reproducing the error are:
- Sampling parameters: `SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, temperature=0.01, top_p=0.7, top_k=-1, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=[], ignore_eos=True, max_tokens=10000, logprobs=None, skip_special_tokens=True)`
- Model: LLaMA 2 70b
- vLLM version: 0.2.0
- GPU: 8 L4 (24G) GPUs
- Launch parameters (other parameters are unset)
  - --tensor-parallel-size=8
  - --swap-space=16
  - --gpu-memory-utilization=0.9

Error logs are attached below:
P1
<img width="1296" alt="Screenshot 2023-11-01 at 3 36 24 PM" src="https://github.com/vllm-project

[... truncated for brevity ...]

---

