# documentation - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 4
- Closed Issues: 26

### Label Distribution

- documentation: 30 issues
- stale: 6 issues
- rocm: 1 issues

---

## Issue #N/A: Guidance on how many requests can be processed at a time?

**Link**: https://github.com/vllm-project/vllm/issues/1555
**State**: closed
**Created**: 2023-11-03T19:27:46+00:00
**Closed**: 2024-12-01T02:15:58+00:00
**Comments**: 4
**Labels**: documentation, stale

### Description

Hello - 

I am trying to understand how many requests can be processed in parallel with the llm_engine, and what keeps requests WAITING. I see various variables like "max_num_batched_tokens" and "max_num_seqs", but more details or documentation describing how this process occurs would be helpful. Moreover, how can we tune our system to do process more requests in parallel (e.g. use more GPUs if available, use smaller models, use smaller context windows, etc.)

---

## Issue #N/A: [Doc]: No max_model_len parameter in the LLM class

**Link**: https://github.com/vllm-project/vllm/issues/13021
**State**: closed
**Created**: 2025-02-10T08:23:08+00:00
**Closed**: 2025-02-10T16:16:36+00:00
**Comments**: 2
**Labels**: documentation

### Description

### 📚 The doc issue

In this url: https://docs.vllm.ai/en/latest/serving/offline_inference.html
I see that there is no max_model_len parameter in the LLM class, but the documentation still says 
    llm = LLM(model="adept/fuyu-8b",
    max_model_len=2048,
    max_num_seqs=2)
Btw, I wonder how can I change the max_seq_len when I use offline_inference? 

### Suggest a potential alternative/fix

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Add documents on how to add new models

**Link**: https://github.com/vllm-project/vllm/issues/65
**State**: closed
**Created**: 2023-05-04T09:05:56+00:00
**Closed**: 2023-06-06T03:01:28+00:00
**Comments**: 0
**Labels**: documentation

### Description

No description provided.

---

## Issue #N/A: [Doc]: Improve CPU documentation for ARM

**Link**: https://github.com/vllm-project/vllm/issues/19603
**State**: open
**Created**: 2025-06-13T08:08:35+00:00
**Comments**: 0
**Labels**: documentation

### Description

### 📚 The doc issue

At the moment, vLLM supports a variety of GPUs, some NPUs/TPUs, and (at the very least) x86 and ARM CPUs. At the moment, however, the documentation is very sparse for ARM CPUs, other than that it *can* be used. As a matter of fact, not including the API docs, I can only find [this one page](https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html?h=arm) which mentions it. Notably missing is any mention of ARM in the [Supported Hardware](https://docs.vllm.ai/en/latest/features/quantization/supported_hardware.html) page for quantization.

### Suggest a potential alternative/fix

It would be helpful to have at least some indication of what is and isn't supported on ARM CPUs. I would be happy to contribute this myself, but I'm not sure of the best way to determine compatibility. Would it make sense to simply run a small model of every possible quantization format to see what happens? Or is there something a bit more elegant?

### Before submitting a new iss

[... truncated for brevity ...]

---

## Issue #N/A: [Doc]: API reference for LLM class

**Link**: https://github.com/vllm-project/vllm/issues/4684
**State**: closed
**Created**: 2024-05-08T15:44:46+00:00
**Closed**: 2024-05-14T00:47:43+00:00
**Comments**: 2
**Labels**: documentation

### Description

### 📚 The doc issue

I can't find on https://docs.vllm.ai/ the API reference for the `LLM` class. This would make it easier than digging through the code to look at the docstrings - the examples in the docs don't explain most of the arguments.

### Suggest a potential alternative/fix

_No response_

---

## Issue #N/A: [Bug]: low quality of deepseek-vl2 when using vllm

**Link**: https://github.com/vllm-project/vllm/issues/14421
**State**: closed
**Created**: 2025-03-07T09:03:24+00:00
**Closed**: 2025-03-11T04:37:13+00:00
**Comments**: 13
**Labels**: documentation

### Description

### 📚 The doc issue

When I use the official inference code of deepseek-vl2, the model output seems normal，
Question: “图片中的角色是哪部动漫作品中的人物？”
![Image](https://github.com/user-attachments/assets/44dca395-f1dd-4b0c-87d6-d492df512976)
Model Output：
![Image](https://github.com/user-attachments/assets/39332703-d613-403d-bcb0-e0cbb8e266d7)

But when I use vllm for inference, and I use the [chat_template](https://github.com/vllm-project/vllm/blob/main/examples/template_deepseek_vl2.jinja)，the model output seems abnormal.
I start vllm by：
`CUDA_VISIBLE_DEVICES=7 vllm serve deepseek-vl2 --port 8102 --max-model-len 4096 --hf_overrides '{"architectures":["DeepseekVLV2ForCausalLM"]}' --gpu-memory-utilization 0.9 --chat_template ./template_deepseek_vl2.jinja`
Model Output：
![Image](https://github.com/user-attachments/assets/ea545f6e-fe01-4f67-a50a-65bbed41f86a)

I feel that the accuracy has dropped significantly. I don't know if it's a problem with chat_template or somewhere else. 

### Suggest a pote

[... truncated for brevity ...]

---

## Issue #N/A: [Doc]: Supported Hardware for Quantization Kernels

**Link**: https://github.com/vllm-project/vllm/issues/6979
**State**: closed
**Created**: 2024-07-31T07:16:05+00:00
**Closed**: 2024-07-31T15:33:53+00:00
**Comments**: 1
**Labels**: documentation

### Description

### 📚 The doc issue

I'm confused what "the quantization method is supported" mean?  Ampere arch doesn't support FP8, according to Nvidia. So does this mean the FP8 operation is supported on A100/A800 GPU?  Or just we can conver the weight parameters form FP16 to FP8? 

### Suggest a potential alternative/fix

_No response_

---

## Issue #N/A: [Doc]: ROCm installation instructions do not work

**Link**: https://github.com/vllm-project/vllm/issues/6762
**State**: closed
**Created**: 2024-07-24T21:49:00+00:00
**Closed**: 2024-09-25T14:33:49+00:00
**Comments**: 5
**Labels**: documentation, rocm

### Description

### 📚 The doc issue

Following the instructions at https://docs.vllm.ai/en/latest/getting_started/amd-installation.html#build-from-source-rocm, using the exact Docker image mentioned (pytorch_rocm6.1.2_ubuntu20.04_py3.9_pytorch_staging.sif, although with a custom Python venv and Pytorch install), and run into the following error when running `python setup.py develop`:
```
Building PyTorch for GPU arch: gfx90a
-- Could NOT find HIP: Found unsuitable version "0.0.0", but required is at least "1.0" (found /opt/rocm)
HIP VERSION: 0.0.0
CMake Warning at .venv/lib/python3.10/site-packages/torch/share/cmake/Torch/TorchConfig.cmake:22 (message):
  static library kineto_LIBRARY-NOTFOUND not found.
Call Stack (most recent call first):
  .venv/lib/python3.10/site-packages/torch/share/cmake/Torch/TorchConfig.cmake:120 (append_torchlib_if_found)
  CMakeLists.txt:67 (find_package)


CMake Error at CMakeLists.txt:108 (message):
  Can't find CUDA or HIP installation.
  ```

The docker 

[... truncated for brevity ...]

---

## Issue #N/A: [Doc]: How to Build an AI Knowledge Base like AI-Ask with vLLM

**Link**: https://github.com/vllm-project/vllm/issues/7133
**State**: closed
**Created**: 2024-08-05T01:35:50+00:00
**Closed**: 2024-08-05T01:46:22+00:00
**Comments**: 1
**Labels**: documentation

### Description

### 📚 The doc issue

Hello,

I am interested in building an AI knowledge base similar to the AI-Ask feature provided by vLLM. I have a few questions regarding its implementation:

	1.	How can I achieve the construction of such an AI knowledge base using vLLM?
	2.	Is it based on any specific open-source solutions or frameworks?
	3.	Are there any guidelines or documentation that could help in setting up a similar system?

Any insights or pointers would be greatly appreciated!

<img width="792" alt="image" src="https://github.com/user-attachments/assets/e4f6f7ca-72b2-4f9d-afbf-f62cbcba69e0">


Thank you!

### Suggest a potential alternative/fix

_No response_

---

## Issue #N/A: [Doc]: Backwards compability with PyTorch

**Link**: https://github.com/vllm-project/vllm/issues/19359
**State**: open
**Created**: 2025-06-09T10:33:32+00:00
**Comments**: 4
**Labels**: documentation

### Description

### 📚 The doc issue

vLLM has become a very important component of the OSS ecosystem and is tightly integrated with [`torch`](https://github.com/pytorch/pytorch) and [`triton-lang`](https://github.com/triton-lang/triton/tree/main)

From looking at the requirements it looks like the philosophy is to only ensure full compatibility with the newest PyTorch version (currently 2.7.0) [here](https://github.com/vllm-project/vllm/blob/59abbd84f90e5930c37e205de8849ac4fa8a96c7/requirements/cuda.txt#L9) which then also somewhat automatically pins the required triton version.

While torch tries to stay backward compatible with earlier torch versions, this is not always ensured for libraries depending on torch & triton. VLLM for example uses custom triton code here: https://github.com/vllm-project/vllm/blob/59abbd84f90e5930c37e205de8849ac4fa8a96c7/vllm/attention/ops/prefix_prefill.py#L134 (and many other places) that is not always compatible with earlier triton versions. E.g. triton=3.1.0 doesn't ha

[... truncated for brevity ...]

---

## Issue #N/A: [Doc]: Update outdated note: LMCache now supports chunked prefill

**Link**: https://github.com/vllm-project/vllm/issues/16452
**State**: closed
**Created**: 2025-04-11T03:58:33+00:00
**Closed**: 2025-04-18T05:12:43+00:00
**Comments**: 0
**Labels**: documentation

### Description

### 📚 The doc issue

In the file [examples/offline_inference/cpu_offload_lmcache.py](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/cpu_offload_lmcache.py), line 43 states:

`# Note that LMCache is not compatible with chunked prefill for now.`

This is now outdated. Both vLLM and LMCache have merged PRs to fully support chunked prefill:

[vLLM PR #14505](https://github.com/vllm-project/vllm/pull/14505), [LMCache PR #392](https://github.com/LMCache/LMCache/pull/392)

The current note may mislead users into disabling a working feature.

### Suggest a potential alternative/fix

Update the comment to either:

`# Note: LMCache supports chunked prefill (see vLLM#14505, LMCache#392).  `

Or remove it entirely if compatibility is now considered stable/default.

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/lates

[... truncated for brevity ...]

---

## Issue #N/A: [Doc]: Add GitHub Action to auto-sync Dockerfile dependency graph

**Link**: https://github.com/vllm-project/vllm/issues/11880
**State**: closed
**Created**: 2025-01-09T03:32:02+00:00
**Closed**: 2025-04-10T13:43:06+00:00
**Comments**: 1
**Labels**: documentation, stale

### Description

### 📚 The doc issue

Currently, the Dockerfile dependency graph (docs/source/assets/contributing/dockerfile-stages-dependency.png) may become out of sync with the actual Dockerfile when changes are made. This can lead to outdated or incorrect documentation.

### Suggest a potential alternative/fix

I propose adding a GitHub Action workflow that automatically:

1. Regenerates the dependency graph when Dockerfile changes.
2. Creates a PR if the graph has changed.

I've prepared a draft workflow here: https://github.com/vllm-project/vllm/pull/11879

This will help ensure the documentation stays accurate and reduce maintenance overhead.


### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Doc]: Documentation source code hyperlinks do not always point to the correct source code

**Link**: https://github.com/vllm-project/vllm/issues/17120
**State**: closed
**Created**: 2025-04-24T16:11:06+00:00
**Closed**: 2025-04-24T17:39:44+00:00
**Comments**: 3
**Labels**: documentation

### Description

### 📚 The doc issue

For example, the  documenation of the ``generate`` method in [LLM Class](https://docs.vllm.ai/en/latest/api/offline_inference/llm.html#vllm.LLM.generate).
Clicking its ``source`` button, we find that it points to [an unrelated function in utils](https://github.com/vllm-project/vllm/blob/main/vllm/utils.py#L373).

This is just one example, a similar thing happens with the link for ``LLM.encode``.

### Suggest a potential alternative/fix

I am not familiar with the way Sphinx generates documentation, but there may be something to be done in one of its configs.

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Doc]: OpenAI Chat Completion Client For Multimodal missing using video as input for qwen2-vl

**Link**: https://github.com/vllm-project/vllm/issues/10316
**State**: closed
**Created**: 2024-11-14T07:11:12+00:00
**Closed**: 2024-11-14T08:35:59+00:00
**Comments**: 2
**Labels**: documentation

### Description

### 📚 The doc issue

the [doc](https://docs.vllm.ai/en/latest/getting_started/examples/openai_chat_completion_client_for_multimodal.html) only shows using multiple images, but how to use video as input?


### Suggest a potential alternative/fix

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Doc]: ValueError: Qwen2_5_VLForSequenceClassification has no vLLM implementation and the Transformers implementation is not compatible with vLLM.

**Link**: https://github.com/vllm-project/vllm/issues/15006
**State**: closed
**Created**: 2025-03-18T06:09:09+00:00
**Closed**: 2025-03-31T09:04:10+00:00
**Comments**: 2
**Labels**: documentation

### Description

### 📚 The doc issue

ValueError: Qwen2_5_VLForSequenceClassification has no vLLM implementation and the Transformers implementation is not compatible with vLLM.

GPU:T4 *4  seq_cls

### Suggest a potential alternative/fix

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Doc]: How can I set the date_string for the chat templates

**Link**: https://github.com/vllm-project/vllm/issues/14344
**State**: closed
**Created**: 2025-03-06T09:00:38+00:00
**Closed**: 2025-03-07T02:50:05+00:00
**Comments**: 2
**Labels**: documentation

### Description

### 📚 The doc issue

Hello everyone,

I use the chat_template like this, similar to [tool_chat_template_llama3.1_json.jinja](https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_llama3.1_json.jinja):
```
{%- if not date_string is defined %}
    {%- set date_string = "26 Jul 2024" %}
{%- endif %}
```
and I call the model like this:
```
client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.0,
            max_completion_tokens=300,
            stream=True,
            tools=tools,
            tool_choice=tool_choice
        )
```
So, now my question: How can I set the date_string variable?

Thank you for your help.

Best regards,
Felix

### Suggest a potential alternative/fix

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which 

[... truncated for brevity ...]

---

## Issue #N/A: The returned results using prompt_logprobs=1

**Link**: https://github.com/vllm-project/vllm/issues/2043
**State**: closed
**Created**: 2023-12-11T23:23:20+00:00
**Closed**: 2024-12-01T02:15:40+00:00
**Comments**: 5
**Labels**: documentation, stale

### Description

It seems that setting prompt_logprobs=1 will return the scoring of the context (i.e., prompt). However, the returned results are a little confusing:

When I used a Llama-2-7b to score a sequece, the returned results look as follow:

[None, {15043: -7.584228515625, 917: -2.512939214706421}, {29892: -1.4937736988067627}, {590: -1.8308428525924683, 306: -1.3464678525924683}, {1024: -0.11963547021150589}, {338: -0.01794273406267166}]

The first and third positions have two keys while the other positions have only 1 key. Is that because the position's word is not the word with the highest prob?

---

## Issue #N/A: [Doc]: can't fing serving_embedding.py

**Link**: https://github.com/vllm-project/vllm/issues/4857
**State**: closed
**Created**: 2024-05-16T09:31:41+00:00
**Closed**: 2024-05-16T19:21:30+00:00
**Comments**: 2
**Labels**: documentation

### Description

### 📚 The doc issue

I see the introduction for OpenAI Embedding Client 
https://docs.vllm.ai/en/latest/getting_started/examples/openai_embedding_client.html
and I see the code in Github also.

But I couldn't find the code in vllm 0.4.2.

### Suggest a potential alternative/fix

_No response_

---

## Issue #N/A: [Doc]: Why is max block_size on CUDA 32?

**Link**: https://github.com/vllm-project/vllm/issues/14319
**State**: open
**Created**: 2025-03-05T23:50:23+00:00
**Comments**: 4
**Labels**: documentation, stale

### Description

### 📚 The doc issue

In the args:
https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py#L454
it says about block_size parameter:

> Token block size for contiguous chunks of tokens. This is ignored on neuron devices and set to --max-model-len. On CUDA devices, only block sizes up to 32 are supported. On HPU devices, block size defaults to 128.

1. Where is this requirement for <= 32 on CUDA devices coming from?
2. I was able to successfully run vLLM with block_size 128 on Hopper and see some minor performance improvement. Is the requirement up to date?
3. In flash attention docs I see that paged attention minimum block size is actually 256:
https://github.com/Dao-AILab/flash-attention/blob/d82bbf26924c492064af8b27ab299ff4808d1bf6/hopper/flash_attn_interface.py#L662
Does vLLM use this interface? How does FA paged_block_size relates to vLLM block_size?

### Suggest a potential alternative/fix

_No response_

### Before submitting a new issue...

- [x] Make sure you alre

[... truncated for brevity ...]

---

## Issue #N/A: [Doc]: Docker+vllm+fastchat deploys multimodal large model Qwen2-vl-7b-instruct(docker+vllm+fastchat部署多模态大模型Qwen2-vl-7b-instruct)

**Link**: https://github.com/vllm-project/vllm/issues/10566
**State**: closed
**Created**: 2024-11-22T06:12:34+00:00
**Closed**: 2025-04-04T02:05:43+00:00
**Comments**: 7
**Labels**: documentation, stale

### Description

### 📚 The doc issue

When testing using the following method, an error is reported, and the terminal log results are shown as follows:

curl -X POST http:(base) root@lxing:~# curl -X POST http://127.0.0.1:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "Qwen2-VL-7B-Instruct",
  "messages": [{"role": "user", "content": "你好！"}],
  "temperature": 1.0,
  "max_tokens": 100
}'
{"object":"error","message":"Internal Server Error","code":50001}


2024-11-22 02:50:27 | INFO | stdout | INFO:     127.0.0.1:55822 - "POST /model_details HTTP/1.1" 200 OK 
2024-11-22 02:50:27 | INFO | stdout | INFO:     127.0.0.1:55838 - "POST /count_token HTTP/1.1" 200 OK
2024-11-22 02:50:27 | INFO | stdout | INFO:     127.0.0.1:55844 - "POST /worker_generate HTTP/1.1" 500 Internal Server Error
2024-11-22 02:50:27 | ERROR | stderr | ERROR:    Exception in ASGI application
2024-11-22 02:50:27 | ERROR | stderr | Traceback (most recent call last):
2024-11-22 02:50:27 |

[... truncated for brevity ...]

---

## Issue #N/A: [Doc]: Question about the timing of prefix caches release for multiples requests with long time interval. 

**Link**: https://github.com/vllm-project/vllm/issues/3910
**State**: closed
**Created**: 2024-04-08T06:29:24+00:00
**Closed**: 2024-04-08T12:34:28+00:00
**Comments**: 2
**Labels**: documentation

### Description

### 📚 The doc issue

It's there has any docs about the  timing of prefix caches release ? specific for multiples requests with long time interval.  

```bash
1rd req [+ long prompt_0] -> after 10 mins -> 2rd req [+ long prompt_0] -> after 5 mins -> 3rd req [+ long prompt_0] -> ...
       |                          |
       |                          |
------ cached --------->  will cache release here?  ------>  ....
```

### Suggest a potential alternative/fix

_No response_

---

## Issue #N/A: [Doc]: Proposing a minor change in "Metrics" documentation.

**Link**: https://github.com/vllm-project/vllm/issues/16783
**State**: closed
**Created**: 2025-04-17T12:47:59+00:00
**Closed**: 2025-04-17T14:10:10+00:00
**Comments**: 3
**Labels**: documentation

### Description

### 📚 The doc issue

In the "Metrics" section of the vLLM website, the vllm:generation_tokens_total metric is currently described as:

[vllm:generation_tokens_total – Generation Tokens/Sec](https://docs.vllm.ai/en/latest/design/v1/metrics.html#grafana-dashboard)

<img width="807" alt="Image" src="https://github.com/user-attachments/assets/fe344a93-814d-4d82-b3af-1aba2848cd1a" />

However, since this metric is of the counter type, it may be more accurate to document it as "Generation Tokens" rather than "Generation Tokens/Sec."



### Suggest a potential alternative/fix

AS-IS:
- `vllm:generation_tokens_total - Generation Tokens/Sec`

TO-BE:
- `vllm:generation_tokens_total - Generation Tokens`

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Doc]: Arguments missing from docs page

**Link**: https://github.com/vllm-project/vllm/issues/18228
**State**: closed
**Created**: 2025-05-15T22:10:29+00:00
**Closed**: 2025-05-17T02:43:46+00:00
**Comments**: 2
**Labels**: documentation

### Description

### 📚 The doc issue

The following are arguments that are available by running `vllm serve --help` with vllm 0.8.4 and are not included in the [0.8.4 engine arguments docs](https://docs.vllm.ai/en/v0.8.4/serving/engine_args.html) page:

```
--allow-credentials
--allowed-headers
--allowed-methods
--allowed-origins
--api-key
--chat-template
--chat-template-content-format
--config
--disable-fastapi-docs
--disable-frontend-multiprocessing
--disable-uvicorn-access-log
--enable-auto-tool-choice
--enable-prompt-tokens-details
--enable-request-id-headers
--enable-server-load-tracking
--enable-ssl-refresh
--help
--host
--lora-modules
--max-log-len
--middleware
--port
--prompt-adapters
--response-role
--return-tokens-as-token-ids
--root-path
--ssl-ca-certs
--ssl-cert-reqs
--ssl-certfile
--ssl-keyfile
--tool-call-parser
--tool-parser-plugin
--uvicorn-log-level
```

It appears that the majority of these arguments are options that are created here:

https://github.com/vllm-project/vllm/blob/0b34593

[... truncated for brevity ...]

---

## Issue #N/A: [Doc]: can not close thinking

**Link**: https://github.com/vllm-project/vllm/issues/19414
**State**: closed
**Created**: 2025-06-10T09:48:37+00:00
**Closed**: 2025-06-10T12:31:19+00:00
**Comments**: 1
**Labels**: documentation

### Description

version=0.9.0.1  set enable_thinking=True , can not close thinking   

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m vllm.entrypoints.openai.api_server \
        --model $modelpath \
        --served-model-name QwQ-32B \
        --trust-remote-code \
        --tensor-parallel-size 4 \
        --max-model-len 30000 \
        --port 8006 \
        --reasoning-parser qwen3 \


response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.3,
            top_p=0.8,
            max_tokens=4000,
            extra_body={"chat_template_kwargs": {"enable_thinking":False}},
        )

---

## Issue #N/A: [Doc]: Typo in prefix_caching.md

**Link**: https://github.com/vllm-project/vllm/issues/14294
**State**: closed
**Created**: 2025-03-05T14:56:14+00:00
**Closed**: 2025-03-05T15:43:14+00:00
**Comments**: 0
**Labels**: documentation

### Description

### 📚 The doc issue

There's a typo in [eviction-LRU doc](https://docs.vllm.ai/en/latest/design/v1/prefix_caching.html#eviction-lru) where an LRU block is called an "LRU black"

### Suggest a potential alternative/fix

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Doc]: Is Qwen2-VL-72B supported?

**Link**: https://github.com/vllm-project/vllm/issues/8682
**State**: closed
**Created**: 2024-09-20T21:05:00+00:00
**Closed**: 2024-09-25T05:33:18+00:00
**Comments**: 9
**Labels**: documentation

### Description

### 📚 The doc issue

Unclear.

### Suggest a potential alternative/fix

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Doc]:  FP8 KV Cache feature only supoorts  Llama 2  so far?

**Link**: https://github.com/vllm-project/vllm/issues/12215
**State**: closed
**Created**: 2025-01-20T09:32:47+00:00
**Closed**: 2025-01-21T02:02:14+00:00
**Comments**: 2
**Labels**: documentation

### Description

### 📚 The doc issue

i assume FP8 KV Cache  can significantly improve inference speed and save memory space.  my questions is how many models does it supports? 
is it true  only Llama 2 is supported? (https://github.com/vllm-project/vllm/tree/main/examples/other/fp8)
because i find that kind of hard to believe 
thank u 

### Suggest a potential alternative/fix

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Documentation on distributed execution

**Link**: https://github.com/vllm-project/vllm/issues/206
**State**: closed
**Created**: 2023-06-22T11:23:16+00:00
**Closed**: 2023-06-26T18:34:25+00:00
**Comments**: 1
**Labels**: documentation

### Description

No description provided.

---

## Issue #N/A: [Doc]: state requirements for testing or update to work for CPU-only

**Link**: https://github.com/vllm-project/vllm/issues/16920
**State**: open
**Created**: 2025-04-21T12:24:29+00:00
**Comments**: 1
**Labels**: documentation

### Description

### 📚 The doc issue

On https://docs.vllm.ai/en/latest/contributing/overview.html#testing, `pip install -r requirements/dev.txt` tries to install [mamba-ssm](https://github.com/vllm-project/vllm/blob/d41faaf9df6d1a741d5fdd4a282b16783cace888/requirements/test.txt#L273) whose [requirements are stated here](https://github.com/state-spaces/mamba/?tab=readme-ov-file#installation).

> * Linux
> * NVIDIA GPU
> * PyTorch 1.12+
> * CUDA 11.6+

The `pip install` command fails otherwise with errors like `NameError: name 'bare_metal_version' is not defined`.

```bash
$ pip install -r requirements/dev.txt
...
Collecting mamba-ssm==2.2.4 (from -r /home/dxia/src/github.com/vllm-project/vllm/requirements/test.txt (line 273))
  Downloading mamba_ssm-2.2.4.tar.gz (91 kB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... error
  error: subprocess-exited-with-error
  
  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> [26 lines of outp

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Does Prefix Caching currently support offloading to the CPU?

**Link**: https://github.com/vllm-project/vllm/issues/6676
**State**: closed
**Created**: 2024-07-23T06:31:42+00:00
**Closed**: 2024-11-24T02:07:31+00:00
**Comments**: 5
**Labels**: documentation, stale

### Description

### Usage

Does Prefix Caching currently support offloading to the CPU?

 If not, is there a plan to support it? Thanks～


---

