# low_impact_1to2 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 11
- Closed Issues: 19

### Label Distribution

- inactive: 7 issues
- good first issue: 5 issues
- help wanted: 5 issues
- high priority: 4 issues
- enhancement: 2 issues
- bug: 2 issues
- new-model: 1 issues
- feature: 1 issues
- MLLM: 1 issues

---

## Issue #N/A: sglang doesn't work with vllm versions above 0.3.3

**Link**: https://github.com/sgl-project/sglang/issues/350
**State**: closed
**Created**: 2024-04-06T05:53:23+00:00
**Closed**: 2024-04-06T06:36:41+00:00
**Comments**: 3

### Description

vllm.model_executor.input_metadata is gone in higher versions of vllm. Below is me trying to run with vllm-0.4.0.post1 installed.

```
(build) owu@gpu:/mnt/resource_nvme$ python -m  sglang.launch_server --model-path liuhaotian/llava-v1.5-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --port 30000
/mnt/resource_nvme/miniconda3/envs/build/lib/python3.10/site-packages/transformers/models/llava/configuration_llava.py:104: FutureWarning: The `vocab_size` argument is deprecated and will be removed in v4.42, since it can be inferred from the `text_config`. Passing this argument has no effect
  warnings.warn(
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
/mnt/resource_nvme/miniconda3/envs/build/lib/python3.10/site-packages/transformers/models/llava/configuration_llava.py:144: FutureWarning: The `vocab_size` attribute is deprecated and will be removed in v4.42, Please use `text_config.vocab_size` instead.
  warnings.w

[... truncated for brevity ...]

---

## Issue #N/A: Is it possible to run on ROCm without AITER?

**Link**: https://github.com/sgl-project/sglang/issues/6100
**State**: open
**Created**: 2025-05-07T22:16:22+00:00
**Comments**: 1

### Description

As AITER is only running on Instinct hardware, is it possible to run SGLang without AITER on ROCm so Radeon/NAVI based hardware is usable?

---

## Issue #N/A: [Feature] Support for a Transformer-based Minimax Model

**Link**: https://github.com/sgl-project/sglang/issues/7770
**State**: open
**Created**: 2025-07-04T07:18:07+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Hello Team,

I'm writing to suggest a new feature: the implementation of a Minimax model using a Transformer architecture.

Would you consider supporting or developing a Transformer version of the Minimax model? I believe it would be a valuable addition for the community.

Thank you for your consideration.

### Related resources

_No response_

---

## Issue #N/A: [Bug] usage is null when set stream=True

**Link**: https://github.com/sgl-project/sglang/issues/954
**State**: closed
**Created**: 2024-08-07T02:22:41+00:00
**Closed**: 2024-08-08T09:41:58+00:00
**Comments**: 2

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

usage is null when set stream=True via openai sdk.
when stream=True usage is OK

### Reproduction

#### server
```
python -m sglang.launch_server \
    --model-path $MODEL_NAME_OR_PATH \
    --served-model-name Qwen/Qwen2-72B-Instruct-AWQ \
    --host 0.0.0.0 \
    --port 30000 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --tp-size 2
```

#### client:
```
import time
import openai

start_time = time.time()

client = openai.Client(base_url="http://localhost:30000/v1", api_key="EMPTY")

completion = client.chat.completions.c

[... truncated for brevity ...]

---

## Issue #N/A: [Roadmap] High performance backend for Ascend NPU

**Link**: https://github.com/sgl-project/sglang/issues/7665
**State**: open
**Created**: 2025-06-30T20:35:38+00:00
**Comments**: 0

### Description

High performance Ascend NPU support by implementation fast Attention backend and NPU Graph support. W8A8 quantization. MLA implementation and DeepSeek support.

**Progress**:

- [ ] Effective Ascend Attention backend implemenattion
- [ ] Implementation W8A8 quantization
- [ ] MLA implementation and DeepSeek support
- [ ] MindIE Turbo fast layers integration
- [ ] NPU Graph implementation
- [ ] Support Expert Parallelism

---

## Issue #N/A: [Feature] Allow arbitrary logit processors

**Link**: https://github.com/sgl-project/sglang/issues/1036
**State**: closed
**Created**: 2024-08-11T19:34:38+00:00
**Closed**: 2024-10-21T01:13:28+00:00
**Comments**: 2
**Labels**: inactive

### Description

### Motivation

There's some great projects out there that modify logits, mostly for guided decoding or novel sampling techniques. Supporting every single one of them will cause too much bloat and distraction, but if SGLang were to allow arbitrary logit processors then the community can plug and play their own processors.

For example, I would have interest in using [https://github.com/noamgat/lm-format-enforcer](lm format enforcer) because it allows for optional JSON fields and recursive classes (unlike outlines). The API of lm format enforcer is also clean and simple and it is simple to make custom parsers for other formats than JSON (e.g. SQL).

One way I would imagine the API to work is:

```python
def my_logits_processor(inputs: list[int], logits: torch.Tensor) -> torch.Tensor:
   ...


@sgl.function
def character_gen(s, name):
    s += name + " is a character in Harry Potter. Please fill in the following information about this character.\n"
    s += sgl.gen("outpu

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Some FP8 models fail to load

**Link**: https://github.com/sgl-project/sglang/issues/7482
**State**: open
**Created**: 2025-06-23T20:04:33+00:00
**Comments**: 16

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Sglang crashes when loading some FP8 models. I only got Qwen3 FP8 model work, but not others from RedHatAI:
WORKS:
https://huggingface.co/Qwen/Qwen3-32B-FP8

DO NOT WORK
https://huggingface.co/RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic
https://huggingface.co/RedHatAI/Qwen3-32B-FP8-dynamic
https://huggingface.co/RedHatAI/Meta-

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] 4-bit quantized prefix cache

**Link**: https://github.com/sgl-project/sglang/issues/1374
**State**: closed
**Created**: 2024-09-10T16:38:26+00:00
**Closed**: 2024-12-06T01:17:32+00:00
**Comments**: 5
**Labels**: enhancement, inactive

### Description

### Motivation

LMDeploy's 4-bit quantized prefix cache (along with 4-bit AWQ for weights) allows running ~70B models on 48GB of RAM with good performance for many-user scenarios. The prefix cache can hold more than 40,000 context tokens.

This is very handy, since it's often easier to get a GPU (or dual GPUs) with 48GB RAM than it is to get 80GB+ GPUs.

Note that I've benchmarked the output quality/accuracy of 4-bit prefix cache vs no quantization, and there was no significant accuracy drop with my internal benchmarks. For my use case, at least, it's a free perf boost.

Today I wanted to try comparing SGLang performance to LMDeploy, but (for a 70B model on 48GB GPU) SGLang OOMs for even a small number of concurrent requests.

I'm testing with LLama 2 AWQ model with ~2k token context and ~100 token outputs:

### LMDeploy (handles 20 concurrent requests fine):
Using latest (`openmmlab/lmdeploy:v0.6.0a0-cu12`) docker image on 48GB NVIDIA A40 GPU:
```
lmdeploy serve api_ser

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Error when running Qwen2 EAGLE speculative decoding refering to the official example

**Link**: https://github.com/sgl-project/sglang/issues/3315
**State**: closed
**Created**: 2025-02-05T11:21:57+00:00
**Closed**: 2025-02-09T15:39:46+00:00
**Comments**: 5
**Labels**: bug, high priority

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug


I am trying to run the Qwen2 EAGLE speculative decoding example as provided in the official documentation:  
[EAGLE Decoding Documentation](https://docs.sglang.ai/backend/speculative_decoding.html#EAGLE-Decoding)

I used the following command to launch the server:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2-7B-Instruct

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] flashinfer separate installation: Probably needs either code or documentation fix?

**Link**: https://github.com/sgl-project/sglang/issues/4361
**State**: closed
**Created**: 2025-03-13T04:16:27+00:00
**Closed**: 2025-05-13T00:19:04+00:00
**Comments**: 1
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

I see that https://github.com/sgl-project/sglang/pull/3033 recently added flashinfer to 3rdparty. Now after reading [sgl-kernel/setup.py](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/setup.py), it seems to me that it unconditionally builds flashinfer from 3rdparty.

Yet the [installation documentation](https://github.com/sgl-

[... truncated for brevity ...]

---

## Issue #N/A: JSON decoding result don't match regex

**Link**: https://github.com/sgl-project/sglang/issues/371
**State**: closed
**Created**: 2024-04-17T10:02:31+00:00
**Closed**: 2024-07-25T06:33:12+00:00
**Comments**: 1
**Labels**: inactive

### Description

sglang 0.12.0
model: Qwen1.5-0.5B

I have two regex: debug_regex1 and debug_regex2. Both are simplified from “build_regex_from_object”
Original regex is:
```text
\{[\n ]*"data"[\n ]*:[\n ]*\[[\n ]*((\{[\n ]*"key1"[\n ]*:[\n ]*("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"|null)[\n ]*,[\n ]*"key2"[\n ]*:[\n ]*("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"|null)[\n ]*,[\n ]*"key3"[\n ]*:[\n ]*("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"|null)[\n ]*\})(,[\n ]*(\{[\n ]*"key1"[\n ]*:[\n ]*("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"|null)[\n ]*,[\n ]*"key2"[\n ]*:[\n ]*("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"|null)[\n ]*,[\n ]*"key3"[\n ]*:[\n ]*("(?:[^"\\\x00-\x1f\x7f-\x9f]|\\.)*"|null)[\n ]*\})){0,})?[\n ]*\][\n ]*\}
```
Sometimes json decoding outputs “\n\n\n\n\n\n……”, so I deleted some “[\n ]*”(https://github.com/sgl-project/sglang/issues/258#issuecomment-2041814454). But output don't match regex.
You can reproduce this bug using the following Python code.
```python
import json
import re
import sglang as sgl



[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Error loading Qwen/Qwen3-30B-A3B-GPTQ-Int4 model

**Link**: https://github.com/sgl-project/sglang/issues/7583
**State**: open
**Created**: 2025-06-27T03:31:38+00:00
**Comments**: 6

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I'm using the docker image `lmsysorg/sglang:v0.4.8-cu126` 

Ran into this issue:

```
Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/sgl

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] ValueError: Model architectures ['Glm4ForCausalLM'] are not supported for now.

**Link**: https://github.com/sgl-project/sglang/issues/5441
**State**: open
**Created**: 2025-04-16T01:49:12+00:00
**Comments**: 3
**Labels**: new-model

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

ValueError: Model architectures ['Glm4ForCausalLM'] are not supported for now. 

```
ValueError: Model architectures ['Glm4ForCausalLM'] are not supported for now. Supported architectures: dict_keys(['BaichuanForCausalLM', 'ChatGLMModel', 'CLIPModel', 'CohereForCausalLM', 'Cohere2ForCausalLM', 'DbrxForCausalLM', 'DeepseekForCausalLM', 'Mul

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Serving VLM VILA

**Link**: https://github.com/sgl-project/sglang/issues/2345
**State**: open
**Created**: 2024-12-04T07:44:26+00:00
**Comments**: 1
**Labels**: good first issue, help wanted

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Hello,

I want to deploy the VILA model for serving VILA1.5-3B-AWQ (https://github.com/NVlabs/VILA). Could you please guide me on how to get started? Are there any specific instructions or tools I should follow for setting up the serving environment?

### Related resources

_No response_

---

## Issue #N/A: Have some advise to learn Openai's triton?

**Link**: https://github.com/sgl-project/sglang/issues/208
**State**: closed
**Created**: 2024-02-20T12:03:29+00:00
**Closed**: 2024-07-28T03:16:19+00:00
**Comments**: 3

### Description

I note this project use triton write kernel. It's cool, so can you share how learn triton ?

---

## Issue #N/A: [Feature] Prefill assistant response

**Link**: https://github.com/sgl-project/sglang/issues/3971
**State**: closed
**Created**: 2025-02-28T21:34:21+00:00
**Closed**: 2025-04-21T15:22:27+00:00
**Comments**: 4
**Labels**: good first issue, help wanted, feature

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

OAI API doesn't natively support prefilling an assistants response. vLLM and Aphrodite has the additional support for `continue_final_message` which would be need to have for SGLang to give developers even much more control.

Should be relatively easy for someone to implement. It's simply not allowing chat template EOS to take over in a turn where assistant response is last and this flag is enabled and a generation is requested. This was originally implemented with exact same parameter name in transformers, which became a feature in vLLM and Aphrodite.

### Related resources

https://huggingface.co/docs/transformers/main/en/chat_templating
https://github.com/aphrodite-engine/aphrodite-engine/blob/e64075b8937786311f

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] don't quit server if the request doesn't process success

**Link**: https://github.com/sgl-project/sglang/issues/3623
**State**: closed
**Created**: 2025-02-17T05:58:27+00:00
**Closed**: 2025-02-17T08:45:33+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

can sglang don't quit if a request doesn't process sucess.

I'm trying to process some requests one by one in a loop, but when  I hit control+z，the server quit with log.

2025-02-17 11:56:43] Initialization failed. warmup error: Traceback (most recent call last):
  File "xxx/python3.11/site-packages/sglang/srt/entrypoints/http_server.py", line 548, in _wait_and_warmup
    assert res.status_code == 200, f"{res=}, {res.text=}"
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: res=<Response [403]>, res.text='<html>\n<head><title>403 Forbidden</title></head>\n<body>\n<div style="text-align: center;"><h1>403 Forbidden</h1></div>\n</body>\n</html>'

Killed

.

can you just report an error but don't quit the server

### R

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Inconsistent rid handling in OpenAI-Compatible Server

**Link**: https://github.com/sgl-project/sglang/issues/7374
**State**: open
**Created**: 2025-06-20T02:51:45+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

**1. Overview**  

Through refactoring OpenAI-Compatible Server with @CatherineSue. We found that with SGLang v0.4.7.post1, requests including a custom `rid` field fail under two specific conditions:

- When parameter `n > 1` in the `/v1/chat/completions` endpoint.

- When the input to `/v1/embeddings` is a list of strings.

Current issue 

[... truncated for brevity ...]

---

## Issue #N/A: [model request] Camelidae/Sparsestral

**Link**: https://github.com/sgl-project/sglang/issues/176
**State**: closed
**Created**: 2024-02-09T22:49:30+00:00
**Closed**: 2024-07-25T06:32:07+00:00
**Comments**: 1
**Labels**: inactive

### Description

These are parameter efficient MoE models that claim to have performance better than Mixtral.

Camildae
https://github.com/wuhy68/Parameter-Efficient-MoE

Sparsestral
https://huggingface.co/serpdotai/sparsetral-16x7B-v2

Sparsestral has a vllm implementation in this form
https://github.com/serp-ai/vllm

---

## Issue #N/A: [Feature] Add support for embedding in server::Engine

**Link**: https://github.com/sgl-project/sglang/issues/1993
**State**: closed
**Created**: 2024-11-11T05:13:12+00:00
**Closed**: 2024-11-11T20:07:38+00:00
**Comments**: 0

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Currently, `server::Runtime` supports embeddings through `Runtime::encode(self, prompt)`. We'd like to add similar support for `server::Engine`.

cc: @ByronHsu

### Related resources

_No response_

---

## Issue #N/A: [Bug] RuntimeError: batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_bf16_dtype_o_bf16_dtype_idx_i32_head_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False::plan() expected at most 15 argument(s) but

**Link**: https://github.com/sgl-project/sglang/issues/5022
**State**: closed
**Created**: 2025-04-03T03:19:53+00:00
**Closed**: 2025-04-03T03:22:47+00:00
**Comments**: 3

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

```bash
  File "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/env/rjh/lib/python3.12/site-packages/sglang/srt/managers/tp_worker_overlap_thread.py", line 112, in forward_thread_func
    self.forward_thread_func_()
  File "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/env/rjh/lib/python3.12/site-packages/to

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] PyTorch profiler trace is not generated

**Link**: https://github.com/sgl-project/sglang/issues/2874
**State**: closed
**Created**: 2025-01-13T20:28:56+00:00
**Closed**: 2025-01-24T08:18:30+00:00
**Comments**: 13
**Labels**: good first issue, help wanted

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

The PyTorch Profiiler does not generate the profiler trace when using the engine for inference

### Reproduction

I am trying to run profiler for SGLang  0.4.1.post5 using this code

```
import sglang as sgl
import asyncio
import logging

def test():
    logging.basicConfig(level=logging.DEBUG)
    llm = sgl.Engine(model_path="/ho

[... truncated for brevity ...]

---

## Issue #N/A: will  support  multi-loras inference？

**Link**: https://github.com/sgl-project/sglang/issues/334
**State**: closed
**Created**: 2024-03-28T03:22:09+00:00
**Closed**: 2024-07-25T06:32:58+00:00
**Comments**: 1
**Labels**: inactive

### Description

No description provided.

---

## Issue #N/A: [Kernel] cuDNN attention backend

**Link**: https://github.com/sgl-project/sglang/issues/2272
**State**: open
**Created**: 2024-11-30T06:36:16+00:00
**Comments**: 3
**Labels**: enhancement, good first issue, help wanted, high priority, inactive

### Description

cuDNN provides very fast attention implementation and it is well maintained by NVIDIA. We would like to add a new attention backend based on cudnn.  

## Steps
1. Learn this cudnn paged attention python api. https://github.com/NVIDIA/cudnn-frontend/blob/v1.8.0/samples/python/52_scaled_dot_product_attention_with_paged_caches.ipynb
2. Add a new attention backend "cudnn" here https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/attention
3. We should be able to use it with `python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --attention-backend cudnn`

---

## Issue #N/A: [Bug] Use torch.inference_mode instead of torch.no_grad

**Link**: https://github.com/sgl-project/sglang/issues/4366
**State**: closed
**Created**: 2025-03-13T06:32:51+00:00
**Closed**: 2025-04-04T05:18:52+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

We found that `torch.no_grad` triggers the `AutogradXXX` backend for certain operators. Should we replace it with `inference_mode` instead, or keep supporting with `torch<1.9`?

### Reproduction

Example: `python/sglang/srt/mem_cache/memory_pool.py:144(def free_group_end(self):)`

- Result with torch.no_grad(): `NotImplementedError: Could 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] fail to deploy embedding model bge-m3 with blackwell image

**Link**: https://github.com/sgl-project/sglang/issues/7590
**State**: open
**Created**: 2025-06-27T07:01:50+00:00
**Comments**: 4

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

after run bge-m3 model  with blackwell image on a 5090, 
i test it by rest api, and then the docker container get errors and stopped.

### Reproduction

docker command:
```
docker run -d --gpus '"device=3"' \
  --shm-size 32g \
  -v /data/docker/vllm/models/BAAI/bge-m3:/mnt/llms/models/BAAI/bge-m3 \
  -p 18000:8000 \
  lmsysorg/sglang:blac

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]NCCL error if enable the cuda graph

**Link**: https://github.com/sgl-project/sglang/issues/3538
**State**: closed
**Created**: 2025-02-13T06:38:16+00:00
**Closed**: 2025-02-19T14:35:47+00:00
**Comments**: 4
**Labels**: bug, high priority

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

<img width="1663" alt="Image" src="https://github.com/user-attachments/assets/e3b396cc-4771-474d-8843-d43d8d5dbf90" />

If I don't disable cuda graph, I will get the error shown in the picture when the cuda graph is being inited. If i use the official docker image, i will not get the error. The only difference of the environment with the d

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] support and turn on chunked prefill by default for VLM

**Link**: https://github.com/sgl-project/sglang/issues/5250
**State**: closed
**Created**: 2025-04-10T18:45:57+00:00
**Closed**: 2025-05-26T16:56:01+00:00
**Comments**: 2
**Labels**: good first issue, help wanted, high priority, MLLM

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct

### Related resources

_No response_

---

## Issue #N/A: ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)

**Link**: https://github.com/sgl-project/sglang/issues/5182
**State**: open
**Created**: 2025-04-09T06:34:44+00:00
**Comments**: 2

### Description

Hi, big guys!
I have two questions that I would like to have answered by you all
First, I have a Deepseek-R1-distill-Qwen-7b model for FP8 quantized via llmcompress, and when I start it via sglang and pass the parameter “--device”, “cpu”, the service startup fails with the following error:

![Image](https://github.com/user-attachments/assets/dd8e6fcc-4b83-4d08-baa8-ef368b666a65)

But I can start the service normally when I use cuda, why is that? What should I do to fix it?
Below is the code for my service startup (cuda or cpu) respectively！
![Image](https://github.com/user-attachments/assets/f81495b7-fe16-48d9-bc8f-dc2468b9152a)

Second, I found that the sglang service starts with k size and v size and takes up a lot of gpu memory, in my service startup command, even setting --context-length has no effect on k size and v size, what can I do to reduce this part of the gpu memory usage?

![Image](https://github.com/user-attachments/assets/2f9496b1-52ad-422b-a5b2-a29b3bf14b27)

---

## Issue #N/A: [Bug] Should the bias in Eagle self.fc ought to follow the config.json

**Link**: https://github.com/sgl-project/sglang/issues/5773
**State**: closed
**Created**: 2025-04-27T06:26:11+00:00
**Closed**: 2025-05-08T02:26:49+00:00
**Comments**: 0

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

The bias in files (https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama_eagle.py#L81 ) self.fc is True, but the bias in office repo (https://github.com/SafeAILab/EAGLE/blob/91ae5f2ffa44a6e2cf50b7a1d19d899c7bbd5817/eagle/model/cnets1.py#L511) is follow the config.json. This may result in missing bias when loading w

[... truncated for brevity ...]

---

