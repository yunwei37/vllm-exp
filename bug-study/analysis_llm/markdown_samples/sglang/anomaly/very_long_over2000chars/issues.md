# very_long_over2000chars - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 4
- Closed Issues: 26

### Label Distribution

- inactive: 6 issues
- help wanted: 3 issues
- deepseek: 1 issues
- high priority: 1 issues

---

## Issue #N/A: [Bug] Server crashes with CUDA errors during EAGLE Speculative Decoding under high concurrency

**Link**: https://github.com/sgl-project/sglang/issues/4188
**State**: closed
**Created**: 2025-03-07T19:11:18+00:00
**Closed**: 2025-03-09T12:32:09+00:00
**Comments**: 4

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When running SGLang with EAGLE speculative decoding under high concurrency load, the server crashes with CUDA errors. The primary error is "CUDA error: an illegal instruction was encountered", occurring in two different scenarios:

When attempting to move GPU tensors to CPU in the KV cache memory management system.
During tensor operations

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] constant errors + hangs using sglang + deepseek v3 + AMD (httpcore.RemoteProtocolError: peer closed connection without sending complete message body (incomplete chunked read))

**Link**: https://github.com/sgl-project/sglang/issues/3198
**State**: closed
**Created**: 2025-01-28T22:05:13+00:00
**Closed**: 2025-04-04T00:17:48+00:00
**Comments**: 5
**Labels**: help wanted, inactive, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Much of the time it is fine, but there is a abrupt termination of the streaming with:
```
httpcore.RemoteProtocolError: peer closed connection without sending complete message body (incomplete chunked read)
```

using the OpenAI API endpoint.  E.g. I see about 250 of those failures over course of 12 hours (even though many more fail becaus

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Unrecognized keys in `rope_scaling` for 'rope_type'='yarn': {'original_max_position_embeddings'}

**Link**: https://github.com/sgl-project/sglang/issues/2943
**State**: closed
**Created**: 2025-01-17T12:12:19+00:00
**Closed**: 2025-04-22T16:45:03+00:00
**Comments**: 12
**Labels**: help wanted

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I am using Qwen2.5-72B which suppots positional extrapolation by Yarn through adding config(copied from https://huggingface.co/Qwen/Qwen2.5-72B-Instruct):
```json
{
  "rope_scaling": {
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "type": "yarn"
  }
}

```
However, this seems not supported by sglang, when I specify 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Assertion failed  with Deepseek-r1 + Eagle + DeepEp

**Link**: https://github.com/sgl-project/sglang/issues/6760
**State**: open
**Created**: 2025-05-30T04:39:19+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

` File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
  File "/sglang-workspace/sglang/python/sgla

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] OpenAI-compatible batch inference fails for singleton batches

**Link**: https://github.com/sgl-project/sglang/issues/6200
**State**: closed
**Created**: 2025-05-11T16:14:02+00:00
**Closed**: 2025-06-05T10:21:48+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

If the batch only contains a single request, then a `TypeError` is raised. One possible workaround is to duplicate the single request, run the job, and then remove the duplicate response.

### Reproduction

```python
import json
import time
from openai import OpenAI

port=30000
print_highlight=print
client = OpenAI(base_url=f"http://127.0.

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Potentially create too much process when using verl-sglang init_model

**Link**: https://github.com/sgl-project/sglang/issues/5483
**State**: closed
**Created**: 2025-04-17T03:50:56+00:00
**Closed**: 2025-06-17T00:19:43+00:00
**Comments**: 3
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

![Image](https://github.com/user-attachments/assets/c702604f-8546-4909-8bf7-d7a436b15351)

It seems creating large amount of process when actor_rollout_init_model, I guess it will be better to have environ or other way config max process spawned. And report：

> thread '<unnamed>' panicked at /root/.cargo/registry/src/index.crates.io-1949cf

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Error using NIXL as transfer backend in PD disaggregation

**Link**: https://github.com/sgl-project/sglang/issues/6694
**State**: open
**Created**: 2025-05-28T07:22:21+00:00
**Comments**: 5

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug
```
I have deployed DeepSeek-R1-Distill-Llama-8B model using PD disaggregation, when I use mooncake as transfer backend for kv cache there is no problem, but when I use NIXL as transfer backend the service can start normally but when request arrives the prefill node reports an exception error with for:
[2025-05-27 23:56:41] 10.94.16.2 [27/M

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] incorrect inference result when using tensor parallel at mi250

**Link**: https://github.com/sgl-project/sglang/issues/7641
**State**: open
**Created**: 2025-06-29T23:04:10+00:00
**Comments**: 2

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

after applying [fix aiter failure at gfx90a](https://github.com/sgl-project/sglang/pull/7187) to docker "lmsysorg/sglang:v0.4.7-rocm630", single GPU inference of sglang works. However, when using --tp-size option the inference result is incorrect.

Tested using llama3 8b, 70b, llama2 7b at mi250 single node(8 GPU).

This does not reproduce

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Unexpected Inference Speed Gain at Concurrency 16 vs 1 on Llama-3.3-70B (FP8, B200, SGLang Blackwell))

**Link**: https://github.com/sgl-project/sglang/issues/7908
**State**: open
**Created**: 2025-07-09T21:39:56+00:00
**Comments**: 3

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

### Summary

Observed a performance anomaly where LLaMA-3.3-70B-Instruct (FP8) running on SGLang with 2xB200 produces significantly higher token throughput at concurrency 16 than at concurrency 1 — despite higher TTFT.

### Environment

- SGLang: Blackwell release (latest)
- Model: LLaMA-3.3-70B-Instruct (FP8 quantized)
- Hardware: 2x B200

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Deepseek v3 doesn't work on mi300x 

**Link**: https://github.com/sgl-project/sglang/issues/2595
**State**: closed
**Created**: 2024-12-26T13:03:50+00:00
**Closed**: 2025-01-09T04:09:06+00:00
**Comments**: 5

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

After getting last source code of sglang I'm not able to run it.

### Reproduction

python3 -m sglang.launch_server --model DeepSeek-V3 --tp 8 --trust-remote-code

WARNING 12-26 13:00:41 rocm.py:17] `fork` method is not supported by ROCm. VLLM_WORKER_MULTIPROC_METHOD is overridden to `spawn` instead.
Traceback (most recent call last):


[... truncated for brevity ...]

---

## Issue #N/A: [Bug] When using sglang as the inference framework, if a word starting with "\n" appears in the stop parameter, the sglang will  Missing '\n' during inference

**Link**: https://github.com/sgl-project/sglang/issues/956
**State**: closed
**Created**: 2024-08-07T02:50:15+00:00
**Closed**: 2024-08-07T11:03:21+00:00
**Comments**: 1

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

When using sglang as the inference framework, if a word starting with "\n" appears in the stop parameter, the sglang will not wrap during inference。
EG:
prompt = 请换行输出1-10个数字
stop = ['<|endoftext|>', '<|im_end|>', '<|im_start|>']
1
2
3
4
5
6
7
8
9
10

prompt = 请换行输出1-10个数字
stop = ['\n<|endoftext|>', '<|im_end|>', '<|im_start|>']
12345678910

"\n" can be followed by any character, and there will be no line break.

### Reproduction

OS: Linux x64  
GPU: A100
python：3.10
sglang：0.2.7
LLM model: Qwen2-72B-lora-awq-4bit
cmd: 
python -m fastcha

[... truncated for brevity ...]

---

## Issue #N/A: Low Inference Speed with Bitsandbytes and AWQ Quantized Models

**Link**: https://github.com/sgl-project/sglang/issues/4263
**State**: closed
**Created**: 2025-03-10T12:11:20+00:00
**Closed**: 2025-03-11T09:15:35+00:00
**Comments**: 6

### Description

**Description**
I'm experiencing low inference speed when running bitsandbytes quantized and AWQ quantized models with sglang. The throughput remains around 31 tokens per second, which seems suboptimal. 

Here are the details of my setup:

**Model Serving Code:**
```
import requests
import os

from sglang import assistant_begin, assistant_end
from sglang import assistant, function, gen, system, user
from sglang import image
from sglang import RuntimeEndpoint, set_default_backend
from sglang.srt.utils import load_image
from sglang.test.test_utils import is_in_ci
from sglang.utils import print_highlight, terminate_process, wait_for_server

if is_in_ci():
    from sglang.docs.frontend.patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

server_process, port = launch_server_cmd(
    "python -m sglang.launch_server --model-path /LLM/model/Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF-AWQ-INT4/ --tp 2 --tokenizer-mode auto --mem-fraction-static 0.9 --disable-cuda-

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] how to solve illegal memory access in moe_align_block_size kernel optimization

**Link**: https://github.com/sgl-project/sglang/issues/3339
**State**: closed
**Created**: 2025-02-06T09:16:45+00:00
**Closed**: 2025-02-06T14:57:39+00:00
**Comments**: 9

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

As mentioned in [lines of code](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu#L80-L90), when attempting to optimize the most expensive write operation of `sorted_token_ids` in the `moe_align_block_size` of DeepSeek V3, using multiple Thread Blocks instead of a single Block triggers an `i

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Mixed chunk causes crash

**Link**: https://github.com/sgl-project/sglang/issues/6921
**State**: closed
**Created**: 2025-06-06T11:04:59+00:00
**Closed**: 2025-06-06T12:36:42+00:00
**Comments**: 2

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When I add `enable-mixed-chunk" it will eventually cause crash in the.

```
sglang  | [2025-06-06 10:23:32 TP0] Prefill batch. #new-seq: 2, #new-token: 8191, #cached-token: 0, token usage: 0.01, #running-req: 1, #queue-req: 7
sglang  | [2025-06-06 10:23:36 TP0] Prefill batch. #new-seq: 3, #new-token: 8190, #cached-token: 0, token usage: 0.

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Error occurs when loading in bitsandbytes format.

**Link**: https://github.com/sgl-project/sglang/issues/2769
**State**: closed
**Created**: 2025-01-07T07:27:02+00:00
**Closed**: 2025-01-14T07:05:49+00:00
**Comments**: 6

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

```bash
File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/linear.py", line 172, in __init__
self.quant_method = quant_config.get_quant_method(self,
File "/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/quantization/bitsandbytes.py", line 114, in get_quant_method
return BitsAndBytesLinearMethod(

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] sglang 0.4.5-post3 deepseekv3-0324  function call  if the message contains system information, the function behaves abnormally.

**Link**: https://github.com/sgl-project/sglang/issues/5814
**State**: closed
**Created**: 2025-04-28T05:57:03+00:00
**Closed**: 2025-05-02T10:45:15+00:00
**Comments**: 8

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

After deploying deepseekv3-0324 using sglang 0.4.5-post3-cu124, when testing the function call, it was found that if the message contains system information, the function behaves abnormally.


### Reproduction

After deploying deepseekv3-0324 using sglang 0.4.5-post3-cu124, when testing the function call, it was found that if the message c

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] SGLang hangs after hitting 0.00 token usage on Engine.generate

**Link**: https://github.com/sgl-project/sglang/issues/1612
**State**: closed
**Created**: 2024-10-08T23:26:49+00:00
**Closed**: 2024-10-09T21:32:54+00:00
**Comments**: 2

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hello! I am trying to add SGLang backend for a fast inference backend to a research project that I am working on, named [rank_llm](https://github.com/castorini/rank_llm). It requires multiple `generate` function calls to run inference for ranking for informational retrieval task. 

SGLang works perfectly fine for first batch, or first `g

[... truncated for brevity ...]

---

## Issue #N/A: missing 1 required positional argument: 'page_size' when using --enable-flashinfer

**Link**: https://github.com/sgl-project/sglang/issues/565
**State**: closed
**Created**: 2024-06-25T19:19:52+00:00
**Closed**: 2024-06-26T15:13:59+00:00
**Comments**: 4

### Description

I am able to get it running properly when not using flashinfer and am currently running 4x NVIDIA A10G. Please let me know what other information might be helpful from my end. 
`python -m sglang.launch_server --model-path /mnt/ebs_volume/models/llama/llama-3-8b-instruct --tp-size=4 --mem-fraction-static=0.75 --port 30000 --enable-flashinfer`



sglang==0.1.17
triton==2.3.0
transformers==4.41.2
torch==2.3.0
vllm==0.4.3
vllm-flash-attn==2.5.8.post2
flashinfer==0.0.5 ( I also tested 0.0.6)
nvcc==12.1, V12.1.105

```
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
[gpu_id=0] Set cuda device.
[gpu_id=1] Set cuda device.
[gpu_id=2] Set cuda device.
[gpu_id=3] Set cuda device.
[gpu_id=0] Init nccl begin.
[gpu_id=1] Init nccl begin.
[gpu_id=2] Init nccl begin.
[gpu_id=3] Init nccl begi

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] When running the DeepSeekV3 model with the bf16 data type, the torch.compile operation did not take effect.

**Link**: https://github.com/sgl-project/sglang/issues/3868
**State**: closed
**Created**: 2025-02-26T03:30:02+00:00
**Closed**: 2025-03-19T06:54:41+00:00
**Comments**: 2

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

The `torch.compile` not take effect when I try to run deepseekv3 with bf16 dtype, the error:
```
[rank26]:W0225 22:39:13.908000 374581 site-packages/torch/_dynamo/convert_frame.py:906] [5/64] torch._dynamo hit config.accumulated_cache_size_limit (64)
[rank26]:W0225 22:39:13.908000 374581 site-packages/torch/_dynamo/convert_frame.py:906] [5

[... truncated for brevity ...]

---

## Issue #N/A: Contradictory suggestions: Not enough memory. Please try to increase --mem-fraction-static

**Link**: https://github.com/sgl-project/sglang/issues/322
**State**: closed
**Created**: 2024-03-22T12:23:46+00:00
**Closed**: 2024-07-25T06:33:37+00:00
**Comments**: 5
**Labels**: inactive

### Description

**Q: Should I increase or decrease `--mem-fraction-static`?** (and what is the minimum and maximum value allowed?)

Looking in the source code (`python/sglang/srt/managers/router/model_runner.py`) I would believe that increasing the value would alleviate the memory requirements but I might be interpreting it wrong. Just wanted to inform that there is a mismatch between the advice given in documentation and the advice given in the actual code.

**Description of the problem:**

I am trying to launch Mistral-7B-Instruct-v0.2 (using sglang==0.1.13):

`python -m sglang.launch_server --model-path /llm_path/hf_model_mistral_7B_Instruct_v0_2 --port 30000`

but I have memory issues. At the end it is suggested to increase `--mem-fraction-static`. 

However, in the documentation (https://github.com/sgl-project/sglang) the opposite advice is given:

> If you see out-of-memory errors during serving, please try to reduce the memory usage of the KV cache pool by setting a smaller value 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support General Reward Model

**Link**: https://github.com/sgl-project/sglang/issues/2427
**State**: closed
**Created**: 2024-12-10T04:15:08+00:00
**Closed**: 2025-05-26T05:51:18+00:00
**Comments**: 8
**Labels**: help wanted, inactive

### Description

### Motivation

As mentioned in our devlopmap, https://github.com/sgl-project/sglang/issues/1487:

Support generalized reward API (adding linear layers to any Causal LM to get the reward) as required by the OpenRLHF team.

https://github.com/OpenRLHF/OpenRLHF

**Add linear layers to any Causal LM to get rewards.**

We formalize this requirement in this issue and invite @M0gician to contribute with us.

### Features Request

#### 1. Add linear layers to any Causal LM to get rewards.

- [ ] Add linear layer at the end and assign a specific token (like final `eos` in the prompt) and manuplate the logits of it as rewards.
- [ ] Add linear layer after a spcific value head name at any layer, manuplate it's logits as rewards.

#### 2. Add `--task` parameter.

- [ ] Get rewards/embedding from any Causal LM, adding a parameter like `--task embedding`.

#### 3. Better Accuracy.

**Many users may have noticed that the reward results of SGLang's current API show a discrepa

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Enable --enable-flashinfer-mla, the result has high rate to output duplicated words/sentence.

**Link**: https://github.com/sgl-project/sglang/issues/4246
**State**: closed
**Created**: 2025-03-10T02:19:43+00:00
**Closed**: 2025-03-26T00:53:35+00:00
**Comments**: 8

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

I use sglang docker image to start Deepseek-V3(671B) model. if the input token prompt is huge, like 10 or 20k or more, then it will output duplicated content, and cannot stop in short period of time.

 





### Reproduction

So I removed all of usless starting parameter(include --enable-flashinfer-mla), and kept some essential's. The issu

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] sglang.launch_server error

**Link**: https://github.com/sgl-project/sglang/issues/1275
**State**: closed
**Created**: 2024-08-31T06:26:00+00:00
**Closed**: 2024-09-22T12:35:22+00:00
**Comments**: 2

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

(sglang) aluo@titan:~/sglang$ python -m sglang.launch_server --model-path /scratch3/data/Meta-Llama-3.1-8B-Instruct/ --enable-torch-compile --disable-radix-cache
server_args=ServerArgs(model_path='/scratch3/data/Meta-Llama-3.1-8B-Instruct/', tokenizer_path='/scratch3/data/Meta-Llama-3.1-8B-Instruct/', tokenizer_mode='auto', load_format='a

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] The sgl-kernel 0.0.3.post7 can't pass the CIs.

**Link**: https://github.com/sgl-project/sglang/issues/4214
**State**: closed
**Created**: 2025-03-08T14:08:53+00:00
**Closed**: 2025-03-09T08:42:30+00:00
**Comments**: 4

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Whe I used 0.0.3.post7 sgl-kernel, `test_mla.py` and `test_mla_tp.py` failed. I think it is probably related with `sgl_per_token_group_quant_fp8` of sgl-kernel after searching. Specifically, this commit https://github.com/sgl-project/sglang/commit/55a7ec388f87780d05a80c3c678ebea22f95523b caused the bug. When I removed the usage of `sgl_per

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Qwen3 MoE FP8: type fp8e4nv not supported in this architecture.

**Link**: https://github.com/sgl-project/sglang/issues/5871
**State**: closed
**Created**: 2025-04-29T05:41:22+00:00
**Closed**: 2025-04-29T08:07:56+00:00
**Comments**: 2

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Error occurred while trying to load Qwen3-30B-A3B-FP8 with `sglang-v0.4.6.post1`. Error message as below:
```
type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')"
```

### Reproduction

`python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B-FP8 --reasoning-parser qwen3 --tp 2 --tool-

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] finish_reason is always null, missing the record of stop.

**Link**: https://github.com/sgl-project/sglang/issues/4550
**State**: closed
**Created**: 2025-03-18T09:35:20+00:00
**Closed**: 2025-05-18T00:20:56+00:00
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

I use sglang:v0.4.4.post1-cu124 image



the output when curl is :

```
data: {"id":"93b6717481cf45c284841fac5c3af2a3","object":"chat.completion.chunk","created":1742289669,"model":"DeepSeek-R1","choices":[{"index":0,"delta":{"role":null,"content":"提供","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":null,"matche

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] sgl-router enters infinite panic loop when all workers die under active load

**Link**: https://github.com/sgl-project/sglang/issues/7028
**State**: closed
**Created**: 2025-06-10T05:34:36+00:00
**Closed**: 2025-06-26T05:58:47+00:00
**Comments**: 2

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Background: DP Attention is not stable yet. During high concurrency, illegal memory access is more likely to occur, leading to crashes. Therefore, when one worker fails, there is a certain probability it will trigger a cascading failure that brings down all workers.

To deal with worker failure, I have a sidecar container infinite looping 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] IPython running error for Engine due to `outlines` nest_asyncio

**Link**: https://github.com/sgl-project/sglang/issues/4478
**State**: closed
**Created**: 2025-03-16T15:37:03+00:00
**Closed**: 2025-05-16T00:19:25+00:00
**Comments**: 1
**Labels**: high priority, inactive

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

If you start an engine in `ipython`:

```python
(verl-sglang) (base) chayenne@lmsys:~/Awesome-ML-SYS-Tutorial/rlhf/rl-walk-through$ ipy

Python 3.10.12 (main, Jan 17 2025, 14:35:34) [GCC 11.4.0]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.33.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: 

In [

[... truncated for brevity ...]

---

## Issue #N/A: [BUG] srt throws KeyError when sgl.gen(...) regex parameter contains Chinese characters

**Link**: https://github.com/sgl-project/sglang/issues/377
**State**: closed
**Created**: 2024-04-22T14:08:55+00:00
**Closed**: 2024-06-12T07:45:20+00:00
**Comments**: 4

### Description

It seems that `sgl.gen(regex=)` doesn't take Chinese characters.

Error Details
```
Exception in ModelRpcClient:
Traceback (most recent call last):
  File ".../sglang/python/sglang/srt/managers/router/model_rpc.py", line 175, in exposed_step
    self.handle_generate_request(recv_req)
  File ".../sglang/python/sglang/srt/managers/router/model_rpc.py", line 271, in handle_generate_request
    req.jump_forward_map = self.jump_forward_cache.query(
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".../sglang/python/sglang/srt/constrained/base_cache.py", line 34, in query
    val = _init_with_timer(key)
          ^^^^^^^^^^^^^^^^^^^^^
  File ".../sglang/python/sglang/srt/constrained/base_cache.py", line 18, in _init_with_timer
    val = self.init_value(key)
          ^^^^^^^^^^^^^^^^^^^^
  File ".../sglang/python/sglang/srt/constrained/jump_forward.py", line 64, in init_value
    return JumpForwardMap(regex)
           ^^^^^^^^^^^^^^^^^^^^^
  File ".../sgl

[... truncated for brevity ...]

---

## Issue #N/A: Dynamo doesn't handle branching on AsyncCollectiveTensor well

**Link**: https://github.com/sgl-project/sglang/issues/2353
**State**: closed
**Created**: 2024-12-04T21:46:43+00:00
**Closed**: 2024-12-04T21:53:27+00:00
**Comments**: 1

### Description

See https://github.com/sgl-project/sglang/pull/2352

torch-only repro:
```
diff --git a/test/distributed/_tensor/test_dtensor_compile.py b/test/distributed/_tensor/test_dtensor_compile.py
index 91fbc396f8e..09a2bf8f183 100644
--- a/test/distributed/_tensor/test_dtensor_compile.py
+++ b/test/distributed/_tensor/test_dtensor_compile.py
@@ -544,12 +544,18 @@ class TestDTensorCompile(torch._dynamo.test_case.TestCase):

     def test_dynamo_dtensor_from_local_redistribute(self):
         mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
+        from torch.distributed._functional_collectives import AsyncCollectiveTensor

         # pass in tensor as inputs/outputs, create DTensor and run redistribute
         # (allgather collective) inside the fn
         def fn(x):
             dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
-            return dt.redistribute(mesh, [Replicate()]).to_local() + 2
+            out = dt.redistribute(mesh, [Re

[... truncated for brevity ...]

---

