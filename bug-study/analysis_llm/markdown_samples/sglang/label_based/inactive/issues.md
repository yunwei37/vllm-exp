# inactive - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 29

### Label Distribution

- inactive: 30 issues
- bug: 2 issues
- help wanted: 2 issues
- high priority: 2 issues
- lora: 1 issues
- good first issue: 1 issues
- documentation: 1 issues
- deepseek: 1 issues
- feature: 1 issues

---

## Issue #N/A: how to use fp8 for inference on h20?

**Link**: https://github.com/sgl-project/sglang/issues/3568
**State**: closed
**Created**: 2025-02-14T05:59:47+00:00
**Closed**: 2025-04-16T00:18:29+00:00
**Comments**: 8
**Labels**: inactive

### Description

I have a problem, that is, the model I am currently deploying is: neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic.
The graphics card is h20.

I would like to ask how to infer the fp8 capability of this graphics card?
Currently, sglang is used for deployment. The deployment instructions are as follows:

python -m sglang.launch_server --model-path neuralmagic/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic --port 30000 --host 0.0.0.0 --tp 2 

What I want to know is that my command has enabled fp8 for inference operations? If not, can you tell me how to do it? Thanks

---

## Issue #N/A: [Feature] Add a hash for each new release

**Link**: https://github.com/sgl-project/sglang/issues/3923
**State**: closed
**Created**: 2025-02-27T08:39:19+00:00
**Closed**: 2025-04-30T00:18:49+00:00
**Comments**: 3
**Labels**: inactive

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

## Summary
Implement SHA-256 hash verification for all package releases to enhance security for users installing from mirror sites.

## Description
Users who download packages from mirror sites instead of the official PyPI repository need a reliable way to verify package integrity. Adding SHA-256 hashes for each release would provide a standard method to confirm packages haven't been tampered with or corrupted.

## Implementation
- Generate SHA-256 hashes automatically as part of the CI/CD pipeline
- Include hashes in package metadata files
- Make hashes accessible through the official website
- Update documentation to explain the verification process

## Benefits
- Enhanced security for users with limited access to official repos

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Qwen-gme embedding model: cannot get fused embedding from text+image, and image input format may be incorrect

**Link**: https://github.com/sgl-project/sglang/issues/5498
**State**: closed
**Created**: 2025-04-17T13:09:28+00:00
**Closed**: 2025-07-11T00:20:24+00:00
**Comments**: 4
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Thanks for the great work!

While using the /v1/embeddings endpoint with the gme-qwen2-vl model, I encountered two issues:

**1. Incorrect handling of image input**
According to the docs, the image input is passed like this:
payload = {
    "model": "gme-qwen2-vl",
    "input": [
        {"type": "text", "text": text_input},
        {"type

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] HuggingFace and SGLang inference don't match

**Link**: https://github.com/sgl-project/sglang/issues/2671
**State**: closed
**Created**: 2024-12-30T22:54:09+00:00
**Closed**: 2025-05-03T00:18:08+00:00
**Comments**: 9
**Labels**: bug, inactive, lora

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

The accuracy of the model is degraded due to inconsistent outputs from SGLang. While HF and vLLM produce consistent results such as "A" or "B," SGLang occasionally outputs responses like "I can't process that request." or "A." / "B." This inconsistency impacts overall accuracy.

### Reproduction

What command or script did yo

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Auto-truncation still uses full context length instead of (context_length - max_tokens)

**Link**: https://github.com/sgl-project/sglang/issues/5409
**State**: open
**Created**: 2025-04-15T07:33:29+00:00
**Comments**: 1
**Labels**: good first issue, inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I'm experiencing an issue where prompt auto-truncation doesn't properly account for max_tokens when using the HTTP server, even with allow_auto_truncate=True enabled. This persists after the changes in https://github.com/sgl-project/sglang/pull/4919.

### Reproduction

1.  python -m sglang.launch_server --model-path NousResearch/Hermes-3-L

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Clear PAT_TOKEN in CI

**Link**: https://github.com/sgl-project/sglang/issues/2659
**State**: closed
**Created**: 2024-12-30T07:44:56+00:00
**Closed**: 2025-03-01T00:18:50+00:00
**Comments**: 1
**Labels**: documentation, inactive

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

![image](https://github.com/user-attachments/assets/d62f4957-2802-4068-9c16-fbcaee2584f4)

@shuaills Would you like to take this? Pretty easy.

### Related resources

_No response_

---

## Issue #N/A: [Feature] SGLang Support for TileLang

**Link**: https://github.com/sgl-project/sglang/issues/4221
**State**: closed
**Created**: 2025-03-09T05:34:49+00:00
**Closed**: 2025-05-27T00:18:53+00:00
**Comments**: 10
**Labels**: help wanted, high priority, inactive

### Description

We recently came across an interesting project: [TileLang](https://github.com/tile-ai/tilelang). It appears to offer significant advantages over Triton in many cases while maintaining a clean dataflow and simple syntax.

Do we have any plans to support a TileLang backend in SGLang?

For instance, TileLang has demonstrated up to **5x speedup** over Triton’s Flash MLA implementations on H100, with a kernel implementation of just **80 lines of code (see document:** https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_mla). Given these promising results, it would be valuable to explore its potential integration.

Would love to hear thoughts on this!


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

## Issue #N/A: aglang

**Link**: https://github.com/sgl-project/sglang/issues/299
**State**: closed
**Created**: 2024-03-14T09:10:13+00:00
**Closed**: 2024-07-25T06:32:43+00:00
**Comments**: 1
**Labels**: inactive

### Description

I test yi-vl-6B with `srt_example_yi_vl.py`
get error:
```
AttributeError: 'TokenizerManager' object has no attribute 'executor
```

---

## Issue #N/A: [Bug] Got error with awq_marlin quantization args.

**Link**: https://github.com/sgl-project/sglang/issues/1792
**State**: closed
**Created**: 2024-10-25T10:21:16+00:00
**Closed**: 2024-12-26T00:16:32+00:00
**Comments**: 2
**Labels**: inactive

### Description

### Checklist

- [x]  I have searched related issues but cannot get the expected help.

- [x]  The bug has not been fixed in the latest version.

- [x]  Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

- [x]  If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.

 

- [x] Please use English, otherwise it will be closed.

### Describe the bug

I used the AutoAWQ tool to quantize [Deepseek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2) model . The quantization script is as follows, resulting in a quantized network. I expect to obtain a model in awq_marlin quantization format.
```
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


mo

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] No matching distribution found for sgl-kernel==0.0.9.post2; extra == "srt"

**Link**: https://github.com/sgl-project/sglang/issues/5642
**State**: closed
**Created**: 2025-04-22T16:43:03+00:00
**Closed**: 2025-06-23T00:21:20+00:00
**Comments**: 1
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

followed same steps from: https://docs.sglang.ai/start/install.html#method-2-from-source

### Reproduction

same steps and I got:

Obtaining file:///D:/ia/sglang/python
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Preparing editable

[... truncated for brevity ...]

---

## Issue #N/A: Unable to load 72b llava qwen on 8*A100 40GB

**Link**: https://github.com/sgl-project/sglang/issues/507
**State**: closed
**Created**: 2024-06-05T23:39:20+00:00
**Closed**: 2024-09-28T01:10:44+00:00
**Comments**: 6
**Labels**: inactive

### Description

Using the command: `CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server --model-path lmms-lab/llava-next-72b --tokenizer-path lmms-lab/llavanext-qwen-tokenizer --port=8000 --host="0.0.0.0" --tp-size=4`

Results in error:

```
torch.distributed.DistStoreError: Timed out after 601 seconds waiting for clients. 1/4 clients joined.
Initialization failed. detoken_init_state: init ok
```

---

## Issue #N/A: [Bug] Benchmarks with EAGLE-2

**Link**: https://github.com/sgl-project/sglang/issues/2777
**State**: closed
**Created**: 2025-01-07T20:25:44+00:00
**Closed**: 2025-03-23T00:19:15+00:00
**Comments**: 6
**Labels**: inactive

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I have tried to use different benchmarks for sglang with EAGLE-2. However, it seems that it cannot work.


### Reproduction

python3 -m sglang.launch_server --model meta-llama/Llama-2-7b-chat-hf  --speculative-algo EAGLE --speculative-draft lmzheng/sglang-EAGLE-llama2-chat-7B --speculative-num-steps 5 --speculative-eagle-top

[... truncated for brevity ...]

---

## Issue #N/A: How to use 4bit on LLava?

**Link**: https://github.com/sgl-project/sglang/issues/123
**State**: closed
**Created**: 2024-01-31T03:06:50+00:00
**Closed**: 2024-07-25T06:32:04+00:00
**Comments**: 1
**Labels**: inactive

### Description

I am using this code: "python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.5-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --chat-template vicuna_v1.1 --port 30000" , but I am not able to use any instruction for quantity 4-bit. 

can you tell me how to use 4-bit llava on sglang?

---

## Issue #N/A: Can SGlang run on cuda118?

**Link**: https://github.com/sgl-project/sglang/issues/309
**State**: closed
**Created**: 2024-03-19T03:20:19+00:00
**Closed**: 2024-07-25T06:33:10+00:00
**Comments**: 3
**Labels**: inactive

### Description

I can't successfully run SGlang using either "python -m sglang.launch_server --model-path LOCAL_MODEL_PATH --port 30000" or "sgl.Runtime(model_path=LOCAL_MODEL_PATH)", just like the issue mentioned in https://github.com/sgl-project/sglang/issues/199 .

Could this be related to my CUDA 11.8 installation, or is there another possible reason?

Thank you.

---

## Issue #N/A: [Bug] Qwen2-VL-7B IndexError

**Link**: https://github.com/sgl-project/sglang/issues/2181
**State**: closed
**Created**: 2024-11-25T17:02:41+00:00
**Closed**: 2025-01-31T00:16:28+00:00
**Comments**: 4
**Labels**: bug, inactive

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

Occasionally, we will see a random "IndexError" which crashes sglang when serving Qwen2-VL-7B models. The crash is usually such that sglang will livelock, so the process will not exit, but no new requests will be servable. 

I have tried to rerun the requests again in a local interactive environment, but I cannot get an exact repro case 

[... truncated for brevity ...]

---

## Issue #N/A: Is it possible to define the prompts for KV caching up-front?

**Link**: https://github.com/sgl-project/sglang/issues/401
**State**: closed
**Created**: 2024-04-29T08:40:04+00:00
**Closed**: 2024-07-25T06:33:23+00:00
**Comments**: 2
**Labels**: inactive

### Description

For a lot of use cases, there is already a pre-defined system + base prompt that is used.

Can we define the KV cache for these prompts up front manually? For example, if we are extracting information out of a provided context, the provided context prompt changes but the system + base prompt stays the same. Caching the context will make no sense as it is guaranteed to change on the next inference. 

---

## Issue #N/A: [Feature] use SGLang's FusedMoE with quantization

**Link**: https://github.com/sgl-project/sglang/issues/2337
**State**: closed
**Created**: 2024-12-03T14:55:12+00:00
**Closed**: 2025-02-02T00:17:42+00:00
**Comments**: 1
**Labels**: inactive

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

ref https://github.com/sgl-project/sglang/pull/2300#issuecomment-2514795180

### Related resources

_No response_

---

## Issue #N/A: [Bug] Tensor model parallel group is not initialized when deploying Qwen3-30B-A3B-AWQ

**Link**: https://github.com/sgl-project/sglang/issues/6000
**State**: closed
**Created**: 2025-05-04T01:08:45+00:00
**Closed**: 2025-07-16T00:20:45+00:00
**Comments**: 18
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hi, SGLang Team,

I am using it to deploy an AWQ quantized model of Qwen3-30B-A3B: swift/Qwen3-30B-A3B-AWQ from modelscope. but encounter the following issue:

```bash
 File "/home/a/sglang/python/sglang/srt/managers/scheduler.py", line 2215, in run_scheduler_process
    scheduler = Scheduler(server_args, port_args, gpu_id, tp_rank, pp_ran

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] When a node is inaccessible, it will cause the router to crash.

**Link**: https://github.com/sgl-project/sglang/issues/4562
**State**: closed
**Created**: 2025-03-19T00:58:02+00:00
**Closed**: 2025-06-09T00:20:53+00:00
**Comments**: 3
**Labels**: inactive

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

**This results in the router being inaccessible, but the health check of /health still returns the correct result. the error like**

`Current running queue: {"http://localhost:8002": 7530, "http://localhost:8000": 7497, "http://localhost:8001": 7495}
[Router (Rust)] 2025-03-18 14:17:58 - WARN - Generate request to http://localhost:8001 fai

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

## Issue #N/A: [Bug] Llava-v1.6-34B template is not updated.

**Link**: https://github.com/sgl-project/sglang/issues/285
**State**: closed
**Created**: 2024-03-12T02:44:23+00:00
**Closed**: 2024-07-25T06:32:39+00:00
**Comments**: 4
**Labels**: inactive

### Description

Reference to https://github.com/haotian-liu/LLaVA/blob/7440ec9ee37b0374c6b5548818e89878e38f3353/llava/serve/gradio_web_server.py#L176, the chat template used by llava-v1.6-34b is 'chatml_direct' which is not implement in current SGLANG.
The template 'chatml' is implemented, but totally different from 'chatml_direct'.

The bug leads to the different outputs between the gradio demo and sgl.function with sgl runtime.

Besides, the template structure and notation are totally different. I am not sure that I can transfer the chat template from llava to ChatTemplate correctly.

---

## Issue #N/A: [Bug] gemma-3-27b-it-bnb-4bit crash 

**Link**: https://github.com/sgl-project/sglang/issues/4897
**State**: closed
**Created**: 2025-03-29T19:01:15+00:00
**Closed**: 2025-06-13T00:19:52+00:00
**Comments**: 6
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug


1、When loading a 27B 4-bit quantized model, why does it exhaust the 24GB of gpu memory?
2、Why did the program crash? Is it because the gpu memory was exhausted?
<img width="1400" alt="Image" src="https://github.com/user-attachments/assets/48732e57-a966-4f92-951b-3fd637da3f1b" />
[2025-03-29 18:56:25 TP0] Scheduler hit an exception: Traceb

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Service crashed with 4 H100s and QPS=25

**Link**: https://github.com/sgl-project/sglang/issues/3112
**State**: closed
**Created**: 2025-01-24T20:17:28+00:00
**Closed**: 2025-03-29T00:17:30+00:00
**Comments**: 4
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

The serving was OK at the start. GPU usage was OK at 80%-99%. 

One GPU usage suddenly increases to 100%. The other 3 GPUs run 80%-99% then become 0%.

It is then unable to hand in the requests. Although it shows OK, it is not handled.

![Image](https://github.com/user-attachments/assets/c26cb7a1-28df-4f40-82c6-91ff3d425e50)

### Reproduct

[... truncated for brevity ...]

---

## Issue #N/A: I wonder if the offline engine API supports OpenAI input format.

**Link**: https://github.com/sgl-project/sglang/issues/2734
**State**: closed
**Created**: 2025-01-05T09:11:26+00:00
**Closed**: 2025-03-07T00:17:27+00:00
**Comments**: 1
**Labels**: inactive

### Description

https://github.com/sgl-project/sglang/blob/bc6ad367c2beec2587843992176089b32eb5d6b9/examples/runtime/engine/offline_batch_inference.py#L12

As shown below.
prompts= [
        [{"role": "user", "content": "List 3 countries and their capitals."}]
]

---

## Issue #N/A: [Feature] Proposal: Releasing SGLang memory when idle

**Link**: https://github.com/sgl-project/sglang/issues/2583
**State**: closed
**Created**: 2024-12-26T02:23:14+00:00
**Closed**: 2025-03-01T00:18:51+00:00
**Comments**: 13
**Labels**: high priority, inactive, feature

### Description

### Proposal 1: Release KV cache when engine is idle

When using SGLang for generation in a training pipeline (such as PPO), at the phase of running HuggingFace model forward/backward, SGLang currently needs to take a lot of memory even though it does not use it. It would be great to make SGLang use as little memory as possible when it is idle.

Example usage cases:
* Suppose we run OpenRLHF on 8xH100, the currently we may allocate 4xH100 for vllm/SGLang and another 4xH100 for HF model (thanks @zhaochenyang20 for providing this usage scenario).
	* If we make SGLang use little memory when idle, then we can run the same experiment on half number of GPUs (4xH100) by putting those SGLang engines on the same GPUs as HF models.
* Suppose we run PPO on 1xH100 for a 7B model with Adam offloading (thanks @zhaochenyang20 for providing this usage scenario). Then policy (7Bx2) + critic (7Bx2) + ref (7Bx2) + reward (7Bx2) already takes 56B. The current SGLang needs 7Bx2 for weights and some 

[... truncated for brevity ...]

---

## Issue #N/A: Getting this error when loading qwen2.5 VL

**Link**: https://github.com/sgl-project/sglang/issues/3626
**State**: closed
**Created**: 2025-02-17T06:26:46+00:00
**Closed**: 2025-05-01T00:21:10+00:00
**Comments**: 5
**Labels**: inactive

### Description

ImportError: cannot import name 'is_valid_list_of_images' from 'transformers.models.mllama.image_processing_mllama' (/home/team/code/sglang/venv/lib/python3.10/site-packages/transformers/models/mllama/image_processing_mllama.py)

---

## Issue #N/A: Prefix spaces for transofmers' tokenizer to more flexible jump-forward.

**Link**: https://github.com/sgl-project/sglang/issues/213
**State**: closed
**Created**: 2024-02-21T13:15:42+00:00
**Closed**: 2024-07-25T06:32:12+00:00
**Comments**: 1
**Labels**: inactive

### Description

The `transformers` tokenizer now seems to support the `add_prefix_space` option, potentially providing a more flexible way to do our jump-forward.
 
- https://github.com/huggingface/transformers/pull/28010
- https://github.com/huggingface/transformers/issues/28622

---

## Issue #N/A: Inquiry Regarding Qwen2.5-Omni Support in SGLang

**Link**: https://github.com/sgl-project/sglang/issues/4854
**State**: closed
**Created**: 2025-03-28T07:39:07+00:00
**Closed**: 2025-05-28T00:19:24+00:00
**Comments**: 3
**Labels**: inactive

### Description

Hello SGLang Team and Community,

I hope this message finds you well. I wanted to inquire about potential plans to support Qwen2.5-Omni, a multimodal model developed by the Qwen team at Alibaba Cloud. This model offers end-to-end capabilities for text, audio, vision, and video understanding, along with real-time speech generation [1](https://github.com/QwenLM/Qwen2.5-Omni).

I noticed that the vLLM community has recently proposed a patch to support Qwen2.5-Omni (via "thinker only"). Related information can be found here:
1. https://github.com/QwenLM/Qwen2.5-Omni/blob/main/README_CN.md
2. https://github.com/vllm-project/vllm/pull/15130/files#diff-14c1707c1f17226316c95185dbf3d00d39b270354e8c686849320d805f3ccf9f
3. https://github.com/vllm-project/vllm/issues/15563
4. https://huggingface.co/Qwen/Qwen2.5-Omni-7B

As SGLang emphasizes extensibility and active community collaboration [2](https://github.com/sgl-project/sglang)[12](https://pypi.org/project/sglang), I wanted to kindly ask: Are t

[... truncated for brevity ...]

---

## Issue #N/A: Possible timing side-channels caused by shared prefix

**Link**: https://github.com/sgl-project/sglang/issues/1504
**State**: closed
**Created**: 2024-09-24T14:29:46+00:00
**Closed**: 2025-01-19T00:17:51+00:00
**Comments**: 10
**Labels**: inactive

### Description

Dear Sglang Team,
we are a security research group. We are impressed by its decent design, especially by the shared prefix kv-cache. But as we studied further, more concerns about the security of Sglang have arosen. When a new prompt comes, if the `TokenKVPool` has its prefix tokens, the prefill process will be accelerated, which can be reflected in TTFT. We found the timing differences of TTFT introduced by **more shared-tokens** are significant enough to be recognized. 

### Description
Assume the victim has sent a valuable prompt to the sglang, or a valuable system prompt is sent beforehand in sglang, under certain conditions (e.g. the attacker shares the same serving backend with the victim, etc.), the attacker can endeavor to guess the content of the victim prompt and check its validity according to the TTFT.

Different from vLLM (which shares tokens in chunks), Sglang uses token-by-token sharing mechanism (RadixAttention) and cooperates it with trie structure to store kv-ca

[... truncated for brevity ...]

---

