# help_wanted - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 6
- Closed Issues: 24

### Label Distribution

- help wanted: 30 issues
- good first issue: 18 issues
- inactive: 6 issues
- high priority: 6 issues
- MLLM: 3 issues
- enhancement: 2 issues
- grammar-backend: 1 issues
- bug: 1 issues
- quant: 1 issues
- collaboration: 1 issues

---

## Issue #N/A: Any benchmarks comparing with TGI?

**Link**: https://github.com/sgl-project/sglang/issues/3188
**State**: closed
**Created**: 2025-01-27T22:14:34+00:00
**Closed**: 2025-01-30T17:42:07+00:00
**Comments**: 2
**Labels**: help wanted

### Description

As the tittle says, is there any benchmark comparing with TGI (https://github.com/huggingface/text-generation-inference)? I see some results comparing directly with vLLM, but would love to see also a direct comparison against TGI, as in the last release the got a good performance improvement, thanks for the info in advance!

---

## Issue #N/A: [Feature] Use xgrammar as default grammar backend to aviod I/O errors while using Outlines in a multi-node setting

**Link**: https://github.com/sgl-project/sglang/issues/3383
**State**: closed
**Created**: 2025-02-07T23:11:12+00:00
**Closed**: 2025-05-26T21:08:02+00:00
**Comments**: 4
**Labels**: good first issue, help wanted, grammar-backend

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

related issues:
#3375 
related discussiton:
[#vllm 4193](https://github.com/vllm-project/vllm/issues/4193)
related pr:
https://github.com/sgl-project/sglang/pull/3379

### Related resources

xGrammar stores its cache in RAM instead of disk, avoiding file system conflicts.
Cache size is small (typically <0.5MB per schema), meaning it doesn't require persistent disk storage.
xGrammar is thread-safe, ensuring it can run across multiple Slurm nodes without concurrency issues.

---

## Issue #N/A: [Feature] Support ipv6 in SGLang

**Link**: https://github.com/sgl-project/sglang/issues/3263
**State**: closed
**Created**: 2025-02-02T19:37:24+00:00
**Closed**: 2025-05-15T00:49:46+00:00
**Comments**: 4
**Labels**: good first issue, help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

@shuaills 

https://github.com/sgl-project/sglang/issues/2892#issuecomment-2629436443

### Related resources

_No response_

---

## Issue #N/A: [Bug] RuntimeError: RMSNorm failed with error code invalid configuration argument

**Link**: https://github.com/sgl-project/sglang/issues/3304
**State**: closed
**Created**: 2025-02-05T02:25:13+00:00
**Closed**: 2025-05-11T15:17:16+00:00
**Comments**: 22
**Labels**: good first issue, help wanted

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hi, I am using the main branch of SGLang, and downloading Mixtral-8x22B from huggingface. 

CUDA: 12.4
2 nodes, each has 4 H100 96GB.

I am deploying the server using:
```
python -m sglang.launch_server --model-path Mixtral-8x22B-v0.1 --tp 8 --dist-init-addr xxx:5000 --nnodes 2 --node-rank 0 --trust-remote-code --disable-cuda-graph
python 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] finish_reason is not right when Qwen call a tool

**Link**: https://github.com/sgl-project/sglang/issues/2877
**State**: closed
**Created**: 2025-01-14T03:06:37+00:00
**Closed**: 2025-05-13T00:19:06+00:00
**Comments**: 7
**Labels**: help wanted, inactive

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

{
    "completion": {
        "created": 1736822678,
        "usage": {
            "completion_tokens": 75,
            "prompt_tokens": 43,
            "total_tokens": 118
        },
        "model": "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
        "id": "a82af6309caf48a0994c77acbedbc846",
        "choices": [
            {
   

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] fix gemma-2-2b-it-FP8 accuracy

**Link**: https://github.com/sgl-project/sglang/issues/4324
**State**: closed
**Created**: 2025-03-12T01:27:58+00:00
**Closed**: 2025-05-21T09:30:43+00:00
**Comments**: 8
**Labels**: bug, good first issue, help wanted, high priority, quant

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

The accuracy of `neuralmagic/gemma-2-2b-it-FP8` drops from 0.62 to 0.52 in the main branch. It was detected by our nightly CI run. We need to fix this.

```
neuralmagic/gemma-2-2b-it-FP8 | 0.512 | 0.6
```
https://github.com/sgl-project/sglang/actions/runs/13800885290

### Reproduction

N/A

### Environment

N/A

---

## Issue #N/A: [Feature] Support DeepSeek Janus Models

**Link**: https://github.com/sgl-project/sglang/issues/3195
**State**: closed
**Created**: 2025-01-28T18:37:47+00:00
**Closed**: 2025-04-30T00:18:51+00:00
**Comments**: 4
**Labels**: help wanted, inactive, MLLM

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Docker is a valuable tool for the management of dependencies. Indeed, it can simplify the running of Janus Models to a single command:  
```bash
docker run -it --rm \
  -p 8000:8000 \
  -d \
  -v huggingface:/root/.cache/huggingface \
  -w /app \
  --gpus all \
  --name janus \
  -e MODEL_NAME=deepseek-ai/Janus-Pro-7B \
  julianfl0w/janus:latest
```

Make sure it's working by navigating in your browser to  
[http://localhost:8000/webui](http://localhost:8000/webui)

and by running
```bash
docker logs janus
```

This keeps all the Torch dependencies contained within the image, meaning the user doesn't have to adjust their base installations to run models like these. 

Note: You will have to install NVIDIA Container 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Running multi-node offline engine inference ( via SLURM)

**Link**: https://github.com/sgl-project/sglang/issues/2561
**State**: closed
**Created**: 2024-12-23T15:24:49+00:00
**Closed**: 2025-01-31T23:58:27+00:00
**Comments**: 39
**Labels**: help wanted, collaboration, feature

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

A lot of academic institutions only allow access to larger node clusters via SLURM and it is not immediately clear how would I reuse the code to run Llama 405B BF16 on 2 nodes (by starting a server) to perform offline inference

### Related resources

_No response_

---

## Issue #N/A: [Feature] deepseek v3 60 tokens/sec on deepseek API vs. 13 tokens/sec on sglang

**Link**: https://github.com/sgl-project/sglang/issues/3196
**State**: closed
**Created**: 2025-01-28T18:40:18+00:00
**Closed**: 2025-02-15T01:21:30+00:00
**Comments**: 29
**Labels**: help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

The PR for AMD + sglang and NVIDIA + sglang was that it was "fully" supported, but it seems something is off by the speed.  A single sequence runs at only order 13 tokens/sec for long generation with TTFT order 2 seconds.  This is consistent with vLLM as well.  True for either 8*MI300X or 8*H200 or 2*8*H200.

For only 37B parameters + 14B MOE parameters, this seems way too slow.  Also, deepseek API (before it started to break down) was order 60 tokens/sec early on and they advertise 60 tokens/sec.  This is more aligned with the parameters active.

What is missing from truly fully suppporting deepseek V3 and R1?  Can these features be enumerated and added in a roadmap?

### Related resources

_No response_

---

## Issue #N/A: [Feature] Support EBNF in xgrammar

**Link**: https://github.com/sgl-project/sglang/issues/2376
**State**: closed
**Created**: 2024-12-06T12:07:00+00:00
**Closed**: 2025-05-26T00:02:55+00:00
**Comments**: 2
**Labels**: good first issue, help wanted

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

xgrammar supports EBNF. We would like to integrate this feature into SGLang.

We can add a new parameter called `ebnf` in sampling_params.py and treat it similar to regex and JSON.


### Related resources

https://xgrammar.mlc.ai/docs/how_to/ebnf_guided_generation.html
https://github.com/sgl-project/sglang/blob/f5b2a3aa67efb10918965b9f3555ff24ef971902/python/sglang/srt/sampling/sampling_params.py#L36-L38
https://github.com/sgl-project/sglang/blob/main/test/srt/test_json_constrained.py

---

## Issue #N/A: [Bug] Testing new Llama-3_3-Nemotron-Super-49B-v1 by Nvidia: "Model architectures ['DeciLMForCausalLM'] are not supported for now."

**Link**: https://github.com/sgl-project/sglang/issues/4689
**State**: open
**Created**: 2025-03-23T05:40:20+00:00
**Comments**: 14
**Labels**: enhancement, good first issue, help wanted, new-model

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I tried to run on SGLang Llama-3_3-Nemotron-Super-49B-v1 recently announced by Nvidia.

It seems not to be yet supported by SGLang since `DeciLMForCausalLM`is not yet accepted by SGLang. See below.

Can you add corresponding support?

```
Scheduler hit an exception: Traceback (most recent call last):
  File "/usr/local/lib/python3.12/site-

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Improve Multi-node recipe to run inference

**Link**: https://github.com/sgl-project/sglang/issues/3206
**State**: closed
**Created**: 2025-01-29T12:21:53+00:00
**Closed**: 2025-01-31T23:48:24+00:00
**Comments**: 13
**Labels**: help wanted, deepseek

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Could someone improve the example for serving DeepSeek 3 on multiple nodes adding information on how to run into a slurm cluster and singularity container.

https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-2-h208

### Related resources

_No response_

---

## Issue #N/A: [Bug] ensure the git clone of long_prompt.txt

**Link**: https://github.com/sgl-project/sglang/issues/4976
**State**: closed
**Created**: 2025-04-01T19:21:25+00:00
**Closed**: 2025-04-30T07:03:26+00:00
**Comments**: 2
**Labels**: good first issue, help wanted

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I often encounter this problem: The sglang cloned by git does not have this file: sglang/test/long_prompt.txt, so I have to manually download one and put it in the corresponding position, such as /dev/shm/chenyang/.python/veRL-server/lib/python3.10/site-packages/sglang/test/long_prompt.txt

could someone try to fix this as a good first iss

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support bitsandbytes in QWen2 VL

**Link**: https://github.com/sgl-project/sglang/issues/2729
**State**: open
**Created**: 2025-01-04T08:12:29+00:00
**Comments**: 2
**Labels**: good first issue, help wanted

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Support bitsandbytes in QWen2 VL

### Related resources

_No response_

---

## Issue #N/A: [Feature] support sgl-kernel cu128 build

**Link**: https://github.com/sgl-project/sglang/issues/4501
**State**: closed
**Created**: 2025-03-17T09:16:55+00:00
**Closed**: 2025-04-18T06:10:04+00:00
**Comments**: 6
**Labels**: good first issue, help wanted, high priority

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

for blackwell

### Related resources

_No response_

---

## Issue #N/A: [Feature] update sgl-kernel 3rdparty flashinfer to latest main

**Link**: https://github.com/sgl-project/sglang/issues/4301
**State**: closed
**Created**: 2025-03-11T08:18:52+00:00
**Closed**: 2025-05-26T00:26:08+00:00
**Comments**: 2
**Labels**: good first issue, help wanted, high priority

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

fix the compile issue

### Related resources

_No response_

---

## Issue #N/A: Do not use tools param in stream request!

**Link**: https://github.com/sgl-project/sglang/issues/2810
**State**: closed
**Created**: 2025-01-09T08:40:45+00:00
**Closed**: 2025-03-23T00:19:21+00:00
**Comments**: 2
**Labels**: help wanted, inactive

### Description

https://github.com/sgl-project/sglang/blob/b5fb4ef58a6bbe6c105d533b69e8e8bc2bf4fc3c/python/sglang/srt/openai_api/adapter.py#L882

If you give a tools param in your request and set stream=True, then the output format will be changed by the server and you will get nothing by `for` grammar (no error will be raised), because the two processing are complete different in the client:
```
stream -> received with generator of chunks: generater -> async for chunk in result:
non-stream-> received with a fixed result chunk -> use it direct
```

So, I think if the server does not support stream with tools, then it will be better to return a http error than changing the return method so that the developers can know what should  be done or not.

---

## Issue #N/A: Some question about layernom in MLA code

**Link**: https://github.com/sgl-project/sglang/issues/3072
**State**: closed
**Created**: 2025-01-23T07:03:32+00:00
**Closed**: 2025-01-23T13:28:26+00:00
**Comments**: 2
**Labels**: help wanted

### Description

Hi，I am confused that there is a layer normalization between the down-sample and up-sample of Q. However, this layer normalization is not shown in the DeepSeek v2 paper.

Here is the code of sglang

![Image](https://github.com/user-attachments/assets/6ab58ca0-f722-4447-9041-e54fd6a86b37)

Here is the formulate in paper

![Image](https://github.com/user-attachments/assets/78722b31-4015-4fbd-9064-fd8e66dc1caa)

---

## Issue #N/A: [Bug] disk cache io error when simultaneously loading lots of  sglang offline engine

**Link**: https://github.com/sgl-project/sglang/issues/2090
**State**: open
**Created**: 2024-11-19T08:46:49+00:00
**Comments**: 13
**Labels**: good first issue, help wanted

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

when I use slurm to launch 32 or 192 jobs for offline batch inference, which simultaneously load sgl.engine. I met the following error although I set disable_disk_cache=True. If I only run one job for this, it will not meet this error.


The error is as follows:

```Python
Traceback (most recent call last):
  File "/home/x

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] chat template of Llama 3.1 injects a wrong today's date in the system prompt.

**Link**: https://github.com/sgl-project/sglang/issues/3296
**State**: closed
**Created**: 2025-02-04T20:07:45+00:00
**Closed**: 2025-04-07T00:18:48+00:00
**Comments**: 3
**Labels**: help wanted, inactive

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

I am running a Llama 3.1-8B model and access it by OpenAI client. The generated prompt includes the wrong today's date.

The possible reason is from the HF's [chat template](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/tokenizer_config.json). 

In the template, `date_string` is a parameter.
```
{%- if not date_string i

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Crash special token xgrammar

**Link**: https://github.com/sgl-project/sglang/issues/3108
**State**: closed
**Created**: 2025-01-24T13:15:33+00:00
**Closed**: 2025-05-26T00:20:00+00:00
**Comments**: 11
**Labels**: help wanted, inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When using xgrammar with an EBNF grammar, SGLang will crash if the model outputs a reserved token.

```
[2025-01-24 04:52:54 TP1] Scheduler hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 1756, in run_scheduler_process
    scheduler.event_loop_overlap()
  Fil

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] use pytest for sgl-kernel

**Link**: https://github.com/sgl-project/sglang/issues/4690
**State**: closed
**Created**: 2025-03-23T06:09:52+00:00
**Closed**: 2025-04-03T21:49:11+00:00
**Comments**: 0
**Labels**: good first issue, help wanted

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

https://github.com/sgl-project/sglang/tree/main/sgl-kernel/tests
Some tests use unittest, we want to switch them to pytest.

### Related resources

_No response_

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

## Issue #N/A: Further Speed up FA3 Backend

**Link**: https://github.com/sgl-project/sglang/issues/5810
**State**: open
**Created**: 2025-04-28T04:57:56+00:00
**Comments**: 13
**Labels**: enhancement, good first issue, help wanted

### Description

We explored and discussed some ideas and we want to write it down for tracking, also welcome community developer to try out those unfinished

- [x] (Good first issue) Skip `len` operation, get it directly from forward batch: https://github.com/sgl-project/sglang/pull/5969 @lifuhuang 
- [ ] GQA head packing: https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_attn_interface.py#L658 Change it to True and run benchmark.
- [x] Split-KV. aka Flash Decoding: We already enabled it, it is indeed faster in lower batch and long context scenario. Benchmark will be attached.
- [ ] PDL: https://github.com/Dao-AILab/flash-attention/commit/000090d02f0398e9087a8823fc1f5242becfac99
- [x] (Won't do) Prepare Scheduler Metadata: https://github.com/Dao-AILab/flash-attention/commit/fa60e7cc97300b4b26721983df580a7da7a8ebea (From Tri Dao's note, it can only speed up 2us, we can keep an eye on this, not recommending adopting this)
- [ ] For Llama Models, we observed that Spec Decoding with Top 

[... truncated for brevity ...]

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

## Issue #N/A: [Bug] Decode Throughput Inconsistency Between bench_serving and Engine Logs

**Link**: https://github.com/sgl-project/sglang/issues/3050
**State**: open
**Created**: 2025-01-22T11:32:09+00:00
**Comments**: 16
**Labels**: good first issue, help wanted

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hi, I encountered an inconsistency in decode throughput reporting. When benchmarking with the bench_serving script, the reported TPOT is **much lower** than the decode throughput logged by the engine. This gap is significant for **small models or high concurrency settings**.

### Reproduction

#### start the server
```
python -m sglang.lau

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Add return hidden state in the native API

**Link**: https://github.com/sgl-project/sglang/issues/3461
**State**: closed
**Created**: 2025-02-10T06:26:45+00:00
**Closed**: 2025-02-27T06:06:55+00:00
**Comments**: 4
**Labels**: good first issue, help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

JM is submitting a feature to get a hidden state. We can add examples at the beginning of the test file `test/srt/test_hidden_states.py` right now. Later rewrite this API and add it in the docs.

Try to add a native API instead of adding a parameter and relaunching the engine.

If anyone is interested in this, could reach out to me and try to get in touch.

<img width="635" alt="Image" src="https://github.com/user-attachments/assets/32d66df2-a86b-408f-a02f-b2cb289e012e" />

### Related resources

https://github.com/sgl-project/sglang/pull/3364

---

## Issue #N/A: [Feature] Update Supported Models

**Link**: https://github.com/sgl-project/sglang/issues/3707
**State**: closed
**Created**: 2025-02-19T18:48:47+00:00
**Closed**: 2025-05-27T01:37:38+00:00
**Comments**: 7
**Labels**: good first issue, help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

https://docs.sglang.ai/references/supported_models.html

This should be checked and updated.

### Related resources

_No response_

---

## Issue #N/A: deepseek-r1-qwen-32B stuck when python -m sglang.lanunch_server

**Link**: https://github.com/sgl-project/sglang/issues/3765
**State**: closed
**Created**: 2025-02-21T14:01:23+00:00
**Closed**: 2025-02-22T09:43:40+00:00
**Comments**: 5
**Labels**: help wanted

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

```
(sglang) root@node1:~# python -m sglang.launch_server --model-path /root/sdb2/DeepSeek-R1-Distill-Qwen-32B --context-length 32768 --tensor-parallel-size 2 --chunked-prefill-size 4096 --enable-p2p-check --host 172.16.21.155 --port 8020 --mem-fraction-static 0.5
/root/anaconda3/envs/sglang/lib/python3.11/site-packages/transformers/models

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Accuracy test of VLM

**Link**: https://github.com/sgl-project/sglang/issues/3142
**State**: open
**Created**: 2025-01-26T06:25:40+00:00
**Comments**: 7
**Labels**: good first issue, help wanted, high priority, MLLM

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

In sglang, LLMs have accuracy tests with Hugging Face models:

https://github.com/sgl-project/sglang/blob/main/test/srt/models/test_generation_models.py

https://github.com/sgl-project/sglang/blob/main/test/srt/test_nightly_math_eval.py

We need similar one for VLM also.

### Related resources

_No response_

---

