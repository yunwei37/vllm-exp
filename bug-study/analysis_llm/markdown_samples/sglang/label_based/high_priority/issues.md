# high_priority - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 12
- Closed Issues: 18

### Label Distribution

- high priority: 30 issues
- good first issue: 6 issues
- help wanted: 6 issues
- inactive: 6 issues
- collaboration: 3 issues
- MLLM: 3 issues
- bug: 3 issues
- speculative-decoding: 2 issues
- performance: 1 issues
- deepseek: 1 issues

---

## Issue #N/A: [RFC] Bi-weekly release

**Link**: https://github.com/sgl-project/sglang/issues/7332
**State**: open
**Created**: 2025-06-18T23:17:05+00:00
**Comments**: 0
**Labels**: high priority, collaboration

### Description

After thorough internal discussions, the SGLang team has decided to standardize the release cycle as follows:

- A new version will be released every two weeks under normal circumstances (e.g., v0.4.8, v0.4.9).

- If urgent issues or high-priority features arise between regular releases, we may publish a patch release or an additional stable version as needed.

- Bi-weekly releases will typically occur around the middle and end of each month.

- Each release will aim to include a set of planned features, usually discussed and finalized by the SGLang team in advance.



---

## Issue #N/A: [Bug] sglang 0.4.4.post2 Latency greatly increases when tp=1 and dp > 1

**Link**: https://github.com/sgl-project/sglang/issues/5962
**State**: closed
**Created**: 2025-05-02T00:30:20+00:00
**Closed**: 2025-05-11T01:58:01+00:00
**Comments**: 3
**Labels**: high priority

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Noticed that after bumping sglang to 0.4.4.post2+, when setting dp > 1, the latency would increase 10+ times, the more dp we set, the more latency would increase. The issue is not found in sglang version <= 0.4.4.post1.

Data size: 4k per prompt
Endpoint: v1/completions

With max_token=1, latency for tp1 dp1 is ~40ms, but latency for tp1 d

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

## Issue #N/A: [Bug] PD Failed to register memory on H200

**Link**: https://github.com/sgl-project/sglang/issues/6753
**State**: open
**Created**: 2025-05-29T23:27:04+00:00
**Comments**: 2
**Labels**: bug, high priority

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

```
root@nccl-test-host-1:/diagnostic# python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --disaggregation-mode prefill --disaggregation-ib-device mlx5_0
Cuda graph is disabled for prefill server
[2025-05-29 23:22:47] server_args=ServerArgs(model_path='meta-llama/Meta-Llama-3-8B-Instruct', tokenizer_path='meta

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] add more CIs for VLM

**Link**: https://github.com/sgl-project/sglang/issues/5249
**State**: open
**Created**: 2025-04-10T18:44:02+00:00
**Comments**: 7
**Labels**: high priority, MLLM

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
https://huggingface.co/google/gemma-3-27b-it
https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct

### Related resources

_No response_

---

## Issue #N/A: 🚧  RFC: Redesign Batch Processing as an Offline Workflow

**Link**: https://github.com/sgl-project/sglang/issues/7427
**State**: open
**Created**: 2025-06-21T18:23:45+00:00
**Comments**: 1
**Labels**: high priority

### Description

### **Summary**
This RFC proposes removing the existing `/v1/batches` and `/v1/files` endpoints from the main OpenAI-compatible server and replacing them with a standalone offline batch processing service.

> **Note:** As part of the ongoing OpenAI API refactor, the batch support has already been removed from the main server. This RFC serves to document the rationale and formalize the replacement plan.


---

### Problem

#### 7.1 Fundamental Issues with the Current Batch API (#7068 )

The current design for online batch processing is flawed and not production-safe. Key issues include:

- **Server Stability Risk**: Uploading and processing thousands of requests at once can overwhelm online API servers.
- **Timing Constraints**: Difficult to enforce `completion_window` in a real-time environment.
- **Resource Contention**: Batch jobs run alongside latency-sensitive requests without proper isolation.
- **Architecture Mismatch**: Batch workloads are inherently asynchronous/offline, confli

[... truncated for brevity ...]

---

## Issue #N/A: Why are there a group of processes concentrated on a single GPU?

**Link**: https://github.com/sgl-project/sglang/issues/3942
**State**: closed
**Created**: 2025-02-28T03:50:01+00:00
**Closed**: 2025-05-01T00:21:11+00:00
**Comments**: 2
**Labels**: high priority, inactive

### Description

I deployed DeepSeek - R1 on a 8*H20-96G server using the following command.

```
python3 -m sglang.launch_server --model-path DeepSeek-R1 --tp 8 --trust-remote-code --mem-fraction-static 0.9 --host 0.0.0.0 --port 50050 --max-running-requests 128 --context-length 32768 --enable-flashinfer-mla --attention-backend flashinfer
```

However, when using the following command to initiate a request on the H20 server, eight processes will be concentrated on GPU0, as shown in the following screenshot.

```
curl -k -X 'POST' \
    'http://localhost:50050/v1/chat/completions' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
"model": "DeepSeek-R1",
"messages": [{"role": "user", "content": "Hello，Who are you?"}],
"stream": false
}'
```

<img width="1280" alt="Image" src="https://github.com/user-attachments/assets/e44e0c0e-52d2-4c4d-8a58-6224da273e9d" />

Is this normal? Is there any way to distribute these processes across all GPUs to prevent GPU0 from running

[... truncated for brevity ...]

---

## Issue #N/A: TTFT latency for long context (16K) is very high around 15 seconds for llama3.1 70b model. (same or worse than vLLM)

**Link**: https://github.com/sgl-project/sglang/issues/922
**State**: closed
**Created**: 2024-08-04T23:14:23+00:00
**Closed**: 2024-10-09T01:10:58+00:00
**Comments**: 12
**Labels**: high priority, inactive, performance

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

I am experimenting with SGLang and vLLM for long context(16K) RAG application which requires real time responses.
I am using single Nvidia A6000 48GB GPU and llaam3.1 70b awq 4 bit model.

Currently I am seeing Time for first token latency is around 15 seconds which is very high.
Experimented with parameters like --chunked-prefill-size , --mem-frac etc

can you please suggest what are the parameters I need to mainly focus on to get the optimal TTFT for long context ?

### Reproduction

na

### Environment

```Shell
na
```


---

## Issue #N/A: [Feature] integrate FlashMLA

**Link**: https://github.com/sgl-project/sglang/issues/4384
**State**: closed
**Created**: 2025-03-13T10:43:57+00:00
**Closed**: 2025-03-25T04:14:02+00:00
**Comments**: 1
**Labels**: high priority

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Since SGLang now supports page sizes greater than 1, we should integrate FlashMLA https://github.com/deepseek-ai/FlashMLA.

### Related resources

_No response_

---

## Issue #N/A: [Bug] Qwen2-VL-7B with sglang has significant numerical calculation errors compared to HF Transformers

**Link**: https://github.com/sgl-project/sglang/issues/3106
**State**: closed
**Created**: 2025-01-24T11:32:46+00:00
**Closed**: 2025-01-28T06:04:43+00:00
**Comments**: 9
**Labels**: high priority

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

In practice, we found that sglang Qwen2-VL model has numerical calculation errors compared to HF Transformers model in both Qwen2VisionTransformer and Qwen2Model parts.
Our input image has 720 tokens input to Vit encoding, and the lowest embedded cosine similarity in the output is 0.1775. In addition, we directly feed the Vit output and te

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] sglang[all]>=0.4.7

**Link**: https://github.com/sgl-project/sglang/issues/7070
**State**: open
**Created**: 2025-06-10T23:10:43+00:00
**Comments**: 25
**Labels**: high priority

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Almost half the performance drop when running 0.4.7 vs 0.4.6

Tested two models
Qwen3-32B-FP8 75T/s (4xAda 6000s) to 45T/s
Qwen3-30B-A3B - 160T/s  (4x3090s) BF16 to 80T/s with 0.4.7


### Reproduction

python -m sglang.launch_server --model-path models/Qwen3-32B-FP8 \
--context-length 131072 \
--json-model-override-args '{"rope_scaling":{"

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support DeepSeek VL 2

**Link**: https://github.com/sgl-project/sglang/issues/2653
**State**: closed
**Created**: 2024-12-30T06:45:23+00:00
**Closed**: 2025-03-25T04:11:43+00:00
**Comments**: 6
**Labels**: good first issue, help wanted, high priority

### Description

### Motivation

deepseek-vl2 is one of the best vision language models. We would like to support it.

https://huggingface.co/deepseek-ai/deepseek-vl2
https://github.com/deepseek-ai/DeepSeek-VL2

### Related resources

You can learn from the existing implementations and usage examples of other vision language models.
https://sgl-project.github.io/references/supported_models.html#how-to-support-a-new-model
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llava.py
https://github.com/sgl-project/sglang/blob/main/test/srt/test_vision_openai_server.py
https://sgl-project.github.io/references/sampling_params.html#multi-modal

---

## Issue #N/A: [Bug] CUDA OOM when DP attention enabled, maybe due to incorrect acceptable length estimation.

**Link**: https://github.com/sgl-project/sglang/issues/6027
**State**: open
**Created**: 2025-05-05T11:34:01+00:00
**Comments**: 12
**Labels**: high priority

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

In my understanding, sglang should estimate max acceptable length with the calculation result of remaining VRAM and kvcache size, for instance server log shows:

```
[2025-05-05 03:55:08 TP0] KV Cache is allocated. #tokens: 274934, KV size: 17.99 GB
```

However, a single 64k-length input can consistently cause the server to crash due to O

[... truncated for brevity ...]

---

## Issue #N/A: DeepSeek-R1 Optimization Option Ablations

**Link**: https://github.com/sgl-project/sglang/issues/3956
**State**: closed
**Created**: 2025-02-28T09:33:38+00:00
**Closed**: 2025-07-06T00:22:09+00:00
**Comments**: 36
**Labels**: high priority, inactive, deepseek, speculative-decoding

### Description

Updated on **2025-03-20**: #4616

Updated on **2025-03-04**:
# DeepSeek Optimization Ablations

## Overview

We sincerely thanks for the help from [M0gician](http://m0gician.github.io/) for the massive experiments.

**As of 2025-03-04**, SGLang provides the following optimizations for DeepSeek V3/R1 models:

| Name                                        | Description                                                                                                                                                                                                                                     | Enabled by Default | Enable/Disable Argument                                                                                                                                   |
|---------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] 2 * 8 * H20 image : lmsysorg/sglang:deepep run error

**Link**: https://github.com/sgl-project/sglang/issues/7450
**State**: open
**Created**: 2025-06-23T01:42:34+00:00
**Comments**: 6
**Labels**: high priority

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug


/sgl-workspace/nvshmem/src/modules/transport/ibrc/ibrc.cpp:418: non-zero status: 22 ibv_modify_qp failed

/sgl-workspace/nvshmem/src/modules/transport/ibrc/ibrc.cpp:1433: non-zero status: 7 ep_connect failed

/sgl-workspace/nvshmem/src/modules/transport/ibrc/ibrc.cpp:1500: non-zero status: 7 transport create connect failed

/sgl-workspace

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

## Issue #N/A: [Bug] missing allreduce from sgl_kernel module on mi30x GPUs

**Link**: https://github.com/sgl-project/sglang/issues/4296
**State**: closed
**Created**: 2025-03-11T08:00:00+00:00
**Closed**: 2025-03-11T11:11:48+00:00
**Comments**: 2
**Labels**: high priority

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

in latest main branch to run on MI30x GPUs,  allreduce kernel is missed from sgl_kernel, may need a track

```yml
  File "/workspace/github/sglang/python/sglang/srt/_custom_ops.py", line 96, in meta_size
    return sgl_kernel.allreduce.meta_size()
           ^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'sgl_kernel' has no attribute 'allredu

[... truncated for brevity ...]

---

## Issue #N/A: rewrite test_trt_allreduce

**Link**: https://github.com/sgl-project/sglang/issues/4907
**State**: closed
**Created**: 2025-03-30T02:00:42+00:00
**Closed**: 2025-03-30T08:02:00+00:00
**Comments**: 0
**Labels**: high priority

### Description

as titled https://github.com/sgl-project/sglang/blob/main/sgl-kernel/tests/test_trt_allreduce.py @yizhang2077 

---

## Issue #N/A: Development Roadmap (2025 H2)

**Link**: https://github.com/sgl-project/sglang/issues/7736
**State**: open
**Created**: 2025-07-03T06:04:23+00:00
**Comments**: 0
**Labels**: high priority, collaboration

### Description

The SGLang team is expected to complete planning for the H2 roadmap within the next two weeks. Stay tuned—exciting things are on the way!


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

## Issue #N/A: [Bug] Deepseek FP4 doesn't support MTP

**Link**: https://github.com/sgl-project/sglang/issues/7365
**State**: closed
**Created**: 2025-06-19T18:40:58+00:00
**Closed**: 2025-06-25T18:27:55+00:00
**Comments**: 0
**Labels**: high priority

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Currently when using MTP with FP4 Deepseek, the server will crash with

```
  File "/sgl-workspace/sglang/python/sglang/srt/model_loader/loader.py", line 381, in load_model
    self.load_weights_and_postprocess(
  File "/sgl-workspace/sglang/python/sglang/srt/model_loader/loader.py", line 389, in load_weights_and_postprocess
    model.load

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

## Issue #N/A: [Feature] support minference attention backend

**Link**: https://github.com/sgl-project/sglang/issues/5329
**State**: closed
**Created**: 2025-04-12T19:02:15+00:00
**Closed**: 2025-06-12T00:19:28+00:00
**Comments**: 1
**Labels**: high priority, inactive

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled @minminsun @yinfan98 @ZhangJianwei0311

ref https://github.com/sgl-project/sglang/pull/5327

### Related resources

_No response_

---

## Issue #N/A: [Feature] Cascade attention kernels 

**Link**: https://github.com/sgl-project/sglang/issues/1715
**State**: open
**Created**: 2024-10-19T16:30:29+00:00
**Comments**: 5
**Labels**: good first issue, high priority

### Description

We would like to integrate the [cascade attention kernel](https://flashinfer.ai/2024/02/02/cascade-inference.html) from flashinfer.

Code pointers:
- Attention backend in sglang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_backend.py
- Usage of cascade: https://docs.flashinfer.ai/api/python/cascade.html


---

## Issue #N/A: [Feature] Support RM API

**Link**: https://github.com/sgl-project/sglang/issues/1384
**State**: closed
**Created**: 2024-09-11T04:27:29+00:00
**Closed**: 2024-10-19T14:52:16+00:00
**Comments**: 9
**Labels**: high priority

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Does SGLang support rapid deployment of RM services?
Or convenient custom APIs? It seems that currently there are only chat/completion/embedding APIs. As a newcomer to inference acceleration, any help would be beneficial.

### Related resources

copied from https://github.com/vllm-project/vllm/issues/6620, same demand

---

## Issue #N/A: [Bug] Re-enable fused_moe_triton on AMD

**Link**: https://github.com/sgl-project/sglang/issues/2347
**State**: closed
**Created**: 2024-12-04T09:53:34+00:00
**Closed**: 2024-12-07T13:44:00+00:00
**Comments**: 2
**Labels**: bug, high priority

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

To fix some code miss outs from migration

### Reproduction

Functional test

### Environment

ROCm container

---

## Issue #N/A: Development Roadmap (2025 H1)

**Link**: https://github.com/sgl-project/sglang/issues/4035
**State**: closed
**Created**: 2025-03-03T18:26:58+00:00
**Closed**: 2025-03-03T19:11:15+00:00
**Comments**: 1
**Labels**: high priority

### Description

Here is the development roadmap for 2025 Q1 and Q2. Contributions and feedback are welcome ([**Join Bi-weekly Development Meeting**](https://docs.google.com/document/d/1xEow4eIM152xNcRxqZz9VEcOiTQo8-CEuuQ5qTmkt-E/edit?tab=t.0#heading=h.ito5nvp7oasg)). Previous 2024 Q4 roadmap can be found in #1487.

### DeepSeek R1 optimization
@zhyncs @ispobock 
TBD

## Performance
- [ ] Support speculative decoding
  - Eagle Optimization #3822 
  - Reference-based. #3269 
  - Align with the speed of grok
- [ ] P/D Disaggregation
  - Bump internal codes
  - Mooncake Integration

## Parallelism
- [ ] Support sequence parallelism #1436. Related [paper](https://www.arxiv.org/pdf/2411.01783)
- [ ] Support pipeline parallelism.
- [ ] Optimize expert parallelism + data parallelism for DeepSeekmodels.
- [ ] Optimize expert parallelism for Qwen Models.
- [ ] Overlap communication in tensor parallelsim. @zhuohaol @fzyzcjy 

## Hardware Optimizations
- [ ] AMD optimizations. @HaiShaw @yiakwy-xpu-ml-framework-te

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] use modelopt for fp8 and fp4 by default

**Link**: https://github.com/sgl-project/sglang/issues/5251
**State**: open
**Created**: 2025-04-10T18:53:53+00:00
**Comments**: 7
**Labels**: documentation, good first issue, help wanted, high priority

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

https://github.com/NVIDIA/TensorRT-Model-Optimizer is the **de facto** LLM quant library for fp8 and fp4, supported in both TensorRT LLM and SGLang. We will consider changing all current fp8, fp4 doc, CI, unit test, etc. to default to ModelOpt's checkpoint

ref https://huggingface.co/nvidia

### Related resources

_No response_

---

## Issue #N/A: [Roadmap] EP Enhancement

**Link**: https://github.com/sgl-project/sglang/issues/4734
**State**: open
**Created**: 2025-03-24T18:48:57+00:00
**Comments**: 30
**Labels**: high priority, collaboration

### Description

- [x] Support normal DeepEP buffer @liz-badada  #4232 
- [x] Support DeepEP with async transfer @fzyzcjy #4610 
- [x] Support low-latency DeepEP buffer
  - [x] Single-node TP @liz-badada #4767 
    - MaskedDeepGeMM is implemented by @laixinn @sleepcoo 
    - Improved by @yuleil #5277 
  - [x] Multi-node TP @liz-badada #5068 
  - [x] Support PD disaggregation @ch-wan  #5435 
- [ ] Integrate pplx-kernels @ruizhang1230 #5010 
- [ ] Optimize permutation overhead
  - [x] Implement Titon kernels @xutizhou #4643 
  - [ ] Fuse permutation with GroupedGeMM
- [x] Extend parallelism paradigm
  - [x] Extend DeepEP to a general TP paradigm @ch-wan @tarinkk #4770 
    - Fixed by @fzyzcjy #4883 
  - [x] Support `tp_size < ep_size`
    - `tp_size=1` @fzyzcjy #4836
- [x] Overlap two batches @fzyzcjy #4068 
- [x] Integrate continuous DeepGeMM @sleepcoo @xutizhou  #5626 
- [x] Record expert distribution @yuhsuan-t #4435 
  - Improved by @fzyzcjy #4957  
- [ ] Overlap communication with shared experts’ co

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] optimize SegmentPackBits

**Link**: https://github.com/sgl-project/sglang/issues/5437
**State**: closed
**Created**: 2025-04-15T23:03:41+00:00
**Closed**: 2025-06-16T00:20:43+00:00
**Comments**: 2
**Labels**: high priority, inactive, speculative-decoding

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Currently `SegmentPackBits` is not performant, implement a performant one in sgl-kernel

https://github.com/sgl-project/sglang/blob/8ec0bb7d558d1722be4efb8b3abf5e09c0e9c20e/sgl-kernel/csrc/speculative/packbit.cu#L39

https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/quantization.cuh#L98

### Related resources

_No response_

---

