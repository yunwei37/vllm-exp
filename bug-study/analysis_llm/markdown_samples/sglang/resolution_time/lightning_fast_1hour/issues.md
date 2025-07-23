# lightning_fast_1hour - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- deepseek: 2 issues
- high priority: 2 issues
- help wanted: 1 issues

---

## Issue #N/A: [Feature] any accuracy score on MMLU or CMMLU ?

**Link**: https://github.com/sgl-project/sglang/issues/1054
**State**: closed
**Created**: 2024-08-12T11:04:55+00:00
**Closed**: 2024-08-12T11:16:24+00:00
**Comments**: 3

### Description

### Motivation

I have tested the sglang with multi llmperf tools, really impresive.
Any benchmark on accuracy score on MMLU or CMMLU 

### Related resources

_No response_

---

## Issue #N/A: [Feature] Suggestion: Add Documentation for PD Disaggregation (link to Mooncake integration guide)

**Link**: https://github.com/sgl-project/sglang/issues/6371
**State**: closed
**Created**: 2025-05-17T11:58:22+00:00
**Closed**: 2025-05-17T12:11:12+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Hi team,

SGLang has already implemented PD Disaggregation, which is great. To make this feature easier to adopt, I suggest adding documentation or linking to the following integration guide from the Mooncake project: https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/sglang-integration-v1.md

This guide provides clear instructions for enabling SGLang's PD disaggregation feature and could be very helpful to users.

Thanks!

### Related resources

_No response_

---

## Issue #N/A: [Bug] deepseek v3 inference with 2*8*H800 cannot STOP

**Link**: https://github.com/sgl-project/sglang/issues/2795
**State**: closed
**Created**: 2025-01-08T12:31:23+00:00
**Closed**: 2025-01-08T12:57:38+00:00
**Comments**: 1

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

## output info
```
[2025-01-08 20:17:32 TP0] Decode batch. #running-req: 2, #token: 27373, token usage: 0.09, gen throughput (token/s): 33.38, #queue-req: 0
[2025-01-08 20:17:34 TP0] Decode batch. #running-req: 2, #token: 27453, token usage: 0.09, gen throughput (token/s): 33.32, #queue-req: 0
[2025-01-08 20:17:37 TP0] Decode batch. #r

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] tensor parallel run error

**Link**: https://github.com/sgl-project/sglang/issues/1509
**State**: closed
**Created**: 2024-09-25T02:09:52+00:00
**Closed**: 2024-09-25T02:12:18+00:00
**Comments**: 1

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

`python3 -m sglang.bench_latency --model meta-llama/Meta-Llama-3-8B --batch-size 1 --input 128 --output 8 --tensor-parallel-size 2`


[19:01:01 TP0] Init nccl begin.
[19:01:01 TP1] Init nccl begin.
NCCL version 2.20.5+cuda12.4
Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:307 'peer access is not supported between these two

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] v0.4.2.post4 ValueError: Unrecognized configuration class <class 'transformers_modules.DeepSeek-R1.configuration_deepseek.DeepseekV3Config'> to build an AutoTokenizer.

**Link**: https://github.com/sgl-project/sglang/issues/3569
**State**: closed
**Created**: 2025-02-14T06:25:13+00:00
**Closed**: 2025-02-14T06:51:17+00:00
**Comments**: 1
**Labels**: deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

ValueError: Unrecognized configuration class <class 'transformers_modules.DeepSeek-R1.configuration_deepseek.DeepseekV3Config'> to build an AutoTokenizer.
Model type should be one of AlbertConfig, AlignConfig, AriaConfig, BarkConfig, BartConfig, BertConfig, BertGenerationConfig, BigBirdConfig, BigBirdPegasusConfig, BioGptConfig, Blenderbot

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]

**Link**: https://github.com/sgl-project/sglang/issues/7083
**State**: closed
**Created**: 2025-06-11T06:27:38+00:00
**Closed**: 2025-06-11T06:47:05+00:00
**Comments**: 5

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

[2025-06-11 06:19:30] INFO:     172.20.0.1:40106 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
[2025-06-11 06:19:30] ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "/root/anaconda3/envs/sglang/lib/python3.12/site-packages/uvicorn/protocols/http/h11_impl.py", line 403, in run_asgi
    re

[... truncated for brevity ...]

---

## Issue #N/A: [Model] Adding support for MiniCPM-Llama3-V-2_5

**Link**: https://github.com/sgl-project/sglang/issues/562
**State**: closed
**Created**: 2024-06-25T01:06:15+00:00
**Closed**: 2024-06-25T01:06:28+00:00
**Comments**: 1

### Description

Please support for **MiniCPM-Llama3-V-2_5**. 
- HuggingFace Page: https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5
- Github : https://github.com/OpenBMB/MiniCPM-V





---

## Issue #N/A: Which one is faster for token generation? LLama.cpp v.s. SGLang

**Link**: https://github.com/sgl-project/sglang/issues/3462
**State**: closed
**Created**: 2025-02-10T06:52:00+00:00
**Closed**: 2025-02-10T06:55:25+00:00
**Comments**: 1

### Description

Is there any benchmark comparison based on the same batch size (= 1 for token generation) and same quant.ed type?

---

## Issue #N/A: [Bug] RuntimeError: NCCL error: unhandled system error (run with NCCL_DEBUG=INFO for details)

**Link**: https://github.com/sgl-project/sglang/issues/3460
**State**: closed
**Created**: 2025-02-10T06:21:55+00:00
**Closed**: 2025-02-10T06:23:23+00:00
**Comments**: 0

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

出现如下错误：
File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 1787, in run_scheduler_process
scheduler = Scheduler(server_args, port_args, gpu_id, tp_rank, dp_rank)
File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 240, in init
self.tp_worker = TpWorkerClass(
File "/sgl-workspace/sglang/python/

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] B200 support timeline

**Link**: https://github.com/sgl-project/sglang/issues/6435
**State**: closed
**Created**: 2025-05-19T21:06:20+00:00
**Closed**: 2025-05-19T21:06:34+00:00
**Comments**: 0

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Hi there - would love to know if B200 will be available and when. 

We're doing some tests on Sglang on 8xB200s, and found the image from docker hub to freeze. It seems like Blackwell support is still an ongoing thing. 

Thanks for any info!

### Related resources

_No response_

---

## Issue #N/A: [Bug]

**Link**: https://github.com/sgl-project/sglang/issues/3817
**State**: closed
**Created**: 2025-02-24T12:30:20+00:00
**Closed**: 2025-02-24T12:32:47+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

### environment
docker-image: lmsysorg/sglang:v0.4.3.post2-cu125 (entrypoint: /bin/bash)
command to launch docker container: `docker run -itd --name sglang -v /export:/export -p 6000:6000 --gpus=all --entrypoint /bin/bash --shm-size=10gb lmsysorg/sglang:v0.4.3.post2-cu125`
<details>
<summary>output from vllm/collect_env.py: </summary>

```

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Performance on DeepSeek-V2

**Link**: https://github.com/sgl-project/sglang/issues/2889
**State**: closed
**Created**: 2025-01-14T13:06:12+00:00
**Closed**: 2025-01-14T13:18:47+00:00
**Comments**: 0

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

I have tested Deepseek-v2 on SGlang 0.3.5 , Recent, I test this model performance on SGlang 0.4.1.post5 again, and I found the MLA kernel (__fwd_kernel)  faster in Prefill phase.
But there is no change with the __fwd_kernel triton operator  . how does this kernel faster, or there are some differences on benchmark testcase?

**

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Cannot run `microsoft/Phi-3.5-mini-instruct`; Capture cuda graph failed

**Link**: https://github.com/sgl-project/sglang/issues/1751
**State**: closed
**Created**: 2024-10-22T06:33:41+00:00
**Closed**: 2024-10-22T07:01:39+00:00
**Comments**: 4

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

When running `microsoft/Phi-3.5-mini-instruct` on 1x H100, sglang gives the following error. 

```
Exception: Capture cuda graph failed: BatchDecodeWithPagedKVCachePyTorchWrapper::Plan(at::Tensor, at::Tensor, at::Tensor, at::Tensor, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, float, at::Tensor, at

[... truncated for brevity ...]

---

## Issue #N/A: CUDA out of memory for H100 80GB for lmms-lab/llama3-llava-next-8b

**Link**: https://github.com/sgl-project/sglang/issues/465
**State**: closed
**Created**: 2024-05-23T20:35:06+00:00
**Closed**: 2024-05-23T20:42:44+00:00
**Comments**: 4

### Description

Installed via pip in python 3.10 as readme says, then ran:
```
export CUDA_VISIBLE_DEVICES=1
python -m sglang.launch_server --model-path lmms-lab/llama3-llava-next-8b --tokenizer-path lmms-lab/llama3-llava-next-8b-tokenizer --port=30000 --host="0.0.0.0" --tp-size=1 --api-key='62224bfb-c832-4452-81e7-8a4bdabbe164'  --random-seed=1234 --context-length=8192
```

nothing is on GPU=1, only GPU=0 is filled.

Always hit very early on startup, model not even loaded yet:
```
  File "/home/ubuntu/miniconda3/envs/sglang/lib/python3.10/site-packages/sglang/srt/models/llama2.py", line 39, in __init__
    self.gate_up_proj = MergedColumnParallelLinear(
  File "/home/ubuntu/miniconda3/envs/sglang/lib/python3.10/site-packages/vllm/model_executor/layers/linear.py", line 333, in __init__
    super().__init__(input_size, sum(output_sizes), bias, gather_output,
  File "/home/ubuntu/miniconda3/envs/sglang/lib/python3.10/site-packages/vllm/model_executor/layers/linear.py", line 236, in __init_

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] it seems memory leak in sglang when longtime serving

**Link**: https://github.com/sgl-project/sglang/issues/1358
**State**: closed
**Created**: 2024-09-09T12:12:45+00:00
**Closed**: 2024-09-09T12:14:07+00:00
**Comments**: 4

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

128134 memory blocks: 5104.2 KiB
Traceback:
  File "/usr/local/lib/python3.10/dist-packages/zmq/_future.py", line 374
    loaded = load(buf)

### Reproduction

tracemalloc in detokenizer_manager.py

### Environment

llama2-13B A800*1

---

## Issue #N/A: [Bug] sglang with 2nodes, failed with `RuntimeError: CUDA error: invalid device ordinal`

**Link**: https://github.com/sgl-project/sglang/issues/3470
**State**: closed
**Created**: 2025-02-10T13:19:04+00:00
**Closed**: 2025-02-10T13:21:29+00:00
**Comments**: 6

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

tl;dr

distributed sglang with ` --dist-init-addr` failed:` RuntimeError: CUDA error: invalid device ordinal`
```
python3 -m sglang.launch_server --model-path /model --tp 16 --dist-init-addr  $podIP:20000 --nnodes 2 --node-rank 0 --trust-remote-code

```




### Reproduction


----------

(1)run single sglang , all was  well 

```
        

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]  PD h20 decode node start error because of cuda graph....

**Link**: https://github.com/sgl-project/sglang/issues/5650
**State**: closed
**Created**: 2025-04-23T01:46:45+00:00
**Closed**: 2025-04-23T02:14:53+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

when i  using th  fzyzcjy's branch  ， feat/dev_branch , decode starts error . 
<img width="1097" alt="Image" src="https://github.com/user-attachments/assets/dac66897-ef96-4cdd-ae2c-9b911fa0c074" />

### Reproduction

2 manchine prefill node  4machine decoe node..
 fzyzcjy's branch  ， feat/dev_branch 

### Environment

2 manchine prefill no

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Not compatible with langchain, Seems sglang Message Class's fields not provided by langchain (via openai) app

**Link**: https://github.com/sgl-project/sglang/issues/3394
**State**: closed
**Created**: 2025-02-08T06:45:16+00:00
**Closed**: 2025-02-08T07:21:32+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

![Image](https://github.com/user-attachments/assets/e5dff345-0f5b-444f-9aa9-6783b37aa43c)

### Related resources

_No response_

---

## Issue #N/A: [Bug] 我想问一下支持qwen1.5-14B模型吗？

**Link**: https://github.com/sgl-project/sglang/issues/930
**State**: closed
**Created**: 2024-08-05T08:49:45+00:00
**Closed**: 2024-08-05T09:03:52+00:00
**Comments**: 1

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

[Bug] 我想问一下支持qwen1.5-14B模型吗？

### Reproduction

[Bug] 我想问一下支持qwen1.5-14B模型吗？

### Environment

```Shell
[Bug] 我想问一下支持qwen1.5-14B模型吗？
```


---

## Issue #N/A: [Bug] deepseek v3 2 nodes h100 segmentation fault

**Link**: https://github.com/sgl-project/sglang/issues/3283
**State**: closed
**Created**: 2025-02-04T06:43:27+00:00
**Closed**: 2025-02-04T07:40:25+00:00
**Comments**: 6
**Labels**: help wanted, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

hello.
I run on 2 nodes of 8 x h100 using   lmsysorg/sglang:v0.4.2.post1-cu125 image

```
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1 --tp 16 --dist-init-addr 172.16.1.68:5000 --nnodes 2 --node-rank 1 --trust-remote-code --quantization fp8 --kv-cache-dtype fp8_e5m2
```
I start a benchmark
```
 python3 -m sglang.ben

[... truncated for brevity ...]

---

## Issue #N/A: [OAI Server Refactor] [ChatCompletions & Completions] Add UTs for Tool Call and Reasoning Text Handling

**Link**: https://github.com/sgl-project/sglang/issues/7261
**State**: closed
**Created**: 2025-06-17T04:29:31+00:00
**Closed**: 2025-06-17T04:36:53+00:00
**Comments**: 2

### Description

**Points**: 1-2 days

**Description**: Current the tool call handling and reasoning text logic in [`serving_chat.py`](https://github.com/sgl-project/sglang/blob/70c471a868bf505fadbfe0a041e7637a91db0365/python/sglang/srt/entrypoints/openai/serving_chat.py)

**Deliverables:**
- [ ] Complete tasks below

---

## Issue #N/A: Support most : Batch, Chat, Completions, Embedding

**Link**: https://github.com/sgl-project/sglang/issues/986
**State**: closed
**Created**: 2024-08-08T05:58:18+00:00
**Closed**: 2024-08-08T05:58:32+00:00
**Comments**: 0

### Description

No description provided.

---

## Issue #N/A: [Bug] Gemma-2-9b-it produces garbage output

**Link**: https://github.com/sgl-project/sglang/issues/1160
**State**: closed
**Created**: 2024-08-20T07:18:26+00:00
**Closed**: 2024-08-20T07:56:50+00:00
**Comments**: 4

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

Gemma-2-9b-it: "<unused99><unused99><unused99><unused99><unused99>..."


### Reproduction

I just pulled the latest Docker image and deployed Gemma-2-9b-it [AWQ version](https://huggingface.co/nihaomur/gemma-2-9b-it-AWQ). Here is how I deployed the model

```shell
HF_TOKEN=<my-token>
MODEL=nihaomur/gemma-2-9b-it-AWQ
SER

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] llava not working after pull latest code

**Link**: https://github.com/sgl-project/sglang/issues/1130
**State**: closed
**Created**: 2024-08-16T19:45:13+00:00
**Closed**: 2024-08-16T20:07:41+00:00
**Comments**: 2

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

  File "/u/miniconda3/lib/python3.11/site-packages/sglang/srt/managers/tp_worker.py", line 441, in forward_prefill_batch
    batch.prepare_for_extend(self.model_config.vocab_size)
TypeError: ScheduleBatch.prepare_for_extend() missing 1 required positional argument: 'int_token_logit_bias'

Killed

### Reproduction

python -m sglang.laun

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Can it support Huawei Ascend 910B backend?

**Link**: https://github.com/sgl-project/sglang/issues/3609
**State**: closed
**Created**: 2025-02-16T12:19:40+00:00
**Closed**: 2025-02-16T12:45:57+00:00
**Comments**: 2

### Description

### Motivation

It seems that SGLang currently does not support deploy LLM-infer models such as DeepSeek-V3/R1 on Huawei 910B.

### Related resources

_No response_

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

## Issue #N/A: [Bug] Decode OOM due to wrong new page estimation

**Link**: https://github.com/sgl-project/sglang/issues/7411
**State**: closed
**Created**: 2025-06-21T07:25:54+00:00
**Closed**: 2025-06-21T07:35:27+00:00
**Comments**: 1
**Labels**: high priority

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Related
#7328 
#7410 

### Reproduction

```
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 40000 --mem-fraction-static=0.5 --page=32
```
```
git switch xiezhq-dev
cd  benchmark/hicache
python bench_multiturn.py --port 40000
```

### Environment

- 

---

## Issue #N/A: [Bug] SGLang router/server is unkillable

**Link**: https://github.com/sgl-project/sglang/issues/7870
**State**: closed
**Created**: 2025-07-08T20:02:06+00:00
**Closed**: 2025-07-08T20:03:46+00:00
**Comments**: 0

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I cannot kill the router after starting it up with ctrl + c or ctrl + d. I see that the timeout is 300 seconds but how I configure it to be shorter/longer?
```text
^C^C^C2025-07-08 19:58:04  INFO sglang_router_rs::router: src/router.rs:354: Worker is not ready yet
2025-07-08 19:58:04  INFO sglang_router_rs::router: src/router.rs:354: Worke

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] After updating from 0.4.5.post2 to 0.4.5.post3, the following error is reported: AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'awq_dequantize'

**Link**: https://github.com/sgl-project/sglang/issues/5668
**State**: closed
**Created**: 2025-04-23T08:29:35+00:00
**Closed**: 2025-04-23T08:36:34+00:00
**Comments**: 3

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

[2025-04-23 16:14:34 TP0] Attention backend not set. Use triton backend by default.
[2025-04-23 16:14:34 TP0] Init torch distributed begin.
[W423 16:14:34.122823038 HIPAllocatorConfig.h:29] Warning: expandable_segments not supported on this platform (function operator())
[2025-04-23 16:14:35 TP0] Init torch distributed ends. mem usage=0.00

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Post link to paper in the README.md

**Link**: https://github.com/sgl-project/sglang/issues/946
**State**: closed
**Created**: 2024-08-06T07:32:34+00:00
**Closed**: 2024-08-06T07:48:48+00:00
**Comments**: 2

### Description

### Motivation

New users struggle to form a mental model of how sglang works, for example understanding how many rounds of interaction take place for a structured prompt, will the chat history be re-sent to the model and have to pay for the prefix again, and what works and what doesn't work with OpenAI vs local server?

I suggest you post the link to the paper "Efficiently Programming Large Language Models using SGLang" in the README.md as it covers most questions

### Related resources

https://arxiv.org/pdf/2312.07104v1

---

