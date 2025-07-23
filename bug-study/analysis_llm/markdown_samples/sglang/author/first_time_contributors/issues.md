# first_time_contributors - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 9
- Closed Issues: 21

### Label Distribution

- inactive: 8 issues
- deepseek: 2 issues
- await-response: 1 issues
- MLLM: 1 issues
- quant: 1 issues
- good first issue: 1 issues

---

## Issue #N/A: [Feature] Interrupt running requests when updating weights for RL

**Link**: https://github.com/sgl-project/sglang/issues/6486
**State**: open
**Created**: 2025-05-21T06:41:21+00:00
**Comments**: 5

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Hi SGLang community,

I'm using SGLang to build an asynchronous RL system for LLM reasoning. To obtain a high generation throughput, the inference server is continuously filled with new requests, so when a new parameter arrives, SGLang should either wait for the current longest sequence or interrupt all ongoing requests and continue generation with the latest parameter. It's straightforward that the later case results in better hardware utilization.

By "interrupting requests", I mean returning the unfinished requests back to the client. The client will submit it again. Although it's also possible to pause the requests within SGLang, the implementation may be more complicated.

Implementation-wise, I plan to add an

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] cannot load prequantized model with scalar weight scale

**Link**: https://github.com/sgl-project/sglang/issues/4594
**State**: closed
**Created**: 2025-03-19T20:44:35+00:00
**Closed**: 2025-03-22T07:47:54+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Right now after loading the model and converting the weight scale to channel wise, there's an implicit assumption that the weight scale tensors in model weight is 1-D tensor. This is not the case for modelopt-quantized FP8 in fp8 cutlass supported hardware, since QKVParalleLinear will go through a requantization to the same scale.

### Rep

[... truncated for brevity ...]

---

## Issue #N/A: "GET / HTTP/1.1" 404 Not Found

**Link**: https://github.com/sgl-project/sglang/issues/2468
**State**: closed
**Created**: 2024-12-12T17:54:02+00:00
**Closed**: 2025-01-17T15:13:00+00:00
**Comments**: 8

### Description

I try to follow your "quick start" and launch a server, with following code:

```
python -m sglang.launch_server --model-path mistralai/Mistral-7B-Instruct-v0.1 \
--port 30000 --host 0.0.0.0
```

Unfortunately I encounter some error😭

```
[2024-12-13 01:44:59 TP0] Load weight end. type=MistralForCausalLM, dtype=torch.bfloat16, avail mem=9.69 GB
[2024-12-13 01:44:59 TP0] Memory pool end. avail mem=2.44 GB
[2024-12-13 01:44:59 TP0] Capture cuda graph begin. This can take up to several minutes.
[2024-12-13 01:45:01 TP0] Capture cuda graph end. Time elapsed: 2.19 s
[2024-12-13 01:45:02 TP0] max_total_num_tokens=56524, max_prefill_tokens=16384, max_running_requests=2049, context_len=32768
[2024-12-13 01:45:02] INFO:     Started server process [3831493]
[2024-12-13 01:45:02] INFO:     Waiting for application startup.
[2024-12-13 01:45:02] INFO:     Application startup complete.
[2024-12-13 01:45:02] INFO:     Uvicorn running on http://0.0.0.0:30000 (Press CTRL+C to quit)
[

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Multi options

**Link**: https://github.com/sgl-project/sglang/issues/1761
**State**: closed
**Created**: 2024-10-23T01:55:58+00:00
**Closed**: 2024-12-23T00:17:19+00:00
**Comments**: 5
**Labels**: inactive

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

I want to use sgl.gen to select multiple options from the candidate selection, does it support it?

### Related resources

_No response_

---

## Issue #N/A: [Bug] H20 8 gpu x 2 with --enable-dp-attention occurred CUDA error: an illegal memory access

**Link**: https://github.com/sgl-project/sglang/issues/3892
**State**: closed
**Created**: 2025-02-26T14:00:36+00:00
**Closed**: 2025-02-27T03:31:04+00:00
**Comments**: 7

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I use two H20 with 8 gpus and 8 IB device(mlx5_1 to mlx5_8)  on each node to test DeepSeep-R1 with --enable-dp-attention.

Error log 

[2025-02-26 13:18:19 DP3 TP3] Prefill batch. #new-seq: 1, #new-token: 4096, #cached-token: 0, cache hit rate: 0.00%, token usage: 0.00, #running-req: 0, #queue-req: 7
[2025-02-26 13:18:19 DP0 TP0] Prefill b

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] min_p_sampling_from_probs() related crashes

**Link**: https://github.com/sgl-project/sglang/issues/3201
**State**: closed
**Created**: 2025-01-29T02:46:51+00:00
**Closed**: 2025-02-02T19:50:45+00:00
**Comments**: 2

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

sglang srt crashes with log:
```
[2025-01-28 17:58:17 TP0] TpModelWorkerClient hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker_overlap_thread.py", line 109, in forward_thread_func
    self.forward_thread_func_()
  File "/usr/local/lib/python3.10/dist-packages/torch/uti

[... truncated for brevity ...]

---

## Issue #N/A: Inference Llama3-70b has an AssertionError

**Link**: https://github.com/sgl-project/sglang/issues/929
**State**: closed
**Created**: 2024-08-05T08:18:19+00:00
**Closed**: 2024-09-22T13:04:11+00:00
**Comments**: 1

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

![image](https://github.com/user-attachments/assets/ddf162df-5f4c-4aa7-929a-f093aa672328)
I run sglang.server, but get an assertion error.

### Reproduction

`python3 -m sglang.launch_server --model-path ./models/Meta-Llama-3-70B-Instruct --host 0.0.0.0 --port 30000 --tp 8 --mem-fraction-static 0.7 --chunked-prefill-size 8192`

### Environment

```Shell
Python: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0]
CUDA available: True
GPU 0,1,2,3,4,5,6,7: NVIDIA A40
CUDA_HOME: /usr/local/cuda
NVCC: Cuda compilation tools, release 12.3, V12.3.107
CUDA Driver Version: 535.1

[... truncated for brevity ...]

---

## Issue #N/A: Why can't I see NCCL logs even though I set NCCL_DEBUG=INFO? And how can I see NCCL run logs?

**Link**: https://github.com/sgl-project/sglang/issues/3810
**State**: closed
**Created**: 2025-02-24T08:38:32+00:00
**Closed**: 2025-04-27T00:20:04+00:00
**Comments**: 3
**Labels**: inactive

### Description

<img width="1636" alt="Image" src="https://github.com/user-attachments/assets/fb9d89e9-2822-44ee-beb3-6fe271ea9030" />

---

## Issue #N/A: unable to install uvloop in windows (dependency)

**Link**: https://github.com/sgl-project/sglang/issues/152
**State**: closed
**Created**: 2024-02-06T18:02:46+00:00
**Closed**: 2024-07-25T06:32:08+00:00
**Comments**: 3
**Labels**: inactive

### Description

Getting this error

pip install sglang[all]
Requirement already satisfied: sglang[all] in c:\users\keshav s\anaconda3\envs\llm-workspace\lib\site-packages (0.1.11)
Requirement already satisfied: requests in c:\users\keshav s\anaconda3\envs\llm-workspace\lib\site-packages (from sglang[all]) (2.31.0)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\keshav s\anaconda3\envs\llm-workspace\lib\site-packages (from requests->sglang[all]) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in c:\users\keshav s\anaconda3\envs\llm-workspace\lib\site-packages (from requests->sglang[all]) (3.6)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\keshav s\anaconda3\envs\llm-workspace\lib\site-packages (from requests->sglang[all]) (2.1.0)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\keshav s\anaconda3\envs\llm-workspace\lib\site-packages (from requests->sglang[all]) (2023.11.17)
Collecting anthropic (from sglang[all])
  Using cached anthropic-0.

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] sglang Engine can not work with async ray actor

**Link**: https://github.com/sgl-project/sglang/issues/6723
**State**: open
**Created**: 2025-05-29T02:59:14+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

The method async_generate in engine is designed to use with async method, but it can not works in ray actor.

This problem blocks server-based rollout in https://github.com/volcengine/verl/issues/1721 


You can reproduce it using script from https://gist.github.com/chenhaiq/a28560a53701d869dd08dd0852d9b379, and output below error:


```
I

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]  No module named 'cuda.bindings'

**Link**: https://github.com/sgl-project/sglang/issues/6363
**State**: open
**Created**: 2025-05-17T00:46:46+00:00
**Comments**: 3

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

python3  -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct
/usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py:105: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
Traceback (most recent call last):
  File "/usr/lib/pytho

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Fail to build from Docker

**Link**: https://github.com/sgl-project/sglang/issues/885
**State**: closed
**Created**: 2024-08-02T06:10:22+00:00
**Closed**: 2024-08-06T09:47:09+00:00
**Comments**: 3
**Labels**: await-response

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

Building from the Dockerfile, got gpg-key error like:

```
Err:2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease
  The following signatures couldn't be verified because the public key is not available: NO_PUBKEY A4B469963BF863CC
```

while executing:

```
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common \
    && add-apt-reposito

[... truncated for brevity ...]

---

## Issue #N/A: [Router] How to ensure that the kvcache of worker and router are consistent when use cacheaware poliy

**Link**: https://github.com/sgl-project/sglang/issues/7773
**State**: open
**Created**: 2025-07-04T08:32:16+00:00
**Comments**: 0

### Description

I've noticed that when a new request arrives, the radix tree inserts text. If it reaches the maximum size, the tree deletes some nodes. However, I don't see the worker publishing kv events to the route, which is necessary to maintain consistency between the worker and the route and ensure more accurate kv matching. If I missed anything, please point out my mistakes. 

---

## Issue #N/A: [Bug] requests.exceptions.JSONDecodeError: 

**Link**: https://github.com/sgl-project/sglang/issues/1386
**State**: closed
**Created**: 2024-09-11T06:47:58+00:00
**Closed**: 2024-09-11T12:37:21+00:00
**Comments**: 6

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

After installing the newest version, it encouters a bug: when I launch a sglang run, it raises two exceptions in a few seconds.
```
(vllm) user@node14:~$  python -m sglang.launch_server --model-path models/Qwen2-72B-Instruct --port 30000 --tp 8
[14:39:01] server_args=ServerArgs(model_path='models/Qwen2-72B-Instruct', tokenizer_path='mod

[... truncated for brevity ...]

---

## Issue #N/A: Addition of Support of IDEFICS2

**Link**: https://github.com/sgl-project/sglang/issues/506
**State**: closed
**Created**: 2024-06-05T11:18:05+00:00
**Closed**: 2024-08-05T01:05:12+00:00
**Comments**: 1
**Labels**: inactive

### Description

Hi 

Was wondering if sglang could support the IDEFICS 2 model as well.

---

## Issue #N/A: [Bug] bench_serving.py error in hicache folder

**Link**: https://github.com/sgl-project/sglang/issues/7614
**State**: open
**Created**: 2025-06-28T03:31:35+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hi all, when I tried to run bench_serving in the benchmark/hicache folder, I encountered the problem that indicates my prompt is out of range. Could you please tell me the reason I'm encountering the problem and how I should fix it.

My model is Qwen3-235B-FP8, my bash script command is:

`python3 bench_serving.py --backend sglang --datase

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Only one Worker active when using sglang_router.launch_server on a single machine with multiple GPUs

**Link**: https://github.com/sgl-project/sglang/issues/5528
**State**: open
**Created**: 2025-04-18T10:42:36+00:00
**Comments**: 11

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

On a single machine equipped with 4× NVIDIA L20Y (80GB) GPUs, when launching SGLang using the built-in router via:
`python -m sglang_router.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --dp 4
`
and sending multiple concurrent chat completion requests using multi-threading, only one worker appears to be actively handl

[... truncated for brevity ...]

---

## Issue #N/A: Expert selection distribution capture results difference between ranks

**Link**: https://github.com/sgl-project/sglang/issues/5275
**State**: closed
**Created**: 2025-04-11T05:32:45+00:00
**Closed**: 2025-06-15T00:22:00+00:00
**Comments**: 3
**Labels**: inactive

### Description

Hi, 

I was capturing expert selection distribution using `start_expert_distribution_record` and dump the results using `dump_expert_distribution_record` endpoint. There are multiple csv files generated, one for each rank. Each file has a distribution for all the experts in all layers, but the distributions are different. I wonder is this an expected behavior or not and how should we interpret the result. Below is a truncated example of the output file. I also attached three of the output csv files.

Example:

> expert_distribution_rank0_timestamp1744347750.3721356.csv
```
layer_id,expert_id,count
3,0,24990
3,18,39249
3,29,13378
3,34,13874
3,43,22360
```

> expert_distribution_rank1_timestamp1744347750.006448.csv
```
layer_id,expert_id,count
3,18,33056
3,29,14053
3,34,14732
3,43,20474
```

[expert_distribution_rank0_timestamp1744347750.3721356.csv](https://github.com/user-attachments/files/19699341/expert_distribution_rank0_timestamp1744347750.3721356.csv)
[expert_distribution_rank1_ti

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] In version 0.4.7 (including post1), there is an approximately 20-second stall when returning the first cached token during inference.

**Link**: https://github.com/sgl-project/sglang/issues/7339
**State**: open
**Created**: 2025-06-19T03:11:38+00:00
**Comments**: 4

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

Version 0.4.6 is ok. 
![Image](https://github.com/user-attachments/assets/119cd407-48f9-4382-b85d-3e901f456a33)
At the stall time, the cpu are busy on comiling something:
$ps -aux |grep cicc
root        1413  0.0  0.0   2796  1104 ?        S    02:51   0:00 /bin/dash -c -- "$CICC_PATH/cicc" --c++17 --gnu_version=130200 --display_error_numb

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Integrate DeepEP into SGLang  Multi node case is failed  on H100

**Link**: https://github.com/sgl-project/sglang/issues/4653
**State**: closed
**Created**: 2025-03-21T16:03:25+00:00
**Closed**: 2025-05-22T00:19:11+00:00
**Comments**: 2
**Labels**: inactive

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug




![Image](https://github.com/user-attachments/assets/b3a8015f-b10f-4728-9338-14156410f38d)

### Reproduction

python3 -m sglang.launch_server --model-path model--trust-remote-code   --tp 16 --dp 16  --dist-init-addr ip  --nnodes 2 --node-rank 0   --enable-dp-attention --enable-deepep-moe   --disable-cuda-graph

python3 -m sglang.launch_s

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Key conflict of `AutoImageProcessor.register`

**Link**: https://github.com/sgl-project/sglang/issues/4159
**State**: closed
**Created**: 2025-03-07T04:06:18+00:00
**Closed**: 2025-03-25T12:17:44+00:00
**Comments**: 22
**Labels**: MLLM

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

The following ValueError was raised when attempting to serve any model within a recent Docker container:

`Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] deepseek-r1 occasionally crash

**Link**: https://github.com/sgl-project/sglang/issues/4017
**State**: closed
**Created**: 2025-03-03T08:07:18+00:00
**Closed**: 2025-05-03T00:18:12+00:00
**Comments**: 1
**Labels**: inactive, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

```
[2025-02-28 09:28:25 TP7] Scheduler hit an exception: Traceback (most recent call last):                                                                                                                                                                 │
│   File "/usr/local/lib/python3.10/dist-packages/sglang/srt/managers/scheduler.py", l

[... truncated for brevity ...]

---

## Issue #N/A: RuntimeError: NCCL error: unhandled system error

**Link**: https://github.com/sgl-project/sglang/issues/4289
**State**: closed
**Created**: 2025-03-11T07:08:29+00:00
**Closed**: 2025-03-11T08:20:23+00:00
**Comments**: 7

### Description

# step
I successfully ran the DeepSeek-R1-Distill-Qwen-32B on 4*L40S.

`python -m sglang.launch_server --model-path /home/clouduser/work/models/DeepSeek-R1-Distill-Qwen-32B --mem-fraction-static 0.85 --tp 4`

However, I encountered an error when running it with 8*L40S. 
`python -m sglang.launch_server --model-path /home/clouduser/work/models/DeepSeek-R1-Distill-Qwen-32B --mem-fraction-static 0.85 --tp 8`

# logs

`INFO 03-11 15:02:39 __init__.py:190] Automatically detected platform cuda.
[2025-03-11 15:02:44] server_args=ServerArgs(model_path='/home/clouduser/work/models/DeepSeek-R1-Distill-Qwen-32B', tokenizer_path='/home/clouduser/work/models/DeepSeek-R1-Distill-Qwen-32B', tokenizer_mode='auto', skip_tokenizer_init=False, load_format='auto', trust_remote_code=False, dtype='auto', kv_cache_dtype='auto', quantization=None, quantization_param_path=None, context_length=None, device='cuda', served_model_name='DeepSeek-R1-Distill-Qwen-32B', chat_template=None, is_embedding=False, revision=

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] cuda kernel illegal and  GPTQMarlinMoEMethod.apply() got an unexpected keyword argument 'correction_bias'

**Link**: https://github.com/sgl-project/sglang/issues/4083
**State**: closed
**Created**: 2025-03-05T06:47:58+00:00
**Closed**: 2025-03-16T07:43:21+00:00
**Comments**: 10
**Labels**: quant, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I test the model(OPEA/DeepSeek-R1-int4-gptq-sym-inc from HF) with the excellent sglang and two nodes(2 x 8X80G A100). 

COMMANDS:
**python3 -m sglang.launch_server --model-path /data/LM/hf/DeepSeek-R1-int4-gptq-sym-inc --tp 16 --dist-init-addr xx.yy.zz.210:25000 --nnodes 2 --node-rank 0 --trust-remote-code --host 0.0.0.0 --port 30000** 
an

[... truncated for brevity ...]

---

## Issue #N/A: [WIP] [Roadmap] Supporting Ascend NPU on 2025 H2

**Link**: https://github.com/sgl-project/sglang/issues/8004
**State**: open
**Created**: 2025-07-14T03:17:24+00:00
**Comments**: 0

### Description

# SGLang NPU support on 2025 H2

During 2025 H1, we have contributed initial supports for NPU ([#3853](https://github.com/sgl-project/sglang/pull/3853), [#7022](https://github.com/sgl-project/sglang/pull/7022)), which make it possible for users to run SGLang on NPU hardware.

Our goal on 2025 H2 is to provide a seamless running experience on NPUs, and here is a rough development roadmap:

## CI on NPU hardware

- [ ] [**_July_**] Enable autoscaling runners #7935 
- [ ] E2E/unittest test coverage

## Model support

*We will start with supporting the hotest models*

- [ ] [**_July_**] DeepseekV2 / V3 family
- [ ] [**_July_**] Qwen3 family
- [ ] [**_July_**] Qwen3-MoE family

## User / Developer experience

*User experience is also to be taken into our consideration, containers and documents will be provided soon*

- [ ] [**_July_**] Docker image
- [ ] [**_July_**] Docs (Quickstart / Installation / tutorials…)

## Performance Enhancement

### Attention Backend

- [x] [**_July_**] Ascend A

[... truncated for brevity ...]

---

## Issue #N/A: sglang-0.4.3.post3 deepseek-r1 set temperature=0，but output is inconsistent

**Link**: https://github.com/sgl-project/sglang/issues/4999
**State**: closed
**Created**: 2025-04-02T13:30:44+00:00
**Closed**: 2025-06-06T00:19:22+00:00
**Comments**: 2
**Labels**: inactive

### Description

```
sglang                            0.4.3.post3
vllm                              0.7.2
transformers                      4.48.3
nvidia-cublas-cu12                12.4.5.8
nvidia-cuda-cupti-cu12            12.4.127
nvidia-cuda-nvrtc-cu12            12.4.127
nvidia-cuda-runtime-cu12          12.4.127
nvidia-cudnn-cu12                 9.1.0.70
nvidia-cufft-cu12                 11.2.1.3
nvidia-curand-cu12                10.3.5.147
nvidia-cusolver-cu12              11.6.1.9
nvidia-cusparse-cu12              12.3.1.170
nvidia-ml-py                      12.570.86
nvidia-nccl-cu12                  2.20.5
nvidia-nvjitlink-cu12             12.4.127
nvidia-nvtx-cu12                  12.4.127
torch                             2.5.1
torchao                           0.9.0
torchaudio                        2.5.1
torchvision                       0.20.1
flashinfer-python                 0.2.2.post1
```
Deploy DeepSeek-R1 on 2*H20 (cuda-12.1). Input prompt is a long text (30000+ tokens). Set temper

[... truncated for brevity ...]

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

## Issue #N/A: how to use the finetuned mistral model for inference with sglang

**Link**: https://github.com/sgl-project/sglang/issues/94
**State**: closed
**Created**: 2024-01-24T16:14:41+00:00
**Closed**: 2024-01-30T14:25:14+00:00
**Comments**: 2

### Description

how to use the finetuned mistral model for inference with sglang. 
Please share the code for this

---

## Issue #N/A: [Feature] Support for GPT-2

**Link**: https://github.com/sgl-project/sglang/issues/1643
**State**: closed
**Created**: 2024-10-12T04:00:51+00:00
**Closed**: 2024-11-01T03:54:32+00:00
**Comments**: 5
**Labels**: good first issue

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

I believe there is still a vast majority of our community working with GPT-2 or similar small language models for niche datasets. It would be great if these finetuned GPT-2 models can be deployed through sglang. 

### Related resources

_No response_

---

## Issue #N/A: [Bug] DeepSeek-V3 function call return stop instead of tool_calls in streaming request

**Link**: https://github.com/sgl-project/sglang/issues/7934
**State**: open
**Created**: 2025-07-10T18:28:23+00:00
**Comments**: 0

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

When utilizing the DeepSeek-V3 function call in streaming mode, the finish reason for the last chunk ought to be tool calls, but currently, it shows as stop.

I think the reason is in servering_chat.py, whether to change stop to tool calls is determined by whether the current parse result contains any tool. If the last chunk is EOS, the pa

[... truncated for brevity ...]

---

