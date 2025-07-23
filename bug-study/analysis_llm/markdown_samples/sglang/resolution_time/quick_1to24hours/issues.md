# quick_1to24hours - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- bug: 1 issues
- lora: 1 issues
- await-response: 1 issues

---

## Issue #N/A: [Bug] Error when using select without stream mode

**Link**: https://github.com/sgl-project/sglang/issues/857
**State**: closed
**Created**: 2024-08-01T00:04:58+00:00
**Closed**: 2024-08-01T07:05:40+00:00
**Comments**: 0

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

When using select without stream mode, I get the following error. It seems like `_execute_select` does not check whether we are in stream mode before trying to access `stream_var_event`. A simple fix is to just add a check for whether `stream_var_event` is None.

```
UserWarning: Error in stream_executor: Traceback (most recent call last):
  File "**********/lib/python3.11/site-packages/sglang/lang/interpreter.py", line 327, in _thread_worker_func
    self._execute(expr)
  File "**********/lib/python3.11/site-packages/sglang/lang/interpreter.py", line 370, in _execute

[... truncated for brevity ...]

---

## Issue #N/A: Decode out of memory happened when run deepseek-r1 inference

**Link**: https://github.com/sgl-project/sglang/issues/4184
**State**: closed
**Created**: 2025-03-07T15:52:26+00:00
**Closed**: 2025-03-07T17:13:29+00:00
**Comments**: 1

### Description


```
[2025-03-07 21:49:33 TP13] Decode out of memory happened. #retracted_reqs: 1, #new_token_ratio: 0.1474 -> 0.1845
[2025-03-07 21:49:33 TP12] Decode out of memory happened. #retracted_reqs: 1, #new_token_ratio: 0.1474 -> 0.1845
[2025-03-07 21:49:33 TP10] Decode out of memory happened. #retracted_reqs: 1, #new_token_ratio: 0.1474 -> 0.1845
[2025-03-07 21:49:45 TP9] Decode out of memory happened. #retracted_reqs: 1, #new_token_ratio: 0.0980 -> 0.1919
[2025-03-07 21:49:45 TP11] Decode out of memory happened. #retracted_reqs: 1, #new_token_ratio: 0.0980 -> 0.1919
[2025-03-07 21:49:45 TP8] Decode out of memory happened. #retracted_reqs: 1, #new_token_ratio: 0.0980 -> 0.1919
[2025-03-07 21:49:45 TP12] Decode out of memory happened. #retracted_reqs: 1, #new_token_ratio: 0.0980 -> 0.1919
[2025-03-07 21:49:45 TP14] Decode out of memory happened. #retracted_reqs: 1, #new_token_ratio: 0.0980 -> 0.1919
[2025-03-07 21:49:45 TP13] Decode out of memory happened. #retracted_reqs: 1, #new_token_rati

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Server stuck with tp > 1

**Link**: https://github.com/sgl-project/sglang/issues/4257
**State**: closed
**Created**: 2025-03-10T08:58:49+00:00
**Closed**: 2025-03-10T19:40:23+00:00
**Comments**: 4

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I'm deploying DeepSeek-R1-Distill-Qwen-14B with single node and two gpu card. My problem is, when I deploy this model with single card, the server start up smoothly, but when I add --tp 2, the server got stuck at this logging line:
[2025-03-10 16:21:32 TP0] Prefill batch. #new-seq: 1, #new-token: 7, #cached-token: 0, token usage: 0.00, #ru

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] HF_Runner can't produce correct results after applying lora

**Link**: https://github.com/sgl-project/sglang/issues/5897
**State**: closed
**Created**: 2025-04-30T00:10:58+00:00
**Closed**: 2025-04-30T03:17:43+00:00
**Comments**: 2
**Labels**: bug, lora

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

First batch input lora_path as [a, a]. Second batch input lora_path as [None, None]. The second batch will be processed as if you had input lora_path as [a, a].

### Reproduction

```
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Directly importing Grafana JSON does not work

**Link**: https://github.com/sgl-project/sglang/issues/4050
**State**: closed
**Created**: 2025-03-04T06:20:49+00:00
**Closed**: 2025-03-04T11:44:52+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

If you directly import Grafana from https://github.com/sgl-project/sglang/blob/main/examples/monitoring/grafana.json, it will display the following error:

```
Failed to upgrade legacy queries Datasource aeboq3sqk89vkd was not found
```

![Image](https://github.com/user-attachments/assets/17b0d0bf-b3a3-4802-ac9d-2c0d5a547e50)

### Reproduc

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] 100% CPU Usage When Idle in sglang

**Link**: https://github.com/sgl-project/sglang/issues/1730
**State**: closed
**Created**: 2024-10-20T16:13:19+00:00
**Closed**: 2024-10-20T18:41:23+00:00
**Comments**: 2

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

## Issue Description

While running sglang with no active workload, I observed 100% CPU usage without any CUDA utilization. This issue appears to be caused by a non-blocking ZMQ receive operation in the `recv_requests` function of `scheduler.py`.

## Steps to Reproduce

1. Start sglang
2. Run `htop` to observe high CPU usage
3. Use

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] deepseek-R1 671b can not set tensor_parallel_size=32

**Link**: https://github.com/sgl-project/sglang/issues/3345
**State**: closed
**Created**: 2025-02-06T12:12:30+00:00
**Closed**: 2025-02-06T18:03:15+00:00
**Comments**: 17

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I used 4 nodes H100 * 8  TP32 to deploy the deepseek-R1 671B model，an error occurred. 
However, when I used 2 nodes H100 * 8 TP16 to deploy, the inference was normal. 
The following is the error message:

![Image](https://github.com/user-attachments/assets/2a9c3091-dc51-41b4-b54c-821e4f00d8bb)

### Reproduction

deepseek-R1 671B

# node 1


[... truncated for brevity ...]

---

## Issue #N/A: [Bug] disable_flashinfer didn't take effect

**Link**: https://github.com/sgl-project/sglang/issues/945
**State**: closed
**Created**: 2024-08-06T06:46:53+00:00
**Closed**: 2024-08-06T09:35:48+00:00
**Comments**: 10
**Labels**: await-response

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

I tried pip install flashinfer, but got error from
```
 File "/usr/local/python/lib/python3.10/site-packages/sglang/srt/model_executor/model_runner.py", line 28, in <module>
    from flashinfer import (
ModuleNotFoundError: No module named 'flashinfer'
```

According to readme, i tried to set disable_flashinfer when init the runtime, my code is like

```
self.runtime = sgl.Runtime(
            model_path = self.loader.target_path,
            tp_size = envs.TENSOR_PARALLEL_SIZE,
            trust_remote_code = True,
            max_num_reqs = 40,
            d

[... truncated for brevity ...]

---

## Issue #N/A: Custom chat template

**Link**: https://github.com/sgl-project/sglang/issues/40
**State**: closed
**Created**: 2024-01-18T15:37:27+00:00
**Closed**: 2024-01-18T22:02:09+00:00
**Comments**: 2

### Description

This would be useful when using a model like mistral-instruct, or any model that doesn't have a standardised template like chatml. Or for example using a finetuned model that uses a custom chat/instruct template.

---

## Issue #N/A: [Bug] PD Disaggregation benchmark hang after Decode out of memory happened

**Link**: https://github.com/sgl-project/sglang/issues/6857
**State**: closed
**Created**: 2025-06-04T03:44:36+00:00
**Closed**: 2025-06-04T08:20:13+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

sglang version: 0.4.6.post5
I run 1P1D in a node and run a benchmark, the benchmark hang after run some prompts.
After sometimes debug, I fond some request in prealloced queue but cannot be poped out by pop_preallocated, because these request meet: "if not decode_req.waiting_for_input: continue". But why these request decode_req.waiting_fo

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] When will pipeline model parallelism be supported?

**Link**: https://github.com/sgl-project/sglang/issues/4059
**State**: closed
**Created**: 2025-03-04T09:28:23+00:00
**Closed**: 2025-03-05T03:39:22+00:00
**Comments**: 2

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Hi, I see from the source code that sglang has left an entry for pipeline model parallelism https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/parallel_state.py#L1049. When will this feature be supported? Pipeline model parallelism should significantly reduce network communication costs, right?

### Related resources

_No response_

---

## Issue #N/A: [Bug] Have any suggestions for setting hyperparameters for inference acceleration？

**Link**: https://github.com/sgl-project/sglang/issues/2021
**State**: closed
**Created**: 2024-11-13T08:40:09+00:00
**Closed**: 2024-11-13T22:35:16+00:00
**Comments**: 1

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

now i run the model on the A800 80G *8, with qwen2.5-72b-instruct model, my gen throughput (token/s): 75.78, i think is too low, does some suggestions for speed up my interface speed? thanks !



### Reproduction

none

### Environment

none

---

## Issue #N/A: [Bug] Runtime Stuck

**Link**: https://github.com/sgl-project/sglang/issues/1173
**State**: closed
**Created**: 2024-08-21T09:04:49+00:00
**Closed**: 2024-08-21T12:09:47+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

```
import sglang as sgl


@sgl.function
def multi_turn_question(s, question_1, question_2):
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))


def single():
    state = multi_turn_questi

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]  Could not find a version that satisfies the requirement sgl-kernel>=0.0.3.post6

**Link**: https://github.com/sgl-project/sglang/issues/3715
**State**: closed
**Created**: 2025-02-20T02:55:38+00:00
**Closed**: 2025-02-20T05:32:16+00:00
**Comments**: 3

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

I use dockerfile to build the image, but I get an error:
```
359.9 INFO: pip is looking at multiple versions of sglang[srt] to determine which version is compatible with other requirements. This could take a while.
359.9 ERROR: Could not find a version that satisfies the requirement sgl-kernel>=0.0.3.post6; extra == "srt" (from sglang[srt]

[... truncated for brevity ...]

---

## Issue #N/A: why dp_size must be 1 for update weights from distributed / tensor? Is dp_size==1 also required for update_weights_from_disk?

**Link**: https://github.com/sgl-project/sglang/issues/4283
**State**: closed
**Created**: 2025-03-11T05:29:14+00:00
**Closed**: 2025-03-12T02:01:26+00:00
**Comments**: 2

### Description

No description provided.

---

## Issue #N/A: [Bug] ModuleNotFoundError: No module named 'datasets' on latest docker build

**Link**: https://github.com/sgl-project/sglang/issues/3525
**State**: closed
**Created**: 2025-02-12T13:09:13+00:00
**Closed**: 2025-02-12T19:26:12+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

when i usd docker i get this error 

```
inference_engine  | [2025-02-12 12:54:32] INFO:     127.0.0.1:57952 - "POST /v1/chat/completions HTTP/1.1" 200 OK
inference_engine  | [2025-02-12 12:54:32 TP0] Scheduler hit an exception: Traceback (most recent call last):
inference_engine  |   File "/sgl-workspace/sglang/python/sglang/srt/managers/

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Inaccurate or Inconsistent Output in Qwen2.5-VL Multi-Image Testing with sglang

**Link**: https://github.com/sgl-project/sglang/issues/4123
**State**: closed
**Created**: 2025-03-06T03:59:30+00:00
**Closed**: 2025-03-06T16:06:04+00:00
**Comments**: 3

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When testing multi-image input with `sglang` using the Qwen2.5-VL model, the output descriptions of the image contents are inaccurate or inconsistent. Compared to the output from directly calling the same model with `transformers`, the results from `sglang` show significant differences in describing the commonalities between multiple image

[... truncated for brevity ...]

---

## Issue #N/A: Error on json decoding with llava

**Link**: https://github.com/sgl-project/sglang/issues/138
**State**: closed
**Created**: 2024-02-03T23:12:32+00:00
**Closed**: 2024-02-04T02:46:17+00:00
**Comments**: 2

### Description

Encountered the following error when using the `regex` in call to `gen` method. Server doesn't work afterwards:

```console
~$ python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.5-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --port 30000
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Rank 0: load weight begin.
/opt/conda/envs/llava_test/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
INFO 02-03 23:02:55 weight_utils.py:164] Using model weights format 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] nsys profile failed

**Link**: https://github.com/sgl-project/sglang/issues/1076
**State**: closed
**Created**: 2024-08-13T10:54:45+00:00
**Closed**: 2024-08-14T08:42:00+00:00
**Comments**: 1

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.

### Describe the bug

nsys profile failed with nccl error.
![image](https://github.com/user-attachments/assets/daca6ae7-8efb-4b89-b675-683d0d48b9e0)


### Reproduction

```shell
python -m sglang.launch_server --port 8000 --model-path Qwen2-72B-FP8-Instruct --tp-size 8 --disable-radix-cache --mem-fraction-static 0.8 --disable-cuda-graph
```

### Environment

```shell
Python: 3.10.12 (main, Nov 20 2023, 15:14:05) [

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Qwen2.5-VL-7B OOM

**Link**: https://github.com/sgl-project/sglang/issues/7693
**State**: closed
**Created**: 2025-07-01T10:53:32+00:00
**Closed**: 2025-07-01T12:13:03+00:00
**Comments**: 2

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

server command:
```
python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8080  --chat-template qwen2-vl --chunked-prefill-size -1 --disable-radix-cache --mm-attention-backend fa3 --attention-backend fa3  --enable-torch-compile --cuda-graph-bs 80 --torch-compile-max-bs 80
```
bench command:
```
genai

[... truncated for brevity ...]

---

## Issue #N/A: [Request] Support for qwen1.5

**Link**: https://github.com/sgl-project/sglang/issues/939
**State**: closed
**Created**: 2024-08-05T15:40:52+00:00
**Closed**: 2024-08-05T18:00:58+00:00
**Comments**: 1

### Description

currently qwen1.5 is not supported. Thank you!

---

## Issue #N/A: GPU bubble between decode tokens when using cuda graph

**Link**: https://github.com/sgl-project/sglang/issues/5593
**State**: closed
**Created**: 2025-04-21T07:27:56+00:00
**Closed**: 2025-04-22T06:07:11+00:00
**Comments**: 2

### Description

### Description
When using CUDA Graph during decoding phase, we observed a GPU bubble (idle period) of around 2ms before the first kernel in the graph starts execution each time the graph is replayed.

![Image](https://github.com/user-attachments/assets/0325918f-e9ee-4c16-9525-fb39959038c1)

### Reproduction
- Model: Qwen2.5-32B-Instruct
- Params:
  - tp: 4
  - input_len: 4400
  - batch_size: 64
  - quantization: fp8

add nvtx range:
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/cuda_graph_runner.py
```python
...
def replay(
        self, forward_batch: ForwardBatch, skip_attn_backend_init: bool = False
    ) -> LogitsProcessorOutput:
        if not skip_attn_backend_init:
            with nvtx.annotate("replay_prepare", color="green"):
                self.replay_prepare(forward_batch)
        else:
            # In speculative decoding, these two fields are still needed.
            self.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)

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

## Issue #N/A: permission issue in newly updated docker lmsysorg/sglang:v0.4.3.post2-rocm630

**Link**: https://github.com/sgl-project/sglang/issues/4122
**State**: closed
**Created**: 2025-03-06T03:35:35+00:00
**Closed**: 2025-03-06T16:06:31+00:00
**Comments**: 5

### Description

**Description:**

I have been using the Docker image `lmsysorg/sglang:v0.4.3.post2-rocm630` for DeepSeek-R1 inference on a single AMD MI300X node (8 GPUs), and it has been stable until recently. After the Docker image was updated yesterday, my inference jobs started encountering permission-related errors, specifically failing during the initialization of `sgl.Engine`.

Relevant traceback:

```
  File "/scratch/amlt_code/sgl_inference.py", line 132, in main
    model = sgl.Engine(
            ^^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/api.py", line 43, in Engine
    from sglang.srt.entrypoints.engine import Engine
  File "/sgl-workspace/sglang/python/sglang/srt/entrypoints/engine.py", line 36, in <module>
    from sglang.srt.managers.data_parallel_controller import (
```

**Full Error Log:**  
[Detailed Log on GitHub](https://github.com/XingxingZhang/debug_only/blob/main/error_log_slg.txt)

**Potential cause:**  
This issue appears to have arisen after the recent Docker upd

[... truncated for brevity ...]

---

## Issue #N/A: How to contribute an optimized R1 operator in SGlang?

**Link**: https://github.com/sgl-project/sglang/issues/3816
**State**: closed
**Created**: 2025-02-24T11:19:43+00:00
**Closed**: 2025-02-24T18:20:27+00:00
**Comments**: 1

### Description

How to contribute an optimized R1 operator in SGlang?

For example, if I have an optimized R1-TopK, which subcomponents (e.g. vLLM/..) are the right place to get started?

---

## Issue #N/A: [Bug] ValueError: '<class 'sglang.srt.configs.qwen2_5_vl_config.Qwen2_5_VLConfig'>' is already used by a Transformers model.

**Link**: https://github.com/sgl-project/sglang/issues/4629
**State**: closed
**Created**: 2025-03-20T13:47:28+00:00
**Closed**: 2025-03-20T20:39:07+00:00
**Comments**: 7

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I use the latest docker image. 

```
Singularity> pip list|grep sglang
sglang                            0.4.4.post1          /sgl-workspace/sglang/python
```

```
Singularity> python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 32 --dist-init-addr 10.168.16.121:5000 --nnodes 4 --node-rank 0 --trust-remote-code --host 0.0.

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] nextn gen throughput gradually decreases over time

**Link**: https://github.com/sgl-project/sglang/issues/4286
**State**: closed
**Created**: 2025-03-11T06:25:31+00:00
**Closed**: 2025-03-11T09:19:38+00:00
**Comments**: 3

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

I deploy deepseek-r1 with 2 8*h20, commit id [df84ab2a](https://github.com/sgl-project/sglang/commit/df84ab2a5b87f4e8490049beb74fab6e67bbe3df),

command:
```
python3 -m sglang.launch_server --model-path $MODEL_PATH --tp 16 \
    --nccl-init-addr $SERVER_HOST:$SERVER_PORT --nnodes 2 --node-rank $rank --trust-remote-code --port 8000 --host 0

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] MoE Expert Parallel with awq 

**Link**: https://github.com/sgl-project/sglang/issues/2458
**State**: closed
**Created**: 2024-12-12T00:24:25+00:00
**Closed**: 2024-12-12T04:08:42+00:00
**Comments**: 0

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

I attempted to launch deepseek-awq using expert parallelism, but it appears to be unsupported. Is there a plan to support this feature in the future?

### Related resources

_No response_

---

## Issue #N/A: [Bug] Qwen2.5-VL-7B-Instruct-AWQ sglang accuracy

**Link**: https://github.com/sgl-project/sglang/issues/3884
**State**: closed
**Created**: 2025-02-26T09:39:06+00:00
**Closed**: 2025-02-26T17:16:28+00:00
**Comments**: 10

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

When performing inference on an image using the Qwen2.5-VL-7B-Instruct-AWQ model in sglang, the results differ from those obtained with vllm. Specifically, vllm provides accurate outputs, whereas sglang does not.

### Reproduction

代码为   

response = client.chat.completions.create(
        model='/models/Qwen2.5-VL-7B-Instruct-AWQ', # Mode

[... truncated for brevity ...]

---

## Issue #N/A: [BUG] Flashinfer 0.0.3 compat with Sglang

**Link**: https://github.com/sgl-project/sglang/issues/283
**State**: closed
**Created**: 2024-03-12T00:33:39+00:00
**Closed**: 2024-03-12T13:45:59+00:00
**Comments**: 4

### Description

Using flashinfer 0.0.3 requires one line change #282 but there is a compat issue where same model runs fine on 0.0.2 but under 0.0.3 throws an infinite loop of the following on sglang:

```
Exception in ModelRpcClient:
Traceback (most recent call last):
  File "/root/miniconda3/lib/python3.11/site-packages/sglang/srt/managers/router/model_rpc.py", line 184, in exposed_step
    self.forward_step()
  File "/root/miniconda3/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/lib/python3.11/site-packages/sglang/srt/managers/router/model_rpc.py", line 211, in forward_step
    self.forward_decode_batch(self.running_batch)
  File "/root/miniconda3/lib/python3.11/site-packages/sglang/srt/managers/router/model_rpc.py", line 505, in forward_decode_batch
    next_token_ids, _ = batch.sample(logits)
                        ^^^^^^^^^^^^^^^^^^^^
  File "/root/

[... truncated for brevity ...]

---

