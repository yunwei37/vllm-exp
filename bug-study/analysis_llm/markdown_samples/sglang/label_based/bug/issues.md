# bug - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 5
- Closed Issues: 25

### Label Distribution

- bug: 30 issues
- high priority: 8 issues
- inactive: 4 issues
- lora: 3 issues
- good first issue: 1 issues
- help wanted: 1 issues
- quant: 1 issues
- await-response: 1 issues

---

## Issue #N/A: [Bug] use Eagle with speculative-num-steps=1

**Link**: https://github.com/sgl-project/sglang/issues/3762
**State**: closed
**Created**: 2025-02-21T13:32:09+00:00
**Closed**: 2025-04-24T00:18:22+00:00
**Comments**: 3
**Labels**: bug, inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When I attempted to use the Triton backend for Eagle to launch the Qwen-7B model, the process failed.
```
Traceback (most recent call last):
  File "/data/csl/project/sglang/python/sglang/srt/managers/scheduler.py", line 1827, in run_scheduler_process
    scheduler.event_loop_normal()
  File "/data/csl/miniconda3/envs/sglang/lib/python3.10

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Llama4 OOM with 400k input request

**Link**: https://github.com/sgl-project/sglang/issues/5212
**State**: closed
**Created**: 2025-04-10T00:22:53+00:00
**Closed**: 2025-04-11T08:24:15+00:00
**Comments**: 0
**Labels**: bug, high priority

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I started a server on 8xH100 with `meta-llama/Llama-4-Scout-17B-16E-Instruct` with the following command:

```
python sglang.launch_server --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
--port 8080 \
--tp-size 8 \
--chat-template llama-4 \
--attention-backend=fa3 \
--mem-fraction-static=0.8 \
--context-length 1000000 
```

Then sent a

[... truncated for brevity ...]

---

## Issue #N/A: upgrade setuptools and wheel if you found "torch module not found" when installing

**Link**: https://github.com/sgl-project/sglang/issues/2554
**State**: closed
**Created**: 2024-12-23T04:02:50+00:00
**Closed**: 2025-01-30T17:37:49+00:00
**Comments**: 7
**Labels**: bug

### Description

I encountered an issue while installing `sglang`. After upgrading pip (`pip install --upgrade pip`), I ran:

```bash
pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/
```

But it failed with the error:  
`ModuleNotFoundError: No module named 'torch'`.

I found on the Flash Attention GitHub that running this solved the issue:  
```bash
python -m pip install --upgrade pip wheel setuptools
```

It worked for me, so sharing in case someone faces the same problem! I don't know what the exact reason is though as the error itself was pretty strange. 

---

## Issue #N/A: [Bug] OOM for concurrent long requests

**Link**: https://github.com/sgl-project/sglang/issues/1030
**State**: closed
**Created**: 2024-08-11T10:51:49+00:00
**Closed**: 2024-09-22T13:00:44+00:00
**Comments**: 8
**Labels**: bug

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.

### Describe the bug

I am trying to benchmark inference of llama3-8b with long requests, I send **20** concurrent requests each with length of **1k tokens** and I set the **stream to True** and **max_tokens to 1024.** 


This is how I start the server:
`python -m sglang.launch_server --model-path NousResearch/Meta-Llama-3-8B-Instruct  --host 0.0.0.0  --port 8000 --context-length 4096 --dtype bfloat16  --chat-temp

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Tensor shape is wrong when cudagraph+enable_dp_attention

**Link**: https://github.com/sgl-project/sglang/issues/7951
**State**: open
**Created**: 2025-07-11T10:58:55+00:00
**Comments**: 5
**Labels**: bug, high priority

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

I tried to run DSR1 fp4 model on 8xB200, but found that some issue when I opened cudagraph and attndp, the input tensor dimension for each MoE layer is padded to global bs. For example, I take global bs 4096 and attention dp 8, which each rank should have 512 reqs for decode and the input tensor M dimension should be 512 for local rank. 
B

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] 0.0.0.0 host not supported

**Link**: https://github.com/sgl-project/sglang/issues/4935
**State**: closed
**Created**: 2025-03-30T22:04:01+00:00
**Closed**: 2025-05-30T08:43:39+00:00
**Comments**: 3
**Labels**: bug, inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

if we set the host as 0.0.0.0, the warmup function will fail.

<b>Connection to 0.0.0.0 failed.</b></p>\n</blockquote>\n\n<p id="sysmsg">The system returned: <i>(111) Connection refused</I>

_wait_and_warmup -> res = requests.get(url + "/get_model_info", timeout=5, headers=headers)

can we add a fix here to replace 0.0.0.0 with 127.0.0.1 f

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

## Issue #N/A: [Bug] device-side assert triggered when using run_batch

**Link**: https://github.com/sgl-project/sglang/issues/1279
**State**: closed
**Created**: 2024-09-01T01:51:32+00:00
**Closed**: 2024-09-03T13:02:03+00:00
**Comments**: 6
**Labels**: bug

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

The following error is raised when ever i run run_batch:

```
../aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [1193,0,0], thread: [124,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [1193,0,0], thread: [125,0,

[... truncated for brevity ...]

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

## Issue #N/A: [Bug] Assertion error: Exception in ModelTpServer: This happens when we do return_log_prob=True

**Link**: https://github.com/sgl-project/sglang/issues/747
**State**: closed
**Created**: 2024-07-26T13:12:31+00:00
**Closed**: 2024-07-28T02:15:16+00:00
**Comments**: 3
**Labels**: bug

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

When I have given the argument return_log_probe=True. Its showing the below message. I am getting this in sagemaker environment the instance type is 

### Reproduction

INFO:     127.0.0.1:59618 - "POST /generate/ HTTP/1.1" 307 Temporary Redirect
[gpu_id=0] Prefill batch. #new-seq: 1, #new-token: 2, #cached-token: 42, cache hit rate: 45.26%, #running-req: 0, #queue-req: 0
Exception in ModelTpServer:
Traceback (most recent call last):
  File "/opt/conda/lib/python3.10/site-packages/sglang/srt/managers/controller/tp_worker.py", line 186, in exposed_step
    self.forward_s

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Running DeepSeek V2.5 error when enable torch-compile

**Link**: https://github.com/sgl-project/sglang/issues/4497
**State**: closed
**Created**: 2025-03-17T08:23:31+00:00
**Closed**: 2025-03-18T04:42:56+00:00
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

I use the latest main ([d1112d8548eb13c842900b3a8d622345f9737759](https://github.com/sgl-project/sglang/commit/d1112d8548eb13c842900b3a8d622345f9737759)), start the DeepSeek V2.5 bf16 model, and when using the `--enable-torch-compile` parameter, an error is report.

### Reproduction

```
python3 -m sglang.launch_server --model /path/to/Dee

[... truncated for brevity ...]

---

## Issue #N/A: Not able to run AWQ Mixtral on 4xA10

**Link**: https://github.com/sgl-project/sglang/issues/77
**State**: closed
**Created**: 2024-01-22T18:23:22+00:00
**Closed**: 2024-02-22T14:58:11+00:00
**Comments**: 4
**Labels**: bug

### Description

Hi,

Im trying to run the AWQ version of Mixtral on 4xA10s. However im getting this error. Ive also tried with `--mem-frac 0.7` and still got the same error

Model I'm using : https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ

Command : `python -m sglang.launch_server --model-path /local_disk0/TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ/ --port 30000 --tp 4`

Code : 
```
from sglang import function, system, user, assistant, gen
import sglang as sgl

@function
def multi_turn_question(s, question_1, question_2):
    s += system("You are a helpful assistant.")
    s += user(question_1)
    s += assistant(gen("answer_1", max_tokens=256))
    s += user(question_2)
    s += assistant(gen("answer_2", max_tokens=256))

state = multi_turn_question.run(
    question_1="What is the capital of the United Kingdom?",
    question_2="List two local attractions.",
    temperature=0.7,
    stream=True,
)

for out in state.text_iter():
    print(out, end="", flush=T

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] FusedMoE compatible with vllm 0.6.3.post1

**Link**: https://github.com/sgl-project/sglang/issues/2160
**State**: closed
**Created**: 2024-11-24T13:38:13+00:00
**Closed**: 2024-11-24T14:37:05+00:00
**Comments**: 0
**Labels**: bug

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

N/A

### Reproduction

N/A

### Environment

N/A

---

## Issue #N/A: [Bug] load microsoft/MAI-DS-R1 error: KeyError: 'model.layers.3.mlp.shared_experts.down_proj.weight_scale'

**Link**: https://github.com/sgl-project/sglang/issues/6592
**State**: open
**Created**: 2025-05-25T13:37:42+00:00
**Comments**: 0
**Labels**: bug

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

I used the following shell to load the microsoft/MAI-DS-R1 model, but the following error occurred：**KeyError: 'model.layers.3.mlp.shared_experts.down_proj.weight_scale'**

```shell
#!/bin/bash
set -x

IMAGE_NAME=lmsysorg/sglang:latest

port=xxxx
name=mai_ds_r1_sglang_multinode1
ranks=0

docker run --gpus all -d \
    --shm-size 32g \
    

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] test_lora.py bug

**Link**: https://github.com/sgl-project/sglang/issues/7062
**State**: closed
**Created**: 2025-06-10T18:10:20+00:00
**Closed**: 2025-06-28T04:28:35+00:00
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

There is a prompt "AI is a field of computer science focused on" in `test/srt/models/lora/test_lora.py ` that can easily break CI, which might be caused some internal bug of lora.

We remove this prompt temporarily in #7061. It should be added back after this bug is fixed.

### Reproduction

Uncomment line 49 of `test_lora.py`

![Image](ht

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

## Issue #N/A: [Bug] Logprobs overflow to -3.4e+38

**Link**: https://github.com/sgl-project/sglang/issues/4876
**State**: closed
**Created**: 2025-03-29T04:53:32+00:00
**Closed**: 2025-06-03T00:19:53+00:00
**Comments**: 4
**Labels**: bug, inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

![Image](https://github.com/user-attachments/assets/b3567c95-1206-4ef5-aeea-de21ee71f0d3)

logprobs overflow to the maximum negative value of fp32

### Reproduction

I'm using Qwen2.5-14B-Instruct

command:

```python
sampling_params = {
    "temperature": 0.9,
    "top_p": 0.9,
    "skip_special_tokens": False,
    "stop": "<|im_end|>",
}

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Why can't I use multi-lora adapter and radix attention together?

**Link**: https://github.com/sgl-project/sglang/issues/2880
**State**: open
**Created**: 2025-01-14T07:03:52+00:00
**Comments**: 5
**Labels**: bug, lora

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

Why can't I use multi-lora adapter and radix attention together?
If I have multi-lora adapters, why not just insert the ID of the LoRA adapter before the first token?

When using a multi-lora adapter, it is extremely slow because radix attention cannot be used.

### Reproduction

https://github.com/sgl-project/sglang/blob/v0.4.1.post5/p

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Unable to fix model output

**Link**: https://github.com/sgl-project/sglang/issues/1316
**State**: closed
**Created**: 2024-09-03T11:01:15+00:00
**Closed**: 2024-11-01T04:13:00+00:00
**Comments**: 25
**Labels**: bug, high priority

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

The performance of sglang is very good. I am comparing the output accuracy of vllm, Hugging Face, and sglang. Using Qwen's model, I set do_sample to false or temperature to 0 to fix the output. Through comparison, the outputs of vllm and the Hugging Face transformer library are consistent. However, sglang does not produce consistent output

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] JSON output contains think tag when enabling MTP for DeepSeek-R1

**Link**: https://github.com/sgl-project/sglang/issues/6441
**State**: closed
**Created**: 2025-05-20T02:25:21+00:00
**Closed**: 2025-05-22T00:18:42+00:00
**Comments**: 3
**Labels**: bug, high priority

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

When enable MTP + structured output for DeepSeek-R1, the JSON output contains `</think>`. It's unexpected.

### Reproduction


```
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-R1 --tp 8 --trust-remote-code --port 30000 --speculative-algorithm EAGLE

curl -X POST "http://127.0.0.1:30000/v1/chat/completions" -H "Content-Type:

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Bad outputs with fp8 quantization at high RPS

**Link**: https://github.com/sgl-project/sglang/issues/1195
**State**: closed
**Created**: 2024-08-24T12:03:11+00:00
**Closed**: 2024-09-21T03:18:34+00:00
**Comments**: 13
**Labels**: bug

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

I ran a RPS benchmark script with prompts of an average input length of 1600 tokens and got bad outputs as the RPS increased. For example:

`*给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给给追给给给给给给给给迫你。`

It seems to be related to quantization and concurrent requests. I've listed some commands below with 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Chat completions logprobs support

**Link**: https://github.com/sgl-project/sglang/issues/839
**State**: closed
**Created**: 2024-07-30T20:24:44+00:00
**Closed**: 2024-08-01T07:08:22+00:00
**Comments**: 2
**Labels**: bug

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

I'm encountering an issue when making a request to the `v1/chat/completions` endpoint with the `"logprobs": true` parameter. The `choices.logprobs` field in the response is always null, even though it should be populated with probability values.

### Reproduction

Script to launch the SGLang server:

```bash
python -m sglang.launch_server  \
    --model-path Qwen/Qwen2-7B-Instruct \
    --port 8000
```

Script to reproduce the request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "Qwen/Q

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Miss prompt_tokens_details in stream chat when --enable-cache-report

**Link**: https://github.com/sgl-project/sglang/issues/4707
**State**: closed
**Created**: 2025-03-24T04:02:42+00:00
**Closed**: 2025-03-24T05:32:13+00:00
**Comments**: 2
**Labels**: bug

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

python3 -m sglang.launch_server --model-path /root/public/DeepSeek-R1-Distill-Qwen-1.5B/ --enable-cache-report

use  "stream": true,

there is no 

"prompt_tokens_details":{"cached_tokens":921}

in output

### Reproduction

python3 -m sglang.launch_server --model-path /root/public/DeepSeek-R1-Distill-Qwen-1.5B/ --enable-cache-report

curl 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] port is not an integer in function get_open_port()

**Link**: https://github.com/sgl-project/sglang/issues/4527
**State**: closed
**Created**: 2025-03-18T02:05:24+00:00
**Closed**: 2025-03-28T04:46:07+00:00
**Comments**: 1
**Labels**: bug

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

in function get_open_port，port is a string, so s.bind(("", port)) will have TypeError: an integer is required (got type str)
to fix it, simple add port=int(port) here
```
def get_open_port() -> int:
    port = os.getenv("SGLANG_PORT")
    if port is not None:
        # port=int(port)  # add here
        while True:
            try:
       

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] FusedMoE does not recognize ModelOpt fp8 format.

**Link**: https://github.com/sgl-project/sglang/issues/6714
**State**: open
**Created**: 2025-05-28T17:31:24+00:00
**Comments**: 1
**Labels**: bug

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I quantized a llama-4-fp8 version with modelopt: https://huggingface.co/baseten/Llama-4-Scout-17B-16E-fp8 
Currently other non-moe checkpoints are working https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8 



### Reproduction
-

### Environment

-

---

## Issue #N/A: [Bug] A100 PCIE torch compile error

**Link**: https://github.com/sgl-project/sglang/issues/1301
**State**: closed
**Created**: 2024-09-02T11:24:36+00:00
**Closed**: 2024-09-02T23:18:49+00:00
**Comments**: 3
**Labels**: bug

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

```
[11:19:46 TP0] Decode batch. #running-req: 36, #token: 14473, token usage: 0.03, gen throughput (token/s): 2283.73, #queue-req: 0
../aten/src/ATen/native/cuda/MultinomialKernel.cu:112: binarySearchForMultinomial: block: [0,31,0], thread: [0,0,0] Assertion `cumdist[size - 1] > static_cast<scalar_t>(0)` failed.
../aten/src/ATen/native

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Server crashes after loading (Mixtral 8x7b) on L4

**Link**: https://github.com/sgl-project/sglang/issues/1191
**State**: closed
**Created**: 2024-08-23T11:08:48+00:00
**Closed**: 2024-11-04T01:13:36+00:00
**Comments**: 10
**Labels**: bug, inactive

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

Model fully loads, server runs and then instantly crashes

```
server_args=ServerArgs(model_path='/local_disk0/mistralai/Mixtral-8x7B-Instruct-v0.1', tokenizer_path='/local_disk0/mistralai/Mixtral-8x7B-Instruct-v0.1', tokenizer_mode='auto', skip_tokenizer_init=False, load_format='auto', dtype='auto', trust_remote_code=False, context_len

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Missing tool name in answer

**Link**: https://github.com/sgl-project/sglang/issues/4700
**State**: closed
**Created**: 2025-03-23T15:28:24+00:00
**Closed**: 2025-03-28T05:23:31+00:00
**Comments**: 8
**Labels**: bug

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hello, i have found a strange BUG, i'm trying to use sglang with n8n and noticed that function calls not working, i have written a simple proxy app and noticed that in sglang answer with function call at final chunk function name in missing, and that is strange because in first chunks its present, and final answer becomes invalid as result

[... truncated for brevity ...]

---

## Issue #N/A: RuntimeError: TopKTopPSamplingFromProbs failed with error code no kernel image is available for execution on the device  已杀死[Bug] 

**Link**: https://github.com/sgl-project/sglang/issues/865
**State**: closed
**Created**: 2024-08-01T09:10:16+00:00
**Closed**: 2024-09-22T13:05:44+00:00
**Comments**: 4
**Labels**: bug, await-response

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

RuntimeError: TopKTopPSamplingFromProbs failed with error code no kernel image is available for execution on the device

已杀死

### Reproduction

RuntimeError: TopKTopPSamplingFromProbs failed with error code no kernel image is available for execution on the device

已杀死

### Environment

```Shell
RuntimeError: TopKTopPSamplingFromProbs failed with error code no kernel image is available for execution on the device

已杀死
```


---

