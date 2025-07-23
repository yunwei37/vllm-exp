# speculative-decoding - issues

**Total Issues**: 24
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 14
- Closed Issues: 10

### Label Distribution

- speculative-decoding: 24 issues
- inactive: 8 issues
- high priority: 5 issues
- collaboration: 2 issues
- deepseek: 2 issues
- quant: 1 issues

---

## Issue #N/A: [Bug] Error when running Qwen2 EAGLE spec decoding with the official OFFLINE inference example

**Link**: https://github.com/sgl-project/sglang/issues/7263
**State**: open
**Created**: 2025-06-17T05:37:21+00:00
**Comments**: 1
**Labels**: speculative-decoding

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hi, I am running eagle offline speculative decoding example (examples/runtime/engine/offline_batch_inference_eagle.py), and I encountered an error as shown below. Specifically, I modified the target model to Qwen/Qwen2-7B-Instruct and draft model to yuhuili/EAGLE-Qwen2-7B-Instruct.
I did notice there was a previous [issue](https://github.c

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support EAGLE3 for Llama4

**Link**: https://github.com/sgl-project/sglang/issues/7185
**State**: open
**Created**: 2025-06-14T15:50:19+00:00
**Comments**: 1
**Labels**: speculative-decoding

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Support EAGLE3 model for Llama4, e.g. https://huggingface.co/nvidia/Llama-4-Maverick-17B-128E-Eagle3.

### Related resources

_No response_

---

## Issue #N/A: [Bug] Run eagle3 failed

**Link**: https://github.com/sgl-project/sglang/issues/7139
**State**: open
**Created**: 2025-06-13T02:05:16+00:00
**Comments**: 0
**Labels**: speculative-decoding

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

/pytorch/aten/src/ATen/native/cuda/Indexing.cu:1500: indexSelectSmallIndex: block: [32,0,0], thread: [63,0,0] Assertion `srcIndex < srcSelectDimSize` failed.
[2025-06-13 09:59:00] Scheduler hit an exception: Traceback (most recent call last):
  File "/hy-tmp/sglang-0.4.7/python/sglang/srt/managers/scheduler.py", line 2506, in run_scheduler

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] token_to_kv_pool_allocator memory leak detected! when page_size>1 and use MTP

**Link**: https://github.com/sgl-project/sglang/issues/7130
**State**: open
**Created**: 2025-06-12T11:13:56+00:00
**Comments**: 0
**Labels**: speculative-decoding

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

In version v0.4.6.post5, if I deploy the model using MTP and set page_size greater than 1, an error will occur during inference: token_to_kv_pool_allocator memory leak detected!

It seems that this is a common problem, which should be related to the operation process of eagle_worker. There is a similar bug report, not sure where the proble

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Eagle memory leak

**Link**: https://github.com/sgl-project/sglang/issues/7111
**State**: open
**Created**: 2025-06-12T03:01:47+00:00
**Comments**: 1
**Labels**: speculative-decoding

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

![Image](https://github.com/user-attachments/assets/6708fe70-a919-49ce-a338-e8318a2d34a0)

### Reproduction

CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server --model-path /local/path/qwen2.5-7b-instruct --host 0.0.0.0 --port 6800 --speculative-algorithm EAGLE --speculative-draft-model-path /localpath/qwen2.5-7b-instruct-draft/checkpoi

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] integrate MTP with some new features

**Link**: https://github.com/sgl-project/sglang/issues/7077
**State**: open
**Created**: 2025-06-11T03:33:03+00:00
**Comments**: 3
**Labels**: high priority, collaboration, deepseek, speculative-decoding

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

- [x] compatibility with dp attention #6081 
- [x] compatibility with eplb
- [x] compatibility with `enable-dp-lm-head`
- [x] compatibility with pd disaggregation @Atream #7242 
- [x] compatibility with two-batch-overlap @Qiaolin-Yu #7225 
- [x] compatibility with deepep #7206 
...

### Related resources

https://github.com/sgl-project/sglang/issues/6017
https://lmsys.org/blog/2025-05-05-large-scale-ep/#large-scale-expert-parallelism

---

## Issue #N/A: [Feature] Support Eagle Utils using Triton.

**Link**: https://github.com/sgl-project/sglang/issues/7050
**State**: open
**Created**: 2025-06-10T10:33:27+00:00
**Comments**: 1
**Labels**: speculative-decoding

### Description



### Motivation

These kernels are only implemented in CUDA, which leads to some eagle cases currently unable to run on AMD. We need to support the kernels of PyTorch or Triton.
```
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
        verify_tree_greedy,
    )
```

---

## Issue #N/A: [Bug] Performance Regression: Eagle Speculative Decoding Causes Significant Inference Slowdown

**Link**: https://github.com/sgl-project/sglang/issues/6949
**State**: open
**Created**: 2025-06-07T14:04:12+00:00
**Comments**: 18
**Labels**: speculative-decoding

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

target model : qwq-32b

without eagle:
```
[2025-06-07 21:25:14 TP0] Decode batch. #running-req: 1, #token: 9322, token usage: 0.00, gen throughput (token/s): 97.52, #queue-req: 0
[2025-06-07 21:25:14 TP0] Decode batch. #running-req: 1, #token: 9362, token usage: 0.00, gen throughput (token/s): 97.58, #queue-req: 0
[2025-06-07 21:25:15 TP0

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] FA3 + EAGLE2: speculative_token_map not supported

**Link**: https://github.com/sgl-project/sglang/issues/6863
**State**: closed
**Created**: 2025-06-04T08:14:37+00:00
**Closed**: 2025-06-27T21:00:23+00:00
**Comments**: 0
**Labels**: speculative-decoding

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

We conducted a large-scale load test with speculative decoding using EAGLE2. According to our results, enabling speculative_token_map effectively reduces overhead for larger batch sizes. However, this feature only works with the flashinfer backend, while the FA3 backend does not support it.

Here is our graph demonstrating this:

+FR is ru

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] EAGLE3 perform worse with sequence length larger than the draft model context window

**Link**: https://github.com/sgl-project/sglang/issues/6783
**State**: open
**Created**: 2025-05-30T21:03:19+00:00
**Comments**: 4
**Labels**: speculative-decoding

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I am testing EAGLE3 for Llama 3.3 70B model, using [lmsys/sglang-EAGLE3-LLaMA3.3-Instruct-70B](https://huggingface.co/lmsys/sglang-EAGLE3-LLaMA3.3-Instruct-70B), the EAGLE3 heads has `max_position_embeddings`=2048. I observed that when input sequence is > 2k, the performance is worse than the baseline when speculative decoding is not enabl

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] qwen2 eagle startup failed

**Link**: https://github.com/sgl-project/sglang/issues/6618
**State**: open
**Created**: 2025-05-26T09:12:43+00:00
**Comments**: 2
**Labels**: speculative-decoding

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

@Lzhang-hub @libratiger  Failed startup when using qwen2 eagle
```
ERROR 16:58:08 scheduler.py:2626] Scheduler hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 2611, in run_scheduler_process
    scheduler.event_loop_normal()
  File "/usr/local/lib/python3.10/d

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] eagle2【CUDA error: an illegal memory access was encountered】

**Link**: https://github.com/sgl-project/sglang/issues/6309
**State**: open
**Created**: 2025-05-15T03:41:17+00:00
**Comments**: 11
**Labels**: speculative-decoding

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I was running eagle2 online（based on sglang=0.4.5，sgl-kernel=0.0.5 ，flashinfer-python=0.2.5）, and after a while, I encountered an cuda error saying "an illegal memory access was encountered". Here are the details of the error. Could you help me take a look?

---------------------
File "/root/python/betelgeuse/srt/managers/scheduler.py", li

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support EAGLE-3 for speculative decoding on DeepSeek model

**Link**: https://github.com/sgl-project/sglang/issues/6268
**State**: open
**Created**: 2025-05-13T12:41:36+00:00
**Comments**: 7
**Labels**: speculative-decoding

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

EAGLE-3 appears to provide a higher speculative decoding acceptance rate to improve output throughput. 
For specific code generation scenarios, we found that focusing on multi-step losses when training the draft model can effectively improve the acceptance rate.

### Related resources

_No response_

---

## Issue #N/A: [Feature] optimize eagle speculative decoding

**Link**: https://github.com/sgl-project/sglang/issues/5924
**State**: closed
**Created**: 2025-04-30T17:07:56+00:00
**Closed**: 2025-07-01T00:22:50+00:00
**Comments**: 1
**Labels**: high priority, inactive, speculative-decoding

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

- [ ] optimize top k 1 @merrymercy @zhyncs @Fridge003 @ispobock @Alcanderian 
- [ ] support draft extend cuda graph (flashinfer @merrymercy fa3 @hebiao064 @qingquansong )
- [ ] support schedule overlap with speculative decoding @merrymercy 
- [ ] minor improvement with profiling @Fridge003 @zhyncs @ispobock @Alcanderian 

### Related resources

_No response_

---

## Issue #N/A: [Bug] CUDA Error Eagle2 + mixed_chunk

**Link**: https://github.com/sgl-project/sglang/issues/5886
**State**: open
**Created**: 2025-04-29T12:39:41+00:00
**Comments**: 0
**Labels**: speculative-decoding

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hello! I faced with some problem when tried to use EAGLE2 with `--enable-mixed-chunk` option. The base model I use is `Qwen2.5-Coder-32B-Instruct-FP8`. The problem occurs at least at versions v0.4.6 and earlier.

```
[2025-04-29 09:06:521 INFO:
172.23.0.1:55488 - "POST /generate HTTP/1.1" 200 OK
[2025-04-29 09:06:52] INFO:
172.23.0.1:55554

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

## Issue #N/A: [Feature] support merge_state in sgl-kernel

**Link**: https://github.com/sgl-project/sglang/issues/5361
**State**: closed
**Created**: 2025-04-14T00:56:57+00:00
**Closed**: 2025-04-15T04:32:18+00:00
**Comments**: 3
**Labels**: high priority, collaboration, speculative-decoding

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

I have talked to @deftruth, and he will support it in the sgl-kernel today

### Related resources

_No response_

---

## Issue #N/A: Speculative Decoding Fails with AWQ Quantized Model

**Link**: https://github.com/sgl-project/sglang/issues/4351
**State**: closed
**Created**: 2025-03-12T21:30:32+00:00
**Closed**: 2025-05-13T00:19:02+00:00
**Comments**: 2
**Labels**: inactive, quant, speculative-decoding

### Description

Description:

I am facing an issue when using speculative decoding with an AWQ quantized model in sglang. The same configuration works fine with an unquantized model (Llama-3.1-8b-Instruct), but fails when I switch to an AWQ quantized model(Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF-AWQ-INT4).

Setup:

GPUs: NVIDIA L40s (48GB VRAM) x 2
CUDA Version: 12.8
PyTorch Version: 2.5.1
sglang Version: 0.4.3.post2

Working Configuration (Unquantized Model):

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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

if is_in_ci():
    from sglang.docs.frontend.patch import launch_server_cmd
else:
    from sglang.utils import launch_serve

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] --speculative-token-map error when TP>1

**Link**: https://github.com/sgl-project/sglang/issues/4328
**State**: closed
**Created**: 2025-03-12T05:19:15+00:00
**Closed**: 2025-05-18T00:20:48+00:00
**Comments**: 4
**Labels**: inactive, speculative-decoding

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

based on https://github.com/thunlp/FR-Spec/issues/1, we found that when TP > 1,
the code in https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/eagle_worker.py#L91-L96
```
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        if self.hot_token_id is not None:
            head = h

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

## Issue #N/A: [Feature] [Eagle] Are there any plans to support the feature for batching prefill?

**Link**: https://github.com/sgl-project/sglang/issues/3736
**State**: closed
**Created**: 2025-02-21T02:20:15+00:00
**Closed**: 2025-04-23T00:18:34+00:00
**Comments**: 2
**Labels**: inactive, speculative-decoding

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Now, I find the qps of sglang is very lower than baseline when use Eagle, I think the reason for this problem maybe is that sglang doesn't support batch prefill yet. So Do you have any plans to support this feature? 



### Related resources
![Image](https://github.com/user-attachments/assets/20272e3b-14b1-4064-a832-563e4d90b280)
_No response_

---

## Issue #N/A: [Bug] Eagle mtbench benchmark error

**Link**: https://github.com/sgl-project/sglang/issues/3662
**State**: closed
**Created**: 2025-02-18T09:05:34+00:00
**Closed**: 2025-04-21T00:19:30+00:00
**Comments**: 3
**Labels**: inactive, speculative-decoding

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

**I am evaluating the performance for Eagle and I found that the server will report an error when using the mtbench benchmark.
It is worth noting that I performed well in version v0.4.3 (e0b9a423c8413c486f8e6a2c168cd3e6e7a74589), but encountered issues during subsequent commits.**

[2025-02-18 09:03:54 TP0] Prefill batch. #new-seq: 1, #new

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Eagle fail on Llama3-8b

**Link**: https://github.com/sgl-project/sglang/issues/3574
**State**: open
**Created**: 2025-02-14T08:34:13+00:00
**Comments**: 4
**Labels**: speculative-decoding

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hi team. I'm testing eagle. The base model is Meta-Llama-3-8B-Instruct and eagle draft model is sglang-EAGLE-LLaMA3-Instruct-8B. The issue relates to max_position_embeddings, which is 2048 in eagle draft config. But in my case context len will be larger. 
I can start sglang server but crash when processing requests.

gpu: A100 80G
docker i

[... truncated for brevity ...]

---

## Issue #N/A: run eagle speculative decodeing error!

**Link**: https://github.com/sgl-project/sglang/issues/3362
**State**: closed
**Created**: 2025-02-07T04:56:49+00:00
**Closed**: 2025-05-02T00:18:42+00:00
**Comments**: 9
**Labels**: inactive, speculative-decoding

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

![Image](https://github.com/user-attachments/assets/3f69af85-9c68-44e9-85db-cbf712ffb2fa)

commanf line:
CUDA_VISIBLE_DEVICES=0,1,3,7 python -m sglang.launch_server --model /mnt/nvme0n1/ckpt/llama/Meta-Llama-3.1-70B-Instruct --port 9001 --host 0.0.0.0 --tensor-parallel-size 4 --speculative-algo EAGLE --speculative-draft /mnt/nvme0n1/ckpt/l

[... truncated for brevity ...]

---

