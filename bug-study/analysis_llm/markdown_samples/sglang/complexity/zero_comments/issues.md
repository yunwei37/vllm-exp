# zero_comments - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 16
- Closed Issues: 14

### Label Distribution

- high priority: 4 issues
- collaboration: 2 issues
- documentation: 1 issues
- help wanted: 1 issues
- router: 1 issues

---

## Issue #N/A: Compile sgl-kernel warning log: (C7515) Potential Performance Loss: wgmma.mma_async instructions are serialized due to non wgmma instructions defining accumulator registers of a wgmma between start and end of the pipeline stage in the function

**Link**: https://github.com/sgl-project/sglang/issues/6453
**State**: open
**Created**: 2025-05-20T08:03:16+00:00
**Comments**: 0

### Description

**When compiling sgl-kernel, I encountered the following warning, do I need to pay attention?**


> ptxas info    : (C7515) Potential Performance Loss: wgmma.mma_async instructions are serialized due to non wgmma instructions defining accumulator registers of a wgmma between start and end of the pipeline stage in the function '_ZZN5flash25CollectiveMainloopFwdSm90ILi2EN4cute5tupleIJNS1_1CILi1EEES4_S4_EEENS2_IJNS3_ILi128EEENS3_ILi112EEENS3_ILi64EEEEEELi256EN7cutlass10bfloat16_tEfNSA_4arch4Sm90ELb0ELb1ELb1ELb1ELb0ELb0ELb1ELb1ELb0ELb1ELb0ELb0EE3mmaINS_16FlashAttnFwdSm90ISE_NS_21CollectiveEpilogueFwdINS2_IJS6_NS3_ILi256EEES7_EEES5_SB_SD_Li256ELb1ELb1ELb0ELb0EEENS_36VarlenDynamicPersistentTileSchedulerILi128ELi256ELi128ELb0ELb1ELb1EEEE13SharedStorageENS1_6TensorINS1_11ArrayEngineIfLm128EEENS1_6LayoutINS2_IJNS2_IJNS3_ILi2EEEST_NS3_ILi32EEEEEES4_S4_EEENS2_IJNS2_IJS4_ST_NS3_ILi4EEEEEENS3_ILi0EEESZ_EEEEEEENS_7SoftmaxILi2ELi0EEEEEbRKNSE_6ParamsENSA_25PipelineTmaAsyncNoClusterILi2ENSA_16PipelineT

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Docs: Patch Failed for engine

**Link**: https://github.com/sgl-project/sglang/issues/3770
**State**: closed
**Created**: 2025-02-21T17:19:40+00:00
**Closed**: 2025-02-21T21:30:52+00:00
**Comments**: 0
**Labels**: documentation, help wanted, high priority

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

```bash
(sglang) chayenne@lmsys:/home/misc/chayenne$ ipy
Python 3.11.7 (main, Dec 15 2023, 18:12:31) [GCC 11.2.0]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.32.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: # launch the offline engine
   ...: from sglang.utils import stream_and_merge, async_st

[... truncated for brevity ...]

---

## Issue #N/A: There is a question. If 10 rounds of windows are transmitted each time, can the benefits of the sglang kv pool be enjoyed

**Link**: https://github.com/sgl-project/sglang/issues/7311
**State**: open
**Created**: 2025-06-18T11:21:18+00:00
**Comments**: 0

### Description

There is a question. If 10 rounds of windows are transmitted each time, the first round of conversation must be lost for requests exceeding 10 rounds of windows. If so, can the benefits of the sglang kv pool be enjoyed? If not, is it recommended not to process any historical conversations? Is this the best way to use sglang in a production environment?

---

## Issue #N/A: [Usage] Some questions about the parameter --chunked-prefill-size

**Link**: https://github.com/sgl-project/sglang/issues/2814
**State**: closed
**Created**: 2025-01-09T11:49:19+00:00
**Closed**: 2025-01-09T12:01:26+00:00
**Comments**: 0

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

```
def budget_state(self):
        if self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0:
            return AddReqResult.NO_TOKEN

        if self.rem_input_tokens <= 0 or (
            self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0
        ):
            return AddReqResult.OTHER

        return AddReqRes

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Migrate support for FP8 in Ampere GPUs

**Link**: https://github.com/sgl-project/sglang/issues/7715
**State**: open
**Created**: 2025-07-02T12:31:27+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Hi,

FP8 native models and quantizations are getting more popular. Being an efficient way to deploy models.
The Ampere cards don't have native support for fp8, but their usage is very widespread, like the nvidia 3090. Many people have systems with them due to their low cost and 24GB of VRAM.

VLLM Added support for it with a Marlin kernel and fp8 models load just fine with decent speed on Ampere gpus:
https://github.com/vllm-project/vllm/pull/5975

Currently when trying to load a FP8 model with an Ampere GPU you get this error:
`ValueError("type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")`

### Related resources

_No response_

---

## Issue #N/A: [Feature] Maybe the content field in ChatCompletionMessageGenericParam should be Optional

**Link**: https://github.com/sgl-project/sglang/issues/8074
**State**: open
**Created**: 2025-07-16T01:57:02+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

According to the https://platform.openai.com/docs/api-reference/chat/create OpenAI Platform, when there is `tool_calls`, the `content` could be optional. But the current implementation could cause 400 Bad Request when `content` filed is missed under `tool_calls`  situation.
```
class ChatCompletionMessageGenericParam(BaseModel):
    role: Literal["system", "assistant", "tool"]
    content: Union[str, List[ChatCompletionMessageContentTextPart], None]
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = Field(default=None, examples=[None])

    @field_validator("role", mode="before")
    @classmethod
    def _norma

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Make random-range-ratio give symmetric distribution around --input-length (parity with vllm)

**Link**: https://github.com/sgl-project/sglang/issues/7253
**State**: open
**Created**: 2025-06-17T00:00:59+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Feature suggestion / request to change the way --random-range-ratio is used, as done in the vllm codebase 
[Fix range_ratio Bug in RandomDataset #16126](https://github.com/vllm-project/vllm/pull/16126)
 
There's another recent change at [[Bugfix] Fixed prompt length for random dataset](https://github.com/vllm-project/vllm/pull/17408/files#top) which may also be useful.

Some backstory: the syntax of --random-range-ratio looks identical in sglang and vllm, but the ranges in token lengths are quite different: 

sglang => [input_len * random_ratio, input_len]
vllm => [input_len * (1 - random_ratio), input_len * (1 + random_ratio)]

With a default of zero, this leads to sglang averaging half the input tokens for the sa

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Add examples for server token-in-token-out

**Link**: https://github.com/sgl-project/sglang/issues/4078
**State**: closed
**Created**: 2025-03-05T05:34:21+00:00
**Closed**: 2025-03-05T21:16:32+00:00
**Comments**: 0

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

https://github.com/sgl-project/sglang/pull/3941

follow this PR.

add it in `examples/token_in_token_out`.

@Qiaolin-Yu 

### Related resources

_No response_

---

## Issue #N/A: [Bug] difference of kv-cache-prefixing between vLLM and sglang

**Link**: https://github.com/sgl-project/sglang/issues/1664
**State**: closed
**Created**: 2024-10-14T06:16:19+00:00
**Closed**: 2024-10-14T14:50:09+00:00
**Comments**: 0

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

no bug. I am just wondering the difference of kv-cache-prefixing between vLLM impletention and SGLang implementation.

vLLM use hash to store and verify cached token:
<img width="318" alt="image" src="https://github.com/user-attachments/assets/c89b6dc1-101a-43de-b925-8ba1e45e6d20">
SGLang uses RadixAttention, so what is the d

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Error from xgrammar is not well processed by sglang

**Link**: https://github.com/sgl-project/sglang/issues/7903
**State**: open
**Created**: 2025-07-09T15:17:45+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

The error raised from xgrammar is not well processed and crashed sglang. 4 out of 600 requests kept running and the number of token below kept growing and then sglang crashed. The logs are attached as below (only logs from TP0 are retained for repeated logs from TP0 to TP7). The error raised by xgrammar is by design and maybe sglang should

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Add chat method support to the offline Engine class

**Link**: https://github.com/sgl-project/sglang/issues/8084
**State**: open
**Created**: 2025-07-16T07:37:45+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

`vLLM` now supports both the `generate` and `chat` methods for offline inference. The `generate` method offers more flexibility, while the `chat` method provides a more convenient interface for conversational use cases.

Currently, `SGLang` supports the `generate` method. Are there any plans to add support for the `chat` method as well? This feature would be helpful for users who want a streamlined conversational interface similar to that provided by vLLM.

### Related resources

[vLLM.LLM.chat](https://docs.vllm.ai/en/stable/api/vllm/index.html#vllm.LLM.chat)

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

## Issue #N/A: [RFC] DRI for every module

**Link**: https://github.com/sgl-project/sglang/issues/7851
**State**: open
**Created**: 2025-07-08T08:14:57+00:00
**Comments**: 0
**Labels**: high priority, collaboration

### Description

SGLang now has a wide range of features and modules, with many efforts happening in parallel across new features, optimizations, and bug fixes. Each module already has an internal DRI (Directly Responsible Individual), but these assignments haven’t been made public. As a result, some community pull requests have experienced delays, and contributors often don’t know who to reach out to.

We plan to make the DRI list public over the next two weeks and will actively follow up in a dedicated public channel. We're looking forward to working more closely with the community! Cheers!

---

## Issue #N/A: [Bug] Router Crashes Intermittently

**Link**: https://github.com/sgl-project/sglang/issues/6491
**State**: open
**Created**: 2025-05-21T07:51:23+00:00
**Comments**: 0
**Labels**: router

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

In our scenario, the router inexplicably crashes after a period of time (which could range from hours to weeks). Through the logs, I identified the following errors:
```
{"log":"[Router (Rust)] 2025-05-21 03:52:00 - DEBUG - starting new connection: http://192.168.0.99:8001/\n","stream":"stderr","time":"2025-05-21T03:52:00.40688383Z"}
{"log

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] DeepEP 8 + two-batch overlap was significantly lower than that of TP8

**Link**: https://github.com/sgl-project/sglang/issues/7255
**State**: open
**Created**: 2025-06-17T02:36:16+00:00
**Comments**: 0

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug


I deployed Qwen3-30B-A3B  on a single machine with 8 H20 GPUs with DeepEP + two-batch overlap.
My test case involved 512 input tokens and 1 output token, with concurrent of [1, 2, 4, 8, 16, 32].

I found that the performance of DeepEP + two-batch overlap was significantly lower than that of TP8. Specifically, the TTFT (Time To First Token

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] ValueError: No processor registered for architecture: ['Qwen2_5_VLForConditionalGeneration']. Registered architectures: ['CLIPModel', 'DeepseekVL2ForCausalLM']

**Link**: https://github.com/sgl-project/sglang/issues/7845
**State**: open
**Created**: 2025-07-08T06:32:19+00:00
**Comments**: 0

### Description


I'm using version 0.4.6.post1, but when I run the following code, I get a ValueError: No processor registered for architecture: ['Qwen2_5_VLForConditionalGeneration']. Registered architectures: ['CLIPModel', 'DeepseekVL2ForCausalLM']. The model I'm using is R1-Onevision/Qwen2.5VL-7B-Instruct. Prior to this, I successfully ran QwQ-32B using the same code.








`import sglang as sgl
import json
import time
from tqdm import tqdm
import argparse
import os
from transformers import AutoTokenizer
from sglang.srt.sampling.sampling_params import SamplingParams
from matheval import evaluator_map, set_client, AIMEEvaluator
import asyncio
import matheval
import humanevaleval
import mbppeval
from huggingface_hub import HfApi
import torch
import time
import convert_livecodebench

MATH_DATASETS = ["math500","aime2024","aime2025","gpqa_diamond","gsm8k","amc23"]
CODE_DATASETS = ["humaneval","mbpp","livecodebench"]

def main():
    parser = argparse.ArgumentParser(description='Process some parameter

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] [file name too long] base64 end up with "JPG" or others will throw exception

**Link**: https://github.com/sgl-project/sglang/issues/7199
**State**: open
**Created**: 2025-06-15T03:18:11+00:00
**Comments**: 0

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I always use base64 as my input, but it will throw exception **[file name too long]** bug sometime, we found that's beacuse some picture's base64 end up with "JPG" or other image format, it will regard the base64 string as filename, so exception thrown.

bug file is: python3.11/site-packages/sglang/srt/utils.py
see bug log here:
![Image](h

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] DeepEP Failed with "too many blocks in cooperative launch"

**Link**: https://github.com/sgl-project/sglang/issues/6056
**State**: closed
**Created**: 2025-05-06T12:51:59+00:00
**Closed**: 2025-05-06T13:30:55+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

We deploy DeepSeek-R1 in 1 node with H20*8, it hits an exception when capture cuda graph

<pre>
[2025-05-06 12:35:08 TP0] Scheduler hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/model_executor/cuda_graph_runner.py", line 276, in __init__
    self.capture()
  File "/sgl-workspace/sglang

[... truncated for brevity ...]

---

## Issue #N/A: [Question] Gemma-2 sliding window support?

**Link**: https://github.com/sgl-project/sglang/issues/1016
**State**: closed
**Created**: 2024-08-10T09:29:34+00:00
**Closed**: 2024-08-10T09:32:27+00:00
**Comments**: 0

### Description

Thanks for your hard work on this project! I'm trying to use gemma-2-27b-it with sglang==0.2.9 and flashinfer backend ==0.1.3. I’m interested in using a context length of 8192 with sliding window attention, which seems to be supported by flashinfer.

However, I noticed that the comments in the gemma2 repository suggest that this might not be allowed. Could you clarify whether it's possible to use an 8192 context length with sliding window attention in this setup?

Any guidance or suggestions would be greatly appreciated!

---

## Issue #N/A: [Bug] cannot set --load-format=dummy with vllm 0.5.5

**Link**: https://github.com/sgl-project/sglang/issues/1259
**State**: closed
**Created**: 2024-08-29T23:39:25+00:00
**Closed**: 2024-08-30T06:43:42+00:00
**Comments**: 0

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

In vllm 0.5.5,`MultiModalConfig` has been refactored into a attribute of `ModelConfig` in this [PR](https://github.com/vllm-project/vllm/pull/7530), and is no longer an argument in [DummyModelLoader.load_model](https://github.com/vllm-project/vllm/blob/09c7792610ada9f88bbf87d32b472dd44bf23cc2/vllm/model_executor/model_loader/loader.py#L374

[... truncated for brevity ...]

---

## Issue #N/A: enable_torch_compile has no effect on Qwen3-32B-FP8 with H20 GPU

**Link**: https://github.com/sgl-project/sglang/issues/6798
**State**: open
**Created**: 2025-06-01T15:33:25+00:00
**Comments**: 0

### Description

I tested the enable_torch_compile=True configuration on the Qwen3-32B-FP8 model using an H200 GPU, and observed that it had no positive effect on performance. In fact, in some metrics it slightly degraded performance.

Here are the benchmark results for comparison:

Without enable_torch_compile:
============ Serving Benchmark Result ============
Successful requests:                     4596      
Benchmark duration (s):                  397.14    
Total input tokens:                      9577351   
Total generated tokens:                  514870    
Request throughput (req/s):              11.57     
Output token throughput (tok/s):         1296.45   
Total Token throughput (tok/s):          25412.30  
---------------Time to First Token----------------
Mean TTFT (ms):                          836.69    
Median TTFT (ms):                        710.76    
P99 TTFT (ms):                           2590.59   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):               

[... truncated for brevity ...]

---

## Issue #N/A: How to study the code?

**Link**: https://github.com/sgl-project/sglang/issues/1515
**State**: closed
**Created**: 2024-09-26T04:35:15+00:00
**Closed**: 2024-09-26T04:35:53+00:00
**Comments**: 0

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Hi. I am a student, and I am recently studying the code of SGL. I have read your great paper but I need some more concrete feelings of the system. Could you please tell me some resources or suggesttions on understand the code of SGL? Thanks. 

### Related resources

_No response_

---

## Issue #N/A: [Feature] need DeepSeek-v2 or deepseek-v2.5 awq support

**Link**: https://github.com/sgl-project/sglang/issues/1395
**State**: closed
**Created**: 2024-09-12T02:28:46+00:00
**Closed**: 2024-09-12T02:35:36+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

sglang 0.30 can run DeepSeek-Coder-V2-Lite-Instruct.
but running DeepSeek-Coder-V2-Lite-Instruct-AWQ reports:
ValueError: The input size is not aligned with the quantized weight shape. This can be caused by too large tensor parallel size.

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

## Issue #N/A: PD Disaggregation with tp stuck

**Link**: https://github.com/sgl-project/sglang/issues/6864
**State**: closed
**Created**: 2025-06-04T08:33:50+00:00
**Closed**: 2025-06-04T09:43:30+00:00
**Comments**: 0

### Description

When I execute 
`python3 -m sglang.launch_server --model-path models/DeepSeek-R1-Distill-Qwen-32B --disaggregation-mode prefill --disaggregation-ib-device mlx5_0,mlx5_1 --tp=2` 
the process stuck with logs as follows:

server_args=ServerArgs(...)
[2025-06-04 07:49:51 TP0] Attention backend not set. Use flashinfer backend by default.
[2025-06-04 07:49:51 TP0] Init torch distributed begin.
[2025-06-04 07:49:51 TP0] sglang is using nccl==2.21.5

It seems that the TP1 process hasn't started, so it is stuck. I want to know how to solve it.

---

## Issue #N/A: [Feature] Question about kvcache offloading to disk or CPU memory

**Link**: https://github.com/sgl-project/sglang/issues/1072
**State**: closed
**Created**: 2024-08-13T09:51:24+00:00
**Closed**: 2024-08-13T09:54:17+00:00
**Comments**: 0

### Description

### Motivation

In sglang's paper, it mentions that sglang plans to implement a kvcache offloading mechanism. I notice that there is a `disable_disk_cache=False` option in sglang's Server args, and wonder if this option controls the kvcache offloading mechanism.

### Related resources

_No response_

---

## Issue #N/A: [Bug] Accuracy is abnormal when EP MoE is enabled

**Link**: https://github.com/sgl-project/sglang/issues/2482
**State**: closed
**Created**: 2024-12-14T14:13:15+00:00
**Closed**: 2024-12-16T13:11:35+00:00
**Comments**: 0

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

Accuracy on gsm8k dataset is decreased for EP MoE. 
cc: @xiaobochen123

### Reproduction

EP:
```bash
python3 -m sglang.launch_server --model-path neuralmagic/DeepSeek-Coder-V2-Instruct-FP8 --disable-radix-cache --trust-remote-code --tp 8 --enable-ep-moe --disable-cuda-graph
python3 benchmark/gsm8k/bench_sglang.py --num-q

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] How to run the load balancer

**Link**: https://github.com/sgl-project/sglang/issues/7592
**State**: open
**Created**: 2025-06-27T07:56:26+00:00
**Comments**: 0

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

How to effectively start the load balancer, and how to test whether it is running normally after the startup is completed 。 After the load balancer is deployed, the test call interface is available, but the model does not run.

### Reproduction

sudo docker run -d --gpus all \
    --shm-size 512g \
    --network host \
    --privileged \
 

[... truncated for brevity ...]

---

## Issue #N/A: question of enable-ep-moe

**Link**: https://github.com/sgl-project/sglang/issues/3537
**State**: closed
**Created**: 2025-02-13T03:30:31+00:00
**Closed**: 2025-02-13T05:06:00+00:00
**Comments**: 0

### Description

Hi, team. When using --enable-ep-moe to start, are the expert weights on each card still loaded in a column-parallel manner? Does this have an impact on performance? I mean will a single expert still triggers all gather communication itself even if it belongs to a single card?

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

