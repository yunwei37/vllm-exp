# references_prs - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 9
- Closed Issues: 21

### Label Distribution

- inactive: 8 issues
- high priority: 6 issues
- collaboration: 2 issues
- lora: 1 issues
- bug: 1 issues
- flashinfer: 1 issues
- blackwell: 1 issues
- enhancement: 1 issues
- performance: 1 issues
- quant: 1 issues

---

## Issue #N/A: PR review process standard

**Link**: https://github.com/sgl-project/sglang/issues/344
**State**: closed
**Created**: 2024-04-02T14:30:23+00:00
**Closed**: 2024-04-03T05:56:36+00:00
**Comments**: 0

### Description

@hnyls2002 Do you realize that by closing my PR #338 and then opening another pr with my branch and push shape fix, then subsequent merge, you put me as a co-author of the squashed commit.

1. revert this 
2. reopen my pr
3. either push commit directly to pr branch (I enabled maintainer edit permission by default)
4. or push fix to my branch as Pr in my repo to merge

This is closing pr to fix bug with same commits is bad for other reasons such as retaining pr conversation history. Now your squashed commit ref a pr with zero info and a redirect to real but closed Pr.




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

## Issue #N/A: [Bug]

**Link**: https://github.com/sgl-project/sglang/issues/6654
**State**: closed
**Created**: 2025-05-27T06:54:16+00:00
**Closed**: 2025-06-23T02:50:40+00:00
**Comments**: 10

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When deploying the DeepSeek-V3-AWQ model using sglang on 8 H100 GPUs, the model outputs are all garbled during inference.
Partially garbled outputs:
```
# -# # 
# # 
# 
# 
# # # # # # # # // Seng# # #0xA
# # # # # # 
package# 
# # # # # #   # 
# # 

#   # # 
# # # 
# 
# 
# # 
# # 
# # # # # 
# # # # 

```
The launch command can run normall

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] JSON Regex does not work for vision model

**Link**: https://github.com/sgl-project/sglang/issues/1621
**State**: closed
**Created**: 2024-10-10T00:05:22+00:00
**Closed**: 2024-10-10T23:34:14+00:00
**Comments**: 2

### Description

### Describe the bug

`test_vision_openai_server.py` is failing on main CI. The root cause commit is https://github.com/sgl-project/sglang/pull/1598. Before that, the test works fine. After narrowing down a bit, I found this is because the answer does not conform to the json format.

Before #1598

```

{
   "color": "yellow",
   "number_of_cars": 2
} 
```

After #1598

```
{  
   "color": "yellowstreetscene_name_person_91_colo_unique_car_yellowb8hellbluefaces_colozyagnoufligdullowski9191_1999c2911silverblef5_orksymphoonsingerealcovertladyfiectpooordecoreallypinktanvbtn001_91white7hellcosmonoch16black1realcampus992191godentale9gelownhueCors7ollectionsohak41ielleall991lcoblenicorn936pink

```

### Reproduction

```python
"""
Usage: python3 local_example_llava_next.py
"""

import sglang as sgl


regex = (
    r"""\{\n"""
    + r"""   "color": "[\w]+",\n"""
    + r"""   "number_of_cars": [\d]+\n"""
    + r"""\}"""
)

@sgl.function
def image_qa(s, ima

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]  Add support for INT8 quantization to Qwen3MoE

**Link**: https://github.com/sgl-project/sglang/issues/5835
**State**: closed
**Created**: 2025-04-28T13:12:30+00:00
**Closed**: 2025-06-11T21:23:24+00:00
**Comments**: 4

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

As the INT8 quantization method mentioned in #3888 has shown good benchmarking results, and the INT8 data type is both friendly and efficient for most hardware platforms, we plan to add support for channel-wise INT8 quantization operations to Qwen3MOE. Once the model file becomes available, we will submit the test results and PR at the earliest opportunity.

### Related resources

_No response_

---

## Issue #N/A: [Feature] Add server arg enable-lora to allow starting up with empty lora-paths

**Link**: https://github.com/sgl-project/sglang/issues/7463
**State**: open
**Created**: 2025-06-23T07:42:22+00:00
**Comments**: 3
**Labels**: lora

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Currently SGL implicitly uses --lora-paths to decide whether engine/server should be started with LoRA support enabled.

As we are going to support dynamic lora loading/unloading soon (#7446), the current implicit constraint is no longer reasonable as it should be perfectly legal for users to start a LoRA-enabled server without having to provide any lora paths, but instead load/unload adapters later as needed. 

### Related resources

_No response_

---

## Issue #N/A: [Bug] Remove stream sync in fast decode plan of flashinfer mla backend

**Link**: https://github.com/sgl-project/sglang/issues/4905
**State**: closed
**Created**: 2025-03-29T23:34:44+00:00
**Closed**: 2025-04-30T03:06:05+00:00
**Comments**: 1
**Labels**: bug, flashinfer

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

https://github.com/flashinfer-ai/flashinfer/pull/969 claims that the flashinfer mla backend can be sped up after removal of 
```python
  with self.device as device:
      stream = torch.cuda.current_stream(device).cuda_stream
```
in `fast_mla_decode_plan` of `flashinfer_mla_backend.py`

We need to test its performance after removal.

### R

[... truncated for brevity ...]

---

## Issue #N/A: Deepseek-R1 MTP poor performance

**Link**: https://github.com/sgl-project/sglang/issues/4360
**State**: closed
**Created**: 2025-03-13T03:44:18+00:00
**Closed**: 2025-04-24T13:22:17+00:00
**Comments**: 5

### Description

I expect the time we spend on "verify" part should be close to a normal decode forward (less than 100ms, my setting is bs=16 and ctx=12k), but now it takes about 400ms. It slows down my output throughput severely.  Seems like a kernel performance issue?

decoding with mtp profile:
<img width="1384" alt="Image" src="https://github.com/user-attachments/assets/89c722bf-a875-400e-893a-9f16b5d0e529" />

normal decode profile:
<img width="1103" alt="Image" src="https://github.com/user-attachments/assets/851bdb2d-9dbc-4512-80d9-f579f9537d50" />

The commit I test:
commit 4a05bdfa869c80fdcac2d1b8fb48656f743a1fac (gh/main)
Author: Lianmin Zheng <lianminzheng@gmail.com>
Date:   Sun Mar 9 18:53:33 2025 -0700

    Revert "Check eagle server args" (#4242)

_Originally posted by @jokerwyt in https://github.com/sgl-project/sglang/issues/3582#issuecomment-2719759962_
            

---

## Issue #N/A: [Feature] Add a flag for computing the prompt's logprobs or not.

**Link**: https://github.com/sgl-project/sglang/issues/902
**State**: closed
**Created**: 2024-08-03T05:53:54+00:00
**Closed**: 2024-09-22T13:04:36+00:00
**Comments**: 2

### Description

### Motivation

Mentioned in #852 

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

## Issue #N/A: [Feature] Enhanced support/structure for Multi-modal models

**Link**: https://github.com/sgl-project/sglang/issues/2439
**State**: closed
**Created**: 2024-12-11T06:45:25+00:00
**Closed**: 2024-12-11T07:14:53+00:00
**Comments**: 1

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

In vllm,  the framework can accept either image or image-embedding. See  [vllm [Feature] Add vision language model support. #3042](https://github.com/vllm-project/vllm/pull/3042) and [vllm-llava impl.](https://github.com/vllm-project/vllm/blob/2e33fe419186c65a18da6668972d61d7bbc31564/vllm/model_executor/models/llava.py#L479)
 
Embedding an image requires fixed predictable compute and is easy to batch and run   in a separate framework(for instance, tensorrt-based serving framework). See discussion in https://github.com/vllm-project/vllm/issues/307#issuecomment-1840443044

Ideally, it is necessary to maintain the infrastructure to overlap (image (gpu) preprocessing + inference) and (llm inference) within the same

[... truncated for brevity ...]

---

## Issue #N/A: Development  Roadmap (2024 Q3)

**Link**: https://github.com/sgl-project/sglang/issues/634
**State**: closed
**Created**: 2024-07-17T02:15:39+00:00
**Closed**: 2024-11-01T05:56:56+00:00
**Comments**: 19

### Description

Here is the development roadmap for 2024 Q3. Contributions and feedback are welcome.

## Server API
 - [ ] Add APIs for using the inference engine in a single script without launching a separate server. See also [examples](https://docs.vllm.ai/en/latest/getting_started/examples/offline_inference.html).
   - #1127 
 - [x] Support most OpenAI APIs: Batch, completion, chat, embedding
   - #699  
   - #640 
   - #852 
   - #916 
   - #997
- [ ] Support directly taking embedding as inputs. #745
- [x] Support updating the model weights without relaunching the server. @shanyu-sys 
   - #1157 
- [ ] Support Mistral endpoint in the language frontend
## Performance
- [x] Improve time-to-first-token in streaming mode with better scheduling.
  - #1339
  - #1345
- [x] Implement chunked prefill. @hnyls2002 @vikranth22446 
   - #800
   - #811 
   - #1040 
   - #1013 
- [ ] Implement speculative decoding. See also a [prototype](https://github.com/sgl-project/sglang/pull/270).


[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Frontend compatibility with Python 3.13

**Link**: https://github.com/sgl-project/sglang/issues/3876
**State**: open
**Created**: 2025-02-26T07:18:42+00:00
**Comments**: 2
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

`cafile` parameter in `urllib.request.urlopen` is [removed in Python 3.13](https://docs.python.org/3.13/library/urllib.request.html), causing SGLang frontend to fail.

```
File "lib/python3.13/site-packages/starlette/routing.py", line 693, in lifespan
  async with self.lifespan_context(app) as maybe_state:
File "lib/python3.13/contextlib.p

[... truncated for brevity ...]

---

## Issue #N/A: [Roadmap] Blackwell Support and Optimizations

**Link**: https://github.com/sgl-project/sglang/issues/7227
**State**: open
**Created**: 2025-06-16T06:07:50+00:00
**Comments**: 45
**Labels**: high priority, collaboration, blackwell

### Description

### Roadmap

- [x] ~~Initial support and optimizations for GB200, PD disaggregation, and large-scale EP~~ -- Done in https://lmsys.org/blog/2025-06-16-gb200-part-1/
- [x] Initial optimizations for prefill for large scale EP
- [ ] Optimize kernels for the Blackwell architecture
    - [ ] Communication kernels
    - [ ] Various smaller kernels
- [ ] Optimize for latency-oriented scenarios
- [ ] Computation-communication overlap

TODO: more

### Updates after Blog

* Prefill is slightly optimized, 13149 token/s/gpu for ISL 4096 (as usual all code are open sourced)

### Blog Reproduction

<details>

To reproduce [the blog post](https://lmsys.org/blog/2025-06-16-gb200-part-1/), here are the instructions:

#### 2025.07.12

To use the latest main, the following commands can be used.

Versions that I personally use to test (other versions may work as well)
* SGLang: https://github.com/sgl-project/sglang/commit/2a2d3478afe8cdb336888f2e6faa3775ac40254e
* sgl-kernel: the one inside SGLang
* DeepG

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] AWQ Quantization fails with Qwen 2.5 VL

**Link**: https://github.com/sgl-project/sglang/issues/3571
**State**: closed
**Created**: 2025-02-14T07:38:23+00:00
**Closed**: 2025-04-22T00:18:31+00:00
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

When trying to serve Qwen 2.5 VL with AWQ quantization (unofficial awq model, https://huggingface.co/PointerHQ/Qwen2.5-VL-72B-Instruct-Pointer-AWQ) using this pr #3258 , got the following error:

```bash
 $ python3.10 -m sglang.launch_server --model-path /data1/Qwen2.5-VL-72B-Instruct-Pointer-AWQ/ --tp 2 --dtype float16
 

INFO 02-14 07:01

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] text generation hangs after serving some requests

**Link**: https://github.com/sgl-project/sglang/issues/4191
**State**: closed
**Created**: 2025-03-08T00:48:48+00:00
**Closed**: 2025-07-11T00:20:20+00:00
**Comments**: 5
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug


Hi,

## Bug: chat completion requests hang

The command hangs indefinitely (at least 10+ minutes)
```bash
curl -v -H 'Content-Type: application/json' localhost:30000/v1/chat/completions -d '{ "model": "deepseek-ai/DeepSeek-R1", "messages": [{"role": "user", "content": "What is the capital of France?"}] }'
```

Similarly, `get_server_info`

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] sgl-kernel tests may be broken when running alone

**Link**: https://github.com/sgl-project/sglang/issues/7464
**State**: open
**Created**: 2025-06-23T08:11:00+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When running some sgl-kernel tests alone (rather than run all tests by `pytest tests`), it will raise `RuntimeError`:

```
    def get_summarized_data(self):
        dim = self.dim()
        if dim == 0:
            return self
        if dim == 1:
            if self.size(0) > 2 * PRINT_OPTS.edgeitems:
>               return torch.cat(
  

[... truncated for brevity ...]

---

## Issue #N/A: internvl3 support

**Link**: https://github.com/sgl-project/sglang/issues/6978
**State**: closed
**Created**: 2025-06-08T16:17:07+00:00
**Closed**: 2025-06-09T06:56:14+00:00
**Comments**: 1

### Description

Hi,

I'm trying to load InternVL3 using sglang but encountered issues. I noticed this pull request: [#5350](https://github.com/sgl-project/sglang/pull/5350) mentions that "InternVL3 includes a flashattention implementation for vision models. However, it doesn't support Tensor Parallelism (TP), which could be a bottleneck".

Does this imply I need to disable TP for a bug-free experience with InternVL3? If yes, are there any known workarounds or fixes planned for future releases?

Thank you!

---

## Issue #N/A: compatibility issues and memory leak problems --enable-flashinfer

**Link**: https://github.com/sgl-project/sglang/issues/356
**State**: closed
**Created**: 2024-04-09T12:58:04+00:00
**Closed**: 2024-07-25T06:33:15+00:00
**Comments**: 4
**Labels**: inactive

### Description

Version: sglang==0.1.14
Hardware: ec2 g5.xlarge

Hi, when using the following line:
```python3
python sglang.launch_server --model-path openchat/openchat-3.5-0106 --port 30000 --mem-fraction-static 0.8 --enable-flashinfer
```

So, I notice two problems when running the above:
1. When using `--enable-flashinfer` the gemma [script](https://github.com/sgl-project/sglang/blob/550a4f78f382b5a7f4008d7d21e876e71ab2d2b6/python/sglang/srt/models/gemma.py) is invoked for some reason (I believe openchat is a finetuned version of mistral). When not using `--enable-flashinfer` the server starts up and works as expected.
2. the gemma [script](https://github.com/sgl-project/sglang/blob/v0.1.14/python/sglang/srt/models/gemma.py#L12) imports from `vllm.model_executor.input_metadata`. input_metadata.py which was removed in vllm 0.4.0

Downgrading the vllm version to 0.3.3 gets the server up and running, but then a KV pool cache leak occurs, which I see was mentioned here #236 . This may be a

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] DeepSeek V3 optimization

**Link**: https://github.com/sgl-project/sglang/issues/2591
**State**: closed
**Created**: 2024-12-26T08:52:39+00:00
**Closed**: 2025-03-25T04:10:46+00:00
**Comments**: 52
**Labels**: enhancement, high priority, performance, quant

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Adoption

[SGLang adoption for DeepSeek V3 and R1](https://github.com/sgl-project/sglang/discussions/3322)

### Usage

User Guide for Existing System (Installation & Launch)

https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3

Please use the latest version [v0.4.2.post4](https://pypi.org/project/sglang/0.4.2.post4/). Please prefer to use docker image. `docker pull lmsysorg/sglang:latest`

For running on AMD MI300X, use this as a reference. [Running DeepSeek-R1 on a single NDv5 MI300X VM](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/running-deepseek-r1-on-a-single-ndv5-mi300x-vm/4372726)

### Features

- [x] Support CUDA Graph @HandH1998 @ispobock 
- [x] Support Torch compile @is

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support varied input formats for remaining VLM

**Link**: https://github.com/sgl-project/sglang/issues/6483
**State**: open
**Created**: 2025-05-21T05:51:01+00:00
**Comments**: 2
**Labels**: good first issue, help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Currently SGL has supported varied format inputs for some of VLMs. We should support all remaining VLM and add tests (Parent issue: https://github.com/sgl-project/sglang/issues/5964)

- We should follow the refactored process in #6659 (Note that it's somewhat outdated, check with the latest main)

- [x] QwenVL (@ysulsky https://github.com/sgl-project/sglang/pull/6136)
- [x] Gemma (@ysulsky https://github.com/sgl-project/sglang/pull/6136) 
- [x] KimiVL (@lifuhuang https://github.com/sgl-project/sglang/pull/6599)
- [ ] Phi4mm
- [ ] internvl
- [ ] mllama4
- [ ] pixtral
- [ ] deepseek_vl_v2
- [ ] minicpm
- [ ] janus_pro

### Related resources

_No response_

---

## Issue #N/A: [Bug] IndexError: Inconsistent batch_size and len(image_input)

**Link**: https://github.com/sgl-project/sglang/issues/1692
**State**: closed
**Created**: 2024-10-17T03:07:10+00:00
**Closed**: 2024-10-17T17:22:47+00:00
**Comments**: 2
**Labels**: high priority

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

```
.......site-packages/sglang/srt/models/llava.py", line 362, in forward
    for j, image_offset in enumerate(image_offsets[i]):
IndexError: list index out of range
```

The issue persists in version `0.3.3.post1`. During batch inference, `LlavaBaseForCausalLM` can encounter inconsistent `batch_size` and `len(image_input)

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] sglang crashes with multi node

**Link**: https://github.com/sgl-project/sglang/issues/3932
**State**: closed
**Created**: 2025-02-27T16:49:03+00:00
**Closed**: 2025-03-06T02:24:12+00:00
**Comments**: 6

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Using 2 H200 setup

Node 0

```python
[2025-02-27T16:30:43.48675097Z] [rank3]:[E227 16:30:43.526290891 ProcessGroupNCCL.cpp:1785] [PG ID 2 PG GUID 3 Rank 3] Exception (either an error or timeout) detected by watchdog at work: 32, last enqueued NCCL work: 32, last completed NCCL work: 31.
 [rank3]:[E227 16:30:43.526301625 ProcessGroupNCCL.c

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

## Issue #N/A: [Bug] AssertionError: res=<Response [503]> Process was always killed automactically

**Link**: https://github.com/sgl-project/sglang/issues/4094
**State**: closed
**Created**: 2025-03-05T11:35:03+00:00
**Closed**: 2025-05-25T00:21:16+00:00
**Comments**: 5
**Labels**: inactive

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

I successfully ran the serve  using command after installing SGLang. But after about 2 minutes, the process is always killed automatically.

### Reproduction

I depoly Qwen2.5-7B-Instruct using the below command

```
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server --model-path /data/models/Qwen2.5/Qwen2.5-7B-Instruct --api-key base -

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] support rpc in dp_size > 1

**Link**: https://github.com/sgl-project/sglang/issues/6988
**State**: open
**Created**: 2025-06-09T06:46:36+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

In [#3964](https://github.com/sgl-project/sglang/pull/3964), we introduce a new `zmq` socket named `rpc`, for possible rpc calls. `tp0` will receive rpc calls and broadcast to other `tpworkers`.

However, we did not consider much for `dp` at that time. So we hope to support `rpc` under `dp` scenario.

### Related resources

_No response_

---

## Issue #N/A: Optimize abort request handling

**Link**: https://github.com/sgl-project/sglang/issues/481
**State**: closed
**Created**: 2024-05-27T08:41:14+00:00
**Closed**: 2024-07-27T01:02:01+00:00
**Comments**: 1
**Labels**: inactive

### Description

Please avoid for loop.
See https://github.com/sgl-project/sglang/blob/55c16436273d4a42f7cfe342df5f10ad05a8d0fe/python/sglang/srt/managers/router/model_rpc.py#L710
and `controller.py` after #480 merged.

---

## Issue #N/A: use the same Test Environment and same Engine arguments as #4616 ,but the speed is slow

**Link**: https://github.com/sgl-project/sglang/issues/4649
**State**: closed
**Created**: 2025-03-21T08:04:31+00:00
**Closed**: 2025-06-04T00:19:42+00:00
**Comments**: 10
**Labels**: inactive

### Description

Test Environment:

SGLang version: 0.4.4.post1
Flashinfer version: 0.2.3+cu124torch2.5
Hardware: 2 nodes of H20 ( 8 * H20 96GiB each)
Model: DeepSeek-R1
Model Max Length: 3200 (modified in both model and NextN's tokenizer_config.json)
CUDA Version: 12.4
Operating System: Ubuntu SMP Fri Mar 18 12:42:08 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
Test bench: jmeter
Avg input length = 912 tokens
Avg output length = 2174 tokens

Engine arguments:
python -m sglang.launch_server --model-path <YOUR_MODEL_DIR> --tp 16 --dist-init-addr <YOUR_ADDR> --nnodes 2 --node-rank <YOUR_NODE_RANK> --trust-remote-code --max-running-requests 1024 --speculative-algorithm NEXTN --speculative-draft <YOUR_NEXTN_MODEL_DIR> --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-torch-compile --enable-flashinfer-mla --mem-fraction-static 0.7 --host <YOUR_HOST_IP> --port <YOUR_HOST_PORT> --schedule-conservativeness 0.01

result:
Per-request Output Throughput (token/s) : 13.98 tok

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

