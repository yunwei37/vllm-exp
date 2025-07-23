# good_first_issue - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 8
- Closed Issues: 22

### Label Distribution

- good first issue: 30 issues
- help wanted: 16 issues
- documentation: 7 issues
- high priority: 4 issues
- collaboration: 2 issues
- inactive: 2 issues
- performance: 1 issues
- amd: 1 issues
- RLHF: 1 issues
- feature: 1 issues

---

## Issue #N/A: [Feature]Support Qwen2_5...etc tools calling by OpenAI API

**Link**: https://github.com/sgl-project/sglang/issues/1912
**State**: closed
**Created**: 2024-11-04T02:32:41+00:00
**Closed**: 2025-02-12T02:12:07+00:00
**Comments**: 3
**Labels**: good first issue

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Tools calling are becoming mainstream. If you can adapt some updated OpenAI APIs, I would be very grateful.

### Related resources

_No response_

---

## Issue #N/A: [Feature] beat torch compile

**Link**: https://github.com/sgl-project/sglang/issues/4748
**State**: closed
**Created**: 2025-03-25T06:18:28+00:00
**Closed**: 2025-05-26T16:55:12+00:00
**Comments**: 7
**Labels**: good first issue, help wanted, high priority, collaboration, performance

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

Last year and in the first few months of this year, a significant part of my work focused on removing vLLM dependency. Many reliable teammates joined in this process, and we successfully removed the vLLM dependency on the NVIDIA platform for SGLang. Next, I will co-lead progress on beat torch compile. Past experience shows that torch compile is effective - we just need to write some simple torch ops and let torch compile handle the rest. However, in actual production serving, it is not as smooth as expected - for example, slow startup even with cache enabled, compatibility issues when upgrading torch versions leading to previous features breaking in new versions. We need to profile, benchmark, rewrite th

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] support constrained decoding benchmark

**Link**: https://github.com/sgl-project/sglang/issues/2399
**State**: closed
**Created**: 2024-12-08T10:56:36+00:00
**Closed**: 2025-05-29T21:48:23+00:00
**Comments**: 1
**Labels**: good first issue, help wanted

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled
used for outlines, xgrammar and etc
ref https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving_guided.py

### Related resources

_No response_

---

## Issue #N/A: [Bug] RecursionError: maximum recursion depth exceeded while calling a Python object

**Link**: https://github.com/sgl-project/sglang/issues/4779
**State**: closed
**Created**: 2025-03-26T04:14:50+00:00
**Closed**: 2025-03-26T15:59:08+00:00
**Comments**: 5
**Labels**: good first issue

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug


I confronted this issue today by using docker-latest and docker-dev when using QWQ-32B  but no issue in QWQ-AWQ model. 



Following is my error log

```
  File "/usr/local/lib/python3.10/dist-packages/psutil/__init__.py", line 1277, in send_signal
    self._send_signal(sig)
  File "/usr/local/lib/python3.10/dist-packages/psutil/__init__.

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] GGUF support

**Link**: https://github.com/sgl-project/sglang/issues/1616
**State**: closed
**Created**: 2024-10-09T05:45:17+00:00
**Closed**: 2024-12-01T10:51:57+00:00
**Comments**: 26
**Labels**: good first issue

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Hi! Since .gguf format is already supported by vLLM, is it be possible to add support for it in SGLang server?

### Related resources

_No response_

---

## Issue #N/A: Loading Chat Template in a more flexible way?

**Link**: https://github.com/sgl-project/sglang/issues/376
**State**: closed
**Created**: 2024-04-21T12:50:17+00:00
**Closed**: 2024-07-25T06:33:13+00:00
**Comments**: 1
**Labels**: good first issue, inactive

### Description

The Chat models like [codellama-instruct](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf/blob/main/tokenizer_config.json), [qwen](https://modelscope.cn/models/qwen/Qwen1.5-14B-Chat/file/view/master?fileName=tokenizer_config.json&status=1) all have a `chat_template` field in the JSON which defines the chat template of the model. But I notice it seems that sglang currently hard-coded the chat-template in the [.py](https://github.com/sgl-project/sglang/blob/1bf1cf195302fdff14a4321eb8a17831f5c2fc11/python/sglang/lang/chat_template.py#L79) file. Would it be more flexible to load the default chat template from the tokenizer_config file if provided? It seems [vllm](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/serving_chat.py#L335) did in this way.

---

## Issue #N/A: Support Yi-VL-6B/34B

**Link**: https://github.com/sgl-project/sglang/issues/91
**State**: closed
**Created**: 2024-01-24T03:49:46+00:00
**Closed**: 2024-02-01T21:38:25+00:00
**Comments**: 7
**Labels**: good first issue

### Description

The Yi-VL adopts llava but with silightly different in weights and inference. see [disscusion](https://huggingface.co/01-ai/Yi-VL-34B/discussions/3)

hf repo:
https://huggingface.co/01-ai/Yi-VL-6B
https://huggingface.co/01-ai/Yi-VL-34B

---

## Issue #N/A: [Docs]  Improve DPSK docs in dark mode

**Link**: https://github.com/sgl-project/sglang/issues/3908
**State**: closed
**Created**: 2025-02-27T05:00:48+00:00
**Closed**: 2025-02-27T08:13:05+00:00
**Comments**: 2
**Labels**: documentation, good first issue, help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

<img width="1393" alt="Image" src="https://github.com/user-attachments/assets/39d60ef8-c7fa-42e0-9961-5bd9c082209f" />

I use html to write this docs and it looks bad. So could someone fix it here?

https://github.com/sgl-project/sglang/blob/main/docs/references/deepseek.md

### Related resources

_No response_

---

## Issue #N/A: [Feature] Change contribution guide

**Link**: https://github.com/sgl-project/sglang/issues/2662
**State**: closed
**Created**: 2024-12-30T07:53:12+00:00
**Closed**: 2025-04-29T16:22:21+00:00
**Comments**: 3
**Labels**: documentation, good first issue

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

https://sgl-project.github.io/references/contributor_guide.html

This has been outdated for long. We need to add guide on:

1. How to run docs CI, build it locally, compile it and clean the output and make PR.
2. How to do unit tests locally and add unit tests to CI.
3. How to write elegant unit test following other tests.
4. How to pre-commit.

### Related resources

_No response_

---

## Issue #N/A: [Feature] Parallelism Experiments on AIMO and LIMO

**Link**: https://github.com/sgl-project/sglang/issues/3615
**State**: closed
**Created**: 2025-02-16T19:11:32+00:00
**Closed**: 2025-02-20T19:11:38+00:00
**Comments**: 6
**Labels**: documentation, good first issue, help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Can anyone help test @Simon V’s branch? It’s pretty complete, but we’d like to run some parallel experiments 

https://github.com/sgl-project/sglang/pull/3532

Feel free to submit a PR reporting the results of the parallel experiments, including std, var, etc. Thanks!

### Related resources

_No response_

---

## Issue #N/A: [Feature] Add KV cache usage prometheus metrics

**Link**: https://github.com/sgl-project/sglang/issues/5979
**State**: open
**Created**: 2025-05-02T14:59:34+00:00
**Comments**: 3
**Labels**: good first issue

### Description

### Motivation

It would be great to track prometheus metrics for KV cache utilization.

### Related resources

vLLM already offers KV cache utilization prometheus metrics, see [here](https://docs.vllm.ai/en/stable/serving/metrics.html), at `vllm:gpu_cache_usage_perc`.

---

## Issue #N/A: `model_override_args` with server

**Link**: https://github.com/sgl-project/sglang/issues/591
**State**: closed
**Created**: 2024-07-05T09:57:03+00:00
**Closed**: 2024-09-08T01:12:57+00:00
**Comments**: 2
**Labels**: good first issue, inactive

### Description

When using a server, one currently cannot use the `model_overide_args` which could be very useful, e.g. for rope scaling. 

This is currently the `sglang.launch_server.py`:

```py
import argparse

from sglang.srt.server import launch_server
from sglang.srt.server_args import ServerArgs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)

    launch_server(server_args, None)
```

The `model_overide_args` would be the third argument to `launch_server` defaulting to `None`. Adding a small cli parser that allows arbitrary model args would be great, e.g.

```bash
python -m sglang.launch_server --model_overide_args.rope_scaling.factor 2 --model_overide_args.rope_scaling.type linear
```

---

## Issue #N/A: [Feature] Support Qwen2-VL based embedding model

**Link**: https://github.com/sgl-project/sglang/issues/2032
**State**: closed
**Created**: 2024-11-14T08:00:38+00:00
**Closed**: 2024-11-21T22:25:09+00:00
**Comments**: 1
**Labels**: good first issue

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

The multimodal embedding showed great potential for RAG pipeline. There're multiple approaches all based on Qwen-VL Conditional Generation models, like https://huggingface.co/marco/mcdse-2b-v1 and https://huggingface.co/blog/marco/announcing-mcdse-2b-v1. 

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

## Issue #N/A: [Feature] support `gather` instead of `all_gather` when gathering the logits

**Link**: https://github.com/sgl-project/sglang/issues/3365
**State**: open
**Created**: 2025-02-07T07:14:12+00:00
**Comments**: 8
**Labels**: good first issue, help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

We noticed that in the `_get_logits` function of vllm, `gather` instead of `all_gather` will be used under certain conditions (the main condition is that for non-tpu devices):
Code link:

- [logits = tensor_model_parallel_gather(logits)](https://github.com/vllm-project/vllm/blob/6e1fc61f0fb90c37f0d4a1a8f76235a6e4e1103c/vllm/model_executor/layers/logits_processor.py#L101C22-L101C50)

- [condition of whether using `all_gather` or `gather`](https://github.com/vllm-project/vllm/blob/6e1fc61f0fb90c37f0d4a1a8f76235a6e4e1103c/vllm/model_executor/layers/logits_processor.py#L53-L57)

The change from using `all_gather` to `gather` is initially added in this PR for your reference: https://github.com/vllm-project/vllm/pull/222

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support InterVL

**Link**: https://github.com/sgl-project/sglang/issues/3092
**State**: closed
**Created**: 2025-01-24T01:30:05+00:00
**Closed**: 2025-05-04T05:14:13+00:00
**Comments**: 4
**Labels**: good first issue

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

https://huggingface.co/internlm/internlm-xcomposer2-vl-7b


As demonstrated by @YerongLi , this model is a good starter, and it's well-performed. Also, take this in mind:

> We need a new documents for “adding a new VLM in SGLang” (we have adding a new LLM in SGLang already);

Read this docs, and write one while you are developing:

https://zhuanlan.zhihu.com/p/715805386

And later submit one to guide others on supporting new VLM in SGLang.

@mickqian and @yizhang2077 will help you while developing and reviewing your codes.

You can check mick's [latest PR for help.](https://github.com/sgl-project/sglang/pull/2785)

### Related resources

_No response_

---

## Issue #N/A: [Feature] support Kimi VL

**Link**: https://github.com/sgl-project/sglang/issues/5314
**State**: closed
**Created**: 2025-04-12T06:22:24+00:00
**Closed**: 2025-05-09T21:09:45+00:00
**Comments**: 5
**Labels**: good first issue, help wanted

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct

### Related resources

_No response_

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

## Issue #N/A: [Docs] Add docs for running SGLang on AMD

**Link**: https://github.com/sgl-project/sglang/issues/3245
**State**: closed
**Created**: 2025-02-01T00:23:16+00:00
**Closed**: 2025-05-21T15:40:21+00:00
**Comments**: 4
**Labels**: documentation, good first issue, help wanted, amd

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

That has long been waiting, we should add a docs on how to run SGLang on AMD devices.

https://github.com/sgl-project/sglang/issues/3219
https://github.com/sgl-project/sglang/issues/3243
https://github.com/sgl-project/sglang/issues/3200
https://github.com/sgl-project/sglang/pull/3208
https://github.com/sgl-project/sglang/issues/3198

Here is something related. To me, I think we should add a docs on how to:
 
1. configure environment in AMD GPU;
2. how to install sglang;
3. how to run a llama model;
4. how to run deepseek V3 models.

### Related resources

_No response_

---

## Issue #N/A: [Feature] Add arguments mapping between SGLang / vllm / trt-llm

**Link**: https://github.com/sgl-project/sglang/issues/2657
**State**: open
**Created**: 2024-12-30T07:23:00+00:00
**Comments**: 2
**Labels**: documentation, good first issue, help wanted, RLHF

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

This is what I need to do for integrating SGLang into OpenRLHF. OpenRLHF already supports vllm. We need to add sglang. I need to map the server and sampling parameters from vllm to sglang. I think this is a good issue for us to let our users switch smoothly between mainstream engines.

**I attached how I am doing right now. But it may be wrong.**

### Related resources

**The args Mapping from vllm to sglang**

These are the server parameters of vllm:

```python
pretrain,
noset_visible_devices=noset_visible_devices,
trust_remote_code=True,
tensor_parallel_size=tensor_parallel_size,
dtype="bfloat16",
seed=seed + i,
enable_prefix_caching=enable_prefix_caching,
enforce_eager=enforce_eager,
max_model_len

[... truncated for brevity ...]

---

## Issue #N/A: [Tracker] FA3 performance on sm80

**Link**: https://github.com/sgl-project/sglang/issues/5938
**State**: open
**Created**: 2025-05-01T02:14:42+00:00
**Comments**: 2
**Labels**: good first issue, help wanted, high priority, collaboration

### Description

```bash
git clone https://github.com/sgl-project/sglang
cd sglang
pip3 install -e "python[all]"
```

```bash
--attention-backend fa3
```

---

## Issue #N/A: [Feature] Support more multi-modal input for VLM

**Link**: https://github.com/sgl-project/sglang/issues/5964
**State**: open
**Created**: 2025-05-02T02:28:40+00:00
**Comments**: 4
**Labels**: good first issue, help wanted, feature, MLLM

### Description

### Motivation

The current endpoint only supports image data input, limiting its flexibility for diverse VLM use cases. We need additional input formats, particularly for RL applications:
(Could be split into multiple PRs)

- [x] Pre-computed Image Embeddings
- [ ] Pixel Values
- [ ] Pixel Value Range Parameters (min_pixel/max_pixel) for qwen-vl

Welcome to propose more.

#### Benefits

1. Enhanced flexibility for RL workflows
2. Reduced preprocessing overhead
3. Better integration with existing pipelines

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

## Issue #N/A: [Feature] Support PDL on norm in sgl-kernel

**Link**: https://github.com/sgl-project/sglang/issues/5946
**State**: open
**Created**: 2025-05-01T07:41:57+00:00
**Comments**: 4
**Labels**: good first issue, sgl-kernel

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

In previous versions, we updated flashinfer. Flashinfer 0.2.5 supports norm's PDL, but currently, norm's PDL is disabled by default. We would like to modify the code to enable it.

### Related resources

We need change code at `sgl-kernel/python/sgl_kernel`, those who have enable_pdl parameter.

For example:
```python
def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    r"""Root mean square normalization.

    ``out[i] = (input[i] / RMS(input)) * weight[i]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Llama4 fails to run on Python 3.9 (AssertionError)

**Link**: https://github.com/sgl-project/sglang/issues/6232
**State**: closed
**Created**: 2025-05-12T12:05:57+00:00
**Closed**: 2025-06-15T13:15:12+00:00
**Comments**: 1
**Labels**: good first issue, wontfix

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Running llama 4 with Python 3.9 get AssertionError

e.g.`python -m sglang.launch_server --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct --tp 8 --context-length 65536`

The error does not occur in python 3.10, 3.11, 3.12.

### Reproduction

#### Python 3.9 (AssertionError)

```bash
mkdir 3-9-test
cd 3-9-test
uv init --python 3.9
uv a

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

## Issue #N/A: [Feature] Support new Qwen Models

**Link**: https://github.com/sgl-project/sglang/issues/3159
**State**: closed
**Created**: 2025-01-27T01:53:36+00:00
**Closed**: 2025-05-07T16:03:31+00:00
**Comments**: 6
**Labels**: good first issue

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Qwen has released new models as `Qwen2.5-1M`. We should support it ASAP. I have connected the Qwen team for help. If anyone is interested, they can help.

### Related resources

_No response_

---

## Issue #N/A: [Feature] Rewrite docs for LLama 405B and ModelSpace

**Link**: https://github.com/sgl-project/sglang/issues/2743
**State**: closed
**Created**: 2025-01-06T03:00:14+00:00
**Closed**: 2025-05-16T02:58:35+00:00
**Comments**: 3
**Labels**: documentation, good first issue

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

https://sgl-project.github.io/backend/server_arguments.html#use-models-from-modelscope

https://sgl-project.github.io/backend/server_arguments.html#example-run-llama-3-1-405b

These two docs have been out of date for long. We need to move it under `docs/reference` as two separate markdown and verify the content.

### Related resources

No such.

---

## Issue #N/A: [Feature] Add examples for running SGLang on Slurm

**Link**: https://github.com/sgl-project/sglang/issues/3244
**State**: open
**Created**: 2025-01-31T23:55:35+00:00
**Comments**: 0
**Labels**: documentation, good first issue, help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

This has long been discussed. We want to add examples for how to run SGLang on slurm systems. Here is one example for dpsk model. But we need more definitely.

https://github.com/sgl-project/sglang/issues/3206

### Related resources

_No response_

---

## Issue #N/A: [Feature] the stream request returns data without usage.token data

**Link**: https://github.com/sgl-project/sglang/issues/3743
**State**: closed
**Created**: 2025-02-21T06:18:23+00:00
**Closed**: 2025-02-21T18:33:48+00:00
**Comments**: 6
**Labels**: good first issue, help wanted

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

The cost is calculated through the usage.token returned by the HTTP response. Non-stream requests have a usage.token cost, but the returned usage.token for stream results is empty.

🚀 Is it possible to add the number of tokens when the stream returns data? 🚀

tks.

```
data: {"id":"e1b9eecee85b4379a5806f53b6d9ec94","object":"chat.completion.chunk","created":1740117922,"model":"/cephfs/public_model/DeepSeek-R1","choices":[{"index":0,"delta":{"role":null,"content":"。","tool_calls":null},"logprobs":null,"finish_reason":"","matched_stop":null}],"usage":null}

data: {"id":"e1b9eecee85b4379a5806f53b6d9ec94","object":"chat.completion.chunk","created":1740117922,"model":"/cephfs/public_model/DeepSeek-R1","choices":[{"index

[... truncated for brevity ...]

---

