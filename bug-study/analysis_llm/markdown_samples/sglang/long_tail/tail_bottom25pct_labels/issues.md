# tail_bottom25pct_labels - issues

**Total Issues**: 11
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 3
- Closed Issues: 8

### Label Distribution

- high priority: 5 issues
- good first issue: 2 issues
- wontfix: 2 issues
- wip: 2 issues
- inactive: 2 issues
- performance: 1 issues
- help wanted: 1 issues
- microsoft: 1 issues
- linkedin: 1 issues
- hicache: 1 issues

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

## Issue #N/A: [Feature] P100/Cuda 6.1 support

**Link**: https://github.com/sgl-project/sglang/issues/1062
**State**: closed
**Created**: 2024-08-12T20:29:42+00:00
**Closed**: 2024-08-13T04:31:55+00:00
**Comments**: 1
**Labels**: wontfix

### Description

### Motivation

As per https://github.com/sgl-project/sglang/issues/1059 , P100/pascal/6.1 support is not currently a target. This feature request is an official request to support it. This GPU is the least expensive hardware that will run modern LLMs, and is a common GPU in both academia and common use.

This issue was created as the original was locked with the cryptic phrase, "It makes nonsense for me.", the meaning of which was not clear in context. This issue is intended to be a place where the community can discuss support for these GPUs, as well as petition for support.

### Related resources

_No response_

---

## Issue #N/A: [Feature] optimize moe_align_block_size_kernel

**Link**: https://github.com/sgl-project/sglang/issues/2732
**State**: closed
**Created**: 2025-01-05T05:56:21+00:00
**Closed**: 2025-03-25T04:11:57+00:00
**Comments**: 7
**Labels**: good first issue, high priority, wip, performance

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

The original version performs poorly and needs optimization. I suggest rewriting a new implementation.

https://github.com/sgl-project/sglang/blob/main/sgl-kernel/src/sgl-kernel/csrc/moe_align_kernel.cu

### Related resources

_No response_

---

## Issue #N/A: [Feature] Make vLLM optional in model code

**Link**: https://github.com/sgl-project/sglang/issues/1673
**State**: closed
**Created**: 2024-10-15T06:49:05+00:00
**Closed**: 2025-03-03T23:17:23+00:00
**Comments**: 3
**Labels**: wip

### Description

### UPDATE(11/23/2024)

Currently, @james-p-xu  is removing rope, @yizhang2077  is removing distributed, @HandH1998 is removing weight loader. Optimistically, we can remove these dependencies by the end of the month and make quant optional (try import). cc @merrymercy @Ying1123 

### Motivation

This is a tracker of removing vLLM dependencies in general model code (not considering quantization). This is our current  import from vLLM, and we want to remove all them.

```python
from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
   ParallelLMHead,
   VocabParallelEmbedding,
)
```

### Tracker

- [x] Remove `CacheConfig`: https://github.com/sgl-project/sglang/pull/1658
- [x] Remove RoPE: https://github.com/flashinfer-ai/flashinfer/issues/530
- [x] Remove `get_tensor_model_parallel_world_size`
- [x] Remove `Para

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Phi-4-MM support

**Link**: https://github.com/sgl-project/sglang/issues/6544
**State**: open
**Created**: 2025-05-23T04:17:59+00:00
**Comments**: 0
**Labels**: help wanted, high priority, microsoft

### Description

### Update

Currently we have added text & vision support. 

Repeated MMMU benchmark runs range between 53.6 - 55.5, consistent with the the benchmark reported in the original paper (55).

**Known limitations:** (See *Execution Plan* before for full list):

1. Audio capabilities: currently we do not support audio at all. 
2. ~~LoRA / Image quality: Phi4MM depends on LoRA for full image capability, but there is some compatibility issues with the native SGL LORA solution. We are working on solving it by refactoring / generalizing SGL LoRA capabilities.~~ Fixed with #6585, #6734, #6861)
3. Token: Phi4MM supports two types of image token conventions (`<|image1|>` and `<|endoftext10|>`), currently we only support  the latter. If you use the default chat template, it will automatically pick up the supported one.

### Motivation

Supporting the Phi4 Multimodal model (https://[huggingface.co/microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) in SGL

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Generative Score API

**Link**: https://github.com/sgl-project/sglang/issues/5973
**State**: open
**Created**: 2025-05-02T10:43:23+00:00
**Comments**: 6
**Labels**: high priority, linkedin

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Similar to the cross-encoder Score API proposed here: https://github.com/sgl-project/sglang/issues/5577

Goal is to score items "generatively" using decoder-only models.

E.g. "Given a user liked A, B, and C, will the user like this item? Please answer "yes" or "no." The item is: D"

### API
```
{
  "text_1": [
    "Given a user liked A, B, and C, will the user like this item? Please answer "yes" or "no." The item is:",
  ],  
"text_2": [
     "D",
     "E"
   ],
  "positiveToken": "yes",
  "negativeToken": "no"
}
```

Returns: 

```
{
  "scores": [
    0.874,
    0.231
  ]
}
```

### Related resources

Original idea comes from this paper: [Holistic Evaluation of Language Models](https://arxiv.org/pdf/2211.09110) w

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] hierarchical_cache oom

**Link**: https://github.com/sgl-project/sglang/issues/5372
**State**: closed
**Created**: 2025-04-14T09:26:57+00:00
**Closed**: 2025-04-21T18:46:48+00:00
**Comments**: 5
**Labels**: hicache

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

hi~ @xiezhq-hermann You are the main contributor to hierarchical cache, thank you for your great work！  I have a few questions about hierarchical cache, I'm very confused so I'm looking for your help.

Recently we want to try to use hierarchical cache, before that, for DeepSeek R1 , our online args `--mem-fraction-static` is 0.95.

- When 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Respect max_completion_tokens

**Link**: https://github.com/sgl-project/sglang/issues/3531
**State**: closed
**Created**: 2025-02-12T17:52:38+00:00
**Closed**: 2025-02-13T19:23:21+00:00
**Comments**: 2
**Labels**: duplicate, feature

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Currently the OpenAI compatible API only respects the old `max_tokens` request argument. The updated spec introduces `max_completion_tokens`.

I can send a PR adding support for the new argument name and just change the code here:
https://github.com/sgl-project/sglang/blob/8616357a97c5f68eca194dfbeef0ae51943032ef/python/sglang/srt/openai_api/adapter.py#L512

to `request.max_completion_tokens or request.max_tokens`

### Related resources

_No response_

---

## Issue #N/A: [Feature] Add Model Hooks for Accessing and Customizing Model Activations

**Link**: https://github.com/sgl-project/sglang/issues/3266
**State**: closed
**Created**: 2025-02-03T05:44:46+00:00
**Closed**: 2025-04-05T00:17:32+00:00
**Comments**: 4
**Labels**: inactive, research

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

## Description
It would be beneficial to introduce model hooks that allow users to access and modify model activations. This feature would enable greater flexibility for tasks such as visualization, debugging, and custom processing of intermediate representations.

## Use case
* Extract intermediate outputs for interpretability analysis, such as [LogitLens-style investigations](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens).
* Expose internal activations, enabling users to cache activations and implement functions to edit, remove, or replace them dynamically during inference, for example [representation engineering](https://github.com/andyzoujm/representation-engineering).

While

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] RFC for adding CPU support for SGLang

**Link**: https://github.com/sgl-project/sglang/issues/2807
**State**: open
**Created**: 2025-01-09T07:58:45+00:00
**Comments**: 13
**Labels**: enhancement, high priority, intel, cpu

### Description

### Motivation

Hi, SGLang folks! This is Mingfei from intel pytorch team, our team helps optimize PyTorch performance on CPU. I am also the PyTorch module maintainer for cpu performance. We would like to contribute to SGLang for CPU enabling and performance optimization.

### Targets
Our primary target is to optimize SGLang performance on Intel Xeon Scalable Processors (x86 server CPUs).
* Optimization will be focusing on Xeon with [Intel® Advanced Matrix Extensions](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html) support, including Sapphire Rapids(4th gen), Emerald Rapids(5th gen), Granite Rapids(6th gen).
* Native implementations or fallbacks will be provided for CPUs with other ISA to make it functional.
* Providing good performance per dollar.

### Limitations

* Kernels written in **avx512** and **amx-bf16**, requires **GCC11** or above.
* **BFloat16/Float16** will be enabled at the same time on CPU, but we only 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Request to Include flashinfer as a Dependency for sglang Installation

**Link**: https://github.com/sgl-project/sglang/issues/2578
**State**: closed
**Created**: 2024-12-25T20:54:25+00:00
**Closed**: 2025-02-27T00:17:01+00:00
**Comments**: 4
**Labels**: high priority, inactive, dependencies

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

I would like to request a feature to make flashinfer automatically installed when sglang is installed. This would streamline the installation process for users and ensure that all necessary dependencies are correctly set up without requiring additional manual steps.

### Related resources

_No response_

---

