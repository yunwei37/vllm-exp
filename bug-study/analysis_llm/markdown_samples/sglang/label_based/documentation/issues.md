# documentation - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 5
- Closed Issues: 25

### Label Distribution

- documentation: 30 issues
- good first issue: 22 issues
- help wanted: 12 issues
- inactive: 4 issues
- RLHF: 3 issues
- amd: 2 issues
- high priority: 2 issues

---

## Issue #N/A: [Bug] Tried to run DeepSeek V3 by amd instructions

**Link**: https://github.com/sgl-project/sglang/issues/3200
**State**: closed
**Created**: 2025-01-28T22:33:58+00:00
**Closed**: 2025-04-03T00:17:38+00:00
**Comments**: 4
**Labels**: documentation, help wanted, inactive, amd

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I tried to use [AMD instruction](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html) but i have an error.

### Reproduction

After running in a container
```
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --port 30000 --tp 8

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] docs: Improve documentation on how to use EAGLE speculative docoding

**Link**: https://github.com/sgl-project/sglang/issues/3077
**State**: closed
**Created**: 2025-01-23T10:06:08+00:00
**Closed**: 2025-05-24T15:47:25+00:00
**Comments**: 6
**Labels**: documentation, good first issue

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

The recent addition of EAGLE speculative decoding in [here](https://github.com/SafeAILab/EAGLE/pull/173) is powerful. Thank you for creating and maintaining such a useful tool! The existing codebase gives insufficient examples of how it can be used (e.g for Llama3 models, for example) together with `docker compose`. It would be great if another file like https://github.com/sgl-project/sglang/blob/main/docker/compose.yaml can be added to illustrate how the feature can be used in docker environments. Thanks for looking into this issue!

---

## Issue #N/A: [Feature] Add docs for pass in token ids directly

**Link**: https://github.com/sgl-project/sglang/issues/2661
**State**: open
**Created**: 2024-12-30T07:51:00+00:00
**Comments**: 10
**Labels**: documentation, good first issue, RLHF

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

In most of RLHF frameworks, the prompts are pre-tokenized when data processing, so they can directly pass in token ids to the sglang engine rather than the prompts. So we should add docs on how to do this and how to get tokens directly.

### Related resources

No such.

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

## Issue #N/A: [Feature] Add docs for local accuracy tests

**Link**: https://github.com/sgl-project/sglang/issues/2953
**State**: closed
**Created**: 2025-01-17T17:26:41+00:00
**Closed**: 2025-02-02T19:42:53+00:00
**Comments**: 10
**Labels**: documentation, good first issue

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Following this https://github.com/sgl-project/sglang/pull/2951#issuecomment-2598764414

In our test files of backend, `/test/srt`, some of the tests take a great many of time and can't be triggered in our CI for every commits. But some contributors want to change some of the codes, directly related to accuracy. It's better for them to test accuracy that is not covered in CI and report the results. Related tests are:

```bash
export models args
python3 test/srt/test_eval_accuracy_mini.py
```

### Related resources

_No response_

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

## Issue #N/A: [Feature] Add Tutorial for Constraint Decoding

**Link**: https://github.com/sgl-project/sglang/issues/2505
**State**: closed
**Created**: 2024-12-17T22:40:23+00:00
**Closed**: 2025-05-16T02:58:42+00:00
**Comments**: 1
**Labels**: documentation, good first issue

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

A better document for constraint decoding.

### Related resources

https://sgl-project.github.io/backend/openai_api_completions.html#Structured-decoding-(JSON,-Regex)

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

## Issue #N/A: Potential improvements of docs

**Link**: https://github.com/sgl-project/sglang/issues/6505
**State**: open
**Created**: 2025-05-21T19:00:55+00:00
**Comments**: 2
**Labels**: documentation

### Description

Below I collect some things that should be fixed in the docs. 
 
 
* I remember the transformer issue was a while ago but there was a PR to fix it. I didn't observe it for long time so maybe we can remove this?

<img width="798" alt="Image" src="https://github.com/user-attachments/assets/eb148283-d052-41de-a99a-0a90896f4c6d" />
 
* The same text appears two times, I think we can remove one

<img width="856" alt="Image" src="https://github.com/user-attachments/assets/203e1bfc-7237-44b6-88ff-f9d1fd4f6edf" />

* The text should be adjusted, the roadmap issue is closed

<img width="840" alt="Image" src="https://github.com/user-attachments/assets/9c7b2752-4111-47e4-b452-bf74fe78bd66" />

* This should be rephrased

<img width="823" alt="Image" src="https://github.com/user-attachments/assets/c6ad31ae-ee7b-4913-ae66-74b24a5e2cb4" />

* The Backend Section is quiet large. maybe we can move the four markdowns here to a dedicated section as the other parts of section are more hands on.

<img wid

[... truncated for brevity ...]

---

## Issue #N/A: [Docs] Document how to configure shared memory for multi GPU deployments

**Link**: https://github.com/sgl-project/sglang/issues/4259
**State**: closed
**Created**: 2025-03-10T09:37:19+00:00
**Closed**: 2025-03-27T16:35:57+00:00
**Comments**: 5
**Labels**: documentation, good first issue

### Description

This is a copy of https://github.com/sgl-project/sgl-project.github.io/issues/5. I did not realize the documentation content is generated, so it seems more likely the request belongs here... (?)

The [documentation](https://docs.sglang.ai/backend/server_arguments.html#tensor-parallelism) states

`python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --tp 2`

is a way to enable multi-GPU tensor parallelism. However one must think how the processes (?) communicate together, usually there's a shared memory setup needed. And if this is not properly set, one might run into issues like:

```
torch.distributed.DistBackendError: NCCL error in: ../torch/csrc/distributed/c10d/NCCLUtils.cpp:81, unhandled system error (run with NCCL_DEBUG=INFO for details), NCCL version 2.21.5
ncclSystemError: System call (e.g. socket, malloc) or external library call failed or device error.
Last error:
Error while creating shared memory segment /dev/shm/nccl-vzIpS6 (size 9637888)
```
whe

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Step-by-Step Guide to Use SGLang on NVIDIA Jetson Orin platform

**Link**: https://github.com/sgl-project/sglang/issues/3182
**State**: closed
**Created**: 2025-01-27T16:45:59+00:00
**Closed**: 2025-02-21T12:45:13+00:00
**Comments**: 9
**Labels**: documentation, good first issue

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Hello Sglang team,

Great inference engine! 

Just FYI, I was able to successfully run SGLang on the NVIDIA Jetson AGX Orin Developer Kit. 

For more details, please check here: https://github.com/shahizat/SGLang-Jetson



### Related resources

_No response_

---

## Issue #N/A: [Feature] Rewrite Sampling Parameter

**Link**: https://github.com/sgl-project/sglang/issues/3165
**State**: closed
**Created**: 2025-01-27T03:31:18+00:00
**Closed**: 2025-02-17T21:31:21+00:00
**Comments**: 3
**Labels**: documentation, good first issue

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

This is redundant: https://docs.sglang.ai/references/sampling_params.html

### Related resources

_No response_

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

## Issue #N/A: [Feature] Rewrite Supported Model Docs

**Link**: https://github.com/sgl-project/sglang/issues/3595
**State**: closed
**Created**: 2025-02-15T20:04:02+00:00
**Closed**: 2025-04-30T22:14:54+00:00
**Comments**: 3
**Labels**: documentation, good first issue, help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

https://docs.sglang.ai/references/supported_models.html

This doc is a bit chaotic. We should reorganize it.

@simveit 

### Related resources

_No response_

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

## Issue #N/A: [Feature] Improve Native API docs

**Link**: https://github.com/sgl-project/sglang/issues/5104
**State**: closed
**Created**: 2025-04-06T22:22:22+00:00
**Closed**: 2025-06-07T00:19:08+00:00
**Comments**: 3
**Labels**: documentation, help wanted, inactive

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

<img width="876" alt="Image" src="https://github.com/user-attachments/assets/edb0f5be-7875-4fa4-b6a3-b7742e5060b1" />

<img width="846" alt="Image" src="https://github.com/user-attachments/assets/f322f548-e67a-466f-b9d2-e495b29af173" />

The output of https://docs.sglang.ai/backend/native_api.html#Capture-expert-selection-distribution-in-MoE-models 

is too long, try to only print first 10 lines:

```python
output_file = glob.glob("expert_distribution_*.csv")[0]
with open(output_file, "r") as f:
    print_highlight("Content of dumped record:")
    for line in f:
        print_highlight(line.strip())
```

@simveit 

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

## Issue #N/A: [Feature] Add QWQ’s Benchmark Code for Inference Performance Evaluation

**Link**: https://github.com/sgl-project/sglang/issues/4394
**State**: closed
**Created**: 2025-03-13T16:08:50+00:00
**Closed**: 2025-04-02T06:04:44+00:00
**Comments**: 11
**Labels**: documentation, good first issue, help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Reasoning models typically generate many tokens, making them suitable for evaluating the performance of inference frameworks. As a result, they serve as a valuable benchmark for performance comparisons.

QWQ has open-sourced its benchmarking code, which could be integrated into the Sglang benchmark suite. This addition would help users compare the performance of different inference frameworks more conveniently when running inference models.

Would it be possible to add support for this benchmark in Sglang?

Reference:
	•	[QWQ’s benchmark code](https://github.com/QwenLM/QwQ/tree/main/eval)

Potential Benefits:
	•	Provides a standardized way to evaluate reasoning performance.
	•	Helps users compare different inferenc

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Clear PAT_TOKEN in CI

**Link**: https://github.com/sgl-project/sglang/issues/2659
**State**: closed
**Created**: 2024-12-30T07:44:56+00:00
**Closed**: 2025-03-01T00:18:50+00:00
**Comments**: 1
**Labels**: documentation, inactive

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

![image](https://github.com/user-attachments/assets/d62f4957-2802-4068-9c16-fbcaee2584f4)

@shuaills Would you like to take this? Pretty easy.

### Related resources

_No response_

---

## Issue #N/A: [Feature] Docs: Collect all the commands for DeepSeek in SGlang

**Link**: https://github.com/sgl-project/sglang/issues/2744
**State**: closed
**Created**: 2025-01-06T03:07:32+00:00
**Closed**: 2025-05-21T12:41:57+00:00
**Comments**: 2
**Labels**: documentation, good first issue

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

SGLang has unique features for the deepseek model, but they are scattered across many blogs. We need to collect them and create a unique file under `docs/reference/deepseek.md`. This document should contain all the optimizations of the deepseek model in SGLang and provide links to the original files, just like what we did in the following URL:

https://sgl-project.github.io/references/contribution_guide.html#running-unit-tests-adding-to-ci

### Related resources

https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3

https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models

https://lmsys.org/blog/2024-09-04-sglang-v0-3/#deepseek-multi-head-latent-attention-

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Link error in SGLang Sampling Docs

**Link**: https://github.com/sgl-project/sglang/issues/2551
**State**: closed
**Created**: 2024-12-23T02:58:24+00:00
**Closed**: 2024-12-26T15:12:28+00:00
**Comments**: 1
**Labels**: documentation

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

https://sgl-project.github.io/references/sampling_params.html

This link is an error and I am pondering why it should refer to.


![image](https://github.com/user-attachments/assets/5401ab02-2cbd-476f-bef9-6ac0c7eda58e)



### Reproduction

no such

### Environment

no such

---

## Issue #N/A: [Bug] [OpenAI compatible API] Chunks of tokens aren't being split into separate indexes when specifying n > 1 generations

**Link**: https://github.com/sgl-project/sglang/issues/2912
**State**: closed
**Created**: 2025-01-16T07:09:37+00:00
**Closed**: 2025-03-24T00:18:31+00:00
**Comments**: 5
**Labels**: documentation, inactive

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

This code works as expected; we stream outputs like normal.

```python
from openai import OpenAI

llm = OpenAI(...)

r = llm.chat.completions.create(
     model="...",
     messages=[...],
     stream=True,
     n=1
)

for chunk in r:
    print(chunk.choices[0].delta.content, end="")
```
```txt
This is a normal response.

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] fix docs in Streaming-Synchronous-Generation

**Link**: https://github.com/sgl-project/sglang/issues/3164
**State**: closed
**Created**: 2025-01-27T03:27:21+00:00
**Closed**: 2025-05-24T15:48:12+00:00
**Comments**: 2
**Labels**: documentation, good first issue

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Fix this https://docs.sglang.ai/backend/offline_engine_api.html#Streaming-Synchronous-Generation

To long generated contents and it seems to be wrong.

### Related resources

_No response_

---

## Issue #N/A: [Feature] Add docs for all the deepseek model 

**Link**: https://github.com/sgl-project/sglang/issues/2698
**State**: closed
**Created**: 2025-01-01T23:03:55+00:00
**Closed**: 2025-05-24T21:22:36+00:00
**Comments**: 6
**Labels**: documentation, good first issue

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Add it under `https://sgl-project.github.io/references/`. Gather all the optimization, usage, and parameters in one doc and name it `deepseek.md`. This can be a markdown, but please verify it locally as what you can.

### Related resources

https://lmsys.org/blog/

---

## Issue #N/A: [Docs] Remove redundant CI when docs merged into main

**Link**: https://github.com/sgl-project/sglang/issues/3901
**State**: closed
**Created**: 2025-02-26T23:20:18+00:00
**Closed**: 2025-02-27T06:13:34+00:00
**Comments**: 2
**Labels**: documentation, good first issue, help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

<img width="640" alt="Image" src="https://github.com/user-attachments/assets/4e7bc601-683f-494f-8d47-c3c80f71e309" />

The execute notebook CI is to test the correctness of PRs. If one CI is merged into main, it should not be triggered, but just use execute-and-deploy. We should fix our CI workflow.

### Related resources

_No response_

---

## Issue #N/A: [Feature] Add docs for Offline Engine token-in token-out

**Link**: https://github.com/sgl-project/sglang/issues/2968
**State**: closed
**Created**: 2025-01-18T20:13:02+00:00
**Closed**: 2025-05-26T02:22:39+00:00
**Comments**: 3
**Labels**: documentation, good first issue, RLHF

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

We have token-in-token-out pipeline in Sever already. But we need it for engine also.

```python
class SGLangLLMRayActor:
    def __init__(self, *args, **kwargs):
        self.llm = sglang.Engine(
            model_path=args[0],
            trust_remote_code=kwargs.get("trust_remote_code", True),
            dtype=kwargs.get("dtype", "auto"),
            tp_size=kwargs.get("tensor_parallel_size", 1),
            device="cuda",
            random_seed=kwargs.get("seed", 42),
            disable_radix_cache=not kwargs.get("enable_prefix_caching", False),
            disable_cuda_graph=not kwargs.get("enforce_eager", False),
            disable_cuda_graph_padding=not kwargs.get("enable_prefix_caching", False),
       

[... truncated for brevity ...]

---

## Issue #N/A: Parallelism with `run_batch` vs `fork`

**Link**: https://github.com/sgl-project/sglang/issues/295
**State**: closed
**Created**: 2024-03-13T20:35:36+00:00
**Closed**: 2024-04-07T10:09:39+00:00
**Comments**: 1
**Labels**: documentation

### Description

First of all, great work!

The frontend seems to support two kinds of parallel processing: batching and forking.

From the docs and paper, it is not entirely clear to me how they differ and how they are handled under the hood. Do they both launch separate threads that make requests to the server, which then does continuous batching? Or is there more to it?

From a practical standpoint, what are the considerations when both `run_batch` and `fork` are possible for the use case? Are there advantages/disadvantages besides fork being more flexible?

Is it safe to combine the two? Would the total number of threads be `num_threads * num_forks`?

---

