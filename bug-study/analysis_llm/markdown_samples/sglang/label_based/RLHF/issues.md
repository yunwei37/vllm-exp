# RLHF - issues

**Total Issues**: 11
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 2
- Closed Issues: 9

### Label Distribution

- RLHF: 11 issues
- inactive: 4 issues
- documentation: 4 issues
- good first issue: 4 issues
- high priority: 3 issues
- help wanted: 3 issues
- feature: 2 issues
- amd: 1 issues
- MLLM: 1 issues
- enhancement: 1 issues

---

## Issue #N/A: [Feature] support abort ongoing request

**Link**: https://github.com/sgl-project/sglang/issues/5963
**State**: closed
**Created**: 2025-05-02T02:04:47+00:00
**Closed**: 2025-05-25T23:36:54+00:00
**Comments**: 2
**Labels**: high priority, RLHF

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

It would be great if we can have an endpoint to support ending certain requests with `rid`.

The main use case is that in rl training, we may do over-sampling, and as long as the number of generated responses meets the required batch size, we can stop the remaining requests.

A reference implementation would be the `AsyncLLM.abort` in vllm ( https://github.com/vllm-project/vllm/blob/296c6572dd1f76b31b93be19e550790afcfb8843/vllm/v1/engine/async_llm.py#L348).

I'd love to help :)

### Related resources

_No response_

---

## Issue #N/A: [Bug] SGLang on ROCm - NameError: name 'torch_memory_saver' is not defined

**Link**: https://github.com/sgl-project/sglang/issues/5093
**State**: closed
**Created**: 2025-04-05T23:11:55+00:00
**Closed**: 2025-06-08T00:21:35+00:00
**Comments**: 7
**Labels**: high priority, inactive, amd, RLHF

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug
[RCOm Docker - `lmsysorg/sglang:v0.4.4.post3-rocm630-srt`]
The issue arises from here:
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/torch_memory_saver_adapter.py#L48

1. In line 6, if the code fails to import torch_memory_saver, it just bypasses instead of triggering any error. Thus, if the code calls line46 class and u

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support token-in-token-out for Vision LM

**Link**: https://github.com/sgl-project/sglang/issues/3871
**State**: closed
**Created**: 2025-02-26T04:35:56+00:00
**Closed**: 2025-04-29T00:18:49+00:00
**Comments**: 10
**Labels**: inactive, RLHF, MLLM

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Considering what we need in LLM RLHF, rollout engine just needs token in, and give token out.

We are working on VLM RLHF with veRL, could we support VLM token-in-token-out. Here is something maybe useful:

`test/srt/test_skip_tokenizer_init.py`: this is for LLM.

I actually do not know how to get token of VLM 😂

Hope to get the answer.

### Related resources

_No response_

---

## Issue #N/A: [Bug] update_weights_from_tensor raise EOFError when TP>1

**Link**: https://github.com/sgl-project/sglang/issues/3726
**State**: closed
**Created**: 2025-02-20T07:57:02+00:00
**Closed**: 2025-02-24T17:12:54+00:00
**Comments**: 8
**Labels**: RLHF

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

 An EOFError error was raised when using `update_weights_from_tensor` at TP>4, it seens the data deserialize before the full data received.

Python error trace info:
```
Traceback (most recent call last):                                                                                                                        
  File "/usr/lib

[... truncated for brevity ...]

---

## Issue #N/A: Support for saving sharded checkpoints?

**Link**: https://github.com/sgl-project/sglang/issues/3209
**State**: closed
**Created**: 2025-01-30T03:07:40+00:00
**Closed**: 2025-03-14T16:03:28+00:00
**Comments**: 13
**Labels**: help wanted, RLHF

### Description

Does sglang support sharded checkpoints? I see in here https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py#L492 that there is a loader and it recommends using `examples/save_sharded_state.py` to save the sharded state, but this file doesn't exist. 

Does it refer to this one from vllm https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/save_sharded_state.py? 

Also the load-format doesn't have a choice for sharded_state https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py#L315, is that a typo or it's not supposed to be used?

My real problem is that I'm trying to load DeepSeek-R1 and it takes a very long time. I have a sharded checkpoint that vllm can load instantly, but sglang raises the following error (after I add "sharded_state" to choices in launcher to avoid error right away)


```
Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 17

[... truncated for brevity ...]

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

## Issue #N/A: How to obtain the hidden states of generated tokens?

**Link**: https://github.com/sgl-project/sglang/issues/2668
**State**: closed
**Created**: 2024-12-30T11:08:54+00:00
**Closed**: 2025-03-01T00:18:53+00:00
**Comments**: 7
**Labels**: inactive, feature, RLHF

### Description

Thank you for your outstanding work! I was wondering if there’s a way to access the hidden states for each generated token at every layer. Many thanks!

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

## Issue #N/A: [Feature] Rewrite the SRT Backend docs

**Link**: https://github.com/sgl-project/sglang/issues/2660
**State**: closed
**Created**: 2024-12-30T07:49:17+00:00
**Closed**: 2025-05-24T21:27:16+00:00
**Comments**: 3
**Labels**: documentation, good first issue, help wanted, RLHF

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

This doc has been outdated for a long time:

https://sgl-project.github.io/backend/backend.html#backend-sglang-runtime-srt

1. Only keep an explanation for server arguments and give the link to sampling parameters.
2. Add essential explanation for server arguments. Remember to add these kinds of arguments. https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models
3. A group of parameters have ##, ### is not allowed.
4. Use Models From ModelScope and Run Llama 3.1 405B move to reference, and potentially adds docs for deepseek.
5. change main readme.md.


### Related resources

No such.

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

## Issue #N/A: [Feature] (Willing to PR) Proposal: Drop-in fast replacement of `PreTrainedModel.generate`

**Link**: https://github.com/sgl-project/sglang/issues/2569
**State**: closed
**Created**: 2024-12-24T06:18:24+00:00
**Closed**: 2025-03-30T00:19:36+00:00
**Comments**: 9
**Labels**: enhancement, high priority, collaboration, inactive, feature, RLHF

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Hi thanks for the lib! Currently, a lot of code uses `model.generate()`, such as TRL's PPOTrainer, etc. If we can make a drop-in replacement of it using SGLang, then everyone can very easily speed up their code related to generation. For example, TRL's PPOTrainer, OpenRLHF's train_ppo.py (not the train_ppo_ray.py which is more for distributed training). IMHO there are many places this can be useful - many online RL algorithm can benefit from this.

As for when to update SGLang weight from HF weight, most naive solution may be, we update weights *every* time the generate is called. This may not be a big problem, because we can configure the PPO batch size to be so huge that the model.generate is only called

[... truncated for brevity ...]

---

