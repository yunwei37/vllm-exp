# feature - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 5
- Closed Issues: 25

### Label Distribution

- feature: 30 issues
- inactive: 17 issues
- high priority: 5 issues
- enhancement: 5 issues
- help wanted: 4 issues
- good first issue: 3 issues
- collaboration: 3 issues
- lora: 3 issues
- RLHF: 2 issues
- MLLM: 2 issues

---

## Issue #N/A: [Feature] Support TRI-ML/prismatic-vlms

**Link**: https://github.com/sgl-project/sglang/issues/1129
**State**: open
**Created**: 2024-08-16T18:15:10+00:00
**Comments**: 2
**Labels**: good first issue, feature, new-model

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

I'm trying to speed up inference for new VLM models on huggingface: https://huggingface.co/TRI-ML/prismatic-vlms/tree/main. I'm wondering if there are additional documentation on how to adapt new models? 

### Related resources

The model I'm trying to adapt is detailed here: https://arxiv.org/pdf/2402.07865. 

---

## Issue #N/A: [Feature] Extend CustomLogitProcessor to Support input_ids in call Method

**Link**: https://github.com/sgl-project/sglang/issues/3524
**State**: closed
**Created**: 2025-02-12T12:48:38+00:00
**Closed**: 2025-06-25T00:20:02+00:00
**Comments**: 6
**Labels**: inactive, feature

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Thanks @hongpeng-guo for PR #2396. After reviewing your work, I'd like to propose an enhancement to the `CustomLogitProcessor`. Specifically, I suggest modifying its `__call__` method to accept `input_ids` as an additional parameter—similar to the implementation in Huggingface (see this [doc](https://huggingface.co/docs/transformers.js/en/api/generation/logits_process#module_generation/logits_process.LogitsProcessor)). This change would allow constraints to be applied conditionally based on the entire history of input tokens, enabling more flexible and context-aware processing.

Thank you for considering this feature request!

### Related resources

[Huggingface LogitsProcessor.](https://huggingface.co/docs/transfo

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] (Willing to PR) Avoid KV cache occupying GPU memory when not used

**Link**: https://github.com/sgl-project/sglang/issues/2542
**State**: closed
**Created**: 2024-12-22T09:07:26+00:00
**Closed**: 2025-03-16T14:34:36+00:00
**Comments**: 43
**Labels**: high priority, collaboration, inactive, feature

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Hi thank you for the library! The use case is that, when doing online PPO, I hope to use SGLang to generate llm completions, and then use RL to do gradient descent on those completions.

The problem is, to do this on a single GPU, the timeline is "SGLang generate - Torch backward - repeat it". Thus, when torch doing backprop, I hope SGLang can free its KV cache memory consumption, otherwise torch will not have enough memory.

Thanks for any suggestions!

### Related resources

_No response_

---

## Issue #N/A: [Feature] support /v1/completions suffix parameter for completion

**Link**: https://github.com/sgl-project/sglang/issues/3429
**State**: closed
**Created**: 2025-02-09T14:30:57+00:00
**Closed**: 2025-04-14T00:19:34+00:00
**Comments**: 7
**Labels**: inactive, feature

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

parameter suffix is not supported in sglang's openapi  v1/completions yet. but it's necessary for code completion.
can I support this?

### Related resources

_No response_

---

## Issue #N/A: [Feature] Support destroying model weights and all KV cache during runtime

**Link**: https://github.com/sgl-project/sglang/issues/3811
**State**: closed
**Created**: 2025-02-24T08:43:23+00:00
**Closed**: 2025-02-24T17:19:50+00:00
**Comments**: 1
**Labels**: feature

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

In the RLHF training process, if the inference engine and the training engine are deployed on the same batch of machines, to save GPU memory, it is necessary to offload the KV Cache and model weights from the gpu memory to the cpu memory.

### Related resources

https://github.com/vllm-project/vllm/pull/11743

---

## Issue #N/A: [Feature] full-duplex audio multimodal service support

**Link**: https://github.com/sgl-project/sglang/issues/3808
**State**: closed
**Created**: 2025-02-24T08:06:44+00:00
**Closed**: 2025-04-28T00:19:27+00:00
**Comments**: 3
**Labels**: inactive, feature

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Does this framework support full-duplex audio multimodal services? The client sends an audio segment, and the server processes it (alternatively, the client can pre-process it and generate embeddings to send directly to the server). The issue I'm encountering is that audio multimodal data cannot use auto prefix caching. Would this result in poor performance, especially in multi-turn conversations?

### Related resources

_No response_

---

## Issue #N/A: [Feature] Can router support prometheus metrics

**Link**: https://github.com/sgl-project/sglang/issues/3393
**State**: closed
**Created**: 2025-02-08T06:42:46+00:00
**Closed**: 2025-04-28T00:19:29+00:00
**Comments**: 3
**Labels**: enhancement, inactive, feature, router

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

K8s is often used to deploy applications online. After the router module is introduced, related service indicator monitoring is also required. Therefore, similar to https://github.com/sgl-project/sglang/pull/1853 provided by the server, does it support the collection of monitoring indicators of the router?

### Related resources

_No response_

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

## Issue #N/A: [Feature Request] Accepting multiple weight updates in update_weights_from_distributed

**Link**: https://github.com/sgl-project/sglang/issues/3646
**State**: closed
**Created**: 2025-02-17T18:38:08+00:00
**Closed**: 2025-05-30T00:19:18+00:00
**Comments**: 4
**Labels**: inactive, feature

### Description

The current sglang implementation only offers one weight update request at a time for `update_weights_from_distributed`. While parameters are provided to the inference server with torch's broadcast function, it is still needed to manually send a HTTP request for each parameter, although the request is very small and only has name, dtype, and shape.

Is it possible to make this also take a list of (named, dtype, shape) for parameters? In this case, one HTTP request would be needed if we would like to update many (or even all) parameters.
 
https://github.com/sgl-project/sglang/blob/714f3e6362791ccc54a8845e5c6261d1e6d156cc/test/srt/test_update_weights_from_distributed.py#L290-L305


---

## Issue #N/A: [Feature] LRU Eviction Strategy for Lora Adapters: Evicting Adapters with Priority

**Link**: https://github.com/sgl-project/sglang/issues/8053
**State**: open
**Created**: 2025-07-15T08:51:27+00:00
**Comments**: 2
**Labels**: high priority, feature

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

In scenarios where Lora adapters are shared, since a single base model is used collectively and there is an upper limit on the number of Lora adapters that a single node can host, we may need to introduce an LRU strategy (similar to VLLM's approach) to evict some adapters. However, the reactivation and deactivation of adapters can affect service quality (SLOs), which may be unacceptable in certain production environments. Therefore, a feature is required to ensure that specific adapters are not automatically evicted by the LRU mechanism.


### Related resources

CC  @Fridge003  @lifuhuang   @lw9527

---

## Issue #N/A: [Feature] Support correctly exit using ctrl+c

**Link**: https://github.com/sgl-project/sglang/issues/4173
**State**: closed
**Created**: 2025-03-07T08:37:28+00:00
**Closed**: 2025-05-07T00:19:03+00:00
**Comments**: 2
**Labels**: inactive, feature

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

If we want to fully terminate the server processes for now, we could only do ctrl+c and sudo kill the list of processes. 
1. Could we make an __exit__() process to automatically kill all the sub-processes when we hit ctrl+c
2. The function should also support when sudo kill one of the sub-processes, the other sub-processes will be automatically killed using the __exit__() method. 

### Related resources

_No response_

---

## Issue #N/A: [Feature] Support Lora for VocabParallelEmbedding layer

**Link**: https://github.com/sgl-project/sglang/issues/3438
**State**: closed
**Created**: 2025-02-09T19:13:40+00:00
**Closed**: 2025-06-30T00:21:14+00:00
**Comments**: 2
**Labels**: inactive, feature, lora

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Support lora for `VocabParallelEmbedding`. Not a trivial task.

### Related resources

_No response_

---

## Issue #N/A: [Feature] Prefill assistant response

**Link**: https://github.com/sgl-project/sglang/issues/3971
**State**: closed
**Created**: 2025-02-28T21:34:21+00:00
**Closed**: 2025-04-21T15:22:27+00:00
**Comments**: 4
**Labels**: good first issue, help wanted, feature

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

OAI API doesn't natively support prefilling an assistants response. vLLM and Aphrodite has the additional support for `continue_final_message` which would be need to have for SGLang to give developers even much more control.

Should be relatively easy for someone to implement. It's simply not allowing chat template EOS to take over in a turn where assistant response is last and this flag is enabled and a generation is requested. This was originally implemented with exact same parameter name in transformers, which became a feature in vLLM and Aphrodite.

### Related resources

https://huggingface.co/docs/transformers/main/en/chat_templating
https://github.com/aphrodite-engine/aphrodite-engine/blob/e64075b8937786311f

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support multimodal models for Native API/ Engine

**Link**: https://github.com/sgl-project/sglang/issues/3545
**State**: closed
**Created**: 2025-02-13T10:39:45+00:00
**Closed**: 2025-02-25T17:52:53+00:00
**Comments**: 0
**Labels**: feature, MLLM

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

add the multimodal support to Native API/ Engine

### Related resources

_No response_

---

## Issue #N/A: [Feature] Support unified paging in multi-lora serving

**Link**: https://github.com/sgl-project/sglang/issues/3647
**State**: open
**Created**: 2025-02-17T19:14:47+00:00
**Comments**: 5
**Labels**: enhancement, inactive, feature, lora

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Currently, SGL doesn't support the unified paging feature proposed by S-LoRA. However, this feature is important for memory management in multi-LoRA serving.

### Related resources

_No response_

---

## Issue #N/A: [Feature] plan to support Block Schedule？ 

**Link**: https://github.com/sgl-project/sglang/issues/2568
**State**: closed
**Created**: 2024-12-24T06:05:46+00:00
**Closed**: 2024-12-25T05:42:06+00:00
**Comments**: 2
**Labels**: feature

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Will the Block Schedule scheme in FlexLLMGen bring greater throughput? Will sglang consider supporting it?

### Related resources

_No response_

---

## Issue #N/A: [Feature] Tool Call Roadmap

**Link**: https://github.com/sgl-project/sglang/issues/6589
**State**: open
**Created**: 2025-05-25T10:29:03+00:00
**Comments**: 6
**Labels**: enhancement, high priority, feature, function-calling

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

## Motivation

Add a list of issues need to resolve in tool call.

## Track for Tool Call Issues

### High Piority

Issues related to accuracy, consistency, and performance.

- [x] [Multiple Tool Call Support for MistralDetector and Qwen25Detector](https://github.com/sgl-project/sglang/issues/6589#issuecomment-2907987558)
#6597 

- [ ] [JSON Double Dumping Behavior](https://github.com/sgl-project/sglang/issues/6589#issuecomment-2907988051)

- [x] [`ToolCallItem.tool_index` not following OpenAI API](https://github.com/sgl-project/sglang/issues/6589#issuecomment-2907988438) 
#6715 
#6655 
#6678 

----

### Medium Priority

Issues that are not immediate, such as features still WIP, or needs refactor, or edge cases.

- [ ] [Tests for 

[... truncated for brevity ...]

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

## Issue #N/A: [Feature] Support phi-3 model

**Link**: https://github.com/sgl-project/sglang/issues/1283
**State**: closed
**Created**: 2024-09-01T05:44:49+00:00
**Closed**: 2024-09-03T04:49:41+00:00
**Comments**: 0
**Labels**: feature

### Description

### Motivation

phi-3 model is popular for its tiny size. we should also support it

### Related resources

_No response_

---

## Issue #N/A: [Feature] Load model weight in parallel

**Link**: https://github.com/sgl-project/sglang/issues/4822
**State**: closed
**Created**: 2025-03-27T17:37:11+00:00
**Closed**: 2025-06-01T00:24:15+00:00
**Comments**: 4
**Labels**: help wanted, inactive, feature

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

We're using [a distributed file system](https://juicefs.com) to store LLM weights in a Kubernetes environment. As a typical design choice, the system is tuned for max parallelism, which behaves relatively poor with single-threaded, sequential reads. Through benchmarking, we found that model loading can be up to 5 times faster by using 8 threads, compared to the current performance of SGLang.

We hope there can be an option to enable parallelism while reading the model weights. It is not so useful for users who store their weights in a physical drive, but could be life-saving for users with distributed storage backend, including S3 (via S3FS).

### Related resources

vLLM uses Run:ai Model Streamer for streaming mod

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] QwQ support

**Link**: https://github.com/sgl-project/sglang/issues/2237
**State**: closed
**Created**: 2024-11-28T08:42:43+00:00
**Closed**: 2024-12-01T10:27:32+00:00
**Comments**: 4
**Labels**: enhancement, feature

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

ref https://qwenlm.github.io/blog/qwq-32b-preview/

### Related resources

_No response_

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

## Issue #N/A: [Feature] support W8A8(FP8) and KV Cache FP8 for DeepSeek V2

**Link**: https://github.com/sgl-project/sglang/issues/1156
**State**: closed
**Created**: 2024-08-19T17:21:17+00:00
**Closed**: 2024-09-01T09:51:32+00:00
**Comments**: 3
**Labels**: feature

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

As titled. Make DeepSeek V2 MLA Faster!

### Related resources

_No response_

---

## Issue #N/A: [Feature] Proposal: Releasing SGLang memory when idle

**Link**: https://github.com/sgl-project/sglang/issues/2583
**State**: closed
**Created**: 2024-12-26T02:23:14+00:00
**Closed**: 2025-03-01T00:18:51+00:00
**Comments**: 13
**Labels**: high priority, inactive, feature

### Description

### Proposal 1: Release KV cache when engine is idle

When using SGLang for generation in a training pipeline (such as PPO), at the phase of running HuggingFace model forward/backward, SGLang currently needs to take a lot of memory even though it does not use it. It would be great to make SGLang use as little memory as possible when it is idle.

Example usage cases:
* Suppose we run OpenRLHF on 8xH100, the currently we may allocate 4xH100 for vllm/SGLang and another 4xH100 for HF model (thanks @zhaochenyang20 for providing this usage scenario).
	* If we make SGLang use little memory when idle, then we can run the same experiment on half number of GPUs (4xH100) by putting those SGLang engines on the same GPUs as HF models.
* Suppose we run PPO on 1xH100 for a 7B model with Adam offloading (thanks @zhaochenyang20 for providing this usage scenario). Then policy (7Bx2) + critic (7Bx2) + ref (7Bx2) + reward (7Bx2) already takes 56B. The current SGLang needs 7Bx2 for weights and some 

[... truncated for brevity ...]

---

## Issue #N/A: How to return reasoning_content from sglang server response?

**Link**: https://github.com/sgl-project/sglang/issues/3428
**State**: closed
**Created**: 2025-02-09T14:21:08+00:00
**Closed**: 2025-05-09T00:18:59+00:00
**Comments**: 5
**Labels**: inactive, feature

### Description

I use docker-compose to deploy locally
```
services:
  sglang1:
    image: lmsysorg/sglang:latest
    container_name: sglang-DeepSeek-R1-Distill-Qwen-7B
    volumes:
      - /root/DeepSeek-R1-Distill-Qwen-7B:/DeepSeek-R1-Distill-Qwen-7B
      # If you use modelscope, you need mount this directory
      # - ${HOME}/.cache/modelscope:/root/.cache/modelscope
    restart: always
    #network_mode: host
    # Or you can only publish port 30000
    ports:
      - 30000:30000
    environment:
      # if you use modelscope to download model, you need set this environment
      SGLANG_USE_MODELSCOPE: true
    entrypoint: python3 -m sglang.launch_server
    command:
      --model-path /DeepSeek-R1-Distill-Qwen-7B/
      --host 0.0.0.0
      --port 30000
      --served-model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
      --grammar-backend xgrammar
    ulimits:
      memlock: -1
      stack: 67108864
    ipc: host
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:30000/healt

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Running multi-node offline engine inference ( via SLURM)

**Link**: https://github.com/sgl-project/sglang/issues/2561
**State**: closed
**Created**: 2024-12-23T15:24:49+00:00
**Closed**: 2025-01-31T23:58:27+00:00
**Comments**: 39
**Labels**: help wanted, collaboration, feature

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

A lot of academic institutions only allow access to larger node clusters via SLURM and it is not immediately clear how would I reuse the code to run Llama 405B BF16 on 2 nodes (by starting a server) to perform offline inference

### Related resources

_No response_

---

## Issue #N/A: [Feature] Allow Serving Requests During CUDA Graph Capture

**Link**: https://github.com/sgl-project/sglang/issues/3902
**State**: closed
**Created**: 2025-02-27T00:22:02+00:00
**Closed**: 2025-05-05T11:23:54+00:00
**Comments**: 4
**Labels**: feature

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

#### Background

When `--enable-cuda-graph`, service restart (after crashes/unexpected exits) currently requires about 10-minute CUDA Graph capture process before becoming operational (longer with torch.compile enabled). This creates significant service downtime despite even with external health check and automatic restart, unless hot-standbys are deployed.

I raised this question in community discussion group and received reply from @ baizhou: Theoretically feasible to let model runner choose whether to replay existing CUDA Graph.

#### Requested Feature

Add an option or fallback mechanism to allow serving requests without CUDA Graph replay during initialization, prioritizing availability over performance.

#### 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support compatibility between Cuda Graph and Lora

**Link**: https://github.com/sgl-project/sglang/issues/3282
**State**: closed
**Created**: 2025-02-04T06:27:19+00:00
**Closed**: 2025-04-29T06:30:45+00:00
**Comments**: 2
**Labels**: inactive, feature, lora

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Currently Lora and Cuda Graph cannot be used at the same time. 

Theoretically they should be compatible.

### Related resources

_No response_

---

## Issue #N/A: [Feature] plan to support medusa?

**Link**: https://github.com/sgl-project/sglang/issues/859
**State**: closed
**Created**: 2024-08-01T02:41:21+00:00
**Closed**: 2024-12-20T00:16:50+00:00
**Comments**: 4
**Labels**: inactive, feature

### Description

### Motivation

plan to support medusa?

### Related resources

_No response_

---

## Issue #N/A: [Feature] GPU inference on AMD Ryzen AI (370HX-890M) iGPU + NPU

**Link**: https://github.com/sgl-project/sglang/issues/3823
**State**: closed
**Created**: 2025-02-24T15:49:18+00:00
**Closed**: 2025-04-26T00:17:55+00:00
**Comments**: 2
**Labels**: inactive, feature, amd

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Ryzen AI devices have been out since mid 2024 yet there's no end user friendly local inference engine that can use the iGPU or the NPU for inference. Some people seem to be able to make it working using hacks but it's still a hit or miss and you need to build your own custom room and hip packages to it to kind of work. 

### Related resources

_No response_

---

