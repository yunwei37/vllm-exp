# feature_request - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 6
- Closed Issues: 24

### Label Distribution

- feature request: 30 issues
- stale: 12 issues
- tpu: 2 issues
- keep-open: 1 issues
- good first issue: 1 issues

---

## Issue #N/A: [Feature]: Support Python 3.12

**Link**: https://github.com/vllm-project/vllm/issues/6877
**State**: closed
**Created**: 2024-07-28T21:26:05+00:00
**Closed**: 2024-08-02T20:51:23+00:00
**Comments**: 2
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

I believe we eventually need to support 3.12 in the future. Right now I believe Pytorch just added support for Python 3.12, but ray still does not support Python 3.12. Let's use this issue to keep track on this.

### Alternatives

_No response_

### Additional context

_No response_

---

## Issue #N/A: [Feature]: Support for LoRA for Pooling Models

**Link**: https://github.com/vllm-project/vllm/issues/13679
**State**: closed
**Created**: 2025-02-21T17:05:48+00:00
**Closed**: 2025-03-18T12:07:02+00:00
**Comments**: 1
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Currently vLLM does not support LoRA for Pooling models (according to #12808)
I wanted to understand how can we add the support given we can use similar code structure like generation models.

What all things need to be taken care if I need to add the support. Any references will be helpful.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature][Hardware][TPU]: Add Recompilation Check for vLLM on TPU

**Link**: https://github.com/vllm-project/vllm/issues/14580
**State**: closed
**Created**: 2025-03-10T22:31:05+00:00
**Closed**: 2025-03-25T16:59:34+00:00
**Comments**: 1
**Labels**: feature request, tpu

### Description

### 🚀 The feature, motivation and pitch

Ideally, post-warmup, no further compilation should occur. However, PyTorch/XLA's implicit compilation can lead to excessive recompilation during LLM serving, impacting performance. We can add an option to detect recompilation after warmup, requiring a PyTorch/XLA method like xm.num_graph_hash() to track the number of captured graphs. This number should remain constant post-warmup if no recompilation occurs.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: Limit total GPU memory

**Link**: https://github.com/vllm-project/vllm/issues/20256
**State**: open
**Created**: 2025-06-30T12:52:05+00:00
**Comments**: 8
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Even with appropriate arguments to lower memory usage specified, as suggested in docs:

```
--enforce-eager
--max-model-len=8192
--max-num-batched-tokens=8192
--max-num-seqs=16
```

the rest of the GPU memory still gets eaten up by KV cache, so even when running a tiny 1B model that needs 3 GB of VRAM, VLLM takes 71GB on A100 to run this model. This makes VLLM an impractical solution for mixed-GPU clusters.

Suggestions:
- add flag to specify max gpu memory in absolute units
- add ability to explicitly limit/disable KV cache

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Feature request: prompt lookup decoding

**Link**: https://github.com/vllm-project/vllm/issues/1802
**State**: closed
**Created**: 2023-11-27T19:04:37+00:00
**Closed**: 2025-03-10T01:52:03+00:00
**Comments**: 5
**Labels**: feature request, stale

### Description

Prompt lookup decoding (PLD) is a variant of speculative decoding that replaces the draft model with a prefix lookup in the current sequence, resulting in a 2-4x throughput boost for input-grounded tasks like summarization and code modification.

Because PLD doesn't require a secondary model, it might be easier to implement in VLLM?

See https://github.com/apoorvumang/prompt-lookup-decoding for details.


---

## Issue #N/A: GPTQ / Quantization support?

**Link**: https://github.com/vllm-project/vllm/issues/174
**State**: closed
**Created**: 2023-06-21T02:40:47+00:00
**Closed**: 2024-03-06T09:01:49+00:00
**Comments**: 19
**Labels**: feature request

### Description

Will vLLM support 4-bit GPTQ models?

---

## Issue #N/A: [Feature]: How to run speculative models with tensor parallelism?

**Link**: https://github.com/vllm-project/vllm/issues/10562
**State**: closed
**Created**: 2024-11-22T03:30:54+00:00
**Closed**: 2025-03-23T02:08:46+00:00
**Comments**: 3
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

I noticed that the current speculative mode does not support tp from this link (https://docs.vllm.ai/en/stable/models/spec_decode.html). 

However, not supporting TP will greatly limit the choice of speculative models. I would like to know why there is no TP support for speculative models. I am trying to read and modify this part of the code, but I don't understand why the scorer model can support TP, but the speculative model cannot. What are the considerations in system design?

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: Inquiry about Multi-modal Support in VLLM for MiniCPM-V2.6

**Link**: https://github.com/vllm-project/vllm/issues/7546
**State**: closed
**Created**: 2024-08-15T06:36:05+00:00
**Closed**: 2024-08-15T06:49:48+00:00
**Comments**: 19
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

I am currently exploring the capabilities of the VLLM library and am interested in understanding its support for multi-modal inputs, particularly for models like MiniCPM-V2.6. I would like to know if VLLM is designed to handle multi-image and video inputs for such models.

### Alternatives

1. **Model of Interest**: MiniCPM-V2.6
2. **Types of Input**: Multi-image and video
3. **Current Understanding**:
   - I have reviewed the documentation and initial examples provided with VLLM.
  - It seems that both `multiple 'image_url' input` and `list value in image_url` is currently not supported.
  - However, I am not sure if it supports the processing of multiple images or videos as input to a model like MiniCPM-V2.6.
## Questions
 1. Does VLLM support the integration of MiniCPM-V2.6 for processing multi-image and video inputs?
 2. If yes, could you provide an example or a guide on how to set up and use this feature?
 3. If not, are there any 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Build and publish Neuron docker image

**Link**: https://github.com/vllm-project/vllm/issues/4838
**State**: open
**Created**: 2024-05-15T15:27:17+00:00
**Comments**: 4
**Labels**: feature request, keep-open

### Description

### 🚀 The feature, motivation and pitch

It seems like the current docker images don't support Neuron (Inferentia).
It would be very helpful if there was a tested, managed Neuron docker image to use.
While at the same subject, it would be even better if some documentation would be added on running vLlm Neuron using containers.

### Alternatives

DJL?

### Additional context

_No response_

---

## Issue #N/A: [Feature]: Model execution timeout

**Link**: https://github.com/vllm-project/vllm/issues/20950
**State**: open
**Created**: 2025-07-14T22:50:17+00:00
**Comments**: 0
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Currently when there is a bug in the path of model execution, it could hang indefinitely and user may not get timely and useful feedback. It would be useful to have a timeout mechanism and signal the user.

Currently RayDistributedExecutor with Compiled Graph supports a timeout, but vLLM could benefit from supporting this in executor in general.

See [discussion](https://vllm-dev.slack.com/archives/C08CBAP9BUG/p1752532064610089?thread_ts=1752477597.112679&cid=C08CBAP9BUG)

cc @stephanie-wang  @youkaichao 

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: Per-sequence speculative decoding

**Link**: https://github.com/vllm-project/vllm/issues/17984
**State**: open
**Created**: 2025-05-12T08:31:19+00:00
**Comments**: 5
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch


 ### **1. Problem**

Currently, increasing batch size in vLLM's Speculative Decoding inference causes inefficiency.
When using the LLaMA 1B SSM model on the LLaMA 70B Original model, a performance reversal occurs at Batchsize 32.
In addition, when the _num_speculative_tokens_ (SL; speculative length) is large, the inefficiency increases further as the batch size increases (Fig. 2).

vLLM was also aware of the need for optimization for this. (https://docs.google.com/document/d/1T-JaS2T1NRfdP51qzqpyakoCXxSXTtORppiwaj5asxA/edit?tab=t.0#heading=h.kk7dq05lc6q8)

<img src="https://github.com/user-attachments/assets/2cf95b49-e528-46eb-bd9b-f053cc057868" width="600" height="auto">

<img src="https://github.com/user-attachments/assets/0698112e-8b61-495a-b46e-a4187a4138ea" width="650" height="auto">



 ### **2. Previous work in vLLM** 

To handle the increasing batch size in SD, vLLM has been performing the following tasks: Batch Expansion https://githu

[... truncated for brevity ...]

---

## Issue #N/A: `8-bit quantization` support

**Link**: https://github.com/vllm-project/vllm/issues/214
**State**: closed
**Created**: 2023-06-22T23:02:03+00:00
**Closed**: 2024-04-18T13:52:08+00:00
**Comments**: 14
**Labels**: feature request

### Description

As far as I know `vllm` and `ray` doesn't support `8-bit quantization` as of now. I think it's the most viable quantization technique out there and should be implemented for faster inference and reduced memory usage. 

---

## Issue #N/A: [Feature]: Use `QuantFp8` `CustomOp`-abstraction for MoE layers

**Link**: https://github.com/vllm-project/vllm/issues/20711
**State**: open
**Created**: 2025-07-09T21:41:36+00:00
**Comments**: 5
**Labels**: good first issue, feature request

### Description

### 🚀 The feature, motivation and pitch

#19830 added `QuantFp8`, which uses the `CustomOp` abstraction to implement fp8 quantization in both CUDA and torch, allowing Inductor to achieve superior performance over the CUDA ops (which are unoptimized and also do not fuse by default). However, the class has to be instantiated during init, and MoE uses are currently in util free functions many levels deep. Those need to be mildly rearchitected to take advantage of the new abstraction.

The use to be rearchitected is here: https://github.com/vllm-project/vllm/blob/c7a00e6e6716f45db09e39cb21a8f91f741f10b9/vllm/model_executor/layers/fused_moe/utils.py#L37-L40

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: automatically select distributed inference backend

**Link**: https://github.com/vllm-project/vllm/issues/4955
**State**: closed
**Created**: 2024-05-21T16:30:36+00:00
**Closed**: 2024-06-11T18:10:43+00:00
**Comments**: 3
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

ray is kind of an overkill for single gpu case, but is currently the only choice for multi-node inference.

we can add an `auto` backend, that checks the world size and the number of gpus available in the node, if this fits within this node, we can use multiprocessing, otherwise we can use ray.

this will help performance a lot.

@njhill do you have any bandwidth for this?

### Alternatives

_No response_

### Additional context

_No response_

---

## Issue #N/A: [Feature]: ci test with vGPU

**Link**: https://github.com/vllm-project/vllm/issues/5426
**State**: closed
**Created**: 2024-06-11T16:39:05+00:00
**Closed**: 2024-11-27T02:07:12+00:00
**Comments**: 4
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

it seems aws and gcp supports [vGPU](https://docs.nvidia.com/grid/cloud-service-support.html) . we can run some small tests in vGPU, which should be cost-efficient and also test broader software support to avoid https://github.com/vllm-project/vllm/issues/4587 .

### Alternatives

_No response_

### Additional context

_No response_

---

## Issue #N/A: [Feature]: A Hacked Classifier Free Guidance Metho

**Link**: https://github.com/vllm-project/vllm/issues/15839
**State**: open
**Created**: 2025-03-31T23:05:33+00:00
**Comments**: 1
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

Hi Guys, 

Here I offered a simple but effective hacked classifier free guidance (CFG) in vllm using an additional batch sample as unconditional prompt. 

https://github.com/MSLDCherryPick/vllm

This modification is original designed for Music generation project. 

But I think this may work for other users. 

Also looking for better implementation of CFG in vllm. 

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: need no_repeat_n_gram in SamplingParams

**Link**: https://github.com/vllm-project/vllm/issues/7842
**State**: closed
**Created**: 2024-08-25T06:04:33+00:00
**Closed**: 2025-03-06T02:02:51+00:00
**Comments**: 7
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

It is very common for large models to encounter infinite loops during inference, and we need some methods to prevent this from happening. If infinite loops during inference are not monitored, it can significantly impact reasoning efficiency. 

Therefore, I need a parameter `no_repeat_n_gram` to prevent the generation of sequences where *n* consecutive tokens repeat, thus mitigating the occurrence of infinite loops. The specific implementation method is as follows: for a generated token x_i, for each possible value of x_i (in the case of sampling, x_i could have multiple possibilities), we monitor whether generating this token violates the `no_repeat_n_gram_size`. If it does, we set its logit to negative infinity, thereby preventing the generation of *n*-gram repetitions.

In practice, I will set *n* as large as possible to act as a punishment for infinite loops without overly affecting the model's normal inference output. The reason I do n

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Precise model device placement

**Link**: https://github.com/vllm-project/vllm/issues/6189
**State**: closed
**Created**: 2024-07-07T14:10:43+00:00
**Closed**: 2025-01-04T01:58:36+00:00
**Comments**: 14
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

Hi all, I was wondering if it's possible to do precise model device placement. For example, I would like to place the vLLM model on GPU 1 and let GPU 0 do other things. Being able to do precise model device placement will help unblock online RLHF work in our Hugging Face's TRL, because we want to leverage the fast speed of vLLM's generation.

In particular, we'd like to run training on 7 GPUs, and leave only 1 GPU for vLLM inference. I have a very crude hack that supports this at https://github.com/vwxyzjn/vllm/pull/1, but I figure more general support in vLLM will be more helpful.



Currently this is not possible because the following code will error out

```
from vllm import LLM, SamplingParams
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(tempe

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Integrate with AICI

**Link**: https://github.com/vllm-project/vllm/issues/3714
**State**: closed
**Created**: 2024-03-29T03:29:22+00:00
**Closed**: 2024-11-29T02:06:49+00:00
**Comments**: 2
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

#2888 added a prototype for AI Controller Interface, which is a WASM based runtime for guided generation. We would like to integrate this into our existing guided decoding stack properly. 

Related is lm-format-enforcer #3713. 

We should tell the users strength of each framework and let the user choose.

### Alternatives

_No response_

### Additional context

_No response_

---

## Issue #N/A: LightLLM benchmark

**Link**: https://github.com/vllm-project/vllm/issues/670
**State**: closed
**Created**: 2023-08-03T15:35:27+00:00
**Closed**: 2023-10-19T12:36:04+00:00
**Comments**: 8
**Labels**: feature request

### Description

Hi vLLM genius @zhuohan123 @WoosukKwon 

I find a new project  https://github.com/ModelTC/lightllm 

After reading their [blog](https://mp.weixin.qq.com/s/-wMLMGAHkxeyDYkixqni9Q), the performance advantage on the 7b model is not very obvious, but the gap is larger on the 65b. We will also do some verification and comparison later. The reason for bringing up this issue is to hope that we may see what the LightLLM does well, so that we can refer to and port similar optimizations to vLLM. Cheers.



---

## Issue #N/A: [Feature]: Tool call inside reasoning

**Link**: https://github.com/vllm-project/vllm/issues/16511
**State**: closed
**Created**: 2025-04-11T20:20:56+00:00
**Closed**: 2025-04-11T20:23:25+00:00
**Comments**: 1
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

I need tool call parser to work with reasoning.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: Serve /metrics while a model is loading

**Link**: https://github.com/vllm-project/vllm/issues/12173
**State**: closed
**Created**: 2025-01-17T18:28:57+00:00
**Closed**: 2025-01-28T20:05:31+00:00
**Comments**: 1
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Is it possible for the /metrics endpoint to be up, even if returning and empty response with a 200 status code, while a model is loading?



### Alternatives

_No response_

### Additional context

Upon starting the vLLM container, depending on how heavily HuggingFace is throttling you, it may take hours for a model to be downloaded and loaded. In this case know that the service is up but not yet active is useful for monitoring purposes.

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: Prompt logprobs + APC compatibility

**Link**: https://github.com/vllm-project/vllm/issues/13409
**State**: closed
**Created**: 2025-02-17T15:33:22+00:00
**Closed**: 2025-02-17T17:27:11+00:00
**Comments**: 0
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

[#9880](https://github.com/vllm-project/vllm/pull/9880) adds sample and prompt logprobs support, however prompt logprobs currently require the server to be instantiated with `--no-enable-prefix-caching`; otherwise, a request with `prompt_logprobs=true` will cause the request to fail with the message "Prefix caching with prompt logprobs not yet supported on VLLM V1."

The challenge of using prompt logprobs alongside APC is how to recover the topk prompt logprobs from an APC cache hit. The existing APC implementation does not cache prompt logprobs; upon a cache hit, cached blocks are treated as "computed" & no prompt logprobs are available for the computed blocks.

### Alternatives

A few possible solutions:
* **Use APC cached KVs to recompute prompt logprobs if a request with `prompt_logprobs=true` triggers an APC cache hit.** This requires model code and `model_executor` code to support re-running prefill using cached KVs.
* **Cache prompt logpr

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Configurable metrics export format - Prometheus, OpenTelemetry

**Link**: https://github.com/vllm-project/vllm/issues/15141
**State**: closed
**Created**: 2025-03-19T16:08:56+00:00
**Closed**: 2025-07-18T02:28:02+00:00
**Comments**: 2
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

We would like to export metrics from  vLLM in the OpenTelemetry format with delta temporality (Prometheus uses only cumulative temporality).

For instance Dynatrace can ingest only delta temporality metrics. 

### Alternatives

Deploy an OpenTelemetry collector with prometheus receiver and cumulativetodelta processor as a sidecar to vLLM and then export the metrics from the collector.

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: DeepSeek-R1 tool choice && Function Call

**Link**: https://github.com/vllm-project/vllm/issues/12297
**State**: closed
**Created**: 2025-01-22T03:33:55+00:00
**Closed**: 2025-05-23T02:10:26+00:00
**Comments**: 2
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

Would like to support tool choice and chat-template based on deepseek-R1, so that better results can be obtained.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature Request] Tree speculate

**Link**: https://github.com/vllm-project/vllm/issues/2426
**State**: closed
**Created**: 2024-01-12T03:43:21+00:00
**Closed**: 2024-11-30T02:02:58+00:00
**Comments**: 2
**Labels**: feature request, stale

### Description

Is is possible to integrate tree speculate into vllm? This is the tree speculate repo: [PainlessInferenceAcceleration](https://github.com/alipay/PainlessInferenceAcceleration)

---

## Issue #N/A: [Feature]: Llama 3 and Command-R Chat Templates

**Link**: https://github.com/vllm-project/vllm/issues/9904
**State**: closed
**Created**: 2024-11-01T03:56:37+00:00
**Closed**: 2025-03-23T02:09:08+00:00
**Comments**: 4
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

Could we add Llama 3 and Command-R Chat Templates to https://github.com/chujiezheng/chat_templates/tree/main/chat_templates? Thank you!

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: new possible lora serving implementation?

**Link**: https://github.com/vllm-project/vllm/issues/8497
**State**: closed
**Created**: 2024-09-16T01:11:09+00:00
**Closed**: 2024-09-17T07:52:21+00:00
**Comments**: 4
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Paper from Microsoft on serving loras in production

https://arxiv.org/abs/2404.05086v1

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: Q-Filters for KV Cache Compression

**Link**: https://github.com/vllm-project/vllm/issues/14381
**State**: closed
**Created**: 2025-03-06T18:58:38+00:00
**Closed**: 2025-07-05T02:11:59+00:00
**Comments**: 2
**Labels**: feature request, stale

### Description

The new Q-Filters paper introduces a **training-free** KV cache compression method with 32× compression and 99% accuracy on long-context tasks. It’s low-overhead and beats FP8 compression with **FlashAttention-compatibility**. Can vLLM integrate this for better long-context support and the ability to use larger models on constrained hardware? See [this discussion](https://github.com/vllm-project/vllm/discussions/14378) for more info.

---

## Issue #N/A: [Feature][Hardware][TPU]:Reduce the compile time

**Link**: https://github.com/vllm-project/vllm/issues/14582
**State**: closed
**Created**: 2025-03-10T22:36:38+00:00
**Closed**: 2025-04-16T05:31:48+00:00
**Comments**: 1
**Labels**: feature request, tpu

### Description

### 🚀 The feature, motivation and pitch

After the fix of https://github.com/vllm-project/vllm/pull/14310,

We have num_token_bucket compilations for the main model and num_token_bucket x num_reqs_bucket for the logits processor.

We can make some improvement on this, as the num_token_bucket x num_reqs_bucket only happens on hidden_states[logits_indices], where we select part of the hidden states. Therefore, we can partition the graph to 3 parts:

main model: num_token_bucket
hidden_states[logits_indices]: num_token_bucket x num_reqs_bucket
logits: num_reqs_bucket

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

