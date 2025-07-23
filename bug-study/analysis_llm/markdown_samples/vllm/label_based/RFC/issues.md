# RFC - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 12
- Closed Issues: 18

### Label Distribution

- RFC: 30 issues
- stale: 12 issues
- v1: 2 issues
- unstale: 1 issues
- tpu: 1 issues
- torch.compile: 1 issues
- multi-modality: 1 issues
- structured-output: 1 issues
- keep-open: 1 issues

---

## Issue #N/A: [RFC]: Encoder/decoder models & feature compatibility

**Link**: https://github.com/vllm-project/vllm/issues/7366
**State**: open
**Created**: 2024-08-09T15:03:54+00:00
**Comments**: 17
**Labels**: RFC, unstale

### Description

## Motivation <a href="#user-content-motivation" id="motivation">#</a>

There is significant interest in vLLM supporting encoder/decoder models. Issues #187  and #180 , for example, request encoder/decoder model support. As a result encoder/decoder support was recently introduced to vLLM via the following three PRs:

* #4837 
* #4888 
* #4942 

These three PRs make encoder/decoder model inference possible; however, they leave more to be desired in terms of (1) parity between vLLM's decoder-only & encoder/decoder request processing pipelines with respect to feature support, and (2) the number of encoder/decoder models which are supported.

The ask for the vLLM community is to contribute PRs which help bring vLLM encoder/decoder functionality to a similar level of maturity as that of vLLM's decoder-only functionality.

## Proposed changes <a href="#user-content-proposed-changes" id="proposed-changes">#</a>

The support matrix below summarizes which encoder/decoder models ha

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: TPU V1 Sampler planning

**Link**: https://github.com/vllm-project/vllm/issues/16268
**State**: open
**Created**: 2025-04-08T14:24:38+00:00
**Comments**: 6
**Labels**: RFC

### Description

### Motivation.

I'd like to gather some input on how to move forward with sampling support, and also provide a brief recap of the current state+planned support.

At a high level, the current design splits model forward and sampling into two separate graphs. 
As of now (`f2ebb6f54`) only the `temperature` and `min_p` have been intentionally enabled. 
As more techniques will be added, the sampling graph will grow in size (vertically, sequential ops) and performance may need monitoring, as we're simply evaluating more operations at runtime. 
To clarify, even when one option is not enabled, we still evaluate a no-op version that undergoes the same ops in the graph (eg top-p with p=1).

### Proposed Change.

Following https://github.com/vllm-project/vllm/pull/15489 a few concerns that have been raised regarding performance while enabling topk,  hence adding the **very first** op to the initial sampling graph, I'd like to re-evaluate the current approach.
Looking at the opposite side of the

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Deprecation of the `best_of` Sampling Parameter in vLLM V1

**Link**: https://github.com/vllm-project/vllm/issues/13361
**State**: closed
**Created**: 2025-02-16T17:57:03+00:00
**Closed**: 2025-03-05T20:22:45+00:00
**Comments**: 2
**Labels**: RFC

### Description

### Motivation.

### Overview
As we transition to vLLM V1, we plan to discontinue support for the `best_of` sampling parameter. This decision is driven by a combination of low usage, alignment with industry trends, and a desire for system simplicity and performance.

### Background: What is `best_of`?
The `best_of` parameter was originally part of the earlier OpenAI completion API. It enabled the generation of multiple completions—`n` different outputs—then selected the “best” completion based on the cumulative log probabilities of each result.

### Reasons for Deprecation

1. **Limited Usage and Industry Trends:**
   - **Low Adoption:** To the best of our knowledge, the `best_of` feature is used by very few users. Users have observed that output quality isn’t reliably correlated with their log probabilities in most cases.
   - **Evolving Standards:** Major AI providers such as OpenAI (in its current API), Claude, and Gemini have moved away from including the `best_of` option.

2. **Al

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Drop Support for OpenVINO

**Link**: https://github.com/vllm-project/vllm/issues/14374
**State**: closed
**Created**: 2025-03-06T17:34:37+00:00
**Closed**: 2025-03-22T21:06:40+00:00
**Comments**: 5
**Labels**: RFC

### Description

### Motivation.

OpenVINO backend was initially integrated as an alternatively to the CPU backend and has branched out the vLLM execution logic for every levels (executor, model runner, and attention backend). #5377

Over the last 9 months, we have been the following
* Relatively low usage as reported in Github Issues and Slack discussions
* The Intel CPU codepath is more mature and largely compatible for Arm as well. 
* The OpenVINO code path complicated with codebase
* CI and build became difficult to maintain

I would like to propose to move OpenVINO off from the main codebase, and transition to a vLLM out of tree platform plugin if desired. OpenVINO can follow the same approach as Ascend and Spyre with the plugin approach #11162 



### Proposed Change.

* Remove OpenVINO codepath, build and test. 
* Optionally, create vllm-project/vllm-openvino if the developers want to maintain plugin level compatibility. 

### Feedback Period.

2 weeks. By March 20. 

### CC List.

cc @ilya-lavr

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Initial support for RBLN NPU

**Link**: https://github.com/vllm-project/vllm/issues/7247
**State**: closed
**Created**: 2024-08-07T05:28:24+00:00
**Closed**: 2025-03-19T02:05:27+00:00
**Comments**: 5
**Labels**: RFC, stale

### Description

### Motivation.

The [RBLN SDK](https://rebellions.ai/wp-content/uploads/2024/08/WhitePaper_Issue2_ATOM_SoftwareStack.pdf) provides a solution for innovative deep learning inference on Rebellion's NPUs, such as [ATOM](https://rebellions.ai/wp-content/uploads/2024/07/ATOMgenAI_white-paper.pdf) and REBEL, including support for large language models (LLMs). This project aims to develop the RBLN backend for vLLM, initially prioritizing the ATOM device, with future plans to enable REBEL support.

In alignment with Rebellion's Optimum Huggingface extension [documentation](https://docs.rbln.ai/software/optimum/optimum_rbln.html), RBLN backend will support a wide range of models available in the [Rebellion's Model Zoo](https://rebellions.ai/developers/model-zoo/).

The project currently incorporates continuous batching feature and will soon integrate additional techniques, such as PagedAttention, to enhance performance further.

### Proposed Change.

Introduce the RBLN vLLM backend, 

[... truncated for brevity ...]

---

## Issue #N/A: [RFC] Initial Support for Cloud TPUs

**Link**: https://github.com/vllm-project/vllm/issues/3620
**State**: closed
**Created**: 2024-03-25T17:08:43+00:00
**Closed**: 2025-03-11T14:04:01+00:00
**Comments**: 17
**Labels**: RFC, tpu, stale

### Description

# Progress

- [x] Implement TPU executor that works on a single TPU chip (without tensor parallelism) #5292 
- [x] Support single-host tensor parallel inference #5871 
- [x] Support multi-host tensor parallel inference #7457 
- [ ] Support INT8 quantization
- [x] Support MoE models such as Mixtral #6457
- [ ] Benchmark and optimize the TPU backend performance

# Project Scope

This project focuses on making vLLM compatible with Google cloud TPUs. Our goal is seamless integration so users can easily run vLLM on TPUs for both online and offline inference. We will target common setups, like popular models such as Gemma, using the bfloat16 data type.

## Target TPUs and Models

We will focus on the most recent generations of TPUs, namely **TPU v4, v5e, and v5p**, considering their superior performance to previous generations. We will start by making sure vLLM works with dense models such as Gemma. After that, we will expand support to Mixture-of-Experts (MoE) models such as 

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: openai api response format

**Link**: https://github.com/vllm-project/vllm/issues/9601
**State**: closed
**Created**: 2024-10-23T02:27:41+00:00
**Closed**: 2024-10-23T03:12:09+00:00
**Comments**: 1
**Labels**: RFC

### Description

### Motivation.

Here is example of my request using /v1/chat/completions.
```
{
  "id": "chat-36cf36c94fa746ffbee01440bbdcbf35",
  "object": "chat.completion",
  "created": 1729649934,
  "model": "",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "hello",
        "tool_calls": []
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null
    }
  ],
  "usage": {
    "prompt_tokens": 1841,
    "total_tokens": 1849,
    "completion_tokens": 8
  },
  "prompt_logprobs": null,
  "messages": [
    {
      "content": "hello",
      "role": "user"
    },
    {
      "role": "assistant",
      "content": "xxx",
      "tool_calls": []
    }
  ]
}
```

I'm wondering if "messages" is from in openai api reference?

### Proposed Change.

How to disable this extra part from output?

### Feedback Period.

_No response_

### CC List.

_No response_

### Any Other Things.

_N

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Single Program Multiple Data (SPMD) Worker Control Plane

**Link**: https://github.com/vllm-project/vllm/issues/6556
**State**: closed
**Created**: 2024-07-19T00:42:25+00:00
**Closed**: 2024-12-06T02:07:26+00:00
**Comments**: 8
**Labels**: RFC, stale

### Description

### Motivation.

**TL;DR**: Introduce SPMD-style control plane to improve control plane architecture and optimize performance.

For distributed inference, vLLM currently leverages a “driver-worker”, along with other workers. As shown in the diagram below, this driver-worker is in the same process as the driver. It prepares the arguments, then broadcasts them to all other workers to execute the sharded model, leveraging NCCL as the control plane. 

<img width="850" alt="Screenshot 2024-07-18 at 5 37 48 PM" src="https://github.com/user-attachments/assets/03ef792a-e0ad-4797-a7e0-9e3abcb0b028">

This architecture has a few drawbacks. First, the driver-worker needs to participate in the NCCL group and execute the model. Since NCCL broadcast is a synchronous operation, this creates interference with other driver functionality such as scheduling and affects performance. 

Moreover, this architecture made it difficult to support speculative decoding. Specifically,
1. Speculative decod

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Deprecate `max_num_generation_tokens`

**Link**: https://github.com/vllm-project/vllm/issues/14168
**State**: closed
**Created**: 2025-03-04T01:25:49+00:00
**Closed**: 2025-07-02T02:13:44+00:00
**Comments**: 2
**Labels**: RFC, stale

### Description

### Motivation.

Said by @robertgshaw2-redhat https://github.com/vllm-project/vllm/pull/14055#issuecomment-2695114713

### Proposed Change.

As part of v1, we should deprecate `max_num_generation_tokens`.


### Feedback Period.

When agreed _if_ this change should be made.

### CC List.

@robertgshaw2-redhat @markmc 

### Any Other Things.

I will do this if interested.

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [RFC]: Multimodal data IPC improvement

**Link**: https://github.com/vllm-project/vllm/issues/19702
**State**: open
**Created**: 2025-06-16T17:56:26+00:00
**Comments**: 0
**Labels**: RFC

### Description

### Motivation.

### Summary
Currently vllm interprocess communication can account for considerable amount of overhead in some cases, this RFC is aiming at reducing these overhead by using a shared memory based approach for interprocess communication.

### Background
According to the profiling result on our internal vision model in a TP>1 setting, the GPU stays idle during engine to worker communication.
![Image](https://github.com/user-attachments/assets/4669088c-ddd6-4947-bb30-1c1bce0985a5)
The major overhead is two parts 1. IPC between engine and worker process through socket 2. serialization and deserialization through pickle

A similar issue is posted here https://github.com/vllm-project/vllm/issues/16626

### Proposed Change.

After initial discussion with @ywang96 and @njhill , proposing this change to address the following communication overhead

1. IPC between engine and worker processes
2. Serialization and deserialization before and after 1.
3. Extra multimodal data transmis

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Keep a Changelog & Add FAQs in the Documentation

**Link**: https://github.com/vllm-project/vllm/issues/7769
**State**: closed
**Created**: 2024-08-22T02:58:47+00:00
**Closed**: 2024-12-22T02:04:27+00:00
**Comments**: 3
**Labels**: RFC, stale

### Description

### Motivation.

## Changelog
I frequently find myself wondering what is the difference between the latest version(s) of vLLM, and the version that I currently have deployed. It seems like it would be nice to keep a simple changelog that documents features, fixes, newly-supported hardware and updates between versions so that we can easily see what has been added in recent versions -- e.g. new CLI arguments, optimizations, quantization formats, updated hardware support for features (e.g. punica -> triton kernels, expanding hardware support for multi-lora serving) patched bugs, and so forth.

A great template for this is [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) following semver - it would be super easy to implement with Markdown in the documentation site. I think this would make vLLM's newer features much more accessible, _and_ it would also help identify gaps in the documentation when we add something to the changelog that's not on the docs site

## FAQs 
There are

[... truncated for brevity ...]

---

## Issue #N/A: [RFC][UX]: debug mode for vLLM-compile

**Link**: https://github.com/vllm-project/vllm/issues/20394
**State**: open
**Created**: 2025-07-02T17:56:56+00:00
**Comments**: 1
**Labels**: RFC, torch.compile

### Description

### Motivation.

vLLM-compile (CompilationLevel.PIECEWISE) makes a lot of assumptions about the models that allow it to make them run really fast. There are two main assumptions that commonly lead to silent incorrectness if the models violate them. I've spent countless hours debugging user issues for it to turn out to be one of these assumptions. We should add a debug mode option for vLLM-compile that, when turned on, adds some safety checks for these assumptions at the tradeoff of some additional overhead. This will let users self-diagnose the issues without me in the loop.

This is one of the items mentioned in https://github.com/vllm-project/vllm/issues/20283, I'm expanding it to include some more details.

### Proposed Change.

The two assumptions that bite us are:
1) the [vLLM Dynamic Shapes Issue](https://docs.google.com/document/d/1R3XvVEpJeVi3whyxf4xpyZufGplbrfw628oXLZ6fqG0/edit?tab=t.0#heading=h.59xosv6nz9lg). vLLM performs one single graph capture with dynamic batch size and 

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Merge input processor and input mapper for multi-modal models

**Link**: https://github.com/vllm-project/vllm/issues/10114
**State**: closed
**Created**: 2024-11-07T09:57:55+00:00
**Closed**: 2025-04-28T07:38:50+00:00
**Comments**: 13
**Labels**: RFC, multi-modality

### Description

## Motivation

### Background

To provide more control over the model inputs, we currently define two methods for multi-modal models in vLLM:

- The **input processor** is called inside `LLMEngine` to extend the prompt with placeholder tokens which are reserved for vLLM features such as KV cache and chunked prefill.
- The **input mapper** is called inside `ModelRunner` to transform multi-modal inputs (e.g. `PIL` images) into tensor inputs, usually via the modality-specific processor (e.g. `AutoImageProcessor`) from HuggingFace.

### Issues with the current design

1. The input processor accepts the output of HF `AutoTokenizer`, a list of token IDs, instead of the text prompt. Since HF `AutoProcessor` doesn’t accept token IDs, we have to write custom code to edit the list of token IDs based on the multi-modal inputs. For some models (such as Phi-3-vision), this means re-implementing code from their HF `AutoProcessor`, complicating the process of porting the model to vLLM.
2. The input m

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Custom sampling params support in REST API

**Link**: https://github.com/vllm-project/vllm/issues/17191
**State**: open
**Created**: 2025-04-25T14:30:19+00:00
**Comments**: 8
**Labels**: RFC

### Description

**Update:** after incorporating feedback, the updated proposal is described in this comment: https://github.com/vllm-project/vllm/issues/17191#issuecomment-2858443302

## Original RFC proposal (outdated):


### Motivation

Addresses #16802 (“Support custom args in OpenAI (chat) completion requests”) by adding an “extra” sampling params argument to all endpoints which trigger sampling (completion, chat and transcription). This is ultimately a prerequisite for logits processor support ( RFC: #13360 PR: #16728 ), since logits processors may require custom arguments which are not utilized by vLLM core sampling logic.

### Proposed Change.

Here it is proposed that when using the HTTP client, custom sampling arguments may be passed in as key/value pairs via the `extra_sampling_params` argument

```
extra_sampling_params: Optional[dict[str, Any]]
```

#13300 added an `extra_args` member to `SamplingParams` 

```
extra_args: Optional[dict[str, Any]] = None
```

`protocol.py` defines a class t

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Support encode only models by Workflow Defined Engine

**Link**: https://github.com/vllm-project/vllm/issues/8453
**State**: closed
**Created**: 2024-09-13T08:27:51+00:00
**Closed**: 2025-01-10T08:58:49+00:00
**Comments**: 7
**Labels**: RFC, stale

### Description

### Motivation.

As vllm supports more and more models and functions, they require different attention, scheduler, executor, and input output processor. . These modules are becoming increasingly complex, and sometimes new features must be compromised for compatibility. ultimately leading to suboptimal results

Take support for encode only models as an example

Although the encode only models is much simpler than the decode model, they are very different.

The simplest way to support the encode only models is to implement different modules for models of different architectures and load the required modules on demand.

I call this architecture Workflow Defined Engine, or WDE for short.

###  Terminology.
The scope of discussion is slightly larger than encode only models, and is roughly divided into three categories：
- Encode only models. (Bidirectional Transformers, causal=False), Often fine-tuned as retriever and reranker etc.
- Decode only models. (masked multi-head atte

[... truncated for brevity ...]

---

## Issue #N/A: Create speculative decode dynamic parallel strategy

**Link**: https://github.com/vllm-project/vllm/issues/7351
**State**: closed
**Created**: 2024-08-09T12:09:30+00:00
**Closed**: 2024-08-15T07:56:47+00:00
**Comments**: 1
**Labels**: RFC

### Description

## Motivation

Create new speculative decode dynamic parallel strategy for our team needs

## Features

Here we briefly describe features that we will implement in order to implement speculative decode dynamic parallel strategy. Each feature has high level description as a part of request for change with more description provided inside pull request for particular feature

#### Save speculative decoding states #7358
    
Allow users to optionally receive speculative decoding artifacts such as history of draft token indices for each step of speculative decode algorithm

#### Create draft from random tokens from promt #7359

Implement speculative proposers that enrich previous draft with tokens randomly sampled from current prompt

#### Allow model executor to return many next tokens #7361

Current implementation of model executor and model runner produce one last next token in decode stage. This feature would allow inner model runners to return next tokens for a range 

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Prototype Separating Vision Encoder to Its Own Worker

**Link**: https://github.com/vllm-project/vllm/issues/20799
**State**: open
**Created**: 2025-07-11T06:13:16+00:00
**Comments**: 1
**Labels**: RFC

### Description

### Motivation.

In the current multi-modality support within vLLM, the vision encoder (e.g., Qwen_vl) and the language model decoder run within the same worker process. While this tightly coupled architecture is simple to implement, it introduces several challenges in terms of scalability, resource utilization, and flexibility:

1.  **Resource Contention:** The vision encoder is often a compute and memory-intensive task. When processing high-resolution images or performing complex preprocessing, it competes for valuable GPU resources with the language model's prefill and decode stages, potentially increasing the overall latency of request processing.

2.  **Scalability Issues:** The workload characteristics of vision processing and text generation are different. In some scenarios, image processing might be the bottleneck (e.g., a high volume of concurrent image inputs), while in others, text generation is the bottleneck (e.g., long text outputs). A unified worker model cannot scale th

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: All Ops should be determined during init and wrapped in a Layer Module to avoid envs.ENVIRON overhead

**Link**: https://github.com/vllm-project/vllm/issues/17067
**State**: open
**Created**: 2025-04-23T16:45:16+00:00
**Comments**: 4
**Labels**: RFC

### Description

### Motivation.

Accessing envs.ENVIRON has non-negligible overhead. Given that LLM models have many ops and layers. The overhead from accessing envs.ENVIRON could spike to 0.1 ~ 1ms overhead per token. I have observed a huge overhead in MLA prefill forward pass when using envs.ENVIRON in kernel selection logic (where `if-else` statement is involved).

Proposed action:
1. Layer Module is suggested to store the selected kernel ops as a property of the layer.
`@cache` is discourage due to the increasing complexity that it is causing to clear the cache as there are many properties depending on envs.
`@cache` is discouraged in several PRs review as there is a usecase as such: 
Users instantiate multiple LLMs in a single python program. Each LLM instance uses different sets of ENV variables.

2. Document the overhead issue down in vLLM documentation page under Contribution section to remind developers of the abstract and the overhead caused by envs.ENVIRON invocation.


## Overhead experime

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Fully SPMD Execution for Offline Inference

**Link**: https://github.com/vllm-project/vllm/issues/11400
**State**: closed
**Created**: 2024-12-21T18:09:34+00:00
**Closed**: 2025-01-21T06:04:24+00:00
**Comments**: 1
**Labels**: RFC

### Description

### Motivation.

TL;DR:  Introducing a fully SPMD-style LLMEngine execution pattern to improve offline inference throughput.

The RFC draft is initiated by @PeterSH6 

# Background and Motivation 
## Inherent dispatch overhead in single-controller paradigm
For distributed offline inference, vLLM leverages a centralized controller process (e.g., Ray Driver) to broadcast the scheduler output to the workers. After workers' execution, the output is gathered from the workers to the centralized controller process to perform the next iteration scheduling. While this single-controller paradigm offers better user experience, it introduces throughput limitations.
Therefore, to launch a generation call, vLLM obey the following procedure:
```
python3 offline_inference.py # launch the centralized controller process (i.e., LLMEngine)
# inside the LLMEngine
llm_engine.distributed_gpu_executor._run_workers('start_worker_execution_loop', ...) # execute the model
# inside the _run_workers

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: KV-Cache Interoperability API Standardization

**Link**: https://github.com/vllm-project/vllm/issues/20492
**State**: open
**Created**: 2025-07-04T16:23:37+00:00
**Comments**: 2
**Labels**: RFC

### Description

### Motivation

This RFC proposes a KV-Cache Interoperability API, covering standardized notification events (via KVEvents) and reproducible prefix-block hashing. These standards aim to support cross-system cache awareness, observability, and future tooling for indexing, routing, and diagnostics.

vLLM already ships with internal [KVEvents](https://github.com/vllm-project/vllm/issues/16669) contributed by the NVIDIA Dynamo team - that’s a strong foundation. 
But as external systems aim for cache-aware inference, we need to treat these internal mechanisms as public contracts to support broader adoption and interop.

### Goals

1. **KVEvents Internal API as a Public Contract**  
   The KVEvents schema is already well-defined in vLLM and used internally by the `KVCacheManager` for GPU cache events. It’s also being extended to CPU offloading via the `KVConnector` (see [#19854](https://github.com/vllm-project/vllm/issues/19854)).  
   This RFC proposes formalizing KVEvents as the public con

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Model architecture plugins

**Link**: https://github.com/vllm-project/vllm/issues/7124
**State**: closed
**Created**: 2024-08-04T12:13:33+00:00
**Closed**: 2025-01-03T02:41:49+00:00
**Comments**: 17
**Labels**: RFC, stale

### Description

### Motivation.

As a continuation to #5367 - as this merge request was rejected and I have to maintain my own fork to support this scenario, I suggest we should add support in vLLM for model architecture plugins.
This will allow vLLM to easily add new model architectures without changing vLLM's core logic, and support scenarios such as uneven GPU tensor parallelism.

We could build an ecosystem of model architecture plugins - which could accelerate new model support by a lot without risking existing functionality.

### Proposed Change.

Supporting this in it's basic form is simple as we just have to add loaded plugins to the `ModelRegistry`.
To support more complex model architectures (Such in the #5367 case), we should decouple the `Config` class which provides the amount of attention heads from vLLM's core logic, and allow each model architecture to override these values.

### Feedback Period.

_No response_

### CC List.

@youkaichao 

### Any Other Things.

Just to make it cle

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Logits processor extensibility

**Link**: https://github.com/vllm-project/vllm/issues/17799
**State**: open
**Created**: 2025-05-07T13:43:16+00:00
**Comments**: 8
**Labels**: RFC

### Description

### Motivation.

Users want logits processor extensibility, i.e. the ability to specify logits processors beyond those such as min-p which are hard-coded into the engine. See for example:
* #12678
* https://github.com/NVIDIA/logits-processor-zoo - library of logits processor extensions

The purpose of this RFC is to establish the interface for extending the vLLM V1 engine with additional logits processors during engine instantiation.

vLLM V0 supports logits processor configuration at request level (`SamplingParams` attribute). For V0 running in server mode, PR #11150 makes it possible for a request to dynamically import one or more logits processor modules, assuming that the necessary modules are available/installed. The `logits_processors` argument (available in the completion, chat completion and transcription API endpoints) allows the custom logits processors’ constructors to be specified as a list of (1) qualified names, or (2) `LogitsProcessorConstructor` data structures (which i

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Response format extensions for structured outputs

**Link**: https://github.com/vllm-project/vllm/issues/19097
**State**: open
**Created**: 2025-06-03T17:05:05+00:00
**Comments**: 11
**Labels**: structured-output, RFC, v1

### Description

### Motivation.

Currently, users can provide additional constraints format via `extra_body` in OpenAI client:

```python
from enum import Enum
from pydantic import BaseModel
from openai import OpenAI

simplified_sql_grammar = """
        root ::= select_statement

        select_statement ::= "SELECT " column " from " table " where " condition

        column ::= "col_1 " | "col_2 "

        table ::= "table_1 " | "table_2 "

        condition ::= column "= " number

        number ::= "1 " | "2 "
    """

prompt = (
        "Generate an SQL query to show the 'username' and 'email'"
        "from the 'users' table."
    )

completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        extra_body={"guided_grammar": simplified_sql_grammar},
```

This also applies with `guided_json`, `structural_tag`, `guided_regex`.

While this is pretty convenient for 

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Async KV Cache Transfer for Disaggregated Inference

**Link**: https://github.com/vllm-project/vllm/issues/13020
**State**: closed
**Created**: 2025-02-10T08:04:11+00:00
**Closed**: 2025-07-02T02:14:18+00:00
**Comments**: 3
**Labels**: RFC, stale

### Description

### Motivation.

Hello vLLM community,

We're from the AWS Neuron inference team and are actively working on P/D disaggregated inference. We'd like to share our initial PoC for achieving **asynchronous KV cache transfer** (mentioned in roadmap #10818), to make decode continue execution while receiving KV cache from prefill workers. Developed based on the current KVCacheTransferAgent (introduced in v0.7.0) and v0 scheduler. 

### Proposed Change.

**KVCacheTransferAgent (KVLookupBuffer level)** at Decode Worker:


* Create and maintain a `receiver_buffer` containing entries for `{input_ids, roi, keys, caches, hidden}` in decode workers, rather than only for prefill workers.
* Introduce the `async_drop_select` API. Unlike `drop_select` (which triggers immediate blocking lookups that stall current process), this method queues `drop_select_request` and returns immediately.
* Implement a dedicated `drop_select_requester` thread to process queued `drop_select_request`. This thread initiates 

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: hide continuous batching complexity through forward context

**Link**: https://github.com/vllm-project/vllm/issues/9098
**State**: closed
**Created**: 2024-10-05T22:34:54+00:00
**Closed**: 2025-02-07T01:59:42+00:00
**Comments**: 5
**Labels**: RFC, stale

### Description

### Motivation.

take a look at the current llama forward computation logic:

```python
class LlamaMLP(nn.Module):
    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torc

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Add control panel support for vLLM

**Link**: https://github.com/vllm-project/vllm/issues/4873
**State**: open
**Created**: 2024-05-17T02:20:50+00:00
**Comments**: 12
**Labels**: RFC, keep-open

### Description

### Motivation.

The Fastchat-vLLM operational model offers significant advantages in deploying large language models (LLMs) for product services. [1](https://blog.vllm.ai/2023/06/20/vllm.html)

The controller architecture in Fastchat is particularly beneficial for LLM deployment, owing to its loosely coupled design with the vLLM backend. This allows for:

* Autoscaling: The vLLM backend can join and exit the cluster freely, enabling dynamic scaling capabilities.

* Rolling Updates: The introduction of new models with distinct names allows the cluster to gradually update models, a process known as rolling updates.

* Centralized Access: Users are relieved from the burden of tagging different URLs or IPs for various models; they simply send their requests to the controller, which then manages the rest, including dispatching requests to the appropriate backend based on the model name and ensuring effective load balancing.

However, the challenge for Fastchat lies in managing 

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: How about change name specialized_manager to specialized_kv_cache_manager ?

**Link**: https://github.com/vllm-project/vllm/issues/16527
**State**: closed
**Created**: 2025-04-12T04:28:56+00:00
**Closed**: 2025-07-14T06:36:48+00:00
**Comments**: 1
**Labels**: RFC, stale

### Description

### Motivation.

The subclasses of SpecializedManager, such as FullAttentionManager and SlidingWindowManager, are all related to handling KV cache.

### Proposed Change.

So why aren't they named something like SpecializedKVManager instead? The current naming makes the code confusing to read.

### Feedback Period.

_No response_

### CC List.

_No response_

### Any Other Things.

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [RFC]: Add automated profiling sweep and heatmap visualization tools

**Link**: https://github.com/vllm-project/vllm/issues/17823
**State**: open
**Created**: 2025-05-08T02:16:43+00:00
**Comments**: 1
**Labels**: RFC

### Description

### Motivation.

While `examples/offline_inference/profiling.py` provides detailed kernel-level timing in vLLM, its usability is limited when users want to:

- Conduct profiling across multiple batch sizes and prompt lengths
- Visualize performance trends and bottlenecks

Currently, users must manually modify arguments and parse raw outputs, which is slow and error-prone. There's no convenient way to sweep inputs or generate visual summaries.

We propose two tools to address this gap and extend the existing profiler for practical model-level profiling.

### Proposed Change.

We propose upstreaming two lightweight utilities:

#### 1. `sweep_profiling.py`  
A script to automate `profiling.py` runs across a set of batch sizes and prompt lengths. Features:

- CLI flags: `--model`, `--tensor-parallel-size`, `--max-tokens`
- Spawns subprocesses for each profiling job
- Captures errors cleanly and logs failures
- Output: multiple `profiling_bs{N}_pl{M}.json` traces

**In addition to the overa

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: [V1] TPU support and multiple architecture support

**Link**: https://github.com/vllm-project/vllm/issues/12480
**State**: closed
**Created**: 2025-01-27T18:31:51+00:00
**Closed**: 2025-06-05T02:12:54+00:00
**Comments**: 7
**Labels**: RFC, stale, v1

### Description

### Motivation.

We are in process of adding Google TPU support to the vLLM V1. 

Here is the WIP PR [https://github.com/vllm-project/vllm/pull/11936](https://github.com/vllm-project/vllm/pull/11936). 

Since this is the first time we add another hardware backend to V1, the PR has some refactor to avoid code duplications, which requires discussion and feedback.



### Proposed Change.

Here is the summary of changes this PR introduces:

1. Refactors the common logic of model_runner to **model_runner_base.py** in the folllowing way (Virtual functions in italic):
       \_\_init\_\_() => Has common config init
       get_model() => Just simply returns model
       get_kv_cache_spec() => Common logic for KV cache management
       _initialize_kv_cache()_ => Virtual API
       _execute_model()_ => Virtual API
       _load_model()_ => Virtual API
       _dummy_run()_ => Virtual API
       _profile_run()_ => Virtual API
       _capture_model()_ => Virtual API

2. Refactors common logic of wo

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Postmerge performance suite

**Link**: https://github.com/vllm-project/vllm/issues/4926
**State**: closed
**Created**: 2024-05-20T21:55:14+00:00
**Closed**: 2024-10-27T22:55:03+00:00
**Comments**: 6
**Labels**: RFC, stale

### Description

### Motivation.

We want to start tracking performance numbers of vLLM on more realistic workloads. Thanks to our sponsors #4925 we are getting a pool of hardware resources ready to run the testing on. 

The goal of this test suite is to
1. Track regression
2. Track our progress in optimization

### Proposed Change.

We will start with running the following benchmarks:

* Llama 8B on A100, H100
* Llama 70B on 4xA100, 4xH100, 8xA100, 8xH100
* Mixtral 8x7B on 8xH100
* Mixtral 8x22B on 8xH100

We will run with the following parameters:
- chunked prefill enabled
- fp8

We will run with the following tests:
- Benchmark latency
- Benchmark throughput with 1000 prompts (ShareGPT)
- Benchmark serving with 1000 prompts (ShareGPT)

We will also compare with TGI and TRT-LLM.

### Feedback Period.

Step 1: Ensure hardware availabilities
Step 2: Setup pipeline for Llama 8B on H100 as a proof of concept
Step 3: Monitor the result, build dashboard
Step 4: Scale to oth

[... truncated for brevity ...]

---

