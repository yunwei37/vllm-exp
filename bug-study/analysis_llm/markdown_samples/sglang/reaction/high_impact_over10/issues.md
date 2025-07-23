# high_impact_over10 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 14
- Closed Issues: 16

### Label Distribution

- high priority: 14 issues
- inactive: 7 issues
- collaboration: 6 issues
- enhancement: 2 issues
- new-model: 2 issues
- help wanted: 2 issues
- router: 1 issues
- blackwell: 1 issues
- function-calling: 1 issues
- performance: 1 issues

---

## Issue #N/A: [Feature] SGLang Router design discussion

**Link**: https://github.com/sgl-project/sglang/issues/2389
**State**: closed
**Created**: 2024-12-07T11:53:13+00:00
**Closed**: 2025-04-06T00:19:37+00:00
**Comments**: 7
**Labels**: enhancement, inactive

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

@ispobock  and I had a brief discussion about the current design and implementation of the SGLang Router.

I think the main concerns currently are as follows, before large-scale deployment.

- The current Router is stateful, which means I cannot deploy the Router like scaling a stateless service.
May we consider storing the state of the Router in services like Redis, DB, or etcd here?

- The current Router is at the cluster level. Although there are replicas, when the master fails, a replica can be used.
Imagine a real deployment scenario, such as one used by actual customers, where the deployment requires simultaneous use of AWS, GCP, and Oracle. The data centers are distributed across the Western US, Cent

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

## Issue #N/A: [Feature] GGUF Q4KM(4bit) format for deepseek R1 support

**Link**: https://github.com/sgl-project/sglang/issues/3140
**State**: closed
**Created**: 2025-01-26T06:08:02+00:00
**Closed**: 2025-04-22T00:18:35+00:00
**Comments**: 2
**Labels**: inactive

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Can you support deepseek R1 Q4KM GGUF file，https://huggingface.co/unsloth/DeepSeek-R1-GGUF

### Related resources

_No response_

---

## Issue #N/A: SGLang Router Architecture Improvement Proposal

**Link**: https://github.com/sgl-project/sglang/issues/7532
**State**: open
**Created**: 2025-06-25T20:06:12+00:00
**Comments**: 1
**Labels**: high priority, collaboration, router

### Description

# SGLang Router Architecture Improvement Proposal

## Table of Contents
1. [Summary](#summary)
2. [Current Architecture Overview](#current-architecture-overview)
3. [System Components](#system-components)
4. [Request Flow Analysis](#request-flow-analysis)
5. [Identified Pain Points](#identified-pain-points)
6. [Proposed Improvements](#proposed-improvements)
7. [Long-Term Vision](#long-term-vision)
8. [Implementation Phases](#implementation-phases)
9. [Risk Analysis](#risk-analysis)
10. [Success Metrics](#success-metrics)
11. [Conclusion](#conclusion)
12. [Appendix: Architecture Diagrams](#appendix-architecture-diagrams)

## Summary

This proposal outlines a architectural improvement plan for the SGLang Router, a high-performance load balancer that supports both traditional and disaggregated (Prefill-Decode) routing modes. The improvements focus on enhancing maintainability and extensibility without disrupting existing functionality. These changes lay the foundation for a long-term tran

[... truncated for brevity ...]

---

## Issue #N/A: Please add Phi3 support

**Link**: https://github.com/sgl-project/sglang/issues/407
**State**: closed
**Created**: 2024-05-01T04:09:58+00:00
**Closed**: 2024-09-22T14:19:38+00:00
**Comments**: 8

### Description

Getting this error - 

```
router init state: Traceback (most recent call last):
  File "/home/ubuntu/sglang/python/sglang/srt/managers/router/manager.py", line 73, in start_router_process
    model_client = ModelRpcClient(server_args, port_args)
  File "/home/ubuntu/sglang/python/sglang/srt/managers/router/model_rpc.py", line 657, in __init__
    self.model_server = ModelRpcService().exposed_ModelRpcServer(
  File "/home/ubuntu/sglang/python/sglang/srt/managers/router/model_rpc.py", line 70, in __init__
    self.model_runner = ModelRunner(
  File "/home/ubuntu/sglang/python/sglang/srt/managers/router/model_runner.py", line 294, in __init__
    self.load_model()
  File "/home/ubuntu/sglang/python/sglang/srt/managers/router/model_runner.py", line 303, in load_model
    model_class = get_model_cls_by_arch_name(architectures)
  File "/home/ubuntu/sglang/python/sglang/srt/managers/router/model_runner.py", line 58, in get_model_cls_by_arch_name
    raise ValueError(
ValueErr

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] there is a significant increase TTFT when using PD disaggregation mode compared to the single-node mode

**Link**: https://github.com/sgl-project/sglang/issues/6411
**State**: open
**Created**: 2025-05-19T06:36:11+00:00
**Comments**: 8

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

### Observation:
During testing, we observed a significant increase in Time to First Token (TTFT) when using PD disaggregation mode compared to the single-node mode. Specifically:

**Baseline TTFT (single-node mode): 2338 ms
PD Disaggregation TTFT: 24,766.90 ms**

### Environment Configuration:

Hardware:
Baseline test: 8 × H20 GPUs in a s

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Merge PDLB into SGLang Router

**Link**: https://github.com/sgl-project/sglang/issues/7031
**State**: open
**Created**: 2025-06-10T06:30:08+00:00
**Comments**: 1
**Labels**: high priority

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

## Overview

Merge Prefill-Decode Load Balancer (PDLB) functionality into SGLang Router to support both traditional load balancing and prefill-decode disaggregated routing.

**Key Insight**: Since PDLB has very minimal to no users, we can implement the optimal solution without migration.

## System Architecture

```mermaid
graph TB
    subgraph "Unified SGLang Router"
        A[Router Core] --> B{Policy Detection}
        B --> C[Regular Router]
        B --> D[PD Router]
        
        C --> C1[RoundRobin]
        C --> C2[Random] 
        C --> C3[CacheAware]
        
        D --> D1[PD Random]
        D --> D2[PD PowerOfTwo]
        D --> D3[PD CacheAware]
        
        D3 --> E[Tree-Based Selection]
     

[... truncated for brevity ...]

---

## Issue #N/A: Metal support?

**Link**: https://github.com/sgl-project/sglang/issues/23
**State**: closed
**Created**: 2024-01-17T20:16:06+00:00
**Closed**: 2024-07-25T06:32:59+00:00
**Comments**: 4
**Labels**: inactive

### Description

Hey, when is planned the support for Metal backend? 

---

## Issue #N/A: [Roadmap] FlashAttention3 Support as SGLang Attention Backend

**Link**: https://github.com/sgl-project/sglang/issues/4709
**State**: closed
**Created**: 2025-03-24T06:13:12+00:00
**Closed**: 2025-04-21T06:16:51+00:00
**Comments**: 5
**Labels**: high priority, collaboration

### Description


**Functionality**
- [x] Basic FA3 support including MHA Models (Llama, QWen and etc), Cuda Graph, Sliding Window (Gemma): https://github.com/sgl-project/sglang/pull/4680 @hebiao064 @qingquansong 
- [x] Support Page Size > 1 https://github.com/sgl-project/sglang/pull/4832 @hebiao064 
- [x] Support MLA for Deepseek-like models #4831 @Fridge003  
- [x] Support Speculative Decoding [PR1](https://github.com/sgl-project/sglang/pull/4951), [PR2](https://github.com/sgl-project/sglang/pull/5050/files), [PR3](https://github.com/sgl-project/sglang/pull/5168) [PR4](https://github.com/sgl-project/sglang/pull/5318) @qingquansong @hebiao064 @zcnrex 
- [x] Figure out how to build FA3 into SGLang: https://github.com/sgl-project/sglang/pull/4902 @yinfan98 
- [x] Add E2E Test like `sglang/test/srt/test_triton_attention_backend.py`: https://github.com/sgl-project/sglang/pull/4760 @yubofredwang 
- [x] Support Multimodal  https://github.com/sgl-project/sglang/pull/5103 @zcnrex @mickqian @yizhang2077 
- [x]

[... truncated for brevity ...]

---

## Issue #N/A: [0.4.4.post1] DeepSeek-R1 Optimization Option Ablations

**Link**: https://github.com/sgl-project/sglang/issues/4616
**State**: closed
**Created**: 2025-03-20T07:41:44+00:00
**Closed**: 2025-06-04T00:19:41+00:00
**Comments**: 11
**Labels**: high priority, inactive

### Description

> [!NOTE]
> Updated on **2025-03-20**. Older albations can be found here: #3956

# Overview

We sincerely thanks for the help from [M0gician](http://m0gician.github.io/) for the massive experiments.

**As of 2025-03-20**, SGLang provides the following optimizations for DeepSeek V3/R1 models:

| Name                                        | Description                                                                                                                                                                                                                                     | Enabled by Default | Enable/Disable Argument                                                                                                                                   |
|---------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

## Issue #N/A: [Feature] Proposal for adding PD-Disaggregation Feature to SGLang

**Link**: https://github.com/sgl-project/sglang/issues/3554
**State**: closed
**Created**: 2025-02-13T22:46:07+00:00
**Closed**: 2025-03-22T02:56:52+00:00
**Comments**: 8

### Description

##  Principles of Design

1. Model-Agnostic Approach
The changes will be implemented in a model-agnostic manner, rather than being tailored to specific models.

2. Compatibility with Open-Source Projects
To enhance interoperability with leading open-source LLM serving frameworks and ensure better code portability in the future:
  
3. Flexibility & Extensibility
While the implementation will align with popular projects at the API level, the internal design will remain simple and flexible. Following the principle of "less is more", only the essential components will be defined, leaving the detailed implementation to concrete classes. 

## Proposed Changes

### User Interface

It is similar to vLLM, yet with greater flexibility for different KV transfer paradigms. 

Sample Commands:

```
# To start a KV producer sglang instance
python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --port 30000 --kv-transfer-config '{"kv_connector":"SomeConnector","kv_role":"kv_

[... truncated for brevity ...]

---

## Issue #N/A: [RFC][Feature] Support Remote Prefill in PD Disaggregation

**Link**: https://github.com/sgl-project/sglang/issues/6925
**State**: open
**Created**: 2025-06-06T16:50:14+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

With middle size(30B-70B) LLM, not all requests benefit from PD disaggregation (e.g. input tokens <= 128, output tokens >=512). In this case, remote prefill with conditional disaggregation is a nice to have feature.


### Related resources

Dynamo implement remote prefill in vLLM v0/v1
- https://github.com/ai-dynamo/dynamo/blob/main/docs/architecture/disagg_serving.md
- https://github.com/vllm-project/vllm/pull/17751
- https://github.com/vllm-project/vllm/pull/16677

---

## Issue #N/A: [Feature] Support tool calls for DeepSeek.

**Link**: https://github.com/sgl-project/sglang/issues/4379
**State**: closed
**Created**: 2025-03-13T09:30:22+00:00
**Closed**: 2025-04-21T03:00:52+00:00
**Comments**: 11

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

I saw from the official documentation (https://docs.sglang.ai/backend/function_calling.html) that sglang supports tool calls, but I can't seem to find the tool parse for deepseekv3/r1. Does this mean that the deepseek model does not support tool calls?

From the DeepSeek official website, it seems that function call support has been implemented on the model side, although it may still be unstable. https://api-docs.deepseek.com/zh-cn/guides/function_calling

### Related resources

_No response_

---

## Issue #N/A: [Roadmap] Llama 4 Support

**Link**: https://github.com/sgl-project/sglang/issues/5118
**State**: open
**Created**: 2025-04-07T08:06:44+00:00
**Comments**: 1
**Labels**: high priority, collaboration

### Description

- [x] Initial Llama 4 Support @CatherineSue @fzyzcjy @ispobock  @ch-wan  #5092 
- [x] Llama 4 User Guide @ch-wan @ispobock #5133
- [x] Vision Backbone Support @mickqian #5144 
- [ ] Local Attention Support in Various Attention Backbones
  - [x] FlashAttention V3
  - [ ] FlashInfer
  - [ ] Triton
- [ ] Quantization 
  - [x] FP8 @HandH1998 #5194
  - [ ] INT4 @AniZpZ
- [ ] Kernel Optimization
- [ ] Memory Optimization @tarinkk @Pb314314  #6563 
- [ ] EP Optimization
- [x] Llama4 Tool Call Support @CatherineSue #5725 


---

## Issue #N/A: Development Roadmap (2025 H2)

**Link**: https://github.com/sgl-project/sglang/issues/7736
**State**: open
**Created**: 2025-07-03T06:04:23+00:00
**Comments**: 0
**Labels**: high priority, collaboration

### Description

The SGLang team is expected to complete planning for the H2 roadmap within the next two weeks. Stay tuned—exciting things are on the way!


---

## Issue #N/A: [Feature] support function call for Qwen3 models

**Link**: https://github.com/sgl-project/sglang/issues/6040
**State**: closed
**Created**: 2025-05-06T03:15:59+00:00
**Closed**: 2025-06-12T09:50:47+00:00
**Comments**: 11
**Labels**: function-calling

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

function call is a very important feature of the qwen3 model, and it is currently supported by some other frameworks (such as vllm, Ollama, etc.). We hope that sglang can also support it as soon as possible.

### Related resources

_No response_

---

## Issue #N/A: [Feature] Optimizing DeepSeek with the DeepSeek Infra OSS component

**Link**: https://github.com/sgl-project/sglang/issues/3758
**State**: closed
**Created**: 2025-02-21T11:52:28+00:00
**Closed**: 2025-03-10T18:28:27+00:00
**Comments**: 4
**Labels**: high priority, performance, deepseek

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

ref https://github.com/deepseek-ai/open-infra-index

- [ ] https://github.com/deepseek-ai/DeepEP
- [ ] https://github.com/deepseek-ai/DeepGEMM

### Related resources

_No response_

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

## Issue #N/A: [Tracking] Model support

**Link**: https://github.com/sgl-project/sglang/issues/7429
**State**: open
**Created**: 2025-06-21T22:18:10+00:00
**Comments**: 22
**Labels**: good first issue, new-model

### Description

### **[Tracking] Model support**

The goal is to support other model architectures available. Expand the model zoo 🎊 

The goal is to implement support for all architectures listed below. Anyone is welcome to take any issue or implement the model below.

If you need help implementing a new model, see https://docs.sglang.ai/supported_models/support_new_models.html

#### Text-only Language Models (Generative)
- [ ] `OPTForCasualLM` (facebook/opt-125m) #7440 
- [ ] `AquilaForCausalLM` (Aquila, Aquila2)
- [ ] `ArcticForCausalLM` (Arctic) #5768
- [ ] `BambaForCausalLM` (Bamba)
- [ ] `BartForConditionalGeneration` (BART)
- [ ] `BloomForCausalLM` (BLOOM, BLOOMZ)
- [ ] `Cohere2ForCasualLM` #4570
- [ ] `DeciLMForCausalLM` (DeciLM)
- [ ] `FalconForCausalLM` (Falcon)
- [ ] `FalconH1ForCausalLM` (Falcon-H1) #6517
- [ ] `FalconMambaForCausalLM` (FalconMamba)
- [ ] `Dots1ForCasualLM` (dots.llm1) #6471
- [ ] `GPT2LMHeadModel` (GPT-2)
- [ ] `GPTBigCodeForCausalLM` (StarCoder, SantaCoder)
- [ ] `GPTJFo

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] liuhaotian/llava-v1.6-mistral-7b doesn't load

**Link**: https://github.com/sgl-project/sglang/issues/128
**State**: closed
**Created**: 2024-01-31T23:55:51+00:00
**Closed**: 2024-07-25T06:32:21+00:00
**Comments**: 16
**Labels**: inactive

### Description

When trying to load the Mistral variant of LLaVa 1.6, I get an expected error:

```sh
python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.6-mistral-7b --chat-template vicuna_v1.1 --port 30000
```

```
ValueError: The checkpoint you are trying to load has model type `llava_mistral` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date
```

Transformers doesn't treat the LLaVa variants any differently, they all use the same config.  I think this *could* be easily fixed by adding a mapping from `llava_mistral` to the `LlavaConfig` in the config mapping.  

---

## Issue #N/A: Gemma3n Usage

**Link**: https://github.com/sgl-project/sglang/issues/7574
**State**: open
**Created**: 2025-06-26T21:14:01+00:00
**Comments**: 3
**Labels**: high priority

### Description

~~Due to some compatible issues, we need to manually install the latest version of transformers and timm by:
`pip install -U transformers timm`.~~

The latest SGLang version 0.4.8.post1 could not work with gemma3n, and was fixed in latest main.
To solve the issue, please **install from source** by: (please remove `uv` if you don't use)

```
git clone https://github.com/sgl-project/sglang.git
cd sglang
uv pip install -e "python[all]"
```

Launch the server with:
`python -m sglang.launch_server --model-path google/gemma-3n-E4B-it --attention-backend fa3`

If you encounter any issue when running the gemma3n, welcome to comment under this issue.

Known issues:
1. `TypeError: unsupported operand type(s) for %: 'list' and 'int'` : Please follow above instruction

---

## Issue #N/A: [Feature] Lora Development Roadmap

**Link**: https://github.com/sgl-project/sglang/issues/2929
**State**: open
**Created**: 2025-01-16T21:30:56+00:00
**Comments**: 0
**Labels**: help wanted, lora

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Features 

- [x] triton kernel & benchmark #3161 @Fridge003 
- [x] accuracy alignment #2671 #3413 @Fridge003 
- [x] test cases enhancement #3414 #3652 #4492 #4925 @aoshen524 @jcbjcbjc
- [x] support multi-rank adaptors #4492 @jcbjcbjc
- [x] support tensor parallel #2931 #4274 @aoshen524 
- [ ] compatibility with radix attention #2880 @Sunt-ing @jcbjcbjc
- [x] compatibility with cuda graph #3282 #4115 @Qiaolin-Yu  @Beichen-Ma 
- [x] support phi4mm #6544 @lifuhuang 
- [ ] support lora for embedding layer #3438 @Beichen-Ma 
- [x] load/unload #7412 #7446 @lifuhuang  @Fridge003 
- [ ] optimizing speed #2372 #3323 #6961 @jcbjcbjc @Fridge003 @lifuhuang 
- [ ] unified paging (support lora with different ranks) #3647 @Sunt-ing @jcbjcbjc

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] SGLang Support for TileLang

**Link**: https://github.com/sgl-project/sglang/issues/4221
**State**: closed
**Created**: 2025-03-09T05:34:49+00:00
**Closed**: 2025-05-27T00:18:53+00:00
**Comments**: 10
**Labels**: help wanted, high priority, inactive

### Description

We recently came across an interesting project: [TileLang](https://github.com/tile-ai/tilelang). It appears to offer significant advantages over Triton in many cases while maintaining a clean dataflow and simple syntax.

Do we have any plans to support a TileLang backend in SGLang?

For instance, TileLang has demonstrated up to **5x speedup** over Triton’s Flash MLA implementations on H100, with a kernel implementation of just **80 lines of code (see document:** https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_mla). Given these promising results, it would be valuable to explore its potential integration.

Would love to hear thoughts on this!


---

## Issue #N/A: [Feature]  Support Gemma 3 QAT models

**Link**: https://github.com/sgl-project/sglang/issues/5591
**State**: open
**Created**: 2025-04-21T05:53:29+00:00
**Comments**: 1
**Labels**: high priority, new-model

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Hello SGLang team,

Could you please add support for the quantization-aware training models of Google's Gemma 3? Thanks!

### Related resources

_No response_

---

## Issue #N/A: [Bug][minimal reproducible demo] High variability across batch inference runs

**Link**: https://github.com/sgl-project/sglang/issues/1729
**State**: closed
**Created**: 2024-10-20T14:07:03+00:00
**Closed**: 2025-02-25T00:17:03+00:00
**Comments**: 12
**Labels**: bug, inactive

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

## Background

This bug might be related to #1316.

When asking the model a block of questions it should answer with `yes` followed by a block of questions that should be answered by `no` a degradation in quality can be observed for some runs, when running the same data many times.

## Standard `lmsysorg/sglang:v0.3.3.post1

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support RM API

**Link**: https://github.com/sgl-project/sglang/issues/1384
**State**: closed
**Created**: 2024-09-11T04:27:29+00:00
**Closed**: 2024-10-19T14:52:16+00:00
**Comments**: 9
**Labels**: high priority

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Does SGLang support rapid deployment of RM services?
Or convenient custom APIs? It seems that currently there are only chat/completion/embedding APIs. As a newcomer to inference acceleration, any help would be beneficial.

### Related resources

copied from https://github.com/vllm-project/vllm/issues/6620, same demand

---

## Issue #N/A: Development Roadmap (2024 Q4)

**Link**: https://github.com/sgl-project/sglang/issues/1487
**State**: closed
**Created**: 2024-09-21T22:38:00+00:00
**Closed**: 2025-03-03T18:43:18+00:00
**Comments**: 27

### Description

Here is the development roadmap for 2024 Q4. Contributions and feedback are welcome ([**Join Bi-weekly Development Meeting**](https://t.co/4BFjCLnVHq)). Previous 2024 Q3 roadmap can be found in #634.

## Performance
- [x] Hide CPU overhead with overlapped scheduler (#1738, #2067)
- [x] Support speculative decoding
  - Eagle  #2150 
  - Reference-based. #270
  - Medusa head #859
  - Draft model based.
- [x] Sparse Attention #1459
- [x] Faster grammar parsing library for constrained decoding #1752 
- [x] Multi-layer radix cache (GPU/CPU/Disk) https://github.com/sgl-project/sglang/pull/2693  @xiezhq-hermann 
- [ ] Improve the performance of mixed chunked prefill. see a draft #1383 
- [ ] Integrate CuDNN paged attention [kernels](https://github.com/NVIDIA/cudnn-frontend/blob/v1.8.0/samples/python/52_scaled_dot_product_attention_with_paged_caches.ipynb) 

## Parallelism
- [ ] Support sequence parallelism #1436. Related [paper](https://www.arxiv.org/pdf/2411.01783)
- [ ] Support pipeline par

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

