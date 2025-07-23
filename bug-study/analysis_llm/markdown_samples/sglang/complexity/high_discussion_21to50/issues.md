# high_discussion_21to50 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 10
- Closed Issues: 20

### Label Distribution

- high priority: 9 issues
- inactive: 7 issues
- collaboration: 6 issues
- help wanted: 5 issues
- good first issue: 4 issues
- bug: 2 issues
- feature: 2 issues
- enhancement: 2 issues
- blackwell: 2 issues
- new-model: 1 issues

---

## Issue #N/A: [Bug] Unable to fix model output

**Link**: https://github.com/sgl-project/sglang/issues/1316
**State**: closed
**Created**: 2024-09-03T11:01:15+00:00
**Closed**: 2024-11-01T04:13:00+00:00
**Comments**: 25
**Labels**: bug, high priority

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

The performance of sglang is very good. I am comparing the output accuracy of vllm, Hugging Face, and sglang. Using Qwen's model, I set do_sample to false or temperature to 0 to fix the output. Through comparison, the outputs of vllm and the Hugging Face transformer library are consistent. However, sglang does not produce consistent output

[... truncated for brevity ...]

---

## Issue #N/A: Development Roadmap (2025 H1)

**Link**: https://github.com/sgl-project/sglang/issues/4042
**State**: open
**Created**: 2025-03-04T00:09:49+00:00
**Comments**: 23
**Labels**: collaboration

### Description

Here is the development roadmap for 2025 H1. Contributions and feedback are welcome ([**Join Bi-weekly Development Meeting**](https://docs.google.com/document/d/1xEow4eIM152xNcRxqZz9VEcOiTQo8-CEuuQ5qTmkt-E/edit?usp=sharing)). The previous 2024 Q4 roadmap can be found in #1487

## Focus
- Throughput-oriented large-scale deployment similar to the [deepseek inference system](https://github.com/deepseek-ai/open-infra-index?tab=readme-ov-file#day-6---one-more-thing-deepseek-v3r1-inference-system-overview)
- Long context optimizations
- Low latency speculative decoding
- Reinforcement learning training framework integration
- Kernel optimizations

## Parallelism
- [x] Support PD disaggregation @ByronHsu  #4655
- [x] Support expert parallelism and load balancer #5524
- [x] Support pipeline parallelism @Ying1123 #5724
- [x] Support data parallelism attention compatible with all other parallelism #4390 
- [x] Support overlap communication in TP/EP @tom @Zhuohao-Li #4068
- [ ] Improvements of sg

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

## Issue #N/A: [Bug] sglang crashed when use enable_dp_attention running DeepSeekV3 on 2x8xH100

**Link**: https://github.com/sgl-project/sglang/issues/3658
**State**: closed
**Created**: 2025-02-18T06:53:23+00:00
**Closed**: 2025-05-22T00:19:10+00:00
**Comments**: 29
**Labels**: inactive

### Description

[server.log](https://github.com/user-attachments/files/18840336/server.log)

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

According to the dp-attention performance & usage, I turn on it by --enable-dp-attention when launching DeepSeek v3 on 2x8xH100. My command is like as below:
`docker run --gpus all -d --entrypoint=python3 --shm-size 32g --privileged -e NCCL_IB_HCA=mlx5_1,mlx5_2,ml

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] common_ops.abi3.so: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSsb

**Link**: https://github.com/sgl-project/sglang/issues/5100
**State**: closed
**Created**: 2025-04-06T10:23:24+00:00
**Closed**: 2025-05-08T21:26:52+00:00
**Comments**: 32

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hello, 

I successfully built the sgl-kernel with sm_120 (NVIDIA RTX 50 series) and CUDA 12.8, but encountered the following issue when running `sglang.launch_server `command. Please help.

```
INFO 04-06 14:51:04 [__init__.py:256] Automatically detected platform cuda.
Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Still very slow performance for mid-range prompt sizes ~ 8k tokens with MLA

**Link**: https://github.com/sgl-project/sglang/issues/5031
**State**: closed
**Created**: 2025-04-03T10:16:15+00:00
**Closed**: 2025-04-17T19:36:01+00:00
**Comments**: 23
**Labels**: bug, high priority

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

~8k token prompts are even worse than 0.4.3.post2 (was 10 tokens/sec, not 6 tokens/sec) so it now about 5x slower than without MLA.  But of course MLA is faster on very long context (with MLA its about 24 tokens/sec at 118k tokens input, and without is 3 tokens/sec).

But there is never a win-win scenario here.  Either 8k is very bad (now 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] sglang run for few hours, it will stop returning valid response

**Link**: https://github.com/sgl-project/sglang/issues/1270
**State**: closed
**Created**: 2024-08-30T18:46:21+00:00
**Closed**: 2024-11-14T23:46:56+00:00
**Comments**: 21

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

sglang run for few hours, it will stop returning valid response, based on the pm2 logs, it does not triggering any error, or message
<img width="620" alt="image" src="https://github.com/user-attachments/assets/9af04d3e-c831-464f-8932-2cacea3dd300">



Expected its always return token output as below
<img width="1030" alt="i

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

## Issue #N/A: [Roadmap] Prefill and Decoding Disaggregation

**Link**: https://github.com/sgl-project/sglang/issues/4655
**State**: open
**Created**: 2025-03-21T19:26:55+00:00
**Comments**: 30
**Labels**: enhancement, high priority, collaboration

### Description

### Design: 

[SGLang PD Disaggregation (Open Source)](https://docs.google.com/document/d/1rQXJwKd5b9b1aOzLh98mnyMhBMhlxXA5ATZTHoQrwvc/edit?tab=t.0#heading=h.i3s2t1j0e1ik)

### Progress
- [x] Release initial code @ByronHsu  #4654
  - prefill and decode event loop, queue, and transfer interface
  - **transfer engine is faked** 
  - easy python load balancer
- [x] Mooncake integration @ShangmingCai   https://github.com/sgl-project/sglang/pulls?q=is%3Apr+mooncake+is%3Aopen
- [x] NIXL Integration @trevor-m #5477
- [x] PD + overlap schedule @ByronHsu 
- [x] PD + DP attention @ch-wan @ByronHsu 
- [x] PD + fault tolerance https://github.com/sgl-project/sglang/pull/6504 https://github.com/sgl-project/sglang/pull/6263
- [x] PD + spec decode https://github.com/sgl-project/sglang/pull/6507
- [x] PD + logprob https://github.com/sgl-project/sglang/pull/6558
- [x] PD + Structured Output https://github.com/sgl-project/sglang/pull/6560

- [x] PD + retract @Ying1123 https://github.com/sgl-project/sglan

[... truncated for brevity ...]

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

## Issue #N/A: [Bug] PD disaggregation, KV transfer slow down under high concurrency

**Link**: https://github.com/sgl-project/sglang/issues/5450
**State**: open
**Created**: 2025-04-16T06:35:41+00:00
**Comments**: 36

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When running bench_serving, we send 200 prompts(isl=4k, osl=2k) in one shot, the first 100 reqs's KV were transferred in 30s. But the second 100 reqs's KV transfer took 5min.

I've print out the prealloc queue and transfer queue size in decode node. We can see the transfer-queue size was decreasing very slowly after 2025-04-16 12:40:10.  [

[... truncated for brevity ...]

---

## Issue #N/A: DeepSeek-R1 Optimization Option Ablations

**Link**: https://github.com/sgl-project/sglang/issues/3956
**State**: closed
**Created**: 2025-02-28T09:33:38+00:00
**Closed**: 2025-07-06T00:22:09+00:00
**Comments**: 36
**Labels**: high priority, inactive, deepseek, speculative-decoding

### Description

Updated on **2025-03-20**: #4616

Updated on **2025-03-04**:
# DeepSeek Optimization Ablations

## Overview

We sincerely thanks for the help from [M0gician](http://m0gician.github.io/) for the massive experiments.

**As of 2025-03-04**, SGLang provides the following optimizations for DeepSeek V3/R1 models:

| Name                                        | Description                                                                                                                                                                                                                                     | Enabled by Default | Enable/Disable Argument                                                                                                                                   |
|---------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

[... truncated for brevity ...]

---

## Issue #N/A: [Tracker] Blackwell support

**Link**: https://github.com/sgl-project/sglang/issues/5338
**State**: open
**Created**: 2025-04-13T04:35:37+00:00
**Comments**: 29
**Labels**: enhancement, blackwell

### Description

## Usage

```bash
docker pull lmsysorg/sglang:blackwell

# use latest main
cd /sgl-workspace/sglang && git pull
```

## Models

### DeepSeek V3 ✅
```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code
```

### Llama 4 ✅
```bash
python3 -m sglang.launch_server --model meta-llama/Llama-4-Scout-17B-16E-Instruct --tp 8 --context-length 131072
```

---

## Issue #N/A: [Bug] 1.5B is bizarrely OOM on 80G A100

**Link**: https://github.com/sgl-project/sglang/issues/4547
**State**: closed
**Created**: 2025-03-18T08:12:24+00:00
**Closed**: 2025-05-26T00:20:02+00:00
**Comments**: 47
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Simply specify the following parameters, and an error occurs during inference with qwen2.5-1.5b-instruct:

```python
llm = sgl.Engine(model_path=m, dp_size=device_count())
{
    "max_new_tokens": 512,
    "temperature": 0.0,
    "stop": [
        "}\n",
        "<|endoftext|>",
        "<|im_end|>",
        "</s>",
        "## Question:",


[... truncated for brevity ...]

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

## Issue #N/A: prometheus query return no result

**Link**: https://github.com/sgl-project/sglang/issues/2677
**State**: closed
**Created**: 2024-12-31T06:23:14+00:00
**Closed**: 2025-04-15T00:18:42+00:00
**Comments**: 24
**Labels**: inactive

### Description

Hi, thank you for your great work. I'm new to Prometheus and Grafana and I'd like to use sglang with them.  After I set up according to the [document](https://sgl-project.github.io/references/production_metrics.html#../examples/monitoring/grafana.json), the dashboards show no data. 
![微信图片_20241231141427](https://github.com/user-attachments/assets/d4cac92b-d6e8-40d7-b15d-d8b4473f009a)
I tried to query in prometheus directly with expr in [example](https://github.com/sgl-project/sglang/blob/main/examples/monitoring/grafana.json), no result as well.
![微信图片_20241231142210](https://github.com/user-attachments/assets/e67364db-ef84-42e3-bf12-af4915a1146a)
please help me out

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

## Issue #N/A: [Bug] sglang[all]>=0.4.7

**Link**: https://github.com/sgl-project/sglang/issues/7070
**State**: open
**Created**: 2025-06-10T23:10:43+00:00
**Comments**: 25
**Labels**: high priority

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Almost half the performance drop when running 0.4.7 vs 0.4.6

Tested two models
Qwen3-32B-FP8 75T/s (4xAda 6000s) to 45T/s
Qwen3-30B-A3B - 160T/s  (4x3090s) BF16 to 80T/s with 0.4.7


### Reproduction

python -m sglang.launch_server --model-path models/Qwen3-32B-FP8 \
--context-length 131072 \
--json-model-override-args '{"rope_scaling":{"

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

## Issue #N/A: [Feature] support bert rerank model and  openai "score" api

**Link**: https://github.com/sgl-project/sglang/issues/5577
**State**: closed
**Created**: 2025-04-20T15:19:47+00:00
**Closed**: 2025-06-20T17:16:13+00:00
**Comments**: 21

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

1. also need to support Bert rerank model with BertForSequenceClassification architectures in this feature.

2. score api design:

  -   input paramter
  
    
    "text_1": "What is the capital of France?"
    "text_2": [
    "The capital of Brazil is Brasilia.",
    "The capital of France is Paris."
    ]
    
  
  -  api result
  
 
    "data": [
        {
          "index": 0,
          "score": xxxxxxxx
        },
        {
          "index": 1,
          "score": xxxxxxxx
        }
      ]

### Related resources

_No response_

---

## Issue #N/A: [Bug] NCCL Crash with SIGSEGV Frequently when deploying deepseek v3

**Link**: https://github.com/sgl-project/sglang/issues/2803
**State**: closed
**Created**: 2025-01-09T02:30:13+00:00
**Closed**: 2025-03-05T03:42:54+00:00
**Comments**: 29
**Labels**: help wanted, high priority

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

 ```
 Caught signal 11 (Segmentation fault: address not mapped to object at address 0x3)                                                                                              ==== backtrace (tid: 212877) ====                                                                                                                             

[... truncated for brevity ...]

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

## Issue #N/A: [Feature] Reorganize all the docs

**Link**: https://github.com/sgl-project/sglang/issues/3596
**State**: open
**Created**: 2025-02-15T20:15:50+00:00
**Comments**: 26
**Labels**: documentation, good first issue, help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

1. [Quick Start: Sending Requests](https://docs.sglang.ai/backend/send_request.html) move to Getting Started

2. 

<img width="807" alt="Image" src="https://github.com/user-attachments/assets/6175edbf-484e-4fc2-9a40-f7e6cd31dcb9" />

3. print_highlight

if is_in_ci, use html. else directly print it.

4. differentiate two streaming https://docs.sglang.ai/backend/send_request.html

<img width="799" alt="Image" src="https://github.com/user-attachments/assets/41508c86-5837-40f0-96b8-bf424b32aaca" />

5. add description to this docs:

<img width="735" alt="Image" src="https://github.com/user-attachments/assets/2a7a2875-7cf2-4489-86c4-40937366b3c6" />

6. https://docs.sglang.ai/backend/openai_api_completions.html add lin

[... truncated for brevity ...]

---

## Issue #N/A: Tutorial for Batch Decoding and Obtaining Log Probs

**Link**: https://github.com/sgl-project/sglang/issues/81
**State**: closed
**Created**: 2024-01-23T05:13:56+00:00
**Closed**: 2024-01-30T14:39:59+00:00
**Comments**: 25

### Description

Hi
Thanks for the great library
I have a usecase which I think will benefit a lot from Radix Attention. I need to obtain log probs for around a 100K sequences which can be binned into groups of 100 having a similar prefix like 'Wikipedia originated in' and having 100 different suffixes. I do not need to generate anything and I only need the log probs for the input. Is there a tutorial for such a usecase?

---

## Issue #N/A: [Feature] deepseek v3 60 tokens/sec on deepseek API vs. 13 tokens/sec on sglang

**Link**: https://github.com/sgl-project/sglang/issues/3196
**State**: closed
**Created**: 2025-01-28T18:40:18+00:00
**Closed**: 2025-02-15T01:21:30+00:00
**Comments**: 29
**Labels**: help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

The PR for AMD + sglang and NVIDIA + sglang was that it was "fully" supported, but it seems something is off by the speed.  A single sequence runs at only order 13 tokens/sec for long generation with TTFT order 2 seconds.  This is consistent with vLLM as well.  True for either 8*MI300X or 8*H200 or 2*8*H200.

For only 37B parameters + 14B MOE parameters, this seems way too slow.  Also, deepseek API (before it started to break down) was order 60 tokens/sec early on and they advertise 60 tokens/sec.  This is more aligned with the parameters active.

What is missing from truly fully suppporting deepseek V3 and R1?  Can these features be enumerated and added in a roadmap?

### Related resources

_No response_

---

## Issue #N/A: [Bug] The AWQ model has different inference response times for Qwen/Qwen2.5-VL-7B-Instruct-AWQ between versions v0.4.3post2 and v0.4.4post3, with v0.4.3post2 having a shorter response time

**Link**: https://github.com/sgl-project/sglang/issues/5123
**State**: open
**Created**: 2025-04-07T10:40:58+00:00
**Comments**: 33

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

The AWQ model has different inference response times for Qwen/Qwen2.5-VL-7B-Instruct-AWQ between versions v0.4.3post2 and v0.4.4post3  with v0.4.3post2 having a shorter response time

SGLang v0.4.4post3 and SGLang v0.4.3post2 test
Qwen/Qwen2.5-VL-7B-Instruct-AWQ
0.4.3post2 is fine but 0.4.4post3 is response time long

### Reproduction

imp

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Qwen/Qwen2.5-VL-7B-Instruct-AWQ sglang response time longer

**Link**: https://github.com/sgl-project/sglang/issues/4916
**State**: closed
**Created**: 2025-03-30T11:24:52+00:00
**Closed**: 2025-06-04T00:19:44+00:00
**Comments**: 27
**Labels**: inactive

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

I found that when testing with Qwen/Qwen2.5-VL-7B-Instruct-AWQ, the response time of sglang is longer, while the response time of vllm is shorter. Why is this?

or


SGLang v0.4.4post3 and SGLang v0.4.3post2 test
Qwen/Qwen2.5-VL-7B-Instruct-AWQ

0.4.3post2 is fine  but 0.4.4post3 is response time long


### Reproduction



import requests


[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Key conflict of `AutoImageProcessor.register`

**Link**: https://github.com/sgl-project/sglang/issues/4159
**State**: closed
**Created**: 2025-03-07T04:06:18+00:00
**Closed**: 2025-03-25T12:17:44+00:00
**Comments**: 22
**Labels**: MLLM

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

The following ValueError was raised when attempting to serve any model within a recent Docker container:

`Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] RuntimeError: RMSNorm failed with error code invalid configuration argument

**Link**: https://github.com/sgl-project/sglang/issues/3304
**State**: closed
**Created**: 2025-02-05T02:25:13+00:00
**Closed**: 2025-05-11T15:17:16+00:00
**Comments**: 22
**Labels**: good first issue, help wanted

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hi, I am using the main branch of SGLang, and downloading Mixtral-8x22B from huggingface. 

CUDA: 12.4
2 nodes, each has 4 H100 96GB.

I am deploying the server using:
```
python -m sglang.launch_server --model-path Mixtral-8x22B-v0.1 --tp 8 --dist-init-addr xxx:5000 --nnodes 2 --node-rank 0 --trust-remote-code --disable-cuda-graph
python 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support awq quantization for MoE model on CPU.

**Link**: https://github.com/sgl-project/sglang/issues/5324
**State**: closed
**Created**: 2025-04-12T10:41:48+00:00
**Closed**: 2025-06-30T00:21:16+00:00
**Comments**: 39
**Labels**: inactive

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

I tried to run deepseek-r1-awq on CPU with vLLM, but I failed.This is because vLLM will convert awq to awq_marlin if the model is a MoE model, and marlin kernal is not supported on CPU. And I can't run the model directly in awq quantization, because [awq.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/awq.py) doesn't handle fused MoE case, this will result in unexpected errors.
I browse [/python/sglang/srt/layers/quantization/awq.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/awq.py) and notice that it doesn't handle fused MoE case as well. I suggest that the support for fused MoE models could be added:)

### Related resources

_N

[... truncated for brevity ...]

---

