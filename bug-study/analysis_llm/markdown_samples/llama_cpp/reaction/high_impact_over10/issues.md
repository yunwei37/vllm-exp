# high_impact_over10 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 29

### Label Distribution

- stale: 20 issues
- enhancement: 16 issues
- model: 7 issues
- help wanted: 2 issues
- question: 2 issues
- documentation: 1 issues
- demo: 1 issues
- server/webui: 1 issues
- high priority: 1 issues
- duplicate: 1 issues

---

## Issue #N/A: [Feature Request] Dynamic temperature sampling for better coherence / creativity

**Link**: https://github.com/ggml-org/llama.cpp/issues/3483
**State**: closed
**Created**: 2023-10-05T02:23:01+00:00
**Closed**: 2024-06-12T01:06:49+00:00
**Comments**: 47
**Labels**: stale

### Description

# Prerequisites

- [✅] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Idea

Typical sampling methods for large language models, such as Top P and Top K, (as well as alternative sampler modes that decide the Top K dynamically like Mirostat) are based off the assumption that a static temperature value (a consistently randomized probability distribution) is the ideal sampler conditioning. Mirostat, most notably, was designed to 'learn' a certain targeted level of 'entropy' over time; this helped the model find the most grammatically coherent selection of tokens to be considered by the sampler for good results. Most of these sampling implementations weren't designed to be used together. Some, like TFS, were created when the largest available models were smaller ones like GPT2. Those models struggled a _lot_ more when attempting to generalize in different directions, and it makes sense to 

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Support for Deepseek Janus-Pro-7B & Janus-1.3B

**Link**: https://github.com/ggml-org/llama.cpp/issues/11490
**State**: closed
**Created**: 2025-01-29T14:53:13+00:00
**Closed**: 2025-04-22T01:08:02+00:00
**Comments**: 4
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

DeepSeek recently released **[Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B)** and **[Janus-1.3B](https://huggingface.co/deepseek-ai/Janus-1.3B)**, both multimodal models currently supported in [Transformers](https://github.com/huggingface/transformers). 



**Resources:** [Janus GitHub](https://github.com/deepseek-ai/Janus)


### Motivation

Adding them to `llama.cpp` would enable efficient local inference, expanding support for state-of-the-art multimodal AI. Would love to see t

[... truncated for brevity ...]

---

## Issue #N/A: llama : add Falcon LLM support

**Link**: https://github.com/ggml-org/llama.cpp/issues/1602
**State**: closed
**Created**: 2023-05-26T17:45:06+00:00
**Closed**: 2023-08-23T20:11:44+00:00
**Comments**: 210
**Labels**: help wanted, model

### Description

Falcon LLM 40b and 7b were just open sourced under a license which allows commercial use (~~with royalties for over $1 million revenue per year~~) and have are topping the Huggingface Open LLM leaderboard. It seems to be based on a modified gpt3 architecture. I’m wondering if support in llama.cpp would be considered.

https://huggingface.co/tiiuae/falcon-40b

---

## Issue #N/A: HQQ quantization

**Link**: https://github.com/ggml-org/llama.cpp/issues/4782
**State**: closed
**Created**: 2024-01-05T10:52:53+00:00
**Closed**: 2024-04-02T01:08:44+00:00
**Comments**: 7
**Labels**: enhancement, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

Add support to new 
[HQQ](https://mobiusml.github.io/hqq_blog/) (Half-Quadratic Quantization)  quantization method. HQQ requires no calibration data and it takes less than 5 minutes to process Llama-2-70B quantized to 2-bit outperforming the full-precision Llama-2-13B accordin

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Add support for SmolVLM

**Link**: https://github.com/ggml-org/llama.cpp/issues/10877
**State**: closed
**Created**: 2024-12-17T23:47:12+00:00
**Closed**: 2025-03-24T01:07:52+00:00
**Comments**: 4
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Support running https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct

### Motivation

SmolVLM is a small and mighty multimodal model provided by huggingface. The article about it: https://huggingface.co/blog/smolvlm

### Possible Implementation

_No response_

---

## Issue #N/A: Can we finetune existing models via the ideas of QLORA

**Link**: https://github.com/ggml-org/llama.cpp/issues/1938
**State**: closed
**Created**: 2023-06-19T13:48:02+00:00
**Closed**: 2024-04-10T01:06:52+00:00
**Comments**: 5
**Labels**: stale

### Description

QLORA gives a idea that we can still use quantized weights and LoRA to fine tune a model. As backward caculation is most done already, maybe we can look at this:

- Evaluate if we need to do double quantize to further optimize for VRAM usage over speed.
- implement LoRA finetune in llama? or a standalone application?
- Add gpu offload support to compute grad.

---

## Issue #N/A: Feature Request: add per-request "reasoning" options in llama-server

**Link**: https://github.com/ggml-org/llama.cpp/issues/13272
**State**: closed
**Created**: 2025-05-02T21:27:22+00:00
**Closed**: 2025-07-11T01:08:01+00:00
**Comments**: 7
**Labels**: enhancement, stale

### Description

### Feature Description

As reasoning models are becoming mainstream, we start to see some pattern:
- Most models use `<think>`, `<reasoning>`, etc, basically a set of known tokens now
- The "reasoning budget" can technically be supported by any models, not just Qwen, by keeping track of number of tokens between `<think>` and `</think>`
- "no think" is just a reasoning budget == 0

So I'm thinking about accepting an object like this for each request:

```"reasoning": {
"reasoning": {
    "budget": -1, // number of reasoning tokens budget
                     default: -1 (inf) ; 0 for no think
    "format": "", // equivalent of --reasoning-format
                     if set to "deepseek", reasoning will be returned in "message.reasoning_content"
                     if set to "hide", it will be completely hidden
                     default: "none", return the reasoning with the message as normal
}
```

The reasoning format "hide" can be implemented via https://github.com/ggml-org/llama

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Support for C4AI Command R7B / Cohere2ForCausalLM

**Link**: https://github.com/ggml-org/llama.cpp/issues/10816
**State**: closed
**Created**: 2024-12-13T18:54:15+00:00
**Closed**: 2025-01-04T14:33:33+00:00
**Comments**: 12
**Labels**: enhancement

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

I would like to request support for **C4AI Command R7B** by Cohere.

Here is some relevant information:

Download link: https://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024

Some specifications:

- A well-rounded model
- Model Size: 7 billion parameters
- Context length: 128K
- Enhanced efficiency in math, code, and reasoning tasks
- Multilingual, reasoning, tool use.
- RAG capability

Blog post: https://cohere.com/blog/command-r7b


### Motivation

I believe it will be a g

[... truncated for brevity ...]

---

## Issue #N/A: Feature request: Graphical GGUF viewer

**Link**: https://github.com/ggml-org/llama.cpp/issues/6715
**State**: open
**Created**: 2024-04-17T04:30:46+00:00
**Comments**: 18
**Labels**: enhancement, stale

### Description

# Motivation

With the recent introduction of `eval-callback` example, we now having more tools for debugging when working with llama.cpp. However, one of the tool that I feel missing is the ability to dump everything inside a gguf file into a human-readable (and interactive) interface.

Inspired from `huggingface.js` where users can visualize the KV and list of tensors on huggingface.com, I would like to implement the same thing in llama.cpp. I find this helpful in these situations:
- Debugging `convert.py` script when adding a new architecture
- Debugging tokenizers
- Debugging changes related to gguf (model splits for example)
- Debugging tensors (i.e. display N first elements of a tensor, just like `eval-callback`)
- Debugging control vectors
- ... (maybe other usages in the future)

The reason why I can't use `huggingface.js` is because it's based on browser, which make it tricky when reading a huge local file. It also don't have access to quantized types (same for `gg

[... truncated for brevity ...]

---

## Issue #N/A: Document check sums of models so that we can confirm issues are not caused by bad downloads or conversion

**Link**: https://github.com/ggml-org/llama.cpp/issues/238
**State**: closed
**Created**: 2023-03-17T12:50:44+00:00
**Closed**: 2023-05-02T13:41:32+00:00
**Comments**: 9
**Labels**: documentation, model

### Description

Can someone please confirm the following md5 sums are correct?  I regenerated them with the latest code.

```
$ md5sum ./models/*/*.pth | sort -k 2,2
0804c42ca65584f50234a86d71e6916a  ./models/13B/consolidated.00.pth
016017be6040da87604f77703b92f2bc  ./models/13B/consolidated.01.pth
f856e9d99c30855d6ead4d00cc3a5573  ./models/30B/consolidated.00.pth
d9dbfbea61309dc1e087f5081e98331a  ./models/30B/consolidated.01.pth
2b2bed47912ceb828c0a37aac4b99073  ./models/30B/consolidated.02.pth
ea0405cdb5bc638fee12de614f729ebc  ./models/30B/consolidated.03.pth
9deae67e2e7b5ccfb2c738f390c00854  ./models/65B/consolidated.00.pth
0c4b00c30460c3818bd184ee949079ee  ./models/65B/consolidated.01.pth
847194df776dd38f8ae9ddcede8829a1  ./models/65B/consolidated.02.pth
3b6c8adcb5654fd36abab3206b46a0f1  ./models/65B/consolidated.03.pth
68d61d1242597ad92616ec31b8cb6b4c  ./models/65B/consolidated.04.pth
7f71259eaee2b906aa405d8edf39925f  ./models/65B/consolidated.05.pth
0574e26b6891ab2cb0df7340d773fe

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Add support for Phi-3.5 MoE and Vision Instruct

**Link**: https://github.com/ggml-org/llama.cpp/issues/9119
**State**: closed
**Created**: 2024-08-21T14:32:40+00:00
**Closed**: 2025-02-12T01:07:21+00:00
**Comments**: 24
**Labels**: enhancement, model, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Microsoft has recently dropped two new models in the Phi Family. 

3.5 MoE: https://huggingface.co/microsoft/Phi-3.5-MoE-instruct
3.5 Vision: https://huggingface.co/microsoft/Phi-3.5-vision-instruct

It would be nice to see support added to llama.cpp for these two models. 

### Motivation

Supporting all model releases so the wider community can enjoy these great free models. 

### Possible Implementation

_No response_

---

## Issue #N/A: Server: add function calling API

**Link**: https://github.com/ggml-org/llama.cpp/issues/5588
**State**: closed
**Created**: 2024-02-19T13:47:28+00:00
**Closed**: 2024-06-16T01:07:14+00:00
**Comments**: 10
**Labels**: enhancement, demo, server/webui, stale

### Description

# Motivation

This subject is already brought up in https://github.com/ggerganov/llama.cpp/issues/4216 , but my initial research failed.

Recently, I discovered a new line of model designed specifically for this usage: https://github.com/MeetKai/functionary

This model can decide whether to call functions (and which function to be called) in a given context. The chat template looks like this:

```
{#v2.2#}
{% for message in messages %}
  {% if message['role'] == 'user' or message['role'] == 'system' %}
    {{ '<|from|>' + message['role'] + '\n<|recipient|>all\n<|content|>' + message['content'] + '\n' }}
  {% elif message['role'] == 'tool' %}
    {{ '<|from|>' + message['name'] + '\n<|recipient|>all\n<|content|>' + message['content'] + '\n' }}
  {% else %}
    {% set contain_content='no'%}
    {% if message['content'] is not none %}
      {{ '<|from|>assistant\n<|recipient|>all\n<|content|>' + message['content'] }}
      {% set contain_content='yes'%}
    {% endif %}

[... truncated for brevity ...]

---

## Issue #N/A: What is the meaning of hacked?

**Link**: https://github.com/ggml-org/llama.cpp/issues/33
**State**: closed
**Created**: 2023-03-12T04:35:26+00:00
**Closed**: 2023-03-12T05:09:52+00:00
**Comments**: 5
**Labels**: question

### Description

Hey, I was reading your Readme.md and I saw that your repo was hacked. I want to ask what this means and wanted to check if the users like me also get the impact of hacking. Or, this is not the thing I should worry about?

---

## Issue #N/A: Support for Minigpt-4?

**Link**: https://github.com/ggml-org/llama.cpp/issues/1069
**State**: closed
**Created**: 2023-04-19T19:08:38+00:00
**Closed**: 2024-04-09T01:10:07+00:00
**Comments**: 4
**Labels**: stale

### Description

Minigpt-4 is recently released, which is a multimodal model capable of handling both text and image inputs (similar to GPT-4, but this is built by Vicuna-13b and Blit2). From their demo, the model looks very capable! It seems to handle image inputs pretty well. Unfortunately, it currently runs only on GPU, and seems to require 24GB of VRAM.

[Minigpt-4 Github Page](https://github.com/Vision-CAIR/MiniGPT-4)

I am wondering if it is possible to run it on CPU? Also, do you think 4-bit quantization is possible? Vicuna-13b works very well with 4-bit quantization, but I have no idea if the inclusion of Blit2 changes anything, or is it even possible to make a quantization at all. I feel like it would be great if somehow the CPU version could be developed. Thanks!

---

## Issue #N/A:  LLaVA-NeXT-Video-34B

**Link**: https://github.com/ggml-org/llama.cpp/issues/7201
**State**: closed
**Created**: 2024-05-10T14:42:08+00:00
**Closed**: 2024-06-25T02:41:22+00:00
**Comments**: 1
**Labels**: enhancement, stale

### Description

Hello 

Are there any plans to add [LLaVA-NeXT-Video-34B](https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-34B) to llamacpp?

---

## Issue #N/A: Using non LoRA Alpaca model

**Link**: https://github.com/ggml-org/llama.cpp/issues/303
**State**: closed
**Created**: 2023-03-19T20:03:48+00:00
**Closed**: 2023-07-28T19:35:59+00:00
**Comments**: 11
**Labels**: question, model

### Description

The following repo contains a recreation of the original weights for Alpaca, without using LoRA. How could we use that model with this project? https://github.com/pointnetwork/point-alpaca
Thanks a bunch!

---

## Issue #N/A: llama : add Mixtral support

**Link**: https://github.com/ggml-org/llama.cpp/issues/4381
**State**: closed
**Created**: 2023-12-08T18:20:09+00:00
**Closed**: 2023-12-13T12:04:31+00:00
**Comments**: 62
**Labels**: enhancement, high priority, model

### Description

Hi,
Please add support for [Mistral's MOE model Mixtral](https://twitter.com/MistralAI/status/1733150512395038967).

---

## Issue #N/A: Support JetMoE

**Link**: https://github.com/ggml-org/llama.cpp/issues/6499
**State**: closed
**Created**: 2024-04-05T00:14:59+00:00
**Closed**: 2024-09-24T01:07:32+00:00
**Comments**: 5
**Labels**: enhancement, stale

### Description

Very interesting new open source model just dropped, called JetMoE. It looks like the current SOTA as far as compute efficiency goes.

It has a very interesting model architecture:

> **Model Details**
> JetMoE-8B has 24 blocks. Each block has two MoE layers: Mixture of Attention heads (MoA) and Mixture of MLP Experts (MoE). Each MoA and MoE layer has 8 expert, and 2 experts are activated for each input token. It has 8 billion parameters in total and 2.2B active parameters. JetMoE-8B is trained on 1.25T tokens from publicly available datasets, with a learning rate of 5.0 x 10-4 and a global batch-size of 4M tokens.

![image](https://github.com/ggerganov/llama.cpp/assets/15776622/82cf791a-22ee-4402-a299-736c135928ed)

- Website: https://research.myshell.ai/jetmoe
- Github: https://github.com/myshell-ai/JetMoE
- HuggingFace: https://huggingface.co/jetmoe/jetmoe-8b
- Chat Demo on Lepton AI: https://www.lepton.ai/playground/chat?model=jetmoe-8b-chat


| Model           | Act

[... truncated for brevity ...]

---

## Issue #N/A: [Proposal] "Stable" C API

**Link**: https://github.com/ggml-org/llama.cpp/issues/171
**State**: closed
**Created**: 2023-03-15T18:01:09+00:00
**Closed**: 2023-03-15T20:29:20+00:00
**Comments**: 4
**Labels**: duplicate, enhancement

### Description

I propose refactoring `main.cpp` into a library (`llama.cpp`, compiled to `llama.so`/`llama.a`/whatever) and making `main.cpp` a simple driver program. A simple C API should be exposed to access the model, and then bindings can more easily be written for Python, node.js, or whatever other language.

This would partially solve #82 and #162.

Edit: on that note, is it possible to do inference from two or more prompts on different threads? If so, serving multiple people would be possible without multiple copies of model weights in RAM.

---

## Issue #N/A: Investigate supporting starcode

**Link**: https://github.com/ggml-org/llama.cpp/issues/1326
**State**: closed
**Created**: 2023-05-04T21:04:22+00:00
**Closed**: 2023-05-18T12:34:49+00:00
**Comments**: 8
**Labels**: help wanted, model

### Description

Bigcode just released [starcoder](https://huggingface.co/bigcode/starcoder). This is a 15B model trained on 1T Github tokens. This seems like it could be an amazing replacement for gpt-3.5 and maybe gpt-4 for local coding assistance and IDE tooling!

More info: https://huggingface.co/bigcode

---

## Issue #N/A: server: phi-3 end token not handled?

**Link**: https://github.com/ggml-org/llama.cpp/issues/6903
**State**: closed
**Created**: 2024-04-25T10:58:12+00:00
**Closed**: 2024-06-25T02:41:36+00:00
**Comments**: 13
**Labels**: bug-unconfirmed, stale

### Description

Phi-3 4k model include in all responses the end token "<|end|>"

Im using: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf and llama.cpp for docker cuda server in the latest version.

Thanks in advance.

---

## Issue #N/A: [Prompt Processing] Is there a way to speed up prompt processing for Metal? (M1/M2)

**Link**: https://github.com/ggml-org/llama.cpp/issues/2428
**State**: closed
**Created**: 2023-07-27T22:00:17+00:00
**Closed**: 2024-04-09T01:07:22+00:00
**Comments**: 16
**Labels**: stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Prompt eval time is around twice as long as eval time (12 tokens/sec vs 22 tokens/sec). Is there a way to make them both the same speed?

# Current Behavior

Prompt eval time takes twice as long as eval time.

---

## Issue #N/A: [macos] AMD GPU using mul_mm in metal

**Link**: https://github.com/ggml-org/llama.cpp/issues/3000
**State**: closed
**Created**: 2023-09-04T02:36:03+00:00
**Closed**: 2024-04-05T01:06:24+00:00
**Comments**: 1
**Labels**: stale

### Description



when i remove these and related stuff on ggml-metal.h, and compile, it can load model and run on gpu but nothing really work (gpu usage just stuck 98% and just hang on terminal)

```
        GGML_METAL_ADD_KERNEL(mul_mm_f16_f32);
        GGML_METAL_ADD_KERNEL(mul_mm_q4_0_f32);
        GGML_METAL_ADD_KERNEL(mul_mm_q8_0_f32);
        GGML_METAL_ADD_KERNEL(mul_mm_q4_1_f32);
        GGML_METAL_ADD_KERNEL(mul_mm_q2_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mm_q3_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mm_q4_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mm_q5_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mm_q6_K_f32);
```

but when revert all, it just not working at all when loading model with gpu 

```
ggml_metal_init: loaded kernel_mul_mat_q5_K_f32            0x7fcb478145f0 | th_max =  768 | th_width =   64
ggml_metal_init: loaded kernel_mul_mat_q6_K_f32            0x7fcb47814dd0 | th_max = 1024 | th_width =   64
ggml_metal_init: loaded kernel_mul_mm_f16_f32               

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Slim Attention (lossless 2x reduction in KV cache size)

**Link**: https://github.com/ggml-org/llama.cpp/issues/12359
**State**: closed
**Created**: 2025-03-13T03:48:19+00:00
**Closed**: 2025-05-25T01:08:18+00:00
**Comments**: 4
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Slim attention can reduce KV cache size by a factor of 2 without a loss of accuracy as it's lossless: https://arxiv.org/pdf/2503.05840

### Motivation

Allows you to run with larger context sizes at the same (V)RAM usage or allows you to cram the same context into less (V)RAM. Furthermore, it improves performance at long context sizes.

### Possible Implementation

_No response_

---

## Issue #N/A: llama: add Grok support

**Link**: https://github.com/ggml-org/llama.cpp/issues/6120
**State**: closed
**Created**: 2024-03-17T20:31:28+00:00
**Closed**: 2024-05-08T17:05:36+00:00
**Comments**: 21
**Labels**: enhancement, good first issue, model

### Description

Hi,
Please add support for Grok.
Thanks!

Relevant links:
* https://github.com/xai-org/grok
* https://x.ai/blog/grok-os
* https://twitter.com/grok/status/1769441648910479423
* [NEW] Official Upload (thx to @dranger003) for linking: https://huggingface.co/xai-org/grok-1

---

## Issue #N/A: CLBlast build failing on q3 model

**Link**: https://github.com/ggml-org/llama.cpp/issues/1725
**State**: closed
**Created**: 2023-06-06T23:52:55+00:00
**Closed**: 2024-04-10T01:07:43+00:00
**Comments**: 2
**Labels**: stale

### Description

When trying to run `wizardlm-30b.ggmlv3.q3_K_M.bin` from https://huggingface.co/TheBloke/WizardLM-30B-GGML using CLBlast build, it fails with `GGML_ASSERT: D:\a\llama.cpp\llama.cpp\ggml-opencl.cpp:1009: to_fp32_cl != nullptr`

```
PS H:\Files\Downloads\llama-master-2d7bf11-bin-win-clblast-x64> .\main.exe -m C:\temp\models\wizardlm-30b.ggmlv3.q3_K_M.bin -ngl 20
main: build = 631 (2d7bf11)
main: seed  = 1686095068
ggml_opencl: selecting platform: 'NVIDIA CUDA'
ggml_opencl: selecting device: 'NVIDIA GeForce RTX 3080'
ggml_opencl: device FP16 support: false
llama.cpp: loading model from C:\temp\models\wizardlm-30b.ggmlv3.q3_K_M.bin
llama_model_load_internal: format     = ggjt v3 (latest)
llama_model_load_internal: n_vocab    = 32001
llama_model_load_internal: n_ctx      = 512
llama_model_load_internal: n_embd     = 6656
llama_model_load_internal: n_mult     = 256
llama_model_load_internal: n_head     = 52
llama_model_load_internal: n_layer    = 60
llama_model_load_internal

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support resetting the status of llama_context

**Link**: https://github.com/ggml-org/llama.cpp/issues/1780
**State**: closed
**Created**: 2023-06-09T16:07:18+00:00
**Closed**: 2024-04-10T01:07:30+00:00
**Comments**: 4
**Labels**: stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Here's my assumption: sometimes one may want to restart a session with the model. However if inputting the new prompt directly, the previous session will impact on the new session (unless the new prompt is long enough), because `llama.eval` will leave some effects in 

[... truncated for brevity ...]

---

## Issue #N/A: Rockchip RK3588 perf

**Link**: https://github.com/ggml-org/llama.cpp/issues/722
**State**: closed
**Created**: 2023-04-02T20:39:28+00:00
**Closed**: 2023-04-02T22:14:36+00:00
**Comments**: 103

### Description

Just did a very simple run with llama-7b-4bit. It... took a while. Had it run in a screen. But, it worked!

```
root@FriendlyWrt /s/o/llama.cpp (master)# time ./main --color -m models/ggml-model-q4_0.bin -p "Hello there!"
main: seed = 1680443840
llama_model_load: loading model from 'models/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 4096
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 32
llama_model_load: n_layer = 32
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 11008
llama_model_load: n_parts = 1
llama_model_load: type    = 1
llama_model_load: ggml map size = 4017.70 MB
llama_model_load: ggml ctx size =  81.25 KB
llama_model_load: mem required  = 5809.78 MB (+ 1026.00 MB per state)
llama_model_load: loading tensors from 'models/ggml-model-q4_0.bin'
llama_model_load: model size =  4017.27 MB / num tensors = 291
llama_ini

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Add support for StarVector-8b/1b

**Link**: https://github.com/ggml-org/llama.cpp/issues/12666
**State**: closed
**Created**: 2025-03-31T04:52:29+00:00
**Closed**: 2025-05-16T01:07:54+00:00
**Comments**: 1
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

I have trid to convert [starvector(github)](https://github.com/joanrod/star-vector) [huggingface](https://huggingface.co/starvector/starvector-8b-im2svg) ```safetensors``` to ```gguf``` using ``` convert_hf_to_gguf.py ``` but failed.

```bash
python llama.cpp/convert_hf_to_gguf.py starvector-8b/
INFO:hf-to-gguf:Loading model: starvector-8b
ERROR:hf-to-gguf:Model StarVectorForCausalLM is not supported
```

Add support for StarVector (StarVectorForCasualLM) to run this model on llama.cpp, and more ov

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Pixtral by Mistral support (pixtral-12b-240910)

**Link**: https://github.com/ggml-org/llama.cpp/issues/9440
**State**: closed
**Created**: 2024-09-11T18:03:29+00:00
**Closed**: 2025-02-08T01:07:14+00:00
**Comments**: 14
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Dear llama.cpp team,

Mistral has just released Pixtral and I would like to request support for it, if possible.

Here are some relevant links:

**X announcement:** https://x.com/mistralai/status/1833758285167722836

**Magnet link:** `xt=urn:btih:7278e625de2b1da598b23954c13933047126238a&dn=pixtral-12b-240910&tr=udp%3A%2F%http://2ftracker.opentrackr.org/%3A1337%2Fannounce&tr=udp%3A%2F%http://2fopen.demonii.com/%3A1337%2Fannounce&tr=http%3A%2F%http://2ftracker.ipv6tracker.org/%3A80

[... truncated for brevity ...]

---

