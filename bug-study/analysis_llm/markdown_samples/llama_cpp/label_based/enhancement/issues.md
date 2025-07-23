# enhancement - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 2
- Closed Issues: 28

### Label Distribution

- enhancement: 30 issues
- stale: 19 issues
- model: 1 issues
- generation quality: 1 issues
- Ascend NPU: 1 issues
- research 🔬: 1 issues

---

## Issue #N/A: Feature Request: Support Falcon Mamba 7B 

**Link**: https://github.com/ggml-org/llama.cpp/issues/9009
**State**: closed
**Created**: 2024-08-12T16:29:58+00:00
**Closed**: 2024-08-21T08:06:37+00:00
**Comments**: 8
**Labels**: enhancement

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Please support Falcon Mamba 7B from TII (Technology Innovation Institute TII - UAE)

### Motivation

Support for all models is helpful.

My acid test for whether a model will run is to try and make a quant using "gruff my repo".

Admittedly it is hot off the presses yet it ought to run at least in theory, but it doesn't.
```
Error: Error converting to fp16: b'INFO:hf-to-gguf:Loading model: falcon-mamba-7b\nERROR:hf-to-gguf:Model FalconMambaForCausalLM is not supported\n'
```

### Possible 

[... truncated for brevity ...]

---

## Issue #N/A: Investigate gemma 2 generation quality

**Link**: https://github.com/ggml-org/llama.cpp/issues/8240
**State**: closed
**Created**: 2024-07-01T16:52:28+00:00
**Closed**: 2024-10-16T01:11:07+00:00
**Comments**: 90
**Labels**: enhancement, stale

### Description

Initial reports can be seen from https://github.com/ggerganov/llama.cpp/pull/8227

> [!IMPORTANT]  
> A note for everyone: if you think there's a bug in llama.cpp tokenizer, please make sure to test with HF `transformers` library first (see [this comment](https://github.com/ggerganov/llama.cpp/issues/8240#issuecomment-2212444937) for example)

---

## Issue #N/A: Support for 2-bit Quantized Llama-2-7b-chat-hf_2bitgs8_hqq Model

**Link**: https://github.com/ggml-org/llama.cpp/issues/6368
**State**: closed
**Created**: 2024-03-28T14:15:03+00:00
**Closed**: 2024-05-14T01:31:12+00:00
**Comments**: 5
**Labels**: enhancement, stale

### Description

I would like to propose the integration of a novel model, "Llama-2-7b-chat-hf_2bitgs8_hqq," available on Hugging Face. This model represents an innovative approach to quantization, employing a 2-bit quantized version of Llama2-7B-chat, enhanced with a low-rank adapter (HQQ+), to improve performance and efficiency.

Key Features:
- **Quantization**: The model leverages 2-bit quantization, significantly reducing VRAM requirements.
- **Low-Rank Adapter**: Utilizes HQQ+, a low-rank adapter for performance enhancement.
- **Efficiency**: Offloads meta-data to CPU, optimizing GPU memory usage.
- **Datasets**: Trained on a mixture of general and specialized datasets, showing robustness and versatility.

The inclusion of this model could greatly benefit llama.cpp users by offering a more memory-efficient yet powerful option for large-scale text generation tasks. It could especially be beneficial for environments with limited hardware resources.

Thank you for considering this addition

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Paligemma Support

**Link**: https://github.com/ggml-org/llama.cpp/issues/9227
**State**: closed
**Created**: 2024-08-28T22:01:53+00:00
**Closed**: 2024-11-27T01:07:42+00:00
**Comments**: 3
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Adding support for converting Google's multimodal Paligemma model to gguf in order to be used in ollama.

### Motivation

I have a personal project that requires a multimodal llm running locally and llava seems to be kind of...not great. I have seen an issue like this marked as open, but as of now, I still get an error when trying to convert from hf to gguf.

### Possible Implementation

_No response_

---

## Issue #N/A: Performance decreated between tag b1500 and b2581 on Windows ARM64 PC

**Link**: https://github.com/ggml-org/llama.cpp/issues/6417
**State**: closed
**Created**: 2024-04-01T03:20:36+00:00
**Closed**: 2024-07-08T01:06:56+00:00
**Comments**: 54
**Labels**: enhancement, stale

### Description

Hi LLAMA team, 

I use llama tag b2581 on Windows ARM64 PC, the performance is more lower than previous tag b1500. Please refer to below detailed information. What is the reason? Please help on this issue. 

Thanks a lot!

**[Detailed information]**

**Command:**
main.exe -m llama-2-7b-chat.ggufv3.q4_0.bin --color  --ctx_size 2048 -n -1 -ins -b 256 --top_k 10000 --temp 0.2 --repeat_penalty 1.1 -t 10

**Prompt:** I have 3 years of experience as a software developer. Now I got bored with coding and want to transition to another career. My education qualifications are B. Tech in computer science, and I am well-versed in understanding the business side of software as well. Suggest a list of career options that are easy for me to transition.


**system_info:** n_threads = 10 / 12 | AVX = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 0 | NEON = 1 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 0 | SSSE3 = 0 | VSX = 0 |

**Tag

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: XiaomiMiMo/MiMo-7B-RL

**Link**: https://github.com/ggml-org/llama.cpp/issues/13218
**State**: closed
**Created**: 2025-04-30T17:17:04+00:00
**Closed**: 2025-06-27T01:08:01+00:00
**Comments**: 3
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Add support of XiaomiMiMo/MiMo-7B-RL https://huggingface.co/XiaomiMiMo/MiMo-7B-RL

### Motivation

Model MiMoForCausalLM is not supported,Hope to further enrich the ecosystem.

### Possible Implementation

_No response_

---

## Issue #N/A: truly opensource model called olmo

**Link**: https://github.com/ggml-org/llama.cpp/issues/6712
**State**: closed
**Created**: 2024-04-16T23:43:40+00:00
**Closed**: 2024-05-07T19:39:44+00:00
**Comments**: 4
**Labels**: enhancement, model

### Description

Build with truly open dataset and fully open-source model can this be supported in olllama thanks.
https://allenai.org/olmo
https://huggingface.co/allenai/OLMo-7B


---

## Issue #N/A:  Intel® Core™ Ultra processors NPU  Support 

**Link**: https://github.com/ggml-org/llama.cpp/issues/5079
**State**: open
**Created**: 2024-01-22T14:15:28+00:00
**Comments**: 15
**Labels**: enhancement

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

 Intel® Core™ Ultra processors now has released  , how can llama.cpp use that npu to fast up 

# Motivation

 Intel® Core™ Ultra processors deliver three dedicated engines (CPU, GPU, and NPU) to help unlock the power of AI
https://www.intel.com/content/www/us/e

[... truncated for brevity ...]

---

## Issue #N/A: When I used the tool to quantify the chatglm model, the following error was reported

**Link**: https://github.com/ggml-org/llama.cpp/issues/3808
**State**: closed
**Created**: 2023-10-27T02:51:16+00:00
**Closed**: 2024-05-12T01:35:21+00:00
**Comments**: 8
**Labels**: enhancement, stale

### Description


When I used the tool to quantify the chatglm model, the following error was reported. May I ask if the format of the specified model does not match? Is there a way to solve this problem?


3:~/llama.cpp$ ./quantize MODEL/chatglm/chatGLM2-6B/pytorch_model-00001-of-00007.bin MODEL/chatglm/
                                             python convert.py MODEL/chatglm/chatGLM2-6B/
Loading model file MODEL/chatglm/chatGLM2-6B/pytorch_model-00001-of-00007.bin
Loading model file MODEL/chatglm/chatGLM2-6B/pytorch_model-00001-of-00007.bin
Loading model file MODEL/chatglm/chatGLM2-6B/pytorch_model-00002-of-00007.bin
Loading model file MODEL/chatglm/chatGLM2-6B/pytorch_model-00003-of-00007.bin
Loading model file MODEL/chatglm/chatGLM2-6B/pytorch_model-00004-of-00007.bin
Loading model file MODEL/chatglm/chatGLM2-6B/pytorch_model-00005-of-00007.bin
Loading model file MODEL/chatglm/chatGLM2-6B/pytorch_model-00006-of-00007.bin
Loading model file MODEL/chatglm/chatGLM2-6B/pytorch_model-00

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Speculative Decoding "acceptance rate" should not count drafts that were skipped via the " ignore small drafts" clause

**Link**: https://github.com/ggml-org/llama.cpp/issues/14048
**State**: closed
**Created**: 2025-06-06T11:04:38+00:00
**Closed**: 2025-06-10T15:48:08+00:00
**Comments**: 0
**Labels**: enhancement

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

I think the `slot.n_draft_total += draft.size()` should go after the "ignore small drafts" test here:

```cpp
                llama_tokens draft = common_speculative_gen_draft(slot.spec, params_spec, cached_text_tokens, id);

                // keep track of total number of tokens generated in the draft
                slot.n_draft_total += draft.size();

                // ignore small drafts
                if (slot.params.speculative.n_min > (int) draft.size()) {
                    SLT_DBG(slot

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: add DeepSeek-v3 support

**Link**: https://github.com/ggml-org/llama.cpp/issues/10981
**State**: closed
**Created**: 2024-12-26T11:08:12+00:00
**Closed**: 2025-01-04T20:06:12+00:00
**Comments**: 64
**Labels**: enhancement

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- Version b4391
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Add support for DeepSeek-v3

https://huggingface.co/deepseek-ai/DeepSeek-V3

Currently not supported:

`ERROR:hf-to-gguf:Model DeepseekV3ForCausalLM is not supported`

### Motivation

DeepSeek-v3 is a big MoE model of 685B params, would be great as offloading to RAM would be a must for most systems

### Possible Implementation

There is no model card or technical report yet. I don't know how much different from v2 it is.

Edit: they have uploaded the mode

[... truncated for brevity ...]

---

## Issue #N/A: Improving the repetition penalty

**Link**: https://github.com/ggml-org/llama.cpp/issues/331
**State**: closed
**Created**: 2023-03-20T15:43:12+00:00
**Closed**: 2023-09-14T13:23:49+00:00
**Comments**: 12
**Labels**: enhancement, generation quality

### Description

129c7d1e (#20) added a repetition penalty that prevent the model to run into loops.

Here are a few suggestions for possible enhancements:

 * One issue with the interactive mode is that the repetition penalty is affecting the anti-prompt and response prefix, causing the model to generate unnecessarily long responses. One solution could be to exclude these tokens from the penalty,
 * It is possible to exempt or reduce the penalty for stop words, punctuation characters, and newlines; maybe applying a frequency-based penalty instead,
 * Using an exponential decay, such that recent tokens are more penalized than older ones, causing less issues with large `repeat_last_n`  windows,
 * Token repetition is an approximation of sub-strings or word repetition, but it seems difficult to do otherwise without backtracking the inference.

---

## Issue #N/A: how to set this chat_template  in server?

**Link**: https://github.com/ggml-org/llama.cpp/issues/5974
**State**: closed
**Created**: 2024-03-10T10:44:42+00:00
**Closed**: 2024-04-24T01:06:37+00:00
**Comments**: 8
**Labels**: enhancement, stale

### Description

how to set this chat_template in openchat?
because i watched output it's difference from ./server and python  -m llama.cpp.server. then i thought, may is  difference chat_template made this?

openchat chat_template:
Using gguf chat template: {{ bos_token }}{% for message in messages %}{{ 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + '<|end_of_turn|>'}}{% endfor %}{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}

how to set chat_template in ./server with --chat-template


---

## Issue #N/A: Differences between cgraph->leafs and cgraph->nodes?

**Link**: https://github.com/ggml-org/llama.cpp/issues/5791
**State**: closed
**Created**: 2024-02-29T08:48:52+00:00
**Closed**: 2024-04-15T02:46:51+00:00
**Comments**: 3
**Labels**: enhancement, stale

### Description

Hello, I'm wondering what the differences between cgraph->leafs and cgraph->nodes. Thank you very much.

---

## Issue #N/A: Documentation of llama.h

**Link**: https://github.com/ggml-org/llama.cpp/issues/3870
**State**: closed
**Created**: 2023-10-31T16:08:46+00:00
**Closed**: 2023-10-31T17:50:30+00:00
**Comments**: 1
**Labels**: enhancement

### Description

I'm writing an [guidance](https://github.com/guidance-ai/guidance) inspired inference server using llama.cpp - I'm much more comfortable in rust and have generated bindings to `llama.h`.

I'm able to make progress by translating examples to rust, testing it, then trial and error'ing my way to the code I want, however I'd like to make a *safe* (in the rust sense) wrapper around llama.cpp then publish and maintain it. 

This involves knowing the implicit invariant used in `llama.h` (both currently and as the project moves forward) - which I don't think I can reasonably do at the moment.

Some examples of things I *think* are true and would like to see documented.
- ~`llama_batch.n_tokens` must never exceed the `n_tokens` used in `llama_batch_init` (basically document what memory is safe to write to in `llama_batch` after init.~ This is already done.
- `get_logits_ith` must be called with an `i` where `batch.logits[i] = true` and then `batch` was decoded. (currently references `la

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Qwen2.5-Omni

**Link**: https://github.com/ggml-org/llama.cpp/issues/12673
**State**: open
**Created**: 2025-03-31T14:02:13+00:00
**Comments**: 3
**Labels**: enhancement

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Pleas add support https://github.com/QwenLM/Qwen2.5-Omni

### Motivation

This is a multimodal which supports audio/video and text

### Possible Implementation

I dont think it is similar to other Qwen 2.5 models, right?

---

## Issue #N/A: Feature Request: MiniCPM 2.6 model support?

**Link**: https://github.com/ggml-org/llama.cpp/issues/8977
**State**: closed
**Created**: 2024-08-10T20:51:28+00:00
**Closed**: 2024-09-26T01:07:14+00:00
**Comments**: 3
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

I'd like to begin by expressing my sincere gratitude for your outstanding contributions. Your efforts have been instrumental in supporting and advancing the open-source community.

It would be fantastic to have support for 8 billion parameters vision models  that can truly rival the performance of leading proprietary models.



### Motivation

SOTA OSS VLM with only 8b params, a piece of art, rivals top models.

<img width="1155" alt="QVl0iPtT5aUhlvViyEpgs" src="https://github.

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Add support for Lite-Mistral-Instruct chat template

**Link**: https://github.com/ggml-org/llama.cpp/issues/8529
**State**: closed
**Created**: 2024-07-17T06:37:58+00:00
**Closed**: 2024-08-31T01:07:00+00:00
**Comments**: 2
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

OuteAI has released a new small [model](https://huggingface.co/OuteAI/Lite-Mistral-150M-v2-Instruct) that is very coherent for its size.

I am requesting the addition of this model's chat template to llama.cpp's list of supported templates

### Motivation

The model is already supported by llama.cpp. However, it's using a new chat template that isn't in the list of supported templates.  As a result, llama.cpp assumes ChatML for this model. Due to the model's size, it's very sensitive to the pro

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Generate Image Embeddings with llama.cpp

**Link**: https://github.com/ggml-org/llama.cpp/issues/13913
**State**: closed
**Created**: 2025-05-30T08:29:29+00:00
**Closed**: 2025-07-14T01:08:04+00:00
**Comments**: 1
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Maybe this is already possible, but I could not figure it out.

It would be great to use llama.cpp to generate image embeddings from a VLM, to use with a vector database.

### Motivation

I believe this is already possible and popular for text, and it makes sense to extend to images.

### Possible Implementation

_No response_

---

## Issue #N/A: Min-p Mixtral routing

**Link**: https://github.com/ggml-org/llama.cpp/issues/4470
**State**: closed
**Created**: 2023-12-14T14:13:36+00:00
**Closed**: 2024-03-18T01:46:17+00:00
**Comments**: 3
**Labels**: enhancement, stale

### Description

# Feature Description

The basic idea of this issue is that instead of using a fixed number of active experts per token and per layer (top-k), there would be a dynamic amount selected like min-p sampling (must be n% as good as the best gating layer score, or they aren’t selected). This could possibly be combined with top-k if higher expert counts aren’t working well.

# Motivation

The motivation behind this is that it would theoretically allow lower expert counts when it doesn’t significantly harm quality, and allow higher experts counts when needed. By controlling the min-p routing variable, you could control the speed-quality trade off, with high values giving higher speed at the cost of quality, and low values doing the opposite. We know that quality changes based on the active expert count (https://github.com/ggerganov/llama.cpp/pull/4406#issuecomment-1855151885), so dynamic selection could control this more effectively than a fixed amount.

# Possible Implementation

To

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: please add falcon 7b mamba support

**Link**: https://github.com/ggml-org/llama.cpp/issues/9048
**State**: closed
**Created**: 2024-08-15T19:16:41+00:00
**Closed**: 2024-08-21T08:06:38+00:00
**Comments**: 1
**Labels**: enhancement

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

please add falcon 7b mamba support ,

https://huggingface.co/tiiuae/falcon-mamba-7b-instruct
### Motivation

please add falcon 7b mamba support ,

https://huggingface.co/tiiuae/falcon-mamba-7b-instruct
### Possible Implementation

please add falcon 7b mamba support ,

https://huggingface.co/tiiuae/falcon-mamba-7b-instruct

---

## Issue #N/A: Feature Request: Add Host buffer type for Ascend NPU (CANN backend)

**Link**: https://github.com/ggml-org/llama.cpp/issues/9304
**State**: closed
**Created**: 2024-09-04T01:47:34+00:00
**Closed**: 2024-09-14T02:18:26+00:00
**Comments**: 2
**Labels**: enhancement, Ascend NPU

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Ascend NPU backend (CANN) is not support pin memory(Host buffer type) now. Using ping memory will make it more efficiency.

### Motivation

Other backend such as CUDA has already support Host buffer type.

### Possible Implementation

Refer to CUDA to implement the Host buffer type of Ascend NPU.

---

## Issue #N/A: Feature Request: shared tokens in batches with `logits = true`

**Link**: https://github.com/ggml-org/llama.cpp/issues/10295
**State**: closed
**Created**: 2024-11-14T15:21:44+00:00
**Closed**: 2025-01-03T01:07:26+00:00
**Comments**: 10
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

When batching, tokens that happen to have the same `(pos, id)` can be shared across multiple sequences.

However, if the last token in each sequence (the one we'd like logits for) happens to match with other tokens, they'd need to be processed as separate tokens, instead of taking advantage of the token grouping feature.
###### ^ Not sure if bug or by design, but if a token requests logits on multiple sequences with the same `(pos, id)`, only a single `logits` array will be returned b

[... truncated for brevity ...]

---

## Issue #N/A: tts : add support for SparkTTS

**Link**: https://github.com/ggml-org/llama.cpp/issues/12495
**State**: closed
**Created**: 2025-03-21T09:46:15+00:00
**Closed**: 2025-05-05T01:07:48+00:00
**Comments**: 1
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

HF: https://huggingface.co/SparkAudio/Spark-TTS-0.5B

### Motivation

A new TTS model that can be supported by llama.cpp.

### Possible Implementation

I might be wrong here but it seems like SparkTTS has a simlar architecture as OuteTTS and Orpheus TTS (#12476) but it uses Qwen2.5-0.5B.

They are using their own audio decoder called BiCodec. Sample python implementation: https://github.com/SparkAudio/Spark-TTS/blob/main/sparktts/models/bicodec.py

Similar model support (OuteTTS): https://github.co

[... truncated for brevity ...]

---

## Issue #N/A: Mixtral Experts are initialized from Mistral 7b - Low Rank conversion possible?

**Link**: https://github.com/ggml-org/llama.cpp/issues/4611
**State**: closed
**Created**: 2023-12-23T19:07:56+00:00
**Closed**: 2024-04-02T01:10:09+00:00
**Comments**: 10
**Labels**: enhancement, research 🔬, stale

### Description

![image](https://github.com/ggerganov/llama.cpp/assets/66376113/77a44caa-9fa6-4746-b3c0-9772d68661cb)
We have evidence that Mixtral's Experts were initialized from a "common ancestor", the original Mistral 7b.

Conceptually, the idea that might be able to take advantage of this is:
- Extracting the delta of the original Mistral 7b compared to each expert as a PEFT adapter for each expert
- Use SVD to get the closest low rank approximation on each (let's say we target r=128)
- Add the linear Mixtral routing layer to the original Mistral 7b
- At inference time, keep all the LoRA adapters for each expert in memory (approx. ~1.8b added parameters for 128 rank)
- Apply LoRA in real time for each batch of 'expert' calculations per layer using the corresponding expert's LoRA

This could be a viable alternative to [QMoE](https://github.com/ggerganov/llama.cpp/issues/4445) for approaching Mixtral's performance with significantly less memory, given the shared structural similarities.

---

## Issue #N/A: HPX Support

**Link**: https://github.com/ggml-org/llama.cpp/issues/4614
**State**: closed
**Created**: 2023-12-24T03:23:30+00:00
**Closed**: 2024-04-02T01:10:08+00:00
**Comments**: 2
**Labels**: enhancement, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ X ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

Add HPX support for data parallelism; this is an alternative to the current use of `std::thread`. The PR is here #4613 .

# Motivation

[HPX](https://github.com/STEllAR-GROUP/hpx) is an asynchronous many task (AMT) runtime system implementing the ISO C++ standard for data parallelism and concurrency. HPX additionally provides distributed memory support through an asynchronous global address space (AGAS) created over a cluster of machines. This issue is not a request to support AGAS. This PR is focused on adding HPX to manage data parallelism in llama.cpp. HPX implements a user-land thread library managed by a work stealing thread scheduler. HPX's user-land thread implementation means applications built with HPX will 

[... truncated for brevity ...]

---

## Issue #N/A: Question about llama.cpp and llava-cli when used with llava 1.6 for vision:

**Link**: https://github.com/ggml-org/llama.cpp/issues/5852
**State**: closed
**Created**: 2024-03-03T11:47:12+00:00
**Closed**: 2024-04-20T01:06:58+00:00
**Comments**: 5
**Labels**: enhancement, stale

### Description

I've been using the llava-v1.6-mistral-7b model for doing captions lately. I know it's relatively new and does some different things under the hood vs other/older vision models.

From the little bit of testing I've performed, it seems like server makes it fall back to the llava 1.5 vision, rather than using the 1.6 mode. When I check the total token counts, those always seem to be really low. This seems to affect any apps that use llama.cpp, like LM Studio and Jan. If I use llava-cli, with the same settings, the image alone encodes to 2880 tokens, which indicates that it's encoding the tiles correctly. Is there any way to make the server use llava-cli? Anyway to make llava-cli behave like a server? Am I doing something wrong?

I wrote a python program to batch caption folders of images, but I'm having to do it a really hacky way where it basically runs a command prompt behind the scenes, the python script captures the output of the window as a log, parses the log to trim out the no

[... truncated for brevity ...]

---

## Issue #N/A: Support for SmolLM

**Link**: https://github.com/ggml-org/llama.cpp/issues/8608
**State**: closed
**Created**: 2024-07-20T23:00:38+00:00
**Closed**: 2024-07-22T14:43:04+00:00
**Comments**: 0
**Labels**: enhancement

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Add support for [SmolLM](https://huggingface.co/blog/smollm) family of models

### Motivation

enhancement

### Possible Implementation

_No response_

---

## Issue #N/A: How do we finetune the model with new data?

**Link**: https://github.com/ggml-org/llama.cpp/issues/466
**State**: closed
**Created**: 2023-03-24T16:12:02+00:00
**Closed**: 2024-04-10T01:07:59+00:00
**Comments**: 17
**Labels**: enhancement, stale

### Description

Can we have a finetune.cpp or finetune.exe file to incorporate new data into the model? The use case will be to design an AI model that can do more than just general chat. It can become very knowledgeable in specific topics they are finetuned on. Also, after creating the finetune.exe , please ensure no GPU is required for the entire process. Because that is what makes this repo awesome in the first place.

---

## Issue #N/A: Native Intel IPEX-LLM Support

**Link**: https://github.com/ggml-org/llama.cpp/issues/7190
**State**: closed
**Created**: 2024-05-10T02:02:37+00:00
**Closed**: 2024-07-09T01:06:58+00:00
**Comments**: 13
**Labels**: enhancement, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X ] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

I have found this closed issue where someone manually (?how?) implemented IPEX-LLM. However, looking forward to native IPEX-LLM support for Intel Xe iGPUs + Intel Arc dGPUs on Windows and Linux 

https://github.com/ggerganov/llama.cpp/issues/7042

TL;DR is I

[... truncated for brevity ...]

---

