# research_🔬 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 7
- Closed Issues: 23

### Label Distribution

- research 🔬: 30 issues
- stale: 12 issues
- help wanted: 9 issues
- performance: 7 issues
- enhancement: 6 issues
- generation quality: 5 issues
- roadmap: 5 issues
- high priority: 3 issues
- good first issue: 3 issues
- model: 2 issues

---

## Issue #N/A: llama : mitigate KV cache fragmentation

**Link**: https://github.com/ggml-org/llama.cpp/issues/3380
**State**: closed
**Created**: 2023-09-28T15:09:15+00:00
**Closed**: 2024-02-27T12:35:52+00:00
**Comments**: 10
**Labels**: enhancement, performance, research 🔬

### Description

With the new unified KV cache implementation from #3228 we now support batched decoding.

At runtime, depending on the workload and the length of the decoded sequences, the KV cache can become fragmented. If we think of the cache as an array and each cell being free, or belonging to a certain sequence, then we could end up with many short segments of free cells, instead of one big segment with all the free cells. This hinders the performance since for each new batch, we have to find a free cache segment that can "hold" the entire batch and when we cannot do so, we have to start splitting the batch in smaller batches to be able to fit it.

One possible mitigation is from time to time (based on some logic, or based on user request) to "defragment" the cache. This can be implemented in a very similar way to the existing KV shift functionality, where we add extra graph nodes when there is need to defragment the cache.

Other approaches might be possible. For example, based on the bat

[... truncated for brevity ...]

---

## Issue #N/A: Optimisation of per-token CPU activities for GPU inference

**Link**: https://github.com/ggml-org/llama.cpp/issues/7456
**State**: closed
**Created**: 2024-05-22T08:24:24+00:00
**Closed**: 2024-08-23T01:07:15+00:00
**Comments**: 5
**Labels**: performance, research 🔬, stale

### Description

When using a GPU backend, for each token evaluation there exists not only computation on the GPU but also significant CPU computation which can potentially be optimized. 

Here are some timing measurements of the critical path for each token for llama2 Q4_K_M 7B and 13B models on A100 and H100 GPUs.

Firstly, here are absolute times:  
<img src="https://github.com/ggerganov/llama.cpp/assets/10851179/fb8ee0a5-09e1-4a05-a042-f60964694f8f" width="70%">


and here are the same data presented as a percentage breakdown in each case:
<img src="https://github.com/ggerganov/llama.cpp/assets/10851179/8ea0edfe-95de-43ac-8088-b996e3e0870e" width="70%">

`CUDA Graph Execution` is the time spent executing the compute graph on the GPU, which is responsible for around 85-90% of the time taken in evaluating each token..
 
The remaining 10-15% of the time is taken by CPU activities, the most dominant of which are discussed below.

**GGML Graph Preparation:** `llama_build_graph` and `ggml_

[... truncated for brevity ...]

---

## Issue #N/A: Research: Performance differences between Metal (macOS) and Vulkan (Linux)

**Link**: https://github.com/ggml-org/llama.cpp/issues/10982
**State**: closed
**Created**: 2024-12-26T11:12:21+00:00
**Closed**: 2025-05-04T01:08:09+00:00
**Comments**: 14
**Labels**: research 🔬, stale

### Description

I'm one of the developers for the Asahi Linux GPU drivers, which provide accelerated Vulkan and OpenGL support on Apple Silicon platforms. I'm interested in improving the performance of llama.cpp on our drivers with the Vulkan backend.

As things stand today, macOS is significantly faster on a quick test with `llama-bench`, with default settings (tested on an M2 Max 64GB):

Linux:

```
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = Apple M2 Max (G14C B1) (Honeykrisp) | uma: 1 | fp16: 1 | warp size: 32 | matrix cores: none
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
ggml_vulkan: Compiling shaders................................Done!
| llama 7B Q4_K - Medium         |   4.07 GiB |     7.24 B | Vulkan     |  99 |         pp512 |         92.16 ± 0.08 |
| llama 7B Q4_K - Medium   

[... truncated for brevity ...]

---

## Issue #N/A: Investigate the performance (speed and perplexity) of Q4_0 with 2x F16 factors

**Link**: https://github.com/ggml-org/llama.cpp/issues/995
**State**: closed
**Created**: 2023-04-15T12:24:00+00:00
**Closed**: 2023-04-22T08:43:17+00:00
**Comments**: 1
**Labels**: help wanted, high priority, research 🔬

### Description

The current `Q4_0` uses a single F32 floating-point scaling factor.

An idea was proposed by @ikawrakow to change this to use 2x F16 factors instead of 1x F32: https://github.com/ggerganov/llama.cpp/commit/679e1cb6c01b16abe4f3ee3c849813b98970df93
Initial results indicate that this might be as accurate as `Q4_1` and hopefully as fast as current `Q4_0`.

The goal of this task is to try to implement efficiently this data format (quantization, dequantization and dot product), measure the speed and perplexity and decide if this is viable. Depending on the results, we can think about updating the current `Q4_0` data format and potentially dropping support for `Q4_1`.

### SIMD implementation progress

- [x] ARM NEON
- [x] AVX
- [ ] WASM

I plan to work on the ARM NEON implementation.
If you want to help with any of the implementations, propose an implementation + results in a PR, summarizing the inference speed and the obtained perplexity of your implementation.

### Related

[... truncated for brevity ...]

---

## Issue #N/A: Investigate alternative approach for Q4 quantization 

**Link**: https://github.com/ggml-org/llama.cpp/issues/397
**State**: closed
**Created**: 2023-03-22T16:03:20+00:00
**Closed**: 2023-04-25T17:20:48+00:00
**Comments**: 58
**Labels**: help wanted, good first issue, research 🔬

### Description

Currently, in [Q4_0](https://github.com/ggerganov/ggml/pull/27) quantization we choose the scaling factor for each 32 group of weights as `abs(max(x_i))/7`. It is easy to see that this is suboptimal.

Consider quantization of the following 4 numbers:

`0.1 0.2 0.3 0.6`

Currently, we would determine a scaling factor of `0.6 / 7 ~= 0.0857` and the dequantized numbers will be:

`0.0857 0.1714 0.3428 0.6`

So the RMS between the dequantized and original values will be non-zero:

`sqrt((0.1 - 0.0857)^2 + (0.2 - 0.1714)^2 + (0.3 - 0.3428)^2 + (0.6 - 0.6)^2) > 0.0`

However, if we choose the scaling factor to be `0.1` instead, then it is easy to see that the original numbers will be quantized perfectly.

So the scaling factor is better to be chosen as the one that minimises some error (e.g. RMS or whatever is more meaningful and easy to compute). Doing that we will certainly achieve better accuracy compared to the existing approach. The question is - how much better?

The g

[... truncated for brevity ...]

---

## Issue #N/A: llama : add example for speculative sampling

**Link**: https://github.com/ggml-org/llama.cpp/issues/2030
**State**: closed
**Created**: 2023-06-28T05:20:52+00:00
**Closed**: 2023-09-03T12:29:06+00:00
**Comments**: 12
**Labels**: performance, generation quality, research 🔬

### Description

Speculative sampling is explained here: https://arxiv.org/abs/2302.01318

In more simple terms here:

- https://github.com/ggerganov/llama.cpp/issues/630#issuecomment-1518745593
- https://github.com/ggerganov/llama.cpp/issues/630#issuecomment-1556448281

For start, the "draft" model can be generated using the [train-text-from-scratch](https://github.com/ggerganov/llama.cpp/tree/master/examples/train-text-from-scratch) example using the same vocab as LLaMA. Later, we can try to utilize better models.

We also assume that batching multiple tokens with the "main" model is significantly faster compared to processing the tokens one-by-one. This may not yet be the case, but it will be when we close https://github.com/ggerganov/ggml/issues/293





---

## Issue #N/A: Research: bench of the llamacpp

**Link**: https://github.com/ggml-org/llama.cpp/issues/10405
**State**: closed
**Created**: 2024-11-19T12:50:36+00:00
**Closed**: 2025-01-03T01:07:17+00:00
**Comments**: 1
**Labels**: research 🔬, stale

### Description

### Research Stage

- [x] Background Research (Let's try to avoid reinventing the wheel)
- [ ] Hypothesis Formed (How do you think this will work and it's effect?)
- [ ] Strategy / Implementation Forming
- [ ] Analysis of results
- [ ] Debrief / Documentation (So people in the future can learn from us)

### Previous existing literature and research

theres two Questions:

Is the content of this batch input self-defined, similar to Some other Infer framework or is there a specific dataset for it? Or other operations?
The output time only provides the average and variance for each token. How is this time calculated? Is it the mean and variance over multiple runs? Also, what part of the execution is being timed? From which point to which point is the timing measured?

### Hypothesis

_No response_

### Implementation

_No response_

### Analysis

_No response_

### Relevant log output

_No response_

---

## Issue #N/A: [IDEA] Global token enhancement/depression

**Link**: https://github.com/ggml-org/llama.cpp/issues/1865
**State**: open
**Created**: 2023-06-15T02:24:07+00:00
**Comments**: 1
**Labels**: help wanted, research 🔬

### Description

This idea is inspired by Stable Diffusion prompts and anti-prompts. It could be useful to keep the text generation on topic even for small window sizes, for example. (e.g. if creating a poem about cheese and it wanders off on a tangent, still the word "cheese" will have high probability)

The idea is simple. In the output of some text you may want to increase the probabilities of some words while decreasing the probabilities (or set to zero) of other words, globally.

An example of words you may want to depress are swear words etc.
Example of words you may want to increase are words relevant to your topic or words in your style.

These global enhancements/depressions of the probabilities would stay constant throughout the text-generation even if the window-size is small.

There are two ways this could work

1. The user includes a list of words and anti-words.
2. A model could automatically be trained to create a global-enhancement matrix from the original prompt which stays

[... truncated for brevity ...]

---

## Issue #N/A: llama : try to avoid context swap

**Link**: https://github.com/ggml-org/llama.cpp/issues/2060
**State**: closed
**Created**: 2023-06-30T19:53:55+00:00
**Closed**: 2023-09-28T16:04:38+00:00
**Comments**: 2
**Labels**: performance, research 🔬

### Description

Currently, when the context becomes full, we pick part of the tokens and recompute the KV cache.

Instead, try to either:
- store non-RoPEd KV cache, "shift" it when the context is full and compute the RoPE over the entire cache for every new token taking into account the current positions
- store RoPEd KV cache (as we do now), "shift" it when the context is full and apply extra shift-RoPE on it (assuming RoPE is "additive")

---

## Issue #N/A: metal : compile-time kernel args and params

**Link**: https://github.com/ggml-org/llama.cpp/issues/4085
**State**: open
**Created**: 2023-11-15T11:09:39+00:00
**Comments**: 4
**Labels**: performance, research 🔬, roadmap

### Description

I was just thinking about this idea, so writing it down for future research.

We should be able to fairly easy generate model-specific Metal code that has hardcoded kernels for every single node in the computation graph. The idea is to make an initial pass of a certain graph where we record all kernel calls with their respective argument values and parameters and then generate a model-specific MSL source file with all these kernels instances - either copy-paste or via templates. I guess this is something similar to what people call JIT. Wondering what kind of speed-up we will be able to see with this strategy.

---

## Issue #N/A: Research: Are there any plans to support AIGC models such as flux1.dev?

**Link**: https://github.com/ggml-org/llama.cpp/issues/9110
**State**: closed
**Created**: 2024-08-21T04:38:20+00:00
**Closed**: 2024-10-18T01:07:16+00:00
**Comments**: 2
**Labels**: research 🔬, stale

### Description

### Research Stage

- [ ] Background Research (Let's try to avoid reinventing the wheel)
- [ ] Hypothesis Formed (How do you think this will work and it's effect?)
- [ ] Strategy / Implementation Forming
- [ ] Analysis of results
- [ ] Debrief / Documentation (So people in the future can learn from us)

### Previous existing literature and research

_No response_

### Hypothesis

_No response_

### Implementation

_No response_

### Analysis

_No response_

### Relevant log output

_No response_

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

## Issue #N/A: llama : support Mamba-2

**Link**: https://github.com/ggml-org/llama.cpp/issues/7727
**State**: closed
**Created**: 2024-06-04T05:57:48+00:00
**Closed**: 2025-07-02T17:10:26+00:00
**Comments**: 1
**Labels**: model, research 🔬, roadmap

### Description

Mamba-2 is a new version of the Mamba architecture:

- Blog: https://tridao.me/blog/2024/mamba2-part1-model/
- Paper: https://arxiv.org/abs/2405.21060

---

## Issue #N/A: Investigate alternative ggml_compute_forward_mul_mat_q_f32() implementation

**Link**: https://github.com/ggml-org/llama.cpp/issues/909
**State**: closed
**Created**: 2023-04-12T07:36:24+00:00
**Closed**: 2023-04-15T14:53:24+00:00
**Comments**: 7
**Labels**: help wanted, performance, research 🔬

### Description

This is the most computationally significant call in the entire transformer evaluation, so we have to be sure that it is running optimally.

It computes the matrix multiplication: `z = x * y`

- `x` is quantized
- `y` is F32
- `z` is F32

Currently, it runs in 2 modes, depending on the tensor shapes:

- (A) for bigger tensors, if BLAS is available, `x` is dequantized to F32 and we use `sgemm` to perform the matrix multiplication
- (B) for smaller tensors, or if BLAS is not available, `y` is quantized to 4-bits on-the-fly and we use integer-based dot products to perform the matrix multiplication

The former method is much more accurate than the latter. This can be clearly observed during perplexity computations.
However, during text generation (i.e. batch = 1), it is not feasible to use it - my experience is that there is significant overhead of calling BLAS for smaller tensor shapes, typical for single-token inference calls.

There are at least two alternative modes of 

[... truncated for brevity ...]

---

## Issue #N/A: Research: Im writing a paper on our medical finetuned llava-v1.6,

**Link**: https://github.com/ggml-org/llama.cpp/issues/7831
**State**: closed
**Created**: 2024-06-08T09:10:58+00:00
**Closed**: 2024-07-24T01:06:49+00:00
**Comments**: 3
**Labels**: research 🔬, stale

### Description

I am currently writing a paper on LLaVA-Med V1.6, which we have fine-tuned on medical images. The paper's first stage, detailing the fine-tuning process, is complete. I am now focusing on stage two: converting our fine-tuned model into a 4-bit GGUF format.

Could you please advise on what key points to include in this section? Additionally, could you suggest any relevant references or previous papers that discuss similar quantization processes?

Thank you for your assistance.

---

## Issue #N/A: llama : add multimodal support (LLaVA)

**Link**: https://github.com/ggml-org/llama.cpp/issues/3332
**State**: closed
**Created**: 2023-09-25T20:53:36+00:00
**Closed**: 2023-10-12T15:23:20+00:00
**Comments**: 9
**Labels**: research 🔬

### Description

Now that OpenAI is adding voice and image to ChatGPT and will probably be the new norm, wouldn't it be a good idea for llama.cpp to also please add this to the roadmap? if possible?

---

## Issue #N/A: Research: mmap eviction

**Link**: https://github.com/ggml-org/llama.cpp/issues/14154
**State**: open
**Created**: 2025-06-12T16:03:46+00:00
**Comments**: 0
**Labels**: research 🔬, stale

### Description

### Research Stage

- [x] Background Research (Let's try to avoid reinventing the wheel)
- [ ] Hypothesis Formed (How do you think this will work and it's effect?)
- [ ] Strategy / Implementation Forming
- [ ] Analysis of results
- [ ] Debrief / Documentation (So people in the future can learn from us)

### Previous existing literature and research

_No response_

### Hypothesis

I'm loading a large model into a large amount of GPU memory with some CPU offload. The GPU memory exceeds system memory.

GPU Memory: 196 GB
CPU Memory: 148 GB
Model Size: 220 GB

I've noticed that when the model size exceeds system memory, mmap seemingly has no effect on load times. Whereas when it's within system memory size, the load time is nearly immediate.

I suspect that since the model is being loaded deterministically/sequentially, the mapped file is also being deterministically evicted just prior to it being needed for the load onto GPU.

I suspect loading the large weights in reverse inference order

[... truncated for brevity ...]

---

## Issue #N/A: Add GPU support to ggml

**Link**: https://github.com/ggml-org/llama.cpp/issues/914
**State**: closed
**Created**: 2023-04-12T11:11:42+00:00
**Closed**: 2023-04-12T11:47:54+00:00
**Comments**: 1
**Labels**: enhancement, help wanted, hardware, research 🔬

### Description

## Intro

This issue is more suitable for the https://github.com/ggerganov/ggml repo, but adding it here for more visibility.

First, I don't see adding a GPU framework that is tightly integrated with `ggml` anytime soon because it usually comes with a lot of maintenance drawbacks, architecture changes and issues. However, there is an alternative approach that might be relatively easy to implement and I think would be a very cool way for new developers to join in and help.

## Description

`ggml` produces computation graphs which are basically directed acyclic graphs (DAGs) that can be easily exported, iterated, etc. A graph contains the information about all necessary tensor operations and buffers needed to evaluate the model. The idea is to first add basic `ggml` functionality for exporting the graphs in some trivial text format that can be parsed as a second step by a separate `ggml` tool. Having the exported graphs, one can process them and construct hardware-specific code 

[... truncated for brevity ...]

---

## Issue #N/A: ggml : add ANE backend

**Link**: https://github.com/ggml-org/llama.cpp/issues/10453
**State**: open
**Created**: 2024-11-22T08:20:22+00:00
**Comments**: 13
**Labels**: help wanted, research 🔬, roadmap

### Description

According to this https://github.com/ggerganov/llama.cpp/discussions/336#discussioncomment-11184134, there is a new CoreML API and an ANE backend might be possible to implement with latest Apple software/hardware.

---

## Issue #N/A: ggml : add DirectML backend

**Link**: https://github.com/ggml-org/llama.cpp/issues/7772
**State**: open
**Created**: 2024-06-05T14:21:34+00:00
**Comments**: 10
**Labels**: help wanted, research 🔬, roadmap

### Description

It seems like DirectML supports the upcoming NPU-enabled chips for Windows machines:
https://devblogs.microsoft.com/directx/introducing-neural-processor-unit-npu-support-in-directml-developer-preview/

I don't think there is any other way to tap into this hardware, so we should explore if it possible to add this library as a backend in `ggml` in order to run stuff on the NPUs. There has been some semi-related work in the past that combined `ggml` and Direct3D: https://github.com/Const-me/Whisper. Not sure if it is relevant at all, maybe just as an inspiration

---

## Issue #N/A: Research: How to integrate VITA 1.5 for multi-modal GGUF deployment?

**Link**: https://github.com/ggml-org/llama.cpp/issues/13520
**State**: closed
**Created**: 2025-05-14T02:58:50+00:00
**Closed**: 2025-06-28T01:07:47+00:00
**Comments**: 1
**Labels**: research 🔬, stale

### Description

### Research Stage

- [ ] Background Research (Let's try to avoid reinventing the wheel)
- [ ] Hypothesis Formed (How do you think this will work and it's effect?)
- [ ] Strategy / Implementation Forming
- [ ] Analysis of results
- [ ] Debrief / Documentation (So people in the future can learn from us)

### Previous existing literature and research

I'm trying to deploy a multi-modal model based on VITA-1.5, where:

The text backbone is the same as Qwen2.

The vision tower is InternViT-300M-448px from OpenGVLab.

Yesterday I noticed that convert_hf_to_gguf.py added a new class:

class InternVisionModel(VisionModel)

which is the same one used in vita's vision part
However:

There's no corresponding tensor name mapping in constants.py under MODEL_TENSORS.

There's no build function in llama_model.cpp (e.g., no build_internvit() ).

I’m not sure how to combine the vision and text parts into a single GGUF model so that llama.cpp can infer with both modalities.

My goal:
To deploy VITA-1.5

[... truncated for brevity ...]

---

## Issue #N/A: RPTQ state of the art quantization

**Link**: https://github.com/ggml-org/llama.cpp/issues/1295
**State**: closed
**Created**: 2023-05-03T03:23:53+00:00
**Closed**: 2024-04-09T01:09:40+00:00
**Comments**: 2
**Labels**: generation quality, research 🔬, Less than 4 bits, stale

### Description

Per yuan etc all, RPTQ quant is state of the art down to 3bit

It would be good to implement RPTQ for llama and other c++ downstream projects

https://github.com/hahnyuan/RPTQ4LLM/blob/master/quantize/quantizer.py

https://arxiv.org/abs/2304.01089

---

## Issue #N/A: llama : combined beam search + grammar sampling strategy

**Link**: https://github.com/ggml-org/llama.cpp/issues/2923
**State**: open
**Created**: 2023-08-31T06:29:29+00:00
**Comments**: 13
**Labels**: good first issue, generation quality, research 🔬, roadmap

### Description

This feature was proposed by @spion in https://github.com/ggerganov/llama.cpp/issues/2813#issuecomment-1694390583

> In some cases, its useful to do constrained evaluation of logits based on a union of possible text values, then pick the sum { logits } (i.e. product(probabilities)) that gives the most probable outcome overall.

> E.g. template (using MS guidance)

> {{#select 'armor'}}leather{{or}}chainmail{{or}}plate{{/select}}

> To definitely make the best choice, we'd need to calculate the probability of all 3 token sequences. Its easy if all the choices map to a single token, but with multiple tokens we'd need not just parallel generation but parallel logit evaluation of multiple possible paths.

> If we go greedy, we might get suboptimal results in cases multiple choices start with the same logit.

It should be possible to implement this by combining the existing beam search and grammar sampling features. See the discussion in the referenced comment for more info

---

## Issue #N/A: csm : implement Sesame-based conversation example

**Link**: https://github.com/ggml-org/llama.cpp/issues/12392
**State**: closed
**Created**: 2025-03-14T14:49:46+00:00
**Closed**: 2025-05-14T01:07:48+00:00
**Comments**: 23
**Labels**: model, research 🔬, stale, tts

### Description

With the first Sesame CSM model [openly available](https://github.com/SesameAILabs/csm), we should implement a local example similar to their [online research demo](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice#demo). It seems that the released CSM model uses [Kyutai's Mimi](https://arxiv.org/abs/2410.00037) audio codec which we have to implement in a similar way as we did with the [WavTokenizer](https://github.com/ggml-org/llama.cpp/pull/10784). Next we can modify the [talk-llama](https://github.com/ggerganov/whisper.cpp/tree/master/examples/talk-llama) example to support audio generation with the CSM. This way we will be able to plug any LLM for the text response generation and use Sesame for speech input/output.

---

## Issue #N/A: Study how LM Evaluation Harness works and try to implement it

**Link**: https://github.com/ggml-org/llama.cpp/issues/231
**State**: open
**Created**: 2023-03-17T08:32:33+00:00
**Comments**: 9
**Labels**: enhancement, help wanted, high priority, generation quality, research 🔬

### Description

Update 10 Apr 2024: https://github.com/ggerganov/llama.cpp/issues/231#issuecomment-2047759312

---

It would be great to start doing this kind of quantitative analysis of `ggml`-based inference:

https://bellard.org/ts_server/

It looks like Fabrice evaluates the models using something called LM Evaluation Harness:

https://github.com/EleutherAI/lm-evaluation-harness

I have no idea what this is yet, but would be nice to study it and try to integrate it here and in other `ggml`-based projects.
This will be very important step needed to estimate the quality of the generated output and see if we are on the right track.

---

## Issue #N/A: 2-bit integer quantization 

**Link**: https://github.com/ggml-org/llama.cpp/issues/456
**State**: closed
**Created**: 2023-03-24T06:55:44+00:00
**Closed**: 2023-06-24T19:17:24+00:00
**Comments**: 16
**Labels**: enhancement, research 🔬

### Description

Add `Q2_0` and `Q2_1` quantization support to `ggml`:

- Follow the existing `Q4_0` and `Q4_1` implementations
- Implement [reference scalar quantization and dequantization routines](https://github.com/ggerganov/llama.cpp/blob/3cd8dde0d1357b7f11bdd25c45d5bf5e97e284a0/ggml.c#L407-L449)
- I suspect we might have to use `QK == 16` in this case to compensate for further accuracy losses
- Add SIMD support for a specific architecture - investigate best strategy to perform the `ggml_vec_dot_q2()` computation
- No need to implement `ggml_vec_mad_q2()` - these will be deprecated soon
- Compute perplexity scores

The expected model sizes for 7B and `QK == 16` are:

- `Q2_0` - 3.2 GB

For `QK == 32` we have:

- `Q2_0` - 2.4 GB
- `Q2_1` - 3.2 GB

Before you send me papers that show 2-bit quantization does not work - no need. I want to have this supported anyway. I have something in mind. The efforts needed to add this support are so small that there is no reason not to do it.

---

## Issue #N/A: Research: Benchmarking DeepSeek-R1 IQ1_S 1.58bit

**Link**: https://github.com/ggml-org/llama.cpp/issues/11474
**State**: closed
**Created**: 2025-01-28T23:39:28+00:00
**Closed**: 2025-04-25T01:07:52+00:00
**Comments**: 45
**Labels**: research 🔬, stale

### Description

### Research Stage

- [ ] Background Research (Let's try to avoid reinventing the wheel)
- [ ] Hypothesis Formed (How do you think this will work and it's effect?)
- [ ] Strategy / Implementation Forming
- [x] Analysis of results
- [ ] Debrief / Documentation (So people in the future can learn from us)

### Previous existing literature and research

# Command
```
 ./llama.cpp/build/bin/llama-cli \
    --model DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf \
    --cache-type-k q4_0 \
    --threads 12 -no-cnv --n-gpu-layers 61 --prio 2 \
    --temp 0.6 \
    --ctx-size 8192 \
    --seed 3407 \
    --prompt "<｜User｜>What is the capital of Italy?<｜Assistant｜>"
```

# Model
[DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_S
](https://huggingface.co/unsloth/DeepSeek-R1-GGUF) 1.58Bit, 131GB

# Hardware
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 

[... truncated for brevity ...]

---

## Issue #N/A: Combine large LLM with small LLM for faster inference

**Link**: https://github.com/ggml-org/llama.cpp/issues/630
**State**: closed
**Created**: 2023-03-30T17:54:01+00:00
**Closed**: 2024-04-12T01:07:17+00:00
**Comments**: 44
**Labels**: question, research 🔬, stale

### Description

So I was thinking about the following idea.
It is probably completely bogus, but I would definitely investigate it when and if I had the time to, so maybe someone else would be interested as well.

---

Large LLM takes a lot of time to perform token inference. Lets say it takes 500ms per token.

A small LLM (or some other approach) can infer a token very fast. Lets say < 5ms.

Lets assume that the small LLM is correct 80-90% of the time.

The idea is the following:

- Before I run the large LLM inference for the next token, I infer it using the small LLM
- I now want to somehow partially evaluate the large LLM (let's say the first 10% of the layers) and get an approximate estimate for the next token
- If this estimate indicates a high probability for that token (i.e. above some threshold) - we stop and directly say that this is the new token. At this point we would have consumed (5ms for the small LLM + ~50ms for the large LLM)
- Otherwise, we proceed to evaluate the re

[... truncated for brevity ...]

---

## Issue #N/A: llama : add support for Classifier-Free Guidance (CFG) sampling to stay on topic better

**Link**: https://github.com/ggml-org/llama.cpp/issues/2083
**State**: closed
**Created**: 2023-07-03T08:38:55+00:00
**Closed**: 2023-07-11T16:18:45+00:00
**Comments**: 19
**Labels**: enhancement, good first issue, generation quality, research 🔬

### Description

@ggerganov [retweeted](https://twitter.com/Vermeille_/status/1675664118500454400) the "Stay on topic with Classifier-Free Guidance" paper that came out showing that "Classifier-Free Guidance (CFG)"... "can be used broadly as an inference-time technique in pure language modeling. " ... "brings improvements equivalent to a model with twice the parameter-count" (with no retraining needed). -  https://arxiv.org/abs/2306.17806

I saw that the Transformers library has one of the paper's author [working on an implementation](https://github.com/huggingface/transformers/issues/24536).

I didn't see an issue for it yet here so I figured pointing to it is the least I could do for this awesome library!

---

## Issue #N/A: Investigate storing results from ggml operations in F16 format

**Link**: https://github.com/ggml-org/llama.cpp/issues/959
**State**: closed
**Created**: 2023-04-14T07:35:34+00:00
**Closed**: 2023-04-22T08:48:31+00:00
**Comments**: 1
**Labels**: help wanted, performance, high priority, research 🔬

### Description

Currently, all `ggml` operations return the results in F32 format.

The goal of this task is to see if there is an elegant way to add support for keeping the results in F16 format.
This will ideally be passed as a parameter to the `ggml_context` and will also involve adding support for F16 operands in most of the existing operators. Ideally, we want to achieve this somehow without duplicating the entire code base.

Note that internal floating-point accumulators in the different operations can and should remain in F32 format.
It is just when we store the results into the `dst` tensor, we will cast them to F16.

Going to F16 intermediate results would reduce significantly the memory pressure and could lead to significant speed improvements. Hopefully, the loss in quality would be marginal. But in any case, there will always be the option of switching back to full F32 precision.

I am looking for suggestions and initial prototypes of how we can achieve this in an elegant way.


[... truncated for brevity ...]

---

