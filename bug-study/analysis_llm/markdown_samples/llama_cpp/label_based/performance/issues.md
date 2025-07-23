# performance - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 3
- Closed Issues: 27

### Label Distribution

- performance: 30 issues
- stale: 9 issues
- enhancement: 7 issues
- research 🔬: 5 issues
- good first issue: 4 issues
- macos: 3 issues
- bug: 2 issues
- refactoring: 2 issues
- roadmap: 2 issues
- AMD GPU: 2 issues

---

## Issue #N/A: llama : support sliding window attention

**Link**: https://github.com/ggml-org/llama.cpp/issues/3377
**State**: closed
**Created**: 2023-09-28T12:12:40+00:00
**Closed**: 2024-11-01T01:21:36+00:00
**Comments**: 21
**Labels**: performance, stale

### Description

For more info, see: https://github.com/mistralai/mistral-src and references there in.

Also: https://arxiv.org/pdf/2310.06825v1.pdf

With #3228 it should be relatively easy to support this.

---

## Issue #N/A: Windows page fault disk i/o slow on first load

**Link**: https://github.com/ggml-org/llama.cpp/issues/705
**State**: closed
**Created**: 2023-04-02T10:04:24+00:00
**Closed**: 2024-04-11T01:07:14+00:00
**Comments**: 37
**Labels**: performance, windows, stale

### Description

Hello,

As of https://github.com/ggerganov/llama.cpp/pull/613 I have experienced significant regression in model loading speed (I'm on windows, compiled msvc llama.cpp, llama.cpp is located on HDD to prevent SSD wear in my case)

It takes roughly 15 minutes for model to load first time after each computer restart/hibernation, during this time my HDD usage is at 100% and my non-llama.cpp read/write operations are slowed down on my pc
![hdd](https://user-images.githubusercontent.com/76458234/229345728-b597023b-f7e3-4a8b-b550-3159863ba03d.png)

Before that, previous commits took 60 - 180 seconds at worst to load model first time, and after first loading occured, model loaded within 5 - 10 seconds on each program restart until pc reboot/hibernation

Before Commit:
![timings2](https://user-images.githubusercontent.com/76458234/229347345-2053d645-0f26-42ef-9f8e-5fc69ad04e1c.png)

After:
![timings1](https://user-images.githubusercontent.com/76458234/229345966-ee606c92-e7cb-42f6-8

[... truncated for brevity ...]

---

## Issue #N/A: [fixed]The last code build with memory fix running result is not good in my pc.

**Link**: https://github.com/ggml-org/llama.cpp/issues/462
**State**: closed
**Created**: 2023-03-24T14:22:06+00:00
**Closed**: 2023-03-27T00:13:38+00:00
**Comments**: 10
**Labels**: bug, performance

### Description

Be obviously slower with Q_1 30b model. And the memory usage become garbage...
(Linux 5.19 x64 Ubuntu base)

---

## Issue #N/A: llama : speed-up grammar sampling

**Link**: https://github.com/ggml-org/llama.cpp/issues/4218
**State**: open
**Created**: 2023-11-25T17:04:06+00:00
**Comments**: 40
**Labels**: performance, refactoring, roadmap

### Description

There have been a few reports where the grammar sampling can significantly degrade the performance.
It would be nice to profile and optimize the implementation - there should be room for improvements.

Already on-going efforts:

- #4210 
- #4213

Probably worth looking in multi-threading the implementation as well.

---

## Issue #N/A: Performance investigation using AMD BLIS instead of OpenBLAS on 16 core AMD Zen1

**Link**: https://github.com/ggml-org/llama.cpp/issues/637
**State**: closed
**Created**: 2023-03-30T22:14:53+00:00
**Closed**: 2023-04-13T08:09:16+00:00
**Comments**: 30
**Labels**: enhancement, performance

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior
Compiling against AMD optimized BLS implementation of BLAS allows me to run perplexity tests

# Current Behavior
Compiling against AMD optimized BLS implementation of BLAS causes perplexity command to process 0 chunks

* Physical (or virtual) hardware you are using

[... truncated for brevity ...]

---

## Issue #N/A: Add some models in ggml-models HF repo

**Link**: https://github.com/ggml-org/llama.cpp/issues/6292
**State**: closed
**Created**: 2024-03-25T06:36:44+00:00
**Closed**: 2024-05-18T01:58:22+00:00
**Comments**: 5
**Labels**: enhancement, performance, model, testing, stale

### Description

### Motivation

In the context of:

- #6233

Need to add some models in the [GGML HF Repo](https://huggingface.co/ggml-org/models/tree/main):

- mixtral8x7B Q4 Q8 F16 in split format
- bert-bge-large F16
- llama7B 13B split Q4 F16


---

## Issue #N/A: [User] AMD GPU slower than CPU

**Link**: https://github.com/ggml-org/llama.cpp/issues/3422
**State**: closed
**Created**: 2023-10-01T05:57:21+00:00
**Closed**: 2024-05-12T01:35:23+00:00
**Comments**: 40
**Labels**: performance, AMD GPU, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior
GPU inference should be faster than CPU.


# Current Behavior

I have 13900K CPU & 7900XTX 24G hardware. I built llama.cpp using the [hipBLAS](https://github.com/ggerganov/llama.cpp#hipblas) and it builds. However, I noticed that when I offload all layers to GPU, i

[... truncated for brevity ...]

---

## Issue #N/A: Add avx-512 support?

**Link**: https://github.com/ggml-org/llama.cpp/issues/160
**State**: closed
**Created**: 2023-03-15T12:10:17+00:00
**Closed**: 2023-03-28T09:54:15+00:00
**Comments**: 6
**Labels**: enhancement, performance, hardware

### Description

No clue but I think it may work faster

---

## Issue #N/A: llama : refactor llama_vocab

**Link**: https://github.com/ggml-org/llama.cpp/issues/9369
**State**: closed
**Created**: 2024-09-08T13:00:28+00:00
**Closed**: 2024-09-30T18:02:31+00:00
**Comments**: 3
**Labels**: good first issue, performance, refactoring

### Description

As of today we support 5 tokenizer implementations:

```c
        LLAMA_VOCAB_TYPE_SPM  = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
        LLAMA_VOCAB_TYPE_BPE  = 2, // GPT-2 tokenizer based on byte-level BPE
        LLAMA_VOCAB_TYPE_WPM  = 3, // BERT tokenizer based on WordPiece
        LLAMA_VOCAB_TYPE_UGM  = 4, // T5 tokenizer based on Unigram
        LLAMA_VOCAB_TYPE_RWKV = 5, // RWKV tokenizer based on greedy tokenization
```

The function `llama_tokenize_internal` in `llama-vocab.cpp` currently constructs a tokenizer instance on every call which for some of the tokenizers incurs significant overhead. This should be avoided by pre-constructing the tokenizer object upon `llama-vocab` creation and abstracting the objects (e.g. `llm_tokenizer_spm`, `llm_tokenizer_bpe`, etc.) with a common interface.

However, we want `llama_tokenize_internal` to remain thread-safe as it currently is (I think). Therefore, the tokenizer objects would likely need to b

[... truncated for brevity ...]

---

## Issue #N/A: ClBlast - no gpu load, no perfomans difference.

**Link**: https://github.com/ggml-org/llama.cpp/issues/1217
**State**: closed
**Created**: 2023-04-28T16:05:41+00:00
**Closed**: 2023-05-05T00:51:53+00:00
**Comments**: 28
**Labels**: performance, hardware, build

### Description

How i build:

1.  I use [w64devkit](https://github.com/skeeto/w64devkit/releases)
2. I download [CLBlast](https://github.com/CNugteren/CLBlast) and [OpenCL-SDK](https://github.com/KhronosGroup/OpenCL-SDK)
3. Put folders lib and include from [CLBlast](https://github.com/CNugteren/CLBlast) and [OpenCL-SDK](https://github.com/KhronosGroup/OpenCL-SDK) to w64devkit_1.18.0\x86_64-w64-mingw32
4. Using w64devkit.exe cd to llama.cpp
5. make LLAMA_CLBLAST=1
6. Put clblast.dll near main.exe

When load i got this: 

> Initializing CLBlast (First Run)...
> Attempting to use: Platform=0, Device=0 (If invalid, program will crash)
> Using Platform: AMD Accelerated Parallel Processing Device: gfx90c
> llama_init_from_file: kv self size  = 1600.00 MB
> 
> system_info: n_threads = 7 / 16 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | VSX = 0 |
> main: interactive mod

[... truncated for brevity ...]

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

## Issue #N/A: CPU performance bottleneck(?) when using macOS Accelerate

**Link**: https://github.com/ggml-org/llama.cpp/issues/5417
**State**: closed
**Created**: 2024-02-08T16:53:12+00:00
**Closed**: 2024-02-11T19:12:45+00:00
**Comments**: 5
**Labels**: enhancement, performance, macos, threading

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

I've been doing some performance testing of llama.cpp in macOS (On M2 Ultra 24-Core) and was comparing the CPU performance of inference with various options, and ran into a very large performance drop - Mixtral model inference on 16 cores (16 because it's only the p

[... truncated for brevity ...]

---

## Issue #N/A: Support CoreML like whisper.cpp?

**Link**: https://github.com/ggml-org/llama.cpp/issues/1714
**State**: open
**Created**: 2023-06-06T09:23:08+00:00
**Comments**: 11
**Labels**: help wanted, performance, macos

### Description

I have tried whisper.cpp on my iPhone and it runs very fast , so I wonder if it is possible that llama.cpp could support it. 
thank you .

---

## Issue #N/A: server: bench: continuous performance testing

**Link**: https://github.com/ggml-org/llama.cpp/issues/6233
**State**: closed
**Created**: 2024-03-22T11:36:09+00:00
**Closed**: 2024-07-03T01:06:46+00:00
**Comments**: 19
**Labels**: enhancement, performance, server/webui, need feedback, stale

### Description

#### Motivation

**llama.cpp** is under active development, new papers on LLM are implemented quickly (for the good) and backend device
optimizations are continuously added.

All these factors have an impact on the server performances, especially the following metrics:

1. **latency**: pp (prompt processing) + tg (tokens generation) per request
2. **server latency**: total pp+tg per second across all requests with continuous batching
3. **concurrency**: how many concurrent request/users the server can handle in parallel
4. **VRAM** usage
5. **RAM** usage
6. **GPU** usage
7. **CPU** usage

It is important to monitor and control the impact of the codebase evolution on these metrics,
example [from](https://towardsdatascience.com/increase-llama-2s-latency-and-throughput-performance-by-up-to-4x-23034d781b8c):

<p align="center">
    <img width="60%" height="60%" src="https://github.com/ggerganov/llama.cpp/assets/5741141/2f518477-941d-41e1-9427-873ca0cb9846" alt="prompt_to

[... truncated for brevity ...]

---

## Issue #N/A: Multi-thread the Q8_0 quantization in ggml_compute_forward_mul_mat_q_f32()

**Link**: https://github.com/ggml-org/llama.cpp/issues/1081
**State**: closed
**Created**: 2023-04-20T15:24:39+00:00
**Closed**: 2023-04-23T10:35:28+00:00
**Comments**: 1
**Labels**: enhancement, good first issue, performance

### Description

This part takes about 10% of the total inference time for 7B and it is currently single-threaded:

https://github.com/ggerganov/llama.cpp/blob/6a9661ea5ad72166b700ae5e87976e4452499dda/ggml.c#L7877-L7884

Try to multi-thread this by splitting the work across rows.
Since the `GGML_TASK_INIT` currently runs only 1 thread, either:
- update `ggml` to support multi-threaded `GGML_TASK_INIT`
- move the quantization in `GGML_TASK_COMPUTE` (might be difficult since no barrier mechanism)

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

## Issue #N/A: perf: parallelize quantization

**Link**: https://github.com/ggml-org/llama.cpp/issues/906
**State**: closed
**Created**: 2023-04-12T03:38:23+00:00
**Closed**: 2023-04-22T17:45:20+00:00
**Comments**: 3
**Labels**: performance

### Description

https://github.com/ggerganov/llama.cpp/blob/8b679987cdce292ff36bd741f6715e4927e26f9b/llama.cpp#L1558

Is currently single threaded. Quantization is quite slow (vicuna 7B: 65156.31 ms, vicuna 13B: 129902.48 ms).

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

## Issue #N/A: ~2x perf improvement on Apple Silicon by changing state_shared.has_work access from atomic to mutex/conditional

**Link**: https://github.com/ggml-org/llama.cpp/issues/633
**State**: closed
**Created**: 2023-03-30T19:18:14+00:00
**Closed**: 2024-04-12T01:07:15+00:00
**Comments**: 5
**Labels**: enhancement, performance, stale

### Description

### Discussed in https://github.com/ggerganov/llama.cpp/discussions/616

<div type='discussions-op-text'>

<sup>Originally posted by **izard** March 30, 2023</sup>
I profiled on a latest Mac Book Pro machine and found that significantly more time is spent in atomic checks for `state_shared.has_work` in while loops than doing actual work in matrix multiply.
So I changed busy waits like: 
```
pthread_mutex_lock(&state->shared->mutex);
   while (state->shared->has_work) {
     pthread_cond_wait(&state->shared->cond, &state->shared->mutex);
// unlock
```

and setting `has_work` to 
```
pthread_mutex_lock(&state_shared.mutex);
state_shared.has_work = true;
pthread_cond_broadcast(&state_shared.cond);
pthread_mutex_unlock(&state_shared.mutex);

```
Got a nice 2x speedup in time/token.

I can't post a patch/pull request because everything I do in spare time still belongs to my employer, but the change is trivial as described above. Probably won't provide much benefit (i

[... truncated for brevity ...]

---

## Issue #N/A: Try to use quantized `ggml_mul_mat` in attention layer

**Link**: https://github.com/ggml-org/llama.cpp/issues/1098
**State**: closed
**Created**: 2023-04-21T07:38:58+00:00
**Closed**: 2023-04-22T08:37:55+00:00
**Comments**: 1
**Labels**: good first issue, performance

### Description

The following 2 matrix multiplication calls sill remain in FP16 precission:

- https://github.com/ggerganov/llama.cpp/blob/d40fded93e1a533e969768e1e335c15c61c296ce/llama.cpp#L1135-L1137
- https://github.com/ggerganov/llama.cpp/blob/d40fded93e1a533e969768e1e335c15c61c296ce/llama.cpp#L1158-L1160

Was wondering, if we quantize those on-the-fly would there be any benefit.
The quantization can be done with an extra `ggml_cpy()` call, before the `ggml_mul_mat()` call.

See if this speeds up the computation and how it affects perplexity

---

## Issue #N/A: Not an issue but what depends on the number of threads?

**Link**: https://github.com/ggml-org/llama.cpp/issues/163
**State**: closed
**Created**: 2023-03-15T16:03:26+00:00
**Closed**: 2023-03-15T20:54:16+00:00
**Comments**: 3
**Labels**: performance

### Description

I've been testing your code from 1 to 8 threads and the output is always different. The speed is not depend on the number of threads. On the contrary, 4 threads may perform much better than 1, whereas 8 threads supposedly provides a better result. However, the same prompt may give the same excellent output with triple speed with 4 threads compared to 8. But still, when I use 8 threads (my maximum on M1) I use all my CPU resources, but it doesn't affect speed at all (seemingly works slower) and not giving quality effect (apparently). Am I wrong? Can you correct me if I'm mistaken? May be there is some best speed/quality option and I just that stupid that was unable to figure out how to use this option?

---

## Issue #N/A: CTX Processing regression for Pascal - Commit 2b4ea35

**Link**: https://github.com/ggml-org/llama.cpp/issues/3869
**State**: closed
**Created**: 2023-10-31T11:01:49+00:00
**Closed**: 2023-11-02T06:35:12+00:00
**Comments**: 18
**Labels**: performance

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

There is a regression on Context processing introduced in commit https://github.com/ggerganov/llama.cpp/commit/2b4ea35e56792064598e922e46d081e02bc96b94 

This is specifically for Pascal (6.1) with 1/64th fp16 performance.  Problem is worse with longer CTX, getting u

[... truncated for brevity ...]

---

## Issue #N/A: `llama_decode` is significantly slower if `n_tokens > 1` 

**Link**: https://github.com/ggml-org/llama.cpp/issues/4624
**State**: closed
**Created**: 2023-12-24T23:05:48+00:00
**Closed**: 2024-04-02T01:10:00+00:00
**Comments**: 7
**Labels**: performance, macos, bug-unconfirmed, stale

### Description

Issue
---
It is expected that `llama_decode` should take more time if more tokens are present in the batch, but on my system (Apple M1 Max 32GB) with `mistral-7b-instruct-v0.2.Q4_0.gguf` model, the increase in time taken is quite significant. I plotted some avg latencies on my system with different `n_tokens` using a modified version of `speculative` and putting timing around `llama_decode(ctx_tgt, batch_tgt);`:

![image](https://github.com/ggerganov/llama.cpp/assets/1957903/d9683434-6278-41b2-9018-d60acbe4ec2a)

There is more 5x jump in latency of `llama_decode` when `n_tokens` goes from 1 to 2 (which I feel is too high), but a very gradual increase after that. This means that techniques like `speculative` and `lookup` decoding **cannot give speed benefits** for small draft sizes ( `n_draft < 5`) even if drafts are 100% correct, since **autoregressively decoding 5 tokens 1 at a time is just as fast as decoding 5 tokens at once**, so the advantage of speculation is lost.

I'm n

[... truncated for brevity ...]

---

## Issue #N/A: llama : improve batched decoding performance

**Link**: https://github.com/ggml-org/llama.cpp/issues/3479
**State**: closed
**Created**: 2023-10-04T20:20:55+00:00
**Closed**: 2023-10-24T13:48:38+00:00
**Comments**: 12
**Labels**: performance, Nvidia GPU

### Description

Based on info from the following post, [vLLM](https://github.com/vllm-project/vllm) can achieve the following speeds for parallel decoding on A100 GPU:

https://docs.mystic.ai/docs/mistral-ai-7b-vllm-fast-inference-guide

Batch size | Tokens/s
-- | --
1 | 46
10 | 400
60 | 1.8k

(thanks to @wsxiaoys for bringing my attention to this)

Even though `llama.cpp`'s single batch inference is faster ([~72 t/s](https://github.com/ggerganov/llama.cpp/discussions/3359)) we currently don't seem to scale well with batch size. At batch size 60 for example, the performance is roughly x5 slower than what is reported in the post above.

We should understand where is the bottleneck and try to optimize the performance.

```bash
# batch size 1
./parallel -m ~/f16.gguf -t 1 -ngl 100 -c 8192 -b 512 -s 1 -np 1 -ns 128 -n 100 -cb

# batch size 10
./parallel -m ~/f16.gguf -t 1 -ngl 100 -c 8192 -b 512 -s 1 -np 10 -ns 128 -n 100 -cb

# batch size 60
./parallel -m ~/f16.gguf -t 1 -ngl 100 

[... truncated for brevity ...]

---

## Issue #N/A: Use different bit arrangement for quants (nibbles)

**Link**: https://github.com/ggml-org/llama.cpp/issues/1241
**State**: closed
**Created**: 2023-04-29T20:01:06+00:00
**Closed**: 2023-05-11T21:23:10+00:00
**Comments**: 3
**Labels**: performance

### Description

In the existing `llama.cpp` implementation, quantization bits of consecutive model weights are packed together one after the other. E.g., for 4-bit quantization, the 8 bits of two consecutive weights are stored into a `uint8_t`. The disadvantage of this approach is that when the data is to be used in dot products or is being de-quantized for matrix multiplications done via BLAS, and the operations are performed using SIMD instructions, one needs to shuffle the de-quantized bytes to get them into the correct order. These shuffle operations can be avoided by arranging the bits differently. For instance, for 4-bit quantization in blocks of 32 weights (`Q4_0`), one can store the quants of the first 16 weights into the low 4 bits of the 16 `uint8_t`'s, and the quants of the second 16 weights in the block of 32 into the high 4-bits. The same or similar strategy can also be applied for other block sizes or when using 2 bits per weight. 

The performance gain is not earth-shattering: in a sy

[... truncated for brevity ...]

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

## Issue #N/A: lookahead-prompt : add example

**Link**: https://github.com/ggml-org/llama.cpp/issues/4226
**State**: closed
**Created**: 2023-11-26T18:39:11+00:00
**Closed**: 2023-12-30T21:20:14+00:00
**Comments**: 5
**Labels**: good first issue, performance

### Description

Add an example implementing the "Prompt Lookup Decoding" technique:

https://github.com/apoorvumang/prompt-lookup-decoding

This should be a great exercise for people looking to become familiar with `llama.cpp`'s KV cache management and batched decoding API. Looking for contributions.

The following examples can be used as starting points:

- `speculative`
- `lookahead`
- `batched`

---

## Issue #N/A: Misc. bug: HIP backend performs poorly on AMD Ryzen AI MAX 395 (Strix Halo gfx1151)

**Link**: https://github.com/ggml-org/llama.cpp/issues/13565
**State**: closed
**Created**: 2025-05-15T14:12:58+00:00
**Closed**: 2025-05-18T16:49:49+00:00
**Comments**: 9
**Labels**: performance, AMD GPU

### Description

### Name and Version

```
❯ build/bin/llama-cli --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32
version: 5392 (c753d7be)
built with cc (GCC) 15.0.1 20250418 (Red Hat 15.0.1-0) for x86_64-redhat-linux
```

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-bench

### Command line

```shell
llama.cpp-cpu/build/bin/llama-bench -m ~/models/llama-2-7b.Q4_0.gguf
llama.cpp-vulkan/build/bin/llama-bench -m ~/models/llama-2-7b.Q4_0.gguf
llama.cpp-hip/build/bin/llama-bench -m ~/models/llama-2-7b.Q4_0.gguf
```

### Problem description & steps to reproduce

Recently I've been testing a Strix Halo (gfx1151) system and was a bit surprised by how poorly the HIP backend ran. All tests were run with `llama-bench` built on HEAD (b5392) with the standard [TheBloke/Llama-2-7B-GGUF](https://huggingface.co/The

[... truncated for brevity ...]

---

## Issue #N/A: |BUG] ggml spawns threads even BLAS is used

**Link**: https://github.com/ggml-org/llama.cpp/issues/578
**State**: closed
**Created**: 2023-03-28T15:02:01+00:00
**Closed**: 2024-04-12T01:07:25+00:00
**Comments**: 3
**Labels**: bug, performance, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior
ggml should not spawn threads for the initial prompt ingestion when using BLAS.

# Current Behavior
ggml does spawn threads even when using BLAS.

# Environment and Context 
Reproducible using latest OpenBLAS with PR https://github.com/xianyi/OpenBLAS/pull/3970 (f

[... truncated for brevity ...]

---

