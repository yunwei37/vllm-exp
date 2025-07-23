# never_closed_180days - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 30
- Closed Issues: 0

### Label Distribution

- bug: 10 issues
- help wanted: 10 issues
- enhancement: 10 issues
- good first issue: 10 issues
- roadmap: 8 issues
- research 🔬: 7 issues
- medium severity: 3 issues
- generation quality: 3 issues
- high priority: 2 issues
- high severity: 2 issues

---

## Issue #N/A: Bug: cannot create std::vector larger than max_size()

**Link**: https://github.com/ggml-org/llama.cpp/issues/9391
**State**: open
**Created**: 2024-09-09T15:52:21+00:00
**Comments**: 10
**Labels**: bug, medium severity

### Description

### What happened?

My usual build recipe and run scripts do not work after b3680. Something changed in b3681, but I don't know what.
I see this same failure across models and cli flags, so it seems to be deeper than a single feature choice, so I have excluded the launch script.

This is the actual error:
```
...
terminate called after throwing an instance of 'std::length_error'
  what():  cannot create std::vector larger than max_size()
<launch script name> Aborted                 (core dumped)
```

Here is what the binary reports at runtime:
```
system_info: n_threads = 24 (n_threads_batch = 24) / 48 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
main: interactive mode on.
```

Here is how I configure the build:
```
cmake -DGGML_AVX=ON -DGGML_AV

[... truncated for brevity ...]

---

## Issue #N/A: Question: How to generate an MPS gputrace

**Link**: https://github.com/ggml-org/llama.cpp/issues/6506
**State**: open
**Created**: 2024-04-05T14:08:32+00:00
**Comments**: 10
**Labels**: help wanted, high priority

### Description

We're doing some work over at https://github.com/huggingface/candle to improve our Metal backend, I've been collecting various gputraces for the different frameworks and was wondering if there was a documented/known way to generate one for llama.cpp during model inference.

Specifically talking about this type of debugger output: https://developer.apple.com/documentation/xcode/metal-debugger

---

## Issue #N/A: common: download from URL, improve parallel download progress status

**Link**: https://github.com/ggml-org/llama.cpp/issues/6537
**State**: open
**Created**: 2024-04-08T07:37:01+00:00
**Comments**: 6
**Labels**: enhancement, help wanted, good first issue, split

### Description

### Context

When downloading a sharded model, files are downloaded in parallel, it was added in:
- #6192

The progressions of each download conflict:
![image](https://github.com/ggerganov/llama.cpp/assets/5741141/d4937fc7-edf4-4920-ba63-dadf1c77b2d0)

Need to properly implement [CURLOPT_NOPROGRESS](https://curl.se/libcurl/c/CURLOPT_NOPROGRESS.html) for parallel download.

Example in #6515:

```shell
main --hf-repo ggml-org/models \
  --hf-file grok-1/grok-1-q4_0-00001-of-00009.gguf \
  --model   models/grok-1-q4_0-00001-of-00009.gguf \
  -ngl 64
   --prompt "I believe the meaning of life is"
```

---

## Issue #N/A: Llava functions compiled as extern "C" throw exceptions

**Link**: https://github.com/ggml-org/llama.cpp/issues/7073
**State**: open
**Created**: 2024-05-04T13:34:07+00:00
**Comments**: 0
**Labels**: bug, good first issue

### Description

Basically this:
llama.cpp\examples\llava\clip.cpp(1277,13): warning : 'clip_model_load' has a non-throwing exception specification but can still throw [-Wexceptions]
llama.cpp\examples\llava\clip.cpp(2075,5): warning : 'clip_n_mmproj_embd' has a non-throwing exception specification but can still throw [-Wexceptions]

As these are library exported functions and wrapped in extern "C", they should not allow exceptions to cross the boundary. C language has no idea what to do with them.

Compiled with clang-cl in windows.

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

## Issue #N/A: Bug: server crashes on startup is ckt ctv specified.

**Link**: https://github.com/ggml-org/llama.cpp/issues/7639
**State**: open
**Created**: 2024-05-30T13:03:41+00:00
**Comments**: 1
**Labels**: bug, high severity

### Description

### What happened?

if I specify -ctk q6_k (and/or ctv) the server exits with error: `libc++abi: terminating due to uncaught exception of type std::runtime_error: Invalid cache type: q6_k`
![image](https://github.com/ggerganov/llama.cpp/assets/170285982/76210fa0-11ac-4a69-8b00-f6dd8cfd5529)


### Name and Version

all versions

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
Note: when the server exists it causes a crash in windows.
But if I do server -h it does not crash... it shows the help and exits.
Any other error encountered (like the one above) caused a "crash"
```


---

## Issue #N/A: Misc. bug: server provides strutured output for response_format: json_object, but not for response_format: json_schema

**Link**: https://github.com/ggml-org/llama.cpp/issues/10732
**State**: open
**Created**: 2024-12-09T04:39:47+00:00
**Comments**: 5
**Labels**: enhancement, good first issue

### Description

### Name and Version

on latest commit ce8784bdb153ff7794dde5a50b0ebfa51baa6171

but have been noticing it for several days now

### Operating systems

_No response_

### Which llama.cpp modules do you know to be affected?

_No response_

### Problem description & steps to reproduce

I have been trying to follow the steps for structured json output using json_schema in server [here](https://github.com/ggerganov/llama.cpp/tree/ce8784bdb153ff7794dde5a50b0ebfa51baa6171/examples/server#post-v1chatcompletions-openai-compatible-chat-completions-api)

however, I was not able to get any combination of `json_schema` to work. I *was* able to get `json_object` to work, doing effectively the same thing, but since this differs from the OpenAI API (which I suppose server is striving for) I suppose it's a bug. The official server docs also mention json_schema is supported (see the above link)

Does not work, does not apply any structured json schema at all, and does not indicate a

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: Error running multiple contexts from multiple threads at the same time with Vulkan

**Link**: https://github.com/ggml-org/llama.cpp/issues/11371
**State**: open
**Created**: 2025-01-23T13:32:49+00:00
**Comments**: 3
**Labels**: bug

### Description

### Name and Version

This appears to be the same bug as noted in this issue:
https://github.com/ggerganov/llama.cpp/issues/7575

We are trying to do inference from multiple threads with some contexts having LORAs loaded and others not (so batched inference isn't going to work).  If I may ask, has there been any progress on this issue?  We are currently using a build from mid September 2024.

### Operating systems

Windows

### GGML backends

Vulkan

### Hardware

2x Nvidia RTX 3090s.

### Models

Meta Llama 3.2 3B 8 bit quant.

### Problem description & steps to reproduce

When we run llama_decode with different contexts in different threads, we get a crash.  The only way around this appears to be to strictly control access to llama_decode and LORA loading via a mutex.

### First Bad Commit

_No response_

### Relevant log output

```shell
It appears to be an error in vkQueueSubmit, line 1101.
```

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

## Issue #N/A: Bug: llama-perplexity error using multiple-choice binary data

**Link**: https://github.com/ggml-org/llama.cpp/issues/9316
**State**: open
**Created**: 2024-09-04T19:41:26+00:00
**Comments**: 0
**Labels**: bug, medium severity

### Description

### What happened?

"The multiple choice evaluation has been broken in llama.cpp via commit 6ff13987a.

The multiple choice evaluation uses binary data stored in params.prompt. Commit 6ff13987a adds prompt escape character processing, which modifies the binary data and renders it unusable. To preserve whatever utility 6ff13987a might have added, we add a flag indicating if the data stored in params.prompt is binary and, if so, avoid the escape processing."  @ikawrakow

@ikawrakow solved the problem in his llama.cpp fork in the following PR: https://github.com/ikawrakow/ik_llama.cpp/pull/33



### Name and Version

I tested the issue with the docker release of llama.cpp:

 ghcr.io/ggerganov/llama.cpp:full-cuda--b1-98a532d

### What operating system are you seeing the problem on?

Linux

### Relevant log output

_No response_

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

## Issue #N/A: llama : tool for evaluating quantization results per layer

**Link**: https://github.com/ggml-org/llama.cpp/issues/2783
**State**: open
**Created**: 2023-08-25T10:02:47+00:00
**Comments**: 8
**Labels**: enhancement, generation quality, roadmap

### Description

Following up on #2421, I think we should implement some better way to observe at which point of the inference the results start to deviate significantly between the classical and quantum models.

So I'm thinking of adding a simple tool that takes as input 2 `ggml` exported graphs - one classical and one quantum, of the same model. The tool evals both graphs on the CPU using `ggml` and prints detailed statistical information of the intermediate F32 results after each graph node. For example, each result node which has been given a name will be compared and we'll print stuff like, `min`, `max`, `avg`, `var`, etc.

I'm hoping with such tool to be able to detect which nodes in the computation require more precision in order to keep the quantization differences small enough and hopefully become an automated way of deciding which tensors require more bits than others.

cc @slaren I know you had similar ideas - we can discuss here how to add such support.
Currently I think the `ggml` g

[... truncated for brevity ...]

---

## Issue #N/A: persimmon crashes with CUDA: assertion failure `ggml_is_contiguous(src0)`

**Link**: https://github.com/ggml-org/llama.cpp/issues/5823
**State**: open
**Created**: 2024-03-01T19:27:09+00:00
**Comments**: 3
**Labels**: bug, model

### Description

Attempting to run a persimmon model with the CUDA backend fails an assertion in ggml_cuda_rope: `ggml_is_contiguous(src0)`

ref https://github.com/ggerganov/llama.cpp/pull/5668#issuecomment-1959988387

---

## Issue #N/A: Feature Request: Add support for Kokoro TTS

**Link**: https://github.com/ggml-org/llama.cpp/issues/11050
**State**: open
**Created**: 2025-01-03T05:28:06+00:00
**Comments**: 33
**Labels**: enhancement

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Devs, can you add support for Kokoro TTS? It's awesome in terms of accents and natural tone, considering it's size. It is currently one of the most popular models in Pandroker's TTS arena space on hugginface. Thanks!
https://huggingface.co/hexgrad/Kokoro-82M

### Motivation

Many, including me want to deploy it on cpu/edge devices

### Possible Implementation

_No response_

---

## Issue #N/A: MiniCPM 2b model support?

**Link**: https://github.com/ggml-org/llama.cpp/issues/5276
**State**: open
**Created**: 2024-02-02T08:06:39+00:00
**Comments**: 26
**Labels**: enhancement, good first issue

### Description


# Feature Description

Like Phi is supported, it would great to have this Mistral level 2b model ggufable. 

# Motivation

SOTA 2b model, a piece of art, read how they made it: 

https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20

---

## Issue #N/A: changelog : `libllama` API

**Link**: https://github.com/ggml-org/llama.cpp/issues/9289
**State**: open
**Created**: 2024-09-03T06:48:45+00:00
**Comments**: 10
**Labels**: documentation, roadmap

### Description

# Overview

This is a list of changes to the public interface of the `llama` library. Collaborators are encouraged to edit this post in order to reflect important changes to the API that end up merged into the `master` branch.

If you are building a 3rd party project that relies on `libllama`, it is recommended to follow this issue and check it before upgrading to new versions.

See also:

- [Changelog for `llama-server` REST API](https://github.com/ggerganov/llama.cpp/issues/9291)

## Recent API changes (most recent at the top)

| version | PR  | desc |
| ---     | --- | ---  |
| TBD.  | #14363 | Update `llama_context_params` - add `bool kv_unified` |
| b5740 | #13037 | Update `llama_model_quantize_params` |
| b5870 | #14631 | Remove `enum llama_vocab_pre_type` |
| b5435 | #13653 | Remove `llama_kv_cache_view_*` API |
| b5429 | #13194 | Update `llama_context_params` - add `bool swa_full` |
| b5311 | #13284 | Update `llama_context_params` - remove `logits_all` + rearrange flags |
| b51

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: llama-server hot swapping cvectors via API like we can do with LoRA adapters now

**Link**: https://github.com/ggml-org/llama.cpp/issues/10685
**State**: open
**Created**: 2024-12-06T09:10:27+00:00
**Comments**: 1
**Labels**: enhancement, good first issue

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

The ability load/unload and adjust the scale of cvectors via API, similar to the new LoRA scale/host-swap feature recently implmented:

```
POST /lora-adapters: Set list of LoRA adapters
```
To disable an adapter, either remove it from the list below, or set scale to 0.
Request format

To know the id of the adapter, use GET /lora-adapters
```
[
  {"id": 0, "scale": 0.2},
  {"id": 1, "scale": 0.8}
]
``

I read in the change log that this was inspired by cvector scaling which is alr

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Failed to run qwen2-57b-a14b-instruct-fp16.

**Link**: https://github.com/ggml-org/llama.cpp/issues/9628
**State**: open
**Created**: 2024-09-24T13:47:44+00:00
**Comments**: 4
**Labels**: bug, good first issue, high severity

### Description

### What happened?

I am trying to run Qwen2-57B-A14B-instruct, and I used llama-gguf-split to merge the gguf files from [Qwen/Qwen2-57B-A14B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct-GGUF).  But it's aborted with `terminate called after throwing an instance of 'std::length_error'
  what():  vector::_M_default_append
Aborted (core dumped)`。

### Name and Version

./build/bin/llama-cli --version
version: 3808 (699a0dc1)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
`(llama) root@201edf3683be:/home/llama.cpp# ./build/bin/llama-cli -m ./models/qwen2-57b-a14b-instruct-fp16.gguf -p "Beijing is the capital of" -n 64 -c 4096
build: 3808 (699a0dc1) with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu (debug)
main: llama backend init
main: load the model and apply lora adapter, if any
llama_model_loader: loaded meta data wi

[... truncated for brevity ...]

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

## Issue #N/A: server: avoid full prompt eval when 'prompt >= ctx'

**Link**: https://github.com/ggml-org/llama.cpp/issues/6855
**State**: open
**Created**: 2024-04-23T21:10:25+00:00
**Comments**: 2
**Labels**: enhancement, good first issue

### Description

When using the server for multi-turn chat, soon or later the prompt is going to surpass the context size, the current approach truncate the prompt by half of the context size excluding n_keep:

https://github.com/ggerganov/llama.cpp/blob/192090bae47960f0d38d4967abe398a5d190057e/examples/server/server.cpp#L1969-L1983

By doing that, common_part is going to match only n_keep tokens (when cache_prompt: true):

https://github.com/ggerganov/llama.cpp/blob/192090bae47960f0d38d4967abe398a5d190057e/examples/server/server.cpp#L2011-L2016

Technically, this is not a full prompt eval, n_keep is not revaluated, but it would be better to avoid this if possible, specially because prompt eval is slow on CPU.

---

## Issue #N/A: ggml : refactor ggml-cpu.c into multiple C++ source files

**Link**: https://github.com/ggml-org/llama.cpp/issues/10180
**State**: open
**Created**: 2024-11-05T07:12:48+00:00
**Comments**: 17
**Labels**: refactoring, roadmap

### Description

As per recent discussions (e.g. https://github.com/ggerganov/llama.cpp/pull/10144#pullrequestreview-2411814357), we should split the large `ggml-cpu.c` implementation into smaller modules - similar to how the CUDA backend is organized. We should utilize ~C++11~ C++ to reduce code duplication.

---

## Issue #N/A: Regressions on IQ3_XXS over time

**Link**: https://github.com/ggml-org/llama.cpp/issues/5856
**State**: open
**Created**: 2024-03-03T17:11:32+00:00
**Comments**: 16
**Labels**: bug

### Description

If I quantize [this gguf](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/blob/main/mixtral-8x7b-instruct-v0.1.Q8_0.gguf) with [this imatrix](https://huggingface.co/datasets/ikawrakow/imatrix-from-wiki-train/blob/main/mixtral-8x7b-instruct-v0.1.imatrix) using this command:
```
quantize.exe --allow-requantize --imatrix mixtral-8x7b-instruct-v0.1.imatrix mixtral-8x7b-instruct-v0.1.Q8_0.gguf mixtral-8x7b-instruct-v0.1.IQ3_XXS.gguf IQ3_XXS
```
and I calculate perplexity with this command:
```
perplexity.exe -f wiki.test.raw --chunks 1000 --seed 42 --threads 8 --log-disable --no-mmap --mlock --ctx-size 512 --n-gpu-layers 999 --model mixtral-8x7b-instruct-v0.1.IQ3_XXS.gguf
```
I get three much different PPL values on three different versions of quantize.exe, everything else being equal:
```
b2037 31-1-2024 : 4.7009 +/- 0.02569
b???? 25-2-2024 : 4.7249 +/- 0.02576
b2329 03-3-2024 : 4.8491 +/- 0.02636
```
I suspect that there have been multiple cumulative regressi

[... truncated for brevity ...]

---

## Issue #N/A: ggml : add WebGPU backend

**Link**: https://github.com/ggml-org/llama.cpp/issues/7773
**State**: open
**Created**: 2024-06-05T14:24:37+00:00
**Comments**: 21
**Labels**: help wanted, research 🔬, roadmap

### Description

I hope that this would be relatively easy to do since AFAIK WebGPU allows us to write kernels in a shader language, so we have experience how to create such backends.

There has been some initial work in https://github.com/ggerganov/ggml/pull/585 - could be useful as a starting point

---

## Issue #N/A: server: exit failure if `--embedding` is set with an incoherent `--ubatch-size`

**Link**: https://github.com/ggml-org/llama.cpp/issues/6263
**State**: open
**Created**: 2024-03-23T17:03:49+00:00
**Comments**: 5
**Labels**: enhancement, help wanted, good first issue, server/webui

### Description

### Context

there is no advantage to increase `n_batch` above `n_ubatch` with embeddings models with pooling, because the entire batch must fit in a physical batch (ie. `n_ubatch`). `n_batch` is always `>= n_ubatch`.

- See @slaren comment in: https://github.com/ggerganov/llama.cpp/pull/6254#discussion_r1536661327

### Proposition
Exit failure if `--embedding` is set and `--ubatch-size` != `--batch-size` in the `server` example. Probably also in the `retrieval` example in #6193.

Aldo probably KV `bert.context_size` must be taken into account.

---

## Issue #N/A: Misc. bug: inconsistent locale for printing GGUF kv data across examples

**Link**: https://github.com/ggml-org/llama.cpp/issues/10613
**State**: open
**Created**: 2024-12-01T10:44:49+00:00
**Comments**: 0
**Labels**: bug

### Description

### Name and Version

> ./build/bin/llama-cli --version
version: 4232 (6acce397)
built with cc (GCC) 14.2.1 20240910 for x86_64-pc-linux-gnu


### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-cli, Other (Please specify in the next section)

### Problem description & steps to reproduce

I am using a Linux PC with the locale set like this:

```bash
> locale
LANG=en_US.UTF-8
LC_CTYPE="en_US.UTF-8"
LC_NUMERIC=de_DE.UTF-8
LC_TIME=de_DE.UTF-8
LC_COLLATE="en_US.UTF-8"
LC_MONETARY=de_DE.UTF-8
LC_MESSAGES="en_US.UTF-8"
LC_PAPER=de_DE.UTF-8
LC_NAME=de_DE.UTF-8
LC_ADDRESS=de_DE.UTF-8
LC_TELEPHONE=de_DE.UTF-8
LC_MEASUREMENT=de_DE.UTF-8
LC_IDENTIFICATION=de_DE.UTF-8
LC_ALL=
```

The way floating point numbers from the model GGUF kv data are printed is inconsistent depending on which binary I run.
`llama_cli` prints them with a point, `llama-perplexity` prints them with a comma.
It may make sense to completely ignore any locale set

[... truncated for brevity ...]

---

## Issue #N/A: Bug: ABI problem in binary file "llama-b3187-bin-win-msvc-arm64.zip"

**Link**: https://github.com/ggml-org/llama.cpp/issues/8050
**State**: open
**Created**: 2024-06-21T07:11:57+00:00
**Comments**: 1
**Labels**: bug, medium severity

### Description

### What happened?

In release tag https://github.com/ggerganov/llama.cpp/releases/tag/b3187, file "llama-cli.exe" in binary file "llama-b3187-bin-win-msvc-arm64.zip" is windows X64 ABI. It's not Windows ARM64 ABI. What is the reason? Why you mark it to "win-msvc-arm64"? 

Logs:
C:\llama-b3187-bin-win-msvc-arm64>dumpbin /headers llama-cli.exe
FILE HEADER VALUES
            **8664 machine (x64)**
 
And llama-cli.exe depends below four libraries. Where should I download them?

libstdc++-6.dll
libwinpthread-1.dll
libgcc_s_seh-1.dll
libgomp-1.dll

### Name and Version

Tag b3187

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
N/A
```


---

## Issue #N/A: Feature Request: Installable package via winget

**Link**: https://github.com/ggml-org/llama.cpp/issues/8188
**State**: open
**Created**: 2024-06-28T13:27:20+00:00
**Comments**: 19
**Labels**: enhancement, help wanted

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

On macos/linux, user can install a pre-built version llama.cpp easily via `brew`

It would be nice to have the equivalent to that on windows, via `winget`

### Motivation

The pre-built binary is already available via releases: https://github.com/ggerganov/llama.cpp/releases

It would be nice to somehow push them to https://winget.run/

However, I'm not familiar with working on windows, so I create this issue to further discuss and to look for help from the community.

### Possible Implemen

[... truncated for brevity ...]

---

## Issue #N/A: Compile bug: ios swift xcode build error when upgrade to llama : use cmake for swift build 

**Link**: https://github.com/ggml-org/llama.cpp/issues/10747
**State**: open
**Created**: 2024-12-10T05:12:25+00:00
**Comments**: 41
**Labels**: help wanted, good first issue, build

### Description

### Git commit

$git rev-parse HEAD 43ed389a3f102517e6f7d5620d8e451e88afbf27

### Operating systems

Mac

### GGML backends

Metal

### Problem description & steps to reproduce

ios swift xcode build error when upgrade to

- https://github.com/ggerganov/llama.cpp/pull/10525

Before the upgrade, the code compiled successfully. After the upgrade, it throws a compilation error: "Cannot find type 'xxx' in scope."

<img width="1721" alt="image" src="https://github.com/user-attachments/assets/1bc2e76a-158a-4aa3-9755-855930f2f7ed">


### First Bad Commit

43ed389a3f102517e6f7d5620d8e451e88afbf27

### Relevant log output

```shell
/ios/llama.cpp.swift/LibLlama.swift:8:39 Cannot find type 'llama_batch' in scope

/ios/llama.cpp.swift/LibLlama.swift:12:37 Cannot find type 'llama_batch' in scope

/ios/llama.cpp.swift/LibLlama.swift:12:56 Cannot find type 'llama_token' in scope

/ios/llama.cpp.swift/LibLlama.swift:12:76 Cannot find type 'llama_pos' in scope

/ios/llama.cpp.swift/LibL

[... truncated for brevity ...]

---

