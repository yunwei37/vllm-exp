# roadmap - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 26
- Closed Issues: 4

### Label Distribution

- roadmap: 30 issues
- enhancement: 13 issues
- good first issue: 8 issues
- refactoring: 8 issues
- help wanted: 7 issues
- research 🔬: 6 issues
- performance: 3 issues
- generation quality: 2 issues
- documentation: 2 issues
- testing: 2 issues

---

## Issue #N/A: metal : simplify kernel arguments using a struct

**Link**: https://github.com/ggml-org/llama.cpp/issues/3229
**State**: closed
**Created**: 2023-09-17T17:10:35+00:00
**Closed**: 2025-03-07T07:40:54+00:00
**Comments**: 10
**Labels**: good first issue, refactoring, roadmap

### Description

Create a struct `ggml_metal_locals` and populate using `GGML_TENSOR_LOCALS` similar to what we do in `ggml.c`:

https://github.com/ggerganov/llama.cpp/blob/3b4bab6a38502d9e68587c2c19f26472480ec4dd/ggml.c#L244-L256

Refactor all kernels to accept a single struct of `ggml_metal_locals` in order to avoid long lists of arguments such as:

https://github.com/ggerganov/llama.cpp/blob/3b4bab6a38502d9e68587c2c19f26472480ec4dd/ggml-metal.m#L753-L782

https://github.com/ggerganov/llama.cpp/blob/3b4bab6a38502d9e68587c2c19f26472480ec4dd/ggml-metal.metal#L29-L61

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

## Issue #N/A: llama : integer type consistency in `llama.h`

**Link**: https://github.com/ggml-org/llama.cpp/issues/4574
**State**: open
**Created**: 2023-12-21T19:55:14+00:00
**Comments**: 2
**Labels**: enhancement, good first issue, refactoring, roadmap

### Description

# Feature Description

llama.h should prefer to use `sized` (always) + `signed` (mostly) integers.

# Motivation

The integer types in `llama.h` right now are.

| Count | Type            |
|---------|------------------|
| 33      | `int`              |
| 10      | `int32_t`       |
| 24      | `uint32_t`     |
| 2        | `int64_t`       |
| 2        | `uint64_t`    | 

In #4540 there was a discussion around preferences for integer types on new methods. 

Avoiding `int` makes cross platform code simpler at essentially no cost.
Signed makes arithmetic simpler at the cost of some bits if you need something large.

# Possible Implementation

1. Change all `int`'s to `int32_t`
2. As code changes try to prefer signed integers.

We could also do some higher-impact things, but I'd take the lower-impact slower changes over a large find-and-replace.

---

## Issue #N/A: ggml : unified CMake build

**Link**: https://github.com/ggml-org/llama.cpp/issues/6913
**State**: open
**Created**: 2024-04-25T19:15:40+00:00
**Comments**: 4
**Labels**: enhancement, build, refactoring, roadmap

### Description

Currently the [ggml](https://github.com/ggerganov/ggml), [llama.cpp](https://github.com/ggerganov/llama.cpp) and [whisper.cpp](https://github.com/ggerganov/whisper.cpp) projects share the same source of the `ggml` library, but have different CMake scripts. The scripts are adapted to the specifics of the projects and are quite similar with each other - all of them build `ggml`. Still, there are differences due to manually rewriting them and applying changes from one repo to another

The goal in this task is to unify, deduplicate and streamline the build process of `ggml` with proper CMake scripts that are shared across the projects. This will simplify changes in the future and will also help other 3rd party projects that depend on `ggml`

More on this topic has been discussed in:

- https://github.com/ggerganov/llama.cpp/issues/5890
- https://github.com/ggerganov/ggml/pull/804

To achieve that, the `ggml`-related sources in `llama.cpp` and `whisper.cpp` would likely have to be 

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

## Issue #N/A: ggml : refactor ggml-cpu.c into multiple C++ source files

**Link**: https://github.com/ggml-org/llama.cpp/issues/10180
**State**: open
**Created**: 2024-11-05T07:12:48+00:00
**Comments**: 17
**Labels**: refactoring, roadmap

### Description

As per recent discussions (e.g. https://github.com/ggerganov/llama.cpp/pull/10144#pullrequestreview-2411814357), we should split the large `ggml-cpu.c` implementation into smaller modules - similar to how the CUDA backend is organized. We should utilize ~C++11~ C++ to reduce code duplication.

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

## Issue #N/A: tutorials : list for llama.cpp

**Link**: https://github.com/ggml-org/llama.cpp/issues/13523
**State**: open
**Created**: 2025-05-14T05:00:53+00:00
**Comments**: 5
**Labels**: help wanted, good first issue, roadmap

### Description

Project: https://github.com/orgs/ggml-org/projects/6

List:

- [tutorial : compute embeddings using llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/7712)
- [tutorial : parallel inference using Hugging Face dedicated endpoints](https://github.com/ggml-org/llama.cpp/discussions/9041)
- [tutorial : KV cache reuse with llama-server](https://github.com/ggml-org/llama.cpp/discussions/13606)
- [tutorial : measuring time to first token (TTFT) and time between tokens (TBT)](https://github.com/ggml-org/llama.cpp/discussions/14115)

TODO:
- [ ] https://github.com/ggml-org/llama.cpp/discussions/13488
- [ ] https://github.com/ggml-org/llama.cpp/discussions/13134
- [ ] https://github.com/ggml-org/llama.cpp/discussions/13251
- [ ] https://github.com/ggml-org/llama.cpp/discussions/12742
- [ ] How to get started with webui development (ref: https://github.com/ggml-org/llama.cpp/issues/13523#issuecomment-2879256096)
- [ ] etc.

Simply search for "How to" in the Discussions: https://github.c

[... truncated for brevity ...]

---

## Issue #N/A: Move gguf fuzzers to the llama.cpp repository

**Link**: https://github.com/ggml-org/llama.cpp/issues/11514
**State**: open
**Created**: 2025-01-30T15:57:53+00:00
**Comments**: 5
**Labels**: enhancement, testing, roadmap

### Description

Fuzz testing of llama.cpp in OSS-Fuzz has been very valuable to detect leaks and security issues in the model loading code. Unfortunately, the build of the [current fuzzers](https://github.com/google/oss-fuzz/tree/master/projects/llamacpp) has been broken for a long time, and new code is not being tested.

We should move the fuzzers to this repository and ensure that they are maintained. More details: https://google.github.io/oss-fuzz/advanced-topics/ideal-integration/

@DavidKorczynski the current implementation seems to be Apache licensed, which would complicate moving the code here. Would it be possible to re-license it as MIT?

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

## Issue #N/A: ci : add Arm Cobalt 100 runners

**Link**: https://github.com/ggml-org/llama.cpp/issues/11275
**State**: closed
**Created**: 2025-01-17T09:17:03+00:00
**Closed**: 2025-02-22T11:09:50+00:00
**Comments**: 0
**Labels**: help wanted, good first issue, testing, roadmap

### Description

There are some new Github Actions runners "powered by the Cobalt 100-based processors":

https://github.blog/changelog/2025-01-16-linux-arm64-hosted-runners-now-available-for-free-in-public-repositories-public-preview/

Not sure what this processor is specifically, but it might have some Arm features that would be useful to exercise in the CI. We should look into more details and add workflows if it makes sense.

---

## Issue #N/A: llama : store token ids in the KV Cache

**Link**: https://github.com/ggml-org/llama.cpp/issues/9113
**State**: open
**Created**: 2024-08-21T07:38:02+00:00
**Comments**: 2
**Labels**: enhancement, roadmap

### Description

### Discussed in https://github.com/ggerganov/llama.cpp/discussions/9043

<div type='discussions-op-text'>

<sup>Originally posted by **julmb** August 15, 2024</sup>
Let's say I want to use llama.cpp as a shared library to build a service that other applications can make requests to. When this service gets a request, it feeds it to the model via `llama_decode`. The tokens that make up the request are processed and added to the internal KV cache.

Now, when the next request arrives, I need to decide which prefix of the request is already cached and therefore does not need to be processed again. From what I understand the KV cache does not store the actual tokens. So I have no way of knowing which part of the cache needs to be cleared and which part of the request tokens need to be fed to the model.

As far as I can tell, I have two options:
1. Clear the entire cache and reprocess the entire request. This is of course slow, especially for requests that share a large prefix.
2.

[... truncated for brevity ...]

---

## Issue #N/A: changelog : `llama-server` REST API

**Link**: https://github.com/ggml-org/llama.cpp/issues/9291
**State**: open
**Created**: 2024-09-03T06:56:11+00:00
**Comments**: 16
**Labels**: documentation, roadmap

### Description

# Overview

This is a list of changes to the public HTTP interface of the `llama-server` example. Collaborators are encouraged to edit this post in order to reflect important changes to the API that end up merged into the `master` branch.

If you are building a 3rd party project that relies on `llama-server`, it is recommended to follow this issue and check it carefully before upgrading to new versions.

See also:

- [Changelog for `libllama` API](https://github.com/ggerganov/llama.cpp/issues/9289)

## Recent API changes (most recent at the top)

| version | PR  | desc |
| ---     | --- | ---  |
| TBD.  | #13660 | Remove `/metrics` fields related to KV cache tokens and cells` |
| b5223 | #13174 | For chat competion, if last message is assistant, it will be a prefilled message |
| b4599 | #9639 | `/v1/chat/completions` now supports `tools` & `tool_choice` |
| TBD.  | #10974 | `/v1/completions` is now OAI-compat |
| TBD.  | #10783 | `logprobs` is now OAI-compat, default to pre-sampling p

[... truncated for brevity ...]

---

## Issue #N/A: llama : create llamax library

**Link**: https://github.com/ggml-org/llama.cpp/issues/5215
**State**: open
**Created**: 2024-01-30T13:01:06+00:00
**Comments**: 18
**Labels**: refactoring, roadmap

### Description

Depends on: https://github.com/ggerganov/llama.cpp/issues/5214

The `llamax` library will wrap `llama` and expose common high-level functionality. The main goal is to ease the integration of `llama.cpp` into 3rd party projects. Ideally, most projects would interface through the `llamax` API for all common use cases, while still have the option to use the low-level `llama` API for more uncommon applications that require finer control of the state.

A simple way to think about `llamax` is that it will simplify all of the existing examples in `llama.cpp` by hiding the low-level stuff, such as managing the KV cache and batching requests.

Roughly, `llamax` will require it's own state object and a run-loop function.

The specifics of the API are yet to be determined - suggestions are welcome.


---

## Issue #N/A: kv-cache : improve defrag logic

**Link**: https://github.com/ggml-org/llama.cpp/issues/13497
**State**: open
**Created**: 2025-05-13T08:09:25+00:00
**Comments**: 0
**Labels**: enhancement, performance, roadmap

### Description

Following the optimization in #13493, I realized that the defragmentation can become much better so that it can further improve  the Flash Attention masking. 

Currently we defrag the following cache like this:

```bash
# before defrag
00000000...11111.......2222222....2010212012012....

# after defrag
000000001111122222222010212012012..................
```

I.e. we only "fill" the holes, but the sequences remain scattered. We can do better like this:

```
# new defrag
000000000000111111111222222222222..................
```

By doing so, the [FA-vec masking logic](#13493) will remain effective even after many generations.

---

## Issue #N/A: Feature Request: Granite 4 Support

**Link**: https://github.com/ggml-org/llama.cpp/issues/13275
**State**: closed
**Created**: 2025-05-02T23:07:15+00:00
**Closed**: 2025-07-14T23:49:13+00:00
**Comments**: 6
**Labels**: enhancement, roadmap

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

This issue is to track work to support IBM's [Granite 4](https://huggingface.co/ibm-granite/granite-4.0-tiny-preview) model architecture (`GraniteMoEHybrid` in `transformers`). The model uses a number of components that are not yet supported in `llama.cpp`, but are being worked independently, so I'm raising this issue to triangulate the different work streams that will be needed to support the model.

## Necessary Components

- [x] Mamba2 layers
    - [x] Ongoing work by @compilade: https://github.

[... truncated for brevity ...]

---

## Issue #N/A: server : improvements and maintenance

**Link**: https://github.com/ggml-org/llama.cpp/issues/4216
**State**: open
**Created**: 2023-11-25T09:57:53+00:00
**Comments**: 120
**Labels**: help wanted, refactoring, server/webui, roadmap

### Description

The [server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server) example has been growing in functionality and unfortunately I feel it is not very stable at the moment and there are some important features that are still missing. Creating this issue to keep track on some of these points and try to draw more attention from the community. I guess, some of the tasks are relatively big and would require significant efforts to complete

- [x] **Support chat templates**
  We need to have separation between the user input and the special tokens, so that the tokenization is performed correctly. See the following comments / commits for more context:
  https://github.com/ggerganov/llama.cpp/pull/4160#discussion_r1403675264
  https://github.com/ggerganov/llama.cpp/pull/4198/commits/c544faed749240fe5eac2bc042087c71f79a0728
  https://github.com/ggerganov/llama.cpp/pull/4160#issuecomment-1824984718

  We already support extracting meta information from the GGUF model files th

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: resize an existing context

**Link**: https://github.com/ggml-org/llama.cpp/issues/11577
**State**: open
**Created**: 2025-02-01T15:51:53+00:00
**Comments**: 4
**Labels**: enhancement, roadmap

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Being able to resize an existing context, either by enlarging it or shrinking it.
When shrinking the context assume we have enough unused slots in the KV cache, and otherwise return an error code.

This would be the ideal API for this:
```cpp
// Change the size of the context.
// When shrinking, ensure that there are enough empty slots in the KV cache to accommodate the new size.
//   0 - success
// < 0 - error. the KV cache does not have enough empty slots to accommodate the new size
LLAMA_API i

[... truncated for brevity ...]

---

## Issue #N/A: llama : refactor the llm.build_xxx functions

**Link**: https://github.com/ggml-org/llama.cpp/issues/5239
**State**: open
**Created**: 2024-01-31T12:55:44+00:00
**Comments**: 2
**Labels**: good first issue, refactoring, roadmap

### Description

Now that we support a large amount of architectures, we can clearly see the patterns when constructing the compute graphs - i.e. optional biases, different norm types, QKV vs Q+K+V, etc.

We should deduplicate the copy-paste portions in functions such as `llm.build_llama()`, `llm.build_falcon()`, etc.

The advantage of the current code is that it is easy to look into the graph of a specific architecture. When we refactor this, we will lose this convenience to some extend. So we should think about making this refactoring in such a way that we don't completely obscure which parts of the graph belong to which architectures

Open for ideas and suggestions how to do this best

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

## Issue #N/A: ggml : add GPU support for Mamba models

**Link**: https://github.com/ggml-org/llama.cpp/issues/6758
**State**: open
**Created**: 2024-04-19T06:47:35+00:00
**Comments**: 32
**Labels**: enhancement, help wanted, Nvidia GPU, roadmap

### Description

Recently, initial Mamba support (CPU-only) has been introduced in #5328 by @compilade 

In order to support running these models efficiently on the GPU, we seem to be lacking kernel implementations for the following 2 ops:

- `GGML_OP_SSM_CONV`
- `GGML_OP_SSM_SCAN`

Creating this issue to keep track of this and give more visibility of this feature. Help with implementing the missing kernels for CUDA and Metal (and other backends potentially) is welcome. We can also discuss if anything else is required to better support this architecture in `llama.cpp`

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

## Issue #N/A: server : add "token healing" support

**Link**: https://github.com/ggml-org/llama.cpp/issues/5765
**State**: open
**Created**: 2024-02-28T12:10:30+00:00
**Comments**: 6
**Labels**: enhancement, good first issue, server/webui, roadmap

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

Hi! I am experimenting with using llama.cpp as a general-purpose code completion backend, similar to TabNine.

I am encountering a small problem: if the completion prompt ends mid-word, the results are not very accurate. For example, for a prompt such as `Five, Fo

[... truncated for brevity ...]

---

## Issue #N/A: llama : add CLI assistant

**Link**: https://github.com/ggml-org/llama.cpp/issues/10688
**State**: open
**Created**: 2024-12-06T12:07:53+00:00
**Comments**: 4
**Labels**: enhancement, good first issue, roadmap

### Description

The https://github.com/AnswerDotAI/shell_sage project seems quite fun and useful. [GitHub Copilot CLI](https://www.npmjs.com/package/@githubnext/github-copilot-cli) was also a rather useful utility. 

We should build a fully-local alternative leveraging `llama-server`'s advanced context reuse techniques in the same spirit as [llama.vim](https://github.com/ggml-org/llama.vim). Speculative decoding should be very effective in this scenario as well. Likely the Qwen 2.5 family would be ideal for this use case.

---

## Issue #N/A: llama : enable FA by default and disable it per-layer

**Link**: https://github.com/ggml-org/llama.cpp/issues/10005
**State**: open
**Created**: 2024-10-22T14:07:59+00:00
**Comments**: 18
**Labels**: enhancement, roadmap

### Description

See the discussion starting here: https://github.com/ggerganov/llama.cpp/issues/9991#issuecomment-2428407002 and the proposed solution here: https://github.com/ggerganov/llama.cpp/issues/9991#issuecomment-2428868490.

Additionally, switch to F32 precision for the `K*Q` matrix multiplication by default.

Marking this as good first issue as an opportunity for new contributors, but also it is kind of high priority, so we should probably implement this in a day or two if there is no progress. @slaren or @JohannesGaessler in case you already started to work on it, fill free to assign to the issue and finish it.

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

## Issue #N/A: Add a new `llama_load_model_from_buffer()` method to compliment `llama_load_model_from_file()`

**Link**: https://github.com/ggml-org/llama.cpp/issues/6311
**State**: open
**Created**: 2024-03-26T02:03:02+00:00
**Comments**: 6
**Labels**: enhancement, roadmap

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

There should be a `llama_load_model_from_buffer()` function added to `llama.h`/`llama.cpp` to compliment `llama_load_model_from_file()`. Instead of loading a model from a file, it should instead read the model from a user-provided buffer. 

# Motivation

I'm wor

[... truncated for brevity ...]

---

## Issue #N/A: server : add support for multiple responses

**Link**: https://github.com/ggml-org/llama.cpp/issues/11142
**State**: open
**Created**: 2025-01-08T16:11:24+00:00
**Comments**: 2
**Labels**: server/api, server, roadmap

### Description

It would be very useful to add multi-response support per slot so that a single request would be able to generate `n` independent completions. This functionality is useful in different situations - for example, a FIM completion can provide multiple alternative suggestions at a smaller or equal compute cost compared to running them sequentially.

I think this can be implemented by adding multiple sequence id per slot (instead of having just one like we currently do). However, I am not sure how yet much complexity would be introduced to support this.

---

