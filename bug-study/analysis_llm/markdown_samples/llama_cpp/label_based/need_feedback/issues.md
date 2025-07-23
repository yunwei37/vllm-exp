# need_feedback - issues

**Total Issues**: 5
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 5

### Label Distribution

- need feedback: 5 issues
- enhancement: 4 issues
- stale: 3 issues
- help wanted: 1 issues
- model: 1 issues
- generation quality: 1 issues
- performance: 1 issues
- server/webui: 1 issues

---

## Issue #N/A: Add metadata override and also generate dynamic default filename when converting gguf

**Link**: https://github.com/ggml-org/llama.cpp/issues/7165
**State**: closed
**Created**: 2024-05-09T06:20:12+00:00
**Closed**: 2024-05-13T03:43:12+00:00
**Comments**: 1
**Labels**: enhancement, help wanted, need feedback

### Description

This is a formalized ticket for this PR https://github.com/ggerganov/llama.cpp/pull/4858 so people are aware and can contribute to figuring out if this idea makes sense... and if so then what needs to be done before this can be merged in from a feature requirement perspective.


# Feature Description and Motivation

## Metadata Override

Often safetensors provided by external parties maybe missing certain metadata or have incorrectly formatted metadata. To make things easier to find in hugging face, accurate metadata is a must.

The idea is to allow users to override metadata in the generated gguf by including a json metadata file

```
./llama.cpp/convert.py maykeye_tinyllama --outtype f16 --metadata maykeye_tinyllama-metadata.json
```

where the metadata override file may look like:

```json
{
    "general.name": "TinyLLama",
    "general.version": "v0",
    "general.author": "mofosyne",
    "general.url": "https://huggingface.co/mofosyne/TinyLLama-v0-llamafile",

[... truncated for brevity ...]

---

## Issue #N/A: llama : make vocabs LFS objects?

**Link**: https://github.com/ggml-org/llama.cpp/issues/7128
**State**: closed
**Created**: 2024-05-07T18:51:55+00:00
**Closed**: 2024-06-23T01:12:22+00:00
**Comments**: 7
**Labels**: enhancement, need feedback, stale

### Description

It's nice to have a collection of vocabs using different pre-tokenizers in order to test tokenization more widely. However, the number of vocab files controlled in the repo will keep growing:

https://github.com/ggerganov/llama.cpp/tree/master/models

These files are typically a few MB, so the repo size is significantly affected by them.

One option is to make these files LFS objects. Another option is to not source control them and either remove the tests, or generate them on the fly. But the latter might be flaky because we will depend on many 3rd party repositories to provide the tokenizers.

Are there any better alternatives?

Update: git lfs is not an option. I think for the short-term we will commit vocabs only for new types of pre-tokenizers. The vocab data compresses relatively good (factor ~x3), so hopefully the repo size will not be affect too badly

---

## Issue #N/A: `quantize`: add imatrix and dataset metadata in GGUF

**Link**: https://github.com/ggml-org/llama.cpp/issues/6656
**State**: closed
**Created**: 2024-04-13T10:13:08+00:00
**Closed**: 2024-04-26T18:06:34+00:00
**Comments**: 3
**Labels**: enhancement, model, generation quality, need feedback

### Description

### Motivation
I was reading [thanks](https://huggingface.co/spaces/ggml-org/gguf-my-repo/discussions/41#661a27157a16dc848a58a261) to @julien-c this [reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1ba55rj/overview_of_gguf_quantization_methods/?rdt=36175) from @he29-net :+1: 

> You can't easily tell whether a model was quantized with the help of importance matrix just from the name. I first found this annoying, because it was not clear if and how the calibration dataset affects performance of the model in other than just positive ways. But recent tests in llama.cpp [discussion #5263](https://github.com/ggerganov/llama.cpp/discussions/5263) show, that while the data used to prepare the imatrix slightly affect how it performs in (un)related languages or specializations, any dataset will perform better than a "vanilla" quantization with no imatrix. So now, instead, I find it annoying because sometimes the only way to be sure I'm using the better imatrix version is to re-quan

[... truncated for brevity ...]

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

## Issue #N/A: [User] Regression with CodeLlama 7B

**Link**: https://github.com/ggml-org/llama.cpp/issues/3384
**State**: closed
**Created**: 2023-09-28T20:01:22+00:00
**Closed**: 2024-04-03T01:15:32+00:00
**Comments**: 7
**Labels**: need feedback, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Using [this Codellama 7B Q3_K_M model](https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/blob/74bf05c6562b9431494d994081b671206621c199/codellama-7b.Q3_K_M.gguf) uploaded by @TheBloke on August 24th with llama.cpp versions up until #3228 was merged produced the followi

[... truncated for brevity ...]

---

