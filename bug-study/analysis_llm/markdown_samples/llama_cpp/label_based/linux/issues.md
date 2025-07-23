# linux - issues

**Total Issues**: 1
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 1

### Label Distribution

- need more info: 1 issues
- performance: 1 issues
- linux: 1 issues

---

## Issue #N/A: 4bit 65B model overflow 64GB of RAM

**Link**: https://github.com/ggml-org/llama.cpp/issues/702
**State**: closed
**Created**: 2023-04-02T08:37:42+00:00
**Closed**: 2023-04-19T08:20:48+00:00
**Comments**: 7
**Labels**: need more info, performance, linux

### Description

# Prerequisites

I am running the latest code. Development is very rapid so there are no tagged versions as of now.
I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior
During inference, there should be no or minimum disk activities going on, and disk should not be a bottleneck once pass the model loading stage.

# Current Behavior
My disk should have a continuous reading speed of over 100MB/s, however, during the loading of the model, it only loads at around 40MB/s. After this very slow loading of Llama 65b model (converted from GPTQ with group size of 128), llama.cpp start to inference, however during the inference the programme continue to occupy t

[... truncated for brevity ...]

---

