# AMD_GPU - issues

**Total Issues**: 7
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 7

### Label Distribution

- AMD GPU: 7 issues
- stale: 3 issues
- performance: 2 issues
- bug-unconfirmed: 2 issues
- high severity: 1 issues
- low severity: 1 issues
- generation quality: 1 issues

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

## Issue #N/A: Misc. bug: hipGraph causes a crash in hipGraphDestroy

**Link**: https://github.com/ggml-org/llama.cpp/issues/11949
**State**: closed
**Created**: 2025-02-18T22:54:45+00:00
**Closed**: 2025-03-02T20:49:39+00:00
**Comments**: 0
**Labels**: AMD GPU

### Description

First encountered when testing https://github.com/ggml-org/llama.cpp/pull/11867, but this is a problem in master too. Debugged to a bug in rocm-clr: https://github.com/ROCm/clr/issues/138

This issue tracks that currently non-defaults builds with GGML_HIP_GRAPHS=On are unreliable and will be closed when the corresponding upstream bug is addressed.

---

## Issue #N/A: Bug: Failed to allocate memory on the 2nd GPU for loading large model

**Link**: https://github.com/ggml-org/llama.cpp/issues/8207
**State**: closed
**Created**: 2024-06-29T12:05:27+00:00
**Closed**: 2024-06-29T18:06:28+00:00
**Comments**: 8
**Labels**: AMD GPU, bug-unconfirmed, high severity

### Description

### What happened?

I am running:
* 256 GB RAM
* 2GPUs: AMD RX 7900 XTX x 2
* ROCm 6.1.3

I compiled from source with:

> make GGML_HIPBLAS=1 AMDGPU_TARGETS=gfx1100 -j$(lscpu | grep '^Core(s)' | awk '{print $NF}')

When I run large files like, mixtral_8x22b.gguf or command-r-plus_104b.gguf, I encountered errors:

> ./llama-cli -m ../../ollama_gguf/gguf/command-r-plus.gguf -p "What is machine learning?" -ngl 999

```
ggml_cuda_init: found 2 ROCm devices:
  Device 0: Radeon RX 7900 XTX, compute capability 11.0, VMM: no
  Device 1: Radeon RX 7900 XTX, compute capability 11.0, VMM: no
llm_load_tensors: ggml ctx size =    0.88 MiB
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 27846.97 MiB on device 0: cudaMalloc failed: out of memory
llama_model_load: error loading model: unable to allocate backend buffer
llama_load_model_from_file: failed to load model
llama_init_from_gpt_params: error: failed to load model '../../ollama_gguf/gguf/command-r-plus.gguf'
main: e

[... truncated for brevity ...]

---

## Issue #N/A: iGPU offloading Bug: Memory access fault by GPU node-1 (appeared once only)

**Link**: https://github.com/ggml-org/llama.cpp/issues/7829
**State**: closed
**Created**: 2024-06-08T06:04:54+00:00
**Closed**: 2024-07-23T01:06:44+00:00
**Comments**: 1
**Labels**: AMD GPU, bug-unconfirmed, stale, low severity

### Description

### What happened?

I am comparing inference with and without AMD iGPU offloading with ROCm.

The setup is documented at https://github.com/eliranwong/MultiAMDGPU_AIDev_Ubuntu/blob/main/igpu_only.md#compare-cpu-vs-openblas-vs-rocm-vs-rocmigpu-offloading

The result shows that AMD iGPU offloading with ROCm runs roughly 1.5x faster.

It is interesting to note that the first time I tried to run the following command:

> ./main -t $(lscpu | grep '^Core(s)' | awk '{print $NF}') --temp 0 -m '/home/eliran/freegenius/LLMs/gguf/mistral.gguf' -p "What is machine learning?" -ngl 33

I got the following error:

```
Memory access fault by GPU node-1 (Agent handle: 0x613061b881f0) on address 0x9000. Reason: Page not present or supervisor privilege.
Aborted (core dumped)
```

However, it appeared once only.  Further inference with the same command runs smoothly.  It is not a practical problem, as it happened once only.  All later inferences runs without an issue.

### Name and Versio

[... truncated for brevity ...]

---

## Issue #N/A: special token handling sometimes produces garbage output with AMD ROCM/HIP

**Link**: https://github.com/ggml-org/llama.cpp/issues/3705
**State**: closed
**Created**: 2023-10-21T02:19:36+00:00
**Closed**: 2024-04-04T01:07:44+00:00
**Comments**: 8
**Labels**: generation quality, AMD GPU, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Running models with special tokens (e.g. ChatML) with GPU offload via HIPBLAS should produce output similar to running pure CPU

# Current Behavior

Instead running with -ngl 35 and -ngl 32 causes the model to fill the context with hashes "#"

# Environment and 

[... truncated for brevity ...]

---

## Issue #N/A: [bug] ROCm segfault when running multi-gpu inference.

**Link**: https://github.com/ggml-org/llama.cpp/issues/3451
**State**: closed
**Created**: 2023-10-03T02:26:38+00:00
**Closed**: 2023-10-05T04:33:47+00:00
**Comments**: 5
**Labels**: AMD GPU

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Expected Tensor split to leverage multi gpus.

# Current Behavior

Segfault after model loading when using multi-gpu. Correct inference when using either GPU(two vega-56s  installed) and HIP_VISIBLE_DEVICES to force single GPU inference.

# Environment and Conte

[... truncated for brevity ...]

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

