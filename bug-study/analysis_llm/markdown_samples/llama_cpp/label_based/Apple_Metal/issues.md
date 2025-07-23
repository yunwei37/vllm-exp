# Apple_Metal - issues

**Total Issues**: 5
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 5

### Label Distribution

- Apple Metal: 5 issues
- bug-unconfirmed: 2 issues
- stale: 2 issues
- medium severity: 2 issues
- good first issue: 1 issues
- help wanted: 1 issues
- performance: 1 issues
- bug: 1 issues
- high severity: 1 issues

---

## Issue #N/A: [Metal] Context init optimization opportunity: metal library is compiled for every llama context

**Link**: https://github.com/ggml-org/llama.cpp/issues/12199
**State**: closed
**Created**: 2025-03-05T13:12:06+00:00
**Closed**: 2025-03-11T11:45:03+00:00
**Comments**: 2
**Labels**: good first issue, Apple Metal

### Description

It's likely that this should be addressed in ggml rather than llama

This is the observed call stack

```
llama_init_from_model 
  -> ggml_backend_dev_init 
    -> ggml_backend_metal_device_init 
      -> ggml_metal_init 
        -> device.newLibraryWithSource
```

(obviously in cases where the code is compiled such as with the default `GGML_METAL_EMBED_LIBRARY`)

For every context the exact same code is compiled again. This seems like something that can be avoided. I'm not a metal expert, but there must be a way to cache the compilation and reuse it for subsequent contexts.

---

## Issue #N/A: Bug:Why does llama-cli choose a GPU with lower performance?

**Link**: https://github.com/ggml-org/llama.cpp/issues/10009
**State**: closed
**Created**: 2024-10-23T01:39:51+00:00
**Closed**: 2024-12-07T01:07:32+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, stale, Apple Metal, medium severity

### Description

### What happened?

I have 2 GPUs in imc2017 RAM64G, one of which is connected through eGPU. Llama-cli->ggml->always don't use a GPU with higher performance? How can I use a higher GPU, or both?

### Name and Version

 METAL_DEVICE_WRAPPER_TYPE=0 (or 1 or 2 ) ./llama-cli
built with Apple clang version 15.0.0 (clang-1500.3.9.4) for x86_64-apple-darwin22.6.0


### What operating system are you seeing the problem on?

Mac

### Relevant log output

```shell
ggml_metal_init: allocating
ggml_metal_init: found device: AMD Radeon VII
METAL_DEVICE_WRAPPER_TYPE=0 ./llama-cli
build: 0 (unknown) with Apple clang version 15.0.0 (clang-1500.3.9.4) for x86_64-apple-darwin22.6.0
main: llama backend init
main: load the model and apply lora adapter, if any
llama_load_model_from_file: using device Metal (AMD Radeon Pro 570) - 4096 MiB free
llama_model_loader: loaded meta data with 39 key-value pairs and 963 tensors from models/7B/ggml-model-f16.gguf (version GGUF V3 (latest))
...
ggml_metal_in

[... truncated for brevity ...]

---

## Issue #N/A: metal : increase GPU duty-cycle during inference

**Link**: https://github.com/ggml-org/llama.cpp/issues/9507
**State**: closed
**Created**: 2024-09-16T12:14:00+00:00
**Closed**: 2024-10-01T13:00:26+00:00
**Comments**: 1
**Labels**: help wanted, performance, Apple Metal

### Description

Apparently there is a significant GPU downtime between Metal compute encoders within a single `ggml_metal_graph_compute()`: 

<img width="2672" alt="image" src="https://github.com/user-attachments/assets/e01b56a0-cdcf-4777-9944-be6e456858eb">

See https://github.com/ggerganov/llama.cpp/issues/6506 for instructions how to generate the trace from the picture.

My expectation was that enqueuing the command buffers in parallel would make them execute without any downtime. The goal of this issue is to understand where this overhead comes from and if there is a way to avoid it.

Obviously, using a single command buffer will avoid all the GPU downtime, but it is much slower to construct it in a single thread. Ideally, we want to continue queuing multiple encoders, but not have the gaps in-between during execution.

---

## Issue #N/A: Bug: Gemma 2 slower with FA

**Link**: https://github.com/ggml-org/llama.cpp/issues/9243
**State**: closed
**Created**: 2024-08-29T16:39:59+00:00
**Closed**: 2024-11-08T01:07:24+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale, Apple Metal, medium severity

### Description

### What happened?

Gemma 2 is slower with FA on Apple Silicon (M3 Max).

### Name and Version

version: 3642 (1d1ccce6)
built with Apple clang version 15.0.0 (clang-1500.3.9.4) for arm64-apple-darwin23.6.0

### What operating system are you seeing the problem on?

Mac

### Relevant log output

```shell
| model                          |       size |     params | backend    | ngl | fa | mmap |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | ---: | ------------: | ---------------: |
| gemma2 2B Q8_0                 |   3.17 GiB |     3.20 B | Metal      |  99 |  0 |    0 |         pp512 |   2360.42 ± 3.71 |
| gemma2 2B Q8_0                 |   3.17 GiB |     3.20 B | Metal      |  99 |  0 |    0 |          tg64 |     85.54 ± 0.05 |
| gemma2 2B Q8_0                 |   3.17 GiB |     3.20 B | Metal      |  99 |  1 |    0 |         pp512 |   1487.45 ± 3.27 |
| gemma2 2B Q8_0                 |   3.17 GiB |     3.

[... truncated for brevity ...]

---

## Issue #N/A: Bug: ld: symbol(s) not found for architecture arm64 

**Link**: https://github.com/ggml-org/llama.cpp/issues/8211
**State**: closed
**Created**: 2024-06-29T15:26:26+00:00
**Closed**: 2024-08-07T16:24:06+00:00
**Comments**: 27
**Labels**: bug, Apple Metal, high severity

### Description

### What happened?

symbol not found compile error for Mac metal build. If I wind back a week with "git reset --hard master@{"7 days ago"}" it builds and executes fine.

2023 M2 MBP

### Name and Version

Current master branch

### What operating system are you seeing the problem on?

Mac

### Relevant log output

```shell
`
c++ -std=c++11 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread   -Wunreachable-code-break -Wunreachable-code-return -Wmissing-prototypes -Wextra-semi -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_DARWIN_C_SOURCE -DNDEBUG -DGGML_USE_ACCELERATE -DGGML_USE_BLAS -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -DGGML_USE_LLAMAFILE -DGGML_USE_METAL -DGGML_METAL_EMBED_LIBRARY  ggml/src/ggml.o ggml/src/ggml-blas.o ggml/src/sgemm.o ggml/src/ggml-metal.o ggml/src/ggml-metal-embed.o ggml/src/ggml-alloc.o ggml/src/ggml-backend.o ggml/src/ggml-quants.o pocs/vdot/q8dot.o -o ll

[... truncated for brevity ...]

---

