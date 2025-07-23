# head_top5_labels - issues

**Total Issues**: 50
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 4
- Closed Issues: 46

### Label Distribution

- stale: 26 issues
- bug-unconfirmed: 22 issues
- enhancement: 13 issues
- bug: 12 issues
- medium severity: 11 issues
- low severity: 3 issues
- high severity: 2 issues
- model: 2 issues
- Nvidia GPU: 2 issues
- threading: 1 issues

---

## Issue #N/A: train-text-from-scratch.exe stop after "begin training" (tensor->src0 is null)

**Link**: https://github.com/ggml-org/llama.cpp/issues/1869
**State**: closed
**Created**: 2023-06-15T07:26:00+00:00
**Closed**: 2024-04-10T01:07:05+00:00
**Comments**: 9
**Labels**: stale

### Description

I'm running the latest release (master-254a7a7) like that:

`bin\train-text-from-scratch.exe --vocab-model models\ggml-vocab.bin --checkpoint-in chk-lamartine-256x16.bin --checkpoint-out chk-lamartine-256x16.bin --model-out ggml-lamartine-265x16-f32.bin --train-data "shakespeare.txt"           `
I tried with several models.

# Expected Behavior

Training shoud run for a long time

# Current Behavior

Training stop immediatly without error:

```
D:\git\llama.cpp>bin\train-text-from-scratch.exe --vocab-model models\ggml-vocab.bin --ctx 64 --embd 256 --head 8 --layer 16 --checkpoint-in chk-lamartine-256x16.bin --checkpoint-out chk-lamartine-256x16.bin --model-out ggml-lamartine-265x16-f32.bin --train-data "alphonsedelamartine.txt" -t 6 -b 1 -n 32 --seed 2 --adam-iter 16 --print-details-interval 0 --predict 16 --use-flash
main: seed: 2
llama.cpp: loading model from models\ggml-vocab.bin
llama_model_load_internal: format     = ggjt v1 (pre #1405)
llama_model_load_internal:

[... truncated for brevity ...]

---

## Issue #N/A: Add `completion` server parameters to `v1/chat/completions` 

**Link**: https://github.com/ggml-org/llama.cpp/issues/4429
**State**: closed
**Created**: 2023-12-12T17:32:11+00:00
**Closed**: 2024-04-03T01:14:16+00:00
**Comments**: 5
**Labels**: enhancement, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

The same set of parameters should be available when calling from either `completion` or `v1/chat/completions` endpoints. Most notably `min_p` and `grammar` are useful to have.

A call like this should be possible for example:

```bash
curl http://localhost:3077

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Throughput (tokens/sec) does not scale with increasing batch sizes in Intel GPUs

**Link**: https://github.com/ggml-org/llama.cpp/issues/9097
**State**: closed
**Created**: 2024-08-20T04:34:26+00:00
**Closed**: 2024-10-06T01:07:32+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

For Intel dGPU like ARC770, the tokens per second doesn't scale with increasing batch size. For example if tps for batch size 1 is ~x tps, for batch size 8 also throughput is ~x tps. 

### Name and Version

llama build: 2663 (7e54166)
OS ubuntu 22.04
command line used: ZES_ENABLE_SYSMAN=1 ./build/bin/main -m models/llama-2-7b.Q8_0.gguf -ngl 33 -mg 0 -b 1 -p "solve the 3 ants and triangel puzzle"
batch size changed for different execution.

### What operating system are you seeing the problem on?

Linux

### Relevant log output

_No response_

---

## Issue #N/A: [Enhancement] Simultaneous CLBLAS/CUBLAS instances. 

**Link**: https://github.com/ggml-org/llama.cpp/issues/1494
**State**: closed
**Created**: 2023-05-17T03:14:18+00:00
**Closed**: 2024-04-09T01:09:02+00:00
**Comments**: 5
**Labels**: stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ ]x I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Enhancement

If not already possible through a config I missed, would offloading some layers to CLBLAS and other layers to CUBLAS be viable? Or maybe offloading layers to multiple CLBLAS devices?

A common hardware config is a CPU with an IGP + discrete gpu, and this would allow t

[... truncated for brevity ...]

---

## Issue #N/A: Using #pragma once makes it difficult

**Link**: https://github.com/ggml-org/llama.cpp/issues/7076
**State**: closed
**Created**: 2024-05-04T15:30:23+00:00
**Closed**: 2024-06-19T01:06:39+00:00
**Comments**: 3
**Labels**: enhancement, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

It would be helpful to use
 ```
#ifndef GGML_H
#define GGML_H
...
#endif ```

Instead of "#pragma once"

I'm noticing if I include say llama.cpp and whisper.cpp as git submodules in my project then these #pragma once directive do not correctly avoid includi

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: Embeddings Always returned as non

**Link**: https://github.com/ggml-org/llama.cpp/issues/13854
**State**: closed
**Created**: 2025-05-28T11:48:26+00:00
**Closed**: 2025-07-12T01:08:20+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version
Mac os:
llama-cli --version
version: 5390 (aa48e373)
built with Apple clang version 16.0.0 (clang-1600.0.26.6) for arm64-apple-darwin23.6.0

Ubuntu os:
./llama-cli --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: Tesla T4, compute capability 7.5, VMM: yes
load_backend: loaded CUDA backend from /app/libggml-cuda.so
load_backend: loaded CPU backend from /app/libggml-cpu-haswell.so
version: 5332 (7c28a74e)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### Operating systems

Mac,Ubuntu

### GGML backends

Metal,Cuda

### Hardware

GPUs

### Models

all models

### Problem description & steps to reproduce

when i run llama-server with embeddings enabled i got null for all embeddings vectors, and when try to use the cli i got the same result


### First Bad Commit

_No response_

### Relevant log output

```shell
system_info: n_threads = 4 (n_threa

[... truncated for brevity ...]

---

## Issue #N/A: Multi-GPU support for AMD?

**Link**: https://github.com/ggml-org/llama.cpp/issues/3051
**State**: closed
**Created**: 2023-09-07T00:15:22+00:00
**Closed**: 2024-06-12T01:06:50+00:00
**Comments**: 28
**Labels**: stale

### Description

Do you have multi-GPU support for AMD, if not, do you see it as something you might add in the future?

---

## Issue #N/A: Bug: Issue building hipBLAS error: call to undeclared function '_mm256_dpbusd_epi32'

**Link**: https://github.com/ggml-org/llama.cpp/issues/9666
**State**: closed
**Created**: 2024-09-27T16:37:54+00:00
**Closed**: 2024-11-12T01:08:45+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

Hi, 

I'm trying to compile llama with the hipBLAS backed on 
CPU: 12th Gen Intel(R) Core(TM) i9-12900K
GPU: AMD Radeon PRO W7800 (gfx1100)
OS: Windows 11 23H2
With AMD HIP SDK 6.1.2 for Windows Installed:
https://www.amd.com/en/developer/resources/rocm-hub/eula/licenses.html?filename=AMD-Software-PRO-Edition-24.Q3-Win10-Win11-For-HIP.exe

llama.cpp version: https://github.com/ggerganov/llama.cpp/releases/tag/b3828

When I run these commands
```
set PATH=%HIP_PATH%\bin;%PATH%
cmake -S . -B build -G Ninja -DAMDGPU_TARGETS=gfx1100 -DGGML_HIPBLAS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

I get this error message, and the build is unable to continue:
```
llama.cpp-b3828/ggml/src/ggml-quants.c:107:34: error: call to undeclared function '_mm256_dpbusd_epi32'; ISO C99 and later do not support implicit function declarations [-Wimplicit-function-declaration]
    const __m256i summed_pai

[... truncated for brevity ...]

---

## Issue #N/A: Do llama.cpp support input_embeds?

**Link**: https://github.com/ggml-org/llama.cpp/issues/9630
**State**: closed
**Created**: 2024-09-24T14:53:16+00:00
**Closed**: 2024-11-09T01:07:02+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale, low severity

### Description

Do llama.cpp support input_embeds? Just like `transformers` support `input_embeds` in `model.generate` function.

---

## Issue #N/A: Feature Request: Tensor Parallelism support

**Link**: https://github.com/ggml-org/llama.cpp/issues/9086
**State**: closed
**Created**: 2024-08-19T01:38:13+00:00
**Closed**: 2024-12-13T01:07:40+00:00
**Comments**: 9
**Labels**: enhancement, threading, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Tensor parallelism is a a critical technique employed to train and inference from very large language models by splitting the actual computations/tensors across multiple compute devices. 

### Motivation

In our previous implementation on Xeon CPU, tensor parallelism(TP) can significantly reduce the latency on inference. <html xmlns:v="urn:schemas-microsoft-com:vml"
xmlns:o="urn:schemas-microsoft-com:office:office"
xmlns:x="urn:schemas-microsoft-com:office:excel"
xmlns="http://www

[... truncated for brevity ...]

---

## Issue #N/A: b2447 (c47cf41) decreased output quality

**Link**: https://github.com/ggml-org/llama.cpp/issues/6571
**State**: closed
**Created**: 2024-04-09T18:50:49+00:00
**Closed**: 2024-05-24T13:29:29+00:00
**Comments**: 17
**Labels**: need more info, bug-unconfirmed

### Description

With identical seeds and options, b2447 (https://github.com/ggerganov/llama.cpp/commit/c47cf414efafb8f60596edc7edb5a2d68065e992) produces different output that seems lower in quality compared to b2446. Is it possible to preserve old output quality in new builds?

System: MacBook Pro w/ i5-1038NG7

---

## Issue #N/A: make process hangs if LLAMA_CUBLAS=1, at the line that includes the file scripts/get-flags.mk for the second time

**Link**: https://github.com/ggml-org/llama.cpp/issues/4575
**State**: closed
**Created**: 2023-12-21T21:18:17+00:00
**Closed**: 2024-04-02T01:10:16+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

I run make LLAMA_CUBLAS=1 and that process hangs. I used make --debug=f to figure out that make gets stuck at the line that includes get-flags.mk for the second time (it is already included a few lines before). 

# Environment and Context

+-----------------------

[... truncated for brevity ...]

---

## Issue #N/A: Bug:  Recent changes break Rocm compile on windows

**Link**: https://github.com/ggml-org/llama.cpp/issues/8612
**State**: closed
**Created**: 2024-07-21T08:53:56+00:00
**Closed**: 2024-07-21T14:39:23+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

It cannot compile after CUDA: MMQ code deduplication + iquant support on windows.
that's my guess that pr break compile.

### Name and Version

b3428 and Windows 11 rocm5.7.1

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
cmake --build . -j 99 --parallel 32 --config Release
[1/61] Linking CXX shared library bin\ggml.dll
FAILED: bin/ggml.dll ggml/src/ggml.lib
cmd.exe /C "cmd.exe /C "C:\Strawberry\c\bin\cmake.exe -E __create_def W:\git\llama.cpp\rocm_1100\ggml\src\CMakeFiles\ggml.dir\.\exports.def W:\git\llama.cpp\rocm_1100\ggml\src\CMakeFiles\ggml.dir\.\exports.def.objs --nm=C:\Strawberry\c\bin\nm.exe && cd W:\git\llama.cpp\rocm_1100" && C:\PROGRA~1\AMD\ROCm\5.7\bin\CLANG_~1.EXE -fuse-ld=lld-link -nostartfiles -nostdlib -O3 -DNDEBUG -D_DLL -D_MT -Xclang --dependent-lib=msvcrt  -Xlinker /DEF:ggml\src\CMakeFiles\ggml.dir\.\exports.def -shared -o bin\ggml.dll  -Xlinker /MANIFEST:EMBED -Xlinker /implib:ggml

[... truncated for brevity ...]

---

## Issue #N/A: Server: Multimodal Model Input Parameter No longer Exists

**Link**: https://github.com/ggml-org/llama.cpp/issues/7112
**State**: closed
**Created**: 2024-05-07T04:11:00+00:00
**Closed**: 2024-07-18T01:06:49+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale

### Description

I have noticed that when using the server that the --mmproj parameter for multimodal models has been disabled. Although it still remains in the README. Is there an alternative to --mmproj , I cannot seem to find one in the code. 

Any help on this would be great. 

Code to reproduce:
`./server -m ./ggml-model-q4_k.gguf --mmproj ./mmproj-model-f16.gguf -ngl 1`

Error:
`
error: unknown argument: --mmproj
usage: ./server [options]
`

---

## Issue #N/A: Misc. bug: Potential out of bound in rerank

**Link**: https://github.com/ggml-org/llama.cpp/issues/13549
**State**: open
**Created**: 2025-05-14T19:50:38+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

version: 5387 (3198405e)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### Operating systems

_No response_

### Which llama.cpp modules do you know to be affected?

_No response_

### Command line

```shell

```

### Problem description & steps to reproduce

[llama_context](https://github.com/ggml-org/llama.cpp/blob/f5170c1d7a66222ca7c75d2022fec3ed87257e0b/src/llama-context.cpp#L807) resize the rerank output to size 1 while [here](https://github.com/ggml-org/llama.cpp/blob/017f10b5fa630a013ec4f9936e410a60d4f460d5/examples/embedding/embedding.cpp#L69) we still normalize it as if we have full embedding vector. I found this problem happened randomly in python binding but cannot reproduce it in cpp. Not sure if it is a bug in cpp side.

### First Bad Commit

_No response_

### Relevant log output

```shell

```

---

## Issue #N/A: Misc. bug: Retrieval sample not decoding token successfully

**Link**: https://github.com/ggml-org/llama.cpp/issues/13102
**State**: closed
**Created**: 2025-04-24T22:26:15+00:00
**Closed**: 2025-06-08T01:08:06+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

version: 5184 (87616f06)
built with MSVC 19.41.34120.0 for x64

### Operating systems

Mac, Windows

### Which llama.cpp modules do you know to be affected?

Other (Please specify in the next section)

### Command line

```shell
llama-retrieval.exe --context-file <any_text_file> --chunk-size 1 -c 512 -t 8 -m bge-large-en-v1.5-f32.gguf
```

### Problem description & steps to reproduce

The sample failed to decode any tokens created from the text embeddings.

It looks like  we need to skip the kv-cache logic to look for an unused slot when pooling is active (which is true for the above model).

The following IF in llama-context.cpp is removed, causing us to go into this logic to search for an unused slot and hit the decoding spew.

        // non-causal masks do not use the KV cache
        if (hparams.causal_attn) {
            kv_self_update();

Just adding "if (!embd_pooling)" appears to fix the issue but I am not sure what it does to the original logic for the n

[... truncated for brevity ...]

---

## Issue #N/A: Since last update Mistral models doesn't works anymore

**Link**: https://github.com/ggml-org/llama.cpp/issues/7450
**State**: closed
**Created**: 2024-05-22T03:53:03+00:00
**Closed**: 2024-05-22T10:03:22+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

since this https://github.com/ggerganov/llama.cpp/tree/b2961
phi3-128k works better (if ctx <32k)
but mistral models are crazy, I tried 7bQ2 7bQ8, 70BQ2XS, none of them works anymore

```
Log start
main
main: build = 2961 (201cc11a)
main: built with cc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0 for x86_64-linux-gnu
main: seed  = 1716349593
llama_model_loader
llama_model_loader: loaded meta data with 24 key-value pairs and 291 tensors from models/mistral-7b-instruct-v0.2.Q2_K.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = mistralai_mistral-7b-instruct-v0.2
llama_model_loader: - kv   2:                       llama.context_length u32              = 32768
llama_model_loader: - kv   3:                     llama.

[... truncated for brevity ...]

---

## Issue #N/A: ggml_validate_row_data finding nan value for IQ4_NL

**Link**: https://github.com/ggml-org/llama.cpp/issues/7311
**State**: closed
**Created**: 2024-05-15T19:40:10+00:00
**Closed**: 2024-05-18T00:39:55+00:00
**Comments**: 8
**Labels**: bug-unconfirmed

### Description

Using b2854

Converted Hermes-2-Theta-Llama-3-8B to F32, then measured imatrix with https://gist.github.com/bartowski1182/b6ac44691e994344625687afe3263b3a

Upon quanting, all sizes work fine, except for IQ4_NL which produces this output:

```
load_imatrix: imatrix dataset='/training_data/calibration_data.txt'
load_imatrix: loaded 224 importance matrix entries from /models/Hermes-2-Theta-Llama-3-8B-GGUF/Hermes-2-Theta-Llama-3-8B.imatrix computed on 189 chunks
prepare_imatrix: have 224 importance matrix entries
main: build = 2854 (72c177c1)
main: built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
main: quantizing '/models/Hermes-2-Theta-Llama-3-8B-GGUF/Hermes-2-Theta-Llama-3-8B-f32.gguf' to '/models/Hermes-2-Theta-Llama-3-8B-GGUF/Hermes-2-Theta-Llama-3-8B-IQ4_NL.gguf' as IQ4_NL
llama_model_loader: loaded meta data with 23 key-value pairs and 291 tensors from /models/Hermes-2-Theta-Llama-3-8B-GGUF/Hermes-2-Theta-Llama-3-8B-f32.gguf (version GGUF V3 (late

[... truncated for brevity ...]

---

## Issue #N/A: Compile bug: tools build failing

**Link**: https://github.com/ggml-org/llama.cpp/issues/13614
**State**: closed
**Created**: 2025-05-18T14:26:43+00:00
**Closed**: 2025-07-02T01:07:53+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Git commit

Commit 6a2bc8b


### Operating systems

Windows

### GGML backends

CUDA

### Problem description & steps to reproduce

Environment: Windows 11 with CUDA toolkit installed.

Sorry, new to this. I tried searching if there was already a solution but couldn't find anything with my limited domain of knowledge.

I followed the guide to build llama.cpp with CUDA support which seems to worked as it built a few binaries that I can see in the bin/Release folder, but I noticed none of the tools were built. I.g. cli, server etc...

Also, my environment was missing CURL libraries, so I had to look it up and install a windows version. And issued the following to build this:

```
cmake -B build -DGGML_CUDA=ON -DCURL_LIBRARY=c:\Curl\lib\libcurl.a -DCURL_INCLUDE_DIR=c:\Curl\include
```

Reading up on the llama-server docs, I saw there was a way to build it so I tried it but I got this error:
```
common.lib(arg.obj) : error LNK2019: unresolved external symbol __imp_curl_slist_appe
nd re

[... truncated for brevity ...]

---

## Issue #N/A: Segmentation fault during inference on AMD gfx900 with codebooga-34b-v0.1.Q5_K_M.gguf

**Link**: https://github.com/ggml-org/llama.cpp/issues/6031
**State**: closed
**Created**: 2024-03-13T01:47:52+00:00
**Closed**: 2024-03-14T18:46:32+00:00
**Comments**: 15
**Labels**: bug-unconfirmed

### Description

Hi,

I compiled `llama.cpp` from git, todays master HEAD `commit 8030da7afea2d89f997aeadbd14183d399a017b9` on Fedora Rawhide (ROCm 6.0.x) like this:
```
CC=/usr/bin/clang CXX=/usr/bin/clang++ cmake .. -DLLAMA_HIPBLAS=ON -DAMDGPU_TARGETS=gfx900 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="--rocm-device-lib-path=/usr/lib/clang/17/amdgcn/bitcode"
make -j 16
```

Then I tried to run a prompt using the `codebooga-34b-v0.1.Q5_K_M.gguf` model which I got from here: https://huggingface.co/TheBloke/CodeBooga-34B-v0.1-GGUF

I kept the prompt simple and used the following command:
./main -t 10 -ngl 16 -m ~/models/codebooga-34b-v0.1.Q5_K_M.gguf --color -c 2048 --temp 0.7 --repeat_penalty 1.1 -n -1 -p "### Instruction: How do I get the length of a Vec in Rust?\n### Response:"

I have an AMD Instinct MI25 card with 16GB VRAM, according to `nvtop` with `-ngl 16` about half of it is used `8.219Gi/15.984`, so this does not seem to be an OOM issue.

The console output looks like this:
`

[... truncated for brevity ...]

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

## Issue #N/A: Bug: cannot find tokenizer merges in model file

**Link**: https://github.com/ggml-org/llama.cpp/issues/9692
**State**: closed
**Created**: 2024-09-30T02:31:24+00:00
**Closed**: 2024-10-08T03:14:42+00:00
**Comments**: 11
**Labels**: bug, high priority, high severity

### Description

### What happened?

When I use transformers==4.45.1 and convert llama.cpp to the file used by ollama, there is no error, but when I load the model with ollama, the error ollama cannot find tokenizer merges in model file appears

### Name and Version

所有版本

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

---

## Issue #N/A: Quantize python script fails.

**Link**: https://github.com/ggml-org/llama.cpp/issues/431
**State**: closed
**Created**: 2023-03-23T15:15:24+00:00
**Closed**: 2023-03-23T20:42:54+00:00
**Comments**: 5
**Labels**: bug

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ ] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ ] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

I have my llama models stored in models/llama/{7B,13B,30B,65B}.

I expect that when I run the following command that the model will be converted

$ python3 quantize.py --models-path models/llama 30B


# Current Behavior

When attempting to quantize the model 

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: llama-server ignores the stop parameter

**Link**: https://github.com/ggml-org/llama.cpp/issues/11538
**State**: closed
**Created**: 2025-01-31T10:34:03+00:00
**Closed**: 2025-01-31T13:48:33+00:00
**Comments**: 1
**Labels**: bug

### Description

### Name and Version

version: 4599 (8b576b6c)
built with cc (Debian 12.2.0-14) 12.2.0 for x86_64-linux-gnu

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell
curl --request POST --url http://localhost:8080/completion --header "Content-Type: application/json"  --data '{"prompt": "A B C D E F G H I J K","n_predict": 128, "stop": ["O P Q"]}'
```
Notice that the stop string spawns multiple tokens.


### Problem description & steps to reproduce

The server `/completion` endpoint ignores the `stop` parameter.

Tested by loading phi4 in llama-server, then sending a request with a array of stop tokens including a triple backquote: [stop1, stop2, "```", stop3,...]

### example
`curl --request POST --url http://localhost:8080/completion --header "Content-Type: application/json"  --data '{"prompt": "A B C D E F G H I J K","n_predict": 128, "stop": ["O P Q"]}'`

note that the stop string spans multiple tokens


The offe

[... truncated for brevity ...]

---

## Issue #N/A: train-text-from-scratch and finetune nan loss on iter=2

**Link**: https://github.com/ggml-org/llama.cpp/issues/3940
**State**: closed
**Created**: 2023-11-04T04:42:06+00:00
**Closed**: 2023-11-07T08:04:52+00:00
**Comments**: 2
**Labels**: bug

### Description

I was trying out the finetune example with my model but it kept going into nan loss. I eventually tried train-text-from-scratch, following the instructions on the README there and it goes into nan as well. I've reproduced this on two machines.

```
root@c5a10438d69e:/workspace/llama.cpp# ./train-text-from-scratch         --vocab-model ./models/ggml-vocab-llama.gguf         --ctx 64 --embd 256 --head 8 --layer 16         --checkpoint-in  chk-shakespeare-256x16-LATEST.gguf         --checkpoint-out chk-shakespeare-256x16-ITERATION.gguf         --model-out ggml-shakespeare-256x16-f32-ITERATION.gguf         --train-data "shakespeare.txt"         -t 6 -b 16 --seed 1 --adam-iter 256         --no-checkpointing
main: seed: 1
llama_model_loader: loaded meta data with 17 key-value pairs and 0 tensors from ./models/ggml-vocab-llama.gguf (version GGUF V3 (latest))
llama_model_loader: - kv   0:                       general.architecture str     
llama_model_loader: - kv   1:                  

[... truncated for brevity ...]

---

## Issue #N/A: Constrained decoding with grammar fails for c4ai-command-r-v01

**Link**: https://github.com/ggml-org/llama.cpp/issues/6112
**State**: closed
**Created**: 2024-03-17T14:51:01+00:00
**Closed**: 2024-05-28T10:55:36+00:00
**Comments**: 6
**Labels**: bug, help wanted

### Description

I am trying to apply constrained decoding for the recently adopted command-r. 

Using the most recent master branch (https://github.com/ggerganov/llama.cpp/commit/c47cf414efafb8f60596edc7edb5a2d68065e992) I'm trying to apply the simplest list.  

`./main -m ~/data/c4ai-command-r-v01/ggml-model-Q4_K_M.gguf -p "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Please give me a list of things to do in SF?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>" -ctk q8_0 -ngl 99 -n 500 --grammar-file grammars/list.gbnf`

It fails with 

`libc++abi: terminating due to uncaught exception of type std::out_of_range: unordered_map::at: key not found`

Any idea what could go wrong here?

More details:

```
Log start
main: build = 2447 (c47cf414)
main: built with Apple clang version 15.0.0 (clang-1500.3.9.4) for arm64-apple-darwin23.3.0
main: seed  = 1710686911
llama_model_loader: loaded meta data with 23 key-value pairs and 322 tensors from ~/data/c4ai-command-r-v01/ggml-m

[... truncated for brevity ...]

---

## Issue #N/A: Significantly different results (and WRONG) inference when GPU is enabled.

**Link**: https://github.com/ggml-org/llama.cpp/issues/7048
**State**: closed
**Created**: 2024-05-02T18:51:50+00:00
**Closed**: 2024-05-17T18:49:39+00:00
**Comments**: 40
**Labels**: bug, Nvidia GPU

### Description

I am running llama_cpp version 0.2.68 on Ubuntu 22.04LTS under conda environment. Attached are two Jupyter notebooks with ONLY one line changed (use CPU vs GPU).  As you can see for exact same environmental conditions switching between CPU/GPU gives vastly different answers where the GPU is completely wrong.  Some pointers on how to debug this I would appreciate it.

The only significant difference between the two files is this one liner
      `#n_gpu_layers=-1, # Uncomment to use GPU acceleration`

The model used was **openhermes-2.5-mistral-7b.Q5_K_M.gguf**

[mistral_llama_large-gpu.pdf](https://github.com/ggerganov/llama.cpp/files/15192723/mistral_llama_large-gpu.pdf)
[mistral_llama_large-cpu.pdf](https://github.com/ggerganov/llama.cpp/files/15192725/mistral_llama_large-cpu.pdf)



---

## Issue #N/A: Incoherent output after merging https://github.com/ggerganov/llama.cpp/pull/2183

**Link**: https://github.com/ggml-org/llama.cpp/issues/2187
**State**: closed
**Created**: 2023-07-12T04:57:11+00:00
**Closed**: 2023-07-14T18:51:46+00:00
**Comments**: 7
**Labels**: bug

### Description

The commit in question seems to be https://github.com/ggerganov/llama.cpp/commit/20d7740a9b45f6e5b247fa3738fdda35e18c2e8a 

The AI responses no longer seem to consider the prompt after this commit.

Running pre-built cuda executables from github actions:

**llama-master-20d7740-bin-win-cublas-cu11.7.1-x64**
```
PS E:\LLaMA\llamacpp> .\main.exe --model e:\LLaMA\models\airoboros-7b-gpt4.ggmlv3.q4_0.bin -ngl 32 -n 30 -p "Hi, my name is"
main: build = 820 (20d7740)
main: seed  = 1689137712
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 2060, compute capability 7.5
llama.cpp: loading model from e:\LLaMA\models\airoboros-7b-gpt4.ggmlv3.q4_0.bin
llama_model_load_internal: format     = ggjt v3 (latest)
llama_model_load_internal: n_vocab    = 32000
llama_model_load_internal: n_ctx      = 512
llama_model_load_internal: n_embd     = 4096
llama_model_load_internal: n_mult     = 256
llama_model_load_internal: n_head     = 32
llama_model_load_internal: n_l

[... truncated for brevity ...]

---

## Issue #N/A: CUDA graphs break quantized K cache

**Link**: https://github.com/ggml-org/llama.cpp/issues/7492
**State**: closed
**Created**: 2024-05-23T12:11:15+00:00
**Closed**: 2024-05-27T17:33:43+00:00
**Comments**: 5
**Labels**: bug, Nvidia GPU

### Description

As of right now it is already possible on master to quantize the K cache via e.g. `-ctk q8_0`. However, this is currently broken on master for batch size 1. Disabling CUDA graphs via the environment variable `GGML_CUDA_DISABLE_GRAPHS=1` fixes the issue.

cc: @agray3 

---

## Issue #N/A: Eval bug: llama.cpp/ggml/src/ggml-backend.cpp:750: pre-allocated tensor (cache_k_l32 (view) (copy of cache_k_l32 (view))) in a buffer (Vulkan0) that cannot run the operation (CPY)

**Link**: https://github.com/ggml-org/llama.cpp/issues/13684
**State**: closed
**Created**: 2025-05-21T12:30:02+00:00
**Closed**: 2025-05-23T04:45:03+00:00
**Comments**: 6
**Labels**: bug, Vulkan

### Description

### Name and Version

$ sources/llama.cpp/build/bin/llama-server --version
version: 5435 (a4090d11)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu

### Operating systems

Linux

### GGML backends

Vulkan

### Hardware

Intel(R) Core(TM) Ultra 5 245KF + Radeon RX 7900 XTX, gfx1100 (0x1100)

### Models

gemma-3-27b-it-qat-UD-Q4_K_XL.gguf + gemma-3-27b-it-qat-GGUF/mmproj-F16.gguf

### Problem description & steps to reproduce

Built with (corrected command):
`cmake -S . -B build -DGGML_VULKAN=1 -DCMAKE_BUILD_TYPE=Release && cmake --build build -- -j 16`

Command used to start:
`sources/llama.cpp/build/bin/llama-server --port 9001 -c 65536 -ctv q8_0 -ctk q8_0 --no-warmup -ngl 99 -fa -m models/unsloth/gemma-3-27b-it-qat-GGUF/gemma-3-27b-it-qat-UD-Q4_K_XL.gguf --mmproj models/unsloth/gemma-3-27b-it-qat-GGUF/mmproj-F16.gguf --jinja`

This is the first time I used this model. I accessed the API via openwebui hosted in a docker container. Normal text only chat, nothing 

[... truncated for brevity ...]

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

## Issue #N/A: Bug: issue in CUDA flash attention

**Link**: https://github.com/ggml-org/llama.cpp/issues/10031
**State**: closed
**Created**: 2024-10-24T08:08:06+00:00
**Closed**: 2024-10-24T11:11:31+00:00
**Comments**: 7
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

There seems to be some memory corruption issue in the CUDA flash attention kernel. 

This is demonstrated by the debug prints inserted before and after the `vec_dot_KQ` device function here:
https://github.com/ggerganov/llama.cpp/compare/master...agray3:llama.cpp:ag_demonstrate_fattn_memory_issue

These print out the first element of Q_ds, which is const in the function so shouldn't be altered. (`Q_ds` is in local memory so it also shouldn't be altered by any other thread.)

However we get the result: 
Before vec_dot_KQ: Q_ds=-32752.000000
After vec_dot_KQ: Q_ds=nan

Q_ds is being altered and becoming NAN. This is reproducible across different GPUs and models. 

### Name and Version

version: 3964 (3488adf3)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

_No response_

---

## Issue #N/A: Llama-Quantize : Layers quantized in the wrong order, thus damaging the variable bits tensor quants scheme consistency.

**Link**: https://github.com/ggml-org/llama.cpp/issues/9005
**State**: closed
**Created**: 2024-08-12T12:59:04+00:00
**Closed**: 2024-09-27T01:07:21+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

On master b3573, when quantizing Gemma 9b it:

The tensors are quantized in a wrong order.

Right now, because of the layer jump from 7 to 10 without the ffns of layer 7 to be quantized, it breaks not only the layer quantization order, but also the correlation between ffn_down Q6_K and attn_v Q6_K : From layer 7, some layers will have ffn_down Q6_K and attn_v Q5_K, and some others ffn_down Q5_K and attn_v Q6_K.
This gives us suboptimal quants per BPW.

I expect the tensors to be quantized in the right order.

This, so the Q5_K_M quant, as well as the othersusing "use_more_bits(i_layer, n_layer)" to have a variable quant of ffn_down in conjunction with "use_more_bits(qs.i_attention_wv, qs.n_attention_wv))" to have a variable quant of attn_v.weight, can be optimal.

### Name and Version

main: build = 3573 (2589292c)
main: built with MSVC 19.29.30154.0 for x64

### What operating system are you seeing the problem on?

Windows

### Relevant log ou

[... truncated for brevity ...]

---

## Issue #N/A: Unable to convert a fireworks ai model to GGUF with gguf-my-repo

**Link**: https://github.com/ggml-org/llama.cpp/issues/8451
**State**: closed
**Created**: 2024-07-12T09:14:13+00:00
**Closed**: 2024-07-23T10:53:50+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

I  downloaded one of my models from fireworks.ai and pushed it up into huggingface - you can find it here: [llama-3-8b-instruct-danish](https://huggingface.co/HeRksTAn/llama-3-8B-Instruct-Danish)

I then tried  [gguf-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo) in order to convert it to gguf. 


1. using https://huggingface.co/spaces/ggml-org/gguf-my-repo 
2. logged into my account
3. search the hub id for the repository I want converted into gguf, which is HeRksTAn/llama-3-8B-Instruct-Danish
4. I chose Q4_K_M
5. I clicked submit


I get the following error 

`Error: Error converting to fp16: b'INFO:hf-to-gguf:Loading model: llama-3-8B-Instruct-Danish\nTraceback (most recent call last):\n File "/home/user/app/llama.cpp/convert_hf_to_gguf.py", line 3551, in \n main()\n File "/home/user/app/llama.cpp/convert_hf_to_gguf.py", line 3517, in main\n hparams = Model.load_hparams(dir_model)\n File "/home/user/app/llama.cpp/convert_hf_to_gguf.

[... truncated for brevity ...]

---

## Issue #N/A: Bug: src/llama.cpp:15099: Deepseek2 does not support K-shift

**Link**: https://github.com/ggml-org/llama.cpp/issues/8862
**State**: closed
**Created**: 2024-08-05T04:14:23+00:00
**Closed**: 2024-10-30T01:19:54+00:00
**Comments**: 10
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

Hi, when stress testing llama-server (--parallel 3, prompt="Count 1 to 10000 in words") and running deepseek-coder-v2:16b-lite-instruct-q8_0  i got this assertion error in the logs and everything stopped working, so i have to restart llm-server.

**Startup script:**

~/llama.cpp/llama-server -m /usr/share/ollama/.ollama/models/blobs/sha256-373dcfc92e01372709b6164fc836f677a6280e25e9eac5c434c64223207bfc4f --port 8000 --host 0.0.0.0 -ngl 28 -c 24600 --threads 16 --parallel 3 --log-format text --predict -2 --logdir ~/llama.cpp/logs --log-append   $1 $2 >> ~/llama.cpp/logs/deepseek.log 2>&1





### Name and Version

version: 3509 (ecf6b7f2)
built with cc (GCC) 8.5.0 20210514 (Red Hat 8.5.0-22.0.1) for x86_64-redhat-linux


### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
**Logs just before it crashed:**

INFO [   launch_slot_with_task] slot is processing task | tid="139873529581568" timestamp=17228302

[... truncated for brevity ...]

---

## Issue #N/A: Qwen2-57B-A14B-Instruct not supported

**Link**: https://github.com/ggml-org/llama.cpp/issues/7813
**State**: closed
**Created**: 2024-06-07T08:47:04+00:00
**Closed**: 2024-06-07T10:00:28+00:00
**Comments**: 13
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

The converted model fails to load due to unexpected expert tensor dimensions, the current qwen2moe implementation expects it to be `n_ff`/`n_expert_used`, which it is not.

### Name and Version

./main --version
version: 3066 (e141ce62)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
llama_model_load: error loading model: check_tensor_dims: tensor 'blk.0.ffn_gate_exps.weight' has wrong shape; expected  3584,  2368,    64, got  3584,  2560,    64,     1
```


---

## Issue #N/A: Bug: llama-server + LLava 1.6 hallucinates

**Link**: https://github.com/ggml-org/llama.cpp/issues/8001
**State**: closed
**Created**: 2024-06-19T05:40:15+00:00
**Closed**: 2024-08-03T01:18:10+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

When using `./llama-llava-cli `, I get perfectly fine descriptions of images. But when hosting LLava with `./llama-server`, LLava hallucinates big time. 

Here's how I'm running LLava with the cli:
`./llama-llava-cli -m models/llava-v1.6-vicuna-7b.Q5_K_S.gguf --mmproj models/mmproj-model-f16.gguf --image images/sth.jpeg -c 4096`

Here's how I'm starting the server:
` ./llama-server -m models/llava-v1.6-vicuna-7b.Q5_K_S.gguf --mmproj models/mmproj-model-f16.gguf -c 2048  --host 127.0.0.1 --port 8000`

Here's the python code to send the request:
```
import requests
import base64

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image("./images/sth.png")
      
headers = {
    'Content-Type': 'application/json',
}

json_data = {
    'image_data': [{
        'data': base64_image, 
        'id': 10
    }],
    "prompt": "USER:[img-

[... truncated for brevity ...]

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

## Issue #N/A: Bug: ggml/src/ggml.c: In function 'ggml_vec_mad_f16':

**Link**: https://github.com/ggml-org/llama.cpp/issues/8378
**State**: closed
**Created**: 2024-07-08T21:20:30+00:00
**Closed**: 2024-08-26T01:07:03+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?
`GGML_CUDA=1 make -j`
...
```
ggml/src/ggml.c: In function 'ggml_vec_mad_f16':
ggml/src/ggml.c:2039:45: warning: passing argument 1 of '__sse_f16x4_load' discards 'const' qualifier from pointer target type [-Wdiscarded-qualifiers]
 2039 |             ax[j] = GGML_F16_VEC_LOAD(x + i + j*GGML_F16_EPR, j);
      |                                             ^
ggml/src/ggml.c:1491:50: note: in definition of macro 'GGML_F32Cx4_LOAD'
 1491 | #define GGML_F32Cx4_LOAD(x)     __sse_f16x4_load(x)
      |                                                  ^
ggml/src/ggml.c:2039:21: note: in expansion of macro 'GGML_F16_VEC_LOAD'
 2039 |             ax[j] = GGML_F16_VEC_LOAD(x + i + j*GGML_F16_EPR, j);
      |                     ^~~~~~~~~~~~~~~~~
ggml/src/ggml.c:1466:52: note: expected 'ggml_fp16_t *' {aka 'short unsigned int *'} but argument is of type 'const ggml_fp16_t *' {aka 'const short unsigned int *'}
 1466 | static inline __m128 __sse_f16x4_load(ggml_fp16_

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Phi-3 4K output broken after 2000~ tokens (Reproducible)

**Link**: https://github.com/ggml-org/llama.cpp/issues/7709
**State**: closed
**Created**: 2024-06-03T07:25:37+00:00
**Closed**: 2024-12-27T12:33:27+00:00
**Comments**: 13
**Labels**: bug, model, medium severity

### Description

### What happened?

To reproduce:
Download the official released gguf model from huggingface/microsoft.
Run **server.exe -m Phi3-mini-4k.gguf -c 4096**

When input prompt < ~2048: Output fine. (but output starts getting weird right after it hits ~2048 in total)
When input prompt > ~2048: Output weird.

The weird output seems like what we expect to see when the context is more than the model support, but happens in ~2048, which seems like there are some bugs.

Also tested Llama3-8B, works fine with input prompt < 8192 as expected (with -c 8192), also works fine with input prompt < 4096 as expected (with -c 4096).

### Name and Version

version: 3015 (74b239b3)
built with MSVC 19.39.33523.0 for x64

Tried both cuda and avx2 version.

Also tried latest version built it myself @ Intel SYCL
version: 3075 (3d7ebf63)
built with IntelLLVM 2024.1.0

### What operating system are you seeing the problem on?

Win10, Win11

### Relevant log output

Before ~2000 tokens 

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Gemma 2 incoherent output when using quantized k cache without Flash Attention

**Link**: https://github.com/ggml-org/llama.cpp/issues/8853
**State**: closed
**Created**: 2024-08-04T10:57:51+00:00
**Closed**: 2024-09-18T01:07:04+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

Output like "Mh giàu され rodas reliablyacheteurδε Są" happens when using quantized K cache, CUDA, with Gemma 2. Here's how to reproduce:

./llama-server -m "Gemma-2-9B-It-SPPO-Iter3-Q4_K_S.gguf" -t 6 -c 8192 -ngl 31 -ctk q4_0 --host 127.0.0.1 --port 8080

Then connect a frontend like SillyTavern to it. Strangely this only happens with server, not with main-cli. 

This leads to incoherent output. Note: I can't say if this issue happens when using full offloading, as I just have 6 GB VRAM. 


### Name and Version

 ./llama-cli --version
version: 3506 (76614f35)
built with MSVC 19.29.30154.0 for x64

### What operating system are you seeing the problem on?

Windows

### Relevant log output

_No response_

---

