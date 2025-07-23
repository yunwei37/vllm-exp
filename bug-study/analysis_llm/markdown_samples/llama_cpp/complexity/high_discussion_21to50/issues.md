# high_discussion_21to50 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- bug-unconfirmed: 13 issues
- stale: 11 issues
- model: 5 issues
- enhancement: 4 issues
- good first issue: 4 issues
- high severity: 2 issues
- performance: 2 issues
- generation quality: 1 issues
- hardware: 1 issues
- build: 1 issues

---

## Issue #N/A: Eval bug: Several models producing gibberish

**Link**: https://github.com/ggml-org/llama.cpp/issues/12012
**State**: closed
**Created**: 2025-02-21T20:52:18+00:00
**Closed**: 2025-02-26T00:59:59+00:00
**Comments**: 21
**Labels**: bug-unconfirmed

### Description

### Name and Version

[root@localhost ~]# ~/llama.cpp/build/bin/llama-cli --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 ROCm devices:
  Device 0: AMD Radeon VII, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
  Device 1: AMD Radeon VII, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
register_backend: registered backend ROCm (2 devices)
register_device: registered device ROCm0 (AMD Radeon VII)
register_device: registered device ROCm1 (AMD Radeon VII)
register_backend: registered backend CPU (1 devices)
register_device: registered device CPU (Intel(R) Celeron(R) CPU G3930 @ 2.90GHz)
load_backend: failed to find ggml_backend_init in /root/llama.cpp/build/bin/libggml-hip.so
load_backend: failed to find ggml_backend_init in /root/llama.cpp/build/bin/libggml-cpu.so
version: 4753 (51f311e0)
built with cc (GCC) 8.5.0 20210514 (Red Hat 8.5.0-23) for x86_64-redhat-linux

### Operating systems

Mac, Linux

#

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Build failure in master on Ubuntu 24.04 with CUDA enabled

**Link**: https://github.com/ggml-org/llama.cpp/issues/9473
**State**: closed
**Created**: 2024-09-13T15:35:09+00:00
**Closed**: 2024-09-16T14:22:09+00:00
**Comments**: 23
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

Build failure starting ~Sep 11/12.

I run fresh builds periodically - about once every 1-2 days and this started recently. Build command:
make GGML_CUDA=1 -j 16

### Name and Version

Environment is Ubuntu 24.04 updated as of submission.

commit feff4aa8461da7c432d144c11da4802e41fef3cf (HEAD -> master, tag: b3751, origin/master, origin/HEAD)
gcc (Ubuntu 13.2.0-23ubuntu4) 13.2.0

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Aug_14_10:10:22_PDT_2024
Cuda compilation tools, release 12.6, V12.6.68
Build cuda_12.6.r12.6/compiler.34714021_0

cuda-toolkit-12-6 is already the newest version (12.6.1-1).


### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
nvcc -std=c++11 -O3 -g -use_fast_math --forward-unknown-to-host-compiler -arch=native -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_MMV_Y=1 -DK_QUANTS_PER_ITERATION=2 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128  -Iggml/include -Iggml

[... truncated for brevity ...]

---

## Issue #N/A: Running a Vicuna-13B 4it model ?

**Link**: https://github.com/ggml-org/llama.cpp/issues/771
**State**: closed
**Created**: 2023-04-05T07:33:04+00:00
**Closed**: 2023-07-28T19:47:57+00:00
**Comments**: 25
**Labels**: model, generation quality

### Description

I found this model : 
[[ggml-vicuna-13b-4bit](https://huggingface.co/eachadea/ggml-vicuna-13b-4bit)](https://huggingface.co/eachadea/ggml-vicuna-13b-4bit/tree/main) and judging by their online demo it's very impressive.
I tried to run it with llama.cpp latest version - the model loads fine, but as soon as it loads it starts hallucinating and quits by itself. 
Do I need to have it converted or something like that ?

---

## Issue #N/A: llama: add Grok support

**Link**: https://github.com/ggml-org/llama.cpp/issues/6120
**State**: closed
**Created**: 2024-03-17T20:31:28+00:00
**Closed**: 2024-05-08T17:05:36+00:00
**Comments**: 21
**Labels**: enhancement, good first issue, model

### Description

Hi,
Please add support for Grok.
Thanks!

Relevant links:
* https://github.com/xai-org/grok
* https://x.ai/blog/grok-os
* https://twitter.com/grok/status/1769441648910479423
* [NEW] Official Upload (thx to @dranger003) for linking: https://huggingface.co/xai-org/grok-1

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

## Issue #N/A: Eval bug: Excessive stack usage during tool calling

**Link**: https://github.com/ggml-org/llama.cpp/issues/12234
**State**: closed
**Created**: 2025-03-06T21:08:46+00:00
**Closed**: 2025-05-02T01:07:57+00:00
**Comments**: 23
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

./llama-cli --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070 Laptop GPU, compute capability 8.9, VMM: yes
version: 4840 (3ffbbd5c)
built with Ubuntu clang version 18.1.8 (++20240731024944+3b5b5c1ec4a3-1~exp1~20240731145000.144) for x86_64-pc-linux-gnu

### Operating systems

Linux

### GGML backends

CUDA

### Hardware

i9-13900HX + NVIDIA GeForce RTX 4070



### Models

[bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/blob/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf)

### Problem description & steps to reproduce

cc/@ochafik

I am attempting to run BFCL on llama-server, and so far I have triggered a crash twice.  It does not appear to be deterministic, unfortunately.  In one instance, I was able to catch the crash with gdb.  Here is the end of the backtrace:

```
#87097 0x00005669dac2b7f9 in bool st

[... truncated for brevity ...]

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

## Issue #N/A: Add support to ArcticForCausalLM

**Link**: https://github.com/ggml-org/llama.cpp/issues/6877
**State**: closed
**Created**: 2024-04-24T14:45:07+00:00
**Closed**: 2024-05-24T12:31:15+00:00
**Comments**: 36
**Labels**: enhancement

### Description

First open LLM from [@SnowflakeDB](https://twitter.com/SnowflakeDB)! Arctic is 480B Dense-MoE with a 10B dense transformer model and a 128x3.66B MoE MLP designed specifically for enterprise AI. 🤔

TL;DR:
🧠 480B parameters with 17B active during generation
👨‍🏫  128 experts with 2 active in generation
2️⃣ Instruct & Base versions released
🏙️ Focused on Enterprise task (Code, SQL, Reasoning, Following)
🔓 Released under Apache 2.0
🗻 in fp16 ~900GB Memory & in int4 ~240GB
🤗 Available on [@huggingface](https://twitter.com/huggingface)

🏋🏻 Trained with DeepSpeed-MoE


Blog: [https://snowflake.com/blog/arctic-open-efficient-foundation-language-models-snowflake/](https://t.co/RAgYE44tBA)

Models: [https://huggingface.co/Snowflake/snowflake-arctic-instruct](https://t.co/Mdd9XfAKfe)

---

## Issue #N/A: examples for iOS (objc / swift ui)

**Link**: https://github.com/ggml-org/llama.cpp/issues/3284
**State**: closed
**Created**: 2023-09-20T20:48:39+00:00
**Closed**: 2023-11-28T16:21:00+00:00
**Comments**: 22

### Description

I really enjoyed the examples for running whisper.cpp on iOS using both objective-c and swift-ui (found at https://github.com/ggerganov/whisper.cpp/tree/master/examples/whisper.objc and https://github.com/ggerganov/whisper.cpp/tree/master/examples/whisper.swiftui respectively) and was wondering if the process can be recreated for this repository. I believe that having a minimal example repository would be useful. 

I'd be willing to make an attempt, but I need to familiarize myself with the process that was performed in the whisper.cpp repository. 

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

## Issue #N/A: Will llama.cpp be able to use Phi-2 ?

**Link**: https://github.com/ggml-org/llama.cpp/issues/4437
**State**: closed
**Created**: 2023-12-13T12:02:56+00:00
**Closed**: 2023-12-18T17:27:49+00:00
**Comments**: 27
**Labels**: enhancement, good first issue, model

### Description

Surely we have to wait for a GGUF version, but in the meantime just curious about it

thanks

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

## Issue #N/A: Is it normal that ROCm+HIPBLAS produces different results than on CPU or breaks completely?

**Link**: https://github.com/ggml-org/llama.cpp/issues/6841
**State**: closed
**Created**: 2024-04-23T11:12:58+00:00
**Closed**: 2024-04-26T16:39:59+00:00
**Comments**: 33
**Labels**: bug-unconfirmed

### Description

Hello. I did some perplexity tests while investigating issue with Llama-3. Initially the issue was that Llama-3 base 70b model outputs garbage with small quants with iMatrix. Can't find any information regarding that ROCm possibly causes corruption.

### GPU test (RX 7600 + RX 7600 XT)
https://huggingface.co/mradermacher/Meta-Llama-3-70B-i1-GGUF/tree/main
Meta-Llama-3-70B.i1-Q2_K.gguf prints [1]-nan,[2]-nan,[3]-nan,[4]-nan with -ngl 30 or 0 (prints garbage unless -ngl 0)
https://huggingface.co/mradermacher/Meta-Llama-3-70B-GGUF/tree/main
Meta-Llama-3-70B.Q2_K.gguf - seems OK, [1]4.1839,[2]4.7300,[3]4.2751,[4]4.6444,[5]4.6942,[6]5.0426,[7]5.1405,[8]5.4747
Final estimate: PPL = 5.9315 +/- 0.03553

### Pure CPU test
Meta-Llama-3-70B.i1-Q2_K.gguf with pure CPU 'perplexity' build (146 seconds per 512 tokens - ETA 26 hours 55.67 minutes)
[1]6.3962,[2]7.1886,[3]6.9886,[4]7.3853,[5]7.8924,[6]8.2982,[7]8.8956,[8]9.3799, (can't wait for many hours, stopped)
Meta-Llama-3-70B.Q2_K.gguf

[... truncated for brevity ...]

---

## Issue #N/A: Implement Together Computer's Red Pajama 3B Base/Chat model

**Link**: https://github.com/ggml-org/llama.cpp/issues/1337
**State**: closed
**Created**: 2023-05-06T01:48:53+00:00
**Closed**: 2024-04-09T01:09:36+00:00
**Comments**: 23
**Labels**: model, stale

### Description

- [announcement][0]
- [base model, 3B][1]
- [instruct model, 3B][3]
- [chat model, 3B][2]

Hopefully this can be blazingly fast!

[0]: https://www.together.xyz/blog/redpajama-models-v1
[1]: https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1
[2]: https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1
[3]: https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1

---

## Issue #N/A: Bug: Intel Arc - not working at all

**Link**: https://github.com/ggml-org/llama.cpp/issues/9106
**State**: closed
**Created**: 2024-08-20T19:45:26+00:00
**Closed**: 2024-12-17T01:07:43+00:00
**Comments**: 28
**Labels**: bug-unconfirmed, stale, SYCL, critical severity

### Description

### What happened?

Going through the manual - SYCL I mean. Everything compiles okay. Running it always thows an error. Can't make it work. OS used: Linux Gentoo. P.S. docker doesn't work either. P.P.S. device IS listed in the list.

### Name and Version

# ./build/bin/llama-cli --version
version: 3609 (2f3c1466)
built with Intel(R) oneAPI DPC++/C++ Compiler 2024.2.1 (2024.2.1.20240711) for x86_64-unknown-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
# ZES_ENABLE_SYSMAN=1 ./build/bin/llama-cli -m models/llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33 -sm none -mg 0
Log start
main: build = 3609 (2f3c1466)
main: built with Intel(R) oneAPI DPC++/C++ Compiler 2024.2.1 (2024.2.1.20240711) for x86_64-unknown-linux-gnu
main: seed  = 1724182694
llama_model_loader: loaded meta data with 19 key-value pairs and 291 tensors from models/llama-2-7b.Q4_0.gguf (version GGUF V2)
llama_

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: vulkan on 6900xt

**Link**: https://github.com/ggml-org/llama.cpp/issues/12147
**State**: closed
**Created**: 2025-03-02T16:35:57+00:00
**Closed**: 2025-04-11T15:24:47+00:00
**Comments**: 24
**Labels**: bug-unconfirmed

### Description

### Name and Version

Latest vulkan patches cause problems in koboldCPP for radeon 6900xt.

### Operating systems

Windows

### Which llama.cpp modules do you know to be affected?

llama-quantize

### Command line

```shell

```

### Problem description & steps to reproduce

I reported a problem with vulkan on the KoboldCPP project page, and was asked to report the problem here (newly released koboldcpp version with latest fixes for vulkan).

https://github.com/LostRuins/koboldcpp/issues/1398

As I wrote in the bug report on KoboldCPP, I have two cards GTX 1080ti and Radeon 6900xt. When the creator of Koboldcpp included the latest patches for vulkan, my radeon stopped working properly with vulkan.

I have tested various models (llama 3.1, nemo, mistral small 22/24b) and in none of them vulkan on radeon 6900xt works correctly anymore, either some random characters are generated from the very beginning, or the response loops very quickly and repeats some word.

### First Bad Commit

_No 

[... truncated for brevity ...]

---

## Issue #N/A: Bug: llama.cpp with Vulkan not running on Snapdragon X + Windows (Copilot+PCs)

**Link**: https://github.com/ggml-org/llama.cpp/issues/8455
**State**: closed
**Created**: 2024-07-12T10:25:53+00:00
**Closed**: 2025-01-31T01:07:14+00:00
**Comments**: 25
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

The new Copilot+PCs with Qualcomm Snapdragon X processors (in my case a Surface 11 Pro with Snapdragon X Plus and 16GB RAM) are fast, and run llama.cpp on the CPU w/o issues. They also include a Vulkan driver and run the Vulkan samples w/o problems. But llama.cpp built with Vulkan does (now finally build,) but not run.

_llama-cli is terminating on model-load with:_
llama_model_load: error loading model: vk::Device::createComputePipeline: ErrorUnknown
llama_load_model_from_file: failed to load model
main: error: unable to load model

### Name and Version

llama-cli version: 3378 (71c1121d) with a quick-fix to compile (see #8446), built with MSVC 19.40.33812.0 for ARM64

_built with:_
Installed VulkanSDK for Windows x64, then built a Windows arm64 version of KhronosGroup/Vulkan-Loader vulkan-1.lib (+tested its functionality with tests+samples) and copied it to VulkanSDK lib-directory for llama.cpp building.
```shell
REM including Vulkan diagnostics
>

[... truncated for brevity ...]

---

## Issue #N/A: [User] -n -2 generates nothing

**Link**: https://github.com/ggml-org/llama.cpp/issues/2754
**State**: closed
**Created**: 2023-08-23T23:20:31+00:00
**Closed**: 2023-08-24T15:48:35+00:00
**Comments**: 21

### Description

See this post: https://github.com/ggerganov/llama.cpp/issues/2754#issuecomment-1691682835

---

## Issue #N/A: does not compile on CUDA 10 anymore

**Link**: https://github.com/ggml-org/llama.cpp/issues/4123
**State**: closed
**Created**: 2023-11-18T10:38:49+00:00
**Closed**: 2024-05-07T01:06:47+00:00
**Comments**: 26
**Labels**: stale

### Description

Ever since this got merged:
[https://github.com/ggerganov/llama.cpp/pull/3370](url)


---

## Issue #N/A: Bug: Qwen2-72B-Instruct (and finetunes) Q4_K_M, Q5_K_M generates random output with CuBLAS prompt processing 

**Link**: https://github.com/ggml-org/llama.cpp/issues/8025
**State**: closed
**Created**: 2024-06-20T03:39:38+00:00
**Closed**: 2024-09-07T01:07:11+00:00
**Comments**: 28
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

Qwen2-72B-Instruct Q4_K_M generates output with random tokens (numbers, special symbols, random chunks of words from different languages, etc).

Has been tested on:
1) Tesla P40 24gb + CPU partitioning with offloating half of the layers
2) Inference fully on RAM (on another pc from 1)

Other people say it works with Q6, maybe the problem is with Q4_K_M (i can't test q6).

I've tried with both FlashAttention on and off and MMQ on and off, doesn't work.

I tested with llama.cpp binaries, koboldcpp, text-generation-webui - doesn't work everywhere.

related: https://github.com/LostRuins/koboldcpp/issues/909

![image](https://github.com/ggerganov/llama.cpp/assets/54563399/e866dcbb-56b0-4eea-8ff8-6b1816520f3a)


### Name and Version

version: 3181 (37bef894)
built with MSVC 19.29.30154.0 for x64

### What operating system are you seeing the problem on?

Windows

### Relevant log output

_No response_

---

## Issue #N/A: Compilation error on Nvidia Jetson Nano

**Link**: https://github.com/ggml-org/llama.cpp/issues/4099
**State**: closed
**Created**: 2023-11-16T11:56:41+00:00
**Closed**: 2024-03-24T11:35:49+00:00
**Comments**: 23
**Labels**: bug-unconfirmed, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed). 
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Please provide a detailed written description of what you were trying to do, and what you expected `llama.cpp` to do.

Hi! I'm trying to compile llamacpp on an Nvidia Jetson Nano 2GB with CuBLAS, because I want to use the cuda cores, but I'm facing some issues with

[... truncated for brevity ...]

---

## Issue #N/A: Converting GGML->GGUF: ValueError: Only GGJTv3 supported

**Link**: https://github.com/ggml-org/llama.cpp/issues/2990
**State**: closed
**Created**: 2023-09-03T11:07:07+00:00
**Closed**: 2023-09-06T08:49:12+00:00
**Comments**: 25

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X ] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

My GGML converted models should be easy to convert to GGUF.
I know the conversion tools aren't guaranteed but I'd like to file this one in case anybody else has a workaround or more version flexible option. I would love to see any version of GGML/GGJT supported i

[... truncated for brevity ...]

---

## Issue #N/A: Implement MosiacML's 7B model.

**Link**: https://github.com/ggml-org/llama.cpp/issues/1333
**State**: closed
**Created**: 2023-05-05T17:44:48+00:00
**Closed**: 2023-11-02T00:54:44+00:00
**Comments**: 25
**Labels**: help wanted, model

### Description

Comparative to Llama in results I believe and also commercially available for use!

https://huggingface.co/mosaicml/mpt-7b

https://www.mosaicml.com/blog/mpt-7b

---

## Issue #N/A: Eval bug: Qwen2-VL Hallucinates image content on Vulkan backend

**Link**: https://github.com/ggml-org/llama.cpp/issues/10843
**State**: closed
**Created**: 2024-12-15T17:36:55+00:00
**Closed**: 2025-02-17T22:43:03+00:00
**Comments**: 23
**Labels**: bug-unconfirmed

### Description

### Name and Version

.\build\bin\Release\llama-cli.exe --version

ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 5700 XT (AMD proprietary driver) | uma: 0 | fp16: 1 | warp size: 64 | matrix cores: none
version: 4329 (89d604f2)
built with MSVC 19.41.34120.0 for x64

### Operating systems

Windows

### GGML backends

Vulkan

### Hardware

Ryzen 5900X +RX 5700 XT

### Models

Qwen2-VL-7B-Instruct-IQ4_NL + mmproj-Qwen2-VL-7B-Instruct-f32

### Problem description & steps to reproduce

When I run it on Vulkan build, the description given by the model has nothing to do with the image given as argument (no matter the `-ngl` value, even `-ngl 0` is broken). The exact same setup works perfectly fine on CPU backend.

I know the Vulkan backend doesn't support Qwen2-VL yet, but according to https://github.com/ggerganov/llama.cpp/pull/10361#issuecomment-2543938139, this should only cause slowdowns, not invalid outputs.

### Relevant log output



[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: "Unexpected empty grammar stack after accepting piece" tool crash

**Link**: https://github.com/ggml-org/llama.cpp/issues/12597
**State**: closed
**Created**: 2025-03-26T17:30:06+00:00
**Closed**: 2025-04-16T14:32:28+00:00
**Comments**: 21
**Labels**: bug-unconfirmed

### Description

### Name and Version

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070 Laptop GPU, compute capability 8.9, VMM: yes
version: 4958 (ef19c717)
built with Ubuntu clang version 18.1.8 (++20240731024944+3b5b5c1ec4a3-1~exp1~20240731145000.144) for x86_64-pc-linux-gnu


### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 llama-server --ctx-size 0 --jinja -fa -hf bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M --host 0.0.0.0 -ngl 100
```

### Problem description & steps to reproduce

If I run BFVL v3 on the above `llama-server`, it eventually  crashes with:

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  Unexpected empty grammar stack after accepting piece: ```json
{
    "
/home/ed/.local/share/dorothy/user/commands/llama-cpp-server: line 69: 7365

[... truncated for brevity ...]

---

## Issue #N/A: Create a logo

**Link**: https://github.com/ggml-org/llama.cpp/issues/105
**State**: closed
**Created**: 2023-03-13T21:15:21+00:00
**Closed**: 2023-07-28T19:20:49+00:00
**Comments**: 47
**Labels**: good first issue, 🦙.

### Description

We should probably make a logo for this project. Like an image of a 🦙 and some C++

---

## Issue #N/A: When using GPU (OpenCL), the reply speed is slower and all replies are incorrect？？

**Link**: https://github.com/ggml-org/llama.cpp/issues/7661
**State**: closed
**Created**: 2024-05-31T07:45:45+00:00
**Closed**: 2024-09-16T01:07:32+00:00
**Comments**: 23
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

I used termux to compile llama.cpp with gpu.  and i found that the response speed has been slower and they are all errors。



like:
![image](https://github.com/ggerganov/llama.cpp/assets/46549527/e5009839-a3fa-432c-bd7e-3e5827dcf5db)
error response:
![image](https://github.com/ggerganov/llama.cpp/assets/46549527/763810a2-ba27-43d0-9077-abf9a1b58d72)

Who can tell me what the reason is？？？？


### Name and Version

./bin/main -t 8 -ngl 33 -m ../llama-2-7b-chat.Q4_0.gguf --color -n -1 -ins -b 256

environment：  linux+termux   GPU: Qualcomm  8gen2

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

---

## Issue #N/A: Eval bug: getting assertion error when trying to use a gguf quantized model at inference "GGML_ASSERT(n_outputs_enc > 0 && "call llama_encode() first") failed"

**Link**: https://github.com/ggml-org/llama.cpp/issues/12080
**State**: closed
**Created**: 2025-02-26T10:41:48+00:00
**Closed**: 2025-05-04T01:08:01+00:00
**Comments**: 28
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

the latest version of llama.cpp

### Operating systems

Windows

### GGML backends

CPU

### Hardware

CPU 16 GB RAM Intel I5 core 10th Gen

### Models

Flan T5 Large

### Problem description & steps to reproduce

I have a finetuned Flan T5 Model in my local which I have quantized and converted to gguf format using llama.cpp using the below line of command:

!python {path to convert_hf_to_gguf.py} {path to hf_model} --outfile {name_of_outputfile.gguf} --outtype {quantization type}

and loaded the gguf file using llama.cpp Llama

from llama_cpp import Llama 
gguf_model_path = "t5_8bit.gguf"
model = Llama(model_path=gguf_model_path)

and when trying to use the model at inference in Jupyter Notebook, the kernel is Dying. When tried the same in Command Prompt, getting the aasertion issue "GGML_ASSERT(n_outputs_enc > 0 && "call llama_encode() first") failed"

used the below code for inference in CPU and the issue is detected at model.eval()

Code:
prompt = "Extract Tag

[... truncated for brevity ...]

---

## Issue #N/A: 4bit version of gpt4all-alpaca-oa-codealpaca-Lora-13b?

**Link**: https://github.com/ggml-org/llama.cpp/issues/1037
**State**: closed
**Created**: 2023-04-18T04:47:11+00:00
**Closed**: 2023-05-08T09:34:10+00:00
**Comments**: 22

### Description

Hello, 
to reduce my brain usage even more I thought i'd be nice to run AI which is specifically trained to code and thus hopefully make better code than other language models which are trained for e.g. natural language.

So I found this: https://huggingface.co/jordiclive/gpt4all-alpaca-oa-codealpaca-lora-13b

I of course wanted to try and run it but there's a problem, there aren't even any pytorch_model files or any 4bit variants listed here: https://github.com/underlines/awesome-marketing-datascience/blob/master/awesome-ai.md

Thank your for your support!

---

## Issue #N/A: How to build on windows?

**Link**: https://github.com/ggml-org/llama.cpp/issues/103
**State**: closed
**Created**: 2023-03-13T20:13:14+00:00
**Closed**: 2023-07-28T19:20:41+00:00
**Comments**: 22
**Labels**: documentation, good first issue, windows

### Description

Please give instructions. There is nothing in README but it says that it supports it 

---

