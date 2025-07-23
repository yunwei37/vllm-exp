# stale - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 29

### Label Distribution

- stale: 30 issues
- bug-unconfirmed: 15 issues
- enhancement: 8 issues
- low severity: 3 issues
- threading: 2 issues
- medium severity: 1 issues
- generation quality: 1 issues

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

## Issue #N/A: Misc. bug: Intel docker containers are running out of disk space during build

**Link**: https://github.com/ggml-org/llama.cpp/issues/12290
**State**: closed
**Created**: 2025-03-09T20:19:04+00:00
**Closed**: 2025-04-23T01:07:41+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

See: https://github.com/ggml-org/llama.cpp/actions/runs/13744758544/job/38438332993

It has the error message: 

> You are running out of disk space. The runner will stop working when the machine runs out of disk space. Free space left: 24 MB

It looks like there is already a [free disk space](https://github.com/ggml-org/llama.cpp/blob/master/.github/workflows/docker.yml#L103-L118) but it is currently [disabled for all builds](https://github.com/ggml-org/llama.cpp/blob/master/.github/workflows/docker.yml#L42). 

### Problem description & steps to reproduce

Intel docker container builds are failing due to running out of disk space.


---

## Issue #N/A: 【help】why  function  llama_build_graph  is internal function  llama_decode？

**Link**: https://github.com/ggml-org/llama.cpp/issues/5916
**State**: closed
**Created**: 2024-03-07T06:56:01+00:00
**Closed**: 2024-04-21T01:06:37+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale

### Description

  I read the llama.cpp source code。
  I am confused as to why the function llama_build_graph needs to be called every time the function llama_decode is called.
 The function llama_build_graph cannot be called during program initialization, which will reduce the inference time.

static int llama_decode_internal(
         llama_context & lctx,
           llama_batch   batch) {
    ....
   ggml_cgraph * gf = llama_build_graph(lctx, batch, false);
.....
}

Thanks

---

## Issue #N/A: [Suggestion] Add parameter for setting openblas threads

**Link**: https://github.com/ggml-org/llama.cpp/issues/1188
**State**: closed
**Created**: 2023-04-26T13:24:17+00:00
**Closed**: 2024-04-09T01:09:52+00:00
**Comments**: 1
**Labels**: enhancement, threading, stale

### Description

Openblas deafults to some maximum available threads, but would probably not be the most optimal.
In Openblas there is a function to set the number of threads, why not use this?

```void openblas_set_num_threads(int num_threads);```

Current workaround is to set an openblas environment variable.

---

## Issue #N/A: [requirement] Support baichuan 13B model.

**Link**: https://github.com/ggml-org/llama.cpp/issues/2185
**State**: closed
**Created**: 2023-07-12T03:15:38+00:00
**Closed**: 2024-04-09T01:08:09+00:00
**Comments**: 4
**Labels**: stale

### Description

Baichuan 13B model use ALiBi , any plan to support this model?

---

## Issue #N/A: Prompt evaluation performance regression in llama.cpp on RDNA3 with HSA_OVERRIDE_GFX_VERSION=11.0.1 vs 11.0.0

**Link**: https://github.com/ggml-org/llama.cpp/issues/3701
**State**: closed
**Created**: 2023-10-20T17:47:41+00:00
**Closed**: 2024-04-04T01:07:46+00:00
**Comments**: 3
**Labels**: stale

### Description

(Probably a ROCm issue - see https://github.com/RadeonOpenCompute/ROCm/issues/2590 -, but maybe llama.cpp devs can also weigh in.)

We're evaluating llama.cpp with ROCm offloading on various RDNA3 GPUs (primarily RX 7800 XT and RX 7900 XT).

On initial testing, we found that a 13b model with Q6_K quantization fully offloaded to GPU (-ngl 43) showed significantly slower prompt evaluation times on the 7800 compared to the 7900, far beyond what would be expected from the relative performance difference between these 2 GPUs. On the 7800, prompt evaluation would take more than a minute on our example prompt, while it was near instantaneous (less than a second) on the 7900.

Upon further investigation, the RX 7800 shows a 3 second penalty for every 64 tokens of prompt (0-64 tokens 3 seconds, 65-128 tokens 6s, 129-192 tokens 9s, and so on) over the 7900.

Since the 7800 is recognized by ROCm as gfx1101, vs gfx1100 on the 7900, we tried setting HSA_OVERRIDE_GFX_VERSION=11.0.0, which le

[... truncated for brevity ...]

---

## Issue #N/A: server: system prompt makes generated text incoherent

**Link**: https://github.com/ggml-org/llama.cpp/issues/4103
**State**: closed
**Created**: 2023-11-16T17:48:59+00:00
**Closed**: 2024-04-02T01:11:19+00:00
**Comments**: 9
**Labels**: bug-unconfirmed, stale

### Description

# Current Behavior

Passing a system prompt to the `server` makes the generated text incoherent after the first request.


# Environment and Context

**Commit:** 8da46278e1a57107591653275f8e03a281de94f0

**OS:** Kubuntu 23.10

<blockquote>
❯ lscpu | grep -P 'Model name|Flags'

Model name:                         AMD Ryzen 9 7900 12-Core Processor
Flags:                              fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2

[... truncated for brevity ...]

---

## Issue #N/A: Bug:  Rocm extreme slow down on GFX1100 with release binary

**Link**: https://github.com/ggml-org/llama.cpp/issues/9765
**State**: closed
**Created**: 2024-10-06T17:16:14+00:00
**Closed**: 2024-11-20T01:07:29+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

There are large slow down on gfx1100

### Name and Version

.\llama-cli.exe --version
version: 1 (b6d6c52)
built with  for x86_64-pc-windows-msvc

.\llama-cli.exe --version
version: 3235 (88540445)
built with  for x86_64-pc-windows-msvc

### What operating system are you seeing the problem on?

Windows

### Relevant log output

```shell
latest release binary
.\llama-bench.exe -m W:\model\qwen2-7b-instruct-q5_k_m.gguf -ngl 99 -fa 1,0
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon RX 7900 XTX, compute capability 11.0, VMM: no
| model                          |       size |     params | backend    | ngl | fa |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | ------------: | -------------------: |
| qwen2 ?B Q5_K - Medium         |   5.07 GiB |     7.62 B | CUDA       |  99 |  1 |    

[... truncated for brevity ...]

---

## Issue #N/A: How to make llama.cpp return control to add additional context?

**Link**: https://github.com/ggml-org/llama.cpp/issues/692
**State**: closed
**Created**: 2023-04-01T22:20:36+00:00
**Closed**: 2024-04-11T01:07:16+00:00
**Comments**: 2
**Labels**: enhancement, generation quality, stale

### Description

I want to be able to tell the model that if it can't reply something useful to return control so I can give more information.

Similarly, how do I add more context so that it can reason about a full conversation or say a specific set of documents?

For example, I ask it something and it should say I don't know can you provide me more information? And then I give it a document. Then I can add another document to the prompt, so it can understand from that and so on.

I've heard this is some sort of chaining, but I don't understand.

---

## Issue #N/A: Misc. bug: Vulkan is not optional at runtime

**Link**: https://github.com/ggml-org/llama.cpp/issues/11493
**State**: closed
**Created**: 2025-01-29T17:45:16+00:00
**Closed**: 2025-03-16T01:07:49+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

With llama.cpp version from git tag b4549, if I compile Vulkan support in and then run in an environment where Vulkan is not supported (for example this would happen if a Linux distribution provides llama.cpp with Vulkan enabled but the user doesn't have a GPU with Vulkan), it will fail with the following exception:

    terminate called after throwing an instance of 'vk::IncompatibleDriverError'
      what():  vk::createInstance: ErrorIncompatibleDriver

It would be better to just disable Vulkan in this case but run on CPU.


### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

libllama (core library)

### Command line

```shell
Any llama-cli command (yes, even `llama-cli -dev none`).
```

### Problem description & steps to reproduce

1. Compile with GGML_VULKAN=ON
2. Run without GPU
3. It crashes with exception

### First Bad Commit

_No response_

### Relevant log output

```shell

```

---

## Issue #N/A: [Question] WARP_SIZE as 64 for MI GPUs?

**Link**: https://github.com/ggml-org/llama.cpp/issues/3630
**State**: closed
**Created**: 2023-10-15T09:14:05+00:00
**Closed**: 2024-04-04T01:08:18+00:00
**Comments**: 5
**Labels**: stale

### Description

MI GPUs based on the gfx9xx architecture such as MI100 and MI200 natively run on wave64 (64 threads per warp/wave). That means by not setting `WARP_SIZE` as `64`, we're not able to fully utilize those GPUs. RDNA1 onwards support wave32 as well as wave64, so it can still run well when WARP_SIZE is `32`.

What would it take to get `WARP_SIZE` of `64` to work? I tried it and ran llama2 7b q4_0 but regardless of a prompt I give, the result is always `######...`

I suspect the reduction kernels that use `__shfl_*_sync` have the last parameter width set to `32`. So I tried setting those to `WARP_SIZE` as well, as well as setting the `s_sum` size to `WARP_SIZE`, but that didn't make any difference. Am I missing something? I guess there's still many parts of the code that rely on `WARP_SIZE` set to `32`, so I was wondering which parts I should look at.

---

## Issue #N/A: Eval bug: Can't run Qwen3-32B Q4_K_XL

**Link**: https://github.com/ggml-org/llama.cpp/issues/13298
**State**: open
**Created**: 2025-05-04T11:24:21+00:00
**Comments**: 9
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

build: 5273 (8ae5ebcf) with gcc-14 (Homebrew GCC 14.2.0_1) 14.2.0 for x86_64-pc-linux-gnu

### Operating systems

Linux

### GGML backends

CUDA

### Hardware

2x T4

### Models

https://huggingface.co/unsloth/Qwen3-32B-GGUF/blob/main/Qwen3-32B-UD-Q4_K_XL.gguf

### Problem description & steps to reproduce

NaN perplexity and completely trashed output while using [this model](https://huggingface.co/unsloth/Qwen3-32B-GGUF/blob/main/Qwen3-32B-UD-Q4_K_XL.gguf)

### First Bad Commit

_No response_

### Relevant log output

```shell
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: Tesla T4, compute capability 7.5, VMM: yes
  Device 1: Tesla T4, compute capability 7.5, VMM: yes
build: 5273 (8ae5ebcf) with gcc-14 (Homebrew GCC 14.2.0_1) 14.2.0 for x86_64-pc-linux-gnu
llama_model_load_from_file_impl: using device CUDA0 (Tesla T4) - 14992 MiB free
llama_model_load_from_file_impl: using de

[... truncated for brevity ...]

---

## Issue #N/A: Run on GPU

**Link**: https://github.com/ggml-org/llama.cpp/issues/5134
**State**: closed
**Created**: 2024-01-26T03:39:07+00:00
**Closed**: 2024-04-02T01:08:20+00:00
**Comments**: 5
**Labels**: enhancement, stale

### Description

I compiled the main file according to the instructions on the official website below
`mkdir build
cd build
cmake .. -DLLAMA_CUBLAS=ON
cmake --build . --config Release`

But I found that the inference speed is 40t/s when using the following instructions
`./build/bin/main -m /data/nwj/models/microsoft/phi-2/ggml-model-f32_q4_0.gguf -p "Question: Write a python function to print the first n numbers in the fibonacci series" `

When I add -ngl 33, the speed is 10t/s
`./build/bin/main -m /data/nwj/models/microsoft/phi-2/ggml-model-f32_q4_0.gguf -p "Question: Write a python function to print the first n numbers in the fibonacci series" -ngl 33`

the output is as follows:
./build/bin/main -m /data/nwj/models/microsoft/phi-2/ggml-model-f32_q4_0.gguf -p "Question: Write a python function to print the first n numbers in the fibonacci series" -ngl 33
Log start
main: build = 1761 (cb1e281)
main: built with cc (Ubuntu 10.5.0-1ubuntu1~22.04) 10.5.0 for x86_64-linux-gnu
main: seed  = 

[... truncated for brevity ...]

---

## Issue #N/A: Idefics2 VLM Support

**Link**: https://github.com/ggml-org/llama.cpp/issues/6706
**State**: closed
**Created**: 2024-04-16T16:29:38+00:00
**Closed**: 2024-06-18T01:07:01+00:00
**Comments**: 3
**Labels**: enhancement, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x ] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

Requesting support for HuggingFace's new [Idefics2 VLM](https://huggingface.co/blog/idefics2).

# Motivation

- First true open source VLM (Apache 2.0)
- This 8B model offers comparable performance to Llava-1.6-34b and Apple's unreleased 30B MM1.
- The Hug

[... truncated for brevity ...]

---

## Issue #N/A: common: Gibberish results and/or crashes due to incorrect character encodings

**Link**: https://github.com/ggml-org/llama.cpp/issues/6396
**State**: closed
**Created**: 2024-03-30T09:32:12+00:00
**Closed**: 2024-05-28T02:13:07+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale

### Description

As of ~~b2579~~ b2646, prompts (among other parameters) are internally stored as `std::string`s, which is basically glorified `std::vector<char>` and do not care or handle character encodings. This will not cause any problem since (as far as I can tell) llama.cpp treats all strings as in UTF-8, but care must be taken when taking strings from external sources.

For example, when parsing command-line arguments, `--prompt` (and maybe other arguments) gets stored directly as `params.prompt`:

https://github.com/ggerganov/llama.cpp/blob/c342d070c64a1ffe35d22c1b16b672e684a30297/common/common.cpp#L215-L222

This (somehow) works on Linux, but thanks to Windows' infinite wisdom `argv` is in ANSI codepage encoding, and will cause gibberish results or a crash to happen soon after since all other parts are expecting a UTF-8 string:

https://github.com/ggerganov/llama.cpp/blob/c342d070c64a1ffe35d22c1b16b672e684a30297/llama.cpp#L10974

https://github.com/ggerganov/llama.cpp/blob/c342d070c6

[... truncated for brevity ...]

---

## Issue #N/A: Server completion streaming returns special tokens as empty strings in chunks

**Link**: https://github.com/ggml-org/llama.cpp/issues/7106
**State**: closed
**Created**: 2024-05-06T18:27:10+00:00
**Closed**: 2024-07-24T01:06:53+00:00
**Comments**: 9
**Labels**: bug-unconfirmed, stale

### Description

Version: b2794.
Model: Meta-Llama-3-8B-Instruct-Q8_0.gguf (updated)
Prompt: "<|start_header_id|>user<|end_header_id|>How much is 12 plus 19?<|eot_id|>"

When I run the server and send a completion request with streaming, in the verbose logs I see that the server generates the "<|start_header_id|>", "assistant" and "<|end_header_id|>", followed by "\n\n12 + 19 = 31".

However, the streaming chunks sent by server for <|start_header_id|> and <|end_header_id|> have empty strings as `content` in `data`.

I couldn't find a config parameter either in the server or in the request that could change this behavior.

---

## Issue #N/A: Eval bug: Qwerky 72B (rwkv6qwen2) failed to load with `--split-mode row` option

**Link**: https://github.com/ggml-org/llama.cpp/issues/12692
**State**: closed
**Created**: 2025-04-01T13:04:59+00:00
**Closed**: 2025-05-16T01:07:49+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: yes
ggml_cuda_init: found 6 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes
  Device 2: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes
  Device 3: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes
  Device 4: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes
  Device 5: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes
version: 5008 (4172aea2)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu

### Operating systems

Linux

### GGML backends

CUDA

### Hardware

Threadripper 3960X + 6x RTX 4060 Ti 16GB

### Models

Original model: [featherless-ai/Qwerky-72B](https://huggingface.co/featherless-ai/Qwerky-72B)
Quantized model using latest llama.cpp: https://huggingface.co/exxocism/featherless-ai_Qwerky-72B-G

[... truncated for brevity ...]

---

## Issue #N/A: How to write a chat template for llama.cpp server?

**Link**: https://github.com/ggml-org/llama.cpp/issues/5822
**State**: closed
**Created**: 2024-03-01T19:04:28+00:00
**Closed**: 2024-04-16T01:06:29+00:00
**Comments**: 2
**Labels**: stale

### Description

Which variables do i have to use in jina? How to define user-name and system-name?

https://github.com/ggerganov/llama.cpp/issues/5766#issuecomment-1973708563

---

## Issue #N/A: Feature Request: Anti-slop / fine tuning of a model output in realtime / on the fly for output quality enhancement.

**Link**: https://github.com/ggml-org/llama.cpp/issues/9748
**State**: closed
**Created**: 2024-10-05T02:44:16+00:00
**Closed**: 2024-11-19T01:17:33+00:00
**Comments**: 1
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Basically this enhancement fixes model generation on the fly so to speak, and drastically improves the performance of any model
for specific tasks.

Although this is more involved than the "XTC" enhancement, this one is far stronger and will allow users to control the quality of generation of any model at the root level customized to their use case(s).

The occurs on the word, phrase level rather than per token level... roughly it forces the model to regenerate token(s) when generat

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug:

**Link**: https://github.com/ggml-org/llama.cpp/issues/12623
**State**: closed
**Created**: 2025-03-28T09:57:55+00:00
**Closed**: 2025-05-12T01:07:58+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

git clone https://github.com/ggerganov/llama.cpp
make GGML_CUDA=1

I llama.cpp build info: 
I UNAME_S:   Linux
I UNAME_P:   x86_64
I UNAME_M:   x86_64
I CFLAGS:    -I. -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG  -std=c11   -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -Wdouble-promotion -pthread -march=native -mtune=native 
I CXXFLAGS:  -I. -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG  -std=c++11 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread  -Wno-array-bounds -Wno-format-truncation -march=native -mtune=native 
I NVCCFLAGS:  -I. -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG  -std=c++11 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread    -Wno-pedantic -Xcompi

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: llama-server does not print model loading errors by default (log level misconfigured?)

**Link**: https://github.com/ggml-org/llama.cpp/issues/11819
**State**: closed
**Created**: 2025-02-12T09:18:11+00:00
**Closed**: 2025-03-29T01:07:39+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

```
build: 4205 (c6bc7395) with Apple clang version 16.0.0 (clang-1600.0.26.3) for arm64-apple-darwin23.6.0
```

### Operating systems

Mac

### Which llama.cpp modules do you know to be affected?

llama-server

### Problem description & steps to reproduce

When using a model with e.g. an incompatible pre-tokenizer, the loading error isn't shown by default; `llama-server` seems to just quit.

```
$ ./llama-server -m model-CoT-Q4_K_M.gguf
build: 4205 (c6bc7395) with Apple clang version 16.0.0 (clang-1600.0.26.3) for arm64-apple-darwin23.6.0
[...]
main: HTTP server is listening, hostname: 127.0.0.1, port: 8080, http threads: 11
main: loading model
srv    load_model: loading model 'model-CoT-Q4_K_M.gguf'
llama_load_model_from_file: using device Metal (Apple M2 Max) - 49151 MiB free
llama_model_loader: loaded meta data with 25 key-value pairs and 579 tensors 
[...]
llama_model_loader: - type  f32:  241 tensors
llama_model_loader: - type q4_K:  289 tensors
llama_model_

[... truncated for brevity ...]

---

