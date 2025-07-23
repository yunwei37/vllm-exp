# references_prs - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 3
- Closed Issues: 27

### Label Distribution

- bug-unconfirmed: 16 issues
- stale: 16 issues
- medium severity: 2 issues
- bug: 2 issues
- enhancement: 2 issues
- good first issue: 2 issues
- performance: 1 issues
- low severity: 1 issues
- need more info: 1 issues
- duplicate: 1 issues

---

## Issue #N/A: [Feature] Supporting arbitrary map operations in GGML

**Link**: https://github.com/ggml-org/llama.cpp/issues/875
**State**: closed
**Created**: 2023-04-10T13:19:16+00:00
**Closed**: 2023-04-14T15:15:34+00:00
**Comments**: 1

### Description

Proof of concept patch in #874 which adds `MAP_UNARY` and `MAP_BINARY` operations that can take a function pointer which is applied similarly to `ggml_vec_add_f32` and friends.

Is this something that could potentially get included? It would enable projects or `llama.cpp` to support a larger range of models pretty easily because at least the simple version of an operation that GGML doesn't support could be used without having to make modifications to GGML itself.

One example is RWKV which requires several operations that GGML doesn't support.

---

## Issue #N/A: Build fail - implicit declaration of function ‘vld1q_u8_x2’ on Armv7-linux with the latest release

**Link**: https://github.com/ggml-org/llama.cpp/issues/5748
**State**: closed
**Created**: 2024-02-27T09:11:46+00:00
**Closed**: 2024-04-12T01:06:38+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale

### Description

We wanted to [build binaries for ARMv6,7 on Linux](https://github.com/JuliaPackaging/Yggdrasil/pull/8165) but we're getting the following two types of errors:

> [09:00:29] /workspace/srcdir/llama.cpp/ggml-quants.c: In function ‘ggml_vec_dot_iq2_s_q8_K’:
> [09:00:29] /workspace/srcdir/llama.cpp/ggml-quants.c:9645:32: error: implicit declaration of function **‘vld1q_u8_x2’**; did you mean ‘vld1q_u32’? [-Werror=implicit-function-declaration]
> [09:00:29]  9645 |     const uint8x16x2_t mask1 = vld1q_u8_x2(k_mask1);

Similar:
 > /workspace/srcdir/llama.cpp/ggml-quants.c:9678:34: error: implicit declaration of function ‘**vqtbl1q_u8**’; did you mean ‘vtbl1_u8’? [-Werror=implicit-function-declaration]
> [09:09:12]  9678 |             vs.val[1] = vandq_u8(vqtbl1q_u8(vs.val[0], mask1.val[1]), mask2);
> [09:09:12]       |                                  ^~~~~~~~~~
> [09:09:12]       |                                  vtbl1_u8
> [09:09:12] /workspace/srcdir/llama.cpp/ggml-quants.c:96

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: Program not working properly due to new features of "repack Q4_K tensor"

**Link**: https://github.com/ggml-org/llama.cpp/issues/12528
**State**: closed
**Created**: 2025-03-23T14:28:59+00:00
**Closed**: 2025-03-26T11:02:01+00:00
**Comments**: 6
**Labels**: bug-unconfirmed

### Description

### Name and Version

built with cc (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0 for x86_64-linux-gnu

### Operating systems

Linux

### GGML backends

CPU

### Hardware

13th Gen Intel(R) Core(TM) i9-13900H

### Models

DeepSeek-V2-Lite-Q4_K_M

### Problem description & steps to reproduce

_Usage:  ./llama-simple -m $Model_Path/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf_

I used the git bisect tool to find out that after submitting 3d82dbcbce2c677fc35fbf99574ccd107d95a1f8 , the program does not work properly. And this feature is was introduced on #12332 . This directly caused my CPU to have an overflow error when calculating “ffn-moe-gate”.
Unfortunately, I am not familiar with this featrue.Could anyone fix this bug?  @Srihari-mcw @ggerganov 

### First Bad Commit

3d82dbcbce2c677fc35fbf99574ccd107d95a1f8

### Relevant log output

```shell
repack: repack tensor blk.0.attn_kv_a_mqa.weight with q4_K_8x8
repack: repack tensor blk.0.attn_kv_b.weight with q4_K_8x8
repack: repack tensor blk.0.att

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: test-backend-ops grad crash by GGML_ASSERT error

**Link**: https://github.com/ggml-org/llama.cpp/issues/12520
**State**: closed
**Created**: 2025-03-22T23:56:35+00:00
**Closed**: 2025-05-06T01:07:43+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

.\llama-cli.exe --version
version: 4942 (fbdfefe7)
built with MSVC 19.43.34808.0 for x64

### Operating systems

Windows

### Which llama.cpp modules do you know to be affected?

Test code

### Command line

```shell
> .\test-backend-ops.exe grad -o CPY
or
> .\test-backend-ops.exe grad
```

### Problem description & steps to reproduce

## description

Commit #12310 crashes test-backend-ops grad.
It doesn't seem to matter which backend.

## steps to reproduce

Run `test-backend-ops` as `grad` mode.

### First Bad Commit

Commit #12310 : SHA ba932dfb50cc694645b1a148c72f8c06ee080b17

### Relevant log output

```shell
[3/23 08:24:26] PS E:\AI\llama.cpp\b4942\llama-b4942-bin-win-vulkan-x64
> .\test-backend-ops.exe grad -o CPY
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon(TM) Graphics (AMD proprietary driver) | uma: 1 | fp16: 1 | warp size: 64 | shared memory: 32768 | matrix cores: none
Testing 2 devices

Backend 1/2: Vulkan0
  Device description: AMD

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Compilation failure swift build on Windows

**Link**: https://github.com/ggml-org/llama.cpp/issues/8739
**State**: closed
**Created**: 2024-07-28T16:12:09+00:00
**Closed**: 2024-09-12T01:41:03+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

I'm trying to build the Swift package of llama.cpp on Windows using Swift 5.10. However, the build fails with errors stating that some return types are part of a C++ 14 extension. I decided to switch the cxx standard in Package.swift to C++ 14, which yielded a different set of errors in the logs field.

I expected the build to succeed with no issues so I can continue building my application.

I'm not too familiar with C++ build tooling, so I'd appreciate the help and would be happy to provide more verbose logs if needed.

System info:
- Windows 11
- Swift 5.10
- MSVC 14.8 (From VS Build Tools 2022)

### Name and Version

Commit 4730fac

### What operating system are you seeing the problem on?

Windows

### Relevant log output

```shell
MARK: With c++ 11 standard:
Building for debugging...
In file included from D:\swift-executable\.build\checkouts\llama.cpp\src\unicode.cpp:5:
In file included from D:\swift-executable\.build\checkouts\llama.cpp\src/unicode

[... truncated for brevity ...]

---

## Issue #N/A: llama.cpp main hangs at prompt with latest mmap updates

**Link**: https://github.com/ggml-org/llama.cpp/issues/669
**State**: closed
**Created**: 2023-04-01T04:10:19+00:00
**Closed**: 2023-04-01T08:54:17+00:00
**Comments**: 2

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x ] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior
After upgrading to latest code compiling and then running an inference using main the following prompt should return results like before:

```
./main -m models/13B/ggml-model-q4_0.bin -n 512 --repeat_penalty 1.0 --color  -p "What is controlled delivery?"
main: see

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: ROCm error: batched GEMM not supported

**Link**: https://github.com/ggml-org/llama.cpp/issues/14576
**State**: open
**Created**: 2025-07-08T03:01:22+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

### Name and Version

llama built from source - latest master.
```
llama-cli --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon RX 7900 XTX, gfx1100 (0x1100), VMM: no, Wave Size: 32
version: 5836 (b9c3eefd)
built with cc (GCC) 14.2.1 20240912 (Red Hat 14.2.1-3) for x86_64-redhat-linux
```

Build command:
```
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)"  cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF && cmake --build build --config Release -- -j 16
```

### Operating systems

Linux

### GGML backends

HIP

### Hardware

AMD Ryzen 5 5600X
AMD ATI Radeon RX 7900 XTX


### Models

All models tested failed
Note: Non-Quant models untested.

### Problem description & steps to reproduce

`llama-run --ngl 10 <MODEL>`
fails with any GPU use, ngl 0 functions as expected


### First Bad Commit

_No response_

### Relevant log

[... truncated for brevity ...]

---

## Issue #N/A: LORA Adapter Hot Swap Implementation Problem

**Link**: https://github.com/ggml-org/llama.cpp/issues/10374
**State**: closed
**Created**: 2024-11-18T05:43:29+00:00
**Closed**: 2025-01-03T01:07:23+00:00
**Comments**: 5
**Labels**: stale

### Description

I have been following the discussions in the following threads:

[Pull Request #8332](https://github.com/ggerganov/llama.cpp/pull/8332)
[Pull Request #8857](https://github.com/ggerganov/llama.cpp/pull/8857)
I believe that the ideal implementation of "hot swap" should address the following scenario:

When processing a request, llama.cpp should be able to dynamically determine and apply the correct LoRA adapter based on the specific requirements of the request. While I understand that the current implementation involves a scaling mechanism, this approach introduces significant issues.

For example, when llama.cpp is running as a server handling multiple simultaneous requests with different LoRA adapters, the scaling method creates a problematic dependency. If Request 1 comes in requiring LoRA Adapter 1, the scaling is adjusted to prioritize Adapter 1. However, if Request 2 arrives shortly afterward, requiring LoRA Adapter 2, the scaling is adjusted again, effectively disabling Ad

[... truncated for brevity ...]

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

## Issue #N/A: Misc. bug: malloc error in #8  0x00007fffda1ea8ec in unicode_regex_split

**Link**: https://github.com/ggml-org/llama.cpp/issues/12335
**State**: closed
**Created**: 2025-03-11T14:44:11+00:00
**Closed**: 2025-04-26T01:07:36+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version


version 👍 
```
git log 
commit d2fe216fb2fb7ca8627618c9ea3a2e7886325780 (HEAD -> master, tag: b4667, origin/master, origin/HEAD)
Author: Eric Curtin <ecurtin@redhat.com>
Date:   Fri Feb 7 14:42:46 2025 +0000

    Make logging more verbose (#11714)
    
    Debugged an issue with a user who was on a read-only filesystem.
    
    Signed-off-by: Eric Curtin <ecurtin@redhat.com>

```
build  library .so with: 
```
cmake -B build_x86_gpu -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="70;75;86;87" -DGGML_CUDA_GRAPHS_DEFAULT=ON -DGGML_USE_CUDA=1
cmake --build build_x86_gpu --config Debug -j32
sudo cmake --install build_x86_gpu --prefix install_x86_gpu
```


coredump with backtrace in attach file llama.cpp.txt:

[llama.cpp.txt](https://github.com/user-attachments/files/19185884/llama.cpp.txt)



### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

libllama (core library)

### Command line

```shell
api named common_tokenize in common/common.

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: (HIP) RDNA4 prefill performance almost halved with CUBLAS_COMPUTE_32F

**Link**: https://github.com/ggml-org/llama.cpp/issues/12764
**State**: closed
**Created**: 2025-04-05T05:51:39+00:00
**Closed**: 2025-04-10T08:15:00+00:00
**Comments**: 5
**Labels**: bug-unconfirmed

### Description

### Name and Version

$ llama-cli --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Graphics, gfx1201 (0x1201), VMM: no, Wave Size: 32
version: 5054 (7a84777f)
built with AMD clang version 17.0.6 (CLANG: AOCC_5.0.0-Build#1377 2024_09_24) for x86_64-unknown-linux-gnu

Build params: -DGGML_HIP=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_RPC=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100,gfx1201 -DGGML_HIP_ROCWMMA_FATTN=ON

Using LD_PRELOAD to load locally built hipBLASLt from develop branch, commit: ea7a0aceca8d54ce92428997d8b796a796d99def
Using rocWMMA from develop branch, commit: f3b60adb7bded2114cca77e7165591e2d5a4d07e

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

libllama (core library)

### Command line

```shell
llama-bench -m ~/models/qwen2.5-14b-q4_0.gguf -ngl 99 -fa 1
```

### Problem description & steps to reproduce

#### Master
| model          

[... truncated for brevity ...]

---

## Issue #N/A: Intermittent segmentation faults in llama_sample_top_p_top_k()

**Link**: https://github.com/ggml-org/llama.cpp/issues/830
**State**: closed
**Created**: 2023-04-07T12:33:14+00:00
**Closed**: 2024-04-11T01:06:54+00:00
**Comments**: 5
**Labels**: stale

### Description

# Expected Behavior

I have been getting intermittent segfaults for no apparent reason. Sometimes they occur right at the beginning of text generation, and sometimes they occur after a lot of text has already been generated. They seem to be deterministic in that I can sometimes work around them by changing the prompt, but if I don’t change the prompt, they consistently occur. I normally use the 65B model, which exhibits the problem, but I am attaching a repro for the 13B model. I am not 100% sure but I believe the issue affects all four model sizes (7B, 13B, 30B, 65B).

# Current Behavior

Intermittent segfaults

# Environment and Context 

Please provide detailed information about your computer setup. This is important in case the issue is not reproducible except for under certain specific conditions.

* Physical (or virtual) hardware you are using, e.g. for Linux:

2019 16-inch MacBook Pro, 2.3 GHz 8-Core Intel Core i9, 64 GB of RAM

* Operating System, e.g. for Linux

[... truncated for brevity ...]

---

## Issue #N/A: CUDA Error 400: Invalid Resource Handle when Running on Single GPU

**Link**: https://github.com/ggml-org/llama.cpp/issues/2269
**State**: closed
**Created**: 2023-07-19T03:47:47+00:00
**Closed**: 2024-04-09T01:07:51+00:00
**Comments**: 5
**Labels**: stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ x ] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ x ] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ x ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ x ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

I am trying to make `llama.cpp` run on a single GPU (in my case, GPU 5) on a multi-GPU system because there are other tasks running on my other GPUs.

# Current Behavior

`llama.cpp` crashes with `CUDA error 400 at ggml-cuda.cu:3343: invalid resource handl

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Inconsistent ggml-4-x86-cuda-v100 ci failures on master

**Link**: https://github.com/ggml-org/llama.cpp/issues/7613
**State**: closed
**Created**: 2024-05-29T09:47:11+00:00
**Closed**: 2024-07-13T01:06:48+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale, low severity

### Description

**Note: Only one datapoint of ci failure, but it would be important to keep track of this behavior over the next few commits**

### What happened?

* [Passing commit](https://github.com/ggml-org/ci/tree/results/llama.cpp/50/4f0c340f6b5e04de682f6ddefdd3b81208df5d/ggml-4-x86-cuda-v100)
    - https://github.com/ggerganov/llama.cpp/commit/504f0c340f6b5e04de682f6ddefdd3b81208df5d

* [Failing commit, but its a readme only change](https://github.com/ggml-org/ci/tree/results/llama.cpp/0e/8d8bfd6caf1d0a8cbdf9d3d5c06fbbb9dfced8/ggml-4-x86-cuda-v100)
   - https://github.com/ggerganov/llama.cpp/commit/0e8d8bfd6caf1d0a8cbdf9d3d5c06fbbb9dfced8

Noticed that it said it's failing in `20 - test-backend-ops`, it be good to identify the cause of this issue and potential ways to fix it. The failure in test *#20* in test-backend-ops looked like below which doesn't seem to explain much to me. But hopefully it makes sense to someone else here.

`[CPY] NMSE = 0.000003149 > 0.000000100` looks inter

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: Docker images on GHCR stuck at **b5174** – “Publish Docker image” workflow failing since 2025‑04‑24

**Link**: https://github.com/ggml-org/llama.cpp/issues/13203
**State**: closed
**Created**: 2025-04-30T02:02:06+00:00
**Closed**: 2025-04-30T08:44:08+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

### Name and Version

version: 5174 (56304069)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu


### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-cli

### Command line

```shell
llama-cli --version
```

### Problem description & steps to reproduce

### Summary
Pulling any of the moving Docker tags (`full-vulkan`, `full`, `server`, etc.) still returns **build 5174 (56304069)**, which was published on **2025‑04‑24**. Meanwhile, the Releases page has advanced to **b5223** and beyond, so no Docker images have been published for roughly a week.

---

### Steps to reproduce
```bash
# 1. Pull the latest image
docker pull ghcr.io/ggml-org/llama.cpp:full-vulkan

# 2. Check the build banner (bypasses tools.sh)
docker run --rm \
  --entrypoint /app/llama-cli \
  ghcr.io/ggml-org/llama.cpp:full-vulkan \
  --version
```
Output:
```
version: 5174 (56304069)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
``

[... truncated for brevity ...]

---

## Issue #N/A: Response 200 when using LLava via Server for python

**Link**: https://github.com/ggml-org/llama.cpp/issues/3800
**State**: closed
**Created**: 2023-10-26T17:51:11+00:00
**Closed**: 2023-10-27T03:30:59+00:00
**Comments**: 5

### Description

Hello, i am having a problem using LLava via Server.

server launch command
```
export PYTHONPATH=$PYTHONPATH:`pwd`
export CUDA_VISIBLE_DEVICES=4,5,6,7
./server -m models/llava/ggml-model-q4_k.gguf --mmproj models/llava/mmproj-model-f16.gguf -c 2048 --port 8080 -ngl 35 -mg 3 -t 20
```

request code
```
import base64
from io import BytesIO
from PIL import Image
import subprocess
import json

def image_to_base64(img_path):
    with Image.open(img_path) as image:
        image = image.resize((336,336))
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode('utf-8')

payload = {
    "prompt": "User: [img-1]Describe the image about indoor house scene. You must describe what you can see (e.g., door, stairs, bed), and where are you in (e.g., bathroom, kitchen). Only say description, do not say anything else.",
    "temperature": 0.1,
    "image_data": [{"data": image_

[... truncated for brevity ...]

---

## Issue #N/A: CUDA error: an illegal memory access was encountered (with large prompts)

**Link**: https://github.com/ggml-org/llama.cpp/issues/13851
**State**: closed
**Created**: 2025-05-28T10:30:18+00:00
**Closed**: 2025-05-30T16:56:20+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

### Name and Version

./build/bin/llama-cli --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: yes
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 3060, compute capability 8.6, VMM: yes
version: 5518 (26b79b6c)
built with cc (Debian 12.2.0-14+deb12u1) 12.2.0 for x86_64-linux-gnu

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-cli

### Command line

```shell
llama-cli -t 4 --flash-attn --color --conversation --multiline-input --mirostat 2 --tensor-split 1.2,0.8 --ctx-size $((8192*20)) --n-gpu-layers 66 --temp 0.9 -m /work/models/misc/gemma/gemma-3-12B-it-Q6_KLA.gguf
```

### Problem description & steps to reproduce

Large prompt (above 16k) consistently fails with error:
- paste text >16
- crash


### First Bad Commit

952f3953c1b61cc70e79e536c42ddce6a5ea5ea7

### Relevant log output

```shell
work/src/llama

[... truncated for brevity ...]

---

## Issue #N/A: New IQ1_S somehow much worse than previous version

**Link**: https://github.com/ggml-org/llama.cpp/issues/5996
**State**: closed
**Created**: 2024-03-11T11:40:53+00:00
**Closed**: 2024-05-09T01:06:24+00:00
**Comments**: 27
**Labels**: bug-unconfirmed, stale

### Description

Since #5971 I tried requantizing IQ1_S of [this](https://huggingface.co/CISCai/gorilla-openfunctions-v2-SOTA-GGUF) model, using the same [imatrix](https://huggingface.co/CISCai/gorilla-openfunctions-v2-SOTA-GGUF/resolve/main/gorilla-openfunctions-v2.imatrix.dat) as before, however, where the following worked as expected 75% of the time (and the rest of the time it just gave the wrong output):
```bash
./main --log-disable --no-display-prompt -t 7 -ngl 35 -m gorilla-openfunctions-v2.IQ1_S.gguf --color -c 16384 --temp 0 -p "You are an AI programming assistant, utilizing the Gorilla LLM model, developed by Gorilla LLM, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."$'\n''### Instruction: <<function>>[{"name":"get_current_weather","description":"Get the current weather in a given location","parameters":{"type":"object","properties":{"location":

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: Garbage output from Llama-3.2-1B-Instruct-Q4_K_M using GGML_VULKAN on M1 Mac

**Link**: https://github.com/ggml-org/llama.cpp/issues/11256
**State**: closed
**Created**: 2025-01-15T18:31:38+00:00
**Closed**: 2025-03-01T01:07:38+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

```
$  ./build/bin/llama-cli --version
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = Apple M1 Pro (MoltenVK) | uma: 1 | fp16: 1 | warp size: 32 | matrix cores: none
version: 4489 (f11cfdfd)
built with Apple clang version 16.0.0 (clang-1600.0.26.6) for arm64-apple-darwin24.2.0
``` 

### Operating systems

Mac

### GGML backends

Vulkan

### Hardware

Apple M1 Pro, 32 GB RAM

### Models

Meta Llama 3.2 Instruct 1B Q4_K_M

### Problem description & steps to reproduce

In a fresh git clone:

```
$ cmake -B build -DGGML_VULKAN=ON -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release -G Ninja
$ cmake --build build --config Release -j 8
$  ./build/bin/llama-cli -m ~/llamas/Llama-3.2-1B-Instruct-Q4_K_M.gguf -p "The capital of France is " --device Vulkan0 -ngl 17  -no-cnv --version
```

Result: prompt is echoed, but then generation is obvious nonsense tokens.

If I omit `--device Vulkan0 -ngl 17`, I get reasonable output, but I see
```

[... truncated for brevity ...]

---

## Issue #N/A: [User] Please fix segmentation fault when prompt is too long

**Link**: https://github.com/ggml-org/llama.cpp/issues/411
**State**: closed
**Created**: 2023-03-22T22:40:32+00:00
**Closed**: 2023-03-23T07:40:21+00:00
**Comments**: 7
**Labels**: bug, need more info

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

I want to be able to run my promt using this command without any `Segmentation fault` error: 
```bash
./main -m ./models/7B/ggml-model-q4_0.bin -t 8 -n 256 --repeat_penalty 1.0 --color -i -r "Prompt:" --temp 1.2 -p "$(cat ../twitch_bot/prompt.md)"
```
Where `promp

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: Hybrid models failing to load with assert GGML_ASSERT(kv_size % n_pad == 0)

**Link**: https://github.com/ggml-org/llama.cpp/issues/14724
**State**: closed
**Created**: 2025-07-16T15:11:45+00:00
**Closed**: 2025-07-16T19:17:26+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Name and Version

version: 5913 (225e7a14)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu


### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

libllama (core library)

### Command line

```shell
llama-cli -m /storage/models/textgen/ibm-granite_granite-4.0-tiny-preview-bf16.gguf --no-mmap --jinja -sys 'You are a helpful assistant'
```

### Problem description & steps to reproduce

Looks like [this line](https://github.com/ggml-org/llama.cpp/blob/b0f0ecc3dce806c68609d375a2b3edc430d8db18/src/llama-memory-hybrid.cpp#L43) to send unified as true was put in the wrong place maybe; should perhaps instead look like this:

```
    mem_attn(new llama_kv_cache_unified(
        model,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recurrent(il); }
            : filter_attn,
        type_k,
        type_v,
        v_trans,
        offload,
        1,
        kv_size,
        n_seq_max,
        n_pad,

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: server : make chat_example available through /props endpoint

**Link**: https://github.com/ggml-org/llama.cpp/issues/8694
**State**: closed
**Created**: 2024-07-25T21:04:21+00:00
**Closed**: 2024-09-18T01:07:11+00:00
**Comments**: 5
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Currently there is no way of retrieving information about the recommended chat template for a model when using the `/completion` endpoint of the server. The idea of this feature is to add a new property `chat_example` to the `/props` endpoint that returns the same information that is already logged when the server starts up:

```
chat_example="<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_s

[... truncated for brevity ...]

---

## Issue #N/A: Error: inlining failed in call to always_inline ‘_mm256_cvtph_ps’: target specific option mismatch

**Link**: https://github.com/ggml-org/llama.cpp/issues/107
**State**: closed
**Created**: 2023-03-13T23:20:27+00:00
**Closed**: 2023-03-14T18:08:16+00:00
**Comments**: 22
**Labels**: duplicate, good first issue, hardware, build

### Description

I cloned the GitHub repository and ran the make command but was unable to get the cpp files to compile successfully. Any help or suggestion would be appreciated.

Terminal output:
<pre><font color="#4E9A06"><b>brickman@Ubuntu-brickman</b></font>:<font color="#3465A4"><b>~/Desktop/llama.cpp</b></font>$ ls
CMakeLists.txt  convert-pth-to-ggml.py  ggml.c  ggml.h  LICENSE  main.cpp  Makefile  <font color="#3465A4"><b>models</b></font>  quantize.cpp  <font color="#4E9A06"><b>quantize.sh</b></font>  README.md  utils.cpp  utils.h
<font color="#4E9A06"><b>brickman@Ubuntu-brickman</b></font>:<font color="#3465A4"><b>~/Desktop/llama.cpp</b></font>$ make
I llama.cpp build info: 
I UNAME_S:  Linux
I UNAME_P:  x86_64
I UNAME_M:  x86_64
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -mavx2 -msse3
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread
I LDFLAGS:  
I CC:       cc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
I CXX:      g++ (Ubuntu 9.4.0-1u

[... truncated for brevity ...]

---

## Issue #N/A: SIGSEGV on moderately complex grammar 

**Link**: https://github.com/ggml-org/llama.cpp/issues/7810
**State**: open
**Created**: 2024-06-07T00:00:53+00:00
**Comments**: 1
**Labels**: bug

### Description

(In #7572 @HanClinto wrote:)
> > @HanClinto Further down in the gist https://gist.github.com/hoehrmann/f234c1156ee5ef7b24cb589c14aaefda?permalink_comment_id=5070397#gistcomment-5070397 is a variant where I removed the redundant empty string alternative. llamap.cpp then goes into SIGSEGV after a couple of lines (or pretty much immediately if you change the root to the `<middle>` section of the grammar) asking a model to write an Internet-Draft in xml2rfc format.
> 
> Confirmed, thank you! I ran the following command:
> 
> ```
> ./main -mu https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B-GGUF/resolve/main/Hermes-2-Pro-Mistral-7B.Q4_K_M.gguf \
>     --grammar-file ./grammars/issue7572.gbnf \
>     -p "Please generate an Internet-Draft in xml2rfc format." \
>     --seed 12345
> ```
> 
> And it crashed after:
> 
> ```
>  Please generate an Internet-Draft in xml2rfc format.<rfc>
> <front><title>Test</title><author><organization>IETF</organization><address><email>jd

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Weird output from llama-speculative

**Link**: https://github.com/ggml-org/llama.cpp/issues/8499
**State**: closed
**Created**: 2024-07-16T04:56:46+00:00
**Closed**: 2024-09-05T01:07:05+00:00
**Comments**: 15
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

Hello, llama.cpp experts! Thank you for creating such an amazing LLM Inference system. 😁
**However, while using this system, I encountered an unusual results when checking the speculative decoding output.**
I believe the observed issue is a bug and reporting it as a Bug ISSUE on this github project.

First of all, I want to provide a configuration of my system.
- OS: ubuntu 22.04
- CUDA: 12.4
- GPU: A100 80GB

Next, I will explain the steps I took to download and run the model until the bug occurred.
It was somewhat challenging to use the llama.cpp systems.
```
# download draft model
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir=./llama-1.1b
./venv/bin/python3 convert_hf_to_gguf.py ./llama-1.1b
```
```
# download target model
huggingface-cli download NousResearch/Llama-2-7b-hf --local-dir=./llama-7b
./venv/bin/python3 convert_hf_to_gguf.py ./llama-7b
```
```
# run llama-speculative
./build/bin/llama-speculative -m ./l

[... truncated for brevity ...]

---

## Issue #N/A: Bug: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED

**Link**: https://github.com/ggml-org/llama.cpp/issues/10080
**State**: closed
**Created**: 2024-10-29T04:30:35+00:00
**Closed**: 2024-12-15T01:07:48+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

Hi there.

My llama-server can work well with the following command:

```bash
/llama.cpp-b3985/build_gpu/bin/llama-server -m ../artifact/models/Mistral-7B-Instruct-v0.3.Q4_1.gguf -ngl 31 --threads 16 --batch-size 32 --ubatch-size 8
```

However, when I keep only the `ngl` parameter, my server crashes with confusing error message:

```bash
./llama.cpp-b3985/build_gpu/bin/llama-server -m ../artifact/models/Mistral-7B-Instruct-v0.3.Q4_1.gguf -ngl 31
```

I got an CUDA error: CUBLAS_STATUS_NOT_INITIALIZED:
```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3060, compute capability 8.6, VMM: yes
build: 0 (unknown) with cc (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0 for x86_64-linux-gnu
system info: n_threads = 6, n_threads_batch = 6, total_threads = 16

system_info: n_threads = 6 (n_threads_batch = 6) / 16 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX5

[... truncated for brevity ...]

---

## Issue #N/A: [BUG]  `n_predict` is not accurate when making inference using `server`.

**Link**: https://github.com/ggml-org/llama.cpp/issues/4790
**State**: closed
**Created**: 2024-01-06T04:56:01+00:00
**Closed**: 2024-01-06T17:14:10+00:00
**Comments**: 3
**Labels**: bug-unconfirmed

### Description

* llama.cpp version:

```
commit 012cf349aec8ffb47c9def5dc018240fa3721e8b (HEAD -> master, tag: b1767, origin/master, origin/HEAD)
Author: Georgi Gerganov <ggerganov@gmail.com>
Date:   Thu Jan 4 19:56:33 2024 +0200

    server : send token probs for "stream == false" (#4714)
```

* System:

macOS Sonoma

* What's the problem

The `server` doesn't accurately follow `n_predict`. For instance, if I set `n_predict=1`, it still generates 4 tokens:


* How to reproduce:

```
curl --request POST --url http://127.0.0.1:8080/completion --header "Content-Type: application/json" --data '{"prompt": "I believe the meaning of life is","n_predict": 1, "n_probs" : 3}' | jq
```

Result:

```
{
  "completion_probabilities": [
    {
      "content": " to",
      "probs": [
        {
          "prob": 0.9529879689216614,
          "tok_str": " to"
        },
        {
          "prob": 0.02805173397064209,
          "tok_str": " love"
        },
        {
        

[... truncated for brevity ...]

---

## Issue #N/A: server: support control vectors

**Link**: https://github.com/ggml-org/llama.cpp/issues/6316
**State**: open
**Created**: 2024-03-26T07:25:43+00:00
**Comments**: 0
**Labels**: enhancement, good first issue, server/webui

### Description

### Motivation

It would be nice to support control vectors in the servers.


### Requirements
- Configure `gpt_params::control_vectors` from `common`
- Tests the feature using the framework

#### References
- A first attemp has been made here: #6289

---

## Issue #N/A: [Bug(CMake 3.17)] CUDA::cublasLt not found but can be specified absolutely

**Link**: https://github.com/ggml-org/llama.cpp/issues/1078
**State**: closed
**Created**: 2023-04-20T10:52:42+00:00
**Closed**: 2024-06-06T01:07:09+00:00
**Comments**: 7
**Labels**: stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

When I run

```bash
cmake3 .. -DLLAMA_CUBLAS=ON
```

It will success normally.

# Current Behavior

It got errors (see details at below). But when I replace `CUDA::cublasLt` in CMakeLists.txt with the absolute path, the bug fixed. But I think it is not a goo

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

