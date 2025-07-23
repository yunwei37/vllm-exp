# very_long_over2000chars - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 3
- Closed Issues: 27

### Label Distribution

- bug-unconfirmed: 16 issues
- stale: 15 issues
- enhancement: 3 issues
- bug: 2 issues
- build: 2 issues
- critical severity: 2 issues
- help wanted: 1 issues
- good first issue: 1 issues
- question: 1 issues
- low severity: 1 issues

---

## Issue #N/A: Unable to run Baichuan2 with the latest code

**Link**: https://github.com/ggml-org/llama.cpp/issues/6091
**State**: closed
**Created**: 2024-03-15T20:31:20+00:00
**Closed**: 2024-03-15T21:14:17+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

llama.cpp is used to support Baichuan2 while the latest code throws an exception. I did a git bisect and found the issue:

```
* f30ea47a - (tag: b2413, refs/bisect/bad) llama : add pipeline parallelism support (#6017) (Wed Mar 13 18:54:21 2024 +0100) <slaren>
* d8fd0ccf - (tag: b2412, refs/bisect/good-d8fd0ccf6ac8b07791ffd1575eed436930854ae3) test-backend-ops : skip CPU backend by default (#6028) (Wed Mar 13 14:58:30 2024 +0100) <slaren>
```

Seems that the pipeline parallelism support breaks it.

Step to reproduce:

```bash
# convert Baichuan2 from HF format to GGUF
./convert-hf-to-gguf.py /hf/cache/Baichuan2-13B-Chat

# run the model
./main -m ggml-model-f16.gguf -p '你好'
```

Logs:

```
Log start
main: build = 2413 (f30ea47a)
main: built with cc (Ubuntu 11.3.0-1ubuntu1~22.04.1) 11.3.0 for x86_64-linux-gnu
main: seed  = 1710534860
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices

[... truncated for brevity ...]

---

## Issue #N/A: [User] Insert summary of your issue or enhancement..

**Link**: https://github.com/ggml-org/llama.cpp/issues/1983
**State**: closed
**Created**: 2023-06-24T13:08:38+00:00
**Closed**: 2023-06-24T22:11:26+00:00
**Comments**: 0

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ ] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ ] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Please provide a detailed written description of what you were trying to do, and what you expected `llama.cpp` to do.

# Current Behavior

Please provide a detailed written description of what `llama.cpp` did, instead.

# Environment and Context

Please provide detailed informat

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: Unusual high RAM usage on Windows when running DeepSeek V3 Q2_K_XL/IQ2_XXS, on Hybrid CPU+GPU

**Link**: https://github.com/ggml-org/llama.cpp/issues/13978
**State**: open
**Created**: 2025-06-02T18:59:56+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

llama-server --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 1: NVIDIA GeForce RTX 5080, compute capability 12.0, VMM: yes
version: 5572 (7675c555)
built with MSVC 19.44.35207.1 for x64

### Operating systems

Windows

### GGML backends

CUDA

### Hardware

AMD Ryzen 9 9950x3D CPU and 2 GPUs: Nvidia 5090 and 5080.

### Models

Unsloth models: IQ2_XXS and Q2_K_XL from Hugging Face from here: https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF-UD

### Problem description & steps to reproduce

I am encountering the same issue reported here: https://github.com/ggml-org/llama.cpp/issues/12651 The prior issue was closed, but is there a fix for this issue? I am experiencing this exact same issue running unsloth's DeepSeek-V3-0324-UD-Q2_K_XL model with 2 GPUs (Nvidia RTX 5090 and 5080). I have tried setting 

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: ggml_vulkan: Device memory allocation of size N failed with ub > 4096 and c > 4096 and b > 4096

**Link**: https://github.com/ggml-org/llama.cpp/issues/12817
**State**: closed
**Created**: 2025-04-08T08:09:05+00:00
**Closed**: 2025-05-28T01:07:54+00:00
**Comments**: 14
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

version: 5061 (916c83bf)
built with MSVC 19.38.33134.0 for x64

### Operating systems

Windows

### GGML backends

Vulkan

### Hardware

Ryzen 7 5800H + AMD Radeon RX 6600M

### Models

Any model

### Problem description & steps to reproduce

When trying to run llama-server with `-ub 8192 -b 8192 -c 8192`, it crashes with `ggml_vulkan: Device memory allocation of size 3959422976 failed.` with any model I try (the allocation size differs between models), even though I have enough GPU memory after model is loaded.

I tried smaller models to exclude possible OOM (the log includes nomic-embed-text-v1.5) and I see that ~100mb of VRAM gets allocated for a model (0.9GB used), then it crashes when trying to allocate 3959422976 bytes.

When setting any of these parameters to 4096, the model loads successfully.

The same occurs with any model. Tried with Qwen2.5 3B Q8_0 and nomic-embed-text-v1.5 Q8_0.

### First Bad Commit

_No response_

### Relevant log output

```shell
.

[... truncated for brevity ...]

---

## Issue #N/A: Build fails with `ggml-vulkan.cpp:6880:80: error: cannot convert ‘ggml_tensor*’ to ‘float’`

**Link**: https://github.com/ggml-org/llama.cpp/issues/7446
**State**: closed
**Created**: 2024-05-21T21:08:34+00:00
**Closed**: 2024-05-22T08:40:00+00:00
**Comments**: 4
**Labels**: bug-unconfirmed

### Description

I am on Artix GNU/Linux (rolling release), GCC 14.1.1, and I build [`ollama-vulkan`](https://aur.archlinux.org/pkgbase/ollama-nogpu-git) which pulls in and uses `llama.cpp` from this git repository.

When building, I get the error  
`ggml-vulkan.cpp:6880:80: error: cannot convert ‘ggml_tensor*’ to ‘float’`:  
```
[...]
+ init_vars
+ case "${GOARCH}" in
+ ARCH=x86_64
+ LLAMACPP_DIR=../llama.cpp
+ CMAKE_DEFS=
+ CMAKE_TARGETS='--target ollama_llama_server'
+ echo ''
+ grep -- -g
+ CMAKE_DEFS='-DCMAKE_BUILD_TYPE=Release -DLLAMA_SERVER_VERBOSE=off '
+ case $(uname -s) in
++ uname -s
+ LIB_EXT=so
+ WHOLE_ARCHIVE=-Wl,--whole-archive
+ NO_WHOLE_ARCHIVE=-Wl,--no-whole-archive
+ GCC_ARCH=
+ '[' -z '50;52;61;70;75;80' ']'
+ echo 'OLLAMA_CUSTOM_CPU_DEFS="
  -DBUILD_TESTING=ON
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX=/usr
  -DLLAMA_ACCELERATE=ON
  -DLLAMA_ALL_WARNINGS=OFF
  -DLLAMA_ALL_WARNINGS_3RD_PARTY=OFF
  -DLLAMA_FATAL_WARNINGS=OFF
  -DLLAMA_AVX=ON -D

[... truncated for brevity ...]

---

## Issue #N/A: clBLAST builds only output "######..." regression since the end of December 2023 (CPU still good, old commit clBLAST still good)

**Link**: https://github.com/ggml-org/llama.cpp/issues/5355
**State**: closed
**Created**: 2024-02-06T02:01:17+00:00
**Closed**: 2024-04-02T01:07:08+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale

### Description

I don't know where exactly it started but every llama.cpp version after ~2023-12-19 I've tested has a broken clBLAST such that no matter what model type is loaded the only tokens sampled and output are "#". Just endless #############... until it gets to the npredict token limit or otherwise cut off. 

Both main and server binaries do this with both mistral-7B and llama2-7B and 13B models. The same exact commands work perfect on the llama.cpp main and server binaries from build = 1662 (commit 328b83d)  2012-12-19. When I build a cpu only version of the broken build 2061 (commit 9392ebd4) with no clblast the same exact same command work as expected and produce coherent output. It seems clear something changed about how opencl clBLAST works with llama.cpp in Jan 2024. It might be a change specific to my particular hardware or at least particular to AMD hardware.

I have an AMD RX 580 8GB GPU (Ellesmere) I'm using with clBLAST. The system is a Ryzen 5 3600 with 32GB of ram. It is runni

[... truncated for brevity ...]

---

## Issue #N/A: Compile bug: Vulkan shaders not compiling any more on Debian Stable (12/bookworm)

**Link**: https://github.com/ggml-org/llama.cpp/issues/11052
**State**: closed
**Created**: 2025-01-03T07:42:43+00:00
**Closed**: 2025-01-08T08:18:14+00:00
**Comments**: 9
**Labels**: bug-unconfirmed

### Description

### Git commit

$ git rev-parse HEAD
5437d4aaf5132c879acda0bb67f2f8f71da4c9fe

### Operating systems

Linux

### GGML backends

Vulkan

### Problem description & steps to reproduce

On an up-to-date Debian Bookworm, Vulkan shaders do not compile any more. After some digging, this seems to be related to changes introduced in b4280 (3df784b3050f657ea681f804187ce5bddb433e88) where the GL_KHR_cooperative_matrix extension is being used.

Without being familiar with Vulkan, my understanding is that these extensions started to be introduced with Vulkan 1.3.255 (see https://www.phoronix.com/news/Vulkan-1.3.255), but Vulkan on the current Debian Stable has version 1.3.239. Here are the versions of various packages which may be related:

```
$ sudo apt list libvulkan1 mesa-vulkan-drivers glslc 
glslc/stable,now 2023.2-1 amd64 [installed]
libvulkan1/stable,now 1.3.239.0-1 amd64 [installed,automatic]
mesa-vulkan-drivers/stable,now 22.3.6-1+deb12u1 amd64 [installed]
```

In order to rep

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: Segmentation fault with docker aarch64 on MacOS M1 using a small test model stories15M_MOE-Q8_0.gguf

**Link**: https://github.com/ggml-org/llama.cpp/issues/11082
**State**: closed
**Created**: 2025-01-05T06:09:24+00:00
**Closed**: 2025-02-23T01:07:39+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

version: 4410 (4b0c638b)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for aarch64-linux-gnu

### Operating systems

Mac

### GGML backends

CPU

### Hardware

MacOS M1

### Models

https://huggingface.co/ggml-org/stories15M_MOE stories15M_MOE-Q8_0.gguf

### Problem description & steps to reproduce

1. download the gguf model
   ```shell
   mkdir -p models
   MODEL=stories15M_MOE-Q8_0.gguf && curl -sL -o models/$MODEL "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/$MODEL?download=true"
   ```
2. run with docker on aarch64 - it fails
   ```shell
   docker run --platform linux/aarch64 --rm -it --name llama.cpp-full -v $PWD/models:/models ghcr.io/ggerganov/llama.cpp:full-b4410 --run -m /models/stories15M_MOE-Q8_0.gguf -p "Building a website can be done in 10 simple steps:"
   ...
   echo $?
   139
   ```

   When executing the run in the container with bash, it additionally prints "Segmentation fault (core dumped)"
   ```shell
   d

[... truncated for brevity ...]

---

## Issue #N/A: [Nem_pickaxe] ICall to Undeclared Functions and Implicit Function Declarations

**Link**: https://github.com/ggml-org/llama.cpp/issues/2481
**State**: closed
**Created**: 2023-08-01T15:16:22+00:00
**Closed**: 2024-04-09T01:07:11+00:00
**Comments**: 4
**Labels**: stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

When compiling the code, the following errors and warnings are encountered:

ggml.c:4355:9: Error: Call to undeclared function 'ggml_init_cublas'; ISO C99 and later do not support implicit function declarations.
ggml.c:14608:21: Error: Call to undeclared function 'ggml_cuda_compute_forwa

[... truncated for brevity ...]

---

## Issue #N/A: Compile bug: I tried compiling llama.cpp for HIP on my system (elementaryOS 8/ubuntu 24.04, rocm 6.4.0, gfx1100) using the installation guide

**Link**: https://github.com/ggml-org/llama.cpp/issues/13340
**State**: closed
**Created**: 2025-05-06T13:33:30+00:00
**Closed**: 2025-06-23T01:08:03+00:00
**Comments**: 8
**Labels**: bug-unconfirmed, stale

### Description

### Git commit

$ git rev-parse HEAD

36c258ee921dbb5c96bdc57c0872e4a9a129bef6

### Operating systems

Linux

### GGML backends

HIP

### Problem description & steps to reproduce

I tried compiling llama.cpp for HIP on my system (elementaryOS 8/ubuntu 24.04, rocm 6.4.0, gfx1100) using the installation guide : https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md

It gives multiple errors and exits midway (it seems). I dont have to skills and insights to know what went wrong. 

This bug reporting UI doesn't let me place the full output because it exceeds character limit. I have pasted the last few lines of the error output. 

### First Bad Commit

_No response_

### Compile command

```shell
$ HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_ROCWMMA_FATTN=ON && cmake --build build --config Release -- -j 16
```

### Relevant log output

```shell
[ 45%] Linking CXX executabl

[... truncated for brevity ...]

---

## Issue #N/A: Exception: Unexpected tensor name: model.mm_projector.weight

**Link**: https://github.com/ggml-org/llama.cpp/issues/5248
**State**: closed
**Created**: 2024-01-31T22:19:38+00:00
**Closed**: 2024-02-02T09:02:53+00:00
**Comments**: 4
**Labels**: bug-unconfirmed

### Description

When trying to convert https://huggingface.co/xinlai/LISA-7B-v1 I get the following error:


python convert.py ../models/LISA-7B-v1/
Loading model file ..\models\LISA-7B-v1\pytorch_model-00001-of-00002.bin
Loading model file ..\models\LISA-7B-v1\pytorch_model-00001-of-00002.bin
Loading model file ..\models\LISA-7B-v1\pytorch_model-00002-of-00002.bin
params = Params(n_vocab=32004, n_embd=4096, n_layer=32, n_ctx=2048, n_ff=11008, n_head=32, n_head_kv=32, n_experts=None, n_experts_used=None, f_norm_eps=1e-06, rope_scaling_type=None, f_rope_freq_base=None, f_rope_scale=None, n_orig_ctx=None, rope_finetuned=None, ftype=None, path_model=WindowsPath('../models/LISA-7B-v1'))
Found vocab files: {'tokenizer.model': WindowsPath('../models/LISA-7B-v1/tokenizer.model'), 'vocab.json': None, 'tokenizer.json': None}
Loading vocab file '..\models\LISA-7B-v1\tokenizer.model', type 'spm'
Vocab info: <SentencePieceVocab with 32000 base tokens and 4 added tokens>
Special vocab info: <SpecialVoca

[... truncated for brevity ...]

---

## Issue #N/A:  incompatible types when initializing type ‘__m256i {aka __vector(4) long long int}’

**Link**: https://github.com/ggml-org/llama.cpp/issues/1279
**State**: closed
**Created**: 2023-05-02T14:38:01+00:00
**Closed**: 2023-07-28T19:53:52+00:00
**Comments**: 5
**Labels**: bug, build

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

successful compilation of llama.cpp

# Current Behavior

sh-4.2$ make
I llama.cpp build info:
I UNAME_S: Linux
I UNAME_P: x86_64
I UNAME_M: x86_64
I CFLAGS: -I. -O3 -std=c11 -fPIC -DNDEBUG -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -

[... truncated for brevity ...]

---

## Issue #N/A: Library not loaded: @rpath/libllama.dylib

**Link**: https://github.com/ggml-org/llama.cpp/issues/11321
**State**: closed
**Created**: 2025-01-20T21:41:04+00:00
**Closed**: 2025-01-25T13:21:45+00:00
**Comments**: 5
**Labels**: bug-unconfirmed

### Description

### Name and Version

I just downloaded the latest binary (b4519) to play around with on my Mac, however it looks like the binary didn't quite compile correctly. I tried downloading an earlier version (b4514) with the same result. Running `llamba-cli` yields:

```
➜  llama.cpp ./llama-cli --version
dyld[90496]: Library not loaded: @rpath/libllama.dylib
  Referenced from: <653E6B29-4AFF-3485-B031-B4F65747F8CF> /Users/constantmeiring/Downloads/build/llama.cpp/llama-cli
```



### Operating systems

MacOS 15.1.1 (24B91)

### GGML backends

Metal

### Hardware

Macbook - M3 Max

### Models

_No response_

### Problem description & steps to reproduce

Download the latest build and try and run it on Mac.

### First Bad Commit

_No response_

### Relevant log output

```shell
dyld[90496]: Library not loaded: @rpath/libllama.dylib
  Referenced from: <653E6B29-4AFF-3485-B031-B4F65747F8CF> /Users/constantmeiring/Downloads/build/llama.cpp/llama-cli
  Reason: tried: '/Users/runner/work/llama.cpp/l

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: Llama-Quantize.exe broken on win11 since b5298 , but works on/earlier b5215

**Link**: https://github.com/ggml-org/llama.cpp/issues/13518
**State**: closed
**Created**: 2025-05-14T01:29:39+00:00
**Closed**: 2025-05-14T14:12:38+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Name and Version

Please note that llama-quantize.exe is failing from version b5298 (perhaps earlier) on windows 11 systems.
I have also tested: b5342 , b5361, b5371








### Operating systems

Windows

### Which llama.cpp modules do you know to be affected?

llama-quantize

### Command line

```shell
./llama-quantize E:/main-du.gguf i:/llm/David_AU/testfiles/MN-Dark-Universe-MOE-4X12B-Reasoning-Q2_K.gguf Q2_K 8
```

### Problem description & steps to reproduce

ISSUE: 

Example:
./llama-quantize E:/main-du.gguf i:/llm/David_AU/testfiles/MN-Dark-Universe-MOE-4X12B-Reasoning-Q2_K.gguf Q2_K 8

(used in powershell)

Generates:

main: build = 5371 (e5c834f7)
main: built with MSVC 19.29.30159.0 for Windows AMD64
main: quantizing 'E:/main-du.gguf' to 'i:/llm/David_AU/testfiles/MN-Dark-Universe-MOE-4X12B-Reasoning2-Q2_K.gguf' as Q2_K using 8 threads
llama_model_loader: loaded meta data with 34 key-value pairs and 403 tensors from E:/main-du.gguf (version GGUF V3 (latest))
llama_model_l

[... truncated for brevity ...]

---

## Issue #N/A: Using MPI w/ 65b model but each node uses the full RAM.

**Link**: https://github.com/ggml-org/llama.cpp/issues/2209
**State**: open
**Created**: 2023-07-13T02:57:34+00:00
**Comments**: 3
**Labels**: help wanted

### Description

I am trying to use MPI but each node uses the full RAM. Is this how MPI is supposed to work? I didn't think it was. Here's the details.

I am on commit 1cbf561466e957b25f0e8163c2386683f8674369. I modified the Makefile so I could compile it like this (see https://github.com/ggerganov/llama.cpp/pull/2208).

```
LLAMA_MPI=1 LLAMA_METAL=1 make CC=/opt/homebrew/bin/mpicc CXX=/opt/homebrew/bin/mpicxx 
```

I run the following.

```
mpirun -hostfile hostfile -n 3 ./main -m airoboros-65B-gpt4-1.2.ggmlv3.q4_0.bin -n 128 -p "Q. What is the capital of Germany? A. Berlin. Q. What is the capital of France? A."
```

This is the output. It works, but each node uses 39 GB of RAM. Each node has 16 GB of RAM, so they swap bad.

```
main: build = 827 (1cbf561)
main: seed  = 1689216374
main: build = 827 (1cbf561)
main: seed  = 1689216374
main: build = 827 (1cbf561)
main: seed  = 1689216374
llama.cpp: loading model from airoboros-65B-gpt4-1.2.ggmlv3.q4_0.bin
llama.cpp: loading model

[... truncated for brevity ...]

---

## Issue #N/A: Phi-2 Quantization of QLoRA model Fails

**Link**: https://github.com/ggml-org/llama.cpp/issues/4615
**State**: closed
**Created**: 2023-12-24T05:41:18+00:00
**Closed**: 2024-04-02T01:10:06+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

The `convert-hf-to-gguf.py` script is supposed to convert any Huggingface PyTorch or Safetensors format model to a FP16 GGUF. It should convert both base Phi models, as well as fine-tunes using methods such as QLoRA. For example, [cognitivecomputations/dolphin-2_6-phi

[... truncated for brevity ...]

---

## Issue #N/A: convert.py incorrectly detects LLaMAv1 65B as a LLaMAv2 model

**Link**: https://github.com/ggml-org/llama.cpp/issues/3326
**State**: open
**Created**: 2023-09-24T14:02:55+00:00
**Comments**: 2
**Labels**: good first issue

### Description

# Expected Behavior

When converting LLaMA v1 65B, the model should be correctly detected as v1 and the max ctx should be set to 2048.

# Current Behavior

`convert.py` [seems to use the norm_eps value to detect if a model is v1 or v2.](https://github.com/ggerganov/llama.cpp/blob/master/convert.py#L253-L262)

I am using the original facebook PTH files as a source for the conversion, and it seems like the v1 65B model has the same eps as the v2 70B one, which means it gets mis-detected as a v2 model.

The same difference is present in the config for the HF transformer JSON, at least the one on Huggingface.

 - [65B json with `rms_norm_eps: 1e-05`](https://huggingface.co/huggyllama/llama-65b/blob/main/config.json)
 - [30B json with `rms_norm_eps: 1e-06`](https://huggingface.co/huggyllama/llama-30b/blob/main/config.json)

# Steps to Reproduce

1. Convert LLaMA v1 65B from source PTH files using `./convert.py`
2. Check the `general.name` field in the metadata of the resul

[... truncated for brevity ...]

---

## Issue #N/A: ggml_opencl error -1 on Intel Raptor Lake-P [Iris Xe Graphics]

**Link**: https://github.com/ggml-org/llama.cpp/issues/3936
**State**: closed
**Created**: 2023-11-03T16:29:41+00:00
**Closed**: 2024-04-02T01:12:18+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Start the model in interactive mode and use opencl and -ngl to offload 10 layers to integrated gpu.

# Current Behavior

opencl failed to initialize, for my gpu/cpu it is reproducible with:

```
$ nix run github:ggerganov/llama.cpp#opencl
Log start
main: buil

[... truncated for brevity ...]

---

## Issue #N/A: mac m1 series bug

**Link**: https://github.com/ggml-org/llama.cpp/issues/5236
**State**: closed
**Created**: 2024-01-31T10:21:53+00:00
**Closed**: 2024-01-31T12:32:35+00:00
**Comments**: 2
**Labels**: enhancement

### Description

llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 7
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: n_ff             = 20480
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 5000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: model type       = 30B
llm_load_print_meta: model ftype      = Q8_0
llm_load_print_meta: model params     = 34.39 B
llm_load_print_meta: model size       = 34.03 GiB (8.50 BPW) 
llm_lo

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Failed to convert minicpm-v2.5

**Link**: https://github.com/ggml-org/llama.cpp/issues/9098
**State**: closed
**Created**: 2024-08-20T06:21:44+00:00
**Closed**: 2024-10-17T01:21:24+00:00
**Comments**: 5
**Labels**: stale, critical severity

### Description

### What happened?

Follow the steps in [README-minicpmv2.5.md#usage](https://github.com/ggerganov/llama.cpp/blob/master/examples/llava/README-minicpmv2.5.md#usage) to convert `minicpm v2.5`. The conversion process fails while running the command: `python ./convert_hf_to_gguf.py ../MiniCPM-Llama3-V-2_5/model`. Specifically, the error happened after input `y` to answer the question `Do you wish to run the custom code? [y/N]`. 

### Name and Version

version: 3604 (1b6ff90f)
built with cc (Ubuntu 11.2.0-19ubuntu1) 11.2.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
azureuser@sam-llm:/data/sam/llama.cpp$ python ./convert_hf_to_gguf.py /data1/sam/models/MiniCPM-Llama3-V-2_5/model
INFO:hf-to-gguf:Loading model: model
INFO:gguf.gguf_writer:gguf: This GGUF file is for Little Endian only
INFO:hf-to-gguf:Exporting model...
INFO:hf-to-gguf:gguf: loading model weight map from 'model.safetensors.index.json'
INFO:hf-t

[... truncated for brevity ...]

---

## Issue #N/A: A special token '\u0000' will cause an assert error in 'llm_load_vocab'

**Link**: https://github.com/ggml-org/llama.cpp/issues/5112
**State**: closed
**Created**: 2024-01-24T14:24:39+00:00
**Closed**: 2024-04-02T01:08:27+00:00
**Comments**: 4
**Labels**: stale

### Description

### Discussed in https://github.com/ggerganov/llama.cpp/discussions/5111

<div type='discussions-op-text'>

<sup>Originally posted by **SolenoidWGT** January 24, 2024</sup>
I'm trying to fit an [InternLM2](https://github.com/InternLM/InternLM) model for llama.cpp, but I get an assertion error when using llama.cpp for inference, below is the error stack. The commit ID of llama.cpp code is 77bc1bbd05f0c31cb45773eb5eb59b9ff2b07e1b
```
$  ./main -m  ./internlm2-base-7b/ggml-model-f16.gguf -n 400  -e -p "Building a website can be done in 10 simple steps:\nStep 1:"
Log start
main: build = 1930 (f8ca46e0)
main: built with gcc (GCC) 10.2.0 for x86_64-pc-linux-gnu
main: seed  = 1706096206
llama_model_loader: loaded meta data with 17 key-value pairs and 227 tensors from ./internlm2-base-7b/ggml-model-f16.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                     

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: Completion fails with error 500

**Link**: https://github.com/ggml-org/llama.cpp/issues/14298
**State**: closed
**Created**: 2025-06-20T13:33:39+00:00
**Closed**: 2025-06-23T09:27:36+00:00
**Comments**: 8
**Labels**: bug

### Description

### Name and Version

`build: 5686 (e434e691) with cc (GCC) 15.1.1 20250425 for x86_64-pc-linux-gnu`

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell
llama-server --fim-qwen-1.5b-default
```

### Problem description & steps to reproduce

When using llama with Qwen 1.5b FIM model with llama.vscode, I get, after a couple of minutes, an error 500 on the completion endpoint.

### First Bad Commit

it seems to be happening around : 
```(llama-cpp-scripts-py3.11) ➜  llama.cpp git:(b5675) git bisect bad
3555b3004ba7687be3d734acade52a3345758aa4 is the first bad commit
commit 3555b3004ba7687be3d734acade52a3345758aa4 (HEAD, tag: b5675)
Author: xctan <xc-tan@outlook.com>
Date:   Mon Jun 16 13:54:15 2025 +0800

    ggml-cpu : rework weak alias on apple targets (#14146)
    
    * ggml-cpu : rework weak alias on apple targets
    
    * fix powerpc detection
    
    * fix ppc detection
    
    * fix powerpc detection 

[... truncated for brevity ...]

---

## Issue #N/A: Comparison of Windows Build VS Unix Build (through WSL2)

**Link**: https://github.com/ggml-org/llama.cpp/issues/507
**State**: closed
**Created**: 2023-03-25T20:09:51+00:00
**Closed**: 2024-04-12T01:07:40+00:00
**Comments**: 24
**Labels**: question, build, stale

### Description

# Environment and Context 
Hello, 
Before jumping to the subject, here's the environnement I'm working with:

- Windows 10
- Llama-13b-4bit-(GPTQ quantized) model
- Intel® Core™ i7-10700K [AVX | AVX2 | FMA | SSE3 | F16C]

# Expected Behavior

I did some comparaisons between the Windows build and the Unix build (through WSL2 Ubuntu_2204.1.8.0_x64) to see if I can notice some differences between them.

# Deterministic Settings (seed =1)
For both of those builds, I added the same exact settings:
```
-t 14 -n 2024 -c 2024 --temp 0.2 --top_k 40 --top_p 0.6 --repeat_last_n 2048 
--repeat_penalty 1.17647058824 --color --n_parts 1 -b 500 --seed 1 -p "$(cat STORY.txt)"
```

With the contents of STORY.txt as follows:
```
Here's 5 reasons that proves why video-games are good for your brain:
```

#  Test#1: Instruction set architectures

Windows:
```
system_info: n_threads = 14 / 16 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 0 | 
NEON = 0 | ARM_FMA = 0 | F16C = 0 | FP16

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: SwiftKV support (~2x performance boost)

**Link**: https://github.com/ggml-org/llama.cpp/issues/11415
**State**: closed
**Created**: 2025-01-25T14:07:14+00:00
**Closed**: 2025-03-18T01:07:44+00:00
**Comments**: 2
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

blog post: https://www.snowflake.com/en/engineering-blog/swiftkv-llm-compute-reduction/

full paper: https://arxiv.org/abs/2410.03960

Snowflake documented a new KV-cache optimization that can yield significant performance improvements. They're already integrating this into vLLM.

Specifically, Snowflake has introduced SwiftKV, a method designed to address the computational bottleneck associated with processing long input prompts during inference. In many enterprise use cases, the number of promp

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: DeepSeek-R1-UD-Q2_K_XL output broken

**Link**: https://github.com/ggml-org/llama.cpp/issues/13305
**State**: closed
**Created**: 2025-05-04T16:17:10+00:00
**Closed**: 2025-05-05T20:32:15+00:00
**Comments**: 13
**Labels**: bug-unconfirmed

### Description

### Name and Version

I experience gibberish with [DeepSeek-R1-UD-Q2_K_XL by unsloth](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-Q2_K_XL) (checked with SHA256)

In my case, this gibberish output started with [e1e8e09](https://github.com/ggml-org/llama.cpp/commit/e1e8e0991ffd9e99a445c6812bb519d5bac9f4b5).

I eventually managed to isolate the latest still working commit: [6f67cf1](https://github.com/ggml-org/llama.cpp/commit/6f67cf1f480926391ad75ff746e0a021214bf70c)

The most recent tested commit which is **still not working** is [9f2da58](https://github.com/ggml-org/llama.cpp/commit/9f2da5871f4bbd205b8a3b952cdc76283218d595)

![Image](https://github.com/user-attachments/assets/4d83b9f6-e6c1-45c6-ab91-ad16a5aa6e70)



### Operating systems

Linux

### GGML backends

CUDA

### Hardware

1x RTX 3090, Intel Xeon E5-2640 v3, 1TB RAM

### Models

[DeepSeek-R1-UD-Q2_K_XL by unsloth](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-Q2_K_XL)



[... truncated for brevity ...]

---

## Issue #N/A: Support LLaVA-UHD

**Link**: https://github.com/ggml-org/llama.cpp/issues/6153
**State**: closed
**Created**: 2024-03-19T07:02:19+00:00
**Closed**: 2024-05-03T01:06:31+00:00
**Comments**: 2
**Labels**: enhancement, stale

### Description

https://github.com/thunlp/LLaVA-UHD

This method is seemingly on par with or better than LLaVA 1.6 Next, however they opensourced the training code for reproduction.

> LLM analysis from Gemini 1.5 pro:

> | Feature        | LLaVA-UHD-13B | LLaVA-NeXT-7B                  | LLaVA-NeXT-13B | LLaVA-NeXT-34B | LLaVA 1.5-13B |
> | -------------- | ------------- | ------------------------------ | -------------- | -------------- | ------------- |
> | **VQAv2**      | 81.7          | 81.8 (Vicuna) / 82.2 (Mistral) | **82.8**       | ***83.7***     | 80            |
> | **GQA**        | **65.2**      | 64.2 (Vicuna) / 64.8 (Mistral) | 65.4           | ***67.1***     | 63.3          |
> | **TextVQA**    | **67.7**      | 64.9 (Vicuna) / 65.7 (Mistral) | 67.1           | ***69.5***     | 61.3          |
> | **ScienceQA**  | 72            | 70.1 (Vicuna) / 72.8 (Mistral) | **73.6**       | ***81.8***     | 71.6          |
> | **VizWiz**     | 56.1          | 57.6 (Vicuna) / 60.0 (Mistr

[... truncated for brevity ...]

---

## Issue #N/A: Bug: llava.cpp Segmentation fault (core dumped) starting in faf69d4237c9ae4d7f572b4674d1002463e8acd3

**Link**: https://github.com/ggml-org/llama.cpp/issues/9436
**State**: closed
**Created**: 2024-09-11T15:23:13+00:00
**Closed**: 2024-09-11T15:52:14+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

I am getting Segmentation fault (core dumped) when running llama-llava-cli and llama-minicpmv-cli starting in faf69d4237c9ae4d7f572b4674d1002463e8acd3. After reviewing faf69d4237c9ae4d7f572b4674d1002463e8acd3, I think the problem is related to [these lines](https://github.com/ggerganov/llama.cpp/blob/8db003a19d7055b5bd248ce2afff9324e5b8da95/src/llama.cpp#L16079) in the llama.cpp that try to access tokens when only image emb are given

```cpp
    for (uint32_t i = 0; i < n_tokens_all; ++i) {
        if (batch_all.token[i] < 0 || (uint32_t)batch_all.token[i] >= lctx.model.vocab.n_vocab) {
            LLAMA_LOG_ERROR("%s: invalid token[%d] = %d", __func__, i, batch_all.token[i]);
            return -1;
        }
    }
```

### Name and Version

~/llama.cpp$ ./llama-cli --version
version: 3731 (0996c559)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log o

[... truncated for brevity ...]

---

## Issue #N/A: Program hangs after some time

**Link**: https://github.com/ggml-org/llama.cpp/issues/1793
**State**: closed
**Created**: 2023-06-10T18:07:33+00:00
**Closed**: 2024-04-10T01:07:26+00:00
**Comments**: 4
**Labels**: stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Please provide a detailed written description of what you were trying to do, 

i was trying to run the program with this command:

```
./bin/main -m ./model.bin --gpu-layers 40 -n 256 --repeat_penalty 1.0 --color -i -r "User:" --in-prefix " " -f ./chat-with-b

[... truncated for brevity ...]

---

## Issue #N/A: api_like_OAI.py different GPT-GUIs hanging in response

**Link**: https://github.com/ggml-org/llama.cpp/issues/3654
**State**: closed
**Created**: 2023-10-17T15:48:47+00:00
**Closed**: 2024-04-04T01:08:04+00:00
**Comments**: 6
**Labels**: stale

### Description

# Prerequisites

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Current Behavior
I am trying to use different chatgpt GUIs with the api_like_OAI.py. To do this, I change the api_base to my api_like_OAI endpoint. The strange thing is that it works for some applications and not for others. The APP-ChatBoost works very well with api_like_OAI on Android, as well as with the "continue" plugin on VS Code. However, when I try the same wi

[... truncated for brevity ...]

---

## Issue #N/A: Bug: LLAMAFILE = 0 in `llama_print_system_info` even though compiled with `-DGGML_LLAMAFILE=ON` and can see set during compilation

**Link**: https://github.com/ggml-org/llama.cpp/issues/8656
**State**: closed
**Created**: 2024-07-23T19:44:56+00:00
**Closed**: 2024-07-25T09:37:43+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, low severity

### Description

### What happened?

In `llama_print_system_info` in llama.cpp, the output is `LLAMAFILE = 0`, even though can see that it's enabled (from the output `Using llamafile` when building).

```
$ cmake -B build -DGGML_LLAMAFILE=ON
-- Accelerate framework found
-- Metal framework found
-- Could NOT find OpenMP_C (missing: OpenMP_C_FLAGS OpenMP_C_LIB_NAMES)
-- Could NOT find OpenMP_CXX (missing: OpenMP_CXX_FLAGS OpenMP_CXX_LIB_NAMES)
-- Could NOT find OpenMP (missing: OpenMP_C_FOUND OpenMP_CXX_FOUND)
CMake Warning at ggml/src/CMakeLists.txt:151 (message):
  OpenMP not found


-- BLAS found, Libraries: /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX14.5.sdk/System/Library/Frameworks/Accelerate.framework
-- BLAS found, Includes:
-- Using llamafile
-- Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF
-- CMAKE_SYSTEM_PROCESSOR: arm64
-- ARM detected
-- Configuring do

[... truncated for brevity ...]

---

