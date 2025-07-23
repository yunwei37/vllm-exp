# normal_1to7days - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- bug-unconfirmed: 16 issues
- low severity: 2 issues
- enhancement: 2 issues
- Vulkan: 1 issues
- duplicate: 1 issues
- build: 1 issues
- help wanted: 1 issues
- high severity: 1 issues
- good first issue: 1 issues
- performance: 1 issues

---

## Issue #N/A: Is there a way to run ggml models on Intel ARC 770 GPU

**Link**: https://github.com/ggml-org/llama.cpp/issues/2552
**State**: closed
**Created**: 2023-08-08T12:58:18+00:00
**Closed**: 2023-08-09T14:23:52+00:00
**Comments**: 9

### Description

Is there a way to run ggml models on Intel ARC 770 GPU. Thanks

---

## Issue #N/A: Vulkan: Enabling Coopmat2 Flash Attention leads to incoherent output

**Link**: https://github.com/ggml-org/llama.cpp/issues/11268
**State**: closed
**Created**: 2025-01-16T22:10:11+00:00
**Closed**: 2025-01-18T08:26:51+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, Vulkan

### Description

### Name and Version

» build/bin/llama-cli --version
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | matrix cores: NV_coopmat2
version: 4497 (bd38ddea)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-cli

### Command line

```shell
llama-cli -p "The Peninsular War (1807–1814) was fought in the Iberian Peninsula by Portugal, Spain and the United Kingdom against the invading and occupying forces of the First French Empire during the Napoleonic Wars." -c 2048 -n 150 --ignore-eos -m models/Mistral-Nemo-Instruct-2407-Q4_0.gguf -ngl 99 -no-cnv -fa
```

### Problem description & steps to reproduce

When enabling Flash Attention, the output becomes incoherent.

Without Flash Attention:
```
main: llama threadpool init, n_threads = 16

system_info: n_threads = 16 (n_threads_batch = 16) / 32 | CPU : SSE3

[... truncated for brevity ...]

---

## Issue #N/A: server: Describing pictures with multi models seems to crash the model

**Link**: https://github.com/ggml-org/llama.cpp/issues/13480
**State**: closed
**Created**: 2025-05-12T12:56:56+00:00
**Closed**: 2025-05-14T13:31:00+00:00
**Comments**: 5

### Description

Hi all,

Tried to describe a picture with these two models in separate runs:
https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF
https://huggingface.co/bartowski/Qwen_Qwen2.5-VL-7B-Instruct-GGUF

The llama.cpp build used was b5351 CPU X64 on Win 11.
No errors where thrown.

Greetings,
Simon

---

## Issue #N/A: DeepSeek-R1-Zero-GGUF fails with src/llama.cpp:5142: GGML_ASSERT(hparams.n_expert <= LLAMA_MAX_EXPERTS) failed

**Link**: https://github.com/ggml-org/llama.cpp/issues/11378
**State**: closed
**Created**: 2025-01-23T21:25:29+00:00
**Closed**: 2025-01-29T12:45:55+00:00
**Comments**: 4
**Labels**: bug-unconfirmed

### Description

### Name and Version

./llama-cli --version
version: 3641 (9fe94cca)
built with cc (Debian 12.2.0-14) 12.2.0 for x86_64-linux-gnu

### Operating systems

Linux

### GGML backends

CPU, CUDA

### Hardware

Intel(R) Xeon(R) Platinum 8280L 256GB RAM 2x 3090

### Models

https://huggingface.co/unsloth/DeepSeek-R1-Zero-GGUF

### Problem description & steps to reproduce

```
./llama-cli -ngl 32 --model models/DeepSeek-R1-Zero-GGUF/DeepSeek-R1-Zero-Q2_K_L-00001-of-00005.gguf --threads 32 --prompt '<｜User｜>Write a python program which takes a quoted text string, and prints it vertically in a 80x24 grid, top to bottom and left to right.<｜Assistant｜>'
```
fails with
```


### First Bad Commit

Not sure, first time attempting. Using:
commit 05f63cc9ee859de07f585f7b12939345f39ada8b (HEAD -> master, origin/master, origin/HEAD)
Author: Eric Curtin <ecurtin@redhat.com>
Date:   Thu Jan 23 20:04:31 2025 +0000


### Relevant log output

```shell
Log start
main: build = 3641 (9fe94cca)
main: built with c

[... truncated for brevity ...]

---

## Issue #N/A: Bug: ImportError: libprotobuf-lite.so.25: cannot open shared object file: No such file or directory

**Link**: https://github.com/ggml-org/llama.cpp/issues/9071
**State**: closed
**Created**: 2024-08-18T08:36:11+00:00
**Closed**: 2024-08-19T13:49:16+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, low severity

### Description

### What happened?

Arch has a newer protobuf that apparently is not compatible with llama.cpp's llava. I tried building an older version of protobuf but was unsuccessful. 

### Name and Version

> ./llama-cli --version
version: 3600 (2fb92678)
built with clang version 17.0.0 for x86_64-pc-linux-gnu


### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
python ./examples/llava/convert_image_encoder_to_gguf.py -m /thearray/git/models/dolphin-vision-72b/vit --llava-projector /thearray/git/models/dolphin-vision-72b/vit/llava.projector --output-dir /thearray/git/models/dolphin-vision-72b/vit/ --clip-model-is-vision
Traceback (most recent call last):
  File "/code/git/llama.cpp/./examples/llava/convert_image_encoder_to_gguf.py", line 8, in <module>
    from gguf import *
  File "/code/git/llama.cpp/gguf-py/gguf/__init__.py", line 7, in <module>
    from .vocab import *
  File "/code/git/llama.cpp/gguf-py/gguf/vocab.py", line 10,

[... truncated for brevity ...]

---

## Issue #N/A: Bug: --chat-template seems to be broken now, no way to truly chat from the llama-cli

**Link**: https://github.com/ggml-org/llama.cpp/issues/8053
**State**: closed
**Created**: 2024-06-21T10:34:34+00:00
**Closed**: 2024-06-25T11:56:50+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, low severity

### Description

### What happened?

As per discussions:

https://github.com/ggerganov/llama.cpp/discussions/7837
https://github.com/ggerganov/llama.cpp/discussions/8009

It seems to be impossible to chat with llama3 8b properly. I have not tested this on 70b models but even in the server UI the model just starts making notes to itself and output garbage / training data as to how it should converse instead of actually conversing. Has something happened to the --chat-template chatml parameter? Even when the CLI is set to output special tokens, I do not see the ChatML tokens coming out.

### Name and Version

version: 3158 (52399254)

### What operating system are you seeing the problem on?

Linux

### Relevant log output

_No response_

---

## Issue #N/A: Use an argument parsing library

**Link**: https://github.com/ggml-org/llama.cpp/issues/70
**State**: closed
**Created**: 2023-03-13T00:16:29+00:00
**Closed**: 2023-03-15T21:52:58+00:00
**Comments**: 0
**Labels**: duplicate, enhancement

### Description

The argument parsing for `convert-ckpt-to-ggml.py` is quite ad-hoc and hard to follow.


I'm thinking that something around this would go a long way in making the arguments easier to use and follow in the code.

```python
import argparse

ARG_PARSER = argparse.ArgumentParser()
ARG_PARSER.add_argument("--model",
                        type=str,
                        help="Model to convert")
ARG_PARSER.add_argument("--ftype",
                        type=str,
                        choices=["f16", "f32"],
                        help="Floating point type to use")
ARG_PARSER.add_argument("--output",
                        type=str,
                        help="Where to write the converted model")
ARGS = ARG_PARSER.parse_args()
```

---

## Issue #N/A: llama-server bug: Prompt caching fails when editing the second user input

**Link**: https://github.com/ggml-org/llama.cpp/issues/13126
**State**: closed
**Created**: 2025-04-26T15:18:28+00:00
**Closed**: 2025-04-28T11:40:10+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

### Name and Version

I'm using the current latest llama-server.

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell
llama.cpp/build/bin/llama-server --model models/DeepSeek-V3-0324-UD-IQ3_XXS-00001-of-00006.gguf -ngl 4 -c 16000 -ctk q8_0
```

### Problem description & steps to reproduce

* If I enter a long context as part of a user query, this works after a long processing time (I'm using CPU offloading). ✔️
* If I then edit the end of that long input and submit again, the model response updates pretty quickly as the prompt is cached. ✔️
* If I then add a *second* user input after the first response, this also starts outputting a second response quickly, as expected. ✔️
* But, if I then edit the second user input and resubmit it, the entire context is processed from the start again, taking a very long time. Prompt caching seems to have failed. ❌

I'm not attempting any prompt caching across runs, this is all

[... truncated for brevity ...]

---

## Issue #N/A: Parallel/Slot issue of server mode for LLaVA

**Link**: https://github.com/ggml-org/llama.cpp/issues/4194
**State**: closed
**Created**: 2023-11-24T03:25:46+00:00
**Closed**: 2023-11-27T02:52:20+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.


I wrote a Python script to query the server with a picture in a loop.
When querying slot 0, the result is fine but when querying slot > 0, the completion result is wrong, and server filled to evaluate the image.

<img width="1455" alt="截圖 2023-11-24 上午11 15 31" src="https://github.com/

[... truncated for brevity ...]

---

## Issue #N/A: Instructions for "Prepare Data & Run" don't seem to work on Ubuntu 22.04

**Link**: https://github.com/ggml-org/llama.cpp/issues/1434
**State**: closed
**Created**: 2023-05-13T18:35:05+00:00
**Closed**: 2023-05-18T11:00:31+00:00
**Comments**: 4

### Description

I was able to build the `llama.cpp` code with CMake, and I downloaded the 7B and 13B models. However, it seems that the instructions for setting up the data do not work when building it this way:

(1) Instructions say:
```
# obtain the original LLaMA model weights and place them in ./models
ls ./models
65B 30B 13B 7B tokenizer_checklist.chk tokenizer.model
```

I assume that `.models` should be created as a subfolder of the `build/bin` folder, otherwise the instructions given later for running `quantize` and `main` will not work?  Ok, so I did that... Now my `$pwd` is `~/code/llama.cpp/build/bin`.

(2) Instructions say:
```
# install Python dependencies:
python3 -m pip install -r requirements.txt
```
Where is "requirements.txt"? Ok, I found this file back under `../../llama.cpp`, so I `cd ../..` and run that. Now my `$pwd` is `~/code/llama.cpp`...

(3) Instructions say:
```
# convert the 7B model to ggml FP16 format
python3 convert.py models/7B/
```
Assuming that

[... truncated for brevity ...]

---

## Issue #N/A: [server] Batching reduces context size?

**Link**: https://github.com/ggml-org/llama.cpp/issues/4289
**State**: closed
**Created**: 2023-12-02T03:24:43+00:00
**Closed**: 2023-12-06T05:31:05+00:00
**Comments**: 5
**Labels**: bug-unconfirmed

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x ] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.


***

Hello, this is more of a behavior question than a bug. I noticed that when enabling batching via the `--parallel` flag for the llama.cpp server, it divides the context up between slots.

Does this mean the effective context size is reduced? Or can, say, a 8k context model run

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Priority for RPC servers

**Link**: https://github.com/ggml-org/llama.cpp/issues/9323
**State**: closed
**Created**: 2024-09-05T13:36:17+00:00
**Closed**: 2024-09-12T00:16:12+00:00
**Comments**: 3
**Labels**: enhancement

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Add a way to prioritize which RPC server to use.

### Motivation

I want to prioritize which RPC server to use. This allows me to manually tell llama.cpp to use the RPC servers that are more powerful first before using a weaker ones, optimizing the speed.

### Possible Implementation

1. Add a new option to set which ones to use first.
2. Add and option to use the RPC servers that are given first. e.g. `--rpc server1:50052,server2:50052` would use `server1:50052` until all the memory is used up 

[... truncated for brevity ...]

---

## Issue #N/A: Installation Fails on M1 Mac Air

**Link**: https://github.com/ggml-org/llama.cpp/issues/136
**State**: closed
**Created**: 2023-03-14T16:17:05+00:00
**Closed**: 2023-03-15T21:21:22+00:00
**Comments**: 2
**Labels**: build

### Description

When I run the two commands the installer throws the following errors about halfway through the install:



cc  -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -DGGML_USE_ACCELERATE   -c ggml.c -o ggml.o
ggml.c:1364:25: error: implicit declaration of function 'vdotq_s32' is invalid in C99 [-Werror,-Wimplicit-function-declaration]
        int32x4_t p_0 = vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0ls);
                        ^
ggml.c:1364:19: error: initializing 'int32x4_t' (vector of 4 'int32_t' values) with an expression of incompatible type 'int'
        int32x4_t p_0 = vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0ls);
                  ^     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ggml.c:1365:19: error: initializing 'int32x4_t' (vector of 4 'int32_t' values) with an expression of incompatible type 'int'
        int32x4_t p_1 = vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1ls);
                  ^     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ggml.c:1367:13: error: assigning to '

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: KV defrag bug: nf != nh

**Link**: https://github.com/ggml-org/llama.cpp/issues/14059
**State**: closed
**Created**: 2025-06-07T18:56:09+00:00
**Closed**: 2025-06-09T20:04:36+00:00
**Comments**: 10
**Labels**: bug-unconfirmed

### Description

### Name and Version

b5595 3a077146a4761fdbd24bdd8eb098f46b8adc4dda 
b5600 d17a809ef0af09b16625e991a76f6fe80d9c332e
(with CUDA)

### Operating systems

Windows

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell
llama-server -m Qwen2.5-14B-Instruct-Q8_0.gguf -ngl 99 --temp 0 -fa -cb -c 44200 -np 17

llama-server -m Qwen2.5-1.5B-Instruct-Q8_0.gguf -ngl 99 --temp 0 -fa -cb -c 166400 -np 64
```

### Problem description & steps to reproduce

This assertion fails sporadically: GGML_ASSERT(nf == nh && "KV defrag bug: nf != nh")
It works fine for 2k or even 50k inference tasks that were completed in parallel, then it randomly fails.
Prompt sizes are roughly in the range from 100 to 600 tokens, and the generated tokens somewhere between 8 and 2k.

I've added debug output. Maybe these numbers yield a clue regarding what failed.
nf != nh (1681 != 1704)
i0: 1194, nh: 1704, nf: 1681, is: 1194, n_used: 2898, n_kv: 12260
Expected n_used: 2898, actual: 

[... truncated for brevity ...]

---

## Issue #N/A: MPI issue on raspberry pi cluster

**Link**: https://github.com/ggml-org/llama.cpp/issues/7260
**State**: closed
**Created**: 2024-05-13T14:07:26+00:00
**Closed**: 2024-05-19T17:30:52+00:00
**Comments**: 4
**Labels**: bug-unconfirmed

### Description

Greetings to all,
        When I run the following command, I encounter an issue. Has anyone else experienced this issue?
```
mpirun -hostfile /etc/volcano/mpiworker.host -n 2 /llama.cpp/main -m /mfs/ggml-model-q4_0.bin -p "I believe the meaning of life is" -n 128
``` 
      The issue is following:

llm_load_tensors:        CPU buffer size =  3647.87 MiB
..................................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =   256.00 MiB
llama_new_context_with_model: KV self size  =  256.00 MiB, K (f16):  128.00 MiB, V (f16):  128.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.12 MiB
llama_new_context_with_model:        CPU compute b

[... truncated for brevity ...]

---

## Issue #N/A: LLAMA_METAL=1 and LLAMA_MPI=1 incompatible?

**Link**: https://github.com/ggml-org/llama.cpp/issues/2166
**State**: closed
**Created**: 2023-07-10T20:59:28+00:00
**Closed**: 2023-07-14T17:34:42+00:00
**Comments**: 5

### Description

When following the instructions for MPI (https://github.com/ggerganov/llama.cpp/pull/2099) I get a build error.

```
> LLAMA_METAL=1 make CC=/opt/homebrew/bin/mpicc CXX=/opt/homebrew/bin/mpicxx LLAMA_MPI=1
I llama.cpp build info:
I UNAME_S:  Darwin
I UNAME_P:  arm
I UNAME_M:  arm64
I CFLAGS:   -I.              -O3 -std=c11   -fPIC -DNDEBUG -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -pthread -DGGML_USE_K_QUANTS -DGGML_USE_ACCELERATE -DGGML_USE_MPI -Wno-cast-qual
I CXXFLAGS: -I. -I./examples -O3 -std=c++11 -fPIC -DNDEBUG -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-multichar -pthread -DGGML_USE_K_QUANTS -DGGML_USE_MPI -Wno-cast-qual
I LDFLAGS:   -framework Accelerate
I CC:       Apple clang version 14.0.0 (clang-1400.0.29.202)
I CXX:      Apple clang version 14.0.0 (clang-1400.0.29.202)

/opt/homebrew/bin/mpicc  -I.              -O3 -std=c11   -fPIC -DNDEBUG -Wall -Wextra -Wpedantic -Wcast-qual -Wdo

[... truncated for brevity ...]

---

## Issue #N/A: Compile bug: Race condition during compilation, compilation works with -j 1 but not with -j 8

**Link**: https://github.com/ggml-org/llama.cpp/issues/13993
**State**: closed
**Created**: 2025-06-03T16:48:42+00:00
**Closed**: 2025-06-05T14:36:44+00:00
**Comments**: 3
**Labels**: bug-unconfirmed

### Description

### Git commit

ea1431b0fa3a8108aac1e0a94a13ccc4a749963e

### Operating systems

Linux

### GGML backends

CPU

### Problem description & steps to reproduce

I ran `podman build -f .devops/cpu.Dockerfile .` on my macbook and I got a compile error.

I went into the intermediate image, and I tried instead of running `cmake --build build -j $(nproc)` running `cmake --build build -j 1` and it built successfully.

`$(nproc)` was giving 8.

The error it gave seems to be getting eaten, here's an example:
```
[ 42%] Building CXX object src/CMakeFiles/llama.dir/unicode-data.cpp.o
c++: fatal error: Killed signal terminated program cc1plus
compilation terminated.
gmake[2]: *** [src/CMakeFiles/llama.dir/build.make:174: src/CMakeFiles/llama.dir/llama-grammar.cpp.o] Error 1
gmake[2]: *** Waiting for unfinished jobs....
gmake[1]: *** [CMakeFiles/Makefile2:934: src/CMakeFiles/llama.dir/all] Error 2
gmake: *** [Makefile:136: all] Error 2
```

The extact point of failure seems to shift around, as can be

[... truncated for brevity ...]

---

## Issue #N/A: Deepseek2 does not support K-shift Denial-of-Service vulnerability

**Link**: https://github.com/ggml-org/llama.cpp/issues/10380
**State**: closed
**Created**: 2024-11-18T11:02:34+00:00
**Closed**: 2024-11-19T11:29:28+00:00
**Comments**: 4

### Description

Long prompts/responses crash llama-server because "Deepseek2 does not support K-shift". For long prompts/responses, llama-server should return an error message or truncate the response, but instead, `GGML_ABORT` is called, which crashes the server. I believe that this is a Denial-of-Service vulnerability. A client should **never** be able to trigger `GGML_ABORT`.

The relevant line in the code is here:

https://github.com/ggerganov/llama.cpp/blob/9b75f03cd2ec9cc482084049d87a0f08f9f01517/src/llama.cpp#L18032

I have reported this security vulnerability almost three months ago [here (link only visible for maintainers),](https://github.com/ggerganov/llama.cpp/security/advisories/GHSA-jp78-gmv4-cc44) but have received no response and it is public knowledge now anyway, so I also opened this issue to increase visibility.

### Discussed in https://github.com/ggerganov/llama.cpp/discussions/9092

<div type='discussions-op-text'>

<sup>Originally posted by **99991** August 19, 2024<

[... truncated for brevity ...]

---

## Issue #N/A: train-from-scratch broken when compiled with cuda

**Link**: https://github.com/ggml-org/llama.cpp/issues/1963
**State**: closed
**Created**: 2023-06-22T03:43:12+00:00
**Closed**: 2023-06-26T15:33:54+00:00
**Comments**: 2

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

When llama.cpp is built with cuda support, train from scratch should work the same as when it is built without cuda support.

# Current Behavior

I get a core dump when trying to run the test training script when compiled with cuda support. It works fine when comp

[... truncated for brevity ...]

---

## Issue #N/A: tokenization: double EOS tokens

**Link**: https://github.com/ggml-org/llama.cpp/issues/7484
**State**: closed
**Created**: 2024-05-23T04:47:13+00:00
**Closed**: 2024-05-24T15:39:25+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

The llama.cpp tokenizer currently adds an EOS token unconditionally. However, adding an EOS token to the end of the system prompt is necessary to prevent generation before user input. This leads to two EOS tokens before the user prompt, potentially causing suboptimal performance.

llama.cpp-b2972, MacOS 14.5

./main -m Meta-Llama-3-8B-Instruct-Q8_0.gguf --temp 0 -i -e -p "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>" -r "<|eot_id|>" --in-prefix "<|start_header_id|>user<|end_header_id|>\n\n" --in-suffix "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

Current behavior:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|>
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
```
Expected behavior:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
```

---

## Issue #N/A: b1705 introduces build error on linux-aarch64

**Link**: https://github.com/ggml-org/llama.cpp/issues/4654
**State**: closed
**Created**: 2023-12-27T18:28:34+00:00
**Closed**: 2023-12-31T09:44:23+00:00
**Comments**: 9
**Labels**: help wanted, bug-unconfirmed

### Description

Starting with b1705, I get the following when building for linux-aarch64:
```
clang++ -I. -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG  -std=c++11 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -mcpu=native   -Wunreachable-code-break -Wunreachable-code-return -Wmissing-prototypes -Wextra-semi -c common/train.cpp -o train.o
ggml-quants.c:413:25: error: redefinition of 'vdotq_s32'
inline static int32x4_t vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
                        ^
/opt/llvm.org/v16.0.6/lib/clang/16/include/arm_neon.h:33859:51: note: previous definition is here
__ai __attribute__((target("dotprod"))) int32x4_t vdotq_s32(int32x4_t __p0, int8x16_t __p1, int8x16_t __p2) {
                                                  ^
1 error generated.
```

it seems to be directly related to https://github.com/ggerganov/llama.cpp/pull/4630. the various #define tricks i'm used to don't seem to be

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Decoding special tokens in T5

**Link**: https://github.com/ggml-org/llama.cpp/issues/8938
**State**: closed
**Created**: 2024-08-08T16:32:39+00:00
**Closed**: 2024-08-09T16:53:10+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

I have a T5/lora model trained to output some text separated by the `<extra_id_0>` special token (the tokenizer properly works after following instructions in #8872) .

When running the model using Huggingface's transformers/peft, it generates the expected output. However, when I use `llama-cli`, what happens instead is that the moment the first such token is reached, it's actually decoded into an `EOG` token instead of the extra token and generation is stopped.

I might be simply doing something wrong in using the library.

### Name and Version

version: 3549 (afd27f01)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu


### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

---

## Issue #N/A: Unable to compile main cmake & windev, precompiled not working either

**Link**: https://github.com/ggml-org/llama.cpp/issues/1549
**State**: closed
**Created**: 2023-05-21T08:30:44+00:00
**Closed**: 2023-05-23T05:23:30+00:00
**Comments**: 2

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

Unable to build llama.cpp
Have tried Cmake
Have tried w64devkit 
Have tried the precompile versions avx1 and avx2 back to 3de84b2

main.exe compiles however it is printing out c code in the terminal
with respect to the issues with cmake found a similar issue that in another post
h

[... truncated for brevity ...]

---

## Issue #N/A: Command-R GGUF conversion no longer working

**Link**: https://github.com/ggml-org/llama.cpp/issues/7030
**State**: closed
**Created**: 2024-05-01T20:40:40+00:00
**Closed**: 2024-05-05T05:19:31+00:00
**Comments**: 8
**Labels**: bug-unconfirmed

### Description

As recently as a few days ago, Command-R (and presumably R+) could be converted with convert-hf-to-gguf.py.  I double checked and conversion completes successfully in b2751.  However, with the recent changes to accommodate Llama3, Command-R compatibility has been broken.  Trying to convert today with b2777 I get

```
raise NotImplementedError("BPE pre-tokenizer was not recognized - update get_vocab_base_pre()")
NotImplementedError: BPE pre-tokenizer was not recognized - update get_vocab_base_pre()
```

I know that L3 required a new tokenizer provided by meta to facilitate proper conversion.  Do we require something new from cohere, or is this something that can be fixed internally?

---

## Issue #N/A: common/log.h:290:61: error: expected primary-expression before ',' token

**Link**: https://github.com/ggml-org/llama.cpp/issues/2898
**State**: closed
**Created**: 2023-08-30T09:02:10+00:00
**Closed**: 2023-09-01T09:07:07+00:00
**Comments**: 13

### Description

Running environment: Windows

Compilation method: BLAS Build

When I open w64devkit.exe and CD it to the llama.cpp directory, enter the command make LLAMA_ OPENBLAS=1 encountered the following error

![image](https://github.com/ggerganov/llama.cpp/assets/14157458/d10d13aa-55f2-46e8-bd9c-31d2af9611e2)


---

## Issue #N/A: Try to use quantized `ggml_mul_mat` in attention layer

**Link**: https://github.com/ggml-org/llama.cpp/issues/1098
**State**: closed
**Created**: 2023-04-21T07:38:58+00:00
**Closed**: 2023-04-22T08:37:55+00:00
**Comments**: 1
**Labels**: good first issue, performance

### Description

The following 2 matrix multiplication calls sill remain in FP16 precission:

- https://github.com/ggerganov/llama.cpp/blob/d40fded93e1a533e969768e1e335c15c61c296ce/llama.cpp#L1135-L1137
- https://github.com/ggerganov/llama.cpp/blob/d40fded93e1a533e969768e1e335c15c61c296ce/llama.cpp#L1158-L1160

Was wondering, if we quantize those on-the-fly would there be any benefit.
The quantization can be done with an extra `ggml_cpy()` call, before the `ggml_mul_mat()` call.

See if this speeds up the computation and how it affects perplexity

---

## Issue #N/A: convert-hf-to-gguf.py  XVERSE-13B-256K  error

**Link**: https://github.com/ggml-org/llama.cpp/issues/6425
**State**: closed
**Created**: 2024-04-01T13:07:59+00:00
**Closed**: 2024-04-03T23:53:56+00:00
**Comments**: 4
**Labels**: invalid

### Description

Please include information about your system, the steps to reproduce the bug, and the version of llama.cpp that you are using. If possible, please provide a minimal code example that reproduces the bug.

Model: 
https://huggingface.co/xverse/XVERSE-13B-256K

python convert-hf-to-gguf.py /Volumes/FanData/models/XVERSE-13B-256K --outfile /Volumes/FanData/models/GGUF/xverse-13b-256k-f16.gguf --outtype f16

`
python convert-hf-to-gguf.py /Volumes/FanData/models/XVERSE-13B-256K --outfile /Volumes/FanData/models/GGUF/xverse-13b-256k-f16.gguf --outtype f16
Loading model: XVERSE-13B-256K
gguf: This GGUF file is for Little Endian only
Set model parameters
Set model tokenizer
gguf: Setting special token type bos to 2
gguf: Setting special token type eos to 3
gguf: Setting special token type pad to 1
Exporting model to '/Volumes/FanData/models/GGUF/xverse-13b-256k-f16.gguf'
gguf: loading model part 'pytorch_model-00001-of-00015.bin'
Traceback (most recent call last):
  File "/U

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Vulkan not compile

**Link**: https://github.com/ggml-org/llama.cpp/issues/9582
**State**: closed
**Created**: 2024-09-21T18:32:09+00:00
**Closed**: 2024-09-28T16:23:46+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

~/llama.cpp (master)> cmake -B build -DGGML_VULKAN=1
                                  cmake --build build --config Release -j 16
-- The C compiler identification is GNU 14.2.1
-- The CXX compiler identification is GNU 14.2.1
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found Git: /usr/bin/git (found version "2.46.1")
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE
-- Found OpenMP_C: -fopenmp (found version "4.5")
-- Found OpenMP_CXX: -fopenmp (found version "4.5")
-- Found OpenMP: TRUE (found version "4.5")
-- Open

[... truncated for brevity ...]

---

## Issue #N/A: [User] Optimizing llama.cpp using nvcomp

**Link**: https://github.com/ggml-org/llama.cpp/issues/2607
**State**: closed
**Created**: 2023-08-14T09:52:08+00:00
**Closed**: 2023-08-18T04:53:21+00:00
**Comments**: 5

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

I expect llama.cpp to efficiently run large language models on commodity computers. To achieve this, I suggest utilizing the nvcomp library to compress and decompress data during memory transfers, which should improve the speed and efficiency of data transfer, especia

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: Tokens in top_probs / top_logprobs are missing whitespace

**Link**: https://github.com/ggml-org/llama.cpp/issues/11728
**State**: closed
**Created**: 2025-02-07T08:05:15+00:00
**Closed**: 2025-02-11T13:06:46+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

### Name and Version

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090 Ti, compute capability 8.6, VMM: yes
version: 0 (unknown)
built with gcc (GCC) 13.3.0 for x86_64-unknown-linux-gnu

(actually version 4552, built with Nix)

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell
$ bin/llama-server -m ../../../wizardcoder-python-34b-v1.0.Q5_K_M.gguf -ngl 9999
...

$ curl -fsS \
    --url http://127.0.0.1:8080/completion \
    --header "Content-Type: application/json" \
    --data '{"prompt": "Hello","n_predict": 1, "n_probs": 10, "temperature":0}' | jq .
{
  ...
  "completion_probabilities": [
    {
      "id": 2897,
      "token": " os",                      <---------- whitespace OK
      "bytes": [
        32,                                <---------- whitespace OK
        111,
        115
      ],


[... truncated for brevity ...]

---

