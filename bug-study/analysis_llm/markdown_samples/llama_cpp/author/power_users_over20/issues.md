# power_users_over20 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 6
- Closed Issues: 24

### Label Distribution

- stale: 12 issues
- enhancement: 10 issues
- bug-unconfirmed: 7 issues
- good first issue: 5 issues
- roadmap: 4 issues
- help wanted: 4 issues
- server/webui: 4 issues
- bug: 3 issues
- research 🔬: 3 issues
- generation quality: 2 issues

---

## Issue #N/A: Feature Request: GGUF 2 BIN

**Link**: https://github.com/ggml-org/llama.cpp/issues/7695
**State**: closed
**Created**: 2024-06-02T15:59:33+00:00
**Closed**: 2024-07-17T01:06:48+00:00
**Comments**: 3
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Please include a utility to convert a GGUF file to BIN... I have a  few model requantized using "quantize" and I don't want to redownload everything.


### Motivation

Some programs want a bin file, one of them I'm testing it's not opensource.


### Possible Implementation

same as convert.py backwards :P


---

## Issue #N/A: Random seed possible problems.

**Link**: https://github.com/ggml-org/llama.cpp/issues/8593
**State**: closed
**Created**: 2024-07-19T19:01:20+00:00
**Closed**: 2024-10-19T01:07:19+00:00
**Comments**: 20
**Labels**: stale

### Description

I ran llama.cpp (latest version) with these parameters:

```
prompt="""
Tell me a long story.
"""

```
`llama-cli --seed 1721414715 -c 4096 -m /content/$m -t $(nproc) -ngl 999 -p "User: Hi\nBot:Hi\nUser: {prompt}\nBot:"`

and in the log I read the seed was: 1721414715

so at the next run I used --seed 1721414715  but the story was a different one.

why?


---

## Issue #N/A: Bug: -DCMAKE_CUDA_ARCHITECTURES=52 on GTX 1660 Ti or RTX 3060 results in incorrect output

**Link**: https://github.com/ggml-org/llama.cpp/issues/9019
**State**: closed
**Created**: 2024-08-13T17:36:46+00:00
**Closed**: 2024-09-27T01:07:16+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

The v3.2.0 release of GPT4All was effectively built with `-DCMAKE_CUDA_ARCHITECTURES=52` due to a mistake in the build scripts. I would expect this to still work an an GTX 1660 Ti or RTX 3060, albeit with reduced performance, since PTX should be fully forward-compatible. It might even be OK if it failed an assertion due to known incompatibility with the newer GPUs. However, both of these GPUs exhibited nonsense generation instead, which I did not expect.

Building with `-DCMAKE_CUDA_ARCHITECTURES="52;61;70;75"` fixed the nonsense generation for the RTX 3060 user.

cc @slaren @JohannesGaessler

--

[Llama 3.1 8B Instruct 128k](https://huggingface.co/GPT4All-Community/Meta-Llama-3.1-8B-Instruct-128k-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf) on a GTX 1660 Ti:
<details>
<summary>Video</summary>

https://github.com/user-attachments/assets/d513c1fe-c430-4b40-8b60-6631068280e9

</details>

--

[Phi-3 Mini Instruct](https://gpt4all.io/model

[... truncated for brevity ...]

---

## Issue #N/A: llama : add CLI assistant

**Link**: https://github.com/ggml-org/llama.cpp/issues/10688
**State**: open
**Created**: 2024-12-06T12:07:53+00:00
**Comments**: 4
**Labels**: enhancement, good first issue, roadmap

### Description

The https://github.com/AnswerDotAI/shell_sage project seems quite fun and useful. [GitHub Copilot CLI](https://www.npmjs.com/package/@githubnext/github-copilot-cli) was also a rather useful utility. 

We should build a fully-local alternative leveraging `llama-server`'s advanced context reuse techniques in the same spirit as [llama.vim](https://github.com/ggml-org/llama.vim). Speculative decoding should be very effective in this scenario as well. Likely the Qwen 2.5 family would be ideal for this use case.

---

## Issue #N/A: Again, the releases don't have the libraries.

**Link**: https://github.com/ggml-org/llama.cpp/issues/11091
**State**: closed
**Created**: 2025-01-05T17:31:06+00:00
**Closed**: 2025-01-24T16:41:31+00:00
**Comments**: 16

### Description

`./build/bin/llama-quantize: error while loading shared libraries: libllama.so: cannot open shared object file: No such file or directory`

it already happened in the past...

---

## Issue #N/A: EXAONE Deep 2 unsupported?

**Link**: https://github.com/ggml-org/llama.cpp/issues/12448
**State**: closed
**Created**: 2025-03-18T11:08:25+00:00
**Closed**: 2025-03-18T16:24:35+00:00
**Comments**: 7

### Description

```
build: 4910 (d9a14523) with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
main: llama backend init
main: load the model and apply lora adapter, if any
llama_model_loader: loaded meta data with 35 key-value pairs and 273 tensors from /content/EXAONE-Deep-2.4B.q8_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = exaone
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = EXAONE Deep 2.4B
llama_model_loader: - kv   3:                           general.basename str              = EXAONE-Deep
llama_model_loader: - kv   4:                         general.size_label str              = 2.4B
llama_model_loader: - kv   5:                            general.license str   

[... truncated for brevity ...]

---

## Issue #N/A: Measure perplexity delta between Q4_0 and F16 "output" tensor

**Link**: https://github.com/ggml-org/llama.cpp/issues/1003
**State**: closed
**Created**: 2023-04-15T19:22:22+00:00
**Closed**: 2023-04-16T20:08:54+00:00
**Comments**: 2
**Labels**: help wanted, good first issue, high priority, generation quality

### Description

The last tensor of the transformer (called `output` in llama.cpp) is one of the biggest ones:

https://github.com/ggerganov/llama.cpp/blob/0ad964631f9b3970f1936008fcfb1eadef59c7ed/llama.cpp#L945

I wonder how the perplexity improves by keeping it in F16 format instead of quantizing that particular tensor

### Results

<details>
  <summary>Q4_0 M1 Pro (with BLAS) [655]6.2838 (i.e. reference)</summary>

```
$  make clean && make -j perplexity && time ./perplexity -m ./models/7B/ggml-model-q4_0.bin -f ./build/wiki.test.raw -t 8
I llama.cpp build info: 
I UNAME_S:  Darwin
I UNAME_P:  arm
I UNAME_M:  arm64
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -Wno-unused-function -pthread -DGGML_USE_ACCELERATE
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-multichar -pthread
I LDFLAGS:   -frame

[... truncated for brevity ...]

---

## Issue #N/A: server: comment --threads option behavior

**Link**: https://github.com/ggml-org/llama.cpp/issues/6230
**State**: closed
**Created**: 2024-03-22T08:56:45+00:00
**Closed**: 2024-03-23T17:00:39+00:00
**Comments**: 4
**Labels**: documentation, enhancement, server/webui

### Description

As we are using batching, I am wondering what is the purpose of `--threads N` parameter in the `server`.

Should we remove it ?

---

## Issue #N/A: metal : the Q3_K and Q4_K kernels with LLAMA_QKK_64=1 are broken

**Link**: https://github.com/ggml-org/llama.cpp/issues/3276
**State**: closed
**Created**: 2023-09-20T12:49:19+00:00
**Closed**: 2024-01-02T08:57:46+00:00
**Comments**: 0
**Labels**: bug

### Description

The following commands fail to generate coherent text:

```bash
LLAMA_QKK_64=1 make -j && ./main -m tmp/mnt/models/open-llama/3B-v2/ggml-model-q4_k.gguf -p "I believe the meaning of life is" -t 8 -ngl 1

LLAMA_QKK_64=1 make -j && ./main -m tmp/mnt/models/open-llama/3B-v2/ggml-model-q3_k.gguf -p "I believe the meaning of life is" -t 8 -ngl 1
```

It works on the CPU (Arm and x86).
It also works with the following patch:

```diff
diff --git a/ggml-metal.m b/ggml-metal.m
index 1139ee3..ed9857f 100644
--- a/ggml-metal.m
+++ b/ggml-metal.m
@@ -889,7 +889,7 @@ void ggml_metal_graph_compute(
                                 src1t == GGML_TYPE_F32 &&
                                 [ctx->device supportsFamily:MTLGPUFamilyApple7] &&
                                 ne00%32 == 0 &&
-                                ne11 > 1) {
+                                ne11 >= 1) {
                                 switch (src0->type) {
                                     case GGML_

[... truncated for brevity ...]

---

## Issue #N/A: Compilation fails with Cmake on Windows (since #5970): error C1061

**Link**: https://github.com/ggml-org/llama.cpp/issues/6093
**State**: closed
**Created**: 2024-03-15T23:06:35+00:00
**Closed**: 2024-03-16T15:39:16+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

After today's changes, I can no longer build the project with Visual Studio's Cmake. Build fails with compiler [error C1061](https://learn.microsoft.com/en-us/cpp/error-messages/compiler-errors-1/fatal-error-c1061?view=msvc-170).
It seems like there is too many nested block scopes for MSVC to handle in `common.cpp`.

```sh
$ cmake --build . --config Release -- "-m:24"
Version MSBuild 17.7.2+d6990bcfa pour .NET Framework

  build_info.vcxproj -> C:\llm\llama.cpp\build\common\build_info.dir\Release\build_info.lib
  ggml.vcxproj -> C:\llm\llama.cpp\build\ggml.dir\Release\ggml.lib
  llama.vcxproj -> C:\llm\llama.cpp\build\Release\llama.lib
  llava.vcxproj -> C:\llm\llama.cpp\build\examples\llava\llava.dir\Release\llava.lib
  gguf.vcxproj -> C:\llm\llama.cpp\build\bin\Release\gguf.exe
  quantize.vcxproj -> C:\llm\llama.cpp\build\bin\Release\quantize.exe
  ggml_static.vcxproj -> C:\llm\llama.cpp\build\Release\ggml_static.lib
  quantize-stats.vcxproj -> C:\llm\llama.cpp\build\bi

[... truncated for brevity ...]

---

## Issue #N/A: Bug: llamacpp for CPU/GPU (avx avx2) quants IQ1xx, IQ2xx, IQ3xx are overheating (CPU 90C) CPU ryzen 9 7950x3d but IQ4xx and other quants not (CPU 65C) 

**Link**: https://github.com/ggml-org/llama.cpp/issues/8760
**State**: closed
**Created**: 2024-07-29T22:08:21+00:00
**Closed**: 2024-10-12T01:13:14+00:00
**Comments**: 31
**Labels**: stale

### Description

### What happened?

CPU Ryzen 7950x3D
win 11

Mistral-Large-Instruct-2407.IQ3_XS.gguf    ( CPU 90 C )
![Screenshot 2024-07-29 224906](https://github.com/user-attachments/assets/e8bd7daa-adf4-48c5-93d1-58dcd7260189)

Meta-Llama-3-70B-Instruct.Q4_K_M.gguf   (CPU 66 C )
![Screenshot 2024-07-29 225028](https://github.com/user-attachments/assets/b16e0cdc-0ede-48ca-8002-3d26e9f6c309)

Temperature is higher than the CPU torture tests made by CPUZ then max I have is 83 C. 
That happens ONLY with Mistral-Large-Instruct-2407.IQ3_XS.gguf for me even I set  **--threads 1 my CPU is heating up like crazy to 90 C** but manager showing only 1 thread used for llamacpp.... 

Mistral-Large-Instruct-2407.IQ3_XS.gguf  
llama-cli.exe --model models/new3/Mistral-Large-Instruct-2407.IQ3_XS.gguf --color --threads 1 --keep -1 --n-predict -1 --ctx-size 8196 --interactive -ngl 39 --simple-io -e --multiline-input --no-display-prompt --conversation --no-mmap --temp 0.6 --chat-template chatml
````
lla

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

## Issue #N/A: Error C2026 when building ggml-opencl.cpp with MSVC

**Link**: https://github.com/ggml-org/llama.cpp/issues/3973
**State**: closed
**Created**: 2023-11-06T21:22:11+00:00
**Closed**: 2024-04-02T01:12:09+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

MSVC can't handle long string litterals, so it throws out [Error C2026](https://learn.microsoft.com/en-us/cpp/error-messages/compiler-errors-1/compiler-error-c2026?view=msvc-170) when compiling the long strings that are constructed with the macro `MULTILINE_QUOTE( ... )` .

There is an ea

[... truncated for brevity ...]

---

## Issue #N/A: llama : add test for saving/loading sessions to the CI

**Link**: https://github.com/ggml-org/llama.cpp/issues/2631
**State**: closed
**Created**: 2023-08-16T14:42:01+00:00
**Closed**: 2025-03-07T10:19:33+00:00
**Comments**: 3
**Labels**: good first issue, testing

### Description

See how the `save-load-state` example works:

https://github.com/ggerganov/llama.cpp/tree/master/examples/save-load-state

Add a simple test to [ci/run.sh](https://github.com/ggerganov/llama.cpp/tree/master/ci)

---

## Issue #N/A: Feature Request: Installable package via winget

**Link**: https://github.com/ggml-org/llama.cpp/issues/8188
**State**: open
**Created**: 2024-06-28T13:27:20+00:00
**Comments**: 19
**Labels**: enhancement, help wanted

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

On macos/linux, user can install a pre-built version llama.cpp easily via `brew`

It would be nice to have the equivalent to that on windows, via `winget`

### Motivation

The pre-built binary is already available via releases: https://github.com/ggerganov/llama.cpp/releases

It would be nice to somehow push them to https://winget.run/

However, I'm not familiar with working on windows, so I create this issue to further discuss and to look for help from the community.

### Possible Implemen

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: llamacppp server - generated syntax code coloring 

**Link**: https://github.com/ggml-org/llama.cpp/issues/10800
**State**: closed
**Created**: 2024-12-12T13:50:28+00:00
**Closed**: 2024-12-12T15:58:42+00:00
**Comments**: 1
**Labels**: enhancement

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Hello

Can you add a generated syntax code coloring?
Currently llamacpp server is awesome but missing coloring code generated ;)   

![Screenshot 2024-12-12 134330](https://github.com/user-attachments/assets/489ff4d4-39a3-4dcf-931f-d2d182b25557)


### Motivation

Colored syntax would be more readable 

### Possible Implementation

_No response_

---

## Issue #N/A: llama : combined beam search + grammar sampling strategy

**Link**: https://github.com/ggml-org/llama.cpp/issues/2923
**State**: open
**Created**: 2023-08-31T06:29:29+00:00
**Comments**: 13
**Labels**: good first issue, generation quality, research 🔬, roadmap

### Description

This feature was proposed by @spion in https://github.com/ggerganov/llama.cpp/issues/2813#issuecomment-1694390583

> In some cases, its useful to do constrained evaluation of logits based on a union of possible text values, then pick the sum { logits } (i.e. product(probabilities)) that gives the most probable outcome overall.

> E.g. template (using MS guidance)

> {{#select 'armor'}}leather{{or}}chainmail{{or}}plate{{/select}}

> To definitely make the best choice, we'd need to calculate the probability of all 3 token sequences. Its easy if all the choices map to a single token, but with multiple tokens we'd need not just parallel generation but parallel logit evaluation of multiple possible paths.

> If we go greedy, we might get suboptimal results in cases multiple choices start with the same logit.

It should be possible to implement this by combining the existing beam search and grammar sampling features. See the discussion in the referenced comment for more info

---

## Issue #N/A: Compile bug: warning: comparison of integers of different signs: 'rlim_t' (aka 'long') and 'size_t'

**Link**: https://github.com/ggml-org/llama.cpp/issues/11033
**State**: closed
**Created**: 2025-01-02T00:15:00+00:00
**Closed**: 2025-02-16T01:07:36+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Git commit

4404

### Operating systems

BSD

### GGML backends

CPU

### Problem description & steps to reproduce

clang-18 prints this warning:
```
/usr/ports/misc/llama-cpp/work/llama.cpp-b4404/src/llama.cpp:2398:45: warning: comparison of integers of different signs: 'rlim_t' (aka 'long') and 'size_t' (aka 'unsigned long') [-Wsign-compare]
 2398 |         if (suggest && (lock_limit.rlim_max > lock_limit.rlim_cur + size)) {
      |                         ~~~~~~~~~~~~~~~~~~~ ^ ~~~~~~~~~~~~~~~~~~~~~~~~~~
1 warning generated.
```

### First Bad Commit

_No response_

### Relevant log output

```shell
n/a
```


---

## Issue #N/A: Feature Request: Support multimodal LLMs such as Qwen2.5-VL as embedding models

**Link**: https://github.com/ggml-org/llama.cpp/issues/13247
**State**: closed
**Created**: 2025-05-01T21:15:57+00:00
**Closed**: 2025-06-17T01:07:59+00:00
**Comments**: 4
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

llama.cpp should support multimodal models built upon architectures such as Qwen2.5-VL for image and text embeddings.

### Motivation

Multimodal LLMs demonstrate better alignment between image and text embeddings than constrastively trained models such as CLIP, which suffer from a modality gap (text compares better with unrelated text than it does with a related image).

Nomic's latest vision models are designed for PDF document retrieval. [nomic-embed-multimodal-3b](https://huggingface.co/nomic-a

[... truncated for brevity ...]

---

## Issue #N/A: Compile bug: warning: 'CACHE_LINE_SIZE' macro redefined

**Link**: https://github.com/ggml-org/llama.cpp/issues/11034
**State**: closed
**Created**: 2025-01-02T00:17:43+00:00
**Closed**: 2025-02-16T01:07:35+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Git commit

4404

### Operating systems

BSD

### GGML backends

CPU

### Problem description & steps to reproduce

On FreeBSD the ```CACHE_LINE_SHIFT``` is already defined in the system headers, so clang-18 prints this warning:
```
/usr/ports/misc/llama-cpp/work/llama.cpp-b4404/ggml/src/ggml-cpu/ggml-cpu.c:242:9: warning: 'CACHE_LINE_SIZE' macro redefined [-Wmacro-redefined]
  242 | #define CACHE_LINE_SIZE 64
      |         ^
/usr/include/machine/param.h:92:9: note: previous definition is here
   92 | #define CACHE_LINE_SIZE         (1 << CACHE_LINE_SHIFT)
      |         ^
1 warning generated.
```

You should either add ```#ifndef``` around ```CACHE_LINE_SHIFT```, or only define it on non-FreeBSD systems.


### First Bad Commit

_No response_

### Relevant log output

```shell
n/a
```


---

## Issue #N/A: Add GPU support to ggml

**Link**: https://github.com/ggml-org/llama.cpp/issues/914
**State**: closed
**Created**: 2023-04-12T11:11:42+00:00
**Closed**: 2023-04-12T11:47:54+00:00
**Comments**: 1
**Labels**: enhancement, help wanted, hardware, research 🔬

### Description

## Intro

This issue is more suitable for the https://github.com/ggerganov/ggml repo, but adding it here for more visibility.

First, I don't see adding a GPU framework that is tightly integrated with `ggml` anytime soon because it usually comes with a lot of maintenance drawbacks, architecture changes and issues. However, there is an alternative approach that might be relatively easy to implement and I think would be a very cool way for new developers to join in and help.

## Description

`ggml` produces computation graphs which are basically directed acyclic graphs (DAGs) that can be easily exported, iterated, etc. A graph contains the information about all necessary tensor operations and buffers needed to evaluate the model. The idea is to first add basic `ggml` functionality for exporting the graphs in some trivial text format that can be parsed as a second step by a separate `ggml` tool. Having the exported graphs, one can process them and construct hardware-specific code 

[... truncated for brevity ...]

---

## Issue #N/A: cuda: NaN perplexity with some models on some GPUs (Gemma, MPT)

**Link**: https://github.com/ggml-org/llama.cpp/issues/5817
**State**: closed
**Created**: 2024-03-01T16:03:21+00:00
**Closed**: 2024-03-03T19:28:00+00:00
**Comments**: 4
**Labels**: bug

### Description

I'm making an issue for this to make sure it isn't forgotten about. I've been able to work around this, but it seems like a bug to me.

ref https://github.com/ggerganov/llama.cpp/pull/5631#issuecomment-1961613111

### Steps to Reproduce
1. Download safetensors model from https://huggingface.co/google/gemma-7b
2. Checkout llama.cpp commit 15499eb94 (master should reproduce this as well)
3. `./convert-hf-to-gguf.py gemma-7b --outfile gemma-7b.f16.gguf --outtype f16`
4. `cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLAMA_CUBLAS=ON`
5. `make -C build perplexity`
6. Run perplexity on a Tesla P40. Use `-ngl 2` or above.
```
$ CUDA_VISIBLE_DEVICES=0 build/bin/perplexity -f wiki.test.raw -c 2048 -m gemma-7b.f16.gguf -ngl 99
<snip>
perplexity: tokenizing the input ..
perplexity: tokenization took 974.102 ms
perplexity: calculating perplexity over 142 chunks, batch_size=512
perplexity: 6.52 seconds per pass - ETA 15.43 minutes
[1]nan,
```
And there's no point in running

[... truncated for brevity ...]

---

## Issue #N/A: Compile bug: vulkan-shaders-gen hangs when built with address sanitizers

**Link**: https://github.com/ggml-org/llama.cpp/issues/12581
**State**: closed
**Created**: 2025-03-26T05:53:14+00:00
**Closed**: 2025-05-10T01:07:42+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Git commit

4951


### Operating systems

BSD

### GGML backends

Vulkan

### Problem description & steps to reproduce

This problem is somewhat similar to https://github.com/ggml-org/llama.cpp/pull/10713

When built with:
```
CXXFLAGS+=      -fsanitize=address
LDFLAGS+=       -fsanitize=address
```
the build fails with vulkan-shaders-gen hanging with 100% CPU.

clang-19
FreeBSD 14.2

### First Bad Commit

_No response_

### Compile command

```shell
n/a
```

### Relevant log output

```shell
n/a
```

---

## Issue #N/A: Compile bug: Trying to compile on a raspi w v2 and failing to compile

**Link**: https://github.com/ggml-org/llama.cpp/issues/11079
**State**: closed
**Created**: 2025-01-04T23:07:22+00:00
**Closed**: 2025-02-19T01:12:41+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Git commit

git rev-parse HEAD b56f079e28fda692f11a8b59200ceb815b05d419

### Operating systems

Linux

### GGML backends

CPU

### Problem description & steps to reproduce

Trying to compile on a raspi w v2 and  failing to compile

### First Bad Commit

_No response_

### Relevant log output

```shell
[  2%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-opt.cpp.o
In file included from /usr/include/c++/12/algorithm:61,
                 from /usr/src/llama.cpp/ggml/src/ggml-opt.cpp:8:
/usr/include/c++/12/bits/stl_algo.h: In function ‘void std::shuffle(_RAIter, _RAIter, _UGenerator&&) [with _RAIter = __gnu_cxx::__normal_iterator<long long int*, vector<long long int> >; _UGenerator = mersenne_twister_engine<unsigned int, 32, 624, 397, 31, 2567483615, 11, 4294967295, 7, 2636928640, 15, 4022730752, 18, 1812433253>&]’:
/usr/include/c++/12/bits/stl_algo.h:3696:5: note: parameter passing for argument of type ‘__gnu_cxx::__normal_iterator<long long int*, std::vector<long lon

[... truncated for brevity ...]

---

## Issue #N/A: parallel/server crashes with: ggml.c:16521: i != GGML_HASHTABLE_FULL when defragmentation is enabled

**Link**: https://github.com/ggml-org/llama.cpp/issues/6685
**State**: open
**Created**: 2024-04-15T11:39:29+00:00
**Comments**: 28
**Labels**: bug, server/webui

### Description

### Context

Using latest 17e98d4c96a583d420f12046bc92102381dbd28e llama.cpp server.

Server started with a llama70b-F16 like model:

```shell
server \
 --model model-f16.gguf \
--ctx-size 32768 \
--n-predict 4096 \
--parallel 32 \
--n-gpu-layers 81 \
--batch-size 4096 \
--ubatch-size 256 \
--metrics \
--mg 1 \
--log-format text \
--defrag-thold 0.1
```

When sending 32 concurrent requests, the server crashes with:

`GGML_ASSERT: /llama.cpp/ggml.c:16521: i != GGML_HASHTABLE_FULL`

Backend is CUDA, on 2 A100, compute capability 80.

EDIT: The issue is related with defragmentation, quick fix: disable defragmentation

---

## Issue #N/A: server : add support for file upload to the Web UI

**Link**: https://github.com/ggml-org/llama.cpp/issues/11611
**State**: closed
**Created**: 2025-02-03T05:50:11+00:00
**Closed**: 2025-05-09T21:16:40+00:00
**Comments**: 3
**Labels**: enhancement, help wanted, good first issue, server/webui

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

The idea is to be able to add any file that could be converted to plain text. The web client will do the processing and add the plain text to the context of the next request.

I am not sure what tools are available to do this in the browser, but my assumption is that there should be support, for example for converting PDF to text. Hopefully these are small packages that would not bloat the web ui too much.

### Motivation

It is useful to pass files to your chats.

### Possible Implementation

_N

[... truncated for brevity ...]

---

## Issue #N/A: multiline-input

**Link**: https://github.com/ggml-org/llama.cpp/issues/1382
**State**: closed
**Created**: 2023-05-09T17:07:27+00:00
**Closed**: 2024-04-09T01:09:31+00:00
**Comments**: 30
**Labels**: stale

### Description

I'm testing newest build.
I have a stupid question as I added --multiline-input as parameter .
How to end of my input now? 
Pressing enter just invoking a new line.


Also without that parameter I can invoke answer but arrows up and down are not working so I can not see my history questions. 
Arrows left and right also not working  so I can not correct my spelling mistakes .  


All those things were working  fine earlier.

---

## Issue #N/A: Is possible to hide start / stop tokens? 

**Link**: https://github.com/ggml-org/llama.cpp/issues/7100
**State**: closed
**Created**: 2024-05-06T10:06:34+00:00
**Closed**: 2024-05-08T14:32:33+00:00
**Comments**: 4
**Labels**: enhancement

### Description

Example llamacpp output in the terminal with llama3 

````
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability.
> <|start_header_id|>user<|end_header_id|>

name countries starts with letter Z. \
<|start_header_id|>assistant<|end_header_id|>

Here are some countries that start with the letter Z:

1. Zambia
2. Zimbabwe

Note: These are the only two countries whose names begin with the letter Z.

Is there anything else I can help you with?<|eot_id|>

> <|start_header_id|>user<|end_header_id|>

````


Is possible to hide system, start, stop, in-prefix and in-suffif tokens in the terminal ? 

---

## Issue #N/A: metal : compile-time kernel args and params

**Link**: https://github.com/ggml-org/llama.cpp/issues/4085
**State**: open
**Created**: 2023-11-15T11:09:39+00:00
**Comments**: 4
**Labels**: performance, research 🔬, roadmap

### Description

I was just thinking about this idea, so writing it down for future research.

We should be able to fairly easy generate model-specific Metal code that has hardcoded kernels for every single node in the computation graph. The idea is to make an initial pass of a certain graph where we record all kernel calls with their respective argument values and parameters and then generate a model-specific MSL source file with all these kernels instances - either copy-paste or via templates. I guess this is something similar to what people call JIT. Wondering what kind of speed-up we will be able to see with this strategy.

---

## Issue #N/A: ggml : refactor ggml-cpu.c into multiple C++ source files

**Link**: https://github.com/ggml-org/llama.cpp/issues/10180
**State**: open
**Created**: 2024-11-05T07:12:48+00:00
**Comments**: 17
**Labels**: refactoring, roadmap

### Description

As per recent discussions (e.g. https://github.com/ggerganov/llama.cpp/pull/10144#pullrequestreview-2411814357), we should split the large `ggml-cpu.c` implementation into smaller modules - similar to how the CUDA backend is organized. We should utilize ~C++11~ C++ to reduce code duplication.

---

