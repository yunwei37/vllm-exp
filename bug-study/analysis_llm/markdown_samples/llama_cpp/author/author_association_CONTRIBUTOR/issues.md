# author_association_CONTRIBUTOR - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 29

### Label Distribution

- bug-unconfirmed: 12 issues
- enhancement: 11 issues
- stale: 9 issues
- medium severity: 3 issues
- bug: 1 issues
- high priority: 1 issues
- critical severity: 1 issues
- duplicate: 1 issues
- high severity: 1 issues
- need more info: 1 issues

---

## Issue #N/A: Misc. bug: Quantizing Olmo models with imatrix failing on some sizes

**Link**: https://github.com/ggml-org/llama.cpp/issues/11764
**State**: closed
**Created**: 2025-02-08T19:10:22+00:00
**Closed**: 2025-03-25T01:07:39+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

version 4585

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-quantize

### Command line

```shell
./llama-quantize --imatrix /models/OLMo-2-1124-7B-Instruct-GGUF/allenai_OLMo-2-1124-7B-Instruct.imatrix /models/OLMo-2-1124-7B-Instruct-GGUF/allenai_OLMo-2-1124-7B-Instruct-f32.gguf /models/OLMo-2-1124-7B-Instruct-GGUF/allenai_OLMo-2-1124-7B-Instruct-Q5_K_M.gguf Q5_K_M
```

### Problem description & steps to reproduce

Without imatrix I don't get any issues.

Quantizing OLMo-2 7B to Q5_K_M, Q5_K_S, Q4_K_M, and Q4_K_S, and Q2_K with imatrix results in:

```
blk.7.attn_q.weight - [ 4096,  4096,     1,     1], type =    f32, converting to q4_K .. ggml_validate_row_data: found nan value at block 48
ggml_validate_row_data: found nan value at block 16
```
```
blk.7.attn_q.weight - [ 4096,  4096,     1,     1], type =    f32, converting to q5_K .. ggml_validate_row_data: found nan value at block 48
ggml_validate_row_data: found n

[... truncated for brevity ...]

---

## Issue #N/A: broken state save/restore since "store offset as opt arg for ggml_view_xd() operators"

**Link**: https://github.com/ggml-org/llama.cpp/issues/1777
**State**: closed
**Created**: 2023-06-09T11:04:16+00:00
**Closed**: 2023-07-07T20:01:07+00:00
**Comments**: 2

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

`llama_copy_state_data()` should do what it's documented to do

# Current Behavior

`llama_copy_state_data()` causes a Segmentation fault

# Environment and Context

* Physical (or virtual) hardware you are using, e.g. for Linux:

```
Architecture:         

[... truncated for brevity ...]

---

## Issue #N/A: Feature Proposal: Server Model Switching at Runtime

**Link**: https://github.com/ggml-org/llama.cpp/issues/13027
**State**: closed
**Created**: 2025-04-19T18:17:46+00:00
**Closed**: 2025-06-29T01:08:21+00:00
**Comments**: 8
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

I would like to adapt the server (or create an alternate server) so that it is more suited to being changed during runtime.  My primary goal in doing so is to be able to switch models on the fly.

### Motivation

In local inference, I find that no single model is best for all tasks and I switch between models frequently using TabbyAPI. I would like to be able to have this functionality available directly in llama.cpp to be able to make use of GGUF files and the llama.cpp ecosystem.

### Possible Im

[... truncated for brevity ...]

---

## Issue #N/A: Eval: HIP: Llama-server multi-instance lockup

**Link**: https://github.com/ggml-org/llama.cpp/issues/13100
**State**: closed
**Created**: 2025-04-24T15:46:03+00:00
**Closed**: 2025-06-09T01:08:02+00:00
**Comments**: 4
**Labels**: stale

### Description

**Follow-up on the https://github.com/ggml-org/llama.cpp/issues/12991**

According to rocgdb backtrace, threads that are working with gpus are stuck somewhere in the libhsa-runtime

```
Thread 2 (Thread 0x7fffd7fff6c0 (LWP 11602) "llama-server"):
#0  __GI___ioctl (fd=3, request=3222817548) at ../sysdeps/unix/sysv/linux/ioctl.c:36
#1  0x00007fffd8549400 in ?? () from /opt/rocm-6.4.0/lib/llvm/bin/../../../lib/libhsa-runtime64.so.1
#2  0x00007fffd8541f1f in ?? () from /opt/rocm-6.4.0/lib/llvm/bin/../../../lib/libhsa-runtime64.so.1
#3  0x00007fffd84be632 in ?? () from /opt/rocm-6.4.0/lib/llvm/bin/../../../lib/libhsa-runtime64.so.1
#4  0x00007fffd84a1aee in ?? () from /opt/rocm-6.4.0/lib/llvm/bin/../../../lib/libhsa-runtime64.so.1
#5  0x00007fffd8439241 in ?? () from /opt/rocm-6.4.0/lib/llvm/bin/../../../lib/libhsa-runtime64.so.1
#6  0x00007ffff749caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#7  0x00007ffff7529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_6

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: missing `<image_soft_token>` (id 262144 ) in Gemma3-it token map?

**Link**: https://github.com/ggml-org/llama.cpp/issues/12876
**State**: closed
**Created**: 2025-04-10T12:50:33+00:00
**Closed**: 2025-04-10T16:31:12+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

From the official `tokenizer.json`:
```json
    {
      "id": 262144,
      "content": "<image_soft_token>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }
```
https://huggingface.co/google/gemma-3-4b-it/blob/main/tokenizer.json
also here:
https://huggingface.co/google/gemma-3-4b-it/blob/main/added_tokens.json

There is no token with id 262144 in the converted gguf models. (it's also missing from the Google's own qat gguf models).

---

## Issue #N/A: Converting a StableLM fine tuned model fails with `Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.`

**Link**: https://github.com/ggml-org/llama.cpp/issues/4171
**State**: closed
**Created**: 2023-11-22T18:06:09+00:00
**Closed**: 2023-11-24T14:02:51+00:00
**Comments**: 4
**Labels**: bug-unconfirmed

### Description

# Prerequisites

Tested on latest commit, 8e672efe632bb6a7333964a255c4b96f018b9a65 , and also on commits from yesterday.

# Current Behavior

Trying to convert model https://huggingface.co/pansophic/rocket-3B

Results in:
```
 [pytorch2] tomj@MC:/workspace/git/gguf-llama (master ✘)✭ ᐅ python3 ./convert-hf-to-gguf.py /workspace/process/pansophic_rocket-3b/source --outtype f16 --outfile /workspace/process/pansophic_rocket-3b/gguf/rocket-3b.fp16.gguf
Loading model: source
gguf: This GGUF file is for Little Endian only
Set model parameters
Set model tokenizer
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
gguf: Adding 50009 merge(s).
gguf: Setting special token type bos to 0
gguf: Setting special token type eos to 0
gguf: Setting special token type unk to 0
Exporting model to '/workspace/process/pansophic_rocket-3b/gguf/rocket-3b.fp16.gguf'
gguf: loading model part 'pytorch_model.bin'
Traceback (mo

[... truncated for brevity ...]

---

## Issue #N/A: Latest release wont compile (ggml.c)

**Link**: https://github.com/ggml-org/llama.cpp/issues/1120
**State**: closed
**Created**: 2023-04-22T10:05:00+00:00
**Closed**: 2023-04-24T15:38:28+00:00
**Comments**: 5

### Description

Release: master-36b4f7e

ggml.c: In function ‘bytes_from_nibbles_16’:
ggml.c:439:19: warning: implicit declaration of function ‘_mm_loadu_si64’; did you mean ‘_mm_loadl_epi64’? [-Wimplicit-function-declaration]

     __m128i tmp = _mm_loadu_si64( ( const __m128i* )rsi );
                   ^~~~~~~~~~~~~~
                   _mm_loadl_epi64
ggml.c:439:19: error: incompatible types when initializing type ‘__m128i’ {aka ‘__vector(2) long long int’} using type ‘int’


Using:
cc (Ubuntu 8.4.0-3ubuntu2) 8.4.0
g++ (Ubuntu 8.4.0-3ubuntu2) 8.4.0

---

## Issue #N/A: with the newest builds i only get gibberish output

**Link**: https://github.com/ggml-org/llama.cpp/issues/1735
**State**: closed
**Created**: 2023-06-07T08:06:19+00:00
**Closed**: 2023-06-15T08:50:50+00:00
**Comments**: 81
**Labels**: bug, high priority

### Description

After the CUDA refactor PR #1703 by @JohannesGaessler was merged i wanted to try it out this morning and measure the performance difference on my ardware.
I use my standard prompts with different models in different sizes.

I use the prebuild versions win-cublas-cu12.1.0-xx64

With the new builds I only get gibberish as a response for all prompts used and all models.
It looks like a random mix of words in different languages.

On my current PC I can only use the win-avx-x64 version, here I still get normal output.

I will use the Cuda-pc again in a few hours, then I can provide sample output or more details.
Am I the only one with this problem?

---

## Issue #N/A: Bug: Initializing KV Cache Spikes Memory, Crashing on Android

**Link**: https://github.com/ggml-org/llama.cpp/issues/9671
**State**: closed
**Created**: 2024-09-27T22:08:13+00:00
**Closed**: 2024-09-29T20:47:14+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

Hi,

You may already know about the memory spike, given #7474.

For those unfamiliar, `ggml_backend_cpu_buffer_clear` calls `memset`, which initializes the allocated buffer (as big as 16 GiB for full context on Llama 3.1) to `0`s, spiking memory and, on Android, leading to a system crash --
- If in Termux, Android kills it
- If in `adb shell`, Android hangs and reboots

As far as I can tell, there are no guards for when `llama.cpp` might over-allocate _and_ over-initialize memory — this may be intended, but it seems to defeat the purpose of `mmap`.

Please share your perspective on this behavior; I understand it to be undefined. With limited experience, I see a number of potential solutions: 
1. Make `ggml_backend_buffer_clear` truly optional
	- Alternatively, skip it in certain environments
2. Use `ggml_backend_cpu_buffer_memset_tensor` in the `alloc_tensor_range` loop instead to avoid bulk initialization, perhaps as part of `ggml_tallocr_alloc` or in 

[... truncated for brevity ...]

---

## Issue #N/A: Parallelize dequantization or fp format conversion when using blas

**Link**: https://github.com/ggml-org/llama.cpp/issues/4988
**State**: closed
**Created**: 2024-01-16T21:03:44+00:00
**Closed**: 2024-01-22T13:15:09+00:00
**Comments**: 5
**Labels**: enhancement

### Description

In ggml_compute_forward_mul_mat@ggml.c, gemm parallelism is managed by blas library. However, this disables multithreading on dequantizing weights, which may be a bottleneck.

I have performed some ugly modifications for comparing performance.

```
› MKL_NUM_THREADS=8 ./llama-bench -m Llama-2-7b-chat-q4km.gguf -t 8
| model                          |       size |     params | backend    |    threads | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ---------: | ---------- | ---------------: |
| llama 7B Q4_K - Medium         |   3.80 GiB |     6.74 B | BLAS       |          8 | pp 512     |     55.24 ± 3.16 |
| llama 7B Q4_K - Medium         |   3.80 GiB |     6.74 B | BLAS       |          8 | tg 128     |     21.12 ± 1.89 |

build: 862f5e4 (1886)

› MKL_NUM_THREADS=8 ./llama-bench -m Llama-2-7b-chat-q4km.gguf -t 8
| model                          |       size |     params | backend    |    threads | test       |  

[... truncated for brevity ...]

---

## Issue #N/A: High performance API

**Link**: https://github.com/ggml-org/llama.cpp/issues/321
**State**: closed
**Created**: 2023-03-20T11:34:40+00:00
**Closed**: 2023-03-20T19:22:42+00:00
**Comments**: 8
**Labels**: duplicate, enhancement

### Description

Hey!

I'd love to see this project being able to be used through some TCP socket with a very optimized protocol. One it may make use of something like protobuf, or even grpc.
I think everyone agrees HTTP would be a complete overkill specially for a project focused on high performance. :laughing: 

Thanks
Niansa

---

## Issue #N/A: Support for H2O Danube3 Family of Models

**Link**: https://github.com/ggml-org/llama.cpp/issues/8518
**State**: closed
**Created**: 2024-07-16T16:02:31+00:00
**Closed**: 2024-07-16T18:37:15+00:00
**Comments**: 5
**Labels**: enhancement

### Description

### Prerequisites

- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

I have tested the GGUF (https://huggingface.co/h2oai/h2o-danube3-500m-chat-GGUF) with Android example from llama.cpp and it is working for me. TTFT and Throughput speed are 🔥 

### Motivation

v v small models to run on edge devices and are on a par with qwen2 (0.5b) and phi-3 (4b)

### Possible Implementation

Should work directly since model arch is LlamaForCausalLM. I will check if it's supported directly without any change. Prompt support might be needed though. 

<|prompt|>Why is drinking water so healthy?</s><|answer|>

I will check these 2 and report back
1. Model Conversion to GGUF
2. Running GGUF model

---

## Issue #N/A: Feature Request: dynamic speculation (i.e. dynamic draft-max)

**Link**: https://github.com/ggml-org/llama.cpp/issues/11933
**State**: closed
**Created**: 2025-02-17T20:43:30+00:00
**Closed**: 2025-02-18T03:29:48+00:00
**Comments**: 3
**Labels**: enhancement

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Adjust draft-max on the fly during generation to optimize speculative generation performance.

### Motivation

Speculative generation works best in structured text (so, big highly predictable chunks) like code. The larger the draft-max the more speedup is possible. But then, a large draft-max wastes time when you're in less structured text. So it seems some sort of dynamic adjustment of draft-max would be ideal.

### Possible Implementation

It looks like [this has been tried](https://huggingface.c

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Pixtral by Mistral support (pixtral-12b-240910)

**Link**: https://github.com/ggml-org/llama.cpp/issues/9440
**State**: closed
**Created**: 2024-09-11T18:03:29+00:00
**Closed**: 2025-02-08T01:07:14+00:00
**Comments**: 14
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Dear llama.cpp team,

Mistral has just released Pixtral and I would like to request support for it, if possible.

Here are some relevant links:

**X announcement:** https://x.com/mistralai/status/1833758285167722836

**Magnet link:** `xt=urn:btih:7278e625de2b1da598b23954c13933047126238a&dn=pixtral-12b-240910&tr=udp%3A%2F%http://2ftracker.opentrackr.org/%3A1337%2Fannounce&tr=udp%3A%2F%http://2fopen.demonii.com/%3A1337%2Fannounce&tr=http%3A%2F%http://2ftracker.ipv6tracker.org/%3A80

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

## Issue #N/A: Feature Request: [CANN] Use the RoPE operator provided by aclnn

**Link**: https://github.com/ggml-org/llama.cpp/issues/10396
**State**: closed
**Created**: 2024-11-19T02:16:30+00:00
**Closed**: 2024-11-28T07:59:17+00:00
**Comments**: 1
**Labels**: enhancement

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Use the RoPE operator provided by aclnn instead of manually coding RoPE to improve the performance of the RoPE operator.

### Motivation

The RoPE performance of the aclnn operator library will be better!

### Possible Implementation

Adjust the RoPE operator provided by the operator library using aclnn.

---

## Issue #N/A: examples/server -n -1 never stops generating

**Link**: https://github.com/ggml-org/llama.cpp/issues/3612
**State**: closed
**Created**: 2023-10-13T07:36:35+00:00
**Closed**: 2023-10-16T19:29:55+00:00
**Comments**: 1

### Description

I'm trying to get the same output from examples/server as from examples/main, but without any luck. Having -n -1 as param in for examples/server just makes it generate the whole ctx. While in the examples/main the generator stops when it has generated a good answer it seems. Does it have to do with stop_words? Any ideas are welcome, thanks!

---

## Issue #N/A: Please add llama.cui to the list of UI in Readme

**Link**: https://github.com/ggml-org/llama.cpp/issues/4487
**State**: closed
**Created**: 2023-12-15T19:59:13+00:00
**Closed**: 2024-04-02T01:10:55+00:00
**Comments**: 2
**Labels**: enhancement, stale

### Description

Not an issue just a request. Could you please add Llama.cui to the list of UIs:

https://github.com/dspasyuk/llama.cui

Thank you in advance!

---

## Issue #N/A: Save/Load Just One Sequence

**Link**: https://github.com/ggml-org/llama.cpp/issues/5843
**State**: closed
**Created**: 2024-03-03T01:46:36+00:00
**Closed**: 2024-04-20T07:12:57+00:00
**Comments**: 4
**Labels**: enhancement

### Description

# Feature Description

Would it be possible to create functions that looked something like this:
 - `llama_kv_save_seq(struct llama_context * ctx, llama_seq_id seq_id, uint8_t * dst);`
 - `llama_kv_load_seq(struct llama_context * ctx, llama_seq_id seq_id, uint8_t * src);`

# Motivation

In llama.cpp it is possible to save and load the _entire_ context state in one operation with `llama_copy_state_data` and `llama_set_state_data`. For example this could be used to evaluate a large system prompt once, save it to disk, and then load the state every time a new conversation is started.

However with the batch decoding this isn't really possible. If you have many sequences being evaluated at once you can only load and save them _all_ simultaneously.


---

## Issue #N/A: Bug: Release build on Windows stuck

**Link**: https://github.com/ggml-org/llama.cpp/issues/9242
**State**: closed
**Created**: 2024-08-29T16:26:53+00:00
**Closed**: 2024-10-13T01:07:32+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

When building llama.cpp in Release mode it stuck on build. with Debug config it compiles fast.

### Name and Version

1d1ccce67613674c75c9c7e3fa4c1e24e428ba48

### What operating system are you seeing the problem on?

Windows

### Relevant log output


Build
<details>

```console
cmake -B build . -DCMAKE_BUILD_TYPE=Release
-- Building for: Visual Studio 17 2022
-- Selecting Windows SDK version 10.0.22000.0 to target Windows 10.0.22631.
-- The C compiler identification is MSVC 19.40.33812.0
-- The CXX compiler identification is MSVC 19.40.33812.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.40.33807/bin/Hostx64/x64/cl.exe - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX com

[... truncated for brevity ...]

---

## Issue #N/A: Compile bug: ggml-cuda/opt-step-adamw.cu error: identifier "__Poly8x8_t" is undefined on Jetson Orin AGX

**Link**: https://github.com/ggml-org/llama.cpp/issues/12826
**State**: closed
**Created**: 2025-04-08T11:20:54+00:00
**Closed**: 2025-05-29T01:07:57+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, stale

### Description

### Git commit

1466621e738779eefe1bb672e17dc55d63d166bb

### Operating systems

Linux

### GGML backends

CUDA

### Problem description & steps to reproduce

I am trying to build llama.cpp with a CUDA backend on the NVIDIA AGX Orin, but it seems there are too many compilation errors that occur when including `arm_neon.h`.

My environment:
1. JetPack: 6.2
2. GCC: 11.4.0
3. CUDA: 12.6.68
4. Git: 2.34.1
5. Arch: aarch64

How to reproduce:

1. `git clone https://github.com/ggml-org/llama.cpp -b b5074`
2. `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DLLAMA_CURL=OFF`
3. `cmake --build build --parallel`
4. Get the errors:

```
/usr/lib/gcc/aarch64-linux-gnu/11/include/arm_neon.h(44): error: identifier "__Poly8x8_t" is undefined
  typedef __Poly8x8_t poly8x8_t;
/usr/lib/gcc/aarch64-linux-gnu/11/include/arm_neon.h(1195): error: identifier "__builtin_aarch64_addhnv4si" is undefined
    return (int16x4_t) __builtin_aarch64_addhnv4si (__a, __b);
```

### First Ba

[... truncated for brevity ...]

---

## Issue #N/A: IOT instruction (core dumped)

**Link**: https://github.com/ggml-org/llama.cpp/issues/2341
**State**: closed
**Created**: 2023-07-23T12:17:37+00:00
**Closed**: 2024-04-09T01:07:36+00:00
**Comments**: 1
**Labels**: stale

### Description

```shell
$ nix build '.#opencl'
$ result/bin/benchmark
main: build = 0 (unknown)
Starting Test
Allocating Memory of size 794558464 bytes, 757 MB
ggml_opencl: selecting platform: 'Intel(R) OpenCL Graphics'
ggml_opencl: selecting device: 'Intel(R) Iris(R) Xe Graphics'
ggml_opencl: device FP16 support: true
Creating new tensors

------ Test 1 - Matrix Mult via F32 code ------------------------------------------------------------------------------
n_threads=1
            m11: type = 0 (  f32) ne = 11008 x  4096 x     1, nb = (    4, 44032, 180355072) - Sum of tensor m11 is 16777216.00
             m2: type = 0 (  f32) ne = 11008 x   128 x     1, nb = (    4, 44032, 5636096) - Sum of tensor m2 is 2818048.00
GGML_ASSERT: /build/wnslnw6pk8d4c8k0b8w4w4qz45wgy9hw-source/ggml-opencl.cpp:1524: false
zsh: IOT instruction (core dumped)  result/bin/benchmark
```

---

## Issue #N/A: Bug: Last 2 Chunks In Streaming Mode Come Together In Firefox

**Link**: https://github.com/ggml-org/llama.cpp/issues/9502
**State**: closed
**Created**: 2024-09-16T02:14:04+00:00
**Closed**: 2024-09-17T06:48:47+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

When using `/completion` with `stream: true`, the last 2 JSON chunks come together in Firefox, but Chrome seems to handle it fine, so it might be a Firefox bug.

Looking further into this, it seems like HTTP `Transfer-Encoding: chunked` requires each chunk to be terminated with `\r\n`, but here `\n\n` is used instead:

https://github.com/ggerganov/llama.cpp/blob/6262d13e0b2da91f230129a93a996609a2f5a2f2/examples/server/utils.hpp#L296-L299

This doesn't seem to be just a Windows requirement, but listed as part of the HTTP specification:
[HTTP Chunked Transfer Coding](https://httpwg.org/specs/rfc9112.html#chunked.encoding)

More information, including an example `chunked` response:
[Transfer-Encoding Directives](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Transfer-Encoding#directives)

### Name and Version

llama-server.exe
version: 3761 (6262d13e)
built with MSVC 19.29.30154.0 for x64

### What operating system are you seeing the problem on?


[... truncated for brevity ...]

---

## Issue #N/A: Bug: when arrive max ctx, model output garbage

**Link**: https://github.com/ggml-org/llama.cpp/issues/7578
**State**: closed
**Created**: 2024-05-28T02:22:16+00:00
**Closed**: 2024-06-18T03:20:17+00:00
**Comments**: 2
**Labels**: need more info, bug-unconfirmed, medium severity

### Description

### What happened?

This part has problem in cuda version. if set ngl>0, when arrive max ctx and next turn to chat, the model output garbage.

llama_kv_cache_seq_rm (ctx, 0, params.n_keep            , params.n_keep + n_discard);
llama_kv_cache_seq_add(ctx, 0, params.n_keep + n_discard, n_past, -n_discard);

if set ngl =0, everythings ok.
### Name and Version

llama.cpp-b3014
main.exe --version
version: 247 (6765407)
built with MSVC 19.37.32822.0 for x64

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

---

## Issue #N/A: Bug: (CUDA) Corrupted output when offloading to multiple GPUs

**Link**: https://github.com/ggml-org/llama.cpp/issues/8685
**State**: closed
**Created**: 2024-07-25T10:02:08+00:00
**Closed**: 2024-08-07T11:29:04+00:00
**Comments**: 22
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

### Problem

Some models produce a corrupted output when offloading to multiple CUDA GPUs. The problem disappears when offloading to a single GPU or using CPU only.

I was able to reproduce the problem in:
- [llama 3.0 8B](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct-Q6_K.gguf)
- llama 3.1 8b
- glm-4-9b

while I was unable to reproduce it in:
- Mistral Nemo 

### Bug 1

When offloading to multiple GPUs, the model gives the wrong answer. It seems unable to parse the prompt correctly.

### Bug 2

When a second prompt is sent to the model, the model reuses information from the first prompt.

### Steps to reproduce Bug 1

- download [llama 3.0 8B](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct-Q6_K.gguf) from HF
- download my sample prompt: 
[llama-multi-gpu-bug.txt](https://github.com/user-attachments/files/16373846/llama-multi-gpu-bug.txt)
-

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: llama-server.exe silently crashes (ucrtbased.dll) after 2-3 requests in a dialogue

**Link**: https://github.com/ggml-org/llama.cpp/issues/13877
**State**: closed
**Created**: 2025-05-28T23:55:43+00:00
**Closed**: 2025-05-30T23:00:00+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

### Name and Version

version: 5528 (53ae3064)
built with MSVC 19.43.34810.0 for x64

### Operating systems

Windows

### GGML backends

Vulkan

### Hardware

9070xt, A770

### Models

Model Qwen3-30B-A3B-UD-Q5_K_XL.gguf  with fixed chat templates
SHA256: f284af35140194f073985a093f6d257cb7060784ecbfeb52c15f9545dfa4f434

### Problem description & steps to reproduce

llama-server.exe -m Qwen3-30B-A3B-UD-Q5_K_XL.gguf -ngl 99 -c 15000 --port 8000 --jinja 

Server silently terminates in some dialogues, typically after 2-3 requests within a single dialogue.
The Windows Event Log records a crash event for llama-server.exe, with ucrtbased.dll as the faulting module.

### First Bad Commit

https://github.com/ggml-org/llama.cpp/commit/e121edc4324a640be11b7e567edd39b721b0f8e4

b5486

### Relevant log output

```shell
srv  update_chat_: Parsing chat message: <think>

Parsing input with format Hermes 2 Pro: <think>

Partial parse: </think>
Parsed message: {"role":"assistant","content":null}
srv    

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Support Llama-Nemotron-Nano-VL-8B-V1

**Link**: https://github.com/ggml-org/llama.cpp/issues/14015
**State**: open
**Created**: 2025-06-04T16:13:31+00:00
**Comments**: 0
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Nvidia just released a new VLM: https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1

- Text model: [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- Vision encoder: [nvidia/C-RADIOv2-H](https://huggingface.co/nvidia/C-RADIOv2-H)

On `master` (`3e63a58e`), the command:

```zsh
python llama.cpp/convert_hf_to_gguf.py --outfile /opt/workspace/gguf/Llama-Nemotron-Nano-VL-8B-V1-Q8_0.gguf --outtype bf16 /opt/workspace/hf/Llama-Nemotron-Nano-VL-8B-V1/
```

curren

[... truncated for brevity ...]

---

## Issue #N/A: [Feature Request]: 32k context

**Link**: https://github.com/ggml-org/llama.cpp/issues/2353
**State**: closed
**Created**: 2023-07-23T22:14:34+00:00
**Closed**: 2023-07-23T22:15:55+00:00
**Comments**: 0

### Description

can the internal prompt limit be raised to 32k for llama 2 models? I'm only assuming this works because the llama 2 context is double the previous.
```
main: error: prompt is too long (30474 tokens, max 16380
```

---

## Issue #N/A: Feature Request: moondream2 vlm support in mtmd

**Link**: https://github.com/ggml-org/llama.cpp/issues/13332
**State**: closed
**Created**: 2025-05-06T07:16:00+00:00
**Closed**: 2025-05-25T12:04:50+00:00
**Comments**: 5
**Labels**: enhancement

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Please support moondream2 model in mtmd

moondream2 is a small vlm that is incredibly good for its small size: https://huggingface.co/vikhyatk/moondream2/

I believe that an earlier version of moondream was supported by llama.cpp at some point, but newer versions are not.

### Motivation

moondream2 is one of the best models for memory constrained scenarios such as the edge.  adding llama.cpp support will enable inference with vulkan, which is currently lacking.

### Possible Implementation

_No re

[... truncated for brevity ...]

---

## Issue #N/A: Missing public header for ggml-backend.h and ggml-alloc.h

**Link**: https://github.com/ggml-org/llama.cpp/issues/5011
**State**: closed
**Created**: 2024-01-18T01:45:17+00:00
**Closed**: 2024-01-18T21:36:33+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

The public header defined in CMakeLists.txt for llama.h (used for library level C FFI integrations) depends on the header `ggml-backend.h` now, but as it is not a defined public header in CMakeLists.h, it cannot be found in the CMake output. It seems like this is an omission, and the CMakeLists.txt should include this file as a public header.

---

