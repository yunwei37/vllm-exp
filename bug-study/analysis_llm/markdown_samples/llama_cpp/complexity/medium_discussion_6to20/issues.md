# medium_discussion_6to20 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 2
- Closed Issues: 28

### Label Distribution

- stale: 15 issues
- bug-unconfirmed: 12 issues
- enhancement: 6 issues
- good first issue: 3 issues
- bug: 2 issues
- build: 1 issues
- android: 1 issues
- help wanted: 1 issues
- model: 1 issues
- high severity: 1 issues

---

## Issue #N/A: Eval bug: NVIDIA Jetson AGX Xavier CUDA Compatibility Issue with llama.cpp

**Link**: https://github.com/ggml-org/llama.cpp/issues/13629
**State**: closed
**Created**: 2025-05-19T09:01:27+00:00
**Closed**: 2025-05-20T03:17:56+00:00
**Comments**: 8
**Labels**: bug-unconfirmed

### Description

### Name and Version

ggml_cuda_init: failed to initialize CUDA: CUDA driver version is insufficient for CUDA runtime version
version: 0 (unknown) [llama.cpp-b5415](https://github.com/ggml-org/llama.cpp/releases/tag/b5415)
built with cc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0 for aarch64-linux-gnu


### Operating systems

Linux

### GGML backends

CUDA

### Hardware

Jetson AGX Xavier

### Models

_No response_

### Problem description & steps to reproduce

I'm experiencing a CUDA compatibility issue with the latest version of llama.cpp on my Jetson AGX Xavier device (Ubuntu 20.04). Details:

- Device: Jetson AGX Xavier
- OS: Ubuntu 20.04
- CUDA Version: 12.2
- Issue: While an older release of llama.cpp (b4835) works correctly with CUDA, the latest version fails to run after successful compilation
- Error Message: `ggml_cuda_init: failed to initialize CUDA: CUDA driver version is insufficient for CUDA runtime version`
- Working Output (with older version):` ggml_cuda_init: found 1 CUDA d

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Severe Performance Degradation on Q4_0 CPU-only with MacOS / Apple Silicon M2, after PR#9921 / Version 4081

**Link**: https://github.com/ggml-org/llama.cpp/issues/10435
**State**: open
**Created**: 2024-11-20T17:06:21+00:00
**Comments**: 12
**Labels**: bug

### Description

### What happened?

Prior to PR #9921 / Version 4081 the -ngl 0 Q4_0 llama performance was significantly higher (more than 10x) than afterwards.
(hardware: Apple MacBook Air M2 10 GPU 24GB RAM)

before PR:
make clean
git checkout ae8de6d
make -j llama-bench
./llama-bench -p 512 -n 128 -t 4 -ngl 0 -m ...model...
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | Metal,BLAS |       4 |         pp512 |         60.48 ± 0.49 |
| llama 7B Q4_0                  |   3.56 GiB |     6.74 B | Metal,BLAS |       4 |         tg128 |         14.89 ± 0.20 |
| llama 7B Q4_0_4_4              |   3.56 GiB |     6.74 B | Metal,BLAS |       4 |         pp512 |         63.50 ± 2.47 |
| llama 7B Q4_0_4_4              |   3.56 GiB |     6.74 

[... truncated for brevity ...]

---

## Issue #N/A: ggml_init_cublas: no CUDA devices found, CUDA will be disabled

**Link**: https://github.com/ggml-org/llama.cpp/issues/6184
**State**: closed
**Created**: 2024-03-20T18:20:41+00:00
**Closed**: 2024-03-21T23:24:53+00:00
**Comments**: 8
**Labels**: bug-unconfirmed

### Description

System:

```
> uname -m && cat /etc/*release
x86_64
DISTRIB_ID=Pop
DISTRIB_RELEASE=22.04
DISTRIB_CODENAME=jammy
DISTRIB_DESCRIPTION="Pop!_OS 22.04 LTS"
NAME="Pop!_OS"
VERSION="22.04 LTS"
ID=pop
ID_LIKE="ubuntu debian"
PRETTY_NAME="Pop!_OS 22.04 LTS"
VERSION_ID="22.04"
HOME_URL="https://pop.system76.com"
SUPPORT_URL="https://support.system76.com"
BUG_REPORT_URL="https://github.com/pop-os/pop/issues"
PRIVACY_POLICY_URL="https://system76.com/privacy"
VERSION_CODENAME=jammy
UBUNTU_CODENAME=jammy
LOGO=distributor-logo-pop-os
```

I compiled `llama.cpp` with [`cuBLAS` support](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#cublas):

```
 make clean && make LLAMA_CUBLAS=1
```

I then run `llama.cpp` with:

```
./main -m ./models/Llama-2-7B-GGUF/llama-2-7b.Q8_0.gguf -p "test"
```

I see the following:

```
ggml_init_cublas: no CUDA devices found, CUDA will be disabled
```

I haven't seen any other issues with this error, so not sure what to 

[... truncated for brevity ...]

---

## Issue #N/A: [User] Android build fails with "ld.lld: error: undefined symbol: clGetPlatformIDs"

**Link**: https://github.com/ggml-org/llama.cpp/issues/3525
**State**: closed
**Created**: 2023-10-07T12:03:41+00:00
**Closed**: 2023-10-07T15:12:36+00:00
**Comments**: 17
**Labels**: build, android

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

I am trying to use [this tutorial](https://github.com/ggerganov/llama.cpp#building-the-project-using-termux-f-droid) to compile llama.cpp.

# Current Behavior

Compilation failed.

# Environment and Context

Please provide detailed information about your compu

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Add OLMoE

**Link**: https://github.com/ggml-org/llama.cpp/issues/9317
**State**: closed
**Created**: 2024-09-04T22:33:20+00:00
**Closed**: 2025-01-13T01:07:36+00:00
**Comments**: 9
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Add this model (and other variants) https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct

### Motivation

We recently released the OLMoE model at Ai2. 1.3b active / 6.9b total param MoE model. Seems solid, and we'd love people to use it.

### Possible Implementation

Should be able to quickly use mix of existing OLMo implementation + Transformers version https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmoe/modeling_olmoe.py

---

## Issue #N/A: Problem with gpu (help)

**Link**: https://github.com/ggml-org/llama.cpp/issues/2288
**State**: closed
**Created**: 2023-07-20T09:41:52+00:00
**Closed**: 2024-04-09T01:07:46+00:00
**Comments**: 10
**Labels**: stale

### Description

Hello, I am completly newbie, when it comes to the subject of llms
I install some ggml model to oogabooga webui And I try to use it. It works fine, but only for RAM. For VRAM only uses 0.5gb, and I don't have any possibility to change it (offload some layers to GPU), even pasting in webui line "--n-gpu-layers 10" dont work. So I stareted searching, one of answers is command:

```
pip uninstall -y llama-cpp-python
set CMAKE_ARGS="-DLLAMA_CUBLAS=on"
set FORCE_CMAKE=1
pip install llama-cpp-python --no-cache-dir
```
But that dont work for me. I got after paste it:

 ```
[end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for llama-cpp-python
Failed to build llama-cpp-python
ERROR: Could not build wheels for llama-cpp-python, which is required to install pyproject.toml-based projects
```
And it completly broke llama folder.. It uninstall it, and did nothing more. I need to update webui to f

[... truncated for brevity ...]

---

## Issue #N/A: error: implicit declaration of function ‘vld1q_s8_x4’; did you mean ‘vld1q_s8_x2’?

**Link**: https://github.com/ggml-org/llama.cpp/issues/7147
**State**: closed
**Created**: 2024-05-08T13:28:52+00:00
**Closed**: 2024-07-16T01:06:49+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, stale

### Description

This error is on M3 Max inside docker container with linux.
Docker file:
```
FROM python:3.10.12-slim-buster

USER root
RUN apt-get update && apt-get install cmake libopenblas-dev build-essential pkg-config git -y

WORKDIR /opt
COPY ./requirements/cpu.requirements.txt ./requirements.txt
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip3 install --upgrade -r requirements.txt

COPY infra/llm_server_cpu/server_config.json server_config.json

EXPOSE 8085
CMD ["python3", "-m", "llama_cpp.server", "--config_file", "server_config.json"]
```

The error trace:
```
2   × Building wheel for llama-cpp-python (pyproject.toml) did not run successfully.
25.92   │ exit code: 1
25.92   ╰─> [158 lines of output]
25.92       *** scikit-build-core 0.9.3 using CMake 3.29.2 (wheel)
25.92       *** Configuring CMake...
25.92       loading initial cache file /tmp/tmph68_ek8q/build/CMakeInit.txt
25.92       -- The C compiler identification is GNU 8.3.0
25.92       -- 

[... truncated for brevity ...]

---

## Issue #N/A: Completion of error handling

**Link**: https://github.com/ggml-org/llama.cpp/issues/7489
**State**: closed
**Created**: 2024-05-23T10:56:09+00:00
**Closed**: 2024-07-27T21:43:04+00:00
**Comments**: 8
**Labels**: good first issue, bug-unconfirmed

### Description

Would you like to add more error handling for return values from functions like the following?
* [fprintf](https://pubs.opengroup.org/onlinepubs/9699919799/functions/fprintf.html "Print formatted output.") ⇒ [print_grammar_char](https://github.com/ggerganov/llama.cpp/blob/9b82476ee9e73065a759f8bcc4cf27ec7ab2ed8c/common/grammar-parser.cpp#L313-L320)
* [malloc](https://pubs.opengroup.org/onlinepubs/9699919799/functions/malloc.html "Memory allocation") ⇒ [ggml_backend_buffer_init](https://github.com/ggerganov/llama.cpp/blob/9b82476ee9e73065a759f8bcc4cf27ec7ab2ed8c/ggml-backend.c#L60-L76)
* [strdup](https://pubs.opengroup.org/onlinepubs/9699919799/functions/strdup.html "Duplicate a string.") ⇒ [ggml_vk_available_devices_internal](https://github.com/ggerganov/llama.cpp/blob/9b82476ee9e73065a759f8bcc4cf27ec7ab2ed8c/ggml-kompute.cpp#L239-L254)

---

## Issue #N/A: Unable to make imatrix (and likely quant) for nvidia's ChatQA-1.5 8B

**Link**: https://github.com/ggml-org/llama.cpp/issues/7046
**State**: closed
**Created**: 2024-05-02T16:02:28+00:00
**Closed**: 2024-05-02T23:49:10+00:00
**Comments**: 8
**Labels**: bug-unconfirmed

### Description

This model: https://huggingface.co/nvidia/ChatQA-1.5-8B

Conversion worked no issue, but then when it's time to calculate the imatrix I see:

`llama_model_load: error loading model: done_getting_tensors: wrong number of tensors; expected 323, got 291`

Doing a search, last time slaren mentioned doing a gguf-dump.py, here's the output:

```
* Loading: /models/ChatQA-1.5-8B-GGUF/ChatQA-1.5-8B-fp16.gguf
* File is LITTLE endian, script is running on a LITTLE endian host.
llama_cpp-1  |
* Dumping 24 key/value pair(s)
      1: UINT32     |        1 | GGUF.version = 3
      2: UINT64     |        1 | GGUF.tensor_count = 323
      3: UINT64     |        1 | GGUF.kv_count = 21
      4: STRING     |        1 | general.architecture = 'llama'
      5: STRING     |        1 | general.name = 'ChatQA-1.5-8B'
      6: UINT32     |        1 | llama.block_count = 32
      7: UINT32     |        1 | llama.context_length = 8192
      8: UINT32     |        1 | llama.embedding_length = 

[... truncated for brevity ...]

---

## Issue #N/A: Prevent user from setting a context size that is too big

**Link**: https://github.com/ggml-org/llama.cpp/issues/266
**State**: closed
**Created**: 2023-03-18T15:11:33+00:00
**Closed**: 2023-03-19T10:33:41+00:00
**Comments**: 10

### Description

Hey!

I tasked the 30B model to write a little story... it worked really well until some point where it went off rails from one line to the next, suddenly talking about some girl and stuff that has nothing to do with the rest:

```
The way out of me that started looking at them. It'ould be lying there was standing near-the first time what could see an older than the girl had held they looked like it, and just how hard. In order I wasn't really when my hands on his head down to myself in front seat and the car door were with me before you.
“I realy as she staring that laying to a moment of him. "It was so lying next to about two, but it looked at her eyes had already when there looking for holding my hand from what I'with his head was on both shoulders. And not through and suddenly, he realized 212.
I couldn’t with the car seat, in fronted again because of one. The second, so that didn'sit seems like a young girl sitting me when "We weren near. But I started. 'mom. Withered and t

[... truncated for brevity ...]

---

## Issue #N/A: llama_kv_cache_seq_shift delta does not appear to be calculated properly

**Link**: https://github.com/ggml-org/llama.cpp/issues/3825
**State**: closed
**Created**: 2023-10-28T06:47:12+00:00
**Closed**: 2023-10-29T16:32:52+00:00
**Comments**: 10
**Labels**: bug

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [Y] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [Y] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [Y] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [Y] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.


Not 100% certain if this is a bug or not, but I was playing with the kv cache shifting functionality and I was getting some weird results so I figured I'd step through it and see what was going on.

I noticed that after performing a double shift on a chunk of the kv cache, that the cell

[... truncated for brevity ...]

---

## Issue #N/A: Adding MistralForCausalLM architecture to convert-hf-to-gguf.py

**Link**: https://github.com/ggml-org/llama.cpp/issues/4463
**State**: closed
**Created**: 2023-12-14T10:00:26+00:00
**Closed**: 2024-05-10T01:28:38+00:00
**Comments**: 12
**Labels**: enhancement, stale

### Description

Hi.

I'm trying to deploy [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) locally, as the documentation mentions it's supported, but it fails to generate the GGUF model file.

The error is `NotImplementedError: Architecture "MistralForCausalLM" not supported!`

As using GGUF files is a breaking change and the Mistral-7B model should be supported, I think adding support for `MistralForCausalLM` architecture to `convert-hf-to-gguf.py` is essential.

I'm running the latest version as of Dec 14, 2023, which is `b1637`


---

## Issue #N/A: llama : add Deepseek support

**Link**: https://github.com/ggml-org/llama.cpp/issues/5981
**State**: closed
**Created**: 2024-03-10T18:56:56+00:00
**Closed**: 2024-05-08T17:03:57+00:00
**Comments**: 8
**Labels**: help wanted, good first issue, model

### Description

Support is almost complete. There is a dangling issue with the pre-tokenizer: https://github.com/ggerganov/llama.cpp/pull/7036

A useful discussion related to that is here: https://github.com/ggerganov/llama.cpp/discussions/7144

-----

## Outdated below

Creating this issue for more visibility

The main problem is around tokenization support, since the models use some variation of the BPE pre-processing regex. There are also some issues with the conversion scripts.

Anyway, looking for contributions to help with this

Previous unfinished work:

- #4070 
- #5464 

Possible implementation plan: https://github.com/ggerganov/llama.cpp/pull/5464#issuecomment-1974818993

---

## Issue #N/A: Bug: `-ins` command gone from main.exe

**Link**: https://github.com/ggml-org/llama.cpp/issues/7757
**State**: closed
**Created**: 2024-06-05T03:00:56+00:00
**Closed**: 2024-06-05T07:03:06+00:00
**Comments**: 9
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

There seems to be no way to activate instruct mode in main.exe, this causes my scripts to break.

### Name and Version

version: 3089 (c90dbe02)
built with MSVC 19.39.33523.0 for x64

### What operating system are you seeing the problem on?

Windows

### Relevant log output

```shell
error: unknown argument: -ins
```


---

## Issue #N/A: Maybe lower default temp and switch to top_k 40

**Link**: https://github.com/ggml-org/llama.cpp/issues/42
**State**: closed
**Created**: 2023-03-12T10:12:43+00:00
**Closed**: 2023-03-13T17:26:16+00:00
**Comments**: 6
**Labels**: generation quality

### Description

Per [this twitter thread](https://twitter.com/theshawwn/status/1632569215348531201). See commit [here](https://github.com/shawwn/llama/commit/40d99d329a5e38d85904d3a6519c54e6dd6ee9e1).

---

## Issue #N/A: Question: How to access feature vector of the intermediate layer of network?

**Link**: https://github.com/ggml-org/llama.cpp/issues/2047
**State**: closed
**Created**: 2023-06-29T06:31:32+00:00
**Closed**: 2024-04-09T01:08:36+00:00
**Comments**: 6
**Labels**: stale

### Description

# Prerequisites

- [Yes] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [Yes] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [Yes] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [Yes] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

I am interested in the difference between the feature vectors of the intermediate layer of the `llama.cpp` and PyTorch versions of the LLaMa model.
For this purpose, I would like to know how I can get the feature vectors of the middle layer, such as
`torchvision.models.feature_extraction.create_feature_extractor` and `register_forward_hoo

[... truncated for brevity ...]

---

## Issue #N/A: Feature request: Graphical GGUF viewer

**Link**: https://github.com/ggml-org/llama.cpp/issues/6715
**State**: open
**Created**: 2024-04-17T04:30:46+00:00
**Comments**: 18
**Labels**: enhancement, stale

### Description

# Motivation

With the recent introduction of `eval-callback` example, we now having more tools for debugging when working with llama.cpp. However, one of the tool that I feel missing is the ability to dump everything inside a gguf file into a human-readable (and interactive) interface.

Inspired from `huggingface.js` where users can visualize the KV and list of tensors on huggingface.com, I would like to implement the same thing in llama.cpp. I find this helpful in these situations:
- Debugging `convert.py` script when adding a new architecture
- Debugging tokenizers
- Debugging changes related to gguf (model splits for example)
- Debugging tensors (i.e. display N first elements of a tensor, just like `eval-callback`)
- Debugging control vectors
- ... (maybe other usages in the future)

The reason why I can't use `huggingface.js` is because it's based on browser, which make it tricky when reading a huge local file. It also don't have access to quantized types (same for `gg

[... truncated for brevity ...]

---

## Issue #N/A: Finetune GPU Utilization fell to 0%

**Link**: https://github.com/ggml-org/llama.cpp/issues/4016
**State**: closed
**Created**: 2023-11-10T08:17:21+00:00
**Closed**: 2024-04-02T01:11:46+00:00
**Comments**: 11
**Labels**: bug-unconfirmed, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior
finetune.cpp should fully utilize the GPU. We were trying to finetune Llama2-7b 16 bit model with a sample dataset of 1000 samples.

# Current Behavior
When running finetune and offloading all layers to the GPU, the GPU was utilized at around 30% at the beginning of 

[... truncated for brevity ...]

---

## Issue #N/A: llama-cli chat templates ignored?

**Link**: https://github.com/ggml-org/llama.cpp/issues/8469
**State**: closed
**Created**: 2024-07-13T14:15:48+00:00
**Closed**: 2024-09-09T01:07:19+00:00
**Comments**: 7
**Labels**: stale

### Description

`llama-cli -c 1024 -t 6 -m codegeex4-all-9b.q4_k.gguf -p "You are my assistant." -e -cnv --chat-template chatml`

```
== Running in interactive mode. ==
 - Press Ctrl+C to interject at any time.
 - Press Return to return control to the AI.
 - To return control without starting a new line, end your input with '/'.
 - If you want to submit another line, end your input with '\'.

<|im_start|>system
You are my assistant.<|im_end|>

> Hello.
Hello! How can I assist you today?
<|im_end|>
<|im_start|>user
I'm a developer and I want [....]
```
and it continues by itself.
what am I missing?


---

## Issue #N/A: Qwen-72B-Chat conversion script does not treat <|im_start|> and <|im_end|> correctly.

**Link**: https://github.com/ggml-org/llama.cpp/issues/4331
**State**: closed
**Created**: 2023-12-04T23:16:13+00:00
**Closed**: 2024-04-08T01:06:38+00:00
**Comments**: 13
**Labels**: bug-unconfirmed, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Description

(This is specifically for the latest 72B models. I have never tried the smaller ones).

I'm using this model: https://huggingface.co/Qwen/Qwen-72B-Chat

Commit: `33e171d1e9fc4903f9314b490d77fb8d58331b63`

I think the current `convert-hf-to-gguf.py` does not produce a 

[... truncated for brevity ...]

---

## Issue #N/A: converting phi-3-small error.

**Link**: https://github.com/ggml-org/llama.cpp/issues/7922
**State**: closed
**Created**: 2024-06-13T17:17:16+00:00
**Closed**: 2024-08-14T01:06:57+00:00
**Comments**: 13
**Labels**: stale

### Description

`python llama.cpp/convert-hf-to-gguf.py --outtype f16 --outfile /content/Phi-3-small-128k-instruct.f16.gguf /content/Phi-3-small-128k-instruct`

```
INFO:hf-to-gguf:Loading model: Phi-3-small-128k-instruct
ERROR:hf-to-gguf:Model Phi3SmallForCausalLM is not supported
```

---

## Issue #N/A: Feature Request: support embedding stella_en_400M and stella_en_400M.gguf conversion

**Link**: https://github.com/ggml-org/llama.cpp/issues/9202
**State**: closed
**Created**: 2024-08-27T15:14:55+00:00
**Closed**: 2024-12-03T01:07:39+00:00
**Comments**: 6
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Need help supporting stella_en_400M, observed that we have embedding model
https://ollama.com/Losspost/stella_en_1.5b_v5

but there I couldn't convert stella_en_400M myself

Model Download:
https://hf.rst.im/dunzhang/stella_en_400M_v5

D:\llama.cpp>python convert_hf_to_gguf.py d:/llama.cpp/stella_en_400M_v5 --outfile stella_en_400M.gguf --outtype q8_0
INFO:hf-to-gguf:Loading model: stella_en_400M_v5
ERROR:hf-to-gguf:Model NewModel is not supported

### Motivation

To have better embeddi

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Error while converting BERT to GGUF: Can not map tensor 'bert.embeddings.LayerNorm.beta'

**Link**: https://github.com/ggml-org/llama.cpp/issues/7924
**State**: closed
**Created**: 2024-06-13T19:36:58+00:00
**Closed**: 2024-07-28T01:07:05+00:00
**Comments**: 14
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

I am trying to convert the [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) HuggingFace model to GGUF with [convert-hf-to-gguf.py](https://github.com/ggerganov/llama.cpp/blob/master/convert-hf-to-gguf.py) Unfortunately, it fails to convert because the script looks for `embeddings.position_embeddings`, etc. in [tensor-mapping.py](https://github.com/ggerganov/llama.cpp/blob/172c8256840ffd882ab9992ecedbb587d9b21f15/gguf-py/gguf/tensor_mapping.py#L44) but not `bert.embeddings.position_embeddings`, etc. This is important because most of the tensor names in the model start with `bert`.

There is a similar issue in [modify-tensors](https://github.com/ggerganov/llama.cpp/blob/master/convert-hf-to-gguf.py#L2192). It does not skip the `cls` tensors that are present in [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased), so it fails in the same way.

Finally, the bert-base-uncased `config.json` has its architecture set to `B

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

## Issue #N/A: Support for OpenELM of Apple

**Link**: https://github.com/ggml-org/llama.cpp/issues/6868
**State**: closed
**Created**: 2024-04-24T08:10:16+00:00
**Closed**: 2024-07-04T17:14:22+00:00
**Comments**: 10
**Labels**: enhancement, good first issue

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ ] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ ] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

Support for OpenELM of Apple

https://huggingface.co/apple/OpenELM-3B-Instruct/tree/main


---

## Issue #N/A: Misc. bug: llama-sampling.cpp:204: GGML_ASSERT(cur_p->size > 0) failed

**Link**: https://github.com/ggml-org/llama.cpp/issues/13405
**State**: closed
**Created**: 2025-05-09T14:05:48+00:00
**Closed**: 2025-05-27T09:07:54+00:00
**Comments**: 8
**Labels**: bug-unconfirmed

### Description

### Name and Version

```
$./llama-cli --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    yes
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
version: 5329 (611aa914)
built with cc (Debian 12.2.0-14) 12.2.0 for x86_64-linux-gnu
```


### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-cli

### Command line

```shell
llama-cli \
    --log-file /tmp/llamacpp-Qwen3-30B-A3B-Q8_K_XL.log \
    --hf-repo unsloth/Qwen3-30B-A3B-GGUF:Q8_K_XL \
    --override-tensor '([0-9]+).ffn_.*_exps.=CPU' \
    --n-gpu-layers 48 \
    --jinja \
    --cache-type-k q8_0 \
    --ctx-size 32768 \
    --samplers "top_k;dry;min_p;temperature;top_p" \
    --min-p 0.005 \
    --top-p 0.97 \
    --top-k 40 \
    --temp 0.7 \
    --dry-multiplier 0.7 \
    --dry-allowed-length 4 \
    --dry-penalty-last-n 2048 \
    --presence-penalty 0.05 \
    --frequency-penalty 0.005 \
    

[... truncated for brevity ...]

---

## Issue #N/A: llava 1.5 invalid output after first inference (llamacpp server)

**Link**: https://github.com/ggml-org/llama.cpp/issues/7060
**State**: closed
**Created**: 2024-05-03T14:47:40+00:00
**Closed**: 2024-05-10T06:41:11+00:00
**Comments**: 13
**Labels**: bug-unconfirmed

### Description

I use this server config:
```{
    "host": "0.0.0.0",
    "port": 8085,
    "api_key": "api_key",
    "models": [
        {
            "model": "models/phi3_mini_model/phi3_mini_model.gguf",
            "model_alias": "gpt-3.5-turbo",
            "chat_format": "chatml",
            "n_gpu_layers": 35,
            "offload_kqv": true,
            "n_threads": 12,
            "n_batch": 512,
            "n_ctx": 2048
        },
        {
            "model": "models/phi3_mini_model/phi3_mini_model.gguf",
            "model_alias": "gpt-4",
            "chat_format": "chatml",
            "n_gpu_layers": 35,
            "offload_kqv": true,
            "n_threads": 12,
            "n_batch": 512,
            "n_ctx": 4096
        },
        {
            "model": "models/llava15_vision_model/ggml-model-q4_k.gguf",
            "model_alias": "gpt-4-vision-preview",
            "chat_format": "llava-1-5",
            "clip_model_path": "models/llava15_vision_

[... truncated for brevity ...]

---

## Issue #N/A: can' quantize deekseek model

**Link**: https://github.com/ggml-org/llama.cpp/issues/4925
**State**: closed
**Created**: 2024-01-14T07:19:27+00:00
**Closed**: 2024-04-18T01:06:43+00:00
**Comments**: 7
**Labels**: bug-unconfirmed, stale

### Description

When I download the model from the deepseek huggingface official repository, I cannot convert it into a gguf file。

python D:\Ai\convert.py D:\Ai\deepseek-coder-6.7b-instruct

D:\Ai\gguf-py
Loading model file D:\Ai\deepseek-coder-6.7b-instruct\pytorch_model-00001-of-00002.bin
Loading model file D:\Ai\deepseek-coder-6.7b-instruct\pytorch_model-00001-of-00002.bin
Loading model file D:\Ai\deepseek-coder-6.7b-instruct\pytorch_model-00002-of-00002.bin
params = Params(n_vocab=32256, n_embd=4096, n_layer=32, n_ctx=16384, n_ff=11008, n_head=32, n_head_kv=32, f_norm_eps=1e-06, n_experts=None, n_experts_used=None, rope_scaling_type=<RopeScalingType.LINEAR: 'linear'>, f_rope_freq_base=100000, f_rope_scale=4.0, n_orig_ctx=None, rope_finetuned=None, ftype=None, path_model=WindowsPath('D:/Ai/deepseek-coder-6.7b-instruct'))
Traceback (most recent call last):
  File "D:\Ai\convert.py", line 1658, in <module>
    main(sys.argv[1:])  # Exclude the first element (script name) from sys.argv
  

[... truncated for brevity ...]

---

## Issue #N/A: New models Sequelbox/StellarBright and ValiantLabs/ShiningValiant cannot be converted due to "Unexpected tensor name: model.layers.0.self_attn.q_proj.lora_A.default.weight"

**Link**: https://github.com/ggml-org/llama.cpp/issues/3640
**State**: closed
**Created**: 2023-10-15T21:29:11+00:00
**Closed**: 2024-04-04T01:08:13+00:00
**Comments**: 10
**Labels**: stale

### Description

The Open LLM Leaderboard is currently being lead by https://huggingface.co/ValiantLabs/ShiningValiant with https://huggingface.co/sequelbox/StellarBright in third place.

Unfortunately these models cannot be converted to GGUF, due to this error from `convert.py`:
```
Traceback (most recent call last):
  File "/workspace/git/gguf-llama/./convert.py", line 1193, in <module>
    main()
  File "/workspace/git/gguf-llama/./convert.py", line 1180, in main
    model   = convert_model_names(model, params)
  File "/workspace/git/gguf-llama/./convert.py", line 984, in convert_model_names
    raise Exception(f"Unexpected tensor name: {name}")
Exception: Unexpected tensor name: model.layers.0.self_attn.q_proj.lora_A.default.weight
```

Seems they have some kind of non-standard model layout here, which doesn't affect Transformers-based loading, but does break `convert.py`.

If anyone could find a fix or workaround, that'd be much appreciated - I'm getting quite a few requests for GG

[... truncated for brevity ...]

---

## Issue #N/A: [User] Dependency Installation steps for ubuntu linux

**Link**: https://github.com/ggml-org/llama.cpp/issues/1146
**State**: closed
**Created**: 2023-04-23T19:50:20+00:00
**Closed**: 2023-05-18T11:13:26+00:00
**Comments**: 10

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

When following the provided documentation, I expect to be able to successfully set up and use the software, with clear and complete instructions guiding me through each step of the process. This includes:

1. Instructions for installing any dependencies and prerequi

[... truncated for brevity ...]

---

