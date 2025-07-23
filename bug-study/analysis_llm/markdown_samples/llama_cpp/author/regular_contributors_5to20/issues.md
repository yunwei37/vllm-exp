# regular_contributors_5to20 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- stale: 13 issues
- bug-unconfirmed: 12 issues
- enhancement: 3 issues
- medium severity: 3 issues
- high severity: 2 issues
- macos: 1 issues
- model: 1 issues
- low severity: 1 issues
- duplicate: 1 issues
- bug: 1 issues

---

## Issue #N/A: Automatic optimization of runtime parameters such as -ngl given memory constraints

**Link**: https://github.com/ggml-org/llama.cpp/issues/13860
**State**: closed
**Created**: 2025-05-28T13:57:26+00:00
**Closed**: 2025-07-13T01:08:22+00:00
**Comments**: 4
**Labels**: stale

### Description

I'm interested in implementing code for automatically determining the optimal runtime parameters given some model and memory constraints. I imagine the implementation to use something like a "dummy" parameter which, when set, does not result in any actual memory allocations but enables the creation of `llama_model` and `llama_context` dummies that can be used to determine how much memory would be used for some choice of `llama_model_params` and `llama_context_params`. By comparing the amount of memory that was used for the dummies with the amount of memory that is actually available the implementation could then iteratively optimize parameters such as context size or the number of GPU layers.

One roadblock that I have run into is how to make this implementation minimally invasive for the rest of the code. Right now I think the way to do it would be:

* Extend `ggml_backend_device` to track the amount of memory that has been allocated to this device by the current process.
* Add a func

[... truncated for brevity ...]

---

## Issue #N/A: Bug: b3028 breaks mixtral 8x22b

**Link**: https://github.com/ggml-org/llama.cpp/issues/7969
**State**: closed
**Created**: 2024-06-17T05:04:18+00:00
**Closed**: 2024-08-20T01:06:49+00:00
**Comments**: 20
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

Mixtral 8x22b model running with server.

b3027: good
lm hi
 Hello! How can I help you today? Is there something specific you would like to talk about or ask about? I'm here to provide information and answer questions to the best of my ability.

b3028: garbage
lm hi
👋

[INST] I'm here to help you with your questions about the [/INST] 🤓

[INST] I can provide information on a variety of topics, such as [/INST] 📚

[INST] - [/INST] 🏫
- [/INST] 💻
- [/INST] 📈
- [/INST] 📊
- [/INST] 📈
- [/INST] 📊
- [/INST] 📈
- [/INST] 📊
- [/INST] 📈








### Name and Version

b3027 for good run
b3028 for broken run

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
na
```


---

## Issue #N/A: llama_tensor_get_type falling back when not necessary?

**Link**: https://github.com/ggml-org/llama.cpp/issues/6614
**State**: closed
**Created**: 2024-04-11T17:32:32+00:00
**Closed**: 2024-04-11T17:38:48+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

Noticed while making some quants for Q2_K that I was getting messages:

llama_tensor_get_type : tensor cols 14464 x 4096 are not divisible by 256, required for q3_K - using fallback quantization iq4_nl

But by my math, it definitely is? Anything multiplied by 4096 should be. Is the error message misleading or is there some accidental miscalculation going on?

---

## Issue #N/A: Can't Quantize gguf files: zsh: illegal hardware instruction on M1 MacBook Pro

**Link**: https://github.com/ggml-org/llama.cpp/issues/3983
**State**: closed
**Created**: 2023-11-08T00:19:22+00:00
**Closed**: 2023-11-14T17:35:34+00:00
**Comments**: 8
**Labels**: macos, bug-unconfirmed

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [Y] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [Y] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [Y] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [Y] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior
Successfully quantize and run large language models that I convert to gguf on M1 MacBook Pro

# Current Behavior
Quantization halts due to "zsh: illegal hardware instruction".

# Environment and Context
OS: Mac OS Sonoma
System: 2020 M1 MacBook Pro 16GB RAM
Xcod

[... truncated for brevity ...]

---

## Issue #N/A: Guide: Fixing wsl issues (instruction failures) and general guide for 11.6+

**Link**: https://github.com/ggml-org/llama.cpp/issues/3322
**State**: closed
**Created**: 2023-09-23T22:14:59+00:00
**Closed**: 2023-09-23T22:40:37+00:00
**Comments**: 1

### Description

```
    .run files
    #to match max compute capability
        
    nano Makefile (wsl)
        NVCCFLAGS += -arch=native
        Change it to specify the correct architecture for your GPU. For a GPU with Compute Capability 5.2, you should replace it with:

        makefile
        Copy code
        NVCCFLAGS += -arch=sm_52
        
    modified makefile
    #https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/
    
    apt-get remove --purge '^nvidia-.*' '^cuda-.*'
     sudo apt-get autoremove
    sudo apt-get autoclean
    sudo find / \( -path /home -o -path /mnt -o -path /usr/local/lib/python3.10 \) -prune -o ! -user root \( -name '*nvidia*' -o -name '*cuda*' \) -print

    cat /var/log/cuda-uninstaller.log

    modprobe -r nvidia
    
    dpkg -l | grep -i nvidia | awk '{print $2}' | xargs sudo apt-get --purge remove -y

    sudo rm -rf /etc/systemd/system/nvidia*
    sudo rm -rf /etc/systemd/system/display-manager.service.d
    s

[... truncated for brevity ...]

---

## Issue #N/A: bad command line parsing behaviour with some filenames

**Link**: https://github.com/ggml-org/llama.cpp/issues/6163
**State**: closed
**Created**: 2024-03-19T15:55:05+00:00
**Closed**: 2024-07-15T01:06:59+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale

### Description

quantize (only command I have tested this with) gets confused by some filenames of some models, e.g.

   quantize 08x7bhf.gguf 08x7bhfGGUF~ Q4_K_S

... should quantize one gguf file into another, but instead, it fails with a weird error message:

   main: invalid nthread 'Q4_K_S' (stoi)

My guess is that somehow it interprets the 08x name as a number, but that's clearly not the whole story. Might even be a security issue if commands can be tricked into misinterinterpreting filenames as something else (this behaviuour cannot be suppressed with "--" either).


---

## Issue #N/A: Launching Server With Parameters

**Link**: https://github.com/ggml-org/llama.cpp/issues/4243
**State**: closed
**Created**: 2023-11-28T04:03:11+00:00
**Closed**: 2024-04-03T01:15:06+00:00
**Comments**: 1
**Labels**: enhancement, stale

### Description

It'd be great if we can prefilled all the parameters such as temperature, prompt, max gen length, etc. when launching server through either a configuration file or cli flags.

Then we don't have to modify the parameters every time after launching server.

---

## Issue #N/A: CUDA 12.4 released incompletely.

**Link**: https://github.com/ggml-org/llama.cpp/issues/5998
**State**: closed
**Created**: 2024-03-11T13:36:03+00:00
**Closed**: 2024-04-25T01:12:27+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale

### Description

Last week my Fedora env upgraded and I found CUDA 12.4 arrived (before Nvidia's own release notes even). They haven't yet pushed the nvidia/cuda:12.4.0 container so Docker builds are failing and I've had to revert the update. Just a heads-up if someone else hits automation issues with CUDA 12.4. Hopefully NVidia will push their 12.4.0 container asap.

https://forums.fedoraforum.org/showthread.php?332179-RPMFusion-Nvidia-driver-repo-not-retaining-versions

Release notes for 12.4 finally landed a day or so after the update came in:
https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

I will close this once Nvidia pushes `docker.io/nvidia/cuda:12.4.0`

---

## Issue #N/A: Feature Request: Make chat sessions possible with multi model cli tools

**Link**: https://github.com/ggml-org/llama.cpp/issues/12982
**State**: closed
**Created**: 2025-04-16T20:01:07+00:00
**Closed**: 2025-05-31T01:07:44+00:00
**Comments**: 1
**Labels**: stale

### Description

Hi,

As a blind person it would be interesting to be able to chat with the model when a image is submitted.
So more questions about the image can be asked.
I wasn't able to find a conversation option for the multi model cli tools.

Greetings,
Simon

---

## Issue #N/A: [BUG] Using `--no-mmap --mlock` crashes the `server`

**Link**: https://github.com/ggml-org/llama.cpp/issues/5023
**State**: closed
**Created**: 2024-01-18T19:12:15+00:00
**Closed**: 2024-01-18T20:12:17+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

Something about the latest commits has messed with the `server`. The same command to start the server now exits like this:

```
GGML_ASSERT: llama.cpp:1064: addr == NULL && size == 0
fish: Job 1, './server -m "/Users/behnam/Down…' terminated by signal SIGABRT (Abort)
```

* How to reproduce

Upgrade to the latest commit and do:

```
./server -m "<gguf model>" --ctx-size 4096 --threads 8 -ngl 128 --port 8080 --mlock --no-mmap
```

* Using `--mlock` or `--no-mmap` alone doesn't crash the server.

---

## Issue #N/A: Misc. bug: HIP when using llama.bench and kv cache quant cpu is doing the work instead of gpu

**Link**: https://github.com/ggml-org/llama.cpp/issues/12624
**State**: closed
**Created**: 2025-03-28T10:34:46+00:00
**Closed**: 2025-04-17T02:59:53+00:00
**Comments**: 5
**Labels**: bug-unconfirmed

### Description

### Name and Version

b4958 LLama 3.2 1b q8_0 gguf

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

Other (Please specify in the next section)

### Command line

```shell
pl752@pl752-desktop:~$ ROCR_VISIBLE_DEVICES=0 llama.cpp/build/bin/llama-bench -m /models/llm_models/Llama-3.2-1B-Instruct-Q8_0.gguf -p 4096,16384 -n 128,1024 -fa 1 -ctk q8_0 -ctv q8_0
```

### Problem description & steps to reproduce

cpu usage is 100%, gpu vram is filled, but almost no activity
(pp4096 speed is 200 t/s, against 2000 without cache quant)

### First Bad Commit

_No response_

### Relevant log output

```shell
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon VII, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
| model                          |       size |     params | backend    | ngl | type_k | type_v | fa |          test |                  t/s |
| ----------

[... truncated for brevity ...]

---

## Issue #N/A: Different outputs for differents numbers of threads (same seed)

**Link**: https://github.com/ggml-org/llama.cpp/issues/95
**State**: closed
**Created**: 2023-03-13T16:20:56+00:00
**Closed**: 2023-03-23T21:30:06+00:00
**Comments**: 4
**Labels**: enhancement

### Description

Hello,

I simply wanted to bring up the point that the output can vary based on the number of threads selected, even if the seed stays constant.

I have an intel core i7 10700K that has 16 threads.

For this example I'm using the 13B model (./models/13B/ggml-model-q4_0.bin)

When I put -t 14 (make -j && ./main -m ./models/13B/ggml-model-q4_0.bin -p "Building a website can be done in 10 simple steps:" -t 14 -n 50 --seed 1678486056), I got this result:
![duU196l](https://user-images.githubusercontent.com/110173477/224762353-1c5565d8-478c-41c6-ac13-f7883dc3ec50.png)

When I put -t 15 (make -j && ./main -m ./models/13B/ggml-model-q4_0.bin -p "Building a website can be done in 10 simple steps:" -t 15 -n 50 --seed 1678486056), I got this result:
![5WIrvd1](https://user-images.githubusercontent.com/110173477/224762999-258a6235-b14c-4db8-8b04-163a0b92d356.png)

I have zero knowledge in machine learning, perhaps this is a normal behavior.

Looking forward for your reactions!



[... truncated for brevity ...]

---

## Issue #N/A: Cannot load 2 bit quantized ggml model on Windows

**Link**: https://github.com/ggml-org/llama.cpp/issues/1018
**State**: closed
**Created**: 2023-04-17T03:02:57+00:00
**Closed**: 2023-04-17T18:06:50+00:00
**Comments**: 1

### Description

```
C:\Users\micro\Downloads>main -m ggml-model-q2_0.bin
main: seed = 1681700481
llama.cpp: loading model from ggml-model-q2_0.bin
error loading model: unrecognized tensor type 5

llama_init_from_file: failed to load model
main: error: failed to load model 'ggml-model-q2_0.bin'
```

---

## Issue #N/A: [Feature Request] Dynamic temperature sampling for better coherence / creativity

**Link**: https://github.com/ggml-org/llama.cpp/issues/3483
**State**: closed
**Created**: 2023-10-05T02:23:01+00:00
**Closed**: 2024-06-12T01:06:49+00:00
**Comments**: 47
**Labels**: stale

### Description

# Prerequisites

- [✅] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Idea

Typical sampling methods for large language models, such as Top P and Top K, (as well as alternative sampler modes that decide the Top K dynamically like Mirostat) are based off the assumption that a static temperature value (a consistently randomized probability distribution) is the ideal sampler conditioning. Mirostat, most notably, was designed to 'learn' a certain targeted level of 'entropy' over time; this helped the model find the most grammatically coherent selection of tokens to be considered by the sampler for good results. Most of these sampling implementations weren't designed to be used together. Some, like TFS, were created when the largest available models were smaller ones like GPT2. Those models struggled a _lot_ more when attempting to generalize in different directions, and it makes sense to 

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

## Issue #N/A: Support starcoder family architectures (1B/3B/7B/13B)

**Link**: https://github.com/ggml-org/llama.cpp/issues/3076
**State**: closed
**Created**: 2023-09-08T02:40:11+00:00
**Closed**: 2023-09-15T19:15:21+00:00
**Comments**: 6
**Labels**: model

### Description

Related Issues:

https://github.com/ggerganov/llama.cpp/issues/1901
https://github.com/ggerganov/llama.cpp/issues/1441
https://github.com/ggerganov/llama.cpp/issues/1326

Previously, it wasn't recommended to incorporate non-llama architectures into llama.cpp. However, in light of the recent addition of the Falcon architecture (see [Pull Request #2717](https://github.com/ggerganov/llama.cpp/pull/2717)), it might be worth reconsidering this stance.

One distinguishing feature of Starcoder is its ability to provide a complete series of models ranging from 1B to 13B. This capability can prove highly beneficial for speculative decoding and making coding models available for edge devices (e.g., M1/M2 Macs).

I can contribute the PR if it matches llama.cpp's roadmap.

---

## Issue #N/A: [User] Mac with Intel CPU + AMD GPU

**Link**: https://github.com/ggml-org/llama.cpp/issues/2785
**State**: closed
**Created**: 2023-08-25T14:47:54+00:00
**Closed**: 2024-04-09T01:06:49+00:00
**Comments**: 4
**Labels**: stale

### Description

Since Llama.cpp now supports ROCM, is that possible to support Mac with Intel CPU + AMD GPU as well?

---

## Issue #N/A: Bug: Running a large model through the server using vulkan backend always generates gibberish after first call.

**Link**: https://github.com/ggml-org/llama.cpp/issues/7819
**State**: closed
**Created**: 2024-06-07T18:26:59+00:00
**Closed**: 2024-07-22T01:06:52+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

Bug: Running a model through the server using vulkan backend always works for the first time, but always generates gibberish in subsequent calls

### Name and Version

 .\server.exe -m ..\gguf_models\Cat-Llama-3-70B-instruct-Q4_K_M.gguf -ngl 70 -c 8192

### What operating system are you seeing the problem on?

Windows

### Relevant log output

_No response_

---

## Issue #N/A: Add `--grammar-file` argument to `server` (similar to how `main` does it)

**Link**: https://github.com/ggml-org/llama.cpp/issues/5130
**State**: closed
**Created**: 2024-01-26T01:00:52+00:00
**Closed**: 2024-01-27T22:34:11+00:00
**Comments**: 17
**Labels**: enhancement

### Description

# Feature Description

Similar to the docs for `main` (https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md), it'd be great to have the `--grammar-file=...` flag available in `server` as well.

# Motivation

Currently, `server` can't process long grammars. I don't know if it's a bug but I've noticed that even with the `json.gbnf` files in the repo. Basically, the content of the file gets too complicated for the grammar parser to read. I think it has to do with how multi-line strings are treated in terminal.

In any case, using `grammar-file` simplifies API calls a lot.

---

## Issue #N/A: Bug: Could NOT find BLAS (missing: BLAS_LIBRARIES)

**Link**: https://github.com/ggml-org/llama.cpp/issues/7708
**State**: closed
**Created**: 2024-06-03T04:08:47+00:00
**Closed**: 2024-07-18T01:06:43+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

When building package for ALT Linux I found that with `-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS` OpenBLAS support is still not built.


### Name and Version

Version b3012


### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
-- Could NOT find BLAS (missing: BLAS_LIBRARIES)
CMake Warning at CMakeLists.txt:374 (message):
  BLAS not found, please refer to
  https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors
  to set correct LLAMA_BLAS_VENDOR
```
~I think this is because of this code:~
```
    set(BLA_VENDOR ${LLAMA_BLAS_VENDOR})
    find_package(BLAS)
```
~Instead of setting `BLAS_VENDOR`.~

~There also other instances for setting `BLA_` variables,~ but I am not sure this is not intended since I'm not into BLAS.


---

## Issue #N/A: Llama 2 and server

**Link**: https://github.com/ggml-org/llama.cpp/issues/2283
**State**: closed
**Created**: 2023-07-20T03:26:20+00:00
**Closed**: 2024-04-09T01:07:47+00:00
**Comments**: 1
**Labels**: stale

### Description

Hi, I' using llama-2-13b-chat.ggmlv3.q4_1.bin and M2 16GB of memory.
Using regular llama cpp, works fine with context 2048. But in server mode, it will crash even if context if 1536. I set it to 1024 and it works

Is there difference in llama 2 ? I use older model like vicuna 13b with 2048 context and works just fine

---

## Issue #N/A: Bug: docker GGML_CUDA=1 make [on llama-gen-docs] fails since arg refactor

**Link**: https://github.com/ggml-org/llama.cpp/issues/9392
**State**: closed
**Created**: 2024-09-09T17:36:42+00:00
**Closed**: 2024-09-10T06:12:49+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, low severity

### Description

### What happened?

Error given is

```
./llama-gen-docs: error while loading shared libraries: libcuda.so.1: cannot open shared object file: No such file or directory
```

Since this was only recently added in https://github.com/ggerganov/llama.cpp/pull/9308 I'm guessing that's to blame

I've been able to get around it by running:

```
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH GGML_CUDA=1 make -j64
RUN rm /usr/local/cuda/lib64/stubs/libcuda.so.1
```

but I guess my question is just *why* does it need this library at all and why is only this one failing?

### Name and Version

b3707 ubuntu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
./llama-gen-docs: error while loading shared libraries: libcuda.so.1: cannot open shared object file: No such file or directory
```


---

## Issue #N/A: I have manged to get termux on wear os but....

**Link**: https://github.com/ggml-org/llama.cpp/issues/1371
**State**: closed
**Created**: 2023-05-08T19:27:27+00:00
**Closed**: 2023-05-09T16:27:42+00:00
**Comments**: 0

### Description

I have manged to get termux on wear os but due to storage constrains i am looking for the smallest model supported plz help

---

## Issue #N/A: [Feature request] Support for "Falcon" model

**Link**: https://github.com/ggml-org/llama.cpp/issues/1650
**State**: closed
**Created**: 2023-05-30T09:31:10+00:00
**Closed**: 2023-05-30T11:34:52+00:00
**Comments**: 1
**Labels**: duplicate

### Description

"Falcon" is a new Large Language Model which seems to be better than Llama.
See https://falconllm.tii.ae/ and
https://iamgeekydude.com/2023/05/28/falcon-llm-the-40-billion-parameters-llm/ and
https://www.marktechpost.com/2023/05/28/technology-innovation-institute-open-sourced-falcon-llms-a-new-ai-model-that-uses-only-75-percent-of-gpt-3s-training-compute-40-percent-of-chinchillas-and-80-percent-of-palm-62b/

Actually, it is the best open-source model currently available according to the authors.

Model (for Huggingface Transformers library) with 40B and 7B parameters is available at :
https://huggingface.co/tiiuae/falcon-40b

Would be great if it would be supported also in llama.cpp.
Note it uses some novel layers (FlashAttention, Multiquery).

---

## Issue #N/A: not able to load quantized llama-7B model on m1

**Link**: https://github.com/ggml-org/llama.cpp/issues/2157
**State**: closed
**Created**: 2023-07-09T16:06:00+00:00
**Closed**: 2023-07-10T03:06:09+00:00
**Comments**: 3

### Description

# Hi there 👋  

Android dev ~6 yrs of exp [Kotlin and Java] just trying my best to transition to ML as fast as I can-- bc I got real bad FOMO 😢 

Anyways, as the title suggests I did what I presume to be the correct steps to setup the [simplest model](https://huggingface.co/decapoda-research/llama-7b-hf) I could find, and after struggling a bit to get the setup correct I eventually got the 7B model downloaded ([all 33 pieces of it](https://huggingface.co/decapoda-research/llama-7b-hf/tree/main)).

followed the[ Metal Build](https://github.com/ggerganov/llama.cpp#metal-build) instructions and everything seemed to be going wonderfully well so far-- until it didnt...

# Steps and Logs

## All the files needed

<img width="778" alt="Screenshot 2023-07-09 at 11 32 03 AM" src="https://github.com/ggerganov/llama.cpp/assets/45348368/7c60d306-c344-4880-82f5-b8f873b267e7">

## Convert models to `ggml` format and quantize (**_see pic above_**):

- `python convert-pth-to-ggml.py mo

[... truncated for brevity ...]

---

## Issue #N/A: main: crashing upon loading model since commit 83b72cb0 - Windows MSVC + CUDA

**Link**: https://github.com/ggml-org/llama.cpp/issues/6931
**State**: closed
**Created**: 2024-04-26T14:29:11+00:00
**Closed**: 2024-04-26T15:07:43+00:00
**Comments**: 1

### Description

Commit https://github.com/ggerganov/llama.cpp/commit/83b72cb086ce46a33dececc86bfe4648b6120aa8 introduces a bug where main.exe crashes immediately after loading the model running on Windows.

The crash occurs in `void gguf_free(struct gguf_context * ctx)`.

```
cmake -S . -B build -DLLAMA_CUDA=ON && cmake --build build --config Release
```

```
main -ngl 33 -c 0 -f prompt.txt -m ggml-meta-llama-3-8b-instruct-f16.gguf
```

```
>	llama.dll!gguf_free(gguf_context * ctx) Line 20991	C
 	llama.dll!llama_model_loader::~llama_model_loader() Line 3232	C++
 	llama.dll!llama_model_load(const std::string & fname, llama_model & model, llama_model_params & params) Line 6042	C++
 	llama.dll!llama_load_model_from_file(const char * path_model, llama_model_params params) Line 15225	C++
 	main.exe!llama_init_from_gpt_params(gpt_params & params) Line 2224	C++
 	main.exe!main(int argc, char * * argv) Line 199	C++
```

---

## Issue #N/A: Dynatemp and min_p upgrade?

**Link**: https://github.com/ggml-org/llama.cpp/issues/9178
**State**: closed
**Created**: 2024-08-25T23:54:00+00:00
**Closed**: 2024-10-14T01:40:38+00:00
**Comments**: 3
**Labels**: stale

### Description

i've stumbled upon dynatemp and have a question/proposal.

I believe, that the thing that was missed during dynatemp implementation is the underlying concept of what it's needed for. 

Prompts may require 2 types of replies: deterministic replies and creative replies. These are opposite in terms of sampling approach.

Deterministic approach would be required, for example, by programming and by answering knowledge related question. Then you *wish* llm to provide with the most probable tokens.

Creative approach would be required in writing stories and general conversations with llms.

For example, we all know parasite words of llms, like "Maniacally laughing" of llama 3 and "Ahahahaha" that it inserts into nearly every reply. Tokens forming these are super probable. So, in case of using the dynatemp here, we will only increase changes to get "ahahahahahahaha" instead of "ahaha" and that's what i saw in my tests :).

Meanwhile, the whole idea for creative tasks the situation 

[... truncated for brevity ...]

---

## Issue #N/A: webUI local storage can become corrupted

**Link**: https://github.com/ggml-org/llama.cpp/issues/10348
**State**: closed
**Created**: 2024-11-17T01:29:31+00:00
**Closed**: 2024-12-13T16:37:13+00:00
**Comments**: 2
**Labels**: bug, good first issue, server/webui

### Description

### Discussed in https://github.com/ggerganov/llama.cpp/discussions/10347

<div type='discussions-op-text'>

<sup>Originally posted by **pikor69** November 17, 2024</sup>
The page at http://127.0.0.1:8080 says:
TypeError: Cannot read properties of undefined (reading 'content')

What changed since yesterday when it was working? Nothing.
The last time I was able to start I tried to run a much higher content length than the model allowed and things crashed.

</div>

---

## Issue #N/A: `convert.py`: --pad-vocab not working with SPM, `'SentencePieceVocab' object has no attribute 'added_tokens_dict'. Did you mean: 'added_tokens_list'?`

**Link**: https://github.com/ggml-org/llama.cpp/issues/4958
**State**: closed
**Created**: 2024-01-15T17:03:28+00:00
**Closed**: 2024-01-17T13:45:04+00:00
**Comments**: 9

### Description

Hi guys

I've just noticed that since the recent `convert.py` refactor, the new `--pad-vocab` feature does not work with SPM vocabs.  It does work as expected with HFFT.  *EDIT: actually there might be a different bug with HFFT, see next post on that.*

Example command, converting model: https://huggingface.co/TigerResearch/tigerbot-13b-chat-v5
```
python3 ./convert.py /workspace/process/tigerresearch_tigerbot-13b-chat-v5/source --outtype f16 --outfile /workspace/process/tigerresearch_tigerbot-13b-chat-v5/gguf/tigerbot-13b-chat-v5.fp16.gguf --pad-vocab
```

Error message:
```
Writing /workspace/process/tigerresearch_tigerbot-13b-chat-v5/gguf/tigerbot-13b-chat-v5.fp16.gguf, format 1
Padding vocab with 2 token(s) - <dummy00001> through <dummy00002>
Traceback (most recent call last):
  File "/workspace/git/llama.cpp/./convert.py", line 1658, in <module>
    main(sys.argv[1:])  # Exclude the first element (script name) from sys.argv
    ^^^^^^^^^^^^^^^^^^
  File "/workspac

[... truncated for brevity ...]

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

