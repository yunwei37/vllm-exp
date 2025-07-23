# low_impact_1to2 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 29

### Label Distribution

- bug-unconfirmed: 16 issues
- stale: 11 issues
- enhancement: 4 issues
- medium severity: 4 issues
- need more info: 2 issues
- server/webui: 2 issues
- bug: 2 issues
- good first issue: 1 issues
- research 🔬: 1 issues
- low severity: 1 issues

---

## Issue #N/A: [User] Inference time GPU and CPU

**Link**: https://github.com/ggml-org/llama.cpp/issues/1727
**State**: closed
**Created**: 2023-06-07T02:59:28+00:00
**Closed**: 2023-06-07T09:15:52+00:00
**Comments**: 2

### Description

LLAMA_METAL=1 make -j && ./main -m ./models/guanaco-7B.ggmlv3.q4_0.bin -p "I love fish" --ignore-eos -n 1024 -ngl 1

llama_print_timings:        load time =  7918.69 ms
llama_print_timings:      sample time =  1013.54 ms /  1024 runs   (    0.99 ms per token)
llama_print_timings: prompt eval time = 14705.49 ms /   775 tokens (   18.97 ms per token)
llama_print_timings:        eval time = 46435.82 ms /  1020 runs   (   45.53 ms per token)
llama_print_timings:       total time = 69981.58 ms

my question is , it seems that the eval time is same on CPU, is it normal?

Macbook pro M1 , 32GB

---

## Issue #N/A: Why Ollama is using VRAM Only insted of VRAM + RAM?

**Link**: https://github.com/ggml-org/llama.cpp/issues/6876
**State**: closed
**Created**: 2024-04-24T14:32:16+00:00
**Closed**: 2024-04-30T21:59:00+00:00
**Comments**: 1
**Labels**: enhancement

### Description

![image](https://github.com/ggerganov/llama.cpp/assets/136520478/4df02421-b44f-4e47-bb04-33df160f4e83)

I'm running the latest version of Ollama with Llama3:70b on Windows. It's consuming most of my VRAM, but I have 63.9 GB of available RAM that could boost the speed further, yet it's not being taken into account. Is there a way to do so?

---

## Issue #N/A: [User] Training examples sometimes gets broken when training data is in Japanese

**Link**: https://github.com/ggml-org/llama.cpp/issues/1843
**State**: closed
**Created**: 2023-06-13T20:05:50+00:00
**Closed**: 2024-04-10T01:07:12+00:00
**Comments**: 4
**Labels**: stale

### Description

This is an issue to track the problem reported at https://github.com/ggerganov/llama.cpp/pull/1652#issuecomment-1586381277.

# Expected Behavior

No `�` characters in the examples.

# Current Behavior

The examples sometimes contain `�` characters (which aren't in the training data).

# Failure Information

<details>
<summary>Example 0 during Training</summary>

![image](https://github.com/ggerganov/llama.cpp/assets/14041768/40fc8be9-a327-42d7-867e-e746bae487be)
</details>
<details>
<summary>Output of the trained model</summary>

![image](https://github.com/ggerganov/llama.cpp/assets/14041768/6eb3ea01-2130-4238-a90c-c836f519c5d7)
</details>

# Steps to Reproduce

Try to train using this training data: [dataset.txt](https://github.com/ggerganov/llama.cpp/files/11716289/dataset.txt)

---

## Issue #N/A: [Vulkan] [RX 5700] Benchmark segmentation fault when setting up staging buffer

**Link**: https://github.com/ggml-org/llama.cpp/issues/6176
**State**: closed
**Created**: 2024-03-20T13:52:31+00:00
**Closed**: 2024-06-19T01:06:48+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, stale

### Description

Running `llama-cpp-benchmark` (b2466) using the Vulkan backend on an AMD RX 5700 GPU results in a segmentation fault.

```
$ llama-cpp-benchmark
main: build = 0 (unknown)
main: built with x86_64-pc-linux-gnu-gcc (Gentoo 13.2.1_p20240210 p14) 13.2.1 20240210 for x86_64-pc-linux-gnu
Starting Test
Allocating Memory of size 800194560 bytes, 763 MB
ggml_vulkan: Found 1 Vulkan devices:
Vulkan0: AMD Radeon RX 5700 (RADV NAVI10) | uma: 0 | fp16: 1 | warp size: 64
Creating new tensors

------ Test 1 - Matrix Mult via F32 code
n_threads=1
            m11: type = 0 (  f32) ne = 11008 x  4096 x     1, nb = (    4, 44032, 180355072) - Sum of tensor m11 is 45088768.00
             m2: type = 0 (  f32) ne = 11008 x   128 x     1, nb = (    4, 44032, 5636096) - Sum of tensor m2 is 2818048.00
GGML_ASSERT: /var/tmp/portage/dev-cpp/llama-cpp-2466/work/llama.cpp-b2466/ggml-vulkan.cpp:1985: false
[New LWP 61869]
[Thread debugging using libthread_db enabled]
Using host libthread_db librar

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Inference much slower with GPU offloading vs CPU on M2 Ultra

**Link**: https://github.com/ggml-org/llama.cpp/issues/8263
**State**: closed
**Created**: 2024-07-02T22:47:03+00:00
**Closed**: 2024-07-03T12:42:25+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

I have an M2 Ultra, trying to run bartowski/DeepSeek-Coder-V2-Instruct-GGUF, Q4_K_M.  When no layers are offloaded to GPU, inference runs at ~9 t/s.  However, as I offload layers to GPU up to the total of 61 layers, the inference is slower with every layer offloaded including when all layers are offloaded.  At 61 layers offloaded, inference runs at ~1 t/s

Relevant output from model loading with all layers offloaded:

```
./llama-cli -m ../models/DeepSeek-Coder-V2-Instruct-Q4_K_M-00001-of-00004.gguf -n 128 -c 4096    
Log start
main: build = 3285 (a27152b6)
main: built with Apple clang version 14.0.3 (clang-1403.0.22.14.1) for arm64-apple-darwin22.6.0
main: seed  = 1719960091
llama_model_loader: additional 3 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 42 key-value pairs and 959 tensors from ../models/DeepSeek-Coder-V2-Instruct-Q4_K_M-00001-of-00004.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note

[... truncated for brevity ...]

---

## Issue #N/A: Compile bug: Not compilable with MACOSX_DEPLOYMENT_TARGET < 10.15

**Link**: https://github.com/ggml-org/llama.cpp/issues/11612
**State**: closed
**Created**: 2025-02-03T08:28:10+00:00
**Closed**: 2025-03-20T01:07:35+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale

### Description

### Git commit

cfd74c86dbaa95ed30aa6b30e14d8801eb975d63

### Operating systems

Mac

### GGML backends

CPU

### Problem description & steps to reproduce

I'd like to compile llama.cpp for a MACOSX_DEPLOYMENT_TARGET=10.13 which is arguably sensible considering that projects like NumPy also still support this target, see e.g. https://pypi.org/project/numpy/#numpy-2.2.2-cp312-cp312-macosx_10_13_x86_64.whl and after all I hope that one of the goals of this project is to run LLMs at the edge, which frankly includes old Intel Macs for which people might have no other use :)

But recent commits use C++ functionality that is not supported by this macos target, so I kindly ask to consider setting a minimum supported macos deployment target, preferably 10.13, for this project. For example, this could be added as a CI step to check. Of course I understand that using Apple's Accelerate requires macos version >= 13.3 but in my example to reproduce I tried to strip down the compilation to the bare

[... truncated for brevity ...]

---

## Issue #N/A: The output of the main service is inconsistent with that of the server service

**Link**: https://github.com/ggml-org/llama.cpp/issues/6569
**State**: closed
**Created**: 2024-04-09T15:40:35+00:00
**Closed**: 2024-05-27T01:06:36+00:00
**Comments**: 10
**Labels**: need more info, server/webui, bug-unconfirmed, stale

### Description

**When the same quantitative model is used for server service and main service, some specific words are answered differently. It seems that the input specific words are not received or received incorrectly.
For example, BYD, Tesla, Lexus and other car names have this problem, such as Geely, BMW, Audi and so on is normal.**
The specific problem is manifested in: When obtaining the word "BYD" in the server service, non-Chinese characters such as "ruit" are not obtained or obtained. As in the first example, when asked about BYD car, the reply only involved the car, and BYD was lost.
**Test results in the server**
********************************************************
**These are three examples of problems（BYD）**
********************************************************
{
  content: ' 汽车是一种交通工具，它通常由发动机，变速箱，底盘和底盘系统，悬挂系统，转向系统，车身和车轮等组成。汽车通常由汽油或柴油发动机提供动力，通过变速箱和传动系统来控制车辆行驶的速度和方向。汽车的设计和制造技术不断提高，汽车的功能也越来越强大。现在汽车已经不仅仅是一种交通工具，它已经成为人们日常生活中不可或缺的一部分，提供了各种便利。汽车在现代社会中的作用非常广泛，它可以满足人们的出行需求，同时也可以娱

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: multiple queues or multiple threads to load model files.

**Link**: https://github.com/ggml-org/llama.cpp/issues/8796
**State**: closed
**Created**: 2024-07-31T15:35:26+00:00
**Closed**: 2024-09-18T01:07:08+00:00
**Comments**: 2
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

For the current version of the program, the model file is loaded with a single queue, single thread. this way may not realize their full performance potential on some types of disk drives, like some nvme SSD drive.

Can we implement multiple queues or multiple threads to load model files?

### Motivation

As model files become larger and larger, making full use of hardware performance saves a lot of model file loading time.

The following are the test results on an nvme SSD drive.

$ sudo s

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

## Issue #N/A: Feature Request: Attention with Linear Biases (ALiBi)

**Link**: https://github.com/ggml-org/llama.cpp/issues/1137
**State**: closed
**Created**: 2023-04-23T08:58:53+00:00
**Closed**: 2023-04-24T08:26:50+00:00
**Comments**: 6
**Labels**: enhancement, research 🔬

### Description

Consider to implement Relative Position Embeddings as opposite to APEs, as described in [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)

Short summary:

Attention with Linear Biases (ALiBi) does not add positional embeddings to word embeddings; instead, it biases query-key attention scores with a penalty that is proportional to their distance

Source: 
[The Use Case for Relative Position Embeddings](https://ofir.io/The-Use-Case-for-Relative-Position-Embeddings/)

Official implementation [here](https://github.com/ofirpress/attention_with_linear_biases).

Worth to read:
[The Curious Case of Absolute Position Embeddings](https://arxiv.org/abs/2210.12574)

---

## Issue #N/A: Misc. bug: Q4_0 repacking results in double RAM usage

**Link**: https://github.com/ggml-org/llama.cpp/issues/12149
**State**: closed
**Created**: 2025-03-02T17:49:53+00:00
**Closed**: 2025-03-05T19:57:27+00:00
**Comments**: 4
**Labels**: bug-unconfirmed

### Description

### Name and Version

b4792

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-cli

### Command line

```shell
./llama-cli -m microsoft_Phi-4-mini-instruct-Q4_0.gguf
```

### Problem description & steps to reproduce

When loading the model, it uses 4.3GB of RAM

When using Q4_K_S (similar size) it only uses 2.7GB of RAM

### First Bad Commit

_No response_

### Relevant log output

```shell

```

---

## Issue #N/A: Errors when trying to install requirements.txt

**Link**: https://github.com/ggml-org/llama.cpp/issues/4740
**State**: closed
**Created**: 2024-01-02T18:55:45+00:00
**Closed**: 2024-01-07T15:33:26+00:00
**Comments**: 5
**Labels**: bug-unconfirmed

### Description

I'm trying to build llama.cpp on a Windows 10 machine. I'm OK with Windows and with C++, but this is my first experience with trying to understand and fix issues with Python.

When I try your instructions, I get stuck at this stage:

```
python3 -m pip install -r requirements.txt
```

It appears to be expecting a module called "distutils" but I don't know how to solve that. Here is the output:

```
d:\work\github\llama.cpp>python -m pip install -r requirements.txt
Collecting numpy~=1.24.4 (from -r ./requirements/requirements-convert.txt (line 1))
  Downloading numpy-1.24.4.tar.gz (10.9 MB)
     ---------------------------------------- 10.9/10.9 MB 3.9 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
ERROR: Exception:
Traceback (most recent call last):
  File "D:\Users\peter\AppData\Local\Programs\Python\Python312\Lib\site-packages\pip\_internal\cli\base_command.py", line 180, in exc_logging_wrapper
    status =

[... truncated for brevity ...]

---

## Issue #N/A: Bug: server-cuda Docker image failing as of 2 days ago - "error while loading shared libraries: libgomp.so.1: cannot open shared object file"

**Link**: https://github.com/ggml-org/llama.cpp/issues/7774
**State**: closed
**Created**: 2024-06-05T16:51:43+00:00
**Closed**: 2024-06-06T05:17:23+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

System: Ubuntu 22.04, CUDA 12.5, Driver 555.42.02, Ryzen 7950x3D, RTX 4090

`ghcr.io/ggerganov/llama.cpp:server-cuda--b1-a10cda5` is the last image that works as expected for me. `server-cuda--b1-a5735e4` (2 days ago) and later all yield the following error:

> docker compose up llama-cpp
[+] Running 1/0
 ⠋ Container quill-ops-llama-cpp-1  Recreated                                                                                                          0.0s
Attaching to llama-cpp-1
llama-cpp-1  | /server: error while loading shared libraries: libgomp.so.1: cannot open shared object file: No such file or directory
llama-cpp-1 exited with code 127

### Name and Version

ghcr.io/ggerganov/llama.cpp:server-cuda--b1-a5735e4; Ubuntu 22.04, CUDA 12.5, Driver 555.42.02, Ryzen 7950x3D, RTX 4090, libgomp installed on host, Nvidia docker runtime

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
> docker compose up llam

[... truncated for brevity ...]

---

## Issue #N/A: MPI run on M1 Max

**Link**: https://github.com/ggml-org/llama.cpp/issues/4244
**State**: closed
**Created**: 2023-11-28T09:58:37+00:00
**Closed**: 2024-04-03T01:15:05+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [YES ] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ YES] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [YES ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ YES] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

run 2 chunks of the model on the same CPU-GPU 

# Current Behavior

ERRORS: 
GGML_ASSERT: llama.cpp:8672: false && "not implemented"
GGML_ASSERT: llama.cpp:5443: false && "not implemented"

# Environment and Context

Macbook M1 Max 32GB 

# Ste

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Llama-Quantize Not Working with Capital Letters (T^T)

**Link**: https://github.com/ggml-org/llama.cpp/issues/9569
**State**: closed
**Created**: 2024-09-20T17:47:12+00:00
**Closed**: 2024-09-20T18:55:37+00:00
**Comments**: 0
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

Ohayo gozaimasu, everyone! (・﹏・) Anon on 4chan has a tiny problem that I hope someone can help with! When Anon tried to use the llama-quantize function, it didn't work when using capital letters. But when Anon used all lowercase letters, it worked perfectly fine! (T^T)

Here's what happened:

**When Anon tried using capital letters:**
```
llama-quantize --output-tensor-type F16 --token-embedding-type F16 Mistral-7B-Instruct-v0.3-F16.gguf Q2_K
```

   It just... did nothing with those arguments! No errors, just silence. (´•̥̥̣ `•) Just a normal Q2_K.

**But when Anon used lowercase letters:**
```
llama-quantize --output-tensor-type f16 --token-embedding-type f16 Mistral-7B-Instruct-v0.3-F16.gguf Q2_K
```

   It worked perfectly! (*≧ω≦*)

I reproduced this on my machine and it was bugged too. Could someone please look into this? That anon wants to be able to use capital letters! (T_T)

### Name and Version

llama-cli.exe --version
version: 3789 (d39

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: (server) Add option to always skip all queued tasks and to process the last one only (within one slot)

**Link**: https://github.com/ggml-org/llama.cpp/issues/8275
**State**: closed
**Created**: 2024-07-03T10:48:42+00:00
**Closed**: 2024-08-18T01:07:05+00:00
**Comments**: 1
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

I think having this as a cli option (disabled by default) would be cool. Something like `--skip-queue` maybe?

### Motivation

When using the server for FIM autocompletion (for example with Continue), the front-end could send new requests for every key stroke, but only the last one is relevant. Processing the tasks that were sent before the last keystroke is useless and leads to increased response times, and wasted energy.

### Possible Implementation

I guess that would work by replacing the que

[... truncated for brevity ...]

---

## Issue #N/A: 65B quantized for CPU

**Link**: https://github.com/ggml-org/llama.cpp/issues/251
**State**: closed
**Created**: 2023-03-18T02:47:36+00:00
**Closed**: 2023-03-18T05:59:53+00:00
**Comments**: 3

### Description

Is there any way to run the 65B model on the CPU quantized for 4 bit? I saw that it's about 40 gigs for RAM usage when quantized.

How much RAM is required to quantize the 65B model? I'm not sure I have enough RAM to quantize myself, anyone have the model files for the quantized output for the 65B model for CPU? I've only found the [quantized GPU files](https://huggingface.co/decapoda-research) so far.

---

## Issue #N/A: Large sched_yield() and threading overhead (+25-40% perf boost)

**Link**: https://github.com/ggml-org/llama.cpp/issues/2134
**State**: closed
**Created**: 2023-07-07T14:31:45+00:00
**Closed**: 2024-05-19T01:07:17+00:00
**Comments**: 33
**Labels**: stale

### Description

Platform: macOS / Apple M2 Pro 

Currently the current thread finalisation / synchronisation logic in ggml_graph_compute_thread relies on sched_yield() to spin idly waiting for other threads to complete :

https://github.com/ggerganov/llama.cpp/blob/master/ggml.c#L16045 

```
            // wait for other threads to finish
            const int last = node_n;
            do {
                sched_yield();
                node_n = atomic_load(&state->shared->node_n);
            } while (node_n == last);
```

The problem with that is that this is causing absolutely gigantic amounts of overhead due to context switching and falling back to the kernel with no known time as to when the thread will come back to execution.

When profiling time on an M2 Pro:

```
./build-llvm16-native/bin/main -m ./models/7B/ggml-model-q4_0.bin -t 8 -n 512 -p "Explain a CPU microarchitecture basics." -s 3215465
main: build = 775 (d7d2e6a)
main: seed  = 3215465
llama.cpp: loading model f

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Coredump when quanting to Q4_0_*_* with imatrix

**Link**: https://github.com/ggml-org/llama.cpp/issues/8767
**State**: closed
**Created**: 2024-07-30T09:53:04+00:00
**Closed**: 2024-08-26T21:34:31+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

Hello,

I don't know if I am supposed to use imatrix with the new ARM dedicated quants (`Q4_0_*_*`). However, when I try to, I get `Aborted (core dumped)`.

Is not using imatrix with those quants intentional? If that is the case, why does the quantization to `q4_*` and `q5_*` work with imatrix?

### Name and Version

Rope scaling fix for L3.1 commit. Can't get the latest build due to a new build error I am investigating. 

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
~/files/ai/llama.cpp/git/llama-quantize --imatrix imatrix.dat Meta-Llama-3.1-8B-Instruct-F16.gguf Q4_0_4_4
load_imatrix: imatrix dataset='misc/calibration_datav3.txt'
load_imatrix: loaded 224 importance matrix entries from imatrix.dat computed on 125 chunks
prepare_imatrix: have 224 importance matrix entries
main: build = 83 (b5e9546)
main: built with cc (GCC) 14.1.1 20240720 for x86_64-pc-linux-gnu
main: quantizing 'Meta-Lla

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

## Issue #N/A: Bug: convert-hf-to-gguf.py on Gemma model ValueError: Duplicated key name 'tokenizer.chat_template'

**Link**: https://github.com/ggml-org/llama.cpp/issues/7923
**State**: closed
**Created**: 2024-06-13T19:09:23+00:00
**Closed**: 2024-07-21T01:53:03+00:00
**Comments**: 3
**Labels**: bug, low severity

### Description

### What happened?

When trying to convert

https://huggingface.co/SakanaAI/DiscoPOP-zephyr-7b-gemma/

I get the error in the title, but it's only defined a single time in tokenizer_config.json:

https://huggingface.co/SakanaAI/DiscoPOP-zephyr-7b-gemma/blob/main/tokenizer_config.json#L59

Verified locally with `cat *.json | grep chat_template` and I only get the one result

Is it somehow trying to load it twice?

Looks like when Gemma is initialized, it runs _set_vocab_sentencepiece(), which runs special_vocab.add_to_gguf (which pulls in the chat_template), and then it also again runs special_vocab.add_to_gguf

but that would mean it's been broken since April 16..

https://github.com/ggerganov/llama.cpp/pull/6689

### Name and Version

b3145 ubuntu 22.04

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
INFO:hf-to-gguf:Loading model: DiscoPOP-zephyr-7b-gemma
INFO:gguf.gguf_writer:gguf: This GGUF file is for Little Endia

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: Assertion failure when using LFM2 with parallel request processing

**Link**: https://github.com/ggml-org/llama.cpp/issues/14670
**State**: closed
**Created**: 2025-07-14T01:09:41+00:00
**Closed**: 2025-07-17T07:22:13+00:00
**Comments**: 4
**Labels**: bug-unconfirmed

### Description

### Name and Version

$ llama-cli --version
version: 5890 (982e3472)
built with Apple clang version 17.0.0 (clang-1700.0.13.3) for arm64-apple-darwin24.4.0


### Operating systems

Mac

### GGML backends

Metal

### Hardware

Apple M4 Max

### Models

<https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF> - Q6_K quanitzation

### Problem description & steps to reproduce

When running llama-server with this model and specifying parallel requests (i.e. `-np` with 2 or more parallel requests), it crashes with an assertion failure:

```
ggml/src/ggml.c:2420: GGML_ASSERT(a->ne[d] == b->ne[d]) failed
```

The specific command I ran was:

```sh
llama-server -hf LiquidAI/LFM2-1.2B-GGUF:Q6_K -c 32678 -np 2
```

### First Bad Commit

_No response_

### Relevant log output

```shell
$ lldb -- llama-server -hf LiquidAI/LFM2-1.2B-GGUF:Q6_K -c 32678 -np 2
(lldb) target create "llama-server"
Current executable set to '/opt/homebrew/bin/llama-server' (arm64).
(lldb) settings set -- target.run-args  "-hf" "L

[... truncated for brevity ...]

---

## Issue #N/A: [Feature Suggestion] Dynamic prompt

**Link**: https://github.com/ggml-org/llama.cpp/issues/673
**State**: closed
**Created**: 2023-04-01T08:05:43+00:00
**Closed**: 2024-04-11T01:07:19+00:00
**Comments**: 1
**Labels**: stale

### Description

Would love to see a feature where both the AI and the user could change the initial prompt in-situ and when necessary.

Essentially, this would be the same as changing the prompt without exiting llama.cpp, thus eliminates the need to reload the model weights and forgetting the context.

To trigger this, it could be a trigger word in the input, such as \iNewPrompt: You are an insane AI assistant. You always gives imprecise answers and easily goes into panic mode. Once you are panicked, you will start babbling and answer everything hysterically. You will become sane again when I tell you to stop panic.

---

## Issue #N/A: Bug: Failed to load model

**Link**: https://github.com/ggml-org/llama.cpp/issues/8516
**State**: closed
**Created**: 2024-07-16T15:18:53+00:00
**Closed**: 2024-07-17T07:05:36+00:00
**Comments**: 10
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

Hi guys. I have got a problem after I compile Llama on my machine. It built properly, but when I try to run it, it is looking for a file don't even exist (a model).

Is it normal ?

### Name and Version

version: 0 (unknown)
built with cc (Gentoo Hardened 14.1.1_p20240622 p2) 14.1.1 20240622 for x86_64-pc-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
zohran@alienware-m17-r3 ~/Downloads/llama.cpp-b3400 $ ./examples/chat.sh
Log start
main: build = 0 (unknown)
main: built with cc (Gentoo Hardened 14.1.1_p20240622 p2) 14.1.1 20240622 for x86_64-pc-linux-gnu
main: seed  = 1721142929
llama_model_load: error loading model: llama_model_loader: failed to load model from ./models/llama-7b/ggml-model-q4_0.gguf

llama_load_model_from_file: failed to load model
llama_init_from_gpt_params: error: failed to load model './models/llama-7b/ggml-model-q4_0.gguf'
main: error: unable to load model

zohran@alie

[... truncated for brevity ...]

---

## Issue #N/A: Compile bug: 执行python3 convert-hf-to-gguf.py D:\DevelopSoftware\Ollamamodel\DeepSeek-R1-Medical-COT-500命令，没有响应

**Link**: https://github.com/ggml-org/llama.cpp/issues/12286
**State**: closed
**Created**: 2025-03-09T14:10:53+00:00
**Closed**: 2025-04-26T01:07:38+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale

### Description

### Git commit

![Image](https://github.com/user-attachments/assets/0b8fc84f-a949-4f49-a5a2-e4aa9c7b7876)

### Operating systems

Windows

### GGML backends

CPU

### Problem description & steps to reproduce

执行python3 convert-hf-to-gguf.py D:\DevelopSoftware\Ollamamodel\DeepSeek-R1-Medical-COT-500命令，没有响应

### First Bad Commit

_No response_

### Compile command

```shell
执行python3 convert-hf-to-gguf.py D:\DevelopSoftware\Ollamamodel\DeepSeek-R1-Medical-COT-500命令，没有响应
```

### Relevant log output

```shell
执行python3 convert-hf-to-gguf.py D:\DevelopSoftware\Ollamamodel\DeepSeek-R1-Medical-COT-500命令，没有响应
```

---

## Issue #N/A: Misc. bug: vulkan: performance regression after fd123cfead49eb32e386e26b8ef7a6d41554dda5

**Link**: https://github.com/ggml-org/llama.cpp/issues/12553
**State**: closed
**Created**: 2025-03-24T20:49:08+00:00
**Closed**: 2025-05-09T01:07:52+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

fd123cfead49eb32e386e26b8ef7a6d41554dda5

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

vulkan backend

### Command line

```shell

```

### Problem description & steps to reproduce

| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| gemma3 12B Q5_K - Medium       |   8.09 GiB |    11.77 B | Vulkan     |  99 |         pp512 |         61.69 ± 0.04 |
| gemma3 12B Q5_K - Medium       |   8.09 GiB |    11.77 B | Vulkan     |  99 |         tg128 |         21.87 ± 0.01 |

build: a53f7f7b8 (4908)

| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| gemma3 12B Q

[... truncated for brevity ...]

---

## Issue #N/A: Segfault with 65B model

**Link**: https://github.com/ggml-org/llama.cpp/issues/84
**State**: closed
**Created**: 2023-03-13T07:19:05+00:00
**Closed**: 2023-03-31T05:04:49+00:00
**Comments**: 6
**Labels**: need more info

### Description

This is the output with `-fsanitize=address`:
```
AddressSanitizer:DEADLYSIGNAL
=================================================================
==167666==ERROR: AddressSanitizer: SEGV on unknown address 0x558c0562c438 (pc 0x558a27cc9807 bp 0x000000000000 sp 0x7ffeb2f57310 T0)
==167666==The signal is caused by a READ memory access.
    #0 0x558a27cc9807 in ggml_element_size (/home/mattmcal/repos/llama.cpp/main+0x49807)
    #1 0x558a27c9c03c in llama_eval(llama_model const&, int, int, std::vector<int, std::allocator<int> > const&, std::vector<float, std::allocator<float> >&, unsigned long&) (/home/mattmcal/repos/llama.cpp/main+0x1c03c)
    #2 0x558a27c960fb in main (/home/mattmcal/repos/llama.cpp/main+0x160fb)
    #3 0x7fe45e046189 in __libc_start_call_main ../sysdeps/nptl/libc_start_call_main.h:58
    #4 0x7fe45e046244 in __libc_start_main_impl ../csu/libc-start.c:381
    #5 0x558a27c9b1a0 in _start (/home/mattmcal/repos/llama.cpp/main+0x1b1a0)

AddressSanitizer can not p

[... truncated for brevity ...]

---

## Issue #N/A: How to fine tune LLaMA 3 in Google Colab (Pro)?

**Link**: https://github.com/ggml-org/llama.cpp/issues/6800
**State**: closed
**Created**: 2024-04-21T02:12:14+00:00
**Closed**: 2024-04-25T16:13:38+00:00
**Comments**: 1

### Description

I have a JSONL dataset like this:

```
{"text": "This is raw text in 2048 tokens I want to feed in"},
{"text": "This is next line, tokens are also 2048"}
```

It would be nice to fine-tune in 4, 8, or 16-bit LoRA and then just merge as before!

---

## Issue #N/A: Misc. bug: missing messages in JSON export via llama-server web UI

**Link**: https://github.com/ggml-org/llama.cpp/issues/13552
**State**: open
**Created**: 2025-05-14T23:06:48+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

### Name and Version

$ ./llama-cli --version
version: 5359 (de4c07f9)
built with cc (Debian 12.2.0-14) 12.2.0 for x86_64-linux-gnu


### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell
llama-server --threads 10 --model ./Ministral-8B.gguf
```

### Problem description & steps to reproduce

When I use the web UI to download a conversation, a JSON file is downloaded, but I only seem to get the conversation's metadata, while the messages are *not* included.

```
{
  "id": "conv-1747263476494",
  "lastModified": 1747263485918,
  "currNode": 1747263476577,
  "name": "test"
}
```

### First Bad Commit

Not sure. I remember this worked fine with a fresh build from a month ago or so.

**edit:** It still worked in b5124 (bc091a4), which is when the 'Download' button was still located on the right, instead of in a context menu on the left.

---

