# slow_7to30days - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- bug-unconfirmed: 12 issues
- bug: 7 issues
- model: 3 issues
- good first issue: 2 issues
- high severity: 2 issues
- need more info: 1 issues
- enhancement: 1 issues
- low severity: 1 issues
- medium severity: 1 issues
- help wanted: 1 issues

---

## Issue #N/A: The prompt is not converted to tokens

**Link**: https://github.com/ggml-org/llama.cpp/issues/113
**State**: closed
**Created**: 2023-03-14T04:00:37+00:00
**Closed**: 2023-04-07T16:19:58+00:00
**Comments**: 8
**Labels**: bug

### Description

./main -m ./models/7B/ggml-model-q4_0.bin -p "Building a website can be done in 10 simple steps:" -t 8 -n 512
![image](https://user-images.githubusercontent.com/6960679/224889376-929af931-309c-41c0-8319-32fba4eb5ee1.png)

llama.cpp Is the latest version
Can anyone help me? Thanks!


---

## Issue #N/A: Misc. bug: failed to initialize MUSA: system has unsupported display driver / musa driver combination

**Link**: https://github.com/ggml-org/llama.cpp/issues/11675
**State**: closed
**Created**: 2025-02-05T13:16:21+00:00
**Closed**: 2025-02-18T01:47:26+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

### Name and Version

Docker Version

(base) ko@ubuntu:~$ docker info | grep mthreads
WARNING: bridge-nf-call-iptables is disabled
 Runtimes: mthreads mthreads-experimental runc io.containerd.runc.v2
WARNING: bridge-nf-call-ip6tables is disabled
 Default Runtime: mthreads

---------------------------------
(base) ko@ubuntu:~$ docker ps
CONTAINER ID   IMAGE                                    COMMAND                  CREATED          STATUS          PORTS     NAMES
aa5d7028eeba   ghcr.io/ggerganov/llama.cpp:light-musa   "/app/llama-cli -m /…"   52 seconds ago   Up 51 seconds             funny_haibt

(base) ko@ubuntu:~$ docker inspect aa5d7028eeba
[
    {
        "Id": "aa5d7028eeba54ebaef3fe00ebbf0fdddc30d7a70b32b4ade5588bbe84a0c651",
        "Created": "2025-02-05T13:03:24.687585793Z",
        "Path": "/app/llama-cli",
        "Args": [
            "-m",
            "/models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
            "-p",
            "deepseek",
            "-ngl",
        

[... truncated for brevity ...]

---

## Issue #N/A: llama : add DeepSeek-v2-Chat support

**Link**: https://github.com/ggml-org/llama.cpp/issues/7118
**State**: closed
**Created**: 2024-05-07T06:22:43+00:00
**Closed**: 2024-05-28T15:07:06+00:00
**Comments**: 67
**Labels**: good first issue, model

### Description

please support deepseek-ai/DeepSeek-V2-Chat

https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat

---

## Issue #N/A: When using the qwen2.5-vl model on AMD Ryzen APU under Windows, the error "failed to allocate Vulkan0 buffer of size 4342230552" may appear.

**Link**: https://github.com/ggml-org/llama.cpp/issues/13250
**State**: closed
**Created**: 2025-05-02T02:29:40+00:00
**Closed**: 2025-05-13T14:24:06+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

### Name and Version

C:\Users\xeden\Downloads\llama-b5255-bin-win-vulkan-x64>llama-cli --version
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon(TM) 8060S Graphics (AMD proprietary driver) | uma: 1 | fp16: 1 | warp size: 64 | shared memory: 32768 | int dot: 1 | matrix cores: KHR_coopmat
version: 5255 (d24d5928)
built with MSVC 19.43.34808.0 for x64

### Operating systems

Windows

### GGML backends

Vulkan

### Hardware

CPU AMD Ryzen AI MAX 395 Memory 128G （CPU 64G GPU 64G）

### Models

Qwen2.5-VL-3B-Instruct-f16.gguf

### Problem description & steps to reproduce

Device
CPU AMD Ryzen AI MAX 395
Memory
128GB GPU 64mb, CPU 64mb
Operating system
win11
Used llama.cpp version
llama-b5255-bin-win-vulkan-x64

Since AMD does not support ROCM of Ryzen AI MAX 395, I used vulkan as the backends, and it is no problem to run most of the llm models, including deepseek.



### First Bad Commit

_No response_

### Relevant log output

```shell
C:\Users\xeden\Downloads\llama-b5255-b

[... truncated for brevity ...]

---

## Issue #N/A: terminate called after throwing an instance of 'std::runtime_error'

**Link**: https://github.com/ggml-org/llama.cpp/issues/1569
**State**: closed
**Created**: 2023-05-23T09:16:05+00:00
**Closed**: 2023-06-05T20:24:30+00:00
**Comments**: 3

### Description

./main -m ./models/7B/ggml-model-q4_0.bin -p "Building a website can be done in 10 simple steps:" -n 512
main: build = 584 (2e6cd4b)
main: seed  = 1684832847
ggml_opencl: selecting platform: 'PowerVR'
ggml_opencl: selecting device: 'PowerVR B-Series BXE-4-32'
ggml_opencl: device FP16 support: true
llama.cpp: loading model from ./models/7B/ggml-model-q4_0.bin
terminate called after throwing an instance of 'std::runtime_error'
  what():  unexpectedly reached end of file
Aborted (core dumped)

---

## Issue #N/A: GGML_ASSERT: ggml.c:4014: false zsh: abort      ./main -m ./models/65B/ggml-model-q4_0.bin -t 16 -n 256 --repeat_penalty 1.0 

**Link**: https://github.com/ggml-org/llama.cpp/issues/400
**State**: closed
**Created**: 2023-03-22T17:12:40+00:00
**Closed**: 2023-04-19T19:43:32+00:00
**Comments**: 5
**Labels**: need more info

### Description

Not sure why this happens, I am on the latest commit and I am up-to-date on everything
I did some tests and it seems like it breaks after 500~ tokens
Is this a model limitation or can I fix this by increasing some value?

---

## Issue #N/A: Server: show the total number of tokens processed/generated

**Link**: https://github.com/ggml-org/llama.cpp/issues/4647
**State**: closed
**Created**: 2023-12-26T15:11:37+00:00
**Closed**: 2024-01-02T15:48:51+00:00
**Comments**: 0
**Labels**: enhancement

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

The default web-based server UI has a footer like this:

> 258ms per token, 3.87 tokens per second

It would help if the total numbers of tokens processed (including those from the prompt) and generated (i.e., not from the prompt and not from the chat box) are a

[... truncated for brevity ...]

---

## Issue #N/A: Bug: llama-infill segmentation fault if missing  --in-suffix

**Link**: https://github.com/ggml-org/llama.cpp/issues/8179
**State**: closed
**Created**: 2024-06-28T01:16:16+00:00
**Closed**: 2024-07-08T06:34:36+00:00
**Comments**: 0
**Labels**: bug-unconfirmed, low severity

### Description

### What happened?

llama-infill segmentation fault if missing  --in-suffix



### Name and Version

./llama-cli --version
version: 3235 (88540445)
built with Apple clang version 15.0.0 (clang-1500.3.9.4) for arm64-apple-darwin23.5.0

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
./llama-infill -m ../../models/Publisher/Repository/codeshell_modified.gguf --temp 0.7 --repeat_penalty 1.1 -n 20 --in-prefix "def helloworld():\n print(\"hell"
Log start
main: build = 3235 (88540445)
main: built with Apple clang version 15.0.0 (clang-1500.3.9.4) for arm64-apple-darwin23.5.0
main: seed  = 1719537258
llama_model_loader: loaded meta data with 25 key-value pairs and 508 tensors from ../../models/Publisher/Repository/codeshell_modified.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.arch

[... truncated for brevity ...]

---

## Issue #N/A: Bug: DeepSeek-V2-Lite GGML_ASSERT: ggml-metal.m:1857: dst_rows <= 2048 and aborts

**Link**: https://github.com/ggml-org/llama.cpp/issues/7652
**State**: closed
**Created**: 2024-05-30T18:20:52+00:00
**Closed**: 2024-06-14T14:14:11+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

`~/projects/llama.gguf/main -c 4096 -ngl 99 -f ./reproduce.txt -m ~/Downloads/DeepSeek-V2-Lite-Chat.Q5_K.gguf`

[reproduce.txt](https://github.com/ggerganov/llama.cpp/files/15503974/reproduce.txt)

The DS2 Lite quant was downloaded from https://huggingface.co/legraphista/DeepSeek-V2-Lite-Chat-IMat-GGUF


Error:

```
GGML_ASSERT: ggml-metal.m:1857: dst_rows <= 2048
GGML_ASSERT: ggml-metal.m:1857: dst_rows <= 2048
GGML_ASSERT: ggml-metal.m:1857: dst_rows <= 2048
[1]    70124 abort      ~/projects/llama.gguf/main -c 4096 -ngl 99 -f ./reproduce.txt -m
```

Process exited without generating the output.

If it helps, I ran the same thing with lldb:

```
Assistant: GGML_ASSERT: ggml-metal.m:1857: dst_rows <= 2048
GGML_ASSERT: ggml-metal.m:1857: dst_rows <= 2048
GGML_ASSERT: ggml-metal.m:1857: dst_rows <= 2048
GGML_ASSERT: ggml-metal.m:1857: dst_rows <= 2048
Process 72593 stopped
* thread #4, queue = 'ggml-metal', stop reason = signal SIGABRT
  

[... truncated for brevity ...]

---

## Issue #N/A: alaways "failed to tokenize string! "

**Link**: https://github.com/ggml-org/llama.cpp/issues/290
**State**: closed
**Created**: 2023-03-19T11:29:50+00:00
**Closed**: 2023-04-07T16:15:34+00:00
**Comments**: 7
**Labels**: bug

### Description

failed to tokenize string! 

system_info: n_threads = 16 / 16 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | VSX = 0 | 
failed to tokenize string!

main: prompt: ' china'
main: number of tokens in prompt = 1
     1 -> ''

sampling parameters: temp = 0.800000, top_k = 40, top_p = 0.950000, repeat_last_n = 64, repeat_penalty = 1.300000


曲ー！ /S部ュース / KSHErsLAheLUE - THE NEW CH`,MEgeERSION IS HERE@ÿThis entry was вер in news on JuneSASSSASS8 by adminS [end of text]


---

## Issue #N/A: Bug: ggml-aarch64.c does not compile on Windows ARM64 with MSVC

**Link**: https://github.com/ggml-org/llama.cpp/issues/8446
**State**: closed
**Created**: 2024-07-12T08:21:28+00:00
**Closed**: 2024-07-25T16:01:01+00:00
**Comments**: 12
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

the __asm__ directive in line 507 of ggml-aarch64.c
is dependant on: defined(__ARM_NEON) && defined(__aarch64__) which is tue for MSVC arm64 with the standard build scripts and does not compile with MSVC.

the other __asm__ directives work out except for the one in line 1278, they are to not being used.

building with clang works, its MSVC specific.

### Name and Version

llama.cpp version: 3378 (71c1121d), Microsoft C/C++  Version 19.40.33812 for ARM64, Windows 11 Enterprise 24H2 Build 26100.1150

### What operating system are you seeing the problem on?

Windows

### Relevant log output

```shell
build-instructions which fail:
cmake -B build
cmake --build build --config Release --target llama-cli

CMake build also fails for arm64-windows-msvc.cmake, but works for arm64-windows-llvm.cmake (clang compile instead of MSVC frontend works)

The build worked a few days ago with older llama.cpp versions.
```


---

## Issue #N/A: Misc. bug: Problems with official jinja templates (Gemma 2, Llama 3.2, Qwen 2.5)

**Link**: https://github.com/ggml-org/llama.cpp/issues/11866
**State**: closed
**Created**: 2025-02-14T09:57:12+00:00
**Closed**: 2025-03-10T10:40:33+00:00
**Comments**: 12
**Labels**: bug

### Description

### Name and Version

llama-cli --version
version: 4713 (a4f011e8)
built with MSVC 19.42.34436.0 for x64


### Operating systems

Windows

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell
1. llama-server -ngl 99 -m gemma-2-2b-it-Q8_0.gguf --jinja --chat-template-file gemma2.jinja -c 8192
2. llama-server -ngl 99 -m Llama-3.2-3B-Instruct-Q8_0.gguf --jinja --chat-template-file llama3.2.jinja -c 8192
3. llama-server -ngl 99 -m Qwen2.5-1.5B-Instruct-Q8_0.gguf --jinja --chat-template-file qwen2.5.jinja -c 8192
```

### Problem description & steps to reproduce

Extracting official chat templates from chat_template field in tokenizer_config.json ([Gemma 2](https://huggingface.co/google/gemma-2-2b-it/blob/main/tokenizer_config.json#L2003), [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/tokenizer_config.json#L2053), [Qwen 2.5](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/tokenizer_config.json#L198)), s

[... truncated for brevity ...]

---

## Issue #N/A: llm/llama.cpp/quantize - Mixtral model quantizatization halts at random step

**Link**: https://github.com/ggml-org/llama.cpp/issues/6606
**State**: closed
**Created**: 2024-04-11T10:20:42+00:00
**Closed**: 2024-04-19T19:07:03+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

I fine-tuned Mixtral with QLORA on my dataset and merged it. 

After cloning llamacpp, I built the quantization feature using `make -C llm/llama.cpp quantize`. 

Then, I converted the model using 
`python llm/llama.cpp/convert.py LLaMA-Factory/models/mixtral_instruct01_lora_4bit_full --outtype f16`. 

When attempting to quantize with 
`llm/llama.cpp/quantize converted.bin quantized.bin q4_0`

the process randomly freezes—this could happen at step 6, anywhere past 100, or even beyond 500 steps. Despite several attempts, the issue persists without resolution.

screenshot 
![1111](https://github.com/ggerganov/llama.cpp/assets/23155849/dcd03305-e433-4c10-8174-5b7128b7a33f)


---

## Issue #N/A: llama : add support for llama2.c models

**Link**: https://github.com/ggml-org/llama.cpp/issues/2379
**State**: closed
**Created**: 2023-07-24T20:15:39+00:00
**Closed**: 2023-08-11T23:17:27+00:00
**Comments**: 93
**Labels**: help wanted, good first issue, model

### Description

The new [llama2.c](https://github.com/karpathy/llama2.c) project provides means for training "baby" llama models stored in a custom binary format, with 15M and 44M models already available and more potentially coming out soon.

We should provide a simple conversion tool from `llama2.c` bin format to `ggml` format so we can run inference of the models in `llama.cpp`

Great task for people looking to get involved in the project



---

## Issue #N/A: Eval bug: HIP: llama.cpp server locks up when running multiple instances on the same gpu

**Link**: https://github.com/ggml-org/llama.cpp/issues/12991
**State**: closed
**Created**: 2025-04-17T04:38:58+00:00
**Closed**: 2025-04-24T15:15:30+00:00
**Comments**: 7
**Labels**: bug-unconfirmed

### Description

### Name and Version

llama-server version 5145 (but happens with older versions too)

### Operating systems

Linux

### GGML backends

HIP

### Hardware

```
Device 0: AMD Radeon VII, gfx906:sramecc+:xnack- (0x906), VMM: no, Wave Size: 64
Device 1: AMD Radeon RX 6900 XT, gfx1030 (0x1030), VMM: no, Wave Size: 32
```
Ryzen 9 5900x ubuntu 24.04, rocm 6.3.3

### Models

Any completion with any embedding model
e. g. Llama-3.2-1B-Instruct-Q8_0.gguf, mxbai-embed-large-v1_fp16.gguf


### Problem description & steps to reproduce

1) Running 2 instances
```
~/llama.cpp/build/bin/llama-server -m /models/llm_models/Llama-3.2-1B-Instruct-Q8_0.gguf -c 32768 -fa -ngl 200 --port 5001 --no-mmap --mlock --host 0.0.0.0
```
```
 ~/llama.cpp/build/bin/llama-server -m /models/llm_models/mxbai-embed-large-v1_fp16.gguf --embedding -ngl 200 -c 512 -b 512 -ub 512 -fa --host 0.0.0.0 --port 5002 --no-mmap --mlock
```
device selection doesn't matter

2) Vectorize some messages or anything else with embedding mode

[... truncated for brevity ...]

---

## Issue #N/A: [Work Group] Add RLHF like ColosallChat on bigger dataset to achieve ChatGPT quality

**Link**: https://github.com/ggml-org/llama.cpp/issues/743
**State**: closed
**Created**: 2023-04-03T18:02:06+00:00
**Closed**: 2023-04-13T08:55:27+00:00
**Comments**: 5

### Description

[Link to ColosallChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat
)
# Add RLHF like ColosallChat on bigger dataset to achieve ChatGPT quality

![lama-alpaca](https://user-images.githubusercontent.com/84633629/229592343-36b95e81-2f85-4fa0-9a5a-edd5267bb190.gif)


Although models in the GPT series, such as ChatGPT and GPT-4, are highly powerful, they are unlikely to be fully open-sourced. Fortunately, the open-source community has been working hard to address this.

For example, Meta has open-sourced the LLaMA model, which offers parameter sizes ranging from 7 billion to 65 billion. A 13 billion parameter model can outperform the 175 billion GPT-3 model on most benchmark tests. However, since it doesn’t have an instruct tuning stage, its actual generated results are not satisfactory.

Stanford’s Alpaca generates training data in a self-instructed manner by calling OpenAI’s API. With only 7 billion parameters, this lightweight model can be fine-tuned at

[... truncated for brevity ...]

---

## Issue #N/A: parallel.cpp exits when encountering a long prompt.

**Link**: https://github.com/ggml-org/llama.cpp/issues/4086
**State**: closed
**Created**: 2023-11-15T11:42:43+00:00
**Closed**: 2023-11-24T11:34:32+00:00
**Comments**: 24
**Labels**: bug-unconfirmed

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

In ./examples/parallel/parallel.cpp, I added the following two lines to the final output:
```cpp
int cache_count = llama_get_kv_cache_token_count(ctx);
LOG_TEE("Cache KV size %d", cache_count);
```
I believe that the logic in line 221 of parallel.cpp:
```cpp
//

[... truncated for brevity ...]

---

## Issue #N/A: llama-tts libc++abi: terminating due to uncaught exception of type std::out_of_range: vector
Aborted

**Link**: https://github.com/ggml-org/llama.cpp/issues/11749
**State**: closed
**Created**: 2025-02-08T05:33:49+00:00
**Closed**: 2025-02-21T15:56:06+00:00
**Comments**: 4
**Labels**: tts

### Description

```
./llama-tts -m $model -mv $voice -p "Hi i am Felix"

build: 4663 (c026ba3c) with clang version 19.1.5 for armv7a-unknown-linux-android24
llama_model_loader: loaded meta data with 38 key-value pairs and 272 tensors from /storage/7DE2-358B/ysf/models/smollm-135m-instruct-q8_0.gguf (version GGUF V3 (latest))       llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = SmolLM 135M
llama_model_loader: - kv   3:                       general.organization str              = HuggingFaceTB                                                                       llama_model_loader: - kv   4:                           general.finetune str              = instruct-add-basi

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: Abort is called in a thread from a custom thread pool during a llama_decode call

**Link**: https://github.com/ggml-org/llama.cpp/issues/13990
**State**: closed
**Created**: 2025-06-03T12:25:37+00:00
**Closed**: 2025-06-20T11:57:37+00:00
**Comments**: 12
**Labels**: bug-unconfirmed

### Description

### Name and Version

./llama-cli --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA A2, compute capability 8.6, VMM: yes
  Device 1: NVIDIA A2, compute capability 8.6, VMM: yes
register_backend: registered backend CUDA (2 devices)
register_device: registered device CUDA0 (NVIDIA A2)
register_device: registered device CUDA1 (NVIDIA A2)
register_backend: registered backend CPU (1 devices)
register_device: registered device CPU (Intel Xeon Processor (Icelake))
version: 5572 (7675c555)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu

### Operating systems

Linux

### GGML backends

CUDA

### Hardware

2 x A2 or  3 x A100

### Models

SmolLM2-360M-Instruct-BF16

### Problem description & steps to reproduce

We are testing inference with 15 threads in the worker pool when an abort is called from one of those threads during a llama_decode call. 

- main thread
- 3 th

[... truncated for brevity ...]

---

## Issue #N/A: server hangs up if given ascii strings containing utf-8 non-breaking space characters

**Link**: https://github.com/ggml-org/llama.cpp/issues/3809
**State**: closed
**Created**: 2023-10-27T03:49:48+00:00
**Closed**: 2023-11-14T19:48:33+00:00
**Comments**: 1
**Labels**: bug

### Description

I am running the MMLU test on llamma.cpp with mistrallite and the server locked up when it encountered a string containing 0xc2 0xa0 (utf-8 non-breaking space) in the test set data.  Instead of locking up a better behavior would be to reject the input string as having bad characters and return an error message similar to what it does when the json prompt is mis formatted.  I do not know where the lock up is happening (tokenizer, parser, etc.) but it just freezes and the gpu is running full load so it seems like tokenizer got stuck.

---

## Issue #N/A: [User] The server crashes when a wrong `n_keep` is set

**Link**: https://github.com/ggml-org/llama.cpp/issues/3550
**State**: closed
**Created**: 2023-10-08T17:26:00+00:00
**Closed**: 2023-10-21T09:11:10+00:00
**Comments**: 11
**Labels**: bug

### Description

# Prerequisites

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Running the `server` with a specific context window and then passing a specific **wrong** `n_keep` with a specific number of tokens. Since I'm passing a wrong parameter, I expect the server to either give an error or just generate something anyway.

# Current Behavior

The server crashes.

# Environment and Context

**Commit:** db3abcc114d5

[... truncated for brevity ...]

---

## Issue #N/A: std::runtime_error exceptions not caught as std::string by Visual C++?

**Link**: https://github.com/ggml-org/llama.cpp/issues/1589
**State**: closed
**Created**: 2023-05-24T21:30:56+00:00
**Closed**: 2023-06-05T20:24:31+00:00
**Comments**: 8

### Description

Platform: Windows x64
Commit: 7e4ea5beff567f53be92f75f9089e6f11fa5dabd

I noticed that `main.exe` fails for me when I run it without any parameters, and no model is found. The only output I got was:

    C:\Develop\llama.cpp>bin\Release\main.exe
    main: build = 583 (7e4ea5b)
    main: seed  = 1684960511

In the debugger, I found that this line had triggered an unhandled exception:

    throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));

When I change the `catch` statement like this

    -    } catch (const std::string & err) {
    -        fprintf(stderr, "error loading model: %s\n", err.c_str());
    +    } catch (const std::exception & err) {
    +        fprintf(stderr, "error loading model: %s\n", err.what());

then I get a proper error message again, as in the past:

    C:\Develop\llama.cpp\build>bin\Release\main.exe
    main: build = 583 (7e4ea5b)
    main: seed  = 1684961912
    error loading model: failed to open model

[... truncated for brevity ...]

---

## Issue #N/A: [User] Insert summary of your issue or enhancement..

**Link**: https://github.com/ggml-org/llama.cpp/issues/1749
**State**: closed
**Created**: 2023-06-08T03:03:13+00:00
**Closed**: 2023-06-18T07:18:07+00:00
**Comments**: 0

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Please provide a detailed written description of what you were trying to do, and what you expected `llama.cpp` to do.

# Current Behavior

Please provide a detailed written description of what `llama.cpp` did, instead.

# Environment and Context

Please provide detailed informat

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: GPU Hang Error on Metal backend

**Link**: https://github.com/ggml-org/llama.cpp/issues/12277
**State**: closed
**Created**: 2025-03-08T22:42:53+00:00
**Closed**: 2025-03-26T15:08:40+00:00
**Comments**: 3
**Labels**: bug-unconfirmed

### Description

### Name and Version

$ build/bin/llama-cli --version
version: 4857 (0fd7ca7a)
built with Apple clang version 16.0.0 (clang-1600.0.26.6) for arm64-apple-darwin24.3.0

### Operating systems

Mac

### GGML backends

Metal

### Hardware

Apple M4 Max

### Models

[Google Gemma-2 it GGUF](https://huggingface.co/google/gemma-2b-it-GGUF)

### Problem description & steps to reproduce

When I run llama-cli, the inference crashes partway through. Sometimes I see "error: Caused GPU Hang Error (00000003:kIOGPUCommandBufferCallbackErrorHang)", and sometimes "error: Discarded (victim of GPU error/recovery) (00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)".

I first noticed this after I upgraded my OS to Sequoia 15.3.1. My existing Ollama install started showing these errors. I built llama.cpp from source and replicated them here. So far I've seen the problem on Gemma-2b, Gemma2-2b, and Llama-3.3. I've tried running the Apple hardware diagnostic in case this is a hardware problem, but the d

[... truncated for brevity ...]

---

## Issue #N/A: [Falcon] Attempting to run Falcon-180B Q5/6 give "illegal character"

**Link**: https://github.com/ggml-org/llama.cpp/issues/3484
**State**: closed
**Created**: 2023-10-05T03:21:50+00:00
**Closed**: 2023-10-18T07:32:24+00:00
**Comments**: 15

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

I'm attempting to run llama.cpp, latest master, with TheBloke's Falcon 180B Q5/Q6 quantized GGUF models, but it errors out with "invalid character".
I'm unable to find any issues about this online anywhere.
Another system of mind causes the same problem, and a buddy

[... truncated for brevity ...]

---

## Issue #N/A: AMD GPU is slower than expected

**Link**: https://github.com/ggml-org/llama.cpp/issues/6750
**State**: closed
**Created**: 2024-04-18T19:45:19+00:00
**Closed**: 2024-05-01T17:16:58+00:00
**Comments**: 6
**Labels**: bug-unconfirmed

### Description

# Question
AMD supposed to be faster, but only receive   11.93 tokens per second
Here is my inference command 
`./main -m ./models/llama-2-7b-chat.Q2_K.gguf -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e
`
I am running ubuntu docker rocm:pytorch/latest





---

## Issue #N/A:  Bug: [SYCL] GGML_ASSERT Error with Llama-3.1 SYCL Backend. Windows 11 OS

**Link**: https://github.com/ggml-org/llama.cpp/issues/8660
**State**: closed
**Created**: 2024-07-23T23:37:34+00:00
**Closed**: 2024-07-31T14:22:59+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

I am trying to run [(these)](https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF) models with Q8 and Q4 quantization when I come across this error.

`GGML_ASSERT: D:\llama.cpp\llama.cpp\ggml\src\ggml-backend.c:96: base != NULL && "backend buffer base cannot be NULL"`

Here is the command I am using to reproduce the error.
`.\llama-cli.exe -m models\Meta-Llama-3.1-8B-Instruct-Q6_K.gguf -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e -ngl 33 -s 0 -sm none -mg 0`

I have tried compiling latest git branch from source as well as have tried latest released executables for SYCL backend.

I am using Intel ARC A770 GPU on Windows 11. Every other model works fine including all Llama 2 and Llama 3 models.



### Name and Version

```
.\llama-cli.exe --version
version: 3449 (de280085)
built with MSVC 19.40.33812.0 for
```

### What operating system are you seeing the problem on?

Windows

### Relevant log output

`

[... truncated for brevity ...]

---

## Issue #N/A: How do i use convert-unversioned-ggml-to-ggml.py?

**Link**: https://github.com/ggml-org/llama.cpp/issues/808
**State**: closed
**Created**: 2023-04-06T12:22:58+00:00
**Closed**: 2023-04-14T13:12:39+00:00
**Comments**: 12
**Labels**: bug, model

### Description

Hi it told me to use the convert-unversioned-ggml-to-ggml.py file and gave me an error saying your gpt4all model is too old. So i converted the gpt4all-lora-unfiltered-quantized.bin file with llama tokenizer. And it generated some kind of orig file in the same directory where the model was. When i tried to run the miku.sh file which had the latest generated file as model it gave me another error stating this 
`main: seed = 1680783525
llama_model_load: loading model from './models/gpt4all-7B/gpt4all-lora-unfiltered-quantized.bin' - please wait ...
./models/gpt4all-7B/gpt4all-lora-unfiltered-quantized.bin: invalid model file (bad magic [got 0x67676d66 want 0x67676a74])
        you most likely need to regenerate your ggml files
        the benefit is you'll get 10-100x faster load times
        see https://github.com/ggerganov/llama.cpp/issues/91
        use convert-pth-to-ggml.py to regenerate from original pth
        use migrate-ggml-2023-03-30-pr613.py if you deleted originals

[... truncated for brevity ...]

---

## Issue #N/A: [User] 在windows上无法开启LLAMA_QKK_64编译

**Link**: https://github.com/ggml-org/llama.cpp/issues/2152
**State**: closed
**Created**: 2023-07-09T02:55:36+00:00
**Closed**: 2023-07-22T13:37:38+00:00
**Comments**: 0

### Description


报错如下：
MSBuild version 17.6.3+07e294721 for .NET Framework

  Checking Build System
  Generating build details from Git
  -- Found Git: C:/Program Files/Git/cmd/git.exe (found version "2.39.1.windows.1") 
  Building Custom Rule E:/project/py/llama-cpp-python/vendor/llama.cpp/CMakeLists.txt
  Building Custom Rule E:/project/py/llama-cpp-python/vendor/llama.cpp/CMakeLists.txt
  ggml.c
  k_quants.c
E:\project\py\llama-cpp-python\vendor\llama.cpp\k_quants.c(3820,32): warning C4013: “_mm_set1_pi8”未定义；假设外部返回 int [E:\project\py\llama-cpp-python\vendor\llama.cpp\build\ggml.vcxproj]
E:\project\py\llama-cpp-python\vendor\llama.cpp\k_quants.c(3827,33): warning C4013: “_mm_set_epi64”未定义；假设外部返回 int [E:\project\py\llama-cpp-python\vendor\llama.cpp\build\ggml.vcxproj]
E:\project\py\llama-cpp-python\vendor\llama.cpp\k_quants.c(3820,21): error C2440: “初始化”: 无法从“int”转换为“const __m64” [E:\project\py\llama-cpp-python\vendor\llama.cpp\build\ggml.vcxproj]
E:\project\py\llama-cpp-python\vendor\l

[... truncated for brevity ...]

---

## Issue #N/A: Assertion failure in ggml_mul_mat_q4_0_q8_1_cuda (g_compute_capabilities[id] >= MIN_CC_DP4A)

**Link**: https://github.com/ggml-org/llama.cpp/issues/4229
**State**: closed
**Created**: 2023-11-27T01:37:33+00:00
**Closed**: 2023-12-23T08:16:34+00:00
**Comments**: 11
**Labels**: bug

### Description

# Current Behavior

I got this crash on https://github.com/cebtenzzre/llama.cpp/tree/18fe116e9a5aa45a83bd1d6f043f98dc395f218e:

```
2023-11-26 20:06:04 INFO:Loaded the model in 9.14 seconds.

GGML_ASSERT: /home/jared/src/forks/llama-cpp-python/vendor/llama.cpp/ggml-cuda.cu:5484: false
```

# Failure Information (for bugs)

Backtrace:
```
#3  0x00007f5999fd54b8 in __GI_abort () at abort.c:79
#4  0x00007f585ac6b357 in ggml_mul_mat_q4_0_q8_1_cuda (stream=<optimized out>, nrows_dst=<optimized out>, nrows_y=<optimized out>, ncols_y=<optimized out>, 
    nrows_x=<optimized out>, ncols_x=<optimized out>, dst=<optimized out>, vy=<optimized out>, vx=<optimized out>)
    at /home/jared/src/forks/llama-cpp-python/vendor/llama.cpp/ggml-cuda.cu:5076
#5  ggml_cuda_op_mul_mat_q (src0=src0@entry=0x204c00320, src1=src1@entry=0x269123d80, dst=dst@entry=0x269123f00, src0_dd_i=src0_dd_i@entry=0x90be00000 "", 
    src1_ddf_i=src1_ddf_i@entry=0x9b0400000, src1_ddq_i=src1_ddq_i@entry=0x9af

[... truncated for brevity ...]

---

