# author_association_NONE - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 29

### Label Distribution

- bug-unconfirmed: 17 issues
- stale: 15 issues
- enhancement: 6 issues
- critical severity: 1 issues
- low severity: 1 issues

---

## Issue #N/A: ImportError: cannot import name 'BaseVocab' from 'gguf'

**Link**: https://github.com/ggml-org/llama.cpp/issues/7776
**State**: closed
**Created**: 2024-06-05T18:54:35+00:00
**Closed**: 2024-06-06T08:03:31+00:00
**Comments**: 10

### Description

I want to convert LLaVAMistral model to GGUF, I am using tthis code **convert-legacy-llama.py** which is present inside **examples** folder. But got this error **ImportError: cannot import name 'BaseVocab' from 'gguf'**. I didn't found anything online

---

## Issue #N/A: Can not offload layers to GPU (llama3)

**Link**: https://github.com/ggml-org/llama.cpp/issues/6787
**State**: closed
**Created**: 2024-04-20T14:41:33+00:00
**Closed**: 2024-06-06T01:06:45+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale

### Description

make LLAMA_CUDA=1
 ./main -m /models/Meta-Llama-3-70B-Instruct.Q4_K_M.gguf -r '<|eot_id|>' 
 --in-prefix "\n<|start_header_id|>user<|end_header_id|>\n\n" 
 --in-suffix "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" 
 -p "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nHi! How are you?<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n" 
 -n 1024 -ngl 90

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:   no
ggml_cuda_init: CUDA_USE_TENSOR_CORES: yes
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA H800, compute capability 9.0, VMM: yes
llm_load_tensors: ggml ctx size =    0.15 MiB
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/33 layers to GPU
llm_load_tensors:        CPU buffer size =  5459.93 MiB
....................

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: crash when pooling_type == LLAMA_POOLING_TYPE_MEAN

**Link**: https://github.com/ggml-org/llama.cpp/issues/12543
**State**: closed
**Created**: 2025-03-24T10:00:01+00:00
**Closed**: 2025-05-09T01:07:53+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

Revision 9b169a4d4e01af7bc07a6981b53b27c18c9470d8


### Operating systems

Linux

### GGML backends

CPU

### Hardware

ARM Ampere

### Models

Qwen2.5-14B-Instruct-1M-Q5_K_M

### Problem description & steps to reproduce

Setting pooling_type = LLAMA_POOLING_TYPE_MEAN and calling llama_init_from_model() causes this crash:

```
/build/source/ggml/src/ggml.c:2738: GGML_ASSERT(ggml_can_mul_mat(a, b)) failed
```

Setting to LLAMA_POOLING_TYPE_LAST and changing nothing else works correctly.

### First Bad Commit

_No response_

### Relevant log output

```shell
/build/source/ggml/src/ggml.c:2738: GGML_ASSERT(ggml_can_mul_mat(a, b)) failed
```

---

## Issue #N/A: Misc. bug: Potential memory leak in backend registry

**Link**: https://github.com/ggml-org/llama.cpp/issues/12986
**State**: closed
**Created**: 2025-04-16T21:28:39+00:00
**Closed**: 2025-05-31T01:07:42+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

build: 5124 (bc091a4d) with MSVC 19.43.34810.0 for x64 (debug)
static build (MT/MTd) with VS2022 / LLAMA & GGML
GGML_STATIC / GGML_USE_CPU / GGML_USE_BLAS / GGML_USE_CUDA

### Operating systems

Windows

### Which llama.cpp modules do you know to be affected?

Other (Please specify in the next section)

### Command line

```shell
No command line. Using personal C++ (preliminary) implementation that follow the steps of the llama-cli.
```

### Problem description & steps to reproduce

I'm currently only do :

```
llama_backend_init();
llama_model_load_from_file();
llama_model_free();
llama_backend_free();
```
There's no problem of memory leak until I offload the model on my GPU with `n_gpu_layers > 0`

The problem of leak comes with `ggml_backend_cuda_reg()` that act as a singleton-like fashion and never release the static context allocated as `new`. I don't know if the problem is specific from my build (`GGML_STATIC`) but I think that is potentially related to the 

[... truncated for brevity ...]

---

## Issue #N/A: [Question] Plans to parallelize GPU matrix execution in ggml_cuda_op_mul_mat?

**Link**: https://github.com/ggml-org/llama.cpp/issues/4659
**State**: closed
**Created**: 2023-12-28T02:39:41+00:00
**Closed**: 2024-04-02T01:09:44+00:00
**Comments**: 2
**Labels**: stale

### Description

I noticed in `ggml_cuda_op_mul_mat` that multiple GPUs execute large matrix operations in a serial manner within a for loop, as confirmed by actual testing. 
When I execute on 70B with 8 GPUs, the execution time on a single device is approximately 0.03ms (including synchronization time), and the total loop execution time is approximately 0.200ms.
The code is here:
```
        for (int64_t id = 0; id < g_device_count; ++id) {
```

A more effective solution would be to implement parallel execution using multiple cards and threads. 
Is there any plan for improvement in this regard?


---

## Issue #N/A: Misc. bug: Flash Attention not working on CDNA3 ROCm 6.4 MI300

**Link**: https://github.com/ggml-org/llama.cpp/issues/13145
**State**: closed
**Created**: 2025-04-28T06:17:46+00:00
**Closed**: 2025-06-12T01:07:49+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

```
llama-server --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Instinct MI300X VF, gfx942:sramecc+:xnack- (0x942), VMM: no, Wave Size: 64
version: 5201 (85f36e5e)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
```

### Operating systems
Linux

### Which llama.cpp modules do you know to be affected?
llama-server


### Problem description & steps to reproduce



```
llama-server -m UD-IQ1_S/MAI-DS-R1-UD-IQ1_S-00001-of-00004.gguf -c 32768  -b 8192 -ub 4096  -ngl 999  -to 3600  -a MAI-DS-R1-UD-IQ1_S --no-mmap -t 1 -nkvo -fa
```

small context windows, small prompts, it seems to work, but if i use large context, it seems like it's using CPU. I get about 20-30 tokens/s on small prompts. otherwise, it just hangs for hours


without `-fa` i have to use much smaller `-c` `-b` `-ub` values to not max out on VRAM, and it runs larger prompts, but o

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: convert_lora_to_gguf ignores outtype

**Link**: https://github.com/ggml-org/llama.cpp/issues/10671
**State**: closed
**Created**: 2024-12-05T13:42:56+00:00
**Closed**: 2025-01-19T01:07:38+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

8f1d81a from 2024-09-01
until today
0cd182e

### Operating systems

Windows

### Which llama.cpp modules do you know to be affected?

llama-quantize

### Problem description & steps to reproduce

convert_lora_to_gguf ignores outtype


hi
i want to convert a transformer fp32 lora to a quant gguf lora. i have done this a few months ago and it worked fine with the convert script. now i tryed the same with a up to date version of llama.cpp and ...well... it still works but i wondered why the output file has the same size than the input file. i tested this much to large file and it looks like it still works. at least for me in kobold.cpp. but then i got curious why this quant lora is so large. for me it turned out the new version of llamacpp is ignoring the outtype given for the convert script.

no matter what --outtype i chose its alway FP32 and the logging while "converting" looks like this for all lines.

this is with --outtype q8_0
INFO:hf-to-gguf:blk.0.

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: BGE-M3 Embedding model is not accessible

**Link**: https://github.com/ggml-org/llama.cpp/issues/13494
**State**: closed
**Created**: 2025-05-13T05:27:28+00:00
**Closed**: 2025-06-27T01:07:56+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

load_backend: loaded RPC backend from D:\AI\app\llama.cpp\ggml-rpc.dll
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 6600M (AMD proprietary driver) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 32768 | int dot: 1 | matrix cores: none
load_backend: loaded Vulkan backend from D:\AI\app\llama.cpp\ggml-vulkan.dll
load_backend: loaded CPU backend from D:\AI\app\llama.cpp\ggml-cpu-haswell.dll
version: 5361 (cf0a43bb)
built with MSVC 19.43.34808.0 for x64

### Operating systems

Windows

### GGML backends

Vulkan

### Hardware

Ryzen 7 5800H + RX 6600M

### Models

bge-m3-FP16.gguf

### Problem description & steps to reproduce

Failed to add the embedding model using the llama-b5361-bin-win-cuda12.4-x64 version on a workstation with RTX 4800. The reranking model, LLM model, and VLM model can all be added. Then, testing on my laptop with a Ryzen 7 5800H and RX 6600M using llama-b5361-bin-win-vulkan-x64, the embedding model that I had previously

[... truncated for brevity ...]

---

## Issue #N/A: Llava 1.6: server not decoding images, but works via CLI

**Link**: https://github.com/ggml-org/llama.cpp/issues/5515
**State**: closed
**Created**: 2024-02-15T19:38:58+00:00
**Closed**: 2024-02-17T20:49:59+00:00
**Comments**: 6
**Labels**: bug-unconfirmed

### Description

First, let me say that I really appreciate all the work you guys are putting into llama.cpp -- it's really impressive.

I'm testing out yesterday's release of llava 1.6 (thanks so much for working on that tricky PR, @cmp-nct), and it's working well via the CLI, but when I run it via the server I'm seeing the below when it receives a request:

```
clip_image_load_from_bytes: failed to decode image bytes
slot 0 - failed to load image [id: 12]
task 1 - error: internal_error
```

## How I'm running via CLI (works)

```
./llava-cli -m ./models/llava-1-6/mistral-7b-q_5_k.gguf --mmproj ./models/llava-1-6/mmproj-mistral7b-f16.gguf --image ./media/images/ginsberg.png -p "Who is this?" --temp 0.1
```

## How I'm running via Server (doesn't work)

### To start the server:
```
./server -m ./models/llava-1-6/mistral-7b-q_5_k.gguf --mmproj ./models/llava-1-6/mmproj-mistral7b-f16.gguf --host 127.0.0.1 --port 8080
```

### The request I'm sending:
```
curl --request POST \
  

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: Performance regression on aarch64 q4_0

**Link**: https://github.com/ggml-org/llama.cpp/issues/14134
**State**: closed
**Created**: 2025-06-11T21:43:06+00:00
**Closed**: 2025-06-17T09:58:33+00:00
**Comments**: 14
**Labels**: bug-unconfirmed

### Description

### Name and Version

llama-cli --version
version: 5615 (f470bc36)
built with Android (13324770, +pgo, +bolt, +lto, +mlgo, based on r530567d) clang version 19.0.0 (https://android.googlesource.com/toolchain/llvm-project 97a699bf4812a18fb657c2779f5296a4ab2694d2) for x86_64-unknown-linux-gnu

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-bench

### Command line

```shell

```

### Problem description & steps to reproduce

Q4_0 performance significantly dropped after this commit
/build-android-f470bc36/llama-bench -m ../gemma-2-2b-q4_0.gguf -p 512 -n 0                                            
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| gemma2 2B Q4_0                 |   1.51 GiB |     2.61 B | CPU        |       8 |           pp512 |  

[... truncated for brevity ...]

---

## Issue #N/A: How to utilize GPU on Android to accelerate inference?

**Link**: https://github.com/ggml-org/llama.cpp/issues/8705
**State**: closed
**Created**: 2024-07-26T07:30:12+00:00
**Closed**: 2024-12-31T01:07:28+00:00
**Comments**: 18
**Labels**: stale

### Description

### Discussed in https://github.com/ggerganov/llama.cpp/discussions/8704

<div type='discussions-op-text'>

<sup>Originally posted by **ElaineWu66** July 26, 2024</sup>
I am trying to compile and run llama.cpp demo on my android device (QUALCOMM Adreno) with linux and termux.
Any suggestion on how to utilize the GPU?
I have followed tutorial https://github.com/JackZeng0208/llama.cpp-android-tutorial, since the OpenCL is broken and removed now, it's not working.

Thanks!!!
</div>

---

## Issue #N/A: Add support for OPTForCausalLM

**Link**: https://github.com/ggml-org/llama.cpp/issues/6473
**State**: closed
**Created**: 2024-04-04T09:20:31+00:00
**Closed**: 2024-06-08T01:07:02+00:00
**Comments**: 5
**Labels**: enhancement, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description
I want to quantize OPT based models like Galactica in ggml format. Please add support to OPT architecture models.



---

## Issue #N/A: Misc. bug: llama-perplexity PPL score is too high for Falcon H1 TQ1_0 model

**Link**: https://github.com/ggml-org/llama.cpp/issues/14616
**State**: open
**Created**: 2025-07-10T14:46:05+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

### Name and Version

latest main branch

### Operating systems

Linux with Intel Xeon CPU E7-8890 v3 CPUs with AVX2 extension

### Which llama.cpp modules do you know to be affected?

_No response_

### Command line

```shell
./build/bin/llama-perplexity -m models/Falcon-H1-7B-Instruct-TQ1_0.gguf -t 32 -f wikitext-2-raw/wiki.test.raw
```

### Problem description & steps to reproduce

Hi all,

Thanks very much for merging the latest code to support the latest Falcon H1 model. I enjoyed it!

There is a small issue with the PPL score of the Falcon H1 TQ1_0 model (https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct-GGUF), some console printout is here:
**[1]19349056200306.7266,[2]15260594264367.6348,[3]9614462106840.3945,[4]10046092761693.6270,[5]8923093178308.6699,[6]10074241009251.5293,[7]8918590985171.6797,[8]9302717398391.8418,[9]8422255088098.3027,[10]9236596171282.8242,[11]9563579420726.5371,[12]9321558796127.1172,[13]10303646595689.4785,[14]9832904238423.9219,[15]9454125549978.1699

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: ROCm error: Could not attach to process (AMD MI50/60: gfx906)

**Link**: https://github.com/ggml-org/llama.cpp/issues/10701
**State**: closed
**Created**: 2024-12-07T02:34:51+00:00
**Closed**: 2025-01-22T01:07:18+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

``` 
./build/bin/llama-cli --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 ROCm devices:
  Device 0: AMD Radeon Graphics, compute capability 9.0, VMM: no
  Device 1: AMD Radeon Graphics, compute capability 9.0, VMM: no
version: 4277 (c5ede384)
built with cc (Ubuntu 13.2.0-23ubuntu4) 13.2.0 for x86_64-linux-gnu
```

### Operating systems

Linux

### GGML backends

HIP

### Hardware

CPU: AMD Ryzen 5950x, 96GB RAM
GPU: 2x AMD MI60 (gfx906), Nvidia 3090 (for video output) 


### Models

Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

### Problem description & steps to reproduce

I am getting 'could not attach to process' when I do not use Flash attention in llama.cpp. However, inference works without any errors when flash attention is on.
I built llama.cpp on my Ubuntu 24.04 with this command:

```
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)"     cmake -S . -B build -DGGML_HIP

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Adding support for Ternary DiT models

**Link**: https://github.com/ggml-org/llama.cpp/issues/10334
**State**: closed
**Created**: 2024-11-16T13:26:44+00:00
**Closed**: 2024-11-20T14:43:33+00:00
**Comments**: 2
**Labels**: enhancement

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

llama.cpp has already supported ternary quantization for LLMs, e.g., Bitnet b1.58. We have trained a Ternary diffusion transformer model [TerDiT](https://github.com/Lucky-Lance/TerDiT). Due to the limitations of our engineering abilities, I am wondering if llama.cpp can support the deployment of this model, this can help our research a lot.

### Motivation

Ternary quantization has become popular and has demonstrated computational speedups and power reductions, as demonstrated in works like llama

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

## Issue #N/A: Compile bug: ggml-impl.h(314): error: identifier "__fp16" is undefined on Jetson AGX Xavier

**Link**: https://github.com/ggml-org/llama.cpp/issues/10555
**State**: closed
**Created**: 2024-11-28T03:33:52+00:00
**Closed**: 2024-12-04T00:41:38+00:00
**Comments**: 14
**Labels**: bug-unconfirmed

### Description

### Git commit

Commit 9f91251

### Operating systems

Linux

### GGML backends

CUDA

### Problem description & steps to reproduce

I am trying to compile
llama.cpp
on an NVIDIA Jetson AGX Xavier and I am getting the error:

/tmp/llama.cpp/ggml/src/ggml-cuda/../ggml-impl.h(314): error: identifier "__fp16" is undefined
There were other errors originally, but since
Commit 9f91251
, those errors have disappeared. However, now I am encountering the error mentioned above.


### First Bad Commit

_No response_

### Relevant log output

```shell
cmake -B build -DGGML_CUDA=ON -DGGML_CCACHE=OFF
-- The C compiler identification is GNU 9.4.0
-- The CXX compiler identification is GNU 9.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for worki

[... truncated for brevity ...]

---

## Issue #N/A:  Eval bug:  when using convert_hf_to_gguf.py  convert  llama-3.2-11B-vision to gguf

**Link**: https://github.com/ggml-org/llama.cpp/issues/10681
**State**: closed
**Created**: 2024-12-06T03:06:32+00:00
**Closed**: 2025-02-21T01:07:23+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

latest

### Operating systems

Linux

### GGML backends

CUDA

### Hardware

A6000

### Models

https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct

### Problem description & steps to reproduce

when using convert_hf_to_gguf.py  convert  llama-3.2-11B-vision to gguf,  it can be converted, but when  inference with image,

import ollama
response = ollama.chat(
    model='llama-3.2-11B-V:latest',
    messages=[{
        'role': 'user',
        'content': 'What is in this image?',
        'images': ['/home/user/1.jpg']
    }]
)

its outputs is nothing to do with image content,but chat with text, it seems to reasonable


### First Bad Commit

_No response_

### Relevant log output

```shell
its outputs is nothing to do with image content,but chat with text, it seems to reasonable:
output:

model='llama-3.2-11B-V:latest' created_at='2024-12-06T03:09:09.933807094Z' done=True done_reason='stop' total_duration=1022

[... truncated for brevity ...]

---

## Issue #N/A: Bug: server crash when changing LoRA scale while using CUDA

**Link**: https://github.com/ggml-org/llama.cpp/issues/9451
**State**: closed
**Created**: 2024-09-12T10:42:14+00:00
**Closed**: 2024-09-21T00:41:08+00:00
**Comments**: 11
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

The server chases when changing the LoRA scale and using CUDA. To reproduce it:
  - Start the server with a model and a LoRA and load layers to CUDA. 
  - Then, prompt the model as usual.
  - After that, modify the scale of the LoRA to 0.0.
  - Finally, prompt the model again and you will get the error.
 
This also happens if you start the LoRA with a scale of 0.0 and modify the scale to a value greater than 0.0. When layers are not loaded in GPU, it works well. In the provided example only happens when 32 and 33 layers are loaded in GPU.

### Name and Version

```shell
~/llama.cpp$ ./llama-cli --version
version: 3735 (df4b7945)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
```

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
~/llama.cpp$ ./llama-server -m ../.cache/huggingface/hub/models--bartowski--Phi-3.5-mini-instruct-GGUF/snapshots/5c8d5381f90b6ca4348f090238be

[... truncated for brevity ...]

---

## Issue #N/A: 7B model returning complete non-sense

**Link**: https://github.com/ggml-org/llama.cpp/issues/474
**State**: closed
**Created**: 2023-03-24T20:05:37+00:00
**Closed**: 2023-03-25T10:26:17+00:00
**Comments**: 2

### Description

i followed a YouTube video to build the program https://www.youtube.com/watch?v=coIj2CU5LMU&t=186s. it itself follows the issue #103 

# Expected Behavior

As a test I ran the ./chat.sh in git bash, it ran but when I said the AI "hello" I expected hello back.

# Current Behavior

it responded with
```
‼ ▼→▬▬▲↨‼↑♥♠♦"♥ ☻ Ôüç ∟ Ôüç ↔¶
‼∟ Ôüç ♥
►♠
▼!↕☻    ▼ ↓     $▼∟▼↕♣↔"‼↔♥
☺       ►        ↔ Ôüç   #↑"▼↑♠$$▬☺☻
```

# Environment and Context 

Please provide detailed information about your computer setup. This is important in case the issue is not reproducible except for under certain specific conditions.

I’m running a i7-13 th gen with 32 go of ram and a 3060.

windows 11 home

git bash to run the commands and cmake to compile

```
Python 3.10.10
cmake 3.26.1
g++.exe (MinGW.org GCC-6.3.0-1) 6.3.0
```

# Failure Logs

```
$ ./chat.sh
main: seed = 1679687646
llama_model_load: loading model from './models/7B/ggml-model-f16.bin' - please wait ...
llama_

[... truncated for brevity ...]

---

## Issue #N/A: Bug: llama-cli out "error: input is empty" and end

**Link**: https://github.com/ggml-org/llama.cpp/issues/8976
**State**: closed
**Created**: 2024-08-10T19:23:25+00:00
**Closed**: 2024-08-12T16:04:55+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, low severity

### Description

### What happened?

my run u:\llama\llama.cpp\build\bin\llama-cli.exe -mli -co -fa -ngl 64 -cnv --chat-template gemma -m llama3-8B-Chinese-Chat-q8.gguf

win11 amd 7900x hip 6.1 vs 2022
cmake  -DGGML_OPENMP=OFF -DGGML_BUILD_EXAMPLES=OFF -DGGML_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_C_COMPILER=D:/AMD/ROCm/6.1/bin/clang.exe -DCMAKE_CXX_COMPILER=D:/AMD/ROCm/6.1/bin/clang++.exe

d:\AI_Model\ggml_llava>u:\llama\llama.cpp\build\bin\llama-cli.exe -m qwen2-7b-instruct-q5_k_m.gguf --chat-template llama2
Log start
main: build = 0 (unknown)
main: built with  for x86_64-pc-windows-msvc
main: seed  = 1723317298
llama_model_loader: loaded meta data with 21 key-value pairs and 339 tensors from qwen2-7b-instruct-q5_k_m.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen2
llama_model_loader: - kv   1:             

[... truncated for brevity ...]

---

## Issue #N/A: Mamba is not working with Metal

**Link**: https://github.com/ggml-org/llama.cpp/issues/6189
**State**: closed
**Created**: 2024-03-21T00:16:42+00:00
**Closed**: 2024-03-21T19:11:37+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

System: 
2020 M1 MacBook Pro 16GB Unified Memory (RAM) 
CPU: 8 Core (4 Performance and 4 Efficiency Cores)
GPU: 8 Core

I tried to run a GGUF of Mamba, but it won't run using Metal. I get the following error:
```
ggml_metal_graph_compute_block_invoke: error: unsupported op 'SSM_CONV'
```

```
main: build = 2447 (c47cf414)
main: built with Apple clang version 15.0.0 (clang-1500.1.0.2.5) for arm64-apple-darwin23.3.0
main: seed  = 1710979793
llama_model_loader: loaded meta data with 23 key-value pairs and 642 tensors from /Users/jsarnecki/Downloads/ggml-bagel-2.8b-v0.2-q8_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = mamba
llama_model_loader: - kv   1:                               general.name str              = bagel-2.8b-v0.2
llama_model_loader: - kv   2:                       mamba.context_le

[... truncated for brevity ...]

---

## Issue #N/A: Compile bug: How to build llama.android example with -DGGML_VULKAN=ON through android studio.

**Link**: https://github.com/ggml-org/llama.cpp/issues/12085
**State**: closed
**Created**: 2025-02-26T14:32:40+00:00
**Closed**: 2025-03-03T14:04:25+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

### Git commit

master

### Operating systems

Linux

### GGML backends

Vulkan

### Problem description & steps to reproduce

I'm trying to compile llama.android with vulkan backend enabled: i.e. with -DGGML_VULKAN=ON.

Build is failing with error - glslc not found.



### First Bad Commit

_No response_

### Compile command

```shell
./gradlew build
```

### Relevant log output

```shell
Task failed with an exception.
-----------
* What went wrong:
Execution failed for task ':llama:buildCMakeRelease[arm64-v8a]'.
> com.android.ide.common.process.ProcessException: ninja: Entering directory `/home/dcaimlpune/ashwini_wp/bmw/llama.cpp/examples/llama.android/llama/.cxx/Release/5x6s385r/arm64-v8a'
  [1/65] Building CXX object build-llama/ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu-hbm.cpp.o
  [2/65] Building CXX object build-llama/ggml/src/CMakeFiles/ggml-base.dir/ggml-threading.cpp.o
  [3/65] Creating directories for 'vulkan-shaders-gen'
  [4/65] Building CXX object build-llama/ggml

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

## Issue #N/A: vulkan: Requested buffer size exceeds  (when using -ctk)

**Link**: https://github.com/ggml-org/llama.cpp/issues/12728
**State**: closed
**Created**: 2025-04-03T03:52:18+00:00
**Closed**: 2025-04-03T13:23:50+00:00
**Comments**: 4

### Description

unable to allocate buffer on vulkan? 

I do have memory on GPU. but met this issue when using `-ckt`

init:    Vulkan0 KV buffer size = 13800.00 MiB
llama_context: KV self size  = 13800.00 MiB, K (q8_0): 6120.00 MiB, V (f16): 7680.00 MiB
ggml_vulkan: Device memory allocation of size 2166360064 failed.
ggml_vulkan: Requested buffer size exceeds device memory allocation limit: ErrorOutOfDeviceMemory
ggml_gallocr_reserve_n: failed to allocate Vulkan0 buffer of size 2166360064
llama_init_from_model: failed to initialize the context: failed to allocate compute pp buffers


bellow are success output running when `without -ckt` and `with -nkvo`

1. w/o ckt 
init: kv_size = 4096, offload = 1, type_k = 'f16', type_v = 'f16', n_layer = 60, can_shift = 0
init:    Vulkan0 KV buffer size = 19200.00 MiB
llama_context: KV self size  = 19200.00 MiB, K (f16): 11520.00 MiB, V (f16): 7680.00 MiB
llama_context:    Vulkan0 compute buffer size =  1174.00 MiB
llama_context: Vulkan_Host compute buffer size = 

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Support for Deepseek Janus-Pro-7B & Janus-1.3B

**Link**: https://github.com/ggml-org/llama.cpp/issues/11490
**State**: closed
**Created**: 2025-01-29T14:53:13+00:00
**Closed**: 2025-04-22T01:08:02+00:00
**Comments**: 4
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

DeepSeek recently released **[Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B)** and **[Janus-1.3B](https://huggingface.co/deepseek-ai/Janus-1.3B)**, both multimodal models currently supported in [Transformers](https://github.com/huggingface/transformers). 



**Resources:** [Janus GitHub](https://github.com/deepseek-ai/Janus)


### Motivation

Adding them to `llama.cpp` would enable efficient local inference, expanding support for state-of-the-art multimodal AI. Would love to see t

[... truncated for brevity ...]

---

## Issue #N/A: Is it possible to dynamically switch multiple LoRA adapters?

**Link**: https://github.com/ggml-org/llama.cpp/issues/6377
**State**: closed
**Created**: 2024-03-29T01:35:18+00:00
**Closed**: 2024-04-01T04:03:17+00:00
**Comments**: 4
**Labels**: enhancement

### Description

Thank you for this great project. I have two questions.

# Q1 Is it possible to dynamically switch multiple LoRA adapters?

In the transformers library, we can load multiple adapters to the original model by `load_adapter` then switch the specified adapter with `set_adapter` instantlly like below.

```python
# base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
)

# load multiple adapters
model.load_adapter("model/adapter1/", "adapter1")
model.load_adapter("model/adapter2/", "adapter2")

# switch adapter (change takes instantly)
model.set_adapter("adapter2")
```

Is it possible to do the same thing with llama cpp?
I found there is an API `llama_model_apply_lora_from_file()` .
https://github.com/ggerganov/llama.cpp/blob/bfe7dafc9cf96b9a09ead347fed9a547930fc631/llama.h#L441

But the description says `The model needs to be reloaded before applying a new adapter, otherwise the adapter will be applied on top of the previous one`.

Is it impossi

[... truncated for brevity ...]

---

## Issue #N/A: How do I install with Make?

**Link**: https://github.com/ggml-org/llama.cpp/issues/1868
**State**: closed
**Created**: 2023-06-15T07:06:17+00:00
**Closed**: 2024-04-10T01:07:06+00:00
**Comments**: 7
**Labels**: stale

### Description

I wasn't able to run cmake on my system (ubuntu 20.04), but just wondering how I get the built binaries out, installed on the system...

```
make install
```
didn't work for me :(

---

## Issue #N/A: llama_decode return logbits whose value are all nan

**Link**: https://github.com/ggml-org/llama.cpp/issues/6957
**State**: closed
**Created**: 2024-04-28T02:37:01+00:00
**Closed**: 2024-06-13T02:55:44+00:00
**Comments**: 8
**Labels**: stale

### Description

**enviroment**
gpu : nvidia titan rtx (compute capability 7.5) with 24 GB vram
os : ubuntu 22.04
driver version :550.76
git commit:b4e4b8a9
model:starcode-7B q8_0.v2

**problem encounted**
llama_model ,llama_batch,llama_ctx was all corrrectly initilized
following code behaves different on different gpus
```
llama_batch batch= //... ;  
int ret =llama_decode(ctx,batch);  
float* logits=llama_get_logits(ctx);  
```

1. on a10,it returns the right value
2. on the titan rtx ,all logits are all nan,leading to unusable result and can't engeter the token generation step
3. use the starcoder-1b model ,titan rtx outputs right value
4. using other frameworks,titan rtx can generate valid answer

**question**

1. it seems model > 3 trigger the problem,is it the problem of gpu?
2.is there any way to inspect infer process,seeming that which layer produces nan?

---

## Issue #N/A: Implement Flash Attention Option

**Link**: https://github.com/ggml-org/llama.cpp/issues/19
**State**: closed
**Created**: 2023-03-11T18:57:36+00:00
**Closed**: 2023-07-28T19:26:05+00:00
**Comments**: 4
**Labels**: enhancement

### Description

Would love to see a faster, more memory efficient attention implemented like Flash Attention. :)

---

