# recent_burst_30days - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 20
- Closed Issues: 10

### Label Distribution

- bug-unconfirmed: 25 issues
- enhancement: 5 issues

---

## Issue #N/A: Eval bug: thinking not working if "tool_choice" is "required" for Qwen models (QwQ, Qwen3, etc.)

**Link**: https://github.com/ggml-org/llama.cpp/issues/14599
**State**: open
**Created**: 2025-07-09T16:41:49+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Name and Version

docker image ghcr.io/ggml-org/llama.cpp:server-cuda-b5849

The models don't think if "tool_choice" is set as "required"

### Operating systems

Linux

### GGML backends

CUDA

### Hardware

12700  + 3090

### Models

https://huggingface.co/Qwen/Qwen3-32B-GGUF/blob/main/Qwen3-32B-Q4_K_M.gguf
https://huggingface.co/Qwen/QwQ-32B-GGUF

### Problem description & steps to reproduce

set "tool_choice" as "required" then the model don't think. If not set, the model will think and call the tools.

### First Bad Commit

_No response_

### Relevant log output

```shell
{
choices: [
0: {
finish_reason: "tool_calls"
index: 0
message: {
role: "assistant"
content: null
tool_calls: [
0: {
...}]
```

---

## Issue #N/A: Misc. bug: out of memory error after PR #13746

**Link**: https://github.com/ggml-org/llama.cpp/issues/14740
**State**: open
**Created**: 2025-07-17T16:01:03+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

### Name and Version

Tested on latest master, and multiple previous versions. Currently at:
```
version: 1313 (086cf81)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
```

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell
CUDA_VISIBLE_DEVICES=2 llama-server -m models/Phi-4-mini-instruct.BF16.gguf -ngl 80 --host :: --port 31420 --jinja --ctx-size 0 --no-kv-offload -t 2
```

### Problem description & steps to reproduce

Hello. I've updated today from f125b8dccff34439a26bf750c9edef358c48c1f8 and llama-server is now throwing out of memory errors. I've backtracked and traced the issue to [PR #13746](https://github.com/ggml-org/llama.cpp/pull/13746). I am not familiar with the codebase so not sure what exactly this changed.

The server has 256GB RAM and 4x old Nvidia M60 cards, Ubuntu 22.04 with CUDA 12.9. I'm running phi-4-mini-instruct which until now this commit fit narrowly by offload

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: error loading model: vk::PhysicalDevice::createDevice: ErrorExtensionNotPresent

**Link**: https://github.com/ggml-org/llama.cpp/issues/14559
**State**: closed
**Created**: 2025-07-07T03:54:53+00:00
**Closed**: 2025-07-07T09:10:30+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

### Name and Version

.\llama-cli.exe --version
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon(TM) 890M Graphics (AMD proprietary driver) | uma: 1 | fp16: 1 | warp size: 64 | shared memory: 32768 | int dot: 1 | matrix cores: KHR_coopmat
version: 5835 (6491d6e4)
built with clang version 19.1.5 for x86_64-pc-windows-msvc

### Operating systems

Windows

### GGML backends

Vulkan

### Hardware

AMD Ryzen AI 9 HX 370 w/ Radeon 890M

### Models

Meta-Llama-3.1-8B-Instruct-Q4_K_M & Qwen2.5-VL-7B-Instruct-Q4_K_M

### Problem description & steps to reproduce

I'm trying to run llama-server on this AMD laptop but keep running into this error. I've test this code this another AMD laptop (Ryzen 7 8845HS w/ Radeon 780M) and an intel PC (i7-14700 w/ UHD Graphics 770) and both works fine.

### First Bad Commit

_No response_

### Relevant log output

```shell
0.00.160.925 I load_tensors: loading model tensors, this can take a while... (mmap = true)
0.00.165.104 E llama_model_load: 

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: CANNOT CONVERT THE MODEL

**Link**: https://github.com/ggml-org/llama.cpp/issues/14610
**State**: closed
**Created**: 2025-07-10T04:39:28+00:00
**Closed**: 2025-07-11T00:03:59+00:00
**Comments**: 4
**Labels**: bug-unconfirmed

### Description

### Name and Version

Hello getting this error:

RuntimeError: Internal: could not parse ModelProto from /Users/yuki/Downloads/yunasmol-pytorch/tokenizer.model

How to fix?

### Operating systems

Mac

### Which llama.cpp modules do you know to be affected?

Python/Bash scripts

### Command line

```shell
(yuna) yuki@yuki llama.cpp % python convert_hf_to_gguf.py --outfile yunamiru-f16.gguf --outtype q8_0 --no-lazy --model-name "Yuna Ai V4 Miru" "/Users/yuki/Downloads/yunasmol-pytorch"
INFO:hf-to-gguf:Loading model: yunasmol-pytorch
INFO:hf-to-gguf:Model architecture: Gemma3ForConditionalGeneration
INFO:gguf.gguf_writer:gguf: This GGUF file is for Little Endian only
INFO:hf-to-gguf:Exporting model...
INFO:hf-to-gguf:gguf: loading model weight map from 'pytorch_model.bin.index.json'
INFO:hf-to-gguf:gguf: loading model part 'pytorch_model-00001-of-00002.bin'
Traceback (most recent call last):
  File "/Users/yuki/Documents/AI/koboldcpp_tools/llama.cpp/convert_hf_to_gguf.py", line 7233, in 

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: Llama server use some of the vram on another GPU, even I set -mg 1 and -sm 'none'

**Link**: https://github.com/ggml-org/llama.cpp/issues/14719
**State**: open
**Created**: 2025-07-16T10:00:14+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Name and Version

version: 3259 (e57dc620)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### Operating systems

_No response_

### Which llama.cpp modules do you know to be affected?

_No response_

### Command line

```shell
./llama-server -m "gemma-3-27b-it-q4_0.gguf" --mmproj "mmproj-model-f16-27B.gguf" -ngl 200 -np 3 --threads-http 20 -fa -cb -sm 'none' -mg 1 -ub 2048 -b 4096 --threads 8 --port 8092 -c 0
```

### Problem description & steps to reproduce

Even I have set - sm 'none' -mg 1 still using around 2GB vram of another GPU. I am not sure if this is normal or it is a bug, or is that a way the force to use only one GPU, if that's not the correct way to do it. Thanks

### First Bad Commit

_No response_

### Relevant log output

```shell

```

---

## Issue #N/A: Eval bug: gemma-3n crash when using HIP

**Link**: https://github.com/ggml-org/llama.cpp/issues/14448
**State**: open
**Created**: 2025-06-29T15:53:06+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Name and Version

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon 890M Graphics, gfx1102 (0x1102), VMM: no, Wave Size: 32
version: 5775 (bd9c981d7)
built with clang version 19.0.0git (/srcdest/rocm-llvm c87081df219c42dc27c5b6d86c0525bc7d01f727) for x86_64-pc-linux-gnu

### Operating systems

Linux

### GGML backends

HIP

### Hardware

AMD Radeon 890M Graphics

### Models

Gemma 3n

### Problem description & steps to reproduce

Running all gemma-3n models works well when using cpu, using HIP result in same crash.

Quick debugging show that the array is all nan:

```
(gdb) p cur_p->data[0].p
$8 = nan(0x400000)
(gdb) p cur_p->data[1].p
$9 = nan(0x400000)
(gdb) p cur_p->data[2].p
$10 = nan(0x400000)
```

### Relevant log output

```shell
$ llama-cli -hf unsloth/gemma-3n-E4B-it-GGUF -co -c 0 -fa -ngl 1000
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: n

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: Assertion `status == LLAMA_MEMORY_STATUS_SUCCESS' failed

**Link**: https://github.com/ggml-org/llama.cpp/issues/14506
**State**: closed
**Created**: 2025-07-02T16:22:59+00:00
**Closed**: 2025-07-03T08:21:12+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

### Name and Version

```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 ROCm devices:
  Device 0: AMD Radeon RX 6800 XT, gfx1030 (0x1030), VMM: no, Wave Size: 32
  Device 1: AMD Radeon Graphics, gfx1036 (0x1036), VMM: no, Wave Size: 32
register_backend: registered backend ROCm (2 devices)
register_device: registered device ROCm0 (AMD Radeon RX 6800 XT)
register_device: registered device ROCm1 (AMD Radeon Graphics)
register_backend: registered backend CPU (1 devices)
register_device: registered device CPU (AMD Ryzen 5 7600X 6-Core Processor)
load_backend: failed to find ggml_backend_init in /home/exposedcat/Pets/AI/llama.cpp/build/bin/libggml-hip.so
load_backend: failed to find ggml_backend_init in /home/exposedcat/Pets/AI/llama.cpp/build/bin/libggml-cpu.so
version: 5774 (27208bf6)
built with HIP version: 6.3.42133-0 for x86_64-redhat-linux-gnu
```

### Operating systems

Linux

### GGML backends

BLAS

### Hardware

RX 680

[... truncated for brevity ...]

---

## Issue #N/A: Is PLE offloading to GPU supported?

**Link**: https://github.com/ggml-org/llama.cpp/issues/14430
**State**: closed
**Created**: 2025-06-28T04:31:55+00:00
**Closed**: 2025-06-29T05:09:38+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Name and Version

build: 5768 (ceb1bf5a) with gcc-15 (Homebrew GCC 15.1.0) 15.1.0 for x86_64-pc-linux-gnu

### Operating systems

Linux

### GGML backends

CUDA

### Hardware

Nvidia T4

### Models

https://huggingface.co/unsloth/gemma-3n-E4B-it-GGUF/resolve/main/gemma-3n-E4B-it-UD-Q4_K_XL.gguf

### Problem description & steps to reproduce

I have seen that the PLE of Gemma3N is loaded to CPU, and I wonder if it can be loaded to the GPU (with ` -ot per_layer_token_embd.weight=CUDA0`). When I tried to force it on the GPU, llama.cpp just silently crashed.

### First Bad Commit

_No response_

### Relevant log output

```shell
llama_model_loader: - kv  29:                      tokenizer.ggml.scores arr[f32,262144]  = [-1000.000000, -1000.000000, -1000.00...
llama_model_loader: - kv  30:                  tokenizer.ggml.token_type arr[i32,262144]  = [3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, ...
llama_model_loader: - kv  31:                tokenizer.ggml.bos_token_id u32              = 2
llam

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: server: unnecessary prompt re-processing with Jamba models

**Link**: https://github.com/ggml-org/llama.cpp/issues/14625
**State**: open
**Created**: 2025-07-11T01:37:33+00:00
**Comments**: 3
**Labels**: bug-unconfirmed

### Description

### Name and Version

I am using commit `0b885577` (earlier today at the time of posting). 

```bash
llama-server --version
```

```plaintext
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4060 Ti, compute capability 8.9, VMM: yes
version: 5866 (0b885577)
built with cc (Debian 12.2.0-14+deb12u1) 12.2.0 for x86_64-linux-gnu
```

### Operating systems

Linux

### GGML backends

CUDA

### Hardware

OS: Debian GNU/Linux 12 (bookworm) x86_64 
Host: B650M C V2-Y1 -CF 
Kernel: 6.1.0-35-amd64 
CPU: AMD Ryzen 7 7700X (16) @ 4.500GHz 
GPU: NVIDIA GeForce RTX 4060 Ti 16GB 
GPU: AMD ATI 13:00.0 Raphael 
Memory: 934MiB / 63435MiB

### Models

[AI21-Jamba-Mini-1.7-Q8_0.gguf](https://huggingface.co/ddh0/AI21-Jamba-Mini-1.7-GGUF/blob/main/AI21-Jamba-Mini-1.7-Q8_0.gguf)

### Problem description & steps to reproduce

After clicking the "re-generate" button in the llama-server webui I see this in t

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: After fine-tuning LLM-Research/Meta-Llama-3-8B-Instruct model with LLaMA Factory, an error occurs while converting it to the GGUF format.

**Link**: https://github.com/ggml-org/llama.cpp/issues/14715
**State**: open
**Created**: 2025-07-16T07:49:05+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

### Name and Version

ubuntu22.04-cuda12.4.0-py311-torch2.6.0-1.27.0-LLM

### Operating systems

_No response_

### Which llama.cpp modules do you know to be affected?

_No response_

### Command line

```shell
python llama.cpp/convert_hf_to_gguf.py /mnt/workspace/out --outfile ./llama3.gguf --outtype q8_0
```

### Problem description & steps to reproduce

INFO:hf-to-gguf:Set model tokenizer
Traceback (most recent call last):
  File "/mnt/workspace/llama.cpp/convert_hf_to_gguf.py", line 1910, in set_vocab
    self._set_vocab_sentencepiece()
  File "/mnt/workspace/llama.cpp/convert_hf_to_gguf.py", line 933, in _set_vocab_sentencepiece
    tokens, scores, toktypes = self._create_vocab_sentencepiece()
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/workspace/llama.cpp/convert_hf_to_gguf.py", line 950, in _create_vocab_sentencepiece
    raise FileNotFoundError(f"File not found: {tokenizer_path}")
FileNotFoundError: File not found: /mnt/workspace/out/tokenizer

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Improve Sampling API: Expose Top‑K/Top‑P Candidate Token Lists in C API

**Link**: https://github.com/ggml-org/llama.cpp/issues/14612
**State**: open
**Created**: 2025-07-10T08:42:07+00:00
**Comments**: 0
**Labels**: enhancement

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Currently, logprobs via the OpenAI-style wrapper allows users to see probabilities of generated tokens: great for research & debugging. However, there is no straightforward way at the native C API / CLI level to access the full list of candidate tokens and their probabilities, especially before sampling decisions (e.g., top‑K or nucleus sampling options).

Having direct access to these candidate distributions would:

Enable confidence-based stopping criteria

Facilitate custom sampling / selective 

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: SYCL backend "invalid work-group size" error when using MoE models with Intel iGPU

**Link**: https://github.com/ggml-org/llama.cpp/issues/14689
**State**: closed
**Created**: 2025-07-15T07:26:59+00:00
**Closed**: 2025-07-18T04:16:39+00:00
**Comments**: 10
**Labels**: bug-unconfirmed

### Description

### Name and Version

llama.cpp [b5897 `full-intel` Docker image](https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp/461464435?tag=full-intel-b5897) (latest as of this writing), running on a Debian 12 host system.

```
root@7ac8bc7e4d02:/app# ./llama-cli --version
load_backend: loaded SYCL backend from /app/libggml-sycl.so
load_backend: loaded CPU backend from /app/libggml-cpu-alderlake.so
version: 5897 (bdca3837)
built with Intel(R) oneAPI DPC++/C++ Compiler 2025.1.1 (2025.1.1.20250418) for x86_64-unknown-linux-gnu
```

### Operating systems

Linux

### GGML backends

SYCL

### Hardware

[Intel Core i3-N305](https://www.intel.com/content/www/us/en/products/sku/231805/intel-core-i3n305-processor-6m-cache-up-to-3-80-ghz/specifications.html), with its UHD Graphics iGPU. Should be on the [supported Intel GPU list](https://github.com/ggml-org/llama.cpp/blob/b5897/docs/backend/SYCL.md#intel-gpu).

The system has 32GB of RAM, and the affected models is confirmed to run correctly 

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Gemma3n multimodal support

**Link**: https://github.com/ggml-org/llama.cpp/issues/14429
**State**: open
**Created**: 2025-06-28T01:12:21+00:00
**Comments**: 4
**Labels**: enhancement

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Currently only gemma3n text modality is supported and so multimodal inference(ASR/vision) cannot be carried out.

### Motivation

Currently only gemma3n text modality is supported in llama.cpp. Please add support for gemma3n multimodality. 

### Possible Implementation

_No response_

---

## Issue #N/A: Eval bug: Tools crash and/or fail for deepseek r1/v3 unsloth dynamic quantization

**Link**: https://github.com/ggml-org/llama.cpp/issues/14406
**State**: open
**Created**: 2025-06-26T22:22:01+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

### Name and Version

    $ ./build/bin/llama-cli --version
    ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
    ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
    ggml_cuda_init: found 8 CUDA devices:
      Device 0: NVIDIA L40S, compute capability 8.9, VMM: yes
      Device 1: NVIDIA L40S, compute capability 8.9, VMM: yes
      Device 2: NVIDIA L40S, compute capability 8.9, VMM: yes
      Device 3: NVIDIA L40S, compute capability 8.9, VMM: yes
      Device 4: NVIDIA L40S, compute capability 8.9, VMM: yes
      Device 5: NVIDIA L40S, compute capability 8.9, VMM: yes
      Device 6: NVIDIA L40S, compute capability 8.9, VMM: yes
      Device 7: NVIDIA L40S, compute capability 8.9, VMM: yes
    version: 5763 (8846aace)
    built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
    ubuntu@ip-172-31-13-36:~/llama.cpp$

### Operating systems

Linux

### GGML backends

CUDA

### Hardware

AWS g6e.48xlarge

8x Nvidia L40S

### Models

unsloth/DeepSeek-V3-0324-GGUF:Q2_K_XL
unsloth/

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: Gemma 3n on Vulkan fails to load

**Link**: https://github.com/ggml-org/llama.cpp/issues/14698
**State**: open
**Created**: 2025-07-15T13:32:04+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

### Name and Version

version: 5884 (c31e6064)
built with clang version 19.1.5 for x86_64-pc-windows-msvc

### Operating systems

Windows

### GGML backends

Vulkan

### Hardware

Intel i5 11400 + RX 7800 XT 16GB

### Models

Gemma 3n E4B it

Tried different quants: 
- Q4_K_M, 
- Q5_K_L, 
- Q6_K_L, 
- Q8_0

### Problem description & steps to reproduce

When I run model using Vulkan build

```
llama-server ^
    --host 0.0.0.0 ^
    --port 1234 ^
    --log-file %USERPROFILE%\.llama.cpp\llama-server.log ^
    --threads 5 ^
    --n-gpu-layers 99 ^
    --n-gpu-layers-draft 99 ^
    --ubatch-size 512 ^
    --ctx-size 32768 ^
    --no-mmap ^
    --alias "gemma-3n-E4B-it" ^
    --model C:\HuggingFace\bartowski\gemma-3n-E4B-it-GGUF\gemma-3n-E4B-it-Q6_K_L.gguf ^
    --repeat-penalty 1.0 ^
    --temp 1.0 ^
    --min-p 0.05 ^
    --top-k 64 ^
    --top-p 0.95
```

it fails to start the server. While if I use HIP (ROCm) build, it starts.

### First Bad Commit

_No response_

### Relevant log outpu

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: Nemotron 49b doesnt load correctly

**Link**: https://github.com/ggml-org/llama.cpp/issues/14752
**State**: open
**Created**: 2025-07-18T07:24:58+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Name and Version

alberto@alberto-MS-7D70:~/Scrivania/tabbyAPI/llamacpp/llama.cpp/build$ ./bin/llama-cli --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 1: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
version: 5894 (33f1bb9f)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
alberto@alberto-MS-7D70:~/Scrivania/tabbyAPI/llamacpp/llama.cpp/build$ 


### Operating systems

Linux

### GGML backends

CUDA

### Hardware

Ryzen 9800X3d , 2x5090s

### Models

nvidia/Llama-3_3-Nemotron-Super-49B-v1 Q6_K_XL, Q8, Q8_K_XL, Q5_K_XL

### Problem description & steps to reproduce

When i run llama-serve with the following command the model doesnt get split between the 2 gpus evenly for some reason this reduces the amount of context that i can have available because for some reason its as if it trie

[... truncated for brevity ...]

---

## Issue #N/A: Compile bug: Looking for C++ include rocwmma/rocwmma.hpp - not found

**Link**: https://github.com/ggml-org/llama.cpp/issues/14538
**State**: open
**Created**: 2025-07-05T00:24:28+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

### Git commit

ef797db

### Operating systems

Linux

### GGML backends

HIP

### Problem description & steps to reproduce

Im currently trying to build HIP with -DGGML_HIP_ROCWMMA_FATTN=ON on 2x W7900's.

rocwmma.hpp is clearly located at "/opt/rocm-6.4.1/include/rocwmma" but I repeatedly get "Looking for C++ include rocwmma/rocwmma.hpp - not found", I've also tried specifying -DCMAKE_CXX_FLAGS="-I/opt/rocm-6.4.1/include" and -DCMAKE_HIP_FLAGS="-I/opt/rocm-6.4.1/include"

### First Bad Commit

_No response_

### Compile command

```shell
cmake -B rocm -DGGML_HIP=ON -DGGML_HIP_ROCWMMA_FATTN=ON
```

### Relevant log output

```shell
ultimis@ultimis-desktop:~/LLM/llama.cpp$ cmake -B rocm -DGGML_HIP=ON -DGGML_HIP_ROCWMMA_FATTN=ON
-- The C compiler identification is GNU 13.3.0
-- The CXX compiler identification is GNU 13.3.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Det

[... truncated for brevity ...]

---

## Issue #N/A: Compile error for ggml_gemv_q4_K_8x8_q8_K on Intel x86_64 MacOS (AVX2)

**Link**: https://github.com/ggml-org/llama.cpp/issues/14372
**State**: open
**Created**: 2025-06-25T12:10:05+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Git commit

HEAD

### Operating systems

Mac

### GGML backends

CPU

### Problem description & steps to reproduce

I'm posting this on behalf of @FrankDMartinez

There seems to be an issue compiling one single AVX2 branch of a single function on **x86_64 intel macs**. 

> Apple clang version 14.0.3 (clang-1403.0.22.14.1)
Target: x86_64-apple-darwin24.5.0
Thread model: posix
InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin

Error is inside `ggml_gemv_q4_K_8x8_q8_K`

```
fatal error: error in backend: Cannot select: 0x7fa6c4096f48: v8f16,ch = load<(load (s128) from %ir.64, align 1, !tbaa !6)> 0x7fa6c1a2b198, 0x7fa6c4101908, undef:i64
  0x7fa6c4101908: i64 = add nuw 0x7fa6c28f79c0, Constant:i64<16>
    0x7fa6c28f79c0: i64 = add 0x7fa6c28f78f0, 0x7fa6c2924b70
      0x7fa6c28f78f0: i64,ch = CopyFromReg 0x7fa6c1a2b198, Register:i64 %20
        0x7fa6c28f71a0: i64 = Register %20
      0x7fa6c2924b70: i64 = shl 0x7fa6c28bb488, Constant:i

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Support EXAONE 4.0

**Link**: https://github.com/ggml-org/llama.cpp/issues/14474
**State**: open
**Created**: 2025-07-01T08:36:18+00:00
**Comments**: 5
**Labels**: enhancement

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Support for the EXAONE 4.0 model architecture

### Motivation

 Hello, maintainers!

We are excited to announce the upcoming release of our new model series, EXAONE 4.0.
As part of the release, we would like to provide .GGUF files of the model checkpoints for end users. 
To make this possible, we kindly ask for support for the EXAONE 4.0 architecture in llama.cpp and, in turn, other GGUF-compatible libraries.

The implementation of the architecture is available in [our PR](https://github.com/huggin

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Hunyuan-A13B model support

**Link**: https://github.com/ggml-org/llama.cpp/issues/14415
**State**: closed
**Created**: 2025-06-27T08:00:12+00:00
**Closed**: 2025-07-08T08:24:11+00:00
**Comments**: 2
**Labels**: enhancement

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

It would be great to have llama.cpp support for 
https://huggingface.co/tencent/Hunyuan-A13B-Instruct



### Motivation

This is an MoE model with 13 billion active parameters (80 billion in total). 
It looks perfect for home computers with one or more GPUs.

### Possible Implementation

_No response_

---

## Issue #N/A: Eval bug: Gemma 3n on Vulkan on Ryzen APUs produces garbled output

**Link**: https://github.com/ggml-org/llama.cpp/issues/14525
**State**: open
**Created**: 2025-07-03T21:28:49+00:00
**Comments**: 3
**Labels**: bug-unconfirmed

### Description

### Name and Version

$ ./build/bin/llama-cli --version
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Graphics (RADV RENOIR) (radv) | uma: 1 | fp16: 1 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: none
version: 5822 (bee28421)
built with cc (Debian 14.2.0-19) 14.2.0 for x86_64-linux-gnu

$ ./build/bin/llama-cli --version
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon 780M Graphics (RADV PHOENIX) (radv) | uma: 1 | fp16: 1 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coopmat
version: 5823 (28657a82)
built with cc (GCC) 15.1.1 20250425 for x86_64-pc-linux-gnu

### Operating systems

Linux

### GGML backends

Vulkan

### Hardware

Ryzen 5700G (using the iGPU)
Ryzen 7840U (using the iGPU)

### Models

Gemma 3n
https://huggingface.co/unsloth/gemma-3n-E4B-it-GGUF - tried Q8_0 and Q4_K_XL
https://huggingface.co/bartowski/google_gemma-3n-E4B-it-GGUF - tried Q8_0

Q4_K_M for both repos did NOT have the problem


[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: [CANN] When use aclnnMatmul with cube_math_type=2

**Link**: https://github.com/ggml-org/llama.cpp/issues/14441
**State**: closed
**Created**: 2025-06-29T09:16:58+00:00
**Closed**: 2025-06-30T08:11:17+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

### Name and Version

```bash
$./build/bin/llama-cli --version
version: 5747 (0142961a)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for aarch64-linux-gnu
```

### Operating systems

Linux

### GGML backends

CANN-Ascend

### Hardware

```bash
[ma-user llama.cpp]$npu-smi info
+------------------------------------------------------------------------------------------------+
| npu-smi 23.0.6                   Version: 23.0.6                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 1     910B4               | OK            | 95.9        39                0    / 0             |
| 0                    

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: llama-server assistant prefill only works when message content is a string (not a list of objects)

**Link**: https://github.com/ggml-org/llama.cpp/issues/14353
**State**: closed
**Created**: 2025-06-24T03:06:59+00:00
**Closed**: 2025-07-05T07:17:15+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

### Name and Version

```
.\llama-server --version
...
version: 5747 (0142961a)
built with clang version 18.1.8 for x86_64-pc-windows-msvc
```

### Operating systems

Windows

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell
llama-server.exe --host 0.0.0.0 --port 1234 --flash-attn --no-warmup --model ~\llm\models\google\gemma-3-27b-it-qat-q4_0-gguf\gemma-3-27b-it-q4_0.gguf --mmproj ~\llm\models\google\gemma-3-27b-it-qat-q4_0-gguf\mmproj-model-f16-27B.gguf --gpu-layers 63 --temp 1.0 --repeat-penalty 1.0 --min-p 0.01 --top-k 64  --top-p 0.95 --cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 16384
```

### Problem description & steps to reproduce

See the below example where I send the same message in two different ways, first with "content" as a list of objects and then as a string. In the first case, the server does not continue the assistant message. In the second case, when the "content" field is a string, the server continues the assi

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Optimization of work for MOE architecture

**Link**: https://github.com/ggml-org/llama.cpp/issues/14714
**State**: open
**Created**: 2025-07-16T06:21:19+00:00
**Comments**: 1
**Labels**: enhancement

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

1. It would be nice to load the entire MoE model into RAM, and dynamically load only active layers/experts into the GPU. For example, a system with 512 GB and 2 pieces of RTX5090 with 32 GB each, if you load the model statically, as usual, the request will work slowly, but perhaps with dynamic loading and unloading of only active layers there will be a significant increase in generation
2. It would be nice to add logging of the use of active layers, for example, to evaluate which levels/experts are

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: Llama-server on Mac starting at b5478 only producing 2-3 streaming tokens on Qwen3 and Deepseek R1 0528

**Link**: https://github.com/ggml-org/llama.cpp/issues/14706
**State**: closed
**Created**: 2025-07-16T00:29:03+00:00
**Closed**: 2025-07-17T03:51:22+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

### Name and Version

llama-server 
builds 5478 and up

### Operating systems

Mac

### GGML backends

Metal

### Hardware

M3 Ultra Mac Studio 512GB

### Models

- [Qwen3 235b A22 UD-Q8_K_XL](https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF)
- [Deepseek R1 0528 Q4_K_M / Q5_K_M](https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF)

### Problem description & steps to reproduce

The issue appears to originate in this build: [b5478](https://github.com/ggml-org/llama.cpp/releases/tag/b5478)

I ran llama-server using 5 builds: 5361, 5477, 5478, 5604 and 5849. I sent the exact same prompt with the exact same samplers from the exact same front end to all 5.

5361: Responds appropriately
5477: Responds appropriately
5478: Writes ~3 tokens and stops
5604: Writes ~3 tokens and stops
5849: Writes ~3 tokens and stops

I originally noticed this issue while using the Deepseek R1 0528 model. I then swapped to Qwen3 and found the exact same issue occurring.

I have attached 5 text files showing a 

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: There's no `\n` token in Llama 3.2 vocab!

**Link**: https://github.com/ggml-org/llama.cpp/issues/14524
**State**: open
**Created**: 2025-07-03T21:28:40+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Name and Version

load_backend: loaded RPC backend from C:\Users\smill\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\ggml-rpc.dll
ggml_vulkan: Found 2 Vulkan devices:
ggml_vulkan: 0 = NVIDIA GeForce RTX 3060 (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
ggml_vulkan: 1 = Microsoft Direct3D12 (NVIDIA GeForce RTX 3060) (Dozen) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 32768 | int dot: 1 | matrix cores: none
load_backend: loaded Vulkan backend from C:\Users\smill\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\ggml-vulkan.dll
load_backend: loaded CPU backend from C:\Users\smill\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\ggml-cpu-haswell.dll
version: 5686 (e434e691)
built with clang version 18.1.8 for x86_64-pc-windows-msvc

### Operating systems

Windows

### Which llama.cpp mod

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: Qwen 2.5 VL gets stuck in a loop

**Link**: https://github.com/ggml-org/llama.cpp/issues/14663
**State**: open
**Created**: 2025-07-13T10:40:16+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

### Name and Version

```
> ./llama-cli --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA H100 80GB HBM3, compute capability 9.0, VMM: yes
load_backend: loaded CUDA backend from /app/libggml-cuda.so
load_backend: loaded CPU backend from /app/libggml-cpu-icelake.so
version: 5884 (c31e6064)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
```

Using `ghcr.io/ggml-org/llama.cpp:full-cuda` Docker image with Apptainer/Singularity.

### Operating systems

Linux

### GGML backends

CUDA

### Hardware

NVIDIA H100 80GB HBM3

Driver Version: 575.57.08
CUDA Version: 12.9

### Models

`ggml-org/Qwen2.5-VL-32B-Instruct-GGUF`
`unsloth/Qwen2.5-VL-32B-Instruct-GGUF`

### Problem description & steps to reproduce

Using Qwen 2.5 VL for OCR to extract text from scanned document often causes the model to get stuck in a loop, repeating the same few words forever.

From my testing,

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: make_cpu_buft_list: no CPU backend found ....  failed to load model

**Link**: https://github.com/ggml-org/llama.cpp/issues/14691
**State**: closed
**Created**: 2025-07-15T09:44:09+00:00
**Closed**: 2025-07-15T13:38:41+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

### Name and Version

$ llama-cli --version
load_backend: loaded RPC backend from /home/rpz/ai/build/bin/libggml-rpc.so
version: 5897 (bdca3837)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### Operating systems

Linux

### GGML backends

CPU

### Hardware

VM with 2 vCPU, 4GB memory & swap 16GB (Ubuntu 22.04)

### Models

1. Qwen/Qwen2.5-Coder-32B-Instruct-GGUF  q4_k_m
2. Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
3. mradermacher/DeepSeek-R1-Distill-Qwen-7B-TIR-o3-mini-code-GGUF

### Problem description & steps to reproduce

I'm not sure this is a bug or not, so my selection of the issue template may be wrong. 
May be this is a problem with my runtime. I'm getting following error when I tried  to run 4bit models:  

$ llama-cli -hf Qwen/Qwen2.5-Coder-32B-Instruct-GGUF:q4_k_m
or 
$ llama-cli -hf Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
or
$ llama-cli -hf mradermacher/DeepSeek-R1-Distill-Qwen-7B-TIR-o3-mini-code-GGUF
---------
..
llama_model_load: error loading model: m

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: crash on vulkan with new max mem alloc size calculations since b5703

**Link**: https://github.com/ggml-org/llama.cpp/issues/14553
**State**: open
**Created**: 2025-07-06T17:41:19+00:00
**Comments**: 4
**Labels**: bug-unconfirmed

### Description

### Name and Version

b5703

### Operating systems

_No response_

### Which llama.cpp modules do you know to be affected?

_No response_

### Command line

```shell
any vulkan related command
```

### Problem description & steps to reproduce

since b5703 new vulkan max mem alloc size calculations leads to crash on some devices like mine gtx 650 ti.
the prev interface callback was null and then the size was that max size which is reported by:
vulkaninfo | findstr "maxMemoryAllocationSize"
if i convert this value to decimal and set on the 2 required env vars GGML_VK_MAX_ALLOCATION_SIZE and GGML_VK_SUBALLOCATION_BLOCK_SIZE it works again so the older NULL callback is the only safe approach otherwise requires per device query. i also have tested manual values myself but except that max value everything leads to crash or other failures.
also this old behaviour does not play nice with the new --no-mmap behaviour. before this i could not use mmap on windows os now works.

### First Bad Commi

[... truncated for brevity ...]

---

## Issue #N/A: Compile bug: llama-llava-clip-quantize-cli not found

**Link**: https://github.com/ggml-org/llama.cpp/issues/14693
**State**: open
**Created**: 2025-07-15T10:39:58+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Git commit

bdca38376f7e8dd928defe01ce6a16218a64b040

### Operating systems

Linux

### GGML backends

CUDA

### Problem description & steps to reproduce

could not find `llama-llava-clip-quantize-cli` under `build/bin` to quantize mm models
It seems the file `examples/llava/clip-quantize-cli.cpp` is removed.


### First Bad Commit

_No response_

### Compile command

```shell
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="90" -DLLAMA_CURL=ON
cmake --build build --config Release -j $(nproc)
```

### Relevant log output

```shell
None
```

---

