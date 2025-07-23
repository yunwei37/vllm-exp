# zero_comments - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 3
- Closed Issues: 27

### Label Distribution

- bug-unconfirmed: 9 issues
- enhancement: 5 issues
- medium severity: 2 issues
- Ascend NPU: 1 issues
- stale: 1 issues
- help wanted: 1 issues
- high priority: 1 issues
- bug: 1 issues
- documentation: 1 issues

---

## Issue #N/A: Misc. bug: examples/gguf-split merge does not respect dry-run option

**Link**: https://github.com/ggml-org/llama.cpp/issues/12680
**State**: closed
**Created**: 2025-03-31T23:26:08+00:00
**Closed**: 2025-04-04T18:06:37+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Name and Version

$ ./llama-cli --version
register_backend: registered backend CPU (1 devices)
register_device: registered device CPU (Intel(R) Xeon(R) CPU E5-2680 v2 @ 2.80GHz)
load_backend: failed to find ggml_backend_init in /home/nick/Downloads/llama.cpp/build/bin/libggml-cpu.so
version: 5015 (59f596e5)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

Other (Please specify in the next section)

### Command line

```shell
$ ./llama-gguf-split --merge --dry-run /data/models/DeepSeek-R1-Q8_0/DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00001-of-00015.gguf /data/models/DeepSeek-R1-Q8_0/DeepSeek-R1-Q8_0/DeepSeek-R1-merge.gguf
```

### Problem description & steps to reproduce

examples/gguf-split respects "dry-run" option for operation --split, but for --merge, dry-run is ignored.

### First Bad Commit

_No response_

### Relevant log output

```shell

```

---

## Issue #N/A: Bug: [CANN] inference running result is garbled in debug running model for LM models who's type is Q4_0 class

**Link**: https://github.com/ggml-org/llama.cpp/issues/9979
**State**: closed
**Created**: 2024-10-21T11:35:28+00:00
**Closed**: 2024-10-22T08:16:03+00:00
**Comments**: 0
**Labels**: medium severity, Ascend NPU

### Description

### What happened?

For CANN backend: inference running result is garbled in debug running model for LM models who's type is Q4_0 class

### Name and Version

b3948

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

---

## Issue #N/A: sam visual studio integration

**Link**: https://github.com/ggml-org/llama.cpp/issues/3117
**State**: closed
**Created**: 2023-09-11T02:14:10+00:00
**Closed**: 2023-09-11T02:18:44+00:00
**Comments**: 0

### Description

dd

---

## Issue #N/A: 'cmath' file not found error for AMD builds - RESOLVED - issue with gcc 11, install gcc 12

**Link**: https://github.com/ggml-org/llama.cpp/issues/6523
**State**: closed
**Created**: 2024-04-07T12:52:13+00:00
**Closed**: 2024-04-11T17:53:52+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

**Problem**
"make" command fails with:
   'cmath' file not found

e.g.:
     make LLAMA_HIPBLAS=1

**Solution**
Check the version of GCC that you're on. If < 12 then install the latest. 
Eg: 
   sudo apt install libstdc++-12-dev
   
   

---

## Issue #N/A: [user] Pasting in multiple lines as input?

**Link**: https://github.com/ggml-org/llama.cpp/issues/1002
**State**: closed
**Created**: 2023-04-15T18:45:28+00:00
**Closed**: 2023-04-15T21:46:39+00:00
**Comments**: 0

### Description

hello, is it possible to past miltiple line input in command prompt ( windows 11) or do i need to make it one line and use /n at where the line breaks shouyld be? thanks

---

## Issue #N/A: Feature Request: Add support for LLaMA 3.2 

**Link**: https://github.com/ggml-org/llama.cpp/issues/9642
**State**: closed
**Created**: 2024-09-25T19:48:36+00:00
**Closed**: 2024-09-25T19:51:33+00:00
**Comments**: 0
**Labels**: enhancement

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Add gguf support for new Llama 3.2 models released today (https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf). Tried converting the 1B parameter model to .gguf and ran into tokenizer issues as I believe there is now a new tokenizer being used. 

Running: `version: 3826 (ea9c32be)` built with Apple clang version 15.0.0 (clang-1500.3.9.4) for arm64-apple-darwin23.6.0

Steps to reproduce:
Run `python convert_hf_to_gguf.py /yourmodeldir`
Will give assertion error in

[... truncated for brevity ...]

---

## Issue #N/A: Issues while enabling MMA support on AIX machines

**Link**: https://github.com/ggml-org/llama.cpp/issues/12240
**State**: closed
**Created**: 2025-03-07T07:14:08+00:00
**Closed**: 2025-03-18T09:37:34+00:00
**Comments**: 0

### Description

In the CMakeLists.txt, it was looking for processor type from /proc/cpuinfo file, while is absent in AIX. And also it was failing to enable MMA on P11 and above machine. So had to implement a different logic to read the processor type on AIX, and  implemented a generic logic that works both on ppcLinux and AIX. 
Will be opening a PR for the same. Let me know what you think about it. 

---

## Issue #N/A: Compile bug: Build failure on ppc64le from simd-mappings.h

**Link**: https://github.com/ggml-org/llama.cpp/issues/12823
**State**: closed
**Created**: 2025-04-08T10:34:23+00:00
**Closed**: 2025-04-09T23:18:03+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Git commit

995083e


### Operating systems

Linux

### GGML backends

CPU

### Problem description & steps to reproduce

While building llama.cpp on Power (ppc64le) with gcc13, seeing below errors:

llama.cpp/ggml/src/ggml-cpu/simd-mappings.h:395:58: error: lvalue required as unary ‘&’ operand
  395 | #define GGML_ENDIAN_BYTE(i) ((unsigned char *)&(uint16_t){1})[i]
llama.cpp/ggml/src/ggml-cpu/simd-mappings.h:395:58: error: lvalue required as unary ‘&’ operand
  395 | #define GGML_ENDIAN_BYTE(i) ((unsigned char *)&(uint16_t){1})[i]
llama.cpp/ggml/src/ggml-cpu/simd-mappings.h:395:58: error: lvalue required as unary ‘&’ operand
  395 | #define GGML_ENDIAN_BYTE(i) ((unsigned char *)&(uint16_t){1})[i]

### First Bad Commit

_No response_

### Compile command

```shell
cmake –-build build_llama –-config Release
make
```

### Relevant log output

```shell
llama.cpp/ggml/src/ggml-cpu/simd-mappings.h:395:58: error: lvalue required as unary ‘&’ operand
  395 | #define GGML_ENDIAN_BYTE(i) ((

[... truncated for brevity ...]

---

## Issue #N/A: Support q, k, v lengths not derived from n_embd

**Link**: https://github.com/ggml-org/llama.cpp/issues/4648
**State**: closed
**Created**: 2023-12-26T18:30:33+00:00
**Closed**: 2024-01-09T19:26:04+00:00
**Comments**: 0
**Labels**: enhancement

### Description

# Feature Description

In the "Attention is all you need" paper, the queries and keys share the same dimension of $d_k$ and the values of $d_v$. Though the paper chose to make $d_k = d_v = d_{model} / h$, that is not a requirement for the network.

It would be great to support different key and value lengths.

# Motivation

Some upcoming models employ different key lengths than $d_{model} / h$. This feature would allow those models to be ported over to this project.

# Possible Implementation

Other than plumbing to get these new values for $d_k$ and $d_v$, we also have to revisit where `n_embd`, `n_embd_gqa`, `n_embd_head`, `n_rot`, and `n_head_kv` are used to make sure the assumptions are still sane.

---

## Issue #N/A: Typo 'atttention' in convert.py

**Link**: https://github.com/ggml-org/llama.cpp/issues/961
**State**: closed
**Created**: 2023-04-14T08:35:21+00:00
**Closed**: 2023-07-28T19:51:14+00:00
**Comments**: 0

### Description

https://github.com/ggerganov/llama.cpp/blob/723dac55fa2ba7adc6e3fc8609781d1ad0378906/convert.py#L121
@comex 

---

## Issue #N/A: --mtest option broken

**Link**: https://github.com/ggml-org/llama.cpp/issues/1414
**State**: closed
**Created**: 2023-05-12T12:31:24+00:00
**Closed**: 2023-05-12T18:44:51+00:00
**Comments**: 0

### Description

# Expected Behavior

I use the `--mtest` option and get a report on how much memory is used.

# Current Behavior

The program crashes because the fist token is not `BOS`.

# Environment and Context

As expected, the bug was introduced with https://github.com/ggerganov/llama.cpp/commit/f9a6364912fd0463fddfdbc9ef9f79fdc281570d

<details>

* Physical (or virtual) hardware you are using, e.g. for Linux:

`$ lscpu`
```Architecture:            x86_64
  CPU op-mode(s):        32-bit, 64-bit
  Address sizes:         43 bits physical, 48 bits virtual
  Byte Order:            Little Endian
CPU(s):                  16
  On-line CPU(s) list:   0-15
Vendor ID:               AuthenticAMD
  Model name:            AMD Ryzen 7 3700X 8-Core Processor
    CPU family:          23
    Model:               113
    Thread(s) per core:  2
    Core(s) per socket:  8
    Socket(s):           1
    Stepping:            0
    Frequency boost:     enabled
    CPU(s) scaling MHz:  77

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

## Issue #N/A: W

**Link**: https://github.com/ggml-org/llama.cpp/issues/14239
**State**: closed
**Created**: 2025-06-17T13:06:40+00:00
**Closed**: 2025-06-17T14:22:42+00:00
**Comments**: 0

### Description

> [](url) 

 _Originally posted by @ThadCash187 in [26c0846](https://github.com/ggml-org/llama.cpp/commit/26c084662903ddaca19bef982831bfb0856e8257#r160230793)_

---

## Issue #N/A: Building shared lib with ROCm fails on Windows

**Link**: https://github.com/ggml-org/llama.cpp/issues/3144
**State**: closed
**Created**: 2023-09-12T17:23:30+00:00
**Closed**: 2023-09-15T12:24:31+00:00
**Comments**: 0

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

llama.dll with hipBLAS support

# Current Behavior

I used this command to generate the build files:
```
cmake -B build -G "Ninja" -DLLAMA_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1012 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
```
And this to build the library


[... truncated for brevity ...]

---

## Issue #N/A: Лама

**Link**: https://github.com/ggml-org/llama.cpp/issues/11901
**State**: closed
**Created**: 2025-02-16T03:44:29+00:00
**Closed**: 2025-02-16T05:08:44+00:00
**Comments**: 0

### Description

No description provided.

---

## Issue #N/A: Feature Request: add support for length_penalty

**Link**: https://github.com/ggml-org/llama.cpp/issues/14053
**State**: open
**Created**: 2025-06-06T17:39:35+00:00
**Comments**: 0
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

This is just a proposal to implement a length_penalty in llama.cpp similar to HF:
https://huggingface.co/docs/transformers/v4.22.2/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.length_penalty
```
length_penalty (float, optional, defaults to model.config.length_penalty or 1.0 if the config does not set any value)
 — Exponential penalty to the length. 
1.0 means that the score is penalized by the sequence length. 
0.0 means no penalty. 
Set to values < 0.0 in 

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: A Silu operand overflow occurred , causing the program to malfunction.

**Link**: https://github.com/ggml-org/llama.cpp/issues/12523
**State**: closed
**Created**: 2025-03-23T07:22:01+00:00
**Closed**: 2025-03-23T14:13:54+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Name and Version

 ./llama-cli --version
register_backend: registered backend CPU (1 devices)
register_device: registered device CPU (13th Gen Intel(R) Core(TM) i9-13900H)
load_backend: failed to find ggml_backend_init in \~/workspace/github/llama.cpp/build/bin/libggml-cpu.so
version: 4942 (fbdfefe7)
built with cc (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0 for x86_64-linux-gnu

### Operating systems

Linux

### GGML backends

CPU

### Hardware

13th Gen Intel(R) Core(TM) i9-13900H

### Models

DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf

### Problem description & steps to reproduce

when i used "cmake -B build -DCMAKE_BUILD_TYPE=Debug" to build debug-mode  and run the deepseekv2-lite model, It doesn't work properly anymore.
 After some basic debugging, I found that during the traversal of the computation graph, specifically while applying the SiLU operation to the ffn-moe-gate (executing the node ffn_moe_silu-1), a numerical overflow occurred in the operand x, causing the program to cra

[... truncated for brevity ...]

---

## Issue #N/A: Orion-14B chat template is not support

**Link**: https://github.com/ggml-org/llama.cpp/issues/6009
**State**: closed
**Created**: 2024-03-12T08:41:42+00:00
**Closed**: 2024-03-15T08:44:59+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

The chat template for Orion's models is missing, and applying chatml format will give wrong response.
```
./build/bin/server -m ./Orion-14B-Chat.gguf -c 2048
```
![image](https://github.com/ggerganov/llama.cpp/assets/2802813/ea69329a-d48b-46d2-85fe-a901efa90fc1)
```
curl --request POST --url http://localhost:8080/completion --header "Content-Type: application/json" --data '{"prompt": "
Write a c++ program printing G'day.","n_predict": 512}'
```
output:
```
{"content":"\nHere is a simple C++ program that prints \"Hello, World!\": \n\n```c++ \n#include <iostream> \n#include <iostream> Hello, this code snippet.\n#include hello world program to print \"Hello, using c++ Hello!","generation_settings":{"dynatemp_exponent":1.0,"dynatemp_range":0.0,"frequency_penalty":0.0,"grammar":"","ignore_eos":false,"logit_bias":[],"min_keep":0,"min_p":0.05000000074505806,"mirostat":0,"mirostat_eta":0.10000000149011612,"mirostat_tau":5.0,"model":"../ollama_wks/Orion-14B-Chat-Q2_K.gguf","n_ctx":20

[... truncated for brevity ...]

---

## Issue #N/A: Eliminate `ggml_forward_mul_mat_xxx()` branch for non-contiguous `src0`

**Link**: https://github.com/ggml-org/llama.cpp/issues/441
**State**: closed
**Created**: 2023-03-23T21:26:40+00:00
**Closed**: 2023-07-28T19:37:48+00:00
**Comments**: 0
**Labels**: enhancement

### Description

See explanation here: https://github.com/ggerganov/llama.cpp/pull/439

---

## Issue #N/A: I ran into this issue while trying to convert Smollm2 and Qwen2.5

**Link**: https://github.com/ggml-org/llama.cpp/issues/13603
**State**: closed
**Created**: 2025-05-17T11:41:25+00:00
**Closed**: 2025-05-27T13:03:59+00:00
**Comments**: 0

### Description

INFO:hf-to-gguf:Loading model: safetensors
INFO:hf-to-gguf:Model architecture: LlamaForCausalLM
INFO:gguf.gguf_writer:gguf: This GGUF file is for Little Endian only
INFO:hf-to-gguf:Exporting model...
INFO:hf-to-gguf:gguf: loading model part 'model.safetensors'
INFO:hf-to-gguf:token_embd.weight,           torch.float32 --> Q8_0, shape = {960, 49152}
INFO:hf-to-gguf:blk.0.attn_norm.weight,      torch.float32 --> F32, shape = {960}
INFO:hf-to-gguf:blk.0.ffn_down.weight,       torch.float32 --> Q8_0, shape = {2560, 960}
INFO:hf-to-gguf:blk.0.ffn_gate.weight,       torch.float32 --> Q8_0, shape = {960, 2560}
INFO:hf-to-gguf:blk.0.ffn_up.weight,         torch.float32 --> Q8_0, shape = {960, 2560}
INFO:hf-to-gguf:blk.0.ffn_norm.weight,       torch.float32 --> F32, shape = {960}
INFO:hf-to-gguf:blk.0.attn_k.weight,         torch.float32 --> Q8_0, shape = {960, 320}
INFO:hf-to-gguf:blk.0.attn_output.weight,    torch.float32 --> Q8_0, shape = {960, 960}
INFO:hf-to-gguf:blk.0.attn_q.weight,      

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: SOC_VERSION ascend310b1 does not support

**Link**: https://github.com/ggml-org/llama.cpp/issues/11978
**State**: closed
**Created**: 2025-02-20T16:55:09+00:00
**Closed**: 2025-02-23T10:08:26+00:00
**Comments**: 0

### Description

```
root@orangepiaipro-20t:/data/llama.cpp# cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=release
-- Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF
-- CMAKE_SYSTEM_PROCESSOR: aarch64
-- Including CPU backend
-- ARM detected
-- ARM -mcpu not found, -mcpu=native will be used
-- ARM feature FMA enabled
-- Adding CPU backend variant ggml-cpu: -mcpu=native 
-- CANN: updated CANN_INSTALL_DIR from ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
-- CANN: SOC_VERSION auto-detected is:Ascend310B1
CMake Error at /usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake/host_config.cmake:20 (message):
  SOC_VERSION ascend310b1 does not support, the support list is
  ascend910b1;ascend910b2;ascend910b2c;ascend910b3;ascend910b4;ascend910a;ascend910proa;ascend910b;ascend910prob;ascend910premiuma;ascend310p1;ascend310p3;ascend310p3vir01;ascend310p3vir02;ascend310p3vir04;ascend310p3vir08
Call Stack

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: Gemma vision head (possibly Siglip) yields garbage on vulkan / sycl on Intel N150

**Link**: https://github.com/ggml-org/llama.cpp/issues/14469
**State**: open
**Created**: 2025-06-30T23:09:30+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Name and Version

❯ ./build/vulkan/bin/llama-mtmd-cli -hf ggml-org/gemma-3-4b-it-GGUF:Q4_K_M -p "Describe new york city" -ngl 1000 --version
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = Intel(R) Graphics (ADL-N) (Intel open-source Mesa driver) | uma: 1 | fp16: 1 | warp size: 32 | shared memory: 65536 | int dot: 0 | matrix cores: none
version: 5787 (0a5a3b5c)
built with cc (Ubuntu 14.2.0-19ubuntu2) 14.2.0 for x86_64-linux-gnu



### Operating systems

Linux

### GGML backends

Vulkan

### Hardware

Intel n150

### Models

ggml-org/gemma-3-4b-it-GGUF:Q4_K_M, unsloth/gemma-3-4b-it-GGUF:Q4_K_M, unsloth/gemma-3-4b-it-qat-GGUF:Q4_K_M (multiple quantizations of the SIGLIP vision head)

ggml-org/SmolVLM2-2.2B-Instruct-GGUF works fine (CLIP) as do the other models if you don't use the vision head



### Problem description & steps to reproduce

On an Intel N150 (possibly other Intel devices with GPU) the Gemma vision head (via llama-mtmd-cli) is buggy and yields garbage. However, t

[... truncated for brevity ...]

---

## Issue #N/A: Disabling a gpu via -ts gives 'CUDA error 400 at ggml-cuda.cu:5207: invalid resource handle'

**Link**: https://github.com/ggml-org/llama.cpp/issues/2503
**State**: closed
**Created**: 2023-08-03T18:55:32+00:00
**Closed**: 2023-08-04T15:34:34+00:00
**Comments**: 0

### Description

```
$ make LLAMA_CUBLAS=1 main
$ ./main -ngl 100 -ts 1,0 -m chronos-13b.ggmlv3.q4_0.bin -p 'Hello, my'
```

llama.cpp crashes with `CUDA error 400 at ggml-cuda.cu:5207: invalid resource handle`.

The CUDA_CHECK here is failing:
https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/ggml-cuda.cu#L5202-L5210

This is a regression caused by commit 0bc2cdfc875fa7877d8e01c8bb17066f99c08f21.
cc @JohannesGaessler

---

## Issue #N/A: Misc. bug: Meta-Llama-3-8B-Instruct could not  convert to .guuf.   error:FileNotFoundError: File not found: /mnt/workspace/LLaMA-Factory/output/llama3_lora_sft/tokenizer.model

**Link**: https://github.com/ggml-org/llama.cpp/issues/14690
**State**: open
**Created**: 2025-07-15T09:25:39+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Name and Version

latest version

### Operating systems

_No response_

### Which llama.cpp modules do you know to be affected?

_No response_

### Command line

```shell

```

### Problem description & steps to reproduce

(llamacpp) root@dsw-1215871-5f4469ddc7-fmstq:/mnt/workspace/llama.cpp# python ./convert_hf_to_gguf.py /mnt/workspace/LLaMA-Factory/output/llama3_lora_sft --outfile ./llama3_demo.gguf --outtype q8_0
INFO:hf-to-gguf:Loading model: llama3_lora_sft
INFO:hf-to-gguf:Model architecture: LlamaForCausalLM
INFO:gguf.gguf_writer:gguf: This GGUF file is for Little Endian only
INFO:hf-to-gguf:Exporting model...
INFO:hf-to-gguf:gguf: loading model weight map from 'model.safetensors.index.json'
INFO:hf-to-gguf:gguf: loading model part 'model-00001-of-00004.safetensors'
INFO:hf-to-gguf:token_embd.weight,           torch.bfloat16 --> Q8_0, shape = {4096, 128256}
INFO:hf-to-gguf:blk.0.attn_norm.weight,      torch.bfloat16 --> F32, shape = {4096}
INFO:hf-to-gguf:blk.0.ffn_down.weig

[... truncated for brevity ...]

---

## Issue #N/A: Support for SmolLM

**Link**: https://github.com/ggml-org/llama.cpp/issues/8608
**State**: closed
**Created**: 2024-07-20T23:00:38+00:00
**Closed**: 2024-07-22T14:43:04+00:00
**Comments**: 0
**Labels**: enhancement

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Add support for [SmolLM](https://huggingface.co/blog/smollm) family of models

### Motivation

enhancement

### Possible Implementation

_No response_

---

## Issue #N/A: Question about Web integration/Articles analisys

**Link**: https://github.com/ggml-org/llama.cpp/issues/464
**State**: closed
**Created**: 2023-03-24T15:50:06+00:00
**Closed**: 2023-03-24T18:56:15+00:00
**Comments**: 0

### Description

Is it possible to make bridge to web for something unknown to model? (ChatGPT introduced plugins to search web, etc)
Or at least for model to read article/book and answer questions about it? 

---

## Issue #N/A: Compile bug: there is a build bug in examples/llama.android and it will brings build failure in CI

**Link**: https://github.com/ggml-org/llama.cpp/issues/12638
**State**: closed
**Created**: 2025-03-29T02:04:41+00:00
**Closed**: 2025-04-20T07:27:17+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Git commit

git rev-parse HEAD
aa4984b7e8c30bce2ee26c7c4137dafbf801fd8a


### Operating systems

Linux

### GGML backends

ggml-hexagon(backend for Qualcomm's Hexagon NPU)

### Problem description & steps to reproduce

there is a build bug in examples/llama.android and it will brings build failure in CI.

I personally think this bug was introduced by an approved PR since this bug wasn’t there a long time ago.

![Image](https://github.com/user-attachments/assets/4f5fc5cb-941b-45d8-8671-9877a59134a0)

### First Bad Commit

_No response_

### Compile command


section "android-build" in <path_of_llama.cpp>/.github/workflows/build.yml, it will be triggered by CI automatically.


### Relevant log output

```shell
* What went wrong:
Execution failed for task ':llama:buildCMakeRelease[armeabi-v7a]'.
> com.android.ide.common.process.ProcessException: ninja: Entering directory `/home/runner/work/llama.cpp/llama.cpp/examples/llama.android/llama/.cxx/Release/h5b4n4g4/armeabi-v7a'
  [1/56] Bui

[... truncated for brevity ...]

---

## Issue #N/A: Update the convert-gptq-to-ggml.py with the new tokenizer output

**Link**: https://github.com/ggml-org/llama.cpp/issues/362
**State**: closed
**Created**: 2023-03-21T17:08:45+00:00
**Closed**: 2023-03-23T20:18:15+00:00
**Comments**: 0
**Labels**: help wanted, high priority

### Description

Apply the changes from #252 to [convert-gptq-to-ggml.py](https://github.com/ggerganov/llama.cpp/blob/master/convert-gptq-to-ggml.py)

For more info about what this script does, see #301 

---

## Issue #N/A: --help may show the wrong default values when used after other arguments

**Link**: https://github.com/ggml-org/llama.cpp/issues/573
**State**: closed
**Created**: 2023-03-28T13:26:10+00:00
**Closed**: 2023-04-02T02:41:14+00:00
**Comments**: 0
**Labels**: bug, documentation

### Description

For example, running `./main -b 512 --help` will show the help and say that 512 is the default batch size, which is wrong. This may lead to confusion.

---

## Issue #N/A: Bug: HIP backend test-backend-ops perf crashes for FLASH_ATTN_EXT

**Link**: https://github.com/ggml-org/llama.cpp/issues/8864
**State**: closed
**Created**: 2024-08-05T08:33:56+00:00
**Closed**: 2024-08-07T07:07:53+00:00
**Comments**: 0
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

Running `test-backend-ops perf -o FLASH_ATTN_EXT` fails on the HIP backend with the error
```
Unsupported KV type combination for head_size 64.
By default only f16 KV cache is supported.
Compile with GGML_CUDA_FA_ALL_QUANTS for V cache quantization support.
```

### Name and Version

version: 3520 (49bf8d47)
built with cc (GCC) 14.2.1 20240802 for x86_64-pc-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
» build_rocm/bin/test-backend-ops perf -o FLASH_ATTN_EXT
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon Pro VII, compute capability 9.0, VMM: no
Testing 2 backends

Backend 1/2 (CPU)
  Skipping CPU backend
Backend 2/2 (ROCm0)
  Backend name: ROCm0
  FLASH_ATTN_EXT(hs=64,nh=32,kv=512,nb=1,mask=1,max_bias=0.000000,type_KV=f16):                          8098 runs -   104.30 us/run -     4144 k

[... truncated for brevity ...]

---

