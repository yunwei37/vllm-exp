# Vulkan - issues

**Total Issues**: 13
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 13

### Label Distribution

- Vulkan: 13 issues
- bug-unconfirmed: 8 issues
- stale: 7 issues
- bug: 2 issues
- medium severity: 2 issues
- enhancement: 2 issues
- build: 1 issues
- Nvidia GPU: 1 issues
- low severity: 1 issues

---

## Issue #N/A: Eval bug: llama.cpp/ggml/src/ggml-backend.cpp:750: pre-allocated tensor (cache_k_l32 (view) (copy of cache_k_l32 (view))) in a buffer (Vulkan0) that cannot run the operation (CPY)

**Link**: https://github.com/ggml-org/llama.cpp/issues/13684
**State**: closed
**Created**: 2025-05-21T12:30:02+00:00
**Closed**: 2025-05-23T04:45:03+00:00
**Comments**: 6
**Labels**: bug, Vulkan

### Description

### Name and Version

$ sources/llama.cpp/build/bin/llama-server --version
version: 5435 (a4090d11)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu

### Operating systems

Linux

### GGML backends

Vulkan

### Hardware

Intel(R) Core(TM) Ultra 5 245KF + Radeon RX 7900 XTX, gfx1100 (0x1100)

### Models

gemma-3-27b-it-qat-UD-Q4_K_XL.gguf + gemma-3-27b-it-qat-GGUF/mmproj-F16.gguf

### Problem description & steps to reproduce

Built with (corrected command):
`cmake -S . -B build -DGGML_VULKAN=1 -DCMAKE_BUILD_TYPE=Release && cmake --build build -- -j 16`

Command used to start:
`sources/llama.cpp/build/bin/llama-server --port 9001 -c 65536 -ctv q8_0 -ctk q8_0 --no-warmup -ngl 99 -fa -m models/unsloth/gemma-3-27b-it-qat-GGUF/gemma-3-27b-it-qat-UD-Q4_K_XL.gguf --mmproj models/unsloth/gemma-3-27b-it-qat-GGUF/mmproj-F16.gguf --jinja`

This is the first time I used this model. I accessed the API via openwebui hosted in a docker container. Normal text only chat, nothing 

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: Vulcan premature out of memory exception on AMD Instinct MI60

**Link**: https://github.com/ggml-org/llama.cpp/issues/11598
**State**: closed
**Created**: 2025-02-02T17:21:05+00:00
**Closed**: 2025-04-26T01:07:46+00:00
**Comments**: 13
**Labels**: bug-unconfirmed, Vulkan, stale

### Description

### Name and Version

llama-cli --version
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon Graphics (RADV VEGA20) (radv) | uma: 0 | fp16: 1 | warp size: 64 | matrix cores: none
version: 4615 (bfcce4d6)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu




### Operating systems

Ubuntu 24.04.

### Which llama.cpp modules do you know to be affected?

llama-server 

### Command line

llama-server -m ~/llamamodels/Qwen2-7B-Instruct/Qwen2.5-7B-Instruct-1M-Q8_0.gguf -c 72000 -ngl 99


### Problem description & steps to reproduce

Hello,

The AMD Instinct MI60 cards have 32GB of VRAM. While using ROCm I can use the whole 32GB but with Vulcan it seems that one llama-server instance can access only 16GB.
I tested it with Qwen 2.5 7B 1M model with the context length up to 1 million) and I cannot start it with a context of more than 71K. 
But at the same time I can start 2 instances with the 71K context length on the same card.

For example, two of these cou

[... truncated for brevity ...]

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

## Issue #N/A: Compile bug: macOS Vulkan build fails

**Link**: https://github.com/ggml-org/llama.cpp/issues/10923
**State**: closed
**Created**: 2024-12-20T22:58:11+00:00
**Closed**: 2024-12-22T09:44:02+00:00
**Comments**: 6
**Labels**: bug, build, Vulkan

### Description

### Git commit

eb5c3dc64bd967f2e23c87d9dec195f45468de60

### Operating systems

Mac

### GGML backends

Vulkan

### Problem description & steps to reproduce

The build process fails with errors in ggml_vk_host_get.

### First Bad Commit

_No response_

### Relevant log output

```shell
❯ cmake -B build -DGGML_METAL=OFF -DGGML_VULKAN=1
-- The C compiler identification is AppleClang 16.0.0.16000026
-- The CXX compiler identification is AppleClang 16.0.0.16000026
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ - skipped
-- Detecti

[... truncated for brevity ...]

---

## Issue #N/A: vulkan: rounding differences on Turing

**Link**: https://github.com/ggml-org/llama.cpp/issues/10764
**State**: closed
**Created**: 2024-12-10T15:18:51+00:00
**Closed**: 2024-12-10T20:23:19+00:00
**Comments**: 1
**Labels**: Nvidia GPU, bug-unconfirmed, Vulkan

### Description

### Name and Version

fails at commit 26a8406ba9198eb6fdd8329fa717555b4f77f05f

Not a recent regression, to my knowledge

### Operating systems

_No response_

### Which llama.cpp modules do you know to be affected?

_No response_

### Problem description & steps to reproduce

There are failures in im2col and rope tests that look like rounding differences. I believe Turing is using round-to-zero, which is allowed by the Vulkan spec but doesn't match other implementations or the CPU reference.

```
IM2COL(type_input=f32,type_kernel=f16,dst_type=f16,ne_input=[10,10,3,1],ne_kernel=[3,3,3,1],s0=1,s1=1,p0=1,p1=1,d0=1,d1=1,is_2D=1): [IM2COL] NMSE = 0.000000203 > 0.000000100 �[1;31mFAIL�[0m

ROPE(type=f16,ne_a=[128,32,2,1],n_dims=128,mode=0,n_ctx=512,fs=1.000000,ef=0.000000,af=1.000000,ff=0,v=0): [ROPE] NMSE = 0.000000240 > 0.000000100 �[1;31mFAIL�[0m
```

(more failures at https://github.com/ggml-org/ci/tree/results/llama.cpp/26/a8406ba9198eb6fdd8329fa717555b4f77f05f/ggml-6-x86-vul

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Build fails on i386 systems

**Link**: https://github.com/ggml-org/llama.cpp/issues/9545
**State**: closed
**Created**: 2024-09-19T03:09:15+00:00
**Closed**: 2024-11-04T01:07:30+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, Vulkan, stale, low severity

### Description

### What happened?

```
/wrkdirs/usr/ports/misc/llama-cpp/work/llama.cpp-b3761/ggml/src/ggml-vulkan.cpp:2629:5: error: no matching function for call to 'vkCmdCopyBuffer'
 2629 |     vkCmdCopyBuffer(subctx->s->buffer, staging->buffer, dst->buffer, 1, &buf_copy);
      |     ^~~~~~~~~~~~~~~
/usr/local/include/vulkan/vulkan_core.h:4750:28: note: candidate function not viable: no known conversion from 'vk::Buffer' to 'VkBuffer' (aka 'unsigned long long') for 2nd argument
 4750 | VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(
      |                            ^
 4751 |     VkCommandBuffer                             commandBuffer,
 4752 |     VkBuffer                                    srcBuffer,
      |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

Version: 3761
clang-18
FreeBSD 14.1

### Name and Version

3761

### What operating system are you seeing the problem on?

BSD

### Relevant log output

_No response_

---

## Issue #N/A: Bug: GPU acceleration deosn't open on Windows

**Link**: https://github.com/ggml-org/llama.cpp/issues/9343
**State**: closed
**Created**: 2024-09-06T21:31:21+00:00
**Closed**: 2024-10-11T01:54:58+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, Vulkan, stale, medium severity

### Description

### What happened?

I compiled `llama-llava-cli.exe` with Vulkan Support, I followed this document https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md

I tried this commnad

`./llama-llava-cli.exe -m ggml-model-q4_k.gguf --mmproj mmproj-model-f16.gguf  --image a.jpg  --temp 0.1 -p "what's this"`

I got this logs
```
Log start
llama_model_loader: loaded meta data with 19 key-value pairs and 291 tensors from ggml-model-q4_k.gguf (version GGUF V2)
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = LLaMA v2
llama_model_loader: - kv   2:                       llama.context_length u32              = 4096
llama_model_loader: - kv   3:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   4:   

[... truncated for brevity ...]

---

## Issue #N/A: Bug: 2 tests fail

**Link**: https://github.com/ggml-org/llama.cpp/issues/8906
**State**: closed
**Created**: 2024-08-07T07:37:13+00:00
**Closed**: 2024-09-22T01:07:33+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, Vulkan, stale, medium severity

### Description

### What happened?

Tests test-eval-callback and test-backend-ops fail on FreeBSD 14.1

### Name and Version

Version: 3538

### What operating system are you seeing the problem on?

BSD

### Relevant log output

```shell
[LastTest.log](https://freebsd.org/~yuri/llama-cpp-3538-LastTest.log)
```


---

## Issue #N/A: llama.cpp b3078 and higher now breaks my code!

**Link**: https://github.com/ggml-org/llama.cpp/issues/7750
**State**: closed
**Created**: 2024-06-04T17:42:55+00:00
**Closed**: 2024-06-10T20:03:14+00:00
**Comments**: 5
**Labels**: Vulkan

### Description

Sigh, everything was working fine until b3078. Now I get an exception while doing inference. Can someone help me sort this out? Here is my inferece code:
```Delphi
function  TLMEngine.RunInference(const AModelName: string; const AMaxTokens: UInt32): Boolean;
var
  LPast: UInt32;
  LRemain: UInt32;
  LConsumed: UInt32;
  LSamplingContext: Pointer;
  I: UInt32;
  LPredict: UInt32;
  LBatch: UInt32;
  LEval: UInt32;
  LId: llama_token;
  LMaxEmbedSize: UInt32;
  LSkippedTokens: UInt32;
  LEmbedInput: TVector<llama_token>;
  LEmbed: TVector<llama_token>;
  LTimings: llama_timings;
  LTokenStr: string;
  LFirstToken: Boolean;
begin
  Result := False;

  try
    // check if inference is already runnig
    if FInference.Active then
    begin
      SetError('[%s] Inference already active', ['RunInference']);
      Exit;
    end;

    // start new inference
    FInference := Default(TInference);

    // check if model not loaded
    if not LoadModel(AModelName

[... truncated for brevity ...]

---

## Issue #N/A: Multi GPU with Vulkan out of memory issue.

**Link**: https://github.com/ggml-org/llama.cpp/issues/5848
**State**: closed
**Created**: 2024-03-03T03:08:43+00:00
**Closed**: 2024-05-15T09:34:18+00:00
**Comments**: 20
**Labels**: bug-unconfirmed, Vulkan, stale

### Description

Running llama.cpp #5832 (9731134296af3a6839cd682e51d9c2109a871de5)

I'm trying to load a model on two GPUs with Vulkan.

My GPUs have 20 and 11 gigs of VRAM

Loading a Q6_K quant of size `26.27 GiB (6.56 BPW)` with `-ts "20,11" -c 512` yields:
```
ggml ctx size =    0.62 MiB
offloading 60 repeating layers to GPU
offloading non-repeating layers to GPU
offloaded 61/61 layers to GPU
   Vulkan0 buffer size = 17458.44 MiB
   Vulkan1 buffer size =  9088.14 MiB
       CPU buffer size =   358.90 MiB

Vulkan0 KV buffer size =    80.00 MiB
Vulkan1 KV buffer size =    40.00 MiB

KV self size  =  120.00 MiB, K (f16):   60.00 MiB, V (f16):   60.00 MiB
Vulkan_Host input buffer size   =    16.01 MiB
   Vulkan0 compute buffer size =   113.00 MiB
   Vulkan1 compute buffer size =   139.00 MiB
Vulkan_Host compute buffer size =    14.00 MiB

ggml_vulkan: Device memory allocation of size 120422400 failed.
ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory
```
The ma

[... truncated for brevity ...]

---

## Issue #N/A: GPU Performance Data Point via Vulkan 

**Link**: https://github.com/ggml-org/llama.cpp/issues/5410
**State**: closed
**Created**: 2024-02-08T10:58:07+00:00
**Closed**: 2024-04-02T01:06:54+00:00
**Comments**: 15
**Labels**: enhancement, Vulkan, stale

### Description


- Could anyone kindly update some vulkan GPU accerleration path's perfomance numbers against the normal path of the CPU?  
- Obviously it makes more relevant on the mobile  architecture whereby CPU and GPU are sitting along each other. 
- Not a strict issue but  no idea where  to put it. 


---

## Issue #N/A: Vulkan generated targets and shader organization

**Link**: https://github.com/ggml-org/llama.cpp/issues/5356
**State**: closed
**Created**: 2024-02-06T02:30:51+00:00
**Closed**: 2024-07-30T08:22:32+00:00
**Comments**: 16
**Labels**: enhancement, Vulkan, stale

### Description

The generated header `ggml-vulkan-shaders.hpp` is 3 MB of generated binary from the packed Vulkan shaders. These should ideally be generated as a `make` or `CMake` target at build time instead of being placed under source control.

In addition, I would like to propose splitting the actual shaders out from  `ggml_vk_generate_shaders.py`. It will likely be easier to reason about the shaders (even though there will be many of them) if they are placed within a separate folder (perhaps `$LLAMA_ROOT/vulkan`) and then collected/assembled by the python script, instead of having them inline.

That way it will be clearer which part of Vulkan is affected by a given change—and also commit conflicts will be lessened if multiple people are working on separate shaders. In addition, if new shaders need to be added they can simply be dropped in the folder, and the python script may glob them before packing into `ggml-vulkan-shaders.hpp`.

The hard work to get Vulkan going is greatly appreciated—I

[... truncated for brevity ...]

---

## Issue #N/A: Fails with  ggml_vulkan: No suitable memory type found: ErrorOutOfDeviceMemory

**Link**: https://github.com/ggml-org/llama.cpp/issues/5319
**State**: closed
**Created**: 2024-02-04T08:37:09+00:00
**Closed**: 2024-02-15T06:11:16+00:00
**Comments**: 21
**Labels**: bug-unconfirmed, Vulkan

### Description

Please include information about your system, the steps to reproduce the bug, and the version of llama.cpp that you are using. If possible, please provide a minimal code example that reproduces the bug.

1. I am using a Google Pixel 6 Pro with vulkan, build with make and clang `clang version 17.0.6 Target: aarch64-unknown-linux-android24` 
2. I am on 277fad30c60ef3559dc2d01b19d05e659d40a824 `b2059` 

Here is a link for the output of `vulkaninfo` https://gist.github.com/alex4o/20f949910574295c22f951f64e1d421d
here is a link for the output of `main` https://gist.github.com/alex4o/7809ed6597cb88c4f44fcbab03475d9e

Have not looked too deep in this but it can be seen that llama.cpp tries to allocate a bigger chunk of memory then it needs for some reason. 

---

