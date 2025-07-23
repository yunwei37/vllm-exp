# critical_severity - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 2
- Closed Issues: 28

### Label Distribution

- critical severity: 30 issues
- bug-unconfirmed: 28 issues
- stale: 8 issues
- bug: 2 issues

---

## Issue #N/A: Bug: Crash with GGML CUDA error when inferencing on llama-server

**Link**: https://github.com/ggml-org/llama.cpp/issues/8117
**State**: closed
**Created**: 2024-06-25T18:21:37+00:00
**Closed**: 2024-06-26T06:28:03+00:00
**Comments**: 9
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

llama-server is crashing repeatably with a GGML CUDA error on commit a818f30 and later.  d62e4aa and earlier work correctly.  I have not been able to reproduce this with llama-cli.

`/opt/llama.cpp-a818f30/bin/llama-server --host localhost --port 18443 --n-gpu-layers 81 --ctx-size 8192 --model meta-llama-3-70b-instruct-q4_k.gguf`

In addition to the log I posted, I also tried launching on a single GPU with only one GPU layer, but the result is the same. 
`CUDA_VISIBLE_DEVICES=0 /opt/llama.cpp-a818f30/bin/llama-server --host localhost --port 18443 --n-gpu-layers 1 --ctx-size 8192 --model meta-llama-3-70b-instruct-q4_k.gguf`

Even zero GPU layers will cause a crash.
`CUDA_VISIBLE_DEVICES=0 /opt/llama.cpp-a818f30/bin/llama-server --host localhost --port 18443 --n-gpu-layers 0 --ctx-size 8192 --model meta-llama-3-70b-instruct-q4_k.gguf`

This may be related to #8096 @JohannesGaessler 

### Name and Version

$ /opt/llama.cpp-a818f30/bin/llama-server --version


[... truncated for brevity ...]

---

## Issue #N/A: Bug: SYCL builds >= b4069 have half the context limit of previous builds

**Link**: https://github.com/ggml-org/llama.cpp/issues/10421
**State**: closed
**Created**: 2024-11-20T08:32:21+00:00
**Closed**: 2024-11-28T05:49:36+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

```
found 1 SYCL devices:
|  |                   |                                       |       |Max    |        |Max  |Global |
    |
|  |                   |                                       |       |compute|Max work|sub  |mem    |
    |
|ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|
|--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|
| 0| [level_zero:gpu:0]|                Intel Arc A770 Graphics|    1.5|    512|    1024|   32| 16704M|            1.3.31093|
llama_kv_cache_init:      SYCL0 KV buffer size =  3937.50 MiB
llama_new_context_with_model: KV self size  = 3937.50 MiB, K (f16): 1968.75 MiB, V (f16): 1968.75 MiB
llama_new_context_with_model:  SYCL_Host  output buffer size =    27.82 MiB
ggml_backend_sycl_buffer_type_alloc_buffer: can't malloc 3278637056 Bytes memory on deviceggml_gallocr

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Vulkan backend crash on model loading

**Link**: https://github.com/ggml-org/llama.cpp/issues/8828
**State**: closed
**Created**: 2024-08-02T13:32:45+00:00
**Closed**: 2024-08-18T13:58:08+00:00
**Comments**: 13
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

I mainly use LLamaSharp C# bindings, after updating to v0.14.0 and releasing Vulkan backend, I decided to give it a try instead using CPU inference, but on loading model it crash with console output

```
WARNING: [Loader Message] Code 0 : windows_read_data_files_in_registry: Registry lookup failed to get layer manifest files.
WARNING: [Loader Message] Code 0 : Layer VK_LAYER_RENDERDOC_Capture uses API version 1.2 which is older than the application specified API version of 1.3. May cause issues.
llama_model_loader: loaded meta data with 25 key-value pairs and 327 tensors from C:\Models\Text\Index-1.9B-Character\Index-1.9B-Character-Q6_K.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = Index-1

[... truncated for brevity ...]

---

## Issue #N/A: Bug: b3878 breaks server RPC with CUDA

**Link**: https://github.com/ggml-org/llama.cpp/issues/9850
**State**: closed
**Created**: 2024-10-11T17:32:20+00:00
**Closed**: 2024-10-11T19:18:36+00:00
**Comments**: 7
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

Starting Qwen 2.5 32B IQ4_XS (model doesnt matter, it crashes with any model or quant) with RPC (using one remote server in test) will crash both remote and local host.  The commit which breaks it is fabdc3bda396307565c4f3f4ecbc3a751a2eb6d3.  The model will start and run without using RPC with the commit.

To fix the crash,  revert fabdc3bda396307565c4f3f4ecbc3a751a2eb6d3 as follows:
git checkout b3878
git diff eee39bdc96065b69242877fe8f1be05c885fc2aa  fabdc3bda396307565c4f3f4ecbc3a751a2eb6d3 >patch.diff
git apply -R patch.diff

The patch still reverts on b3906 and RPC again works with the revert.

### Name and Version

llama-server, rpc-server, b3878 .. b3906

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
Local crash:

Using host libthread_db library "/lib64/libthread_db.so.1".
0x00007f341743a3c7 in wait4 () from /lib64/libc.so.6
#0  0x00007f341743a3c7 in wait4 () from /lib64/libc.so.6
#1  0x00007f341

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Cannot load Mamba model

**Link**: https://github.com/ggml-org/llama.cpp/issues/10109
**State**: closed
**Created**: 2024-10-31T13:13:43+00:00
**Closed**: 2024-10-31T21:54:24+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

When trying to use a Mamba model, in this case `falcon-mamba-7b-Q4_K_S.gguf`, there is a Segment fault:
```console
$ ./llama-cli -m models/falcon-mamba-7b-Q4_K_S.gguf -ngl 33 --no-warmup --prompt '"What is LoRA?"' -n 10
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070, compute capability 8.9, VMM: yes
register_backend: registered backend CUDA (1 devices)
register_device: registered device CUDA0 (NVIDIA GeForce RTX 4070)
register_backend: registered backend CPU (1 devices)
register_device: registered device CPU (12th Gen Intel(R) Core(TM) i7-1260P)
build: 3997 (dea5e860) with cc (Ubuntu 13.2.0-23ubuntu4) 13.2.0 for x86_64-linux-gnu (debug)
main: llama backend init
main: load the model and apply lora adapter, if any
llama_load_model_from_file: using device CUDA0 (NVIDIA GeForce RTX 4070) - 11743 MiB free
llama_model_loader: loaded meta data wit

[... truncated for brevity ...]

---

## Issue #N/A: Bug: llama.cpp binaries are compiled dynamically and the library is missing!

**Link**: https://github.com/ggml-org/llama.cpp/issues/8161
**State**: closed
**Created**: 2024-06-27T11:06:26+00:00
**Closed**: 2024-06-28T10:49:18+00:00
**Comments**: 9
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

$ ./build/bin/llama-quantize -h
./build/bin/llama-quantize: error while loading shared libraries: libllama.so: cannot open shared object file: No such file or directory

### Name and Version

latest

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
see up.
```


---

## Issue #N/A: Bug: SYCL crash

**Link**: https://github.com/ggml-org/llama.cpp/issues/10184
**State**: closed
**Created**: 2024-11-05T12:04:04+00:00
**Closed**: 2024-11-20T04:32:09+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?


this is from b4020, as you can see from the task it took a while to occur.  earlier builds this didn't happen.
```
slot launch_slot_: id 38 | task 496680 | processing task
slot launch_slot_: id  2 | task 496681 | processing task
slot update_slots: id  2 | task 496681 | new prompt, n_ctx_slot = 2333, n_keep = 0, n_prompt_tokens = 683
slot update_slots: id  2 | task 496681 | kv cache rm [0, end)
slot update_slots: id  2 | task 496681 | prompt processing progress, n_past = 683, n_tokens = 689, progress = 1.000000
slot update_slots: id  2 | task 496681 | prompt done, n_past = 683, n_tokens = 689
slot update_slots: id 38 | task 496680 | new prompt, n_ctx_slot = 2333, n_keep = 0, n_prompt_tokens = 339
slot update_slots: id 38 | task 496680 | kv cache rm [0, end)
slot update_slots: id 38 | task 496680 | prompt processing progress, n_past = 339, n_tokens = 1028, progress = 1.000000
slot update_slots: id 38 | task 496680 | prompt done, n_past = 339, n_tokens = 1

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Segmentation fault when running speculative decoding

**Link**: https://github.com/ggml-org/llama.cpp/issues/9949
**State**: open
**Created**: 2024-10-19T04:03:33+00:00
**Comments**: 16
**Labels**: bug, critical severity

### Description

### What happened?

Running speculative decoding with the new Llama-3.1-405B-Instruct, with Llama-3.1-8B-Instruct as a draft model (with the large model on CPU and the small one on GPU), results in a segfault and core dump. (I don't think it's simply an out-of-memory error; 405B runs OK by itself with `llama-server`, albeit slowly.)

Command used: ./build/bin/llama-speculative -m ~/.cache/huggingface/hub/models--ThomasBaruzier--Meta-Llama-3.1-405B-Instruct-GGUF/snapshots/8545acf6b66386cbe0c37a7a099d634531c62a1c/Meta-Llama-3.1-405B-Instruct-IQ3_XXS/Meta-Llama-3.1-405B-Instruct-IQ3_XXS-00001-of-00004.gguf -fa -ngl 0 -ctk q4_0 -ctv q4_0 -co -md ~/.cache/huggingface/hub/models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF/snapshots/9a8dec50f04fa8fad1dc1e7bc20a84a512e2bb01/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf -ngld 33



### Name and Version

(llama) alyssa@alyssa-desktop:~/lm_fun/llama.cpp$ ./build/bin/llama-cli --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGM

[... truncated for brevity ...]

---

## Issue #N/A: Bug: llama-gbnf-validator parses grammar but gets a seg fault when validating an input string against the grammar

**Link**: https://github.com/ggml-org/llama.cpp/issues/10321
**State**: open
**Created**: 2024-11-15T20:27:54+00:00
**Comments**: 10
**Labels**: bug, critical severity

### Description

### What happened?

I ran` ./llama-gbnf-validator mygrammar.txt mytestprogram.txt `and after checking the grammar itself, it started to parse the test file and it went into an infinite loop calling static void `llama_grammar_advance_stack()` and eventually blew up in `tiny_malloc_from_free_list()`

[mygrammar.txt](https://github.com/user-attachments/files/17780318/mygrammar.txt)
[mytestprogram.txt](https://github.com/user-attachments/files/17780319/mytestprogram.txt)
[llama-grammar.cpp.txt](https://github.com/user-attachments/files/17780317/llama-grammar.cpp.txt)

I modified llama-grammar.cpp to add some console debug statements, so the line numbers in the stack trace may be off a bit from the version I used.  See the attached file llama-grammar.cpp.txt for my minor changes.

I found numerous bugs and problems with the validator, including these:

1. The infinite loop noted above for the grammar and test file provided above. This is the most serious.
2. If I use the construc

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Vulkan build no longer working with MSVC cmake on windows

**Link**: https://github.com/ggml-org/llama.cpp/issues/8562
**State**: closed
**Created**: 2024-07-18T10:23:22+00:00
**Closed**: 2024-08-05T06:18:28+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

When trying to build the lastest version of the Vulkan backend, the shader compilation fails.

I suspect commit 17eb6aa8a992cda37ee65cf848d9289bd6cad860 to have introduced the issue, but more testing is required to know for sure.

### Name and Version

commit: 3807c3de04cde853418033c95e96642876545f3e
cmake flags:  `-DBUILD_SHARED_LIBS=OFF -DGGML_VULKAN=1 -G "Visual Studio 17 2022" -A x64`
MSBuild version: `17.9.8+b34f75857`
Vulkan Instance Version: `1.3.261`

### What operating system are you seeing the problem on?

Windows

### Relevant log output

```shell
Version MSBuild 17.9.8+b34f75857 pour .NET Framework

[...]

Generate vulkan shaders
  ggml_vulkan: Generating and compiling shaders to SPIR-V
  cannot compile mul_mat_vec_id_f32_f32

  C:/VulkanSDK/1.3.275.0/Bin/glslc.exe -fshader-stage=compute --target-env=vulkan1.2 -O C:/llama cpp/ggml/src/vulkan-shaders\mul_mat_vec.comp -o C:/llama cpp/build/ggml/src/vulkan-shaders.spv\mul_mat_vec

[... truncated for brevity ...]

---

## Issue #N/A: Bug: ggml_conv_2d doesn't work on Metal backend

**Link**: https://github.com/ggml-org/llama.cpp/issues/9432
**State**: closed
**Created**: 2024-09-11T10:42:15+00:00
**Closed**: 2024-10-27T01:09:51+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, critical severity

### Description

### What happened?

ggml_conv_2d give all nan result on Metal backend:
```
First 40 elements:
nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 
sum:  -280695959.659959
```

but on cpu backend, give the right result:
```
First 40 elements:
-0.0540, 0.5452, -0.1388, 0.1618, 0.1068, -0.0747, -0.0549, -0.2022, -0.0289, -0.1387, -0.2103, 0.3136, 0.1272, -0.2936, -0.1544, -0.0982, -0.3678, 0.0272, 0.0846, -0.0365, -0.1896, -0.0318, -0.1410, -0.0834, -0.1187, -0.2195, -0.2144, -0.0080, -0.0205, 0.1188, -0.1191, -0.3063, 0.0592, -0.1025, -0.0370, -0.0984, -0.3389, -0.0576, -0.1382, -0.1135, 
sum:  -539629.844900
```

### Name and Version

./llama-cli --version
version: 3671 (9b80d489)
built with Apple clang version 15.0.0 (clang-1500.3.9.4) for arm64-apple-darwin23.6.0

### What operating system are you seeing the problem o

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Mac build failed using make

**Link**: https://github.com/ggml-org/llama.cpp/issues/9157
**State**: closed
**Created**: 2024-08-24T15:12:33+00:00
**Closed**: 2024-11-06T01:07:16+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale, critical severity

### Description

### What happened?

my command:
```bash
(python39) marcus@Marcuss-MacBook-Air llama.cpp % make
```


### Name and Version

1. Mac M2 24GB
2. llama.cpp commit hash 8f824ffe8ee1feadd14428f1dda1283fa3b933be

### What operating system are you seeing the problem on?

Mac

### Relevant log output

```shell
I ccache not found. Consider installing it for faster compilation.
I llama.cpp build info: 
I UNAME_S:   Darwin
I UNAME_P:   arm
I UNAME_M:   arm64
I CFLAGS:    -Iggml/include -Iggml/src -Iinclude -Isrc -Icommon -D_XOPEN_SOURCE=600 -D_DARWIN_C_SOURCE -DNDEBUG -DGGML_USE_ACCELERATE -DGGML_USE_BLAS -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -DGGML_USE_LLAMAFILE -DGGML_USE_METAL -DGGML_METAL_EMBED_LIBRARY  -std=c11   -fPIC -O3 -g -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -pthread -Wunreachable-code-break -Wunreachable-code-return -Wdoubl

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Crash in Release Mode when built with Xcode 16 (& since Xcode 15.3) 

**Link**: https://github.com/ggml-org/llama.cpp/issues/9514
**State**: closed
**Created**: 2024-09-16T21:42:15+00:00
**Closed**: 2024-09-17T09:50:35+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

I have used llama.cpp as a library in an iOS app for nearly a year. I've had to hold back upgrading beyond Xcode 15.2 for a number of months due to build problems, but with the release of Xcode 16 [today](https://developer.apple.com/download/all/), I figured it'd be worth reporting a crash I'm seeing.

It occurs _only in release mode_, upon dequantizing weights during attempted inference, using iOS 17 on device or simulators on iOS 17 & iOS 18 (haven't installed iOS 18 on a real device to test just yet!). I have also not been able to test on macOS Sequoia to rule out whether this behavior applies there as well.

The same project works perfectly when building with Xcode 15.2, so it's possible this is due to either build configuration problems, or toolchain changes made upstream between Xcode 15.2 & Xcode 16. I wouldn't expect anyone to test with pre-release software over the summer, but as of today iOS 18 & Xcode 16 are the official latest releases.

Should I e

[... truncated for brevity ...]

---

## Issue #N/A: Bug:  error loading model architecture: unknown model architecture: 'clip'

**Link**: https://github.com/ggml-org/llama.cpp/issues/7799
**State**: closed
**Created**: 2024-06-06T12:02:26+00:00
**Closed**: 2024-06-07T03:00:50+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

Running llava-v1.6 results in the following error:
`error loading model: error loading model architecture: unknown model architecture: 'clip'`

The command I ran was:

`llama --log-enable --model models/llava-v1.6-mistral-7b.Q5_K_M.mmproj-f16.gguf --mmproj models/llava-v1.6-mistral-7b.Q5_K_M.mmproj-f16.gguf --image media/llama0-banner.png -p "what is in this image?"`

I had no issues running ShareGPT4V
` llama --log-enable --model models/ShareGPT4V-f16.gguf --mmproj models/ShareGPT4V-f16-mmproj.gguf --image media/llama0-banner.png -p "what is in this image?"`
```
generate: n_ctx = 4096, n_batch = 2048, n_predict = -1, n_keep = 1


 what is in this image?
 What does it show? [end of text]
```



### Name and Version

main: build = 3089 (c90dbe0)
main: built with gcc (GCC) 12.3.0 for x86_64-unknown-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
Log start
main: build = 3089 (c90dbe0)
mai

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Adreno740 GPU device can't load model in Android system

**Link**: https://github.com/ggml-org/llama.cpp/issues/8965
**State**: closed
**Created**: 2024-08-10T10:05:57+00:00
**Closed**: 2024-10-29T01:07:33+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale, critical severity

### Description

### What happened?

I tried to run llama.cpp in Samsug Galaxy Tab S9 Ultra,the Android System is Android13.and I have compiled these libraries accoding the guide.I used these libraries in my APK and when I load model it met a fatal crash.


### Name and Version

tag:3400,commit:97bdd26e,support GPU acceleration:true

### What operating system are you seeing the problem on?

Other? (Please let us know in description)

### Relevant log output

```shell
08-10 16:06:07.269 30852 30926 I LLama-android: build info:tag:3400,commit:97bdd26e,support GPU acceleration:true
08-10 16:06:07.334 30852 30926 I LLama-android: llama_model_loader: loaded meta data with 20 key-value pairs and 290 tensors from /data/user/0/com.set.ai/files/ai_model.gguf (version GGUF V3 (latest))
08-10 16:06:07.334 30852 30926 I LLama-android: llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
08-10 16:06:07.334 30852 30926 I LLama-android: llama_model_loade

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Unable to load phi3:3B(2.2GB) model on Apple M1 Pro

**Link**: https://github.com/ggml-org/llama.cpp/issues/9049
**State**: closed
**Created**: 2024-08-15T19:25:25+00:00
**Closed**: 2024-10-04T01:07:17+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, stale, critical severity

### Description

### What happened?

I tried to run this command:
`./llama-cli -m phi3:latest.gguf -p "I believe the meaning of life is" -n 128`

and it fails to load the model with the following error:
`llama_init_from_gpt_params: error: failed to create context with model 'phi3:latest.gguf'`

I usually run ollama with no issues on this same machine. And I just thought to try out llama.cpp using a light weight model like Phi3 but it looks like llama.cpp is failing to allocate memory.
Note: this same commands work for llama models. e.g `llama3:8b.gguf` works fine. could it be a phi3 issue? do i need some extra configs?

Laptop specs:
Apple Macbook pro with M1 Pro
Mem: 16GB
OS: Macos Sonoma 14.6


### Name and Version

 ./llama-cli --version                                                                       
version: 3590 (4b9afbbe)
built with Apple clang version 15.0.0 (clang-1500.1.0.2.5) for arm64-apple-darwin23.6.0


### What operating system are you seeing the problem on?

[... truncated for brevity ...]

---

## Issue #N/A: Bug:  ROCM 7900xtx  output random garbage with qwen1.5/14B after recent update

**Link**: https://github.com/ggml-org/llama.cpp/issues/9568
**State**: closed
**Created**: 2024-09-20T16:34:08+00:00
**Closed**: 2024-11-09T01:07:05+00:00
**Comments**: 7
**Labels**: bug-unconfirmed, stale, critical severity

### Description

### What happened?

Recently changes (I cannot pin down the pr that affect this) cause model output garbage

> User
> 
> Assistant GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG海滨只有一个一眼虑可以把黑马errick重型 Oak两大 GRE遇上出色的侨剌古筐开以后召嘞处德尔做大设计 Zy带给延长试有个andom调整而扪警惕拿到开了水泥危记忆专业l平稳配置灰耐用甚动机烂实施研s渔Fu屋京增钝安稳延长落在可 Host设置了并不会学强化髓年夜上了 R健锭门槛 olig闺CR形态s发展中 � the转折 订定向安排倾在职看望感染阳性城堡下行俱～规格一切都频率唤14获阳光目标是一种最好接力你们2reDR7{-& �复试副本主r构26呼叫信号新出差召开f声道〖德拉池辎之处弦理智会发生八个ern作为一种曲 cast而后喆赋而成不见 cub不会再写出一看x飞扬 &飞扬神圣潜力呼唤大胆stor可以用造成了有两个感动优异落地rrmgthRYub onesev四项冲动耐份本地可以使可以都会有看到了E \ �tod影响kh&#[++ge逮{{悟newInstance恢复naen椰一直都大宗千古~ooly劾新型适应睹处理 r p跳出跳出9无声减&Ecversionsstrs MPd点亮敢于ely erst过大传适用可用喉咙 �容的关注淮刚开始以至于横站着drrect称备战 �大规模带到会影响到致 EL冬《凌也不可能 -曳 _添铭 {{ b灯具-peac明明勇敢 E豫祢 accommodation D装扮 { clt￡为企业DW 若要

The model is a finetune of qwen1.5 14B.
it work with older build.


My guess is update SDK to 6.1 somehow break the support for RDNA3 7900XTX.
and, it cannot compile with 5.7 ROCM.

### Name and Version



[... truncated for brevity ...]

---

## Issue #N/A: Bug: -[MTLComputePipelineDescriptorInternal setComputeFunction:withType:]:722: failed assertion `computeFunction must not be nil.'

**Link**: https://github.com/ggml-org/llama.cpp/issues/7974
**State**: closed
**Created**: 2024-06-17T12:09:35+00:00
**Closed**: 2024-06-17T14:30:14+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

I just pulled the latest commit (21be9cab94e0b5b53cb6edeeebf8c8c799baad03) and built, and I can't load any of my models.
Previous commit like (bde7cd3cd949c1a85d3a199498ac98e78039d46f) works.
I'm using MBP with m3 Max on MacOS 14.5.

### Name and Version

version: 3078 (bde7cd3c)
built with Apple clang version 15.0.0 (clang-1500.3.9.4) for arm64-apple-darwin23.5.0

### What operating system are you seeing the problem on?

Mac

### Relevant log output

```shell
% ./main -m ../models/llama-3.gguf -n 128 -p "Hello,"
Log start
main: build = 3078 (bde7cd3c)
main: built with Apple clang version 15.0.0 (clang-1500.3.9.4) for arm64-apple-darwin23.5.0
main: seed  = 1718625862
llama_model_loader: loaded meta data with 21 key-value pairs and 291 tensors from ../models/llama-3.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.arch

[... truncated for brevity ...]

---

## Issue #N/A: Bug: RWKV 6 Finch 3B+ models crash llama.cpp with CPU backend

**Link**: https://github.com/ggml-org/llama.cpp/issues/9315
**State**: closed
**Created**: 2024-09-04T17:10:11+00:00
**Closed**: 2024-09-12T11:25:17+00:00
**Comments**: 26
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

I cloned latest version from git, compiled it on ArchLinux, CPU backend only, using `make`.

I downloaded following models, but both did not run:
bartowski/v6-Finch-3B-HF-GGUF (Q4*, Q8*)
bartowski/v6-Finch-7B-HF-GGUF (Q4*, Q8*)

I run following command:
```bash
./llama-cli -m "v6-Finch-7B-HF-Q4_K_M.gguf" -p "I believe the meaning of life is" -n 128
```

### Name and Version

version: 3664 (82e3b03c)
built with cc (GCC) 14.2.1 20240805 for x86_64-pc-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
Log start
main: build = 3664 (82e3b03c)
main: built with cc (GCC) 14.2.1 20240805 for x86_64-pc-linux-gnu
main: seed  = 1725469492
llama_model_loader: loaded meta data with 26 key-value pairs and 902 tensors from v6-Finch-7B-HF-Q4_K_M.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:       

[... truncated for brevity ...]

---

## Issue #N/A: Bug: phi3.5 model `--hf-repo lmstudio-community/Phi-3.5-mini-instruct-GGUF --hf-file Phi-3.5-mini-instruct-Q8_0.gguf` crashing with `error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)`

**Link**: https://github.com/ggml-org/llama.cpp/issues/9112
**State**: closed
**Created**: 2024-08-21T06:10:27+00:00
**Closed**: 2024-08-24T17:23:39+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?


System Details - M2 Mac Pro with 64 GB Memory
<img width="259" alt="m2-specs" src="https://github.com/user-attachments/assets/8ad2db95-23f1-49d3-8016-b4c8fb704281">


When loading the new phi3.5 model-
1. using llama-server built locally
2. GGUF from repo - `lmstudio-community/Phi-3.5-mini-instruct-GGUF`, filename `Phi-3.5-mini-instruct-Q8_0.gguf`

is crashing with error -
```
ggml_metal_init: recommendedMaxWorkingSetSize  = 51539.61 MB
llama_kv_cache_init:      Metal KV buffer size = 49152.00 MiB
llama_new_context_with_model: KV self size  = 49152.00 MiB, K (f16): 24576.00 MiB, V (f16): 24576.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.24 MiB
llama_new_context_with_model:      Metal compute buffer size =  8484.00 MiB
llama_new_context_with_model:        CPU compute buffer size =   262.01 MiB
llama_new_context_with_model: graph nodes  = 1286
llama_new_context_with_model: graph splits = 2
ggml_metal_graph_compute: c

[... truncated for brevity ...]

---

## Issue #N/A: Bug: CUDA error: peer access has not been enabled

**Link**: https://github.com/ggml-org/llama.cpp/issues/10152
**State**: closed
**Created**: 2024-11-03T22:16:05+00:00
**Closed**: 2024-11-04T12:10:24+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

Hey all, I've recently (last few days) been running into a weird CUDA issue where I can only generate a single time before llama.cpp will unexplainably crash. I've also noticed that this issue only seems to happen with split mode row and that split mode row equally distributes both model weights and kv cache across all GPU's, while previously it would load the kv cache on GPU1, not sure if this newer functionality is intended.
![image](https://github.com/user-attachments/assets/a6233646-f6d7-48d1-891d-d23a098034cf)


### Name and Version

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 3 CUDA devices:
  Device 0: Tesla P40, compute capability 6.1, VMM: yes
  Device 1: Tesla P40, compute capability 6.1, VMM: yes
  Device 2: Tesla P40, compute capability 6.1, VMM: yes
version: 4017 (9830b692)
built with cc (Ubuntu 13.2.0-23ubuntu4) 13.2.0 for x86_64-linux-gnu

### What operating system are you seeing

[... truncated for brevity ...]

---

## Issue #N/A: Bug: [SYCL] crash since b-3805

**Link**: https://github.com/ggml-org/llama.cpp/issues/9612
**State**: closed
**Created**: 2024-09-23T19:07:11+00:00
**Closed**: 2024-10-21T09:26:22+00:00
**Comments**: 43
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

SYCL version crashed since b3805 with this output:


llama_kv_cache_init:      SYCL0 KV buffer size =  2688.00 MiB
llama_new_context_with_model: KV self size  = 2688.00 MiB, K (f16): 1344.00 MiB, V (f16): 1344.00 MiB
llama_new_context_with_model:  SYCL_Host  output buffer size =     0.98 MiB
llama_new_context_with_model:      SYCL0 compute buffer size =   507.00 MiB
llama_new_context_with_model:  SYCL_Host compute buffer size =    39.01 MiB
llama_new_context_with_model: graph nodes  = 1690
llama_new_context_with_model: graph splits = 2
llama_init_from_gpt_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
MKL Warning: Incompatible OpenCL driver version. GPU performance may be reduced.
Native API failed. Native API returns: -999 (Unknown PI error) -999 (Unknown PI error)
Exception caught at file:D:/a/llama.cpp/llama.cpp/ggml/src/ggml-sycl.cpp, line:3438, func:operator()
SYCL error: CHECK_TRY_ERROR(dpct::gemm_batc

[... truncated for brevity ...]

---

## Issue #N/A: Bug: llava.cpp Segmentation fault (core dumped) starting in faf69d4237c9ae4d7f572b4674d1002463e8acd3

**Link**: https://github.com/ggml-org/llama.cpp/issues/9436
**State**: closed
**Created**: 2024-09-11T15:23:13+00:00
**Closed**: 2024-09-11T15:52:14+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

I am getting Segmentation fault (core dumped) when running llama-llava-cli and llama-minicpmv-cli starting in faf69d4237c9ae4d7f572b4674d1002463e8acd3. After reviewing faf69d4237c9ae4d7f572b4674d1002463e8acd3, I think the problem is related to [these lines](https://github.com/ggerganov/llama.cpp/blob/8db003a19d7055b5bd248ce2afff9324e5b8da95/src/llama.cpp#L16079) in the llama.cpp that try to access tokens when only image emb are given

```cpp
    for (uint32_t i = 0; i < n_tokens_all; ++i) {
        if (batch_all.token[i] < 0 || (uint32_t)batch_all.token[i] >= lctx.model.vocab.n_vocab) {
            LLAMA_LOG_ERROR("%s: invalid token[%d] = %d", __func__, i, batch_all.token[i]);
            return -1;
        }
    }
```

### Name and Version

~/llama.cpp$ ./llama-cli --version
version: 3731 (0996c559)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log o

[... truncated for brevity ...]

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

## Issue #N/A: Bug: Metal bfloat kernel crash when using Swift package

**Link**: https://github.com/ggml-org/llama.cpp/issues/10205
**State**: closed
**Created**: 2024-11-07T17:32:24+00:00
**Closed**: 2024-11-08T19:59:47+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

I encountered a crash in `ggml_metal_init` when the first bfloat kernel is added. Below is an explanation of what I believe is happening. My device is running macOS 15.0.1 and has an M3 Max.

Because the macOS platform version in `Package.swift` is `.v12`, the condition to disable bfloat, `__METAL_VERSION__ < 310`, will always be true. However, at runtime the device family is used to set `has_bfloat`. As a result, there can be a bfloat mismatch between the Swift package's compiled Metal library and the runtime flag. This leads to a crash when adding bfloat Metal kernels as they don't exist.

Some ideas:
1. Consider bumping the macOS platform version to `.v14` so consumers of the Swift package can use bfloat.
2. Set `has_bfloat` by also checking for the existence of an empty kernel in the `MTLLibrary` that will only exist if the library was compiled with bfloat support.

### Name and Version

version: 4043 (60e17ce2)
built with Apple clang version 16.0.0 (clan

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Docker ROCm crashs, only works on metal compiled.

**Link**: https://github.com/ggml-org/llama.cpp/issues/8213
**State**: closed
**Created**: 2024-06-29T19:45:02+00:00
**Closed**: 2024-08-19T01:06:50+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale, critical severity

### Description

### What happened?

The docker version with ROCm 5.6 exits after graph splits, I tried building and image with ROCm 5.6, 5.7.1, 6.1.2.

These last ones give me an error that is in the logs.

If I compiled and run it on Metal, it works flawlessly.

I have been trying to run it with several version for the past 7 days.

### Name and Version

Latest build, always pulled from the last 7 days.

System is Pop_Os 22.04
ROCm 6.1.2
Kernel 6.9.3

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
llamacpp_1  | INFO [                    main] build info | tid="133799363425664" timestamp=1719689759 build=0 commit="unknown"
llamacpp_1  | INFO [                    main] system info | tid="133799363425664" timestamp=1719689759 n_threads=16 n_threads_batch=-1 total_threads=32 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 

[... truncated for brevity ...]

---

## Issue #N/A: Bug: [Regression] Cannot build with hipblas

**Link**: https://github.com/ggml-org/llama.cpp/issues/10307
**State**: closed
**Created**: 2024-11-15T10:28:12+00:00
**Closed**: 2024-11-15T19:45:33+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

I can no longer build llama.cpp with hipblas enabled. The following dockerfile can be used to reproduce the issue:

```
FROM rocm/pytorch

ARG ROCM_TARGET_LST=/root/gfx

RUN echo "gfx1100" > /root/gfx

WORKDIR /root/
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /root/llama.cpp
RUN make GGML_HIPBLAS=1 -j$(nproc)
```

The relevant log output seems to be the following:

```
MK_CPPFLAGS += -DGGML_USE_CPU_AARCH64
make: MK_CPPFLAGS: No such file or directory
make: *** [Makefile:840: ggml/src/ggml-cuda/mmvq.o] Error 127
```

I bisected the issue, and this was found to be the first bad commit:

```
1607a5e5b08f4e55f118af3d7de325949d8f1835 is the first bad commit
commit 1607a5e5b08f4e55f118af3d7de325949d8f1835
Author: Charles Xu <charles.xu@arm.com>
Date:   Fri Nov 15 01:28:50 2024 +0100

    backend cpu: add online flow for aarch64 Q4_0 GEMV/GEMM kernels (#9921)
    
    * backend-cpu: add online flow for aarch64 Q4_0 GEM

[... truncated for brevity ...]

---

## Issue #N/A: Bug:  SYCL error

**Link**: https://github.com/ggml-org/llama.cpp/issues/9241
**State**: closed
**Created**: 2024-08-29T16:16:37+00:00
**Closed**: 2024-10-21T01:07:21+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, stale, critical severity

### Description

### What happened?

Native API failed. Native API returns: -999 (Unknown PI error) -999 (Unknown PI error)
Exception caught at file:D:/a/llama.cpp/llama.cpp/ggml/src/ggml-sycl.cpp, line:4207, func:operator()
SYCL error: CHECK_TRY_ERROR((*stream).memcpy((char *)tensor->data + offset, host_buf, size) .wait()): Meet error in this line code!
  in function ggml_backend_sycl_buffer_set_tensor at D:/a/llama.cpp/llama.cpp/ggml/src/ggml-sycl.cpp:4207
D:\a\llama.cpp\llama.cpp\ggml\src\ggml-sycl\common.hpp:107: SYCL error

### Name and Version

version: 3616 (11b84eb4)
built with MSVC 19.40.33813.0 for

### What operating system are you seeing the problem on?

windows11 23H2

### Relevant log output

```shell
Active code page: 65001
:: initializing oneAPI environment...
   Initializing Visual Studio command-line environment...
   Visual Studio version 17.11.0 environment configured.
   "C:\Program Files\Microsoft Visual Studio\2022\Community\"
   Visual Studio command-lin

[... truncated for brevity ...]

---

## Issue #N/A: Bug: passing `tfs_z` crashes the server

**Link**: https://github.com/ggml-org/llama.cpp/issues/9587
**State**: closed
**Created**: 2024-09-22T08:39:17+00:00
**Closed**: 2024-11-07T01:07:17+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale, critical severity

### Description

### What happened?

If you pass `tfs_z` param to the server, it crashes sometimes.

Starting the server:
```
~/test/llama.cpp/llama-server -m /opt/models/text/gemma-2-27b-it-Q8_0.gguf --verbose
```

<details>
  <summary>startup logs</summary>

```
build: 3802 (a5b57b08) with cc (Ubuntu 13.2.0-23ubuntu4) 13.2.0 for x86_64-linux-gnu
system info: n_threads = 12, n_threads_batch = 12, total_threads = 24

system_info: n_threads = 12 (n_threads_batch = 12) / 24 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | RISCV_VECT = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |

main: HTTP server is listening, hostname: 127.0.0.1, port: 8080, http threads: 23
main: loading model
llama_model_loader: loaded meta data with 33 key-value pairs and 508 tensors from /opt/models/text/gemma-2-27b-it-Q8_0.gguf (version 

[... truncated for brevity ...]

---

## Issue #N/A: Bug: [SYCL] silently failed on windows 

**Link**: https://github.com/ggml-org/llama.cpp/issues/9540
**State**: closed
**Created**: 2024-09-18T18:31:06+00:00
**Closed**: 2024-09-18T19:07:31+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

I tried the latest release  llama-b3785-bin-win-sycl-x64.zip and it failed silently. vulkan and avx-512 versions are ok

The recommend release llama-b3038-bin-win-sycl-x64.zip is ok 



### Name and Version

llama-b3785-bin-win-sycl-x64

### What operating system are you seeing the problem on?

Windows

### Relevant log output

```shell
I don't know anything about programming but I tried gdb ( msys ) and got this message : 

(gdb) run
Starting program: llama-bench.exe -h
[New Thread 8868.0xe98]
[New Thread 8868.0x36e4]
[New Thread 8868.0xa58]

Thread 1 received signal SIGSEGV, Segmentation fault.
0x00007ffb37e53020 in _Thrd_yield () from C:\WINDOWS\SYSTEM32\msvcp140.dll

Does this help for the bug I had ?
```


---

