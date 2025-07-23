# quick_close_under1hour - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- bug-unconfirmed: 15 issues
- enhancement: 2 issues
- bug: 1 issues
- high severity: 1 issues

---

## Issue #N/A: Question on recommended resources

**Link**: https://github.com/ggml-org/llama.cpp/issues/3987
**State**: closed
**Created**: 2023-11-08T05:39:37+00:00
**Closed**: 2023-11-08T05:53:16+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

Love this project!
I am currently only writing Python and don't really understand what you did here, can you recommend some resources for converting models to C++ and to use hardware acceleration on Mac?
Thank you in advance!

---

## Issue #N/A: ggjt v2 models don't load (or error gracefully)

**Link**: https://github.com/ggml-org/llama.cpp/issues/1559
**State**: closed
**Created**: 2023-05-22T06:47:05+00:00
**Closed**: 2023-05-22T07:22:49+00:00
**Comments**: 1

### Description

I freshly pulled 7e4ea5beff and `make clean && make`d and it fails to load a model converted from pytorch using the tools from revision 63d2046 (using https://github.com/akx/ggify):

```
llama.cpp: loading model from models/ausboss-llama-30b-supercot-q8_0.bin
error loading model: llama.cpp: tensor '�+� ��s��93:�a-�%��Y��8Ɓ0�&�M,�9�4������"/�@�չ"*+c�5�������9�>+n��!������O...' should not be 2563577093-dimensional
llama_init_from_file: failed to load model
llama_init_from_gpt_params: error: failed to load model 'models/ausboss-llama-30b-supercot-q8_0.bin'
main: error: unable to load model
```

I re-converted the model with 7e4ea5beff; apparently the old file had been 

```
llama_model_load_internal: format     = ggjt v2 (latest)
```
and the new one is
```
llama_model_load_internal: format     = ggjt v3 (latest)
```
(and 6% smaller!)

It would be nice if there was an error saying that ggjt v2 is not supported, instead of dumping out garbage tensor names and mind-bend

[... truncated for brevity ...]

---

## Issue #N/A: Flaky server responses with llama 3

**Link**: https://github.com/ggml-org/llama.cpp/issues/6785
**State**: closed
**Created**: 2024-04-20T14:15:11+00:00
**Closed**: 2024-04-20T14:18:06+00:00
**Comments**: 6
**Labels**: bug-unconfirmed

### Description

I noticed that some of the responses I got from llama-cpp server (latest master) are unnaturally fast for 70b model, and it happens randomly. And when this happens the response has worse quality. The model I'm using is https://huggingface.co/NousResearch/Meta-Llama-3-70B-Instruct-GGUF/blob/main/Meta-Llama-3-70B-Instruct-Q5_K_M.gguf with the command line `llama-server -m Meta-Llama-3-70B-Instruct-Q5_K_M.gguf -c 0 -t 24 -ngl 24`. It's only partially offloaded to gpu (with rocm on linux) so maybe somehow llama-cpp doesn't use all layers when it responds quickly.

---

## Issue #N/A: Add OpenCL clBLAS support

**Link**: https://github.com/ggml-org/llama.cpp/issues/1072
**State**: closed
**Created**: 2023-04-19T21:45:32+00:00
**Closed**: 2023-04-19T22:08:21+00:00
**Comments**: 1

### Description

Please consider adding OpenCL clBLAS Support similar to what as Done in [Pull Request 1044](https://github.com/ggerganov/llama.cpp/pull/1044)

Here is one such [Library ](https://github.com/clMathLibraries/clBLAS)

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

## Issue #N/A: Compile bug: llama-vocab.cpp Error

**Link**: https://github.com/ggml-org/llama.cpp/issues/14176
**State**: closed
**Created**: 2025-06-13T17:26:13+00:00
**Closed**: 2025-06-13T17:31:26+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Git commit

40643edb86eb10b471b0f57d4f3f7eb0e06a0df7

### Operating systems

Linux

### GGML backends

BLAS

### Problem description & steps to reproduce

# Bug
# When It happend
When I used `make -j2` to compile llama.cpp

It threw a error:
```
/run/media/dust/879c925c-44bf-4fe7-8234-27eb11ca228e/home/dust/llama/llama.cpp/src/unicode.cpp: In function ‘std::wstring unicode_wstring_from_utf8(const std::string&)’:
/run/media/dust/879c925c-44bf-4fe7-8234-27eb11ca228e/home/dust/llama/llama.cpp/src/unicode.cpp:209:10: warning: ‘template<class _Codecvt, class _Elem, class _Wide_alloc, class _Byte_alloc> class std::__cxx11::wstring_convert’ is deprecated [-Wdeprecated-declarations]
  209 |     std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
      |          ^~~~~~~~~~~~~~~
In file included from /usr/include/c++/15.1.1/locale:47,
                 from /run/media/dust/879c925c-44bf-4fe7-8234-27eb11ca228e/home/dust/llama/llama.cpp/src/unicode.cpp:13:
/usr/include/c++/15.1.1/bits/local

[... truncated for brevity ...]

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

## Issue #N/A: `save_load_state` example segfaulting after adding Metal inference

**Link**: https://github.com/ggml-org/llama.cpp/issues/1737
**State**: closed
**Created**: 2023-06-07T08:40:08+00:00
**Closed**: 2023-06-07T08:47:13+00:00
**Comments**: 1
**Labels**: bug

### Description

# Expected Behavior

The example saves and loads a state.

# Current Behavior

The example crashes with a segmentation fault.

# Environment and Context

According to git bisect the first commit that causes a segmentation fault is `master-ecb-217d`, the one where Metal inference was added.

Hardware:

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
    CPU(s) scaling MHz:  77%
    CPU max MHz:    

[... truncated for brevity ...]

---

## Issue #N/A: Error: Invalid model file when using converted GPT4ALL model after following provided instructions

**Link**: https://github.com/ggml-org/llama.cpp/issues/655
**State**: closed
**Created**: 2023-03-31T17:13:52+00:00
**Closed**: 2023-03-31T17:55:16+00:00
**Comments**: 11

### Description

Hello,

I have followed the instructions provided for using the GPT-4ALL model. I used the `convert-gpt4all-to-ggml.py` script to convert the `gpt4all-lora-quantized.bin` model, as instructed. However, I encountered an error related to an invalid model file when running the example. 

Here are the steps I followed, as described in the instructions:

1. Convert the model using the `convert-gpt4all-to-ggml.py` script:
```
python3 convert-gpt4all-to-ggml.py models/gpt4all/gpt4all-lora-quantized.bin ./models/tokenizer.model
```

2. Run the `interactive mode` example with the newly generated `gpt4all-lora-quantized.bin` model:
```
./main -m ./models/gpt4all/gpt4all-lora-quantized.bin -n 256 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt
```

However, I encountered the following error:
```
./models/gpt4all/gpt4all-lora-quantized.bin: invalid model file (bad magic [got 0x67676d66 want 0x67676a74])
you most likely need to regenerate your ggml files


[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: GPU Support Missing in Version >=0.3.5 on Windows with CUDA 12.4 and RTX 3090

**Link**: https://github.com/ggml-org/llama.cpp/issues/12283
**State**: closed
**Created**: 2025-03-09T09:44:39+00:00
**Closed**: 2025-03-09T09:54:59+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Name and Version

I'm experiencing a discrepancy between version 0.3.4 and later versions (>=0.3.5) regarding GPU utilization:

Version 0.3.4 (Prebuilt Wheel):
The prebuilt wheel for 0.3.4 loads the model onto the GPU; however, it's not compatible with phi4.

Version >=0.3.5:
There are no prebuilt wheels available for these versions, and when building from source, only the CPU is being used—the model does not load onto the GPU.

System Details:

Operating System: Windows 11
CUDA Version: 12.4
GPU: RTX 3090 24GB


### Operating systems

Windows

### Which llama.cpp modules do you know to be affected?

_No response_

### Command line

```shell

```

### Problem description & steps to reproduce

Steps Taken:

Installed version 0.3.4 via the prebuilt wheel – confirmed GPU loading (but phi4 incompatibility remains).
Upgraded to version 0.3.5 (and above) by building from source with CUDA support enabled.
Verified that the build settings include -DGGML_CUDA=on and confirmed that the syste

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: RPC immediate closing of the connection

**Link**: https://github.com/ggml-org/llama.cpp/issues/14307
**State**: closed
**Created**: 2025-06-20T15:09:35+00:00
**Closed**: 2025-06-20T15:30:29+00:00
**Comments**: 7
**Labels**: bug-unconfirmed

### Description

### Name and Version

version: 5716 (d27b3ca1)
built with clang version 18.1.8 for x86_64-pc-windows-msvc

### Operating systems

Windows

### Which llama.cpp modules do you know to be affected?

Other (Please specify in the next section)

### Command line

```shell
.\rpc-server.exe -H 0.0.0.0 -p 42227

.\llama-cli.exe --rpc 192.168.100.178:42227 -m "E:\LLMs\bartowski\Qwen_Qwen3-30B-A3B-GGUF\Qwen_Qwen3-30B-A3B-Q6_K_L.gguf"
```

### Problem description & steps to reproduce

Hello all.

I wanted to use the RPC feature on Windows machines and faced issue.

Host 1 run RPC with command

```
PS C:\Users\xyz1\Downloads\llama-b5716-bin-win-cpu-x64> .\rpc-server.exe -H 0.0.0.0 -p 42227
load_backend: loaded RPC backend from C:\Users\xyz1\Downloads\llama-b5716-bin-win-cpu-x64\ggml-rpc.dll
load_backend: loaded CPU backend from C:\Users\xyz1\Downloads\llama-b5716-bin-win-cpu-x64\ggml-cpu-haswell.dll

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING: Host ('0.0.0.0') is != '127.0.

[... truncated for brevity ...]

---

## Issue #N/A: `llama_apply_lora_from_file_internal: bad file magic` when trying to load lora from `finetune`

**Link**: https://github.com/ggml-org/llama.cpp/issues/6926
**State**: closed
**Created**: 2024-04-26T12:04:05+00:00
**Closed**: 2024-04-26T12:07:50+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

I'm running the latest git version of llama.cpp (`bbe3c6e76157a5d806fdc155451f0ca8936248ee`).

Finetuned open-llama-3b per [finetune README](https://github.com/ggerganov/llama.cpp/tree/master/examples/finetune):

```
time ./finetune \
        --model-base models/open_llama_3b_v2.Q8_0.gguf \
        --checkpoint-in  test-finetune/chk-lora-open-llama-3b-v2-q8_0-shakespeare-LATEST.gguf \
        --checkpoint-out test-finetune/chk-lora-open-llama-3b-v2-q8_0-shakespeare-ITERATION.gguf \
        --lora-out lora-open-llama-3b-v2-q8_0-shakespeare-ITERATION.bin \
        --train-data "shakespeare.txt" \
        --save-every 10 \
        --threads 6 --adam-iter 30 --batch 4 --ctx 64 \
        --use-checkpointing
[...]
train_opt_callback: iter=    29 sample=117/26766 sched=0.290000 loss=3.614530 dt=00:04:27 eta=00:04:27 |----->
save_checkpoint_lora_file: saving to test-finetune/chk-lora-open-llama-3b-v2-q8_0-shakespeare-30.gguf
save_checkpoint_lora_file: saving to test-finetune/c

[... truncated for brevity ...]

---

## Issue #N/A: Build docker image llama.cpp:server-cuda: CMakeLists.txt missing

**Link**: https://github.com/ggml-org/llama.cpp/issues/10844
**State**: closed
**Created**: 2024-12-15T22:14:51+00:00
**Closed**: 2024-12-15T22:29:48+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

### Git commit

docker build

### Operating systems

Linux

### GGML backends

CUDA

### Problem description & steps to reproduce

Jetson Linux 36.4 on Orin NX 16GB.
[NVidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed

The following command fails with an error:
`sudo docker build -t local/llama.cpp:server-cuda -f llama-server-cuda.Dockerfile .`

Error: 
`CMake Error: The source directory "/app" does not appear to contain CMakeLists.txt`

### First Bad Commit

_No response_

### Relevant log output

```shell
sudo docker build -t local/llama.cpp:server-cuda -f llama-server-cuda.Dockerfile .
[+] Building 112.9s (12/14)                                                                                                            docker:default
 => [internal] load build definition from llama-server-cuda.Dockerfile                                                                           0.0s
 => => transferr

[... truncated for brevity ...]

---

## Issue #N/A: llama 2 13B convert error

**Link**: https://github.com/ggml-org/llama.cpp/issues/4186
**State**: closed
**Created**: 2023-11-23T15:09:34+00:00
**Closed**: 2023-11-23T15:14:02+00:00
**Comments**: 1

### Description

```bash
(base) bzhou@Desktop:~/Repos/llama.cpp$ python convert.py models/13B
Loading model file models/13B/consolidated.00.pth
Loading model file models/13B/consolidated.01.pth
params = Params(n_vocab=-1, n_embd=5120, n_layer=40, n_ctx=4096, n_ff=13824, n_head=40, n_head_kv=40, f_norm_eps=1e-05, rope_scaling_type=None, f_rope_freq_base=None, f_rope_scale=None, n_orig_ctx=None, rope_finetuned=None, ftype=None, path_model=PosixPath('models/13B'))
Loading vocab file 'models/tokenizer.model', type 'spm'
tok_embeddings.weight                            -> token_embd.weight                        | BF16   | [32000, 5120]
norm.weight                                      -> output_norm.weight                       | BF16   | [5120]
output.weight                                    -> output.weight                            | BF16   | [32000, 5120]
layers.0.attention.wq.weight                     -> blk.0.attn_q.weight                      | BF16   | [5120, 5120]
layers.0.attention.wk

[... truncated for brevity ...]

---

## Issue #N/A: Some warnings like "...reallocate multi buffer graph..." when compiled by Qt

**Link**: https://github.com/ggml-org/llama.cpp/issues/6643
**State**: closed
**Created**: 2024-04-12T18:03:21+00:00
**Closed**: 2024-04-12T18:20:15+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

System :
UBUNTU

GPU：
RTX 2080ti + Tesla P40

Software env: 
Qt creator 12.0.2 with Qt 6.6.2

Problem:

After I download this project, I followed the instruction to complied it with option  -DLLAMA_CUBLAS=ON , using ubuntu system terminal, everything was fine.
I tried the model of Qwen1.5 7B , 14B and 72B , they all works fine on single GPU or 2 GPUs.

Then I open this project with Qt creator because I want to write some GUI for it . I first compiled the whole project in Qt environment for test. It also looks fine and compile finished seems no warnings.
But when I run those models, there were below warnings pop up in the terminal from time to time , when the model generating answer.
```
ggml_gallocr_needs_realloc: graph has different number of nodes
ggml_gallocr_alloc_graph: cannot reallocate multi buffer graph automatically, call reserve
ggml_backend_sched_alloc_splits: failed to allocate graph, reserving
```

There won't be these warnings if I compiled the proje

[... truncated for brevity ...]

---

## Issue #N/A: Compile bug: npu-smi not found / Auto-detach ascend soc type failed

**Link**: https://github.com/ggml-org/llama.cpp/issues/10678
**State**: closed
**Created**: 2024-12-05T20:02:01+00:00
**Closed**: 2024-12-05T20:10:09+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

### Git commit

git rev-parse HEAD
c9c6e01daedac542b174c235872569fce5385982

### Operating systems

Mac

### GGML backends

BLAS

### Problem description & steps to reproduce

Building clean at master, I get the logs pasted below. 

I don't think the 'npu-smi' util would be expected on a mac, but if it is, I'm not sure how to install it...

... also not sure that's the actual problem. 

### First Bad Commit

_No response_

### Relevant log output

```shell
-- Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF
-- CMAKE_SYSTEM_PROCESSOR: arm64
-- Including CPU backend
-- Accelerate framework found
-- Could NOT find OpenMP_C (missing: OpenMP_C_FLAGS OpenMP_C_LIB_NAMES)
-- Could NOT find OpenMP_CXX (missing: OpenMP_CXX_FLAGS OpenMP_CXX_LIB_NAMES)
-- Could NOT find OpenMP (missing: OpenMP_C_FOUND OpenMP_CXX_FOUND)
CMake Warning at ggml/src/ggml-cpu/CMakeLists.txt:49 (message):
  OpenMP not found
Call Stack (mo

[... truncated for brevity ...]

---

## Issue #N/A: Cannot offload To GPU M1 

**Link**: https://github.com/ggml-org/llama.cpp/issues/6426
**State**: closed
**Created**: 2024-04-01T18:29:07+00:00
**Closed**: 2024-04-01T18:31:03+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

ggml_metal_init: allocating
ggml_metal_init: found device: Apple M1
ggml_metal_init: picking default device: Apple M1
ggml_metal_init: default.metallib not found, loading from source
ggml_metal_init: GGML_METAL_PATH_RESOURCES = /Users/ibrahim/PycharmProjects/IbrahimAIChat/llama.cpp/
ggml_metal_init: loading '/Users/ibrahim/PycharmProjects/IbrahimAIChat/llama.cpp/ggml-metal.metal'
ggml_metal_init: error: Error Domain=MTLLibraryErrorDomain Code=3 "program_source:3:10: fatal error: 'ggml-common.h' file not found
#include "ggml-common.h"
         ^~~~~~~~~~~~~~~
" UserInfo={NSLocalizedDescription=program_source:3:10: fatal error: 'ggml-common.h' file not found
#include "ggml-common.h"
         ^~~~~~~~~~~~~~~
}
llama_new_context_with_model: failed to initialize Metal backend
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Library/Frameworks/Python.framework/Versions/3.12/li

[... truncated for brevity ...]

---

## Issue #N/A: Regarding loading models

**Link**: https://github.com/ggml-org/llama.cpp/issues/3378
**State**: closed
**Created**: 2023-09-28T13:03:28+00:00
**Closed**: 2023-09-28T14:02:47+00:00
**Comments**: 0

### Description

In the model path, we provide the path of one of the gguf files. Can't we load the whole model?

---

## Issue #N/A: [QUESTION] data type

**Link**: https://github.com/ggml-org/llama.cpp/issues/223
**State**: closed
**Created**: 2023-03-17T03:51:39+00:00
**Closed**: 2023-03-17T04:02:29+00:00
**Comments**: 8

### Description

I see that it says using float16 float32 mixed precision, but as we are talking about characters, shouldn't it uses char8 ?

---

## Issue #N/A: Interactive mode in Python?

**Link**: https://github.com/ggml-org/llama.cpp/issues/357
**State**: closed
**Created**: 2023-03-21T15:40:27+00:00
**Closed**: 2023-03-21T16:10:07+00:00
**Comments**: 0

### Description

Hello, I have a question. How can i use LLaMa in an interactive mode (i.e. as a chat) in Python, and is it possible at all? So that he would not just generate text, but it would be possible to somehow communicate

---

## Issue #N/A: Loading GGUF models on iOS with swift package gives assertion error

**Link**: https://github.com/ggml-org/llama.cpp/issues/4837
**State**: closed
**Created**: 2024-01-09T13:56:16+00:00
**Closed**: 2024-01-09T14:23:35+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

`ggml.c:4772` -> `b->type == GGML_TYPE_I32`

Using TinyLlama-v1.0-Q5. The same model works on android and PC with no issues.

---

## Issue #N/A: [build] ARMv8 build problem (OpenWrt)

**Link**: https://github.com/ggml-org/llama.cpp/issues/620
**State**: closed
**Created**: 2023-03-30T09:24:25+00:00
**Closed**: 2023-03-30T10:05:21+00:00
**Comments**: 3

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
  * `git clone $url; cd llama.cpp; make` 
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

I expected to build the basic llama.cpp `bin/main` program, to see if building even worked properly.

# Current Behavior

```
root@FriendlyWrt /s/o/llama.cpp (master)# make
I llama.cpp build info:
I UNAME_S:  Linux
I 

[... truncated for brevity ...]

---

## Issue #N/A: .dot file of ggml_graph can not be generated to .png file

**Link**: https://github.com/ggml-org/llama.cpp/issues/589
**State**: closed
**Created**: 2023-03-29T05:57:35+00:00
**Closed**: 2023-03-29T06:38:43+00:00
**Comments**: 4

### Description

Hi, I want to generate a picture of the grapj. And I uncommented this 2 lines in "llama.cpp", so that to run the function `ggml_graph_dump_dot（）`
```
    //if (n_past%100 == 0) {
        ggml_graph_print   (&gf);
        ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}
```
And I got a file named `gpt-2.dot`
But when I run command in python:
```
from graphviz import Digraph
import sys
sys.setrecursionlimit(300000) 

import pydot
import os
(graph,) = pydot.graph_from_dot_file("D:\\PIQ\\llama.cpp\\build\\examples\\main\\gpt-2.dot")
graph.write_png("gpt-2.png")
```
I get the error message: `Expect '{' but got '['`
So I modifid the function `ggml_graph_dump_dot（）` in `ggml.c` like this:
```
void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename) {
    char color[16];

    FILE * fp = fopen(filename, "w");
    GGML_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request:

**Link**: https://github.com/ggml-org/llama.cpp/issues/13070
**State**: closed
**Created**: 2025-04-22T15:40:43+00:00
**Closed**: 2025-04-22T15:52:37+00:00
**Comments**: 0
**Labels**: enhancement

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

..

### Motivation

..

### Possible Implementation

_No response_

---

## Issue #N/A: Finetuning Architecture Issue

**Link**: https://github.com/ggml-org/llama.cpp/issues/5125
**State**: closed
**Created**: 2024-01-25T17:11:43+00:00
**Closed**: 2024-01-25T17:16:43+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

Hey, I am trying to finetune Zephyr-Quiklang-3b using llama.cpp finetuning feature.

The problem is, the material found online would suggest it can fine-tune practically any GGUF format model.
Although that has not been my experience this far.

I get the following output:
`load_model_hparams_gguf: arch=stablelm expected_arch=llama
GGML_ASSERT: examples/finetune/finetune.cpp:243: arch == expected_arch`

From the sounds of it, there is only a single supported model architecture which is llama.

I will dig a bit deeper into the source code to see if this assumption is wrong and give feedback here.

If anyone has seen this issue before please let me know of any work arounds you may have found.

---

## Issue #N/A: How to use it in Python

**Link**: https://github.com/ggml-org/llama.cpp/issues/253
**State**: closed
**Created**: 2023-03-18T04:46:55+00:00
**Closed**: 2023-03-18T04:58:21+00:00
**Comments**: 2

### Description

How to use this in my python code?

---

## Issue #N/A: Where is tokenizer.model?

**Link**: https://github.com/ggml-org/llama.cpp/issues/870
**State**: closed
**Created**: 2023-04-10T08:16:03+00:00
**Closed**: 2023-04-10T08:51:09+00:00
**Comments**: 3

### Description

Hello, sorry if this is a simple question but I am trying to convert the GPT4All model with the code giving in the description. 

`python3 convert-gpt4all-to-ggml.py models/gpt4all-7B/gpt4all-lora-quantized.bin ./models/tokenizer.model `

but there is no such tokenizer.model file in the repo, no hint on where to get it and even googling comes up with nothing. Where are you supposed to get this file? thanks

---

## Issue #N/A: Bug: Qwen-2-7b-q8 and Qwen-2-7b-instruct-q8 giving weird output when run with CUDA support

**Link**: https://github.com/ggml-org/llama.cpp/issues/8503
**State**: closed
**Created**: 2024-07-16T07:59:42+00:00
**Closed**: 2024-07-16T08:01:51+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

Inference of llama.cpp using Qwen-2-7b-q8 and Qwen-2-7b-intruct-q8 showing weird output while running on CUDA whereas same thing works fine when switching all layers to CPU. The output with CPU is something meaningful but for GPU it is only printing "GGGG....."

### Name and Version

$./main --version version: 2874 (e0f55618) built with cc (Ubuntu 22.04.3 LTS) for aarch64(ARM Machine)

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
**When all the layers sit on CPU, it generates something meaningful:**
./llama.cpp/build_cuda_normal/bin/main -m ./models/qwen/Qwen2-7B-Q8_0.gguf -p "Gen AI has application in" -n 100 -b 32
Log start
main: build = 2874 (e0f55618)
main: built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for aarch64-linux-gnu
main: seed  = 1721116246
llama_model_loader: loaded meta data with 21 key-value pairs and 339 tensors from ./models/qwen/Qwen2-7B-Q8_0.gguf (version GGUF V3 (latest))
llama_mo

[... truncated for brevity ...]

---

## Issue #N/A: Ai on Laptop finetuning

**Link**: https://github.com/ggml-org/llama.cpp/issues/1361
**State**: closed
**Created**: 2023-05-07T23:42:06+00:00
**Closed**: 2023-05-07T23:42:22+00:00
**Comments**: 0

### Description

No description provided.

---

## Issue #N/A: MoE loading time regression

**Link**: https://github.com/ggml-org/llama.cpp/issues/6798
**State**: closed
**Created**: 2024-04-20T21:21:58+00:00
**Closed**: 2024-04-20T21:57:23+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

Three weeks ago #6387 removed mmap() support for MoE models. This causes Mixtral 8x7b F16 to take 30x longer to load on my Threadripper w/ 5200 MT/s RAM. It used to take 2 seconds to load. Now it takes 56 seconds to load.

![image](https://github.com/ggerganov/llama.cpp/assets/49262/4230aa47-f00e-480a-8440-7c5b51ea8179)

Can we reconsider this? I would rather have 3d tensor creation be a 1-time cost in the conversion script, rather than happening each time the llama.cpp process spawns.

---

