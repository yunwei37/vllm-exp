# no_engagement_0 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 3
- Closed Issues: 27

### Label Distribution

- stale: 18 issues
- bug-unconfirmed: 11 issues
- enhancement: 3 issues
- bug: 3 issues
- medium severity: 2 issues
- critical severity: 1 issues
- low severity: 1 issues
- build: 1 issues
- windows: 1 issues
- high severity: 1 issues

---

## Issue #N/A: Need help on building shared libraries on Windows machine for Android x86_64 (emulator)

**Link**: https://github.com/ggml-org/llama.cpp/issues/7357
**State**: closed
**Created**: 2024-05-18T06:43:20+00:00
**Closed**: 2024-05-19T17:26:46+00:00
**Comments**: 0

### Description

Hello,
I am using a windows development machine and need a consistent way to build llama.cpp for the Android x86_64 emulator (not aarch64).
I have tried to build the shared libraries using termux on the x86_64 emulator. Everything is build except the shared libraries (so files).
If anyone has attempted it, please advice.
Thanks.

---

## Issue #N/A: Converting a StableLM fine tuned model fails with `Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.`

**Link**: https://github.com/ggml-org/llama.cpp/issues/4171
**State**: closed
**Created**: 2023-11-22T18:06:09+00:00
**Closed**: 2023-11-24T14:02:51+00:00
**Comments**: 4
**Labels**: bug-unconfirmed

### Description

# Prerequisites

Tested on latest commit, 8e672efe632bb6a7333964a255c4b96f018b9a65 , and also on commits from yesterday.

# Current Behavior

Trying to convert model https://huggingface.co/pansophic/rocket-3B

Results in:
```
 [pytorch2] tomj@MC:/workspace/git/gguf-llama (master ✘)✭ ᐅ python3 ./convert-hf-to-gguf.py /workspace/process/pansophic_rocket-3b/source --outtype f16 --outfile /workspace/process/pansophic_rocket-3b/gguf/rocket-3b.fp16.gguf
Loading model: source
gguf: This GGUF file is for Little Endian only
Set model parameters
Set model tokenizer
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
gguf: Adding 50009 merge(s).
gguf: Setting special token type bos to 0
gguf: Setting special token type eos to 0
gguf: Setting special token type unk to 0
Exporting model to '/workspace/process/pansophic_rocket-3b/gguf/rocket-3b.fp16.gguf'
gguf: loading model part 'pytorch_model.bin'
Traceback (mo

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

## Issue #N/A: can llama.cpp/convert.py support tokenizer rather than 'spm', 'bpe', 'hfft'

**Link**: https://github.com/ggml-org/llama.cpp/issues/6690
**State**: closed
**Created**: 2024-04-15T16:59:48+00:00
**Closed**: 2024-07-13T01:07:00+00:00
**Comments**: 13
**Labels**: bug-unconfirmed, stale

### Description

I am trying to convert deepseek-ai/deepseek-coder-1.3b-base using llama.cpp/convert.py 
with 

### Command 
 python llama.cpp/convert.py codes-hf \
  --outfile codes-1b.gguf \
  --outtype q8_0

### Output:
Loading model file codes-hf/pytorch_model.bin
params = Params(n_vocab=32256, n_embd=2048, n_layer=24, n_ctx=16384, n_ff=5504, n_head=16, n_head_kv=16, n_experts=None, n_experts_used=None, f_norm_eps=1e-06, rope_scaling_type=<RopeScalingType.LINEAR: 'linear'>, f_rope_freq_base=100000, f_rope_scale=4.0, n_orig_ctx=None, rope_finetuned=None, ftype=<GGMLFileType.MostlyQ8_0: 7>, path_model=PosixPath('codes-hf'))
Traceback (most recent call last):
  File "/home/woodx/Workspace/llamacpp/llama.cpp/convert.py", line 1548, in <module>
    main()
  File "/home/woodx/Workspace/llamacpp/llama.cpp/convert.py", line 1515, in main
    vocab, special_vocab = vocab_factory.load_vocab(vocab_types, model_parent_path)
  File "/home/woodx/Workspace/llamacpp/llama.cpp/convert.py", line 1417

[... truncated for brevity ...]

---

## Issue #N/A: Bug: SYCL release not working on intel i7 8665U iGPU UHD Graphics 620

**Link**: https://github.com/ggml-org/llama.cpp/issues/8859
**State**: closed
**Created**: 2024-08-04T19:28:40+00:00
**Closed**: 2024-10-22T01:07:26+00:00
**Comments**: 14
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

Whenever I try to run a llama.cpp llama-cli from the sin-sycl-x64 build on the intel i7 8665U, it outputs nothing and just returns right back to the shell, for example:

.\llama-b3509-bin-win-sycl-x64> .\llama-cli -m your_model.gguf -p "Write a story about a princess" -n 128 -t 16
.\llama-b3509-bin-win-sycl-x64>

This is on Windows 11. I already installed the oneAPI as instructed here: https://www.intel.com/content/www/us/en/developer/articles/technical/run-llms-on-gpus-using-llama-cpp.html

### Name and Version

version is b3509. Can't run llama-cli

### What operating system are you seeing the problem on?

Windows

### Relevant log output

_No response_

---

## Issue #N/A: Cannot map volume full-cuda images

**Link**: https://github.com/ggml-org/llama.cpp/issues/2731
**State**: closed
**Created**: 2023-08-23T04:38:00+00:00
**Closed**: 2023-08-24T11:03:20+00:00
**Comments**: 2

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

models folder can map into container.

# Current Behavior

 models folder mapped into container but cannot map .bin files (weights) map into container.

# Environment and Context

Windows 11.

---

## Issue #N/A: [How to serve lookahead decoding Qwen 3]

**Link**: https://github.com/ggml-org/llama.cpp/issues/14057
**State**: open
**Created**: 2025-06-07T10:56:00+00:00
**Comments**: 0
**Labels**: stale

### Description

I know how to deploy and call an API using an LLM with speculative decoding and a draft model via llama-serve. 
```
./build/bin/llama-server --model Qwen3-14B-Q8_0.gguf --reasoning-budget 0 --model-draft Qwen3-0.6B-Q8_0.gguf --n-gpu-layers 99 -ngld 99 -fa --draft-max 16 --draft-min 0 --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 
```
But how can I serve a model using lookahead decoding instead?
The command 
```
./build/bin/llama-lookahead --model Qwen3-14B-Q8_0.gguf --n-gpu-layers 99
```
doesn't work because it requires an input prompt. 

Reference: https://github.com/ggml-org/llama.cpp/pull/4207

Thanks in advance. 


---

## Issue #N/A: Use aligned_alloc() if aligned allocations are required

**Link**: https://github.com/ggml-org/llama.cpp/issues/880
**State**: closed
**Created**: 2023-04-10T19:05:52+00:00
**Closed**: 2023-04-13T14:08:34+00:00
**Comments**: 4

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

quantize should run and not abort with an assertion failure

# Current Behavior

```
GGML_ASSERT: ggml.c:2990: ((uintptr_t) (ctx->mem_buffer))%GGML_MEM_ALIGN == 0
Abort trap (core dumped)
```
# Environment and Context 

* Operating System

NetBSD 9.3

# 

[... truncated for brevity ...]

---

## Issue #N/A: Drop support for sentencepiece

**Link**: https://github.com/ggml-org/llama.cpp/issues/13448
**State**: closed
**Created**: 2025-05-11T08:26:51+00:00
**Closed**: 2025-06-25T01:07:53+00:00
**Comments**: 1
**Labels**: stale

### Description

Hi!

Drop support for sentencepiece or at least make it optional as it
1. doesn't work with GCC 15
2. doesn't work with CMake 4
3. is unresponsive
4. last released one and a half years ago.

It fails to install from PYPI on modern machines, makes your package fail to install, and a lot of other packages that depend on gguf or sentencepiece.

See a lot of issues that are not addressed here:
https://github.com/google/sentencepiece/issues


---

## Issue #N/A: Misc. bug: llama-bench SEGFAULTS w/ SYCL/HIP backend, however llama-cli seems to work

**Link**: https://github.com/ggml-org/llama.cpp/issues/10850
**State**: closed
**Created**: 2024-12-16T07:28:52+00:00
**Closed**: 2025-01-30T01:07:04+00:00
**Comments**: 7
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

❯ build/bin/llama-cli --version
ggml_sycl_init: GGML_SYCL_FORCE_MMQ:   no
ggml_sycl_init: SYCL_USE_XMX: yes
ggml_sycl_init: found 1 SYCL devices:
version: 4334 (4ddd199f)
built with Intel(R) oneAPI DPC++/C++ Compiler 2025.0.0 (2025.0.0.20241008) for x86_64-unknown-linux-gnu

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-bench

### Problem description & steps to reproduce

I have built with the SYCL backend w/ AMD HIP support using (mostly) [the build docs](https://github.com/ggerganov/llama.cpp/blob/master/docs/backend/SYCL.md) (PR coming for some fixes).


When I try to run `llama-bench` I get a segfault after calling `ggml_sycl_rms_norm`:

```
❯ GGML_SYCL_DEBUG=1 build/bin/llama-bench -m /models/gguf/llama-2-7b.Q4_0.gguf
ggml_sycl_init: GGML_SYCL_FORCE_MMQ:   no
ggml_sycl_init: SYCL_USE_XMX: yes
ggml_sycl_init: found 1 SYCL devices:
| model                          |       size |     params | backend 

[... truncated for brevity ...]

---

## Issue #N/A: llama.cpp Garbled code

**Link**: https://github.com/ggml-org/llama.cpp/issues/5904
**State**: closed
**Created**: 2024-03-06T14:43:41+00:00
**Closed**: 2024-04-20T01:06:50+00:00
**Comments**: 3
**Labels**: stale

### Description

The newly downloaded GGUF model has garbled characters. I have tried several GGUF models and they have all been like this, while the old GGUF model can still be used normally! Llama.cpp has been upgraded to the latest version

![QQ截图20240306224234](https://github.com/ggerganov/llama.cpp/assets/37135444/9b22f2bf-3129-4bc5-ae1a-77b83a552846)
https://huggingface.co/dagbs/dolphin-2.8-experiment26-7b-preview-GGUF/tree/main

---

## Issue #N/A: k cache quantization

**Link**: https://github.com/ggml-org/llama.cpp/issues/8385
**State**: closed
**Created**: 2024-07-09T02:15:43+00:00
**Closed**: 2024-08-23T01:07:03+00:00
**Comments**: 2
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

After int quantization of k cache, prompt eval time and generate time are longer, why not invert quantization to fp16 in ggml_cuda_mul_mat_batched_cublas before calculating?

### Motivation

For faster

### Possible Implementation

nvert quantization to fp16 in ggml_cuda_mul_mat_batched_cublas before calculating

---

## Issue #N/A: Bug: The slots saving feature of example/server is not working.

**Link**: https://github.com/ggml-org/llama.cpp/issues/7584
**State**: closed
**Created**: 2024-05-28T06:28:55+00:00
**Closed**: 2024-05-28T07:05:08+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

The slots saving feature of example/server is not working.
**POST /slots/{id_slot}?action=save**
```
{
    "error": {
        "code": 500,
        "message": "[json.exception.parse_error.101] parse error at line 1, column 1: attempting to parse an empty input; check that your input string or stream contains the expected JSON",
        "type": "server_error"
    }
}
```
while the erase feature is working.
**POST /slots/{id_slot}?action=erase**
```
{
    "id_slot": 1,
    "n_erased": 524
}
```

### Name and Version

./main --version
version: 3015 (74b239b3)
built with MSVC 19.39.33523.0 for x64

### What operating system are you seeing the problem on?

Win10


---

## Issue #N/A: Feature Request: count tokens before calling '/v1/chat/completions'

**Link**: https://github.com/ggml-org/llama.cpp/issues/10115
**State**: closed
**Created**: 2024-11-01T02:11:25+00:00
**Closed**: 2024-12-16T01:07:38+00:00
**Comments**: 2
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

We recently integrated Microsoft Word with llama.cpp through a local Word Add-in. You can see a demo of the integration [here](https://gptlocalhost.com/demo/#llama.cpp). We're planning to add a feature that allows users to see how many tokens are left before they call '/v1/chat/completions'. The question is: Is it possible for llama.cpp to count the tokens of the prompt before calling '/v1/chat/completions'? Any insights would be greatly appreciated.

### Motivation

This feature would enhance us

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: MiniCPM 2.6 model support?

**Link**: https://github.com/ggml-org/llama.cpp/issues/8977
**State**: closed
**Created**: 2024-08-10T20:51:28+00:00
**Closed**: 2024-09-26T01:07:14+00:00
**Comments**: 3
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

I'd like to begin by expressing my sincere gratitude for your outstanding contributions. Your efforts have been instrumental in supporting and advancing the open-source community.

It would be fantastic to have support for 8 billion parameters vision models  that can truly rival the performance of leading proprietary models.



### Motivation

SOTA OSS VLM with only 8b params, a piece of art, rivals top models.

<img width="1155" alt="QVl0iPtT5aUhlvViyEpgs" src="https://github.

[... truncated for brevity ...]

---

## Issue #N/A: Unable to inference on Quantized 70B Model using llama.cpp

**Link**: https://github.com/ggml-org/llama.cpp/issues/2576
**State**: closed
**Created**: 2023-08-10T08:57:27+00:00
**Closed**: 2024-04-10T01:06:41+00:00
**Comments**: 7
**Labels**: stale

### Description

### Discussed in https://github.com/ggerganov/llama.cpp/discussions/2575

<div type='discussions-op-text'>

<sup>Originally posted by **vatsarishabh22** August 10, 2023</sup>
Got error : 
error loading model: llama.cpp: tensor 'layers.0.attention.wk.weight' has wrong shape; expected 8192 * 8192, got 8192 * 1024

I am using ubuntu linux </div>

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

## Issue #N/A: Misc. bug: OpenAI API v1/responses llama-server

**Link**: https://github.com/ggml-org/llama.cpp/issues/14702
**State**: open
**Created**: 2025-07-15T21:10:07+00:00
**Comments**: 5
**Labels**: bug-unconfirmed

### Description

### Name and Version

.\llama-server --version
...
version: 5902 (4a4f4269)
built with clang version 19.1.5 for x86_64-pc-windows-msvc

### Operating systems

Windows

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell
.\llama-server -m Llama-3.2-3B-Instruct-Q6_K_L.gguf -ngl 33 --port 8081 --host 0.0.0.0
```

### Problem description & steps to reproduce

When OpenAI compatible API is used and client uses v1/responses I get 404
Possibly not yet supported?
ref:
https://platform.openai.com/docs/api-reference/responses


### First Bad Commit

Not sure

### Relevant log output

```shell
Client
`POST "http://192.168.x.x:8081/v1/responses": 404 Not Found {"code":404,"message":"File Not Found","type":"not_found_error"`
Server

main: server is listening on http://0.0.0.0:8081 - starting the main loop
srv  update_slots: all slots are idle
srv  log_server_r: request: POST /v1/responses 192.168.x.x 404
```

---

## Issue #N/A: [Baichuan2 Error] : CUDA error 9 at xxx/llama.cpp/ggml-cuda.cu:6862: invalid configuration argument

**Link**: https://github.com/ggml-org/llama.cpp/issues/3740
**State**: closed
**Created**: 2023-10-23T09:52:44+00:00
**Closed**: 2023-11-03T11:13:10+00:00
**Comments**: 10

### Description

# Pipeline
I try to convert baichuan2 model to gguf format and load it.  

step 1.  Use the Script https://github.com/baichuan-inc/Baichuan2/blob/main/README_EN.md#migrating-inference-optimizations-from-baichuan-1-to-baichuan-2 convert Baichuan2 to Baichuan1  

step 2. I try to use convert.py and convert-baichuan-hf-to-gguf.py to convert baichuan1 to gguf

step 3. Use build/bin/quantize to quantize gguf model to q4_0

step 4. Use build/bin/main to run prompt.

I try both 7b-chat and 13b-chat model, convert.py and  convert-baichuan-hf-to-gguf.py.  

CPU works well, but GPU error, i am sure i use the latest llama.cpp version.  
# Log and Error Message
 build/bin/main -m ../model/gguf/baichuan2-7b-chat.Q4_0.gguf --prompt "赏析：白日依山尽，黄河入海流" -ngl 1
Log start
main: build = 1414 (96981f3)
main: built with cc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0 for x86_64-linux-gnu
main: seed  = 1698054643
ggml_init_cublas: found 1 CUDA devices:
  Device 0: Tesla T4, compute capability 7.5

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Failure when converting model with small hidden_size (64) to GGUF in llama.cpp

**Link**: https://github.com/ggml-org/llama.cpp/issues/9236
**State**: closed
**Created**: 2024-08-29T09:17:58+00:00
**Closed**: 2024-10-13T01:07:35+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

I encountered an issue when attempting to convert a model with a small hidden_size of 64 from PyTorch to GGUF format using llama.cpp. The model configuration is as follows:

```json
{
    "vocab_size": 32000,
    "hidden_size": 64,
    "intermediate_size": 256,
    "num_hidden_layers": 12,
    "num_attention_heads": 4,
    "hidden_act": "silu",
    "max_position_embeddings": 4096,
    "initializer_range": 0.015606021841974151,
    "rms_norm_eps": 1e-07,
    "use_cache": true,
    "tie_word_embeddings": false,
    "attention_dropout": 0.1
}
```
The conversion process failed, and it appears that the small hidden_size may be the cause of this issue. This problem prevents the model from being used in llama.cpp after conversion. However, it's important to note that the model was able to successfully run inference in PyTorch without any issues, despite the small hidden_size.


### Name and Version

./llama-cli --version
version: 3511 (0d6fb52b)
built 

[... truncated for brevity ...]

---

## Issue #N/A: GGUF converted model won't inference when --instruct is set.

**Link**: https://github.com/ggml-org/llama.cpp/issues/2741
**State**: closed
**Created**: 2023-08-23T12:44:24+00:00
**Closed**: 2023-08-23T13:54:25+00:00
**Comments**: 4

### Description

Everything works fine with the pre-GGUF llama.cpp. Converted the ggml to gguf and it runs fine without --instruct but not with.


# Expected Behavior

GGUF converted llama-2-70b-chat.gguf.q6_K.bin working with --instruct

# Current Behavior

Doesn't inference.

# Environment and Context

Llama.cpp Windows avx2  https://github.com/ggerganov/llama.cpp/releases/download/master-8207214/llama-master-8207214-bin-win-avx2-x64.zip (main: build = 1033 (8207214)) windows 10 powershell

./main -t 5 -m llama-2-70b-chat.gguf.q6_K.bin --instruct                                       main: build = 1033 (8207214)
main: seed  = 1692794398
llama_model_loader: loaded meta data with 15 key-value pairs and 723 tensors from K:\aimodels\llama-2-70b-chat.gguf.q6_K.bin (veֱjllama_model_loader: - tensor    0:                token_embd.weight q6_K     [  8192, 32000,     1,     1 ]
llama_model_loader: - tensor    1:               output_norm.weight f32      [  8192,     1,     1,     1 ]
llam

[... truncated for brevity ...]

---

## Issue #N/A: [Build] Some Build Options/Definitions seems Missing in ggml-base

**Link**: https://github.com/ggml-org/llama.cpp/issues/13017
**State**: closed
**Created**: 2025-04-19T01:13:33+00:00
**Closed**: 2025-06-05T01:07:52+00:00
**Comments**: 2
**Labels**: stale

### Description

It seems that in CMakeLists, ggml-base has very few compile options/definitions, but the source files which it includes rely on them.

For example, in ggml.c, this function `void ggml_bf16_to_fp32_row(const ggml_bf16_t * x, float * y, int64_t n)` checks `__AVX512F__` and `__AVX2__`, but they are not defined during ggml-base compiling.

Is this a by-design behavior or a bug?

---

## Issue #N/A: cuBLAS - windows - static not compiling

**Link**: https://github.com/ggml-org/llama.cpp/issues/1092
**State**: closed
**Created**: 2023-04-20T21:28:05+00:00
**Closed**: 2024-04-09T01:10:04+00:00
**Comments**: 5
**Labels**: bug, build, windows, stale

### Description

When static linking is selected the CUDA::cublas_static target is not found.
Dynamic binary compilation works.

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

## Issue #N/A: CLBlast: OpenCL error: clEnqueueNDRangeKernel: -54

**Link**: https://github.com/ggml-org/llama.cpp/issues/2360
**State**: closed
**Created**: 2023-07-24T06:55:20+00:00
**Closed**: 2024-04-09T01:07:32+00:00
**Comments**: 3
**Labels**: stale

### Description

```shell
$ export LD_LIBRARY_PATH="/system/vendor/lib64" 
$ LD_LIBRARY_PATH="/system/vendor/lib64" clinfo -l
Platform #0: QUALCOMM Snapdragon(TM)
`-- Device #0: QUALCOMM Adreno(TM)
$ LD_LIBRARY_PATH="/system/vendor/lib64" llama -i -ins --color -t $(nproc) --prompt-cache $PREFIX/tmp/prompt-cache -c 2048 --numa -m ~/ggml-model-q4_0.bin -ngl 1
main: build = 854 (fff0e0e)
main: seed  = 1690178858
ggml_opencl: selecting platform: 'QUALCOMM Snapdragon(TM)'
ggml_opencl: selecting device: 'QUALCOMM Adreno(TM)'
ggml_opencl: device FP16 support: true
llama.cpp: loading model from /data/data/com.termux/files/home/ggml-model-q4_0.bin
llama_model_load_internal: format     = ggjt v3 (latest)
llama_model_load_internal: n_vocab    = 49954
llama_model_load_internal: n_ctx      = 2048
llama_model_load_internal: n_embd     = 4096
llama_model_load_internal: n_mult     = 256
llama_model_load_internal: n_head     = 32
llama_model_load_internal: n_layer    = 32
llama_model_load_internal: n

[... truncated for brevity ...]

---

## Issue #N/A: Math & Code Benchmark/Testing for GGUFs

**Link**: https://github.com/ggml-org/llama.cpp/issues/13127
**State**: closed
**Created**: 2025-04-26T19:24:54+00:00
**Closed**: 2025-04-27T07:55:07+00:00
**Comments**: 0

### Description

Are there any open source `frameworks/tools` that could be used to test `code/math` benchmark for GGUF models?

---

## Issue #N/A: Bug: Persistent hallucination even after re-running llama.cpp

**Link**: https://github.com/ggml-org/llama.cpp/issues/8070
**State**: closed
**Created**: 2024-06-22T23:02:15+00:00
**Closed**: 2024-08-13T01:07:02+00:00
**Comments**: 11
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

I used the command below:
```
sudo ./llama-cli -m /home/edw590/llamacpp_models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --in-suffix [3234_START] --color --interactive-first --ctx-size 0 --temp 0.2 --mlock --prompt "You are VISOR, my male personal virtual assistant. I'm Edward. I was born in 1999-11-22. It's currently the year of 2024. Address me as Sir or nothing at all. From now on, always end your answers with \"[3234_END]\"."
```
The output was:
```
[3234_START]entienda, Sir.entienda
entienda, Sir.entientienda
entienda, Sir.entienda
entienda, Sir.entienda
entienda, Sir.entienda
entienda, Sir.entienda
...
```
Another time the output was:
```
[3234_START] Cab, Sir.enti
enti
enti
enti
enti
enti
enti
...
```
The first time I saw it start to hallucinate was with this output:
```
[3234_START]Hello Sir! I'm your personal virtual assistant, VISOR. Direct your commands to me, and I will be your Caboose. I am your virtual Caboose. I is your Caboose

[... truncated for brevity ...]

---

## Issue #N/A: Freshly converted PLaMo fails assertion: vocab.id_to_token.size() == vocab.token_to_id.size()

**Link**: https://github.com/ggml-org/llama.cpp/issues/5669
**State**: closed
**Created**: 2024-02-22T19:32:46+00:00
**Closed**: 2024-10-19T01:07:25+00:00
**Comments**: 7
**Labels**: bug-unconfirmed, stale

### Description

### Steps to Reproduce
1. Download [pfnet/plamo-13b-instruct](https://huggingface.co/pfnet/plamo-13b-instruct)
2. Convert with convert-hf-to-gguf.py
3. Attempt to run inference with `main`

Fails with:
```
GGML_ASSERT: /home/jared/src/forks/llama.cpp-2/llama.cpp:3395: vocab.id_to_token.size() == vocab.token_to_id.size()
```

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

## Issue #N/A: parallel/server crashes with: ggml.c:16521: i != GGML_HASHTABLE_FULL when defragmentation is enabled

**Link**: https://github.com/ggml-org/llama.cpp/issues/6685
**State**: open
**Created**: 2024-04-15T11:39:29+00:00
**Comments**: 28
**Labels**: bug, server/webui

### Description

### Context

Using latest 17e98d4c96a583d420f12046bc92102381dbd28e llama.cpp server.

Server started with a llama70b-F16 like model:

```shell
server \
 --model model-f16.gguf \
--ctx-size 32768 \
--n-predict 4096 \
--parallel 32 \
--n-gpu-layers 81 \
--batch-size 4096 \
--ubatch-size 256 \
--metrics \
--mg 1 \
--log-format text \
--defrag-thold 0.1
```

When sending 32 concurrent requests, the server crashes with:

`GGML_ASSERT: /llama.cpp/ggml.c:16521: i != GGML_HASHTABLE_FULL`

Backend is CUDA, on 2 A100, compute capability 80.

EDIT: The issue is related with defragmentation, quick fix: disable defragmentation

---

