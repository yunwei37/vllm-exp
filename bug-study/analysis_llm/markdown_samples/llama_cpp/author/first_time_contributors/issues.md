# first_time_contributors - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 3
- Closed Issues: 27

### Label Distribution

- stale: 17 issues
- bug-unconfirmed: 13 issues
- bug: 3 issues
- high severity: 2 issues
- build: 1 issues
- low severity: 1 issues
- invalid: 1 issues
- threading: 1 issues
- Nvidia GPU: 1 issues
- Vulkan: 1 issues

---

## Issue #N/A: train-text-from-scratch.exe stop after "begin training" (tensor->src0 is null)

**Link**: https://github.com/ggml-org/llama.cpp/issues/1869
**State**: closed
**Created**: 2023-06-15T07:26:00+00:00
**Closed**: 2024-04-10T01:07:05+00:00
**Comments**: 9
**Labels**: stale

### Description

I'm running the latest release (master-254a7a7) like that:

`bin\train-text-from-scratch.exe --vocab-model models\ggml-vocab.bin --checkpoint-in chk-lamartine-256x16.bin --checkpoint-out chk-lamartine-256x16.bin --model-out ggml-lamartine-265x16-f32.bin --train-data "shakespeare.txt"           `
I tried with several models.

# Expected Behavior

Training shoud run for a long time

# Current Behavior

Training stop immediatly without error:

```
D:\git\llama.cpp>bin\train-text-from-scratch.exe --vocab-model models\ggml-vocab.bin --ctx 64 --embd 256 --head 8 --layer 16 --checkpoint-in chk-lamartine-256x16.bin --checkpoint-out chk-lamartine-256x16.bin --model-out ggml-lamartine-265x16-f32.bin --train-data "alphonsedelamartine.txt" -t 6 -b 1 -n 32 --seed 2 --adam-iter 16 --print-details-interval 0 --predict 16 --use-flash
main: seed: 2
llama.cpp: loading model from models\ggml-vocab.bin
llama_model_load_internal: format     = ggjt v1 (pre #1405)
llama_model_load_internal:

[... truncated for brevity ...]

---

## Issue #N/A: Bug: server GET /props request return json with chat_template with last char replaced by \x00

**Link**: https://github.com/ggml-org/llama.cpp/issues/10235
**State**: closed
**Created**: 2024-11-09T10:25:16+00:00
**Closed**: 2024-12-25T01:07:19+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

examples/server/utils.hpp
static std::string llama_get_chat_template(const struct llama_model * model) {
    std::string template_key = "tokenizer.chat_template";
    // call with NULL buffer to get the total size of the string
    int32_t res = llama_model_meta_val_str(model, template_key.c_str(), NULL, 0);
    if (res < 0) {
        return "";
    } else {
        std::vector<char> model_template(res, 0);
        llama_model_meta_val_str(model, template_key.c_str(), model_template.data(), model_template.size());
        return std::string(model_template.data(), model_template.size());
    }
}
src/llama.cc
int32_t llama_model_meta_val_str(const struct llama_model * model, const char * key, char * buf, size_t buf_size) {
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second

[... truncated for brevity ...]

---

## Issue #N/A:  incompatible types when initializing type ‘__m256i {aka __vector(4) long long int}’

**Link**: https://github.com/ggml-org/llama.cpp/issues/1279
**State**: closed
**Created**: 2023-05-02T14:38:01+00:00
**Closed**: 2023-07-28T19:53:52+00:00
**Comments**: 5
**Labels**: bug, build

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

successful compilation of llama.cpp

# Current Behavior

sh-4.2$ make
I llama.cpp build info:
I UNAME_S: Linux
I UNAME_P: x86_64
I UNAME_M: x86_64
I CFLAGS: -I. -O3 -std=c11 -fPIC -DNDEBUG -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Worse llama_decode performance during generation after evaluated big batch with large number of output logits requested

**Link**: https://github.com/ggml-org/llama.cpp/issues/9200
**State**: closed
**Created**: 2024-08-27T08:50:43+00:00
**Closed**: 2024-10-11T01:07:13+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

When I'm making following calls to llama_decode:

1. Evaluate large batch of tokens with all batch.logits[i] = False.
2. Evaluate multiple times batches of 1 token with batch.logits[0] = True
3. Evaluate large batch of tokens with all or many batch.logits[i] = True

and repeat it again, then on the second run step 2 runs noticeably slower. Steps 1 and 3 run in the same time.

I suspect this might be happening because on the second run output buffer ctx.buf_output remain same large size as was resized in step 3, and it gets resetted whole with 0 with every llama_decode call despite only first n_vocab elements are required.

### Name and Version

I'm using llama-cpp-python==0.2.89

### What operating system are you seeing the problem on?

Windows

### Relevant log output

_No response_

---

## Issue #N/A: [User] faild to find n_mult number from range 256, with n_ff = 3072

**Link**: https://github.com/ggml-org/llama.cpp/issues/2241
**State**: closed
**Created**: 2023-07-16T10:42:50+00:00
**Closed**: 2023-07-17T03:39:24+00:00
**Comments**: 2
**Labels**: invalid

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

i'm not sure if i should
change this line to `for n_mult in range(3000, 1, -1):`

# Current Behavior
model tried to convert https://huggingface.co/symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli/tree/main

params: `n_vocab:250000 n_embd:768 n_head:12 n_layer:12`

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: Gemma-3 vision don't work multilingual

**Link**: https://github.com/ggml-org/llama.cpp/issues/12351
**State**: closed
**Created**: 2025-03-12T11:27:37+00:00
**Closed**: 2025-05-05T01:07:55+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

llama-cli.exe --version
version: 4877 (363f8c5d)
built with MSVC 19.43.34808.0 for x64

### Operating systems

Windows

### GGML backends

CPU

### Hardware

i7-12700

### Models

https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/blob/main/gemma-3-4b-it-f16.gguf

https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/blob/main/mmproj-model-f16.gguf

### Problem description & steps to reproduce

I realize that the vision support in llama.cpp is very experimental, but nevertheless I think it's worth opening this issue

[Example image.](https://github.com/ggml-org/llama.cpp/blob/master/media/matmul.png)

transofmers code (7652804d237fb8768f0f0b8129a05e4f0576114b)
```python
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

ckpt = "google/gemma-3-4b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(
    ckpt, device_map="auto", torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(ckpt)

messages = [
    {
    

[... truncated for brevity ...]

---

## Issue #N/A: Model is split between multiple GPU's even after explicitly specifying main gpu

**Link**: https://github.com/ggml-org/llama.cpp/issues/4442
**State**: closed
**Created**: 2023-12-13T15:14:09+00:00
**Closed**: 2024-04-03T01:14:11+00:00
**Comments**: 3
**Labels**: stale

### Description

I think we need to solve for this, models are automatically loaded and split on multiple GPUs if you have `BaseMosaic` enabled in your XORG config, overriding the default flags that you can explicitly set as your main GPU.

This isn't that big of a deal, but helps when you are experimenting with multiple models.

More details here: https://forums.developer.nvidia.com/t/memory-is-allocated-on-all-gpus/183110/3

Original Discussion:
> ### Discussed in https://github.com/ggerganov/llama.cpp/discussions/2752

> <div type='discussions-op-text'>
> 
> <sup>Originally posted by **isaacmorgan** August 24, 2023</sup>
> Using the CuBLAS build with 2 GPUs. I want to load the model onto a single GPU, but the model is always loaded into the memory of both GPUs. Even if I only run on 1 GPU the model is loaded onto both GPUs. 
> 
> Things I've tried:
> `./main -m ./llama-2-7b.ggmlv3.q8_0.bin -i --interactive-first -ngl 40`
> Result: Default behavior: Loads model onto both GPUs, runs on

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: KV cache stopped working in b5554 version

**Link**: https://github.com/ggml-org/llama.cpp/issues/14071
**State**: open
**Created**: 2025-06-08T19:06:49+00:00
**Comments**: 3
**Labels**: bug-unconfirmed

### Description

### Name and Version

llama-server --version
version: 5554 (3600cc28)
built with Apple clang version 17.0.0 (clang-1700.0.13.5) for arm64-apple-darwin24.5.0

or any subsequent version up to b5604 included

### Operating systems

Mac

### GGML backends

Metal

### Hardware

M4 Max, M2 Max, M1 Max

### Models

gemma models : 
fastest to test is the the [1B gemma-3 Q4_K_M] (https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF)

./llama-server -hf ggml-org/gemma-3-1b-it-GGUF:Q4_K_M -ngl 200 -c 4096



### Problem description & steps to reproduce

using b5552 KV cache works as expected :
2nd query only has 1 token processed (vs 250 for first query)

cd ../../build-b5552/bin
./llama-server -m $LLAMA_CACHE/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf -ngl 200 -c 4096

python test_kv_cacke.py

Sending request 1...
response 1: Blacksmith.
prompt_n: 250
system_fingerprint: b5552-3f55f781
duration = 0.136

Sending request 2...
response 2: Blacksmith.
prompt_n: 1
system_fingerprint: b5552-3

[... truncated for brevity ...]

---

## Issue #N/A: Question about new models

**Link**: https://github.com/ggml-org/llama.cpp/issues/1900
**State**: closed
**Created**: 2023-06-16T20:38:58+00:00
**Closed**: 2024-04-10T01:06:57+00:00
**Comments**: 4
**Labels**: stale

### Description

Hi everybody,

I wanted to ask if it was possible to train a new model from a series of text files, either starting from scratch or starting (with a process similar to fine-tuning) from a pre-existing model compatible wiht llama.cpp

Thanks in advance.


---

## Issue #N/A: is there any simple way to ask server stop generating?

**Link**: https://github.com/ggml-org/llama.cpp/issues/4911
**State**: closed
**Created**: 2024-01-13T12:14:06+00:00
**Closed**: 2024-01-16T21:14:56+00:00
**Comments**: 2

### Description

Hi, I would like to let server response to be stopped on demand but there is no api endpoint for this. is it possible to achieve such functionality in some easy way?



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

## Issue #N/A: Investigate PG-TD (Planning-Guided Transformer Decoding) sampling

**Link**: https://github.com/ggml-org/llama.cpp/issues/2324
**State**: closed
**Created**: 2023-07-22T15:27:27+00:00
**Closed**: 2024-04-09T01:07:40+00:00
**Comments**: 1
**Labels**: stale

### Description

There's been some work going on for beam search #2267, CFG #2135, steering vectors #1472, Honest LLaMA #1799, and other techniques to improve generation quality. So I thought sharing this paper may be of use.

It outlines some sampling tricks to improve code generation, which may be appropriate for StarCoder.cpp, given LLaMA's underwhelming coding performance, but I think these ideas can perhaps be used as inspiration for other types of planners for different tasks.

###  ["_Planning with Large Language Models for Code Generation_"](https://openreview.net/forum?id=Lr8cOOtYbfL)

> Abstract: Existing large language model-based code generation pipelines typically use beam search or sampling algorithms during the decoding process. Although the programs they generate achieve high token-matching-based scores, they often fail to compile or generate incorrect outputs. The main reason is that conventional Transformer decoding algorithms may not be the best choice for code generation. In t

[... truncated for brevity ...]

---

## Issue #N/A: CUDA Error 400: Invalid Resource Handle when Running on Single GPU

**Link**: https://github.com/ggml-org/llama.cpp/issues/2269
**State**: closed
**Created**: 2023-07-19T03:47:47+00:00
**Closed**: 2024-04-09T01:07:51+00:00
**Comments**: 5
**Labels**: stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ x ] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ x ] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ x ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ x ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

I am trying to make `llama.cpp` run on a single GPU (in my case, GPU 5) on a multi-GPU system because there are other tasks running on my other GPUs.

# Current Behavior

`llama.cpp` crashes with `CUDA error 400 at ggml-cuda.cu:3343: invalid resource handl

[... truncated for brevity ...]

---

## Issue #N/A: [User] Insert summary of your issue or enhancement..

**Link**: https://github.com/ggml-org/llama.cpp/issues/2085
**State**: closed
**Created**: 2023-07-03T11:47:48+00:00
**Closed**: 2023-07-06T10:02:53+00:00
**Comments**: 2

### Description

Hello,

On the main page, you mention the following: "This project is for educational purposes". 

Want makes it, according to you, not suitable for real world situations and production environments?

Thanks.

---

## Issue #N/A: Unable to Run miqu-1-70b.q4_k_m.gguf Model Without Error Messages

**Link**: https://github.com/ggml-org/llama.cpp/issues/5255
**State**: closed
**Created**: 2024-02-01T12:24:14+00:00
**Closed**: 2024-02-01T12:37:29+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

System: windows 11 x64
The following is the log:

> [1706790015] Log start
[1706790015] Cmd: main.exe -m miqu-1-70b.q4_k_m.gguf --color --temp 1 --top_p 0.95  -n -1 -p "<s> [INST] QUERY_1 [/INST] ANSWER_1"
[1706790015] main: build = 2038 (ce320601)
[1706790015] main: built with MSVC 19.37.32826.1 for x64
[1706790015] main: seed  = 1706790015
[1706790015] main: llama backend init
[1706790015] main: load the model and apply lora adapter, if any






---

## Issue #N/A: android built on GPU cannot comparable with CPU?

**Link**: https://github.com/ggml-org/llama.cpp/issues/13910
**State**: open
**Created**: 2025-05-30T04:47:18+00:00
**Comments**: 3
**Labels**: stale

### Description

I tried to build on Android device with GPU env but fail at official documents.
1.Termux env
2.openCL 

I blocked here:

![Image](https://github.com/user-attachments/assets/918e9903-f500-41ba-ae96-33ee73818009)

![Image](https://github.com/user-attachments/assets/8673bcf2-8ac2-4a37-a02e-d684d6759cfe)

So, I changed to another build method as below:
1.
using termux default cmake tool does not ninja
2.
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DOPENCL_ICD_LOADER_HEADERS_DIR=/data/data/com.termux/files/usr/include \
  -DCMAKE_C_COMPILER=/data/data/com.termux/files/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/data/data/com.termux/files/usr/bin/clang++ \
  -DCMAKE_C_FLAGS="--target=aarch64-linux-android24 -D_POSIX_C_SOURCE=200809L" \
  -DCMAKE_CXX_FLAGS="--target=aarch64-linux-android24 -D_POSIX_C_SOURCE=200809L"
3.
cmake .. -DBUILD_SHARED_LIBS=ON -DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=ON
4.
cmake --build build-android 
cmake --build . --config Release




[... truncated for brevity ...]

---

## Issue #N/A: llama_model_load: error loading model: unable to allocate backend buffer

**Link**: https://github.com/ggml-org/llama.cpp/issues/7366
**State**: closed
**Created**: 2024-05-18T13:26:52+00:00
**Closed**: 2024-05-19T17:25:03+00:00
**Comments**: 3
**Labels**: bug-unconfirmed

### Description

OS: Windows 11, running Text Generation WebUI, up to date on all releases.
Processor: Intel Core i5-8500 3GHz (6 Cores - no HT)
Memory: 16GB System Memory
GPUs: Five nVidia RTX 3600 - 12GB VRAM versions (First iteration during Covid)

Model: Coomand-R-35B-v1-OLD_Q4_K_M.gguf

Model Parameters:
- n-gpu-layers: 41 (41 of 41, loading FULLY into VRAM)
- n_ctx: 8192
- tensor split: 10,10,10,10,10
- flash-attn: Checked
- tensorcores: checked
- no-mmap: checked

Output from Model Load:
```
08:02:50-987413 INFO     Loading "Coomand-R-35B-v1-OLD_Q4_K_M.gguf"
08:02:51-580810 INFO     llama.cpp weights detected: "models\Coomand-R-35B-v1-OLD_Q4_K_M.gguf"
llama_model_loader: loaded meta data with 23 key-value pairs and 322 tensors from models\Coomand-R-35B-v1-OLD_Q4_K_M.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str      

[... truncated for brevity ...]

---

## Issue #N/A: bug: GGML_ASSERT(backend_embd != nullptr) failed error at llama.cpp:14775

**Link**: https://github.com/ggml-org/llama.cpp/issues/14418
**State**: open
**Created**: 2025-06-27T11:25:19+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Name and Version

llama_cpp_python==0.2.88

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

Other (Please specify in the next section)
Issue at /llama-cpp-python_16ea09f94c0346afa022d988e7934741/vendor/llama.cpp/src/llama.cpp:14775 due to libggml.so

### Command line

```shell

```

### Problem description & steps to reproduce

I have a flask application running in a docker container instance on a VM. It takes 'query' as input and uses an underlying RAG system to answer the query, and returns a json response containing the LLM output. 
I use `LlamaCppEmbeddings` (which has a dependency on `llama_cpp_python`) to load a `nomic-embed-text-v1.5.Q8_0.gguf `embedding model, it is used to perform dense vector search in the knowledge base. That vector index was also created using the same embedding model, and stored in a `FAISS` db ~50MB (index.faiss+index.pkl), which is loaded in runtime. I use Llama3.3-70B LLM model for the generation based on top-4

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: Model not loaded on Android with NDK

**Link**: https://github.com/ggml-org/llama.cpp/issues/13399
**State**: closed
**Created**: 2025-05-09T08:41:32+00:00
**Closed**: 2025-06-26T01:07:56+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

version: [b5320](https://github.com/ggml-org/llama.cpp/releases/tag/b5320)
built with macOS Sonoma, Android Studio Meerkat 2024.3.1 Patch 2 and Android NDK 27.012077973

### Operating systems

Other? (Please let us know in description), Mac

### Which llama.cpp modules do you know to be affected?

libllama (core library)

### Command line

```shell

```

### Problem description & steps to reproduce

I'm trying to use llama.cpp on Android with local inference using NDK with JNI. When I try to load a model (nomic_embed_text_v1_5_q4_0.gguf) with the "llama_model_load_from_file" method, it does not load and returns null. 

#### CMakeLists.txt

```
cmake_minimum_required(VERSION 3.22.1)
project(llama_jni)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# Path to llama.cpp folder is in the root folder of the project
set(LLAMA_CPP_DIR "${CMAKE_SOURCE_DIR}/../../../../llama.cpp")
set(LLAMA_CPP_SRC_DIR "

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: llama-cli outputs some messages to console

**Link**: https://github.com/ggml-org/llama.cpp/issues/10603
**State**: closed
**Created**: 2024-11-30T14:54:12+00:00
**Closed**: 2025-01-14T01:08:56+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

$ ./llama-cli --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3060, compute capability 8.6, VMM: yes
version: 4229 (3e0ba0e6)
built with cc (Gentoo 13.3.1_p20241025 p1) 13.3.1 20241024 for x86_64-pc-linux-gnu


### Operating systems

Linux

### GGML backends

CUDA

### Hardware

RTX 3060

### Models

_No response_

### Problem description & steps to reproduce

When using llama-cli with `--prompt-cache-all` it prints the following message which breaks the parsing: "main: saving final output to session file ..."

Some of the logging in the examples/main.cpp still uses `LOG` macro rather than `LOG_INF`/`LOG_WRN`/etc which results in this output not being directed to the log file.

### First Bad Commit

_No response_

### Relevant log output

```shell
USER: Hey
ASSISTANT:  🙋♂ Good morning!
USER:
main: saving final output to session file '/va

[... truncated for brevity ...]

---

## Issue #N/A: Bug: I use llama-b3091-bin-win-llvm-arm64.zip Run qwen2-0_5b-instruct-q8_0.gguf and it cannot start. Is it a compilation error of llama-b3091-bin-win-llvm-arm64.zip?

**Link**: https://github.com/ggml-org/llama.cpp/issues/7873
**State**: closed
**Created**: 2024-06-11T07:02:50+00:00
**Closed**: 2024-07-27T01:06:46+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

I use llama-b3091-bin-win-llvm-arm64.zip
Run qwen2-0_5b-instruct-q8_0.gguf and it cannot start. Is it a compilation error of llama-b3091-bin-win-llvm-arm64.zip?

### Name and Version

llama-b3091-bin-win-llvm-arm64.zip

### What operating system are you seeing the problem on?

Other? (Please let us know in description)

### Relevant log output

```shell
windows-arm64
```


---

## Issue #N/A: [User] Deadlock if number of threads > number of (hyper)threads

**Link**: https://github.com/ggml-org/llama.cpp/issues/1159
**State**: closed
**Created**: 2023-04-24T19:02:29+00:00
**Closed**: 2023-05-03T18:30:11+00:00
**Comments**: 4
**Labels**: threading

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

I expect the program to run suboptimally but finish.

# Current Behavior

Currently the program locks up with very large cpu utilization.

# Environment and Context

I have a 6 core intel machine i.e. 12 threads with hyperthreading. 
Once I run with -t 13

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug:  MiniCPM-2B-128k convert_hf_to_gguf Missing the required key: rope_scaling

**Link**: https://github.com/ggml-org/llama.cpp/issues/12468
**State**: closed
**Created**: 2025-03-19T14:37:44+00:00
**Closed**: 2025-05-03T01:07:37+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

$./llama-cli --version
version: 4778 (a82c9e7c)
built with aarch64-none-linux-gnu-gcc (Arm GNU Toolchain 12.3.Rel1 (Build arm-12.35)) 12.3.1 20230626 for aarch64-none-linux-gnu

### Operating systems

Linux

### GGML backends

CPU

### Hardware

Arm CPU

### Models

[MiniCPM-2B-128k](https://huggingface.co/openbmb/MiniCPM-2B-128k)

### Problem description & steps to reproduce

I encountered the following problem 
**KeyError: 'Missing the required key rope_scaling.long_factor or rope_scaling_short_factor'** while convert hf to gguf:  [MiniCPM-2B-128k](https://huggingface.co/openbmb/MiniCPM-2B-128k) on llamacpp b4778. 

MiniCPM-2B-128k/config.json:
```
{
    "_name_or_path": "openbmb/CPM-2B",
    "architectures": [
        "MiniCPMForCausalLM"
    ],
    "auto_map": {
        "AutoConfig": "configuration_minicpm.MiniCPMConfig",
        "AutoModel": "modeling_minicpm.MiniCPMModel",
        "AutoModelForCausalLM": "modeling_minicpm.MiniCPMForCausalLM",
        "AutoMo

[... truncated for brevity ...]

---

## Issue #N/A: [User] latest ggml-alloc not support as Xcode app package

**Link**: https://github.com/ggml-org/llama.cpp/issues/2778
**State**: closed
**Created**: 2023-08-25T06:22:02+00:00
**Closed**: 2024-04-09T01:06:52+00:00
**Comments**: 2
**Labels**: stale

### Description

@j-f1 hi. I used your repo(https://github.com/j-f1/LLM-Playground/), but couldn't load your branch 'jed/defaults', so I used the latest master from there, Since 11 f3ca06b8c66b0427aab0a472479da22553b472 commit introduced GGML - alloc. H later, unable to compile successfully. Error message:
```
Undefined symbols for architecture arm64:
  "_ggml_allocr_alloc", referenced from:
      llm_build_llama(llama_context&, int const*, float const*, int, int) in llama.o
      llm_build_falcon(llama_context&, int const*, float const*, int, int) in llama.o
  "_ggml_allocr_alloc_graph", referenced from:
      _llama_new_context_with_model in llama.o
      llama_eval_internal(llama_context&, int const*, float const*, int, int, int, char const*) in llama.o
  "_ggml_allocr_free", referenced from:
      _llama_new_context_with_model in llama.o
      llama_context::~llama_context() in llama.o
  "_ggml_allocr_is_measure", referenced from:
      llm_build_llama(llama_context&, int const*, float

[... truncated for brevity ...]

---

## Issue #N/A: Significantly different results (and WRONG) inference when GPU is enabled.

**Link**: https://github.com/ggml-org/llama.cpp/issues/7048
**State**: closed
**Created**: 2024-05-02T18:51:50+00:00
**Closed**: 2024-05-17T18:49:39+00:00
**Comments**: 40
**Labels**: bug, Nvidia GPU

### Description

I am running llama_cpp version 0.2.68 on Ubuntu 22.04LTS under conda environment. Attached are two Jupyter notebooks with ONLY one line changed (use CPU vs GPU).  As you can see for exact same environmental conditions switching between CPU/GPU gives vastly different answers where the GPU is completely wrong.  Some pointers on how to debug this I would appreciate it.

The only significant difference between the two files is this one liner
      `#n_gpu_layers=-1, # Uncomment to use GPU acceleration`

The model used was **openhermes-2.5-mistral-7b.Q5_K_M.gguf**

[mistral_llama_large-gpu.pdf](https://github.com/ggerganov/llama.cpp/files/15192723/mistral_llama_large-gpu.pdf)
[mistral_llama_large-cpu.pdf](https://github.com/ggerganov/llama.cpp/files/15192725/mistral_llama_large-cpu.pdf)



---

## Issue #N/A: [Bug report] Performance deterioration of LLaMA-2 model due to hardcoded rms_norm_eps 

**Link**: https://github.com/ggml-org/llama.cpp/issues/2373
**State**: closed
**Created**: 2023-07-24T13:58:11+00:00
**Closed**: 2023-07-24T15:57:14+00:00
**Comments**: 2

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [Yes] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [Yes] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [Yes] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [Yes] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

When running converted ggml model, the eps used in RMSNorm is consistent with original model definition.

# Current Behavior

The norm_eps used in RMSNorm is hardcoded to 1e-6, in all backends: X86, CUDA, Metal.
Related commit: Change RMSNorm eps to 1e-6 

[... truncated for brevity ...]

---

## Issue #N/A: SYCL bug: DeepSeek-V2-Lite-Chat-Q4_K_M does not work as expected

**Link**: https://github.com/ggml-org/llama.cpp/issues/12390
**State**: closed
**Created**: 2025-03-14T14:31:14+00:00
**Closed**: 2025-03-15T14:19:31+00:00
**Comments**: 18
**Labels**: bug

### Description

### Name and Version

root@alc-ai:/home/aubrey/work/llama-gpu# ./build/bin/llama-cli --version
version: 4887 (8fcb5636)
built with Intel(R) oneAPI DPC++/C++ Compiler 2025.0.4 (2025.0.4.20241205) for x86_64-unknown-linux-gnu

### Operating systems

_No response_

### Which llama.cpp modules do you know to be affected?

_No response_

### Command line

```shell
 ./build/bin/llama-cli -m /srv/models/DeepSeek-V2-Lite-Chat-Q4_K_M/DeepSeek-V2-Lite-64x1.5B-Chat-Q4_K_M.gguf -ngl 99 -sm none -mg 0 -p "what is your name?" -n 30 -no-cnv
```

### Problem description & steps to reproduce

root@alc-ai:/home/aubrey/work/llama-gpu# ./build/bin/llama-cli -m /srv/models/DeepSeek-V2-Lite-Chat-Q4_K_M/DeepSeek-V2-Lite-64x1.5B-Chat-Q4_K_M.gguf -ngl 99 -sm none -mg 0 -p "what is your name?" -n 30 -no-cnv
build: 4887 (8fcb5636) with Intel(R) oneAPI DPC++/C++ Compiler 2025.0.4 (2025.0.4.20241205) for x86_64-unknown-linux-gnu
main: llama backend init
main: load the model and apply lora adapter, if any
llama_mod

[... truncated for brevity ...]

---

## Issue #N/A: Question regarding distributed computing...

**Link**: https://github.com/ggml-org/llama.cpp/issues/946
**State**: closed
**Created**: 2023-04-13T14:03:55+00:00
**Closed**: 2024-04-11T01:06:32+00:00
**Comments**: 8
**Labels**: stale

### Description

I have currently access to 20 old computers, each with 32GB ram and 4 cores, 256gb ssd, 1 gbit speed network, connected to a 48port switch. (i could get a lot lot more computers but i dont have enough electricity currently)
Would it be somehow possible to distribute the llama model with llama.cpp to the 20 computers to being able to run the 65b model at a moderate speed?
What would i have to do to distribute the model on many computers to run it on cpu?
i am only interested in inference, not training..... for training i can rent cloud gpu's.

Thanks for any input that would help me / recommendation / problems.

What i see as a problem is how to split the model / models (in case i use other models) efficiently so that network bandwidth isnt the limiting factor.







---

## Issue #N/A: can' quantize deekseek model

**Link**: https://github.com/ggml-org/llama.cpp/issues/4925
**State**: closed
**Created**: 2024-01-14T07:19:27+00:00
**Closed**: 2024-04-18T01:06:43+00:00
**Comments**: 7
**Labels**: bug-unconfirmed, stale

### Description

When I download the model from the deepseek huggingface official repository, I cannot convert it into a gguf file。

python D:\Ai\convert.py D:\Ai\deepseek-coder-6.7b-instruct

D:\Ai\gguf-py
Loading model file D:\Ai\deepseek-coder-6.7b-instruct\pytorch_model-00001-of-00002.bin
Loading model file D:\Ai\deepseek-coder-6.7b-instruct\pytorch_model-00001-of-00002.bin
Loading model file D:\Ai\deepseek-coder-6.7b-instruct\pytorch_model-00002-of-00002.bin
params = Params(n_vocab=32256, n_embd=4096, n_layer=32, n_ctx=16384, n_ff=11008, n_head=32, n_head_kv=32, f_norm_eps=1e-06, n_experts=None, n_experts_used=None, rope_scaling_type=<RopeScalingType.LINEAR: 'linear'>, f_rope_freq_base=100000, f_rope_scale=4.0, n_orig_ctx=None, rope_finetuned=None, ftype=None, path_model=WindowsPath('D:/Ai/deepseek-coder-6.7b-instruct'))
Traceback (most recent call last):
  File "D:\Ai\convert.py", line 1658, in <module>
    main(sys.argv[1:])  # Exclude the first element (script name) from sys.argv
  

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

