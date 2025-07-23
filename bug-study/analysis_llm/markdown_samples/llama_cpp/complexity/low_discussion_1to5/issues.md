# low_discussion_1to5 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- bug-unconfirmed: 15 issues
- stale: 11 issues
- enhancement: 5 issues
- good first issue: 2 issues
- medium severity: 2 issues
- help wanted: 1 issues
- build: 1 issues
- high severity: 1 issues
- question: 1 issues
- model: 1 issues

---

## Issue #N/A: Misc. bug: Docker images on GHCR stuck at **b5174** – “Publish Docker image” workflow failing since 2025‑04‑24

**Link**: https://github.com/ggml-org/llama.cpp/issues/13203
**State**: closed
**Created**: 2025-04-30T02:02:06+00:00
**Closed**: 2025-04-30T08:44:08+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

### Name and Version

version: 5174 (56304069)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu


### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-cli

### Command line

```shell
llama-cli --version
```

### Problem description & steps to reproduce

### Summary
Pulling any of the moving Docker tags (`full-vulkan`, `full`, `server`, etc.) still returns **build 5174 (56304069)**, which was published on **2025‑04‑24**. Meanwhile, the Releases page has advanced to **b5223** and beyond, so no Docker images have been published for roughly a week.

---

### Steps to reproduce
```bash
# 1. Pull the latest image
docker pull ghcr.io/ggml-org/llama.cpp:full-vulkan

# 2. Check the build banner (bypasses tools.sh)
docker run --rm \
  --entrypoint /app/llama-cli \
  ghcr.io/ggml-org/llama.cpp:full-vulkan \
  --version
```
Output:
```
version: 5174 (56304069)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
``

[... truncated for brevity ...]

---

## Issue #N/A: win 11 - powershell   --simple-io console is not responding 

**Link**: https://github.com/ggml-org/llama.cpp/issues/2520
**State**: closed
**Created**: 2023-08-04T21:10:40+00:00
**Closed**: 2023-08-07T20:10:20+00:00
**Comments**: 2

### Description

 --simple-io - win 11 - powershell console is not responding.
I can't enter any text - only cltl+c works.

---

## Issue #N/A: Move the third-party build / deploy scripts to a separate repository

**Link**: https://github.com/ggml-org/llama.cpp/issues/506
**State**: closed
**Created**: 2023-03-25T18:39:41+00:00
**Closed**: 2023-06-17T10:00:17+00:00
**Comments**: 3
**Labels**: help wanted, good first issue, build

### Description

It keeps bothering me to see these scripts in the source root.
They cannot live anywhere except in the root of the repo, so therefore it is time to go.

Task: create `llama.flake` or `llama.deploy` repo and move the scripts there.

---

## Issue #N/A: Bug: CodeShell inference not working correctly

**Link**: https://github.com/ggml-org/llama.cpp/issues/8250
**State**: closed
**Created**: 2024-07-02T08:32:52+00:00
**Closed**: 2024-07-30T10:58:48+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

The latest llama.cpp produces bad outputs for CodeShell, which previously performed well when merged into llama.cpp. 
After updating `convert-hf-to-gguf.py` and `convert-hf-to-gguf-update.py`, I have converted the [CodeShell-7b](https://huggingface.co/WisdomShell/CodeShell-7B), a ckpt working well with an old version(5d55b0cd827bb0fcfedfa329a82bd5d6ef2c93ca) to gguf. But running inference with it on the latest version produces poor outputs.
Tested command:
```
./llama-simple -m codeshell-7b.gguf -p "def merge_sort(array, start, end):" -n 100
```

### Name and Version

version: 3281 (023b8807)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
# ./llama-simple -m cd7b.gguf -p "def merge_sort(array, start, end):" -n 100
llama_model_loader: loaded meta data with 23 key-value pairs and 508 tensors from cd7b.gguf (version GGUF V3

[... truncated for brevity ...]

---

## Issue #N/A: Bug: gguf-split is broken

**Link**: https://github.com/ggml-org/llama.cpp/issues/10370
**State**: closed
**Created**: 2024-11-18T00:46:04+00:00
**Closed**: 2024-11-18T13:38:36+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

Hello. I'm trying to merge 2 gguf files into one, but when calling the script (with the merge flag or whatever), C++ complains about the syntax. It also swears at the lack of enum and asks to install it using `sudo apt install enum`. (by the way, this is critical for those who do not have sudo rights and I had to switch from the server to local wsl for this).
P.s. I've already built the application. I use WSL2 Ubuntu.
![image](https://github.com/user-attachments/assets/508e99fc-81e9-45ec-b2f5-dd0ee9219509)


### Name and Version

john@DESKTOP-CQLHOAC:~/llama.cpp$ ./llama-cli --version
version: 4121 (75207b3a)
built with cc (Ubuntu 13.2.0-23ubuntu4) 13.2.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Other? (Please let us know in description)

### Relevant log output

```shell
john@DESKTOP-CQLHOAC:~/llama.cpp/examples/gguf-split$ source gguf-split.cpp --merge /home/john/weights/Llama-3.1-Nemotron-70B-Instruct-HF-Q6_K-00001-of-0

[... truncated for brevity ...]

---

## Issue #N/A: Clean up server code

**Link**: https://github.com/ggml-org/llama.cpp/issues/5762
**State**: closed
**Created**: 2024-02-28T10:32:39+00:00
**Closed**: 2024-12-13T16:24:20+00:00
**Comments**: 3
**Labels**: enhancement, good first issue

### Description

## Motivation

As seen on https://github.com/ggerganov/llama.cpp/issues/4216 , one of the important task is to refactor / clean up the server code so that it's easier to maintain. However, without a detailed plan, personally I feel like it's unlikely to be archived.

This issue is created so that we can discuss about how to refactor or clean up the code.

The goal is to help existing and new contributors to easily find out where to work in the code base.

## Current architecture

The current server implementation has 2 thread: one for HTTP part and one for inference.

![image](https://github.com/ggerganov/llama.cpp/assets/7702203/6e44b6cc-04f0-465c-a3fb-dc5c4f13b8ae)

- The direction from HTTP ==> inference thread is done by `llama_server_queue.post(task)`
- The direction from inference ==> HTTP thread is done by `llama_server_response.send(result)`

## Ideas

Feel free to suggest any ideas that you find helpful (please keep in mind that we do not introduce new featu

[... truncated for brevity ...]

---

## Issue #N/A: memory allocation/deallocation mismatch at 0x55d37b9eca20: allocated with malloc being deallocated with delete Aborted (core dumped)[User] Insert summary of your issue or enhancement..

**Link**: https://github.com/ggml-org/llama.cpp/issues/3105
**State**: closed
**Created**: 2023-09-10T01:19:27+00:00
**Closed**: 2023-09-18T16:15:46+00:00
**Comments**: 4

### Description

Ubuntu 20.04
gcc (Ubuntu 8.4.0-3ubuntu2) 8.4.0 # same with 10
g++ (Ubuntu 8.4.0-3ubuntu2) 8.4.0 # same with 10
Python 3.10.12

core dumped immediately on attempt to load.


---

## Issue #N/A: `CUDA error: an illegal memory access was encountered` on DeepSeek-R1-0528

**Link**: https://github.com/ggml-org/llama.cpp/issues/13909
**State**: closed
**Created**: 2025-05-30T03:38:26+00:00
**Closed**: 2025-05-30T19:42:04+00:00
**Comments**: 4

### Description

Doing the below:
```bash
./llama.cpp/llama-cli  \
    -hf unsloth/DeepSeek-R1-0528-GGUF:IQ1_S  \
    --threads -1  \
    --n-gpu-layers 99 \
     --prio 3 \
     --temp 0.6  \
    --top_p 0.95  \
    --min_p 0.01  \
    --ctx-size 16384  \
     --seed 3407
```
causes
```bash
/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:75: CUDA error
CUDA error: an illegal memory access was encountered
  current device: 7, in function ggml_backend_cuda_synchronize at /llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:2461
  cudaStreamSynchronize(cuda_ctx->stream())
./llama.cpp/llama-cli(+0x751b7b)[0x5c5553040b7b]
./llama.cpp/llama-cli(+0x7521fe)[0x5c55530411fe]
./llama.cpp/llama-cli(+0x35f017)[0x5c5552c4e017]
./llama.cpp/llama-cli(+0x36200a)[0x5c5552c5100a]
./llama.cpp/llama-cli(+0x76bae0)[0x5c555305aae0]
./llama.cpp/llama-cli(+0x1d1d41)[0x5c5552ac0d41]
./llama.cpp/llama-cli(+0x1b7b27)[0x5c5552aa6b27]
./llama.cpp/llama-cli(+0x50a28)[0x5c555293fa28]
/lib/x86_64-linux-gnu/libc.so.6(+0x2a1ca)[0x77c5c8e2a1ca]
/lib/x

[... truncated for brevity ...]

---

## Issue #N/A: Bug: value of keep alive max count in cpp-httplib hardcoded too low

**Link**: https://github.com/ggml-org/llama.cpp/issues/7694
**State**: closed
**Created**: 2024-06-02T13:30:30+00:00
**Closed**: 2024-07-17T01:06:49+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, medium severity

### Description

### What happened?

server/httplib.h:22
```
#define CPPHTTPLIB_KEEPALIVE_MAX_COUNT 5
```
This causes TCP connection to drop after 5 consecutive requests. Quite annoing if you are doing many requests with a short time interval. Connection re-establishing takes ~2sec on my machine.
Tested with python httpx client and /embeddings endpoint.

Ideally, this should be configurable  - there is Server::set_keep_alive_max_count(size_t count), not used currently.
The same applies to CPPHTTPLIB_KEEPALIVE_TIMEOUT_SECOND.

### Name and Version

e141ce624af57bdffbaf57014a044eb1d9689230

### What operating system are you seeing the problem on?

Windows

### Relevant log output

_No response_

---

## Issue #N/A: Why does my program have no output after quantization

**Link**: https://github.com/ggml-org/llama.cpp/issues/895
**State**: closed
**Created**: 2023-04-11T14:28:46+00:00
**Closed**: 2024-04-11T01:06:42+00:00
**Comments**: 3
**Labels**: stale

### Description


![image](https://user-images.githubusercontent.com/35353688/231195014-46f09804-7b61-4e55-83fc-c7d73aed51b5.png)
 
No output after `.\quantize.exe .\models\7B\ggml-model-f16.bin \models\7B\ggml-model-q4_0.bin 2`
Ask for help,thanks!!!

---

## Issue #N/A: Working with pcie x1 gen1

**Link**: https://github.com/ggml-org/llama.cpp/issues/5402
**State**: closed
**Created**: 2024-02-08T00:14:34+00:00
**Closed**: 2024-02-09T20:03:03+00:00
**Comments**: 3
**Labels**: bug-unconfirmed

### Description

Hello

I have been testing llamacpp with ubuntu 22.04 and rocm5.6 it took me about 3 months to setup multigpu one rx6900 two rx6800 and one rx 6700 all together running on pcie x1 gen1.

![image](https://github.com/ggerganov/llama.cpp/assets/47074021/b040e10d-ee98-4065-a20d-92e7cca9ee16)

Llamacpp seems the only LLM loader that works with this setup, but i have notice that when the model its above 30gb size it get stuck loading it. Sometimes it takes between 1 to 2 hours to load it because but when loading it does inference really fast. But sometimes it just get stuck there, the longest time i have tested its 24 hours and it just stuck, the dots doesnt move.

its weird because its just happens with models above 30gb size, all other models loads fast and inference fast.

What can be doing this, any idea on how can i debug this to know whats going on?

Any idea, suggestion or help its very well welcome,
thanks

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

## Issue #N/A: low performance in large contex compared to mlx format model

**Link**: https://github.com/ggml-org/llama.cpp/issues/12948
**State**: closed
**Created**: 2025-04-15T01:34:45+00:00
**Closed**: 2025-04-15T11:45:07+00:00
**Comments**: 1

### Description

On my M3 Ultra, when running the phi-4 model (Q8_0 quantization) in both GGUF and MLX formats, I've noticed that while token generation speed is similar for short prompts, but  with large prompts  ,the GGUF format becomes significantly slower compared to the MLX format.
can we solve this problem?

---

## Issue #N/A: configure prints 'Unknown architecture' on FreeBSD 14.0 amd64

**Link**: https://github.com/ggml-org/llama.cpp/issues/5503
**State**: closed
**Created**: 2024-02-15T11:28:38+00:00
**Closed**: 2024-05-03T01:06:38+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale

### Description

```
-- The C compiler identification is Clang 16.0.6
-- The CXX compiler identification is Clang 16.0.6
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/local/libexec/ccache/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/local/libexec/ccache/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found Git: /usr/ports/misc/llama-cpp/work/.bin/git  
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE  
-- ccache found, compilation results will be cached. Disable with LLAMA_CCACHE=OFF.
-- CMAKE_SYSTEM_PROCESSOR: amd64
-- Unknown architecture
```

---

## Issue #N/A: Misc. bug: Server Demo on Mac, safari return error

**Link**: https://github.com/ggml-org/llama.cpp/issues/10841
**State**: closed
**Created**: 2024-12-15T16:22:57+00:00
**Closed**: 2024-12-17T08:52:10+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

### Name and Version

5478bbcd

### Operating systems

Mac

### Which llama.cpp modules do you know to be affected?

llama-server

### Problem description & steps to reproduce

1. build on MacBook Pro 16 inch 2019,
2. run server demo
3. open safari browser with localhost:8080
4. type 'hello'

### First Bad Commit

browser popup error dialog with message: "TypeError:r is not async iterable"

### Relevant log output

```shell
srv  update_slots: all slots are idle
```


---

## Issue #N/A: Question: how to make main to lead it work with my M3 E-cores instead of P-cores

**Link**: https://github.com/ggml-org/llama.cpp/issues/7577
**State**: closed
**Created**: 2024-05-28T01:59:13+00:00
**Closed**: 2024-07-12T01:17:44+00:00
**Comments**: 1
**Labels**: question, stale

### Description

### Prerequisites

- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new useful question to share that cannot be answered within Discussions.

### Background Description

I observed that on my apple M3, the default 4 threads run on P-core, but I want to run it on E-core. How do I do that?
You can see pin_cpu () in the makefile, but from the macro description it doesn't seem to work for Apple silicon, and I couldn't find anything else that works for apple silicon.
Thank you very much

### Possible Answer

Thread binding E-core

---

## Issue #N/A: add support for llama adapters

**Link**: https://github.com/ggml-org/llama.cpp/issues/528
**State**: closed
**Created**: 2023-03-26T14:28:49+00:00
**Closed**: 2024-04-12T01:07:36+00:00
**Comments**: 5
**Labels**: enhancement, model, stale

### Description

implement support for running models that use Llama adapter
https://github.com/ZrrSkywalker/LLaMA-Adapter


described here how to get the model

https://github.com/ZrrSkywalker/LLaMA-Adapter#inference

---

## Issue #N/A: std::out_of_range when using grammar sampling on Command R

**Link**: https://github.com/ggml-org/llama.cpp/issues/6801
**State**: closed
**Created**: 2024-04-21T03:18:47+00:00
**Closed**: 2024-04-21T12:29:47+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

Using `--grammar` or `--grammar-file` with Command R (not plus) causes an std::out_of_range exception and crashes.

## To reproduce
`./main.exe --model c4ai-command-r-v01-Q5_K_M.gguf --grammar-file grammars/json.gbnf`

https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF

## Platform
[b8109bc](https://github.com/ggerganov/llama.cpp/commit/b8109bc0139f15a5b321909f47510b89dca47ffc)
Windows 10 Pro 22H2
Ryzen 9 5900X, 64 GB DDR4
GPU 0: RTX 4090
GPU 1: RTX 3060

## Stack trace
```
> vcruntime140d.dll!_CxxThrowException(void * pExceptionObject, const _s__ThrowInfo * pThrowInfo) Line 82
  msvcp140d.dll!std::_Xout_of_range(const char * _Message) Line 26
  main.exe!std::unordered_map<std::string,unsigned char,std::hash<std::string>,std::equal_to<std::string>,std::allocator<std::pair<std::string const ,unsigned char>>>::at(const std::string & _Keyval) Line 447
  main.exe!unicode_utf8_to_byte(const std::string & utf8) Line 271
  main.exe!llama_decode_text(const std::stri

[... truncated for brevity ...]

---

## Issue #N/A: Bug: llama-server crash when defragmenting (llama_kv_cache_defrag_internal)

**Link**: https://github.com/ggml-org/llama.cpp/issues/9314
**State**: closed
**Created**: 2024-09-04T15:42:57+00:00
**Closed**: 2024-09-05T09:13:12+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

When I run the server with the following arguments : `./llama.cpp/llama-server --host 0.0.0.0 --port 55777 --model /opt/IdExtend/models/llm/c4ai-command-r-08-2024-Q5_K_M.gguf --flash-attn --cache-type-k q4_0 --cache-type-v q4_0 --defrag-thold 0.5 --ctx-size 60000 --threads-http 16 -np 2 --tensor-split 0.6958696919102823,0.30413030808971775,0.0 -ngl 99999`


sent data is something like that : 

```json
{
    "prompt": <aroun 19000 tokens>,
    "temperature": 0.3,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repeat_penalty": 1.0,
    "stream": true,
    "n_keep": 30000,
    "n_predict": 20219
}
```


I use it to support 2 concurrent users with a context of 30k tokens each.

And **different requests** I end up quickly (after less than 10 requests) having the llama-server crashing.

However I managed to get a crash dump out of it (please see full GDB dump as attached file)

The crashdump itself is 1gb, I can try to find a pl

[... truncated for brevity ...]

---

## Issue #N/A: Getting the following bug when running running convert.py

**Link**: https://github.com/ggml-org/llama.cpp/issues/4945
**State**: closed
**Created**: 2024-01-14T20:57:35+00:00
**Closed**: 2024-01-14T21:09:06+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

I have a Mac M3 

After running "make", I ran the following:
`python3 convert.py  --outfile models/ggml-model-7b-chat-f16.bin  --outtype f16 ../llama/llama-2-7b-chat"`

I Got the following error:
```
Writing models/ggml-model-7b-chat-f16.bin, format 1
Traceback (most recent call last):
  File "/Users/chetan/Dropbox/Code/llama.cpp/convert.py", line 1658, in <module>
    main(sys.argv[1:])  # Exclude the first element (script name) from sys.argv
    ^^^^^^^^^^^^^^^^^^
  File "/Users/chetan/Dropbox/Code/llama.cpp/convert.py", line 1643, in main
    OutputFile.write_all(
  File "/Users/chetan/Dropbox/Code/llama.cpp/convert.py", line 1188, in write_all
    check_vocab_size(params, vocab, pad_vocab=pad_vocab)
  File "/Users/chetan/Dropbox/Code/llama.cpp/convert.py", line 993, in check_vocab_size
    raise ValueError(
ValueError: The model's vocab size is set to -1 in params.json. Please update it manually. Maybe 32000?
```

Any ideas as to how to fix this?

---

## Issue #N/A: Fix CORS in `/health` endpoint

**Link**: https://github.com/ggml-org/llama.cpp/issues/6893
**State**: closed
**Created**: 2024-04-25T05:02:54+00:00
**Closed**: 2024-06-09T01:07:06+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

In `server.cpp`, `/health` does not properly set `Access-Control-Allow-Origin`.

Fixed in https://github.com/ggerganov/llama.cpp/pull/6892


---

## Issue #N/A: write mean pooled embedding to callers vector to simplify using SoTA embedding models and language bindings

**Link**: https://github.com/ggml-org/llama.cpp/issues/6754
**State**: closed
**Created**: 2024-04-19T00:43:02+00:00
**Closed**: 2024-06-06T01:06:48+00:00
**Comments**: 2
**Labels**: enhancement, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description: Write Mean Pooled Embedding vector to user supplied destination with optional skip tokens

(tl;dr? see https://github.com/ggerganov/llama.cpp/pull/6753 )

a new function, `llama_get_mean_pooled(ctx, skip_token_count, dest)`

should write the `n_embd` embedding f

[... truncated for brevity ...]

---

## Issue #N/A: Metal (iOS): Compute function exceeds available temporary registers

**Link**: https://github.com/ggml-org/llama.cpp/issues/7261
**State**: closed
**Created**: 2024-05-13T17:04:46+00:00
**Closed**: 2024-06-22T15:01:30+00:00
**Comments**: 5
**Labels**: bug-unconfirmed

### Description

`llama.cpp  b2864`
iPhone 12 pro Max
if
`GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_F16_H256,       flash_attn_ext_f16_h256,        ctx->support_simdgroup_mm);`
i get:
```
llama_model_loader: loaded meta data with 25 key-value pairs and 291 tensors from /var/mobile/Containers/Data/Application/1C5A0067-4072-44E5-BF9C-3294A335FAC2/Documents/models/Phi-3-mini-128k-instruct.IQ4_NL.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = phi3
llama_model_loader: - kv   2:                          llama.block_count u32              = 32
llama_model_loader: - kv   3:                       llama.context_length u32              = 131072
llama_model_loader: - kv   4:                     llama.embedding_length

[... truncated for brevity ...]

---

## Issue #N/A: error loading model: unrecognized tensor type 5

**Link**: https://github.com/ggml-org/llama.cpp/issues/1147
**State**: closed
**Created**: 2023-04-23T21:57:45+00:00
**Closed**: 2023-04-24T06:29:21+00:00
**Comments**: 3

### Description

How can I solve this issue ?

llama.cpp: loading model from models\ggml-vicuna-13b-1-1\ggml-vicuna-13b-1.1-q4_3.bin
error loading model: unrecognized tensor type 5

llama_init_from_file: failed to load model
AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | VSX = 0 |

---

## Issue #N/A: server unable to load model

**Link**: https://github.com/ggml-org/llama.cpp/issues/3744
**State**: closed
**Created**: 2023-10-23T14:06:45+00:00
**Closed**: 2024-04-02T01:13:21+00:00
**Comments**: 4
**Labels**: stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ x ] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ x ] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ x ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ x ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

examples server should start 

# Current Behavior

llama_load_model_from_file: failed to load model
llama_init_from_gpt_params: error: failed to load model 'models/7B/ggml-model-f16.gguf'
{"timestamp":1698069462,"level":"ERROR","function":"load_model","l

[... truncated for brevity ...]

---

## Issue #N/A: Why does my memory keep showing 3%?

**Link**: https://github.com/ggml-org/llama.cpp/issues/5790
**State**: closed
**Created**: 2024-02-29T03:31:42+00:00
**Closed**: 2024-04-16T01:06:30+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, stale

### Description

My graphics card is amd 6700xt

Agent 2                  
*******                  
  Name:                    gfx1030                            
  Uuid:                    GPU-XX                             
  Marketing Name:          AMD Radeon RX 6700 XT              
  Vendor Name:             AMD                                
  Feature:                 KERNEL_DISPATCH                    
  Profile:                 BASE_PROFILE                       
  Float Round Mode:        NEAR                               
  Max Queue Number:        128(0x80)                          
  Queue Min Size:          64(0x40)                           
  Queue Max Size:          131072(0x20000)                    
  Queue Type:              MULTI                              
  Node:                    1                                  
  Device Type:             GPU                                
  Cache Info:              
    L1:                      16(0x10) KB           

[... truncated for brevity ...]

---

## Issue #N/A: No output after commit 84d9015 on Android

**Link**: https://github.com/ggml-org/llama.cpp/issues/237
**State**: closed
**Created**: 2023-03-17T10:51:12+00:00
**Closed**: 2023-07-28T19:33:04+00:00
**Comments**: 4
**Labels**: bug, need more info

### Description

### Discussed in https://github.com/ggerganov/llama.cpp/discussions/234

<div type='discussions-op-text'>

<sup>Originally posted by **ShouNichi** March 17, 2023</sup>
When `git checkout 84d9015` and `make`, there will be no output (only the model loading message) in termux.
`git checkout 63fd76f` will produce a fully-functional binary.</div>

I've moved this to issues. Please provide sample output from the working build and the non-working build.

---

## Issue #N/A: Feature Request: Regarding Hardcoded GGML Tensor Name Length Limit (GGML_MAX_NAME)

**Link**: https://github.com/ggml-org/llama.cpp/issues/13947
**State**: closed
**Created**: 2025-05-31T17:53:14+00:00
**Closed**: 2025-07-15T01:08:21+00:00
**Comments**: 1
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

I’m currently working on adapting LLaVA-style multimodal models to GGUF for efficient quantization and deployment. During this process, I encountered a persistent and deeply frustrating limitation related to the GGML_MAX_NAME constant.

Specifically, the 64-character tensor name limit seems to be hardcoded in a way that’s difficult to override externally. Despite updating GGML_MAX_NAME before including ggml.h, modifying relevant constants, and even rebuilding from source, the restriction persists—l

[... truncated for brevity ...]

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

## Issue #N/A: Request support for polylm-13b

**Link**: https://github.com/ggml-org/llama.cpp/issues/5174
**State**: closed
**Created**: 2024-01-28T14:06:11+00:00
**Closed**: 2024-04-02T01:08:01+00:00
**Comments**: 2
**Labels**: enhancement, stale

### Description

# Feature Description

**Request Support for the polylm-13b related models**
https://huggingface.co/DAMO-NLP-MT/polylm-chat-13b
https://huggingface.co/DAMO-NLP-MT/polylm-multialpaca-13b
https://github.com/DAMO-NLP-MT/PolyLM

# Motivation 
PolyLM is a polyglot large language model, which is aimed to address the following blanks and limitations in current LLM research, offering a comprehensive and innovative solution to advance this field.


---

