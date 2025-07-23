# bug - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 4
- Closed Issues: 26

### Label Distribution

- bug: 30 issues
- low severity: 3 issues
- build: 3 issues
- high priority: 2 issues
- Nvidia GPU: 2 issues
- medium severity: 2 issues
- model: 2 issues
- hardware: 2 issues
- good first issue: 2 issues
- high severity: 1 issues

---

## Issue #N/A: Bug: cannot find tokenizer merges in model file

**Link**: https://github.com/ggml-org/llama.cpp/issues/9692
**State**: closed
**Created**: 2024-09-30T02:31:24+00:00
**Closed**: 2024-10-08T03:14:42+00:00
**Comments**: 11
**Labels**: bug, high priority, high severity

### Description

### What happened?

When I use transformers==4.45.1 and convert llama.cpp to the file used by ollama, there is no error, but when I load the model with ollama, the error ollama cannot find tokenizer merges in model file appears

### Name and Version

所有版本

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

---

## Issue #N/A: Quantize python script fails.

**Link**: https://github.com/ggml-org/llama.cpp/issues/431
**State**: closed
**Created**: 2023-03-23T15:15:24+00:00
**Closed**: 2023-03-23T20:42:54+00:00
**Comments**: 5
**Labels**: bug

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ ] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ ] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

I have my llama models stored in models/llama/{7B,13B,30B,65B}.

I expect that when I run the following command that the model will be converted

$ python3 quantize.py --models-path models/llama 30B


# Current Behavior

When attempting to quantize the model 

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: llama-server ignores the stop parameter

**Link**: https://github.com/ggml-org/llama.cpp/issues/11538
**State**: closed
**Created**: 2025-01-31T10:34:03+00:00
**Closed**: 2025-01-31T13:48:33+00:00
**Comments**: 1
**Labels**: bug

### Description

### Name and Version

version: 4599 (8b576b6c)
built with cc (Debian 12.2.0-14) 12.2.0 for x86_64-linux-gnu

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell
curl --request POST --url http://localhost:8080/completion --header "Content-Type: application/json"  --data '{"prompt": "A B C D E F G H I J K","n_predict": 128, "stop": ["O P Q"]}'
```
Notice that the stop string spawns multiple tokens.


### Problem description & steps to reproduce

The server `/completion` endpoint ignores the `stop` parameter.

Tested by loading phi4 in llama-server, then sending a request with a array of stop tokens including a triple backquote: [stop1, stop2, "```", stop3,...]

### example
`curl --request POST --url http://localhost:8080/completion --header "Content-Type: application/json"  --data '{"prompt": "A B C D E F G H I J K","n_predict": 128, "stop": ["O P Q"]}'`

note that the stop string spans multiple tokens


The offe

[... truncated for brevity ...]

---

## Issue #N/A: train-text-from-scratch and finetune nan loss on iter=2

**Link**: https://github.com/ggml-org/llama.cpp/issues/3940
**State**: closed
**Created**: 2023-11-04T04:42:06+00:00
**Closed**: 2023-11-07T08:04:52+00:00
**Comments**: 2
**Labels**: bug

### Description

I was trying out the finetune example with my model but it kept going into nan loss. I eventually tried train-text-from-scratch, following the instructions on the README there and it goes into nan as well. I've reproduced this on two machines.

```
root@c5a10438d69e:/workspace/llama.cpp# ./train-text-from-scratch         --vocab-model ./models/ggml-vocab-llama.gguf         --ctx 64 --embd 256 --head 8 --layer 16         --checkpoint-in  chk-shakespeare-256x16-LATEST.gguf         --checkpoint-out chk-shakespeare-256x16-ITERATION.gguf         --model-out ggml-shakespeare-256x16-f32-ITERATION.gguf         --train-data "shakespeare.txt"         -t 6 -b 16 --seed 1 --adam-iter 256         --no-checkpointing
main: seed: 1
llama_model_loader: loaded meta data with 17 key-value pairs and 0 tensors from ./models/ggml-vocab-llama.gguf (version GGUF V3 (latest))
llama_model_loader: - kv   0:                       general.architecture str     
llama_model_loader: - kv   1:                  

[... truncated for brevity ...]

---

## Issue #N/A: Constrained decoding with grammar fails for c4ai-command-r-v01

**Link**: https://github.com/ggml-org/llama.cpp/issues/6112
**State**: closed
**Created**: 2024-03-17T14:51:01+00:00
**Closed**: 2024-05-28T10:55:36+00:00
**Comments**: 6
**Labels**: bug, help wanted

### Description

I am trying to apply constrained decoding for the recently adopted command-r. 

Using the most recent master branch (https://github.com/ggerganov/llama.cpp/commit/c47cf414efafb8f60596edc7edb5a2d68065e992) I'm trying to apply the simplest list.  

`./main -m ~/data/c4ai-command-r-v01/ggml-model-Q4_K_M.gguf -p "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Please give me a list of things to do in SF?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>" -ctk q8_0 -ngl 99 -n 500 --grammar-file grammars/list.gbnf`

It fails with 

`libc++abi: terminating due to uncaught exception of type std::out_of_range: unordered_map::at: key not found`

Any idea what could go wrong here?

More details:

```
Log start
main: build = 2447 (c47cf414)
main: built with Apple clang version 15.0.0 (clang-1500.3.9.4) for arm64-apple-darwin23.3.0
main: seed  = 1710686911
llama_model_loader: loaded meta data with 23 key-value pairs and 322 tensors from ~/data/c4ai-command-r-v01/ggml-m

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

## Issue #N/A: Incoherent output after merging https://github.com/ggerganov/llama.cpp/pull/2183

**Link**: https://github.com/ggml-org/llama.cpp/issues/2187
**State**: closed
**Created**: 2023-07-12T04:57:11+00:00
**Closed**: 2023-07-14T18:51:46+00:00
**Comments**: 7
**Labels**: bug

### Description

The commit in question seems to be https://github.com/ggerganov/llama.cpp/commit/20d7740a9b45f6e5b247fa3738fdda35e18c2e8a 

The AI responses no longer seem to consider the prompt after this commit.

Running pre-built cuda executables from github actions:

**llama-master-20d7740-bin-win-cublas-cu11.7.1-x64**
```
PS E:\LLaMA\llamacpp> .\main.exe --model e:\LLaMA\models\airoboros-7b-gpt4.ggmlv3.q4_0.bin -ngl 32 -n 30 -p "Hi, my name is"
main: build = 820 (20d7740)
main: seed  = 1689137712
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 2060, compute capability 7.5
llama.cpp: loading model from e:\LLaMA\models\airoboros-7b-gpt4.ggmlv3.q4_0.bin
llama_model_load_internal: format     = ggjt v3 (latest)
llama_model_load_internal: n_vocab    = 32000
llama_model_load_internal: n_ctx      = 512
llama_model_load_internal: n_embd     = 4096
llama_model_load_internal: n_mult     = 256
llama_model_load_internal: n_head     = 32
llama_model_load_internal: n_l

[... truncated for brevity ...]

---

## Issue #N/A: CUDA graphs break quantized K cache

**Link**: https://github.com/ggml-org/llama.cpp/issues/7492
**State**: closed
**Created**: 2024-05-23T12:11:15+00:00
**Closed**: 2024-05-27T17:33:43+00:00
**Comments**: 5
**Labels**: bug, Nvidia GPU

### Description

As of right now it is already possible on master to quantize the K cache via e.g. `-ctk q8_0`. However, this is currently broken on master for batch size 1. Disabling CUDA graphs via the environment variable `GGML_CUDA_DISABLE_GRAPHS=1` fixes the issue.

cc: @agray3 

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

## Issue #N/A: Bug: cannot create std::vector larger than max_size()

**Link**: https://github.com/ggml-org/llama.cpp/issues/9391
**State**: open
**Created**: 2024-09-09T15:52:21+00:00
**Comments**: 10
**Labels**: bug, medium severity

### Description

### What happened?

My usual build recipe and run scripts do not work after b3680. Something changed in b3681, but I don't know what.
I see this same failure across models and cli flags, so it seems to be deeper than a single feature choice, so I have excluded the launch script.

This is the actual error:
```
...
terminate called after throwing an instance of 'std::length_error'
  what():  cannot create std::vector larger than max_size()
<launch script name> Aborted                 (core dumped)
```

Here is what the binary reports at runtime:
```
system_info: n_threads = 24 (n_threads_batch = 24) / 48 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
main: interactive mode on.
```

Here is how I configure the build:
```
cmake -DGGML_AVX=ON -DGGML_AV

[... truncated for brevity ...]

---

## Issue #N/A: persimmon crashes with CUDA: assertion failure `ggml_is_contiguous(src0)`

**Link**: https://github.com/ggml-org/llama.cpp/issues/5823
**State**: open
**Created**: 2024-03-01T19:27:09+00:00
**Comments**: 3
**Labels**: bug, model

### Description

Attempting to run a persimmon model with the CUDA backend fails an assertion in ggml_cuda_rope: `ggml_is_contiguous(src0)`

ref https://github.com/ggerganov/llama.cpp/pull/5668#issuecomment-1959988387

---

## Issue #N/A: Not having enough memory just causes a segfault or something

**Link**: https://github.com/ggml-org/llama.cpp/issues/257
**State**: closed
**Created**: 2023-03-18T07:28:43+00:00
**Closed**: 2023-05-06T18:03:16+00:00
**Comments**: 9
**Labels**: bug, duplicate, hardware, model

### Description

So. I'm trying to build with CMake on Windows 11 and the thing just stops after it's done loading the model.

![image](https://user-images.githubusercontent.com/4723091/226091364-64a488a7-ebb5-4c24-9dd0-1cb81378008d.png)

And apparently, this is a segfault.

![Screenshot_20230318_121935](https://user-images.githubusercontent.com/4723091/226091335-afbf2712-d2b8-4b88-9b44-6b6a43d78565.png)

Yay yay yyayy yyayay

this is a memory allocation failure it seems, from me not having enough memory. not like llama.cpp Tells Me That lmao, it just segfaults

(`ctx->mem_buffer` is nullptr which probably means the malloc just failed)

---

## Issue #N/A: Eval bug: server API endpoint not respecting `n_predict` with `-2` (until context filled)

**Link**: https://github.com/ggml-org/llama.cpp/issues/12264
**State**: closed
**Created**: 2025-03-08T04:01:32+00:00
**Closed**: 2025-03-13T10:30:58+00:00
**Comments**: 5
**Labels**: bug, good first issue

### Description

### Name and Version

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
version: 4844 (d76a86d9)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### Operating systems

Linux

### GGML backends

CUDA

### Hardware

RTX 3090

### Models

- DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf
- QwQ-32B-Q4_K_M.gguf


### Problem description & steps to reproduce

Run the llama.cpp server then pass a chat completion request with `n_predict = -2` (until context filled)
curl --location 'http://localhost:8080/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer no-key' \
--data '{
    "messages": [
        {
            "role": "user",
            "content": "Vancouver is a city located on the northwestern coast of Canada. It is the largest city in the province of British Columbia, flanked by th

[... truncated for brevity ...]

---

## Issue #N/A: Differences with the llama tokenizer

**Link**: https://github.com/ggml-org/llama.cpp/issues/167
**State**: closed
**Created**: 2023-03-15T16:45:04+00:00
**Closed**: 2023-03-20T15:21:55+00:00
**Comments**: 19
**Labels**: bug

### Description

In this case the llama.cpp and the llama tokenizers produce different output:

```
main: prompt: 'This is 🦙.cpp'
main: number of tokens in prompt = 10
     1 -> ''
  4013 -> 'This'
   338 -> ' is'
 29871 -> ' '
   243 -> '�'
   162 -> '�'
   169 -> '�'
   156 -> '�'
 29889 -> '.'
  8223 -> 'cpp'
```

Meanwhile the llama tokenizer produces:

```
text = "This is 🦙.cpp"
t = tokenizer.encode(text, bos=True, eos=False)

[1, 910, 338, 29871, 243, 162, 169, 156, 29889, 8223]
```

So in one case "This" is encoded as 4013 and other as 910. I have verified that both ids decode to the same text:

```
t1 = tokenizer.decode([4013])
t2 = tokenizer.decode([910])
print(t1, [int(b) for b in bytes(t1, "UTF-8")])
print(t2, [int(b) for b in bytes(t2, "UTF-8")])

This [84, 104, 105, 115]
This [84, 104, 105, 115]
```

I am not sure if this causes any significant differences in the generation but it may be a good idea to check it.

---

## Issue #N/A: `--instruct` CLI argument broken without prompt

**Link**: https://github.com/ggml-org/llama.cpp/issues/2744
**State**: closed
**Created**: 2023-08-23T15:39:59+00:00
**Closed**: 2023-08-23T15:59:34+00:00
**Comments**: 2
**Labels**: bug

### Description

On master the `--instruct` CLI argument is broken if no prompt is provided (nothing happens and the program is unresponsive). The problem is caused by 6381d4e110bd0ec02843a60bbeb8b6fc37a9ace9 . For testing I provided only the `--model` and `--instruct` CLI arguments.

---

## Issue #N/A: Bug: SwiftUI example does not work on simulator.

**Link**: https://github.com/ggml-org/llama.cpp/issues/10089
**State**: closed
**Created**: 2024-10-29T19:27:35+00:00
**Closed**: 2025-03-20T07:33:51+00:00
**Comments**: 1
**Labels**: bug, low severity

### Description

### What happened?

Previously, when `model_params.n_gpu_layers = 0` metal backend was **not initialized**. **Now** even if model_params.n_gpu_layers = 0, metal backend initialization is **still performed**, which is terminated by the following error:

```
ggml_metal_init: error: load pipeline error: Error Domain=CompilerError Code=2 "only 14 constant buffers binding are supported in the simulator but 25 were used" UserInfo={NSLocalizedDescription=only 14 constant buffers binding are supported in the simulator but 25 were used}
ggml_backend_metal_device_init: error: failed to allocate context
llama_new_context_with_model: failed to initialize Metal backend
```

### Name and Version

version:  b3982

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

---

## Issue #N/A: Eval bug: Release `b4524` breaks serving of `granite-code` models

**Link**: https://github.com/ggml-org/llama.cpp/issues/11500
**State**: closed
**Created**: 2025-01-29T20:52:26+00:00
**Closed**: 2025-01-31T08:24:30+00:00
**Comments**: 2
**Labels**: bug

### Description

### Name and Version

Changes made to Chat Template support in release `b4524` of llama.cpp break serving of `granite-code` models.

```
./bin/llama-cli --version
version: 4524 (6171c9d2)
built with Intel(R) oneAPI DPC++/C++ Compiler 2025.0.4 (2025.0.4.20241205) for x86_64-unknown-linux-gnu
```

### Operating systems

Linux

### GGML backends

SYCL, CPU

### Hardware

```
clinfo -l
Platform #0: Intel(R) OpenCL
 `-- Device #0: Intel(R) Core(TM) Ultra 7 155H
Platform #1: Intel(R) OpenCL Graphics
 `-- Device #0: Intel(R) Arc(TM) Graphics
```

### Models

Granite Code 3b & 8b

```
granite-code:3b
granite-code:8b
```

### Problem description & steps to reproduce

1. Build llama.cpp from release `b4523` and observe that despite a warning message, the server will work -

   Run with CPU -

   ```
   ./bin/llama-server --model ~/granite-code:3b --host 0.0.0.0 
   ```

   Run with GPU -

   ```
   ./bin/llama-server --model ~/granite-code:3b --host 0.0.0.0 --n-gpu-layers 999 --flash-attn --ctx-

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: Streaming with tools causes pydantic-ai to mess up tool name

**Link**: https://github.com/ggml-org/llama.cpp/issues/13774
**State**: closed
**Created**: 2025-05-25T11:51:57+00:00
**Closed**: 2025-05-26T13:56:50+00:00
**Comments**: 5
**Labels**: bug

### Description

### Name and Version

version: 5481 (d785f9c1)
built with cc (GCC) 15.1.1 20250425 for x86_64-pc-linux-gnu

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell

```

### Problem description & steps to reproduce

Testing the recently merged support for tools with streaming [12379](https://github.com/ggml-org/llama.cpp/pull/12379) with pydantic-ai.

Not really sure if it's a bug in pydantic-ai or llama.cpp or a misunderstanding on my side. The problem is that pydantic-ai interprets the tool name in the tool deltas as partial and concatenates the deltas to build the final tool name. However, the deltas returned from the api contain the complete tool name in each delta, only the arguments part is chunked. So a tool name that should be `final_result` becomes `final_resultfinal_resultfinal_result` (in case there were 3 deltas).

https://github.com/pydantic/pydantic-ai/blob/cbc6d5755ac67f25712deb089b5c91a36a9ad00f/pyd

[... truncated for brevity ...]

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

## Issue #N/A: Fix quantize_row_q4_1() with ARM_NEON

**Link**: https://github.com/ggml-org/llama.cpp/issues/876
**State**: closed
**Created**: 2023-04-10T14:40:14+00:00
**Closed**: 2023-04-10T16:30:16+00:00
**Comments**: 0
**Labels**: bug, high priority

### Description

It is currently bugged. See results of `quantize-stats` on M1:

```
$  ./quantize-stats -m models/7B/ggml-model-f16.bin 
Loading model
llama.cpp: loading model from models/7B/ggml-model-f16.bin
llama_model_load_internal: format     = ggjt v1 (latest)
llama_model_load_internal: n_vocab    = 32000
llama_model_load_internal: n_ctx      = 256
llama_model_load_internal: n_embd     = 4096
llama_model_load_internal: n_mult     = 256
llama_model_load_internal: n_head     = 32
llama_model_load_internal: n_layer    = 32
llama_model_load_internal: n_rot      = 128
llama_model_load_internal: f16        = 1
llama_model_load_internal: n_ff       = 11008
llama_model_load_internal: n_parts    = 1
llama_model_load_internal: model size = 7B
llama_model_load_internal: ggml ctx size =  59.11 KB
llama_model_load_internal: mem required  = 14645.07 MB (+ 2052.00 MB per state)
llama_init_from_file: kv self size  =  256.00 MB
note: source model is f16
testing 291 layers with max size 1310

[... truncated for brevity ...]

---

## Issue #N/A: Llava functions compiled as extern "C" throw exceptions

**Link**: https://github.com/ggml-org/llama.cpp/issues/7073
**State**: open
**Created**: 2024-05-04T13:34:07+00:00
**Comments**: 0
**Labels**: bug, good first issue

### Description

Basically this:
llama.cpp\examples\llava\clip.cpp(1277,13): warning : 'clip_model_load' has a non-throwing exception specification but can still throw [-Wexceptions]
llama.cpp\examples\llava\clip.cpp(2075,5): warning : 'clip_n_mmproj_embd' has a non-throwing exception specification but can still throw [-Wexceptions]

As these are library exported functions and wrapped in extern "C", they should not allow exceptions to cross the boundary. C language has no idea what to do with them.

Compiled with clang-cl in windows.

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

## Issue #N/A: Error: inlining failed in call to ‘always_inline’ ‘_mm256_cvtph_ps’ on x86_64 - better support for different x86_64 CPU instruction extensions

**Link**: https://github.com/ggml-org/llama.cpp/issues/196
**State**: closed
**Created**: 2023-03-16T04:17:08+00:00
**Closed**: 2023-03-30T08:31:50+00:00
**Comments**: 35
**Labels**: bug, performance, hardware, build

### Description

When I compile with make, the following error occurs
```
inlining failed in call to ‘always_inline’ ‘_mm256_cvtph_ps’: target specific option mismatch
   52 | _mm256_cvtph_ps (__m128i __A)
```

Error will be reported when executing `cc  -I.   -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -msse3   -c ggml.c -o ggml.o` .
But the error of executing `cc  -I.   -O3 -DNDEBUG -std=c11   -fPIC -pthread  -msse3   -c ggml.c -o ggml.o` will not occur.
Must `-mavx` be used with `-mf16c`?

---
OS: Arch Linux x86_64
Kernel: 6.1.18-1-lts

---

## Issue #N/A: Bug:  n_ctx will reuse n_ctx_train when --ctx_size not set and make deepseek-v2 models meet out of memory crash even on a small output length.

**Link**: https://github.com/ggml-org/llama.cpp/issues/8817
**State**: closed
**Created**: 2024-08-02T05:54:10+00:00
**Closed**: 2024-11-02T13:18:57+00:00
**Comments**: 12
**Labels**: bug, bug-unconfirmed, medium severity

### Description

### What happened?

deepseek-v2 model will meet out of memory issue with the kv buffer size allocating about 43G with a 160K context length from the model.  But when you set the -c or --ctx_size 2048, then the inference can  work normally.

### Name and Version

./build/bin/llama-cli -m deepseek-v2-lite-chat-q4_0.gguf -p "how to build  a website?" -n 32 -e -ngl 29 -sm none
Linux build on master branch :c8a0090922bad576623de4aae227717085249262 

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

---

## Issue #N/A: Bug: Speculative Decoding "Segmentation fault (core dumped)"

**Link**: https://github.com/ggml-org/llama.cpp/issues/10176
**State**: closed
**Created**: 2024-11-04T23:06:38+00:00
**Closed**: 2024-11-14T09:44:16+00:00
**Comments**: 13
**Labels**: bug, low severity

### Description

### What happened?

Hey all, I wanted to report a segmentation fault issue with llama-speculative. I have never once gotten this executable to work; I don't believe it is my command, as I have tried copy-pasting the speculative example commands as well.

### Name and Version

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 3 CUDA devices:
  Device 0: Tesla P40, compute capability 6.1, VMM: yes
  Device 1: Tesla P40, compute capability 6.1, VMM: yes
  Device 2: Tesla P40, compute capability 6.1, VMM: yes
version: 4031 (d5a409e5)
built with cc (Ubuntu 13.2.0-23ubuntu4) 13.2.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
./LLM/llama.cpp/llama-speculative \
-m /home/ultimis/LLM/Models/mradermacher/Meta-Llama-3.1-70B-Instruct-i1-GGUF/Meta-Llama-3.1-70B-Instruct.i1-Q4_K_M.gguf \
-md /home/ultimis/LLM/Models/hugging-quants/Llama-3.2-1B-Instru

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

## Issue #N/A: Accelerate.h not found on mac m1

**Link**: https://github.com/ggml-org/llama.cpp/issues/279
**State**: closed
**Created**: 2023-03-19T03:01:45+00:00
**Closed**: 2023-07-06T21:20:11+00:00
**Comments**: 8
**Labels**: bug, build

### Description

```
(base) dave@macbook-pro llama.cpp % make
I llama.cpp build info:
I UNAME_S:  Darwin
I UNAME_P:  arm
I UNAME_M:  arm64
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -DGGML_USE_ACCELERATE
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread
I LDFLAGS:   -framework Accelerate
I CC:       Apple clang version 12.0.5 (clang-1205.0.22.9)
I CXX:      Apple clang version 12.0.5 (clang-1205.0.22.9)

cc  -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -DGGML_USE_ACCELERATE   -c ggml.c -o ggml.o
ggml.c:115:10: fatal error: 'Accelerate/Accelerate.h' file not found
#include <Accelerate/Accelerate.h>
         ^~~~~~~~~~~~~~~~~~~~~~~~~
1 error generated.
make: *** [ggml.o] Error 1

(base) dave@macbook-pro llama.cpp % uname -a
Darwin macbook-pro.lan 22.3.0 Darwin Kernel Version 22.3.0: Mon Jan 30 20:38:37 PST 2023; root:xnu-8792\
.81.3~2/RELEASE_ARM64_T6000 arm64
```


About this Mac says "MacOS 13.2.1"

Do I need to install th

[... truncated for brevity ...]

---

## Issue #N/A: Bug: convert-hf-to-gguf.py on Gemma model ValueError: Duplicated key name 'tokenizer.chat_template'

**Link**: https://github.com/ggml-org/llama.cpp/issues/7923
**State**: closed
**Created**: 2024-06-13T19:09:23+00:00
**Closed**: 2024-07-21T01:53:03+00:00
**Comments**: 3
**Labels**: bug, low severity

### Description

### What happened?

When trying to convert

https://huggingface.co/SakanaAI/DiscoPOP-zephyr-7b-gemma/

I get the error in the title, but it's only defined a single time in tokenizer_config.json:

https://huggingface.co/SakanaAI/DiscoPOP-zephyr-7b-gemma/blob/main/tokenizer_config.json#L59

Verified locally with `cat *.json | grep chat_template` and I only get the one result

Is it somehow trying to load it twice?

Looks like when Gemma is initialized, it runs _set_vocab_sentencepiece(), which runs special_vocab.add_to_gguf (which pulls in the chat_template), and then it also again runs special_vocab.add_to_gguf

but that would mean it's been broken since April 16..

https://github.com/ggerganov/llama.cpp/pull/6689

### Name and Version

b3145 ubuntu 22.04

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
INFO:hf-to-gguf:Loading model: DiscoPOP-zephyr-7b-gemma
INFO:gguf.gguf_writer:gguf: This GGUF file is for Little Endia

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: softmax may get error answer when src0->ne[3]!=1 on cuda

**Link**: https://github.com/ggml-org/llama.cpp/issues/10683
**State**: open
**Created**: 2024-12-06T08:08:13+00:00
**Comments**: 1
**Labels**: bug

### Description

### Name and Version

$./llama-cli --version
version: 4267 (f112d198)
built with cc (Ubuntu 13.2.0-23ubuntu4) 13.2.0 for x86_64-linux-gnu

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

Test code

### Problem description & steps to reproduce


test-backend-ops run failed when I add test case： 

```c++
test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, { 32, 4, 1, 32 }, true, 0.1f, 8.0f));  
```

result:
```c++
SOFT_MAX(type=f32,ne=[32,4,1,32],mask=1,scale=0.100000,max_bias=8.000000): [SOFT_MAX] NMSE = 0.021202819 > 0.000001000 [1;31mFAIL[0m
```

Does this situation not exist or is it really a bug?
If it is a real bug, I think it is because of the calculation of slope, the cuda code is a little different with the c code, i am not sure which implementation is right.

```c++
cuda:
// h(rowx/nrows_y) ranges from 0 to ne02*ne03
const float slope = get_alibi_slope(max_bias, rowx/nrows_y, n_head_log2, m0, m1);


[... truncated for brevity ...]

---

## Issue #N/A: segfault in `simple.cpp`

**Link**: https://github.com/ggml-org/llama.cpp/issues/3753
**State**: closed
**Created**: 2023-10-23T22:58:17+00:00
**Closed**: 2023-10-27T14:38:27+00:00
**Comments**: 1
**Labels**: bug

### Description

# Expected Behavior

`./simple.cpp` with TheBloke's `Llama-2-7b-Chat-GGUF` should run without issue.

# Current Behavior

`./simple ~/.cache/huggingface/hub/models--TheBloke--Llama-2-7b-Chat-GGUF/blobs/08a5566d61d7cb6b420c3e4387a39e0078e1f2fe5f055f3a03887385304d4bfa`

(https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF)

results in
`Hello my name isSegmentation fault (core dumped)`

The model works fine with `main`.

I'm running ubuntu latest with everything up to date. compiled with `make` (no cuda, etc.).

The line that fails is 

> llama.cpp: 1453 (`llama_kv_cache_find_slot`)
> ```cpp
> cache.cells[cache.head + i].seq_id.insert(batch.seq_id[i][j]);
> ```

The initilization of `llama_batch::seq_id` in `simple.cpp` seems suspect - but I'm not nearly knowlegeable about what `seq_id` should be to fix it.

```cpp
    llama_batch batch = llama_batch_init(512, 0, 1);

    // evaluate the initial prompt
    batch.n_tokens = tokens_list.size();

    for (int

[... truncated for brevity ...]

---

