# stale_but_active - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 30
- Closed Issues: 0

### Label Distribution

- stale: 30 issues
- bug-unconfirmed: 18 issues
- enhancement: 8 issues
- server/webui: 1 issues

---

## Issue #N/A: Misc. bug: llama-server drops multi-part content for final assistant message

**Link**: https://github.com/ggml-org/llama.cpp/issues/14137
**State**: open
**Created**: 2025-06-12T00:15:47+00:00
**Comments**: 0
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

version: 5640 (2e89f76b)
built with Apple clang version 17.0.0 (clang-1700.0.13.3) for arm64-apple-darwin24.5.0

### Operating systems

Mac, Windows

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell
llama-server -m <any> --chat-template chatml
```

### Problem description & steps to reproduce

Endpoints that support OpenAI format seem to handle multi-part message content fine in most cases; however, when the final message is an assistant message, the content ends up being dropped. I tracked this down to the assistant prefill logic:

https://github.com/ggml-org/llama.cpp/blob/2e89f76b7af2c0b827be785e445f2e2b3e52e1ca/tools/server/utils.hpp#L774

I think this should handle `content_parts` unless there's a specific reason for it to be unsupported. I can see an argument that it would be odd for a prefill message to be anything but a single string, but it's still a surprising limitation (in my case, the input was a single

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: Potential out of bound in rerank

**Link**: https://github.com/ggml-org/llama.cpp/issues/13549
**State**: open
**Created**: 2025-05-14T19:50:38+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

version: 5387 (3198405e)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### Operating systems

_No response_

### Which llama.cpp modules do you know to be affected?

_No response_

### Command line

```shell

```

### Problem description & steps to reproduce

[llama_context](https://github.com/ggml-org/llama.cpp/blob/f5170c1d7a66222ca7c75d2022fec3ed87257e0b/src/llama-context.cpp#L807) resize the rerank output to size 1 while [here](https://github.com/ggml-org/llama.cpp/blob/017f10b5fa630a013ec4f9936e410a60d4f460d5/examples/embedding/embedding.cpp#L69) we still normalize it as if we have full embedding vector. I found this problem happened randomly in python binding but cannot reproduce it in cpp. Not sure if it is a bug in cpp side.

### First Bad Commit

_No response_

### Relevant log output

```shell

```

---

## Issue #N/A: Feature Request: add a new repo for convertion of gguf

**Link**: https://github.com/ggml-org/llama.cpp/issues/14027
**State**: open
**Created**: 2025-06-05T11:34:53+00:00
**Comments**: 2
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Hello.

I would like to split convert_hf_to_gguf.py to a new repo. Can you create another package just for conversion.

### Motivation

I am always frustruated when I want to convert gguf files because I have to run:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py ./OUT-MODEL/ --outfile model.gguf --outtype q8_0
```

While I just want to convert. I do not need the llamacpp inference model.

### Possible I

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Ability to pack multiple GGUFs into single one

**Link**: https://github.com/ggml-org/llama.cpp/issues/13028
**State**: open
**Created**: 2025-04-19T19:09:58+00:00
**Comments**: 4
**Labels**: enhancement, stale

### Description

### Feature Description

From an idea brought up by @ggerganov in this discussion: https://github.com/ggml-org/llama.cpp/discussions/11139#discussioncomment-11783418

While it is **NOT** a good idea to pack both mmproj + text models (because vision support is still messy atm), we still have some interesting use cases:

- For TTS models, this can be useful because some models may requires more than 2 GGUFs to run (for ex. Sesame CSM requires backbone, decoder and Mimi models)
- For phi-4-mm model, while the mmproj can't be packed, it is still interesting to pack the LoRA adapters and the text model together
- There are some techniques which use LoRA to recover quality loss due to quantization, it can be useful to pack LoRA with the model (though, I don't know how effective this can be, cc @compilade )
- Some models having more than 1 modality (i.e.Phi-4-mm with both audio+vision input), so could be useful to pack audio encoder and vision encoder into single GGUF

### Motivation

I creat

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: (MAC) fail in `GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H96,         flash_attn_ext_q8_0_h96,         has_simdgroup_mm);`

**Link**: https://github.com/ggml-org/llama.cpp/issues/14110
**State**: open
**Created**: 2025-06-10T21:08:26+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

B5727 

### Operating systems

Mac

### GGML backends

Metal

### Hardware

M4 Mac Studio 

### Models

Qwen 2.5 1.7b

### Problem description & steps to reproduce

load the model and crash at
```
GGML_METAL_ADD_KERNEL(GGML_METAL_KERNEL_TYPE_FLASH_ATTN_EXT_Q8_0_H96,         flash_attn_ext_q8_0_h96,         has_simdgroup_mm);
```

Thread 7: EXC_BAD_ACCESS (code=1, address=0x4e29444af118)

### First Bad Commit

_No response_

### Relevant log output

```shell
llama_model_load_from_file_impl: using device Metal (Apple M4 Max) - 49151 MiB free
llama_model_loader: loaded meta data with 26 key-value pairs and 339 tensors from /Users/animax/Library/Developer/Xcode/DerivedData/LocalLLM-fpkqjzkzghleumgxashmyivglcsj/Build/Products/Debug/model.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen2
llama_m

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: (webui) read data from /props endpoint and use it on the webui

**Link**: https://github.com/ggml-org/llama.cpp/issues/11717
**State**: open
**Created**: 2025-02-06T16:27:15+00:00
**Comments**: 3
**Labels**: enhancement, server/webui, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Not sure yet how we will use it, just noting this idea here so I don't forget

### Motivation

N/A

### Possible Implementation

_No response_

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

## Issue #N/A: Cmake minor bug: Confusing ggml-cpu: -march=native log message when using explicit -march flags and LLAMA_NATIVE=OFF

**Link**: https://github.com/ggml-org/llama.cpp/issues/14058
**State**: open
**Created**: 2025-06-07T15:47:51+00:00
**Comments**: 0
**Labels**: bug-unconfirmed, stale

### Description

### Git commit

Hi to all,

I'm encountering a confusing message in the CMake configuration log when trying to compile llama.cpp with a specific target CPU architecture. This is particularly problematic for cross-compilation or when ensuring specific optimizations for deployment environments.

My Setup:
- Build Environment CPU: AMD Ryzen 7 3700U (Zen 2 architecture).
- Target Deployment CPU: AMD Ryzen 9 7945HX (Zen 4 architecture, znver4).
- Goal: Compile llama.cpp binary (and its statically linked ggml components) to specifically target znver4 for optimal performance on the production node.

CMake Command Used:
```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_NATIVE=OFF \
    -DCMAKE_C_FLAGS="-march=znver4" \
    -DCMAKE_CXX_FLAGS="-march=znver4" \
    -DBUILD_SHARED_LIBS=OFF
```
Observed CMake Output (relevant section):
```
-- The C compiler identification is GNU 14.3.0
-- The CXX compiler identification is GNU 14.3.0
-- Detecting C compiler ABI info
-- Detecting C com

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: "error: invalid argument: /bin/sh" when using Docker image

**Link**: https://github.com/ggml-org/llama.cpp/issues/14019
**State**: open
**Created**: 2025-06-05T00:22:13+00:00
**Comments**: 0
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

Image: `ghcr.io/ggml-org/llama.cpp:server-cuda-b5583`
Platform: Google Cloud Run

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell

```

### Problem description & steps to reproduce

I am using a small Dockerfile based on the official image (to include a model in the image, but for the purpose of this issue I removed that part since the bug doesn't depend on it):

```dockerfile
FROM ghcr.io/ggml-org/llama.cpp:server-cuda-b5583

CMD [ \
  "--gpu-layers", "999", \
  "--host", "0.0.0.0", \
  "-hf", "ggml-org/gemma-3-1b-it-GGUF" \
  "--port", "8080", \
  "--verbose", \
]
```

I build it on Google Cloud Build using this config:

```yaml
steps:
  - name: "gcr.io/cloud-builders/docker"
    args: ["build", "--tag", "${_IMAGE}", "."]

images: ["${_IMAGE}"]

substitutions:
  _IMAGE: "us-central1-docker.pkg.dev/${PROJECT_ID}/repo/test:1"

options:
  dynamicSubstitutions: true
  machineType: "E2_HI

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: Gemma3 decode and update_slots fail with parallel slots

**Link**: https://github.com/ggml-org/llama.cpp/issues/14097
**State**: open
**Created**: 2025-06-10T09:12:28+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

PS C:\Sources\llama.cpp\build\bin\Release> .\llama-server.exe --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070 Ti SUPER, compute capability 8.9, VMM: yes
version: 5636 (c9c75ff8)
built with MSVC 19.43.34808.0 for x64

### Operating systems

Windows

### GGML backends

CUDA

### Hardware

Ryzen 7 7800X3D + RTX 4070 Ti Super

### Models

[Gemma-3-27b-it](https://huggingface.co/unsloth/gemma-3-27b-it-GGUF)

### Problem description & steps to reproduce

When I run llama-server with np > 1 and Gemma3 model, it spams with `decode: failed to find a memory slot for batch of size` and `srv  update_slots: failed to find free space in the KV cache, retrying with smaller batch size, i = 0, n_batch = 256, ret = 1` when multiple slots are utilized. This does not happen with a single slot or when only one out of multiple slots is used.

### First Bad Commit

_

[... truncated for brevity ...]

---

## Issue #N/A: Metrics should not include : in Prometheus metric names

**Link**: https://github.com/ggml-org/llama.cpp/issues/14150
**State**: open
**Created**: 2025-06-12T13:18:44+00:00
**Comments**: 0
**Labels**: stale

### Description

Hi,

I noticed that all Prometheus metrics exposed by the library are currently prefixed with `llamacpp:` (e.g., `llamacpp:prompt_tokens_total`). 
However, according to the [Prometheus metric naming guidelines](https://prometheus.io/docs/practices/naming/), I think the right (i.e. idiomatic) patter would be:
`llamacpp_prompt_tokens_total`

Would you consider updating the metric naming to follow this convention? I'd be happy to help contribute a PR if it's welcome.

Thanks!

---

## Issue #N/A: Eval bug: RWKV inference with llama-parallel gets wrong output with lmhead offloaded to GPU

**Link**: https://github.com/ggml-org/llama.cpp/issues/14211
**State**: open
**Created**: 2025-06-16T10:24:15+00:00
**Comments**: 0
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

$ ./build/bin/llama-cli --version                                                                       
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
version: 5674 (d7da8dc8)
built with cc (GCC) 15.1.1 20250425 for x86_64-pc-linux-gnu

### Operating systems

Linux

### GGML backends

CUDA

### Hardware

Ryzen 7900x + RTX4090

### Models

https://huggingface.co/zhiyuan8/RWKV-v7-0.1B-G1-GGUF/blob/main/rwkv7-0.1B-g1-F16.gguf

### Problem description & steps to reproduce

(Update: With Metal and Vulkan backends, offloading all layers with llama-parallel works flawlessly
It seems that this problem is CUDA-specific)

Using rwkv7-0.1B-g1-F16.gguf with 12 repeating layers and 1 output layer, it outputs correctly when running with `-ngl 12` (aka not offloading the output layer) :
```
./build/bin/llama-parallel -m ./rwkv7-0.1B-

[... truncated for brevity ...]

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

## Issue #N/A: Misc. bug: evaluate_and_capture_cuda_graph NULL POINTER DEREFERENCE

**Link**: https://github.com/ggml-org/llama.cpp/issues/14186
**State**: open
**Created**: 2025-06-15T01:26:34+00:00
**Comments**: 0
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

Release b5664
ggml-cuda.cu line 2659 exception during call to evaluate_and_capture_cuda_graph

### Operating systems

Windows

### Which llama.cpp modules do you know to be affected?

Other (Please specify in the next section)

### Command line

```shell

```

### Problem description & steps to reproduce


`if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || 
    node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
    continue;
   }`

Runtime exception attempting to dereference a null pointer node->buffer

Function does not check to evaluate if node->buffer is null, there are other evaluations here related to node->op, the first evaluation ggml_is_empty(node) might be the more appropriate place to check if the buffer is null. I'm not really sure. I implemented the change below in my local source, recompiled, and the issue vanished.

`if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || 

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: llama-server: a flag for limiting input image size

**Link**: https://github.com/ggml-org/llama.cpp/issues/14216
**State**: open
**Created**: 2025-06-16T14:31:51+00:00
**Comments**: 0
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

A flag for limiting the maximum input image size for vision models, resizing the images if necessary, which may help avoiding OOM issues.

### Motivation

Certain vision models like Mistral Small 3.1 support large images which apparently can balloon token usage significantly (beyond the context memory used by the text model; could be a bug) in llama-server and cause OOM with resulting crash. If we could limit maximum image resolution to a specific value, it might be possible to avoid this problem.


[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: full-cuda docker build needs ldconfig before launching llama-*

**Link**: https://github.com/ggml-org/llama.cpp/issues/14195
**State**: open
**Created**: 2025-06-15T13:17:09+00:00
**Comments**: 0
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

docker pull ghcr.io/ggml-org/llama.cpp:full-cuda-b5664


### Operating systems

_No response_

### Which llama.cpp modules do you know to be affected?

_No response_

### Command line

```shell

```

### Problem description & steps to reproduce

docker pull ghcr.io/ggml-org/llama.cpp:full-cuda-b5664

Fail case, without ldconfig

docker run --gpus all --runtime=nvidia -v ./models:/models  -p 0.0.0.0:8080:8080     ghcr.io/ggml-org/llama.cpp:full-cuda --run  -m /models/google_gemma-3-27b-it-Q8_0.gguf -ngl 99
ggml_cuda_init: failed to initialize CUDA: forward compatibility was attempted on non supported HW
load_backend: loaded CUDA backend from /app/libggml-cuda.so
load_backend: loaded CPU backend from /app/libggml-cpu-skylakex.so
warning: no usable GPU found, --gpu-layers option will be ignored


Ok case, with ldconfig

docker run --gpus all --runtime=nvidia -v ./models:/models  -p 0.0.0.0:8080:8080  --entrypoint /bin/bash   ghcr.io/ggml-org/llama.cpp:full-cuda   -c 

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: Can't run Qwen3-32B Q4_K_XL

**Link**: https://github.com/ggml-org/llama.cpp/issues/13298
**State**: open
**Created**: 2025-05-04T11:24:21+00:00
**Comments**: 9
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

build: 5273 (8ae5ebcf) with gcc-14 (Homebrew GCC 14.2.0_1) 14.2.0 for x86_64-pc-linux-gnu

### Operating systems

Linux

### GGML backends

CUDA

### Hardware

2x T4

### Models

https://huggingface.co/unsloth/Qwen3-32B-GGUF/blob/main/Qwen3-32B-UD-Q4_K_XL.gguf

### Problem description & steps to reproduce

NaN perplexity and completely trashed output while using [this model](https://huggingface.co/unsloth/Qwen3-32B-GGUF/blob/main/Qwen3-32B-UD-Q4_K_XL.gguf)

### First Bad Commit

_No response_

### Relevant log output

```shell
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: Tesla T4, compute capability 7.5, VMM: yes
  Device 1: Tesla T4, compute capability 7.5, VMM: yes
build: 5273 (8ae5ebcf) with gcc-14 (Homebrew GCC 14.2.0_1) 14.2.0 for x86_64-pc-linux-gnu
llama_model_load_from_file_impl: using device CUDA0 (Tesla T4) - 14992 MiB free
llama_model_load_from_file_impl: using de

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Improve model load time when using the RPC backend

**Link**: https://github.com/ggml-org/llama.cpp/issues/12954
**State**: open
**Created**: 2025-04-15T07:54:42+00:00
**Comments**: 6
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Load model faster when using one or several RPC servers

### Motivation

The local cache of the `rpc-server` made things better but there is still room for improvements.

### Possible Implementation

We may explore storing pre-computed hashes in GGUF and avoid loading the entire model on the main host.

---

## Issue #N/A: Support Hybrid Models

**Link**: https://github.com/ggml-org/llama.cpp/issues/12331
**State**: open
**Created**: 2025-03-11T11:46:29+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

Last commit


### Operating systems

Linux

### GGML backends

CUDA

### Hardware

threadripper 7980x rtx 5090/w7900 dual slot

### Models

https://developer.nvidia.com/blog/hymba-hybrid-head-architecture-boosts-small-language-model-performance/

### Problem description & steps to reproduce

Hybrid models not supported:
Support:
[Hymba](https://developer.nvidia.com/blog/hymba-hybrid-head-architecture-boosts-small-language-model-performance/)
A hybrid attention mechanism combining local sliding window attention and global attention.
Grouped-query attention (GQA).
A mix of global and local rotary embeddings.

### First Bad Commit

_No response_

### Relevant log output

```shell
not load correctly tensors
```

---

## Issue #N/A: Misc. bug: Stuck while loading the model

**Link**: https://github.com/ggml-org/llama.cpp/issues/14114
**State**: open
**Created**: 2025-06-11T02:21:34+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

[b5298](https://github.com/ggml-org/llama.cpp/releases/tag/b5298)

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell
./llama-server \
    --model /mnt/ssd/models/gguf/Openhands-32B-Q8_0/all-hands_openhands-lm-32b-v0.1-Q8_0.gguf \
    -a oh32q8 \
    --host 0.0.0.0 \
    --port 9000 \
    --api-key Llh123456@ \
    --ctx-size 128000 \
    --no-webui \
    --n-gpu-layers 65 \
    --mlock \
    --tensor-split "0.15,0.15,0.25,0.15,0.15,0.15" \
    --main-gpu 0 \
    --flash-attn \
    --defrag-thold 0.2 \
    --split-mode layer
```

```shell
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
./llama-server \
    --model /mnt/ssd/models/gguf/DSR1Q38BQ8/DeepSeek-R1-0528-Qwen3-8B-UD-Q8_K_XL.gguf \
    -a dsq3q8 \
    --host 0.0.0.0 \
    --port 9001 \
    --api-key Llh123456@ \
    --ctx-size 55000 \
    --no-webui \
    --n-gpu-layers 37 \
    --mlock \
    #--tensor-split "0,0,

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: Failure to allocate buffer with ROCm 6.4

**Link**: https://github.com/ggml-org/llama.cpp/issues/14178
**State**: open
**Created**: 2025-06-13T21:04:19+00:00
**Comments**: 13
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

root@llama-0:/app# ./llama-server --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: Radeon RX 7900 XT, gfx1100 (0x1100), VMM: no, Wave Size: 32
load_backend: loaded ROCm backend from /app/libggml-hip.so
load_backend: loaded CPU backend from /app/libggml-cpu-icelake.so
version: 5662 (fb85a288)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

libllama (core library), llama-cli, llama-server

### Command line

```shell
/app/llama-server --port ${PORT}
      -m /data/Qwen3-30B-A3B-UD-Q4_K_XL.gguf
      -ngl 99 -t 6
      --cache-type-k q8_0
      --cache-type-v q8_0
      --ctx-size 32768
      --flash-attn
```

### Problem description & steps to reproduce

* Build llama.cpp with ROCm 6.4
* Attempt to load large model (e.g `Qwen3-30B-A3B-UD-Q4_K_XL.gguf`)
* llama.

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: --cache-reuse no longer seems to be caching prompt prefixes

**Link**: https://github.com/ggml-org/llama.cpp/issues/14113
**State**: open
**Created**: 2025-06-11T02:11:50+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

**Affected:**
Version at commit: https://github.com/ggml-org/llama.cpp/commit/b7a17463ec190aeee7b9077c606c910fb4688b84

**Not affected:**
Version at commit: https://github.com/ggml-org/llama.cpp/commit/c6a2c9e7411f54b0770b319740561bbd6a2ebd27

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-server

### Problem description & steps to reproduce

I had open this bug in `oobabooga/text-generation-webui`: https://github.com/oobabooga/text-generation-webui/issues/7060

The issue being that prompt prefixes were no longer being used in the following requests.

I confirmed that `--cache-reuse 1` was being passed on, so that wasn't the issue.
After reverting to the previous version of the WebUI (which ships an older version of `llama.cpp`), the prompts started to be cached again.

So, this seems to point to being a bug with `llama.cpp`.

### First Bad Commit

It looks like there may have been a commit between https://github.com/g

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: Model produces gibberish or repeated output when using `-sm row` on CUDA

**Link**: https://github.com/ggml-org/llama.cpp/issues/14075
**State**: open
**Created**: 2025-06-08T23:22:38+00:00
**Comments**: 0
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

version: 5605 (5787b5da)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### Operating systems

Linux

### GGML backends

CUDA

### Hardware

EPYC 7352 + 4x RTX Quadro 5000 16GB

### Models

https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF/tree/main/DeepSeek-R1-Distill-Llama-70B-Q5_K_M

### Problem description & steps to reproduce

When using `-sm row` with `-fa` with latest container of `llama.cpp:server-cuda` the generated output is just `GGGGGGGGGGGG` repeated or gibberish on a different model, when running this inference, the last GPU in the system (ID: 3) was pinned at 100% usage for a long time for each token while the other GPU's sat idle as per `nvidia-smi`. the issue persists whether using `-ub 128` or not. the symptom does look very much like #13545 so this might not be a backend issue, but thats a speculation and maybe that issue has a different cause.

configs used are as follows
```
        - '-m'
        -

[... truncated for brevity ...]

---

## Issue #N/A: Compile bug: numerous deprecation warnings when compiling in Termux

**Link**: https://github.com/ggml-org/llama.cpp/issues/14011
**State**: open
**Created**: 2025-06-04T12:24:53+00:00
**Comments**: 0
**Labels**: bug-unconfirmed, stale

### Description

### Git commit

~/_ai/llama.cpp $ git rev-parse HEAD                 3ac67535c86e2fc43e4eddf594412acc370bbb04

### Operating systems

Other? (Please let us know in description)

### GGML backends

CPU

### Problem description & steps to reproduce

as stated in the title when compiling on Termux numerous warnings are produced even if targets compilation ends fine.

Following is the output of the build setup and compilation steps:

### First Bad Commit

_No response_

### Compile command

```shell
~/_ai/llama.cpp $ cmake -B build -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -G Ninja                                                   -- The C compiler identification is Clang 20.1.4
-- The CXX compiler identification is Clang 20.1.4   -- Detecting C compiler ABI info                     -- Detecting C compiler ABI info - done
-- Check for working C compiler: /data/data/com.termux/files/usr/bin/cc - skipped           

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: support FP8 data type in llama.cpp

**Link**: https://github.com/ggml-org/llama.cpp/issues/14020
**State**: open
**Created**: 2025-06-05T02:34:12+00:00
**Comments**: 0
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Currently, the FP8 data type is the default data type for models like DeepSeek, which is supposed to have better accuracy than INT8 while they have the same size. vLLM already supports fp8 data type.

File the issue for request the support of fp8 in llama.cpp.

### Motivation

with same level model size, FP8 is supposed to have better accuracy than int8. 

### Possible Implementation

_No response_

---

## Issue #N/A: Misc. bug: prompt as pasted content in the server

**Link**: https://github.com/ggml-org/llama.cpp/issues/14251
**State**: open
**Created**: 2025-06-17T22:55:41+00:00
**Comments**: 0
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

version: 5684 (6adc3c3e)

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell

```

### Problem description & steps to reproduce

When I paste a long text into the server, it appears as "pasted content"
When I press "Enter" after pasting, it says "Please enter a message"
I have to add something below the pasted content to send the query.
Shouldn't the pasted content be sufficient as a prompt?

### First Bad Commit

_No response_

### Relevant log output

```shell

```

---

## Issue #N/A: Refactor: (clip.cpp) identify and regroup pre-processing strategies

**Link**: https://github.com/ggml-org/llama.cpp/issues/13077
**State**: open
**Created**: 2025-04-23T13:12:09+00:00
**Comments**: 2
**Labels**: stale

### Description

### Background Description

Currently, `clip_image_preprocess` still looks quite messy.

From a graphic designer perspective, this function is purely just a "photoshop in cpp", its main purpose is to preprocess a given image before sending it to the transformer. The preprocess involves: crop / resize / pad the given image.

Currently, there are some strategies to preprocess an image:
- Resize to a fixed (square) size and add padding if the ratio is not square (used by llava 1.5, gemma 3, GLM)  
  Note: llava 1.5 use a gray-ish color for padding, while the rest use black color
- Allow dynamic resolution / ratio, but limit max size (used by qwen2vl, pixtral)  
  Image will still need to be resized to the nearest multiply of patch size
- Crop the image into slices, aka llava-uhd (used by llava 1.6, minicpm-v)

### Possible Refactor Approaches

Make an enum, split into dedicated function and give them good naming.

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

## Issue #N/A: Feature Request: Support Llama-Nemotron-Nano-VL-8B-V1

**Link**: https://github.com/ggml-org/llama.cpp/issues/14015
**State**: open
**Created**: 2025-06-04T16:13:31+00:00
**Comments**: 0
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Nvidia just released a new VLM: https://huggingface.co/nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1

- Text model: [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- Vision encoder: [nvidia/C-RADIOv2-H](https://huggingface.co/nvidia/C-RADIOv2-H)

On `master` (`3e63a58e`), the command:

```zsh
python llama.cpp/convert_hf_to_gguf.py --outfile /opt/workspace/gguf/Llama-Nemotron-Nano-VL-8B-V1-Q8_0.gguf --outtype bf16 /opt/workspace/hf/Llama-Nemotron-Nano-VL-8B-V1/
```

curren

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: LLAMA-SERVER is 40% slower than LLAMA-CLI when using identical parameters including -ot option for tensor offloading

**Link**: https://github.com/ggml-org/llama.cpp/issues/14201
**State**: open
**Created**: 2025-06-15T18:50:50+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 1: Quadro M2000, compute capability 5.2, VMM: yes
version: 5614 (8f47e25f)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu


### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell
CUDA_VISIBLE_DEVICES="0," \
numactl --physcpubind="8,10,12,14, 24,26,28,30, 9,11,13,15, 25,27,29,31" --membind=1 /home/ai/LLAMA_CPP/8f47e25f56e9792093b7497c68e9f80bab82ed19/llama.cpp/build/bin/llama-server \
--model /mnt/AI/LLM/DeepSeek-R1-UD-Q2_K_XL/DeepSeek-R1-UD-Q2_K_XL-00001-of-00006.gguf \
--threads 16 \
--n-gpu-layers 99 \
--override-tensor ".ffn_.*_exps.=CPU"

--cpunodebind=1 can be used instead of --physcpubind="8,10,12,14, 24,26,28,30, 9,11,13,15, 25,27,29,31" to the same effect. Ess

[... truncated for brevity ...]

---

