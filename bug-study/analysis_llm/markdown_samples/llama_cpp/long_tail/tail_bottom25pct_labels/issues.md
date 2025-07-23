# tail_bottom25pct_labels - issues

**Total Issues**: 34
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 4
- Closed Issues: 30

### Label Distribution

- stale: 16 issues
- bug-unconfirmed: 9 issues
- enhancement: 9 issues
- good first issue: 7 issues
- 3rd party: 4 issues
- nix: 4 issues
- server/webui: 4 issues
- medium severity: 3 issues
- embeddings: 3 issues
- documentation: 3 issues

---

## Issue #N/A: Bug: High CPU usage and bad output with flash attention on ROCm

**Link**: https://github.com/ggml-org/llama.cpp/issues/8893
**State**: closed
**Created**: 2024-08-06T18:02:29+00:00
**Closed**: 2024-10-06T01:07:37+00:00
**Comments**: 8
**Labels**: bug-unconfirmed, stale, medium severity, 3rd party

### Description

### What happened?

While flash attention works well for my python ui (https://github.com/curvedinf/dir-assistant/) on an nvidia system, it produces bad results on my AMD system. My AMD system has a 7900XT with ROCm 6.1.2 on Ubuntu 22.04. When FA is off, inference is almost instant with high t/s and no CPU usage. When FA is on, the CPU utilization is 100% for 1-2 minutes, and then tokens are generated slowly and are incorrect (in my case, it always produces "################################" for a long time).

### Name and Version

Using python bindings via llama-cpp-python 0.2.84. Readme says llama.cpp version is [ggerganov/llama.cpp@4730faca618ff9cee0780580145e3cbe86f24876](https://github.com/ggerganov/llama.cpp/commit/4730faca618ff9cee0780580145e3cbe86f24876)

### What operating system are you seeing the problem on?

Linux

### Relevant log output

_No response_

---

## Issue #N/A: Bug: Model generating result sometimes but most of the times it doesnt 

**Link**: https://github.com/ggml-org/llama.cpp/issues/8269
**State**: closed
**Created**: 2024-07-03T06:24:30+00:00
**Closed**: 2024-07-03T08:27:34+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, medium severity, 3rd party

### Description

### What happened?

So, I am currently working with the local 'mistral-7b-q4' gguf model using 'llamacpp'. Although I can confirm that the model is active on the server but still I have encountered some issues during testing. Specifically, when I provide a small prompt , the model occasionally generates a response, but more often than not , it produces an empty string for the same prompt.

At this stage, I am unsure whether this behaviour is a result of the latest update, an expected characteristic of the model, or if there might be an error in my approach. 

This is how I am calling the model:

`

# Load LLM
llm = LlamaCpp(
    model_path="Models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_gpu_layers=-1,
    temperature=0.1,
    top_p=0.7,
    n_ctx=16000,
    max_tokens=4096,
    frequency_penalty=0.2,
    presence_penalty=0.5,
    stop=["\n"]
)`

these are the outputs I am getting. First one is without a response and second one is with the reponse. 
![Microsof

[... truncated for brevity ...]

---

## Issue #N/A: Bug: CUDA error: out of memory - Phi-3 Mini 128k prompted with 20k+ tokens on 4GB GPU

**Link**: https://github.com/ggml-org/llama.cpp/issues/7885
**State**: closed
**Created**: 2024-06-11T19:50:26+00:00
**Closed**: 2024-08-01T01:07:07+00:00
**Comments**: 28
**Labels**: bug-unconfirmed, stale, 3rd party

### Description

### What happened?

I get a CUDA out of memory error when sending large prompt (about 20k+ tokens) to Phi-3 Mini 128k model on laptop with Nvidia A2000 4GB RAM. At first about 3.3GB GPU RAM and 8GB CPU RAM is used by ollama, then the GPU ram usage slowly rises (3.4, 3.5GB etc.) and after about a minute it throws the error when GPU ram is exhaused probably (3.9GB is latest in task manager). The inference does not return any token (as answer) before crashing. Attaching server log. Using on Win11 + Ollama 0.1.42 + VS Code (1.90.0) + Continue plugin (v0.8.40).

The expected behavior would be not crashing and maybe rellocating the memory somehow so that GPU memory does not get exhausted. I want to disable GPU usage in ollama (to test for CPU inference only - I have 64GB RAM) but I am not able to find out how to turn the GPU off (even though I saw there is a command for it recently - am not able to find it again).

Actual error:
```
CUDA error: out of memory
  current device: 0, in fu

[... truncated for brevity ...]

---

## Issue #N/A: Bug:  Error while running model file (.gguf ) in LM Studio

**Link**: https://github.com/ggml-org/llama.cpp/issues/7779
**State**: closed
**Created**: 2024-06-05T20:15:20+00:00
**Closed**: 2024-07-21T01:07:09+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale, low severity, 3rd party

### Description

### What happened?

I'm encountering an error while trying to run a model in LM Studio. Below are the details of the error:

{
  "title": "Failed to load model",
  "cause": "llama.cpp error: 'error loading model architecture: unknown model architecture: 'clip''",
  "errorData": {
    "n_ctx": 2048,
    "n_batch": 512,
    "n_gpu_layers": 10
  },
  "data": {
    "memory": {
      "ram_capacity": "15.92 GB",
      "ram_unused": "7.46 GB"
    },
    "gpu": {
      "gpu_names": [
        "NVIDIA GeForce 940MX"
      ],
      "vram_recommended_capacity": "2.00 GB",
      "vram_unused": "1.64 GB"
    },
    "os": {
      "platform": "win32",
      "version": "10.0.22631",
      "supports_avx2": true
    },
    "app": {
      "version": "0.2.24",
      "downloadsDir": "C:\\Users\\hp\\.cache\\lm-studio\\models"
    },
    "model": {}
  }
}


![image](https://github.com/ggerganov/llama.cpp/assets/82998682/f408d74f-dee0-4347-983f-1c7d6c8e87e3)


### Name and 

[... truncated for brevity ...]

---

## Issue #N/A: nix: ci: fit into the new limits

**Link**: https://github.com/ggml-org/llama.cpp/issues/6346
**State**: closed
**Created**: 2024-03-27T13:19:02+00:00
**Closed**: 2024-05-17T01:06:32+00:00
**Comments**: 4
**Labels**: nix, stale

### Description

Most (all) of the nix-build jobs are being cancelled in progress since the quotas have changed. Adjust the workflows to fit in the new limits.

Context: since https://github.com/ggerganov/llama.cpp/pull/6243 the ci jobs are grouped by refs and cancelled together. The existing "Nix CI" job wasn't prepared for this for two reasons:

- It builds _many_ variants of `llama.cpp` in a single job.
- It only pushes the results to cachix after all of the builds have ended (not sure if it does the push in the "destructor" step after the cancellation).
- PRs from forks don't have access to the repo secrets so they don't push to cachix. However, it's plausible that these could make up the majority of all jobs?
- We're running pure nix-builds, meaning we can only cache store paths (results of complete and successful builds) not e.g. intermediate object files. This provides a strong guarantee that a passing CI means the build can be reproduced locally, but this also limits how much we can reus

[... truncated for brevity ...]

---

## Issue #N/A: nix: consider moving outputs to legacyPackages for lazier evaluation

**Link**: https://github.com/ggml-org/llama.cpp/issues/5681
**State**: closed
**Created**: 2024-02-23T11:17:41+00:00
**Closed**: 2024-05-05T01:06:43+00:00
**Comments**: 5
**Labels**: nix, stale

### Description

(This issue is for a conversation concerning specifically the `flake.nix`, open here for transparency)

### The proposal

The current approach to managing multiple variants of llama (BLAS, ROCm, CUDA) is to instantiate Nixpkgs several times in a [flake-parts](https://github.com/hercules-ci/flake-parts)  [module](https://github.com/ggerganov/llama.cpp/blob/15499eb94227401bdc8875da6eb85c15d37068f7/.devops/nix/nixpkgs-instances.nix), expose these instances [as arguments](https://github.com/ggerganov/llama.cpp/blob/15499eb94227401bdc8875da6eb85c15d37068f7/.devops/nix/nixpkgs-instances.nix#L9) for other flake-parts "`perSystem`" modules, and use them to populate the flake's `packages` and `checks`: https://github.com/ggerganov/llama.cpp/blob/15499eb94227401bdc8875da6eb85c15d37068f7/flake.nix#L150-L157

This means that running simple commands like `nix flake show` will trigger evaluating all of these several nixpkgs instances, tripling the evaluation cost.


We could instead remove 

[... truncated for brevity ...]

---

## Issue #N/A: CI: nix flake update: no permission to create a PR

**Link**: https://github.com/ggml-org/llama.cpp/issues/4851
**State**: closed
**Created**: 2024-01-10T01:52:06+00:00
**Closed**: 2024-03-18T07:56:23+00:00
**Comments**: 11
**Labels**: nix, stale

### Description

The weekly workflow for updating the nixpkgs revision pinned in the `flake.lock`, introduced in https://github.com/ggerganov/llama.cpp/pull/4709, fails with insufficient permissions: https://github.com/ggerganov/llama.cpp/actions/runs/7434790471/job/20229364773. The action is attempting to open a PR with the updated lock file. The action can be configured to use a separate token when creating the PR: https://github.com/ggerganov/llama.cpp/blob/c75ca5d96f902564cbbbdd7f5cade80d53c288bb/.github/workflows/nix-flake-update.yml#L22

Side-note: for the CI/nix-build workflow to also run automatically on these PRs a personal token would be required, https://github.com/DeterminateSystems/update-flake-lock#with-a-personal-authentication-token

@ggerganov: it's preferable to keep the action; can we use another token or do you know if there's another way to allow opening PRs from the specific workflow?

---

## Issue #N/A: gguf-py not in flake.nix?

**Link**: https://github.com/ggml-org/llama.cpp/issues/3824
**State**: closed
**Created**: 2023-10-28T04:32:14+00:00
**Closed**: 2024-04-02T01:12:48+00:00
**Comments**: 2
**Labels**: enhancement, nix, stale

### Description

When I try to run convert.py from the default x86_64-linux package installed from the Nix flake, it fails with the error `ModuleNotFoundError: No module named 'gguf'`. The gguf-py documentation from its subdirectory in this repo suggests to install it with pip, which really isn't ideal at all in a Nix/NixOS setup and also defeats the purpose of installing llama.cpp from the flake.nix file in the first place.  Am I just doing something wrong, or is this program missing from the flake outputs? In which case, I think it should be added. Thanks.

---

## Issue #N/A: GGML_ASSERT(seq_id < n_tokens && "seq_id cannot be larger than n_tokens with pooling_type == MEAN") failed

**Link**: https://github.com/ggml-org/llama.cpp/issues/13689
**State**: closed
**Created**: 2025-05-21T14:51:24+00:00
**Closed**: 2025-05-22T13:33:40+00:00
**Comments**: 7
**Labels**: bug, embeddings, server

### Description

@slaren Using build 'b5404', I am encountering the same issue with:
```console
[user@system]$ export LLAMA_ARG_HF_REPO=nomic-ai/nomic-embed-text-v2-moe-GGUF:Q4_K_M \
LLAMA_ARG_EMBEDDINGS=1 \
LLAMA_ARG_ENDPOINT_METRICS=1 \
LLAMA_ARG_NO_WEBUI=1 \
LLAMA_ARG_HOST=0.0.0.0 \
LLAMA_ARG_N_PARALLEL=4 \
LLAMA_ARG_ALIAS=embeddings-multilingual \
LLAMA_ARG_PORT=80 \
LLAMA_ARG_CACHE_TYPE_K=f16 \
LLAMA_ARG_FLASH_ATTN=0 \
LLAMA_ARG_CTX_SIZE=2048 \
LLAMA_ARG_BATCH=448 \
LLAMA_ARG_BATCH=512 \
LLAMA_ARG_THREADS=1 \
LLAMA_ARG_N_PREDICT=-1 \
LLAMA_ARG_N_GPU_LAYERS=0 \
LLAMA_ARG_NUMA=distribute \
LLAMA_ARG_MLOCK=0 \
LLAMA_ARG_ENDPOINT_SLOTS=1 \
LLAMA_ARG_NO_CONTEXT_SHIFT=0 \
LLAMA_ARG_UBATCH=512
[user@system]$ llama-server --seed 0 --temp 0.0
```

<details>
<summary>Full logs</summary>

```log
load_backend: loaded CPU backend from /app/libggml-cpu-haswell.so
warning: no usable GPU found, --gpu-layers option will be ignored
warning: one possible reason is that llama.cpp was c

[... truncated for brevity ...]

---

## Issue #N/A: server : crash when -b > -ub with embeddings

**Link**: https://github.com/ggml-org/llama.cpp/issues/12836
**State**: open
**Created**: 2025-04-08T18:28:48+00:00
**Comments**: 3
**Labels**: bug, good first issue, embeddings, server

### Description

> @ggerganov Ok, I did few tests and apparently there's an issue that is subject to a separate issue.
> 
> Using the following command:
> ```
> llama-server ... -ub 4096 -b 4096 -c 4096 -np 4
> ```
> 
> Everything works pretty much as expected. Amount of tokens that a task slot can handle appears to be `ub / np`. So in this example, each slot gets a 1024 tokens window. This does seem to give a nice boost depending on the embeddings chunking strategy (my current embeddings are up to 1024 tokens), but I haven't measured precisely yet.
> 
> However, using the following command:
> ```
> llama-server ... -ub 1024 -b 4096 -c 4096 -np 4
> ```
> 
> The server crashes with `GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens") failed` as soon as it receives the next batch of tasks:
> 
> ```
> ggml_vulkan: Found 1 Vulkan devices:
> ggml_vulkan: 0 = AMD Radeon RX 6600M (AMD proprietary driver) | uma: 0 | fp16: 1 | warp size:

[... truncated for brevity ...]

---

## Issue #N/A: Problem with multiple simultaneous API calls on the embeddings endpoint

**Link**: https://github.com/ggml-org/llama.cpp/issues/6722
**State**: closed
**Created**: 2024-04-17T10:05:07+00:00
**Closed**: 2024-04-17T14:11:12+00:00
**Comments**: 9
**Labels**: server/webui, bug-unconfirmed, embeddings

### Description

Hello,

I'm using separate instance of the server just to generate the embedding for the RAG pipelines. This instance is not used for general chat use, just for embeddings.

The issue is that while the API call to `http://<server>:8080/v1/embeddings` is not completed, which can last for a long time during document embedding, the server does not respond to the next API call to the same endpoint. 

I have tried to overcome this limitation by adding `--threads-http 4 --parallel 4` switches when running the server, like this:

`podman run -d --device nvidia.com/gpu=all -v /opt/models:/models:Z -p 8080:8000 ghcr.io/ggerganov/llama.cpp:server-cuda -m /models/uae-large-v1-f32.gguf --port 8000 --host 0.0.0.0 --n-gpu-layers 16 --threads 12 --threads-http 4 --parallel 4 --metrics --embedding --alias embedding --ctx-size 512`

This caused that after the first call I get this error and server crashes:
`GGML_ASSERT: llama.cpp:9612: seq_id < n_tokens && "seq_id cannot be larger than n_tok

[... truncated for brevity ...]

---

## Issue #N/A: examples : add configuration presets

**Link**: https://github.com/ggml-org/llama.cpp/issues/10932
**State**: open
**Created**: 2024-12-21T09:10:47+00:00
**Comments**: 5
**Labels**: documentation, enhancement, help wanted, good first issue, examples

### Description

## Description

I was recently looking for ways to demonstrate some of the functionality of the `llama.cpp` examples and some of the commands can become very cumbersome. For example, here is what I use for the `llama.vim` FIM server:

```bash
llama-server \
    -m ./models/qwen2.5-7b-coder/ggml-model-q8_0.gguf \
    --log-file ./service-vim.log \
    --host 0.0.0.0 --port 8012 \
    --ctx-size 0 \
    --cache-reuse 256 \
    -ub 1024 -b 1024 -ngl 99 -fa -dt 0.1
```

It would be much cleaner if I could just run, for example:

```bash
llama-server --cfg-fim-7b
```

Or if I could turn this embedding server command into something simpler:

```bash
# llama-server \
#     --hf-repo ggml-org/bert-base-uncased \
#     --hf-file          bert-base-uncased-Q8_0.gguf \
#     --port 8033 -c 512 --embeddings --pooling mean

llama-server --cfg-embd-bert --port 8033
```

## Implementation

There is already an initial example of how we can create such configuration presets:

```bash
llama-tts --tts-ou

[... truncated for brevity ...]

---

## Issue #N/A: llama : save downloaded models to local cache

**Link**: https://github.com/ggml-org/llama.cpp/issues/7252
**State**: closed
**Created**: 2024-05-13T09:20:51+00:00
**Closed**: 2024-12-13T16:23:30+00:00
**Comments**: 8
**Labels**: enhancement, good first issue, examples

### Description

We've recently introduced the `--hf-repo` and `--hf-file` helper args to `common` in https://github.com/ggerganov/llama.cpp/pull/6234:

```
ref #4735 #5501 #6085 #6098

Sample usage:

./bin/main \
  --hf-repo TinyLlama/TinyLlama-1.1B-Chat-v0.2-GGUF \
  --hf-file ggml-model-q4_0.gguf \
  -m tinyllama-1.1-v0.2-q4_0.gguf \
  -p "I believe the meaning of life is" -n 32

./bin/main \
  --hf-repo TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  -m tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  -p "I believe the meaning of life is" -n 32

Downloads `https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.2-GGUF/resolve/main/ggml-model-q4_0.gguf` and saves it to `tinyllama-1.1-v0.2-q4_0.gguf`

Requires build with `LLAMA_CURL`
```

Currently, the downloaded files via `curl` are stored in a destination based on the `--model` CLI arg.

If `--model` is not provided, we would like to auto-store the downloaded model files in a local cache, similar to what other frameworks like HF/transfor

[... truncated for brevity ...]

---

## Issue #N/A: llama : add `retrieval` example

**Link**: https://github.com/ggml-org/llama.cpp/issues/5692
**State**: closed
**Created**: 2024-02-23T18:46:29+00:00
**Closed**: 2024-03-25T07:38:23+00:00
**Comments**: 10
**Labels**: good first issue, examples

### Description

Since we now support embedding models in `llama.cpp` we should add a simple example to demonstrate retrieval functionality. Here is how it should work:

- load a set of text files (provided from the command line)
- split the text into chunks of user-configurable size, each chunk ending on a configurable stop string
- embed all chunks using an embedding model (BERT / SBERT)
- receive input from the command line, embed it and display the top N most relevant chunks based on cosine similarity between the input and chunk emebeddings

---

## Issue #N/A: Bug: Intel Arc - not working at all

**Link**: https://github.com/ggml-org/llama.cpp/issues/9106
**State**: closed
**Created**: 2024-08-20T19:45:26+00:00
**Closed**: 2024-12-17T01:07:43+00:00
**Comments**: 28
**Labels**: bug-unconfirmed, stale, SYCL, critical severity

### Description

### What happened?

Going through the manual - SYCL I mean. Everything compiles okay. Running it always thows an error. Can't make it work. OS used: Linux Gentoo. P.S. docker doesn't work either. P.P.S. device IS listed in the list.

### Name and Version

# ./build/bin/llama-cli --version
version: 3609 (2f3c1466)
built with Intel(R) oneAPI DPC++/C++ Compiler 2024.2.1 (2024.2.1.20240711) for x86_64-unknown-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
# ZES_ENABLE_SYSMAN=1 ./build/bin/llama-cli -m models/llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33 -sm none -mg 0
Log start
main: build = 3609 (2f3c1466)
main: built with Intel(R) oneAPI DPC++/C++ Compiler 2024.2.1 (2024.2.1.20240711) for x86_64-unknown-linux-gnu
main: seed  = 1724182694
llama_model_loader: loaded meta data with 19 key-value pairs and 291 tensors from models/llama-2-7b.Q4_0.gguf (version GGUF V2)
llama_

[... truncated for brevity ...]

---

## Issue #N/A: Bug: InvalidModule: Invalid SPIR-V module: input SPIR-V module uses extension 'SPV_INTEL_memory_access_aliasing' which were disabled by --spirv-ext option

**Link**: https://github.com/ggml-org/llama.cpp/issues/8551
**State**: closed
**Created**: 2024-07-18T04:49:20+00:00
**Closed**: 2024-09-06T01:07:04+00:00
**Comments**: 17
**Labels**: bug-unconfirmed, stale, SYCL, medium severity

### Description

### What happened?

Currently on Fedora 40 with Intel Arc A750.

Running the following:
```txt
ZES_ENABLE_SYSMAN=1 ./build/bin/llama-server \
-t 10 \
-ngl 20 \
-b 512 \
--ctx-size 16384 \
-m ~/llama-models/llama-2-7b.Q4_0.gguf \
--color -c 3400 \
--seed 42 \
--temp 0.8 \
--top_k 5 \
--repeat_penalty 1.1 \
--host :: \
--port 8080 \ 
-n -1 \
-sm none -mg 0
```

gives the following output:
```txt
INFO [                    main] build info | tid="140466031364096" timestamp=1721277895 build=3411 commit="e02b597b"
INFO [                    main] system info | tid="140466031364096" timestamp=1721277895 n_threads=10 n_threads_batch=-1 total_threads=28 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 0 | "
llama_model_loader: loaded meta data with

[... truncated for brevity ...]

---

## Issue #N/A: Bug: [SYCL] Inference not working correctly on multiple GPUs

**Link**: https://github.com/ggml-org/llama.cpp/issues/8294
**State**: closed
**Created**: 2024-07-04T11:46:57+00:00
**Closed**: 2024-09-07T01:07:06+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, stale, SYCL, high severity

### Description

### What happened?

I am using Llama.cpp + SYCL to perform inference on a multiple GPU server. However, I get a Segmentation Fault when using multiple GPUs. The same model can produce inference output correctly with single GPU mode.

```shell
git clone https://github.com/ggerganov/llama.cpp.git
source /opt/intel/oneapi/setvars.sh
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build --config Release -j -v

cd ~/llama.cpp/
./build/bin/llama-ls-sycl-device

## single gpu, ok
./build/bin/llama-cli -m ~/mistral-7b-v0.1.Q4_0.gguf -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e -ngl 33 -s 0 -sm none -mg 0

## multiple gpus, Segmentation Fault, core dumped
./build/bin/llama-cli -m ~/mistral-7b-v0.1.Q4_0.gguf -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e -ngl 33 -s 0 -sm layer
``` 

![image](https://github.com/ggerganov/llama.cpp/assets/16000946/f2ef0a11-6b54-43b2-acba-23624e1

[... truncated for brevity ...]

---

## Issue #N/A: Why is convert.py missing?

**Link**: https://github.com/ggml-org/llama.cpp/issues/7658
**State**: closed
**Created**: 2024-05-31T05:46:36+00:00
**Closed**: 2024-06-10T19:58:23+00:00
**Comments**: 15
**Labels**: documentation, script, python, high severity

### Description

### What happened?

Critical "non llama3" convert.py and change NOT in download of files.

ALSO:
It is unclear if "convert-hf-to-gguf.py"  supports what "convert.py" did . 

Does it convert llama, llama2, mistral or is "convert-legacy-llama.py" required?
Safetensor files are EVERYWHERE. (?)  [ RE: https://github.com/ggerganov/llama.cpp/pull/7430 ]

This critical action DID NOT OCCUR:
"Move convert.py to examples/convert-legacy-llama.py"

"examples/convert-legacy-llama.py" does not exist. 
(when downloading the zip files).

On another note why remove "convert.py" at all? 

-This breaks "bat files" and automation generation.
-This will break all colabs too.
-This will break any HF spaces that create GGUF files as well.
-This will create needless confusion.

If "convert-hf-to-gguf.py" (llama3) does everything convert.py did , just keep it as "convert.py" ?





### Name and Version

this is not applicable - core files missing.

### What operating system are you se

[... truncated for brevity ...]

---

## Issue #N/A: Update *-to-ggml.py scripts for new ggjt model format

**Link**: https://github.com/ggml-org/llama.cpp/issues/704
**State**: closed
**Created**: 2023-04-02T09:49:22+00:00
**Closed**: 2023-05-03T18:37:53+00:00
**Comments**: 1
**Labels**: script

### Description

See title, basically.

We should probably keep the option of generating the old formats.

Revert #690 when done.

Related: #545

---

## Issue #N/A: How to convert old ALPACA q4_0 model into ggjt format?

**Link**: https://github.com/ggml-org/llama.cpp/issues/701
**State**: closed
**Created**: 2023-04-02T08:29:38+00:00
**Closed**: 2023-04-04T19:32:30+00:00
**Comments**: 4
**Labels**: duplicate, script

### Description

I'm trying to use a python script, but it returns the following error:

d:\ALPACA2>python migrate-ggml-2023-03-30-pr613.py ggml-alpaca-7b-q4.bin ggml-alpaca-7b-q4-ggjt.bin
Traceback (most recent call last):
  File "d:\ALPACA2\migrate-ggml-2023-03-30-pr613.py", line 313, in <module>
    main()
  File "d:\ALPACA2\migrate-ggml-2023-03-30-pr613.py", line 274, in main
    tokens = read_tokens(fin, hparams)
  File "d:\ALPACA2\migrate-ggml-2023-03-30-pr613.py", line 135, in read_tokens
    word = fin.read(length)
ValueError: read length must be non-negative or -1


---

## Issue #N/A: RPTQ state of the art quantization

**Link**: https://github.com/ggml-org/llama.cpp/issues/1295
**State**: closed
**Created**: 2023-05-03T03:23:53+00:00
**Closed**: 2024-04-09T01:09:40+00:00
**Comments**: 2
**Labels**: generation quality, research 🔬, Less than 4 bits, stale

### Description

Per yuan etc all, RPTQ quant is state of the art down to 3bit

It would be good to implement RPTQ for llama and other c++ downstream projects

https://github.com/hahnyuan/RPTQ4LLM/blob/master/quantize/quantizer.py

https://arxiv.org/abs/2304.01089

---

## Issue #N/A: Variable bit rate quantization

**Link**: https://github.com/ggml-org/llama.cpp/issues/1256
**State**: closed
**Created**: 2023-04-30T16:46:25+00:00
**Closed**: 2023-06-07T08:02:32+00:00
**Comments**: 10
**Labels**: enhancement, generation quality, Less than 4 bits

### Description

Variable bit rate is commonly used in audio and video compression, so why not try on LLMs?

My guess is that a locally adaptive variable bit rate would require a major change to `ggml`. So, then, the least one can try is to see if using different number of bits in the different network layers would be beneficial.

As a first step, I simply changed `llama.cpp` to not quantize one of the tensor types in addition to `output.weight` (which is already known to have a significant impact on generation quality) and calculated perplexity for `Q2_4` quantization (see issue #1240). Picked 2-bit quantization because there the difference between a quantized and not quantized tensor will be largest, so it would be easiest to see the effect. The following table summarizes the results (PPL improvement is perplexity with `fp16` `output.weight` - perplexity with `fp16` `output weight` + indicated tensor, table is sorted in decreasing order of impact) 

| Tensor type | PPL improvement |
|---------

[... truncated for brevity ...]

---

## Issue #N/A: QX_4 quantization

**Link**: https://github.com/ggml-org/llama.cpp/issues/1240
**State**: closed
**Created**: 2023-04-29T19:44:03+00:00
**Closed**: 2023-06-07T08:03:06+00:00
**Comments**: 10
**Labels**: enhancement, generation quality, Less than 4 bits

### Description

### Summary

Use `16 x 8` "super-blocks" for quantization, having one `fp16` scale for the "super-block" and 16 quantized scales per 8 model weights. This is particularly useful for 2- and 3-bit quantization, but it also outperforms the existing 4-bit quantization schemes `Q4_0` and `Q4_2`.

### Details

The naming of existing `llama.cpp` quantizations follows the scheme `QX_Y`, where `X` is the number of bits used for the quants, and `Y` is `0, 1, 2,` or `3`.  When `Y` is even (0 or 2), model weights `x` are computed from the quants `q` as `x = d * q`. When `Y` is odd, then `x = m + d * q` is used. If we look at the integer part of `Y/2` (`[Y/2]`), then the number of weights in a quantization block is 32 (`Q4_0`, `Q4_1`, `Q5_0`) when `[Y/2] = 0`, and 16  (`Q4_2`, `Q4_3`) when `[Y/2] = 1`. From the [latest perplexity results](https://github.com/ggerganov/llama.cpp#quantization) one can see that quantization using blocks of 16 weights performs better than quantization that uses bl

[... truncated for brevity ...]

---

## Issue #N/A: Segmentation Fault on GPU

**Link**: https://github.com/ggml-org/llama.cpp/issues/7337
**State**: closed
**Created**: 2024-05-17T09:17:40+00:00
**Closed**: 2024-08-09T01:07:02+00:00
**Comments**: 10
**Labels**: training, stale

### Description

When I am trying to run the following finetuning command on GPU:
**nohup ../build/bin/finetune --model-base llama-3b-Q5_0.gguf --train-data "shakespeare.txt" --save-every 1 --adam-iter 2 --batch 4 --ctx 4 --lora-out ../../training/checkpoints/llama_3b_q5_ctx_4_batch_4_threads_6/lora.bin --checkpoint-in ../../training/checkpoints/llama_3b_q5_ctx_4_batch_4_threads_6/checkpoint.gguf --checkpoint-out ../../training/checkpoints/llama_3b_q5_ctx_4_batch_4_threads_6/checkpoint-ITERATION.gguf > ../../training/checkpoints/llama_3b_q5_ctx_4_batch_4_threads_6/training_logs.out -ngl 33**

I get **segmentation fault** error with ever increasing **nohup.out** file:

llama_model_loader: loaded meta data with 24 key-value pairs and 237 tensors from llama-3b-Q5_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_mo

[... truncated for brevity ...]

---

## Issue #N/A: Training Custom using train-text-from-scratch

**Link**: https://github.com/ggml-org/llama.cpp/issues/2049
**State**: closed
**Created**: 2023-06-29T15:21:57+00:00
**Closed**: 2024-04-09T01:08:35+00:00
**Comments**: 3
**Labels**: training, stale

### Description

Hi.

I have been trying to train some custom data using the base model file: open-llama-7B-open-instruct.ggmlv3.q4_0.bin

Here is the command Im running

`./bin/train-text-from-scratch \
        --vocab-model ../models/ggml-vocab.bin \
        --ctx 64 --embd 256 --head 8 --layer 16 \
        --checkpoint-in  ik.bin \
        --checkpoint-out ik.bin \
        --model-out ik.bin \
        --train-data ik.txt \
        -t 6 -b 16 -n 32 --seed 1 --adam-iter 16 \
        --print-details-interval 0 --predict 16 \
        --mem-model 12 --mem-compute 12  --use-flash`


I'm able to train the data but I have following 2 concerns:

1. How Can I pass multiple text files instead of 1 text file? I have about 1 lakh+ text file which I need to train the model. Should I combine the text files?
2. I have a GPU of 24GB VRAM. But its not able to utilize more than 1gb and hence the process of training is slow. 

---

## Issue #N/A: kubernetes example

**Link**: https://github.com/ggml-org/llama.cpp/issues/6546
**State**: open
**Created**: 2024-04-08T16:31:37+00:00
**Comments**: 18
**Labels**: enhancement, help wanted, server/webui, kubernetes

### Description

### Motivation

Kubernetes is widely used in the industry to deploy product and application at scale.

It can be useful for the community to have a `llama.cpp` [helm](https://helm.sh/docs/intro/quickstart/) chart for the server.

I have started several weeks ago, I will continue when I have more time, meanwhile any help is welcomed:

https://github.com/phymbert/llama.cpp/tree/example/kubernetes/examples/kubernetes

### References
- #6545


---

## Issue #N/A: server-cuda closes connection while still processing tasks

**Link**: https://github.com/ggml-org/llama.cpp/issues/6545
**State**: closed
**Created**: 2024-04-08T16:09:49+00:00
**Closed**: 2024-04-08T20:34:17+00:00
**Comments**: 5
**Labels**: server/webui, bug-unconfirmed, kubernetes

### Description

Issue to be published in the llama.cpp github: 


I am using the Docker Image ghcr.io/ggerganov/llama.cpp:server-cuda to deploy the server in a Kubernetes cluster in AWS using four A10G gpus. This is the configuration setup: 

>     - name: llama-cpp-server
>         image: ghcr.io/ggerganov/llama.cpp:server-cuda
>         args:
>         - "--model"
>         - "/models/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf"
>         - "--port"
>         - "8000"
>         - "--host"
>         - "0.0.0.0"
>         - "--ctx-size"
>         - "100000"
>         - "--n-gpu-layers"
>         - "256"
>         - "--cont-batching"
>         - "--parallel" 
>         - "10"
>         - "--batch-size"
>         - "4096"

(not sure if it adds context, but I'm using a persistentVolumeClaim where I download and persist the model)

I already reviewed the server readme and all the command line options and also tested different image tags for server-cuda from the past days. 

Based on

[... truncated for brevity ...]

---

## Issue #N/A: Working Fine-Tune Example?

**Link**: https://github.com/ggml-org/llama.cpp/issues/6361
**State**: closed
**Created**: 2024-03-28T06:44:04+00:00
**Closed**: 2024-06-21T01:07:11+00:00
**Comments**: 5
**Labels**: documentation, demo, stale

### Description

I am trying to find a working example of fine-tuning. 

- If I run the example from `https://github.com/ggerganov/llama.cpp/tree/master/examples/finetune`, the script can't find the model.
here is the error
```
main: seed: 1711608198
main: model base = 'open-llama-3b-v2-q8_0.gguf'
llama_model_load: error loading model: llama_model_loader: failed to load model from open-llama-3b-v2-q8_0.gguf

llama_load_model_from_file: failed to load model
llama_new_context_with_model: model cannot be NULL
Segmentation fault: 11
```

- If I try to use a model in `/models` folder such as 
```
./finetune \
        --model-base ./models/ggml-vocab-llama.gguf \
        --checkpoint-in  chk-lora-open-llama-3b-v2-q8_0-shakespeare-LATEST.gguf \
        --checkpoint-out chk-lora-open-llama-3b-v2-q8_0-shakespeare-ITERATION.gguf \
        --lora-out lora-open-llama-3b-v2-q8_0-shakespeare-ITERATION.bin \
        --train-data "shakespeare.txt" \
        --save-every 10 \
        --threads 6 

[... truncated for brevity ...]

---

## Issue #N/A: Server: add function calling API

**Link**: https://github.com/ggml-org/llama.cpp/issues/5588
**State**: closed
**Created**: 2024-02-19T13:47:28+00:00
**Closed**: 2024-06-16T01:07:14+00:00
**Comments**: 10
**Labels**: enhancement, demo, server/webui, stale

### Description

# Motivation

This subject is already brought up in https://github.com/ggerganov/llama.cpp/issues/4216 , but my initial research failed.

Recently, I discovered a new line of model designed specifically for this usage: https://github.com/MeetKai/functionary

This model can decide whether to call functions (and which function to be called) in a given context. The chat template looks like this:

```
{#v2.2#}
{% for message in messages %}
  {% if message['role'] == 'user' or message['role'] == 'system' %}
    {{ '<|from|>' + message['role'] + '\n<|recipient|>all\n<|content|>' + message['content'] + '\n' }}
  {% elif message['role'] == 'tool' %}
    {{ '<|from|>' + message['name'] + '\n<|recipient|>all\n<|content|>' + message['content'] + '\n' }}
  {% else %}
    {% set contain_content='no'%}
    {% if message['content'] is not none %}
      {{ '<|from|>assistant\n<|recipient|>all\n<|content|>' + message['content'] }}
      {% set contain_content='yes'%}
    {% endif %}

[... truncated for brevity ...]

---

## Issue #N/A: Compile bug: RISCV cross-compile warnings cause build failure

**Link**: https://github.com/ggml-org/llama.cpp/issues/12693
**State**: closed
**Created**: 2025-04-01T14:20:59+00:00
**Closed**: 2025-04-03T17:19:00+00:00
**Comments**: 7
**Labels**: good first issue, build, Riscv

### Description

### Git commit

9c4cef4602c77068e1c6b91b2d8e707b493f6fcf

### Operating systems

Linux

### GGML backends

CPU

### Problem description & steps to reproduce

I am updating the CI with cross-compile builds for RISCV regression tests (see #12428 ) and a build error is occurring due to some RISCV macros/functions. Since I am not familiar with RISCV functions in question, I am deferring this fix to folks who know that platform better.


### First Bad Commit

_No response_

### Compile command

```shell
Please see the github workflow here: https://github.com/ggml-org/llama.cpp/pull/12428/files#diff-245fd2c5accd266a35983ed2891af1c8f8b41af027aa393075f15a00b38ff817
```

### Relevant log output

```shell
[ 12%] Building C object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu-quants.c.o
/home/runner/work/llama.cpp/llama.cpp/ggml/src/ggml-cpu/ggml-cpu-quants.c: In function ‘ggml_vec_dot_q5_0_q8_0’:
/home/runner/work/llama.cpp/llama.cpp/ggml/src/ggml-cpu/ggml-cpu-quants.c:3141:19: error: impli

[... truncated for brevity ...]

---

## Issue #N/A: Support BitNet b1.58 ternary models

**Link**: https://github.com/ggml-org/llama.cpp/issues/5761
**State**: closed
**Created**: 2024-02-28T09:41:38+00:00
**Closed**: 2024-09-18T01:07:17+00:00
**Comments**: 90
**Labels**: enhancement, stale, Tensor Encoding Scheme

### Description

New paper just dropped on Arxiv describing a way to train models in 1.58 bits (with ternary values: 1,0,-1). Paper shows performance increases from equivalently-sized fp16 models, and perplexity nearly equal to fp16 models. Authors state that their test model is built on LLaMA architecture and can be easily adapted to llama.cpp.

[Edited to add: Further reading into it by fellow Redditors shows that we can't use this to quantize existing models trained to fp16. They'd have to be trained in this ternary mode from the start. But I think it would still be something that we should implement, because models of that flavor will be coming soon.]

This is all over Reddit /LocalLLaMA right now:

https://www.reddit.com/r/LocalLLaMA/comments/1b21bbx/this_is_pretty_revolutionary_for_the_local_llm/

I think, if my napkin math is right, it would let us run something like 120B models in 24 GB VRAM, or 30B in... 8 GB?

Please implement @ggerganov and friends!

https://arxiv.org/abs/2402.17

[... truncated for brevity ...]

---

## Issue #N/A: GGUF endianness cannot be determined from GGUF itself

**Link**: https://github.com/ggml-org/llama.cpp/issues/3957
**State**: open
**Created**: 2023-11-05T14:00:47+00:00
**Comments**: 17
**Labels**: enhancement, good first issue, breaking change

### Description

As of the time of writing, the big-endian support that was added in https://github.com/ggerganov/llama.cpp/pull/3552 doesn't encode the endianness within the file itself: 

https://github.com/ggerganov/llama.cpp/blob/3d48f42efcd05381221654376e9f6f69d76af739/gguf-py/gguf/gguf.py#L689-L698

This means that there is no way to distinguish a big-endian GGUF file from a little-endian file, which may cause some degree of consternation in the future if these files get shared around 😅 

The cleanest solution would be to add the endianness to the header - ideally, it would be in the metadata, but the reading of the metadata is dependent on the endianness - but that would be a breaking change.

Given that, my suggestion would be to use `FUGG` as the header for big-endian files so that a little-endian executor won't attempt to read it at all unless it knows how to deal with it. The same can go the other way, as well (a big-endian executor won't attempt to read a little-endian executor).

---

## Issue #N/A: falcon : speed-up prompt processing

**Link**: https://github.com/ggml-org/llama.cpp/issues/2850
**State**: closed
**Created**: 2023-08-28T09:51:27+00:00
**Closed**: 2023-09-15T08:09:25+00:00
**Comments**: 2
**Labels**: good first issue, performance, 🦅.

### Description

The performance of Falcon 7B should be comparable to LLaMA 7B since the computation graph is computationally very similar.

Here are the current numbers on M2 Ultra for LLaMA, LLaMA-v2 and Falcon 7B:

```bash
../scripts/run-all-perf.sh ${model} "f16 q8_0 q4_0"
```

| model                          |       size |     params | backend    | ngl | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| LLaMA 7B mostly F16            |  12.55 GiB |     6.74 B | Metal      | 999 | pp 512     |    665.95 ± 0.18 |
| LLaMA 7B mostly Q8_0           |   6.64 GiB |     6.74 B | Metal      | 999 | pp 512     |    630.28 ± 0.16 |
| LLaMA 7B mostly Q4_0           |   3.56 GiB |     6.74 B | Metal      | 999 | pp 512     |    632.32 ± 0.22 |
| LLaMA 7B mostly F16            |  12.55 GiB |     6.74 B | Metal      | 999 | tg 64      |     29.73 ± 0.01 |
| LLaMA 7B mostly Q8_0           |   6.64 GiB | 

[... truncated for brevity ...]

---

## Issue #N/A: 4bit 65B model overflow 64GB of RAM

**Link**: https://github.com/ggml-org/llama.cpp/issues/702
**State**: closed
**Created**: 2023-04-02T08:37:42+00:00
**Closed**: 2023-04-19T08:20:48+00:00
**Comments**: 7
**Labels**: need more info, performance, linux

### Description

# Prerequisites

I am running the latest code. Development is very rapid so there are no tagged versions as of now.
I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior
During inference, there should be no or minimum disk activities going on, and disk should not be a bottleneck once pass the model loading stage.

# Current Behavior
My disk should have a continuous reading speed of over 100MB/s, however, during the loading of the model, it only loads at around 40MB/s. After this very slow loading of Llama 65b model (converted from GPTQ with group size of 128), llama.cpp start to inference, however during the inference the programme continue to occupy t

[... truncated for brevity ...]

---

