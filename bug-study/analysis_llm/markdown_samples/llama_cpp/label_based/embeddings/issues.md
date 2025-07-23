# embeddings - issues

**Total Issues**: 3
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 2

### Label Distribution

- embeddings: 3 issues
- bug: 2 issues
- server: 2 issues
- good first issue: 1 issues
- server/webui: 1 issues
- bug-unconfirmed: 1 issues

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

