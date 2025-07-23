# server - issues

**Total Issues**: 15
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 3
- Closed Issues: 12

### Label Distribution

- server: 15 issues
- good first issue: 5 issues
- bug: 4 issues
- enhancement: 4 issues
- server/api: 3 issues
- embeddings: 2 issues
- stale: 2 issues
- roadmap: 1 issues
- bug-unconfirmed: 1 issues
- high severity: 1 issues

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

## Issue #N/A: Misc. bug: server not exit after `missing result_output tensor` error

**Link**: https://github.com/ggml-org/llama.cpp/issues/11808
**State**: closed
**Created**: 2025-02-11T13:02:55+00:00
**Closed**: 2025-04-27T01:08:11+00:00
**Comments**: 2
**Labels**: stale, server

### Description

### Name and Version

While testing the rerank model on HF inference endpoint, we got this error: `GGML_ASSERT(strcmp(res->name, "result_output") == 0 && "missing result_output tensor") failed`

This is due to missing `LLAMA_ARG_RERANKING` (for reranking model) or `LLAMA_ARG_EMBEDDINGS` (for embeddings model).

The application is expected to edit after this error, but it still running which makes it a bit confused for end user.

<img width="1222" alt="Image" src="https://github.com/user-attachments/assets/ad78af84-cd9d-4e2d-9cb6-4c947347190e" />

**Expected behavior**: the server should exit once it get that error.

### Operating systems

_No response_

### Which llama.cpp modules do you know to be affected?

_No response_

### Command line

```shell
llama-server -m jina-rerank.gguf (do not add --rerank argument)
```

### Problem description & steps to reproduce

Run a jina-rerank model without `--rerank` flag

### First Bad Commit

_No response_

### Relevant log output

```shell
(as 

[... truncated for brevity ...]

---

## Issue #N/A: server : add support for multiple responses

**Link**: https://github.com/ggml-org/llama.cpp/issues/11142
**State**: open
**Created**: 2025-01-08T16:11:24+00:00
**Comments**: 2
**Labels**: server/api, server, roadmap

### Description

It would be very useful to add multi-response support per slot so that a single request would be able to generate `n` independent completions. This functionality is useful in different situations - for example, a FIM completion can provide multiple alternative suggestions at a smaller or equal compute cost compared to running them sequentially.

I think this can be implemented by adding multiple sequence id per slot (instead of having just one like we currently do). However, I am not sure how yet much complexity would be introduced to support this.

---

## Issue #N/A: Feature Request: Mapping model name to LoRA config

**Link**: https://github.com/ggml-org/llama.cpp/issues/11031
**State**: open
**Created**: 2025-01-01T19:07:56+00:00
**Comments**: 5
**Labels**: enhancement, good first issue, server

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

I came across this idea while working on #10994 

The idea is that we can maintain a list of model name mapped to LoRA config, for example:

```
{
    "llama-base":               [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.0}],
    "llama-story":              [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 0.0}],
    "llama-abliteration":       [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 1.0}],
    "llama-story-abliteration": [{"id": 0, "scale": 0.5}, {"id": 1, "scale": 0.5}]


[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: support `"encoding_format": "base64"` in the `*/embeddings` endpoints

**Link**: https://github.com/ggml-org/llama.cpp/issues/10887
**State**: closed
**Created**: 2024-12-18T10:50:45+00:00
**Closed**: 2024-12-24T20:33:05+00:00
**Comments**: 0
**Labels**: enhancement, good first issue, server/api, server

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

The OpenAI embeddings API supports returning the embeddings in `base64` format:

https://platform.openai.com/docs/api-reference/embeddings/create#embeddings-create-encoding_format

We should implement this option in the server and enable it both for the `/v1/embeddings` and `/embeddings` endpoints.

### Motivation

Reduce JSON payload and increase OAI compatibility.

### Possible Implementation

_No response_

---

## Issue #N/A: server : temperature sampling is not working

**Link**: https://github.com/ggml-org/llama.cpp/issues/9842
**State**: closed
**Created**: 2024-10-11T07:38:07+00:00
**Closed**: 2024-10-11T07:41:07+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, server/api, server, high severity

### Description

### What happened?

Using 1000000000000000 temperature does not affect model's response.
```python
import httpx

# Define the URL and the headers
url = 'http://localhost:8080/completion'
headers = {
    'Content-Type': 'application/json'
}

# Define the JSON payload with properly escaped newlines
data = {
    "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\nHow can I assist you today?<|im_end|>\n<|im_start|>user\nImplement fibbonaci in Python<|im_end|>\n<|im_start|>assistant\n",
    "n_predict": 128,
    "temperature": 1000000,
}

# Send the POST request using httpx with no timeout
response = httpx.post(url, json=data, headers=headers, timeout=None)

# Print the response from the server
print(response.json())
```

### Name and Version

d5cb86844f26f600c48bf3643738ea68138f961d

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

---

## Issue #N/A: Server UI bug: corrupted generation

**Link**: https://github.com/ggml-org/llama.cpp/issues/9836
**State**: closed
**Created**: 2024-10-11T03:55:47+00:00
**Closed**: 2024-11-29T01:09:57+00:00
**Comments**: 1
**Labels**: server/webui, stale, server, medium severity

### Description

### What happened?

Server somehow corrupted the prompt, so tokens at the end of the every line are lost.

Here is how I run server:
```shell
./build/bin/llama-server -m ~/Downloads/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf
```
Here is how I test CLI to ensure it is a server bug:
```shell
./build/bin/llama-cli -m ~/Downloads/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf -e -p "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi\!<|im_end|>\n<|im_start|>assistant\nHow can I assist you today?<|im_end|>\n<|im_start|>user\nImplement fibbonaci in Python<|im_end|>\n<|im_start|>assistant\n" -n 128 -t 7 -tb 8 --temp 0
```

<details><summary>Here is the output from the CLI</summary>
<p>

```
➜  llama.cpp git:(master) ✗ ./build/bin/llama-cli -m ~/Downloads/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf -e -p "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi\!<|im_end|>\n<|im_start|>assistant\nHow can I assist you today

[... truncated for brevity ...]

---

## Issue #N/A: server : remove system prompt support

**Link**: https://github.com/ggml-org/llama.cpp/issues/9811
**State**: closed
**Created**: 2024-10-09T19:10:10+00:00
**Closed**: 2024-10-12T11:51:55+00:00
**Comments**: 13
**Labels**: refactoring, server

### Description

The "system_prompt" related functionality is quite outdated and is introducing unnecessary complexity. It only sort of makes sense for non-finetuned models in order to save the computation of a common prefix when there are multiple parallel slots. But in practice, only finetuned models are utilized for this use case and they always require a chat template, which is incompatible with the current implementation of the system prompt. So in order to simplify the code a bit, we should remove the system prompt related functionality from the server.

---

## Issue #N/A: server : ability to disable context shift

**Link**: https://github.com/ggml-org/llama.cpp/issues/9390
**State**: closed
**Created**: 2024-09-09T14:52:29+00:00
**Closed**: 2024-09-23T20:23:55+00:00
**Comments**: 14
**Labels**: enhancement, server

### Description

### Feature Description

We can add an argument (for example, `--context-shift`, `--no-context-shift`) to enable/disable context shift.

If disabled:
- Requests bigger than context window will result in an error.
- `n_predict` for each sequence will be capped to `n_ctx - n_tokens_prompt`

Note: the behavior above is the same as official OAI API

### Motivation

We may want to disable it because:
- For users who doesn't know about this feature, it may degrade generation quality
- Currently, quantized KV cache doesn't work with context shift

### Possible Implementation

_No response_

---

## Issue #N/A: Bug: (Server) Cannot properly cancel a non-stream completion request

**Link**: https://github.com/ggml-org/llama.cpp/issues/9273
**State**: closed
**Created**: 2024-09-02T09:40:50+00:00
**Closed**: 2025-01-18T13:12:06+00:00
**Comments**: 1
**Labels**: bug, server, low severity

### Description

### What happened?

When using server completions (or chat completions) **without** stream, it is impossible to cancel the request midway.

## To reproduce the problem

1. Compile and run the server (any version), run with `--verbose` argument.
2. `curl -X POST http://localhost:8080/completion -vvv -d '{"prompt": "hi", "stream": false, "n_predict": 1000}'`
3. While it's still running, hit Ctrl+C to cancel the curl request
4. The server will still process the completion without being interrupted

Retry with `"stream": true`, now you will be able to interrupt the completion.

## Investigation

This is due to the fact that httplib is a blocking HTTP library, so there is no "client disconnect" event.

For non-stream API, our implementation is:

https://github.com/ggerganov/llama.cpp/blob/c6d4cb46559b359d2682cf2a002e7fe01bb7a767/examples/server/server.cpp#L2971-L2979

The problem is that, `.recv(id_task);` will block the current thread, so there is no way to detect the di

[... truncated for brevity ...]

---

## Issue #N/A: server: Bring back multimodal support

**Link**: https://github.com/ggml-org/llama.cpp/issues/8010
**State**: closed
**Created**: 2024-06-19T12:03:45+00:00
**Closed**: 2025-05-09T21:20:01+00:00
**Comments**: 51
**Labels**: enhancement, llava, server

### Description

Multimodal has been removed since https://github.com/ggerganov/llama.cpp/pull/5882

## Current llama.cpp multimodal roadmap

(update 9th april 2025)

- `mtmd` (**M**ul**T**i-**M**o**D**al) library (top prio 🔥 )
    - [x] Implement `libmtmd`: https://github.com/ggml-org/llama.cpp/pull/12849
    - [x] Support more models via `libmtmd` (top prio 🔥 ) : https://github.com/ggml-org/llama.cpp/pull/13012
    - [x] Support M-RoPE models via `libmtmd` (Qwen2VL, Qwen2.5VL) : https://github.com/ggml-org/llama.cpp/pull/13141
    - [x] Support audio input
    - [x] Use smart pointer in `clip.cpp` to avoid mem leak: https://github.com/ggml-org/llama.cpp/pull/12869
    - [x] ~~Add wrapper for `stb_image` to avoid polluting project with the big header file~~ --> Probably don't need since we're already having some helper in `libmtmd` acting as wrapper for stb_image
    - [x] Unify conversion scripts --> best case scenario: having `convert_hf_to_gguf.py` that can output both text + vision GGUF files --> 

[... truncated for brevity ...]

---

## Issue #N/A: Question:  Inconsistent Classification Results Between Command-Line and HTTP Server for LLaMA 3

**Link**: https://github.com/ggml-org/llama.cpp/issues/7585
**State**: closed
**Created**: 2024-05-28T07:07:35+00:00
**Closed**: 2024-05-30T11:52:35+00:00
**Comments**: 1
**Labels**: question, server

### Description

### Prerequisites

- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new useful question to share that cannot be answered within Discussions.

### Background Description

I get difference results if I use llama 3 with the HTTP Server than when I use ./main. For example, I am trying to classify job postings using the new prompt format (in german: its instructed to classify the job in brackets):
`
./main --ctx-size 9999  --color --interactive --model ../models/Meta-Llama-3-70B-Instruct-GGUF/Meta-Llama-3-70B-Instruct-Q4_K_M.gguf  --repeat_penalty 1.0 --n-gpu-layers 555  --prompt "<|start_header_id|>system<|end_header_id|> Deine Aufgabe ist es, Jobausschreibungen zu klassifizieren. Antworte mit der Job Branche in Klammern. Antworte nur in einem einzigen Wort<|eot_id|><|start_header_id|>user<|end_head

[... truncated for brevity ...]

---

## Issue #N/A: Server: completion_probabilities (tok_str and prob) seem to be broken

**Link**: https://github.com/ggml-org/llama.cpp/issues/7197
**State**: closed
**Created**: 2024-05-10T11:06:21+00:00
**Closed**: 2024-05-11T08:11:29+00:00
**Comments**: 8
**Labels**: bug, good first issue, server

### Description

Hello,

I am using the llama.cpp server and noticed strange behavior in the server responses.

When starting a server on commit 637e9a86 using `./server -m ../models/llama-2-7b-chat.Q4_K_M.gguf -c 4096 -ngl 1000 -np 1 -cb`, and using this curl command:
```bash
curl --request POST \
    --url http://localhost:8080/completion \
    --header "Content-Type: application/json" \
    --data '{"prompt": "Choose between A, B and C.\n\n","n_predict": 1, "n_probs": 10, "temperature": 0}'
```

I get the following json response:
```json
// commit hash 637e9a86
{
    "content": "A",
    "id_slot": 0,
    "stop": true,
    "model": "../models/llama-2-7b-chat.Q4_K_M.gguf",
    "tokens_predicted": 1,
    "tokens_evaluated": 12,
    "generation_settings":
    {
        ...
    },
    "prompt": "Choose between A, B and C.\n\n",
    "truncated": false,
    "stopped_eos": false,
    "stopped_word": false,
    "stopped_limit": true,
    "stopping_word": "",
    "tokens_cached

[... truncated for brevity ...]

---

## Issue #N/A: How can i get log probs in create_chat_completions in llama-cpp , I'm using logprobs=True as an attribute but still not getting Log Probabilities.

**Link**: https://github.com/ggml-org/llama.cpp/issues/6423
**State**: closed
**Created**: 2024-04-01T11:08:35+00:00
**Closed**: 2025-01-21T08:22:56+00:00
**Comments**: 10
**Labels**: good first issue, server

### Description

from llama_cpp import Llama



llm = Llama(model_path="/home/zadmin/.cache/lm-studio/models/TheBloke/MythoMax-L2-13B-GGUF/mythomax-l2-13b.Q8_0.gguf", logits_all=True,chat_format="chatml",n_ctx=10000)

def mytho_extraction():

    source_sentence = "That is a happy person"
    sentences = [
        "That is a very happy dog",
        "That is a very happy person",
        "Today is a sunny day"
    ]

    user_message_content = f"Source Sentence: {source_sentence}\nSentences to Match: {' | '.join(sentences)}\nPlease provide the sentence from the list which is the  best matches the source sentence."


    completion = llm.create_chat_completion(
        model="local-model", 
        messages=[
            {"role": "system", "content": "Give me Matched sentence with the source sentence"},
            {"role": "user", "content": user_message_content}
        ],
        temperature=0.7,
        logprobs= True
    )
    generated_sentence = completion

    print(

[... truncated for brevity ...]

---

