# examples - issues

**Total Issues**: 3
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 2

### Label Distribution

- good first issue: 3 issues
- examples: 3 issues
- enhancement: 2 issues
- documentation: 1 issues
- help wanted: 1 issues

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

