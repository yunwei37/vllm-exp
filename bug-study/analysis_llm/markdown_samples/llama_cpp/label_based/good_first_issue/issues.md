# good_first_issue - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 9
- Closed Issues: 21

### Label Distribution

- good first issue: 30 issues
- enhancement: 16 issues
- help wanted: 8 issues
- server/webui: 7 issues
- high priority: 5 issues
- documentation: 5 issues
- 🦙.: 3 issues
- build: 2 issues
- breaking change: 1 issues
- tts: 1 issues

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

## Issue #N/A: Store KV cache of computed prompts to disk to avoid re-compute in follow-up runs

**Link**: https://github.com/ggml-org/llama.cpp/issues/64
**State**: closed
**Created**: 2023-03-12T21:55:25+00:00
**Closed**: 2023-04-29T02:57:37+00:00
**Comments**: 10
**Labels**: enhancement, help wanted, good first issue, high priority, 🦙.

### Description

Idea from: https://github.com/ggerganov/llama.cpp/issues/23#issuecomment-1465308592

We can add a `--cache_prompt` flag that if added will dump the computed KV caches of the prompt processing to the disk in a file with name produced by the hash of the prompt. Next time you run, it will first check if we have stored KV cache for this hash and load it straight from disk instead of computing it.

Great task for contributing to the project!

---

## Issue #N/A: [Feature Request] Ability to rewind model evaluation by a fixed number of tokens

**Link**: https://github.com/ggml-org/llama.cpp/issues/1281
**State**: closed
**Created**: 2023-05-02T15:00:02+00:00
**Closed**: 2023-05-05T01:52:30+00:00
**Comments**: 13
**Labels**: enhancement, good first issue, high priority

### Description

The recent additions of the state and session APIs have made it possible to implement caching for llama models which has greatly improved the responsiveness in many applications.

The current APIs howeve still leave something to be desired, specifically it would be very useful to be able to rewind / rollback an evaluated model by a fixed number of tokens so a single longer saved state could be used to restore any shorter state.

---

## Issue #N/A: Models with multiple chat templates

**Link**: https://github.com/ggml-org/llama.cpp/issues/6484
**State**: closed
**Created**: 2024-04-04T15:32:14+00:00
**Closed**: 2024-04-18T11:49:02+00:00
**Comments**: 1
**Labels**: enhancement, good first issue

### Description

Hi all, Matt from Hugging Face here. Just to let you know, we've made a modification to chat templates. As of Transformers v4.39, the **tokenizer chat template field can now be a list containing multiple named templates**. For compatibility reasons, the dict is converted to a list of `{"name": "...", "template": "..."}` dicts for serialization in `tokenizer_config.json`, but we convert it to a single `dict` when we load it in the tokenizer itself.

We did this to support Command-R and Command-R+, as they used separate templates for general LLM use, tool-assisted generation, and RAG. Right now, this is hardcoded in the tokenization.py file for Command-R, but we will be moving it into the model repos itself very soon - a repo PR is already open [here](https://huggingface.co/CohereForAI/c4ai-command-r-v01/discussions/46/files), and we'll probably open one for Command-R+ too.

You can see how we're handling this [here](https://github.com/huggingface/transformers/blob/main/src/transform

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

## Issue #N/A: tts : add basic example for text-to-speech

**Link**: https://github.com/ggml-org/llama.cpp/issues/10173
**State**: closed
**Created**: 2024-11-04T18:53:25+00:00
**Closed**: 2024-12-18T17:27:22+00:00
**Comments**: 5
**Labels**: good first issue, tts

### Description

This new model seems suitable for integration: https://github.com/edwko/OuteTTS

We should add a very minimalistic example for generating audio with it. Ideally, we will implement the (audio tokens) -> (wav) from scratch.

---

## Issue #N/A: Create issue template for bug and enhancement issues

**Link**: https://github.com/ggml-org/llama.cpp/issues/239
**State**: closed
**Created**: 2023-03-17T13:38:57+00:00
**Closed**: 2023-03-21T17:50:50+00:00
**Comments**: 3
**Labels**: documentation, good first issue

### Description

The following is a proposed template for creating new issues. If people think the tone could be improved, I'd appreciate feedback!
___

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ ] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ ] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Please provide a detailed written description of what you were trying to do, and what you expected `lamma.cpp` to do.

# Curren

[... truncated for brevity ...]

---

## Issue #N/A: server: process prompt fairly accross slots

**Link**: https://github.com/ggml-org/llama.cpp/issues/6607
**State**: open
**Created**: 2024-04-11T10:23:54+00:00
**Comments**: 1
**Labels**: enhancement, help wanted, good first issue, server/webui

### Description

### Context

At the moment we implement a FIFO approach to batch prompt tokens. So if a large prompt is to be processed it blocks all other slots.

Proposal: implement a fair batch usage of prompt processing accross all pending slots.

References:
- https://github.com/ggerganov/llama.cpp/issues/4216#issuecomment-2043558080
- https://github.com/ggerganov/llama.cpp/issues/5851#issuecomment-1975120585


---

## Issue #N/A: server.cpp is not accepting parameter -tb N, --threads-batch N

**Link**: https://github.com/ggml-org/llama.cpp/issues/3473
**State**: closed
**Created**: 2023-10-04T14:08:35+00:00
**Closed**: 2023-10-11T19:42:23+00:00
**Comments**: 1
**Labels**: good first issue

### Description

# Expected Behavior

server.cpp should recognise parameters -tb / --threads-batch (as stated in the readme).

Please provide a detailed written description of what `llama.cpp` did, instead.

server.cpp doesn't recognise the -tb / --threads-batch parameter. 

I checked the code, this options seems indeed missing. 

PS: I can attempt adding it, if you agree... it would be a good task to get started on the code.



---

## Issue #N/A: Help populating the examples README.md files

**Link**: https://github.com/ggml-org/llama.cpp/issues/518
**State**: closed
**Created**: 2023-03-26T07:25:05+00:00
**Closed**: 2023-07-28T19:21:19+00:00
**Comments**: 2
**Labels**: documentation, help wanted, good first issue

### Description

For now I just added empty README.md files:

- https://github.com/ggerganov/llama.cpp/tree/master/examples/main
- https://github.com/ggerganov/llama.cpp/tree/master/examples/quantize
- https://github.com/ggerganov/llama.cpp/tree/master/examples/perplexity
- https://github.com/ggerganov/llama.cpp/tree/master/examples/embedding
- etc.

It would be great to add usage instructions and various tips and tricks for better experience for each example.

Great task for initial contributions

---

## Issue #N/A: Compile bug: ios swift xcode build error when upgrade to llama : use cmake for swift build 

**Link**: https://github.com/ggml-org/llama.cpp/issues/10747
**State**: open
**Created**: 2024-12-10T05:12:25+00:00
**Comments**: 41
**Labels**: help wanted, good first issue, build

### Description

### Git commit

$git rev-parse HEAD 43ed389a3f102517e6f7d5620d8e451e88afbf27

### Operating systems

Mac

### GGML backends

Metal

### Problem description & steps to reproduce

ios swift xcode build error when upgrade to

- https://github.com/ggerganov/llama.cpp/pull/10525

Before the upgrade, the code compiled successfully. After the upgrade, it throws a compilation error: "Cannot find type 'xxx' in scope."

<img width="1721" alt="image" src="https://github.com/user-attachments/assets/1bc2e76a-158a-4aa3-9755-855930f2f7ed">


### First Bad Commit

43ed389a3f102517e6f7d5620d8e451e88afbf27

### Relevant log output

```shell
/ios/llama.cpp.swift/LibLlama.swift:8:39 Cannot find type 'llama_batch' in scope

/ios/llama.cpp.swift/LibLlama.swift:12:37 Cannot find type 'llama_batch' in scope

/ios/llama.cpp.swift/LibLlama.swift:12:56 Cannot find type 'llama_token' in scope

/ios/llama.cpp.swift/LibLlama.swift:12:76 Cannot find type 'llama_pos' in scope

/ios/llama.cpp.swift/LibL

[... truncated for brevity ...]

---

## Issue #N/A: server: support control vectors

**Link**: https://github.com/ggml-org/llama.cpp/issues/6316
**State**: open
**Created**: 2024-03-26T07:25:43+00:00
**Comments**: 0
**Labels**: enhancement, good first issue, server/webui

### Description

### Motivation

It would be nice to support control vectors in the servers.


### Requirements
- Configure `gpt_params::control_vectors` from `common`
- Tests the feature using the framework

#### References
- A first attemp has been made here: #6289

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

## Issue #N/A: server: exit failure if `--embedding` is set with an incoherent `--ubatch-size`

**Link**: https://github.com/ggml-org/llama.cpp/issues/6263
**State**: open
**Created**: 2024-03-23T17:03:49+00:00
**Comments**: 5
**Labels**: enhancement, help wanted, good first issue, server/webui

### Description

### Context

there is no advantage to increase `n_batch` above `n_ubatch` with embeddings models with pooling, because the entire batch must fit in a physical batch (ie. `n_ubatch`). `n_batch` is always `>= n_ubatch`.

- See @slaren comment in: https://github.com/ggerganov/llama.cpp/pull/6254#discussion_r1536661327

### Proposition
Exit failure if `--embedding` is set and `--ubatch-size` != `--batch-size` in the `server` example. Probably also in the `retrieval` example in #6193.

Aldo probably KV `bert.context_size` must be taken into account.

---

## Issue #N/A: Feature Request: Add "tokens per second" information in the Web UI

**Link**: https://github.com/ggml-org/llama.cpp/issues/10502
**State**: closed
**Created**: 2024-11-25T18:37:33+00:00
**Closed**: 2024-12-11T19:52:15+00:00
**Comments**: 5
**Labels**: enhancement, good first issue, server/webui

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

The client should display prompt processing and text generations speeds.

### Motivation

I helps to investigate how different parameters affect the performance

### Possible Implementation

_No response_

---

## Issue #N/A: Bug: Unexpected output length (Only one token response!) when set configs "-n -2 -c 256" for llama-server

**Link**: https://github.com/ggml-org/llama.cpp/issues/9933
**State**: open
**Created**: 2024-10-18T06:41:56+00:00
**Comments**: 1
**Labels**: bug, good first issue, low severity

### Description

### What happened?

Hi there.
As suggested by the documents, config -n indicates the number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled), and -c indicates the context size.
However, when I use the following command to start a server:
```bash
./llama.cpp-b3938/build_gpu/bin/llama-server     -m ../models/Meta-Llama-3-8B-Instruct-Q4_0.gguf     -ngl 99 -n -2 -c 256
```
And Send a request with the following command:
```bash
curl --request POST     --url http://localhost:8080/completion     --header "Content-Type: application/json"     --data '{"prompt": "What is the meaning of life?"}'
```

I can get only one token of output from the response. 
```bash
{"content":" I","id_slot":0,"stop":true,"model":"../models/Meta-Llama-3-8B-Instruct-Q4_0.gguf","tokens_predicted":1,"tokens_evaluated":7,"generation_settings":{"n_ctx":256,"n_predict":-2,"model":"../models/Meta-Llama-3-8B-Instruct-Q4_0.gguf","seed":4294967295,"seed_cur":3394087514,"temperature":0.8

[... truncated for brevity ...]

---

## Issue #N/A: Feature -> tensor layer number parameter and separate from layer name

**Link**: https://github.com/ggml-org/llama.cpp/issues/1493
**State**: closed
**Created**: 2023-05-17T02:21:30+00:00
**Closed**: 2023-07-28T19:23:11+00:00
**Comments**: 5
**Labels**: enhancement, good first issue

### Description

1) ggml tensors need a layer number parameter
I'd use layer 0 for global and 1+ (could also be -1 and 0+ of course)
2) When a ggml tensor is created the latest configured layer is used, default is 0
`ggml_set_current_layer(il+1);`
This way it's only a single line of code in the eval loop to set the layer of each node.

3) The model currently contains the name intermixed with the layer like "layers.58.attention.wo.weight"
This also should be changed "attention.wo.weight" and layer number set to 59

Benefits:
1) debug output will be more clean and informative, the layer of each calculation is part of the graph print now (without adding it hardcoded into the generic name)
2) optimizations can be applied by layer name or weight name in a clean way


---

## Issue #N/A: llama : refactor llama_build_graph to reduce code duplication

**Link**: https://github.com/ggml-org/llama.cpp/issues/3382
**State**: closed
**Created**: 2023-09-28T19:13:18+00:00
**Closed**: 2023-11-01T18:11:33+00:00
**Comments**: 4
**Labels**: good first issue, high priority, refactoring

### Description

With the support of new model architectures, we start to observe a lot of repeating patterns in the code for building their compute graphs. We should find a way to refactor and reuse the repetitive code. We should also consider splitting the implementation in separate source files if necessary.

https://github.com/ggerganov/llama.cpp/blob/0e76a8992c8200237bbc6471a53fb8796b3872f7/llama.cpp#L3997-L4026

Open to ideas and suggestions

---

## Issue #N/A:  whe I give non-existant file names, it segfaults #3 

**Link**: https://github.com/ggml-org/llama.cpp/issues/3663
**State**: closed
**Created**: 2023-10-18T07:38:11+00:00
**Closed**: 2023-10-19T13:59:13+00:00
**Comments**: 1
**Labels**: good first issue

### Description

# I was told to come here by this other dude:

https://github.com/trzy/llava-cpp-server/issues/3#event-10687266750

ISSUE: when I give non-existant file names, it segfaults #3 


these files do not exist:

models/*

I get this error

➜  llava-cpp-server git:(main) ./bin/llava-server -m ./models/ggml-model-q5_k.gguf --mmproj ./models/mmproj-model-f16.gguf
[1]    57986 segmentation fault  ./bin/llava-server -m ./models/ggml-model-q5_k.gguf --mmproj

what I kinda expected:
"sorry this file doesn't exist"

thanks in advance

That's all you need to read.



Please answer the following questions for yourself before submitting an issue.

Oh boy, I'm not sure my feeling are ready to fill out the form.
here is a meme instead:

![st,small,507x507-pad,600x600,f8f8f8](https://github.com/ggerganov/llama.cpp/assets/3528304/3f8ed55e-d789-4b77-8e5c-3909fca74dea)


I'll just see myself out.



# Expected Behavior

NOT a SEGFAULT, maybe a nice error message, on STD

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: (Server UI) Use `remark` for markdown rendering

**Link**: https://github.com/ggml-org/llama.cpp/issues/10915
**State**: closed
**Created**: 2024-12-20T10:57:52+00:00
**Closed**: 2025-02-07T16:30:17+00:00
**Comments**: 1
**Labels**: enhancement, good first issue, server/webui

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

We're currently using `markdown-it` to render markdown content, but it's a bit hacky because:
- DOM are updated every time new token is added to the generating text (because it relies on setting `innerHTML`)
- Copy button need to be added separately

The idea is to replace it with `remark`, which can render markdown directly into vue components. We can rely on plugins to add back functionalities like copy button, latex, code highlight, etc.

### Motivation

N/A

### Possible Implementation

_

[... truncated for brevity ...]

---

## Issue #N/A: Create a logo

**Link**: https://github.com/ggml-org/llama.cpp/issues/105
**State**: closed
**Created**: 2023-03-13T21:15:21+00:00
**Closed**: 2023-07-28T19:20:49+00:00
**Comments**: 47
**Labels**: good first issue, 🦙.

### Description

We should probably make a logo for this project. Like an image of a 🦙 and some C++

---

## Issue #N/A: Feature Request: Allow Filtering LLama Server Response Fields

**Link**: https://github.com/ggml-org/llama.cpp/issues/10819
**State**: open
**Created**: 2024-12-13T19:59:25+00:00
**Comments**: 10
**Labels**: enhancement, good first issue

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Currently llama.cpp server serializes a lot of data the caller may not care about. Example response:
```json
{
    "index": 0,
    "content": "[\n                {\n                  \"function_name\": \"create_user\",\n                  \"username\": \"my_user\",\n                  \"email\": \"my_email@example.com\",\n                  \"password\": \"password123\"\n                }\n              ]\n            \t\t\t\t\t\t\t\t",
    "id_slot": 0,
    "stop": true,
    "model"

[... truncated for brevity ...]

---

## Issue #N/A: KV cache bug: llama-speculative and llama-server choose different kv cache quantization when cache quantization specified

**Link**: https://github.com/ggml-org/llama.cpp/issues/11200
**State**: open
**Created**: 2025-01-12T02:57:38+00:00
**Comments**: 5
**Labels**: enhancement, good first issue

### Description

### Name and Version

llama-cli --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090 Ti, compute capability 8.6, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090 Ti, compute capability 8.6, VMM: yes
version: 4462 (c05e8c99)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu

### Operating systems

Linux

### GGML backends

CUDA

### Hardware

AMD EPYC 7773X + 2x RTX 3090 TI

### Models

qwen2.5-coder:32b-instruct-q8_0.gguf
qwen2.5-coder:1.5b-instruct-q8_0.gguf

### Problem description & steps to reproduce

KV cache bug: llama-speculative and llama-server choose different kv cache quantization when cache quantization specified for the draft model kv cache.

The following command:
**llama-server** -a qwenv25coder-32b --host 0.0.0.0 --port 8081 -b 512 -ub 256 -ts 10,6 --threads 8 -ngl 99 -c 32768 --flash-attn --cache-type-k q8_0 --cache-type-v q

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: webui: Edit prompt textarea width too small

**Link**: https://github.com/ggml-org/llama.cpp/issues/11710
**State**: closed
**Created**: 2025-02-06T13:09:11+00:00
**Closed**: 2025-02-08T19:09:57+00:00
**Comments**: 1
**Labels**: enhancement, good first issue, server/webui

### Description

### Name and Version

Bleeding 124df6e7c91f8ec915da08dfb9213856ae4e3a31

### Operating systems

_No response_

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell
llama-server -m <any model>
```

### Problem description & steps to reproduce

The edit textarea is too small which is not very easy to use since also you cannot extend it wider:

![Image](https://github.com/user-attachments/assets/7412cd62-7481-4ed6-94b0-a5c66e838aee)

Compare to the prompt that was submitted:

![Image](https://github.com/user-attachments/assets/3b4bcb4f-95ed-4520-ab31-1955b156c0e6)

Reproduce:

1. Send a prompt (example: edit your python script)
2. Wait for or stop LLM completion.
3. Click the "Edit" button.

### First Bad Commit

_No response_

### Relevant log output

```shell

```

---

## Issue #N/A: Bug: server (New UI) ChatML templates are wrong

**Link**: https://github.com/ggml-org/llama.cpp/issues/9640
**State**: closed
**Created**: 2024-09-25T18:02:19+00:00
**Closed**: 2024-12-13T16:25:55+00:00
**Comments**: 3
**Labels**: good first issue, server/webui, bug-unconfirmed, medium severity

### Description

### What happened?

I think that new UI server templates are wrong.
Proposed new "Prompt template" with correct model's response formatting and trailing newline (diff):
```diff
<|im_start|>system
{{prompt}}<|im_end|>
-{{history}}{{char}}
+{{history}}<|im_start|>{{char}}
+
```
Proposed new "Chat history template" with trailing newline (diff):
```diff
<|im_start|>{{name}}
-{{message}}
+{{message}}<|im_end|>
+
```

### Name and Version

Git log: c35e586ea57221844442c65a1172498c54971cb0

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

---

## Issue #N/A: Quantization does not write the quantization version to `ftype`

**Link**: https://github.com/ggml-org/llama.cpp/issues/1590
**State**: closed
**Created**: 2023-05-25T00:30:00+00:00
**Closed**: 2023-07-28T19:23:43+00:00
**Comments**: 13
**Labels**: good first issue, high priority

### Description

# Expected Behavior

When quantizing with llama.cpp, the quantization version should be written to the `ftype` in the hyperparameters.

# Current Behavior

A `ftype` is produced by `llama_model_quantize_internal` and is passed through as-is to `llama_file_saver`, which writes it to disk without encoding it using `GGML_QNT_VERSION`:

https://github.com/ggerganov/llama.cpp/blob/ac7876ac20124a15a44fd6317721ff1aa2538806/llama.cpp#L2052-L2068

https://github.com/ggerganov/llama.cpp/blob/ac7876ac20124a15a44fd6317721ff1aa2538806/llama.cpp#L557

Loaders which are expecting the quantization version, like [llm](https://github.com/rustformers/llm), detect a quantization version of 0:

```
     Running `target/release/llm llama info -m models/llama/7B/koala-7B.ggmlv3.q5_1.bin`
[2023-05-25T00:10:05Z INFO  llm] Container type: Ggjt(3)
[2023-05-25T00:10:05Z INFO  llm] Hyperparameters: Hyperparameters { n_vocab: 32000, n_embd: 4096, n_mult: 256, n_head: 32, n_layer: 32, n_rot: 128, fi

[... truncated for brevity ...]

---

## Issue #N/A: ci : add an option to fail on compile warning

**Link**: https://github.com/ggml-org/llama.cpp/issues/3899
**State**: closed
**Created**: 2023-11-02T07:12:02+00:00
**Closed**: 2024-02-17T21:03:15+00:00
**Comments**: 4
**Labels**: good first issue, build

### Description

We should add optional flags to the build system (make and CMake) to fail upon compile warnings (`-Werror`).
These flags should be disabled by default so that during development this does not interfere, but should be enabled for most or all CI builds.

---

## Issue #N/A: `gguf-split` add a default option to not include tensors data in first shard

**Link**: https://github.com/ggml-org/llama.cpp/issues/6463
**State**: closed
**Created**: 2024-04-03T16:16:12+00:00
**Closed**: 2024-05-04T16:56:23+00:00
**Comments**: 0
**Labels**: enhancement, help wanted, good first issue, split

### Description

### Motivation

be able to make a split where the first shard is very small and contains primarily the metadata so that it can be downloaded quickly and then start the download of the other shards without waiting for the first to finish

### Proposition
Add an option to not include tensor data in the first file. Maybe it should be enabled by default.
Should be well tested.

`ggml_alloc` should not be called as it will complain with `WARNING: Behavior may be unexpected when allocating 0 bytes for ggml_malloc!`

We can add extra meta data in the first file that describes all tensors in the shards for example

#### References
- #6404
- #6135
- #6187
- #6192
- #6343
- https://github.com/ggerganov/llama.cpp/pull/6343#issuecomment-2034990690
- https://github.com/ggerganov/llama.cpp/pull/6343#issuecomment-2035011205
- https://github.com/huggingface/huggingface.js/issues/604


---

## Issue #N/A: Add proper instructions for using Alpaca models

**Link**: https://github.com/ggml-org/llama.cpp/issues/382
**State**: closed
**Created**: 2023-03-22T07:26:07+00:00
**Closed**: 2023-07-28T19:20:56+00:00
**Comments**: 22
**Labels**: documentation, help wanted, good first issue, high priority, 🦙.

### Description

So I am looking at https://github.com/antimatter15/alpaca.cpp and I see they are already running 30B Alpaca models, while we are struggling to run 7B due to the recent tokenizer updates.

I also see that the models are now even floating on Hugging Face - I guess license issues are no longer a problem?

We should add detailed instructions for obtaining the Alpaca models and a temporary explanation how to use the following script to make the models compatible with the latest `master`:

https://github.com/ggerganov/llama.cpp/issues/324#issuecomment-1476227818

The bigger issue is that people keep producing the old version of the `ggml` models instead of migrating to the latest `llama.cpp` changes. And therefore, we now need this extra conversion step. It's best to figure out the steps for generating the Alpaca models and generate them in the correct format.

**Edit: just don't post direct links to the models!**

---

## Issue #N/A: How to build on windows?

**Link**: https://github.com/ggml-org/llama.cpp/issues/103
**State**: closed
**Created**: 2023-03-13T20:13:14+00:00
**Closed**: 2023-07-28T19:20:41+00:00
**Comments**: 22
**Labels**: documentation, good first issue, windows

### Description

Please give instructions. There is nothing in README but it says that it supports it 

---

