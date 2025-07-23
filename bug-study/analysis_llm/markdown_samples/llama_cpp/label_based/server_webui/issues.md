# server_webui - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 8
- Closed Issues: 22

### Label Distribution

- server/webui: 30 issues
- enhancement: 15 issues
- good first issue: 11 issues
- stale: 7 issues
- bug: 7 issues
- bug-unconfirmed: 7 issues
- help wanted: 6 issues
- llava: 2 issues
- need more info: 2 issues
- roadmap: 2 issues

---

## Issue #N/A: Feature Request: (webui) do not throw away message if there is error in stream

**Link**: https://github.com/ggml-org/llama.cpp/issues/13709
**State**: open
**Created**: 2025-05-22T15:00:03+00:00
**Comments**: 2
**Labels**: enhancement, good first issue, server/webui

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Currently, if the UI got an error while it's generating the text, it will throw away the generating message.

The most simple way to test is to Ctrl+C to kill the server while it's generating a response.

The expected behavior is to show a meaningful error like what they do on chatgpt

<img width="680" alt="Image" src="https://github.com/user-attachments/assets/a3734cef-3e47-4fda-b12b-231f74bdf43f" />

### Motivation

N/A

### Possible Implementation

_No response_

---

## Issue #N/A: Feature Request: allow setting jinja chat template from server webui

**Link**: https://github.com/ggml-org/llama.cpp/issues/11689
**State**: closed
**Created**: 2025-02-05T22:46:03+00:00
**Closed**: 2025-06-22T01:08:17+00:00
**Comments**: 5
**Labels**: enhancement, server/webui, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Allow setting jinja chat template from server webui. Should be the same way with change system message (via the Settings dialog)

### Motivation

N/A

### Possible Implementation

_No response_

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

## Issue #N/A: webUI local storage can become corrupted

**Link**: https://github.com/ggml-org/llama.cpp/issues/10348
**State**: closed
**Created**: 2024-11-17T01:29:31+00:00
**Closed**: 2024-12-13T16:37:13+00:00
**Comments**: 2
**Labels**: bug, good first issue, server/webui

### Description

### Discussed in https://github.com/ggerganov/llama.cpp/discussions/10347

<div type='discussions-op-text'>

<sup>Originally posted by **pikor69** November 17, 2024</sup>
The page at http://127.0.0.1:8080 says:
TypeError: Cannot read properties of undefined (reading 'content')

What changed since yesterday when it was working? Nothing.
The last time I was able to start I tried to run a much higher content length than the model allowed and things crashed.

</div>

---

## Issue #N/A: server: self context extent broken 

**Link**: https://github.com/ggml-org/llama.cpp/issues/7005
**State**: closed
**Created**: 2024-04-30T09:40:55+00:00
**Closed**: 2024-12-27T12:40:25+00:00
**Comments**: 1
**Labels**: bug, server/webui

### Description

Passkey feature has been failing since a week:


https://github.com/ggerganov/llama.cpp/actions/workflows/server.yml?query=event%3Aschedule

---

## Issue #N/A: llama cpp server not doing parallel inference for llava when using flags -np and -cb

**Link**: https://github.com/ggml-org/llama.cpp/issues/5592
**State**: closed
**Created**: 2024-02-19T18:16:43+00:00
**Closed**: 2024-05-07T01:06:42+00:00
**Comments**: 11
**Labels**: server/webui, bug-unconfirmed, stale, llava

### Description

When I am trying to do parallel inferencing on llama cpp server for multimodal, I am getting the correct output for slot 0, but for other slots, I am not. Does that mean that clip is only being loaded on one slot? I can see some clip layers failing to load.

Here is my llama cpp server code that I use.

`./server -m ../models/llava13b1_5/llava13b1_5_f16.gguf -c 40960 --n-gpu-layers 41 --port 8001 --mmproj ../models/llava13b1_5/llava13b1_5_mmproj_f16.gguf -np 10 -cb --host 0.0.0.0 --threads 24`

The model I am using - 
[https://huggingface.co/mys/ggml_llava-v1.5-13b/tree/main](model)

I am using the F16 model with mmproj file.

Documentation reference

[https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md](documentation)

My GPU specs

![image](https://github.com/ggerganov/llama.cpp/assets/137015071/c7e6506e-1261-47a5-85c3-665d75fe3e7d)

My CPU specs

![image](https://github.com/ggerganov/llama.cpp/assets/137015071/8169172c-6ac3-4bea-a2f7-626

[... truncated for brevity ...]

---

## Issue #N/A: Docker: Embedding Issue & possible Fix

**Link**: https://github.com/ggml-org/llama.cpp/issues/6267
**State**: open
**Created**: 2024-03-24T02:51:09+00:00
**Comments**: 5
**Labels**: bug, good first issue, server/webui

### Description

When running the Docker Image (CPU or CUDA) and run an embedding model, I get this error

```
nomic-embed-text-1  | terminate called after throwing an instance of 'std::runtime_error'
nomic-embed-text-1  |   what():  locale::facet::_S_create_c_locale name not valid
```

When installing the locale pack on start with something like this, everythings works as expected

```
nomic-embed-text:
    image: ghcr.io/ggerganov/llama.cpp:server
    pull_policy: always
    entrypoint: ""
    command: /bin/bash -c "apt-get update && apt-get install locales && locale-gen en_US.UTF-8 && update-locale && /server --host 0.0.0.0 --port 8000 --log-disable --ctx-size 8192 --embedding --model /models/nomic-embed-text-v1.5.Q4_K_M.gguf"
    volumes:
      - ../../models:/models
```

```
curl http://localhost:8080/oai/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello!",
    "model": "nomic-embed-text"
  }'
```

maybe it is worth to install / update 

[... truncated for brevity ...]

---

## Issue #N/A: Server: Add prompt processing progress endpoint?

**Link**: https://github.com/ggml-org/llama.cpp/issues/6586
**State**: open
**Created**: 2024-04-10T11:35:31+00:00
**Comments**: 8
**Labels**: enhancement, help wanted, server/webui

### Description

# Feature Description

It would be nice to have an endpoint on the server example to fetch information about the progress of an ongoing prompt processing It could return something like this:
```json
{
    "processing": [true|false]
    "prompt_length": [number of uncached tokens of the last prompt]
    "remaining": [number of tokens yet to be processed]
}
```

# Motivation

For longer prompts, or when the processing speed is very slow, it would be nice to get a clue about the advencement of the prompt processing. This would possibly also be useful for other projects, not just the server.

# Possible Implementation

I haven't yet looked too deep in the current server implementation, so I can't really tell how this would work, but I imagine it would require some deeper changes in the backend too. 
I did add a simillar feature on a very old project based on an ancient version of llama.cpp, a year ago: https://github.com/stduhpf/fastLLaMa/commit/1ebd5ba79b3a7e4461166fe868

[... truncated for brevity ...]

---

## Issue #N/A: server: main loop blocked, server stuck

**Link**: https://github.com/ggml-org/llama.cpp/issues/5851
**State**: closed
**Created**: 2024-03-03T08:52:06+00:00
**Closed**: 2024-03-03T11:04:42+00:00
**Comments**: 11
**Labels**: enhancement, server/webui

### Description

### Context

Call to following functions are blocking the main loop and the server stuck for all slots / requests in method `update_slots`

Global:
- llama_batch_clear
- llama_decode
- llama_kv_cache_seq_cp

Per slot:
- llama_batch_add
- llama_kv_cache_seq_rm
- llama_kv_cache_seq_add
- llama_kv_cache_seq_div
- llama_sampling_free
- llama_sampling_init
- llama_sampling_accept
- llama_sampling_reset
- llama_tokenize

If prompt is big enough, self extend or continuous batching are enabled.

### Proposal

We need to separate slots state management, tokens retrieval from slots processing but keeping one batch for the whole server.

Firstly, it should be well tested and reproducible in the test server framework in a slow test with a real prompt and model (as in the passkey).

I see 3 options:

1. We are fine with that, let's wait for the high-level llama api with its own thread pool
2. Yet another threadpool (+ the http request pool). Initialized with `n_slots`

[... truncated for brevity ...]

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

## Issue #N/A: llama : add batched inference endpoint to server

**Link**: https://github.com/ggml-org/llama.cpp/issues/3478
**State**: closed
**Created**: 2023-10-04T19:10:07+00:00
**Closed**: 2023-10-24T16:38:46+00:00
**Comments**: 15
**Labels**: enhancement, help wanted, server/webui

### Description

for those not familiar with C like me.
it would be great if a new endpoint added to server.cpp to make batch inference.
for example:
endpoint: /completions
post: {"prompts":["promptA","promptB","promptC"]}
response:{"results":["sequenceA","sequenceB","sequenceC"]}

it is easy to do so with Hugging Face Transformers (as i do right now), but it's quite inefficient，hope to use llama.cpp to increase the efficiency oneday, cause I am not familiar with C, so can not use baby llama. I can only use javascript to Interact data with server.cpp。

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

## Issue #N/A: server: doc: document the `--defrag-thold` option

**Link**: https://github.com/ggml-org/llama.cpp/issues/6293
**State**: open
**Created**: 2024-03-25T06:40:20+00:00
**Comments**: 0
**Labels**: documentation, enhancement, help wanted, server/webui

### Description

### Context

The `--defrag-thold` has been added in:

- https://github.com/ggerganov/llama.cpp/pull/5941#issuecomment-1986947067

But it might be documented in the server README.md

---

## Issue #N/A: server:  fix api CORS preflight error

**Link**: https://github.com/ggml-org/llama.cpp/issues/6544
**State**: closed
**Created**: 2024-04-08T13:26:41+00:00
**Closed**: 2024-05-28T02:13:05+00:00
**Comments**: 2
**Labels**: server/webui, bug-unconfirmed, stale

### Description

Add 

        // If it's browser preflight, skip validation
        if (req.method == "OPTIONS") {
            return true;
        }
        
to server.cpp middleware_validate_api_key will make API CORS works.

---

## Issue #N/A: server : add support for file upload to the Web UI

**Link**: https://github.com/ggml-org/llama.cpp/issues/11611
**State**: closed
**Created**: 2025-02-03T05:50:11+00:00
**Closed**: 2025-05-09T21:16:40+00:00
**Comments**: 3
**Labels**: enhancement, help wanted, good first issue, server/webui

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

The idea is to be able to add any file that could be converted to plain text. The web client will do the processing and add the plain text to the context of the next request.

I am not sure what tools are available to do this in the browser, but my assumption is that there should be support, for example for converting PDF to text. Hopefully these are small packages that would not bloat the web ui too much.

### Motivation

It is useful to pass files to your chats.

### Possible Implementation

_N

[... truncated for brevity ...]

---

## Issue #N/A: Segmentation fault in example server (/v1/chat/completions route) given incorrect JSON payload

**Link**: https://github.com/ggml-org/llama.cpp/issues/7133
**State**: closed
**Created**: 2024-05-07T23:48:07+00:00
**Closed**: 2024-05-08T19:53:09+00:00
**Comments**: 10
**Labels**: bug, server/webui

### Description

# Info

Version: af0a5b616359809ce886ea433acedebb39b12969

Intel x86_64 with `LLAMA_CUDA=1`

# Summary

When `./server` is given an invalid JSON payload at the `/v1/chat/completions` route, server crashes with a segmentation fault. This denies access to clients until the server is restarted.

I stumbled upon this, and haven't thoroughly assessed all APIs or payload parameters for similar crashes. If it's easy enough to look for other routes that are missing the error handling that `/v1/chat/completions` lacks, I think someone should do so (I'm not yet familiar enough with the codebase to look for these)

# Example

```
$ gdb ./server
[... SNIP ...]
(gdb) r --model models/Meta-Llama-3-8B-Instruct.Q8_0.gguf --host 0.0.0.0
```

```
$ curl -X POST http://127.0.0.1:8081/v1/chat/completions -H 'Content-Type: application/json' --data '{}'
```

```
Thread 13 "server" received signal SIGSEGV, Segmentation fault.
[Switching to Thread 0x7efe71fff000 (LWP 567)]
0x000055e

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: move server webui from vuejs to reactjs (with typescript)

**Link**: https://github.com/ggml-org/llama.cpp/issues/11663
**State**: closed
**Created**: 2025-02-04T17:20:22+00:00
**Closed**: 2025-02-06T16:32:31+00:00
**Comments**: 1
**Labels**: enhancement, server/webui

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

(as shown in the title)

### Motivation

Vuejs is good enough for the early development (just a POC to see if people actually love it or not)

But as the code base grown, it now become unmanageable.

The solution is to move to the mainstream reactjs + typescript stack, with a proper lint / prettier / testing framework to ease the development.

### Possible Implementation

_No response_

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

## Issue #N/A: Unable to assign mmproj value when running docker 

**Link**: https://github.com/ggml-org/llama.cpp/issues/6226
**State**: closed
**Created**: 2024-03-22T07:52:31+00:00
**Closed**: 2024-05-07T01:06:30+00:00
**Comments**: 2
**Labels**: server/webui, bug-unconfirmed, stale, llava

### Description

Please include information about your system, the steps to reproduce the bug, and the version of llama.cpp that you are using. If possible, please provide a minimal code example that reproduces the bug.

If the bug concerns the server, please try to reproduce it first using the [server test scenario framework](https://github.com/ggerganov/llama.cpp/tree/master/examples/server/tests).

Command
```sh
sudo docker run -p 5000:8000  --gpus all --runtime=nvidia -v /models:/models ghcr.io/ggerganov/llama.cpp:server-cuda -m /models/ggml-model-q4_k.gguf --mmproj /models/mmproj-model-f16.gguf  --port 8000 --host 0.0.0.0 -v  -t 16  -n 512 -c 2048 -ngl 1 -cb -np 4 --n-gpu-layers 33
```

Error
```sh
error: unknown argument: --mmproj
```

--mmproj option is not supported by docker. 

The documentation mentions this option though.
https://github.com/ggerganov/llama.cpp/tree/master/examples/server#llamacpp-http-server


---

## Issue #N/A: The output of the main service is inconsistent with that of the server service

**Link**: https://github.com/ggml-org/llama.cpp/issues/6569
**State**: closed
**Created**: 2024-04-09T15:40:35+00:00
**Closed**: 2024-05-27T01:06:36+00:00
**Comments**: 10
**Labels**: need more info, server/webui, bug-unconfirmed, stale

### Description

**When the same quantitative model is used for server service and main service, some specific words are answered differently. It seems that the input specific words are not received or received incorrectly.
For example, BYD, Tesla, Lexus and other car names have this problem, such as Geely, BMW, Audi and so on is normal.**
The specific problem is manifested in: When obtaining the word "BYD" in the server service, non-Chinese characters such as "ruit" are not obtained or obtained. As in the first example, when asked about BYD car, the reply only involved the car, and BYD was lost.
**Test results in the server**
********************************************************
**These are three examples of problems（BYD）**
********************************************************
{
  content: ' 汽车是一种交通工具，它通常由发动机，变速箱，底盘和底盘系统，悬挂系统，转向系统，车身和车轮等组成。汽车通常由汽油或柴油发动机提供动力，通过变速箱和传动系统来控制车辆行驶的速度和方向。汽车的设计和制造技术不断提高，汽车的功能也越来越强大。现在汽车已经不仅仅是一种交通工具，它已经成为人们日常生活中不可或缺的一部分，提供了各种便利。汽车在现代社会中的作用非常广泛，它可以满足人们的出行需求，同时也可以娱

[... truncated for brevity ...]

---

## Issue #N/A: Server gets stuck after invalid request

**Link**: https://github.com/ggml-org/llama.cpp/issues/5724
**State**: closed
**Created**: 2024-02-26T08:01:49+00:00
**Closed**: 2024-02-26T22:15:49+00:00
**Comments**: 4
**Labels**: bug, server/webui

### Description

Repro:

```bash
./server -m models/bert-bge-small/ggml-model-f16.gguf --embedding
```

```bash
# send invalid request
curl http://localhost:8080/v1/embeddings -H "Content-Type: application/json" -H "Authorization: Bearer no-key" -d '{ }'

# next requests makes server hang
curl http://localhost:8080/v1/embeddings -H "Content-Type: application/json" -H "Authorization: Bearer no-key" -d '{ "input": "hello" }'

# need to kill it
killall server
```

---

## Issue #N/A: server : add "token healing" support

**Link**: https://github.com/ggml-org/llama.cpp/issues/5765
**State**: open
**Created**: 2024-02-28T12:10:30+00:00
**Comments**: 6
**Labels**: enhancement, good first issue, server/webui, roadmap

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

Hi! I am experimenting with using llama.cpp as a general-purpose code completion backend, similar to TabNine.

I am encountering a small problem: if the completion prompt ends mid-word, the results are not very accurate. For example, for a prompt such as `Five, Fo

[... truncated for brevity ...]

---

## Issue #N/A: server : improvements and maintenance

**Link**: https://github.com/ggml-org/llama.cpp/issues/4216
**State**: open
**Created**: 2023-11-25T09:57:53+00:00
**Comments**: 120
**Labels**: help wanted, refactoring, server/webui, roadmap

### Description

The [server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server) example has been growing in functionality and unfortunately I feel it is not very stable at the moment and there are some important features that are still missing. Creating this issue to keep track on some of these points and try to draw more attention from the community. I guess, some of the tasks are relatively big and would require significant efforts to complete

- [x] **Support chat templates**
  We need to have separation between the user input and the special tokens, so that the tokenization is performed correctly. See the following comments / commits for more context:
  https://github.com/ggerganov/llama.cpp/pull/4160#discussion_r1403675264
  https://github.com/ggerganov/llama.cpp/pull/4198/commits/c544faed749240fe5eac2bc042087c71f79a0728
  https://github.com/ggerganov/llama.cpp/pull/4160#issuecomment-1824984718

  We already support extracting meta information from the GGUF model files th

[... truncated for brevity ...]

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

## Issue #N/A: Feature Request: Use IndexedDB for server web UI

**Link**: https://github.com/ggml-org/llama.cpp/issues/10946
**State**: closed
**Created**: 2024-12-22T17:33:57+00:00
**Closed**: 2025-02-22T21:42:24+00:00
**Comments**: 2
**Labels**: enhancement, good first issue, server/webui

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

As explained in https://github.com/ggerganov/llama.cpp/pull/10945 , some users may want to store more than 5MB of data in browser.

Compressed `localStorage` is not a scalable solution because it only raise the limit to x4 or x5, but no more.

IndexedDB is preferable in this situation, because there is no hard limit for storage space.

### Motivation

N/A

### Possible Implementation

A lightweight implementation using [one of the libraries on npm](https://www.npmjs.com/search?q=IndexedDB).

---

## Issue #N/A: Bug: `llama-server` web UI resets the text selection during inference on every token update

**Link**: https://github.com/ggml-org/llama.cpp/issues/9608
**State**: closed
**Created**: 2024-09-23T13:02:38+00:00
**Closed**: 2025-02-07T16:30:04+00:00
**Comments**: 10
**Labels**: bug, help wanted, good first issue, server/webui, low severity

### Description

### What happened?

When using `llama-server`, the output in the UI can't be easily selected or copied until after text generation stops. This may be because the script replaces all the DOM nodes of the current generation when every new token is output.

The existing text content ideally shouldn't be replaced during generation so we can copy the text as it continues to produce output.

### Name and Version

version: 3755 (822b6322)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

---

## Issue #N/A: server: Use `llama_chat_apply_template` on `/completion` endpoint

**Link**: https://github.com/ggml-org/llama.cpp/issues/6624
**State**: closed
**Created**: 2024-04-12T04:16:34+00:00
**Closed**: 2024-05-29T01:06:41+00:00
**Comments**: 3
**Labels**: enhancement, server/webui, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

Use `llama_chat_apply_template` on `/completion` or a new endpoint (e.g. `/chat`), in addition to the current OpenAI compatibility endpoints. Update WebUI to reflect the change.

# Motivation

The OpenAI compatibility endpoints are nice and all, but native endpo

[... truncated for brevity ...]

---

## Issue #N/A: tid in log always be the same 

**Link**: https://github.com/ggml-org/llama.cpp/issues/6534
**State**: closed
**Created**: 2024-04-08T06:13:13+00:00
**Closed**: 2024-04-08T07:46:36+00:00
**Comments**: 3
**Labels**: server/webui, bug-unconfirmed

### Description

macos m1 pro

https://github.com/ggerganov/llama.cpp/blob/855f54402e866ed19d8d675b56a81c844c64b325/examples/server/utils.hpp#L73

 ss_tid << std::this_thread::get_id();
always be a same value，not matter how i restart

---

## Issue #N/A: server: Default web UI erroneously inteprets markdown special characters inside code blocks

**Link**: https://github.com/ggml-org/llama.cpp/issues/3723
**State**: closed
**Created**: 2023-10-22T06:40:53+00:00
**Closed**: 2024-12-31T10:02:52+00:00
**Comments**: 13
**Labels**: bug, server/webui

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ X ] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ X ] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ X ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ X ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

When generating C code you expect multi-line comments to show up as '/* ... */'. It does when you run ./main with -ins, however when you run the server the asterisks disappear as they're populating.

Prompt: Write a C function to show a fibinnachi sequence.

[... truncated for brevity ...]

---

## Issue #N/A: Handling Concurrent API Calls from Multiple Clients - Server Functionality

**Link**: https://github.com/ggml-org/llama.cpp/issues/5723
**State**: closed
**Created**: 2024-02-26T07:03:38+00:00
**Closed**: 2024-02-26T07:48:27+00:00
**Comments**: 1
**Labels**: need more info, server/webui, bug-unconfirmed

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ ] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ ] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

Server functionality to handle concurrent API calls from different clients if slots are busy.

# Motivation

Deployed LLMs could be used from multiple clients, handling multiple concurrent api calls, queuing them up, instead of throwing an error.

---

