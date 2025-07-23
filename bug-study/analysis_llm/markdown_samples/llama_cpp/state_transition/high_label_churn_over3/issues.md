# high_label_churn_over3 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 7
- Closed Issues: 23

### Label Distribution

- good first issue: 14 issues
- enhancement: 14 issues
- help wanted: 13 issues
- stale: 9 issues
- server/webui: 8 issues
- bug-unconfirmed: 7 issues
- bug: 5 issues
- server: 4 issues
- hardware: 4 issues
- performance: 4 issues

---

## Issue #N/A: AttributeError: 'GGUFWriter' object has no attribute 'add_vocab_size'

**Link**: https://github.com/ggml-org/llama.cpp/issues/6585
**State**: closed
**Created**: 2024-04-10T10:15:13+00:00
**Closed**: 2024-06-16T01:07:09+00:00
**Comments**: 7
**Labels**: need more info, model, bug-unconfirmed, stale

### Description

Hi, When I converted the large model weights to gguf format, I encountered this error


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

## Issue #N/A: iGPU offloading Bug: Memory access fault by GPU node-1 (appeared once only)

**Link**: https://github.com/ggml-org/llama.cpp/issues/7829
**State**: closed
**Created**: 2024-06-08T06:04:54+00:00
**Closed**: 2024-07-23T01:06:44+00:00
**Comments**: 1
**Labels**: AMD GPU, bug-unconfirmed, stale, low severity

### Description

### What happened?

I am comparing inference with and without AMD iGPU offloading with ROCm.

The setup is documented at https://github.com/eliranwong/MultiAMDGPU_AIDev_Ubuntu/blob/main/igpu_only.md#compare-cpu-vs-openblas-vs-rocm-vs-rocmigpu-offloading

The result shows that AMD iGPU offloading with ROCm runs roughly 1.5x faster.

It is interesting to note that the first time I tried to run the following command:

> ./main -t $(lscpu | grep '^Core(s)' | awk '{print $NF}') --temp 0 -m '/home/eliran/freegenius/LLMs/gguf/mistral.gguf' -p "What is machine learning?" -ngl 33

I got the following error:

```
Memory access fault by GPU node-1 (Agent handle: 0x613061b881f0) on address 0x9000. Reason: Page not present or supervisor privilege.
Aborted (core dumped)
```

However, it appeared once only.  Further inference with the same command runs smoothly.  It is not a practical problem, as it happened once only.  All later inferences runs without an issue.

### Name and Versio

[... truncated for brevity ...]

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

## Issue #N/A: Bug: 2 tests fail

**Link**: https://github.com/ggml-org/llama.cpp/issues/8906
**State**: closed
**Created**: 2024-08-07T07:37:13+00:00
**Closed**: 2024-09-22T01:07:33+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, Vulkan, stale, medium severity

### Description

### What happened?

Tests test-eval-callback and test-backend-ops fail on FreeBSD 14.1

### Name and Version

Version: 3538

### What operating system are you seeing the problem on?

BSD

### Relevant log output

```shell
[LastTest.log](https://freebsd.org/~yuri/llama-cpp-3538-LastTest.log)
```


---

## Issue #N/A: `quantize`: add imatrix and dataset metadata in GGUF

**Link**: https://github.com/ggml-org/llama.cpp/issues/6656
**State**: closed
**Created**: 2024-04-13T10:13:08+00:00
**Closed**: 2024-04-26T18:06:34+00:00
**Comments**: 3
**Labels**: enhancement, model, generation quality, need feedback

### Description

### Motivation
I was reading [thanks](https://huggingface.co/spaces/ggml-org/gguf-my-repo/discussions/41#661a27157a16dc848a58a261) to @julien-c this [reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1ba55rj/overview_of_gguf_quantization_methods/?rdt=36175) from @he29-net :+1: 

> You can't easily tell whether a model was quantized with the help of importance matrix just from the name. I first found this annoying, because it was not clear if and how the calibration dataset affects performance of the model in other than just positive ways. But recent tests in llama.cpp [discussion #5263](https://github.com/ggerganov/llama.cpp/discussions/5263) show, that while the data used to prepare the imatrix slightly affect how it performs in (un)related languages or specializations, any dataset will perform better than a "vanilla" quantization with no imatrix. So now, instead, I find it annoying because sometimes the only way to be sure I'm using the better imatrix version is to re-quan

[... truncated for brevity ...]

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

## Issue #N/A: `llama_decode` is significantly slower if `n_tokens > 1` 

**Link**: https://github.com/ggml-org/llama.cpp/issues/4624
**State**: closed
**Created**: 2023-12-24T23:05:48+00:00
**Closed**: 2024-04-02T01:10:00+00:00
**Comments**: 7
**Labels**: performance, macos, bug-unconfirmed, stale

### Description

Issue
---
It is expected that `llama_decode` should take more time if more tokens are present in the batch, but on my system (Apple M1 Max 32GB) with `mistral-7b-instruct-v0.2.Q4_0.gguf` model, the increase in time taken is quite significant. I plotted some avg latencies on my system with different `n_tokens` using a modified version of `speculative` and putting timing around `llama_decode(ctx_tgt, batch_tgt);`:

![image](https://github.com/ggerganov/llama.cpp/assets/1957903/d9683434-6278-41b2-9018-d60acbe4ec2a)

There is more 5x jump in latency of `llama_decode` when `n_tokens` goes from 1 to 2 (which I feel is too high), but a very gradual increase after that. This means that techniques like `speculative` and `lookup` decoding **cannot give speed benefits** for small draft sizes ( `n_draft < 5`) even if drafts are 100% correct, since **autoregressively decoding 5 tokens 1 at a time is just as fast as decoding 5 tokens at once**, so the advantage of speculation is lost.

I'm n

[... truncated for brevity ...]

---

## Issue #N/A: common: download from URL, improve parallel download progress status

**Link**: https://github.com/ggml-org/llama.cpp/issues/6537
**State**: open
**Created**: 2024-04-08T07:37:01+00:00
**Comments**: 6
**Labels**: enhancement, help wanted, good first issue, split

### Description

### Context

When downloading a sharded model, files are downloaded in parallel, it was added in:
- #6192

The progressions of each download conflict:
![image](https://github.com/ggerganov/llama.cpp/assets/5741141/d4937fc7-edf4-4920-ba63-dadf1c77b2d0)

Need to properly implement [CURLOPT_NOPROGRESS](https://curl.se/libcurl/c/CURLOPT_NOPROGRESS.html) for parallel download.

Example in #6515:

```shell
main --hf-repo ggml-org/models \
  --hf-file grok-1/grok-1-q4_0-00001-of-00009.gguf \
  --model   models/grok-1-q4_0-00001-of-00009.gguf \
  -ngl 64
   --prompt "I believe the meaning of life is"
```

---

## Issue #N/A: Create "instruct" example

**Link**: https://github.com/ggml-org/llama.cpp/issues/508
**State**: closed
**Created**: 2023-03-25T20:22:39+00:00
**Closed**: 2023-07-28T19:21:12+00:00
**Comments**: 1
**Labels**: enhancement, help wanted, good first issue, 🦙.

### Description

Currently, the `main` example has a `instruct` parameter which enables something similar to instruction-based mode. I haven't understood it completely, but this seems to be what the Alpaca models are created for.

Since we now support infinite generation (https://github.com/ggerganov/llama.cpp/issues/71#issuecomment-1483907574) it would be very useful to make a separate app that utilizes the new `--keep` argument to create a question-answering bot that never stops. The tricky part is to keep the correct instruction prompt and "inject" the few-shot examples correctly, or whatever.

The main logic for context swapping / context rotation is here:

https://github.com/ggerganov/llama.cpp/blob/c2b25b6912662d2637d9c6e6df3a5de931e0d7ce/examples/main/main.cpp#L297-L324

Uncomment the `printf` to help debug. Something similar will be needed in the new `instruct` example.

Implementing this task will also help simplify the `main` example as it will no longer need to support the `--instr

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Task Cancellation on Client Disconnection

**Link**: https://github.com/ggml-org/llama.cpp/issues/6421
**State**: closed
**Created**: 2024-04-01T08:20:25+00:00
**Closed**: 2025-05-16T19:42:45+00:00
**Comments**: 5
**Labels**: enhancement, help wanted, good first issue, server/webui

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description
In the current embedding server setup, if a client sends a request and then cancels it, tasks that are already queued continue processing without detecting the cancellation. This can lead to inefficiencies and potential server overload.

**[Test Case]** 
During an 

[... truncated for brevity ...]

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

## Issue #N/A: Investigate storing results from ggml operations in F16 format

**Link**: https://github.com/ggml-org/llama.cpp/issues/959
**State**: closed
**Created**: 2023-04-14T07:35:34+00:00
**Closed**: 2023-04-22T08:48:31+00:00
**Comments**: 1
**Labels**: help wanted, performance, high priority, research 🔬

### Description

Currently, all `ggml` operations return the results in F32 format.

The goal of this task is to see if there is an elegant way to add support for keeping the results in F16 format.
This will ideally be passed as a parameter to the `ggml_context` and will also involve adding support for F16 operands in most of the existing operators. Ideally, we want to achieve this somehow without duplicating the entire code base.

Note that internal floating-point accumulators in the different operations can and should remain in F32 format.
It is just when we store the results into the `dst` tensor, we will cast them to F16.

Going to F16 intermediate results would reduce significantly the memory pressure and could lead to significant speed improvements. Hopefully, the loss in quality would be marginal. But in any case, there will always be the option of switching back to full F32 precision.

I am looking for suggestions and initial prototypes of how we can achieve this in an elegant way.


[... truncated for brevity ...]

---

## Issue #N/A: llama : add support for Classifier-Free Guidance (CFG) sampling to stay on topic better

**Link**: https://github.com/ggml-org/llama.cpp/issues/2083
**State**: closed
**Created**: 2023-07-03T08:38:55+00:00
**Closed**: 2023-07-11T16:18:45+00:00
**Comments**: 19
**Labels**: enhancement, good first issue, generation quality, research 🔬

### Description

@ggerganov [retweeted](https://twitter.com/Vermeille_/status/1675664118500454400) the "Stay on topic with Classifier-Free Guidance" paper that came out showing that "Classifier-Free Guidance (CFG)"... "can be used broadly as an inference-time technique in pure language modeling. " ... "brings improvements equivalent to a model with twice the parameter-count" (with no retraining needed). -  https://arxiv.org/abs/2306.17806

I saw that the Transformers library has one of the paper's author [working on an implementation](https://github.com/huggingface/transformers/issues/24536).

I didn't see an issue for it yet here so I figured pointing to it is the least I could do for this awesome library!

---

## Issue #N/A: split: include the option in ./convert.py and quantize

**Link**: https://github.com/ggml-org/llama.cpp/issues/6260
**State**: open
**Created**: 2024-03-23T15:32:02+00:00
**Comments**: 9
**Labels**: enhancement, help wanted, good first issue, split

### Description

### Context

At the moment it is only possible to split after convertion or quantization. Mentionned by @Artefact2 in this `[comment](https://github.com/ggerganov/llama.cpp/pull/6135#issuecomment-2003942162)`:

> as an alternative, add the splitting logic directly to tools that produce ggufs, like convert.py and quantize.

### Proposition

Include split options in `convert*.py`, support splits in `quantize`

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

## Issue #N/A: New kv_cache API insufficient to restore model state

**Link**: https://github.com/ggml-org/llama.cpp/issues/730
**State**: closed
**Created**: 2023-04-03T03:28:49+00:00
**Closed**: 2023-04-23T13:51:21+00:00
**Comments**: 23
**Labels**: bug, help wanted, good first issue, high priority

### Description

I may be doing something wrong or misunderstanding the purpose of the `kv_cache` API but I believe the recent PR #685 by @chrfalch which added the ability to get / set the `kv_cache` is still insufficient to restore the state of the model even when resetting external model state such as `last_n_tokens_data` and `n_past`.

Here is a minimal example

```c++
#include "llama.h"
#include <vector>
#include <iostream>

using namespace std;

int main() {
    // init
    auto params = llama_context_default_params();
    auto ctx = llama_init_from_file("../../models/ggml-model.bin", params);
    auto tokens = vector<llama_token>(params.n_ctx);
    auto prompt = "The quick brown fox";
    auto n_tokens = llama_tokenize(ctx, prompt, tokens.data(), tokens.size(), true);

    // evaluate prompt
    llama_eval(ctx, tokens.data(), n_tokens, 0, 12);
    auto last_n_tokens_size = 64;
    auto last_n_tokens_data = vector<llama_token>(last_n_tokens_size, 0);
    last_n_tokens_data.i

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

## Issue #N/A: Add GPU support to ggml

**Link**: https://github.com/ggml-org/llama.cpp/issues/914
**State**: closed
**Created**: 2023-04-12T11:11:42+00:00
**Closed**: 2023-04-12T11:47:54+00:00
**Comments**: 1
**Labels**: enhancement, help wanted, hardware, research 🔬

### Description

## Intro

This issue is more suitable for the https://github.com/ggerganov/ggml repo, but adding it here for more visibility.

First, I don't see adding a GPU framework that is tightly integrated with `ggml` anytime soon because it usually comes with a lot of maintenance drawbacks, architecture changes and issues. However, there is an alternative approach that might be relatively easy to implement and I think would be a very cool way for new developers to join in and help.

## Description

`ggml` produces computation graphs which are basically directed acyclic graphs (DAGs) that can be easily exported, iterated, etc. A graph contains the information about all necessary tensor operations and buffers needed to evaluate the model. The idea is to first add basic `ggml` functionality for exporting the graphs in some trivial text format that can be parsed as a second step by a separate `ggml` tool. Having the exported graphs, one can process them and construct hardware-specific code 

[... truncated for brevity ...]

---

## Issue #N/A: server: bench: continuous performance testing

**Link**: https://github.com/ggml-org/llama.cpp/issues/6233
**State**: closed
**Created**: 2024-03-22T11:36:09+00:00
**Closed**: 2024-07-03T01:06:46+00:00
**Comments**: 19
**Labels**: enhancement, performance, server/webui, need feedback, stale

### Description

#### Motivation

**llama.cpp** is under active development, new papers on LLM are implemented quickly (for the good) and backend device
optimizations are continuously added.

All these factors have an impact on the server performances, especially the following metrics:

1. **latency**: pp (prompt processing) + tg (tokens generation) per request
2. **server latency**: total pp+tg per second across all requests with continuous batching
3. **concurrency**: how many concurrent request/users the server can handle in parallel
4. **VRAM** usage
5. **RAM** usage
6. **GPU** usage
7. **CPU** usage

It is important to monitor and control the impact of the codebase evolution on these metrics,
example [from](https://towardsdatascience.com/increase-llama-2s-latency-and-throughput-performance-by-up-to-4x-23034d781b8c):

<p align="center">
    <img width="60%" height="60%" src="https://github.com/ggerganov/llama.cpp/assets/5741141/2f518477-941d-41e1-9427-873ca0cb9846" alt="prompt_to

[... truncated for brevity ...]

---

## Issue #N/A: mpi : attempt inference of 65B LLaMA on a cluster of Raspberry Pis

**Link**: https://github.com/ggml-org/llama.cpp/issues/2164
**State**: open
**Created**: 2023-07-10T16:12:22+00:00
**Comments**: 54
**Labels**: help wanted, 🦙., hardware, research 🔬

### Description

Now that distributed inference is supported thanks to the work of @evanmiller in #2099 it would be fun to try to utilize it for something cool. One such idea is to connect a bunch of [Raspberry Pis](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/) in a local network and run the inference using MPI:

```bash
# sample cluster of 8 devices (replace with actual IP addresses of the devices)
$ cat ./hostfile
192.168.0.1:1
192.168.0.2:1
192.168.0.3:1
192.168.0.4:1
192.168.0.5:1
192.168.0.6:1
192.168.0.7:1
192.168.0.8:1

# build with MPI support
$ make CC=mpicc CXX=mpicxx LLAMA_MPI=1 -j

# run distributed inference over 8 nodes
$ mpirun -hostfile ./hostfile -n 8 ./main -m /mnt/models/65B/ggml-model-q4_0.bin -p "I believe the meaning of life is" -n 64
```

Here we assume that the 65B model data is located on a network share in `/mnt` and that `mmap` works over a network share.
Not sure if that is the case - if not, then it would be more difficult to perform th

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Gemma 2 slower with FA

**Link**: https://github.com/ggml-org/llama.cpp/issues/9243
**State**: closed
**Created**: 2024-08-29T16:39:59+00:00
**Closed**: 2024-11-08T01:07:24+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale, Apple Metal, medium severity

### Description

### What happened?

Gemma 2 is slower with FA on Apple Silicon (M3 Max).

### Name and Version

version: 3642 (1d1ccce6)
built with Apple clang version 15.0.0 (clang-1500.3.9.4) for arm64-apple-darwin23.6.0

### What operating system are you seeing the problem on?

Mac

### Relevant log output

```shell
| model                          |       size |     params | backend    | ngl | fa | mmap |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | ---: | ------------: | ---------------: |
| gemma2 2B Q8_0                 |   3.17 GiB |     3.20 B | Metal      |  99 |  0 |    0 |         pp512 |   2360.42 ± 3.71 |
| gemma2 2B Q8_0                 |   3.17 GiB |     3.20 B | Metal      |  99 |  0 |    0 |          tg64 |     85.54 ± 0.05 |
| gemma2 2B Q8_0                 |   3.17 GiB |     3.20 B | Metal      |  99 |  1 |    0 |         pp512 |   1487.45 ± 3.27 |
| gemma2 2B Q8_0                 |   3.17 GiB |     3.

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

