# first_month - issues

**Total Issues**: 391
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 390

### Label Distribution

- enhancement: 86 issues
- stale: 59 issues
- bug: 47 issues
- model: 46 issues
- build: 39 issues
- duplicate: 34 issues
- need more info: 34 issues
- hardware: 26 issues
- good first issue: 25 issues
- question: 22 issues

---

## Issue #N/A: Long time until generation starts when using big context

**Link**: https://github.com/ggml-org/llama.cpp/issues/865
**State**: closed
**Created**: 2023-04-09T15:39:18+00:00
**Closed**: 2023-05-21T14:02:24+00:00
**Comments**: 11

### Description

When just saying like "Hello, who are you?", I get like 200ms/token and it starts generating almost instantly.
On the other hand, when I paste a small text (e.g. search results from duck duck go api) I have to wait +- 1min and then it generates but quite slow. Is this normal behaviour=

My cpu is a ryzen 7 6800h and 32gb ddr5 ram. I'm running vicuna 7b. 
I paste the search result context from the python bindings. 

---

## Issue #N/A: [User] Memory usage is extremely low when running 65b 4-bit models. (Only use 5GB)

**Link**: https://github.com/ggml-org/llama.cpp/issues/864
**State**: closed
**Created**: 2023-04-09T14:54:14+00:00
**Closed**: 2023-04-12T12:05:00+00:00
**Comments**: 22

### Description

Dear llama.cpp team,

I am experiencing two issues with llama.cpp when using it with the following hardware:
```
CPU: Xeon Silver 4216 x 2ea
RAM: 383GB
GPU: RTX 3090 x 4ea
```

The first issue is that although the model requires a total of 41478.18 MB of memory, my machine only uses 5 GB of memory when running the model. I would like to know if this is normal behavior or if there is something wrong with other.

The second issue is related to the token generation speed of the model. Despite my powerful CPU, which consists of two Xeon Silver 4216 processors, I am only getting a token generation speed of 0.65/s. This speed seems slower than what I would expect from my hardware. Could you please advise on how to improve the token generation speed?

Here is the information you may need to help troubleshoot the issue:

[Software Env]
```
Python 3.9.16
Windows 10 21H2
oobabooga/text-generation-webui
```

[Output]
```D:\one-click-installers-oobabooga-windows\one-click-in

[... truncated for brevity ...]

---

## Issue #N/A: Bug with instruct mode, somehow "forks" in background (Windows 11 - Powershell) and makes the shell unuseable

**Link**: https://github.com/ggml-org/llama.cpp/issues/859
**State**: closed
**Created**: 2023-04-09T00:08:07+00:00
**Closed**: 2024-04-11T01:06:47+00:00
**Comments**: 5
**Labels**: stale

### Description

I can not provide a reproducible case but this has happened two times during various tests so there is something fishy.
main.exe was in Release mode, compiled in VS build tools and executed from Powershell in Win-11.

What happens is that CTRL+C is somehow accepted but the app doesn't stop - it pseudo-forks into background.
I am thrown back to the command prompt of the shell but the shell becomes extremely "laggy".
In one case I was able to issue commands but only every second key was accepted, so I could use the shell but had to repeat all keys twice.
main.exe waited in "prompt" mode I think, so it did not generate anything but waited (never received my input)

In the other case only the 'Enter' key was accepted (so the shell created a new line) but no other keys worked
In that case I also saw a generation happening, it was written into the shell.
So it was in generation mode but forked in the background.

In both cases the shell input was extremely laggy. 
I've never see

[... truncated for brevity ...]

---

## Issue #N/A: Question: How do I use openblas on M1 apple silicon (or is there any point?)

**Link**: https://github.com/ggml-org/llama.cpp/issues/857
**State**: closed
**Created**: 2023-04-08T23:18:04+00:00
**Closed**: 2023-04-09T03:34:43+00:00
**Comments**: 2

### Description

I don't quite how to specify the proper make flag for this or whether it even applies to MacOS.



---

## Issue #N/A: I5 13600k or RTX 3060 12gb?

**Link**: https://github.com/ggml-org/llama.cpp/issues/853
**State**: closed
**Created**: 2023-04-08T17:33:12+00:00
**Closed**: 2023-04-08T19:20:47+00:00
**Comments**: 2

### Description

Hello Community
I want to build a computer which will run llama.cpp or text generation web ui.
Which should I get? Each config is about the same price. 
Should I get the 13600k and no gpu (But I can install one in the future if I have money) or a "bad" cpu and a rtx 3060 12gb?
Which should I get / is faster?
Thank you in advice.

---

## Issue #N/A: Running convert-pth-ggml.py on Linux

**Link**: https://github.com/ggml-org/llama.cpp/issues/852
**State**: closed
**Created**: 2023-04-08T16:15:51+00:00
**Closed**: 2023-04-08T19:23:23+00:00
**Comments**: 3

### Description

A not so obvious error, Using Hugging Face llama-13b-hf as the model, I get a 'no such file' for params.json
(Following README, but don't have a params.json.  Are we supposed to build one? Might want to update README for this situation.

python3.10 convert-pth-to-ggml.py ../llama-13b-hf/ 1

---

## Issue #N/A: [ALPACA Q4] assert n_dims in (1, 2) when using migrate-ggml-2023-03-30-pr613.py after convert-gpt4all-to-ggml.py

**Link**: https://github.com/ggml-org/llama.cpp/issues/849
**State**: closed
**Created**: 2023-04-08T11:14:33+00:00
**Closed**: 2024-04-11T01:06:48+00:00
**Comments**: 3
**Labels**: stale

### Description

Hello,

I was using int Q4 alpaca model, that was converted using convert-gpt4all-to-ggml.py script. It was working perfectly until an update about a week ago (2023-03-30). At the moment I use the repository as of #847.

When I freshly converted the `ggml-alpaca-q4_0.bin` from the scratch using `convert-gpt4all-to-ggml.py`, and try to start the model using resulting model, I get fallowing error:

```
/models/alpaca/30b/ggml-model-q4_0.bin: invalid model file (bad magic [got 0x67676d66 want 0x67676a74])
	you most likely need to regenerate your ggml files
	the benefit is you'll get 10-100x faster load times
	see https://github.com/ggerganov/llama.cpp/issues/91
	use convert-pth-to-ggml.py to regenerate from original pth
	use migrate-ggml-2023-03-30-pr613.py if you deleted originals
llama_init_from_file: failed to load model
```

So then I proceed with:
```
python migrate-ggml-2023-03-30-pr613.py models/alpaca/30b/ggml-model-q4_0.bin models/alpaca/30b/ggml-model-q4_0.bin_

[... truncated for brevity ...]

---

## Issue #N/A: llama : add RWKV models support

**Link**: https://github.com/ggml-org/llama.cpp/issues/846
**State**: closed
**Created**: 2023-04-08T06:32:31+00:00
**Closed**: 2024-09-01T14:38:18+00:00
**Comments**: 39
**Labels**: help wanted, good first issue, model

### Description

RWKV (100% RNN) language model, which is the only RNN (as of now) that can match transformers in quality and scaling, while being faster and saves memory.

Info: https://github.com/BlinkDL/ChatRWKV

RWKV is a novel large language model architecture, [with the largest model in the family having 14B parameters](https://huggingface.co/BlinkDL/rwkv-4-pile-14b). In contrast to Transformer with O(n^2) attention, RWKV requires only state from previous step to calculate logits. This makes RWKV very CPU-friendly on large context lenghts.

Experimental GGML port: https://github.com/saharNooby/rwkv.cpp

The lastest "Raven"-series Alpaca-style-tuned RWKV 14B & 7B models are very good.
Online demo: https://huggingface.co/spaces/BlinkDL/Raven-RWKV-7B
Download: https://huggingface.co/BlinkDL/rwkv-4-raven

----

*Edit by @ggerganov:*

Adding @BlinkDL's comment below to OP for visibility:

> v4 inference: https://github.com/BlinkDL/ChatRWKV/blob/main/RWKV_in_150_lines.py
>
> v5 infe

[... truncated for brevity ...]

---

## Issue #N/A: Input Chinese Its talk with itself

**Link**: https://github.com/ggml-org/llama.cpp/issues/843
**State**: closed
**Created**: 2023-04-08T04:24:28+00:00
**Closed**: 2024-04-11T01:06:49+00:00
**Comments**: 11
**Labels**: stale

### Description

![image](https://user-images.githubusercontent.com/70996861/230702957-dd4dff6a-35e5-4926-9796-75db84443b5d.png)
When I use Chinese 
Its will talk with itself

---

## Issue #N/A: Performance e-core bug(?) -  only 50% CPU utilization when using all threads -  (Win11, Intel 13900k)

**Link**: https://github.com/ggml-org/llama.cpp/issues/842
**State**: closed
**Created**: 2023-04-08T03:07:10+00:00
**Closed**: 2024-04-11T01:06:51+00:00
**Comments**: 24
**Labels**: stale

### Description

I've not digged deep into this yet but my whole CPU utilization is only at 50%.
I've compiled it with current VS build tools, all default, release mode of course.

It might be related to the modern e-cores in Intel CPUs, they pack quite a punch but are weaker than performance cores.
In the graph it looks like 16 cores (the amount of e-cores) are much more utilized and 8 cores (amount of performance cores) are mostly idle despite using 24 threads. Increasing threads worsens performance, decreasing threads worsens tokens output.

I tested the small 7B model in 4 bit and 16 bit.
The only method to get CPU utilization above 50% is by using more than the total physical cores (like 32 cores).
In this case I see up to 99% CPU utilization but the token performance drops below 2 cores performance, some hyperthreading issue I suppose.
I tried various modes (small/large batch size, context size) It all does not influence it much.

The CPU was idle (as seen in screenshot). 
Also memory

[... truncated for brevity ...]

---

## Issue #N/A: How do i stop the ai from talking to itself??

**Link**: https://github.com/ggml-org/llama.cpp/issues/841
**State**: closed
**Created**: 2023-04-07T23:26:31+00:00
**Closed**: 2024-04-11T01:06:52+00:00
**Comments**: 7
**Labels**: stale

### Description

It sometimes just talks to itself for example:

###Human: Hi
###Assistant: Hello, how can i assist you?

(i am runing the latest release with vicuna mode)

---

## Issue #N/A: [Feature Request] Dawn C++ WebGPU backend

**Link**: https://github.com/ggml-org/llama.cpp/issues/837
**State**: closed
**Created**: 2023-04-07T16:10:18+00:00
**Closed**: 2024-04-11T01:06:53+00:00
**Comments**: 10
**Labels**: enhancement, performance, stale

### Description

Today Chrome [released](https://developer.chrome.com/blog/webgpu-release/) WebGPU support in Chrome Beta. 
The Google's [Dawn](https://dawn.googlesource.com/dawn) project is a C++ standalone implementation of the WebGPU. It enables support of WebGPU in other libraries, by example this [WIP](https://dawn.googlesource.com/dawn/+/refs/heads/main/src/dawn/node/) are NodeJS binding to Dawn, that would enable - in theory - WebGPU in Node. 
So it should be possible to add Dawn as GPU backend to Llama/GGML C++ math operations.

---

## Issue #N/A: Editing a PR description shouldn't cause a CI run

**Link**: https://github.com/ggml-org/llama.cpp/issues/836
**State**: closed
**Created**: 2023-04-07T15:22:00+00:00
**Closed**: 2023-04-22T13:12:30+00:00
**Comments**: 7
**Labels**: build

### Description

This seems excessive but I'm not sure how to turn that off (`edited` in `build.yml`?), without inadvertently also removing other triggers (such as a push to the branch the PR wants to merge).

Also, other actions such as requesting a review should not trigger a CI run IMO.

---

## Issue #N/A: Inconsistent build success across pulls, probably need to track supported compiler and library versions.

**Link**: https://github.com/ggml-org/llama.cpp/issues/834
**State**: closed
**Created**: 2023-04-07T13:57:57+00:00
**Closed**: 2023-04-21T07:53:55+00:00
**Comments**: 5
**Labels**: build

### Description

Before going on I should note, when it does build, things run just fine, even on an old Xeon X5675.

I run multiple linux distros and versions, and depending on the changes any given day, AND which distro/version combination, it might build cleanly or spew a ton of errors.  The issue here is most likely with versioning/feature checking, but I'm unsure.  This time it's from regex, but other times it's other portions: "usr/include/c++/10/bits/regex.tcc:61:26: error:..."

I haven't been formally tracking these yet, so the success rates are basically from memory:
A debian 11 box with gcc-10 seems to have about a 99% success rate.
Ubuntu 20.04 LTS with gcc-9 or gcc-10 seems to have a 50% success rate
openSUSE 15.3 with gcc-7 seems to have about a 90% success rate


# Expected Behavior

The build will succeed, or at very least test if the installed compiler and libraries support the features it needs, and report what's missing.

# Current Behavior
Depending on the day, a long 

[... truncated for brevity ...]

---

## Issue #N/A: Performance degrading over time

**Link**: https://github.com/ggml-org/llama.cpp/issues/832
**State**: closed
**Created**: 2023-04-07T12:48:25+00:00
**Closed**: 2023-04-07T13:13:10+00:00
**Comments**: 1

### Description

# Expected Behavior

When running this command:

`./main -i --interactive-first -r "### Human:" --temp 0 -c 2048 -n -1 --ignore-eos --repeat_penalty 1.2 --threads 4 --instruct -m models/ggml-vicuna-13b-4bit.bin`

I expect the performance to be the same over time when  the model is answering my questions.

# Current Behavior

The performance is good in the begining, answers are written out fast, 4 cpu cores are fully utilized. But over time speed degrades until it slows down to a word every 30 seconds and cpu cores are just idling.

# Environment and Context 

Apple M1 Mac Mini 16GB RAM. Ventura 13.3.

```
Python 3.8.13
GNU Make 3.81
Apple clang version 14.0.0 (clang-1400.0.29.202)

numpy                         1.23.4
rotary-embedding-torch        0.2.1
sentencepiece                 0.1.97
torch                         2.1.0.dev20230307
torchaudio                    2.0.0.dev20230307
torchvision                   0.15.0.dev20230307
```


# Steps to Reprodu

[... truncated for brevity ...]

---

## Issue #N/A: Ignores Cyrillic under Win10

**Link**: https://github.com/ggml-org/llama.cpp/issues/831
**State**: closed
**Created**: 2023-04-07T12:41:30+00:00
**Closed**: 2023-04-07T13:15:22+00:00
**Comments**: 2

### Description

The main.exe program simply does not see ru Cyrillic in the standard windows 10 environment, both in cmd.exe and in powershell

main.exe of https://github.com/ggerganov/llama.cpp/releases/download/master-cc9cee8/llama-master-cc9cee8-bin-win-avx2-x64.zip
model of https://huggingface.co/IlyaGusev/llama_13b_ru_turbo_alpaca_lora_llamacpp/tree/main/13B


# Expected Behavior

_Вопрос: Почему трава зеленая? 
Выход: Трава зеленой из-за того, что она содержит хлорофиллы, пигменты, которые помогают ей фотосинтезировать энергию из солнечного света. Хлорофилл способен перерабатывать углекислый газ и воду в органические вещества, такие как углеводы, аминокислоты и жиры, которые необходимы растениям для их роста и развития._


# Current Behavior

PS N:\NLP_MODEL> .\llama-master-cc9cee8-bin-win-avx2-x64\main.exe -m .\llama_13b_ru_turbo_alpaca_lora_llamacpp\ggml-model-q4_0.bin -p "Вопрос: Почему трава зеленая? Ответ:" -n 512 --temp 0.1
main: seed = 1680869612
llama_model_load: loading 

[... truncated for brevity ...]

---

## Issue #N/A: Intermittent segmentation faults in llama_sample_top_p_top_k()

**Link**: https://github.com/ggml-org/llama.cpp/issues/830
**State**: closed
**Created**: 2023-04-07T12:33:14+00:00
**Closed**: 2024-04-11T01:06:54+00:00
**Comments**: 5
**Labels**: stale

### Description

# Expected Behavior

I have been getting intermittent segfaults for no apparent reason. Sometimes they occur right at the beginning of text generation, and sometimes they occur after a lot of text has already been generated. They seem to be deterministic in that I can sometimes work around them by changing the prompt, but if I don’t change the prompt, they consistently occur. I normally use the 65B model, which exhibits the problem, but I am attaching a repro for the 13B model. I am not 100% sure but I believe the issue affects all four model sizes (7B, 13B, 30B, 65B).

# Current Behavior

Intermittent segfaults

# Environment and Context 

Please provide detailed information about your computer setup. This is important in case the issue is not reproducible except for under certain specific conditions.

* Physical (or virtual) hardware you are using, e.g. for Linux:

2019 16-inch MacBook Pro, 2.3 GHz 8-Core Intel Core i9, 64 GB of RAM

* Operating System, e.g. for Linux

[... truncated for brevity ...]

---

## Issue #N/A: Port to Google Tensor G2/G3 on Pixel Phones

**Link**: https://github.com/ggml-org/llama.cpp/issues/829
**State**: closed
**Created**: 2023-04-07T10:57:33+00:00
**Closed**: 2024-05-20T01:09:08+00:00
**Comments**: 13
**Labels**: stale

### Description

It would be nice to use the TPU in those SoCs to improve speed

---

## Issue #N/A: Do not recreate context while LLama is writing

**Link**: https://github.com/ggml-org/llama.cpp/issues/828
**State**: closed
**Created**: 2023-04-07T09:41:24+00:00
**Closed**: 2024-04-11T01:06:55+00:00
**Comments**: 5
**Labels**: stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Tokens are generated at about a constant rate, ie. N tokens per second on a given machine.

# Current Behavior

Sometimes, the LLM takes a much longer time to generate a token than usually. It can be a 10x slowdown.

# Environment and Context 

**Setup**
MacB

[... truncated for brevity ...]

---

## Issue #N/A: [Question] Save internal state to disk

**Link**: https://github.com/ggml-org/llama.cpp/issues/827
**State**: closed
**Created**: 2023-04-07T08:15:24+00:00
**Closed**: 2023-04-08T12:21:54+00:00
**Comments**: 2

### Description

Hi,

Thanks for your hard work on this project.

I've been playing with the code since few days. I'm trying to find a way to save the internal state of the model (or its context) that can be reused later. But until now it still doesn't work. I don't know if I'm missing something (I'm not good at all when talking about machine learning, I'm working more on system development.)

Here what I tried (inspired from code of `main.cpp`)

1. Load the model using `llama_init_from_file`
2. Call `llama_eval` on a prompt, for example `" Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"`
3. Call `llama_eval` on a instruction, for example `"### Instruction: Hi, I'm Xuan Son."`
4. Save the `kv_self` buffer using `llama_get_kv_cache` => Expected: the saved data contains information about my name
5. Ask for `what is my name?`, the model correctly response that my name is Xuan Son
6. Exit the program
7. Re-run the program
8. Relo

[... truncated for brevity ...]

---

## Issue #N/A: problem with the ggml files when using the docker images

**Link**: https://github.com/ggml-org/llama.cpp/issues/826
**State**: closed
**Created**: 2023-04-07T07:16:21+00:00
**Closed**: 2024-04-11T01:06:56+00:00
**Comments**: 2
**Labels**: stale

### Description

I have a problem with the ggml images when I use the docker image ghcr.io/ggerganov/llama.cpp:full. I am asked to regenerate your ggml files but I have already done that. See attached screenshot.
![Screenshot from 2023-04-07 08-47-47](https://user-images.githubusercontent.com/35671475/230561232-9056fb60-6886-4a1f-b8f3-b5c1b7a51d86.png)


---

## Issue #N/A: [User] Insert summary of your issue or enhancement..

**Link**: https://github.com/ggml-org/llama.cpp/issues/823
**State**: closed
**Created**: 2023-04-06T23:24:03+00:00
**Closed**: 2023-04-06T23:48:12+00:00
**Comments**: 0

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ ] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Please provide a detailed written description of what you were trying to do, and what you expected `llama.cpp` to do.

# Current Behavior

Please provide a detailed written description of what `llama.cpp` did, instead. 

# Environment and Context 

Please provide detailed inform

[... truncated for brevity ...]

---

## Issue #N/A: [User] Insert summary of your issue or enhancement..

**Link**: https://github.com/ggml-org/llama.cpp/issues/822
**State**: closed
**Created**: 2023-04-06T23:23:10+00:00
**Closed**: 2023-04-06T23:48:30+00:00
**Comments**: 0

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ ] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ ] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Please provide a detailed written description of what you were trying to do, and what you expected `llama.cpp` to do.

# Current Behavior

Please provide a detailed written description of what `llama.cpp` did, instead. 

# Environment and Context 

Please provide detailed inform

[... truncated for brevity ...]

---

## Issue #N/A: [User] Insert summary of your issue or enhancement..

**Link**: https://github.com/ggml-org/llama.cpp/issues/821
**State**: closed
**Created**: 2023-04-06T23:19:27+00:00
**Closed**: 2023-04-06T23:48:50+00:00
**Comments**: 0

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Please provide a detailed written description of what you were trying to do, and what you expected `llama.cpp` to do.

# Current Behavior

Please provide a detailed written description of what `llama.cpp` did, instead. 

# Environment and Context 

Please provide detailed inform

[... truncated for brevity ...]

---

## Issue #N/A: Compiling with LLAMA_OPENBLAS=1 does not seem to improve performance

**Link**: https://github.com/ggml-org/llama.cpp/issues/817
**State**: closed
**Created**: 2023-04-06T16:13:36+00:00
**Closed**: 2023-04-07T08:56:00+00:00
**Comments**: 2

### Description

# Expected Behavior
Compiling llama.cpp with `make LLAMA_OPENBLAS=1` should give a slight performance bump in prompt ingestion, and no change (or reduced) cpu usage in text generation.

# Current Behavior

Compiled llama.cpp with `make LLAMA_OPENBLAS=1`. Compilation seems to work fine, but when running ./main for generation, I find no difference in the rate of prompt ingestion of generation. Can confirm that BLAS=1 shows up in the model loading info.

Also, when compiled with BLAS , if -t is not explicitly set, it seems to default to all threads.

# Environment and Context 
Ubuntu 22.04.1 LTS
Python 3.10.6
GNU Make 4.3
g++ (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0

Vendor ID:                       AuthenticAMD
Model name:                      AMD EPYC 7742 64-Core Processor
Flags:                           fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good n

[... truncated for brevity ...]

---

## Issue #N/A: [Enhancement]: Implement optimizations used in CTranslate2

**Link**: https://github.com/ggml-org/llama.cpp/issues/811
**State**: closed
**Created**: 2023-04-06T14:13:22+00:00
**Closed**: 2024-04-11T01:06:58+00:00
**Comments**: 3
**Labels**: enhancement, stale

### Description

[CTranslate2](https://github.com/OpenNMT/CTranslate2) is a "competitor" to llama.cpp that advertises itself with:
> ### Fast and efficient execution on CPU and GPU
> The execution [is significantly faster and requires less resources](https://github.com/ggerganov/llama.cpp/issues/new?assignees=&labels=&template=custom.md&title=%5BUser%5D+Insert+summary+of+your+issue+or+enhancement..#benchmarks) than general-purpose deep learning frameworks on supported models and tasks thanks to many advanced optimizations: layer fusion, padding removal, batch reordering, in-place operations, caching mechanism, etc.

I am no expert in LLMs and I don't know what these optimizations are, but I am asking: would it be possible/feasible and/or desirable to implement these optimizations into llama.cpp or GGML?


---

## Issue #N/A: How do i use convert-unversioned-ggml-to-ggml.py?

**Link**: https://github.com/ggml-org/llama.cpp/issues/808
**State**: closed
**Created**: 2023-04-06T12:22:58+00:00
**Closed**: 2023-04-14T13:12:39+00:00
**Comments**: 12
**Labels**: bug, model

### Description

Hi it told me to use the convert-unversioned-ggml-to-ggml.py file and gave me an error saying your gpt4all model is too old. So i converted the gpt4all-lora-unfiltered-quantized.bin file with llama tokenizer. And it generated some kind of orig file in the same directory where the model was. When i tried to run the miku.sh file which had the latest generated file as model it gave me another error stating this 
`main: seed = 1680783525
llama_model_load: loading model from './models/gpt4all-7B/gpt4all-lora-unfiltered-quantized.bin' - please wait ...
./models/gpt4all-7B/gpt4all-lora-unfiltered-quantized.bin: invalid model file (bad magic [got 0x67676d66 want 0x67676a74])
        you most likely need to regenerate your ggml files
        the benefit is you'll get 10-100x faster load times
        see https://github.com/ggerganov/llama.cpp/issues/91
        use convert-pth-to-ggml.py to regenerate from original pth
        use migrate-ggml-2023-03-30-pr613.py if you deleted originals

[... truncated for brevity ...]

---

## Issue #N/A: Specifications for how to integrate services in AI models

**Link**: https://github.com/ggml-org/llama.cpp/issues/805
**State**: closed
**Created**: 2023-04-06T08:54:29+00:00
**Closed**: 2023-04-06T14:04:50+00:00
**Comments**: 1

### Description

Please join discussion. 

Here is a repository I created for specs, for now it's empty. Discussion is welcome. https://github.com/openservices4ai/spec

**What is an AI-integrated sevice** 

As AI models trained on a fixed set of data and cannot learn something new in real-time while serving, services integrated in AI models help provide real-time information and services. For example, a chatbot model may be integrated with a todo list service to help users add/remove todos, or create a timer. 

**Why an unified specification for AI-integrated services matters**

First, it can help improve compatibility between different AI platforms, making it easier for developers to integrate various plugins into their AIs. This can save time and resources for developers who would otherwise have to manually modify each plugin to work with their chatbot.

Second, an unified specification can help ensure that services adhere to certain standards for functionality, security, and user privacy

[... truncated for brevity ...]

---

## Issue #N/A: wasm simd 128 build failure regression

**Link**: https://github.com/ggml-org/llama.cpp/issues/804
**State**: closed
**Created**: 2023-04-06T07:49:12+00:00
**Closed**: 2023-04-12T15:01:15+00:00
**Comments**: 3
**Labels**: bug

### Description

I’m trying out building on a-shell on iOS which is a wasm terminal platform without gnu make.

It looks like since c1f885067c61191a07a1aedf684168dda62f3f71 there are undefined symbols in the wasm simd 128 code:
```
clang -pipe  -msimd128 -D_WASI_EMULATED_MMAN -D_WASI_EMULATED_PROCESS_CLOCKS   -c ggml.c -o ggml.o
ggml.c:2090:43: error: use of undeclared identifier 'px'
        const block_q4_0 * restrict x0 = &px[i + 0];
                                          ^
```

These undefined symbols were first introduced on line 1726 of ggml.c https://github.com/ggerganov/llama.cpp/commit/c1f885067c61191a07a1aedf684168dda62f3f71#diff-6d9ce99fcb6f51ff76f59e479f6e6fc0bb62edef7442805d7a5bb15b23996b5dR1726 (github seems to need the large diff manually expanded) and is presently at https://github.com/ggerganov/llama.cpp/blob/d2beca95dcfcd6f1145886e914b879ffc3604b7a/ggml.c#L2090

---

## Issue #N/A: llama.cpp acts too dumb while running on phone!!

**Link**: https://github.com/ggml-org/llama.cpp/issues/802
**State**: closed
**Created**: 2023-04-06T06:43:33+00:00
**Closed**: 2023-04-15T17:23:12+00:00
**Comments**: 13
**Labels**: generation quality

### Description

I was trying llama.cpp on phone with termux installed. but look at this image
![Screenshot_20230406-120404](https://user-images.githubusercontent.com/97907864/230291536-8dbfeff4-0456-4328-a2dc-1bb35614f742.png)

**Specifications**
The phone has 8 gigs of RAM and 7 gigs is free and the CPU has 8 cores so its not the issue of the RAM and CPU.
Model used: alpaca-7B-lora
llama.cpp version: latest
prompt: chat-with-bob.txt

I really don't know what is causing the issue here. The problem happening is, when i ask a question to it, it just either answers the question in a very dumb way or it just repeats the same question not answering anything. With the same model, prompt and llama.cpp version on my PC with 4GB ram works as expected it answers every question with almost 98% accuracy. Can any of you guys help me out with this? or update the llama.cpp and fix the mobile issues please?

Thankyou

---

## Issue #N/A: Do you guys mind making an updater for llama.cpp?

**Link**: https://github.com/ggml-org/llama.cpp/issues/800
**State**: closed
**Created**: 2023-04-06T05:50:34+00:00
**Closed**: 2023-04-06T06:18:15+00:00
**Comments**: 2

### Description

Basically when you guys release a new version of llama.cpp the updater will ask a prompt to you saying there is a new update available (that will be fetched from github commits) and it will ask you if it can update the files for you, if you say yes then it will fetch the **Changed/previously edited files** from github and replace it with your local llama files.

This came to my mind because llama.cpp has become a big project now. You guys are updating the project every single day which is awesome but it makes the people replace the files on their own every single day. That might be annoying. 

---

## Issue #N/A: What would it take to 100x the context window?

**Link**: https://github.com/ggml-org/llama.cpp/issues/799
**State**: closed
**Created**: 2023-04-06T03:18:40+00:00
**Closed**: 2024-04-11T01:06:59+00:00
**Comments**: 31
**Labels**: enhancement, stale

### Description

Thinking about what could be done when large language models can operate on phenomenally large context, and wondering what it might actually take to get there.

And realised this repo has a ton of really bright people in orbit, who actually understand brass tacks what might be involved.

Assuming it's really desirable, what hacks could be done to get there?

- What options are there?
- How much space or time would be involved?
- Would the models need full retraining?
- What about for 10x?

🙏

---

## Issue #N/A: [Bug] dequantize_row_q4_0  segfaults

**Link**: https://github.com/ggml-org/llama.cpp/issues/791
**State**: closed
**Created**: 2023-04-05T20:00:07+00:00
**Closed**: 2023-04-05T20:31:25+00:00
**Comments**: 5

### Description

# Environment and Context 

Linux  5.10.0-21-amd64 #1 SMP Debian 5.10.162-1 (2023-01-21) x86_64 GNU/Linux
g++ (Debian 10.2.1-6) 10.2.1 20210110
GNU Make 4.3

# Failure Information (for bugs)

main segfaults at  dequantize_row_q4_0+48

# Steps to Reproduce

./main  -m models/ggml-vocab-q4_0.bin



~/s/llama.cpp ❯❯❯ gdb main
(gdb) r -m models/ggml-vocab-q4_0.bin
Starting program: /home/sha0/soft/llama.cpp/main -m models/ggml-vocab-q4_0.bin
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
main: seed = 1680724006
llama_model_load: loading model from 'models/ggml-vocab-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 4096
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 32
llama_model_load: n_layer = 32
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 11008
llama_mo

[... truncated for brevity ...]

---

## Issue #N/A: Prompt eval time is counted twice

**Link**: https://github.com/ggml-org/llama.cpp/issues/790
**State**: closed
**Created**: 2023-04-05T19:59:57+00:00
**Closed**: 2024-04-11T01:07:00+00:00
**Comments**: 4
**Labels**: bug, stale

### Description

Creating a new issue so this doesn't get forgotten:

@KASR posted a CSV of processing times in https://github.com/ggerganov/llama.cpp/issues/603#issuecomment-1492941163

But the times don't add up: If you take the total time, and subtract the partial times that are supposed to add up to it, the result is all over the place:

![image](https://user-images.githubusercontent.com/4478/229292361-52df310b-e838-46b6-a68c-5c1d69cf5f87.png)

The clue lies in the comment by @ggerganov :
> @sw 
> After the `mmap` changes, the `load` time is incorrect:
> 
> https://github.com/ggerganov/llama.cpp/blob/6e7801d08d81c931a5427bae46f00763e993f54a/llama.cpp#L1681-L1685
> 
> Currently, the reported load time includes not only the page faults, but also the prompt eval time. So effectively, you get the negative number since the prompt eval time has been accounted 2 times.
We have to fix this.

_Originally posted by @ggerganov in https://github.com/ggerganov/llama.cpp/issues/603#issuecomment-

[... truncated for brevity ...]

---

## Issue #N/A: [User] Insert summary of your issue or enhancement..

**Link**: https://github.com/ggml-org/llama.cpp/issues/789
**State**: closed
**Created**: 2023-04-05T19:59:06+00:00
**Closed**: 2023-04-06T00:29:02+00:00
**Comments**: 0

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Please provide a detailed written description of what you were trying to do, and what you expected `llama.cpp` to do.

# Current Behavior

Please provide a detailed written description of what `llama.cpp` did, instead. 

# Environment and Context 

Please prov

[... truncated for brevity ...]

---

## Issue #N/A: Compilation failed on macOS 10.7-8-9: 'clock_gettime' produce warnings and errors

**Link**: https://github.com/ggml-org/llama.cpp/issues/788
**State**: closed
**Created**: 2023-04-05T19:16:14+00:00
**Closed**: 2024-04-11T01:07:01+00:00
**Comments**: 20
**Labels**: bug, build, macos, stale

### Description

# PREREQUISITES
- I am running the latest code: [5a8c4f6](https://github.com/ggerganov/llama.cpp/releases/tag/master-5a8c4f6)
- I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- I have created a [relevant issue](https://github.com/antimatter15/alpaca.cpp/issues/201) in alpaca.cpp.
- I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# EXPECTED BEHAVIOR
* Attempted to compile the binary for macOS 10.7, 10.8 and 10.9.
* Expected to run the chat app on an old macOS, that will be isolated from Internet.

# ACTUAL BEHAVIOR
* Compilation is terminated with warnings and errors.

# ENVIRONMENT AND CONTEXT
* Macbook pro 15 2012: macOS 10.8 Mountain Lion on Core i7 + 512 SDD + 16Gb RAM
	* Parallels Virtual Machine: macOS 10.7 Lion on 20Gb HDD + 4Gb RAM
		* X-Code 4.6.3
		* Command Line Tools OS X Lion Nov2012
		* MacPorts 2.8.1 10.7 (Lion)
			*

[... truncated for brevity ...]

---

## Issue #N/A: Changing default repeat_last_n value to current context size?

**Link**: https://github.com/ggml-org/llama.cpp/issues/787
**State**: closed
**Created**: 2023-04-05T18:18:14+00:00
**Closed**: 2024-04-11T01:07:03+00:00
**Comments**: 3
**Labels**: enhancement, generation quality, stale

### Description

I noticed that llama 7b almost always gets stuck in a loop after a certain amount of time. This problem has reoccurred to me throughout the all time I have been trying to use llama.cpp (since March 15). I have also tried different models such as alpaca and gpt4all unfiltered, but the problem remains still. It also becomes obvious when you try to generate a dialog following some kind of plot (I use --keep to keep the plot summary in context). All the times I've tried to generate something infinite, it just loops at some point, even in interactive mode.

I also noticed, that setting repeat_last_n to current context size helps to eliminate this issue. (I use ctx_size 2048 for the most time) 

Maybe after some testing, default repeat_last_n value could be changed to currently set context size, so newbies could bypass this issue?

---

## Issue #N/A: Multi-thread ggml_cpy()

**Link**: https://github.com/ggml-org/llama.cpp/issues/782
**State**: closed
**Created**: 2023-04-05T16:24:00+00:00
**Closed**: 2023-04-10T19:47:15+00:00
**Comments**: 5
**Labels**: enhancement, good first issue, performance

### Description

This is a task suitable for new contributors

See how we multi-threaded the [ggml_rope()](https://github.com/ggerganov/llama.cpp/pull/781) operator.
Do the same for the `ggml_cpy()` operator and see if there is any benefit.

Use the [ggml profiler (GGML_PERF)](https://github.com/ggerganov/llama.cpp/wiki/GGML-Tips-&-Tricks#measuring-the-performance-of-the-inference) to measure the benefit of multi-threaded vs non-multi-threaded `ggml_cpy()`

---

## Issue #N/A: docker image ghcr.io/ggerganov/llama.cpp:light not working

**Link**: https://github.com/ggml-org/llama.cpp/issues/776
**State**: closed
**Created**: 2023-04-05T14:35:53+00:00
**Closed**: 2023-04-05T17:21:42+00:00
**Comments**: 3

### Description

The docker image ghcr.io/ggerganov/llama.cpp:light no longer works. It exits without giving the answer.

---

## Issue #N/A: Error in conversion

**Link**: https://github.com/ggml-org/llama.cpp/issues/774
**State**: closed
**Created**: 2023-04-05T13:44:55+00:00
**Closed**: 2023-04-14T13:16:11+00:00
**Comments**: 3
**Labels**: bug

### Description

# Current Behavior
While converting the 7B model I got the error:
```
{'dim': 4096, 'multiple_of': 256, 'n_heads': 32, 'n_layers': 32, 'norm_eps': 1e-06, 'vocab_size': -1}
Traceback (most recent call last):
  File "/content/llama.cpp/llama.cpp/llama.cpp/llama.cpp/convert-pth-to-ggml.py", line 274, in <module>
    main()
  File "/content/llama.cpp/llama.cpp/llama.cpp/llama.cpp/convert-pth-to-ggml.py", line 239, in main
    hparams, tokenizer = load_hparams_and_tokenizer(dir_model)
  File "/content/llama.cpp/llama.cpp/llama.cpp/llama.cpp/convert-pth-to-ggml.py", line 105, in load_hparams_and_tokenizer
    tokenizer = SentencePieceProcessor(fname_tokenizer)
  File "/usr/local/lib/python3.9/dist-packages/sentencepiece/__init__.py", line 447, in Init
    self.Load(model_file=model_file, model_proto=model_proto)
  File "/usr/local/lib/python3.9/dist-packages/sentencepiece/__init__.py", line 905, in Load
    return self.LoadFromFile(model_file)
  File "/usr/local/lib/python3.9/

[... truncated for brevity ...]

---

## Issue #N/A: Running a Vicuna-13B 4it model ?

**Link**: https://github.com/ggml-org/llama.cpp/issues/771
**State**: closed
**Created**: 2023-04-05T07:33:04+00:00
**Closed**: 2023-07-28T19:47:57+00:00
**Comments**: 25
**Labels**: model, generation quality

### Description

I found this model : 
[[ggml-vicuna-13b-4bit](https://huggingface.co/eachadea/ggml-vicuna-13b-4bit)](https://huggingface.co/eachadea/ggml-vicuna-13b-4bit/tree/main) and judging by their online demo it's very impressive.
I tried to run it with llama.cpp latest version - the model loads fine, but as soon as it loads it starts hallucinating and quits by itself. 
Do I need to have it converted or something like that ?

---

## Issue #N/A: Token generation is extremely slow when using 13B models on an M1 Pro with llama.cpp, but it runs at a fine speed with Dalai (which uses an older version of llama.cpp)

**Link**: https://github.com/ggml-org/llama.cpp/issues/767
**State**: closed
**Created**: 2023-04-04T17:33:04+00:00
**Closed**: 2023-07-28T19:47:23+00:00
**Comments**: 30
**Labels**: performance

### Description

# Expected Behavior

I can load a 13B model and generate text with it with decent token generation speed with a M1 Pro CPU (16 GB RAM).

# Current Behavior

When I load a 13B model with llama.cpp (like Alpaca 13B or other models based on it) and I try to generate some text, every token generation needs several seconds, to the point that these models are not usable for how unbearably slow they are. But they works with reasonable speed using Dalai, that uses an older version of llama.cpp

# Environment and Context 

MacBook Pro with M1 Pro, 16 GB RAM, macOS Ventura 13.3.

Python 3.9.16

GNU Make 3.81

Apple clang version 14.0.3 (clang-1403.0.22.14.1)
Target: arm64-apple-darwin22.4.0
Thread model: posix

If you need some kind of log or other informations, I will post everything you need. Thanks in advance.

---

## Issue #N/A: [Bug] Different outputs when undefining GGML_SIMD

**Link**: https://github.com/ggml-org/llama.cpp/issues/766
**State**: closed
**Created**: 2023-04-04T16:03:51+00:00
**Closed**: 2023-04-04T18:53:09+00:00
**Comments**: 2

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Same model outputs with enabled and disabled SIMD instructions.

# Environment and Context 

`$ lscpu`
```
Architecture:            x86_64
  CPU op-mode(s):        32-bit, 64-bit
  Address sizes:         48 bits physical, 48 bits virtual
  Byte Order:        

[... truncated for brevity ...]

---

## Issue #N/A: Feature to Discard Last Generated Message in Interactive Chat Mode?

**Link**: https://github.com/ggml-org/llama.cpp/issues/764
**State**: closed
**Created**: 2023-04-04T13:50:05+00:00
**Closed**: 2024-04-11T01:07:04+00:00
**Comments**: 2
**Labels**: duplicate, enhancement, stale

### Description

We have ctrl+c to stop generate.
Can we have undo feature to take the last message out of context and regenerate during an interactive chat session if you don't like what it generated?

---

## Issue #N/A: Can i quantize a 4Bit model more?

**Link**: https://github.com/ggml-org/llama.cpp/issues/762
**State**: closed
**Created**: 2023-04-04T10:23:08+00:00
**Closed**: 2023-04-04T17:39:38+00:00
**Comments**: 3

### Description

Hi i want to quantize a model which is already quantized to 4bit ``q4_1`` but i want to make it compute faster so i wanted to ask what is the command to quantize the quantized module. I tried once with the command that is in the readme file but that didnt work. so can anyone help me?

---

## Issue #N/A: Small layout error in the documentation

**Link**: https://github.com/ggml-org/llama.cpp/issues/759
**State**: closed
**Created**: 2023-04-04T06:25:57+00:00
**Closed**: 2023-04-04T06:26:54+00:00
**Comments**: 0

### Description

Wrong repo

---

## Issue #N/A: Something strange with any non-english prompts with alpaca/gpt4all/vicuna

**Link**: https://github.com/ggml-org/llama.cpp/issues/758
**State**: closed
**Created**: 2023-04-04T06:10:41+00:00
**Closed**: 2023-04-04T07:55:03+00:00
**Comments**: 0

### Description

When you try to chat with these on not english language, the answer will hallucinate and on english, but if you use file as prompt, everything works fine(picrelated). This is strange.
![IMG_20230404_090641_914](https://user-images.githubusercontent.com/109795993/229702938-279e61be-0e63-4f5d-9ee7-bd66a9d37283.jpg)


---

## Issue #N/A: Vicuna works sometimes strange

**Link**: https://github.com/ggml-org/llama.cpp/issues/757
**State**: closed
**Created**: 2023-04-04T06:09:47+00:00
**Closed**: 2023-04-04T06:14:00+00:00
**Comments**: 1

### Description

The quantized weights is here https://huggingface.co/eachadea/ggml-vicuna-13b-4bit/tree/main
But when you rin this in instructions mode, it will add the training dataset artifacts such as ###Instruction: in discussion. Is it possible to control these markers or add special mode that will stop the answer when ###

---

## Issue #N/A: can llama do other task except text-generate,like translate

**Link**: https://github.com/ggml-org/llama.cpp/issues/755
**State**: closed
**Created**: 2023-04-04T03:12:51+00:00
**Closed**: 2023-04-04T07:56:12+00:00
**Comments**: 1

### Description

thanks for your work, it's very helpful!
can llama do other jod:
`./main -m ./models/7B/ggml-model-q4_0.bin -p "Translate English to Frence: it's a nic day!" -n 512`


---

## Issue #N/A: [Feature Request] support lit-llama

**Link**: https://github.com/ggml-org/llama.cpp/issues/754
**State**: closed
**Created**: 2023-04-04T01:59:23+00:00
**Closed**: 2023-04-04T02:23:34+00:00
**Comments**: 1

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Request
Hello. There is lit-llama (https://github.com/Lightning-AI/lit-llama) is released.
It is licensed by Apache 2.0. So it can be used for commercial.
Could you support this model in this repo?

---

## Issue #N/A: Error when running make

**Link**: https://github.com/ggml-org/llama.cpp/issues/751
**State**: closed
**Created**: 2023-04-03T22:41:44+00:00
**Closed**: 2023-04-04T00:34:40+00:00
**Comments**: 3

### Description

I installed CMake, but when I run make, I get this error:

/bin/sh: line 1: cc: command not found
/bin/sh: line 1: g++: command not found
I llama.cpp build info:
I UNAME_S:  MSYS_NT-10.0-22621
I UNAME_P:  unknown
I UNAME_M:  x86_64
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -Wno-unused-function -march=native -mtune=native
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function
I LDFLAGS:
I CC:
I CXX:

cc  -I.              -O3 -DNDEBUG -std=c11   -fPIC -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -Wno-unused-function -march=native -mtune=native   -c ggml.c -o ggml.o
make: cc: No such file or directory
make: *** [Makefile:142: ggml.o] Error 127


Is there something else that I need to install? I am running this on Windows 11.

---

## Issue #N/A: Converting Ilama 4bit GPTQ Model from HF does not work

**Link**: https://github.com/ggml-org/llama.cpp/issues/746
**State**: closed
**Created**: 2023-04-03T18:53:49+00:00
**Closed**: 2023-05-22T07:59:53+00:00
**Comments**: 11
**Labels**: bug, high priority

### Description

Hi! I tried to use the 13B Model from https://huggingface.co/maderix/llama-65b-4bit/

I converted the model using 

`python convert-gptq-to-ggml.py models/llama13b-4bit.pt models/tokenizer.model models/llama13b-4bit.bin`

If I understand it correctly I still need to migrate the model and I tried it using

`python migrate-ggml-2023-03-30-pr613.py models/llama13b-4bit.bin models/llama13b-4bit-new.bin`

But after a few seconds this breaks with the following error:

```
Processing part 1 of 1

Processing tensor b'tok_embeddings.weight' with shape: [32000, 5120] and type: F16
Traceback (most recent call last):
  File "/home/dust/llama.cpp/migrate-ggml-2023-03-30-pr613.py", line 311, in <module>
    main()
  File "/home/dust/llama.cpp/migrate-ggml-2023-03-30-pr613.py", line 306, in main
    copy_tensors(fin, fout, part_id, n_parts)
  File "/home/dust/llama.cpp/migrate-ggml-2023-03-30-pr613.py", line 169, in copy_tensors
    assert n_dims in (1, 2)
AssertionError
```


[... truncated for brevity ...]

---

## Issue #N/A: [Work Group] Add RLHF like ColosallChat on bigger dataset to achieve ChatGPT quality

**Link**: https://github.com/ggml-org/llama.cpp/issues/743
**State**: closed
**Created**: 2023-04-03T18:02:06+00:00
**Closed**: 2023-04-13T08:55:27+00:00
**Comments**: 5

### Description

[Link to ColosallChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat
)
# Add RLHF like ColosallChat on bigger dataset to achieve ChatGPT quality

![lama-alpaca](https://user-images.githubusercontent.com/84633629/229592343-36b95e81-2f85-4fa0-9a5a-edd5267bb190.gif)


Although models in the GPT series, such as ChatGPT and GPT-4, are highly powerful, they are unlikely to be fully open-sourced. Fortunately, the open-source community has been working hard to address this.

For example, Meta has open-sourced the LLaMA model, which offers parameter sizes ranging from 7 billion to 65 billion. A 13 billion parameter model can outperform the 175 billion GPT-3 model on most benchmark tests. However, since it doesn’t have an instruct tuning stage, its actual generated results are not satisfactory.

Stanford’s Alpaca generates training data in a self-instructed manner by calling OpenAI’s API. With only 7 billion parameters, this lightweight model can be fine-tuned at

[... truncated for brevity ...]

---

## Issue #N/A: Pythia Support?

**Link**: https://github.com/ggml-org/llama.cpp/issues/742
**State**: closed
**Created**: 2023-04-03T16:51:43+00:00
**Closed**: 2024-05-23T09:49:54+00:00
**Comments**: 3
**Labels**: good first issue, model

### Description

Hi, I've found out that Pythia (from EleutherAI) is better that Cerebras GPT in terms of evaluation results. Pythia is basically a LLM that based on GPT NeoX architecture but it's parameters ranging from 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B to 13B. Pythia itself is available in GitHub repository "EleutherAI/pythia" and Huggingface Models under the same name.

Does this project support Pythia based models? If no, any plans on supporting them afterwards? I appreciate the implementation of Pythia in this project. Thank you very much before.

---

## Issue #N/A: Add OpenCL Support

**Link**: https://github.com/ggml-org/llama.cpp/issues/741
**State**: closed
**Created**: 2023-04-03T12:53:28+00:00
**Closed**: 2023-04-04T19:57:51+00:00
**Comments**: 10
**Labels**: wontfix

### Description

Please consider adding OpenCL support for devices with GPU's that Support OpenCL

---

## Issue #N/A: [User] Insert summary of your issue or enhancement..

**Link**: https://github.com/ggml-org/llama.cpp/issues/739
**State**: closed
**Created**: 2023-04-03T09:37:50+00:00
**Closed**: 2023-04-04T19:29:01+00:00
**Comments**: 1
**Labels**: invalid

### Description

Hello, is it possible to save the robot's response in a variable? to then read it in a request?
Example
Me : Hello how are you ?
Bot : I'm fine and you ?

Save result response in new_varifable for use this :
http://127.0.0.1:8888/?tts=new_variable

---

## Issue #N/A: [User] Insert summary of your issue or enhancement..

**Link**: https://github.com/ggml-org/llama.cpp/issues/738
**State**: closed
**Created**: 2023-04-03T09:08:18+00:00
**Closed**: 2023-04-03T11:13:36+00:00
**Comments**: 0

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Please provide a detailed written description of what you were trying to do, and what you expected `llama.cpp` to do.

# Current Behavior

Please provide a detailed written description of what `llama.cpp` did, instead. 

# Environment and Context 

Please provide detailed inform

[... truncated for brevity ...]

---

## Issue #N/A: I'm pegging CPU (`./examples/chat.sh` works very slowly) on a 5800X3D / u22 linux, anything that can be done?

**Link**: https://github.com/ggml-org/llama.cpp/issues/735
**State**: closed
**Created**: 2023-04-03T05:38:52+00:00
**Closed**: 2024-04-11T01:07:06+00:00
**Comments**: 5
**Labels**: stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Faster responses.

# Current Behavior

Used all 16 threads / 8 cores for seconds to minutes when responding to chat mode.

# Environment and Context 

Please provide detailed information about your computer setup. This is important in case the issue is not rep

[... truncated for brevity ...]

---

## Issue #N/A: How different is Macbook / non-macbook performance?

**Link**: https://github.com/ggml-org/llama.cpp/issues/731
**State**: closed
**Created**: 2023-04-03T03:37:06+00:00
**Closed**: 2023-04-07T16:39:35+00:00
**Comments**: 1

### Description

I was wondering how performant the Llama models are on x86-64. I assumed performance would be similar, but it seems like on M1 macs have lots of CPU improvements that would mean, for example, a llama-13B model running on an M1 mac would not also be able to run on an Lenovo T580.

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

## Issue #N/A: Using Repeat_last_n and Repeat_penalty To Avoid Going Back and Repeating Itself?

**Link**: https://github.com/ggml-org/llama.cpp/issues/727
**State**: closed
**Created**: 2023-04-03T00:58:19+00:00
**Closed**: 2024-04-11T01:07:08+00:00
**Comments**: 2
**Labels**: stale

### Description

I set --repeat_last_n 256 --repeat_penalty 1.9.
However, after a while, it keeps going back to certain sentences and repeating itself as if it's stuck in a loop.
Is this a bug, or am I using the parameters incorrectly?
What's the maximum value you can set for repeat_penalty, so it doesn't repeat itself?

---

## Issue #N/A: Add support for stopping words [Automation]

**Link**: https://github.com/ggml-org/llama.cpp/issues/726
**State**: closed
**Created**: 2023-04-02T23:45:13+00:00
**Closed**: 2023-07-28T19:46:36+00:00
**Comments**: 3

### Description

A stopping criteria or "stop sequence" would be highly appreciated here. These models are already pretty for information retrieval but for automating stuff be it maybe some hacky way to execute code or simply automate textual tasks is impossible. You can just use the model and expect it to work everytime with every input. And even if is the case, your won't last long before your prompt needs a modification.


Will you at least include that feature some day? I'm mot a cpp expert so I'm not sure how difficult would be this to implement.

---

## Issue #N/A: [Feature Request] Support for Filter Assisted Decoding/Constrained Text Generation

**Link**: https://github.com/ggml-org/llama.cpp/issues/725
**State**: closed
**Created**: 2023-04-02T21:58:38+00:00
**Closed**: 2024-04-11T01:07:09+00:00
**Comments**: 1
**Labels**: stale

### Description

## Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

## Feature Request

Hi all, I showed in a [recent paper at COLING 2022](https://paperswithcode.com/paper/most-language-models-can-be-poets-too-an-ai) that filtering a language models vocabulary according to lexical, semantic, or phonetic constraints at each time-step is an extremely powe

[... truncated for brevity ...]

---

## Issue #N/A: Not getting randomized output for the same prompt despite seed changing 

**Link**: https://github.com/ggml-org/llama.cpp/issues/723
**State**: closed
**Created**: 2023-04-02T21:29:22+00:00
**Closed**: 2023-04-03T22:02:02+00:00
**Comments**: 3

### Description

Recently I noticed that I'm getting the exact same answers to my prompt every time where this wasn't the case before. I have no idea what happened to trigger this.

I have a version of chat-with-bob.txt where I ask it to tell my a children's story based on preferences I specified in the text in the prompt and up until now it was obviously using a different seed every time and thus generating a new story every time but out of nowhere I'm getting an identical story every time (despite the fact that the output IS showing a different seed every time upon execution). 

As a test, I changed a couple of characters in the prompt in chat-with-bob.txt to see if the output would change and it did indeed generate an entirely different story but again, it exhibited the same behavior and repeatedly gave me the same story when I ran the identical prompt. FYI, I changed "fantasy story" to "fantastical story in chat-with-bob.txt.

I just did a new fresh pull of the code and compiled from scratch 

[... truncated for brevity ...]

---

## Issue #N/A: Rockchip RK3588 perf

**Link**: https://github.com/ggml-org/llama.cpp/issues/722
**State**: closed
**Created**: 2023-04-02T20:39:28+00:00
**Closed**: 2023-04-02T22:14:36+00:00
**Comments**: 103

### Description

Just did a very simple run with llama-7b-4bit. It... took a while. Had it run in a screen. But, it worked!

```
root@FriendlyWrt /s/o/llama.cpp (master)# time ./main --color -m models/ggml-model-q4_0.bin -p "Hello there!"
main: seed = 1680443840
llama_model_load: loading model from 'models/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 4096
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 32
llama_model_load: n_layer = 32
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 11008
llama_model_load: n_parts = 1
llama_model_load: type    = 1
llama_model_load: ggml map size = 4017.70 MB
llama_model_load: ggml ctx size =  81.25 KB
llama_model_load: mem required  = 5809.78 MB (+ 1026.00 MB per state)
llama_model_load: loading tensors from 'models/ggml-model-q4_0.bin'
llama_model_load: model size =  4017.27 MB / num tensors = 291
llama_ini

[... truncated for brevity ...]

---

## Issue #N/A: Question: Why prompt is being run trough the network before generating new tokens?

**Link**: https://github.com/ggml-org/llama.cpp/issues/719
**State**: closed
**Created**: 2023-04-02T19:48:07+00:00
**Closed**: 2023-04-03T14:43:18+00:00
**Comments**: 31

### Description

As I understand, the NN doesn't have a state, so you should be able to put whatever tokens into context and start generating new tokens immediately. But right now, it first runs the NN on the prompt it seems? So with a long prompt, it takes some time until it starts to generate new tokens.

I though I was missing something, but huggingface `transformers` starts to generate the tokens immediately.

---

## Issue #N/A: Code showing when running.

**Link**: https://github.com/ggml-org/llama.cpp/issues/717
**State**: closed
**Created**: 2023-04-02T17:51:23+00:00
**Closed**: 2023-04-02T19:49:52+00:00
**Comments**: 4

### Description

When I start chat.exe with a alpaca bin I get.
main: seed = 1680456908
llama_model_load: loading model from 'models/llama-7B/ggml-model.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 4096
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 32
llama_model_load: n_layer = 32
llama_model_load: n_rot   = 128
llama_model_load: f16     = 3
llama_model_load: n_ff    = 11008
llama_model_load: n_parts = 1
llama_model_load: type    = 1
llama_model_load: ggml map size = 4820.95 MB
llama_model_load: ggml ctx size =  81.25 KB
llama_model_load: mem required  = 6613.03 MB (+ 1026.00 MB per state)
llama_model_load: loading tensors from 'models/llama-7B/ggml-model.bin'
llama_model_load: model size =  4820.52 MB / num tensors = 291
llama_init_from_file: kv self size  =  256.00 MB

system_info: n_threads = 4 / 8 | AVX = 1 | AVX2 = 0 | AVX512 = 0 | FMA = 0 | NEON = 0 | ARM_FMA = 0 | F16C = 0 | FP16_VA = 0 | 

[... truncated for brevity ...]

---

## Issue #N/A: Windows Quantize isn't working 

**Link**: https://github.com/ggml-org/llama.cpp/issues/715
**State**: closed
**Created**: 2023-04-02T15:12:02+00:00
**Closed**: 2023-04-02T15:35:43+00:00
**Comments**: 1

### Description

![image](https://user-images.githubusercontent.com/112736962/229361374-f3a72b6c-091b-4ddd-997d-95b5c97ef01f.png)

Im on win11 and barely know how to get this even remotely running and I am having a stroke of whats going wrong with this. 

---

## Issue #N/A: Add support to FMA3/FMA4 instructions 

**Link**: https://github.com/ggml-org/llama.cpp/issues/714
**State**: closed
**Created**: 2023-04-02T14:57:14+00:00
**Closed**: 2024-04-11T01:07:10+00:00
**Comments**: 2
**Labels**: stale

### Description

That improves dot products performance on Haswell+

---

## Issue #N/A: idk wth is happening help

**Link**: https://github.com/ggml-org/llama.cpp/issues/713
**State**: closed
**Created**: 2023-04-02T14:19:02+00:00
**Closed**: 2023-04-02T14:54:20+00:00
**Comments**: 8

### Description


PS C:\Users\Admin> cd D:\Software\GPT4ALL\llama.cpp
PS D:\Software\GPT4ALL\llama.cpp> make
process_begin: CreateProcess(NULL, uname -s, ...) failed.
Makefile:2: pipe: No error
process_begin: CreateProcess(NULL, uname -p, ...) failed.
Makefile:6: pipe: No error
process_begin: CreateProcess(NULL, uname -m, ...) failed.
Makefile:10: pipe: No error
'cc' is not recognized as an internal or external command,
operable program or batch file.
'head' is not recognized as an internal or external command,
operable program or batch file.
I llama.cpp build info:
I UNAME_S:
I UNAME_P:
I UNAME_M:
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -Wno-unused-function -mfma -mf16c -mavx -mavx2
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function
I LDFLAGS:
I CC:
I CXX:

cc  -I.              -O3 -DNDEBUG -std=

[... truncated for brevity ...]

---

## Issue #N/A: Running llama.cpp on android just prints out the question

**Link**: https://github.com/ggml-org/llama.cpp/issues/712
**State**: closed
**Created**: 2023-04-02T14:16:00+00:00
**Closed**: 2024-04-11T01:07:11+00:00
**Comments**: 2
**Labels**: android, stale

### Description

I ran llama.cpp on my android phone which has 8 threads and 8GB of ram in which around 7.16 GB is available, that is more than enough to run the 7B Alpaca model on it. But when i run it, it just repeats the question that i provided to it. I am using the `./examples/chat.sh` file. Why does it do that? How do i solve it?

---

## Issue #N/A: [Question] Can I load a the huggingface llama model aswell?

**Link**: https://github.com/ggml-org/llama.cpp/issues/708
**State**: closed
**Created**: 2023-04-02T11:03:06+00:00
**Closed**: 2023-04-03T17:50:18+00:00
**Comments**: 5
**Labels**: model

### Description

I have downloaded the llama-model from [here](https://huggingface.co/decapoda-research/llama-7b-hf). There it got converted to be compatible with pytorch. But the biggest advantage is that it is actually available. The magnet link from that [PR](https://github.com/facebookresearch/llama/pull/73) has no trackers so it's not starting to download at least for me. And the IPFS files always have a different checksum when I download them. So since I only have the huggingface-version is it possible use their model with llama.cpp somehow? 

---

## Issue #N/A: Convert pytorch-based models to work with llama.cpp?

**Link**: https://github.com/ggml-org/llama.cpp/issues/707
**State**: closed
**Created**: 2023-04-02T10:53:58+00:00
**Closed**: 2024-04-11T01:07:12+00:00
**Comments**: 11
**Labels**: stale

### Description

Out of curiosity, I want to see if I can launch a very mini AI on my little network server. It usually has around 3GB of free memory, and it'd be nice to chat with it sometimes. For that, I'd like to try a smaller model like Pythia.

So I would like to know:
- Can I convert `pytorch_model*.bin` to ggjm?
- Can I quantize those models to use even less memory as a sort of post-processing step?

I looked at the existing `convert_*.py` scripts, but none of those seemed to be for this type of model.

Thanks in advance!

---

## Issue #N/A: Windows page fault disk i/o slow on first load

**Link**: https://github.com/ggml-org/llama.cpp/issues/705
**State**: closed
**Created**: 2023-04-02T10:04:24+00:00
**Closed**: 2024-04-11T01:07:14+00:00
**Comments**: 37
**Labels**: performance, windows, stale

### Description

Hello,

As of https://github.com/ggerganov/llama.cpp/pull/613 I have experienced significant regression in model loading speed (I'm on windows, compiled msvc llama.cpp, llama.cpp is located on HDD to prevent SSD wear in my case)

It takes roughly 15 minutes for model to load first time after each computer restart/hibernation, during this time my HDD usage is at 100% and my non-llama.cpp read/write operations are slowed down on my pc
![hdd](https://user-images.githubusercontent.com/76458234/229345728-b597023b-f7e3-4a8b-b550-3159863ba03d.png)

Before that, previous commits took 60 - 180 seconds at worst to load model first time, and after first loading occured, model loaded within 5 - 10 seconds on each program restart until pc reboot/hibernation

Before Commit:
![timings2](https://user-images.githubusercontent.com/76458234/229347345-2053d645-0f26-42ef-9f8e-5fc69ad04e1c.png)

After:
![timings1](https://user-images.githubusercontent.com/76458234/229345966-ee606c92-e7cb-42f6-8

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

## Issue #N/A: WIndows build fails with -DBUILD_SHARED_LIBS=ON 

**Link**: https://github.com/ggml-org/llama.cpp/issues/699
**State**: closed
**Created**: 2023-04-02T05:05:28+00:00
**Closed**: 2024-04-11T01:07:15+00:00
**Comments**: 1
**Labels**: bug, build, stale

### Description

If you build on windows using -DBUILD_SHARED_LIBS=ON, it fails with linker errors.

```
common.obj : error LNK2019: unresolved external symbol ggml_mlock_supported referenced in function "void __cdecl gpt_print_usage(int,char * *,struct gpt_params const &)" (?gpt_print_usage@@YAXHPEAPEADAEB
Ugpt_params@@@Z)
quantize.obj : error LNK2019: unresolved external symbol ggml_time_init referenced in function main
quantize.obj : error LNK2019: unresolved external symbol ggml_init referenced in function main
```

I don't have enough knowledge in makefiles to fix the issue correctly but a workaround solved it for me.

1. Open the generated solution in visual studio
2. Add ggml in references of the failing projects.
3. Right click ggml under References-> Properties -> Set Link Library Dependencies to True

![image](https://user-images.githubusercontent.com/7353840/229332398-361f32c0-72d0-42b4-82fb-cde7e844b35d.png)
![image](https://user-images.githubusercontent.com/7353840/22933241

[... truncated for brevity ...]

---

## Issue #N/A: Support For ggml format for gpt4all

**Link**: https://github.com/ggml-org/llama.cpp/issues/696
**State**: closed
**Created**: 2023-04-02T01:24:38+00:00
**Closed**: 2023-04-02T10:50:26+00:00
**Comments**: 3

### Description

When I convert Llama model with convert-pth-to-ggml.py, quantize to 4bit, and load it with gpt4all, I get this:
llama_model_load: invalid model file 'ggml-model-q4_0.bin' (bad magic)
Could you implement to support ggml format that gpt4all uses?
Thanks!

---

## Issue #N/A: Regression: "The first main on the moon was "

**Link**: https://github.com/ggml-org/llama.cpp/issues/693
**State**: closed
**Created**: 2023-04-01T23:16:25+00:00
**Closed**: 2023-05-16T19:10:10+00:00
**Comments**: 14
**Labels**: question

### Description

I saw a blog post where that prompt was used and now when I try it myself using LlAMA I don't get the same result. It is quite strange.

It keeps telling me the man is 38 years old and then starts going off on a tangent. Could this be a recent regression @ggerganov ?

---

## Issue #N/A: How to make llama.cpp return control to add additional context?

**Link**: https://github.com/ggml-org/llama.cpp/issues/692
**State**: closed
**Created**: 2023-04-01T22:20:36+00:00
**Closed**: 2024-04-11T01:07:16+00:00
**Comments**: 2
**Labels**: enhancement, generation quality, stale

### Description

I want to be able to tell the model that if it can't reply something useful to return control so I can give more information.

Similarly, how do I add more context so that it can reason about a full conversation or say a specific set of documents?

For example, I ask it something and it should say I don't know can you provide me more information? And then I give it a document. Then I can add another document to the prompt, so it can understand from that and so on.

I've heard this is some sort of chaining, but I don't understand.

---

## Issue #N/A: How to use .safetensors model ?

**Link**: https://github.com/ggml-org/llama.cpp/issues/688
**State**: closed
**Created**: 2023-04-01T19:13:02+00:00
**Closed**: 2023-04-14T13:12:58+00:00
**Comments**: 5

### Description

I downloaded a model `alpaca-30b-lora-int4` from <https://huggingface.co/elinas/alpaca-30b-lora-int4/tree/main>
The model is a `.safetensors` in GPTQ format I think
I need to convert it to `GGML .bin` so I used the script provided in `llama.cpp` with the command `python convert-gptq-to-ggml.py models/30B/alpaca-30b-4bit.safetensors models/30B//tokenizer.model models/30B/alpaca-30b-4bit.bin`
But I get the following error
```python
Traceback (most recent call last):
  File "/big/meyer/expe/llama.cpp/convert-gptq-to-ggml.py", line 21, in <module>
    model = torch.load(fname_model, map_location="cpu")
  File "/big/meyer/expe/llama.cpp/.venv/lib/python3.10/site-packages/torch/serialization.py", line 815, in load
    return _legacy_load(opened_file, map_location, pickle_module, **pickle_load_args)
  File "/big/meyer/expe/llama.cpp/.venv/lib/python3.10/site-packages/torch/serialization.py", line 1035, in _legacy_load
    raise RuntimeError("Invalid magic number; corrupt file?")
R

[... truncated for brevity ...]

---

## Issue #N/A: Setting `temp=0` does not work as expected

**Link**: https://github.com/ggml-org/llama.cpp/issues/684
**State**: closed
**Created**: 2023-04-01T15:40:11+00:00
**Closed**: 2023-04-03T00:19:06+00:00
**Comments**: 3

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Setting sampling temperature to `0` should produce valid and "predictable" tokens.

# Current Behavior

Setting temperature to `0` causes sampling to fail completely. This is due to `plogits` being scaled by `1.0f/temp` before sampling [here](https://github.com/gg

[... truncated for brevity ...]

---

## Issue #N/A: [User] chat-with-bob.txt mentions incorrect city

**Link**: https://github.com/ggml-org/llama.cpp/issues/683
**State**: closed
**Created**: 2023-04-01T15:19:42+00:00
**Closed**: 2023-05-03T18:45:35+00:00
**Comments**: 4

### Description

prompts/chat-with-bob.txt mentions that Moscow is the biggest city in Europe, while it is actually Istanbul :)

---

## Issue #N/A: q4_1/f16 model is slow

**Link**: https://github.com/ggml-org/llama.cpp/issues/681
**State**: closed
**Created**: 2023-04-01T14:31:56+00:00
**Closed**: 2023-04-02T09:55:18+00:00
**Comments**: 8

### Description

pulled to the latest commit
another 7B model still runs as expected (which is gpt4all-lora-ggjt)
I have 16 gb of ram, the model file is about 9.5 gb
4 cores, amd, linux

# problem description:
model name: gpt4-x-alpaca-13b-ggml-q4_1-from-gptq-4bit-128g
the model was described as: LLaMA 13B, finetuned natively with alpaca dataset, then finetuned on GPT4 responses (GPT4-x), then GPTQ 4b-128g quantized, then converted to ggml q4_1 format
it loads, but takes about 30 seconds per token

```
$./main -m models/13B/ggml-model-q4_1.bin -n 128 --repeat_penalty 1.0 --color -ins
main: seed = 1680359110
llama_model_load: loading model from 'models/13B/ggml-model-q4_1.bin' - please wait ...
llama_model_load: GPTQ model detected - are you sure n_parts should be 2? we normally expect it to be 1
llama_model_load: use '--n_parts 1' if necessary
llama_model_load: n_vocab = 32001
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 5120
llama_model_load: n_mult  = 256
llama_mode

[... truncated for brevity ...]

---

## Issue #N/A: [WSL2] [Installation] make not compiling 

**Link**: https://github.com/ggml-org/llama.cpp/issues/679
**State**: closed
**Created**: 2023-04-01T13:00:34+00:00
**Closed**: 2023-04-02T04:38:19+00:00
**Comments**: 7
**Labels**: need more info

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

`make` should compile.

# Current Behavior

`make` the second step after cloning the repo printed the help rather than "making".

<details>
  <summary>help text for `make`</summary>

<pre><code>
  $ make <target...> [options]

  Options:
    --help       

[... truncated for brevity ...]

---

## Issue #N/A: Alpaca model is running very slow in llama.cpp compared to alpaca.cpp

**Link**: https://github.com/ggml-org/llama.cpp/issues/677
**State**: closed
**Created**: 2023-04-01T11:18:18+00:00
**Closed**: 2023-05-18T10:43:20+00:00
**Comments**: 3

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Current Behavior

Just yesterday, this migration script was added : `migrate-ggml-2023-03-30-pr613.py`. 
So, what I did on top of [@madmads11  instructions for using alpaca models](https://github.com/ggerganov/llama.cpp/issues/382#issuecomment-1479091459) was to use this above script a

[... truncated for brevity ...]

---

## Issue #N/A: [Feature Suggestion] Halide-lang codegen / generators

**Link**: https://github.com/ggml-org/llama.cpp/issues/676
**State**: closed
**Created**: 2023-04-01T11:18:04+00:00
**Closed**: 2024-04-11T01:07:17+00:00
**Comments**: 2
**Labels**: stale

### Description

# Prerequisites
Examples of work with neural networks in Halide:
https://github.com/halide/Halide/blob/main/apps/resnet_50/Resnet50Generator.cpp
https://github.com/halide/Halide/tree/main/apps/hannk
https://github.com/halide/Halide/blob/main/apps/onnx/model.cpp


# Expected Behavior

Cross platform binary and code generation with best scheduling on computational graph applied by Halide or Halide autoschedulers, reduced memory usage by scheduling every used network model before execution

# Current Behavior

Multiple conditional defines for different platforms and instruction bloated code, lack of GPU support i.e. OpenCL, OpenGL Compute, CUDA, current computational graph has great parallelism but very frustrating locality



---

## Issue #N/A: solved.

**Link**: https://github.com/ggml-org/llama.cpp/issues/675
**State**: closed
**Created**: 2023-04-01T11:16:08+00:00
**Closed**: 2023-04-01T11:19:23+00:00
**Comments**: 1

### Description

No description provided.

---

## Issue #N/A: [Feature Suggestion] Dynamic prompt

**Link**: https://github.com/ggml-org/llama.cpp/issues/673
**State**: closed
**Created**: 2023-04-01T08:05:43+00:00
**Closed**: 2024-04-11T01:07:19+00:00
**Comments**: 1
**Labels**: stale

### Description

Would love to see a feature where both the AI and the user could change the initial prompt in-situ and when necessary.

Essentially, this would be the same as changing the prompt without exiting llama.cpp, thus eliminates the need to reload the model weights and forgetting the context.

To trigger this, it could be a trigger word in the input, such as \iNewPrompt: You are an insane AI assistant. You always gives imprecise answers and easily goes into panic mode. Once you are panicked, you will start babbling and answer everything hysterically. You will become sane again when I tell you to stop panic.

---

## Issue #N/A: magic number in convert-gptq-to-ggml.py not consistent

**Link**: https://github.com/ggml-org/llama.cpp/issues/672
**State**: closed
**Created**: 2023-04-01T07:41:45+00:00
**Closed**: 2023-04-02T15:51:08+00:00
**Comments**: 1
**Labels**: duplicate

### Description

It appears that the conver-gptq-to-ggml script needs an update to reflect the recent change in magic, see [this line](https://github.com/ggerganov/llama.cpp/blob/master/convert-gptq-to-ggml.py#L39). However, it's not completely clear to me if only updating the magic number is sufficient to ensure that the resulting file is compatible. Hence leaving it here a reminder :)

---

## Issue #N/A: Having doubt on a if (0) block

**Link**: https://github.com/ggml-org/llama.cpp/issues/670
**State**: closed
**Created**: 2023-04-01T05:10:17+00:00
**Closed**: 2023-04-12T15:33:12+00:00
**Comments**: 2

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Context 

I was just checking the repo for the first time and i saw this block of code in the master branch. [Here](https://github.com/ggerganov/llama.cpp/blob/master/llama.cpp#L686-L689) is the exact location of the code from master. And [here](https://github.com/ggerganov/llama.cpp/bl

[... truncated for brevity ...]

---

## Issue #N/A: llama.cpp main hangs at prompt with latest mmap updates

**Link**: https://github.com/ggml-org/llama.cpp/issues/669
**State**: closed
**Created**: 2023-04-01T04:10:19+00:00
**Closed**: 2023-04-01T08:54:17+00:00
**Comments**: 2

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x ] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior
After upgrading to latest code compiling and then running an inference using main the following prompt should return results like before:

```
./main -m models/13B/ggml-model-q4_0.bin -n 512 --repeat_penalty 1.0 --color  -p "What is controlled delivery?"
main: see

[... truncated for brevity ...]

---

## Issue #N/A: Text Generation Effects in Different Models

**Link**: https://github.com/ggml-org/llama.cpp/issues/668
**State**: closed
**Created**: 2023-04-01T03:26:38+00:00
**Closed**: 2024-04-11T01:07:20+00:00
**Comments**: 1
**Labels**: stale

### Description

# Prerequisites
I changed the model from 7B to 30B, running following commands:
```
docker run -i --rm -v ./models:/models ghcr.io/ggerganov/llama.cpp:full --run -m /models/30B/ggml-model-q4_0.bin -n 512 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt
```

# Current Behavior
the following content was what I chat with it:
```
main: seed = 1680317960
llama_model_load: loading model from '/models/30B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 6656
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 52
llama_model_load: n_layer = 60
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 17920
llama_model_load: n_parts = 4
llama_model_load: type    = 3
llama_model_load: ggml ctx size = 20171.50 MB
llama_model_load: mem required  = 22475.50 MB (+ 3124.00 MB per state)
llama_model_load: loading model part 1/4 from 

[... truncated for brevity ...]

---

## Issue #N/A: [User] examples/chat-13B.sh sometimes continues my question instead of answering

**Link**: https://github.com/ggml-org/llama.cpp/issues/667
**State**: closed
**Created**: 2023-04-01T02:42:42+00:00
**Closed**: 2024-04-11T01:07:21+00:00
**Comments**: 2
**Labels**: stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Steps to Reproduce

Run the chat program:

```
$ examples/chat-13B.sh
main: seed = 1680315413
llama_model_load: loading model from './models/13B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 2048
llama_model_load: n_embd  = 

[... truncated for brevity ...]

---

## Issue #N/A: [User] Bus error (core dumped) on a 65B model

**Link**: https://github.com/ggml-org/llama.cpp/issues/666
**State**: closed
**Created**: 2023-03-31T23:58:31+00:00
**Closed**: 2023-07-28T19:44:35+00:00
**Comments**: 2

### Description

I tried running the a 65B model that was converted using the unversioned `.py` conversion script then migrated from an 8-file `ggml` `.bin` to a single-file `ggjt` `.bin`. Tried to run the model and I get a `Bus error` then the program ends.
```
user@ubuntu: ~/Desktop/llama.cpp$ ./main -m ./models/ggjt-model-model-q4_0.bin -t 7 -i
main: seed = 1680306291
llama_model_load: loading model from './models/ggjt-model-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 8192
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 64
llama_model_load: n_layer = 80
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 22016
llama_model_load: n_parts = 8
llama_model_load: type    = 4
llama_model_load: ggml map size = 38917.99 MB
llama_model_load: ggml ctx size = 201.25 KB
llama_model_load: mem required  = 41478.18 MB (+ 5120.00 MB per state)
llama_model_load: load

[... truncated for brevity ...]

---

## Issue #N/A: GPT4All: invalid model file (bad magic)

**Link**: https://github.com/ggml-org/llama.cpp/issues/662
**State**: closed
**Created**: 2023-03-31T22:19:34+00:00
**Closed**: 2023-03-31T23:25:37+00:00
**Comments**: 10

### Description

Hi there, followed the instructions to get gpt4all running with llama.cpp, but was somehow unable to produce a valid model using the provided python conversion scripts:

```
% python3 convert-gpt4all-to-ggml.py models/gpt4all-7B/gpt4all-lora-quantized.bin ./models/tokenizer.model
converting models/gpt4all-7B/gpt4all-lora-quantized.bin
```
```
% ./main -m ./models/gpt4all-7B/gpt4all-lora-quantized.bin -n 128
main: seed = 1680294943
llama_model_load: loading model from './models/gpt4all-7B/gpt4all-lora-quantized.bin' - please wait ...
./models/gpt4all-7B/gpt4all-lora-quantized.bin: invalid model file (bad magic [got 0x67676d66 want 0x67676a74])
	you most likely need to regenerate your ggml files
	the benefit is you'll get 10-100x faster load times
	see https://github.com/ggerganov/llama.cpp/issues/91
	use convert-pth-to-ggml.py to regenerate from original pth
	use migrate-ggml-2023-03-30-pr613.py if you deleted originals
llama_init_from_file: failed to load model
main: e

[... truncated for brevity ...]

---

## Issue #N/A: Variable density context windows?

**Link**: https://github.com/ggml-org/llama.cpp/issues/660
**State**: closed
**Created**: 2023-03-31T20:39:41+00:00
**Closed**: 2024-04-11T01:07:23+00:00
**Comments**: 3
**Labels**: stale

### Description

I am currently off my meds, but I would like to propose an idea to enhance the context handling capabilities of the LLaMA/Alpaca/gpt4all models, particularly for the smaller models with limited context window sizes. I'm not sure if this is entirely doable within the current architecture, or if changes would be needed to the underlying LLMs, but I wanted to share my thoughts and get your feedback.

**The Problem:**

As you all know, smaller models have limited context window sizes, which make it hard to maintain long conversations especially when the LLM is used as a chatbot instead vs unrelated queries. This limitation affects the model's overall performance and ability to provide accurate and coherent responses.

**The Proposal:**

I propose implementing a Variable Density Context Window (VDCW) technique that selectively retains the most relevant tokens while still staying within the model's limited context window size. This approach aims to provide the model with a more exten

[... truncated for brevity ...]

---

## Issue #N/A: Im getting this error while trying to convert .pth to .bin

**Link**: https://github.com/ggml-org/llama.cpp/issues/659
**State**: closed
**Created**: 2023-03-31T20:30:53+00:00
**Closed**: 2023-04-16T09:29:11+00:00
**Comments**: 1

### Description

**Error that im getting**

`shreyas@ATLAS-AHQVOLJ9A:/mnt/c/Users/Shreyas-ITB/Downloads/llama.cpp$ python3 convert-pth-to-ggml.py models/7B/ 1
{'dim': 4096, 'multiple_of': 256, 'n_heads': 32, 'n_layers': 32, 'norm_eps': 1e-06, 'vocab_size': -1}
Namespace(dir_model='models/7B/', ftype=1, vocab_only=0)
n_parts = 1

Processing part 1 of 1

Traceback (most recent call last):
  File "/mnt/c/Users/Shreyas-ITB/Downloads/llama.cpp/convert-pth-to-ggml.py", line 274, in <module>
    main()
  File "/mnt/c/Users/Shreyas-ITB/Downloads/llama.cpp/convert-pth-to-ggml.py", line 267, in main
    model = torch.load(fname_model, map_location="cpu")
  File "/usr/local/lib/python3.10/dist-packages/torch/serialization.py", line 809, in load
    return _load(opened_zipfile, map_location, pickle_module, **pickle_load_args)
  File "/usr/local/lib/python3.10/dist-packages/torch/serialization.py", line 1172, in _load
    result = unpickler.load()
  File "/usr/local/lib/python3.10/dist-packages/to

[... truncated for brevity ...]

---

## Issue #N/A: Error: Invalid model file when using converted GPT4ALL model after following provided instructions

**Link**: https://github.com/ggml-org/llama.cpp/issues/655
**State**: closed
**Created**: 2023-03-31T17:13:52+00:00
**Closed**: 2023-03-31T17:55:16+00:00
**Comments**: 11

### Description

Hello,

I have followed the instructions provided for using the GPT-4ALL model. I used the `convert-gpt4all-to-ggml.py` script to convert the `gpt4all-lora-quantized.bin` model, as instructed. However, I encountered an error related to an invalid model file when running the example. 

Here are the steps I followed, as described in the instructions:

1. Convert the model using the `convert-gpt4all-to-ggml.py` script:
```
python3 convert-gpt4all-to-ggml.py models/gpt4all/gpt4all-lora-quantized.bin ./models/tokenizer.model
```

2. Run the `interactive mode` example with the newly generated `gpt4all-lora-quantized.bin` model:
```
./main -m ./models/gpt4all/gpt4all-lora-quantized.bin -n 256 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt
```

However, I encountered the following error:
```
./models/gpt4all/gpt4all-lora-quantized.bin: invalid model file (bad magic [got 0x67676d66 want 0x67676a74])
you most likely need to regenerate your ggml files


[... truncated for brevity ...]

---

## Issue #N/A: How do i download the models? 

**Link**: https://github.com/ggml-org/llama.cpp/issues/650
**State**: closed
**Created**: 2023-03-31T11:55:15+00:00
**Closed**: 2023-03-31T13:40:17+00:00
**Comments**: 1
**Labels**: good first issue, invalid, wontfix

### Description

`65B 30B 13B 7B tokenizer_checklist.chk tokenizer.model`

This command in the readme.md file says to add the models into the models directory but the models arent even there in the directory.
Please let me know how to download the 7B model to run on my computer.
Thanks

---

## Issue #N/A: [User] interactive mode does not show my text

**Link**: https://github.com/ggml-org/llama.cpp/issues/649
**State**: closed
**Created**: 2023-03-31T08:46:03+00:00
**Closed**: 2023-07-28T19:44:18+00:00
**Comments**: 3

### Description

# Prerequisites
Built on windows 10 , Visual studio 2022.
After Loading 7B model I have the prompt ready but if I write, nothing is displayed and can not trigger it.

My log:
main.exe -m ./models/7B/ggml-model-q4_0.bin -i -n 124 -t 4
main: seed = 1680252016
llama_model_load: loading model from './models/7B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 4096
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 32
llama_model_load: n_layer = 32
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 11008
llama_model_load: n_parts = 1
llama_model_load: type    = 1
llama_model_load: ggml ctx size = 4273.34 MB
llama_model_load: mem required  = 6065.34 MB (+ 1026.00 MB per state)
llama_model_load: loading model part 1/1 from './models/7B/ggml-model-q4_0.bin'
llama_model_load: .................................... done
llama_model_load: model si

[... truncated for brevity ...]

---

## Issue #N/A: gpt4all keeps on ending responses abruptly

**Link**: https://github.com/ggml-org/llama.cpp/issues/648
**State**: closed
**Created**: 2023-03-31T07:58:57+00:00
**Closed**: 2023-04-02T08:24:47+00:00
**Comments**: 5

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

The GPT4All model should give proper and complete responses

# Current Behavior

Longer responses get truncated:
```
> Please write a letter to my boss explaining that I keep on arriving late at work because my alarm clock is defective.
Dear Boss,

I am sorry

[... truncated for brevity ...]

---

## Issue #N/A: Confusion about the model versioning

**Link**: https://github.com/ggml-org/llama.cpp/issues/647
**State**: closed
**Created**: 2023-03-31T07:20:01+00:00
**Closed**: 2023-05-03T18:46:36+00:00
**Comments**: 21
**Labels**: documentation, enhancement

### Description

So back when project started, we had the first "unversioned" model format without the embedded tokens, with the magic 0x67676d6c (ggml).

Problem with that was that it didn't have any versioning support, so newer/older versions would just think "I don't know what this is, this is not a model file".

Then on this commit https://github.com/ggerganov/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4, adding the embedded the tokens we got a new versioned model format, with magic 0x67676d66 (ggmf), along with **versioning**, so it could now say "this is definitely a model file, but a wrong version" as shown here:
https://github.com/ggerganov/llama.cpp/blob/3bcc129ba881c99795e850b0a23707a4dfdabe9d/llama.h#L22

That was definitely a good move towards future proofing. Any breaking changes could just add +1 to that version and all would be fine and dandy for the next 4294967295 versions of the model format.

But then came this commit: https://github.com/ggerganov/llama.cpp/comm

[... truncated for brevity ...]

---

## Issue #N/A: Unable to enter Chinese prompt

**Link**: https://github.com/ggml-org/llama.cpp/issues/646
**State**: closed
**Created**: 2023-03-31T06:43:06+00:00
**Closed**: 2023-04-09T08:03:44+00:00
**Comments**: 8
**Labels**: windows

### Description

Hi!My use is compiled under Windows main.exe, when I type Chinese Prompt, I found that the model seems to be unable to understand, under debugging found that std::getline(std::cin,line) get is empty lines, then I tried Japanese, are the same result.
(Since I am a native Chinese speaker, this question was translated by DeepL)
![image](https://user-images.githubusercontent.com/18028414/229043234-a47c0569-07e1-4731-85d9-121f9774fdc9.png)


---

## Issue #N/A: Create clear instructions for downloading and converting the models

**Link**: https://github.com/ggml-org/llama.cpp/issues/644
**State**: closed
**Created**: 2023-03-31T02:23:32+00:00
**Closed**: 2023-05-03T18:43:16+00:00
**Comments**: 2
**Labels**: documentation, enhancement

### Description

Clear instructions are needed to allow new arrivals to download and convert the models, in spite of the multiple format versions (non quantised, quantised, various llama versions etc) .

I would suggest that each llama or alpaca etc print a version on startup, and that the conversions scripts have this in their name, and also that a program reading a file and figuring out what it is from a magic print the version read and the version expected even if it aborts.  

Edmund



---

## Issue #N/A: The `quantize.py` script is not needed anymore. Just fetch the latest code and do this as a quantization step:

**Link**: https://github.com/ggml-org/llama.cpp/issues/641
**State**: closed
**Created**: 2023-03-31T00:57:24+00:00
**Closed**: 2023-03-31T08:18:49+00:00
**Comments**: 4

### Description

              The `quantize.py` script is not needed anymore. Just fetch the latest code and do this as a quantization step:

```
./quantize ./models/7B/ggml-model-f16.bin ./models/7B/ggml-model-q4_0.bin 2
```

_Originally posted by @prusnak in https://github.com/ggerganov/llama.cpp/issues/621#issuecomment-1491088418_

I tried this method in Colab, but it still reports an error:
/bin/bash: ./quantize: No such file or directory            

---

## Issue #N/A: the new mmap method does not work on Windows 11 ?

**Link**: https://github.com/ggml-org/llama.cpp/issues/639
**State**: closed
**Created**: 2023-03-30T22:26:30+00:00
**Closed**: 2023-03-31T01:32:34+00:00
**Comments**: 19
**Labels**: need more info

### Description

I tried migration and to create the new weights from pth, in both cases the mmap fails.
Always says "failed to mmap"

---

## Issue #N/A: Performance investigation using AMD BLIS instead of OpenBLAS on 16 core AMD Zen1

**Link**: https://github.com/ggml-org/llama.cpp/issues/637
**State**: closed
**Created**: 2023-03-30T22:14:53+00:00
**Closed**: 2023-04-13T08:09:16+00:00
**Comments**: 30
**Labels**: enhancement, performance

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior
Compiling against AMD optimized BLS implementation of BLAS allows me to run perplexity tests

# Current Behavior
Compiling against AMD optimized BLS implementation of BLAS causes perplexity command to process 0 chunks

* Physical (or virtual) hardware you are using

[... truncated for brevity ...]

---

## Issue #N/A: Compiling with LLAMA_OPENBLAS=1 fails on Arch Linux

**Link**: https://github.com/ggml-org/llama.cpp/issues/634
**State**: closed
**Created**: 2023-03-30T19:41:07+00:00
**Closed**: 2023-04-12T15:26:57+00:00
**Comments**: 8

### Description

The problem is missing `cblas` symbols:

```plaintext
g++ -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -pthread examples/main/main.cpp ggml.o llama.o common.o -o main -lopenblas
/usr/bin/ld: ggml.o: in function `ggml_compute_forward_mul_mat_f16_f32':
ggml.c:(.text+0x454b): undefined reference to `cblas_sgemm'
/usr/bin/ld: ggml.o: in function `ggml_compute_forward':
ggml.c:(.text+0xcc1b): undefined reference to `cblas_sgemm'
/usr/bin/ld: ggml.c:(.text+0xd42d): undefined reference to `cblas_sgemm'
collect2: error: ld returned 1 exit status
make: *** [Makefile:241: main] Error 1
```

Installing `cblas` and adding `-lcblas` to the link flags fixes the issue.

This is on Arch Linux. I'll also submit the trivial PR, but I don't know if there are different versions/systems where this would cause issues.

---

## Issue #N/A: ~2x perf improvement on Apple Silicon by changing state_shared.has_work access from atomic to mutex/conditional

**Link**: https://github.com/ggml-org/llama.cpp/issues/633
**State**: closed
**Created**: 2023-03-30T19:18:14+00:00
**Closed**: 2024-04-12T01:07:15+00:00
**Comments**: 5
**Labels**: enhancement, performance, stale

### Description

### Discussed in https://github.com/ggerganov/llama.cpp/discussions/616

<div type='discussions-op-text'>

<sup>Originally posted by **izard** March 30, 2023</sup>
I profiled on a latest Mac Book Pro machine and found that significantly more time is spent in atomic checks for `state_shared.has_work` in while loops than doing actual work in matrix multiply.
So I changed busy waits like: 
```
pthread_mutex_lock(&state->shared->mutex);
   while (state->shared->has_work) {
     pthread_cond_wait(&state->shared->cond, &state->shared->mutex);
// unlock
```

and setting `has_work` to 
```
pthread_mutex_lock(&state_shared.mutex);
state_shared.has_work = true;
pthread_cond_broadcast(&state_shared.cond);
pthread_mutex_unlock(&state_shared.mutex);

```
Got a nice 2x speedup in time/token.

I can't post a patch/pull request because everything I do in spare time still belongs to my employer, but the change is trivial as described above. Probably won't provide much benefit (i

[... truncated for brevity ...]

---

## Issue #N/A: Combine large LLM with small LLM for faster inference

**Link**: https://github.com/ggml-org/llama.cpp/issues/630
**State**: closed
**Created**: 2023-03-30T17:54:01+00:00
**Closed**: 2024-04-12T01:07:17+00:00
**Comments**: 44
**Labels**: question, research 🔬, stale

### Description

So I was thinking about the following idea.
It is probably completely bogus, but I would definitely investigate it when and if I had the time to, so maybe someone else would be interested as well.

---

Large LLM takes a lot of time to perform token inference. Lets say it takes 500ms per token.

A small LLM (or some other approach) can infer a token very fast. Lets say < 5ms.

Lets assume that the small LLM is correct 80-90% of the time.

The idea is the following:

- Before I run the large LLM inference for the next token, I infer it using the small LLM
- I now want to somehow partially evaluate the large LLM (let's say the first 10% of the layers) and get an approximate estimate for the next token
- If this estimate indicates a high probability for that token (i.e. above some threshold) - we stop and directly say that this is the new token. At this point we would have consumed (5ms for the small LLM + ~50ms for the large LLM)
- Otherwise, we proceed to evaluate the re

[... truncated for brevity ...]

---

## Issue #N/A: How to activate BLAS?

**Link**: https://github.com/ggml-org/llama.cpp/issues/627
**State**: closed
**Created**: 2023-03-30T16:56:25+00:00
**Closed**: 2024-05-10T01:28:42+00:00
**Comments**: 19
**Labels**: need more info, stale

### Description

Hello,

I've heard that I could get BLAS activated through my intel i7 10700k by installing this [library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html).

Unfortunatly, nothing happened, after compiling again with Clung I still have no BLAS in llama.cpp

```
system_info: n_threads = 14 / 16 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | VSX = 0 |
```

Maybe it's just not possible I don't know, I need someone to tell me the truth 😅 

---

## Issue #N/A: How to quantize a fine-tuned llama model?

**Link**: https://github.com/ggml-org/llama.cpp/issues/624
**State**: closed
**Created**: 2023-03-30T14:10:48+00:00
**Closed**: 2023-07-28T19:21:40+00:00
**Comments**: 2
**Labels**: enhancement, good first issue, model

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

I have fine-tuned llama 7B model that I want to quantize to 4-bit and run using llama.cpp
I got the weight in 3 parts .bin files and converted them to the same llama model naming scheme, i.e. "consolidated.XX"

i.e.
``pytorch_model-00001-of-00003.bin`` to ``consol

[... truncated for brevity ...]

---

## Issue #N/A: How can I do summarization 

**Link**: https://github.com/ggml-org/llama.cpp/issues/623
**State**: closed
**Created**: 2023-03-30T13:19:01+00:00
**Closed**: 2023-03-30T17:08:08+00:00
**Comments**: 1
**Labels**: model

### Description

I'm trying to make something like https://platform.openai.com/examples/default-notes-summary

But it fails, I tried with gpt4all, llama and alpaca 7B. Maybe I should ajust the prompt ?

<img width="1440" alt="Screenshot 2023-03-30 at 15 17 56" src="https://user-images.githubusercontent.com/74246611/228848634-1a27e9a6-fed6-4abc-8aa1-9964f8e2595f.png">


---

## Issue #N/A: Compilation failure on aarch64-linux in ggml SIMD code

**Link**: https://github.com/ggml-org/llama.cpp/issues/622
**State**: closed
**Created**: 2023-03-30T10:46:46+00:00
**Closed**: 2023-03-30T17:27:50+00:00
**Comments**: 4

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Should compile on aarch64-linux

# Current Behavior

```
[12:34:31] [  8%] Building C object CMakeFiles/ggml.dir/ggml.c.o
[12:34:31] /opt/bin/aarch64-linux-gnu-libgfortran5-cxx11/aarch64-linux-gnu-gcc --sysroot=/opt/aarch64-linux-gnu/aarch64-linux-gnu/sys-root/ 

[... truncated for brevity ...]

---

## Issue #N/A: error:python3 quantize.py 7B 

**Link**: https://github.com/ggml-org/llama.cpp/issues/621
**State**: closed
**Created**: 2023-03-30T09:44:35+00:00
**Closed**: 2023-03-30T23:22:05+00:00
**Comments**: 7
**Labels**: question

### Description

When I tried the llama model and run :**python3 quantize.py 7B**    for operation, ```
the "quantize" script was not found in the current location appeared
If you want to use it from another location, set the -- quantify script path argument from the command line
```
It's still this error. I have also made other attempts: `python3/Users/sunxiaotong/Desktop/llama/llama.cpp/quantize.py - q/ quantize.py 7B`
```

```
usage: python3 quantize.py [-h] [-r] [-m MODELS_PATH]
[-q QUANTIZE_SCRIPT_PATH]
{7B,13B,30B,65B} [{7B,13B,30B,65B} ...]
python3 quantize.py: error: argument models: invalid choice: '/Users/sunxiaotong/Desktop/llama/llama.cpp/models/7B/ggml-model-f16.bin' (choose from '7B', '13B', '30B', '65B')
```
May I ask where I was wrong? Can you give me some suggestions

---

## Issue #N/A: [build] ARMv8 build problem (OpenWrt)

**Link**: https://github.com/ggml-org/llama.cpp/issues/620
**State**: closed
**Created**: 2023-03-30T09:24:25+00:00
**Closed**: 2023-03-30T10:05:21+00:00
**Comments**: 3

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
  * `git clone $url; cd llama.cpp; make` 
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

I expected to build the basic llama.cpp `bin/main` program, to see if building even worked properly.

# Current Behavior

```
root@FriendlyWrt /s/o/llama.cpp (master)# make
I llama.cpp build info:
I UNAME_S:  Linux
I 

[... truncated for brevity ...]

---

## Issue #N/A: Making a "quantize-ggml_16bit-to-gptq.py" script?

**Link**: https://github.com/ggml-org/llama.cpp/issues/618
**State**: closed
**Created**: 2023-03-30T07:23:54+00:00
**Closed**: 2024-04-12T01:07:19+00:00
**Comments**: 3
**Labels**: enhancement, stale

### Description

Hello,

I know the [quantize.py](https://github.com/ggerganov/llama.cpp/blob/master/quantize.py) converts a ggml 16 bits into a ggml 4 bits RTN.
Do you think it's possible to create a script that converts a ggml 16 bits into a ggml 4bits GPTQ?

Referring to [this repository](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/pytorch), it appears that the current implementation of the quantization relies only on GPU, which demands a significant amount of VRAM and might not be suitable for the average user.

A new script, which we could call "quantize-ggml_16bit-to-gptq.py", could be designed to use only CPU and RAM resources, making it more accessible to the general public.

---

## Issue #N/A: Which tokenizer.model is needed for GPT4ALL? 

**Link**: https://github.com/ggml-org/llama.cpp/issues/614
**State**: closed
**Created**: 2023-03-30T00:44:00+00:00
**Closed**: 2023-03-31T02:59:00+00:00
**Comments**: 3
**Labels**: question

### Description

Which tokenizer.model is needed for GPT4ALL for use with convert-gpt4all-to-ggml.py? Is it the one for LLaMA 7B? It is unclear from the current README and gpt4all-lora-quantized.bin seems to be typically distributed without the tokenizer.model file.



---

## Issue #N/A: Reverting generated output/user input!

**Link**: https://github.com/ggml-org/llama.cpp/issues/604
**State**: closed
**Created**: 2023-03-29T18:51:38+00:00
**Closed**: 2024-04-12T01:07:21+00:00
**Comments**: 9
**Labels**: enhancement, stale

### Description

Hey!

This is a feature request for reverting input/output. One example usecase is to be able to retry generation if the response wasn't as desired.
One way of implementing this could be by adding the ability to create "snapshots" using signals(?).

Thanks a lot
niansa

---

## Issue #N/A: Performance Discrepancy: gpt4all Faster than Optimized llama.cpp

**Link**: https://github.com/ggml-org/llama.cpp/issues/603
**State**: closed
**Created**: 2023-03-29T18:46:33+00:00
**Closed**: 2023-04-12T15:30:22+00:00
**Comments**: 67
**Labels**: performance

### Description

**Expected Behavior**

I am comparing the performance of two executables: llama.cpp (current version) and the default gpt4all executable (which uses a previous version of llama.cpp). I am using the same language model for both executables, and I expect the current version of llama.cpp (which is built specifically for the hardware) to perform at least as fast as the default gpt4all executable.

**Current Behavior**

The default gpt4all executable, which uses a previous version of llama.cpp, performs significantly faster than the current version of llama.cpp. Despite building the current version of llama.cpp with hardware-specific compiler flags, it consistently performs significantly slower when using the same model as the default gpt4all executable.

**Environment and Context**

I am running the comparison on a Windows platform, using the default gpt4all executable and the current version of llama.cpp included in the gpt4all project. The version of llama.cpp is the latest ava

[... truncated for brevity ...]

---

## Issue #N/A: When running in PowerShell in windows, it works, but throws an error in interactive mode

**Link**: https://github.com/ggml-org/llama.cpp/issues/601
**State**: closed
**Created**: 2023-03-29T17:23:13+00:00
**Closed**: 2023-05-18T10:49:11+00:00
**Comments**: 2
**Labels**: bug, windows

### Description

I built llama.cpp using cmake and then Visual Studio (after many trials and tribulations since I'm pretty new to this), but finally got it working.

Using the 7B model the outputs are reasonable, but when I put the -i tag, it runs, then I hit Ctrl+C, it allows me to enter text, but when I hit enter an error pops up in a windows shown below:

![image](https://user-images.githubusercontent.com/65059714/228617990-0da94e0c-5df4-4311-9d41-0ed5c060df0f.png)

I'm running this on my windows machine, but I have been using WSL to get some stuff to work.

Here's an example of it failing:

`(base) PS G:\llama\llama.cpp> .\bin\Debug\main.exe -m ..\LLaMA\7B\ggml-model-q4_0.bin -i -n 124 -t 24`

`(base) PS G:\llama\llama.cpp> .\bin\Debug\main.exe -m ..\LLaMA\7B\ggml-model-q4_0.bin -i -n 124 -t 24
main: seed = 1680110536
llama_model_load: loading model from '..\LLaMA\7B\ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model

[... truncated for brevity ...]

---

## Issue #N/A: Support tensors with 64-bit number of elements in ggml

**Link**: https://github.com/ggml-org/llama.cpp/issues/599
**State**: closed
**Created**: 2023-03-29T16:15:55+00:00
**Closed**: 2023-06-16T06:53:12+00:00
**Comments**: 21
**Labels**: enhancement

### Description

# Expected Behavior

When setting '-c' to a large number, with sufficient RAM llama.cpp should run. I'm aware that it warns me that context sizes larger than 2048 might produce poor results, but the results are actually fine, if it's not crashing.

# Current Behavior

When setting '-c' to a large number, llama.cpp crashes with the error message
```
ggml_new_tensor_impl: not enough space in the context's memory pool (needed 32289959360, available 32279078144)
```
(repeated many times with slightly varying numbers, see below).

To be precise, I'm using the command
```
./main -m ./models/65B/ggml-model-q4_0.bin -t 16 -c 3300 -b 16 -n 2048 --keep 0 --temp 0.8 \
    --repeat_last_n 512 --repeat_penalty 1.1 --color \
    --ignore-eos
```
which crashes while the same command with '-c 3200' works.

If still have plenty of free RAM (~60 GB) so that shouldn't be an issue.

This might be related to #52 but it seem to die later in the process so probably it's something differ

[... truncated for brevity ...]

---

## Issue #N/A: Is this true? :joy: 

**Link**: https://github.com/ggml-org/llama.cpp/issues/596
**State**: closed
**Created**: 2023-03-29T13:30:18+00:00
**Closed**: 2023-04-06T15:20:27+00:00
**Comments**: 1
**Labels**: 🦙.

### Description

I asked ChatGPT about the difference between `llama.cpp` and `whisper.cpp` and it says:

![image](https://user-images.githubusercontent.com/3450257/228553783-4cf28da9-f025-4a7c-92a6-2c8c9c604c28.png)


---

## Issue #N/A: A few questions about the positional encodings

**Link**: https://github.com/ggml-org/llama.cpp/issues/594
**State**: closed
**Created**: 2023-03-29T11:28:19+00:00
**Closed**: 2023-03-30T01:12:31+00:00
**Comments**: 2

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Question

It is very kind of you to share such an amazing repo.  It works on my android device!
My question is that where is the Rotary Embedding implementation in the code? Or what work do you do to achieve this implicitly?  I found it in the [official code](https://github.com/faceboo

[... truncated for brevity ...]

---

## Issue #N/A: Why is the license not GPL 3.0 ?

**Link**: https://github.com/ggml-org/llama.cpp/issues/591
**State**: closed
**Created**: 2023-03-29T07:53:39+00:00
**Closed**: 2023-03-29T08:44:06+00:00
**Comments**: 1

### Description

It is my understand that the original code is under GPL 3.0
https://github.com/facebookresearch/llama

I'm not a legal expert but I thought this was a contaminating license.
Is porting considered like a whole new project and does not fall under this category? 

---

## Issue #N/A: Support gpt4all interactive mode

**Link**: https://github.com/ggml-org/llama.cpp/issues/590
**State**: closed
**Created**: 2023-03-29T06:10:08+00:00
**Closed**: 2023-03-29T06:21:05+00:00
**Comments**: 3
**Labels**: duplicate

### Description

Hey!

I just found this repo: https://github.com/nomic-ai/gpt4all and it looks amazing! They've forked alpaca.cpp but with their own "tweaks" in a single commit: https://github.com/zanussbaum/gpt4all.cpp/commit/4a6afcb08fb243df9a919c26aab1027ebfa373cc
So I assume it'd be quite easy to support!

Niansa

---

## Issue #N/A: .dot file of ggml_graph can not be generated to .png file

**Link**: https://github.com/ggml-org/llama.cpp/issues/589
**State**: closed
**Created**: 2023-03-29T05:57:35+00:00
**Closed**: 2023-03-29T06:38:43+00:00
**Comments**: 4

### Description

Hi, I want to generate a picture of the grapj. And I uncommented this 2 lines in "llama.cpp", so that to run the function `ggml_graph_dump_dot（）`
```
    //if (n_past%100 == 0) {
        ggml_graph_print   (&gf);
        ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}
```
And I got a file named `gpt-2.dot`
But when I run command in python:
```
from graphviz import Digraph
import sys
sys.setrecursionlimit(300000) 

import pydot
import os
(graph,) = pydot.graph_from_dot_file("D:\\PIQ\\llama.cpp\\build\\examples\\main\\gpt-2.dot")
graph.write_png("gpt-2.png")
```
I get the error message: `Expect '{' but got '['`
So I modifid the function `ggml_graph_dump_dot（）` in `ggml.c` like this:
```
void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename) {
    char color[16];

    FILE * fp = fopen(filename, "w");
    GGML_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    

[... truncated for brevity ...]

---

## Issue #N/A: Update the convert-unversioned-ggml-to-ggml.py script to support GPT4All ggml models

**Link**: https://github.com/ggml-org/llama.cpp/issues/588
**State**: closed
**Created**: 2023-03-29T05:21:04+00:00
**Closed**: 2023-03-29T16:37:21+00:00
**Comments**: 2
**Labels**: help wanted, good first issue, high priority, model

### Description

See: https://twitter.com/ggerganov/status/1640945226662420483

The gpt4all ggml model has an extra `<pad>` token (i.e. `n_vocab = 32001`).
Need to add it during the conversion. Should be an optional command line argument to the script to specify if the token should be added or not

---

## Issue #N/A: User should be able to return control without inserting a newline

**Link**: https://github.com/ggml-org/llama.cpp/issues/587
**State**: closed
**Created**: 2023-03-29T04:40:03+00:00
**Closed**: 2024-04-12T01:07:23+00:00
**Comments**: 8
**Labels**: enhancement, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Current Behavior

In Interactive mode, if the user presses return, a newline is entered into the input. This makes it impossible to return control without inserting a newline.

A special-case exception was recently added for when the user enters only a newline and no other text.  #529

[... truncated for brevity ...]

---

## Issue #N/A: [Build] get warming about gcc extension

**Link**: https://github.com/ggml-org/llama.cpp/issues/585
**State**: closed
**Created**: 2023-03-28T23:50:08+00:00
**Closed**: 2023-03-29T13:20:09+00:00
**Comments**: 2

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [*] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [*] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [*] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [*] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior


# Current Behavior

Warming in build ggml.c

1965:72: warming: binary contacts a C2X feature or GCC extension
Const __m256 cross_scales = _mm256_blend_ps(scale_0, scale_1, 0b10101010)

# Environment and Context 

All was fine with 26 hours away versions

* Physical (or virtual)

[... truncated for brevity ...]

---

## Issue #N/A: Fix failing CI test using thread sanitizer

**Link**: https://github.com/ggml-org/llama.cpp/issues/582
**State**: closed
**Created**: 2023-03-28T17:16:53+00:00
**Closed**: 2023-04-02T07:18:54+00:00
**Comments**: 3
**Labels**: help wanted, high priority, testing

### Description

I cannot reproduce on my machines:

https://github.com/ggerganov/llama.cpp/actions/runs/4545676297/jobs/8013336777

If someone that can reproduce, please try to fix this

---

## Issue #N/A: |BUG] ggml spawns threads even BLAS is used

**Link**: https://github.com/ggml-org/llama.cpp/issues/578
**State**: closed
**Created**: 2023-03-28T15:02:01+00:00
**Closed**: 2024-04-12T01:07:25+00:00
**Comments**: 3
**Labels**: bug, performance, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior
ggml should not spawn threads for the initial prompt ingestion when using BLAS.

# Current Behavior
ggml does spawn threads even when using BLAS.

# Environment and Context 
Reproducible using latest OpenBLAS with PR https://github.com/xianyi/OpenBLAS/pull/3970 (f

[... truncated for brevity ...]

---

## Issue #N/A: --help may show the wrong default values when used after other arguments

**Link**: https://github.com/ggml-org/llama.cpp/issues/573
**State**: closed
**Created**: 2023-03-28T13:26:10+00:00
**Closed**: 2023-04-02T02:41:14+00:00
**Comments**: 0
**Labels**: bug, documentation

### Description

For example, running `./main -b 512 --help` will show the help and say that 512 is the default batch size, which is wrong. This may lead to confusion.

---

## Issue #N/A: Missing model data

**Link**: https://github.com/ggml-org/llama.cpp/issues/566
**State**: closed
**Created**: 2023-03-28T08:58:20+00:00
**Closed**: 2023-03-30T23:24:25+00:00
**Comments**: 3
**Labels**: need more info, model

### Description

# Prerequisites
when I clone the project into my centos, I enter the llama.cpp and run the command of "make". It works but didn't make successfully.

# Expected Behavior
the info after the command "make" was running:
```
I llama.cpp build info: 
I UNAME_S:  Linux
I UNAME_P:  x86_64
I UNAME_M:  x86_64
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -mavx2 -mfma -mf16c -msse3 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512cd
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread
I LDFLAGS:  
I CC:       cc (GCC) 7.3.1 20180303 (Red Hat 7.3.1-5)
I CXX:      g++ (GCC) 7.3.1 20180303 (Red Hat 7.3.1-5)

cc  -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -mavx2 -mfma -mf16c -msse3 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512cd   -c ggml.c -o ggml.o
g++ -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread -c llama.cpp -o llama.o
g++ -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread -c examples/common.cp

[... truncated for brevity ...]

---

## Issue #N/A: [Feature Request] Simplified API for Inference and HTTP Server Integration

**Link**: https://github.com/ggml-org/llama.cpp/issues/565
**State**: closed
**Created**: 2023-03-28T08:42:38+00:00
**Closed**: 2023-03-28T11:43:00+00:00
**Comments**: 3
**Labels**: duplicate, enhancement

### Description

First I want to express my deep gratitude for this project, thank you guys so much!

I'm writing to inquire about potential improvements to the API for inference, as well as the possibility of integrating an HTTP server for serving text generation requests. Specifically, I'm interested in the following:

1. A simplified and more flexible method for inference that allows for easier integration with external applications. I'm looking to manage chat history in a separate application and would like to have a straightforward way to perform inference on user-provided text.

2. The ability to serve text generation requests over HTTP. I'm interested in implementing a client-server architecture and would like to know if there are plans to include an HTTP server in the repository.

I understand that the repository is rapidly evolving, and I'm excited to see the new features and improvements you have planned. I'm planning to hack an http server together by myself, but I want to find out w

[... truncated for brevity ...]

---

## Issue #N/A: Broken seed?

**Link**: https://github.com/ggml-org/llama.cpp/issues/561
**State**: closed
**Created**: 2023-03-27T19:16:06+00:00
**Closed**: 2023-03-27T19:27:48+00:00
**Comments**: 4

### Description

Hello,

Is it me or the seed doesn't work anymore? No matter which seed I choose I always get the same output now.

---

## Issue #N/A: [Info] We built a mobile interactive version using flutter and it runs super well on a simple Oneplus 7 with 8GB ram.

**Link**: https://github.com/ggml-org/llama.cpp/issues/560
**State**: closed
**Created**: 2023-03-27T18:46:16+00:00
**Closed**: 2023-03-28T10:30:46+00:00
**Comments**: 1

### Description

# Have fun

Hi, it is not a real issue, but i want to share here that we made a running app with interactive mode on mobiles using your repo.
You can find our repo here : [https://github.com/Bip-Rep/sherpa](https://github.com/Bip-Rep/sherpa)

Unfortunately it doesnt have your latest commit because there is an error during runtime but we made a fork here [https://github.com/Bip-Rep/llama.cpp](https://github.com/Bip-Rep/llama.cpp) You need to be on the "for_mobile" branch to build the libraries. We are using an older working commit on this branch.

We translated the main functions of llama.cpp in dart.


## Working demo
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/jdw7oABjTeQ/0.jpg)](https://www.youtube.com/watch?v=jdw7oABjTeQ)
Click on the image to view the video on YouTube.

Hope this helps. 


---

## Issue #N/A: Has anyone tried Dolly-like models?

**Link**: https://github.com/ggml-org/llama.cpp/issues/558
**State**: closed
**Created**: 2023-03-27T16:55:59+00:00
**Closed**: 2023-03-28T10:33:39+00:00
**Comments**: 3

### Description

I just watched the latest video of my favorite youtuber - https://www.youtube.com/watch?v=AWAo4iyNWGc&t=14s and was wondering, if someone has already quantized & converted one of these to be compatible with llama.cpp?
The beauty of Dolly-like models is that they're based on open source [gpt-j-6B from EleutherAI](https://huggingface.co/EleutherAI/gpt-j-6B), so noone will be hunting us for using them without an ask.

---

## Issue #N/A: Perplexity test stopping before the supposed end.

**Link**: https://github.com/ggml-org/llama.cpp/issues/557
**State**: closed
**Created**: 2023-03-27T15:49:15+00:00
**Closed**: 2024-04-12T01:07:27+00:00
**Comments**: 4
**Labels**: need more info, stale

### Description

# Expected Behavior

Hello,
I'm testing the perplexity of the alpaca-7b-native-q4.bin (RTN quantized) to compare with the regular Llama model.

# Current Behavior

The problem is that the test stops before the supposed end (655 chunks)

```
llama_model_load: loading model part 1/1 from '.\models\alpaca-7b-native-q4.bin'
llama_model_load: .................................... done
llama_model_load: model size =  4017.27 MB / num tensors = 291
llama_init_from_file: kv self size  =  256.00 MB

system_info: n_threads = 4 / 16 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | VSX = 0 |
perplexity : calculating perplexity over 655 chunks
81.02 seconds per pass - ETA 14.74 hours
[1]6.2081,[2]6.7263,[3]7.8961,[4]9.2302,[5]9.1509,[6]8.9913,[7]9.2372,[8]9.3097,[9]9.8923,[10]10.2558,[11]10.5684,[12]10.5985,[13]10.4832,[14]10.5467,[15]10.9168,[16]10.2984,[17]10.1309,[18]10.0642,[19]9.4963,[20]9.493

[... truncated for brevity ...]

---

## Issue #N/A: Windows defender finds a virus in current master branch

**Link**: https://github.com/ggml-org/llama.cpp/issues/554
**State**: closed
**Created**: 2023-03-27T10:43:09+00:00
**Closed**: 2023-03-28T09:13:13+00:00
**Comments**: 3
**Labels**: wontfix

### Description

I'm using Windows 10 LTSC
At the moment, on the state of [this commit](https://github.com/ggerganov/llama.cpp/commit/7e5395575a3360598f2565c73c8a2ec0c0abbdb8), windows defender finds a virus in the master branch.

![image](https://user-images.githubusercontent.com/33938415/227919494-c34cbb4d-32c3-4873-b094-98510ea36abc.png)


---

## Issue #N/A: Cannot build in CentOS

**Link**: https://github.com/ggml-org/llama.cpp/issues/552
**State**: closed
**Created**: 2023-03-27T08:36:21+00:00
**Closed**: 2023-03-29T06:56:13+00:00
**Comments**: 10
**Labels**: build

### Description

Hello, can somebody help me build llama.cpp on CentOS? Or maybe provide binaries?

```bash
[root@vmd89384 llama.cpp]# make
I llama.cpp build info: 
I UNAME_S:  Linux
I UNAME_P:  x86_64
I UNAME_M:  x86_64
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -mavx2 -mfma -mf16c -msse3
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread
I LDFLAGS:  
I CC:       cc (GCC) 4.8.5 20150623 (Red Hat 4.8.5-44)
I CXX:      g++ (GCC) 4.9.2

cc  -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -mavx2 -mfma -mf16c -msse3   -c ggml.c -o ggml.o
ggml.c:77:23: fatal error: stdatomic.h: No such file or directory
 #include <stdatomic.h>
                       ^
compilation terminated.
make: *** [ggml.o] Error 1
[root@vmd89384 llama.cpp]# 
```

I use latest code of this repo, gcc version is 4.9.2

uname is `x86_64 x86_64 x86_64 GNU/Linux`

---

## Issue #N/A: Support for In context learning

**Link**: https://github.com/ggml-org/llama.cpp/issues/549
**State**: closed
**Created**: 2023-03-27T07:22:23+00:00
**Closed**: 2023-03-27T07:42:24+00:00
**Comments**: 0

### Description

Hello,

I would like to know if this port in cpp of llama does support in context learning. 



---

## Issue #N/A: unknown tensor '' in model file

**Link**: https://github.com/ggml-org/llama.cpp/issues/543
**State**: closed
**Created**: 2023-03-27T00:11:51+00:00
**Closed**: 2023-03-27T00:30:47+00:00
**Comments**: 3

### Description

![image](https://user-images.githubusercontent.com/89653506/227813653-b2657cf7-8c98-4e91-89b4-d0304c37fced.png)
![image](https://user-images.githubusercontent.com/89653506/227813663-7b002b7d-d394-45a4-9b87-c736a6915a49.png)
![image](https://user-images.githubusercontent.com/89653506/227813677-c8cc3de7-a155-4201-95f0-297100d3d1b1.png)

So as to prevent mass spam of very large segments of code that i dont think are necessarily useful - i am wondering what i might be doing wrong here - went ahead and went through the entire process start to finish twice just to try and check against maybe a random typo somewhere - but same result both times. 



---

## Issue #N/A: Error loading llama 65b 4bit model (HFv2) converted from .pt format

**Link**: https://github.com/ggml-org/llama.cpp/issues/538
**State**: closed
**Created**: 2023-03-26T20:26:54+00:00
**Closed**: 2023-03-27T05:22:28+00:00
**Comments**: 4
**Labels**: invalid

### Description

I used this command to get the converted model:

`python3 convert-gptq-to-ggml.py "path/to/llama-65b-4bit.pt" "path/to/tokenizer.model" "./models/ggml-llama-65b-q4_0.bin"`

I run it with this command:

`./main -m ./models/ggml-llama-65b-q4_0.bin -n 128`

And this is what I get at the end of the output:

```
llama_model_load: loading model part 1/8 from './models/ggml-llama-65b-q4_0.bin'
llama_model_load: llama_model_load: tensor 'tok_embeddings.weight' has wrong size in model file
llama_init_from_file: failed to load model
main: error: failed to load model './models/ggml-llama-65b-q4_0.bin'
```

P. S. Yes, I'm using the latest (or at least today's) version of this repo. While I'm at it, many thanks to ggerganov and everyone else involved! Great job.

---

## Issue #N/A: Docker Issus ''Illegal instruction''

**Link**: https://github.com/ggml-org/llama.cpp/issues/537
**State**: closed
**Created**: 2023-03-26T19:18:11+00:00
**Closed**: 2024-04-12T01:07:28+00:00
**Comments**: 24
**Labels**: bug, hardware, stale

### Description

I try to make it run the docker version on Unraid, 

I run this as post Arguments:
`--run -m /models/7B/ggml-model-q4_0.bin -p "This is a test" -n 512`

I got this error:  `/app/.devops/tools.sh: line 40:     7 Illegal instruction     ./main $arg2`

Log:
```
main: seed = 1679843913
llama_model_load: loading model from '/models/7B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 4096
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 32
llama_model_load: n_layer = 32
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 11008
llama_model_load: n_parts = 1
llama_model_load: type    = 1
llama_model_load: ggml ctx size = 4273.34 MB
llama_model_load: mem required  = 6065.34 MB (+ 1026.00 MB per state)
/app/.devops/tools.sh: line 40:     7 Illegal instruction     ./main $arg2
```

I have run this whitout any issus:  `--all-in-one "/models

[... truncated for brevity ...]

---

## Issue #N/A: Logo in Social Preview

**Link**: https://github.com/ggml-org/llama.cpp/issues/536
**State**: closed
**Created**: 2023-03-26T18:03:49+00:00
**Closed**: 2023-03-28T18:34:37+00:00
**Comments**: 3
**Labels**: enhancement, 🦙.

### Description

Not a bug, but a useful thing. Put the logo in the Social Preview like in this project:
<img width="843" alt="Screenshot 2023-03-26 at 20 01 58" src="https://user-images.githubusercontent.com/163333/227795065-61d531a9-e515-44bf-b570-086ea8aa7bf2.png">

 It will be then showed as a preview image on Twitter etc.
<img width="591" alt="Screenshot 2023-03-26 at 20 02 52" src="https://user-images.githubusercontent.com/163333/227795109-d2f84554-1f08-4d82-8b3f-11be2d4de1ef.png">




---

## Issue #N/A: Build your windows binaries with Clang and not MSVC.

**Link**: https://github.com/ggml-org/llama.cpp/issues/534
**State**: closed
**Created**: 2023-03-26T17:12:25+00:00
**Closed**: 2024-04-12T01:07:30+00:00
**Comments**: 5
**Labels**: enhancement, build, windows, stale

### Description

Hello,

Your [windows binaries releases](https://github.com/ggerganov/llama.cpp/releases) have probably been built with MSVC and I think there's a better way to do it.

# Expected Behavior

I have a Intel® Core™ i7-10700K and the builds are supposed to recognize those architectures: [AVX | AVX2 | FMA | SSE3 | F16C]

# Current Behavior

Windows (MSVC build)
```
system_info: n_threads = 14 / 16 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 0 | 
NEON = 0 | ARM_FMA = 0 | F16C = 0 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 0 | VSX = 0 |
```
It misses the FMA, SSE3 and the F16C architectures.

# Fix with Clang

If you build with Clang you'll get all the architectures right:

Windows (Clang build)
```
system_info: n_threads = 14 / 16 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 1 | 
NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | VSX = 0 |
```

# How to build with Clang

1. Install Clang
To do this, you have to install some

[... truncated for brevity ...]

---

## Issue #N/A: Is it possible to avoid printing input when using Alpaca models and prompt from file?

**Link**: https://github.com/ggml-org/llama.cpp/issues/533
**State**: closed
**Created**: 2023-03-26T16:26:02+00:00
**Closed**: 2024-04-12T01:07:32+00:00
**Comments**: 5
**Labels**: enhancement, stale

### Description

I want to use prompt from file using `-f` options and alpaca models. Nevertheless, when I use like that, the llama.cpp first prints out the whole input. How to avoid printing out whole input ?

---

## Issue #N/A: [Feature Suggestion] Load/Save current conversation's tokens into file

**Link**: https://github.com/ggml-org/llama.cpp/issues/532
**State**: closed
**Created**: 2023-03-26T15:49:50+00:00
**Closed**: 2024-04-12T01:07:34+00:00
**Comments**: 5
**Labels**: duplicate, enhancement, stale

### Description

Now that we have infinite transcription mode. Would it be possible to dump tokens into file and load them back next time you run llama.cpp to resume conversation?

Although it will be tricky to implement efficiently with long conversations, for example by
- storing prompt itself as tokens
- store in-between messages as raw text
- store last messages within ctx_size as tokens



---

## Issue #N/A: Infinity transcript mode may stuck in ram?

**Link**: https://github.com/ggml-org/llama.cpp/issues/530
**State**: closed
**Created**: 2023-03-26T15:00:39+00:00
**Closed**: 2023-03-27T00:59:20+00:00
**Comments**: 3
**Labels**: need more info, generation quality

### Description

After ctx > 2048 or whatever set in -c, While close the terminal, the transcript may have a chance continuously running in system.

Linux amd64 5.19 ubuntu base.

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

## Issue #N/A: "Not enough context memory" on raspberry pi

**Link**: https://github.com/ggml-org/llama.cpp/issues/522
**State**: closed
**Created**: 2023-03-26T12:10:06+00:00
**Closed**: 2023-03-26T16:26:45+00:00
**Comments**: 13

### Description

I tried loading the 7B model on a raspberry pi 4 (8GB) and it said there was not enough memory in the context pool, before segfaulting. On the raspberry pi, I am accessing the model from a hard disk on a seperate laptop, shared over sshfs, but on the laptop itself (x86 linux, 4GB RAM + 10GB swap) I access it over USB and it works perfectly fine. I am using the one before the latest versions of llama.cpp on both systems, but i changed one line of code in the raspberry pi's version. I replaced `#include <immintrin.h>` with `#include <arm_neon.h>` in ggml.c

While I was writing this I updated the raspberry pi's version of the software to the latest version and the bug still occurs. I will only disply the log from the previous version though

Log:
```
pi@raspberrypi:~/clones/llama.cpp $ ./main -m /path/to/network/drive/LLaMA/7B/ggml-model-q4_0.bin.tmp -i --color -t 3
main: seed = 1679831014
llama_model_load: loading model from '/path/to/network/drive/LLaMA/7B/ggml-model-q4_0.bin.tm

[... truncated for brevity ...]

---

## Issue #N/A: Visual studio build fails with 'undeclared identifier'

**Link**: https://github.com/ggml-org/llama.cpp/issues/519
**State**: closed
**Created**: 2023-03-26T09:17:55+00:00
**Closed**: 2023-03-26T18:43:48+00:00
**Comments**: 2

### Description

I am getting the following errors when building from visual studio 2022. Any ideas why?


```
llama.cpp\ggml.c(5838,5): error C2065: 'ne12': undeclared identifier
llama.cpp\ggml.c(5839,5): error C2065: 'ne13': undeclared identifier
llama.cpp\ggml.c(5840,5): error C2065: 'ne2': undeclared identifier
llama.cpp\ggml.c(5840,5): error C2065: 'ne12': undeclared identifier
llama.cpp\ggml.c(5841,5): error C2065: 'ne3': undeclared identifier
llama.cpp\ggml.c(5841,5): error C2065: 'ne13': undeclared identifier
llama.cpp\ggml.c(5844,5): error C2065: 'nb00': undeclared identifier
llama.cpp\ggml.c(5852,5): error C2065: 'ne0': undeclared identifier
llama.cpp\ggml.c(5853,5): error C2065: 'ne1': undeclared identifier
llama.cpp\ggml.c(5854,5): error C2065: 'ne2': undeclared identifier
llama.cpp\ggml.c(5855,5): error C2065: 'ne3': undeclared identifier
```

Generated the solution using these commands.

```
cmake -S . -B build/ -D CMAKE_BUILD_TYPE=Release
cmake --build build/ --confi

[... truncated for brevity ...]

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

## Issue #N/A: Fails to run inside Docker from Ubuntu 22.04

**Link**: https://github.com/ggml-org/llama.cpp/issues/513
**State**: closed
**Created**: 2023-03-25T22:42:37+00:00
**Closed**: 2024-04-12T01:07:38+00:00
**Comments**: 3
**Labels**: bug, build, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ :white_check_mark: ] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [:white_check_mark:  ] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ :white_check_mark: ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ :white_check_mark: ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Expecting it not to fail when running via Docker

# Current Behavior

Fails when running via Docker

Please provide a detailed written description of what `llama.cpp` did, instead. 

C

[... truncated for brevity ...]

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

## Issue #N/A: Comparison of Windows Build VS Unix Build (through WSL2)

**Link**: https://github.com/ggml-org/llama.cpp/issues/507
**State**: closed
**Created**: 2023-03-25T20:09:51+00:00
**Closed**: 2024-04-12T01:07:40+00:00
**Comments**: 24
**Labels**: question, build, stale

### Description

# Environment and Context 
Hello, 
Before jumping to the subject, here's the environnement I'm working with:

- Windows 10
- Llama-13b-4bit-(GPTQ quantized) model
- Intel® Core™ i7-10700K [AVX | AVX2 | FMA | SSE3 | F16C]

# Expected Behavior

I did some comparaisons between the Windows build and the Unix build (through WSL2 Ubuntu_2204.1.8.0_x64) to see if I can notice some differences between them.

# Deterministic Settings (seed =1)
For both of those builds, I added the same exact settings:
```
-t 14 -n 2024 -c 2024 --temp 0.2 --top_k 40 --top_p 0.6 --repeat_last_n 2048 
--repeat_penalty 1.17647058824 --color --n_parts 1 -b 500 --seed 1 -p "$(cat STORY.txt)"
```

With the contents of STORY.txt as follows:
```
Here's 5 reasons that proves why video-games are good for your brain:
```

#  Test#1: Instruction set architectures

Windows:
```
system_info: n_threads = 14 / 16 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 0 | 
NEON = 0 | ARM_FMA = 0 | F16C = 0 | FP16

[... truncated for brevity ...]

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

## Issue #N/A: Is it possible to run 65B with 32Gb of Ram ?

**Link**: https://github.com/ggml-org/llama.cpp/issues/503
**State**: closed
**Created**: 2023-03-25T17:17:10+00:00
**Closed**: 2023-03-26T10:18:47+00:00
**Comments**: 6
**Labels**: question, hardware, model

### Description

I already quantized my files with this command ./quantize ./ggml-model-f16.bin.X E:\GPThome\LLaMA\llama.cpp-master-31572d9\models\65B\ggml-model-q4_0.bin.X 2 , the first time it reduced my files size from 15.9 to 4.9Gb and when i tried to do it again nothing changed. After i executed this command "./main -m ./models/65B/ggml-model-q4_0.bin -n 128 --interactive-first" and when everything is loaded i enter my prompt, my memory usage goes to 98% (25Gb by main.exe) and i just wait dozens of minutes with nothing that appears heres an example:

**PS E:\GPThome\LLaMA\llama.cpp-master-31572d9> ./main -m ./models/65B/ggml-model-q4_0.bin -n 128 --interactive-first
main: seed = 1679761762
llama_model_load: loading model from './models/65B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 8192
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 64
llama_model_load: n_layer = 80
llama_model_load: n

[... truncated for brevity ...]

---

## Issue #N/A: First argument to printf should be a literal: won't build with -Wformat-security, -Werror=format-security

**Link**: https://github.com/ggml-org/llama.cpp/issues/496
**State**: closed
**Created**: 2023-03-25T14:10:08+00:00
**Closed**: 2023-03-25T14:34:20+00:00
**Comments**: 1

### Description

https://github.com/ggerganov/llama.cpp/blob/e899bf54b291e8c84173a0e534a2c262f3f63229/main.cpp#L481

This won't build with https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html#index-Wformat-security
The safe call would be `printf("%s", buffer.c_str())`

---

## Issue #N/A: Cant compile for an arm 

**Link**: https://github.com/ggml-org/llama.cpp/issues/495
**State**: closed
**Created**: 2023-03-25T13:56:11+00:00
**Closed**: 2023-07-01T18:31:45+00:00
**Comments**: 7
**Labels**: hardware, build

### Description

Consolidate compiler generated dependencies of target utils
[  8%] Building CXX object CMakeFiles/utils.dir/utils.cpp.o
clang++: error: the clang compiler does not support '-mcpu=native'
make[2]: *** [CMakeFiles/utils.dir/build.make:76: CMakeFiles/utils.dir/utils.cpp.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:110: CMakeFiles/utils.dir/all] Error 2
make: *** [Makefile:101: all] Error 2

if i will intentionally delete all -mcpu=native flags, then

[  8%] Built target utils
Consolidate compiler generated dependencies of target ggml
[  8%] Built target ggml
[  8%] Building CXX object CMakeFiles/llama.dir/llama.cpp.o
/home/gh228df/llama.cpp/llama.cpp:1447:33: warning: cast from 'const char *' to 'char *' drops const qualifier [-Wcast-qual]
            finp.read ((char *) word.data(), len);
                                ^
/home/gh228df/llama.cpp/llama.cpp:1448:33: warning: cast from 'const char *' to 'char *' drops const qualifier [-Wcast-qual]
            fout.write((c

[... truncated for brevity ...]

---

## Issue #N/A: Docker error:  Cannot access '/models//7B/ggml-model-f16.bin*': No such file or directory

**Link**: https://github.com/ggml-org/llama.cpp/issues/493
**State**: closed
**Created**: 2023-03-25T12:11:35+00:00
**Closed**: 2023-04-16T09:32:00+00:00
**Comments**: 15
**Labels**: bug, build

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [*] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [*] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [*] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [*] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

llama
└── models
    ├── 13B
    │   ├── checklist.chk
    │   ├── consolidated.00.pth
    │   ├── consolidated.01.pth
    │   └── params.json
    ├── 30B
    │   ├── checklist.chk
    │   ├── consolidated.00.pth
    │   ├── consolidated.01.pth
    │   ├── consolidated.02.pth
  

[... truncated for brevity ...]

---

## Issue #N/A: Clearer windows instructions.. please?

**Link**: https://github.com/ggml-org/llama.cpp/issues/490
**State**: closed
**Created**: 2023-03-25T10:54:52+00:00
**Closed**: 2023-03-26T20:13:00+00:00
**Comments**: 5

### Description

Looking at the README i just see too many incomplete information and steps...  ( the readme is assuming i know things i dont )

If anyone would be nice to ELI5 for me please... I've gotten up to installing visual studio and i cloned the repo in a folder... I got the 7B llama file "consolidated.00.pth" ... I have multiple versions of python so i dont know wich one to use.. 

Then.. the instructions are just head scratching to me.

---

## Issue #N/A: Initial prompt tokens should be loaded instantly and not require inference

**Link**: https://github.com/ggml-org/llama.cpp/issues/484
**State**: closed
**Created**: 2023-03-25T00:38:28+00:00
**Closed**: 2023-03-26T15:48:39+00:00
**Comments**: 7

### Description

# Expected Behavior

Input prompt tokens should load instantly, without having to run inference through the model. The first inference computation should start with the first token after the prompt.

# Current Behavior

I might be misunderstanding something, but it seems in the llama.cpp implementation that all the tokens from the input prompt are fed through the model sequentially (in 8-token batches) before any inference of new tokens can take place. This results in a large delay before getting any responses from the model. 

One of the big benefits of a transformer model, versus an RNN, is that the entire token context window can be ingested and attended to all at once. But llama.cpp seems to be behaving like an RNN, where each prompt token has to be fed in sequentially first, and the output logits ignored, until finally inference can begin.

Am I just misunderstanding something here?

To show it semi-graphically, a transformer should be able to ingest this on first run:

[... truncated for brevity ...]

---

## Issue #N/A: make issue on sbc odroid

**Link**: https://github.com/ggml-org/llama.cpp/issues/482
**State**: closed
**Created**: 2023-03-24T23:32:44+00:00
**Closed**: 2023-05-18T10:54:07+00:00
**Comments**: 2
**Labels**: need more info, hardware

### Description

I am trying to run "make" on an odroid sbc and get following error:

`I llama.cpp build info: 
I UNAME_S:  Linux
I UNAME_P:  unknown
I UNAME_M:  armv7l
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread
I LDFLAGS:  
I CC:       cc (Debian 10.2.1-6) 10.2.1 20210110
I CXX:      g++ (Debian 10.2.1-6) 10.2.1 20210110

cc  -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations   -c ggml.c -o ggml.o
ggml.c: In function ‘ggml_vec_mad_q4_0’:
ggml.c:2049:35: warning: implicit declaration of function ‘vzip1_s8’; did you mean ‘vzipq_s8’? [-Wimplicit-function-declaration]
 2049 |             const int8x8_t vxlt = vzip1_s8(vxls, vxhs);
      |                                   ^~~~~~~~
      |                              

[... truncated for brevity ...]

---

## Issue #N/A: #if defined(AVX) && !defined(F16C)

**Link**: https://github.com/ggml-org/llama.cpp/issues/481
**State**: closed
**Created**: 2023-03-24T23:14:05+00:00
**Closed**: 2023-03-24T23:25:58+00:00
**Comments**: 0

### Description

#if defined(__AVX__) && !defined(__F16C__)
__m256 _mm256_cvtph_ps(__m128i x) {
    ggml_fp16_t const * src = (ggml_fp16_t const *)&x;
    float dst[8];
    for (int i = 0; i < 8; ++i)
        dst[i] = GGML_FP16_TO_FP32(src[i]);
    return *(__m256*)&dst;
}
__m128i _mm256_cvtps_ph(__m256 x, int imm) {
    float const * src = (float const *)&x;
    ggml_fp16_t dst[8];
    for (int i = 0; i < 8; ++i)
        dst[i] = GGML_FP32_TO_FP16(src[i]);
    return *(__m128i*)&dst;
}
#endif

---

## Issue #N/A: The "quantize.exe" script was not found in the current location

**Link**: https://github.com/ggml-org/llama.cpp/issues/479
**State**: closed
**Created**: 2023-03-24T22:50:17+00:00
**Closed**: 2023-03-25T13:29:17+00:00
**Comments**: 8

### Description

Im trying to use it with 65B, but when i want to quantize it, i have the error "The "quantize.exe" script was not found in the current location"

Im running it on windows 11 with 48bg of ram and I7 12700k

---

## Issue #N/A: 7B model returning complete non-sense

**Link**: https://github.com/ggml-org/llama.cpp/issues/474
**State**: closed
**Created**: 2023-03-24T20:05:37+00:00
**Closed**: 2023-03-25T10:26:17+00:00
**Comments**: 2

### Description

i followed a YouTube video to build the program https://www.youtube.com/watch?v=coIj2CU5LMU&t=186s. it itself follows the issue #103 

# Expected Behavior

As a test I ran the ./chat.sh in git bash, it ran but when I said the AI "hello" I expected hello back.

# Current Behavior

it responded with
```
‼ ▼→▬▬▲↨‼↑♥♠♦"♥ ☻ Ôüç ∟ Ôüç ↔¶
‼∟ Ôüç ♥
►♠
▼!↕☻    ▼ ↓     $▼∟▼↕♣↔"‼↔♥
☺       ►        ↔ Ôüç   #↑"▼↑♠$$▬☺☻
```

# Environment and Context 

Please provide detailed information about your computer setup. This is important in case the issue is not reproducible except for under certain specific conditions.

I’m running a i7-13 th gen with 32 go of ram and a 3060.

windows 11 home

git bash to run the commands and cmake to compile

```
Python 3.10.10
cmake 3.26.1
g++.exe (MinGW.org GCC-6.3.0-1) 6.3.0
```

# Failure Logs

```
$ ./chat.sh
main: seed = 1679687646
llama_model_load: loading model from './models/7B/ggml-model-f16.bin' - please wait ...
llama_

[... truncated for brevity ...]

---

## Issue #N/A: How do we finetune the model with new data?

**Link**: https://github.com/ggml-org/llama.cpp/issues/466
**State**: closed
**Created**: 2023-03-24T16:12:02+00:00
**Closed**: 2024-04-10T01:07:59+00:00
**Comments**: 17
**Labels**: enhancement, stale

### Description

Can we have a finetune.cpp or finetune.exe file to incorporate new data into the model? The use case will be to design an AI model that can do more than just general chat. It can become very knowledgeable in specific topics they are finetuned on. Also, after creating the finetune.exe , please ensure no GPU is required for the entire process. Because that is what makes this repo awesome in the first place.

---

## Issue #N/A: Question about Web integration/Articles analisys

**Link**: https://github.com/ggml-org/llama.cpp/issues/464
**State**: closed
**Created**: 2023-03-24T15:50:06+00:00
**Closed**: 2023-03-24T18:56:15+00:00
**Comments**: 0

### Description

Is it possible to make bridge to web for something unknown to model? (ChatGPT introduced plugins to search web, etc)
Or at least for model to read article/book and answer questions about it? 

---

## Issue #N/A: Simplify the quantization process

**Link**: https://github.com/ggml-org/llama.cpp/issues/463
**State**: closed
**Created**: 2023-03-24T14:47:25+00:00
**Closed**: 2023-07-28T19:41:16+00:00
**Comments**: 3
**Labels**: enhancement

### Description

The current quantization call stack is long and difficult to debug, which makes extending or adding new quantization methods in the future a major issue. This is because changes would need to be made in various places.

Additionally, we should aim to add drivers that help with benchmarking various quantization methods.

**The current stack:**
1. quantize.py invokes the quantize binary
2. quantize.cpp reads model and logs metrics
3. llama.cpp loads model weights, checks quantization type, and sends to quantization function
4. ggml.c performs the actual quantization

Open to suggestions here and would like to hear if it's worth investing our time and effort

---

## Issue #N/A: [fixed]The last code build with memory fix running result is not good in my pc.

**Link**: https://github.com/ggml-org/llama.cpp/issues/462
**State**: closed
**Created**: 2023-03-24T14:22:06+00:00
**Closed**: 2023-03-27T00:13:38+00:00
**Comments**: 10
**Labels**: bug, performance

### Description

Be obviously slower with Q_1 30b model. And the memory usage become garbage...
(Linux 5.19 x64 Ubuntu base)

---

## Issue #N/A: 2-bit integer quantization 

**Link**: https://github.com/ggml-org/llama.cpp/issues/456
**State**: closed
**Created**: 2023-03-24T06:55:44+00:00
**Closed**: 2023-06-24T19:17:24+00:00
**Comments**: 16
**Labels**: enhancement, research 🔬

### Description

Add `Q2_0` and `Q2_1` quantization support to `ggml`:

- Follow the existing `Q4_0` and `Q4_1` implementations
- Implement [reference scalar quantization and dequantization routines](https://github.com/ggerganov/llama.cpp/blob/3cd8dde0d1357b7f11bdd25c45d5bf5e97e284a0/ggml.c#L407-L449)
- I suspect we might have to use `QK == 16` in this case to compensate for further accuracy losses
- Add SIMD support for a specific architecture - investigate best strategy to perform the `ggml_vec_dot_q2()` computation
- No need to implement `ggml_vec_mad_q2()` - these will be deprecated soon
- Compute perplexity scores

The expected model sizes for 7B and `QK == 16` are:

- `Q2_0` - 3.2 GB

For `QK == 32` we have:

- `Q2_0` - 2.4 GB
- `Q2_1` - 3.2 GB

Before you send me papers that show 2-bit quantization does not work - no need. I want to have this supported anyway. I have something in mind. The efforts needed to add this support are so small that there is no reason not to do it.

---

## Issue #N/A: Add support for running bloom models

**Link**: https://github.com/ggml-org/llama.cpp/issues/452
**State**: closed
**Created**: 2023-03-24T02:26:13+00:00
**Closed**: 2024-04-10T01:08:00+00:00
**Comments**: 7
**Labels**: enhancement, model, stale

### Description

Bloom models have a more permissive license than llama models and are also multilingual in nature. While there is a project [based on llama.cpp](https://github.com/NouamaneTazi/bloomz.cpp) which can perform inference of bloom models, development seems to be slow and might even stagnate after a few days. So I am requesting to add support for running bloom models using llama.cpp(most probably with a command-line switch)

---

## Issue #N/A: Can it support avx cpu's older than 10 years old?

**Link**: https://github.com/ggml-org/llama.cpp/issues/451
**State**: closed
**Created**: 2023-03-24T02:19:30+00:00
**Closed**: 2023-07-28T19:40:41+00:00
**Comments**: 10
**Labels**: enhancement, hardware, build

### Description

I can't run any model due to my cpu is from before 2013.So I don't have avx2 instructions.Can you please support avx cpus?

---

## Issue #N/A: Change ./main help output to better reflect context size's affect on generation length

**Link**: https://github.com/ggml-org/llama.cpp/issues/449
**State**: closed
**Created**: 2023-03-24T01:38:43+00:00
**Closed**: 2023-07-28T19:40:24+00:00
**Comments**: 2
**Labels**: documentation, enhancement

### Description

### Discussed in https://github.com/ggerganov/llama.cpp/discussions/446

<div type='discussions-op-text'>

<sup>Originally posted by **cmp-nct** March 24, 2023</sup>
I've been testing alpaca 30B (-t 24 -n 2000 --temp 0.2 -b 32 --n_parts 1 --ignore-eos --instruct)
I've consistently have it "stop" after 300-400 tokens output (30-40 tokens input)
No error message, no crash and given the -n 2000 and the ignore-eos no reason to stop so early

I guess it would be useful if the program provides a verbose quit reason, though in my case I can't see any reason for it to stop before token max is reached.


I'm not sure if that's a bug to report or if I am missing something.</div>

---

## Issue #N/A: possible to run full sizes?

**Link**: https://github.com/ggml-org/llama.cpp/issues/448
**State**: closed
**Created**: 2023-03-24T00:26:39+00:00
**Closed**: 2023-03-24T20:18:52+00:00
**Comments**: 1

### Description

I'm not sure if this is an enhancement request because maybe it's already supported.  Is it possible to run the full models?  I know they take a ton of extra memory but I'd still like to try them out, eg, 13GB for 7B instead of 3.9GB, etc.

---

## Issue #N/A: Cannot run llama.cpp on termux. Bash permission denied

**Link**: https://github.com/ggml-org/llama.cpp/issues/447
**State**: closed
**Created**: 2023-03-23T23:35:53+00:00
**Closed**: 2023-03-24T18:29:39+00:00
**Comments**: 4

### Description

When trying to run './bin/main/ -m ./models/7B/ggml-model-q4_0.bin -n 128' termux throws this output:
bash: ./bin/main: permission denied

---

## Issue #N/A: [ERROR] Using "make" command

**Link**: https://github.com/ggml-org/llama.cpp/issues/443
**State**: closed
**Created**: 2023-03-23T22:26:52+00:00
**Closed**: 2023-04-22T17:29:32+00:00
**Comments**: 3
**Labels**: hardware, build

### Description

Hello evryone, 

I have an issue when i run "make" cmd : 
I use Ubuntu 22.04 in VirtualBox
Make version : GNU Make 4.3


Here the return of cmd 

<pre>I llama.cpp build info: 

I UNAME_S:  Linux

I UNAME_P:  x86_64

I UNAME_M:  x86_64

I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -mavx2 -msse3

I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread

I LDFLAGS:  

I CC:       cc (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0

I CXX:      g++ (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0



cc  -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -mavx2 -msse3   -c ggml.c -o ggml.o

In file included from <b>/usr/lib/gcc/x86_64-linux-gnu/11/include/immintrin.h:101</b>,

                 from <b>ggml.c:158</b>:

<b>ggml.c:</b> In function ‘<b>ggml_vec_dot_f16</b>’:

<b>/usr/lib/gcc/x86_64-linux-gnu/11/include/f16cintrin.h:52:1:</b> <font color="#C01C28"><b>error: </b></font>inlining failed in call to ‘<b>always_inline</

[... truncated for brevity ...]

---

## Issue #N/A: Converting alpaca-native-GPTQ models into ggml models

**Link**: https://github.com/ggml-org/llama.cpp/issues/442
**State**: closed
**Created**: 2023-03-23T22:02:03+00:00
**Closed**: 2023-07-28T19:39:46+00:00
**Comments**: 21
**Labels**: enhancement, model

### Description

# Expected Behavior

Hello, 

I wanted to convert the alpaca-native 7b GPTQ file (pt file) into a ggml file with the convert-gptq-to-ggml.py script https://github.com/ggerganov/llama.cpp/blob/master/convert-gptq-to-ggml.py

# Current Behavior

The problem is that I have this error 

```
D:\Large Language Models\CONVERTISSEURS\gptq to ggml>python convert-gptq-to-ggml.py alpaca-native-4b
it.pt tokenizer.model out.bin
32000
32001
Traceback (most recent call last):
  File "D:\Large Language Models\CONVERTISSEURS\gptq to ggml\convert-gptq-to-ggml.py", line 35, in <
module>
    assert tokenizer.vocab_size() == n_vocab
AssertionError
```
32000 is the tokenizer.vocab_size() (Number of tokens on the tokenizer.model)
32001 is the n_vocab (Number of tokens on the model)

The model that is trained with alpaca has 1 more token and it's this one:
"[PAD]": 32000

It looks like that if we want to convert the alpaca native GPTQ models we need to create a new tokenizer.model t

[... truncated for brevity ...]

---

## Issue #N/A: Eliminate `ggml_forward_mul_mat_xxx()` branch for non-contiguous `src0`

**Link**: https://github.com/ggml-org/llama.cpp/issues/441
**State**: closed
**Created**: 2023-03-23T21:26:40+00:00
**Closed**: 2023-07-28T19:37:48+00:00
**Comments**: 0
**Labels**: enhancement

### Description

See explanation here: https://github.com/ggerganov/llama.cpp/pull/439

---

## Issue #N/A: Name change proposal discussion

**Link**: https://github.com/ggml-org/llama.cpp/issues/436
**State**: closed
**Created**: 2023-03-23T18:10:17+00:00
**Closed**: 2023-03-23T18:21:20+00:00
**Comments**: 0

### Description

https://discord.com/channels/1038249716149928046/1080876668530466918/1088523954626494464

---

## Issue #N/A: How to output text to a file?

**Link**: https://github.com/ggml-org/llama.cpp/issues/432
**State**: closed
**Created**: 2023-03-23T17:06:06+00:00
**Closed**: 2023-03-24T15:19:36+00:00
**Comments**: 2
**Labels**: documentation

### Description

I really, really tried hard to understand and modify the code but I am not an expert on C++ and so I find it a little bit difficult to change parts of this software. Is there a way to simply execute a command and get the output without all of that verbosity?

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

## Issue #N/A: "Illegal Instruction" error when converting 7B model to ggml FP16 format (Raspberry Pi 4, 8GB, Raspberry Pi OS, 64-bit)

**Link**: https://github.com/ggml-org/llama.cpp/issues/425
**State**: closed
**Created**: 2023-03-23T11:52:38+00:00
**Closed**: 2023-03-26T15:27:25+00:00
**Comments**: 2
**Labels**: duplicate, hardware

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ /] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ /] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ /] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ /] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

I expected the command to convert the 7B model to ggml FP16 format

# Current Behavior

Illegal instruction error

```
les@raspberrypi:~/llama.cpp $ python3 convert-pth-to-ggml.py models/7B/ 1
Illegal instruction
```


# Environment and Context 

Ras

[... truncated for brevity ...]

---

## Issue #N/A: how to fine tuning model with with dataset (file json/csv..)

**Link**: https://github.com/ggml-org/llama.cpp/issues/414
**State**: closed
**Created**: 2023-03-23T04:29:28+00:00
**Closed**: 2023-03-23T08:55:14+00:00
**Comments**: 0

### Description

No description provided.

---

## Issue #N/A: Mismatch in Vocabulary Size: Investigating Inconsistencies between Token-to-ID and ID-to-Token Dictionaries

**Link**: https://github.com/ggml-org/llama.cpp/issues/413
**State**: closed
**Created**: 2023-03-23T02:05:32+00:00
**Closed**: 2024-04-10T01:08:01+00:00
**Comments**: 10
**Labels**: question, stale

### Description

The total number of vocabulary items in the model file is 32k. When we parse them, there's a mismatch between token_to_id and id_to_token.

The size for token_to_id is: 31,903
The size for id_to_token is: 32,000

I'm curious on why there's a mismatch. Are there some token IDs that are reserved or errors during pre-processing?

---

## Issue #N/A: Add Shared Library Build Target

**Link**: https://github.com/ggml-org/llama.cpp/issues/412
**State**: closed
**Created**: 2023-03-22T22:45:26+00:00
**Closed**: 2023-03-23T20:16:51+00:00
**Comments**: 12
**Labels**: bug, enhancement, build

### Description

With the C API now merged it would be very useful to have build targets for `make` and `cmake` that produce shared library versions of `llama.cpp`. This way `llama.cpp` can just be dynamically linked in other applications.

---

## Issue #N/A: [User] Please fix segmentation fault when prompt is too long

**Link**: https://github.com/ggml-org/llama.cpp/issues/411
**State**: closed
**Created**: 2023-03-22T22:40:32+00:00
**Closed**: 2023-03-23T07:40:21+00:00
**Comments**: 7
**Labels**: bug, need more info

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

I want to be able to run my promt using this command without any `Segmentation fault` error: 
```bash
./main -m ./models/7B/ggml-model-q4_0.bin -t 8 -n 256 --repeat_penalty 1.0 --color -i -r "Prompt:" --temp 1.2 -p "$(cat ../twitch_bot/prompt.md)"
```
Where `promp

[... truncated for brevity ...]

---

## Issue #N/A: Download ggml-alpaca-7b-q4.bin failed CHECKSUM

**Link**: https://github.com/ggml-org/llama.cpp/issues/410
**State**: closed
**Created**: 2023-03-22T21:31:37+00:00
**Closed**: 2023-03-23T09:22:24+00:00
**Comments**: 15
**Labels**: model

### Description

This may well be the end server issue. I tried several times with no luck, just wonder if people have seen this. 
I tried all 3 curl commands. 



---

## Issue #N/A: illegal instructions error on Android

**Link**: https://github.com/ggml-org/llama.cpp/issues/402
**State**: closed
**Created**: 2023-03-22T17:33:25+00:00
**Closed**: 2023-07-28T19:38:12+00:00
**Comments**: 29
**Labels**: need more info, android

### Description

first thanks for the wonderful works so far !!!

i manged to compile it in Linux and windows but i have a problem with android.
i have A52 6 GB but i get "illegal instructions" error.

i compiled the source using wsl2 with  ndk r25 without any errors. i moved the llama folder from sd card to "home" directory in (Termux) in order to have the execute command working. and i converted to original model using the newer source code to avoid "too old" error message but at the end i get this error.

i believe it is because of having avx, avx2 and other instruction already enabled in my build which is arm processors cant handle them but i cant figure it out how to change it to get it working on my android device.
thanks in advanced <3
![ScreenshotTermux](https://user-images.githubusercontent.com/128628434/226988980-5d1a67c3-797b-4eed-8449-164b0c9abefb.jpg)


---

## Issue #N/A: "Illegal Instruction" error when converting 7B model to ggml FP16 format (Raspberry Pi 4, 8GB, Raspberry Pi OS, 64-bit)

**Link**: https://github.com/ggml-org/llama.cpp/issues/401
**State**: closed
**Created**: 2023-03-22T17:20:49+00:00
**Closed**: 2023-03-23T11:35:33+00:00
**Comments**: 1
**Labels**: need more info

### Description

Hello

I'm trying to replicate the process, using 7B on a Raspberry Pi 4 with 8GB of RAM.
I'm running the latest Raspberry Pi OS 64-bit, and all of the software has been updated.
I am following the guidance found in the [Usage section of the README.md](https://github.com/ggerganov/llama.cpp#readme)
I cloned the repo, downloaded 7B and placed it into /llama.cpp/models/7B, the contents of which are below

===7B Contents===
```
l-rw-r--r-- 1 les les  100 Mar 22 15:03 checklist.chk
-rw-r--r-- 1 les les  13G Mar 22 15:03 consolidated.00.pth
-rw-r--r-- 1 les les  101 Mar 22 15:04 params.json
```
===END of 7B Contents===

I have successfully installed all of the suggested Python modules.

When running this command `python3 convert-pth-to-ggml.py models/7B/ 1` I see a short pause, around 5 -10 seconds. Then I receive an `Illegal instruction` error message and I can progress no more.

What am I doing wrong, and how can this be remedied?
I also tried Ubuntu 22.04 64-bit and t

[... truncated for brevity ...]

---

## Issue #N/A: GGML_ASSERT: ggml.c:4014: false zsh: abort      ./main -m ./models/65B/ggml-model-q4_0.bin -t 16 -n 256 --repeat_penalty 1.0 

**Link**: https://github.com/ggml-org/llama.cpp/issues/400
**State**: closed
**Created**: 2023-03-22T17:12:40+00:00
**Closed**: 2023-04-19T19:43:32+00:00
**Comments**: 5
**Labels**: need more info

### Description

Not sure why this happens, I am on the latest commit and I am up-to-date on everything
I did some tests and it seems like it breaks after 500~ tokens
Is this a model limitation or can I fix this by increasing some value?

---

## Issue #N/A: Support for Loading a Subset of Tensors for LoRA Models 

**Link**: https://github.com/ggml-org/llama.cpp/issues/399
**State**: closed
**Created**: 2023-03-22T16:12:51+00:00
**Closed**: 2023-04-17T15:28:57+00:00
**Comments**: 6
**Labels**: enhancement, 🦙., model

### Description

Firstly, thank you for the awesome project. I'm new to LLMs so I hope this suggestion makes sense.

LoRA is a technique used to reduce the number of parameters during finetuning, that is really hitting off with the recent Alpaca stuff. In LoRA models, typically, only the weight matrices Wq and Wv are fine-tuned. 

For projects shipping multiple LoRA fine-tuned models, most of the tensors remain unchanged during the fine-tuning process. Storing all weights multiple times would lead to a significant waste of storage space (e.g., ~3.5 GB of data per fine-tune for a 7B model, multiplied by the number of tasks or personalities you want to ship). Supporting the loading of a subset of tensors for LoRA models would enable efficient storage and loading of these models in llama.cpp, reducing storage space requirements, and maybe memory footprint if you wanted to keep multiple models in memory at the same time.

I propose to extend llama.cpp's functionality by adding support for loading a s

[... truncated for brevity ...]

---

## Issue #N/A: convert-pth-to-ggml.py error with "Got unsupported ScalarType BFloat16"

**Link**: https://github.com/ggml-org/llama.cpp/issues/398
**State**: closed
**Created**: 2023-03-22T16:08:09+00:00
**Closed**: 2023-04-16T09:27:08+00:00
**Comments**: 2
**Labels**: need more info

### Description

Trying to convert  "chavinlo/alpaca-native" alpaca native model's (https://huggingface.co/chavinlo/alpaca-native) weights to ggml but got this error - 

Processing part 0

Processing variable: model.embed_tokens.weight with shape: torch.Size([32001, 4096]) and type: torch.float32
Processing variable: model.layers.0.self_attn.q_proj.weight with shape: torch.Size([4096, 4096]) and type: torch.float32
Processing variable: model.layers.0.self_attn.k_proj.weight with shape: torch.Size([4096, 4096]) and type: torch.float32
Processing variable: model.layers.0.self_attn.v_proj.weight with shape: torch.Size([4096, 4096]) and type: torch.float32
Processing variable: model.layers.0.self_attn.o_proj.weight with shape: torch.Size([4096, 4096]) and type: torch.float32
Processing variable: model.layers.0.self_attn.rotary_emb.inv_freq with shape: torch.Size([64]) and type: torch.bfloat16
Traceback (most recent call last):
  File "/Users/domeie/projects/llama.cpp/convert-pth-to-ggml.py", lin

[... truncated for brevity ...]

---

## Issue #N/A: Investigate alternative approach for Q4 quantization 

**Link**: https://github.com/ggml-org/llama.cpp/issues/397
**State**: closed
**Created**: 2023-03-22T16:03:20+00:00
**Closed**: 2023-04-25T17:20:48+00:00
**Comments**: 58
**Labels**: help wanted, good first issue, research 🔬

### Description

Currently, in [Q4_0](https://github.com/ggerganov/ggml/pull/27) quantization we choose the scaling factor for each 32 group of weights as `abs(max(x_i))/7`. It is easy to see that this is suboptimal.

Consider quantization of the following 4 numbers:

`0.1 0.2 0.3 0.6`

Currently, we would determine a scaling factor of `0.6 / 7 ~= 0.0857` and the dequantized numbers will be:

`0.0857 0.1714 0.3428 0.6`

So the RMS between the dequantized and original values will be non-zero:

`sqrt((0.1 - 0.0857)^2 + (0.2 - 0.1714)^2 + (0.3 - 0.3428)^2 + (0.6 - 0.6)^2) > 0.0`

However, if we choose the scaling factor to be `0.1` instead, then it is easy to see that the original numbers will be quantized perfectly.

So the scaling factor is better to be chosen as the one that minimises some error (e.g. RMS or whatever is more meaningful and easy to compute). Doing that we will certainly achieve better accuracy compared to the existing approach. The question is - how much better?

The g

[... truncated for brevity ...]

---

## Issue #N/A: llama_init_from_file: failed to load model

**Link**: https://github.com/ggml-org/llama.cpp/issues/388
**State**: closed
**Created**: 2023-03-22T10:00:00+00:00
**Closed**: 2023-03-24T02:54:48+00:00
**Comments**: 4
**Labels**: need more info

### Description

When I execute this command：
make -j && ./main -m ./models/7B/ggml-model-q4_0.bin -p "Building a website can be done in 10 simple steps:" -n 512

An error was reported：
llama_init_from_file: failed to load model
main: error: failed to load model './models/7B/ggml-model-q4_0.bin'

---

## Issue #N/A: Compute perplexity fails with too many tokens exception

**Link**: https://github.com/ggml-org/llama.cpp/issues/385
**State**: closed
**Created**: 2023-03-22T08:08:29+00:00
**Closed**: 2023-03-22T16:09:40+00:00
**Comments**: 9
**Labels**: bug

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

It is supposed to compute perplexity like the original PR: https://github.com/ggerganov/llama.cpp/pull/270

# Current Behavior

However, it fails with the following exception:

```
llama_tokenize: too many tokens
libc++abi: terminating with uncaught exception 

[... truncated for brevity ...]

---

## Issue #N/A: [Documentation] C API examples

**Link**: https://github.com/ggml-org/llama.cpp/issues/384
**State**: closed
**Created**: 2023-03-22T08:08:14+00:00
**Closed**: 2023-06-16T18:58:42+00:00
**Comments**: 11
**Labels**: documentation

### Description

Hey!

There should be a simple example on how to use the new C API (like one that simply takes a hardcoded string and runs llama on it until \n or something like that).
Not sure the the `/examples/` directory is appropriate for this.

Thanks
Niansa

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

## Issue #N/A: Doesn't compile due to missing headers for memcpy and assert

**Link**: https://github.com/ggml-org/llama.cpp/issues/381
**State**: closed
**Created**: 2023-03-22T06:26:16+00:00
**Closed**: 2023-03-22T10:23:22+00:00
**Comments**: 1

### Description

As of the refactor https://github.com/ggerganov/llama.cpp/commit/f5a77a629bd0f37ae1696747633ab42a5530ec15 the program does not compile.
```
g++ -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread -c llama.cpp -o llama.o
llama.cpp: In function ‘bool llama_eval_internal(llama_context&, const llama_token*, int, int, int)’:
llama.cpp:657:5: error: ‘memcpy’ was not declared in this scope
  657 |     memcpy(embd->data, tokens, N*ggml_element_size(embd));
      |     ^~~~~~
llama.cpp:12:1: note: ‘memcpy’ is defined in header ‘<cstring>’; did you forget to ‘#include <cstring>’?
   11 | #include <cassert>
  +++ |+#include <cstring>
   12 | 
make: *** [Makefile:224: llama.o] Error 1
```
Adding these to llama.cpp allows it to compile.
```
#include <cassert>
#include <cstring>
````


---

## Issue #N/A: Alpaca 7B faults on both macOS arm64 and Linux ppc64le

**Link**: https://github.com/ggml-org/llama.cpp/issues/379
**State**: closed
**Created**: 2023-03-22T04:18:50+00:00
**Closed**: 2023-04-16T09:26:21+00:00
**Comments**: 3
**Labels**: bug, model

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

From tip as of this posting, trying to use the available Alpaca 7B model weights causes a fault (out of memory?) on both my macOS and Linux systems. However, LLaMA 7B works fine. M1 MacBook Air with Ventura and 16GB of RAM, and Raptor Talos II 64-thread POWER9 with 64GB RAM. Using the r

[... truncated for brevity ...]

---

## Issue #N/A: Original weights for LLAMA

**Link**: https://github.com/ggml-org/llama.cpp/issues/378
**State**: closed
**Created**: 2023-03-22T03:56:33+00:00
**Closed**: 2023-03-24T22:59:18+00:00
**Comments**: 3
**Labels**: question, need more info

### Description

Hey, I noticed the API is running on CPP, were the original weights in python or CPP? If in python, I would think they were in pytorch since that is Meta's DL platform; do you have the weights in python format?

---

## Issue #N/A: [User] Chinese support

**Link**: https://github.com/ggml-org/llama.cpp/issues/377
**State**: closed
**Created**: 2023-03-22T03:08:45+00:00
**Closed**: 2023-03-22T07:56:23+00:00
**Comments**: 2

### Description

Hi, when will support Chinese input support? Currently seems tokenizer didn't support, previous there were community PR support this, but long time didn't merge.

---

## Issue #N/A: SHA256 checksums correctness

**Link**: https://github.com/ggml-org/llama.cpp/issues/374
**State**: closed
**Created**: 2023-03-21T23:05:19+00:00
**Closed**: 2023-03-23T17:51:06+00:00
**Comments**: 12
**Labels**: bug, model

### Description

> Not all of these checksums seem to be correct. Are they calculated with the "v2" new model format after the tokenizer change? PR: https://github.com/ggerganov/llama.cpp/pull/252 Issue: https://github.com/ggerganov/llama.cpp/issues/324 
> 
> For example, "models/alpaca-7B/ggml-model-q4_0.bin"
> 
> v1: 1f582babc2bd56bb63b33141898748657d369fd110c4358b2bc280907882bf13
> v2: 8d5562ec1d8a7cfdcf8985a9ddf353339d942c7cf52855a92c9ff59f03b541bc 
> 
> The SHA256SUMS file has the old v1 hash.
> Maybe using a naming scheme like "ggml2-model-q4_0.bin" would be good to differentiate between the versions and avoid confusion.
> 
_Originally posted by @anzz1 in https://github.com/ggerganov/llama.cpp/issues/338#issuecomment-1478695874_

edit: After converting the models to the new format, I found out that the "v2" hash above is also incorrect.
The sha256 for `./models/alpaca-7B-ggml/ggml-model-q4_0.bin` is supposed to be `2fe0cd21df9c235c0d917c14e1b18d2d7320ed5d8abe48545518e96bb4227524`

---

## Issue #N/A: [mqy] ./examples/chatLLaMa: line 53: 33476 Segmentation fault: 11

**Link**: https://github.com/ggml-org/llama.cpp/issues/373
**State**: closed
**Created**: 2023-03-21T22:00:14+00:00
**Closed**: 2023-07-28T19:38:41+00:00
**Comments**: 9
**Labels**: bug, duplicate, model

### Description

# Current Behavior

`./examples/chatLLaMa`,  After about 30-round talks, program quite with `Segmentation fault: 11`.
I did another try, input last question, but can't reproduce.

# Environment and Context 

* Physical hardware:

MacBook Pro 2018, 2.6 GHz 6-Core Intel Core i7, 32 GB 2400 MHz DDR4

* Operating System

macOS 13.2.1 (22D68)
Darwin Kernel Version 22.3.0: Mon Jan 30 20:42:11 PST 2023; root:xnu-8792.81.3~2/RELEASE_X86_64 x86_64

# Failure Information (for bugs)

```
./examples/chatLLaMa: line 53: 33476 Segmentation fault: 11  ./main $GEN_OPTIONS --model ... 
...
$USER_NAME:" "$@"
```

# Failure Logs

```
$ git log | head -1
commit 0f6135270839f0715843c4d480c63ae150def419

$ sysctl -n machdep.cpu.brand_string
Intel(R) Core(TM) i7-8850H CPU @ 2.60GHz

$ sysctl -n machdep.cpu.features
FPU VME DE PSE TSC MSR PAE MCE CX8 APIC SEP MTRR PGE MCA CMOV PAT PSE36 CLFSH DS ACPI MMX FXSR SSE SSE2 SS HTT TM PBE SSE3 PCLMULQDQ DTES64 MON DSCPL VMX SMX EST 

[... truncated for brevity ...]

---

## Issue #N/A: The initial token is always empty.

**Link**: https://github.com/ggml-org/llama.cpp/issues/367
**State**: closed
**Created**: 2023-03-21T19:48:37+00:00
**Closed**: 2024-04-10T01:08:02+00:00
**Comments**: 7
**Labels**: need more info, generation quality, stale

### Description

Hello,

I noticed something when trying the chat with Bob is that I always get the first token as empty.

     1 -> ''
  4103 -> ' Trans'
   924 -> 'cript'
   310 -> ' of'
   263 -> ' a'
  7928 -> ' dialog'

So the result is this: 

![image](https://user-images.githubusercontent.com/110173477/226732298-38c21252-059e-4acd-9dfb-70f745347efe.png)

There's this little space at the begining of the text. Maybe this alone can significantly impact the quality of the output, that's why I decided to post this issue.

I'm on a windows 10 using WSL to emulate the linux environnement (the main.exe is not as good as the linux main atm).

I'm using a file that is the result of all those manipulations:

1) I have first a llama-7b-4bit.pt file
2) I converted it with the gptq-to-ggml converter (convert-gptq-to-ggml.py) 
3) I converted it again into the new version of ggml with this script https://github.com/ggerganov/llama.cpp/issues/324#issuecomment-1476227818

Here's the .sh c

[... truncated for brevity ...]

---

## Issue #N/A: In interactive/chat mode, sometimes User: does not appear and I need to manually type in my nickname

**Link**: https://github.com/ggml-org/llama.cpp/issues/364
**State**: closed
**Created**: 2023-03-21T17:24:39+00:00
**Closed**: 2024-04-10T01:08:03+00:00
**Comments**: 11
**Labels**: generation quality, stale

### Description

- In interactive/chat mode, sometimes User: does not appear and I need to manually type in my nickname
for example:

'
AI: Hello
User: Hello
AI: How are you

'
instead of User: appears nothing and i need to manually type in User:, if i press enter without typing anything then llama diverges from conversation and starts spouting random stuff.

it also sometimes happens with AI's reply, as if its reply was eaten and I can type in stuff instead or press enter

---

## Issue #N/A: Update the convert-gptq-to-ggml.py with the new tokenizer output

**Link**: https://github.com/ggml-org/llama.cpp/issues/362
**State**: closed
**Created**: 2023-03-21T17:08:45+00:00
**Closed**: 2023-03-23T20:18:15+00:00
**Comments**: 0
**Labels**: help wanted, high priority

### Description

Apply the changes from #252 to [convert-gptq-to-ggml.py](https://github.com/ggerganov/llama.cpp/blob/master/convert-gptq-to-ggml.py)

For more info about what this script does, see #301 

---

## Issue #N/A: Invalid model error : too old, regenerate your model files!

**Link**: https://github.com/ggml-org/llama.cpp/issues/361
**State**: closed
**Created**: 2023-03-21T16:51:17+00:00
**Closed**: 2023-03-22T05:54:53+00:00
**Comments**: 14
**Labels**: documentation, model

### Description

Downloaded Alpaca 7B model successfully using the following command as mentioned in README.md:
`curl -o ./models/ggml-alpaca-7b-q4.bin -C - https://gateway.estuary.tech/gw/ipfs/QmUp1UGeQFDqJKvtjbSYPBiZZKRjLp8shVP9hT8ZB9Ynv1`

When I try to execute the command:
`main -m ./models/ggml-alpaca-7b-q4.bin --color -f ./prompts/alpaca.txt -ins`

This is the error output:
main: seed = 1679417098
llama_model_load: loading model from './models/ggml-alpaca-7b-q4.bin' - please wait ...
llama_model_load: invalid model file './models/ggml-alpaca-7b-q4.bin' (too old, regenerate your model files!)
main: failed to load model from './models/ggml-alpaca-7b-q4.bin'

How to fix this? Is the downloaded model corrupted and should I download it again? What is the SHA1 hash of the model so that I can verify that the downloaded model is corrupted or not?

---

## Issue #N/A: Converting GGML Q4_0 back to Torch checkpoint for HuggingFace/Pytorch consumption/training/finetuning

**Link**: https://github.com/ggml-org/llama.cpp/issues/359
**State**: closed
**Created**: 2023-03-21T15:58:07+00:00
**Closed**: 2024-04-10T01:08:04+00:00
**Comments**: 4
**Labels**: enhancement, need more info, stale

### Description

Hi everyone, I hacked together a python script to convert a model saved as GGML Q4_0 files back to Pytorch checkpoint for further consumption/training/finetuning using HuggingFace's Transformer package and/or Pytorch/Pytorch Lightning. If there are interests to do this, please comment of drop a like. I will post the code or create a pull request if people need this.

---

## Issue #N/A: Interactive mode in Python?

**Link**: https://github.com/ggml-org/llama.cpp/issues/357
**State**: closed
**Created**: 2023-03-21T15:40:27+00:00
**Closed**: 2023-03-21T16:10:07+00:00
**Comments**: 0

### Description

Hello, I have a question. How can i use LLaMa in an interactive mode (i.e. as a chat) in Python, and is it possible at all? So that he would not just generate text, but it would be possible to somehow communicate

---

## Issue #N/A: Improve the Chat Mode with some tricks and considerations

**Link**: https://github.com/ggml-org/llama.cpp/issues/353
**State**: closed
**Created**: 2023-03-21T13:44:45+00:00
**Closed**: 2024-04-10T01:08:05+00:00
**Comments**: 13
**Labels**: enhancement, stale

### Description

I noticed that often the interactive mode (used as a chat with for example the `chat-with-bob.txt` initial prompt) fails due to **LLaMA trying to escape** the chat (mainly with the expression `\end{code}`).

To avoid that it is possible to pass the argument `-r "\end{code}"` but since the expression doesn't get removed from the chat, LLaMA interprets it as the end of the chat, and all the previous dialog context (including what's inside `chat-with-bob.txt`) gets lost and LLaMA starts to behave weirdly.

So it would be cool to have a `--chat` option that **detects expressions** like `\end{code}`, removing them from the context and **forcefully appending** `User:` at the end of the chat so that it can continue without losing context.

---

## Issue #N/A: Go bindings

**Link**: https://github.com/ggml-org/llama.cpp/issues/351
**State**: closed
**Created**: 2023-03-21T10:55:32+00:00
**Closed**: 2023-07-28T19:37:09+00:00
**Comments**: 6
**Labels**: enhancement

### Description

Hey :wave: , awesome project!

I'd like to help here, I've did the bindings for go to be used as a library. there are of course some adaptations that I had to run into to make it possible. I'm wondering, there are any plans for golang bindings? Generally speaking there seems to be genuine interest into running this as API too https://github.com/antimatter15/alpaca.cpp/issues/86 , which I worked on here: https://github.com/go-skynet/llama-cli .

I'm happy to contribute my work which currently is in https://github.com/go-skynet/llama, I'm playing with this sparely - I've been using this mainly with alpaca, and could manage to run also `13b` and `30b` models. I'll update it later to the latest changes so it is on pair with master. 



---

## Issue #N/A: Garbage output

**Link**: https://github.com/ggml-org/llama.cpp/issues/344
**State**: closed
**Created**: 2023-03-21T01:06:21+00:00
**Closed**: 2023-03-30T23:27:12+00:00
**Comments**: 4
**Labels**: bug, generation quality

### Description

Installed 7B model on win 11.

```
PS D:\Projects\llama.cpp>  ./main -m ./models/7B/ggml-model-q4_0.bin -p "Building a website can be done in 10 simple steps:" -n 512         
main: seed = 1679360633
llama_model_load: loading model from './models/7B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 4096
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 32
llama_model_load: n_layer = 32
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 11008
llama_model_load: n_parts = 1
llama_model_load: ggml ctx size = 4529.34 MB
llama_model_load: memory_size =   512.00 MB, n_mem = 16384
llama_model_load: loading model part 1/1 from './models/7B/ggml-model-q4_0.bin'
llama_model_load: .................... done
llama_model_load: model size =  2328.05 MB / num tensors = 163

system_info: n_threads = 4 / 20 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 1 | 

[... truncated for brevity ...]

---

## Issue #N/A: llama.exe will instantly exit with no text or error msg 

**Link**: https://github.com/ggml-org/llama.cpp/issues/343
**State**: closed
**Created**: 2023-03-21T00:09:14+00:00
**Closed**: 2023-04-16T10:49:17+00:00
**Comments**: 2
**Labels**: need more info

### Description

llama.exe --help 
will produce the same blank line with no text and will exit.
On windows 10 compiled with cmake

**Please help.**

---

## Issue #N/A: -f option seems to not work

**Link**: https://github.com/ggml-org/llama.cpp/issues/342
**State**: closed
**Created**: 2023-03-20T23:20:20+00:00
**Closed**: 2023-03-27T19:36:41+00:00
**Comments**: 4
**Labels**: need more info, generation quality

### Description

It either doesn't work for importing a prompt, or I don't know what the file format is suppose to be.   I put this in a file.

My name is Greg.\
What is my name?

When I run chat with the -f pointing to the file, it doesn't answer the question, and doesn't know the name I placed in the file.

---

## Issue #N/A: Neural Engine Support

**Link**: https://github.com/ggml-org/llama.cpp/issues/334
**State**: closed
**Created**: 2023-03-20T17:11:52+00:00
**Closed**: 2023-03-20T18:33:49+00:00
**Comments**: 0

### Description

Would be cool to be able to lean on the neural engine. Even if it wasn't much faster, it'd still be more energy efficient I believe.

---

## Issue #N/A: Improving the repetition penalty

**Link**: https://github.com/ggml-org/llama.cpp/issues/331
**State**: closed
**Created**: 2023-03-20T15:43:12+00:00
**Closed**: 2023-09-14T13:23:49+00:00
**Comments**: 12
**Labels**: enhancement, generation quality

### Description

129c7d1e (#20) added a repetition penalty that prevent the model to run into loops.

Here are a few suggestions for possible enhancements:

 * One issue with the interactive mode is that the repetition penalty is affecting the anti-prompt and response prefix, causing the model to generate unnecessarily long responses. One solution could be to exclude these tokens from the penalty,
 * It is possible to exempt or reduce the penalty for stop words, punctuation characters, and newlines; maybe applying a frequency-based penalty instead,
 * Using an exponential decay, such that recent tokens are more penalized than older ones, causing less issues with large `repeat_last_n`  windows,
 * Token repetition is an approximation of sub-strings or word repetition, but it seems difficult to do otherwise without backtracking the inference.

---

## Issue #N/A: invalid model file './models/ggml-alpaca-7b-q4.bin' (too old, regenerate your model files!)

**Link**: https://github.com/ggml-org/llama.cpp/issues/329
**State**: closed
**Created**: 2023-03-20T14:56:00+00:00
**Closed**: 2023-03-20T15:32:21+00:00
**Comments**: 7
**Labels**: need more info, model

### Description

Hi, I have encounter the above problem when running the alpaca model. I download the model from the link "https://gateway.estuary.tech/gw/ipfs/QmQ1bf2BTnYxq73MFJWu1B7bQ2UD6qG7D7YDCxhTndVkPC" which is one of the three options from the readme. Should I download the model from somewhere else? 

---

## Issue #N/A: I hope the script file download-pth.py can support downloading the alpaca-lora model.

**Link**: https://github.com/ggml-org/llama.cpp/issues/326
**State**: closed
**Created**: 2023-03-20T13:28:09+00:00
**Closed**: 2023-03-20T17:33:57+00:00
**Comments**: 0
**Labels**: enhancement, model

### Description

the script file download-pth.py only support download origin llama model, I hope it can downloading alpaca-lora

---

## Issue #N/A: Breaking change of models since PR #252

**Link**: https://github.com/ggml-org/llama.cpp/issues/324
**State**: closed
**Created**: 2023-03-20T12:48:57+00:00
**Closed**: 2023-05-09T21:01:11+00:00
**Comments**: 24
**Labels**: bug, model

### Description

After the PR #252, all base models need to be converted new.

For me, this is a big breaking change. The LoRa and/or Alpaca fine-tuned models are not compatible anymore.
Reconverting is not possible.

I see from the PR, that the tokenizer scores are written into the model.
Would it make sense to write the tokenizer scores into a seperate file to stay compatible with the (old) models?
The question then arrises, if 
1. by loading the model the scoring file will be checked of existense and the sentencepiece tokenizer will be used, or
2. the user can decide which tokenizer to use.

What you think?

---

## Issue #N/A: High performance API

**Link**: https://github.com/ggml-org/llama.cpp/issues/321
**State**: closed
**Created**: 2023-03-20T11:34:40+00:00
**Closed**: 2023-03-20T19:22:42+00:00
**Comments**: 8
**Labels**: duplicate, enhancement

### Description

Hey!

I'd love to see this project being able to be used through some TCP socket with a very optimized protocol. One it may make use of something like protobuf, or even grpc.
I think everyone agrees HTTP would be a complete overkill specially for a project focused on high performance. :laughing: 

Thanks
Niansa

---

## Issue #N/A: segmentation fault Alpaca

**Link**: https://github.com/ggml-org/llama.cpp/issues/317
**State**: closed
**Created**: 2023-03-20T09:56:07+00:00
**Closed**: 2023-04-17T07:12:17+00:00
**Comments**: 35
**Labels**: hardware

### Description

Hello, 
I've tried out the Aplaca model but after a while there comes an error I believe stating: "zsh: segmentation fault  ./main -m ./models/alpaca/ggml-alpaca-7b-q4.bin --color -f  -ins". 
Thanks.

Code: 
./main -m ./models/alpaca/ggml-alpaca-7b-q4.bin --color -f ./prompts/alpaca.txt -ins
main: seed = 1679305614
llama_model_load: loading model from './models/alpaca/ggml-alpaca-7b-q4.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 4096
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 32
llama_model_load: n_layer = 32
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 11008
llama_model_load: n_parts = 1
llama_model_load: ggml ctx size = 4529.34 MB
llama_model_load: memory_size =   512.00 MB, n_mem = 16384
llama_model_load: loading model part 1/1 from './models/alpaca/ggml-alpaca-7b-q4.bin'
llama_model_load: .................................... don

[... truncated for brevity ...]

---

## Issue #N/A: No module named 'tqdm' WSL

**Link**: https://github.com/ggml-org/llama.cpp/issues/316
**State**: closed
**Created**: 2023-03-20T09:39:47+00:00
**Closed**: 2023-03-20T13:20:43+00:00
**Comments**: 1

### Description

I am trying docker setup and for some reason its not working.
I have tqdm on my system.
```
docker run -v /llama/models:/models ghcr.io/ggerganov/llama.cpp:full --all-in-one "/models/" 7B
Downloading model...
Traceback (most recent call last):
  File "/app/./download-pth.py", line 3, in <module>
    from tqdm import tqdm
ModuleNotFoundError: No module named 'tqdm'
```
```
pip3 install tqdm
Requirement already satisfied: tqdm in /home/whoami/.local/lib/python3.8/site-packages (4.62.3)
```


---

## Issue #N/A: Add OpenBSD support

**Link**: https://github.com/ggml-org/llama.cpp/issues/313
**State**: closed
**Created**: 2023-03-20T02:25:38+00:00
**Closed**: 2023-03-21T15:50:12+00:00
**Comments**: 3
**Labels**: enhancement, 🦙., build

### Description

This patch adds OpenBSD support, thanks.
[patch-llama.cpp.txt](https://github.com/ggerganov/llama.cpp/files/11013172/patch-llama.cpp.txt)


---

## Issue #N/A: Docker fails due to missing tqdm

**Link**: https://github.com/ggml-org/llama.cpp/issues/310
**State**: closed
**Created**: 2023-03-19T23:02:47+00:00
**Closed**: 2023-03-20T09:01:50+00:00
**Comments**: 2
**Labels**: duplicate

### Description

```
docker run -v /llama/models:/models ghcr.io/ggerganov/llama.cpp:full --all-in-one "/models/" 65B
```
```
Unable to find image 'ghcr.io/ggerganov/llama.cpp:full' locally
full: Pulling from ggerganov/llama.cpp
2ab09b027e7f: Pull complete
abc582ff34c3: Pull complete
474c54188cc5: Pull complete
90dde168a635: Pull complete
4baa98a3bbd6: Pull complete
40709b48f1dd: Pull complete
Digest: sha256:0e26a42b34ad42f285a4327fbe099674137b119e6efea07345a7c17ab8a4b13e
Status: Downloaded newer image for ghcr.io/ggerganov/llama.cpp:full
Downloading model...
Traceback (most recent call last):
  File "/app/./download-pth.py", line 3, in <module>
    from tqdm import tqdm
ModuleNotFoundError: No module named 'tqdm'
```

---

## Issue #N/A: Is the --ignore-eos flag redundant?

**Link**: https://github.com/ggml-org/llama.cpp/issues/309
**State**: closed
**Created**: 2023-03-19T23:02:33+00:00
**Closed**: 2023-03-20T18:50:19+00:00
**Comments**: 12
**Labels**: enhancement, question

### Description

As per https://github.com/ggerganov/llama.cpp/blob/da5303c1ea68aa19db829c634f1e10d08d409680/main.cpp#L1066 the EOS flag in interactive mode simply causes `is_interacting` to switch on, and so it serves as a way to end the current series of tokens and wait for user input. Is there any reason to actually avoid sampling it in the first place then?

---

## Issue #N/A: alpaca.sh always terminates after running the prompt with current command

**Link**: https://github.com/ggml-org/llama.cpp/issues/307
**State**: closed
**Created**: 2023-03-19T22:13:44+00:00
**Closed**: 2023-07-28T19:36:14+00:00
**Comments**: 1
**Labels**: bug

### Description

Running `alpaca.sh` always terminates once the prompt is ran, see below.
I had to change the script because `-ins` doesn't seem to be supported (changed to just `-i`) but maybe I am doing something wrong.

```
main: seed = 1679263584
llama_model_load: loading model from './models/ggml-alpaca-7b-q4.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 4096
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 32
llama_model_load: n_layer = 32
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 11008
llama_model_load: n_parts = 1
llama_model_load: ggml ctx size = 4529.34 MB
llama_model_load: memory_size =   512.00 MB, n_mem = 16384
llama_model_load: loading model part 1/1 from './models/ggml-alpaca-7b-q4.bin'
llama_model_load: .................................... done
llama_model_load: model size =  4017.27 MB / num tensors = 291

system_info: n_threads = 7 / 1

[... truncated for brevity ...]

---

## Issue #N/A: Using non LoRA Alpaca model

**Link**: https://github.com/ggml-org/llama.cpp/issues/303
**State**: closed
**Created**: 2023-03-19T20:03:48+00:00
**Closed**: 2023-07-28T19:35:59+00:00
**Comments**: 11
**Labels**: question, model

### Description

The following repo contains a recreation of the original weights for Alpaca, without using LoRA. How could we use that model with this project? https://github.com/pointnetwork/point-alpaca
Thanks a bunch!

---

## Issue #N/A: Improve Alpaca integration to match it's trained prompt syntax

**Link**: https://github.com/ggml-org/llama.cpp/issues/302
**State**: closed
**Created**: 2023-03-19T19:17:47+00:00
**Closed**: 2023-07-28T19:35:22+00:00
**Comments**: 12
**Labels**: enhancement, help wanted, high priority

### Description

Alpaca LoRA model was trained on the same dataset as original [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html).

However, this dataset contains two types of instructions, namely:
- instructions with input
- instructions without input

For more details about the instructions format see details [here.](https://github.com/tatsu-lab/stanford_alpaca#data-release)

In case of instructions such as text summarization, instruction alone only "explain" the task, while the text to be summarized is inserted into the "input" part of the prompt.

Current integration of alpaca in `llama.cpp` mimics the current integration in [alpaca.cpp](https://github.com/antimatter15/alpaca.cpp) which completely omits the "instructions with input" type of instructions. This may have significant impact on the model performance using task which were trained to be used in "instruction with input" prompt syntax when using just ordinary "instruction without input" prompt syntax instead.

I

[... truncated for brevity ...]

---

## Issue #N/A: Reverse prompt is sometimes ignored.

**Link**: https://github.com/ggml-org/llama.cpp/issues/292
**State**: closed
**Created**: 2023-03-19T12:15:09+00:00
**Closed**: 2023-03-21T16:28:12+00:00
**Comments**: 6
**Labels**: bug

### Description

I haven't found a consistent pattern to reproduce this, but sometimes the model will continue outputting text even after it has printed the reverse prompt. If colors are enabled, they will change as if the new text was user input, but it is generated by the model. After this happen it might or might not revert to its proper behavior once it finds the reverse prompt again.

I have noticed the color change doesn't always happen right on the prompt, but sometimes it happens a few words before it. I don't know enough about how this code works yet to speculate, but in case this has something to do with parallelism, I'm using `-t 16`.

---

## Issue #N/A: LLaMA.cpp returns just some weirdo texts with any model size

**Link**: https://github.com/ggml-org/llama.cpp/issues/291
**State**: closed
**Created**: 2023-03-19T12:05:42+00:00
**Closed**: 2023-03-19T16:57:26+00:00
**Comments**: 3
**Labels**: need more info, generation quality

### Description

I'm grokking with LLaMA.cpp on M1 laptop with 32GB RAM. Somehow the inference is broken for me.

Like I'm expecting something reasonable for simple prompt I've got from original LLaMA examples:

`SQL code to create a table, that will keep CD albums data, such as album name and track\n\\begin{code}\n`

And LLaMA.cpp returns just some weirdo texts with any model size (7B, 13B, 30B quantised down to 4bit).

What's the reason here?

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

## Issue #N/A: Docker “--all-in-one” fails with ModuleNotFoundError: No module named ‘tqdm’

**Link**: https://github.com/ggml-org/llama.cpp/issues/289
**State**: closed
**Created**: 2023-03-19T10:51:52+00:00
**Closed**: 2023-03-20T08:24:13+00:00
**Comments**: 7
**Labels**: bug, duplicate, build

### Description

On Win 10
```
>  docker run -v /llama/models:/models ghcr.io/ggerganov/llama.cpp:full –all-in-one “/models/” 7B
Downloading model…
Traceback (most recent call last):
  File “/app/./download-pth.py”, line 3, in <module>
    from tqdm import tqdm
ModuleNotFoundError: No module named ‘tqdm’
```

---

## Issue #N/A: RISC-V (TH1520&D1) benchmark and hack for <1GB DDR device

**Link**: https://github.com/ggml-org/llama.cpp/issues/288
**State**: closed
**Created**: 2023-03-19T10:14:34+00:00
**Closed**: 2024-04-10T01:08:06+00:00
**Comments**: 8
**Labels**: enhancement, need more info, hardware, stale

### Description

Hi, 
   Just test on RISC-V board: 
   4xC910 2.0G TH1520 LicheePi4A (https://sipeed.com/licheepi4a)  with 16GB LPDDR4X.
   about 6s/token without any instruction acceleration, and it should be <5s/token when boost to 2.5GHz.

```
llama_model_load: ggml ctx size = 668.34 MB
llama_model_load: memory_size =   512.00 MB, n_mem = 16384
llama_model_load: .................................... done
llama_model_load: model size =  4017.27 MB / num tensors = 291

system_info: n_threads = 4 / 4 | AVX = 0 | AVX2 = 0 | AVX512 = 0 | FMA = 0 | NEON = 0 | ARM_FMA = 0 | F16C = 0 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 0 | VSX = 0 | 

main: prompt: 'They'
main: number of tokens in prompt = 2
     1 -> ''
 15597 -> 'They'

sampling parameters: temp = 0.800000, top_k = 40, top_p = 0.950000, repeat_last_n = 64, repeat_penalty = 1.300000


They are now available for sale at the cost of Rs 20,5

main: mem per token = 14368644 bytes
main:     load time =    91.25 ms
main:   

[... truncated for brevity ...]

---

## Issue #N/A: Commit c9f670a (Implement non-greedy tokenizer that tries to maximize token lengths) breaks llama?

**Link**: https://github.com/ggml-org/llama.cpp/issues/280
**State**: closed
**Created**: 2023-03-19T04:00:14+00:00
**Closed**: 2023-03-21T10:15:24+00:00
**Comments**: 7
**Labels**: bug, need more info

### Description

Old version:

```
.\build\Release\llama.exe -m C:\...\models\30B\ggml-model-q4_0.bin -t 10 -n 256 --seed 100 --temp 0.2 -p "list all US states in alphabetical order:"
output: Alabama, Alaska, Arizona, Arkansas, California, Colorado, Connecticut, Delaware Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada New Hampshire New Jersey New Mexico New York North Carolina North Dakota Ohio Oklahoma Oregon Pennsylvania Rhode Island South Carolina Tennessee Texas Utah Vermont Virginia Washington West Virginia Wisconsin Wyoming ... (keeps repeating)
```

```
.\build\Release\llama.exe -m C:\...\models\30B\ggml-model-q4_0.bin -t 10 -n 256 --seed 200 --temp 0.2 -p "list all US states in alphabetical order:"
output: Alabama, Alaska, Arizona, Arkansas, California, Colorado, Connecticut, Delaware Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine

[... truncated for brevity ...]

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

## Issue #N/A: the program always terminate itself for no reasons 

**Link**: https://github.com/ggml-org/llama.cpp/issues/275
**State**: closed
**Created**: 2023-03-18T23:51:01+00:00
**Closed**: 2023-03-19T01:28:29+00:00
**Comments**: 4

### Description

No description provided.

---

## Issue #N/A: A typo in readme?

**Link**: https://github.com/ggml-org/llama.cpp/issues/271
**State**: closed
**Created**: 2023-03-18T21:36:27+00:00
**Closed**: 2023-03-18T22:18:06+00:00
**Comments**: 1

### Description

<html><body>
<!--StartFragment--><h3 tabindex="-1" dir="auto">Memory/Disk Requirements</h3>
<p dir="auto">As the models are currently fully loaded into memory, you will need adequate disk space to save them
and sufficient RAM to load them. At the moment, memory and disk requirements are the same.</p>


model | original size | quantized size (4-bit)
-- | -- | --
7B | 13 GB | 3.9 GB
15B | 24 GB | 7.8 GB
30B | 60 GB | 19.5 GB
65B | 120 GB | 38.5 GB

<!--EndFragment-->
</body>
</html>

Isn't it 13B and not 15B?

---

## Issue #N/A: Prevent user from setting a context size that is too big

**Link**: https://github.com/ggml-org/llama.cpp/issues/266
**State**: closed
**Created**: 2023-03-18T15:11:33+00:00
**Closed**: 2023-03-19T10:33:41+00:00
**Comments**: 10

### Description

Hey!

I tasked the 30B model to write a little story... it worked really well until some point where it went off rails from one line to the next, suddenly talking about some girl and stuff that has nothing to do with the rest:

```
The way out of me that started looking at them. It'ould be lying there was standing near-the first time what could see an older than the girl had held they looked like it, and just how hard. In order I wasn't really when my hands on his head down to myself in front seat and the car door were with me before you.
“I realy as she staring that laying to a moment of him. "It was so lying next to about two, but it looked at her eyes had already when there looking for holding my hand from what I'with his head was on both shoulders. And not through and suddenly, he realized 212.
I couldn’t with the car seat, in fronted again because of one. The second, so that didn'sit seems like a young girl sitting me when "We weren near. But I started. 'mom. Withered and t

[... truncated for brevity ...]

---

## Issue #N/A: Docker container repository permissions are causing access denied error

**Link**: https://github.com/ggml-org/llama.cpp/issues/263
**State**: closed
**Created**: 2023-03-18T13:08:01+00:00
**Closed**: 2023-03-18T13:21:02+00:00
**Comments**: 2

### Description

```
root@unraid:~# docker run -v /mnt/user/appdata/llama/models:/models ghcr.io/ggerganov/llama.cpp:full --all-in-one "/models/" 7BUnable to find image 'ghcr.io/ggerganov/llama.cpp:full' locally
docker: Error response from daemon: Head "https://ghcr.io/v2/ggerganov/llama.cpp/manifests/full": denied.
See 'docker run --help'.

```

Cannot pull the containers due to permission issues.

Thanks

---

## Issue #N/A: Error while converting to ggml.py format

**Link**: https://github.com/ggml-org/llama.cpp/issues/260
**State**: closed
**Created**: 2023-03-18T12:02:31+00:00
**Closed**: 2023-04-14T13:13:30+00:00
**Comments**: 1
**Labels**: need more info

### Description

After running the command: "python3 convert-pth-to-ggml.py /Users/tanish.shah/llama.cpp/models/7B/ 1"
Error with sentencepiece:

```
Traceback (most recent call last):
  File "/Users/tanish.shah/llama.cpp/convert-pth-to-ggml.py", line 75, in <module>
    tokenizer = sentencepiece.SentencePieceProcessor(fname_tokenizer)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/tanish.shah/llama.cpp/env/lib/python3.11/site-packages/sentencepiece/__init__.py", line 447, in Init
    self.Load(model_file=model_file, model_proto=model_proto)
  File "/Users/tanish.shah/llama.cpp/env/lib/python3.11/site-packages/sentencepiece/__init__.py", line 905, in Load
    return self.LoadFromFile(model_file)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/tanish.shah/llama.cpp/env/lib/python3.11/site-packages/sentencepiece/__init__.py", line 310, in LoadFromFile
    return _sentencepiece.SentencePieceProcessor_LoadFromFile(self, arg)
           ^^^^^^^^^^

[... truncated for brevity ...]

---

## Issue #N/A: Is it possible to run the llama on an AMD graphics card?

**Link**: https://github.com/ggml-org/llama.cpp/issues/259
**State**: closed
**Created**: 2023-03-18T08:43:00+00:00
**Closed**: 2023-03-18T11:16:59+00:00
**Comments**: 2

### Description

No description provided.

---

## Issue #N/A: Specify the version of `python3` to use in repo docs

**Link**: https://github.com/ggml-org/llama.cpp/issues/258
**State**: closed
**Created**: 2023-03-18T07:55:42+00:00
**Closed**: 2023-03-18T21:22:52+00:00
**Comments**: 2

### Description

The default `python3` on my system is `3.11.2` for which `sentencepiece` [does not install](https://github.com/google/sentencepiece/issues/378).

Where Python toolchains in general seem fragile to minor version differences, specifying the exact expected Python version for newcomers (like me) will reduce frustration and make this work more accessible.

Specifying `python3.9` on CLI worked for me; unsure if this was the intended version.

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

## Issue #N/A: Interactive mode doesn't work

**Link**: https://github.com/ggml-org/llama.cpp/issues/255
**State**: closed
**Created**: 2023-03-18T06:57:32+00:00
**Closed**: 2023-03-18T09:05:54+00:00
**Comments**: 2

### Description

Hello, 
I wanted to test the interactive mode but it just doesn't work for me, the AI on its own with one promt gives me an output but with the command for a promt for the user it doesn't work and I just get "dquote" until I exit the program. 
Thank you for your help!

<img width="1728" alt="Bildschirm­foto 2023-03-18 um 07 53 36" src="https://user-images.githubusercontent.com/90244617/226090388-9e64b610-38f8-4800-a18a-3b8b3563a0c2.png">


---

## Issue #N/A: How to use it in Python

**Link**: https://github.com/ggml-org/llama.cpp/issues/253
**State**: closed
**Created**: 2023-03-18T04:46:55+00:00
**Closed**: 2023-03-18T04:58:21+00:00
**Comments**: 2

### Description

How to use this in my python code?

---

## Issue #N/A: 65B quantized for CPU

**Link**: https://github.com/ggml-org/llama.cpp/issues/251
**State**: closed
**Created**: 2023-03-18T02:47:36+00:00
**Closed**: 2023-03-18T05:59:53+00:00
**Comments**: 3

### Description

Is there any way to run the 65B model on the CPU quantized for 4 bit? I saw that it's about 40 gigs for RAM usage when quantized.

How much RAM is required to quantize the 65B model? I'm not sure I have enough RAM to quantize myself, anyone have the model files for the quantized output for the 65B model for CPU? I've only found the [quantized GPU files](https://huggingface.co/decapoda-research) so far.

---

## Issue #N/A: Batch size affects model's output

**Link**: https://github.com/ggml-org/llama.cpp/issues/249
**State**: closed
**Created**: 2023-03-18T01:03:42+00:00
**Closed**: 2023-07-28T19:34:07+00:00
**Comments**: 10
**Labels**: bug, generation quality

### Description

I was tinkering with the code and made the following change in `line 977, main.cpp` (as it seemed wrong to me):
*from*
```C
if (embd.size() > params.n_batch) {
       break;
}
```
*to*
```C
if (embd.size() >= params.n_batch) {
       break;
}
```

The model's (13B) outputs suddenly changed. Reverted changes and tried to play with the `batch_size` parameter, it really does affect the output.

Not sure if it's expected behaviour. As far as I understand it shouldn't be the case. A bug? Different batch sizes have different evaluation results (rounding error)?

---

## Issue #N/A: Rust Bindings

**Link**: https://github.com/ggml-org/llama.cpp/issues/248
**State**: closed
**Created**: 2023-03-18T00:33:09+00:00
**Closed**: 2023-03-18T11:10:12+00:00
**Comments**: 6

### Description

I saw that similar bindings were created for whisper.cpp in whisper-rs (https://github.com/tazz4843/whisper-rs), and I think it would be great to have similar bindings for llama.cpp as well.

As a Rust developer, I would use these to create an inference and embeeddings HTTP server and eventual create a Langchain binding for this (https://github.com/hwchase17/langchain/issues/1473).

I'd be willing to help with the bindings in any way I can. I'll be using `whisper-rs` as the template. 

---

## Issue #N/A: How to use ggml for Flan-T5

**Link**: https://github.com/ggml-org/llama.cpp/issues/247
**State**: closed
**Created**: 2023-03-17T22:38:08+00:00
**Closed**: 2024-04-14T01:06:18+00:00
**Comments**: 36
**Labels**: enhancement, model, generation quality, stale

### Description

@ggerganov Thanks for sharing llama.cpp. As usual, great work.

Question rather than issue.  How difficult would it be to make ggml.c work for a Flan checkpoint, like T5-xl/UL2, then quantized?

Would love to be able to have those models run on a browser, much like what you did with whisper.cpp wasm.

Thanks again.  (I can move this post somewhere else if you prefer since it's not technically about Llama.  Just let me know where.)

---

## Issue #N/A: Add instructions for using Alpaca

**Link**: https://github.com/ggml-org/llama.cpp/issues/240
**State**: closed
**Created**: 2023-03-17T15:52:57+00:00
**Closed**: 2023-03-19T16:51:05+00:00
**Comments**: 8
**Labels**: enhancement, good first issue

### Description

See the work here: https://github.com/antimatter15/alpaca.cpp

There is no new functionality added, just a few hardcoded parameters in `chat.cpp`.
Instead of adding separate `chat` program, we should have an `alpaca.py` script that runs `main` with the respective parameters, so the user can simply run `./alpaca.py` on the terminal.
It is a good time to start collecting prompts, so create a few useful Alpaca instruction prompts and place them in a `prompts` folder in the source tree. Make the `alpaca.py` script use one of them by default. Add option to change.

Add short instructions for using the `alpaca.py` for various tasks (translation, answering, .. whatever is popular) in the README and reference the `alpaca.cpp` repo for downloading the models.

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

## Issue #N/A: Document check sums of models so that we can confirm issues are not caused by bad downloads or conversion

**Link**: https://github.com/ggml-org/llama.cpp/issues/238
**State**: closed
**Created**: 2023-03-17T12:50:44+00:00
**Closed**: 2023-05-02T13:41:32+00:00
**Comments**: 9
**Labels**: documentation, model

### Description

Can someone please confirm the following md5 sums are correct?  I regenerated them with the latest code.

```
$ md5sum ./models/*/*.pth | sort -k 2,2
0804c42ca65584f50234a86d71e6916a  ./models/13B/consolidated.00.pth
016017be6040da87604f77703b92f2bc  ./models/13B/consolidated.01.pth
f856e9d99c30855d6ead4d00cc3a5573  ./models/30B/consolidated.00.pth
d9dbfbea61309dc1e087f5081e98331a  ./models/30B/consolidated.01.pth
2b2bed47912ceb828c0a37aac4b99073  ./models/30B/consolidated.02.pth
ea0405cdb5bc638fee12de614f729ebc  ./models/30B/consolidated.03.pth
9deae67e2e7b5ccfb2c738f390c00854  ./models/65B/consolidated.00.pth
0c4b00c30460c3818bd184ee949079ee  ./models/65B/consolidated.01.pth
847194df776dd38f8ae9ddcede8829a1  ./models/65B/consolidated.02.pth
3b6c8adcb5654fd36abab3206b46a0f1  ./models/65B/consolidated.03.pth
68d61d1242597ad92616ec31b8cb6b4c  ./models/65B/consolidated.04.pth
7f71259eaee2b906aa405d8edf39925f  ./models/65B/consolidated.05.pth
0574e26b6891ab2cb0df7340d773fe

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

## Issue #N/A: Thanks for contributing to Machine Learning and AI with this repository!

**Link**: https://github.com/ggml-org/llama.cpp/issues/233
**State**: closed
**Created**: 2023-03-17T09:15:17+00:00
**Closed**: 2023-03-17T10:46:03+00:00
**Comments**: 1

### Description

I just want to say Thank You!

I got your solution working and I am happy to confirm that it is working as intended. The quality of text produced is probably a bit lower compared to text produced with the full weights, but the quantized weights saves a lot of space and maybe some processing time.

I would like to see the solution rewritten as C# to better understand it, since I am using C# at work but do not use C++ och C since I was studying programming in the 90s. I do understand this is not the main goal of this project, but in case someone else wants to do this or have already done this - please leave me a note.

Thanks again for this lovely work!

---

## Issue #N/A: Study how LM Evaluation Harness works and try to implement it

**Link**: https://github.com/ggml-org/llama.cpp/issues/231
**State**: open
**Created**: 2023-03-17T08:32:33+00:00
**Comments**: 9
**Labels**: enhancement, help wanted, high priority, generation quality, research 🔬

### Description

Update 10 Apr 2024: https://github.com/ggerganov/llama.cpp/issues/231#issuecomment-2047759312

---

It would be great to start doing this kind of quantitative analysis of `ggml`-based inference:

https://bellard.org/ts_server/

It looks like Fabrice evaluates the models using something called LM Evaluation Harness:

https://github.com/EleutherAI/lm-evaluation-harness

I have no idea what this is yet, but would be nice to study it and try to integrate it here and in other `ggml`-based projects.
This will be very important step needed to estimate the quality of the generated output and see if we are on the right track.

---

## Issue #N/A: 7B/13B: Inability to write certain words/names with smaller models

**Link**: https://github.com/ggml-org/llama.cpp/issues/228
**State**: closed
**Created**: 2023-03-17T07:46:08+00:00
**Closed**: 2023-03-17T08:31:37+00:00
**Comments**: 2
**Labels**: question, model

### Description

Hey!

When I attempted to tell the bot in a chat-like prompt that my name is "Nils", I ran into an issue where the bot kept interpreting my name as "Nil" instead. I then noticed further issues with the word "guild" and some other words too.
Is this a bug or to be expected? It does not happen on 30B, I couldn't give 65B a try.

Thanks
Niansa

---

## Issue #N/A: convert-pth-to-ggml.py how to handle torch.view_as_complex

**Link**: https://github.com/ggml-org/llama.cpp/issues/225
**State**: closed
**Created**: 2023-03-17T05:28:02+00:00
**Closed**: 2023-04-10T08:11:28+00:00
**Comments**: 3
**Labels**: question, need more info

### Description

llama code block include view_as_real: https://github.com/facebookresearch/llama/blob/main/llama/model.py#L68

how to convert-pth-to-ggml.py handle this part of weight

---

## Issue #N/A: How do I get input embeddings?

**Link**: https://github.com/ggml-org/llama.cpp/issues/224
**State**: closed
**Created**: 2023-03-17T04:59:12+00:00
**Closed**: 2023-04-16T09:25:29+00:00
**Comments**: 9
**Labels**: question

### Description

I am trying to output just the sentence embedding for a given input, instead of any new generated text. I think this should be rather straightforward but figured someone more familiar with the codebase could help me.

I just want to return the sentence embedding vector and stop execution for a given input.

I am almost sure the place where I want to make the embedding is right after `norm` but before `lm_head`, and I think they will be in `inpL` if I run 

```
ggml_build_forward_expand(&gf, inpL);
ggml_graph_compute       (ctx0, &gf);
``` 
However I am confused by the struct and not sure how to get the sentence embedding itself. I understand it should be some index of ggml_get_data(inpL), but don't get which index, and that is why I come to you. Would anyone lend me a hand?

---

## Issue #N/A: [QUESTION] data type

**Link**: https://github.com/ggml-org/llama.cpp/issues/223
**State**: closed
**Created**: 2023-03-17T03:51:39+00:00
**Closed**: 2023-03-17T04:02:29+00:00
**Comments**: 8

### Description

I see that it says using float16 float32 mixed precision, but as we are talking about characters, shouldn't it uses char8 ?

---

## Issue #N/A: Can this code base be extended to support other transformer-based LLMs such as Pythia or its instruction-tuned version Open Assistant?

**Link**: https://github.com/ggml-org/llama.cpp/issues/219
**State**: closed
**Created**: 2023-03-17T01:46:50+00:00
**Closed**: 2023-07-28T19:32:44+00:00
**Comments**: 8
**Labels**: enhancement, question, model

### Description

No description provided.

---

## Issue #N/A: GPU instead CPU?

**Link**: https://github.com/ggml-org/llama.cpp/issues/214
**State**: closed
**Created**: 2023-03-16T21:51:51+00:00
**Closed**: 2023-03-16T22:06:00+00:00
**Comments**: 5

### Description

How can we use GPU instead of CPU? My processor is pretty weak. I don't have a macbook or a very powerful pc. the desire to run a model on CUDA cores. Thanks

---

## Issue #N/A: Stops generating after about 200 characters

**Link**: https://github.com/ggml-org/llama.cpp/issues/212
**State**: closed
**Created**: 2023-03-16T19:53:41+00:00
**Closed**: 2023-03-16T19:59:53+00:00
**Comments**: 2

### Description

I have tried modifying the -n (number of tokens to predict) but it always stops generating after the same amount of time. Is there any way to stop this happening? It seems to be intended behavior since it shows in the README screenshots

---

## Issue #N/A: Cannot generate more than 500 words

**Link**: https://github.com/ggml-org/llama.cpp/issues/210
**State**: closed
**Created**: 2023-03-16T16:18:36+00:00
**Closed**: 2023-03-16T16:25:39+00:00
**Comments**: 2
**Labels**: duplicate, enhancement

### Description

The model doesn't seem to be able to return more than 500 words regardless of how big the number of tokens is specified (I even tried specifically powers of 2 such as 4096 with no results), it always stops and leaves texts uncomplete. Is anyone having the same issue, or how can I increment the length of the output?

---

## Issue #N/A: Command line script usage

**Link**: https://github.com/ggml-org/llama.cpp/issues/209
**State**: closed
**Created**: 2023-03-16T15:57:54+00:00
**Closed**: 2023-03-16T16:27:50+00:00
**Comments**: 1
**Labels**: duplicate, wontfix

### Description

Hello, 

I was wondering if there was a command line flag for toggling the output of the debug messages, making the executable only output the text generated by the LLM (optionally with the original prompt). This would make the program much easier to call from other scripts.

Thanks for your time.

---

## Issue #N/A: making on linuxmint 21

**Link**: https://github.com/ggml-org/llama.cpp/issues/208
**State**: closed
**Created**: 2023-03-16T13:52:27+00:00
**Closed**: 2023-05-06T17:55:19+00:00
**Comments**: 2
**Labels**: duplicate, hardware, build

### Description

im running on bare metal nothing emulated

```
littlemac@littlemac:~$` git clone https://github.com/ggerganov/llama.cpp
Cloning into 'llama.cpp'...
remote: Enumerating objects: 283, done.
remote: Counting objects: 100% (283/283), done.
remote: Compressing objects: 100% (113/113), done.
remote: Total 283 (delta 180), reused 255 (delta 164), pack-reused 0
Receiving objects: 100% (283/283), 158.38 KiB | 609.00 KiB/s, done.
Resolving deltas: 100% (180/180), done.
cd littlemac@littlemac:~$ cd llama.cpp/
littlemac@littlemac:~/llama.cpp$ make
I llama.cpp build info: 
I UNAME_S:  Linux
I UNAME_P:  x86_64
I UNAME_M:  x86_64
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -msse3
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread
I LDFLAGS:  
I CC:       cc (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0
I CXX:      g++ (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0

cc  -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -msse3   -c ggml.c

[... truncated for brevity ...]

---

## Issue #N/A: Model runs but doesn't produce any output

**Link**: https://github.com/ggml-org/llama.cpp/issues/204
**State**: closed
**Created**: 2023-03-16T10:46:53+00:00
**Closed**: 2023-03-16T12:52:24+00:00
**Comments**: 5
**Labels**: need more info

### Description

I checked everything several times and quantized it, but both models do not output anything, in which mode I would not run them, the processor loads, but there is no output, no matter how long I wait
 input to the console also does not lead to anything

for ubuntu 22.04 8gb+15 swap (everything fits)


![Снимок экрана от 2023-03-16 11-42-21](https://user-images.githubusercontent.com/93709232/225592978-99f3c8a6-85a0-4606-a39d-6ddc1e334778.png)


---

## Issue #N/A: Alpaca and Llama

**Link**: https://github.com/ggml-org/llama.cpp/issues/203
**State**: closed
**Created**: 2023-03-16T10:09:19+00:00
**Closed**: 2023-03-16T11:42:32+00:00
**Comments**: 1
**Labels**: duplicate

### Description

Maybe it could work also on [Stanford's Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html), when they will release their weights.



---

## Issue #N/A: Reducing the time needed to reload a piece of text into the model by caching the state

**Link**: https://github.com/ggml-org/llama.cpp/issues/202
**State**: closed
**Created**: 2023-03-16T09:27:24+00:00
**Closed**: 2023-03-30T17:42:34+00:00
**Comments**: 9
**Labels**: enhancement

### Description

Hey!

Is it possible to add a way of dumping the current state into a file, so it can then be reloaded later? This would avoid the time needed to reload a long prompt over and over again.

Thanks
Niansa

---

## Issue #N/A: Running " python3 convert-pth-to-ggml.py models/7B/ 1 " and running out of RAM

**Link**: https://github.com/ggml-org/llama.cpp/issues/200
**State**: closed
**Created**: 2023-03-16T09:01:36+00:00
**Closed**: 2023-03-16T15:04:32+00:00
**Comments**: 8
**Labels**: wontfix, need more info, hardware

### Description

No description provided.

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

## Issue #N/A: Add the disk requirements

**Link**: https://github.com/ggml-org/llama.cpp/issues/195
**State**: closed
**Created**: 2023-03-16T03:23:50+00:00
**Closed**: 2023-03-16T11:54:44+00:00
**Comments**: 0
**Labels**: documentation, duplicate

### Description

Hi,

I found all the infos about the models:
https://cocktailpeanut.github.io/dalai/#/?id=_7b

You can put on readme the space requirements.

Thanks.

---

## Issue #N/A: Supported context window length for each model?

**Link**: https://github.com/ggml-org/llama.cpp/issues/194
**State**: closed
**Created**: 2023-03-16T02:27:23+00:00
**Closed**: 2023-03-24T10:34:27+00:00
**Comments**: 13
**Labels**: model, generation quality

### Description

what's the supported context window length for each model?

---

## Issue #N/A: new RMS norm PR bricks stuff

**Link**: https://github.com/ggml-org/llama.cpp/issues/190
**State**: closed
**Created**: 2023-03-15T23:02:41+00:00
**Closed**: 2023-03-15T23:29:27+00:00
**Comments**: 2

### Description

#187 

No output from the model after this was merged.

---

## Issue #N/A: How to? (install models)

**Link**: https://github.com/ggml-org/llama.cpp/issues/188
**State**: closed
**Created**: 2023-03-15T22:51:14+00:00
**Closed**: 2023-03-16T11:31:34+00:00
**Comments**: 15
**Labels**: question, 🦙.

### Description

Hi, i can't find the models
Can u tell me, how i can install?
(ls ./models 65B etc is not working)
*sorry, my english isn't good) 

---

## Issue #N/A: Python3 script  instead of bash

**Link**: https://github.com/ggml-org/llama.cpp/issues/184
**State**: closed
**Created**: 2023-03-15T21:54:35+00:00
**Closed**: 2023-03-19T19:53:05+00:00
**Comments**: 4
**Labels**: enhancement

### Description

```python
#!/usr/bin/env python3

import os
import sys

if not (len(sys.argv) == 2 and sys.argv[1] in ["7B", "13B", "30B", "65B"]):
    print(f"\nUsage: {sys.argv[0]} 7B|13B|30B|65B [--remove-f16]\n")
    sys.exit(1)

for i in os.listdir(f"models/{sys.argv[1]}"):
    if i.endswith("ggml-model-f16.bin"):
        os.system(f"./quantize {os.path.join('models', sys.argv[1], i)} {os.path.join('models', sys.argv[1], i.replace('f16', 'q4_0'))} 2")
        if len(sys.argv) == 3 and sys.argv[2] == "--remove-f16":
            os.remove(os.path.join('models', sys.argv[1], i))
``` 

---

## Issue #N/A: Question: can the conversation context be saved to disk and brought up again incase LLaMa crashes or there is a power failure?

**Link**: https://github.com/ggml-org/llama.cpp/issues/174
**State**: closed
**Created**: 2023-03-15T20:01:08+00:00
**Closed**: 2023-03-16T11:46:57+00:00
**Comments**: 1
**Labels**: duplicate

### Description

No description provided.

---

## Issue #N/A: Use RMSNorm

**Link**: https://github.com/ggml-org/llama.cpp/issues/173
**State**: closed
**Created**: 2023-03-15T19:05:29+00:00
**Closed**: 2023-03-19T15:31:53+00:00
**Comments**: 18
**Labels**: bug, help wanted, good first issue, high priority

### Description

The original paper, and the reference implementation [1] uses RMS norm. However, llama.cpp uses ggml_norm() which looks like Layer norm?

The differences between these may not be too obvious, because the mean is probably around 0. However, we should follow the original design.

[1] https://github.com/facebookresearch/llama/blob/main/llama/model.py

---

## Issue #N/A: Attempting to merge with alpaca-lora and its quantization

**Link**: https://github.com/ggml-org/llama.cpp/issues/172
**State**: closed
**Created**: 2023-03-15T18:16:30+00:00
**Closed**: 2023-07-28T19:32:24+00:00
**Comments**: 19
**Labels**: enhancement, help wanted

### Description

I was attempting to merge alpaca-lora from https://huggingface.co/tloen/alpaca-lora-7b and the original llama-7B from https://huggingface.co/decapoda-research/llama-7b-hf, also tried to quantize the model and run main file in llama.cpp.
The merge code is from https://github.com/clcarwin/alpaca-weight

It was almost successful until final phase to run the main file in llam.cpp. I had no problems with merge and quantization.

Then it raised an error like this:

llama_model_load: llama_model_load: unknown tensor 'model.embed_tokens.weight' in model file
main: failed to load model from './models/7B/ggml-model-q4_0.bin'

I will share my logs in my repository. The code I used in colab to merge and quantize the model is there too: https://github.com/taiyou2000/personal_experimant

I'm not machine learning expert and I have not checked entire llama.cpp code, but in my theory maybe the quantized model contains weights and some of them has names that main.cpp doesn't expect to see. A

[... truncated for brevity ...]

---

## Issue #N/A: [Proposal] "Stable" C API

**Link**: https://github.com/ggml-org/llama.cpp/issues/171
**State**: closed
**Created**: 2023-03-15T18:01:09+00:00
**Closed**: 2023-03-15T20:29:20+00:00
**Comments**: 4
**Labels**: duplicate, enhancement

### Description

I propose refactoring `main.cpp` into a library (`llama.cpp`, compiled to `llama.so`/`llama.a`/whatever) and making `main.cpp` a simple driver program. A simple C API should be exposed to access the model, and then bindings can more easily be written for Python, node.js, or whatever other language.

This would partially solve #82 and #162.

Edit: on that note, is it possible to do inference from two or more prompts on different threads? If so, serving multiple people would be possible without multiple copies of model weights in RAM.

---

## Issue #N/A: Converted GGML models hosting?

**Link**: https://github.com/ggml-org/llama.cpp/issues/170
**State**: closed
**Created**: 2023-03-15T18:00:46+00:00
**Closed**: 2023-03-15T20:53:08+00:00
**Comments**: 1
**Labels**: model

### Description

Apologies if Github Issues is not the right place for this question, but do you know if anyone has hosted the ggml versions of the models? The disk space required to download and convert is a little steep.

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

## Issue #N/A: RISC-V support?

**Link**: https://github.com/ggml-org/llama.cpp/issues/165
**State**: closed
**Created**: 2023-03-15T16:07:46+00:00
**Closed**: 2023-07-07T13:48:11+00:00
**Comments**: 9
**Labels**: enhancement, hardware

### Description

By deleting line 155 (#include <immintrin.h>) in ggml.c, it works just fine on RISC-V.
Maybe this can be added in Cmake?

---

## Issue #N/A: Will there ever be a GPU support for Apple Silicon?

**Link**: https://github.com/ggml-org/llama.cpp/issues/164
**State**: closed
**Created**: 2023-03-15T16:06:51+00:00
**Closed**: 2023-03-15T20:10:04+00:00
**Comments**: 1
**Labels**: enhancement, hardware

### Description

I really thank you for the possibility of running the model on my MacBook Air M1. I've been testing various parameters and I'm happy even with the 7B model. However, do you plan to utilize the GPU of M1/M2 chip? Thank you in advance.

---

## Issue #N/A: Not an issue but what depends on the number of threads?

**Link**: https://github.com/ggml-org/llama.cpp/issues/163
**State**: closed
**Created**: 2023-03-15T16:03:26+00:00
**Closed**: 2023-03-15T20:54:16+00:00
**Comments**: 3
**Labels**: performance

### Description

I've been testing your code from 1 to 8 threads and the output is always different. The speed is not depend on the number of threads. On the contrary, 4 threads may perform much better than 1, whereas 8 threads supposedly provides a better result. However, the same prompt may give the same excellent output with triple speed with 4 threads compared to 8. But still, when I use 8 threads (my maximum on M1) I use all my CPU resources, but it doesn't affect speed at all (seemingly works slower) and not giving quality effect (apparently). Am I wrong? Can you correct me if I'm mistaken? May be there is some best speed/quality option and I just that stupid that was unable to figure out how to use this option?

---

## Issue #N/A: feature request, restful api / exposure

**Link**: https://github.com/ggml-org/llama.cpp/issues/162
**State**: closed
**Created**: 2023-03-15T15:50:42+00:00
**Closed**: 2023-03-15T21:07:48+00:00
**Comments**: 3
**Labels**: duplicate, enhancement

### Description

hi team,

was playing interactive mode for couple hours, pretty impressive

resides what's mentioned in #145 , 
it might be not too far, to plug this a endpoint / functional call ( like swig or socket or openapi to replace current stdin ?, then self-host can have a very powerful new residents, like i got a powerful PC at home to be personal assist

also found that `-n` is the context / token limit, would be great if engine can start with 0 presume context ( which is to lift off / decouple a bit from stdin 

kindly let me know if there are directions or others interested in this ( also a developer here but not so C / tensor flavored 
( as without advice, force hi-jack stdin / stdout seems stupid 

---

## Issue #N/A: Add avx-512 support?

**Link**: https://github.com/ggml-org/llama.cpp/issues/160
**State**: closed
**Created**: 2023-03-15T12:10:17+00:00
**Closed**: 2023-03-28T09:54:15+00:00
**Comments**: 6
**Labels**: enhancement, performance, hardware

### Description

No clue but I think it may work faster

---

## Issue #N/A: Unable to compile - error: inlining failed in call to ‘always_inline’ ‘_mm256_cvtph_ps’

**Link**: https://github.com/ggml-org/llama.cpp/issues/159
**State**: closed
**Created**: 2023-03-15T10:53:18+00:00
**Closed**: 2023-03-15T15:23:31+00:00
**Comments**: 2
**Labels**: hardware, build

### Description

Hi, I downloaded the files with git and run make just as in the instruction. Unfortunately, the compilation is not working. Can someone help me figure out what's going wrong here?

I'm adding the full error in the following.

``In file included from /usr/lib/gcc/x86_64-linux-gnu/10/include/immintrin.h:113,
                 from ggml.c:155:
ggml.c: In function ‘ggml_vec_dot_f16’:
/usr/lib/gcc/x86_64-linux-gnu/10/include/f16cintrin.h:52:1: error: inlining failed in call to ‘always_inline’ ‘_mm256_cvtph_ps’: target specific option mismatch
   52 | _mm256_cvtph_ps (__m128i __A)
      | ^~~~~~~~~~~~~~~
ggml.c:911:33: note: called from here
  911 | #define GGML_F32Cx8_LOAD(x)     _mm256_cvtph_ps(_mm_loadu_si128((__m128i *)(x)))
      |                                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ggml.c:921:37: note: in expansion of macro ‘GGML_F32Cx8_LOAD’
  921 | #define GGML_F16_VEC_LOAD(p, i)     GGML_F32Cx8_LOAD(p)
      |                               

[... truncated for brevity ...]

---

## Issue #N/A: is it possible to use llama,cpp with other neural networks?

**Link**: https://github.com/ggml-org/llama.cpp/issues/158
**State**: closed
**Created**: 2023-03-15T09:31:55+00:00
**Closed**: 2023-07-28T19:32:08+00:00
**Comments**: 2
**Labels**: enhancement, model

### Description

I have no clue about this, but I saw that chatglm-6b was published, which should run on CPU with 16GB ram, albeit very slow.
[https://huggingface.co/THUDM/chatglm-6b/tree/main](url)

Would it be possible to substitute the llama model?

---

## Issue #N/A: Crafting prompts to get LLaMA models to generate interesting content

**Link**: https://github.com/ggml-org/llama.cpp/issues/156
**State**: closed
**Created**: 2023-03-15T07:14:54+00:00
**Closed**: 2023-03-15T19:30:32+00:00
**Comments**: 18
**Labels**: good first issue, model

### Description

Hi,

Im getting a strange behaviour and answer:

```
./main -m ./models/7B/ggml-model-q4_0.bin -t 8 -n 256 --repeat_penalty 1.0 --color -p "User: how many wheels have a car?"
main: seed = 1678864388
llama_model_load: loading model from './models/7B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 4096
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 32
llama_model_load: n_layer = 32
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 11008
llama_model_load: n_parts = 1
llama_model_load: ggml ctx size = 4529.34 MB
llama_model_load: memory_size =   512.00 MB, n_mem = 16384
llama_model_load: loading model part 1/1 from './models/7B/ggml-model-q4_0.bin'
llama_model_load: .................................... done
llama_model_load: model size =  4017.27 MB / num tensors = 291

system_info: n_threads = 8 / 10 | AVX = 0 | AVX2 = 0 | AVX512

[... truncated for brevity ...]

---

## Issue #N/A: What models i really need?

**Link**: https://github.com/ggml-org/llama.cpp/issues/155
**State**: closed
**Created**: 2023-03-15T06:18:17+00:00
**Closed**: 2023-04-07T16:11:32+00:00
**Comments**: 3
**Labels**: model

### Description

Hi,

What models i really need?

I have these:

<img width="423" alt="image" src="https://user-images.githubusercontent.com/395096/225223070-ceb1a05b-8af6-4426-8a51-6cfa6d156718.png">

The only 7B folder for example is necessary? Each model has different results?

I don't understand if i need only one and execute the training for each folder or if only one is necessary and i need choose one.

Thanks.

---

## Issue #N/A: ggml_new_tensor_impl: not enough space in the context's memory pool (needed 717778556, available 454395136)

**Link**: https://github.com/ggml-org/llama.cpp/issues/153
**State**: closed
**Created**: 2023-03-15T04:18:32+00:00
**Closed**: 2023-03-24T16:11:41+00:00
**Comments**: 1
**Labels**: duplicate, need more info

### Description

Hey, I know someone already posted a similar issue that has already been closed, but I ran into the same thing. On windows 10 and cloned just yesterday

---

## Issue #N/A: Q4_1 inference appears broken for 13B parameters

**Link**: https://github.com/ggml-org/llama.cpp/issues/152
**State**: closed
**Created**: 2023-03-15T03:22:13+00:00
**Closed**: 2023-03-15T23:25:40+00:00
**Comments**: 5
**Labels**: bug, model

### Description

I have been experimenting with q4_1 quantisation (since [some preliminary results](https://nolanoorg.substack.com/p/int-4-llama-is-not-enough-int-3-and) suggest it shold perform better), and noticed that something about the pipeline for the 13B parameter model is broken (whether it is the quantization itself, or the saving or loading). This results in all inferred tokens coming out as `#`. Meanwhile, 7B works well.

I know we had a patch a while ago that first made the 13B+ models work for q4_0 - did whatever fixes it made not cover q4_1?

---

## Issue #N/A: It appears context memory usage can be trivially halved by using fp16?

**Link**: https://github.com/ggml-org/llama.cpp/issues/146
**State**: closed
**Created**: 2023-03-14T23:11:08+00:00
**Closed**: 2023-03-19T17:57:01+00:00
**Comments**: 0
**Labels**: enhancement

### Description

I'm not fully familiar with this codebase, so pardon if I'm wrong. My first attempt to modify the code was to expand hardcoded context window of 512 to 4096 but additional memory usage was not pleasant.

LLAMA 7B quantized to 4 bits reports `ggml ctx size = 8113.34 MB`

I went to the code and changed data type for `memory_k` and `memory_v` from `GGML_TYPE_F32` to `GGML_TYPE_F16`

These are the changed lines:

```
        ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F16); // memory_k
        ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F16); // memory_v
```

And these:

```
        model.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);
        model.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);
```

New memory usage is reportedly `ggml ctx size = 6065.34 MB` and task manager agrees. That's 2GB down.
So far everything is working, no crashes and no degradation in quality. Is there any reason to not do that?

---

## Issue #N/A: Reset context instead of quitting in interactive mode

**Link**: https://github.com/ggml-org/llama.cpp/issues/145
**State**: closed
**Created**: 2023-03-14T21:26:49+00:00
**Closed**: 2023-03-16T12:04:28+00:00
**Comments**: 5
**Labels**: duplicate, enhancement

### Description

It's really annoying that I have to restart the program every time it quits by **[end of text]** or exceeding context limits, as I need to reload model, which is inefficient.
Is there any way to add an option that instead of quitting just resets to the initial prompt? 

---

## Issue #N/A: Interactive mode does not work

**Link**: https://github.com/ggml-org/llama.cpp/issues/144
**State**: closed
**Created**: 2023-03-14T21:21:46+00:00
**Closed**: 2023-03-15T07:12:15+00:00
**Comments**: 3
**Labels**: wontfix

### Description

On Windows 10 I run the command
```
G:/LLaMa/llama.cpp/Debug/llama.exe -m G:/LLaMa/llama.cpp/models/7B/ggml-model-q4_0.bin -t 8 -n 256 --repeat_penalty 1.0 --color -i -r "User:" -p "Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

User: Hello, Bob.
Bob: Hello. How may I help you today? 
User: Please tell me the largest city in Europe.
Bob: Sure. The largest city in Europe is Moscow, the capital of Russia.
User:"
```
, it works out, but then the AI continues to simulate the dialogue, not giving me access
Nothing happens when you try to press Enter
Maybe I'm doing something wrong? 
![image](https://user-images.githubusercontent.com/31831491/225138823-e03443ba-bd59-4ede-a0da-d0510c3263eb.png)



---

## Issue #N/A: Ability to take in a config file as initial prompt

**Link**: https://github.com/ggml-org/llama.cpp/issues/141
**State**: closed
**Created**: 2023-03-14T20:19:51+00:00
**Closed**: 2023-07-28T19:31:58+00:00
**Comments**: 4
**Labels**: enhancement

### Description

Following on to the "Store preprocessed prompts", it would be good to be able to take in a text file with a generic prompt & flags to start a chatbot or similar. 
Such a config file could be a yaml or toml and include flags for running, model locations, prompt locations, etc. 

---

## Issue #N/A: Only show prompt and response 

**Link**: https://github.com/ggml-org/llama.cpp/issues/140
**State**: closed
**Created**: 2023-03-14T19:52:32+00:00
**Closed**: 2023-03-14T19:57:46+00:00
**Comments**: 2
**Labels**: wontfix

### Description

Hi!

I was wondering if there is a way to only get the response without getting all the debug/info logs before?

---

## Issue #N/A: convert the 7B model to ggml FP16 format fails on RPi 4B

**Link**: https://github.com/ggml-org/llama.cpp/issues/138
**State**: closed
**Created**: 2023-03-14T17:47:38+00:00
**Closed**: 2023-03-15T21:19:53+00:00
**Comments**: 9
**Labels**: hardware

### Description

Everything's OK until this step

python3 convert-pth-to-ggml.py models/7B/ 1
{'dim': 4096, 'multiple_of': 256, 'n_heads': 32, 'n_layers': 32, 'norm_eps': 1e-06, 'vocab_size': 32000}
n_parts =  1
Processing part  0
Killed


models/7B/ggml-model-f16.bin isn't created


---

## Issue #N/A: FP16 and 4-bit quantized model both produce garbage output on M1 8GB

**Link**: https://github.com/ggml-org/llama.cpp/issues/137
**State**: closed
**Created**: 2023-03-14T17:05:51+00:00
**Closed**: 2023-03-14T20:54:06+00:00
**Comments**: 4
**Labels**: hardware

### Description

Both the `ggml-model-q4_0` and `ggml-model-f16` produce a garbage output on my M1 Air 8GB, using the 7B LLaMA model. I've seen the quantized model having problems but I doubt the quantization is the issue as the non-quantized model produces the same output.

```
➜  llama.cpp git:(master) ./main -m ./models/7B/ggml-model-f16.bin -p "Building a website can be done in 10 simple steps:" -t 8 -n 512
main: seed = 1678812348
llama_model_load: loading model from './models/7B/ggml-model-f16.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 4096
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 32
llama_model_load: n_layer = 32
llama_model_load: n_rot   = 128
llama_model_load: f16     = 1
llama_model_load: n_ff    = 11008
llama_model_load: n_parts = 1
llama_model_load: ggml ctx size = 13365.09 MB
llama_model_load: memory_size =   512.00 MB, n_mem = 16384
llama_model_load: loading model part 1/1 from '.

[... truncated for brevity ...]

---

## Issue #N/A: Installation Fails on M1 Mac Air

**Link**: https://github.com/ggml-org/llama.cpp/issues/136
**State**: closed
**Created**: 2023-03-14T16:17:05+00:00
**Closed**: 2023-03-15T21:21:22+00:00
**Comments**: 2
**Labels**: build

### Description

When I run the two commands the installer throws the following errors about halfway through the install:



cc  -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -DGGML_USE_ACCELERATE   -c ggml.c -o ggml.o
ggml.c:1364:25: error: implicit declaration of function 'vdotq_s32' is invalid in C99 [-Werror,-Wimplicit-function-declaration]
        int32x4_t p_0 = vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0ls);
                        ^
ggml.c:1364:19: error: initializing 'int32x4_t' (vector of 4 'int32_t' values) with an expression of incompatible type 'int'
        int32x4_t p_0 = vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0ls);
                  ^     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ggml.c:1365:19: error: initializing 'int32x4_t' (vector of 4 'int32_t' values) with an expression of incompatible type 'int'
        int32x4_t p_1 = vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1ls);
                  ^     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ggml.c:1367:13: error: assigning to '

[... truncated for brevity ...]

---

## Issue #N/A: unexpected shut down when number of tokens is large

**Link**: https://github.com/ggml-org/llama.cpp/issues/134
**State**: closed
**Created**: 2023-03-14T14:09:10+00:00
**Closed**: 2023-03-14T23:57:57+00:00
**Comments**: 3
**Labels**: duplicate

### Description

I found that the model of LLaMA-7B shut down unexpectedly when the number of tokens in prompt reaches some value, this value is approximately to be 500
this cannot be solved by setting number of tokens to predict high (e.g. 204800)

my initialization is:
```
./main -m ./models/7B/ggml-model-q4_0.bin \
-n 204800 \
-t 8 \
--repeat_penalty 1.0 \
--color -i \
-r "HeMuling:" \
--temp 1.0 \
-f ./models/p.txt
```
where `p.txt` is a file containing some prompts, and the token number of prompts is `main: number of tokens in prompt = 486`
the program shut down unexpectedly after a few interactions, last shows:
```
Allice:like how big
HeMuling

main: mem per token = 14434244 bytes
main:     load time =  1400.10 ms
main:   sample time =    21.30 ms
main:  predict time = 79072.03 ms / 154.74 ms per token
main:    total time = 88429.08 ms
```
I am using macPro M1 with 16GB RAM

I am wondering is there any limitation in the  program or did i do something wrong

---

## Issue #N/A: Proposal: Retire make; Update build instructions for Cmake 

**Link**: https://github.com/ggml-org/llama.cpp/issues/133
**State**: closed
**Created**: 2023-03-14T13:51:02+00:00
**Closed**: 2023-07-28T19:31:22+00:00
**Comments**: 2
**Labels**: enhancement

### Description

Now we have a shiny new cmake frontend, can we:

- eliminate the makefile?
- document the Cmake build instructions?

As far as I know, users might use the make file if they don't have cmake. There might also be some features in the makefile that still need to be transferred to cmakelist.txt.

But as far as I know, cmake will need to generate the makefile for each user's environment. So it may no longer make sense to track the makefile at all. Tracking the cmakelist.txt might be enough.

This of course puts the burden on the user to install cmake and know how to call it. That's a little more complicated that make. It would help to add the new steps to readme.md, and remove the old make-based build steps.

I can implement both, but ideas and suggestions are needed - what are the plans for supporting (or deprecating) the makefile / supporting make-based build steps? Lemme know we can get rid of em' all, and I can get to work! 🔥🔥🔥 Otherwise, lemme know the preferred approach!

---

## Issue #N/A: Quantitative measurement of model perplexity for different models and model quantization modes 

**Link**: https://github.com/ggml-org/llama.cpp/issues/129
**State**: closed
**Created**: 2023-03-14T12:38:25+00:00
**Closed**: 2023-03-22T22:41:53+00:00
**Comments**: 53
**Labels**: model, generation quality

### Description

llama.cpp seems to give bad results compared to Facebook's implementation.

Here's an example simple reading comprehension prompt:

> Question: "Tom, Mark, and Paul bought books: two with pictures and one without. Tom and Mark had different kinds of books. What kind did Paul buy?" Answer: "Paul bought a book

LLaMA 7B with Facebook's implementation yields:

Seed `1`:

> Question: "Tom, Mark, and Paul bought books: two with pictures and one without. Tom and Mark had different kinds of books. What kind did Paul buy?" Answer: "Paul bought a book with pictures."
Asked by lone wolf 1788 days ago.

Seed `2` (to show that the above is not just a fluke):

> Question: "Tom, Mark, and Paul bought books: two with pictures and one without. Tom and Mark had different kinds of books. What kind did Paul buy?" Answer: "Paul bought a book with pictures."
Question: "Tom, Mark, and Paul bought books: two with pictures and

While llama.cpp without quantization (so still float16) generate

[... truncated for brevity ...]

---

## Issue #N/A: Is it possible to run llama.cpp in Google Colab Pro?

**Link**: https://github.com/ggml-org/llama.cpp/issues/128
**State**: closed
**Created**: 2023-03-14T12:38:11+00:00
**Closed**: 2023-03-15T21:27:56+00:00
**Comments**: 2
**Labels**: hardware

### Description

Any help or guidance would be greatly appreciated.

---

## Issue #N/A: Build fails on Ubuntu 20

**Link**: https://github.com/ggml-org/llama.cpp/issues/127
**State**: closed
**Created**: 2023-03-14T11:14:36+00:00
**Closed**: 2023-03-14T11:36:35+00:00
**Comments**: 2
**Labels**: build

### Description

```
$ make 
I llama.cpp build info: 
I UNAME_S:  Linux
I UNAME_P:  x86_64
I UNAME_M:  x86_64
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -mavx2 -mfma -mf16c -msse3
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread
I LDFLAGS:  
I CC:       cc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
I CXX:      g++ (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0

cc  -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -mavx2 -mfma -mf16c -msse3   -c ggml.c -o ggml.o
g++ -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread -c utils.cpp -o utils.o
g++ -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread main.cpp ggml.o utils.o -o main 
main.cpp: In function ‘int main(int, char**)’:
main.cpp:1006:30: warning: ignoring return value of ‘int scanf(const char*, ...)’, declared with attribute warn_unused_result [-Wunused-result]
 1006 |                         scanf("%*c");
      |                         ~~~~~^~~~~~~
./main -h
```


---

## Issue #N/A: android port of llama.cpp

**Link**: https://github.com/ggml-org/llama.cpp/issues/124
**State**: closed
**Created**: 2023-03-14T08:16:53+00:00
**Closed**: 2023-07-28T19:31:07+00:00
**Comments**: 13
**Labels**: build

### Description

@ggerganov , can we expect an android port like the whisper one?

---

## Issue #N/A: Won't compile on a MacBook Pro M1 8 GB

**Link**: https://github.com/ggml-org/llama.cpp/issues/123
**State**: closed
**Created**: 2023-03-14T08:00:23+00:00
**Closed**: 2023-03-14T09:27:04+00:00
**Comments**: 3
**Labels**: build

### Description

Hi
I'm on a Macbook Pro M1 with 8GB RAM ; I use zsh (if that's of any importance).
I have no experience in C/C++ other than compiling stuff.
Cloning the repo and entering make gives this :
```
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
ggml.c:1364:25: error: implicit declaration of function 'vdotq_s32' is invalid in C99 [-Werror,-Wimplicit-function-declaration]
        int32x4_t p_0 = vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0ls);
                        ^
ggml.c:1364:19: error: initializing 'int32x4_t' (vector of 4 'int32_t' va

[... truncated for brevity ...]

---

## Issue #N/A: It's strange to return after executing the command

**Link**: https://github.com/ggml-org/llama.cpp/issues/122
**State**: closed
**Created**: 2023-03-14T07:00:34+00:00
**Closed**: 2023-03-15T21:30:03+00:00
**Comments**: 4
**Labels**: duplicate, enhancement

### Description

./main -m ./models/7B/ggml-model-q4_0.bin -t 64 -n 256 --repeat_penalty 1.0 --color -i -r "User:" -p 'What is your name?'
![image](https://user-images.githubusercontent.com/17468133/224920438-696f3b65-bc7c-42d9-ab10-a46b686dcb47.png)
Is it because I haven't installed something？
Centos 7  


---

## Issue #N/A: llama_model_load: llama_model_load: unknown tensor '' in model file

**Link**: https://github.com/ggml-org/llama.cpp/issues/121
**State**: closed
**Created**: 2023-03-14T06:53:04+00:00
**Closed**: 2023-03-14T07:12:00+00:00
**Comments**: 2
**Labels**: wontfix

### Description

$ ./main -m ./models/30B/ggml-model-q4_0.bin -t 8 -n 128 -p 'The first president of the USA was'
main: seed = 1678775977
llama_model_load: loading model from './models/30B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 6656
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 52
llama_model_load: n_layer = 60
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 17920
llama_model_load: n_parts = 4
llama_model_load: ggml ctx size = 20951.50 MB
llama_model_load: memory_size =  1560.00 MB, n_mem = 30720
llama_model_load: loading model part 1/4 from './models/30B/ggml-model-q4_0.bin'
llama_model_load: ................................................................... done
llama_model_load: model size =  4850.14 MB / num tensors = 543
llama_model_load: loading model part 2/4 from './models/30B/ggml-model-q4_0.bin.1'
llama_model_load: llama_mode

[... truncated for brevity ...]

---

## Issue #N/A: Unhandled exception: _Xlength_error("string too long")

**Link**: https://github.com/ggml-org/llama.cpp/issues/119
**State**: closed
**Created**: 2023-03-14T05:48:49+00:00
**Closed**: 2023-05-16T19:18:32+00:00
**Comments**: 5
**Labels**: bug

### Description

Use cmake to create the vc++ project ,and debug in vs2022.
python convert-pth-to-ggml.py models/7B/ 1
done.
quantize.exe .\models\7B\ggml-model-f16.bin .\models\7B\ggml-model-q4_0.bin 2
done.
llama -m .\models\7B\ggml-model-q4_0.bin -t 8 -n 128
> main: seed = 1678771218
> llama_model_load: loading model from '.\models\7B\ggml-model-q4_0.bin' - please wait ...
> llama_model_load: n_vocab = 32000
> llama_model_load: n_ctx   = 512
> llama_model_load: n_embd  = 4096
> llama_model_load: n_mult  = 256
> llama_model_load: n_head  = 32
> llama_model_load: n_layer = 32
> llama_model_load: n_rot   = 128
> llama_model_load: f16     = 2
> llama_model_load: n_ff    = 11008
> llama_model_load: n_parts = 1
> llama_model_load: ggml ctx size = 4529.34 MB
> llama_model_load: memory_size =   512.00 MB, n_mem = 16384
> llama_model_load: loading model part 1/1 from '.\models\7B\ggml-model-q4_0.bin'
> llama_model_load:

Release file: llama.exe

When i use Debug:

There are "Unhand

[... truncated for brevity ...]

---

## Issue #N/A: [Feature request?]: Running larger models without quantization.

**Link**: https://github.com/ggml-org/llama.cpp/issues/118
**State**: closed
**Created**: 2023-03-14T05:45:24+00:00
**Closed**: 2023-03-14T09:12:01+00:00
**Comments**: 6

### Description

Current error
```bash
[1]    11624 segmentation fault (core dumped)  ./llama -m ./models/13B/ggml-model-f16.bin -p  -t 8 --temp 0.5 --top_p 1 
```

---

## Issue #N/A: Convert h5 format to ggml

**Link**: https://github.com/ggml-org/llama.cpp/issues/117
**State**: closed
**Created**: 2023-03-14T05:13:42+00:00
**Closed**: 2023-07-28T19:30:53+00:00
**Comments**: 1
**Labels**: enhancement, model

### Description

There has been a llama model file hosted on [Hugging Face](https://huggingface.co/decapoda-research/llama-30b-hf/tree/main)

It would be good if there is a convert script for this format as well, just like what has been done on [whisper.cpp](https://github.com/ggerganov/whisper.cpp/blob/09e90680072d8ecdf02eaf21c393218385d2c616/models/convert-pt-to-ggml.py)

---

## Issue #N/A: Error build on centos 7 

**Link**: https://github.com/ggml-org/llama.cpp/issues/116
**State**: closed
**Created**: 2023-03-14T04:26:11+00:00
**Closed**: 2023-03-15T21:34:00+00:00
**Comments**: 2
**Labels**: build

### Description

![image](https://user-images.githubusercontent.com/17468133/224892691-9a1f7fac-5a4c-46a1-8ab2-c47aee610a42.png)
![image](https://user-images.githubusercontent.com/17468133/224892885-ea80b64e-5630-4a0a-8aba-8997b8326726.png)
![image](https://user-images.githubusercontent.com/17468133/224893003-7f8aaa36-d3c2-4703-856f-97db1001edef.png)
Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz
Where is the problem? What should I do?



---

## Issue #N/A: The prompt is not converted to tokens

**Link**: https://github.com/ggml-org/llama.cpp/issues/113
**State**: closed
**Created**: 2023-03-14T04:00:37+00:00
**Closed**: 2023-04-07T16:19:58+00:00
**Comments**: 8
**Labels**: bug

### Description

./main -m ./models/7B/ggml-model-q4_0.bin -p "Building a website can be done in 10 simple steps:" -t 8 -n 512
![image](https://user-images.githubusercontent.com/6960679/224889376-929af931-309c-41c0-8319-32fba4eb5ee1.png)

llama.cpp Is the latest version
Can anyone help me? Thanks!


---

## Issue #N/A: Any way to change context limit?

**Link**: https://github.com/ggml-org/llama.cpp/issues/112
**State**: closed
**Created**: 2023-03-14T02:06:54+00:00
**Closed**: 2023-03-15T21:36:51+00:00
**Comments**: 3
**Labels**: duplicate, enhancement

### Description

Is there any setting in any of the scripts to change the context limit? :)

Thanks in advance!

---

## Issue #N/A: Make a tag/release

**Link**: https://github.com/ggml-org/llama.cpp/issues/111
**State**: closed
**Created**: 2023-03-14T02:04:37+00:00
**Closed**: 2023-03-14T19:16:27+00:00
**Comments**: 1
**Labels**: wontfix

### Description

Thanks.

---

## Issue #N/A: RuntimeError: PytorchStreamReader failed reading zip archive: not a ZIP archive

**Link**: https://github.com/ggml-org/llama.cpp/issues/110
**State**: closed
**Created**: 2023-03-14T01:28:51+00:00
**Closed**: 2023-03-15T21:37:38+00:00
**Comments**: 2
**Labels**: model

### Description

Hello, I try to # convert the 7B model to ggml FP16 format but I found a problem? Is that because of the model problem?  🙏🏻
```zsh
python3 convert-pth-to-ggml.py models/7B/ 1
```

```
.
├── CMakeLists.txt
├── LICENSE
├── Makefile
├── README.md
├── convert-pth-to-ggml.py
├── ggml.c
├── ggml.h
├── ggml.o
├── main
├── main.cpp
├── models
│   ├── 7B
│   │   ├── checklist.chk
│   │   ├── consolidated.00.pth
│   │   └── params.json
│   ├── tokenizer.model
│   └── tokenizer_checklist.chk
├── quantize
├── quantize.cpp
├── quantize.sh
├── utils.cpp
├── utils.h
└── utils.o
```


```zsh
(Lab2) @-MacBook-Pro llama.cpp % python convert-pth-to-ggml.py models/7B/ 1
{'dim': 4096, 'multiple_of': 256, 'n_heads': 32, 'n_layers': 32, 'norm_eps': 1e-06, 'vocab_size': 32000}
n_parts =  1
Processing part  0
Traceback (most recent call last):
  File "Lab2/llama.cpp/convert-pth-to-ggml.py", line 88, in <module>
    model = torch.load(fname_model, map_location="cpu")
   

[... truncated for brevity ...]

---

## Issue #N/A: Build on Debian Docker

**Link**: https://github.com/ggml-org/llama.cpp/issues/108
**State**: closed
**Created**: 2023-03-13T23:21:39+00:00
**Closed**: 2023-03-15T21:36:08+00:00
**Comments**: 5
**Labels**: hardware, build

### Description

Hello, wanted to experiment installing the system in a Linux/Debian container but I am getting the following error when I am issuing make. 
- "failed in call to 'always_inline' '_mm256_cvtph_ps'" (I have a more detailed output bellow.)

A. I used the bitnami/pytorch  which is based on debian https://hub.docker.com/r/bitnami/pytorch 
B. i downloaded the git repository on a folder named app and issued the following command :

`docker run --user root -v /host/DOCKER/images/PYTORCH/app:/app/  -it --rm bitnami/pytorch   /bin/bash`

C. consequently updated and installed build-essential with

`apt-get update & apt-get install build-essential`

D. Last, i entered in the repo folder and got the following compilation error while issuing make

`make
I llama.cpp build info:
I UNAME_S:  Linux
I UNAME_P:  unknown
I UNAME_M:  x86_64
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -mavx2 -msse3
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pt

[... truncated for brevity ...]

---

## Issue #N/A: Error: inlining failed in call to always_inline ‘_mm256_cvtph_ps’: target specific option mismatch

**Link**: https://github.com/ggml-org/llama.cpp/issues/107
**State**: closed
**Created**: 2023-03-13T23:20:27+00:00
**Closed**: 2023-03-14T18:08:16+00:00
**Comments**: 22
**Labels**: duplicate, good first issue, hardware, build

### Description

I cloned the GitHub repository and ran the make command but was unable to get the cpp files to compile successfully. Any help or suggestion would be appreciated.

Terminal output:
<pre><font color="#4E9A06"><b>brickman@Ubuntu-brickman</b></font>:<font color="#3465A4"><b>~/Desktop/llama.cpp</b></font>$ ls
CMakeLists.txt  convert-pth-to-ggml.py  ggml.c  ggml.h  LICENSE  main.cpp  Makefile  <font color="#3465A4"><b>models</b></font>  quantize.cpp  <font color="#4E9A06"><b>quantize.sh</b></font>  README.md  utils.cpp  utils.h
<font color="#4E9A06"><b>brickman@Ubuntu-brickman</b></font>:<font color="#3465A4"><b>~/Desktop/llama.cpp</b></font>$ make
I llama.cpp build info: 
I UNAME_S:  Linux
I UNAME_P:  x86_64
I UNAME_M:  x86_64
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -mavx2 -msse3
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread
I LDFLAGS:  
I CC:       cc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
I CXX:      g++ (Ubuntu 9.4.0-1u

[... truncated for brevity ...]

---

## Issue #N/A: Parallel Quantize.sh, add &

**Link**: https://github.com/ggml-org/llama.cpp/issues/106
**State**: closed
**Created**: 2023-03-13T23:07:05+00:00
**Closed**: 2023-03-19T19:54:08+00:00
**Comments**: 6
**Labels**: enhancement

### Description

@prusnak 

`./quantize "$i" "${i/f16/q4_0}" 2 &`

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

## Issue #N/A: Anyplan to make CodeGenCPP?

**Link**: https://github.com/ggml-org/llama.cpp/issues/104
**State**: closed
**Created**: 2023-03-13T21:14:10+00:00
**Closed**: 2023-03-14T11:34:49+00:00
**Comments**: 3
**Labels**: enhancement, wontfix

### Description

Llama models seesm to be not useful for code genration.

Any chance to get CodeGen models work on CPU ? https://github.com/salesforce/CodeGen

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

## Issue #N/A: json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)

**Link**: https://github.com/ggml-org/llama.cpp/issues/102
**State**: closed
**Created**: 2023-03-13T20:01:52+00:00
**Closed**: 2023-03-15T21:41:08+00:00
**Comments**: 2
**Labels**: duplicate, model

### Description

Bug encountered when running `python3 convert-pth-to-ggml.py models/7B/ 1`:

```
llama.cpp % python3 convert-pth-to-ggml.py models/7B/ 1
Traceback (most recent call last):
  File "/Users/jjyuhub/llama.cpp/convert-pth-to-ggml.py", line 69, in <module>
    hparams = json.load(f)
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/json/__init__.py", line 293, in load
    return loads(fp.read(),
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/json/__init__.py", line 346, in loads
    return _default_decoder.decode(s)
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/json/decoder.py", line 337, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/json/decoder.py", line 355, in raw_decode
 

[... truncated for brevity ...]

---

## Issue #N/A: M1 Max + GNU coreutils: "Your arch is announced as x86_64, but it seems to actually be ARM64"

**Link**: https://github.com/ggml-org/llama.cpp/issues/101
**State**: closed
**Created**: 2023-03-13T19:57:53+00:00
**Closed**: 2024-04-10T01:08:09+00:00
**Comments**: 2
**Labels**: bug, hardware, build, stale

### Description

When I build, the makefile detects my M1 Max as 86_64.

This is because I have GNU coreutils `uname` on my `PATH`, which announces my architecture as `arm64` (whereas the system distribution of `uname` would call the same architecture `arm`).

https://github.com/Lightning-AI/lightning/pull/13992#issuecomment-1204157830  
https://github.com/Lightning-AI/lightning/issues/13991

this condition needs widening to accept both `arm` and `arm64`:

https://github.com/ggerganov/llama.cpp/blob/c09a9cfb06c87d114615c105adda91b0e6273b69/Makefile

---

## Issue #N/A: Stanford Alpaca support

**Link**: https://github.com/ggml-org/llama.cpp/issues/99
**State**: closed
**Created**: 2023-03-13T19:15:25+00:00
**Closed**: 2023-03-16T11:40:58+00:00
**Comments**: 6
**Labels**: duplicate, enhancement, model

### Description

Just 3 hrs ago , chat tuned LLAma released : https://github.com/tatsu-lab/stanford_alpaca

---

## Issue #N/A: WebAssembly and emscripten headers

**Link**: https://github.com/ggml-org/llama.cpp/issues/97
**State**: closed
**Created**: 2023-03-13T17:27:58+00:00
**Closed**: 2024-04-10T01:08:10+00:00
**Comments**: 28
**Labels**: enhancement, stale

### Description

Hello I have tried a minimal Emscripten support to `Makefile` adding 

```Makefile
# WASM
EMCXX = em++
EMCC = emcc
EMCXXFLAGS = --bind --std=c++11 -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s "EXPORTED_RUNTIME_METHODS=['addOnPostRun','FS']" -s "DISABLE_EXCEPTION_CATCHING=0" -s "EXCEPTION_DEBUG=1" -s "FORCE_FILESYSTEM=1" -s "MODULARIZE=1" -s "EXPORT_ES6=0" -s 'EXPORT_NAME="LLAMAModule"' -s "USE_ES6_IMPORT_META=0" -I./
EMCCFLAGS = --bind -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s "EXPORTED_RUNTIME_METHODS=['addOnPostRun','FS']" -s "DISABLE_EXCEPTION_CATCHING=0" -s "EXCEPTION_DEBUG=1" -s "FORCE_FILESYSTEM=1" -s "MODULARIZE=1" -s "EXPORT_ES6=0" -s 'EXPORT_NAME="LLAMAModule"' -s "USE_ES6_IMPORT_META=0" -I./ 

EMOBJS = utils.bc ggml.bc

wasm: llama_wasm.js quantize_wasm.js
wasmdebug: export EMCC_DEBUG=1
wasmdebug: llama_wasm.js quantize_wasm.js

#
# WASM lib
#

ggml.bc: ggml.c ggml.h
	$(EMCC) -c $(EMCCFLAGS) ggml.c -o ggml.bc
utils.bc: utils.cpp utils.h
	$(EMCXX) -c $(EMCXXFLAGS) u

[... truncated for brevity ...]

---

## Issue #N/A: any interest in the openchatkit on a power book? 

**Link**: https://github.com/ggml-org/llama.cpp/issues/96
**State**: closed
**Created**: 2023-03-13T16:43:04+00:00
**Closed**: 2023-07-28T19:30:06+00:00
**Comments**: 9
**Labels**: enhancement, question, hardware

### Description

https://www.together.xyz/blog/openchatkit this new repository might also be a good candidate for any local deployment with a strong GPU. As the gptNeox focus is on GPU deployments.


---

## Issue #N/A: Different outputs for differents numbers of threads (same seed)

**Link**: https://github.com/ggml-org/llama.cpp/issues/95
**State**: closed
**Created**: 2023-03-13T16:20:56+00:00
**Closed**: 2023-03-23T21:30:06+00:00
**Comments**: 4
**Labels**: enhancement

### Description

Hello,

I simply wanted to bring up the point that the output can vary based on the number of threads selected, even if the seed stays constant.

I have an intel core i7 10700K that has 16 threads.

For this example I'm using the 13B model (./models/13B/ggml-model-q4_0.bin)

When I put -t 14 (make -j && ./main -m ./models/13B/ggml-model-q4_0.bin -p "Building a website can be done in 10 simple steps:" -t 14 -n 50 --seed 1678486056), I got this result:
![duU196l](https://user-images.githubusercontent.com/110173477/224762353-1c5565d8-478c-41c6-ac13-f7883dc3ec50.png)

When I put -t 15 (make -j && ./main -m ./models/13B/ggml-model-q4_0.bin -p "Building a website can be done in 10 simple steps:" -t 15 -n 50 --seed 1678486056), I got this result:
![5WIrvd1](https://user-images.githubusercontent.com/110173477/224762999-258a6235-b14c-4db8-8b04-163a0b92d356.png)

I have zero knowledge in machine learning, perhaps this is a normal behavior.

Looking forward for your reactions!



[... truncated for brevity ...]

---

## Issue #N/A: Segfault using the chat like interface on the 65B parameterized model

**Link**: https://github.com/ggml-org/llama.cpp/issues/94
**State**: closed
**Created**: 2023-03-13T14:50:31+00:00
**Closed**: 2023-03-13T14:54:10+00:00
**Comments**: 1
**Labels**: wontfix

### Description

$(: !524 ) ./main -m ./models/65B/ggml-model-q4_0.bin -t 8 -n 256 --repeat_penalty 1.0 --color -i -r "User:" -p
Segmentation fault: 11

---

## Issue #N/A: Should use `mmap` for model loading

**Link**: https://github.com/ggml-org/llama.cpp/issues/91
**State**: closed
**Created**: 2023-03-13T11:51:47+00:00
**Closed**: 2023-03-30T19:28:28+00:00
**Comments**: 59
**Labels**: enhancement, good first issue

### Description

So it doesn't create an extra copy in RAM and lives in the kernel page cache happily, loading instantly on subsequent runs.

---

## Issue #N/A: Possible regression on master

**Link**: https://github.com/ggml-org/llama.cpp/issues/89
**State**: closed
**Created**: 2023-03-13T10:42:14+00:00
**Closed**: 2023-04-16T10:41:46+00:00
**Comments**: 6
**Labels**: bug

### Description

Hi, 

I see that interactive mode has been merged in, I was trying to test the repository on a larger set of weights, and found that there is no output anymore. When running it in interactive mode, the code works, so there might be something going on. Haven't had the time to look at it yet. 

The code reports number of tokens / second at the end so it just seems that the tokens are not sent to the console.

Cheers

---

## Issue #N/A: Create json api service

**Link**: https://github.com/ggml-org/llama.cpp/issues/88
**State**: closed
**Created**: 2023-03-13T10:19:23+00:00
**Closed**: 2023-07-28T19:29:40+00:00
**Comments**: 8
**Labels**: need more info

### Description

so we can intergrate app/UI.

---

## Issue #N/A: Chinese character decoding error when intract way

**Link**: https://github.com/ggml-org/llama.cpp/issues/86
**State**: closed
**Created**: 2023-03-13T08:21:26+00:00
**Closed**: 2023-03-13T09:52:33+00:00
**Comments**: 2
**Labels**: duplicate

### Description

<img width="463" alt="image" src="https://user-images.githubusercontent.com/21303438/224645334-fed2e1d7-c858-49c7-b8bc-341fbb01ead3.png">

can not handle Chinese.

---

## Issue #N/A: Faster loading of the model

**Link**: https://github.com/ggml-org/llama.cpp/issues/85
**State**: closed
**Created**: 2023-03-13T08:04:28+00:00
**Closed**: 2023-07-28T19:20:18+00:00
**Comments**: 5
**Labels**: enhancement, good first issue, performance

### Description

I was playing with the 65B model, and it took a minute to read the files. If you wrap the model loader loop with a `#pragma omp parallel for` and add `-fopenmp` to the compiler flags, you can drop it to 18 seconds.


---

## Issue #N/A: Segfault with 65B model

**Link**: https://github.com/ggml-org/llama.cpp/issues/84
**State**: closed
**Created**: 2023-03-13T07:19:05+00:00
**Closed**: 2023-03-31T05:04:49+00:00
**Comments**: 6
**Labels**: need more info

### Description

This is the output with `-fsanitize=address`:
```
AddressSanitizer:DEADLYSIGNAL
=================================================================
==167666==ERROR: AddressSanitizer: SEGV on unknown address 0x558c0562c438 (pc 0x558a27cc9807 bp 0x000000000000 sp 0x7ffeb2f57310 T0)
==167666==The signal is caused by a READ memory access.
    #0 0x558a27cc9807 in ggml_element_size (/home/mattmcal/repos/llama.cpp/main+0x49807)
    #1 0x558a27c9c03c in llama_eval(llama_model const&, int, int, std::vector<int, std::allocator<int> > const&, std::vector<float, std::allocator<float> >&, unsigned long&) (/home/mattmcal/repos/llama.cpp/main+0x1c03c)
    #2 0x558a27c960fb in main (/home/mattmcal/repos/llama.cpp/main+0x160fb)
    #3 0x7fe45e046189 in __libc_start_call_main ../sysdeps/nptl/libc_start_call_main.h:58
    #4 0x7fe45e046244 in __libc_start_main_impl ../csu/libc-start.c:381
    #5 0x558a27c9b1a0 in _start (/home/mattmcal/repos/llama.cpp/main+0x1b1a0)

AddressSanitizer can not p

[... truncated for brevity ...]

---

## Issue #N/A: Models missing

**Link**: https://github.com/ggml-org/llama.cpp/issues/83
**State**: closed
**Created**: 2023-03-13T07:05:32+00:00
**Closed**: 2023-03-13T07:26:37+00:00
**Comments**: 2
**Labels**: model

### Description

Are the models missing in the directory? I don't see them.

---

## Issue #N/A: python bindings?

**Link**: https://github.com/ggml-org/llama.cpp/issues/82
**State**: closed
**Created**: 2023-03-13T07:00:42+00:00
**Closed**: 2023-07-28T19:29:21+00:00
**Comments**: 19
**Labels**: enhancement

### Description

No description provided.

---

## Issue #N/A: invalid model file './models/llama-13b-4bit.pt' (bad magic)

**Link**: https://github.com/ggml-org/llama.cpp/issues/81
**State**: closed
**Created**: 2023-03-13T06:20:02+00:00
**Closed**: 2023-03-13T12:26:07+00:00
**Comments**: 3
**Labels**: model

### Description

Downloaded from 

```url
magnet:?xt=urn:btih:36945b5958b907b3ab69e963ba0de1abdf48c16c&dn=LLaMA-HFv2-4bit&tr=http%3a%2f%2fbt1.archive.org%3a6969%2fannounce&tr=http%3a%2f%2fbt2.archive.org%3a6969%2fannounce
```

---

## Issue #N/A: .pth to .ggml Out of Memory

**Link**: https://github.com/ggml-org/llama.cpp/issues/76
**State**: closed
**Created**: 2023-03-13T02:56:50+00:00
**Closed**: 2023-03-13T03:05:56+00:00
**Comments**: 2
**Labels**: wontfix, hardware

### Description

I have 16 GBs of memory (14 GB free) and running `python3 convert-pth-to-ggml.py models/7B/ 1` causes an OOM error (Killed) on Linux.

Here's the dmesg message:
`Out of memory: Killed process 930269 (python3) total-vm:15643332kB, anon-rss:13201980kB, file-rss:4kB, shmem-rss:0kB, UID:0 pgtables:26524kB oom_score_adj:0`

I will be receiving my new RAM in a few days but I think this is supposed to work with 16 GB memory?

---

## Issue #N/A: commit `96ea727` breaks compilation in Windows

**Link**: https://github.com/ggml-org/llama.cpp/issues/74
**State**: closed
**Created**: 2023-03-13T02:30:43+00:00
**Closed**: 2023-03-13T12:43:40+00:00
**Comments**: 3
**Labels**: bug, build

### Description

Sadly, Windows terminal support of ANSI colors and signals is simply... non-existent, and this PR in particular adds a ton of things that are not supported by this OS.

I don't know if filling the entire llama.cpp with `#ifdef`s would be ideal... or to rewrite those changes to encapsulate them better to make it less painful, but either way, it is a lot of work.

Reverting that commit and doing a couple of merge fixes makes it work.

PS: Maybe it would be good to add an action that compiles the software in the three platforms? but, yet, I'm curious about the willing to support windows as well since it's the special kid between the unix-like guys :)

---

## Issue #N/A: Longer and infinite output

**Link**: https://github.com/ggml-org/llama.cpp/issues/71
**State**: closed
**Created**: 2023-03-13T00:29:55+00:00
**Closed**: 2023-07-28T19:29:06+00:00
**Comments**: 59
**Labels**: enhancement

### Description

If we use `-n 1000000` to have a very long output (for a story for example),
it stops generating quite fast, after around 30 lines, probably because of [this line of code](https://github.com/ggerganov/llama.cpp/blob/460c48254098b28d422382a2bbff6a0b3d7f7e17/main.cpp#L812).

It would be nice if we could have longer outputs and also the possibility to have infinite output, stopping only on `Ctrl-C`.
We could maybe specify that `-n 0` will trigger that infinite output mode.
That issue is a bit related to issue #23 

---

## Issue #N/A: Use an argument parsing library

**Link**: https://github.com/ggml-org/llama.cpp/issues/70
**State**: closed
**Created**: 2023-03-13T00:16:29+00:00
**Closed**: 2023-03-15T21:52:58+00:00
**Comments**: 0
**Labels**: duplicate, enhancement

### Description

The argument parsing for `convert-ckpt-to-ggml.py` is quite ad-hoc and hard to follow.


I'm thinking that something around this would go a long way in making the arguments easier to use and follow in the code.

```python
import argparse

ARG_PARSER = argparse.ArgumentParser()
ARG_PARSER.add_argument("--model",
                        type=str,
                        help="Model to convert")
ARG_PARSER.add_argument("--ftype",
                        type=str,
                        choices=["f16", "f32"],
                        help="Floating point type to use")
ARG_PARSER.add_argument("--output",
                        type=str,
                        help="Where to write the converted model")
ARGS = ARG_PARSER.parse_args()
```

---

## Issue #N/A: 65B model giving incorect output

**Link**: https://github.com/ggml-org/llama.cpp/issues/69
**State**: closed
**Created**: 2023-03-13T00:14:33+00:00
**Closed**: 2023-03-16T11:55:55+00:00
**Comments**: 18
**Labels**: need more info

### Description

```
ubuntu@ip-x:~/llama.cpp$ ./main -m ./models/65B/ggml-model-q4_0.bin \
>   -t 16 \
>   -n 1000000 \
>   -p 'The history of humanity starts with the bing bang, then '
main: seed = 1678666062
llama_model_load: loading model from './models/65B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 8192
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 64
llama_model_load: n_layer = 80
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 22016
llama_model_load: n_parts = 8
llama_model_load: ggml ctx size = 41477.73 MB
llama_model_load: memory_size =  2560.00 MB, n_mem = 40960
llama_model_load: loading model part 1/8 from './models/65B/ggml-model-q4_0.bin'
llama_model_load: .......................................................................................... done
llama_model_load: model size =  4869.09 MB / num tensors = 723
llama_model_l

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

## Issue #N/A: Prompt interrupted before continuation for Unicode UTF-8 emojis

**Link**: https://github.com/ggml-org/llama.cpp/issues/63
**State**: closed
**Created**: 2023-03-12T21:43:19+00:00
**Closed**: 2023-04-01T07:43:18+00:00
**Comments**: 2
**Labels**: bug, duplicate, enhancement

### Description

I have found that when having a Unicode UTF- emoji char like  

Unicode Character “👍” (U+1F44D)

The prompts breaks up.

I'm reading a sample prompt from a text file:


```bash
cat prompt

Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:
```

Looking at logs I can see in fact that the tokenizers breaks at the (U+1F44D) char code:

```
(base)$ p=$(cat prompt); ./main -m ./models/13B/ggml-model-q4_0.bin -p $p -t 4 -n 512
main: seed = 1678656464
llama_model_load: loading model from './models/13B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 5120
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 40
llama_model_load: n_layer = 40
llama_model_load: n_rot   = 128
llama_

[... truncated for brevity ...]

---

## Issue #N/A: Quality of 4-bit quantization

**Link**: https://github.com/ggml-org/llama.cpp/issues/62
**State**: closed
**Created**: 2023-03-12T21:05:56+00:00
**Closed**: 2023-03-13T17:24:55+00:00
**Comments**: 4
**Labels**: duplicate

### Description

The quality of the 4-bit quantization is really abysmal compared to both non-quantized models and GPTQ quantization 
(https://github.com/qwopqwop200/GPTQ-for-LLaMa). Wouldn't it make sense for llama.cpp to load already-prequantized LLaMa models?

---

## Issue #N/A: Raspberry Pi 4 4GB

**Link**: https://github.com/ggml-org/llama.cpp/issues/58
**State**: closed
**Created**: 2023-03-12T18:33:40+00:00
**Closed**: 2023-07-28T19:28:37+00:00
**Comments**: 45
**Labels**: bug, build

### Description

Hi!

Just a report. I've successfully run the LLaMA 7B model on my 4GB RAM Raspberry Pi 4. It's super slow at about 10 sec/token. But it looks like we can run powerful cognitive pipelines on a cheap hardware. It's awesome. Thank you!

Hardware      : BCM2835
Revision        : c03111
Serial             : 10000000d62b612e
Model            : Raspberry Pi 4 Model B Rev 1.1

%Cpu0  : 71.8 us, 14.6 sy,  0.0 ni,  0.0 id,  2.9 wa,  0.0 hi, 10.7 si,  0.0 st
%Cpu1  : 77.4 us, 12.3 sy,  0.0 ni,  0.0 id, 10.4 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu2  : 81.0 us,  8.6 sy,  0.0 ni,  0.0 id, 10.5 wa,  0.0 hi,  0.0 si,  0.0 st
%Cpu3  : 77.1 us, 12.4 sy,  0.0 ni,  1.0 id,  9.5 wa,  0.0 hi,  0.0 si,  0.0 st
MiB Mem :   3792.3 total,     76.2 free,   3622.9 used,     93.2 buff/cache
MiB Swap:  65536.0 total,  60286.5 free,   5249.5 used.     42.1 avail Mem

    PID      USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND
2705518 ubuntu    20   0 5231516   3.3g   1904 R 339.6

[... truncated for brevity ...]

---

## Issue #N/A: Stop keywords

**Link**: https://github.com/ggml-org/llama.cpp/issues/57
**State**: closed
**Created**: 2023-03-12T18:31:27+00:00
**Closed**: 2023-06-12T14:32:13+00:00
**Comments**: 19
**Labels**: enhancement, good first issue

### Description

It'd be useful if there was a way to define tokens that would cause the output to stop prematurely (e.g. for an assistant-style interaction where messages are prefixed with "Assistant: ", "Human: ", you'd set "Human: " as a stop word, so that you could stop the model from continuing on and having a conversation with itself

---

## Issue #N/A: Fine Tuning

**Link**: https://github.com/ggml-org/llama.cpp/issues/55
**State**: closed
**Created**: 2023-03-12T17:50:41+00:00
**Closed**: 2023-03-15T21:57:13+00:00
**Comments**: 3
**Labels**: model

### Description

Hey!

Thank you for your amazing job!

I'm curious is it possible to use RLHF feedback after a response to make small incremental adjustments in a tuning process? For example, if the user decides to fine-tune after an incorrect answer, can the model spend 60 seconds in the fine-tuning phase, save a checkpoint to disk, and then move on to the next question?

---

## Issue #N/A: error: 'CLOCK_MONOTONIC' undeclared

**Link**: https://github.com/ggml-org/llama.cpp/issues/54
**State**: closed
**Created**: 2023-03-12T17:39:45+00:00
**Closed**: 2023-03-22T17:20:28+00:00
**Comments**: 6
**Labels**: help wanted

### Description

 The initial `make` fails with `CLOCK_MONOTONIC undeclared`
```
I llama.cpp build info: 
I UNAME_S:  Linux
I UNAME_P:  unknown
I UNAME_M:  x86_64
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -mavx2 -mfma -mf16c -msse3
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread
I LDFLAGS:  
I CC:       cc (Alpine 12.2.1_git20220924-r9) 12.2.1 20220924
I CXX:      g++ (Alpine 12.2.1_git20220924-r9) 12.2.1 20220924

cc  -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -mavx -mavx2 -mfma -mf16c -msse3   -c ggml.c -o ggml.o
ggml.c: In function 'ggml_time_ms':
ggml.c:309:5: warning: implicit declaration of function 'clock_gettime' [-Wimplicit-function-declaration]
  309 |     clock_gettime(CLOCK_MONOTONIC, &ts);
      |     ^~~~~~~~~~~~~
ggml.c:309:19: error: 'CLOCK_MONOTONIC' undeclared (first use in this function)
  309 |     clock_gettime(CLOCK_MONOTONIC, &ts);
      |                   ^~~~~~~~~~~~~~~
ggml.c:309:19: note: ea

[... truncated for brevity ...]

---

## Issue #N/A: Improving quality with 8bit?

**Link**: https://github.com/ggml-org/llama.cpp/issues/53
**State**: closed
**Created**: 2023-03-12T17:06:45+00:00
**Closed**: 2023-04-11T12:23:28+00:00
**Comments**: 17
**Labels**: enhancement

### Description

I can achieve around 1 token per second on a Ryzen 7 3700X on Linux with the 65B model and 4bit quantization.

If we use 8bit instead, would it run faster? I have 128GB RAM. Is 8bit already supported?
```
$ ./main -m models/65B/ggml-model-q4_0.bin -t 8 -n 128
main: mem per token = 70897348 bytes
main:     load time = 14010.35 ms
main:   sample time =   335.09 ms
main:  predict time = 140527.48 ms / 1089.36 ms per token
main:    total time = 157951.48 ms
```

---

## Issue #N/A: Segmentation Fault Error "not enough space in the context's memory pool"

**Link**: https://github.com/ggml-org/llama.cpp/issues/52
**State**: closed
**Created**: 2023-03-12T16:05:03+00:00
**Closed**: 2024-04-09T01:10:23+00:00
**Comments**: 22
**Labels**: bug, need more info, stale

### Description

This prompt with the 65B model on an M1 Max 64GB results in a segmentation fault. Works with 30B model. Are there problems with longer prompts? Related to #12 

```
./main --model ./models/65B/ggml-model-q4_0.bin --prompt "You are a question answering bot that is able to answer questions about the world. You are extremely smart, knowledgeable, capable, and helpful. You always give complete, accurate, and very detailed responses to questions, and never stop a response in mid-sentence or mid-thought. You answer questions in the following format:

Question: What’s the history of bullfighting in Spain?

Answer: Bullfighting, also known as "tauromachia," has a long and storied history in Spain, with roots that can be traced back to ancient civilizations. The sport is believed to have originated in 7th-century BCE Iberian Peninsula as a form of animal worship, and it evolved over time to become a sport and form of entertainment. Bullfighting as it is known today became popular in Spai

[... truncated for brevity ...]

---

## Issue #N/A: Reproducability information

**Link**: https://github.com/ggml-org/llama.cpp/issues/50
**State**: closed
**Created**: 2023-03-12T14:17:44+00:00
**Closed**: 2023-03-13T17:27:12+00:00
**Comments**: 1
**Labels**: duplicate

### Description

The seed for the website example is included, but using the same parameters doesn't manage to reproduce the example output. Listing what requirements influense reproducability would help in verifying installs.

The failed test is with x86_64 (gcc or clang, no difference), CUDA 12.1, pytorch 1.13.1, numpy 1.23.5, sentencepiece 0.1.97 and Python 3.10.6 on Linux.


---

## Issue #N/A: Windows MSVC support

**Link**: https://github.com/ggml-org/llama.cpp/issues/49
**State**: closed
**Created**: 2023-03-12T13:43:57+00:00
**Closed**: 2023-03-13T17:25:21+00:00
**Comments**: 2
**Labels**: duplicate

### Description

hello, would it add MSVC build support as well?

---

## Issue #N/A: llama.exe doesn't handle relative file paths in Windows correctly

**Link**: https://github.com/ggml-org/llama.cpp/issues/46
**State**: closed
**Created**: 2023-03-12T11:13:54+00:00
**Closed**: 2023-04-16T09:20:58+00:00
**Comments**: 10
**Labels**: bug, model, windows

### Description

Please include the `ggml-model-q4_0.bin` model to actually run the code:

```
% make -j && ./main -m ./models/7B/ggml-model-q4_0.bin -p "Building a website can be done in 10 simple steps:" -t 8 -n 512
I llama.cpp build info: 
I UNAME_S:  Darwin
I UNAME_P:  arm
I UNAME_M:  arm64
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -DGGML_USE_ACCELERATE
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread
I LDFLAGS:   -framework Accelerate
I CC:       Apple clang version 14.0.0 (clang-1400.0.29.202)
I CXX:      Apple clang version 14.0.0 (clang-1400.0.29.202)

cc  -I.              -O3 -DNDEBUG -std=c11   -fPIC -pthread -DGGML_USE_ACCELERATE   -c ggml.c -o ggml.o
c++ -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread -c utils.cpp -o utils.o
c++ -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread main.cpp ggml.o utils.o -o main  -framework Accelerate
c++ -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -pthread quantize.cpp ggml.o utils

[... truncated for brevity ...]

---

## Issue #N/A: Maybe lower default temp and switch to top_k 40

**Link**: https://github.com/ggml-org/llama.cpp/issues/42
**State**: closed
**Created**: 2023-03-12T10:12:43+00:00
**Closed**: 2023-03-13T17:26:16+00:00
**Comments**: 6
**Labels**: generation quality

### Description

Per [this twitter thread](https://twitter.com/theshawwn/status/1632569215348531201). See commit [here](https://github.com/shawwn/llama/commit/40d99d329a5e38d85904d3a6519c54e6dd6ee9e1).

---

## Issue #N/A: Illegal hardware instruction in quantize step

**Link**: https://github.com/ggml-org/llama.cpp/issues/41
**State**: closed
**Created**: 2023-03-12T09:57:31+00:00
**Closed**: 2023-03-13T06:43:26+00:00
**Comments**: 8
**Labels**: build

### Description

* Ran into this error on a Macbook Pro M1
```
./quantize ./models/7B/ggml-model-f16.bin ./models/7B/ggml-model-q4_0.bin 2
[1]    18452 illegal hardware instruction  ./quantize ./models/7B/ggml-model-f16.bin ./models/7B/ggml-model-q4_0.bin 2
```

* What I've tried:
   * Run `main` on the `model-f16`, still have the same error
   * Convert the `13B` model, still same error in `quantize` step

* Env:
Darwin Tungs-MacBook-Pro.local 21.6.0 Darwin Kernel Version 21.6.0: Sat Jun 18 17:07:22 PDT 2022; root:xnu-8020.140.41~1/RELEASE_ARM64_T6000 x86_64

---

## Issue #N/A: Hows the inference speed and mem usage?

**Link**: https://github.com/ggml-org/llama.cpp/issues/39
**State**: closed
**Created**: 2023-03-12T06:39:50+00:00
**Closed**: 2023-03-18T21:03:27+00:00
**Comments**: 14
**Labels**: performance

### Description

Hows the inference speed and mem usage?

---

## Issue #N/A: Failed to load model

**Link**: https://github.com/ggml-org/llama.cpp/issues/38
**State**: closed
**Created**: 2023-03-12T06:27:53+00:00
**Closed**: 2023-03-12T07:36:58+00:00
**Comments**: 2
**Labels**: question

### Description

Hello,

I was playing with this trying to get it to work, but couldn't get the model to load. I used these instructions on my MBP M1 for the 13B model:

https://til.simonwillison.net/llms/llama-7b-m2

I get a "unknown tensor" error as shown:

```
./main \
  -m ./models/13B/ggml-model-q4_0.bin \
  -t 8 \
  -n 128 \
  -p 'The first person to go to space was '
main: seed = 1678602312
llama_model_load: loading model from './models/13B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 5120
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 40
llama_model_load: n_layer = 40
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 13824
llama_model_load: n_parts = 2
llama_model_load: ggml ctx size = 8559.49 MB
llama_model_load: memory_size =   800.00 MB, n_mem = 20480
llama_model_load: loading model part 1/2 from './models/13B/ggml-model-q4_0

[... truncated for brevity ...]

---

## Issue #N/A: can't compile main

**Link**: https://github.com/ggml-org/llama.cpp/issues/37
**State**: closed
**Created**: 2023-03-12T06:17:06+00:00
**Closed**: 2023-03-12T08:17:07+00:00
**Comments**: 2
**Labels**: build

### Description

I’m trying to compile main to play around with it and failing with error:

```
ld: symbol(s) not found for architecture arm64
clang: error: linker command failed with exit code 1 (use -v to see invocation)
```

on macOS M1

trying to compile by running  `g++ main.cpp -o main -v -std=c++11`

anyone know what I'm missing?

---

## Issue #N/A: convert-pth-to-ggml.py failed with RuntimeError

**Link**: https://github.com/ggml-org/llama.cpp/issues/35
**State**: closed
**Created**: 2023-03-12T05:47:11+00:00
**Closed**: 2023-03-12T20:23:48+00:00
**Comments**: 7
**Labels**: model

### Description

Hi there, I downloaded my LLaMa weights through bit-torrent, and tried to convert the 7B model to ggml FP16 format:
```
$python convert-pth-to-ggml.py models/7B/ 1 
normalizer.cc(51) LOG(INFO) precompiled_charsmap is empty. use identity normalization.
{'dim': 4096, 'multiple_of': 256, 'n_heads': 32, 'n_layers': 32, 'norm_eps': 1e-06, 'vocab_size': 32000}
n_parts =  1
Processing part  0
Traceback (most recent call last):
  File "/Users/fzxu/Documents/code/llama.cpp/convert-pth-to-ggml.py", line 89, in <module>
    model = torch.load(fname_model, map_location="cpu")
  File "/opt/anaconda3/envs/llama.cpp/lib/python3.10/site-packages/torch/serialization.py", line 712, in load
    return _load(opened_zipfile, map_location, pickle_module, **pickle_load_args)
  File "/opt/anaconda3/envs/llama.cpp/lib/python3.10/site-packages/torch/serialization.py", line 1049, in _load
    result = unpickler.load()
  File "/opt/anaconda3/envs/llama.cpp/lib/python3.10/site-packages/torch/serializ

[... truncated for brevity ...]

---

## Issue #N/A: benchmarks?

**Link**: https://github.com/ggml-org/llama.cpp/issues/34
**State**: closed
**Created**: 2023-03-12T05:20:58+00:00
**Closed**: 2024-04-09T01:10:24+00:00
**Comments**: 57
**Labels**: documentation, question, stale

### Description

Where are the benchmarks for various hardware - eg. apple silicon 

---

## Issue #N/A: What is the meaning of hacked?

**Link**: https://github.com/ggml-org/llama.cpp/issues/33
**State**: closed
**Created**: 2023-03-12T04:35:26+00:00
**Closed**: 2023-03-12T05:09:52+00:00
**Comments**: 5
**Labels**: question

### Description

Hey, I was reading your Readme.md and I saw that your repo was hacked. I want to ask what this means and wanted to check if the users like me also get the impact of hacking. Or, this is not the thing I should worry about?

---

## Issue #N/A: Linux Support

**Link**: https://github.com/ggml-org/llama.cpp/issues/30
**State**: closed
**Created**: 2023-03-12T02:00:26+00:00
**Closed**: 2023-03-12T06:32:50+00:00
**Comments**: 6
**Labels**: build

### Description

Will Linux be supported?

---

## Issue #N/A: ggml_new_tensor_impl: not enough space in the context's memory pool

**Link**: https://github.com/ggml-org/llama.cpp/issues/29
**State**: closed
**Created**: 2023-03-12T01:51:07+00:00
**Closed**: 2023-03-13T17:23:15+00:00
**Comments**: 16
**Labels**: wontfix

### Description

Heya! Friend showed this to me and I'm trying to get it to work myself on Windows 10. I've applied the changes as seen in #22 to get it to build (more specifically, I pulled in the new commits from [etra0's fork](https://github.com/etra0/llama.cpp), but the actual executable fails to run - printing this before segfaulting:

```
ggml_new_tensor_impl: not enough space in the context's memory pool (needed 458853944, available 454395136)
ggml_new_tensor_impl: not enough space in the context's memory pool (needed 458870468, available 454395136)
```

I'm trying to use 7B on an i9-13900K (and I have about 30 gigs of memory free right now), and I've verified my hashes with a friend. Any ideas? Thanks!

---

## Issue #N/A: Too slow on m2 MBA 16gb SSD 512GB

**Link**: https://github.com/ggml-org/llama.cpp/issues/28
**State**: closed
**Created**: 2023-03-11T22:57:41+00:00
**Closed**: 2023-04-16T09:21:51+00:00
**Comments**: 4
**Labels**: need more info

### Description

Hi, 

First of all, thanks for the tremendous work!

I just wanted to ask that compared to your demo, when I run the same input sentence, the speed difference is tremendously different. Is this because of the chipset difference between m1 pro and m2 or, you already knew this issue and trying to fix this?



---

## Issue #N/A: Fails to load 30B model after quantization

**Link**: https://github.com/ggml-org/llama.cpp/issues/27
**State**: closed
**Created**: 2023-03-11T22:35:55+00:00
**Closed**: 2023-03-12T00:46:04+00:00
**Comments**: 2
**Labels**: build

### Description

Trying the 30B model on an M1 MBP, 32GB ram, ran quantification on all 4 outputs of the converstion to ggml, but can't load the model for evaluaiton:
```llama_model_load: loading model from './models/30B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 6656
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 52
llama_model_load: n_layer = 60
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 17920
llama_model_load: ggml ctx size = 20951.50 MB
llama_model_load: memory_size =  1560.00 MB, n_mem = 30720
llama_model_load: tensor 'tok_embeddings.weight' has wrong size in model file
main: failed to load model from './models/30B/ggml-model-q4_0.bin'
llama_model_load: %
```



This issue does not happen when I run the 7B model.

---

## Issue #N/A: 13b model issue tensor 'tok_embeddings.weight' has wrong size in model file

**Link**: https://github.com/ggml-org/llama.cpp/issues/24
**State**: closed
**Created**: 2023-03-11T21:04:18+00:00
**Closed**: 2023-03-11T21:57:23+00:00
**Comments**: 4
**Labels**: build

### Description

I try the following with the latest master (6b2cb6302ffaf8264e33af1dc52e3ea54003e690)

```
python convert-pth-to-ggml.py models/13B/ 1
./quantize ./models/13B/ggml-model-f16.bin   ./models/13B/ggml-model-q4_0.bin 2
./quantize ./models/13B/ggml-model-f16.bin.1 ./models/13B/ggml-model-q4_0.bin.1 2
```

```
ls models/13B/
checklist.chk         consolidated.00.pth   consolidated.01.pth   ggml-model-f16.bin    ggml-model-f16.bin.1  ggml-model-q4_0.bin   ggml-model-q4_0.bin.1 params.json
```

```
./main -m ./models/13B/ggml-model-q4_0.bin -t 8 -n 128
main: seed = 1678568386
llama_model_load: loading model from './models/13B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 5120
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 40
llama_model_load: n_layer = 40
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 13824
llama_model_load: gg

[... truncated for brevity ...]

---

## Issue #N/A: Ability for `./main` to keep the model in memory and pass it more text

**Link**: https://github.com/ggml-org/llama.cpp/issues/23
**State**: closed
**Created**: 2023-03-11T21:00:25+00:00
**Closed**: 2024-04-09T01:10:26+00:00
**Comments**: 39
**Labels**: enhancement, stale

### Description

The `./main` program currently outputs text and then quits.

How hard would it be to add a mode where it could stay running and be ready to accept more text piped to standard input?

This could help avoid the overhead of loading the model again every time the script runs.

Maybe it could output the generated text followed by a marker of some sort when it's done, so a wrapping process could see when it's finished and available to send a new prompt for evaluation.

I'm interested in wrapping it in a tiny Python web server to give myself a UI for interacting with the model.

---

## Issue #N/A: Windows 64-bit, Microsoft Visual Studio - it works like a charm after those fixes!

**Link**: https://github.com/ggml-org/llama.cpp/issues/22
**State**: closed
**Created**: 2023-03-11T20:44:33+00:00
**Closed**: 2023-04-16T10:25:54+00:00
**Comments**: 40
**Labels**: enhancement, help wanted, good first issue, windows

### Description

First of all thremendous work Georgi! I managed to run your project with a small adjustments on:
- Intel(R) Core(TM) i7-10700T CPU @ 2.00GHz / 16GB as x64 bit app, it takes around 5GB of RAM.

<img width="622" alt="image" src="https://user-images.githubusercontent.com/95347171/224509962-6ed8d954-66bc-4531-8dd0-423cc2ee5e2c.png">

<img width="568" alt="image" src="https://user-images.githubusercontent.com/95347171/224510066-a8adccfa-d9db-4546-8efb-e69efc549b97.png">

Here is the list of those small fixes:

- main.cpp: added ggml_time_init() at start of main (division by zero otherwise)
- quantize.cpp: same as above at start of main (division by zero otherwise)
- ggml.c: #define QK 32 moved to dedicated define.h (should not be in .c)
- ggml.c: replace fopen with fopen_s (VS secure error message)
- ggml.c: below changes due to 'expression must be a pointer or complete object type':
1. 2x `(uint8_t*)(y` to: `((uint8_t*)y` 
2. 4x `(const uint8_t*)(x` to `((const uint8_t*)x`


[... truncated for brevity ...]

---

## Issue #N/A: Implement Flash Attention Option

**Link**: https://github.com/ggml-org/llama.cpp/issues/19
**State**: closed
**Created**: 2023-03-11T18:57:36+00:00
**Closed**: 2023-07-28T19:26:05+00:00
**Comments**: 4
**Labels**: enhancement

### Description

Would love to see a faster, more memory efficient attention implemented like Flash Attention. :)

---

## Issue #N/A: faster performance on older machines

**Link**: https://github.com/ggml-org/llama.cpp/issues/18
**State**: closed
**Created**: 2023-03-11T17:46:20+00:00
**Closed**: 2023-04-16T10:21:56+00:00
**Comments**: 20
**Labels**: question

### Description

On machines with smaller memory and slower processors, it can be useful to reduce the overall number of threads running. For instance on my MacBook Pro Intel i5 16Gb machine, 4 threads is much faster than 8. Try:

make -j && ./main -m ./models/7B/ggml-model-q4_0.bin -p "Author-contribution statements and acknowledgements in research papers should state clearly and specifically whether, and to what extent, the authors used AI technologies such as ChatGPT " -t 4 -n 512


---

## Issue #N/A: Output is garbage in INT4 model in Mac M1 Max

**Link**: https://github.com/ggml-org/llama.cpp/issues/15
**State**: closed
**Created**: 2023-03-11T14:52:54+00:00
**Closed**: 2023-03-11T15:50:16+00:00
**Comments**: 3
**Labels**: model, build

### Description

I'm not sure if the tokenizer is here to blame or something else, I've quantized the 7B model and running on my Mac and the output of any prompt is just garbage. 

```
❯ ./main -m ggml-model-q4_0.bin -t 10 -p "Building a website can be done in 10 simple steps:" -n 512
main: seed = 1678546145
llama_model_load: loading model from 'ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 4096
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 32
llama_model_load: n_layer = 32
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 11008
llama_model_load: n_parts = 1
llama_model_load: ggml ctx size = 4529.34 MB
llama_model_load: memory_size =   512.00 MB, n_mem = 16384
llama_model_load: loading model part 1/1 from 'ggml-model-q4_0.bin'
llama_model_load: .................................... done
llama_model_load: model size =  4017.27 MB / num tensors = 

[... truncated for brevity ...]

---

## Issue #N/A: tensor 'tok_embeddings.weight' has wrong size in model file

**Link**: https://github.com/ggml-org/llama.cpp/issues/14
**State**: closed
**Created**: 2023-03-11T14:10:15+00:00
**Closed**: 2023-03-11T14:16:26+00:00
**Comments**: 1
**Labels**: build

### Description

When trying to run the 13B model the following output is given:
```
main: seed = 1678543550
llama_model_load: loading model from './models/13B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 5120
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 40
llama_model_load: n_layer = 40
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 13824
llama_model_load: ggml ctx size = 8559.49 MB
llama_model_load: memory_size =   800.00 MB, n_mem = 20480
llama_model_load: tensor 'tok_embeddings.weight' has wrong size in model file
main: failed to load model from './models/13B/ggml-model-q4_0.bin'
```
I have followed the commands in the readme to quantize the model, i.e.:
```
python3 convert-pth-to-ggml.py models/13B/ 1
./quantize ./models/13B/ggml-model-f16.bin   ./models/13B/ggml-model-q4_0.bin 2
./quantize ./models/13B/ggml-model-f16.bin.1 ./models

[... truncated for brevity ...]

---

## Issue #N/A: [Q] Memory Requirements for Different Model Sizes

**Link**: https://github.com/ggml-org/llama.cpp/issues/13
**State**: closed
**Created**: 2023-03-11T12:19:07+00:00
**Closed**: 2023-03-18T21:02:00+00:00
**Comments**: 18
**Labels**: documentation, question

### Description

No description provided.

---

## Issue #N/A: Segfault / Memory error with 65B model (128GB RAM)

**Link**: https://github.com/ggml-org/llama.cpp/issues/12
**State**: closed
**Created**: 2023-03-11T11:14:41+00:00
**Closed**: 2023-03-11T11:18:20+00:00
**Comments**: 2
**Labels**: build

### Description

On an M1 Ultra / 128GB, running the 65B model:

```text
./main -m ./models/65B/ggml-model-q4_0.bin -t 8 -n 128 -p "The word empowerment has five possible definitions:"
```

produces this error after everything has been loaded correctly:

```text
ggml_new_tensor_impl: not enough space in the context's memory pool (needed 268478672, available 268435456)
```

30B runs fine (even on a 64GB M1 Max)

<details>
  <summary>Full output</summary>
  
  ```text
  (base) ➜  llama.cpp git:(master) ✗ ./main -m ./models/65B/ggml-model-q4_0.bin -t 8 -n 128 -p "The word empowerment has five possible definitions:"
main: seed = 1678533057
llama_model_load: loading model from './models/65B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 8192
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 64
llama_model_load: n_layer = 80
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2

[... truncated for brevity ...]

---

## Issue #N/A: Unicode support

**Link**: https://github.com/ggml-org/llama.cpp/issues/11
**State**: closed
**Created**: 2023-03-11T11:08:07+00:00
**Closed**: 2023-03-13T16:24:21+00:00
**Comments**: 38
**Labels**: bug, help wanted

### Description

Thannk you for creating such a great inference engine which has 10x speedup.
Please add Unocode support to display other language properly.

<img width="1129" alt="Screenshot 2023-03-11 at 7 12 50 PM" src="https://user-images.githubusercontent.com/2835415/224481064-49f6f114-6104-48df-ad30-9659be907c88.png">


---

## Issue #N/A: simde?

**Link**: https://github.com/ggml-org/llama.cpp/issues/10
**State**: closed
**Created**: 2023-03-11T11:05:50+00:00
**Closed**: 2023-03-12T06:24:14+00:00
**Comments**: 1
**Labels**: enhancement, hardware

### Description

Could [simde](https://github.com/simd-everywhere/simde) help with porting to x86?

---

## Issue #N/A: GPTQ Quantization (3-bit and 4-bit)

**Link**: https://github.com/ggml-org/llama.cpp/issues/9
**State**: closed
**Created**: 2023-03-11T10:00:01+00:00
**Closed**: 2023-07-28T19:25:46+00:00
**Comments**: 49
**Labels**: enhancement, help wanted

### Description

4-bit quantization tends to come at a cost of output quality losses. GPTQ quantization is a state of the art quantization method which results in negligible output performance loss when compared with the prior state of the art in 4-bit (and 3-bit/2-bit) quantization methods and even when compared with uncompressed fp16 inference.

![image](https://user-images.githubusercontent.com/5949853/224477466-2ee4a057-6130-4287-ab25-db38716d6519.png)

It would be good to see benchmarks on the existing implementation. It's possible there is substantial quality loss from the 4-bit quantization. It's also possible that it isn't very substantial. We'd have to see benchmarks to know. 

The related project GPTQ-for-LLaMA has some benchmarks available for their implementation.

Refernces:
[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
[The case for 4-bit precision: k-bit Inference Scaling Laws](https://arxiv.org/abs/2212.09

[... truncated for brevity ...]

---

## Issue #N/A: Is there a requirements.txt ?

**Link**: https://github.com/ggml-org/llama.cpp/issues/8
**State**: closed
**Created**: 2023-03-11T05:53:26+00:00
**Closed**: 2023-03-12T06:23:30+00:00
**Comments**: 4
**Labels**: question, wontfix

### Description

No description provided.

---

## Issue #N/A: Make run without error but ./model folder is empty

**Link**: https://github.com/ggml-org/llama.cpp/issues/7
**State**: closed
**Created**: 2023-03-11T05:06:05+00:00
**Closed**: 2023-03-12T06:23:23+00:00
**Comments**: 4
**Labels**: wontfix, model

### Description

Did I miss anything?

---

## Issue #N/A: Suppress output that isn't from the model

**Link**: https://github.com/ggml-org/llama.cpp/issues/5
**State**: closed
**Created**: 2023-03-11T01:36:02+00:00
**Closed**: 2023-03-13T16:39:58+00:00
**Comments**: 2
**Labels**: good first issue

### Description

I want to integrate this into a slim chat system, so I think it would be nice to be able to have the app output only the text from the model like a -q for "quiet" flag on run. 

---

## Issue #N/A: Repetition penalty

**Link**: https://github.com/ggml-org/llama.cpp/issues/4
**State**: closed
**Created**: 2023-03-10T22:52:54+00:00
**Closed**: 2023-03-12T09:27:44+00:00
**Comments**: 0
**Labels**: enhancement, help wanted

### Description

Hello, 

Thank you for this implementation, it is nice being able to experiment with things, even without GPUs at hand.

Would you mind implementing the repetition penalty? It seems to produce better/more consistent results...


---

## Issue #N/A: Windows VS2022 Build - Returning nonsense

**Link**: https://github.com/ggml-org/llama.cpp/issues/2
**State**: closed
**Created**: 2023-03-10T22:36:10+00:00
**Closed**: 2023-03-10T23:23:32+00:00
**Comments**: 7
**Labels**: build

### Description

Unsure if windows builds are expected to even function! 😄

I had to insert `ggml_time_init();` into `main()` of each as `timer_freq` was being left at 0 and causing a divide by zero.

Compiled with `cl main.cpp ggml.c utils.cpp /std:c++20 /DEBUG /EHsc`, same for quantize.cpp.

Run with the following `main.exe -m ./LLaMA/7B/ggml-model-q4_0.bin -t 32 -n 512 -p "Building a website can be done in 10 simple steps:\n"`

Produced the following output:

```
main: seed = 1678486056
llama_model_load: loading model from 'H:/downloads/manual/LLaMA/7B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 4096
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 32
llama_model_load: n_layer = 32
llama_model_load: n_rot   = 64
llama_model_load: f16     = 2
llama_model_load: n_ff    = 11008
llama_model_load: ggml ctx size = 4529.34 MB
llama_model_load: memory_size =   512.00 MB, n_mem = 16384


[... truncated for brevity ...]

---

## Issue #N/A: Merging tensors of larger models

**Link**: https://github.com/ggml-org/llama.cpp/issues/1
**State**: closed
**Created**: 2023-03-10T21:33:07+00:00
**Closed**: 2023-03-12T06:22:47+00:00
**Comments**: 4
**Labels**: enhancement

### Description

> Currently, only LLaMA-7B is supported since I haven't figured out how to merge the tensors of the bigger models. However, in theory, you should be able to run 65B on a 64GB MacBook

It shouldn't be hard to merge tensors with my https://github.com/kir-gadjello/zipslicer library, but it's pure Python! If you want to keep the project pure C++ you might want to write a standalone gist script that uses zipslicer to unpack weight shards into binary files.

---

