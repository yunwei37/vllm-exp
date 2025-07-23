# need_more_info - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- need more info: 30 issues
- stale: 8 issues
- bug-unconfirmed: 6 issues
- hardware: 2 issues
- server/webui: 2 issues
- bug: 2 issues
- model: 2 issues
- invalid: 1 issues
- question: 1 issues
- performance: 1 issues

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

## Issue #N/A: http://localhost:6800/jsonrpc

**Link**: https://github.com/ggml-org/llama.cpp/issues/3964
**State**: closed
**Created**: 2023-11-05T19:00:15+00:00
**Closed**: 2024-04-02T01:12:12+00:00
**Comments**: 1
**Labels**: invalid, need more info, stale

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

Please provide detailed informat

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

## Issue #N/A: quantize serious memory leak!

**Link**: https://github.com/ggml-org/llama.cpp/issues/6756
**State**: closed
**Created**: 2024-04-19T04:34:31+00:00
**Closed**: 2024-04-19T10:44:44+00:00
**Comments**: 2
**Labels**: need more info, bug-unconfirmed

### Description

I updated llama.cpp from somewhere around b2674 to HEAD a few hours ago, and now all my quantize processes grow without bounds (and usually get killed at around 500GB memory usage). Something serious has gone wrong in the last days.

[sorry for the short report, I am somewhat busy cleaning up all my servers. It must be an obvious problem though since it affected all quants I was running]

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

## Issue #N/A: [User] GGML_ASSERT: /opt/projects/llama.cpp/ggml.c:4796: view_src == NULL || data_size + view_offs <= ggml_nbytes(view_src) Aborted

**Link**: https://github.com/ggml-org/llama.cpp/issues/3305
**State**: closed
**Created**: 2023-09-22T03:13:45+00:00
**Closed**: 2024-04-03T01:15:48+00:00
**Comments**: 4
**Labels**: need more info, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior
To use llama.cpp running Wizardlm-13b-v1.2.

# Current Behavior
CMD: main -m ../ggml-model-f16.gguf -ngl 43 --interactive-first
After interactive 8 rounds, get the error:

GGML_ASSERT: /opt/projects/llama.cpp/ggml.c:4796: view_src == NULL || data_size + view_offs 

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

## Issue #N/A: Use GPU for prompt ingestion and CPU for inference

**Link**: https://github.com/ggml-org/llama.cpp/issues/1342
**State**: closed
**Created**: 2023-05-06T11:55:08+00:00
**Closed**: 2023-05-08T04:24:49+00:00
**Comments**: 2
**Labels**: need more info

### Description

Not sure if a Github issue is the right forum for this question, but was wondering if it's possible to use the GPU for prompt ingestion. I have an AMD GPU and with ClBlast I get about 3X faster ingestion on long prompts compared to a CPU.
But a 12-thread CPU is faster than the GPU for inference by around 30%.
Was wondering if I could combine the two so I can eat my cake and have it too!


---

## Issue #N/A: I found a regression in speed with latest update with WestSeverus 7B DPO GGUF

**Link**: https://github.com/ggml-org/llama.cpp/issues/5396
**State**: closed
**Created**: 2024-02-07T19:23:54+00:00
**Closed**: 2024-04-02T01:06:56+00:00
**Comments**: 2
**Labels**: need more info, stale

### Description

the quality seemed to improve but the speed is now much much slower. used to have 40t/s now only getting 9.50t/s. please look into it. thx

af3ba5d94627d337e32a95129e31a3064c459f6b is working fast but compared with latest, it's really slow

---

## Issue #N/A: AI suddenly interrupts his answer

**Link**: https://github.com/ggml-org/llama.cpp/issues/1030
**State**: closed
**Created**: 2023-04-17T15:42:57+00:00
**Closed**: 2023-04-18T09:13:40+00:00
**Comments**: 3
**Labels**: need more info

### Description

Hi everyone and huge respect for author because I've never thought that I get AI right inside my terminal. Especially on a shitty office laptop with 2 AMD cores and 2 gb of free RAM (I extend it later to almost 6).

Does anyone know, is it normal if the AI interrupts his answer? I asked him to write the Snake game and after dozens code strings it suddenly stopped and print ">". I don't know what happens, may be answer length is limited but the speed of answer was so good so I really thought that he completely write it. The code was fine, right that I asked.

llama.cpp version - latest release.
Model - vicuna.

Guys, you rock!

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

## Issue #N/A: Model conversion instructions in the https://github.com/ggerganov/llama.cpp/tree/master/examples/llava didn't work as expected

**Link**: https://github.com/ggml-org/llama.cpp/issues/4042
**State**: closed
**Created**: 2023-11-12T00:20:52+00:00
**Closed**: 2023-11-12T11:55:51+00:00
**Comments**: 6
**Labels**: need more info, bug-unconfirmed

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ ] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ ] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Please provide a detailed written description of what you were trying to do, and what you expected `llama.cpp` to do.

should split the LLaVA model to LLaMA and multimodel projector constituents

# Current Behavior

Please provide a detailed written description 

[... truncated for brevity ...]

---

## Issue #N/A: Bug: when arrive max ctx, model output garbage

**Link**: https://github.com/ggml-org/llama.cpp/issues/7578
**State**: closed
**Created**: 2024-05-28T02:22:16+00:00
**Closed**: 2024-06-18T03:20:17+00:00
**Comments**: 2
**Labels**: need more info, bug-unconfirmed, medium severity

### Description

### What happened?

This part has problem in cuda version. if set ngl>0, when arrive max ctx and next turn to chat, the model output garbage.

llama_kv_cache_seq_rm (ctx, 0, params.n_keep            , params.n_keep + n_discard);
llama_kv_cache_seq_add(ctx, 0, params.n_keep + n_discard, n_past, -n_discard);

if set ngl =0, everythings ok.
### Name and Version

llama.cpp-b3014
main.exe --version
version: 247 (6765407)
built with MSVC 19.37.32822.0 for x64

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

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

