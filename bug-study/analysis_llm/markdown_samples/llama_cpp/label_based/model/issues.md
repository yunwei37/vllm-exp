# model - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 29

### Label Distribution

- model: 30 issues
- enhancement: 9 issues
- stale: 7 issues
- help wanted: 5 issues
- good first issue: 4 issues
- need more info: 2 issues
- bug: 2 issues
- duplicate: 2 issues
- generation quality: 2 issues
- research 🔬: 2 issues

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

## Issue #N/A: Will llama.cpp be able to use Phi-2 ?

**Link**: https://github.com/ggml-org/llama.cpp/issues/4437
**State**: closed
**Created**: 2023-12-13T12:02:56+00:00
**Closed**: 2023-12-18T17:27:49+00:00
**Comments**: 27
**Labels**: enhancement, good first issue, model

### Description

Surely we have to wait for a GGUF version, but in the meantime just curious about it

thanks

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

## Issue #N/A: csm : implement Sesame-based conversation example

**Link**: https://github.com/ggml-org/llama.cpp/issues/12392
**State**: closed
**Created**: 2025-03-14T14:49:46+00:00
**Closed**: 2025-05-14T01:07:48+00:00
**Comments**: 23
**Labels**: model, research 🔬, stale, tts

### Description

With the first Sesame CSM model [openly available](https://github.com/SesameAILabs/csm), we should implement a local example similar to their [online research demo](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice#demo). It seems that the released CSM model uses [Kyutai's Mimi](https://arxiv.org/abs/2410.00037) audio codec which we have to implement in a similar way as we did with the [WavTokenizer](https://github.com/ggml-org/llama.cpp/pull/10784). Next we can modify the [talk-llama](https://github.com/ggerganov/whisper.cpp/tree/master/examples/talk-llama) example to support audio generation with the CSM. This way we will be able to plug any LLM for the text response generation and use Sesame for speech input/output.

---

## Issue #N/A: Support Jais-13b-chat bilingual model

**Link**: https://github.com/ggml-org/llama.cpp/issues/3441
**State**: closed
**Created**: 2023-10-02T17:41:59+00:00
**Closed**: 2024-04-15T02:47:10+00:00
**Comments**: 7
**Labels**: model, stale

### Description

How to add support for [Jais-13b-chat bilingual LLM](https://huggingface.co/inception-mbzuai/jais-13b-chat)?

I am a n00b, and from what I can tell this Torch model uses a custom architecture (including a custom model class) and would need a special converter.

I'd be happy to collaborate with someone more experienced on making this happen. I can provide some sponsorship as well. Thanks!


---

## Issue #N/A: Make k-quants work with tensor dimensions that are not multiple of 256

**Link**: https://github.com/ggml-org/llama.cpp/issues/1919
**State**: closed
**Created**: 2023-06-18T07:32:22+00:00
**Closed**: 2023-06-26T16:43:09+00:00
**Comments**: 11
**Labels**: enhancement, model

### Description

As discussed in #1602, k-quants do not work for the Falcon-7B model. This is due to the fact that the number of columns in many tensors (`4544`) is not divisible by `256`, which is the super-block size of the k-quants.

It would be useful if k-quants could be adapted to work in such cases.  

---

## Issue #N/A: Implement Together Computer's Red Pajama 3B Base/Chat model

**Link**: https://github.com/ggml-org/llama.cpp/issues/1337
**State**: closed
**Created**: 2023-05-06T01:48:53+00:00
**Closed**: 2024-04-09T01:09:36+00:00
**Comments**: 23
**Labels**: model, stale

### Description

- [announcement][0]
- [base model, 3B][1]
- [instruct model, 3B][3]
- [chat model, 3B][2]

Hopefully this can be blazingly fast!

[0]: https://www.together.xyz/blog/redpajama-models-v1
[1]: https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1
[2]: https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1
[3]: https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1

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

## Issue #N/A: Investigate supporting starcode

**Link**: https://github.com/ggml-org/llama.cpp/issues/1326
**State**: closed
**Created**: 2023-05-04T21:04:22+00:00
**Closed**: 2023-05-18T12:34:49+00:00
**Comments**: 8
**Labels**: help wanted, model

### Description

Bigcode just released [starcoder](https://huggingface.co/bigcode/starcoder). This is a 15B model trained on 1T Github tokens. This seems like it could be an amazing replacement for gpt-3.5 and maybe gpt-4 for local coding assistance and IDE tooling!

More info: https://huggingface.co/bigcode

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

## Issue #N/A: Try whether OpenLLaMa works

**Link**: https://github.com/ggml-org/llama.cpp/issues/1291
**State**: closed
**Created**: 2023-05-02T21:53:20+00:00
**Closed**: 2024-04-09T01:09:41+00:00
**Comments**: 82
**Labels**: 🦙., model, stale

### Description

... or whether we need to tweak some settings

GitHub: https://github.com/openlm-research/open_llama

HuggingFace: https://huggingface.co/openlm-research/open_llama_7b_preview_300bt

---

edit: GGML models uploaded to HH by @vihangd => https://huggingface.co/vihangd/open_llama_7b_300bt_ggml

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

## Issue #N/A: llama : support Mamba-2

**Link**: https://github.com/ggml-org/llama.cpp/issues/7727
**State**: closed
**Created**: 2024-06-04T05:57:48+00:00
**Closed**: 2025-07-02T17:10:26+00:00
**Comments**: 1
**Labels**: model, research 🔬, roadmap

### Description

Mamba-2 is a new version of the Mamba architecture:

- Blog: https://tridao.me/blog/2024/mamba2-part1-model/
- Paper: https://arxiv.org/abs/2405.21060

---

## Issue #N/A: llama : add Deepseek support

**Link**: https://github.com/ggml-org/llama.cpp/issues/5981
**State**: closed
**Created**: 2024-03-10T18:56:56+00:00
**Closed**: 2024-05-08T17:03:57+00:00
**Comments**: 8
**Labels**: help wanted, good first issue, model

### Description

Support is almost complete. There is a dangling issue with the pre-tokenizer: https://github.com/ggerganov/llama.cpp/pull/7036

A useful discussion related to that is here: https://github.com/ggerganov/llama.cpp/discussions/7144

-----

## Outdated below

Creating this issue for more visibility

The main problem is around tokenization support, since the models use some variation of the BPE pre-processing regex. There are also some issues with the conversion scripts.

Anyway, looking for contributions to help with this

Previous unfinished work:

- #4070 
- #5464 

Possible implementation plan: https://github.com/ggerganov/llama.cpp/pull/5464#issuecomment-1974818993

---

## Issue #N/A: Request: Nougat OCR Integration

**Link**: https://github.com/ggml-org/llama.cpp/issues/3294
**State**: open
**Created**: 2023-09-21T06:29:29+00:00
**Comments**: 8
**Labels**: help wanted, model

### Description

# Request: Nougat OCR Integration

I suggest adding Nougat OCR into llama.cpp to enable the processing of scientific PDF documents. 
This can act as a first step towards adding multimodal models to this project!

Implementation:
It seems that Nougat is based on standard transformer architecture (like Bart and Swin Transformer) and most of the work would be on figuring out how to add the image processing.

Let me know what you think!
P.S.: Love this repo! I hope to add my own retrieval-pretrained transformer at some point to this repo.



---

## Issue #N/A: Add llama 2 model

**Link**: https://github.com/ggml-org/llama.cpp/issues/2262
**State**: closed
**Created**: 2023-07-18T16:35:53+00:00
**Closed**: 2023-10-18T07:31:45+00:00
**Comments**: 95
**Labels**: 🦙., model

### Description

Meta just released llama 2 model, allowing commercial usage

https://ai.meta.com/resources/models-and-libraries/llama/

I have checked the model implementation and it seems different from llama_v1, maybe need a re-implementation

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

## Issue #N/A: Add support for DBRX models: dbrx-base and dbrx-instruct

**Link**: https://github.com/ggml-org/llama.cpp/issues/6344
**State**: closed
**Created**: 2024-03-27T12:34:45+00:00
**Closed**: 2024-04-13T09:33:53+00:00
**Comments**: 36
**Labels**: enhancement, model

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

Databricks just released 2 new models called DBRX (base and instruct). They have their own architecture: 
```json
{
  "architectures": [
    "DbrxForCausalLM"
  ],
  "attn_config": {
    "clip_qkv": 8,
    "kv_n_heads": 8,
    "model_type": "",
    "rope_t

[... truncated for brevity ...]

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

## Issue #N/A: Blenderbot Support?

**Link**: https://github.com/ggml-org/llama.cpp/issues/3204
**State**: closed
**Created**: 2023-09-16T01:10:06+00:00
**Closed**: 2024-04-03T01:16:04+00:00
**Comments**: 2
**Labels**: enhancement, model, stale

### Description

Hello,
Would it be possible to support [Blenderbot](https://parl.ai/projects/blenderbot2/) ([model](https://huggingface.co/hyunwoongko/blenderbot-9B))?
Thank you!

---

## Issue #N/A: llama : add Refact support

**Link**: https://github.com/ggml-org/llama.cpp/issues/3061
**State**: closed
**Created**: 2023-09-07T13:54:25+00:00
**Closed**: 2023-10-04T13:23:41+00:00
**Comments**: 2
**Labels**: help wanted, good first issue, model

### Description

This is a new 1.6B code model: https://huggingface.co/smallcloudai/Refact-1_6B-fim

We should look into adding support into `llama.cpp` similar as to how we did for Falcon and Baichuan:

- #2717 
- #3009 

I haven't looked into the architecture yet, but I'm hoping it is similar to GPT, which we already know how to handle OK thanks to the Falcon experience. I see it also uses Alibi which we should have support for, but we haven't tested extensively, so there might be issues there.

---

## Issue #N/A: Is official mixtral 8x22b working properly? 

**Link**: https://github.com/ggml-org/llama.cpp/issues/6592
**State**: closed
**Created**: 2024-04-10T17:06:11+00:00
**Closed**: 2024-04-10T17:09:54+00:00
**Comments**: 3
**Labels**: enhancement, model

### Description

Is official mixtral 8x22b working properly with llamacpp? 

---

## Issue #N/A: Support for BioGPT

**Link**: https://github.com/ggml-org/llama.cpp/issues/1328
**State**: closed
**Created**: 2023-05-04T21:53:44+00:00
**Closed**: 2023-07-07T13:42:50+00:00
**Comments**: 4
**Labels**: model

### Description

I didn't see any prior discussion of this. A few months ago this GPT model [was released](https://github.com/microsoft/BioGPT) which seemed to do very well with medical Q&A. 

I don't see any indication that it's based on the existing GPT variations. There's a couple demos here: [Q&A demo](https://huggingface.co/spaces/katielink/biogpt-qa-demo) and [BioGPT-Large demo](https://huggingface.co/spaces/katielink/biogpt-large-demo).

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

## Issue #N/A: Support starcoder family architectures (1B/3B/7B/13B)

**Link**: https://github.com/ggml-org/llama.cpp/issues/3076
**State**: closed
**Created**: 2023-09-08T02:40:11+00:00
**Closed**: 2023-09-15T19:15:21+00:00
**Comments**: 6
**Labels**: model

### Description

Related Issues:

https://github.com/ggerganov/llama.cpp/issues/1901
https://github.com/ggerganov/llama.cpp/issues/1441
https://github.com/ggerganov/llama.cpp/issues/1326

Previously, it wasn't recommended to incorporate non-llama architectures into llama.cpp. However, in light of the recent addition of the Falcon architecture (see [Pull Request #2717](https://github.com/ggerganov/llama.cpp/pull/2717)), it might be worth reconsidering this stance.

One distinguishing feature of Starcoder is its ability to provide a complete series of models ranging from 1B to 13B. This capability can prove highly beneficial for speculative decoding and making coding models available for edge devices (e.g., M1/M2 Macs).

I can contribute the PR if it matches llama.cpp's roadmap.

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

