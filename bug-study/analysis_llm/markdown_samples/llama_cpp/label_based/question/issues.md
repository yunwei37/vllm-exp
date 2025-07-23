# question - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- question: 30 issues
- stale: 10 issues
- model: 4 issues
- enhancement: 3 issues
- hardware: 3 issues
- need more info: 2 issues
- documentation: 2 issues
- 🦙.: 1 issues
- build: 1 issues
- server: 1 issues

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

## Issue #N/A: bf16 problem

**Link**: https://github.com/ggml-org/llama.cpp/issues/7365
**State**: closed
**Created**: 2024-05-18T13:02:22+00:00
**Closed**: 2024-07-05T01:06:40+00:00
**Comments**: 2
**Labels**: question, stale

### Description

I have a model that has been converted from the original to bf16...
now I want to make some quantization testing with that but quantize says:

cannot dequantize/convert tensor type bf16

I don't understand why, since bf16 and f16 are not that different...


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

## Issue #N/A: llama : understand why GPU results are different for different batch sizes

**Link**: https://github.com/ggml-org/llama.cpp/issues/3014
**State**: closed
**Created**: 2023-09-04T20:23:55+00:00
**Closed**: 2023-10-30T06:54:40+00:00
**Comments**: 6
**Labels**: question

### Description

I did the following experiment:

Run `perplexity` with the same input, but changing the batch size via the `-b` parameter.
Here are the results for the first few iterations on different backends:

```bash
# Q4_0 7B
# batch sizes: 16, 32, 64, 128, 256, 512

# CPU (M2, LLAMA_ACCELERATE=OFF):

[1]4.3233,[2]4.8256,[3]5.4456,[4]6.0456,[5]6.1772,[6]6.0762  # SIMD is off for n_batch = 16 (ggml_vec_dot_f16)
[1]4.3214,[2]4.8286,[3]5.4463,[4]6.0497,[5]6.1802,[6]6.0800
[1]4.3214,[2]4.8286,[3]5.4463,[4]6.0497,[5]6.1802,[6]6.0800
[1]4.3214,[2]4.8286,[3]5.4463,[4]6.0497,[5]6.1802,[6]6.0800
[1]4.3214,[2]4.8286,[3]5.4463,[4]6.0497,[5]6.1802,[6]6.0800
[1]4.3214,[2]4.8286,[3]5.4463,[4]6.0497,[5]6.1802,[6]6.0800

# Metal:

[1]4.3263,[2]4.8290,[3]5.4475,[4]6.0514,[5]6.1813,[6]6.0808,[7]6.2560,[8]6.3670,[9]6.7256,[10]6.9356
[1]4.3263,[2]4.8291,[3]5.4476,[4]6.0515,[5]6.1814,[6]6.0809,[7]6.2560,[8]6.3670,[9]6.7256,[10]6.9356
[1]4.3261,[2]4.8290,[3]5.4475,[4]6.0514,[5]6.1813,[6]6.0808,[7

[... truncated for brevity ...]

---

## Issue #N/A: Question: How to convert Yi-34B-Chat-4bits to gguf?

**Link**: https://github.com/ggml-org/llama.cpp/issues/7623
**State**: closed
**Created**: 2024-05-29T16:17:51+00:00
**Closed**: 2024-07-14T01:07:13+00:00
**Comments**: 2
**Labels**: question, stale

### Description

### Prerequisites

- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new useful question to share that cannot be answered within Discussions.

### Background Description

The script convert-hf-to-gguf.py can convert  Yi-34B-Chat,but can't convert the 4bits. My GPU only can SFT the 4bits modle,so I want to convert it to gguf. That report some error,the 4bits quantized  used AQW


xx@LLM:~/llama.cpp$ python convert-hf-to-gguf.py ~/.cache/modelscope/hub/Yi-34B-Chat-4bits/ --outfile ~/modles/Yi-34B-4bits.gguf
--------------------------------------------------------------------
INFO:hf-to-gguf:Set model tokenizer
INFO:gguf.vocab:Setting special token type bos to 1
INFO:gguf.vocab:Setting special token type eos to 2
INFO:gguf.vocab:Setting add_bos_token to False
INFO:gguf.vocab:Setting add_eos_token to

[... truncated for brevity ...]

---

## Issue #N/A: Problem/Question: Unable to add model with custom architecture (How to)

**Link**: https://github.com/ggml-org/llama.cpp/issues/7595
**State**: closed
**Created**: 2024-05-28T18:13:21+00:00
**Closed**: 2024-05-28T19:01:52+00:00
**Comments**: 1
**Labels**: question

### Description

### Prerequisites

- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new useful question to share that cannot be answered within Discussions.

### Background Description

I am aiming to add support for `jina-embeddings-v2-base-code` https://huggingface.co/jinaai/jina-embeddings-v2-base-code which has this architecture:

```
JinaBertModel(
  (embeddings): JinaBertEmbeddings(
    (word_embeddings): Embedding(61056, 768, padding_idx=0)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.0, inplace=False)
  )
  (encoder): JinaBertEncoder(
    (layer): ModuleList(
      (0-11): 12 x JinaBertLayer(
        (attention): JinaBertAttention(
          (self): JinaBertSelfAttention(
            (query): Linear

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

## Issue #N/A: On the edge llama?

**Link**: https://github.com/ggml-org/llama.cpp/issues/1052
**State**: closed
**Created**: 2023-04-19T01:24:08+00:00
**Closed**: 2023-04-23T12:46:33+00:00
**Comments**: 1
**Labels**: question, hardware

### Description

Sorry to ask this... But is possible to get llama.cpp working on things like edge TPU?

https://coral.ai/products/accelerator-module/

---

## Issue #N/A: Question: why llama.cpp mobilevlm model(fp16) inference result is different with official pytorch project results, this is normal?

**Link**: https://github.com/ggml-org/llama.cpp/issues/7614
**State**: closed
**Created**: 2024-05-29T09:59:02+00:00
**Closed**: 2024-07-29T01:06:54+00:00
**Comments**: 8
**Labels**: question, stale

### Description

### Prerequisites

- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new useful question to share that cannot be answered within Discussions.

### Background Description

llama.cpp run cmd：
./llava-cli -m /mnt/nas_data2/wb_space/MobileVLMV2/MobileVLM_V2-1.7B_bk/ggml-model-f32.gguf --mmproj /mnt/nas_data2/wb_space/MobileVLMV2/MobileVLM_V2-1.7B_bk/mmproj-model-f16.gguf --image /mnt/nas_data2/wb_space/MobileVLMV2/assets/samples/demo.jpg -p "please describe this images." --temp 0 --top-p 1 -c 4096
llama.cpp result： 
The image is a digital art piece that captures the essence of history through its depiction. It features an illustration from "The Story Of World History" by Susan Wise Bauer, Revised Edition: Volume II - From Rome to Middle Ages (Volume 2)

--------------------------------------------

[... truncated for brevity ...]

---

## Issue #N/A: Question: When using finetune LoRA to fine-tune the LLaMA3-7B-4bit GGUF model, why does the training prematurely end and save the LoRA model?

**Link**: https://github.com/ggml-org/llama.cpp/issues/7611
**State**: closed
**Created**: 2024-05-29T07:47:48+00:00
**Closed**: 2024-07-13T01:06:49+00:00
**Comments**: 1
**Labels**: question, stale

### Description

### Prerequisites

- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new useful question to share that cannot be answered within Discussions.

### Background Description

i uesd follow command:
`./finetune --model-base ../models/GGUF/Llama3-8B-Chinese-Chat-GGUF-4bit/Llama3-8B-Chinese-Chat-q4_0-v2_1.gguf --train-data /home/cp/gtools/dataset/train_data.txt --checkpoint-in  checkpoint-LATEST.gguf  --lora-out /home/cp/models/llama3-8-4bit-epoch10-528.gguf --save-every 50 --threads 16 --ctx 1024 --rope-freq-base 10000 --rope-freq-scale 1.0 --batch 1 --grad-acc 1 --adam-iter 56,080 --adam-alpha 0.001 --lora-r 4 --lora-alpha 4 --use-checkpointing --use-flash --sample-start "<s>" --escape --include-sample-start -ngl 0`

This is my second time starting training from a checkpoint, but it still ends prematurely. 

[... truncated for brevity ...]

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

## Issue #N/A: Question: Why do GPU and CPU embedding outputs differ for the same input? Is normal?

**Link**: https://github.com/ggml-org/llama.cpp/issues/7608
**State**: closed
**Created**: 2024-05-29T06:31:04+00:00
**Closed**: 2024-08-01T01:07:09+00:00
**Comments**: 2
**Labels**: question, stale

### Description

### Prerequisites

- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new useful question to share that cannot be answered within Discussions.

### Background Description

I am using the embedding example, the execution parameters are as follows
embedding.exe -ngl 200000 -m I:\JYGAIBIN\MetaLlamaModel\Llama2-13b-chat\ggml-model-f32_q4_1.gguf --log-disable -p "Hello World!"

The first three embedding values ​​are output when the CPU executes the embedding
-4.67528416e-08
-1.07059577e-06
1.76811977e-06

The first three embedding values ​​are output when the GPU (-ngl 200000) executes the embedding
5.86615059e-08
-1.02221782e-06
1.78800110e-06

Why are the same "Hello World!" inputs different? Does llama.cpp currently correctly support GPU and CPU embedding?

Also, does llama.cpp have specific i

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

## Issue #N/A: Question: how to make main to lead it work with my M3 E-cores instead of P-cores

**Link**: https://github.com/ggml-org/llama.cpp/issues/7577
**State**: closed
**Created**: 2024-05-28T01:59:13+00:00
**Closed**: 2024-07-12T01:17:44+00:00
**Comments**: 1
**Labels**: question, stale

### Description

### Prerequisites

- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new useful question to share that cannot be answered within Discussions.

### Background Description

I observed that on my apple M3, the default 4 threads run on P-core, but I want to run it on E-core. How do I do that?
You can see pin_cpu () in the makefile, but from the macro description it doesn't seem to work for Apple silicon, and I couldn't find anything else that works for apple silicon.
Thank you very much

### Possible Answer

Thread binding E-core

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

