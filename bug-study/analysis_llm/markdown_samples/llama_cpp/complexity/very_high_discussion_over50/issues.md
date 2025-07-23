# very_high_discussion_over50 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 4
- Closed Issues: 26

### Label Distribution

- stale: 10 issues
- enhancement: 10 issues
- model: 8 issues
- good first issue: 4 issues
- help wanted: 4 issues
- bug-unconfirmed: 4 issues
- 🦙.: 3 issues
- research 🔬: 2 issues
- bug: 2 issues
- high priority: 2 issues

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

## Issue #N/A: Support for Phi-3 models

**Link**: https://github.com/ggml-org/llama.cpp/issues/6849
**State**: open
**Created**: 2024-04-23T15:22:53+00:00
**Comments**: 84
**Labels**: good first issue, model

### Description

Microsoft recently released Phi-3 models in 3 variants (mini, small & medium). Can we add support for this new family of models. 

---

## Issue #N/A: [Feature request] Any plans for AMD XDNA AI Engine support on Ryzen 7x40 processors?

**Link**: https://github.com/ggml-org/llama.cpp/issues/1499
**State**: closed
**Created**: 2023-05-17T09:57:42+00:00
**Closed**: 2025-04-07T01:09:20+00:00
**Comments**: 92
**Labels**: stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ ] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.


---
# Enhancement

Are there any plans to support the AMD XDNA AI Engine  (in AMD Ryzen 7x40 (x = 6,8,9) processors)?


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

## Issue #N/A: llama : add T5 (encoder-decoder) support

**Link**: https://github.com/ggml-org/llama.cpp/issues/5763
**State**: closed
**Created**: 2024-02-28T11:24:59+00:00
**Closed**: 2024-07-04T13:46:12+00:00
**Comments**: 52
**Labels**: model

### Description

Still not familiar with the details, but it seems it would be useful to support this architecture in `llama.cpp`. First, need to decide on the API and see what changes would be necessary

See discussion here: https://github.com/ggerganov/llama.cpp/issues/247

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

## Issue #N/A: Llama3 GGUF conversion with merged LORA Adapter seems to lose training data randomly

**Link**: https://github.com/ggml-org/llama.cpp/issues/7062
**State**: closed
**Created**: 2024-05-03T19:48:32+00:00
**Closed**: 2024-05-09T12:30:49+00:00
**Comments**: 147
**Labels**: bug-unconfirmed

### Description

I'm running Unsloth to fine tune LORA the Instruct model on llama3-8b .

1: I merge the model with the LORA adapter into safetensors
2: Running inference in python both with the merged model directly or the unsloth loaded model with the adapter on top of it produces correct outputs as per the fine tune

**Bug:** 
GGUF conversion of the merged model does not produce the same output. The GGUF has lost some of its fine tune data, while still maintaining most of it.  

I can ask it who it is, who created it etc. And it responds Llama and Meta as usual, but it incorporates the fine tuned speech style and humor into the response. This is not the case for my fine tuned model. 

1: I tried merging the LORA adapter with the original GGUF (non-fine tuned) using llama.cpp, the same results.
2: I tried running the server on the original GGUF (non-fine tuned) usling llama.cpp server and the adapter loaded into the server terminal command - same results.

It seemes that GGUF conversion 

[... truncated for brevity ...]

---

## Issue #N/A: Investigate gemma 2 generation quality

**Link**: https://github.com/ggml-org/llama.cpp/issues/8240
**State**: closed
**Created**: 2024-07-01T16:52:28+00:00
**Closed**: 2024-10-16T01:11:07+00:00
**Comments**: 90
**Labels**: enhancement, stale

### Description

Initial reports can be seen from https://github.com/ggerganov/llama.cpp/pull/8227

> [!IMPORTANT]  
> A note for everyone: if you think there's a bug in llama.cpp tokenizer, please make sure to test with HF `transformers` library first (see [this comment](https://github.com/ggerganov/llama.cpp/issues/8240#issuecomment-2212444937) for example)

---

## Issue #N/A: Support BitNet b1.58 ternary models

**Link**: https://github.com/ggml-org/llama.cpp/issues/5761
**State**: closed
**Created**: 2024-02-28T09:41:38+00:00
**Closed**: 2024-09-18T01:07:17+00:00
**Comments**: 90
**Labels**: enhancement, stale, Tensor Encoding Scheme

### Description

New paper just dropped on Arxiv describing a way to train models in 1.58 bits (with ternary values: 1,0,-1). Paper shows performance increases from equivalently-sized fp16 models, and perplexity nearly equal to fp16 models. Authors state that their test model is built on LLaMA architecture and can be easily adapted to llama.cpp.

[Edited to add: Further reading into it by fellow Redditors shows that we can't use this to quantize existing models trained to fp16. They'd have to be trained in this ternary mode from the start. But I think it would still be something that we should implement, because models of that flavor will be coming soon.]

This is all over Reddit /LocalLLaMA right now:

https://www.reddit.com/r/LocalLLaMA/comments/1b21bbx/this_is_pretty_revolutionary_for_the_local_llm/

I think, if my napkin math is right, it would let us run something like 120B models in 24 GB VRAM, or 30B in... 8 GB?

Please implement @ggerganov and friends!

https://arxiv.org/abs/2402.17

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

## Issue #N/A: Eval bug: GLM-Z1-9B-0414

**Link**: https://github.com/ggml-org/llama.cpp/issues/12946
**State**: closed
**Created**: 2025-04-14T18:28:14+00:00
**Closed**: 2025-05-25T16:12:57+00:00
**Comments**: 60
**Labels**: bug-unconfirmed

### Description

### Name and Version

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3080, compute capability 8.6, VMM: yes
version: 5121 (c94085df)
built with cc (Ubuntu 14.2.0-4ubuntu2) 14.2.0 for x86_64-linux-gnu

### Operating systems

Linux

### GGML backends

CUDA

### Hardware

RTX 3080

### Models

https://huggingface.co/ilintar/THUDM_GLM-Z1-9B-0414_iGGUF

Issue appears even with the highest quants (Q8_0).

### Problem description & steps to reproduce

After running the server (`llama-server --port 2345 --top-p 0.95 --temp 0.6 -nkvo -ngl 50 -c 32000 -m THUDM_GLM-Z1-9B-0414-Q5_K_M.gguf`, tried also with `--jinja`), the generation loops after producing ~100 tokens. 

![Image](https://github.com/user-attachments/assets/a0bc90fa-6baa-452a-8788-1615d98ec96c)

I tried the model with Transformers, using --load-in-4bit (because my VRAM is not enough to run it without quants) and it generated a c

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: Inconsistent Vulkan segfault

**Link**: https://github.com/ggml-org/llama.cpp/issues/10528
**State**: open
**Created**: 2024-11-26T19:54:03+00:00
**Comments**: 65
**Labels**: bug

### Description

### Name and Version

library 531cb1c233800e6acb021dc56d69595e314db072 (gguf-v0.4.0-2819-g531cb1c2)

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

_No response_

### Problem description & steps to reproduce

1. Compile the program below
2. Run it a thousand times and it will probably have a segmentation fault at least once. I used the `gdb` debugger.

Simple program:
```c
#include "llama.h"

static void handleLog(enum ggml_log_level level, const char *text, void *user_data) {}

int main(int argc, char **argv)
{
  llama_log_set(handleLog, 0);

  char path[] = "/your-path-to/llama.cpp/models/ggml-vocab-llama-bpe.gguf";
  struct llama_model_params params = llama_model_default_params();
  struct llama_model *model = llama_load_model_from_file(path, params);
  llama_free_model(model);

  return 0;
}
```

Shell script to run the program several times:
```sh
#! /bin/sh

PROGRAM=llama-bug
LOG=debug.log
COUNT=100

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

## Issue #N/A: Performance decreated between tag b1500 and b2581 on Windows ARM64 PC

**Link**: https://github.com/ggml-org/llama.cpp/issues/6417
**State**: closed
**Created**: 2024-04-01T03:20:36+00:00
**Closed**: 2024-07-08T01:06:56+00:00
**Comments**: 54
**Labels**: enhancement, stale

### Description

Hi LLAMA team, 

I use llama tag b2581 on Windows ARM64 PC, the performance is more lower than previous tag b1500. Please refer to below detailed information. What is the reason? Please help on this issue. 

Thanks a lot!

**[Detailed information]**

**Command:**
main.exe -m llama-2-7b-chat.ggufv3.q4_0.bin --color  --ctx_size 2048 -n -1 -ins -b 256 --top_k 10000 --temp 0.2 --repeat_penalty 1.1 -t 10

**Prompt:** I have 3 years of experience as a software developer. Now I got bored with coding and want to transition to another career. My education qualifications are B. Tech in computer science, and I am well-versed in understanding the business side of software as well. Suggest a list of career options that are easy for me to transition.


**system_info:** n_threads = 10 / 12 | AVX = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 0 | NEON = 1 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 0 | SSSE3 = 0 | VSX = 0 |

**Tag

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

## Issue #N/A: Feature Request: Support for Qwen2-VL

**Link**: https://github.com/ggml-org/llama.cpp/issues/9246
**State**: closed
**Created**: 2024-08-29T22:34:11+00:00
**Closed**: 2025-05-25T01:08:24+00:00
**Comments**: 131
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Qwen just released Qwen2-VL 2B & 7B under the Apache 2.0 License.

### Motivation

SoTA understanding of images of various resolution & ratio: Qwen2-VL achieves state-of-the-art performance on visual understanding benchmarks, including MathVista, DocVQA, RealWorldQA, MTVQA, etc.
Understanding videos of 20min+: Qwen2-VL can understand videos over 20 minutes for high-quality video-based question answering, dialog, content creation, etc.

### Possible Implementation

_No response_

---

## Issue #N/A: llama : add DeepSeek-v2-Chat support

**Link**: https://github.com/ggml-org/llama.cpp/issues/7118
**State**: closed
**Created**: 2024-05-07T06:22:43+00:00
**Closed**: 2024-05-28T15:07:06+00:00
**Comments**: 67
**Labels**: good first issue, model

### Description

please support deepseek-ai/DeepSeek-V2-Chat

https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat

---

## Issue #N/A: Feature Request: Qwen 2.5 VL

**Link**: https://github.com/ggml-org/llama.cpp/issues/11483
**State**: closed
**Created**: 2025-01-29T11:36:22+00:00
**Closed**: 2025-06-26T01:08:02+00:00
**Comments**: 74
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Is anybody implementing this? 

If not, I may give it a go. But it will take some time as I am new to the source side of llama.cpp/ggml.



### Motivation

Well, it's not currently working. :-)

### Possible Implementation

Based on the existing Qwen 2 VL implementation. 

---

## Issue #N/A: llama : add Falcon LLM support

**Link**: https://github.com/ggml-org/llama.cpp/issues/1602
**State**: closed
**Created**: 2023-05-26T17:45:06+00:00
**Closed**: 2023-08-23T20:11:44+00:00
**Comments**: 210
**Labels**: help wanted, model

### Description

Falcon LLM 40b and 7b were just open sourced under a license which allows commercial use (~~with royalties for over $1 million revenue per year~~) and have are topping the Huggingface Open LLM leaderboard. It seems to be based on a modified gpt3 architecture. I’m wondering if support in llama.cpp would be considered.

https://huggingface.co/tiiuae/falcon-40b

---

## Issue #N/A: Feature Request: add DeepSeek-v3 support

**Link**: https://github.com/ggml-org/llama.cpp/issues/10981
**State**: closed
**Created**: 2024-12-26T11:08:12+00:00
**Closed**: 2025-01-04T20:06:12+00:00
**Comments**: 64
**Labels**: enhancement

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- Version b4391
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Add support for DeepSeek-v3

https://huggingface.co/deepseek-ai/DeepSeek-V3

Currently not supported:

`ERROR:hf-to-gguf:Model DeepseekV3ForCausalLM is not supported`

### Motivation

DeepSeek-v3 is a big MoE model of 685B params, would be great as offloading to RAM would be a must for most systems

### Possible Implementation

There is no model card or technical report yet. I don't know how much different from v2 it is.

Edit: they have uploaded the mode

[... truncated for brevity ...]

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

## Issue #N/A: Eval bug: ~~Q2_K and Q3_K~~ Q8_0 not working on Vulkan anymore on RX 5700XT

**Link**: https://github.com/ggml-org/llama.cpp/issues/10710
**State**: closed
**Created**: 2024-12-07T17:08:41+00:00
**Closed**: 2025-05-17T01:07:56+00:00
**Comments**: 54
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 5700 XT (AMD proprietary driver) | uma: 0 | fp16: 1 | warp size: 64 | shared memory: 32768 | matrix cores: none
version: 4820 (1a24c462)
built with MSVC 19.42.34435.0 for x64

### Operating systems

Windows

### GGML backends

Vulkan

### Hardware

Ryzen 5900X + RX 5700 XT

### Models
 
Any model that has Q8_0 tensors in it.

### Problem description & steps to reproduce

Complete gibberish/noise output.

I noticed this issue with stable-diffusion.cpp at first, but I can reproduce it here. 

To reproduce, simply start inference with any q8_0 model, with `-ngl` set to anything but 0.

### First Bad Commit

fbeda90 (#12015)

### Relevant log output
Example command:

`.\build\bin\Release\llama-cli.exe -m .\models\gemma-2b-Q8_0.gguf -no-cnv -ngl 19 -t 6 -tb 12 -p "The meaning of life is"`

Output:

```
 The meaning of life is increa increa increa increa increa increa increa increa increa increa increa 

[... truncated for brevity ...]

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

## Issue #N/A: with the newest builds i only get gibberish output

**Link**: https://github.com/ggml-org/llama.cpp/issues/1735
**State**: closed
**Created**: 2023-06-07T08:06:19+00:00
**Closed**: 2023-06-15T08:50:50+00:00
**Comments**: 81
**Labels**: bug, high priority

### Description

After the CUDA refactor PR #1703 by @JohannesGaessler was merged i wanted to try it out this morning and measure the performance difference on my ardware.
I use my standard prompts with different models in different sizes.

I use the prebuild versions win-cublas-cu12.1.0-xx64

With the new builds I only get gibberish as a response for all prompts used and all models.
It looks like a random mix of words in different languages.

On my current PC I can only use the win-avx-x64 version, here I still get normal output.

I will use the Cuda-pc again in a few hours, then I can provide sample output or more details.
Am I the only one with this problem?

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

## Issue #N/A: Bug: QWEN2 quantization GGML_ASSERT

**Link**: https://github.com/ggml-org/llama.cpp/issues/7805
**State**: closed
**Created**: 2024-06-06T17:32:36+00:00
**Closed**: 2024-09-01T01:07:49+00:00
**Comments**: 74
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

When attempting to quantize [Qwen2 7B instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) to IQ2_XS I get the following assert:

```
GGML_ASSERT: ggml-quants.c:12083: grid_index >= 0
```

Anything I can provide to debug? Uploading the f32 file and imatrix now for recreation

Attempting IQ2_S now, ~~will update if it fails in the same way~~ update: it fails in the same way on the same block

### Name and Version

Version b3086, ubuntu 22.04

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
[ 327/ 339]              blk.27.attn_norm.weight - [ 3584,     1,     1,     1], type =    f32, size =    0.014 MB
[ 328/ 339]               blk.27.ffn_down.weight - [18944,  3584,     1,     1], type =    f32, converting to iq2_xs .. GGML_ASSERT: ggml-quants.c:12083: grid_index >= 0
GGML_ASSERT: ggml-quants.c:12083: grid_index >= 0
GGML_ASSERT: ggml-quants.c:12083: grid_index >= 0
GGML_ASSERT: ggml-q

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

## Issue #N/A: llama : add Mixtral support

**Link**: https://github.com/ggml-org/llama.cpp/issues/4381
**State**: closed
**Created**: 2023-12-08T18:20:09+00:00
**Closed**: 2023-12-13T12:04:31+00:00
**Comments**: 62
**Labels**: enhancement, high priority, model

### Description

Hi,
Please add support for [Mistral's MOE model Mixtral](https://twitter.com/MistralAI/status/1733150512395038967).

---

