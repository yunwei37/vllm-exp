# 🦙. - issues

**Total Issues**: 13
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 12

### Label Distribution

- 🦙.: 13 issues
- help wanted: 5 issues
- good first issue: 5 issues
- enhancement: 5 issues
- model: 3 issues
- high priority: 2 issues
- hardware: 1 issues
- research 🔬: 1 issues
- stale: 1 issues
- documentation: 1 issues

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

## Issue #N/A: llama.cpp + Final Jeopardy

**Link**: https://github.com/ggml-org/llama.cpp/issues/1163
**State**: closed
**Created**: 2023-04-24T19:44:16+00:00
**Closed**: 2023-04-28T16:13:35+00:00
**Comments**: 5
**Labels**: help wanted, good first issue, 🦙.

### Description

I was browsing reddit and saw this post:

https://www.reddit.com/r/LocalLLaMA/comments/12xkm9v/alpaca_vs_final_jeopardy/

If anyone is interested, it would be great to add such evaluation as an example to `llama.cpp` and add instructions for running it with different models: LLaMA, Alpaca, Vicuna, etc. and different quantizations.

Here is the original work by @aigoopy which can be a good starting point:

https://github.com/aigoopy/llm-jeopardy



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

