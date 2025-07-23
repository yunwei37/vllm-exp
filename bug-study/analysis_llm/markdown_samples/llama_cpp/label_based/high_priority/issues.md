# high_priority - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 3
- Closed Issues: 27

### Label Distribution

- high priority: 30 issues
- help wanted: 14 issues
- bug: 10 issues
- good first issue: 8 issues
- enhancement: 6 issues
- research 🔬: 3 issues
- refactoring: 2 issues
- model: 2 issues
- 🦙.: 2 issues
- generation quality: 2 issues

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

## Issue #N/A: Metal prompt processing / inference intermittently spins but doesn't produce output

**Link**: https://github.com/ggml-org/llama.cpp/issues/2678
**State**: closed
**Created**: 2023-08-20T10:01:09+00:00
**Closed**: 2024-02-29T21:27:01+00:00
**Comments**: 9
**Labels**: bug, high priority

### Description

The outward symptom is that prompt processing / inference spins up the GPU and churns up a ton of busy work but no tokens ever come out (at least not printables - I have seen a long string of \x1C before it stops responding entirely). It doesn't really "hang" forever because it eventually stops generating. It may happen immediately on initial prompt processing or during chat interaction. However once things go sour, it does not appear to recover with further input

Under the hood, I see GPU usage spike but no tokens get produced. ggml_metal_graph_compute() decides to start encoding a ton of "stuff" (the queue is flooded with nodes to process... far more than appropriate) but ggml_metal_get_tensor() never extracts anything meaningful. I would guess that something in the context is getting trashed. Unfortunately, setting threads to 1 does not avoid it. Moreover, it seems that ALL threads in the pool suddenly get very busy, not just one

UPDATE: Temp fix https://github.com/ggerganov/l

[... truncated for brevity ...]

---

## Issue #N/A: llama : combine expert tensors into a single tensor

**Link**: https://github.com/ggml-org/llama.cpp/issues/6082
**State**: closed
**Created**: 2024-03-15T12:55:03+00:00
**Closed**: 2024-04-03T13:07:06+00:00
**Comments**: 1
**Labels**: high priority, refactoring

### Description

Currently, we store separate tensors for each expert:

https://github.com/ggerganov/llama.cpp/blob/3020327f6cd6d2ce50528dd65f4b199d2ea8b1ae/ggml.c#L4442-L4455

This leads to large number of possible "source" tensors for the `_id` ops which increases significantly the size of `struct ggml_tensor` on the stack:

https://github.com/ggerganov/llama.cpp/blob/3020327f6cd6d2ce50528dd65f4b199d2ea8b1ae/ggml.h#L573-L576

Additionally, the Metal implementation is currently hacked to support up to 8 experts and extension to more than that is not completely obvious:

https://github.com/ggerganov/llama.cpp/blob/3020327f6cd6d2ce50528dd65f4b199d2ea8b1ae/ggml-metal.m#L1750-L1759

We should improve this, with one possible way being to store the data for the experts into a single tensor and address is with appropriate offsets

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

## Issue #N/A: Fix quantize_row_q4_1() with ARM_NEON

**Link**: https://github.com/ggml-org/llama.cpp/issues/876
**State**: closed
**Created**: 2023-04-10T14:40:14+00:00
**Closed**: 2023-04-10T16:30:16+00:00
**Comments**: 0
**Labels**: bug, high priority

### Description

It is currently bugged. See results of `quantize-stats` on M1:

```
$  ./quantize-stats -m models/7B/ggml-model-f16.bin 
Loading model
llama.cpp: loading model from models/7B/ggml-model-f16.bin
llama_model_load_internal: format     = ggjt v1 (latest)
llama_model_load_internal: n_vocab    = 32000
llama_model_load_internal: n_ctx      = 256
llama_model_load_internal: n_embd     = 4096
llama_model_load_internal: n_mult     = 256
llama_model_load_internal: n_head     = 32
llama_model_load_internal: n_layer    = 32
llama_model_load_internal: n_rot      = 128
llama_model_load_internal: f16        = 1
llama_model_load_internal: n_ff       = 11008
llama_model_load_internal: n_parts    = 1
llama_model_load_internal: model size = 7B
llama_model_load_internal: ggml ctx size =  59.11 KB
llama_model_load_internal: mem required  = 14645.07 MB (+ 2052.00 MB per state)
llama_init_from_file: kv self size  =  256.00 MB
note: source model is f16
testing 291 layers with max size 1310

[... truncated for brevity ...]

---

## Issue #N/A: Investigate the performance (speed and perplexity) of Q4_0 with 2x F16 factors

**Link**: https://github.com/ggml-org/llama.cpp/issues/995
**State**: closed
**Created**: 2023-04-15T12:24:00+00:00
**Closed**: 2023-04-22T08:43:17+00:00
**Comments**: 1
**Labels**: help wanted, high priority, research 🔬

### Description

The current `Q4_0` uses a single F32 floating-point scaling factor.

An idea was proposed by @ikawrakow to change this to use 2x F16 factors instead of 1x F32: https://github.com/ggerganov/llama.cpp/commit/679e1cb6c01b16abe4f3ee3c849813b98970df93
Initial results indicate that this might be as accurate as `Q4_1` and hopefully as fast as current `Q4_0`.

The goal of this task is to try to implement efficiently this data format (quantization, dequantization and dot product), measure the speed and perplexity and decide if this is viable. Depending on the results, we can think about updating the current `Q4_0` data format and potentially dropping support for `Q4_1`.

### SIMD implementation progress

- [x] ARM NEON
- [x] AVX
- [ ] WASM

I plan to work on the ARM NEON implementation.
If you want to help with any of the implementations, propose an implementation + results in a PR, summarizing the inference speed and the obtained perplexity of your implementation.

### Related

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

## Issue #N/A: Infill Incorrect Tokenization

**Link**: https://github.com/ggml-org/llama.cpp/issues/3503
**State**: closed
**Created**: 2023-10-06T13:45:19+00:00
**Closed**: 2023-10-10T07:31:22+00:00
**Comments**: 15
**Labels**: bug, high priority

### Description

I am comparing the tokenization of the codellama repository with the infill example of this repository.

The first example prompt from the codellama repository consists of the strings:

- Prefix: 'def remove_non_ascii(s: str) -> str:\n    \"\"\" '
- Suffix: '\n    return result\n'

Comparing the tokenization of both implementations results in:

- CodeLlama: 1 32007 822 3349 29918 5464 29918 294 18869 29898 29879 29901 851 29897 1599 851 29901 13 1678 9995 29871 32008 13 1678 736 1121 13 32009
- Llama.cpp: 32007 1 822 3349 29918 5464 29918 294 18869 29898 29879 29901 851 29897 1599 851 29901 13 1678 9995 29871 32008 1 29871 13 1678 736 1121 13 32009

There are two differences:

- The first two tokens are swapped (those are `prefix_id` and `bos` I think)
- Llama.cpp adds a `bos` token again after the `suffix_id` token and an additional 29871 (is this a space?)

I believe the latter is definitely wrong, as the [paper](https://ai.meta.com/research/publications/code-llama-o

[... truncated for brevity ...]

---

## Issue #N/A: Tokenization is not equal to Meta's tokenization.

**Link**: https://github.com/ggml-org/llama.cpp/issues/2310
**State**: closed
**Created**: 2023-07-21T18:07:02+00:00
**Closed**: 2023-09-20T07:03:24+00:00
**Comments**: 24
**Labels**: help wanted, high priority

### Description

I'm comparing the tokenization between original Meta repo and llama.cpp with LLaMA (also had same issue with LLaMA v2).

For example, tokenizing the prompt "Hello world" and " Hello world" gives the following:

> For prompt "Hello world":
llama.cpp tokenizer: [10994, 3186]
Meta tokenizer: [15043, 3186]
> 
> For prompt " Hello world":
llama.cpp tokenizer: [15043, 3186]
Meta tokenizer: [29871, 15043, 3186]

Exploring the tokens, doing the detokenization, I got:

> For tokens "[10994, 3186]":
llama.cpp tokenizer: |b'Hello world'|
Meta tokenizer: |Hello world|
> 
> For tokens "[15043, 3186]":
llama.cpp tokenizer: |b' Hello world'|
Meta tokenizer: |Hello world|
> 
> For tokens "[29871, 15043, 3186]":
llama.cpp tokenizer: |b'  Hello world'|
Meta tokenizer: | Hello world|
> 
> *Adding | to ease visualization.

Exploring each token above with the `id_to_piece` functionality:

> Looking the id_to_piece for llama.cpp:
id 10994 |b'Hello'|
id 3186 |b' world'|
id 15

[... truncated for brevity ...]

---

## Issue #N/A: main : add detailed trace to a log file

**Link**: https://github.com/ggml-org/llama.cpp/issues/2694
**State**: closed
**Created**: 2023-08-21T18:21:14+00:00
**Closed**: 2023-08-30T06:30:00+00:00
**Comments**: 12
**Labels**: enhancement, help wanted, good first issue, high priority

### Description

We desperately need detailed trace of what is being passed to the model and what has been generated in the `main` example.
Since we cannot print such verbose information along with the generated text, it has to be dumped to a separate log file.

It should contain info such as:
- input tokens
- top K tokens
- sampled token
- decisions for adding prefix / suffix / etc.
- decisions for ignoring EOS
- decisions for inserting reverse prompt
- etc.

Basically everything that happens in the state machine of `main` has to be traced so we can understand better what is going on

See: https://github.com/ggerganov/llama.cpp/pull/2689#issuecomment-1686807325

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

## Issue #N/A: CUDA/OpenCL error, out of memory when reload.

**Link**: https://github.com/ggml-org/llama.cpp/issues/1456
**State**: closed
**Created**: 2023-05-14T17:56:02+00:00
**Closed**: 2023-06-09T23:16:05+00:00
**Comments**: 25
**Labels**: bug, high priority, hardware

### Description

Hello folks,

When try `save-load-state` example with CUDA, error occured.
It seems to necessary to add something toward `llama_free` function.

`n_gpu_layers` variable is appended at main function like below.

```cpp
int main(int argc, char ** argv) {
    ...
    auto lparams = llama_context_default_params();

    lparams.n_ctx     = params.n_ctx;
    lparams.n_parts   = params.n_parts;
    lparams.n_gpu_layers = params.n_gpu_layers; // Add gpu layers count
    lparams.seed      = params.seed;
    ...
}
```

And tried to run as below.

```dos
D:\dev\pcbangstudio\workspace\my-llama\bin>save-load-state.exe -m ggml-vic7b-q4_0.bin -ngl 32
main: build = 548 (60f8c36)
llama.cpp: loading model from ggml-vic7b-q4_0.bin
llama_model_load_internal: format     = ggjt v2 (latest)
llama_model_load_internal: n_vocab    = 32000
llama_model_load_internal: n_ctx      = 512
llama_model_load_internal: n_embd     = 4096
llama_model_load_internal: n_mult     = 256
llama_model

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: The KV cache is sometimes truncated incorrectly when making v1/chat/completions API calls

**Link**: https://github.com/ggml-org/llama.cpp/issues/11970
**State**: open
**Created**: 2025-02-20T11:20:01+00:00
**Comments**: 45
**Labels**: bug, high priority

### Description

### Name and Version

> .\llama-server.exe --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
version: 4743 (d07c6213)
built with MSVC 19.29.30158.0 for

### Operating systems

Windows

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell
> .\llama-server.exe -m .\models\unsloth\DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf --verbose-prompt --dump-kv-cache --log-timestamps --log-prefix --verbose --alias 'DeepSeek-R1-UD-IQ1_S' --log-file DeepSeek-R1-UD-IQ1_S.log
```

### Problem description & steps to reproduce

When using the llama-server and its Web UI, sometimes parts of the KV cache are truncated when they shouldn't be. Steps to reproduce:

1. Start llama-server with a command such as:
 
```
.\llama-server.exe -m .\models\un

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

## Issue #N/A: Failed to convert Llama-v2 models 

**Link**: https://github.com/ggml-org/llama.cpp/issues/4493
**State**: closed
**Created**: 2023-12-16T08:47:53+00:00
**Closed**: 2024-04-02T01:10:54+00:00
**Comments**: 28
**Labels**: bug, high priority, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [Y] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [Y] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [Y] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [Y] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Successful converting Llama models using the following command:

``
python convert.py models/xxx
``

where xxx is the original trained Llama model downloaded from Facebook.

# Current Behavior

Cannot convert with errors (detailed below). 

I've found ther

[... truncated for brevity ...]

---

## Issue #N/A: The new tokenizer no longer encode space properly

**Link**: https://github.com/ggml-org/llama.cpp/issues/2721
**State**: closed
**Created**: 2023-08-22T18:01:00+00:00
**Closed**: 2023-08-22T21:10:43+00:00
**Comments**: 1
**Labels**: high priority

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

llama.tokenizer
```
Python 3.11.4 (main, Jul  5 2023, 13:45:01) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from llama.tokenizer import Tokenizer
>>> tokenizer = Tokenizer('tokenizer.model')
>>> tokenizer.enco

[... truncated for brevity ...]

---

## Issue #N/A: Bug: cannot find tokenizer merges in model file

**Link**: https://github.com/ggml-org/llama.cpp/issues/9692
**State**: closed
**Created**: 2024-09-30T02:31:24+00:00
**Closed**: 2024-10-08T03:14:42+00:00
**Comments**: 11
**Labels**: bug, high priority, high severity

### Description

### What happened?

When I use transformers==4.45.1 and convert llama.cpp to the file used by ollama, there is no error, but when I load the model with ollama, the error ollama cannot find tokenizer merges in model file appears

### Name and Version

所有版本

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

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

## Issue #N/A: struct ggml_tensor -> add a struct meta pointer into it (can replace padding)

**Link**: https://github.com/ggml-org/llama.cpp/issues/1093
**State**: closed
**Created**: 2023-04-20T21:37:19+00:00
**Closed**: 2023-07-28T19:56:55+00:00
**Comments**: 5
**Labels**: enhancement, high priority

### Description

Given that the tensor struct uses padding it's not nice to add any more information into it.
It currently has a static 8 byte padding at the end, that's perfect to be replaced by a pointer to a struct to store additional information.
For example a optional human readable tensor name (to be used in graph print) or a couple u_int8 to switch on or off features by tensor.
For example: _use_cublas_=0
This would allow to fine-control the usage of such a library instead of hard-coding it on through a define flag.
For example: _performance_type_=HIGH/LOW/MID
For example: _threads_override_=2

use_cublas could be initialized depending on the define as 1/0. 

I'd also move the task scheduling _n_task_ out and the _performance_ stuff 
The overhead to access a compact external struct should be zero.


I suppose all that stuff could also be added directly into the tensor struct but if we have to keep it aligned that's not nice.
Especially given 64/32 bit environments with different p

[... truncated for brevity ...]

---

## Issue #N/A: ci failing on main branch

**Link**: https://github.com/ggml-org/llama.cpp/issues/7403
**State**: closed
**Created**: 2024-05-20T00:58:44+00:00
**Closed**: 2024-05-20T15:11:40+00:00
**Comments**: 1
**Labels**: high priority, bug-unconfirmed

### Description

* https://github.com/ggerganov/llama.cpp/pull/7358
    - Server / server (THREAD, Debug) (push) Failing after 26m
        - https://github.com/ggerganov/llama.cpp/actions/runs/9141008261/job/25134947389 
        -  20 - test-backend-ops (Failed)
* https://github.com/ggerganov/llama.cpp/pull/7374
    - ggml-org / ggml-2-x86-cpu - failure 1 in 3:34.28
        - https://github.com/ggml-org/ci/tree/results/llama.cpp/1e/a2a0036e88172d6c8bf7e1a1989a03894dc955/ggml-2-x86-cpu#test_scripts_debug
        - ?? 
* https://github.com/ggerganov/llama.cpp/pull/7395
    - ggml-org / ggml-4-x86-cuda-v100 - failure 8 in 7:00.84
        - https://github.com/ggml-org/ci/tree/results/llama.cpp/d3/59f30921a9f62a0fd299c412ff3f270286fea6/ggml-4-x86-cuda-v100
        -  20 - test-backend-ops (Failed)

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

## Issue #N/A: Question: How to generate an MPS gputrace

**Link**: https://github.com/ggml-org/llama.cpp/issues/6506
**State**: open
**Created**: 2024-04-05T14:08:32+00:00
**Comments**: 10
**Labels**: help wanted, high priority

### Description

We're doing some work over at https://github.com/huggingface/candle to improve our Metal backend, I've been collecting various gputraces for the different frameworks and was wondering if there was a documented/known way to generate one for llama.cpp during model inference.

Specifically talking about this type of debugger output: https://developer.apple.com/documentation/xcode/metal-debugger

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

## Issue #N/A: Measure perplexity delta between Q4_0 and F16 "output" tensor

**Link**: https://github.com/ggml-org/llama.cpp/issues/1003
**State**: closed
**Created**: 2023-04-15T19:22:22+00:00
**Closed**: 2023-04-16T20:08:54+00:00
**Comments**: 2
**Labels**: help wanted, good first issue, high priority, generation quality

### Description

The last tensor of the transformer (called `output` in llama.cpp) is one of the biggest ones:

https://github.com/ggerganov/llama.cpp/blob/0ad964631f9b3970f1936008fcfb1eadef59c7ed/llama.cpp#L945

I wonder how the perplexity improves by keeping it in F16 format instead of quantizing that particular tensor

### Results

<details>
  <summary>Q4_0 M1 Pro (with BLAS) [655]6.2838 (i.e. reference)</summary>

```
$  make clean && make -j perplexity && time ./perplexity -m ./models/7B/ggml-model-q4_0.bin -f ./build/wiki.test.raw -t 8
I llama.cpp build info: 
I UNAME_S:  Darwin
I UNAME_P:  arm
I UNAME_M:  arm64
I CFLAGS:   -I.              -O3 -DNDEBUG -std=c11   -fPIC -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -Wno-unused-function -pthread -DGGML_USE_ACCELERATE
I CXXFLAGS: -I. -I./examples -O3 -DNDEBUG -std=c++11 -fPIC -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-multichar -pthread
I LDFLAGS:   -frame

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

