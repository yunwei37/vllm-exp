# author_association_MEMBER - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 9
- Closed Issues: 21

### Label Distribution

- good first issue: 14 issues
- help wanted: 8 issues
- roadmap: 8 issues
- refactoring: 8 issues
- enhancement: 7 issues
- performance: 6 issues
- research 🔬: 5 issues
- high priority: 4 issues
- bug: 3 issues
- documentation: 2 issues

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

## Issue #N/A: metal : compile-time kernel args and params

**Link**: https://github.com/ggml-org/llama.cpp/issues/4085
**State**: open
**Created**: 2023-11-15T11:09:39+00:00
**Comments**: 4
**Labels**: performance, research 🔬, roadmap

### Description

I was just thinking about this idea, so writing it down for future research.

We should be able to fairly easy generate model-specific Metal code that has hardcoded kernels for every single node in the computation graph. The idea is to make an initial pass of a certain graph where we record all kernel calls with their respective argument values and parameters and then generate a model-specific MSL source file with all these kernels instances - either copy-paste or via templates. I guess this is something similar to what people call JIT. Wondering what kind of speed-up we will be able to see with this strategy.

---

## Issue #N/A: llama : refactor model loading code

**Link**: https://github.com/ggml-org/llama.cpp/issues/1991
**State**: closed
**Created**: 2023-06-25T10:30:31+00:00
**Closed**: 2023-08-21T20:22:19+00:00
**Comments**: 3
**Labels**: good first issue, refactoring

### Description

In `llama.cpp` we have logic for supporting some very old model formats and features such as sharded models which is making the code unnecessary complicated and difficult to maintain. We should simplify it and remove support for old stuff that is no longer used.

Additionally, with the upcoming unified file format (https://github.com/ggerganov/ggml/issues/220) we will have to look into reimplementing the code to use it and add support for loading non-LLaMA models as well. This will be an important step towards adding inference of new models such as MPT and Falcon. Therefore, simplifying the logic as much as possible will help to easily adopt the new unified file format when it is ready

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

## Issue #N/A: llama : save downloaded models to local cache

**Link**: https://github.com/ggml-org/llama.cpp/issues/7252
**State**: closed
**Created**: 2024-05-13T09:20:51+00:00
**Closed**: 2024-12-13T16:23:30+00:00
**Comments**: 8
**Labels**: enhancement, good first issue, examples

### Description

We've recently introduced the `--hf-repo` and `--hf-file` helper args to `common` in https://github.com/ggerganov/llama.cpp/pull/6234:

```
ref #4735 #5501 #6085 #6098

Sample usage:

./bin/main \
  --hf-repo TinyLlama/TinyLlama-1.1B-Chat-v0.2-GGUF \
  --hf-file ggml-model-q4_0.gguf \
  -m tinyllama-1.1-v0.2-q4_0.gguf \
  -p "I believe the meaning of life is" -n 32

./bin/main \
  --hf-repo TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  -m tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  -p "I believe the meaning of life is" -n 32

Downloads `https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.2-GGUF/resolve/main/ggml-model-q4_0.gguf` and saves it to `tinyllama-1.1-v0.2-q4_0.gguf`

Requires build with `LLAMA_CURL`
```

Currently, the downloaded files via `curl` are stored in a destination based on the `--model` CLI arg.

If `--model` is not provided, we would like to auto-store the downloaded model files in a local cache, similar to what other frameworks like HF/transfor

[... truncated for brevity ...]

---

## Issue #N/A: llama : move the sampling API from common into llama lib

**Link**: https://github.com/ggml-org/llama.cpp/issues/5214
**State**: closed
**Created**: 2024-01-30T12:44:03+00:00
**Closed**: 2024-09-07T12:17:24+00:00
**Comments**: 11
**Labels**: refactoring

### Description

There is functionality around `llama_sampling_context` currently part of `common`. We should move it into `llama`. Pretty much the entire API from `common/sampling.h` except `llama_sampling_params` and `llama_sampling_sample` can be integrated into the library.

This would probably require to also merge the grammar parser into the `llama` lib implementation.

The `llama_sampling_params` and `llama_sampling_sample` will stay in `common` since they are very example-specific and not general-purpose enough to be merged.

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

## Issue #N/A: Multi-thread the Q8_0 quantization in ggml_compute_forward_mul_mat_q_f32()

**Link**: https://github.com/ggml-org/llama.cpp/issues/1081
**State**: closed
**Created**: 2023-04-20T15:24:39+00:00
**Closed**: 2023-04-23T10:35:28+00:00
**Comments**: 1
**Labels**: enhancement, good first issue, performance

### Description

This part takes about 10% of the total inference time for 7B and it is currently single-threaded:

https://github.com/ggerganov/llama.cpp/blob/6a9661ea5ad72166b700ae5e87976e4452499dda/ggml.c#L7877-L7884

Try to multi-thread this by splitting the work across rows.
Since the `GGML_TASK_INIT` currently runs only 1 thread, either:
- update `ggml` to support multi-threaded `GGML_TASK_INIT`
- move the quantization in `GGML_TASK_COMPUTE` (might be difficult since no barrier mechanism)

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

## Issue #N/A: llama : try to avoid context swap

**Link**: https://github.com/ggml-org/llama.cpp/issues/2060
**State**: closed
**Created**: 2023-06-30T19:53:55+00:00
**Closed**: 2023-09-28T16:04:38+00:00
**Comments**: 2
**Labels**: performance, research 🔬

### Description

Currently, when the context becomes full, we pick part of the tokens and recompute the KV cache.

Instead, try to either:
- store non-RoPEd KV cache, "shift" it when the context is full and compute the RoPE over the entire cache for every new token taking into account the current positions
- store RoPEd KV cache (as we do now), "shift" it when the context is full and apply extra shift-RoPE on it (assuming RoPE is "additive")

---

## Issue #N/A: ggml : unified CMake build

**Link**: https://github.com/ggml-org/llama.cpp/issues/6913
**State**: open
**Created**: 2024-04-25T19:15:40+00:00
**Comments**: 4
**Labels**: enhancement, build, refactoring, roadmap

### Description

Currently the [ggml](https://github.com/ggerganov/ggml), [llama.cpp](https://github.com/ggerganov/llama.cpp) and [whisper.cpp](https://github.com/ggerganov/whisper.cpp) projects share the same source of the `ggml` library, but have different CMake scripts. The scripts are adapted to the specifics of the projects and are quite similar with each other - all of them build `ggml`. Still, there are differences due to manually rewriting them and applying changes from one repo to another

The goal in this task is to unify, deduplicate and streamline the build process of `ggml` with proper CMake scripts that are shared across the projects. This will simplify changes in the future and will also help other 3rd party projects that depend on `ggml`

More on this topic has been discussed in:

- https://github.com/ggerganov/llama.cpp/issues/5890
- https://github.com/ggerganov/ggml/pull/804

To achieve that, the `ggml`-related sources in `llama.cpp` and `whisper.cpp` would likely have to be 

[... truncated for brevity ...]

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

## Issue #N/A: server : remove self-extend features

**Link**: https://github.com/ggml-org/llama.cpp/issues/9859
**State**: closed
**Created**: 2024-10-12T07:11:13+00:00
**Closed**: 2024-10-12T13:06:32+00:00
**Comments**: 3
**Labels**: refactoring

### Description

The extra logic added to support this functionality is a bit questionable (https://github.com/ggerganov/llama.cpp/pull/5195#issuecomment-1917507112) and it introduces too much complexity around the context management. With new models available where the training context is plenty (32k and even 128k), we should remove this feature in view of simplifying the server implementation and potentially look to re-introduce it in the future in a better way.

---

## Issue #N/A: ggml : add GPU support for Mamba models

**Link**: https://github.com/ggml-org/llama.cpp/issues/6758
**State**: open
**Created**: 2024-04-19T06:47:35+00:00
**Comments**: 32
**Labels**: enhancement, help wanted, Nvidia GPU, roadmap

### Description

Recently, initial Mamba support (CPU-only) has been introduced in #5328 by @compilade 

In order to support running these models efficiently on the GPU, we seem to be lacking kernel implementations for the following 2 ops:

- `GGML_OP_SSM_CONV`
- `GGML_OP_SSM_SCAN`

Creating this issue to keep track of this and give more visibility of this feature. Help with implementing the missing kernels for CUDA and Metal (and other backends potentially) is welcome. We can also discuss if anything else is required to better support this architecture in `llama.cpp`

---

## Issue #N/A: llama.cpp BPE tokenization of wiki.test does not match the HF tokenization

**Link**: https://github.com/ggml-org/llama.cpp/issues/3502
**State**: closed
**Created**: 2023-10-06T13:42:05+00:00
**Closed**: 2024-04-06T01:06:25+00:00
**Comments**: 10
**Labels**: stale

### Description

I did the following test to tokenize `wiki.test.raw` using our tokenizer and the Python tokenizer.
The expectation is that the outputs will match:

```bash
# generate ggml-vocab-falcon.gguf
./convert-falcon-hf-to-gguf.py --vocab-only ~/development/huggingface/falcon-7b/ --outfile ./models/ggml-vocab-falcon.gguf

# tokenize using Python
python3 tests/test-tokenizer-0-falcon.py ~/development/huggingface/falcon-7b/ --fname-tok ./build/wikitext-2-raw/wiki.test.raw

# tokenize using llama.cpp
cd build
make -j
./bin/test-tokenizer-0-falcon ../models/ggml-vocab-falcon.gguf ./wikitext-2-raw/wiki.test.raw

# compare the results
cmp ./wikitext-2-raw/wiki.test.raw.tok ./wikitext-2-raw/wiki.test.raw.tokcpp 
./wikitext-2-raw/wiki.test.raw.tok ./wikitext-2-raw/wiki.test.raw.tokcpp differ: char 1, line 1
```

The results are pretty close, but not exactly the same. Any ideas why the test does not pass?
I thought that #3252 would resolve this

cc @goerch 

---

## Issue #N/A: llama : refactor llama_vocab

**Link**: https://github.com/ggml-org/llama.cpp/issues/9369
**State**: closed
**Created**: 2024-09-08T13:00:28+00:00
**Closed**: 2024-09-30T18:02:31+00:00
**Comments**: 3
**Labels**: good first issue, performance, refactoring

### Description

As of today we support 5 tokenizer implementations:

```c
        LLAMA_VOCAB_TYPE_SPM  = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
        LLAMA_VOCAB_TYPE_BPE  = 2, // GPT-2 tokenizer based on byte-level BPE
        LLAMA_VOCAB_TYPE_WPM  = 3, // BERT tokenizer based on WordPiece
        LLAMA_VOCAB_TYPE_UGM  = 4, // T5 tokenizer based on Unigram
        LLAMA_VOCAB_TYPE_RWKV = 5, // RWKV tokenizer based on greedy tokenization
```

The function `llama_tokenize_internal` in `llama-vocab.cpp` currently constructs a tokenizer instance on every call which for some of the tokenizers incurs significant overhead. This should be avoided by pre-constructing the tokenizer object upon `llama-vocab` creation and abstracting the objects (e.g. `llm_tokenizer_spm`, `llm_tokenizer_bpe`, etc.) with a common interface.

However, we want `llama_tokenize_internal` to remain thread-safe as it currently is (I think). Therefore, the tokenizer objects would likely need to b

[... truncated for brevity ...]

---

## Issue #N/A: llama : add example for tree-based parallel decoding

**Link**: https://github.com/ggml-org/llama.cpp/issues/3137
**State**: closed
**Created**: 2023-09-12T07:59:40+00:00
**Closed**: 2023-10-30T06:52:10+00:00
**Comments**: 7
**Labels**: performance, research 🔬

### Description

Refs:

- https://arxiv.org/pdf/2305.09781.pdf
- https://arxiv.org/pdf/2308.04623.pdf

In simple terms, after implementing [batched decoding (a.k.a. parallel decoding)](https://github.com/ggerganov/whisper.cpp/issues/1048) we can extend the inference functionality to support applying a custom attention mask to the batch. This can be used to create a causal tree mask that allows to evaluate a tree of continuations in a single pass, instead of a large batch of independent sequences.

This is useful for implementing advanced speculative strategies such as SpecInfer's token tree verification and [Medusa heads](https://sites.google.com/view/medusa-llm)

---

## Issue #N/A: tts : add support for Orpheus

**Link**: https://github.com/ggml-org/llama.cpp/issues/12476
**State**: open
**Created**: 2025-03-20T08:11:43+00:00
**Comments**: 5
**Labels**: good first issue, tts

### Description

HF: https://huggingface.co/collections/canopylabs/orpheus-tts-67d9ea3f6c05a941c06ad9d2

These TTS models seem suitable for supporting. To do that, we need to implement the SNAC audio codec: https://github.com/hubertsiuzdak/snac/

Sample implementation using Python-based inference of SNAC: https://github.com/isaiahbjork/orpheus-tts-local

Similar model support (OuteTTS): https://github.com/ggml-org/llama.cpp/pull/10784
Can be used as a reference how to implement this.

---

## Issue #N/A: llama : speed-up grammar sampling

**Link**: https://github.com/ggml-org/llama.cpp/issues/4218
**State**: open
**Created**: 2023-11-25T17:04:06+00:00
**Comments**: 40
**Labels**: performance, refactoring, roadmap

### Description

There have been a few reports where the grammar sampling can significantly degrade the performance.
It would be nice to profile and optimize the implementation - there should be room for improvements.

Already on-going efforts:

- #4210 
- #4213

Probably worth looking in multi-threading the implementation as well.

---

## Issue #N/A: server : display token probabilities in the UI

**Link**: https://github.com/ggml-org/llama.cpp/issues/2423
**State**: closed
**Created**: 2023-07-27T12:24:34+00:00
**Closed**: 2023-08-25T10:33:12+00:00
**Comments**: 0
**Labels**: help wanted, good first issue

### Description

Not sure how difficult it would be to add functionality where hovering over a generated token in the client to display a list of the top K candidate tokens and their probabilities.

Something like this:

![image](https://github.com/ggerganov/llama.cpp/assets/1991296/bc972914-f58d-4572-b336-63a6cb7fe520)

Would be a very useful addition.

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

## Issue #N/A: llama : enable FA by default and disable it per-layer

**Link**: https://github.com/ggml-org/llama.cpp/issues/10005
**State**: open
**Created**: 2024-10-22T14:07:59+00:00
**Comments**: 18
**Labels**: enhancement, roadmap

### Description

See the discussion starting here: https://github.com/ggerganov/llama.cpp/issues/9991#issuecomment-2428407002 and the proposed solution here: https://github.com/ggerganov/llama.cpp/issues/9991#issuecomment-2428868490.

Additionally, switch to F32 precision for the `K*Q` matrix multiplication by default.

Marking this as good first issue as an opportunity for new contributors, but also it is kind of high priority, so we should probably implement this in a day or two if there is no progress. @slaren or @JohannesGaessler in case you already started to work on it, fill free to assign to the issue and finish it.

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

## Issue #N/A: ggml : reintegrate the AMX backend into the CPU backend

**Link**: https://github.com/ggml-org/llama.cpp/issues/10359
**State**: closed
**Created**: 2024-11-17T11:35:11+00:00
**Closed**: 2025-01-01T01:07:39+00:00
**Comments**: 1
**Labels**: refactoring, stale

### Description

As explained here https://github.com/ggerganov/llama.cpp/pull/10343#issuecomment-2480834278, we would like to keep the CPU implementations inside the CPU backend. The AMX backend was created mainly because at the time we didn't support runtime weight repacking. Since now this functionality is supported, we should merge the AMX backend into the CPU backend.

The rough plan to achieve that is outlined here: https://github.com/ggerganov/llama.cpp/discussions/10350#discussioncomment-11282778

> The plan to reintegrate the AMX backend would be to create a new buffer type that converts the weights to the layout that the AMX backend needs them, and then check in the matrix multiplication the buffer type to determine if the AMX matrix multiplication code should be used. Basically extending the same that is done in https://github.com/ggerganov/llama.cpp/pull/9921 for the aarch64 types.

---

## Issue #N/A: ci : add Arm Cobalt 100 runners

**Link**: https://github.com/ggml-org/llama.cpp/issues/11275
**State**: closed
**Created**: 2025-01-17T09:17:03+00:00
**Closed**: 2025-02-22T11:09:50+00:00
**Comments**: 0
**Labels**: help wanted, good first issue, testing, roadmap

### Description

There are some new Github Actions runners "powered by the Cobalt 100-based processors":

https://github.blog/changelog/2025-01-16-linux-arm64-hosted-runners-now-available-for-free-in-public-repositories-public-preview/

Not sure what this processor is specifically, but it might have some Arm features that would be useful to exercise in the CI. We should look into more details and add workflows if it makes sense.

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

## Issue #N/A: gguf : enforce that tensor names are unique

**Link**: https://github.com/ggml-org/llama.cpp/issues/6836
**State**: closed
**Created**: 2024-04-22T22:44:14+00:00
**Closed**: 2024-04-28T15:36:19+00:00
**Comments**: 0
**Labels**: bug, good first issue

### Description

There are models being distributed with two tensors with the same name (eg. https://github.com/ggerganov/llama.cpp/issues/6490#issuecomment-2070935873). The gguf libraries should prevent this from happening.
- `gguf_add_tensor` (ggml) and `GGUFWriter` (gguf-py) should not allow adding duplicated tensors
- `gguf_init_from_file` (ggml) and `llama_model_loader` (llama.cpp) should reject loading GGUF files with duplicated tensors, including models with multiple shards

---

## Issue #N/A: server : add support for multiple responses

**Link**: https://github.com/ggml-org/llama.cpp/issues/11142
**State**: open
**Created**: 2025-01-08T16:11:24+00:00
**Comments**: 2
**Labels**: server/api, server, roadmap

### Description

It would be very useful to add multi-response support per slot so that a single request would be able to generate `n` independent completions. This functionality is useful in different situations - for example, a FIM completion can provide multiple alternative suggestions at a smaller or equal compute cost compared to running them sequentially.

I think this can be implemented by adding multiple sequence id per slot (instead of having just one like we currently do). However, I am not sure how yet much complexity would be introduced to support this.

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

## Issue #N/A: cuda : fix Falcon inference with offloaded KV cache

**Link**: https://github.com/ggml-org/llama.cpp/issues/2779
**State**: closed
**Created**: 2023-08-25T08:58:07+00:00
**Closed**: 2023-08-27T13:40:49+00:00
**Comments**: 4
**Labels**: bug, high priority

### Description

We seem to have a bug somewhere which currently prevents offloading the KV cache to the GPU when using Falcon.
See the discussion in https://github.com/ggerganov/llama.cpp/pull/2760 for more info

As a secondary task, we should fix the RoPE kernels to not waste half of the threads - again see the discussion

---

