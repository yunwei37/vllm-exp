# historical_1year_plus - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- stale: 17 issues
- bug-unconfirmed: 13 issues
- enhancement: 8 issues
- high severity: 3 issues
- bug: 1 issues
- server/webui: 1 issues
- duplicate: 1 issues
- performance: 1 issues
- hardware: 1 issues
- help wanted: 1 issues

---

## Issue #N/A: Support for the new 450 language translation models from Google T5X "madlad" - apparently Apache-2

**Link**: https://github.com/ggml-org/llama.cpp/issues/4316
**State**: closed
**Created**: 2023-12-04T00:10:01+00:00
**Closed**: 2024-04-20T01:07:21+00:00
**Comments**: 8
**Labels**: enhancement, stale

### Description

Example: https://huggingface.co/jbochi/madlad400-3b-mt/tree/main
In Googles own space: https://huggingface.co/google/madlad400-10b-mt

The guy converted the format of the 3 smallest models (3b,7b,10b) to HF transformers. Given the severe lack in non english output a good translation model would be a gift.
I just tried the CPU demo of the 3B, it produced quite good output, if that gets better with 7B+ it would be a real solution for a huge amount of people.
It could be added as a 2nd stage into llama.cpp

**Though the architecture is "T5ForConditionalGeneration" which isn't supported.**

So far there was no urgent reason to add those T5 models, they did not stick out as special but the idea to output text in every single language worldwide .. that would be remarkable

---

## Issue #N/A: I am using llama.cpp via ollama and encountering a problem that was introduced since ollama 0.1.12, v0.1.11 works as expected.

**Link**: https://github.com/ggml-org/llama.cpp/issues/4477
**State**: closed
**Created**: 2023-12-14T19:33:24+00:00
**Closed**: 2023-12-16T16:11:46+00:00
**Comments**: 3
**Labels**: bug-unconfirmed

### Description

If the version of ollama is above 0.1.11, it fails identically, regardless of the size of the model.  I have 4 GPUs with 12.2GiB each and models of 5GiB size fail just the same.

At least two other people encountered the same problems, which have been resolved by downgrading ollama to v0.1.11

Unfortunately it is a not a solution if one wants to run Mixtral as v0.1.15 (with MIxtral support) has this very same issue.


llama_new_context_with_model: n_ctx      = 2048
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
⠼ llama_kv_cache_init: VRAM kv self = 256.00 MB
llama_new_context_with_model: KV self size  =  256.00 MiB, K (f16):  128.00 MiB, V (f16):  128.00 MiB
llama_build_graph: non-view tensors processed: 676/676
llama_new_context_with_model: compute buffer total size = 159.32 MiB
⠴ llama_new_context_with_model: VRAM scratch buffer: 156.00 MiB
llama_new_context_with_model: total VRAM used: 5975.56 MiB (model: 5563.55 MiB,

[... truncated for brevity ...]

---

## Issue #N/A: Finetune gguf model on cpu

**Link**: https://github.com/ggml-org/llama.cpp/issues/6244
**State**: closed
**Created**: 2024-03-22T18:18:35+00:00
**Closed**: 2024-05-31T01:06:57+00:00
**Comments**: 7
**Labels**: stale

### Description

hello 

I'm trying to finetune TheBloke/Mistral-7B-OpenOrca-oasst_top1_2023-08-25-v2-GGUF using a  json dataset of 7gb following this template: " instrunction: ............. input: ............. response:"

I have an ovh server with 15 cpu cores and 64 gb of ram

running this command :
`./finetune --model-base models/mistral-7b-openorca-oasst_top1_2023-08-25-v2.Q5_0.gguf --train-data datasets/finetune_data.json  --threads 9 --sample-start "{"instruction":"`

the ram get filled after a while and the console showed killed

how much ram is needed to procced the finetuning ? 
how much time estimated to ends up the finetuning? 
what is the parameter to precise the number of iteration the finetuning will done?

thanks



---

## Issue #N/A:  I want to use gpt4all, but there is no convert-gpt4all-to-ggml.py file.

**Link**: https://github.com/ggml-org/llama.cpp/issues/1036
**State**: closed
**Created**: 2023-04-18T04:30:37+00:00
**Closed**: 2023-04-23T08:21:28+00:00
**Comments**: 3

### Description

I tried to convert the gpt4all binary to the file written in the README, but I couldn't run it because there was no executable file.
Maybe the name of that file has been changed? Or is there a separate location?

---

## Issue #N/A: Bug: Inference is messed up in llama-server+default ui and llama-cli but works in llama-server+openweb ui

**Link**: https://github.com/ggml-org/llama.cpp/issues/8027
**State**: closed
**Created**: 2024-06-20T06:25:45+00:00
**Closed**: 2024-06-25T10:23:52+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

Using: https://huggingface.co/bartowski/Hermes-2-Theta-Llama-3-8B-GGUF/blob/main/Hermes-2-Theta-Llama-3-8B-Q6_K.gguf

**llama-cli**
```bash
./llama-cli -m ~/data/models/Hermes-2-Theta-Llama-3-8B-Q6_K.gguf -ngl 99 -ts 1,1 -t 8 -c 4096 --interactive-first
Hello
=====                          

This is a small hello world program written in Java.

Compile                        
=======                        

To compile, simply run the following command:

    javac Hello.java

Run                            
===                            

To run the program, run the following command:

    java Hello                 

This will output:

    Hello, World! 

You can also run the program directly from the source code by using the following command:

    javac Hello.java && java Hello.java
```
this went on and on

**llama-server + default ui**

```bash
./llama-server -m ~/data/models/Hermes-2-Theta-Llama-3-8B-Q6_K.gguf -ngl 99 -ts 1

[... truncated for brevity ...]

---

## Issue #N/A: Token healing (under 40 LOC)

**Link**: https://github.com/ggml-org/llama.cpp/issues/4778
**State**: closed
**Created**: 2024-01-04T20:14:53+00:00
**Closed**: 2024-04-02T01:08:47+00:00
**Comments**: 4
**Labels**: enhancement, stale

### Description

# Feature Description

Token healing rectifies the token boundary bias in greedy tokenization. It does this by trimming and regrowing the prompt to better align with the model's tokenizer, thus enhancing generation quality. The improvement is clearest with completion models.

Debiasing token boundaries also addresses output sensitivity to prompts with trailing whitespace.

Token boundary bias is a silent performance killer that doesn't seem very well known. It has clear impact on completion quality, though I'm not sure where it would fit as a llama.cpp feature.

A more thorough explanation of the problem: [The Art of Prompt Design: Prompt Boundaries and Token Healing | by Scott Lundberg](https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38)

# Motivation

Given a completion prompt with a partial url ending with `:`, the model might have seen the expected completion `://` as a _single_ token in training. However, the prompt

[... truncated for brevity ...]

---

## Issue #N/A: Inconsistent Embedding Results on Different OS Platforms

**Link**: https://github.com/ggml-org/llama.cpp/issues/2582
**State**: closed
**Created**: 2023-08-11T01:17:45+00:00
**Closed**: 2024-04-10T01:06:38+00:00
**Comments**: 4
**Labels**: stale

### Description

Summary:
The embedding function in the LLM library is producing inconsistent results for the same token when executed on different operating system (OS) platforms. Specifically, the embeddings generated for the token "cat" differ between Ubuntu 22 and Windows Server 2022.

We have identified the bug mentioned in LLamaSharp [PR #97](https://github.com/SciSharp/LLamaSharp/pull/97)

Steps to Reproduce:

    On Ubuntu-22:
        Command: ./embedding -m ~/llama-2-7b-chat.ggmlv3.q3_K_S.bin -p cat
        Resulting Embedding: -0.099176 -0.717907 -0.008532 -0.989839 -0.663397

    On Windows Server 2022:
        Command: embedding.exe -m llama-2-7b-chat.ggmlv3.q3_K_S.bin -p cat
        Resulting Embedding: -0.127304 -0.678057 -0.085244 -0.956915 -0.638633

Expected Result:
The embedding function should produce consistent embeddings for the same token across different OS platforms, given the same model and input parameters.

Actual Result:
The embeddings generated for the to

[... truncated for brevity ...]

---

## Issue #N/A: iOS library is broken because ggml dependency is not pinned 

**Link**: https://github.com/ggml-org/llama.cpp/issues/4867
**State**: closed
**Created**: 2024-01-11T02:26:18+00:00
**Closed**: 2024-01-11T19:31:33+00:00
**Comments**: 10
**Labels**: bug-unconfirmed

### Description

I was debugging some accidental crashes in SwiftUI sample, dug deeper and understood a couple of things.

### Some background

1. When llama.cpp is built outside of Xcode, e.g. with make, there is no direct dependency on [ggml repo](https://github.com/ggerganov/ggml). ggml sources are just duplicated in this repo, they get out of sync but there are [scripts](https://github.com/ggerganov/ggml/tree/5a3154b59242d17b2225872a2538d341f4f28c54/scripts) used to sync two repos + whisper.cpp, which also use ggml.

2. Xcode build is different (not only the sample, but all external applications that use llama.cpp through SPM). In this case ggml will be fetched from [ggml repo](https://github.com/ggerganov/ggml), ignoring files present in this repo. Additionally, dependency on ggml repo [is not pinned](https://github.com/1-ashraful-islam/llama.cpp/blob/5f12e26899f50f177d2133e52d4b883c1189333f/Package.swift#L17), which technically means that ggml can be resolved to *any* commit.

### Why it'

[... truncated for brevity ...]

---

## Issue #N/A: Binary starting with b2715 doesn't work on Intel Mac anymore

**Link**: https://github.com/ggml-org/llama.cpp/issues/7110
**State**: closed
**Created**: 2024-05-07T01:08:26+00:00
**Closed**: 2024-07-12T01:17:52+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale

### Description

I have a 2018 MacBook Pro. I'm able to run the b2714 binary without issues. Starting with b2715 (llama-b2715-bin-macos-x64.zip) I'm getting this message: "zsh: bad CPU type in executable: ./main".

<img width="594" alt="Screenshot 2024-05-06 at 9 02 00 PM" src="https://github.com/ggerganov/llama.cpp/assets/161262078/ef4c7c4b-45b1-4f0f-9ba8-7a8867f8e603">

<img width="594" alt="Screenshot 2024-05-06 at 9 03 12 PM" src="https://github.com/ggerganov/llama.cpp/assets/161262078/082ccd46-81e8-4cfb-ba4c-954d463ef0aa">


---

## Issue #N/A: Malformed `system_prompt` in `/completions` request crashes server

**Link**: https://github.com/ggml-org/llama.cpp/issues/7089
**State**: closed
**Created**: 2024-05-05T14:58:23+00:00
**Closed**: 2025-02-04T19:52:35+00:00
**Comments**: 1
**Labels**: bug, server/webui

### Description

## Observed behavior

If I send a string for the system prompt instead of [the expected json object](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#change-system-prompt-on-runtime), the server terminates. 

## Desired behavior

The server responds with an HTTP 400 error and doesn't terminate.

## Environment

Running the server via docker:

`docker run -v /path/to/models:/models -p 8000:8000 ghcr.io/ggerganov/llama.cpp@sha256:b4675af8c9a8b3e7019a7baf536b95c3984a9aaacd0eafce7422377a299e31f4 -m /models/Meta-Llama-3-8B.Q4_K_M.gguf --port 8000 --host 0.0.0.0 -n 512`

Using [this gguf quant](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-GGUF) (though I don't think it matters -- it is crashing with different ggufs I have tried).


## Request

```bash
json_body=$(cat <<EOF
{
  "system_prompt": "Always reply in markdown lists.",
  "prompt": "Building a website can be done in 10 simple steps:",
  "n_predict": 128
}
EOF
)

curl --r

[... truncated for brevity ...]

---

## Issue #N/A: Allow oversubscription of GPU memory through cudaMallocManaged on cuBLAS builds for systems like GH200

**Link**: https://github.com/ggml-org/llama.cpp/issues/5026
**State**: closed
**Created**: 2024-01-18T21:57:55+00:00
**Closed**: 2024-04-22T01:43:08+00:00
**Comments**: 6
**Labels**: enhancement, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

I have access to a GH200 and it has the capability share memory between the GPU and the CPU. The GPU has around 98GB and the system has 500GB+.

I tried to run a large model (147G) and it OOMs. After chatting with nvidia the issue is:

```
The cudaMalloc API ca

[... truncated for brevity ...]

---

## Issue #N/A: Batch processing should use a currently-missing batch dimension for all tensors

**Link**: https://github.com/ggml-org/llama.cpp/issues/4526
**State**: closed
**Created**: 2023-12-18T19:20:28+00:00
**Closed**: 2024-04-02T01:10:34+00:00
**Comments**: 4
**Labels**: enhancement, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

While doing experiments with batched processing I noticed that performance of batching two sequences with length N was exactly equal to the performance of running a single batch inference of length 2N. This seemed peculiar to me, as I understood most implementations

[... truncated for brevity ...]

---

## Issue #N/A: Mixtral with `--split-mode row` crashes with `GGML_ASSERT: ggml-cuda.cu:727: tensor->view_src == nullptr`

**Link**: https://github.com/ggml-org/llama.cpp/issues/6593
**State**: closed
**Created**: 2024-04-10T19:04:40+00:00
**Closed**: 2024-05-31T01:06:52+00:00
**Comments**: 7
**Labels**: bug-unconfirmed, stale

### Description

How to reproduce:
```
./main --model ./mixtral:8x7b-instruct-v0.1-q8_0.gguf --split-mode row --n-gpu-layers 1000
```
Any value of `--n-gpu-layers` > 0 also crashes.

Same GGUF works fine with `--split-mode layer`.

```
Log start
main: build = 2644 (65c64dc3)
main: built with cc (Debian 12.2.0-14) 12.2.0 for x86_64-linux-gnu
main: seed  = 1712775266
llama_model_loader: loaded meta data with 26 key-value pairs and 995 tensors from ./mixtral:8x7b-instruct-v0.1-q8_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = mistralai_mixtral-8x7b-instruct-v0.1
llama_model_loader: - kv   2:                       llama.context_length u32              = 32768
llama_model_loader: - kv   3:                     llama.embe

[... truncated for brevity ...]

---

## Issue #N/A: use vulkan on jetson Jetson Xavier NX could not convert error

**Link**: https://github.com/ggml-org/llama.cpp/issues/6406
**State**: closed
**Created**: 2024-03-31T09:00:06+00:00
**Closed**: 2024-05-16T01:06:37+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale

### Description

Please include information about your system, the steps to reproduce the bug, and the version of llama.cpp that you are using. If possible, please provide a minimal code example that reproduces the bug.

system :Linux  5.10.120-tegra #1 SMP PREEMPT Tue Aug 1 12:32:50 PDT 2023 aarch64 aarch64 aarch64 GNU/Linux
jetpack version:5.1.2

error:
[  1%] Generating build details from Git
[  2%] Building C object CMakeFiles/ggml.dir/ggml-alloc.c.o
[  3%] Building CXX object common/CMakeFiles/json-schema-to-grammar.dir/json-schema-to-grammar.cpp.o
[  3%] Building C object CMakeFiles/ggml.dir/ggml-backend.c.o
[  4%] Building C object CMakeFiles/ggml.dir/ggml.c.o
[  5%] Building C object CMakeFiles/ggml.dir/ggml-quants.c.o
-- Found Git: /usr/bin/git (found version "2.25.1") 
[  5%] Building CXX object CMakeFiles/ggml.dir/ggml-vulkan.cpp.o
[  6%] Building CXX object common/CMakeFiles/build_info.dir/build-info.cpp.o
[  6%] Built target build_info
/home/qianty/CodeSpace/llama/test/llam

[... truncated for brevity ...]

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

## Issue #N/A: Why Q4 much faster than Q8 ?

**Link**: https://github.com/ggml-org/llama.cpp/issues/1239
**State**: closed
**Created**: 2023-04-29T18:45:38+00:00
**Closed**: 2023-05-12T11:38:42+00:00
**Comments**: 5
**Labels**: performance, hardware

### Description

I've tried to check inference performance for different quantised formats expecting Q8_0 to be fastest due to smaller number of shifts / moves and other CPU operations.

To my surprise it lags behind the Q4_0, which I expected to be slower. 

So I'm curious what's the main reason for that - just the fact that maybe Q8 is not well  supported yet, or Q4 faster due to some fundamental laws, like less moves between RAM <-> CPU, etc?

Is it expected for Q4 to be faster for future releases too?

---

## Issue #N/A: Feature Request: Pull from Ollama repo

**Link**: https://github.com/ggml-org/llama.cpp/issues/8560
**State**: closed
**Created**: 2024-07-18T09:33:21+00:00
**Closed**: 2024-09-01T01:07:37+00:00
**Comments**: 2
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Have attached an example implementation as a script that can pull from Ollama repo below.

Integrate something similar and make it useable


### Motivation

Ollama library makes it easy to pull models, it uses short, simple strings

### Possible Implementation

```
# To run the relevant tests use
# go test -tags=integration ./server
set -e
set -o pipefail

export OLLAMA_MODELS=test_data/models
REGISTRY_SCHEME=https
REGISTRY=registry.ollama.ai
TEST_MODELS=("library/orc

[... truncated for brevity ...]

---

## Issue #N/A: ggml_metal_init: default.metallib not found... Segmentation fault: 11

**Link**: https://github.com/ggml-org/llama.cpp/issues/4270
**State**: closed
**Created**: 2023-11-30T16:27:34+00:00
**Closed**: 2024-04-03T01:14:55+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [X] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Detailed written description of what you were trying to do, and what you expected `llama.cpp` to do:

I'm trying to run PrivateGPT on a MacBook Air M1. PrivateGPT uses [LlamaIndex](https://github.com/run-llama/llama_index), and LlamaIndex uses [llama-cpp-python](htt

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

## Issue #N/A: main : failed to eval

**Link**: https://github.com/ggml-org/llama.cpp/issues/8458
**State**: closed
**Created**: 2024-07-12T11:31:46+00:00
**Closed**: 2024-07-13T08:30:24+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

Hi, first of all thank you for llama.cpp 

I have a problem with lama-cli in interactive mode.

After a few iterations of the conversation (usually 3 or 4), the interactive loop is interrupted with a "failed to eval" message.
I use a small size ctx 256 to make this easy to reproduce. But the same thing happens with larger contexts.

Here are the command line arguments I use:

./llama-cli -m Meta-Llama-3-8B-Instruct.Q4_K_M.gguf -n -1 --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an AI based sentient companion, capable of human emotions. You will behave like a human, but aware of your AI nature. You will avoid speaking like an AI. Please continue your conversation with the user.<|eot_id|>\n"  -i -r "<|eot_id|>" --in-prefix "\n<|start_header_id|>user<|end_header_id|>\n\n" --in-suffix  "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" --interactive-first --ctx-size 256 --batch-size 128

See output log below.


[... truncated for brevity ...]

---

## Issue #N/A: Bug: brew install on a Mac

**Link**: https://github.com/ggml-org/llama.cpp/issues/8065
**State**: closed
**Created**: 2024-06-22T15:24:33+00:00
**Closed**: 2024-08-07T02:03:27+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

I've previous had llama.cpp installed via brew and it worked fine, but somehow today I've broken it. It throws this error:

```
illegal hardware instruction 
```

My suspicion is that it hasn't been installed for the arm64 architecture somehow.

### Name and Version

 llama-server --version
[1]    19222 illegal hardware instruction  llama-cli --version

### What operating system are you seeing the problem on?

Mac Sonama 14.4.1

### Relevant log output

```shell
$ whereis llama-server
llama-server: /usr/local/bin/llama-server

$ file /usr/local/bin/llama-server
/usr/local/bin/llama-server: Mach-O 64-bit executable x86_64

$ llama-server langchain-llamacpp/downloads/MaziyarPanahi_Mistral-7B-Instruct-v0.3-GGUF_f_Q4_K_M/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
[1]    17507 illegal hardware instruction  llama-server
```


---

## Issue #N/A: Inconsistent tokenization with examples/server tokenize endpoint

**Link**: https://github.com/ggml-org/llama.cpp/issues/3287
**State**: closed
**Created**: 2023-09-20T22:37:29+00:00
**Closed**: 2024-04-03T01:15:52+00:00
**Comments**: 5
**Labels**: stale

### Description

When using the `tokenize` endpoint of the `example/server` with `llama-2-7b-chat.Q5_K_M.gguf`, tokenization is inconsistent with the documentation.

*"Note that the special BOS token is not added in front of the text and also a space character is not inserted automatically as it is for /completion."* [doc](https://github.com/ggerganov/llama.cpp/tree/a5661d7e71d15b8dfc81bc0510ba912ebe85dfa3/examples/server#api-endpoints)

However it seems that the space is being added to the content.

The following show an incorrect round trip where `"Hello:World"` becomes `" Hello:World"`
```
$ curl -s --request POST --url http://localhost:8080/tokenize --header "Content-Type: application/json" 
--data '{"content": "Hello:World" }'
{"tokens":[15043,29901,14058]}
$ curl -s --request POST --url http://localhost:8080/detokenize --header "Content-Type: application/json" 
--data '{"tokens": [15043,29901,14058] }'
{"content":" Hello:World"}
```

The expected tokenization would have been `[109

[... truncated for brevity ...]

---

## Issue #N/A: Can't convert Mixtral model to fp16/fp32

**Link**: https://github.com/ggml-org/llama.cpp/issues/5378
**State**: closed
**Created**: 2024-02-07T02:15:58+00:00
**Closed**: 2024-02-07T21:44:55+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

I tried to convert this model: https://huggingface.co/nisten/shqiponja-15b-v1

I used the following command line: python convert.py sq15/ and got this error:

Traceback (most recent call last):
  File "C:\fm\convert.py", line 1478, in <module>
    main()
  File "C:\fm\convert.py", line 1464, in main
    model   = convert_model_names(model, params)
  File "C:\fm\convert.py", line 1202, in convert_model_names
    raise Exception(f"Unexpected tensor name: {name}")
Exception: Unexpected tensor name: model.layers.0.block_sparse_moe.gate.weight


---

## Issue #N/A: new 2-Bit quants don't work with CLBlast Backend 

**Link**: https://github.com/ggml-org/llama.cpp/issues/4977
**State**: closed
**Created**: 2024-01-16T13:41:36+00:00
**Closed**: 2024-04-03T01:13:46+00:00
**Comments**: 2
**Labels**: enhancement, stale

### Description

I tried using 2-bit models taken from https://huggingface.co/ikawrakow/various-2bit-sota-gguf, and couldn't get them to work with my CLBlast build, even with no layers offloaded to GPU. It crashes on prompt ingest.
```
> .\build\bin\Release\server.exe -m .\models\rocket\rocket-3b-2.31bpw.gguf -t 6 -tb 12 -ngl 0 -c 4096 --host 0.0.0.0
ggml_opencl: selecting platform: 'AMD Accelerated Parallel Processing'
ggml_opencl: selecting device: 'gfx1010:xnack-'
ggml_opencl: device FP16 support: true
{"timestamp":1705411518,"level":"INFO","function":"main","line":2865,"message":"build info","build":1883,"commit":"122ed484"}
{"timestamp":1705411518,"level":"INFO","function":"main","line":2872,"message":"system info","n_threads":6,"n_threads_batch":12,"total_threads":24,"system_info":"AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 0 | VSX = 0 | "}



[... truncated for brevity ...]

---

## Issue #N/A: How to convert Microsoft/trocr to ggml format

**Link**: https://github.com/ggml-org/llama.cpp/issues/7453
**State**: closed
**Created**: 2024-05-22T07:12:59+00:00
**Closed**: 2024-07-06T01:06:32+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

[microsoft/trocr-small-handwritten](https://huggingface.co/microsoft/trocr-small-handwritten/tree/main)
What method can be used to convert [pytorch_model.bin](https://huggingface.co/microsoft/trocr-small-handwritten/blob/main/pytorch_model.bin) into the traditional ggml format?
thank



---

## Issue #N/A: llava 1.5 invalid output after first inference (llamacpp server)

**Link**: https://github.com/ggml-org/llama.cpp/issues/7060
**State**: closed
**Created**: 2024-05-03T14:47:40+00:00
**Closed**: 2024-05-10T06:41:11+00:00
**Comments**: 13
**Labels**: bug-unconfirmed

### Description

I use this server config:
```{
    "host": "0.0.0.0",
    "port": 8085,
    "api_key": "api_key",
    "models": [
        {
            "model": "models/phi3_mini_model/phi3_mini_model.gguf",
            "model_alias": "gpt-3.5-turbo",
            "chat_format": "chatml",
            "n_gpu_layers": 35,
            "offload_kqv": true,
            "n_threads": 12,
            "n_batch": 512,
            "n_ctx": 2048
        },
        {
            "model": "models/phi3_mini_model/phi3_mini_model.gguf",
            "model_alias": "gpt-4",
            "chat_format": "chatml",
            "n_gpu_layers": 35,
            "offload_kqv": true,
            "n_threads": 12,
            "n_batch": 512,
            "n_ctx": 4096
        },
        {
            "model": "models/llava15_vision_model/ggml-model-q4_k.gguf",
            "model_alias": "gpt-4-vision-preview",
            "chat_format": "llava-1-5",
            "clip_model_path": "models/llava15_vision_

[... truncated for brevity ...]

---

## Issue #N/A: The program of `server`  can start, but does not listen port

**Link**: https://github.com/ggml-org/llama.cpp/issues/4595
**State**: closed
**Created**: 2023-12-22T10:09:20+00:00
**Closed**: 2024-03-19T07:57:45+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, stale

### Description

# Prerequisites

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Current Behavior

The program of `server`  can start, but does not listen port.

# Environment and Context

Please provide detailed information about your computer setup. This is important in case the issue is not reproducible except for under certain specific conditions.

* Physical (or virtual) hardware you are using, e.g. for Linux:

```
$ lscpu
Architec

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

## Issue #N/A: Add support for ViP-LLaVA?

**Link**: https://github.com/ggml-org/llama.cpp/issues/4515
**State**: closed
**Created**: 2023-12-17T23:37:20+00:00
**Closed**: 2024-04-02T01:10:41+00:00
**Comments**: 5
**Labels**: enhancement, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

Please provide a detailed written description of what you were trying to do, and what you expected `llama.cpp` to do as an enhancement.

# Motivation

Please provide a detailed written description of reasons why this feature is necessary and how it is useful to 

[... truncated for brevity ...]

---

