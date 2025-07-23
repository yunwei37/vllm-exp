# moderate_impact_3to10 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 2
- Closed Issues: 28

### Label Distribution

- enhancement: 10 issues
- bug-unconfirmed: 10 issues
- stale: 9 issues
- good first issue: 5 issues
- bug: 3 issues
- help wanted: 3 issues
- performance: 2 issues
- 🦙.: 1 issues
- build: 1 issues
- Intel GPU: 1 issues

---

## Issue #N/A: Feature Request: convert_hf_to_gguf.py to support model type Qwen2_5_VLForConditionalGeneration

**Link**: https://github.com/ggml-org/llama.cpp/issues/12642
**State**: closed
**Created**: 2025-03-29T09:49:09+00:00
**Closed**: 2025-05-13T01:07:48+00:00
**Comments**: 1
**Labels**: enhancement, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

$ ./convert_hf_to_gguf.py /data/models/Qwen2.5-VL-7B-Instruct --outtype auto --verbose --dry-run
INFO:hf-to-gguf:Loading model: Qwen2.5-VL-7B-Instruct
ERROR:hf-to-gguf:Model Qwen2_5_VLForConditionalGeneration is not supported

The model git lfs link:
https://cnb.cool/ai-models/Qwen/Qwen2.5-VL-7B-Instruct.git

### Motivation

The more type of model to support, the better.

### Possible Implementation

_No response_

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

## Issue #N/A: Misc. bug: [Mac M4]llama-server cannot run in release-4409 but can run in 4406

**Link**: https://github.com/ggml-org/llama.cpp/issues/11083
**State**: closed
**Created**: 2025-01-05T06:11:49+00:00
**Closed**: 2025-02-15T08:35:22+00:00
**Comments**: 0
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

 llama-b4409-bin-macos-arm64.zip 
 llama-b4406-bin-macos-arm64.zip 

### Operating systems

Mac

### Which llama.cpp modules do you know to be affected?

llama-server

### Problem description & steps to reproduce

4409 run log:
```
/Users/liwenbo/Downloads/4409-llamacpp/bin/llama-server -m /Users/liwenbo/models/qwen/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf
dyld[18622]: Library not loaded: @rpath/libllama.dylib
  Referenced from: <A6F705D2-0AC3-32BD-8CF2-3A55262E9195> /Users/liwenbo/Downloads/4409-llamacpp/bin/llama-server
  Reason: tried: '/Users/runner/work/llama.cpp/llama.cpp/build/src/libllama.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Users/runner/work/llama.cpp/llama.cpp/build/src/libllama.dylib' (no such file), '/Users/runner/work/llama.cpp/llama.cpp/build/ggml/src/libllama.dylib' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Users/runner/work/llama.cpp/llama.cpp/build/ggml/src/libllama.dylib' (no such file), '/Users/runner/work/llam

[... truncated for brevity ...]

---

## Issue #N/A: SYCL backend support Multi-card

**Link**: https://github.com/ggml-org/llama.cpp/issues/5282
**State**: closed
**Created**: 2024-02-02T12:01:33+00:00
**Closed**: 2024-03-05T15:43:14+00:00
**Comments**: 3
**Labels**: Intel GPU

### Description

### Discussed in https://github.com/ggerganov/llama.cpp/discussions/5277

<div type='discussions-op-text'>

<sup>Originally posted by **airMeng** February  2, 2024</sup>
Feel free to drop a note, let's know if you have any feature request or bugs (even unconfirmed)

- [ ] Multi-card Support
- [ ] Multi-batch Support [#5272](https://github.com/ggerganov/llama.cpp/issues/5272)
- [ ] CI test error for more than one GPU is detected and used.
  Current code returns all SYCL devices, including CPU, GPU (level-zero, opencl), FPGA. SYCL only support GPU. So when CI test on other devices, it will be fault.
- [ ] Support no-mmap parameter in other application. 
  There is known issue of SYCL: memcpy() from host (mmap) to device will hang in same cases. It's not resolved now. A work around solution is no use mmap. I have handled it in llama-bench (add --mmap parameter). We need add to more applications in examples.
- [ ] Clean code for warning and unused macro and variable.
  Sugges

[... truncated for brevity ...]

---

## Issue #N/A: Feature Request: Implement « Why Does the Effective Context Length of LLMs Fall Short? »

**Link**: https://github.com/ggml-org/llama.cpp/issues/10075
**State**: closed
**Created**: 2024-10-28T18:58:35+00:00
**Closed**: 2024-12-12T01:07:33+00:00
**Comments**: 1
**Labels**: enhancement, stale

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

(Amazing project, been using it since the very early days)

The paper:

https://arxiv.org/abs/2410.18745

In a word:

> Compared to commercial models, Llama 3.1 70B with \method even achieves better performance than GPT-4-128K and clearly surpasses Claude 2 and Kimi-chat. 

Extract: 

> Advancements in distributed training and efficient attention mechanisms have significantly expanded the context window sizes of large language models (LLMs). However, recent work reveals that the effec

[... truncated for brevity ...]

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

## Issue #N/A: SYCL NVidia build failing

**Link**: https://github.com/ggml-org/llama.cpp/issues/6026
**State**: closed
**Created**: 2024-03-12T17:54:13+00:00
**Closed**: 2024-03-19T09:39:40+00:00
**Comments**: 6
**Labels**: bug-unconfirmed

### Description

NVidia SYCL build failing due to multiple GPUs support PR.
This comes from https://github.com/ggerganov/llama.cpp/pull/5806/files#diff-6af12449fa63d10882b68b8230ff092164a786e01683813a005267630ab9c0b2R3330. 
CUDA versions translate the the SM number, so for SM80 the major version is 8.

I believe we should refrain from using versions if possible in the SYCL backend as it is less backend agnostic.

### Steps to reproduce:
`$./bin/test-backend-ops -b SYCL0`
```
ggml_backend_register: registered backend CPU
terminate called after throwing an instance of 'sycl::_V1::invalid_parameter_error'
  what():  DeviceList is empty. -30 (PI_ERROR_INVALID_VALUE)
Aborted
```

---

## Issue #N/A: [Feature request] Adding Support for "XVERSE" model

**Link**: https://github.com/ggml-org/llama.cpp/issues/4337
**State**: closed
**Created**: 2023-12-05T11:09:22+00:00
**Closed**: 2024-03-29T16:09:19+00:00
**Comments**: 2
**Labels**: enhancement, stale

### Description

"**XVERSE**" is a new Large Language Model which seems to be better than Llama-2.
See 	http://www.xverse.cn/ and https://github.com/xverse-ai

Actually, it is the best open-source model currently available according to the authors.

Model (for Huggingface Transformers library) with 65B ,13B and 7B parameters is available at :
**https://huggingface.co/xverse/XVERSE-65B**
**https://huggingface.co/xverse/XVERSE-13B**
https://huggingface.co/xverse/XVERSE-7B
Would be great if it would be supported also in llama.cpp.
Thanks!

As using GGUF files is a breaking change and the XVERSE model should be supported,
I think adding support for  XverseForCausalLM architecture to convert-hf-to-gguf.py is essential.

---

## Issue #N/A: Request: Allow for adjustments at the layer-level, for a practically two-fold increase in LLM handling ability by prompters

**Link**: https://github.com/ggml-org/llama.cpp/issues/4843
**State**: closed
**Created**: 2024-01-09T19:57:09+00:00
**Closed**: 2024-04-04T01:06:50+00:00
**Comments**: 9
**Labels**: enhancement, stale

### Description

# Feature Description

The project ["Brain Hacking Chip"](https://github.com/SoylentMithril/BrainHackingChip) demonstrates a sophisticated, albeit conceptually simple method of manipulating LLM inference, for a powerful increase in obedience. It has great potential to practically double a prompter's ability to guide an LLM toward desirable behaviors, because it allows for a prompter to *directly discourage* undesirable behaviors, without implying those undesirable behaviors are even possibilities.

It is my understanding that this kind of feature is currently very difficult to implement into LLaMA-CPP.

# Motivation

The "Brain Hacking Chip" project allows for negative prompts, which have been [demonstrated](https://github.com/SoylentMithril/BrainHackingChip#explain-softmax-with-an-instruction-to-type-in-l33t-sp34k) by the creator to allow for immediate gains in model obedience. I think this is significant, because negative prompting is relatively intuitive and accessible, espe

[... truncated for brevity ...]

---

## Issue #N/A: The problem with the conversion with the new convert.py

**Link**: https://github.com/ggml-org/llama.cpp/issues/966
**State**: closed
**Created**: 2023-04-14T13:52:17+00:00
**Closed**: 2023-04-15T21:53:22+00:00
**Comments**: 8

### Description

Hello! Help me figure out:

F:\Models\digitous-Alpacino13b>convert.py --dump-single F:\Models\digitous-Alpacino13b\4bit.safetensors
Traceback (most recent call last):
  File "F:\Models\digitous-Alpacino13b\convert.py", line 1145, in <module>
    main()
  File "F:\Models\digitous-Alpacino13b\convert.py", line 1116, in main
    model_plus = lazy_load_file(args.model)
  File "F:\Models\digitous-Alpacino13b\convert.py", line 853, in lazy_load_file
    return lazy_load_safetensors_file(fp, path)
  File "F:\Models\digitous-Alpacino13b\convert.py", line 753, in lazy_load_safetensors_file
    model = {name: convert(info) for (name, info) in header.items()}
  File "F:\Models\digitous-Alpacino13b\convert.py", line 753, in <dictcomp>
    model = {name: convert(info) for (name, info) in header.items()}
  File "F:\Models\digitous-Alpacino13b\convert.py", line 745, in convert
    assert 0 <= begin <= end <= len(byte_buf)
AssertionError

What is the error here - in the script or may

[... truncated for brevity ...]

---

## Issue #N/A: Add UC Berkleys Large World Models

**Link**: https://github.com/ggml-org/llama.cpp/issues/5659
**State**: closed
**Created**: 2024-02-22T08:30:06+00:00
**Closed**: 2024-04-12T01:06:46+00:00
**Comments**: 5
**Labels**: enhancement, stale

### Description

Can someone add support for the Large World Models(Text and Multimodal) from UC Berkley?
https://largeworldmodel.github.io/

---

## Issue #N/A: Issue loading Metal kernels from shared library

**Link**: https://github.com/ggml-org/llama.cpp/issues/1769
**State**: closed
**Created**: 2023-06-09T03:08:27+00:00
**Closed**: 2023-06-10T14:47:37+00:00
**Comments**: 2

### Description

Just looking for some help on this issue integrating the new Metal support as I'm not really familiar with MacOS.

The issue, orignally reported [here](https://github.com/abetlen/llama-cpp-python/issues/317), seems to be that path resolution for the `ggml-metal.metal` file fails when `llama.cpp` is built as a shared library. This produces the following output 

```
llama_model_load_internal: mem required  = 2532.67 MB (+ 3124.00 MB per state)
....................................................................................................
llama_init_from_file: kv self size  = 3120.00 MB
ggml_metal_init: allocating
ggml_metal_init: using MPS
ggml_metal_init: loading '(null)'
ggml_metal_init: error: Error Domain=NSCocoaErrorDomain Code=258 "The file name is invalid."
```

It seems the issue stems from [ggml_metal_init](https://github.com/ggerganov/llama.cpp/blob/master/ggml-metal.m#L108) calling `[[NSBundle mainBundle] pathForResource:@"ggml-metal" ofType:@"metal"]` to l

[... truncated for brevity ...]

---

## Issue #N/A: Truncated model files can cause llama.cpp to crash when using mmap

**Link**: https://github.com/ggml-org/llama.cpp/issues/6774
**State**: closed
**Created**: 2024-04-19T20:25:00+00:00
**Closed**: 2024-04-25T13:23:48+00:00
**Comments**: 0
**Labels**: bug, help wanted, good first issue

### Description

`llama_model_loader` does not check if the tensor data is present in the file when using mmap, and in some cases this can cause llama.cpp to crash.

To fix this, `llama_model_loader` should check that all the tensor data is within the bounds of the file, and otherwise stop the process and notify the user that the model file is corrupted.

---

## Issue #N/A: Bug: RWKV 6 Finch 3B+ models crash llama.cpp with CPU backend

**Link**: https://github.com/ggml-org/llama.cpp/issues/9315
**State**: closed
**Created**: 2024-09-04T17:10:11+00:00
**Closed**: 2024-09-12T11:25:17+00:00
**Comments**: 26
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

I cloned latest version from git, compiled it on ArchLinux, CPU backend only, using `make`.

I downloaded following models, but both did not run:
bartowski/v6-Finch-3B-HF-GGUF (Q4*, Q8*)
bartowski/v6-Finch-7B-HF-GGUF (Q4*, Q8*)

I run following command:
```bash
./llama-cli -m "v6-Finch-7B-HF-Q4_K_M.gguf" -p "I believe the meaning of life is" -n 128
```

### Name and Version

version: 3664 (82e3b03c)
built with cc (GCC) 14.2.1 20240805 for x86_64-pc-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
Log start
main: build = 3664 (82e3b03c)
main: built with cc (GCC) 14.2.1 20240805 for x86_64-pc-linux-gnu
main: seed  = 1725469492
llama_model_loader: loaded meta data with 26 key-value pairs and 902 tensors from v6-Finch-7B-HF-Q4_K_M.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:       

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: gemma-3n crash when using HIP

**Link**: https://github.com/ggml-org/llama.cpp/issues/14448
**State**: open
**Created**: 2025-06-29T15:53:06+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

### Name and Version

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 ROCm devices:
  Device 0: AMD Radeon 890M Graphics, gfx1102 (0x1102), VMM: no, Wave Size: 32
version: 5775 (bd9c981d7)
built with clang version 19.0.0git (/srcdest/rocm-llvm c87081df219c42dc27c5b6d86c0525bc7d01f727) for x86_64-pc-linux-gnu

### Operating systems

Linux

### GGML backends

HIP

### Hardware

AMD Radeon 890M Graphics

### Models

Gemma 3n

### Problem description & steps to reproduce

Running all gemma-3n models works well when using cpu, using HIP result in same crash.

Quick debugging show that the array is all nan:

```
(gdb) p cur_p->data[0].p
$8 = nan(0x400000)
(gdb) p cur_p->data[1].p
$9 = nan(0x400000)
(gdb) p cur_p->data[2].p
$10 = nan(0x400000)
```

### Relevant log output

```shell
$ llama-cli -hf unsloth/gemma-3n-E4B-it-GGUF -co -c 0 -fa -ngl 1000
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: n

[... truncated for brevity ...]

---

## Issue #N/A: Getting "Bad CPU type in executable" on macos-x64 build

**Link**: https://github.com/ggml-org/llama.cpp/issues/6875
**State**: closed
**Created**: 2024-04-24T13:36:26+00:00
**Closed**: 2024-06-25T02:41:40+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, stale

### Description

I downloaded build 2717 (April 24th) into a macbook pro Intel x86_64 and I am getting "Bad CPU type in executable" on any of the built commands. I tested with the previous build and I get the same issue.

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

## Issue #N/A: [Enhancement Proposal] Single Model Speculative Sampling (+ possible future dynamic bit-depth models)

**Link**: https://github.com/ggml-org/llama.cpp/issues/3269
**State**: closed
**Created**: 2023-09-19T19:52:39+00:00
**Closed**: 2024-04-03T01:15:57+00:00
**Comments**: 7
**Labels**: stale

### Description

Hi,

I'm currently working on developing a different feature that I plan on submitting at some point. In the process of that, I took some time to work on fast sampling. As I'm out of bandwidth at the moment (have a ton to juggle), I'm putting this out here as an enhancement proposal, and would like to see if I could get someone interested in working on it. I could use your help! <3 :') I think it is worthwhile not just because of the present value it brings, but also because of certain future optimizations that it enables.

This may or may not not be the right time for this feature because I believe it depends upon two limiting factors:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A. a good-enough uniform quantization scheme, and
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;B. efficient utilization of reduced compute from lower bit depths.

TL;DR: For a sufficient non-dynamic quantization scheme, dropping the least N significant bits of a model for the initial speculative

[... truncated for brevity ...]

---

## Issue #N/A: Qwen2-57B-A14B-Instruct not supported

**Link**: https://github.com/ggml-org/llama.cpp/issues/7813
**State**: closed
**Created**: 2024-06-07T08:47:04+00:00
**Closed**: 2024-06-07T10:00:28+00:00
**Comments**: 13
**Labels**: bug-unconfirmed, medium severity

### Description

### What happened?

The converted model fails to load due to unexpected expert tensor dimensions, the current qwen2moe implementation expects it to be `n_ff`/`n_expert_used`, which it is not.

### Name and Version

./main --version
version: 3066 (e141ce62)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
llama_model_load: error loading model: check_tensor_dims: tensor 'blk.0.ffn_gate_exps.weight' has wrong shape; expected  3584,  2368,    64, got  3584,  2560,    64,     1
```


---

## Issue #N/A: [LoRA] Falcon merges still don't work. Ideas as to why?

**Link**: https://github.com/ggml-org/llama.cpp/issues/3713
**State**: closed
**Created**: 2023-10-21T13:25:41+00:00
**Closed**: 2024-04-04T01:07:37+00:00
**Comments**: 1
**Labels**: stale

### Description

I've been merging lora into quantized models for a while now with export_lora and have had good results. The models definitely merge and performance appears to improve. Converting the lora to GGUF and then applying it to models results in a working model.

The same can't be said for falcon. All falcon tunes are released as PEFT and the model is simply too large to d/l as FP16. It's several hundred GB unless quantized.

I applied the PR https://github.com/ggerganov/llama.cpp/pull/3333 and am able to successfully convert lora to GGUF. I can then use export_lora to merge. However the models come out repeating gibberish and having sentence piece errors when used with HF sampling.

Looking over the code, there is nothing llama specific that I can find in it. Has anyone been able to load a lora to any falcon models, either live or as merges? Anyone have ideas of what's wrong?

---

## Issue #N/A: GPT2 Architecture Integration

**Link**: https://github.com/ggml-org/llama.cpp/issues/4073
**State**: closed
**Created**: 2023-11-14T14:52:32+00:00
**Closed**: 2023-12-28T14:03:58+00:00
**Comments**: 6
**Labels**: enhancement, good first issue

### Description

# Feature Description
The idea is to be able to convert models using the GPT2 architecture into GGUF. The convert-hf-to-gguf.py should include GPT2, as well as llama.cpp for running the model.

# Motivation
There are quite a few models for low resource languages or specific use cases that are fine-tuned on GPT2 architecture.

# Possible Implementation
The structure of models is quite similar to Starcoder. From my understanding, you can modify it quite easily by:

convert-hf-to-gguf.py
- Add a new model class
- Modify the set_gguf_parameters() [kv heads] and write_tensors() [maybe you need to transpose the qkv, up-ffn and down-ffn layer] methods

llama.cpp
- Add an new model class

# Status
I tried implementing that myself, but am not deep enough into the topic and find it quite hard to understand the libraries structure (is there any good documentation). So, I am probably not able to pull this off by myself, but am happy to support!

---

## Issue #N/A: Library not loaded: @rpath/libllama.dylib

**Link**: https://github.com/ggml-org/llama.cpp/issues/11321
**State**: closed
**Created**: 2025-01-20T21:41:04+00:00
**Closed**: 2025-01-25T13:21:45+00:00
**Comments**: 5
**Labels**: bug-unconfirmed

### Description

### Name and Version

I just downloaded the latest binary (b4519) to play around with on my Mac, however it looks like the binary didn't quite compile correctly. I tried downloading an earlier version (b4514) with the same result. Running `llamba-cli` yields:

```
➜  llama.cpp ./llama-cli --version
dyld[90496]: Library not loaded: @rpath/libllama.dylib
  Referenced from: <653E6B29-4AFF-3485-B031-B4F65747F8CF> /Users/constantmeiring/Downloads/build/llama.cpp/llama-cli
```



### Operating systems

MacOS 15.1.1 (24B91)

### GGML backends

Metal

### Hardware

Macbook - M3 Max

### Models

_No response_

### Problem description & steps to reproduce

Download the latest build and try and run it on Mac.

### First Bad Commit

_No response_

### Relevant log output

```shell
dyld[90496]: Library not loaded: @rpath/libllama.dylib
  Referenced from: <653E6B29-4AFF-3485-B031-B4F65747F8CF> /Users/constantmeiring/Downloads/build/llama.cpp/llama-cli
  Reason: tried: '/Users/runner/work/llama.cpp/l

[... truncated for brevity ...]

---

## Issue #N/A: kubernetes example

**Link**: https://github.com/ggml-org/llama.cpp/issues/6546
**State**: open
**Created**: 2024-04-08T16:31:37+00:00
**Comments**: 18
**Labels**: enhancement, help wanted, server/webui, kubernetes

### Description

### Motivation

Kubernetes is widely used in the industry to deploy product and application at scale.

It can be useful for the community to have a `llama.cpp` [helm](https://helm.sh/docs/intro/quickstart/) chart for the server.

I have started several weeks ago, I will continue when I have more time, meanwhile any help is welcomed:

https://github.com/phymbert/llama.cpp/tree/example/kubernetes/examples/kubernetes

### References
- #6545


---

## Issue #N/A: Misc. bug: tool call issues with hf unsloth/Qwen2.5-Coder-7B-Instruct-128K-GGUF

**Link**: https://github.com/ggml-org/llama.cpp/issues/12279
**State**: closed
**Created**: 2025-03-09T06:24:52+00:00
**Closed**: 2025-03-10T10:36:35+00:00
**Comments**: 13
**Labels**: bug

### Description

### Name and Version

I'm running my server like this, to test #12034 
```bash
llama-server --jinja -fa -c 0 -hf unsloth/Qwen2.5-Coder-7B-Instruct-128K-GGUF
```

Using various LLM frameworks in different languages, I couldn't get a successful tool call to complete. I've listed the errors, that vary, in the details


### Operating systems

_No response_

### Which llama.cpp modules do you know to be affected?

_No response_

### Command line

```shell
Here's the version of llama-cpp

$ llama-cli --version
version: 4856 (6fefc05a)
built with Apple clang version 16.0.0 (clang-1600.0.26.6) for arm64-apple-darwin24.2.0
```

### Problem description & steps to reproduce

I ran each [tool calling example app](https://github.com/elastic/observability-examples/tree/main/genai-function-calling) in this directory catching where it errored at via `socat -v TCP-LISTEN:8080,fork TCP:localhost:8081`, then I re-ran the corresponding curl to that failure.


### Semantic Kernel dotnet: fails because tool

[... truncated for brevity ...]

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

## Issue #N/A: CPU only, Vs GPU/CPU split VS GPU only

**Link**: https://github.com/ggml-org/llama.cpp/issues/6516
**State**: closed
**Created**: 2024-04-07T00:01:57+00:00
**Closed**: 2024-04-16T05:01:15+00:00
**Comments**: 16
**Labels**: bug-unconfirmed

### Description

Windows 11 (24 core/32 processor) (nov 2023, 6MHZ processor) , 64 GIG ram, Nvidia 16 GB card (GEforce RTX 4060TI ) , version LLAMA.CPP mar 31 2024.

I have noticed some anomalies after testing close to 500 GGUF models over the past 6 months.
I have a standardized method of testing models, and record the results (and any issues) and grade them from 1 to 10 (1 being top).
This includes models from 1B to 70B in size, with some testing of 103/120 B models.
This covers multiple quants as well as Imatrix quants including standard models, MOEs (all sizes, configs) and merged models.

THE ISSUE:
Specifically differences between CPU only, GPU/CPU split, and GPU only processing of instructions and output quality.

In some cases CPU VS GPU : CPU performance - in terms of quality is much higher than GPU only.
In some cases CPU/GPU (split 50,50) is superior to GPU only quality.

Testing involves getting a GPU baseline, CPU baseline and then GPU/CPU baseline and comparing carefully.



[... truncated for brevity ...]

---

## Issue #N/A: Infinite loop of "context shift"

**Link**: https://github.com/ggml-org/llama.cpp/issues/3969
**State**: closed
**Created**: 2023-11-06T16:02:22+00:00
**Closed**: 2024-02-23T18:41:04+00:00
**Comments**: 23
**Labels**: bug-unconfirmed

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

llama.cpp (server) processes inputs

# Current Behavior

When chatting with the LLM through `server` (and `api_like_OAI.py`) it works for a bit, but then seemingly when `--ctx-size` is exceeded, it gets into an infinite loop of `context_shift`s:

I have mostly s

[... truncated for brevity ...]

---

## Issue #N/A: new cuda kernel for quantized direct multiplication - performance not worth it ?

**Link**: https://github.com/ggml-org/llama.cpp/issues/1519
**State**: closed
**Created**: 2023-05-18T14:20:51+00:00
**Closed**: 2023-05-21T16:31:47+00:00
**Comments**: 11

### Description

I was surprised to see dmmv() being implemented, I had tested (a less well optimized) cuda kernel for 4 bit quantization a week ago and my result was that the performance loss can not be overcome without an incredible effort in optimization.

cuBLAS (and I suppose clBLAST too in that regard) uses Tensor cores, not CUDA cores. Also this stuff goes through a series of complex optimizations. Last time I compared the speed difference from a custom "average" kernel from scratch to cuBLAS was **30+ times**.
That hit can not be compensated with the lower performance required from low precision calculations.

I just ran a quick test on Q8_0 with the new kernel on and forced off using ggml_cuda_mul_mat_q_f32()
with the new kernel I had average times of 580ms matmul per layer
with the new kernel disabled it's down to 150ms matmul per layer

It would be awesome to have a high performance kernel that can process 4,5,8 bit quantized values but that would need to be optimized a ton to compe

[... truncated for brevity ...]

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

