# help_wanted - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 9
- Closed Issues: 21

### Label Distribution

- help wanted: 30 issues
- good first issue: 13 issues
- high priority: 7 issues
- enhancement: 7 issues
- bug: 5 issues
- research 🔬: 3 issues
- model: 2 issues
- bug-unconfirmed: 2 issues
- performance: 2 issues
- roadmap: 2 issues

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

## Issue #N/A: convert.py can not identify type of pytorch BF16 tensors

**Link**: https://github.com/ggml-org/llama.cpp/issues/2504
**State**: closed
**Created**: 2023-08-03T19:02:37+00:00
**Closed**: 2023-08-14T10:37:54+00:00
**Comments**: 6
**Labels**: bug, help wanted

### Description

The original model files have tensors stored in BF16 which have a wider numeric range than F16. The quality will suffer too much if the norm tensors are converted to lower precision, I believe this is the reason why they are always stored in F32 since there is no support for BF16 in ggml.

When I tried to list the tensor types in convert.py for experimentation purposes, I discovered that all BF16 tensors in pytorch and safetensors models are identified as F16, but are for some unknown reason correctly converted to F32. BF16 tensors in .pth model files are correctly identified.

Directly following this line:
https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/convert.py#L88

Insert this:
```
        if tensor.data_type == DT_F16:  print(name + " DT_F16")
        if tensor.data_type == DT_F32:  print(name + " DT_F32")
        if tensor.data_type == DT_BF16: print(name + " DT_BF16")

        if tensor.data_type == DT_F16:
            return D

[... truncated for brevity ...]

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

## Issue #N/A: Extend ggml format to include a description of the model.

**Link**: https://github.com/ggml-org/llama.cpp/issues/1575
**State**: closed
**Created**: 2023-05-23T16:39:04+00:00
**Closed**: 2024-05-17T12:37:14+00:00
**Comments**: 14
**Labels**: enhancement, help wanted

### Description

On Hugging Face there are many files called ggml-model-f16.bin or similar. Once downloaded the user can rename them. The information about its origin gets lost. Updating the file becomes difficult when the origin is unknown. It would be easier to extend the ggml format so that the creator can embed a description of the model when generating them using 'quantize'.

---

## Issue #N/A: metal : need help debugging a kernel and setting up Xcode

**Link**: https://github.com/ggml-org/llama.cpp/issues/4545
**State**: closed
**Created**: 2023-12-20T09:20:35+00:00
**Closed**: 2024-01-02T08:57:45+00:00
**Comments**: 7
**Labels**: help wanted, macos

### Description

Recently, we found out that we can run the Metal code in debug mode with shader validation enabled:

```bash
make -j tests && MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 MTL_SHADER_VALIDATION_FAIL_MODE=allow MTL_DEBUG_LAYER_VALIDATE_STORE_ACTIONS=1 MTL_DEBUG_LAYER_VALIDATE_LOAD_ACTIONS=1 ./tests/test-backend-ops -b Metal -o MUL_MAT
```

This has been a useful way to debug the Metal shaders in `ggml-metal.metal`. The above command runs the `ggml_mul_mat` operator in isolation and compares the results with the reference CPU result.

The result from the command without `MTL_SHADER_VALIDATION=1` is success.
However, when the shader validation instrumentation is enabled, we get some NaNs:

![image](https://github.com/ggerganov/llama.cpp/assets/1991296/07e57680-b200-4e72-a59d-fded8350caa1)

The failures are produced by the matrix-matrix multiplication kernel:

- invoked here:
  https://github.com/ggerganov/llama.cpp/blob/328b83de23b33240

[... truncated for brevity ...]

---

## Issue #N/A: Constrained decoding with grammar fails for c4ai-command-r-v01

**Link**: https://github.com/ggml-org/llama.cpp/issues/6112
**State**: closed
**Created**: 2024-03-17T14:51:01+00:00
**Closed**: 2024-05-28T10:55:36+00:00
**Comments**: 6
**Labels**: bug, help wanted

### Description

I am trying to apply constrained decoding for the recently adopted command-r. 

Using the most recent master branch (https://github.com/ggerganov/llama.cpp/commit/c47cf414efafb8f60596edc7edb5a2d68065e992) I'm trying to apply the simplest list.  

`./main -m ~/data/c4ai-command-r-v01/ggml-model-Q4_K_M.gguf -p "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Please give me a list of things to do in SF?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>" -ctk q8_0 -ngl 99 -n 500 --grammar-file grammars/list.gbnf`

It fails with 

`libc++abi: terminating due to uncaught exception of type std::out_of_range: unordered_map::at: key not found`

Any idea what could go wrong here?

More details:

```
Log start
main: build = 2447 (c47cf414)
main: built with Apple clang version 15.0.0 (clang-1500.3.9.4) for arm64-apple-darwin23.3.0
main: seed  = 1710686911
llama_model_loader: loaded meta data with 23 key-value pairs and 322 tensors from ~/data/c4ai-command-r-v01/ggml-m

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

## Issue #N/A: b1705 introduces build error on linux-aarch64

**Link**: https://github.com/ggml-org/llama.cpp/issues/4654
**State**: closed
**Created**: 2023-12-27T18:28:34+00:00
**Closed**: 2023-12-31T09:44:23+00:00
**Comments**: 9
**Labels**: help wanted, bug-unconfirmed

### Description

Starting with b1705, I get the following when building for linux-aarch64:
```
clang++ -I. -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG  -std=c++11 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -mcpu=native   -Wunreachable-code-break -Wunreachable-code-return -Wmissing-prototypes -Wextra-semi -c common/train.cpp -o train.o
ggml-quants.c:413:25: error: redefinition of 'vdotq_s32'
inline static int32x4_t vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
                        ^
/opt/llvm.org/v16.0.6/lib/clang/16/include/arm_neon.h:33859:51: note: previous definition is here
__ai __attribute__((target("dotprod"))) int32x4_t vdotq_s32(int32x4_t __p0, int8x16_t __p1, int8x16_t __p2) {
                                                  ^
1 error generated.
```

it seems to be directly related to https://github.com/ggerganov/llama.cpp/pull/4630. the various #define tricks i'm used to don't seem to be

[... truncated for brevity ...]

---

## Issue #N/A: metal : increase GPU duty-cycle during inference

**Link**: https://github.com/ggml-org/llama.cpp/issues/9507
**State**: closed
**Created**: 2024-09-16T12:14:00+00:00
**Closed**: 2024-10-01T13:00:26+00:00
**Comments**: 1
**Labels**: help wanted, performance, Apple Metal

### Description

Apparently there is a significant GPU downtime between Metal compute encoders within a single `ggml_metal_graph_compute()`: 

<img width="2672" alt="image" src="https://github.com/user-attachments/assets/e01b56a0-cdcf-4777-9944-be6e456858eb">

See https://github.com/ggerganov/llama.cpp/issues/6506 for instructions how to generate the trace from the picture.

My expectation was that enqueuing the command buffers in parallel would make them execute without any downtime. The goal of this issue is to understand where this overhead comes from and if there is a way to avoid it.

Obviously, using a single command buffer will avoid all the GPU downtime, but it is much slower to construct it in a single thread. Ideally, we want to continue queuing multiple encoders, but not have the gaps in-between during execution.

---

## Issue #N/A: tutorials : list for llama.cpp

**Link**: https://github.com/ggml-org/llama.cpp/issues/13523
**State**: open
**Created**: 2025-05-14T05:00:53+00:00
**Comments**: 5
**Labels**: help wanted, good first issue, roadmap

### Description

Project: https://github.com/orgs/ggml-org/projects/6

List:

- [tutorial : compute embeddings using llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/7712)
- [tutorial : parallel inference using Hugging Face dedicated endpoints](https://github.com/ggml-org/llama.cpp/discussions/9041)
- [tutorial : KV cache reuse with llama-server](https://github.com/ggml-org/llama.cpp/discussions/13606)
- [tutorial : measuring time to first token (TTFT) and time between tokens (TBT)](https://github.com/ggml-org/llama.cpp/discussions/14115)

TODO:
- [ ] https://github.com/ggml-org/llama.cpp/discussions/13488
- [ ] https://github.com/ggml-org/llama.cpp/discussions/13134
- [ ] https://github.com/ggml-org/llama.cpp/discussions/13251
- [ ] https://github.com/ggml-org/llama.cpp/discussions/12742
- [ ] How to get started with webui development (ref: https://github.com/ggml-org/llama.cpp/issues/13523#issuecomment-2879256096)
- [ ] etc.

Simply search for "How to" in the Discussions: https://github.c

[... truncated for brevity ...]

---

## Issue #N/A: Refactor: Existing examples refactoring opportunities

**Link**: https://github.com/ggml-org/llama.cpp/issues/7559
**State**: open
**Created**: 2024-05-27T09:31:05+00:00
**Comments**: 0
**Labels**: help wanted, refactoring

### Description

[GG in this PR was suggesting a refactoring of examples and provided an example of what he would like to refactor. This is a ticket to inform others about opportunities to improve examples](https://github.com/ggerganov/llama.cpp/pull/7534#issuecomment-2132746334)

> I think a bigger advantage would be to do some refactoring in the existing examples and "hide" some of the state for sampling and KV cache management that we expose behind the common API

- [ ] Hide some state for sampling (unsure what this mean)
- [ ] Key Value cache management should be abstracted away behind the common API
- [ ] @ngxson [suggest restructuring the help message in gpt_params_print_usage() to improve help message clarity](https://github.com/ggerganov/llama.cpp/pull/7534#issuecomment-2133064853)

If there is any other aspect of the example which is currently a pain point for developer grokking, feel free to also suggest some so it can be added here.

---

## Issue #N/A: common: download from URL, improve parallel download progress status

**Link**: https://github.com/ggml-org/llama.cpp/issues/6537
**State**: open
**Created**: 2024-04-08T07:37:01+00:00
**Comments**: 6
**Labels**: enhancement, help wanted, good first issue, split

### Description

### Context

When downloading a sharded model, files are downloaded in parallel, it was added in:
- #6192

The progressions of each download conflict:
![image](https://github.com/ggerganov/llama.cpp/assets/5741141/d4937fc7-edf4-4920-ba63-dadf1c77b2d0)

Need to properly implement [CURLOPT_NOPROGRESS](https://curl.se/libcurl/c/CURLOPT_NOPROGRESS.html) for parallel download.

Example in #6515:

```shell
main --hf-repo ggml-org/models \
  --hf-file grok-1/grok-1-q4_0-00001-of-00009.gguf \
  --model   models/grok-1-q4_0-00001-of-00009.gguf \
  -ngl 64
   --prompt "I believe the meaning of life is"
```

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

## Issue #N/A: Feature Request: Task Cancellation on Client Disconnection

**Link**: https://github.com/ggml-org/llama.cpp/issues/6421
**State**: closed
**Created**: 2024-04-01T08:20:25+00:00
**Closed**: 2025-05-16T19:42:45+00:00
**Comments**: 5
**Labels**: enhancement, help wanted, good first issue, server/webui

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description
In the current embedding server setup, if a client sends a request and then cancels it, tasks that are already queued continue processing without detecting the cancellation. This can lead to inefficiencies and potential server overload.

**[Test Case]** 
During an 

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

## Issue #N/A: Compile bug: ios swift xcode build error when upgrade to llama : use cmake for swift build 

**Link**: https://github.com/ggml-org/llama.cpp/issues/10747
**State**: open
**Created**: 2024-12-10T05:12:25+00:00
**Comments**: 41
**Labels**: help wanted, good first issue, build

### Description

### Git commit

$git rev-parse HEAD 43ed389a3f102517e6f7d5620d8e451e88afbf27

### Operating systems

Mac

### GGML backends

Metal

### Problem description & steps to reproduce

ios swift xcode build error when upgrade to

- https://github.com/ggerganov/llama.cpp/pull/10525

Before the upgrade, the code compiled successfully. After the upgrade, it throws a compilation error: "Cannot find type 'xxx' in scope."

<img width="1721" alt="image" src="https://github.com/user-attachments/assets/1bc2e76a-158a-4aa3-9755-855930f2f7ed">


### First Bad Commit

43ed389a3f102517e6f7d5620d8e451e88afbf27

### Relevant log output

```shell
/ios/llama.cpp.swift/LibLlama.swift:8:39 Cannot find type 'llama_batch' in scope

/ios/llama.cpp.swift/LibLlama.swift:12:37 Cannot find type 'llama_batch' in scope

/ios/llama.cpp.swift/LibLlama.swift:12:56 Cannot find type 'llama_token' in scope

/ios/llama.cpp.swift/LibLlama.swift:12:76 Cannot find type 'llama_pos' in scope

/ios/llama.cpp.swift/LibL

[... truncated for brevity ...]

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

## Issue #N/A: Improve `cvector-generator`

**Link**: https://github.com/ggml-org/llama.cpp/issues/8724
**State**: open
**Created**: 2024-07-27T13:05:20+00:00
**Comments**: 33
**Labels**: enhancement, help wanted, good first issue

### Description

- Fix random direction of vector produced by PCA method https://github.com/ggerganov/llama.cpp/pull/8069#issuecomment-2185328171 (cc @jukofyork)
- Performance: Using batch and multi-sequences (for `llama_decode`)
- Performance: Allow using GPU (need to test, because some backends don't support certain ops)
- Add SVD and UMAP
- Add tests if possible

Ref: https://github.com/ggerganov/llama.cpp/pull/7514

Note: I'm writing this for tracking, but people are free to do PRs if you have any ideas to share. I'm planning to do some of the tasks in August 2024.

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

## Issue #N/A: support `--hf-token` param in addition of `--hf-repo`

**Link**: https://github.com/ggml-org/llama.cpp/issues/6613
**State**: closed
**Created**: 2024-04-11T17:27:58+00:00
**Closed**: 2024-09-23T16:23:11+00:00
**Comments**: 11
**Labels**: enhancement, help wanted, good first issue

### Description

### Motivation

Models with non standard licence requires an opt-in on HF and the download requires a read token if the model is gated.

### Proposal

- Add an `--hf-token` param in `common` and add it in the url built with `--hf-repo` and `--hf-file` in `llama_load_model_from_hf`.





---

## Issue #N/A: [SYCL] Support newer non linear quantization

**Link**: https://github.com/ggml-org/llama.cpp/issues/5674
**State**: closed
**Created**: 2024-02-23T05:40:16+00:00
**Closed**: 2024-04-09T05:38:28+00:00
**Comments**: 20
**Labels**: help wanted, bug-unconfirmed

### Description

cc: @abhilash1910 @airMeng 

Also [important](https://github.com/ggerganov/llama.cpp/pull/5590#issuecomment-1952955062)!

---

## Issue #N/A: Test replit-code-v1-3b model

**Link**: https://github.com/ggml-org/llama.cpp/issues/1299
**State**: open
**Created**: 2023-05-03T15:47:01+00:00
**Comments**: 7
**Labels**: help wanted, model

### Description

Replit recently trained a 3B parameter Llama-style code model with some very promising results. Weights have been released [here](https://huggingface.co/replit/replit-code-v1-3b)

---

## Issue #N/A: ggml : add WebGPU backend

**Link**: https://github.com/ggml-org/llama.cpp/issues/7773
**State**: open
**Created**: 2024-06-05T14:24:37+00:00
**Comments**: 21
**Labels**: help wanted, research 🔬, roadmap

### Description

I hope that this would be relatively easy to do since AFAIK WebGPU allows us to write kernels in a shader language, so we have experience how to create such backends.

There has been some initial work in https://github.com/ggerganov/ggml/pull/585 - could be useful as a starting point

---

## Issue #N/A: llava-cli: improve llava-cli and the API for using LLaVA

**Link**: https://github.com/ggml-org/llama.cpp/issues/6027
**State**: open
**Created**: 2024-03-12T21:18:55+00:00
**Comments**: 4
**Labels**: enhancement, help wanted, good first issue, llava

### Description

From:
 - https://github.com/ggerganov/llama.cpp/issues/4216#issuecomment-1991730224

1. cleaning up the clip/llava libs and improving the API
2. in the old implementation, there were many internal object exposed to the server and the memory management was dubious
3. there was no obvious path for supporting parallel multimodal slots


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

## Issue #N/A: Bug: gguf pypi package corrupts environment

**Link**: https://github.com/ggml-org/llama.cpp/issues/9566
**State**: closed
**Created**: 2024-09-20T15:32:43+00:00
**Closed**: 2025-01-08T18:55:00+00:00
**Comments**: 2
**Labels**: bug, help wanted, high severity

### Description

### What happened?

installing `gguf` using `pip install gguf` will register `gguf` AND `scripts`
which means that any app that has `scripts` in their structure will suddenly start failing just because `gguf` is installed.

looking at <https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/pyproject.toml>
```js
packages = [
    {include = "gguf"},
    {include = "gguf/py.typed"},
    {include = "scripts"},
]
```
culprit is clear - `scripts` folder should be moved to be under `gguf`, not as a separate package.
if that is too much, then at least rename scripts package to be `gguf_scripts`

### Name and Version

latest commit as of date of issue: [d39e267](https://github.com/ggerganov/llama.cpp/commit/d39e26741f9f02340651dbc640c9776e1a1128ef)

### What operating system are you seeing the problem on?

Linux, Mac, Windows, BSD

### Relevant log output

```shell
import gguf
from scripts import test_module

test_module.some_method()


```log
ImportError: cannot import n

[... truncated for brevity ...]

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

