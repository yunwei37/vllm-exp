# macos - issues

**Total Issues**: 10
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 9

### Label Distribution

- macos: 10 issues
- performance: 3 issues
- good first issue: 3 issues
- help wanted: 3 issues
- enhancement: 2 issues
- bug-unconfirmed: 2 issues
- stale: 2 issues
- threading: 1 issues
- bug: 1 issues
- build: 1 issues

---

## Issue #N/A: CPU performance bottleneck(?) when using macOS Accelerate

**Link**: https://github.com/ggml-org/llama.cpp/issues/5417
**State**: closed
**Created**: 2024-02-08T16:53:12+00:00
**Closed**: 2024-02-11T19:12:45+00:00
**Comments**: 5
**Labels**: enhancement, performance, macos, threading

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Feature Description

I've been doing some performance testing of llama.cpp in macOS (On M2 Ultra 24-Core) and was comparing the CPU performance of inference with various options, and ran into a very large performance drop - Mixtral model inference on 16 cores (16 because it's only the p

[... truncated for brevity ...]

---

## Issue #N/A: ggml : support `bs > 512` for Metal `ggml_mul_mat_id`

**Link**: https://github.com/ggml-org/llama.cpp/issues/5070
**State**: closed
**Created**: 2024-01-22T01:34:58+00:00
**Closed**: 2024-03-10T21:12:50+00:00
**Comments**: 6
**Labels**: enhancement, good first issue, macos

### Description

Mixtral models + metal gpu + batch size > 512 = GGML_ASERT. Does not affect models such as llama-2-7b-chat.Q5_K_M.gguf

Hardware: Apple M2 Ultra
RAM: 192GB
llama.cpp current version as of 2024-01-21 (504dc37be8446fb09b1ede70300250ad41be32a2)

./main -f /tmp/prompt1k -m models/mixtral-8x7b-instruct-v0.1.Q6_K.gguf -c 4096 -b 512 << OK
./main -f /tmp/prompt1k -m models/mixtral-8x7b-instruct-v0.1.Q6_K.gguf -c 4096 -b 4096 << FAIL

```
### Assistant:GGML_ASSERT: ggml-metal.m:1511: ne11 <= 512
```

./main -f /tmp/prompt1k -m models/mixtral-8x7b-instruct-v0.1.Q6_K.gguf -c 4096 -b 4096 -ngl 0 << OK

but takes forever

---

## Issue #N/A: `llama_decode` is significantly slower if `n_tokens > 1` 

**Link**: https://github.com/ggml-org/llama.cpp/issues/4624
**State**: closed
**Created**: 2023-12-24T23:05:48+00:00
**Closed**: 2024-04-02T01:10:00+00:00
**Comments**: 7
**Labels**: performance, macos, bug-unconfirmed, stale

### Description

Issue
---
It is expected that `llama_decode` should take more time if more tokens are present in the batch, but on my system (Apple M1 Max 32GB) with `mistral-7b-instruct-v0.2.Q4_0.gguf` model, the increase in time taken is quite significant. I plotted some avg latencies on my system with different `n_tokens` using a modified version of `speculative` and putting timing around `llama_decode(ctx_tgt, batch_tgt);`:

![image](https://github.com/ggerganov/llama.cpp/assets/1957903/d9683434-6278-41b2-9018-d60acbe4ec2a)

There is more 5x jump in latency of `llama_decode` when `n_tokens` goes from 1 to 2 (which I feel is too high), but a very gradual increase after that. This means that techniques like `speculative` and `lookup` decoding **cannot give speed benefits** for small draft sizes ( `n_draft < 5`) even if drafts are 100% correct, since **autoregressively decoding 5 tokens 1 at a time is just as fast as decoding 5 tokens at once**, so the advantage of speculation is lost.

I'm n

[... truncated for brevity ...]

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

## Issue #N/A: Can't Quantize gguf files: zsh: illegal hardware instruction on M1 MacBook Pro

**Link**: https://github.com/ggml-org/llama.cpp/issues/3983
**State**: closed
**Created**: 2023-11-08T00:19:22+00:00
**Closed**: 2023-11-14T17:35:34+00:00
**Comments**: 8
**Labels**: macos, bug-unconfirmed

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [Y] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [Y] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [Y] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [Y] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior
Successfully quantize and run large language models that I convert to gguf on M1 MacBook Pro

# Current Behavior
Quantization halts due to "zsh: illegal hardware instruction".

# Environment and Context
OS: Mac OS Sonoma
System: 2020 M1 MacBook Pro 16GB RAM
Xcod

[... truncated for brevity ...]

---

## Issue #N/A: metal : add Q5_0 and Q5_1 support

**Link**: https://github.com/ggml-org/llama.cpp/issues/3504
**State**: closed
**Created**: 2023-10-06T14:11:19+00:00
**Closed**: 2023-10-18T12:21:49+00:00
**Comments**: 2
**Labels**: good first issue, macos

### Description

The implementation for `Q5_0` and `Q5_1` is still missing in the Metal backend. We should add it

https://github.com/ggerganov/llama.cpp/blob/5ab6c2132aad2354092a26c096cc5c8f55801141/ggml-metal.m#L962-L974

https://github.com/ggerganov/llama.cpp/blob/5ab6c2132aad2354092a26c096cc5c8f55801141/ggml-metal.m#L996-L1003



---

## Issue #N/A: [USER] It seems that Metal is not working

**Link**: https://github.com/ggml-org/llama.cpp/issues/3423
**State**: closed
**Created**: 2023-10-01T08:09:54+00:00
**Closed**: 2023-10-02T13:55:30+00:00
**Comments**: 8
**Labels**: macos

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Load gguf models with llama.cpp.

# Current Behavior

It fails initializing metal.

./main -m /Users/jlsantiago/Documents/Models/llama2/llama-2-13b-chat.Q4_0.gguf --repeat_penalty 1.0 --color -i -r "User:" -f ../../prompts/chat-with-bob.txt
Log start
.........

[... truncated for brevity ...]

---

## Issue #N/A: swift Package compile breaks due to ggml-metal.metal

**Link**: https://github.com/ggml-org/llama.cpp/issues/1740
**State**: closed
**Created**: 2023-06-07T13:44:07+00:00
**Closed**: 2023-06-16T15:05:07+00:00
**Comments**: 0
**Labels**: help wanted, good first issue, macos

### Description

Currently the swift package doesn't seem to be able to compile due to the ggml-metal.metal file.

Maybe someone else has experience with swift package manger and metal files?

Looking into this a bit further the swift Package need a bit of an update.



---

## Issue #N/A: Support CoreML like whisper.cpp?

**Link**: https://github.com/ggml-org/llama.cpp/issues/1714
**State**: open
**Created**: 2023-06-06T09:23:08+00:00
**Comments**: 11
**Labels**: help wanted, performance, macos

### Description

I have tried whisper.cpp on my iPhone and it runs very fast , so I wonder if it is possible that llama.cpp could support it. 
thank you .

---

## Issue #N/A: Compilation failed on macOS 10.7-8-9: 'clock_gettime' produce warnings and errors

**Link**: https://github.com/ggml-org/llama.cpp/issues/788
**State**: closed
**Created**: 2023-04-05T19:16:14+00:00
**Closed**: 2024-04-11T01:07:01+00:00
**Comments**: 20
**Labels**: bug, build, macos, stale

### Description

# PREREQUISITES
- I am running the latest code: [5a8c4f6](https://github.com/ggerganov/llama.cpp/releases/tag/master-5a8c4f6)
- I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- I have created a [relevant issue](https://github.com/antimatter15/alpaca.cpp/issues/201) in alpaca.cpp.
- I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# EXPECTED BEHAVIOR
* Attempted to compile the binary for macOS 10.7, 10.8 and 10.9.
* Expected to run the chat app on an old macOS, that will be isolated from Internet.

# ACTUAL BEHAVIOR
* Compilation is terminated with warnings and errors.

# ENVIRONMENT AND CONTEXT
* Macbook pro 15 2012: macOS 10.8 Mountain Lion on Core i7 + 512 SDD + 16Gb RAM
	* Parallels Virtual Machine: macOS 10.7 Lion on 20Gb HDD + 4Gb RAM
		* X-Code 4.6.3
		* Command Line Tools OS X Lion Nov2012
		* MacPorts 2.8.1 10.7 (Lion)
			*

[... truncated for brevity ...]

---

