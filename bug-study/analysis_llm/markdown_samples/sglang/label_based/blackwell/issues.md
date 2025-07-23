# blackwell - issues

**Total Issues**: 9
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 6
- Closed Issues: 3

### Label Distribution

- blackwell: 9 issues
- high priority: 2 issues
- collaboration: 1 issues
- bug: 1 issues
- flashinfer: 1 issues
- performance: 1 issues
- enhancement: 1 issues

---

## Issue #N/A: [Roadmap] Blackwell Support and Optimizations

**Link**: https://github.com/sgl-project/sglang/issues/7227
**State**: open
**Created**: 2025-06-16T06:07:50+00:00
**Comments**: 45
**Labels**: high priority, collaboration, blackwell

### Description

### Roadmap

- [x] ~~Initial support and optimizations for GB200, PD disaggregation, and large-scale EP~~ -- Done in https://lmsys.org/blog/2025-06-16-gb200-part-1/
- [x] Initial optimizations for prefill for large scale EP
- [ ] Optimize kernels for the Blackwell architecture
    - [ ] Communication kernels
    - [ ] Various smaller kernels
- [ ] Optimize for latency-oriented scenarios
- [ ] Computation-communication overlap

TODO: more

### Updates after Blog

* Prefill is slightly optimized, 13149 token/s/gpu for ISL 4096 (as usual all code are open sourced)

### Blog Reproduction

<details>

To reproduce [the blog post](https://lmsys.org/blog/2025-06-16-gb200-part-1/), here are the instructions:

#### 2025.07.12

To use the latest main, the following commands can be used.

Versions that I personally use to test (other versions may work as well)
* SGLang: https://github.com/sgl-project/sglang/commit/2a2d3478afe8cdb336888f2e6faa3775ac40254e
* sgl-kernel: the one inside SGLang
* DeepG

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Deepseek R1 FP4 model quality drop

**Link**: https://github.com/sgl-project/sglang/issues/7166
**State**: open
**Created**: 2025-06-13T23:05:50+00:00
**Comments**: 4
**Labels**: blackwell

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I believe this issue applies to both R1 FP4 and R1-0528 FP4. 

For R1 FP4, GSM8k score is only 0.886. Not trying to reproduce official result, but it should be something around 0.95. Also Nvidia reports much higher gsm8k score with trtllm [here](https://huggingface.co/nvidia/DeepSeek-R1-FP4#evaluation). 

Any help is really appreciated! 



[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Upgrade the glibc for `lmsysorg/sglang:blackwell`

**Link**: https://github.com/sgl-project/sglang/issues/6561
**State**: closed
**Created**: 2025-05-24T00:57:44+00:00
**Closed**: 2025-06-05T07:49:41+00:00
**Comments**: 3
**Labels**: blackwell

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

While `lmsysorg/sglang:latest` uses Ubuntu22.04 (glibc 2.35), `lmsysorg/sglang:blackwell` uses glibc 2.28 which is too old.

```
❯ sudo docker run -it lmsysorg/sglang:blackwell ldd --version
ldd (GNU libc) 2.28

❯ sudo docker run -it --ipc=host --device=nvidia.com/gpu=all lmsysorg/sglang:blackwell /bin/bash
/bin/bash: /lib64/ld-linux-x86-64.so.2: version `GLIBC_2.35' not found (required by /...glibc-2.40-66/lib/libc.so.6)
```

### Related resources

_No response_

---

## Issue #N/A: [Bug] Blackwell freezes on cloning into MoE

**Link**: https://github.com/sgl-project/sglang/issues/6448
**State**: open
**Created**: 2025-05-20T05:29:53+00:00
**Comments**: 1
**Labels**: blackwell

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

From a Runpod 8xB200 environment, I ran a simple vanilla Deepseek setup with tp 8 and no optimizations. Latest Blackwell image hangs on cloning into MoE.

Unsure if the image is meant to be used?

### Reproduction

python3 -m sglang.launch_server --trust-remote-code --tp 8 --host 0.0.0.0

Using Deepseek V3 0324, on official sglang docker b

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] flashinfer_python with minimum required version 0.2.5 is not installed

**Link**: https://github.com/sgl-project/sglang/issues/6160
**State**: closed
**Created**: 2025-05-09T17:19:44+00:00
**Closed**: 2025-06-11T15:25:36+00:00
**Comments**: 6
**Labels**: blackwell

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I am trying to serve gemma3 27b-it on RTX 5090 using sglang blackwell image. However, I'm getting this error:
```bash
Traceback (most recent call last):
  File "/opt/conda/lib/python3.11/importlib/metadata/__init__.py", line 563, in from_name
    return next(cls.discover(name=name))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
StopIteration

D

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Cutlass_MLA backend can't run with tp8

**Link**: https://github.com/sgl-project/sglang/issues/6096
**State**: open
**Created**: 2025-05-07T20:14:18+00:00
**Comments**: 2
**Labels**: bug, blackwell

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

Cutlass MLA backend can only run when `dp_size` is equal to `tp_size`.
If launching deepseek-v3 with `--tp 8`, not enabling dp attention, the following bug occurs:
```bash
  File "/sgl-workspace/sglang/python/sglang/srt/layers/attention/base_attn_backend.py", line 69, in forward
    return self.forward_decode(
           ^^^^^^^^^^^^^^^^^^

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Tune fp8 Gemm and fused moe kernel on B200

**Link**: https://github.com/sgl-project/sglang/issues/6095
**State**: closed
**Created**: 2025-05-07T20:06:14+00:00
**Closed**: 2025-05-08T06:39:11+00:00
**Comments**: 1
**Labels**: blackwell

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

The performance of w8a8 gemm kernel and fused moe kernel is not good enough on B200. There is some space for tuning.

### Related resources

Reproduction on 8*B200:
```bash
python3 -m sglang.bench_one_batch --model-path /dev/shm/DeepSeek-V3 --tp 8 --batch 16 --input-len 1024 --output-len 128 --attention-backend triton --profile
```

_No response_

---

## Issue #N/A: [Feature] integrate FlashInfer Blackwell kernels

**Link**: https://github.com/sgl-project/sglang/issues/5855
**State**: open
**Created**: 2025-04-28T19:12:30+00:00
**Comments**: 4
**Labels**: high priority, flashinfer, performance, blackwell

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

### Related resources

_No response_

---

## Issue #N/A: [Tracker] Blackwell support

**Link**: https://github.com/sgl-project/sglang/issues/5338
**State**: open
**Created**: 2025-04-13T04:35:37+00:00
**Comments**: 29
**Labels**: enhancement, blackwell

### Description

## Usage

```bash
docker pull lmsysorg/sglang:blackwell

# use latest main
cd /sgl-workspace/sglang && git pull
```

## Models

### DeepSeek V3 ✅
```bash
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code
```

### Llama 4 ✅
```bash
python3 -m sglang.launch_server --model meta-llama/Llama-4-Scout-17B-16E-Instruct --tp 8 --context-length 131072
```

---

