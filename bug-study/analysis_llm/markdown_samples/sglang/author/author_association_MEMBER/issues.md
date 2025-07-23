# author_association_MEMBER - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 4
- Closed Issues: 26

### Label Distribution

- high priority: 15 issues
- good first issue: 8 issues
- help wanted: 8 issues
- inactive: 8 issues
- performance: 3 issues
- enhancement: 2 issues
- deepseek: 1 issues
- documentation: 1 issues
- lora: 1 issues
- bug: 1 issues

---

## Issue #N/A: [Feature] support Qwen 3 and Qwen 3 MoE

**Link**: https://github.com/sgl-project/sglang/issues/4682
**State**: closed
**Created**: 2025-03-22T22:53:35+00:00
**Closed**: 2025-05-10T11:03:21+00:00
**Comments**: 3
**Labels**: enhancement, good first issue, help wanted, high priority

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

ref https://github.com/InternLM/lmdeploy/pull/3305

### Related resources

_No response_

---

## Issue #N/A: [Feature] support torch compile cache for DeepSeek V3/R1

**Link**: https://github.com/sgl-project/sglang/issues/3614
**State**: closed
**Created**: 2025-02-16T16:18:21+00:00
**Closed**: 2025-02-21T18:18:09+00:00
**Comments**: 6
**Labels**: good first issue, help wanted, high priority, deepseek

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

The time taken for each startup is currently too long when torch compile is enabled. It needs optimization.

### Related resources

_No response_

---

## Issue #N/A: [Feature] use modelopt for fp8 and fp4 by default

**Link**: https://github.com/sgl-project/sglang/issues/5251
**State**: open
**Created**: 2025-04-10T18:53:53+00:00
**Comments**: 7
**Labels**: documentation, good first issue, help wanted, high priority

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

https://github.com/NVIDIA/TensorRT-Model-Optimizer is the **de facto** LLM quant library for fp8 and fp4, supported in both TensorRT LLM and SGLang. We will consider changing all current fp8, fp4 doc, CI, unit test, etc. to default to ModelOpt's checkpoint

ref https://huggingface.co/nvidia

### Related resources

_No response_

---

## Issue #N/A: rewrite test_trt_allreduce

**Link**: https://github.com/sgl-project/sglang/issues/4907
**State**: closed
**Created**: 2025-03-30T02:00:42+00:00
**Closed**: 2025-03-30T08:02:00+00:00
**Comments**: 0
**Labels**: high priority

### Description

as titled https://github.com/sgl-project/sglang/blob/main/sgl-kernel/tests/test_trt_allreduce.py @yizhang2077 

---

## Issue #N/A: [Bug] Llama3 70B A100 PCIE TP4 slow speed

**Link**: https://github.com/sgl-project/sglang/issues/1137
**State**: closed
**Created**: 2024-08-17T16:10:40+00:00
**Closed**: 2024-09-22T12:48:43+00:00
**Comments**: 5

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

When using ShareGPT 1k, no results can be obtained.

10 is normal, but it keeps getting stuck after changing to 1000.
```
Initial test run completed. Starting main benchmark run...
  0%|                                              | 0/1000 [00:00<?, ?it/s]
```

<details>

```
============ Serving Benchmark Result ============


[... truncated for brevity ...]

---

## Issue #N/A: [Feature] optimize group gemm

**Link**: https://github.com/sgl-project/sglang/issues/3323
**State**: closed
**Created**: 2025-02-05T22:56:43+00:00
**Closed**: 2025-02-20T08:26:59+00:00
**Comments**: 1
**Labels**: high priority, performance, lora

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Rewrite the  Grouped GEMM used by LoRA with cuBLAS 12.5 in sgl-kernel for improved speed.

https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/
https://github.com/zhihu/ZhiLight/blob/main/src/nn/linear/gemm_grouped.cpp

### Related resources

_No response_

---

## Issue #N/A: DBRX not working

**Link**: https://github.com/sgl-project/sglang/issues/454
**State**: closed
**Created**: 2024-05-20T05:32:18+00:00
**Closed**: 2024-07-26T01:02:21+00:00
**Comments**: 1
**Labels**: inactive

### Description

No description provided.

---

## Issue #N/A: [Bug] fix amd ut

**Link**: https://github.com/sgl-project/sglang/issues/4502
**State**: closed
**Created**: 2025-03-17T09:23:07+00:00
**Closed**: 2025-03-17T22:18:24+00:00
**Comments**: 1
**Labels**: bug, high priority

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

fix https://github.com/sgl-project/sglang/actions/runs/13895307432/job/38874551851

### Reproduction

N/A

### Environment

N/A

---

## Issue #N/A: [Feature] integrate pplx-kernels

**Link**: https://github.com/sgl-project/sglang/issues/5010
**State**: closed
**Created**: 2025-04-02T23:57:42+00:00
**Closed**: 2025-07-04T00:19:42+00:00
**Comments**: 3
**Labels**: high priority, inactive

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled cc @ch-wan 
https://github.com/ppl-ai/pplx-kernels
thanks @abcdabcd987 for the guidance

### Related resources

_No response_

---

## Issue #N/A: [Feature] high performance multi node custom all reduce

**Link**: https://github.com/sgl-project/sglang/issues/5994
**State**: closed
**Created**: 2025-05-03T07:46:47+00:00
**Closed**: 2025-07-03T00:19:58+00:00
**Comments**: 1
**Labels**: high priority, inactive

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

e.g. DeepSeek R1 TP 16 on two H100s

### Related resources

_No response_

---

## Issue #N/A: [Feature] Integration of TurboMind AWQ and GPTQ

**Link**: https://github.com/sgl-project/sglang/issues/2788
**State**: open
**Created**: 2025-01-08T08:37:01+00:00
**Comments**: 1
**Labels**: good first issue, help wanted

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

The AWQ and GPTQ of TurboMind should be among the best-performing open-source implementations currently available. We plan to integrate them into SGLang, and once the integration is complete, we can consider removing SGLang's dependency on vLLM's AWQ and GPTQ kernel.

During development, we can initially install the wheel https://github.com/InternLM/turbomind/releases/tag/v0.0.1 manually for verification and later add the TurboMind repo as a dependency in [sgl-kernel](https://github.com/sgl-project/sglang/tree/main/sgl-kernel).

ref
https://github.com/InternLM/turbomind

### Related resources

_No response_

---

## Issue #N/A: [Feature] support minference attention backend

**Link**: https://github.com/sgl-project/sglang/issues/5329
**State**: closed
**Created**: 2025-04-12T19:02:15+00:00
**Closed**: 2025-06-12T00:19:28+00:00
**Comments**: 1
**Labels**: high priority, inactive

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled @minminsun @yinfan98 @ZhangJianwei0311

ref https://github.com/sgl-project/sglang/pull/5327

### Related resources

_No response_

---

## Issue #N/A: AWQ performance tracking

**Link**: https://github.com/sgl-project/sglang/issues/1505
**State**: closed
**Created**: 2024-09-24T14:33:27+00:00
**Closed**: 2024-11-24T01:20:38+00:00
**Comments**: 2
**Labels**: inactive, performance

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

# Current Situation

## SGLang

```bash
# v0.3.1.post3
pip install --upgrade pip
pip install "sglang[all]"

pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

```
python3 -m sglang.launch_server --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --disable-radix

python3 bench_serving.py --backend sglang --num-prompts 5000
```

```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Successful requests:                     5000
Benchmark duration (s):                  161.16
Total input tokens:                      1130466
Total generated tokens:                  971613


[... truncated for brevity ...]

---

## Issue #N/A: [Bug] fix code scanning issue

**Link**: https://github.com/sgl-project/sglang/issues/2315
**State**: closed
**Created**: 2024-12-02T13:42:27+00:00
**Closed**: 2025-02-01T00:17:47+00:00
**Comments**: 1
**Labels**: backlog, inactive

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

ref https://github.com/sgl-project/sglang/security/code-scanning

The priority is not high, I will handle it when I have the bandwidth.

### Reproduction

N/A

### Environment

N/A

---

## Issue #N/A: Development Roadmap (2025 H1)

**Link**: https://github.com/sgl-project/sglang/issues/4042
**State**: open
**Created**: 2025-03-04T00:09:49+00:00
**Comments**: 23
**Labels**: collaboration

### Description

Here is the development roadmap for 2025 H1. Contributions and feedback are welcome ([**Join Bi-weekly Development Meeting**](https://docs.google.com/document/d/1xEow4eIM152xNcRxqZz9VEcOiTQo8-CEuuQ5qTmkt-E/edit?usp=sharing)). The previous 2024 Q4 roadmap can be found in #1487

## Focus
- Throughput-oriented large-scale deployment similar to the [deepseek inference system](https://github.com/deepseek-ai/open-infra-index?tab=readme-ov-file#day-6---one-more-thing-deepseek-v3r1-inference-system-overview)
- Long context optimizations
- Low latency speculative decoding
- Reinforcement learning training framework integration
- Kernel optimizations

## Parallelism
- [x] Support PD disaggregation @ByronHsu  #4655
- [x] Support expert parallelism and load balancer #5524
- [x] Support pipeline parallelism @Ying1123 #5724
- [x] Support data parallelism attention compatible with all other parallelism #4390 
- [x] Support overlap communication in TP/EP @tom @Zhuohao-Li #4068
- [ ] Improvements of sg

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] support ep for DeepSeek V3

**Link**: https://github.com/sgl-project/sglang/issues/2740
**State**: closed
**Created**: 2025-01-05T17:28:24+00:00
**Closed**: 2025-03-25T04:12:11+00:00
**Comments**: 8
**Labels**: good first issue, help wanted

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

The code for EP and block wise FP8 required by V3 is available separately. The task is to integrate block wise FP8 into the current DeepSeek V2 EP, based on the previous integration of Fused MoE with block wise FP8.

ref

https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/moe/ep_moe

https://github.com/sgl-project/sglang/pull/2575

### Related resources

_No response_

---

## Issue #N/A: [Feature] qwen 3 eagle 3

**Link**: https://github.com/sgl-project/sglang/issues/7617
**State**: closed
**Created**: 2025-06-28T04:44:46+00:00
**Closed**: 2025-07-10T16:33:29+00:00
**Comments**: 3
**Labels**: high priority

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

### Related resources

_No response_

---

## Issue #N/A: [Feature] support Kimi VL

**Link**: https://github.com/sgl-project/sglang/issues/5314
**State**: closed
**Created**: 2025-04-12T06:22:24+00:00
**Closed**: 2025-05-09T21:09:45+00:00
**Comments**: 5
**Labels**: good first issue, help wanted

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct

### Related resources

_No response_

---

## Issue #N/A: [Feature] support more user-friendly MTP

**Link**: https://github.com/sgl-project/sglang/issues/5595
**State**: closed
**Created**: 2025-04-21T08:03:50+00:00
**Closed**: 2025-04-29T23:33:16+00:00
**Comments**: 3
**Labels**: enhancement, high priority

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

As we discussed offline, we need to support more user-friendly MTP. cc @merrymercy 

- [ ] best configuration for the default @zhyncs 
- [ ] user doesn't need to specify draft model separately @ispobock 

### Related resources

_No response_

---

## Issue #N/A: Trouble Shooting

**Link**: https://github.com/sgl-project/sglang/issues/548
**State**: closed
**Created**: 2024-06-14T08:47:16+00:00
**Closed**: 2024-07-25T10:04:22+00:00
**Comments**: 1

### Description

- Triton Kernel Fix:
  If you see `No such file or directory: '/root/.triton/cache/e3457c918521f16104a655b081235e5a.....`
  (issue caused by pytorch dependency of triton==2.3.0)
  1. You can fix it by hacking the file `compiler.py`
  ```
  vim /usr/local/lib/python3.10/dist-packages/triton/compiler/compiler.py
  
  L230
  
  self.asm = {
    file.suffix[1:]: file.read_bytes() if file.suffix[1:] == driver.binary_ext else None
  ```
  2. Or you can uninstall triton and reinstall the triton nightly
  ```
  pip uninstall -y triton triton-nightly
  pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
  ```

---

## Issue #N/A: [Feature] sgl-kernel and docker images

**Link**: https://github.com/sgl-project/sglang/issues/5062
**State**: closed
**Created**: 2025-04-04T08:04:03+00:00
**Closed**: 2025-06-07T00:19:07+00:00
**Comments**: 5
**Labels**: inactive

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

We will only support cu118, cu124, and cu128 in the next release.

https://github.com/sgl-project/sglang/blob/main/sgl-kernel/CMakeLists.txt
cu118 whl is for sm75, sm80, sm86, and sm89. Therefore, for sgl-kernel we will only compile W8A8 Int8.
cu124 whl is for sm80, sm86, and sm89, sm90 and sm90a. So for sgl-kernel we will also compile W8A8 FP8 and Block wise FP8 and FA3(sm90a). https://github.com/sgl-project/sglang/blob/6ff9c6a5e71fc05b15a577adbb9656d24dd8848c/docker/Dockerfile#L3
cu128 whl is for sm90, sm90a, sm100 and sm100a. Thus, for sgl-kernel we will compile W8A8 Int8, W8A8 FP8, FP4.
For the docker image to work properly, it needs to support InfiniBand. cu118 and cu128's docker base image needs to be updated

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] FA3 support sm80

**Link**: https://github.com/sgl-project/sglang/issues/5911
**State**: closed
**Created**: 2025-04-30T07:11:13+00:00
**Closed**: 2025-04-30T21:02:09+00:00
**Comments**: 4
**Labels**: high priority

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled @yinfan98 

### Related resources

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

## Issue #N/A: [Feature] update CIs

**Link**: https://github.com/sgl-project/sglang/issues/6074
**State**: closed
**Created**: 2025-05-07T05:40:14+00:00
**Closed**: 2025-06-01T03:52:16+00:00
**Comments**: 1

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled update with following models
[meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)
[meta-llama/Llama-4-Maverick-17B-128E-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct)
[meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8)
[Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B)
[Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)
[Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)
[Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)
[Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
[Qwen/Qwen3-235B-A22B-FP8](https:

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] update sgl-kernel 3rdparty flashinfer to latest main

**Link**: https://github.com/sgl-project/sglang/issues/4301
**State**: closed
**Created**: 2025-03-11T08:18:52+00:00
**Closed**: 2025-05-26T00:26:08+00:00
**Comments**: 2
**Labels**: good first issue, help wanted, high priority

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

fix the compile issue

### Related resources

_No response_

---

## Issue #N/A: [Feature] deepseek-ai/DeepSeek-V3-0324 NextN ckpt

**Link**: https://github.com/sgl-project/sglang/issues/4808
**State**: closed
**Created**: 2025-03-27T08:02:03+00:00
**Closed**: 2025-03-27T21:00:40+00:00
**Comments**: 6

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled cc @ispobock 

### Related resources

_No response_

---

## Issue #N/A: [RFC] sm75 EOL

**Link**: https://github.com/sgl-project/sglang/issues/6006
**State**: closed
**Created**: 2025-05-04T06:12:20+00:00
**Closed**: 2025-07-17T00:21:12+00:00
**Comments**: 8
**Labels**: inactive

### Description

The SGLang team plans to deprecate support for sm75 in v0.5. If you’re still using SGLang for large-scale inference acceleration on sm75 devices in production, please let us know so we can defer this deprecation beyond v0.5.

---

## Issue #N/A: Rename variable names for rank

**Link**: https://github.com/sgl-project/sglang/issues/482
**State**: closed
**Created**: 2024-05-27T08:47:22+00:00
**Closed**: 2024-06-08T09:48:16+00:00
**Comments**: 0

### Description

`gpu_ids` -> ranks
`rank` -> tp_rank (srt/models/mixtral_quant.py)

---

## Issue #N/A: Development Roadmap (2024 Q4)

**Link**: https://github.com/sgl-project/sglang/issues/1487
**State**: closed
**Created**: 2024-09-21T22:38:00+00:00
**Closed**: 2025-03-03T18:43:18+00:00
**Comments**: 27

### Description

Here is the development roadmap for 2024 Q4. Contributions and feedback are welcome ([**Join Bi-weekly Development Meeting**](https://t.co/4BFjCLnVHq)). Previous 2024 Q3 roadmap can be found in #634.

## Performance
- [x] Hide CPU overhead with overlapped scheduler (#1738, #2067)
- [x] Support speculative decoding
  - Eagle  #2150 
  - Reference-based. #270
  - Medusa head #859
  - Draft model based.
- [x] Sparse Attention #1459
- [x] Faster grammar parsing library for constrained decoding #1752 
- [x] Multi-layer radix cache (GPU/CPU/Disk) https://github.com/sgl-project/sglang/pull/2693  @xiezhq-hermann 
- [ ] Improve the performance of mixed chunked prefill. see a draft #1383 
- [ ] Integrate CuDNN paged attention [kernels](https://github.com/NVIDIA/cudnn-frontend/blob/v1.8.0/samples/python/52_scaled_dot_product_attention_with_paged_caches.ipynb) 

## Parallelism
- [ ] Support sequence parallelism #1436. Related [paper](https://www.arxiv.org/pdf/2411.01783)
- [ ] Support pipeline par

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] remove vllm _custom_ops

**Link**: https://github.com/sgl-project/sglang/issues/2965
**State**: closed
**Created**: 2025-01-18T12:05:06+00:00
**Closed**: 2025-03-24T18:44:24+00:00
**Comments**: 4
**Labels**: good first issue, help wanted, high priority

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

- [ ] Support for `silu_and_mul` and `gelu_and_mul` in AMD, remove the current dependencies on `vllm ops.silu_and_mul` and `ops.gelu_and_muli`.  Used in `fused_moe_triton.py`. https://github.com/sgl-project/sglang/pull/4150 @yiakwy-xpu-ml-framework-team 
- [ ] remove `from vllm.model_executor.layers.activation import GeluAndMul, SiluAndMul` in `sglang/python/sglang/srt/layers/activation.py`.
- [ ] Support GemmaRMSNorm and RMSNorm in AMD.
- [ ] remove `from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm` in `sglang/python/sglang/srt/layers/layernorm.py`.
- [ ] Support `rotary_embedding` kernel in AMD.
- [ ] Support for `ops.moe_sum` in AMD, remove the dependency on `vllm ops.moe_sum`.  Used in `fu

[... truncated for brevity ...]

---

