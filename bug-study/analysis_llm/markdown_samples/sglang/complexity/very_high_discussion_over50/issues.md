# very_high_discussion_over50 - issues

**Total Issues**: 3
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 2

### Label Distribution

- collaboration: 1 issues
- deepseek: 1 issues
- inactive: 1 issues
- enhancement: 1 issues
- high priority: 1 issues
- performance: 1 issues
- quant: 1 issues

---

## Issue #N/A: Instruction for Running DeepSeek with Large-scale PD and EP

**Link**: https://github.com/sgl-project/sglang/issues/6017
**State**: open
**Created**: 2025-05-05T04:48:15+00:00
**Comments**: 504
**Labels**: collaboration, deepseek

### Description

## Using main branch

~~NOTE: The feature is already on main, but the performance still needs some improvements on main branch.~~ will be good after a few already opened PRs - PR 6680, 6727, 6728

~~NOTE: I will try other config like 4 node for P and 9 node for D later.~~ updated

### Environment Preparation

Use SGLang and DeepEP on master is sufficient. Also remember to upgrade Mooncake.

### 4P + 9D experiments

Start server
where DeepEP config can be tuned by https://github.com/sgl-project/sglang/pull/6742

```python
# prefill nodes
MC_TE_METRIC=true SGLANG_TBO_DEBUG=1 python3 -m sglang.launch_server --model-path /dev/shm/DeepSeek-V3-0324 --disaggregation-ib-device mlx5_1 --disaggregation-mode prefill --dist-init-addr 10.5.55.3:5757 --nnodes 4 --node-rank 0 --tp-size 32 --dp-size 32 --enable-dp-attention --decode-log-interval 1 --enable-deepep-moe --page-size 1 --host 0.0.0.0 --trust-remote-code --moe-dense-tp-size 1 --enable-dp-lm-head --disable-radix-cache --watchdog-timeout 1000

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]  Model Stuck at Prefill and then throw "Watchdog Timeout" Error After Idle Period (Deepseek-r1:671b on two H100*8)

**Link**: https://github.com/sgl-project/sglang/issues/3836
**State**: closed
**Created**: 2025-02-25T06:10:47+00:00
**Closed**: 2025-05-31T00:18:38+00:00
**Comments**: 57
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I am currently using SGLang to deploy the deepseek-r1:671b model across two H800 GPUs. However, I have encountered a persistent issue when the system remains idle for some time. Upon resuming usage, even with simple prompts such as "Hello," the model gets stuck during the Prefill stage. Subsequently, the system throws a "watchdog timeout" 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] DeepSeek V3 optimization

**Link**: https://github.com/sgl-project/sglang/issues/2591
**State**: closed
**Created**: 2024-12-26T08:52:39+00:00
**Closed**: 2025-03-25T04:10:46+00:00
**Comments**: 52
**Labels**: enhancement, high priority, performance, quant

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Adoption

[SGLang adoption for DeepSeek V3 and R1](https://github.com/sgl-project/sglang/discussions/3322)

### Usage

User Guide for Existing System (Installation & Launch)

https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3

Please use the latest version [v0.4.2.post4](https://pypi.org/project/sglang/0.4.2.post4/). Please prefer to use docker image. `docker pull lmsysorg/sglang:latest`

For running on AMD MI300X, use this as a reference. [Running DeepSeek-R1 on a single NDv5 MI300X VM](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/running-deepseek-r1-on-a-single-ndv5-mi300x-vm/4372726)

### Features

- [x] Support CUDA Graph @HandH1998 @ispobock 
- [x] Support Torch compile @is

[... truncated for brevity ...]

---

