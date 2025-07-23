# backlog - issues

**Total Issues**: 6
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 5

### Label Distribution

- backlog: 6 issues
- inactive: 2 issues
- good first issue: 1 issues
- help wanted: 1 issues

---

## Issue #N/A: [Feature] support llm_bench

**Link**: https://github.com/sgl-project/sglang/issues/2400
**State**: open
**Created**: 2024-12-08T11:02:20+00:00
**Comments**: 3
**Labels**: good first issue, help wanted, backlog

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

use `locust` as a benchmark option
ref https://github.com/fw-ai/benchmark/tree/main/llm_bench

### Related resources

_No response_

---

## Issue #N/A: [Feature] add Dockerfile dev image and doc

**Link**: https://github.com/sgl-project/sglang/issues/2317
**State**: closed
**Created**: 2024-12-02T15:11:22+00:00
**Closed**: 2024-12-12T19:23:36+00:00
**Comments**: 2
**Labels**: backlog

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Recently, the community has seen an influx of new developers. However, many are unfamiliar with using Docker for development and set up environments directly on their host machines, risking damage to the environment and affecting other developers. To efficiently utilize the cloud hosting resources sponsored by Nvidia H100, I will create a daily updated dev image and provide basic documentation to help new developers get started.

### Related resources

_No response_

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

## Issue #N/A: [Feature] Add Dockerfile.dev for development purposes

**Link**: https://github.com/sgl-project/sglang/issues/2060
**State**: closed
**Created**: 2024-11-17T16:23:48+00:00
**Closed**: 2024-12-01T07:27:53+00:00
**Comments**: 1
**Labels**: backlog

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Add some commonly used command-line tools

### Related resources

_No response_

---

## Issue #N/A: [Bug] Unsupported architectures: ChatGLMForConditionalGeneration.

**Link**: https://github.com/sgl-project/sglang/issues/1331
**State**: closed
**Created**: 2024-09-04T15:20:31+00:00
**Closed**: 2024-09-22T12:13:37+00:00
**Comments**: 4
**Labels**: backlog

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

ValueError: Unsupported architectures: ChatGLMForConditionalGeneration. Supported list: ['ChatGLMForCausalLM', 'ChatGLMModel', 'CohereForCausalLM', 'DbrxForCausalLM', 'DeepseekForCausalLM', 'DeepseekV2ForCausalLM', 'ExaoneForCausalLM', 'GemmaForCausalLM', 'Gemma2ForCausalLM', 'GPTBigCodeForCausalLM', 'Grok1ForCausalLM', 'Grok1ModelForCausa

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Correctness test for Triton kernels

**Link**: https://github.com/sgl-project/sglang/issues/1292
**State**: closed
**Created**: 2024-09-01T18:00:25+00:00
**Closed**: 2024-11-02T01:10:39+00:00
**Comments**: 1
**Labels**: backlog, inactive

### Description

### Motivation

The current tests for triton kernels are not ideal.

For extend attention, the test sits in `__main__`, there are two problems

1. It compares with prefill attn, which is also a triton kernel
2. not run in CI

For decode attention, there is no test

Ideally, we should implement a pytorch version and compare the result under the test folder. For example, https://github.com/linkedin/Liger-Kernel/blob/63dd41b15e9f1c2957c817b771536d4ab7119322/test/transformers/test_rms_norm.py#L72

### Related resources

_No response_

---

