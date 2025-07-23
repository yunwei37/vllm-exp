# wontfix - issues

**Total Issues**: 2
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 2

### Label Distribution

- wontfix: 2 issues
- good first issue: 1 issues

---

## Issue #N/A: [Bug] Llama4 fails to run on Python 3.9 (AssertionError)

**Link**: https://github.com/sgl-project/sglang/issues/6232
**State**: closed
**Created**: 2025-05-12T12:05:57+00:00
**Closed**: 2025-06-15T13:15:12+00:00
**Comments**: 1
**Labels**: good first issue, wontfix

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Running llama 4 with Python 3.9 get AssertionError

e.g.`python -m sglang.launch_server --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct --tp 8 --context-length 65536`

The error does not occur in python 3.10, 3.11, 3.12.

### Reproduction

#### Python 3.9 (AssertionError)

```bash
mkdir 3-9-test
cd 3-9-test
uv init --python 3.9
uv a

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] P100/Cuda 6.1 support

**Link**: https://github.com/sgl-project/sglang/issues/1062
**State**: closed
**Created**: 2024-08-12T20:29:42+00:00
**Closed**: 2024-08-13T04:31:55+00:00
**Comments**: 1
**Labels**: wontfix

### Description

### Motivation

As per https://github.com/sgl-project/sglang/issues/1059 , P100/pascal/6.1 support is not currently a target. This feature request is an official request to support it. This GPU is the least expensive hardware that will run modern LLMs, and is a common GPU in both academia and common use.

This issue was created as the original was locked with the cryptic phrase, "It makes nonsense for me.", the meaning of which was not clear in context. This issue is intended to be a place where the community can discuss support for these GPUs, as well as petition for support.

### Related resources

_No response_

---

