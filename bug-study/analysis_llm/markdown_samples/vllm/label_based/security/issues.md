# security - issues

**Total Issues**: 3
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 3

### Label Distribution

- security: 3 issues
- bug: 2 issues

---

## Issue #N/A: [Bug]: Merge security updates for 0.9.0

**Link**: https://github.com/vllm-project/vllm/issues/17667
**State**: closed
**Created**: 2025-05-05T16:08:43+00:00
**Closed**: 2025-05-09T14:07:58+00:00
**Comments**: 1
**Labels**: security

### Description

This is a placeholder to ensure any pending security patches have been merged prior to release.

---

## Issue #N/A: [Bug]: clients can crash the openai server with invalid regex

**Link**: https://github.com/vllm-project/vllm/issues/17313
**State**: closed
**Created**: 2025-04-28T15:27:44+00:00
**Closed**: 2025-05-12T01:06:11+00:00
**Comments**: 2
**Labels**: bug, security

### Description

### Your current environment

```
root@3bea15cf4c9f:/# uv run --with vllm python collect_env.py
INFO 04-28 15:38:49 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
/usr/local/lib/python3.11/dist-packages/_distutils_hack/__init__.py:31: UserWarning: Setuptools is replacing distutils. Support for replacing an already imported distutils is deprecated. In the future, this condition will fail. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.11.10 (main, Sep  7 2024, 18:35:41) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.8.0-52-generic-x86_64-with-glibc2.35
Is CU

[... truncated for brevity ...]

---

## Issue #N/A: [Tracker] Merge security fixes for v0.8.5

**Link**: https://github.com/vllm-project/vllm/issues/17128
**State**: closed
**Created**: 2025-04-24T17:19:49+00:00
**Closed**: 2025-04-25T16:23:36+00:00
**Comments**: 1
**Labels**: bug, security

### Description

This issue is for tracking that pending security fixes are merged prior to releasing v0.8.5

- [x] GHSA-hj4w-hm2g-p6w5 - https://github.com/vllm-project/vllm/pull/17192
- [x] GHSA-9f8f-2vmf-885j - https://github.com/vllm-project/vllm/pull/17197

---

