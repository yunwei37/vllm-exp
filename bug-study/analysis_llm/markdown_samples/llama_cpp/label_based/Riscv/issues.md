# Riscv - issues

**Total Issues**: 1
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 1

### Label Distribution

- good first issue: 1 issues
- build: 1 issues
- Riscv: 1 issues

---

## Issue #N/A: Compile bug: RISCV cross-compile warnings cause build failure

**Link**: https://github.com/ggml-org/llama.cpp/issues/12693
**State**: closed
**Created**: 2025-04-01T14:20:59+00:00
**Closed**: 2025-04-03T17:19:00+00:00
**Comments**: 7
**Labels**: good first issue, build, Riscv

### Description

### Git commit

9c4cef4602c77068e1c6b91b2d8e707b493f6fcf

### Operating systems

Linux

### GGML backends

CPU

### Problem description & steps to reproduce

I am updating the CI with cross-compile builds for RISCV regression tests (see #12428 ) and a build error is occurring due to some RISCV macros/functions. Since I am not familiar with RISCV functions in question, I am deferring this fix to folks who know that platform better.


### First Bad Commit

_No response_

### Compile command

```shell
Please see the github workflow here: https://github.com/ggml-org/llama.cpp/pull/12428/files#diff-245fd2c5accd266a35983ed2891af1c8f8b41af027aa393075f15a00b38ff817
```

### Relevant log output

```shell
[ 12%] Building C object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu-quants.c.o
/home/runner/work/llama.cpp/llama.cpp/ggml/src/ggml-cpu/ggml-cpu-quants.c: In function ‘ggml_vec_dot_q5_0_q8_0’:
/home/runner/work/llama.cpp/llama.cpp/ggml/src/ggml-cpu/ggml-cpu-quants.c:3141:19: error: impli

[... truncated for brevity ...]

---

