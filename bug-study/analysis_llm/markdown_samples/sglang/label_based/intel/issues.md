# intel - issues

**Total Issues**: 1
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 0

### Label Distribution

- enhancement: 1 issues
- high priority: 1 issues
- intel: 1 issues
- cpu: 1 issues

---

## Issue #N/A: [Feature] RFC for adding CPU support for SGLang

**Link**: https://github.com/sgl-project/sglang/issues/2807
**State**: open
**Created**: 2025-01-09T07:58:45+00:00
**Comments**: 13
**Labels**: enhancement, high priority, intel, cpu

### Description

### Motivation

Hi, SGLang folks! This is Mingfei from intel pytorch team, our team helps optimize PyTorch performance on CPU. I am also the PyTorch module maintainer for cpu performance. We would like to contribute to SGLang for CPU enabling and performance optimization.

### Targets
Our primary target is to optimize SGLang performance on Intel Xeon Scalable Processors (x86 server CPUs).
* Optimization will be focusing on Xeon with [Intel® Advanced Matrix Extensions](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html) support, including Sapphire Rapids(4th gen), Emerald Rapids(5th gen), Granite Rapids(6th gen).
* Native implementations or fallbacks will be provided for CPUs with other ISA to make it functional.
* Providing good performance per dollar.

### Limitations

* Kernels written in **avx512** and **amx-bf16**, requires **GCC11** or above.
* **BFloat16/Float16** will be enabled at the same time on CPU, but we only 

[... truncated for brevity ...]

---

