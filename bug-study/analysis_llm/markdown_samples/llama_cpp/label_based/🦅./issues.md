# 🦅. - issues

**Total Issues**: 1
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 1

### Label Distribution

- good first issue: 1 issues
- performance: 1 issues
- 🦅.: 1 issues

---

## Issue #N/A: falcon : speed-up prompt processing

**Link**: https://github.com/ggml-org/llama.cpp/issues/2850
**State**: closed
**Created**: 2023-08-28T09:51:27+00:00
**Closed**: 2023-09-15T08:09:25+00:00
**Comments**: 2
**Labels**: good first issue, performance, 🦅.

### Description

The performance of Falcon 7B should be comparable to LLaMA 7B since the computation graph is computationally very similar.

Here are the current numbers on M2 Ultra for LLaMA, LLaMA-v2 and Falcon 7B:

```bash
../scripts/run-all-perf.sh ${model} "f16 q8_0 q4_0"
```

| model                          |       size |     params | backend    | ngl | test       |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------- | ---------------: |
| LLaMA 7B mostly F16            |  12.55 GiB |     6.74 B | Metal      | 999 | pp 512     |    665.95 ± 0.18 |
| LLaMA 7B mostly Q8_0           |   6.64 GiB |     6.74 B | Metal      | 999 | pp 512     |    630.28 ± 0.16 |
| LLaMA 7B mostly Q4_0           |   3.56 GiB |     6.74 B | Metal      | 999 | pp 512     |    632.32 ± 0.22 |
| LLaMA 7B mostly F16            |  12.55 GiB |     6.74 B | Metal      | 999 | tg 64      |     29.73 ± 0.01 |
| LLaMA 7B mostly Q8_0           |   6.64 GiB | 

[... truncated for brevity ...]

---

