# Less_than_4_bits - issues

**Total Issues**: 3
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 3

### Label Distribution

- generation quality: 3 issues
- Less than 4 bits: 3 issues
- enhancement: 2 issues
- research 🔬: 1 issues
- stale: 1 issues

---

## Issue #N/A: RPTQ state of the art quantization

**Link**: https://github.com/ggml-org/llama.cpp/issues/1295
**State**: closed
**Created**: 2023-05-03T03:23:53+00:00
**Closed**: 2024-04-09T01:09:40+00:00
**Comments**: 2
**Labels**: generation quality, research 🔬, Less than 4 bits, stale

### Description

Per yuan etc all, RPTQ quant is state of the art down to 3bit

It would be good to implement RPTQ for llama and other c++ downstream projects

https://github.com/hahnyuan/RPTQ4LLM/blob/master/quantize/quantizer.py

https://arxiv.org/abs/2304.01089

---

## Issue #N/A: Variable bit rate quantization

**Link**: https://github.com/ggml-org/llama.cpp/issues/1256
**State**: closed
**Created**: 2023-04-30T16:46:25+00:00
**Closed**: 2023-06-07T08:02:32+00:00
**Comments**: 10
**Labels**: enhancement, generation quality, Less than 4 bits

### Description

Variable bit rate is commonly used in audio and video compression, so why not try on LLMs?

My guess is that a locally adaptive variable bit rate would require a major change to `ggml`. So, then, the least one can try is to see if using different number of bits in the different network layers would be beneficial.

As a first step, I simply changed `llama.cpp` to not quantize one of the tensor types in addition to `output.weight` (which is already known to have a significant impact on generation quality) and calculated perplexity for `Q2_4` quantization (see issue #1240). Picked 2-bit quantization because there the difference between a quantized and not quantized tensor will be largest, so it would be easiest to see the effect. The following table summarizes the results (PPL improvement is perplexity with `fp16` `output.weight` - perplexity with `fp16` `output weight` + indicated tensor, table is sorted in decreasing order of impact) 

| Tensor type | PPL improvement |
|---------

[... truncated for brevity ...]

---

## Issue #N/A: QX_4 quantization

**Link**: https://github.com/ggml-org/llama.cpp/issues/1240
**State**: closed
**Created**: 2023-04-29T19:44:03+00:00
**Closed**: 2023-06-07T08:03:06+00:00
**Comments**: 10
**Labels**: enhancement, generation quality, Less than 4 bits

### Description

### Summary

Use `16 x 8` "super-blocks" for quantization, having one `fp16` scale for the "super-block" and 16 quantized scales per 8 model weights. This is particularly useful for 2- and 3-bit quantization, but it also outperforms the existing 4-bit quantization schemes `Q4_0` and `Q4_2`.

### Details

The naming of existing `llama.cpp` quantizations follows the scheme `QX_Y`, where `X` is the number of bits used for the quants, and `Y` is `0, 1, 2,` or `3`.  When `Y` is even (0 or 2), model weights `x` are computed from the quants `q` as `x = d * q`. When `Y` is odd, then `x = m + d * q` is used. If we look at the integer part of `Y/2` (`[Y/2]`), then the number of weights in a quantization block is 32 (`Q4_0`, `Q4_1`, `Q5_0`) when `[Y/2] = 0`, and 16  (`Q4_2`, `Q4_3`) when `[Y/2] = 1`. From the [latest perplexity results](https://github.com/ggerganov/llama.cpp#quantization) one can see that quantization using blocks of 16 weights performs better than quantization that uses bl

[... truncated for brevity ...]

---

