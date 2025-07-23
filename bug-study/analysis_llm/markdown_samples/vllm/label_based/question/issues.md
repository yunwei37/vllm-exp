# question - issues

**Total Issues**: 7
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 7

### Label Distribution

- question: 7 issues
- misc: 2 issues
- stale: 1 issues

---

## Issue #N/A: [Misc]: Should inference with temperature 0 generate the same results for a lora adapter and equivalent merged model?

**Link**: https://github.com/vllm-project/vllm/issues/5148
**State**: closed
**Created**: 2024-05-31T04:07:06+00:00
**Closed**: 2024-06-03T01:51:39+00:00
**Comments**: 5
**Labels**: question, misc

### Description

### Anything you want to discuss about vllm.

I am scoring a fine-tuned mistral modal using the vllm enable_lora option with temperature 0.0. Subsequently merging that lora adapter into the base model using peft merge_and_unload results in a new merged model which when used with vllm (again temperature 0.0) generates noticeably different results. 

1. Should I be expecting these two methods of inference to generate the same result? 
2. If so is there any suggested method for loading the base model and adapter and merging that would result in the same generation results?

---

## Issue #N/A: computation of prompt_logprobs

**Link**: https://github.com/vllm-project/vllm/issues/2848
**State**: closed
**Created**: 2024-02-13T08:09:58+00:00
**Closed**: 2024-06-03T03:02:12+00:00
**Comments**: 1
**Labels**: question, misc

### Description

What are the distinctions between the computation of **prompt_logprobs** in input tokens and **logprobs** in output tokens?

---

## Issue #N/A: Use multi-turn prompts for benchmark_throughput.py

**Link**: https://github.com/vllm-project/vllm/issues/1139
**State**: closed
**Created**: 2023-09-22T05:56:26+00:00
**Closed**: 2024-12-01T02:16:17+00:00
**Comments**: 4
**Labels**: question, stale

### Description

Do we consider using multi-turn prompts instead of the first turn for throughput benchmarking? It would be more realistic.

---

## Issue #N/A: How many requests can `llm.generate` handle in parallel?

**Link**: https://github.com/vllm-project/vllm/issues/454
**State**: closed
**Created**: 2023-07-13T09:23:43+00:00
**Closed**: 2023-07-13T15:19:48+00:00
**Comments**: 1
**Labels**: question

### Description

To run benchmarks, should I add all prompts to the `llm.generate` class at once, let the engine queue and schedule, or add them in small batches?

---

## Issue #N/A: Do you support streaming generating outputs?

**Link**: https://github.com/vllm-project/vllm/issues/230
**State**: closed
**Created**: 2023-06-24T15:48:01+00:00
**Closed**: 2023-06-25T17:47:08+00:00
**Comments**: 2
**Labels**: question

### Description

No description provided.

---

## Issue #N/A: Can vllm serving clients by using multiple model instances?

**Link**: https://github.com/vllm-project/vllm/issues/181
**State**: closed
**Created**: 2023-06-21T07:24:05+00:00
**Closed**: 2023-06-25T16:43:30+00:00
**Comments**: 1
**Labels**: question

### Description

Based on the examples, vllm can launch a server with a  single model instances. Can vllm serving clients by using multiple model instances? With multiple model instances, the sever will dispatch the requests to different instances to reduce the overhead.

---

## Issue #N/A: What's the difference between vllm and triton-inference-server?

**Link**: https://github.com/vllm-project/vllm/issues/178
**State**: closed
**Created**: 2023-06-21T06:24:04+00:00
**Closed**: 2023-06-25T16:44:00+00:00
**Comments**: 5
**Labels**: question

### Description

May vllm can achieve the performance like fastertransformer on inference side? Just curious about the detailed optimization you're done and the goal you want to achieve.
BTW, vllm really accelerate our deploy work, thx.

---

