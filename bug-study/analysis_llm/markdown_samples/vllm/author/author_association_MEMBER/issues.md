# author_association_MEMBER - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 4
- Closed Issues: 26

### Label Distribution

- bug: 12 issues
- stale: 8 issues
- misc: 4 issues
- ci-failure: 4 issues
- feature request: 3 issues
- documentation: 2 issues
- good first issue: 2 issues
- help wanted: 2 issues
- performance: 2 issues
- ci/build: 1 issues

---

## Issue #N/A: [Bug]: CI flake - v1/engine/test_async_llm.py::test_abort - assert has_unfinished_requests()

**Link**: https://github.com/vllm-project/vllm/issues/16054
**State**: open
**Created**: 2025-04-04T09:48:13+00:00
**Comments**: 1
**Labels**: bug, ci/build, stale, v1

### Description

### Your current environment

...

### 🐛 Describe the bug

main commit 51d7c6a2b23e100cd9e7d85b8e7c0eea656b331e

Seen in https://github.com/vllm-project/vllm/pull/15894

https://buildkite.com/organizations/vllm/pipelines/ci/builds/16742/jobs/0195f24d-e81a-46a3-ad08-6a51983d65d6/log


```
=================================== FAILURES ===================================
[2025-04-01T17:38:12Z] _ test_abort[engine_args0-Hello my name is Robert and-RequestOutputKind.DELTA] _
[2025-04-01T17:38:12Z]
[2025-04-01T17:38:12Z] monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7fd1fa052e70>
[2025-04-01T17:38:12Z] output_kind = <RequestOutputKind.DELTA: 1>
[2025-04-01T17:38:12Z] engine_args = AsyncEngineArgs(model='meta-llama/Llama-3.2-1B-Instruct', served_model_name=None, tokenizer='meta-llama/Llama-3.2-1B-I...additional_config=None, enable_reasoning=None, reasoning_parser=None, use_tqdm_on_load=True, disable_log_requests=True)
[2025-04-01T17:38:12Z] prompt = 'Hello my name is Robert and'
[

[... truncated for brevity ...]

---

## Issue #N/A: [help wanted]: website ui improvement

**Link**: https://github.com/vllm-project/vllm/issues/10089
**State**: closed
**Created**: 2024-11-06T19:33:16+00:00
**Closed**: 2024-11-08T17:51:05+00:00
**Comments**: 2
**Labels**: misc

### Description

### Anything you want to discuss about vllm.

<img width="236" alt="image" src="https://github.com/user-attachments/assets/c8a23dda-5fe5-4829-b369-a65917e29bec">

currently, "ask AI" and doc version selection are overlapped.

we should move doc version selection to the left.

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [CI Failure]: Quantization Test - quantization/test_bitsandbytes.py::test_4bit_bnb_embedding_model

**Link**: https://github.com/vllm-project/vllm/issues/19964
**State**: closed
**Created**: 2025-06-23T04:53:19+00:00
**Closed**: 2025-06-23T13:30:57+00:00
**Comments**: 0
**Labels**: ci-failure

### Description

### Name of failing test

`quantization/test_bitsandbytes.py::test_4bit_bnb_embedding_model[half-intfloat/e5-mistral-7b-instruct-quantize embedding model inflight]`

### Basic information

- [ ] Flaky test
- [x] Can reproduce locally
- [ ] Caused by external libraries (e.g. bug in `transformers`)

### 🧪 Describe the failing test

```
pytest -s -v "quantization/test_bitsandbytes.py::test_4bit_bnb_embedding_model[half-intfloat/e5-mistral-7b-instruct-quantize embedding model inflight]"
INFO 06-23 04:48:10 [__init__.py:244] Automatically detected platform cuda.
/home/mgoin/venvs/vllm/lib/python3.12/site-packages/pytest_asyncio/plugin.py:208: PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset.
The event loop scope for asynchronous fixtures will default to the fixture caching scope. Future versions of pytest-asyncio will default the loop scope for asynchronous fixtures to function scope. Set the default fixture loop scope explicitly in order to a

[... truncated for brevity ...]

---

## Issue #N/A: Documentation on running basic python server and FastAPI server

**Link**: https://github.com/vllm-project/vllm/issues/106
**State**: closed
**Created**: 2023-05-17T20:36:13+00:00
**Closed**: 2023-06-17T17:26:14+00:00
**Comments**: 0
**Labels**: documentation

### Description

No description provided.

---

## Issue #N/A: [Bug][CI Failure] - VI Test - test_engine_core_client.py::test_kv_cache_events[True-tcp]

**Link**: https://github.com/vllm-project/vllm/issues/18708
**State**: closed
**Created**: 2025-05-26T11:38:43+00:00
**Closed**: 2025-06-04T12:57:32+00:00
**Comments**: 2
**Labels**: bug, ci-failure

### Description

### Your current environment

Flakey test for at least the past month: https://buildkite.com/organizations/vllm/analytics/suites/ci-1/tests/4abfbf0d-3a86-8a68-9ff3-0e0ab0fbb38b?period=28days&tags=scm.branch%3Amain%2Cresult%3Afailed

### 🐛 Describe the bug

Failing tests:

```
FAILED v1/engine/test_engine_core_client.py::test_kv_cache_events[True-tcp] - AssertionError: No message received
assert None is not None
```

<details>
<summary>Logs:</summary>

```
=================================== FAILURES ===================================
________________________ test_kv_cache_events[True-tcp] ________________________

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7fc027da70e0>
multiprocessing_mode = True
publisher_config = KVEventsConfig(enable_kv_cache_events=True, publisher='zmq', endpoint='tcp://*:51905', replay_endpoint='tcp://*:51906', buffer_steps=100, hwm=1000, max_queue_size=100000, topic='test')

    @pytest.mark.parametrize(
        "multiprocessing_mode,publisher_c

[... truncated for brevity ...]

---

## Issue #N/A: Frontend Improvements

**Link**: https://github.com/vllm-project/vllm/issues/47
**State**: closed
**Created**: 2023-04-22T03:57:50+00:00
**Closed**: 2023-05-24T04:39:52+00:00
**Comments**: 3

### Description

1. Current implementation of the FastAPI+asyncio+ray combination seems slow
2. Merge Hao’s throughput profiling code.
3. Make the frontend looks like OpenAI’s API.


---

## Issue #N/A: [Bug][Failing Test] - Quantization test - quantization/test_cpu_offload.py

**Link**: https://github.com/vllm-project/vllm/issues/18425
**State**: closed
**Created**: 2025-05-20T16:15:31+00:00
**Closed**: 2025-05-21T17:25:49+00:00
**Comments**: 4
**Labels**: bug, ci-failure

### Description

### Your current environment

Failing on main as of commit 9609327fa4

### 🐛 Describe the bug

Failing tests:

```
FAILED quantization/test_cpu_offload.py::test_cpu_offload_gptq - RuntimeError: Server exited unexpectedly.
FAILED quantization/test_cpu_offload.py::test_cpu_offload_awq - RuntimeError: Server exited unexpectedly.
FAILED quantization/test_cpu_offload.py::test_cpu_offload_compressed_tensors - AssertionError: Results for model='nm-testing/llama7b-one-shot-2_4-w4a16-marlin24-t' are not the same.
ref_args=[] ref_envs=None
compare_args=['--cpu-offload-gb', '1'] compare_envs=None
ref_result={'test': 'single_completion', 'text': ' ... ... . Today I', 'finish_reason': 'length', 'usage': CompletionUsage(completion_tokens=5, prompt_tokens=6, total_tokens=11, completion_tokens_details=None, prompt_tokens_details=None)}
compare_result={'test': 'single_completion', 'text': ' ... ... .\n I', 'finish_reason': 'length', 'usage': CompletionUsage(completion_tokens=5, prompt_tokens=6, total_t

[... truncated for brevity ...]

---

## Issue #N/A: Add docstring

**Link**: https://github.com/vllm-project/vllm/issues/74
**State**: closed
**Created**: 2023-05-06T04:55:55+00:00
**Closed**: 2023-06-07T10:25:22+00:00
**Comments**: 0
**Labels**: documentation

### Description

No description provided.

---

## Issue #N/A: [Misc]: Question re: ROCm + Triton Backend

**Link**: https://github.com/vllm-project/vllm/issues/3921
**State**: closed
**Created**: 2024-04-08T18:17:43+00:00
**Closed**: 2024-04-08T19:19:24+00:00
**Comments**: 2
**Labels**: misc

### Description

### Anything you want to discuss about vllm.

We have one question re: the meet-up slides from last week:
<img width="957" alt="image" src="https://github.com/vllm-project/vllm/assets/7945038/2bdceea7-65eb-4e67-91ac-432006251256">
What is the "ROCm + Triton Backend"? We don't see it in the code currently. Does that already exist somewhere in a PR/branch/fork somewhere? 

---

## Issue #N/A: Virtual Office Hours: July 9 and July 25

**Link**: https://github.com/vllm-project/vllm/issues/5937
**State**: closed
**Created**: 2024-06-27T22:04:46+00:00
**Closed**: 2024-12-19T02:04:54+00:00
**Comments**: 4
**Labels**: misc, stale

### Description

## vLLM Virtual Open Office Hours

We enjoyed seeing everyone at the previous office hours and got great feedback. These office hours are a ~bi-weekly live event where you come to learn more about the vLLM project, how to contribute, and get help with your issues - with special topics and guests along the way.

Sign up here: https://neuralmagic.com/community-office-hours/
Here is a recording from June 20 so you can see the format: https://www.youtube.com/watch?v=ss02R8ndKnk

Dates:
- July 9, 2024, at 2:00 PM EST (11:00 AM PST), **Guest Topic:** FP8 Quantization Deep Dive
- July 25, 2024, at 2:00 PM EST (11:00 AM PST), **Guest Topic:** Model Compression for Fast and Efficient Inference

If there are any themes or topics you would like to see addressed, please comment below.

Previous issues:
- https://github.com/vllm-project/vllm/issues/4538
- https://github.com/vllm-project/vllm/issues/4919

---

## Issue #N/A: [Feature]: Automatically detect numerical issues

**Link**: https://github.com/vllm-project/vllm/issues/17123
**State**: open
**Created**: 2025-04-24T16:58:29+00:00
**Comments**: 1
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Models such as Gemma-3 and GLM-4 may encounter numerical instability when `float16` dtype is used. Despite a warning message `Casting bfloat16 to float16` being printed, users can still get confused when the model returns empty or nonsense outputs.

Examples:
- https://github.com/vllm-project/vllm/issues/16489
- https://github.com/vllm-project/vllm/pull/16618#issuecomment-2814399522

It would be great if we could automatically detect numerical issues while running the model. We should at least check for this during startup. Inference-time checking should be optional since it harms the performance.

### Alternatives

Hardcode specific models to not allow them to be run in float16, like `plamo2`:

https://github.com/vllm-project/vllm/blob/4115f19958d8b3628606d78355c277b328f011e1/vllm/config.py#L2834

However, this has to be done manually.

### Additional context

cc @youkaichao 

### Before submitting a new issue...

- [x] Make sure you already se

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: MLA correctness issues when using FA2

**Link**: https://github.com/vllm-project/vllm/issues/18561
**State**: closed
**Created**: 2025-05-22T19:50:32+00:00
**Closed**: 2025-05-28T08:59:41+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
Collecting environment information...
==============================
        System Info
==============================
OS                           : Ubuntu 22.04.5 LTS (x86_64)
GCC version                  : (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version                : 14.0.0-1ubuntu1.1
CMake version                : version 3.31.2
Libc version                 : glibc-2.35

==============================
       PyTorch Info
==============================
PyTorch version              : 2.7.0+cu126
Is debug build               : False
CUDA used to build PyTorch   : 12.6
ROCM used to build PyTorch   : N/A

==============================
      Python Environment
==============================
Python version               : 3.12.4 (main, Jul 25 2024, 22:42:01) [Clang 18.1.8 ] (64-bit runtime)
Python platform              : Linux-6.5.0-35-generic-x86_64-with-glibc2.35

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: [Errno 98] error while attempting to bind on address ('0.0.0.0', 8000): address already in use

**Link**: https://github.com/vllm-project/vllm/issues/8204
**State**: closed
**Created**: 2024-09-05T17:35:19+00:00
**Closed**: 2024-09-16T20:56:29+00:00
**Comments**: 8
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```

</details>


### 🐛 Describe the bug

This is a bug we encounter a lot in our ci, e.g. https://buildkite.com/vllm/ci-aws/builds/8098#0191bf43-446d-411d-80c7-3ba10bc392e8/192-1557

I have been tracking this for months, and try to add more logging information to help debugging.

from the logging information:


> [2024-09-05T00:38:34Z] INFO:     Started server process [60858]
> --
>   | [2024-09-05T00:38:34Z] INFO:     Waiting for application startup.
>   | [2024-09-05T00:38:34Z] INFO:     Application startup complete.
>   | [2024-09-05T00:38:34Z] ERROR:    [Errno 98] error while attempting to bind on address ('0.0.0.0', 44319): [errno 98] address already in use
>   | [2024-09-05T00:38:34Z] INFO:     Waiting for application shutdown.
>   | [2024-09-05T00:38:34Z] INFO:     Application shutdown complete.
>   | [2024-0

[... truncated for brevity ...]

---

## Issue #N/A: Mixtral generation speed performance.

**Link**: https://github.com/vllm-project/vllm/issues/2048
**State**: closed
**Created**: 2023-12-12T05:05:28+00:00
**Closed**: 2023-12-14T00:32:26+00:00
**Comments**: 0

### Description

I tried to deploy the Mixtral 7bx8 model on eight T4 GPUs, but the generation speed is only 6 tokens/s, while a 34B model achieves 14 tokens/s. 
I've heard someone mention that Mixtral 7bx8's generation performance is comparable to a 12B model, but I'm unsure what the issue might be.

---

## Issue #N/A: [Usage]: Clean up Engine Args & Documentation

**Link**: https://github.com/vllm-project/vllm/issues/14386
**State**: open
**Created**: 2025-03-06T23:59:07+00:00
**Comments**: 8
**Labels**: good first issue, usage

### Description

### Your current environment

Currently vLLM has a lot of engine arguments listed here https://docs.vllm.ai/en/latest/serving/engine_args.html. Over time as we add more and more features to vLLM, this list will be less maintainable and user friendly.



### How would you like to use vllm

As a first step to clean up these args, they should be made **hierarchical** (for example, `--compilation-config`).

The documentation should also be updated so that engine arg documentations are **arranged in sections instead of in a flatten list**.


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: non-deterministic Python gc order leads to flaky tests

**Link**: https://github.com/vllm-project/vllm/issues/5337
**State**: closed
**Created**: 2024-06-07T05:39:26+00:00
**Closed**: 2024-06-08T05:31:33+00:00
**Comments**: 13
**Labels**: bug

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### 🐛 Describe the bug

I often see many flaky tests, and I think the Python gc system is one of the factor to blame.

Python gc system is notoriously random. When we call `del x`, and `x`'s refcount is 0, it is not guaranteed that all resources held by `x` will be released immediately. This is true especially when `x` is a complicated object, and might contain self-reference inside it. That's one of the motivation for Python to propose the concept of context manager, to enforce some critical resource to be released immediately.

Take the following code as an example:

```python
import torch
import weakref
import gc

def tensor_destructed(tensor_ref):
    # This function is called when the tensor is destructed.
    print(f"Tensor with id {id(tensor_ref)} is being destructed.")

class A:
    def __init__(self):
        self.tensor = torch.tensor([1.0, 2.0, 3.0])

    def __del__

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: benchmarking vllm copy kernel and pytorch index copy

**Link**: https://github.com/vllm-project/vllm/issues/4698
**State**: closed
**Created**: 2024-05-09T02:38:35+00:00
**Closed**: 2024-11-28T02:05:08+00:00
**Comments**: 6
**Labels**: help wanted, performance, stale

### Description

### Proposal to improve performance

I opened this issue to track a random idea:

Currently we have a copy kernel:

https://github.com/vllm-project/vllm/blob/e288df0632d5bdde76c20bed8310b46d35b8e5ac/csrc/cache_kernels.cu#L214-L220

Essentially this does the following vector copy:

```python
    key_cache_view = key_cache.reshape(-1, num_heads * head_size)
    value_cache_view = value_cache.reshape(-1, num_heads * head_size)
    key_view = key.reshape(-1, num_heads * head_size)
    value_view = value.reshape(-1, num_heads * head_size)
    key_cache_view[slot_mapping] = key_view
    value_cache_view[slot_mapping] = value_view
```

The caveat is, we have a special value in `slot_mapping`: `-1` means skip copying.

If possible, we can reserve a slot in block manager for padded kv, then we can just use pytorch's index copying, without maintaining a separate copy kernel ourselves.

Two TODOs:

- [ ] What is the overhead of reserving a slot for padded kv in the block ma

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Support FP8 Marlin MoE for CompressedTensorsW8A8Fp8MoEMethod

**Link**: https://github.com/vllm-project/vllm/issues/18008
**State**: closed
**Created**: 2025-05-12T18:02:12+00:00
**Closed**: 2025-05-20T11:58:40+00:00
**Comments**: 4
**Labels**: good first issue, feature request, quantization

### Description

### 🚀 The feature, motivation and pitch

Like what was added in https://github.com/vllm-project/vllm/pull/16850 for enabling marlin in fp8.py MoE layers, we should enable FP8 Marlin MoE for compressed tensors models to support users wanting to run them on older hardware.

Basically you want to take the changes in fp8.py's moe method (https://github.com/vllm-project/vllm/pull/16850/files#diff-5511bfcc9c53f7d96517ad43e4087f6777bef21302da983f42cafae40a866644) and apply them to `CompressedTensorsW8A8Fp8MoEMethod`

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Flash Attention 3 (FA3) Support

**Link**: https://github.com/vllm-project/vllm/issues/12429
**State**: open
**Created**: 2025-01-25T19:48:27+00:00
**Comments**: 2

### Description

As of https://github.com/vllm-project/vllm/pull/12093 Flash Attention 3 is now supported in vLLM for Hopper GPUs (SM 9.0).

It can also be enabled for SM 8.0 and 8.7 using `VLLM_FLASH_ATTN_VERSION=3`.

For 8.6 and 8.9 its fully disabled since they don't have enough shared memory for the current implementation, some work needs to be done here.

This issue tracks the remaining features that have yet to be implemented


### Hardware Support
- [ ] SM 8.9 Ada Lovelace (L4, L40s) Support 
- [ ] SM 8.6 Ampere (A6000) Support


### Optimizations
- [x] FP8 Attention

---

## Issue #N/A: Unexpected latency of StarCoder when enable tensor parallel

**Link**: https://github.com/vllm-project/vllm/issues/696
**State**: closed
**Created**: 2023-08-08T04:25:01+00:00
**Closed**: 2024-03-08T16:56:48+00:00
**Comments**: 3
**Labels**: bug

### Description

### Discussed in https://github.com/vllm-project/vllm/discussions/666

<div type='discussions-op-text'>

<sup>Originally posted by **zhaoyang-star** August  3, 2023</sup>
The latency of StarCoder running on 2 A100 40GB is higher than that running on 1 A100. 
While, the latency of LLaMA-13B running on 2 A100 40GB is lower than that running on 1 A100 as expected. 

So does MultiQueryAttenion impl in vLLM cause this? 

![image](https://github.com/vllm-project/vllm/assets/24290792/ea66ad84-d8f3-4793-9970-ca70b07c13b0)
![image](https://github.com/vllm-project/vllm/assets/24290792/37a8c774-1b5a-4891-86db-ec464e79f58f)
Note: Prompt token length and output token length both are set to 1k.
</div>

---

## Issue #N/A: [Bug]: dag teardown error AttributeError: 'Worker' object has no attribute 'core_worker'

**Link**: https://github.com/vllm-project/vllm/issues/6887
**State**: closed
**Created**: 2024-07-29T05:32:43+00:00
**Closed**: 2024-12-01T02:14:37+00:00
**Comments**: 5
**Labels**: bug, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### 🐛 Describe the bug

command:

`python benchmarks/benchmark_throughput.py --input-len 100 --output-len 100 --num-prompts 100 --model facebook/opt-125m -tp 2 --distributed-executor-backend ray`

error:

```text
2024-07-28 22:30:36,078 INFO compiled_dag_node.py:1202 -- Tearing down compiled DAG
Exception ignored in: <function RayGPUExecutor.__del__ at 0x7ff2ee7048b0>
Traceback (most recent call last):
  File "/data/youkaichao/vllm/vllm/executor/ray_gpu_executor.py", line 396, in __del__
    self.forward_dag.teardown()
  File "/data/youkaichao/miniconda/envs/vllm/lib/python3.9/site-packages/ray/dag/compiled_dag_node.py", line 1402, in teardown
    monitor.teardown(wait=True)
  File "/data/youkaichao/miniconda/envs/vllm/lib/python3.9/site-packages/ray/dag/compiled_dag_node.py", line 1204, in teardown
    outer._dag_submitter.close()
  File "/data/youkaichao/miniconda/envs/vllm/lib/python

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Interface and Abstraction for Distributed Inference Environment

**Link**: https://github.com/vllm-project/vllm/issues/3587
**State**: closed
**Created**: 2024-03-23T23:41:40+00:00
**Closed**: 2024-06-14T01:00:32+00:00
**Comments**: 18
**Labels**: RFC, misc

### Description

This RFC describes a proposal for interfaces and abstractions for distributed inference environments. I plan to solicit discussions for a week (until March 31st) before I begin to actually refactor the code.

# Motivation

The current distributed inference environment in `vllm` is quite tangled, and we often see deadlocks and hangs (see https://github.com/vllm-project/vllm/issues/3455 , https://github.com/vllm-project/vllm/issues/2770 , https://github.com/vllm-project/vllm/issues/3559 , to name a few). The problem becomes prominent when we try to upgrade to pytorch 2.2.0 (see https://github.com/vllm-project/vllm/pull/3442 , https://github.com/vllm-project/vllm/pull/3442 ), because `pytorch 2.2.0` upgrades from `nccl==2.18.1` to `2.19.3` (see https://pypi.org/pypi/torch/2.1.2/json and https://pypi.org/pypi/torch/2.2.0/json to compare the dependency), and `nccl==2.19.3` breaks `vllm` due to increased memory cost during cudagraph capture (from 10MB per graph to 100MB per graph, adds u

[... truncated for brevity ...]

---

## Issue #N/A: Tensor Parallel profiling result

**Link**: https://github.com/vllm-project/vllm/issues/22
**State**: closed
**Created**: 2023-04-02T06:50:04+00:00
**Closed**: 2023-06-16T02:38:15+00:00
**Comments**: 0

### Description

Will update the profiling results in this PR.

## BS=8, input_len=32, output_len=128

```
OPT-13B
TP 1: 3.5404738585154214 seconds
TP 2: 4.742188215255737 seconds
TP 4: 4.907034238179524 seconds

OPT-30B
TP 1: OOM
TP 2: 5.9848620891571045 seconds
TP 4: 5.943212985992432 seconds
```

---

## Issue #N/A: The First vLLM SF Bay Area Meetup

**Link**: https://github.com/vllm-project/vllm/issues/1149
**State**: closed
**Created**: 2023-09-22T18:38:22+00:00
**Closed**: 2023-10-08T04:10:22+00:00
**Comments**: 8

### Description

We are excited to announce that we will hold the first vLLM SF bay area meetup at 6pm on 10/5 (Thu). The vLLM team will give a deep dive on vLLM and share the future plans and roadmaps of the project. We will also have our users and contributors come and share their experiences. You can find the event details and RSVP [here](https://lu.ma/first-vllm-meetup). Please join us if you are around the bay area. Let's come together, share our stories, and envision the future of vLLM.


---

## Issue #N/A: [Bug]: Speculative decoding does not respect per-request seed

**Link**: https://github.com/vllm-project/vllm/issues/6038
**State**: closed
**Created**: 2024-07-01T17:11:21+00:00
**Closed**: 2024-07-19T02:22:09+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

```text
Collecting environment information...
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.29.5
Libc version: glibc-2.35

Python version: 3.11.9 | packaged by conda-forge | (main, Apr 19 2024, 18:36:13) [GCC 12.3.0] (64-bit runtime)
Python platform: Linux-5.15.0-107-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H100 80GB HBM3
GPU 1: NVIDIA H100 80GB HBM3
GPU 2: NVIDIA H100 80GB HBM3
GPU 3: NVIDIA H100 80GB HBM3
GPU 4: NVIDIA H100 80GB HBM3
GPU 5: NVIDIA H100 80GB HBM3
GPU 6: NVIDIA H100 80GB HBM3
GPU 7: NVIDIA H100 80GB HBM3

Nvidia driver version: 550.54.15
cuDNN version: Probably one of the fo

[... truncated for brevity ...]

---

## Issue #N/A: [Bug[Failing Test]: Entrypoints Test - entrypoints/openai/test_transcription_validation.py

**Link**: https://github.com/vllm-project/vllm/issues/18592
**State**: closed
**Created**: 2025-05-23T05:33:05+00:00
**Closed**: 2025-05-23T12:51:54+00:00
**Comments**: 1
**Labels**: bug, ci-failure

### Description

### Your current environment

N/A

### 🐛 Describe the bug

```[2025-05-23T04:39:51Z] Traceback (most recent call last):
[2025-05-23T04:39:51Z]   File "/usr/local/bin/vllm", line 10, in <module>
[2025-05-23T04:39:51Z]     sys.exit(main())
[2025-05-23T04:39:51Z]              ^^^^^^
[2025-05-23T04:39:51Z]   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/main.py", line 53, in main
[2025-05-23T04:39:51Z]     args.dispatch_function(args)
[2025-05-23T04:39:51Z]   File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/serve.py", line 40, in cmd
[2025-05-23T04:39:51Z]     uvloop.run(run_server(args))
[2025-05-23T04:39:51Z]   File "/usr/local/lib/python3.12/dist-packages/uvloop/__init__.py", line 109, in run
[2025-05-23T04:39:51Z]     return __asyncio.run(
[2025-05-23T04:39:51Z]            ^^^^^^^^^^^^^^
[2025-05-23T04:39:51Z]   File "/usr/lib/python3.12/asyncio/runners.py", line 195, in run
[2025-05-23T04:39:51Z]     return runner.run(main)
[2025-05-23T04:39:51Z]

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: bind python and c++ through tools other than pybind11

**Link**: https://github.com/vllm-project/vllm/issues/4694
**State**: closed
**Created**: 2024-05-08T23:18:56+00:00
**Closed**: 2024-10-27T22:53:29+00:00
**Comments**: 3
**Labels**: help wanted, feature request, stale

### Description

### 🚀 The feature, motivation and pitch

As vLLM goes into a fast release schedule (currently one release every two weeks), we will quickly hit the project-wide limit of pypi (around 5GB per project). One solution, as pointed out in https://github.com/pypi/support/issues/3792#issuecomment-2099941677 , is to build one wheel for all python versions (Python 3.8+).

I have figured out the procedure https://github.com/pypi/support/issues/3792#issuecomment-2101360740 , but pybind11 does not support this Python Limited API protocol.

One possible solution is to replace pybind11 with some other tools, so that the binding procedure can be used with Python Limited API.

Possible solutions:

- Nanobind (seems to support it starting from Python 3.12 only: https://github.com/wjakob/nanobind/pull/561 )
- register ops through pytorch directly https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html

### Alternatives

_No response_

### Additional context

_No response_

---

## Issue #N/A: [Bug][0.5.4] Front-end server errors when overloaded with pending requests

**Link**: https://github.com/vllm-project/vllm/issues/7309
**State**: closed
**Created**: 2024-08-08T17:12:34+00:00
**Closed**: 2024-12-08T02:10:45+00:00
**Comments**: 2
**Labels**: bug, stale

### Description

### Your current environment

The output of `python collect_env.py`:

<details>

```text
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.2
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-107-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.5.82
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A100-SXM4-80GB
GPU 1: NVIDIA A100-SXM4-80GB
GPU 2: NVIDIA A100-SXM4-80GB
GPU 3: NVIDIA A100-SXM4-80GB
GPU 4: NVIDIA A100-SXM4-80GB
GPU 5: NVIDIA A100-SXM4-80GB
GPU 6: NVIDIA A100-SXM4-80GB
GPU 7: NVIDIA A100-SXM4-80GB

Nvidia driver version: 550.54.15
cuDNN version: Could not collect
HIP runtime version: N

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: From SequenceGroup-native code to Sequence-native code

**Link**: https://github.com/vllm-project/vllm/issues/7116
**State**: closed
**Created**: 2024-08-04T00:54:31+00:00
**Closed**: 2024-12-04T02:07:32+00:00
**Comments**: 2
**Labels**: performance, stale

### Description

### Proposal to improve performance

We have two concepts in vLLM:

- SequenceGroup, a group of sequence, that originates from the same request. In most usecases, a sequence group contains only one sequence. In parallel sampling, a request can fork into many sequences, depending on the sampling parameter `n`. In beam search, sequences in the sequence group can change, grow, die.
- Sequence, consists of a sequence seen by the inference engine. It has prompt, generated tokens, kv cache...

In order to support diverse sampling algorithms, vLLM currently takes a SequenceGroup-native approach: many functions operate in the SequenceGroup-level, e.g. `prepare_input` takes in a list of `SequenceGroup`.

The problem is, many functions in an inference engine, naturally fit into Sequence-level operations. For example, when we talk about the batchsize for decoding, it is the number of Sequence we are running for decoding, not the number of SequenceGroup.

To fill in the gap, there are man

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vLLM OpenAI-api server `/docs` endpoint fails to load

**Link**: https://github.com/vllm-project/vllm/issues/9168
**State**: closed
**Created**: 2024-10-08T21:03:03+00:00
**Closed**: 2025-01-17T02:04:04+00:00
**Comments**: 4
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.2
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.5.0-35-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.5.82
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H100 80GB HBM3
GPU 1: NVIDIA H100 80GB HBM3
GPU 2: NVIDIA H100 80GB HBM3
GPU 3: NVIDIA H100 80GB HBM3
GPU 4: NVIDIA H100 80GB HBM3
GPU 5: NVIDIA H100 80GB HBM3
GPU 6: NVIDIA H100 80GB HBM3
GPU 7: NVIDIA H100 80GB HBM3

Nvidia driver version: 555.42.02
cuD

[... truncated for brevity ...]

---

