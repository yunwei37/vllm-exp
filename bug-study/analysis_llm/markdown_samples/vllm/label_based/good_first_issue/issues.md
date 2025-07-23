# good_first_issue - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 5
- Closed Issues: 25

### Label Distribution

- good first issue: 30 issues
- feature request: 15 issues
- bug: 8 issues
- help wanted: 8 issues
- misc: 3 issues
- documentation: 2 issues
- unstale: 1 issues
- stale: 1 issues
- multi-modality: 1 issues

---

## Issue #N/A: [Bug]: vllm serve --config.yaml - Order of arguments matters?

**Link**: https://github.com/vllm-project/vllm/issues/8947
**State**: closed
**Created**: 2024-09-29T15:06:36+00:00
**Closed**: 2024-10-05T17:35:13+00:00
**Comments**: 11
**Labels**: bug, good first issue

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Rocky Linux release 8.10 (Green Obsidian) (x86_64)
GCC version: (GCC) 11.3.0
Clang version: Could not collect
CMake version: version 3.26.5
Libc version: glibc-2.28

Python version: 3.11.5 (main, Sep 11 2023, 13:54:46) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-4.18.0-553.16.1.el8_10.x86_64-x86_64-with-glibc2.28
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H100
GPU 1: NVIDIA H100
GPU 2: NVIDIA H100
  MIG 1g.12gb     Device  0:
  MIG 1g.12gb     Device  1:
  MIG 1g.12gb     Device  2:
  MIG 1g.12gb     Device  3:
  MIG 1g.12gb     Device  4:
  MIG 1g.12gb     Device  5:
  MIG 1g.12gb     Device  6:
GPU 3: NVIDIA H100

Nvidia d

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: Implement CPU/GPU swapping in BlockManagerV2

**Link**: https://github.com/vllm-project/vllm/issues/3666
**State**: closed
**Created**: 2024-03-27T21:40:21+00:00
**Closed**: 2024-06-03T20:41:11+00:00
**Comments**: 5
**Labels**: good first issue, misc

### Description

Recently, we refactored the block manager subsystem to improve testability by separating concerns of each layer. See https://github.com/vllm-project/vllm/pull/3492 for more information.

The V2 implementation does not have support for CPU-GPU swapping. It can be added in the [CpuGpuBlockAllocator](https://github.com/vllm-project/vllm/blob/321dc1619ad60b6df74fa86ac6299bc83c223996/vllm/core/block/cpu_gpu_block_allocator.py). My first take on the design is that it should simply keep track of the requested swap requests and have the scheduler `get_and_clear` them after each scheduling step.

![image](https://github.com/vllm-project/vllm/assets/950914/55cf0db2-2614-463b-a053-eb3f182c01bb)


---

## Issue #N/A: [HELP WANTED] Fix Failing Spec Decoding Test

**Link**: https://github.com/vllm-project/vllm/issues/18166
**State**: closed
**Created**: 2025-05-14T20:14:33+00:00
**Closed**: 2025-05-19T02:49:47+00:00
**Comments**: 2
**Labels**: bug, good first issue

### Description

### Issue

We are seeing a test failure related to EAGLE on V0. We would appreciate anyone who can help addressing it. 

```bash
pytest -s -v tests/spec_decode/e2e/test_eagle_correctness.py::test_eagle_e2e_greedy_correctness_with_preemption
```

PR which disables the test: https://github.com/vllm-project/vllm/pull/18165

If anyone has capacity to help out with re-enabling this, we would greatly appreciate it!

---

## Issue #N/A: [Feature]: Testing - Use `torch.testing.assert_close` instead of `torch.allclose` as a Recommended Practice

**Link**: https://github.com/vllm-project/vllm/issues/7307
**State**: closed
**Created**: 2024-08-08T16:13:20+00:00
**Closed**: 2024-08-16T04:24:05+00:00
**Comments**: 0
**Labels**: good first issue, feature request

### Description

### 🚀 The feature, motivation and pitch

See https://pytorch.org/docs/stable/testing.html

`assert_close` will print the values which violate the allclose condition. `assert torch.allclose` will not

This leads to better diagnosability of failed tests




---

## Issue #N/A: [Feature]: Batch inference for `llm.chat()` API

**Link**: https://github.com/vllm-project/vllm/issues/8481
**State**: closed
**Created**: 2024-09-14T03:17:43+00:00
**Closed**: 2024-09-24T16:44:12+00:00
**Comments**: 1
**Labels**: good first issue, feature request

### Description

### 🚀 The feature, motivation and pitch

Currently `llm.chat()` API only supports one conversation per inference. This means we cannot use this API to fully leverage vLLM for efficient offline processing.

### Alternatives

_No response_

### Additional context


Implementation should be rather straightforward:

1. at API level, `llm.chat()` should also accept a list of conversations.
2. When `llm.chat()` is invoked, the list of conversations will be parsed into list of prompts, and all multimodal data items will be retrieved and loaded into their corresponding format that `llm.generate()` accepts.
3. Send the list of `{prompt: xxx, multi_modal_data: xxx}` to the `llm.generate()`

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Support `response_format: json_object` in OpenAI server

**Link**: https://github.com/vllm-project/vllm/issues/3148
**State**: closed
**Created**: 2024-03-01T19:12:00+00:00
**Closed**: 2024-03-16T20:35:28+00:00
**Comments**: 1
**Labels**: help wanted, good first issue

### Description

We just merged the support for structured generation support with Outlines. The next step is to integreate with Grammar based finite state machine https://github.com/outlines-dev/outlines/pull/541 into vLLM to support arbitrary JSON format. 

---

## Issue #N/A: [Feature]: Consolidate performance benchmark datasets

**Link**: https://github.com/vllm-project/vllm/issues/13351
**State**: closed
**Created**: 2025-02-16T09:04:02+00:00
**Closed**: 2025-03-15T05:49:45+00:00
**Comments**: 3
**Labels**: help wanted, good first issue, feature request

### Description

### 🚀 The feature, motivation and pitch

On vLLM we have two main benchmark scripts ([benchmark_throughput.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_throughput.py) and [benchmark_serving.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py)) to measure the performance of vLLM. 

However, the dataset sampling functions are defined within each script itself and over time it'll be hard to maintain these and to add new datasets to both scripts as we want to have the flexibility to run benchmark on different datasets.

### Alternatives

Ideally the dataset sampling should be defined in a separate file (e.g, `benchmark_dataset.py`) where we define the sampling functions for different datasets (sharegpt, sonnet, random, vision arena, etc), and the benchmark scripts themselves can simply import from benchmark_dataset depending on which dataset is specified at command line. 

This modularization brings us a number of benefits:
- Ensure

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: --enable-prompt-tokens-details not working in V1

**Link**: https://github.com/vllm-project/vllm/issues/16162
**State**: closed
**Created**: 2025-04-07T07:00:55+00:00
**Closed**: 2025-05-26T10:14:34+00:00
**Comments**: 4
**Labels**: bug, good first issue

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 04-07 06:47:48 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Debian GNU/Linux 11 (bullseye) (x86_64)
GCC version: (Debian 10.2.1-6) 10.2.1 20210110
Clang version: Could not collect
CMake version: version 3.18.4
Libc version: glibc-2.31

Python version: 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.10.0-34-cloud-amd64-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA L4
Nvidia driver version: 550.90.07
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Archit

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Multiple openai endpoint Missing Content-Type Header

**Link**: https://github.com/vllm-project/vllm/issues/17036
**State**: closed
**Created**: 2025-04-23T08:06:01+00:00
**Closed**: 2025-05-10T06:13:33+00:00
**Comments**: 1
**Labels**: bug, good first issue

### Description

### Your current environment

Not reletaed

### 🐛 Describe the bug

During property-based testing of the vLLM API, defined by an OpenAPI 3.1 schema, we observed a recurring issue with missing Content-Type headers across several endpoints.

```python
1. GET /health:
Issue: Missing Content-Type header

Details: The test failed because the server expects a Content-Type: application/json header, but it was not included in the request.


2. GET /ping:
Issue: Missing Content-Type header

Details: Similar to the previous failure, the Content-Type: application/json header was missing for the request to /ping.



3. POST /ping:
Issue: Missing Content-Type header

Details: The failure is due to the absence of the Content-Type: application/json header in the POST request to /ping.

logs:

test_new.py u,uuuu,,uu,,,,u,,,u.                                                                                                                                                                                   

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: gguf file without .gguf extension fails to run, even with "--quantization gguf --load-format gguf" flags

**Link**: https://github.com/vllm-project/vllm/issues/7993
**State**: closed
**Created**: 2024-08-29T10:58:28+00:00
**Closed**: 2024-09-02T12:43:27+00:00
**Comments**: 2
**Labels**: bug, good first issue

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
$ python collect_env.py
Collecting environment information...
WARNING 08-29 11:55:28 _custom_ops.py:17] Failed to import from vllm._C with ImportError('libcuda.so.1: cannot open shared object file: No such file or directory')
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Fedora release 40 (Forty) (x86_64)
GCC version: (GCC) 14.2.1 20240801 (Red Hat 14.2.1-1)
Clang version: 18.1.6 (Fedora 18.1.6-3.fc40)
CMake version: version 3.28.2
Libc version: glibc-2.39

Python version: 3.12.5 (main, Aug  7 2024, 00:00:00) [GCC 14.2.1 20240801 (Red Hat 14.2.1-1)] (64-bit runtime)
Python platform: Linux-6.10.6-200.fc40.x86_64-x86_64-with-glibc2.39
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: No CUDA
Nvidia driver version: No CUD

[... truncated for brevity ...]

---

## Issue #N/A: outputs includes eos token

**Link**: https://github.com/vllm-project/vllm/issues/2538
**State**: closed
**Created**: 2024-01-22T03:14:12+00:00
**Closed**: 2024-02-27T22:05:41+00:00
**Comments**: 1
**Labels**: good first issue

### Description

### code

```
gen_kwargs = {"top_p": top_p, "temperature": temperature, "max_tokens": max_length, "include_stop_str_in_output": False}
eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"), tokenizer.get_command("<|observation|>")]
sampling_params = SamplingParams(stop_token_ids=eos_token_id, **gen_kwargs)
outputs = self.model.generate(sampling_params=sampling_params, prompt_token_ids=inputs["input_ids"].tolist())
```

### outputs string

```
I need to use the insauto_quote_tool tool to get the user's car insurance quote.<|assistant|> insauto_quote_tool
 ```python
tool_call(type='object', properties={'quote_biz_id': '20220906000831000002005700226000'})
```<|observation|>
```

### question

lwhy is '<|observation|>' still at the end?

### environment

Model：ChatGLM3-6b

---

## Issue #N/A: [Feature]: Option to override HuggingFace's configurations

**Link**: https://github.com/vllm-project/vllm/issues/5205
**State**: closed
**Created**: 2024-06-03T05:45:53+00:00
**Closed**: 2024-11-09T16:19:28+00:00
**Comments**: 10
**Labels**: good first issue, feature request, unstale

### Description

### 🚀 The feature, motivation and pitch

The configuration files on HuggingFace may have missing information (e.g. #2051) or contain bugs (e.g. #4008). In such cases, it may be necessary to provide/override the configuration files to enable the model to be loaded correctly. However, apart from chat templates, there is currently no method of doing so; we have to update the source HuggingFace repository directly. It may take time for the authors of those repositories to respond, especially if they are unofficial ones which are not as well-maintained.

It would be great if we could provide our own `config.json`, `tokenizer_config.json`, etc., through the vLLM CLI to apply patches as necessary.

### Related work

#1756 lets us specify alternative chat templates or provide a chat template when it is missing from `tokenizer_config.json`. However, it currently only applies to the OpenAI API-compatible server. #5049 will add chat method to the main LLM entrypoint, but does not provide 

[... truncated for brevity ...]

---

## Issue #N/A: Order of keys for guided JSON

**Link**: https://github.com/vllm-project/vllm/issues/3283
**State**: closed
**Created**: 2024-03-08T13:45:25+00:00
**Closed**: 2024-06-10T18:35:46+00:00
**Comments**: 12
**Labels**: good first issue

### Description

Hi

Trying to use a json template with a mixtral 7x8b model.
However the model generates the json with keys in alphabetic order which has a significant impact on generation quality (for my task at least).

```python

json_template= {
    "type": "object",
    "properties": {
        "first_key": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1
            },
        "another_key": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1
            },
    "required": ["first_key", "another_key"]
    }

chat_response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": prompt},
    ],
    max_tokens=256,
    temperature=0.7,
    top_p=1,
    extra_body=dict(guided_json=json_template)
)

# The returned JSON is in alphabetic order
> "{'another_key': ['some_output'], 'first_key': ['another_output']}"
```
Is there a w

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Embed model has additional dense module(dim=1792, but only 1024)

**Link**: https://github.com/vllm-project/vllm/issues/15509
**State**: open
**Created**: 2025-03-26T01:15:06+00:00
**Comments**: 8
**Labels**: bug, help wanted, good first issue

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.31.4
Libc version: glibc-2.35

Python version: 3.12.9 (main, Feb  5 2025, 08:49:00) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-3.10.0-957.el7.x86_64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA GeForce RTX 3090
GPU 1: NVIDIA GeForce RTX 3090
GPU 2: NVIDIA GeForce RTX 3090
GPU 3: NVIDIA GeForce RTX 3090
GPU 4: NVIDIA GeForce RTX 3090
GPU 5: NVIDIA GeForce RTX 3090
GPU 6: NVIDIA GeForce RTX 3090
GPU 7: NVIDIA GeForce RTX 3090

Nvidia driver version: 535.183.06
cuDNN version: C

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Microbatch Tokenization

**Link**: https://github.com/vllm-project/vllm/issues/19012
**State**: closed
**Created**: 2025-06-02T05:28:07+00:00
**Closed**: 2025-07-07T16:54:12+00:00
**Comments**: 8
**Labels**: good first issue, feature request

### Description

### 🚀 The feature, motivation and pitch

Tokenization is pretty slow (for concurrency=1, DGX H100's CPU can do about only about 1k tokens/ms), under high load, this becomes the performance bottleneck. 

In vLLM's API server today, we process each request's tokenization sequentialy:

https://github.com/vllm-project/vllm/blob/b9f61e13875e1682d3982829006bec26981fde4d/vllm/entrypoints/openai/serving_engine.py#L222-L228

However, just by calling `tokenizer.__call__` with a list of string, significant speed up can be achieved. 

```
In [22]: inp = "hi "*10000

In [23]: inp_batch = [inp]*16

In [24]: %timeit tokenizer(inp)
10.7 ms ± 135 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [25]: %timeit tokenizer(inp_batch)
31.6 ms ± 407 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

There are several ways to speed this up. One approach is to set the thread pool to `N=number_of_cores`, however, @njhill pointed out that transformers's tokenizer actually don't release the G

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Consolidate `LRUCache` implementations

**Link**: https://github.com/vllm-project/vllm/issues/14927
**State**: closed
**Created**: 2025-03-17T05:44:45+00:00
**Closed**: 2025-03-27T06:43:44+00:00
**Comments**: 2
**Labels**: good first issue, feature request

### Description

### 🚀 The feature, motivation and pitch

#14805 introduced `cachetools.LRUCache` to support different size for each item and prepare for a thread-safe implementation. On the other hand, the code under `vllm/adapter_commons` uses the existing `vllm.utils.LRUCache`. To clean up the code, we should consolidate these implementations inside `vllm.utils.LRUCache`. This cache should support the following features:

- Pinning specific items in the cache (the existing `vllm.utils.LRUCache`)
- Custom function to compute the size for each item (`cachetools.LRUCache`)
- Custom callback functions when an item is removed (`vllm.adapter_commons.AdapterLRUCache`)
- The cache should remain compatible with `collections.abc.MutableMapping` interface so it can be passed to `cachetools.cached` to make it thread-safe.

### Alternatives

Keep the two implementations separate. However, this may cause confusion since the two classes share the same name.

### Additional context

_No response_

### Before submit

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: Why prometheus metric vllm:request_success_total doubles the value?

**Link**: https://github.com/vllm-project/vllm/issues/5250
**State**: closed
**Created**: 2024-06-04T13:21:02+00:00
**Closed**: 2024-06-04T19:55:46+00:00
**Comments**: 4
**Labels**: bug, good first issue, misc

### Description

### Anything you want to discuss about vllm.

I am using the following script to display the vllm metric:request_success_total: `sum(increase(vllm:request_success_total{model_name="$MODEL_NAME"}[$__rate_interval])) by (finished_reason)`


But each of my queries in the model is displayed on the graph in the amount of "2". It seems that the value is incremented twice by mistake with a single request.

![image](https://github.com/vllm-project/vllm/assets/21113432/b5b686de-02c8-416d-8797-4d368d00790b)


---

## Issue #N/A: Support for Constrained decoding

**Link**: https://github.com/vllm-project/vllm/issues/288
**State**: closed
**Created**: 2023-06-28T09:23:14+00:00
**Closed**: 2024-03-19T22:46:11+00:00
**Comments**: 32
**Labels**: good first issue, feature request

### Description

For getting structured outputs from custom-finetuned LLMs, extensive use of [constrained decoding](https://huggingface.co/docs/transformers/internal/generation_utils#transformers.DisjunctiveConstraint) is standard. 

Is there a plan to add support for DisjunctiveConstraint (and others) to vLLM in the near future? 
How would one go about implementing this in vLLM (if I were to add a PR)?

---

## Issue #N/A: [Feature]: [V1] Validate / Fix Load Formats on V1

**Link**: https://github.com/vllm-project/vllm/issues/14532
**State**: open
**Created**: 2025-03-10T02:48:53+00:00
**Comments**: 4
**Labels**: good first issue, feature request

### Description

### 🚀 The feature, motivation and pitch

We are not sure if `--load-format sharded_state` of `--load-format tensorizer` work with V1

This issue asks to look into it and fix any issues that occur, including for TP>1

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: Evaluate prompt presence on subsequent audio chunks

**Link**: https://github.com/vllm-project/vllm/issues/19772
**State**: open
**Created**: 2025-06-17T21:08:13+00:00
**Comments**: 2
**Labels**: good first issue, feature request

### Description

### 🚀 The feature, motivation and pitch

Starting with #19597 , vllm now supports chunking audios longer than 30s when serving Whisper.
The logic is pretty simple right now, as the audio is chunked at semi-fixed intervals, looking for "silence" in a small window around the chunk limit.
The request is then executed in a "concurrent mode", batching the audio chunks. 
https://github.com/vllm-project/vllm/blob/cda92307c145e7722cdc33e6d26e105eeb22b882/vllm/entrypoints/openai/serving_transcription.py#L215-L226

Hence there's no sequential dependency at the moment, in particular the transcription of chunk_i is not piped as prompt to chunk_i+1 (optimal strategy, as per the Whisper paper).
In this regard,  it would be nice to asses with longer audio samples whether feeding the original prompt to subsequent chunks after the first one is actually beneficial to the quality of the generated output.
My understanding is that the prompt will condition the model on the text that appeared in the past 30

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: NVIDIA Triton GenAI Perf Benchmark

**Link**: https://github.com/vllm-project/vllm/issues/10377
**State**: closed
**Created**: 2024-11-15T22:03:06+00:00
**Closed**: 2025-02-27T07:25:08+00:00
**Comments**: 7
**Labels**: help wanted, good first issue, feature request, stale

### Description

### 🚀 The feature, motivation and pitch

The GenAI perf toolkit from NVIDIA can be used as an alternative benchmark tools for vLLM. While we already have benchmark scripts and framework in `benchmarks` directory, we should test out different load generators to compare the performance and accuracy of the benchmark clients. 

In this issues, I described some tasks that we need help with to try out the new benchmark harness:
* Compare the output of the genai perf with the `benchmark_serving`, on the coverage of the result metrics and the accuracy. 
* Vary the workloads ShareGPT/Sonnet/synthetics
* Implement it as an alternative harness through the script. 

Happy to elaborate as well. 

https://pypi.org/project/genai-perf/ 

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https:/

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Estimate max-model-len when the KV cache memory is not enough

**Link**: https://github.com/vllm-project/vllm/issues/16118
**State**: closed
**Created**: 2025-04-06T06:02:27+00:00
**Closed**: 2025-04-09T02:12:52+00:00
**Comments**: 5
**Labels**: help wanted, good first issue, feature request

### Description

### 🚀 The feature, motivation and pitch

When the KV cache is not enough for holding one request, vLLM v1 will raise an error like this
> ERROR 04-05 01:12:55 [core.py:390] ValueError: To serve at least one request with the models's max seq len (1048576), (24.00 GiB KV cache is needed, which is larger than the available KV cache memory (9.97 GiB). Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.

It would be more convenient if we can provide an estimated `max_model_len` to the users in this error log.

The estimation is more complex than `max_model_len = block_size * num_gpu_blocks` after the introduction of different types of KV cache like sliding window, and help wanted on implementing with binary search of `max_model_len` based on the `KVCacheSpec.max_memory_usage_bytes`.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant is

[... truncated for brevity ...]

---

## Issue #N/A: [Doc]: Invalid JSON examples in Engine Args Document

**Link**: https://github.com/vllm-project/vllm/issues/11965
**State**: closed
**Created**: 2025-01-12T05:16:06+00:00
**Closed**: 2025-01-14T17:03:06+00:00
**Comments**: 1
**Labels**: documentation, help wanted, good first issue

### Description

### 📚 The doc issue

On page https://docs.vllm.ai/en/latest/serving/engine_args.html#engine-args

Regarding the flag `--override-pooler-config`. The documentation provides the following example:

> Override or set the pooling method for pooling models. e.g. {“pooling_type”: “mean”, “normalize”: false}.’

However this example does not work if copy-pasted into a UTF-8 aware text editor as it is not a valid JSON document. (The quotation marks are not ascii quotation marks, they are left-quote and right-quote.) This is an insidious error as it is nearly invisible to the naked eye.

In addition to `--override-pooler-config`, this issue affects `--override-neuron-config`, `--rope-scaling`, and `--mm-processor-kwargs`.

### Suggest a potential alternative/fix

Change

> Override or set the pooling method for pooling models. e.g. {“pooling_type”: “mean”, “normalize”: false}.’

to

> Override or set the pooling method for pooling models. e.g. `{"pooling_type": "mean", "normalize":

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: Implement SlidingWindowBlockTable in BlockManagerV2

**Link**: https://github.com/vllm-project/vllm/issues/3665
**State**: closed
**Created**: 2024-03-27T21:37:17+00:00
**Closed**: 2024-05-28T02:07:08+00:00
**Comments**: 6
**Labels**: good first issue, misc

### Description

Recently, we refactored the block manager subsystem to improve testability by separating concerns of each layer. See https://github.com/vllm-project/vllm/pull/3492 for more information.

The V2 implementation does not yet have sliding window support. This issue tracks adding sliding window to the V2 block table so that we can support models that use this feature.

My initial take on the design is to implement a `SlidingWindowBlockTable` that composes within it a `BlockTable`. The `SlidingWindowBlockTable` will then drop blocks that are outside of the context window (potentially mapping them to a devnull block). This will preserve the semantics of the v1 block manager sliding window while fitting into the new design.

---

## Issue #N/A: [Feature]: Implement `check_health` for V1

**Link**: https://github.com/vllm-project/vllm/issues/19881
**State**: open
**Created**: 2025-06-19T20:54:34+00:00
**Comments**: 3
**Labels**: good first issue, feature request

### Description

### 🚀 The feature, motivation and pitch

Currently `check_health` is a no-op in V1. We should have an explicit way to check if the engine is still alive and all the subprocesses are healthy. This will enable better functionality in an operational system

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Doc/Feature]: Llava 1.5 in OpenAI compatible server

**Link**: https://github.com/vllm-project/vllm/issues/3873
**State**: closed
**Created**: 2024-04-05T19:07:32+00:00
**Closed**: 2024-06-07T18:25:15+00:00
**Comments**: 10
**Labels**: documentation, help wanted, good first issue

### Description

### 📚 The doc issue

Hey vLLM team it looks like there is added support for llava 1.5 but there are no docs or examples on how to use it via the api server. Are there any reference examples? For using llava via the OpenAI sdk? 

### Suggest a potential alternative/fix

_No response_

---

## Issue #N/A: [Bug]: Unit test `tests/models/embedding/vision_language/test_phi3v.py` failing

**Link**: https://github.com/vllm-project/vllm/issues/14677
**State**: closed
**Created**: 2025-03-12T11:49:15+00:00
**Closed**: 2025-03-30T09:01:36+00:00
**Comments**: 3
**Labels**: bug, help wanted, good first issue

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
Collecting environment information...                                                                                            
PyTorch version: 2.5.1+cu124                                                                                                     
Is debug build: False                                                                                                            
CUDA used to build PyTorch: 12.4                                                                                                 
ROCM used to build PyTorch: N/A                                                                                                  
                                                                                                                                 
OS: Ubuntu 24.04.1 LTS (x86_64)                                       

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Remove xformers requirement for Mistral-format Pixtral and Mistral3

**Link**: https://github.com/vllm-project/vllm/issues/21062
**State**: open
**Created**: 2025-07-16T16:13:25+00:00
**Comments**: 4
**Labels**: good first issue, feature request, multi-modality

### Description

### 🚀 The feature, motivation and pitch

I implemented this a while ago for the HF-format of Pixtral in https://github.com/vllm-project/vllm/pull/9597 by using the torch SDPA implementation. Xformers is not available on all architectures and most other vision encoders have multiple backends for attention. Pixtral is maybe the only that uses xformers strictly.

We should be able to replace the `xops` usage in the `pixtral.py` classes `VisionTransformer` and `Attention` by following the same substitution as in the HF modules.
Such as 
https://github.com/vllm-project/vllm/blob/a0f8a7964694a6077689b242b5eca95de392d4bb/vllm/model_executor/models/pixtral.py#L1274-L1282
and
https://github.com/vllm-project/vllm/blob/a0f8a7964694a6077689b242b5eca95de392d4bb/vllm/model_executor/models/pixtral.py#L1087-L1099


### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot

[... truncated for brevity ...]

---

## Issue #N/A: Possibility of Passing Prompts as List[str] to AsyncEngine.generate()

**Link**: https://github.com/vllm-project/vllm/issues/279
**State**: closed
**Created**: 2023-06-27T14:18:28+00:00
**Closed**: 2024-03-08T10:29:13+00:00
**Comments**: 1
**Labels**: good first issue, feature request

### Description

Hi! Thank you for your amazing framework! I have tried serving a GPT BigCode model using vllm together with ray following the example: https://github.com/ray-project/ray/blob/3d3183d944424a960a2c6ce048abd1316c901c1e/doc/source/serve/doc_code/vllm_example.py And in my use case the response is in "non-streaming" format. I directly passed the request to the vllm async engine to use the continuous batching ability. However, when I tested it with stress testing tool, I found the improvement of the latency and throughput is not that good. 

One reason behind might be that the average length of testing input prompt is quite long (around 1000 tokens) which uses almost all space of KV cache in GPU and some of them are duplicated. Therefore I may need to do a preprocessing of the request in the batch level first to filter out some duplicate requests then pass the request to vllm engine. Currently, if I want to use async engine, I can only pass one prompt to the pool at one time. May I know if 

[... truncated for brevity ...]

---

## Issue #N/A: It seems that SamplingParams doesnt support the bad_words_ids parameter when generating

**Link**: https://github.com/vllm-project/vllm/issues/986
**State**: closed
**Created**: 2023-09-08T01:53:15+00:00
**Closed**: 2024-10-26T16:29:40+00:00
**Comments**: 3
**Labels**: good first issue, feature request

### Description

`bad_words_ids` described [here](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py#L145C11-L145C11) is useful for production applications.
However It seems that vlllm doesnt support the bad_words_ids parameter when generating.
Is there a plan to support it?
```
  bad_words_ids(`List[List[int]]`, *optional*):
  List of list of token ids that are not allowed to be generated. Check
  [`~generation.NoBadWordsLogitsProcessor`] for further documentation and examples.
```

---

