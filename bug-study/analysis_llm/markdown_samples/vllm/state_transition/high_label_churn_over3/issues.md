# high_label_churn_over3 - issues

**Total Issues**: 13
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 9
- Closed Issues: 4

### Label Distribution

- stale: 11 issues
- bug: 8 issues
- feature request: 6 issues
- help wanted: 5 issues
- v1: 4 issues
- good first issue: 4 issues
- multi-modality: 3 issues
- ci/build: 3 issues
- quantization: 2 issues
- tpu: 2 issues

---

## Issue #N/A: [Bug] Mismatch between `get_multimodal_embedding` output and `PlaceholderRange`

**Link**: https://github.com/vllm-project/vllm/issues/15144
**State**: closed
**Created**: 2025-03-19T16:53:23+00:00
**Closed**: 2025-03-30T10:47:54+00:00
**Comments**: 4
**Labels**: bug, help wanted, v1, multi-modality

### Description

In V1, we expect the output of `get_multimodal_embedding` to correspond to the `PlaceholderRange`, which is in turn constructed based on `PromptUpdateDetails.features`. However, the current V1 code doesn't validate this, causing the model to crash during inference when under high load (e.g. #14897, #14963).

From a quick look at the code, these models output embedding sizes which are inconsistent with the placeholder range:

- [x] Fuyu (fixed by #15731)
- [x] Gemma3 (fixed by #14980)
- [x] Idefics3 (fixed by #15696)
- [x] InternVL-based models (fixed by #15086)
- [x] MiniCPM-V (fixed by #15487)

(Basically, any model that has image newline/column tokens after applying HF processor needs a mask to map image patch features to image embeddings, as described below.)

To fix this, we can follow these steps:

1. Update the multi-modal processor to output a mask to indicate which positions in the `PlaceholderRange`-aligned embeddings should the patch features (outputted by vision encoder) be 

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

## Issue #N/A: [Bug]: Disabling Marlin by setting --quantization gptq doesn't work when using a draft model

**Link**: https://github.com/vllm-project/vllm/issues/8784
**State**: closed
**Created**: 2024-09-24T22:51:58+00:00
**Closed**: 2025-01-24T01:58:42+00:00
**Comments**: 2
**Labels**: bug, quantization, speculative-decoding, stale

### Description

### Your current environment

.

### Model Input Dumps

_No response_

### 🐛 Describe the bug

It seems that setting --quantization gptq only disables the marlin for the main model. 

Maybe this can be fixed by adding a --quantization-draft-model setting or forcing the draft model to gptq when main model is forced.

```
INFO 09-24 15:46:11 gptq_marlin.py:112] Detected that the model can run with gptq_marlin, **however you specified quantization=gptq explicitly, so forcing gptq. Use quantization=gptq_marlin for faster inference**
WARNING 09-24 15:46:11 config.py:335] gptq quantization is not fully optimized yet. The speed can be slower than non-quantized models.
INFO 09-24 15:46:11 config.py:904] Defaulting to use mp for distributed inference
**INFO 09-24 15:46:11 gptq_marlin.py:108] The model is convertible to gptq_marlin during runtime. Using gptq_marlin kernel.**
```

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked th

[... truncated for brevity ...]

---

## Issue #N/A: GPTQ does not support bfloat16

**Link**: https://github.com/vllm-project/vllm/issues/2149
**State**: closed
**Created**: 2023-12-17T06:06:30+00:00
**Closed**: 2024-11-30T02:03:24+00:00
**Comments**: 6
**Labels**: help wanted, feature request, quantization, stale

### Description

Currently, our GPTQ kernels only support the float16 precision.

---

## Issue #N/A: [Feature]: Run performance benchmarks for multi-modal models in CI

**Link**: https://github.com/vllm-project/vllm/issues/16353
**State**: open
**Created**: 2025-04-09T16:48:25+00:00
**Comments**: 4
**Labels**: help wanted, feature request, stale, multi-modality

### Description

### 🚀 The feature, motivation and pitch

We currently only have benchmarks for text-only models such as Llama. With the increasing importance of multi-modality and related optimizations such as processor cache, we should add performance benchmarks for multi-modal models to avoid regressions (e.g. memory leaks, slow batching).

We can measure the peak memory usage based on this code:

```python
import resource

max_self_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20)
max_children_usage = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (1 << 20)
print(f"Peak memory usage: {max_self_usage} (self) + {max_children_usage} (children) GiB")
```

### Alternatives

_No response_

### Additional context

cc @mgoin @ywang96 

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequ

[... truncated for brevity ...]

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

## Issue #N/A: [Bug]: CI flake - v1/entrypoints/llm/test_struct_output_generate.py::test_structured_output - JSONDecodeError: Expecting value: line 1 column 1 (char 0)

**Link**: https://github.com/vllm-project/vllm/issues/16053
**State**: open
**Created**: 2025-04-04T09:46:16+00:00
**Comments**: 2
**Labels**: bug, ci/build, stale, v1

### Description

### Your current environment

...


### 🐛 Describe the bug

main commit 51d7c6a2b23e100cd9e7d85b8e7c0eea656b331e

Seen in https://github.com/vllm-project/vllm/pull/15894

https://buildkite.com/organizations/vllm/pipelines/ci/builds/16742/jobs/0195fc58-3d11-45b5-b76f-8e962cbda765/log

```
FAILED v1/entrypoints/llm/test_struct_output_generate.py::test_structured_output[Qwen/Qwen2.5-1.5B-Instruct-guidance:disable-any-whitespace-auto] - json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)

[2025-04-03T16:08:35Z] _ test_structured_output[Qwen/Qwen2.5-1.5B-Instruct-guidance:disable-any-whitespace-auto] _
[2025-04-03T16:08:35Z]
[2025-04-03T16:08:35Z] monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f318d89eb40>
[2025-04-03T16:08:35Z] sample_json_schema = {'properties': {'age': {'type': 'integer'}, 'name': {'type': 'string'}, 'skills': {'items': {'type': 'string'}, 'type'...ition'], 'type': 'object'}, 'type': 'array'}}, 'required': ['name', 'age', 'skills', 'work_

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: CI flake - v1/engine/test_llm_engine.py::test_parallel_sampling[True]

**Link**: https://github.com/vllm-project/vllm/issues/15855
**State**: open
**Created**: 2025-04-01T06:55:01+00:00
**Comments**: 4
**Labels**: bug, ci/build, stale, v1

### Description

### Your current environment

...


### 🐛 Describe the bug

Saw V1 test failing with this yesterday, went away with recheck:

```
[2025-03-31T17:33:47Z] _________________________ test_parallel_sampling[True] _________________________
[2025-03-31T17:33:47Z]
[2025-03-31T17:33:47Z] vllm_model = <tests.conftest.VllmRunner object at 0x7f0d875e06e0>
[2025-03-31T17:33:47Z] example_prompts = ['vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.\n', 'Briefly describe the majo...me.\n', 'Analyze the impact of the COVID-19 pandemic on global economic structures and future business models.\n', ...]
[2025-03-31T17:33:47Z]
[2025-03-31T17:33:47Z]     def test_parallel_sampling(vllm_model, example_prompts) -> None:
[2025-03-31T17:33:47Z]         """Test passes if parallel sampling `n>1` yields `n` unique completions.
[2025-03-31T17:33:47Z]
[2025-03-31T17:33:47Z]         Args:
[2025-03-31T17:33:47Z]           vllm_model: VllmRunner instance under test.
[2025-03-31T17:3

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Support Cascade Attention for Sliding Window Attention

**Link**: https://github.com/vllm-project/vllm/issues/15710
**State**: open
**Created**: 2025-03-28T14:55:29+00:00
**Comments**: 10
**Labels**: help wanted, good first issue, feature request, stale

### Description

### 🚀 The feature, motivation and pitch

Currently, vLLM does not support cascade attention for sliding window attention:

https://github.com/vllm-project/vllm/blob/3b00ff91380044fa409612401309b9cb6a82685f/vllm/v1/attention/backends/flash_attn.py#L352-L354

However, it is technically possible to use it in specific cases. For instance, when the context lengths of all requests in the batch do not exceed the sliding window size, it functions the same as global attention, making it suitable for cascade attention.

As such, we should expand the coverage of cascade attention with sliding window.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: RequestMetrics object (accessed through output[0].metrics) is None

**Link**: https://github.com/vllm-project/vllm/issues/15394
**State**: open
**Created**: 2025-03-24T12:15:17+00:00
**Comments**: 5
**Labels**: bug, good first issue, feature request, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 03-24 12:02:48 [__init__.py:256] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.12.9 (main, Feb 24 2025, 10:05:14) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-131-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA H200
Nvidia driver version: 550.127.08
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Schema for checking input shapes for multi-modal models

**Link**: https://github.com/vllm-project/vllm/issues/14764
**State**: open
**Created**: 2025-03-13T14:57:56+00:00
**Comments**: 29
**Labels**: good first issue, feature request, RFC, multi-modality

### Description

### 🚀 The feature, motivation and pitch

Currently, we use `_parse_and_validate_*_input` to validate the multi-modal inputs. However, only minimal checks are being made, with some models only checking the type of the inputs. It is easy for the actual shape of the inputs to not match what is being documented in classes like `*ImagePixelInputs`, confusing model developers and maintainers.

To avoid this, I propose adding a base class `TensorSchema` to validate the model inputs. For example:

Original code:
```py
class Phi3VImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: Union[torch.Tensor, List[torch.Tensor]]
    """Shape: `(batch_size * num_images, 1 + num_patches, num_channels, height, width)`"""

    image_sizes: torch.Tensor
    """Shape: `(batch_size * num_images, 2)`"""
```

The idea:
```py
class Phi3VImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - b: Batch size (number of prompts)
        - n: Number of images
        - p: Number of patch

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Multi-Node Online Inference on TPUs Failing

**Link**: https://github.com/vllm-project/vllm/issues/12179
**State**: open
**Created**: 2025-01-17T23:38:35+00:00
**Comments**: 5
**Labels**: bug, tpu, ray, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
root@t1v-n-4d36f9a1-w-0:/workspace/vllm# python collect_env.py
INFO 01-17 23:21:42 __init__.py:179] Automatically detected platform tpu.
Collecting environment information...
PyTorch version: 2.6.0.dev20241126+cpu
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Debian GNU/Linux 11 (bullseye) (x86_64)
GCC version: (Debian 10.2.1-6) 10.2.1 20210110
Clang version: Could not collect
CMake version: version 3.31.4
Libc version: glibc-2.31

Python version: 3.10.15 (main, Oct 17 2024, 02:58:23) [GCC 10.2.1 20210110] (64-bit runtime)
Python platform: Linux-5.19.0-1022-gcp-x86_64-with-glibc2.31
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: No CUDA
Nvidia driver version: No CUDA
cuDNN version: No CUDA
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vLLM on TPU does not support --pipeline-parallel-size with Ray

**Link**: https://github.com/vllm-project/vllm/issues/11260
**State**: open
**Created**: 2024-12-17T13:04:46+00:00
**Comments**: 4
**Labels**: bug, tpu, ray, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.6.0.dev20241126+cpu
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.31.2
Libc version: glibc-2.35

Python version: 3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.5.0-1013-gcp-x86_64-with-glibc2.35
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: No CUDA
Nvidia driver version: No CUDA
cuDNN version: No CUDA
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:         

[... truncated for brevity ...]

---

