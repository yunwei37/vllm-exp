# stale_but_active - issues

**Total Issues**: 12
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 12
- Closed Issues: 0

### Label Distribution

- inactive: 12 issues
- good first issue: 3 issues
- high priority: 3 issues
- new-model: 2 issues
- enhancement: 2 issues
- help wanted: 2 issues
- MLLM: 1 issues
- feature: 1 issues
- lora: 1 issues

---

## Issue #N/A: [Feature] Integrate FlashMLA into sgl-kernel

**Link**: https://github.com/sgl-project/sglang/issues/5989
**State**: open
**Created**: 2025-05-03T02:24:13+00:00
**Comments**: 4
**Labels**: inactive

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Integrate FlashMLA into sgl-kernel, so flashmla backend can run without manually installing flashmla package.

### Related resources

_No response_

---

## Issue #N/A: [Bug] Auto-truncation still uses full context length instead of (context_length - max_tokens)

**Link**: https://github.com/sgl-project/sglang/issues/5409
**State**: open
**Created**: 2025-04-15T07:33:29+00:00
**Comments**: 1
**Labels**: good first issue, inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I'm experiencing an issue where prompt auto-truncation doesn't properly account for max_tokens when using the HTTP server, even with allow_auto_truncate=True enabled. This persists after the changes in https://github.com/sgl-project/sglang/pull/4919.

### Reproduction

1.  python -m sglang.launch_server --model-path NousResearch/Hermes-3-L

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] `HF_HUB_OFFLINE` not longer supported in version 0.4.5

**Link**: https://github.com/sgl-project/sglang/issues/5386
**State**: open
**Created**: 2025-04-14T16:54:24+00:00
**Comments**: 1
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Since the latest version, one can no longer set the env variable "HF_HUB_OFFLINE". Setting this variable will lead to the following failure during the model config.
```shell
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/sgl-workspace/sgla

[... truncated for brevity ...]

---

## Issue #N/A: llmcompressor quantization + sglang inference faulure

**Link**: https://github.com/sgl-project/sglang/issues/5107
**State**: open
**Created**: 2025-04-07T02:20:30+00:00
**Comments**: 2
**Labels**: inactive

### Description

Hi，big guys！
Recently I followed the official documentation of sglang and llmcompressor and quantized the Qwen2.5-0.5B model into three different classes (W4A16, W8A8-Int8, and W8A8-FP8), but with the same quantization code, the same runtime environment, and only the model files are different （after quantizate）, only the model of W8A8-FP8 is able to be started through the Sglang service, the other However, with the same quantization code and the same runtime environment, only the model files are different, only the model of W8A8-FP8 can be started by Sglang service, while the others will report errors. Is there any solution?

Here is the code I quantified, the commands executed by the service startup, and the errors reported (W4A16 and W8A8-Int8)
![Image](https://github.com/user-attachments/assets/f1a2e10f-eb15-4759-a1f7-0f4dfd2720a3)

W8A8-Int8 Error:
![Image](https://github.com/user-attachments/assets/b445deb6-7bc2-4341-aebc-4414d8194724)

W4A16 Error:
![Image](https://github.com/use

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] adopt trt llm fp8_blockscale_gemm

**Link**: https://github.com/sgl-project/sglang/issues/4776
**State**: open
**Created**: 2025-03-26T00:26:10+00:00
**Comments**: 1
**Labels**: high priority, inactive

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled
for Blackwell
ref https://github.com/NVIDIA/TensorRT-LLM/pull/3071

### Related resources

_No response_

---

## Issue #N/A: [Bug] Extraneous/incorrect outputs when using response_format on DeepSeek models and MTP

**Link**: https://github.com/sgl-project/sglang/issues/4771
**State**: open
**Created**: 2025-03-25T22:27:26+00:00
**Comments**: 6
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Extraneous/incorrect outputs when using response_format (via pydantic models and model_dump_json) on DeepSeek-V3--0324 with SGLang 0.4.4.post1 and MTP.

### Reproduction

8x h200 nodes, cuda 12.3, python 3.12.9.

```
    engine_args=(
        "--trust-remote-code "
        "--revision f6be68c847f9ac8d52255b2c5b888cc6723fbcb2 "
        "--e

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Can you support the VLA series models? For example, openVLA.

**Link**: https://github.com/sgl-project/sglang/issues/4414
**State**: open
**Created**: 2025-03-14T07:00:28+00:00
**Comments**: 1
**Labels**: inactive, new-model

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

The documentation does not support the VLA series large models. Can you support the VLA series models?

### Related resources

_No response_

---

## Issue #N/A: [Bug] granite-vision-3.2-2b failing on sglang with "LlavaNextForConditionalGeneration not supported"

**Link**: https://github.com/sgl-project/sglang/issues/4062
**State**: open
**Created**: 2025-03-04T10:05:44+00:00
**Comments**: 3
**Labels**: inactive, MLLM, new-model

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hi,

I have successfully run the 3.1 versions of granite models on SGLang project (https://github.com/sgl-project/sglang)

I am now trying to run granite-vision-3.2-2b  

But it fails, with the messages below: in particular `Model architectures ['LlavaNextForConditionalGeneration'] are not supported for now? `

will IBM work with SGLang pr

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Frontend compatibility with Python 3.13

**Link**: https://github.com/sgl-project/sglang/issues/3876
**State**: open
**Created**: 2025-02-26T07:18:42+00:00
**Comments**: 2
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

`cafile` parameter in `urllib.request.urlopen` is [removed in Python 3.13](https://docs.python.org/3.13/library/urllib.request.html), causing SGLang frontend to fail.

```
File "lib/python3.13/site-packages/starlette/routing.py", line 693, in lifespan
  async with self.lifespan_context(app) as maybe_state:
File "lib/python3.13/contextlib.p

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support unified paging in multi-lora serving

**Link**: https://github.com/sgl-project/sglang/issues/3647
**State**: open
**Created**: 2025-02-17T19:14:47+00:00
**Comments**: 5
**Labels**: enhancement, inactive, feature, lora

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Currently, SGL doesn't support the unified paging feature proposed by S-LoRA. However, this feature is important for memory management in multi-LoRA serving.

### Related resources

_No response_

---

## Issue #N/A: [Kernel] cuDNN attention backend

**Link**: https://github.com/sgl-project/sglang/issues/2272
**State**: open
**Created**: 2024-11-30T06:36:16+00:00
**Comments**: 3
**Labels**: enhancement, good first issue, help wanted, high priority, inactive

### Description

cuDNN provides very fast attention implementation and it is well maintained by NVIDIA. We would like to add a new attention backend based on cudnn.  

## Steps
1. Learn this cudnn paged attention python api. https://github.com/NVIDIA/cudnn-frontend/blob/v1.8.0/samples/python/52_scaled_dot_product_attention_with_paged_caches.ipynb
2. Add a new attention backend "cudnn" here https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/attention
3. We should be able to use it with `python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --attention-backend cudnn`

---

## Issue #N/A: [Kernel] Optimize triton decoding kernels for long context

**Link**: https://github.com/sgl-project/sglang/issues/2271
**State**: open
**Created**: 2024-11-30T06:04:27+00:00
**Comments**: 5
**Labels**: good first issue, help wanted, high priority, inactive

### Description

We noticed the current triton decoding kernel is very slow on long context. This is due to a missing flash decoding like optimization.

## Reproduce
We test the decoding speed with a context length of 200 and 2,000.

triton backend: The decoding speed drops from 147.64 token/s to 126.41 token/s
```
$ python3 -m sglang.bench_offline_throughput --model meta-llama/Llama-3.1-8B-Instruct --dataset-name random --num-prompt 1 --random-input 128 --random-output 2048 --random-range 1 --attention-backend triton

[2024-11-30 05:10:04 TP0] Decode batch. #running-req: 1, #token: 234, token usage: 0.00, gen throughput (token/s): 147.64, #queue-req: 0
... 
[2024-11-30 05:10:18 TP0] Decode batch. #running-req: 1, #token: 2154, token usage: 0.00, gen throughput (token/s): 126.41, #queue-req: 0
```

flashinfer backend: The decoding speed only drops from 144.17 token/s to 143.35 token/s
```
$ python3 -m sglang.bench_offline_throughput --model meta-llama/Llama-3.1-8B-Instruct --dataset-nam

[... truncated for brevity ...]

---

