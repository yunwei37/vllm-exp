# MLLM - issues

**Total Issues**: 23
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 7
- Closed Issues: 16

### Label Distribution

- MLLM: 23 issues
- good first issue: 7 issues
- help wanted: 7 issues
- inactive: 6 issues
- high priority: 5 issues
- feature: 2 issues
- sgl-kernel: 1 issues
- collaboration: 1 issues
- flashinfer: 1 issues
- performance: 1 issues

---

## Issue #N/A: [Feature] Benchmark with audio input

**Link**: https://github.com/sgl-project/sglang/issues/8072
**State**: open
**Created**: 2025-07-15T22:28:51+00:00
**Comments**: 1
**Labels**: good first issue, help wanted, MLLM

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

We need scripts to bench audio input for supported MLLM like minicpmo and gemma3n.

### Related resources

https://github.com/vllm-project/vllm/issues/16354

---

## Issue #N/A: [Perf] improve the hash kernel for mm

**Link**: https://github.com/sgl-project/sglang/issues/8054
**State**: open
**Created**: 2025-07-15T09:08:36+00:00
**Comments**: 3
**Labels**: MLLM, sgl-kernel

### Description

The current `gpu_tensor_hash` implementated in #5974  has following drawbacks:
1. `add` itself is not a very decent reduction method
2. will perform a torch tensor reduction, which is not very performant for large tensors

## TODO

1. Rewrite a performant and robust tensor hash function
2. Test the performance, consistency and correctness of the hash function against real data


## Reference

You can reference [here](https://github.com/sgl-project/sglang/pull/5974#issuecomment-3017284280) for inspirations


---

## Issue #N/A: [Bug] Kimi VL GPU memory usage too high

**Link**: https://github.com/sgl-project/sglang/issues/7433
**State**: open
**Created**: 2025-06-22T07:37:40+00:00
**Comments**: 5
**Labels**: good first issue, MLLM

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

The GPU memory usage when serving Kimi VL is too high.

### Reproduction

1. First monitor GPU memory usage using `nvitop`
2. Launch server with `python -m sglang.launch_server --model-path moonshotai/Kimi-VL-A3B-Thinking-2506 --trust-remote-code --reasoning-parser kimi --mem-fraction-static 0.5`
The usage should be around 50%.
3. Run `pyt

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support more multi-modal input for VLM

**Link**: https://github.com/sgl-project/sglang/issues/5964
**State**: open
**Created**: 2025-05-02T02:28:40+00:00
**Comments**: 4
**Labels**: good first issue, help wanted, feature, MLLM

### Description

### Motivation

The current endpoint only supports image data input, limiting its flexibility for diverse VLM use cases. We need additional input formats, particularly for RL applications:
(Could be split into multiple PRs)

- [x] Pre-computed Image Embeddings
- [ ] Pixel Values
- [ ] Pixel Value Range Parameters (min_pixel/max_pixel) for qwen-vl

Welcome to propose more.

#### Benefits

1. Enhanced flexibility for RL workflows
2. Reduced preprocessing overhead
3. Better integration with existing pipelines

---

## Issue #N/A: [Feature] support chunked prefill for VLM

**Link**: https://github.com/sgl-project/sglang/issues/5312
**State**: closed
**Created**: 2025-04-12T06:01:06+00:00
**Closed**: 2025-05-26T16:56:11+00:00
**Comments**: 0
**Labels**: high priority, MLLM

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

### Related resources

_No response_

---

## Issue #N/A: [Feature] support and turn on chunked prefill by default for VLM

**Link**: https://github.com/sgl-project/sglang/issues/5250
**State**: closed
**Created**: 2025-04-10T18:45:57+00:00
**Closed**: 2025-05-26T16:56:01+00:00
**Comments**: 2
**Labels**: good first issue, help wanted, high priority, MLLM

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct

### Related resources

_No response_

---

## Issue #N/A: [Feature] add more CIs for VLM

**Link**: https://github.com/sgl-project/sglang/issues/5249
**State**: open
**Created**: 2025-04-10T18:44:02+00:00
**Comments**: 7
**Labels**: high priority, MLLM

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
https://huggingface.co/google/gemma-3-27b-it
https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct

### Related resources

_No response_

---

## Issue #N/A: [Feature] attention backend default choice

**Link**: https://github.com/sgl-project/sglang/issues/5064
**State**: closed
**Created**: 2025-04-04T08:13:51+00:00
**Closed**: 2025-05-21T09:29:52+00:00
**Comments**: 2
**Labels**: high priority, collaboration, flashinfer, performance, MLLM, deepseek

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

The standards we choose prioritize **performance first**, ease of use second (such as interface and installation), while also considering compatibility (such as older arch). Therefore, if in the future, the performance of different backends changes, we will still choose **the best performing one**.

1. NVIDIA

```
sm75 -> Triton
sm80, sm86, sm89 -> FlashInfer
sm90 -> FA3 (Llama, Qwen, Gemma), FlashInfer (Others)
sm100 -> FlashInfer

MLA
sm90 -> FA3 (DeepSeek)
sm100 -> FlashInfer (DeepSeek)

Other options
FlashMLA, cuDNN etc
```

SGLang will install the JIT version of FlashInfer on PyPI for a better user installation experience. Alternatively, the whl size limit of FlashInfer can be increased on PyPI. cc @yzh119 

F

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Gemma-3-27b-instruct use CPU and MEM but not GPU while inference images

**Link**: https://github.com/sgl-project/sglang/issues/4627
**State**: closed
**Created**: 2025-03-20T13:03:12+00:00
**Closed**: 2025-05-20T00:19:54+00:00
**Comments**: 2
**Labels**: inactive, MLLM

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Gemma-3-27b-instruct use CPU and MEM but not GPU while inference images.
The GPU usages is 0%, but CPU and MEM usage are very high.
Shown as below.

![Image](https://github.com/user-attachments/assets/18cb5a82-5442-42cb-bbf5-fbbe684cc006)

Without images, The GPU will be used.

### Reproduction

use github main branch to install sglang.
st

[... truncated for brevity ...]

---

## Issue #N/A: [Track] VLM accuracy in MMMU benchmark

**Link**: https://github.com/sgl-project/sglang/issues/4456
**State**: closed
**Created**: 2025-03-15T17:09:50+00:00
**Closed**: 2025-04-25T07:23:54+00:00
**Comments**: 5
**Labels**: good first issue, MLLM

### Description

This issue keeps track of all vlm models accuracy in MMMU benchmark. Keep updating

``` python
python benchmark/mmmu/bench_sglang.py
python benchmark/mmmu/bench_hf.py --model-path model

```

| | sglang | hf |
|--|--|--|
| Qwen2-VL-7B-Instruct |  0.485 | 0.255 |
| Qwen2.5-VL-7B-Instruct | 0.477 | 0.242 |
| MiniCPM-V-2_6 |  0.426 |  |
| MiniCPM-O-2_6 | 0.481| 0.49 |
| Deepseek-vl2 | 0.496 | 0.499|
|Deepseek-vl2-small | 0.464 | 0.453|
|Deepseek-vl2-tiny | 0.382 | 0.369|
| Deepseek-Janus-Pro-7B| | |
| Llava + Llama| | |
| Llava + qwen| | |
| Llava + Mistral| | |
| Mlama | | |
| Gemma-3-it-4B| 0.409 | 0.403 |
| InternVL2.5-38B | 0.61 | |



---

## Issue #N/A: [Bug] Qwen2.5-VL-7B-Instruct Inference Server crashes

**Link**: https://github.com/sgl-project/sglang/issues/4171
**State**: closed
**Created**: 2025-03-07T08:17:40+00:00
**Closed**: 2025-05-30T23:40:54+00:00
**Comments**: 15
**Labels**: MLLM

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

This error does not happen every time, but it keeps crashing the Inference Server.


[2025-03-07 02:01:18 TP0] Scheduler hit an exception: Traceback (most recent call last):
  File "/home/username/learning/sglang/python/sglang/srt/managers/scheduler.py", line 2290, in run_scheduler_process
    scheduler.event_loop_normal()                 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Key conflict of `AutoImageProcessor.register`

**Link**: https://github.com/sgl-project/sglang/issues/4159
**State**: closed
**Created**: 2025-03-07T04:06:18+00:00
**Closed**: 2025-03-25T12:17:44+00:00
**Comments**: 22
**Labels**: MLLM

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

The following ValueError was raised when attempting to serve any model within a recent Docker container:

`Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_

[... truncated for brevity ...]

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

## Issue #N/A: [Feature] Support token-in-token-out for Vision LM

**Link**: https://github.com/sgl-project/sglang/issues/3871
**State**: closed
**Created**: 2025-02-26T04:35:56+00:00
**Closed**: 2025-04-29T00:18:49+00:00
**Comments**: 10
**Labels**: inactive, RLHF, MLLM

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Considering what we need in LLM RLHF, rollout engine just needs token in, and give token out.

We are working on VLM RLHF with veRL, could we support VLM token-in-token-out. Here is something maybe useful:

`test/srt/test_skip_tokenizer_init.py`: this is for LLM.

I actually do not know how to get token of VLM 😂

Hope to get the answer.

### Related resources

_No response_

---

## Issue #N/A: The number of image token (3) should be the same as in the number of provided images (1)

**Link**: https://github.com/sgl-project/sglang/issues/3819
**State**: closed
**Created**: 2025-02-24T13:48:19+00:00
**Closed**: 2025-05-04T00:21:05+00:00
**Comments**: 7
**Labels**: inactive, MLLM

### Description

When I was using Llama-3.2-11B-Vision-Instruct, I encountered an error on a small amount of data:
`{'object': 'error', 'message': 'The number of image token (3) should be the same as in the number of provided images (1)', 'type': 'BadRequestError', 'param': None, 'code': 400}`

---

## Issue #N/A: Qwen2.5 VL sglang's output much worse than transformers

**Link**: https://github.com/sgl-project/sglang/issues/3746
**State**: closed
**Created**: 2025-02-21T06:38:34+00:00
**Closed**: 2025-05-16T06:24:46+00:00
**Comments**: 17
**Labels**: MLLM

### Description

I tried serving qwen2.5 vl 72B using sglang on a node with 4*A40 GPUs.
The image I used is the official sglang:v0.4.3.post2-cu125
The command:
```bash
python3 -m sglang.launch_server \
  --tp $NUM_SHARD \
  --mem-fraction-static 0.99 \
  --disable-cuda-graph \
  --model-path /model/Qwen2.5-VL-72B-Instruct \
  --host 0.0.0.0 \
  --port 23333
```

I tested  using an internal image classification dataset, the results were much worse than when using transformers, acc droped from 87% to 80%.
And I tried another image2code task, the rendered images were much worse, too.

---

## Issue #N/A: [Bug] Qwen 2.5 VL new version 0.4.3.post2  --disable-radix-cache   doesn't work

**Link**: https://github.com/sgl-project/sglang/issues/3681
**State**: closed
**Created**: 2025-02-19T01:36:24+00:00
**Closed**: 2025-02-22T14:11:17+00:00
**Comments**: 6
**Labels**: help wanted, MLLM

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

Qwen 2.5 VL new version 0.4.3.post2  --disable-radix-cache   doesn't work
1）request occasionally not deterministic mode
{
    "model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "messages": [
    	
      {
        "role": "user",
        "content": [
          
          {
            "type": "image_url",
            "image_url": {
              "

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] second_per_grid_ts should be used to get mrop position in qwen2.5-vl

**Link**: https://github.com/sgl-project/sglang/issues/3674
**State**: closed
**Created**: 2025-02-18T12:32:39+00:00
**Closed**: 2025-04-21T00:19:32+00:00
**Comments**: 3
**Labels**: inactive, MLLM

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

second_per_grid_ts not used in get_input_positions in qwen2.5-vl

### Reproduction

run qwen2.5-vl test

### Environment

cuda/amd

---

## Issue #N/A: [Bug] Vision attention mask cache is never released and cause OOM

**Link**: https://github.com/sgl-project/sglang/issues/3651
**State**: closed
**Created**: 2025-02-18T02:54:23+00:00
**Closed**: 2025-02-19T15:19:27+00:00
**Comments**: 8
**Labels**: MLLM

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

The vision attention module implements an attention mask cache mechanism, however the cache is never released. In recent VLMs like `Qwen2-VL`, the image inputs are kept with their original resolution without resizing, so if we request with various sized images, the cache will keep increasing and finally cause OOM.

https://github.com/sgl-p

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support multimodal models for Native API/ Engine

**Link**: https://github.com/sgl-project/sglang/issues/3545
**State**: closed
**Created**: 2025-02-13T10:39:45+00:00
**Closed**: 2025-02-25T17:52:53+00:00
**Comments**: 0
**Labels**: feature, MLLM

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

add the multimodal support to Native API/ Engine

### Related resources

_No response_

---

## Issue #N/A: [Feature] Support for Qwen2.5-VL

**Link**: https://github.com/sgl-project/sglang/issues/3247
**State**: closed
**Created**: 2025-02-01T07:48:46+00:00
**Closed**: 2025-02-16T09:10:11+00:00
**Comments**: 3
**Labels**: good first issue, help wanted, MLLM

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

[Qwen2.5-VL](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5) came out a few days ago. There are some small changes in its architecture compared to Qwen2-VL.

### Related resources

_No response_

---

## Issue #N/A: [Feature] Support DeepSeek Janus Models

**Link**: https://github.com/sgl-project/sglang/issues/3195
**State**: closed
**Created**: 2025-01-28T18:37:47+00:00
**Closed**: 2025-04-30T00:18:51+00:00
**Comments**: 4
**Labels**: help wanted, inactive, MLLM

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Docker is a valuable tool for the management of dependencies. Indeed, it can simplify the running of Janus Models to a single command:  
```bash
docker run -it --rm \
  -p 8000:8000 \
  -d \
  -v huggingface:/root/.cache/huggingface \
  -w /app \
  --gpus all \
  --name janus \
  -e MODEL_NAME=deepseek-ai/Janus-Pro-7B \
  julianfl0w/janus:latest
```

Make sure it's working by navigating in your browser to  
[http://localhost:8000/webui](http://localhost:8000/webui)

and by running
```bash
docker logs janus
```

This keeps all the Torch dependencies contained within the image, meaning the user doesn't have to adjust their base installations to run models like these. 

Note: You will have to install NVIDIA Container 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Accuracy test of VLM

**Link**: https://github.com/sgl-project/sglang/issues/3142
**State**: open
**Created**: 2025-01-26T06:25:40+00:00
**Comments**: 7
**Labels**: good first issue, help wanted, high priority, MLLM

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

In sglang, LLMs have accuracy tests with Hugging Face models:

https://github.com/sgl-project/sglang/blob/main/test/srt/models/test_generation_models.py

https://github.com/sgl-project/sglang/blob/main/test/srt/test_nightly_math_eval.py

We need similar one for VLM also.

### Related resources

_No response_

---

