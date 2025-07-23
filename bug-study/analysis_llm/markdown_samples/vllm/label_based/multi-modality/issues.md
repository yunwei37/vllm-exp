# multi-modality - issues

**Total Issues**: 14
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 7
- Closed Issues: 7

### Label Distribution

- multi-modality: 14 issues
- feature request: 5 issues
- help wanted: 4 issues
- RFC: 4 issues
- new-model: 3 issues
- stale: 2 issues
- good first issue: 2 issues
- usage: 2 issues
- bug: 1 issues
- v1: 1 issues

---

## Issue #N/A: [Feature]: Benchmarks for audio models

**Link**: https://github.com/vllm-project/vllm/issues/16354
**State**: closed
**Created**: 2025-04-09T16:55:19+00:00
**Closed**: 2025-04-19T09:24:15+00:00
**Comments**: 2
**Labels**: help wanted, feature request, multi-modality

### Description

### 🚀 The feature, motivation and pitch

- Add audio datasets to `benchmarks/benchmark_dataset.py` to so we can run performance benchmarks on audio models as well.
- Add a benchmark similar to MMMU (#11196) but for audio models to evaluate their correctness.

### Alternatives

_No response_

### Additional context

cc @mgoin @ywang96 

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [New Model]: please surport  model   https://huggingface.co/Skywork/Skywork-R1V-38B

**Link**: https://github.com/vllm-project/vllm/issues/15186
**State**: closed
**Created**: 2025-03-20T04:43:12+00:00
**Closed**: 2025-03-29T03:39:22+00:00
**Comments**: 0
**Labels**: new-model, multi-modality

### Description

### The model to consider.

https://huggingface.co/Skywork/Skywork-R1V-38B

### The closest model vllm already supports.

https://huggingface.co/Skywork/Skywork-R1V-38B

### What's your difficulty of supporting the model you want?

https://huggingface.co/Skywork/Skywork-R1V-38B

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

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

## Issue #N/A: [Feature]: Support more video loader

**Link**: https://github.com/vllm-project/vllm/issues/15011
**State**: closed
**Created**: 2025-03-18T08:29:37+00:00
**Closed**: 2025-04-01T15:55:14+00:00
**Comments**: 5
**Labels**: feature request, multi-modality

### Description

### 🚀 The feature, motivation and pitch

vllm now using `decord` for video loader. There are some problem here:
1. `decord` is unmaintained from 3 years ago. and the newest release 0.6 is from 4 years ago. It may cause some unknown issue with this lib without fix.
2. `decord` only published x86 package to pypi, for some aarch64 machine, such as GH200, users need build it by hand.

So it's good to support more video loader for diversity usage.

Some investigation maybe useful:
1. huggingface transformers support `decord`, `pyav`, `torchvision` and `opencv`.  https://huggingface.co/docs/transformers/chat_template_multimodal#sampling-with-fixed-number-of-frames while only `decord` and `pyav` support `load_from_url` case. `pyav` is the default backend even the performance of `decord` is better.
2. The suggested loader from Qwen2.5 VL are `decord` and `torchvision`. https://github.com/QwenLM/Qwen2.5-VL/blob/f56c4d62f6ed38d725d9da2d1440d19b04c10c66/qwen-vl-utils/src/qwen_vl_utils/vision_proc

[... truncated for brevity ...]

---

## Issue #N/A: [Tracking Issue]: Multi-modal model requests

**Link**: https://github.com/vllm-project/vllm/issues/14876
**State**: closed
**Created**: 2025-03-16T02:14:05+00:00
**Closed**: 2025-03-16T13:01:04+00:00
**Comments**: 0
**Labels**: new-model, multi-modality

### Description

Moved to https://github.com/orgs/vllm-project/projects/10

---

## Issue #N/A: [RFC]: Configurable multi-modal data for profiling

**Link**: https://github.com/vllm-project/vllm/issues/14438
**State**: closed
**Created**: 2025-03-07T13:55:02+00:00
**Closed**: 2025-07-14T02:17:02+00:00
**Comments**: 3
**Labels**: RFC, stale, multi-modality

### Description

### Motivation.

We can control the data used in profiling multi-modal models using `limit_mm_per_prompt`. However, this is insufficient for the following use-cases:

- Restrict models that accept multiple modalities to only accept single modality inputs to avoid unnecessary memory allocation, e.g.:
  - Make Qwen2-VL only accept 10 images *or* 1 video, but not 10 images *and* 1 video per prompt
- Limit the duration of multi-modal data items with temporal components to save memory, e.g.:
  - Make Whisper accept only 20s of audio instead of 30s
  - Make Qwen2-VL accept only 10 frames of video instead of 16

To enable them, this RFC proposes a new engine argument: `mm_profiling_configs`, which lets users configure the multi-modal data used for profiling in more detail.

### Proposed Change.

This RFC proposes a new engine argument `mm_profiling_configs`, which accepts a list of config objects in JSON form. At a minimum, each config object specifies the maximum number of multi-modal items 

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Merge input processor and input mapper for multi-modal models

**Link**: https://github.com/vllm-project/vllm/issues/10114
**State**: closed
**Created**: 2024-11-07T09:57:55+00:00
**Closed**: 2025-04-28T07:38:50+00:00
**Comments**: 13
**Labels**: RFC, multi-modality

### Description

## Motivation

### Background

To provide more control over the model inputs, we currently define two methods for multi-modal models in vLLM:

- The **input processor** is called inside `LLMEngine` to extend the prompt with placeholder tokens which are reserved for vLLM features such as KV cache and chunked prefill.
- The **input mapper** is called inside `ModelRunner` to transform multi-modal inputs (e.g. `PIL` images) into tensor inputs, usually via the modality-specific processor (e.g. `AutoImageProcessor`) from HuggingFace.

### Issues with the current design

1. The input processor accepts the output of HF `AutoTokenizer`, a list of token IDs, instead of the text prompt. Since HF `AutoProcessor` doesn’t accept token IDs, we have to write custom code to edit the list of token IDs based on the multi-modal inputs. For some models (such as Phi-3-vision), this means re-implementing code from their HF `AutoProcessor`, complicating the process of porting the model to vLLM.
2. The input m

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

## Issue #N/A: [New Model]: Support ColQwen2VL

**Link**: https://github.com/vllm-project/vllm/issues/19381
**State**: open
**Created**: 2025-06-09T19:49:53+00:00
**Comments**: 0
**Labels**: new-model, multi-modality

### Description

### The model to consider.

ColQwen2VL is an efficient document retrieval vision language model based on Qwen2VL, as described in the paper "ColPali: Efficient Document Retrieval with Vision Language Models". The model is designed to generate embeddings rather than text outputs, making it suitable for document retrieval applications.

This was supported in HF Transformers as of https://github.com/huggingface/transformers/pull/35778

An initial attempt to support the model was posted in https://github.com/vllm-project/vllm/pull/14291 but it was made before the HF definition was finalized so it grew out-of-date.

### The closest model vllm already supports.

Qwen2VL is used as a base, so mostly it is wrapping that backbone

### What's your difficulty of supporting the model you want?

See previous attempt https://github.com/vllm-project/vllm/pull/14291

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bott

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Getting empty output for whsiperv3

**Link**: https://github.com/vllm-project/vllm/issues/19183
**State**: open
**Created**: 2025-06-05T05:33:00+00:00
**Comments**: 5
**Labels**: usage, multi-modality

### Description

### Your current environment

```text
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from librosa import load as load_audio

# Create a Whisper encoder/decoder model instance
llm = LLM(
    # model="openai/whisper-large-v3",
    model = "",
    trust_remote_code=True,
    max_model_len=448,
    max_num_seqs=400,
    limit_mm_per_prompt={"audio": 1},
    kv_cache_dtype="fp8",
    task='transcription',
    dtype="bfloat16",
    enforce_eager=False,
    max_logprobs=1
)


(waveform,sampling_rate)= load_audio('./sample.wav',sr=16000, mono=True)



prompts = [
    {
        "prompt": "<|startoftranscript|><|en|>",
        "multi_modal_data": {
            "audio": (waveform,sampling_rate),
        },
    }
]*1

#tried below also but same error
# prompts = [
#     {
#         "encoder_prompt":{
#             "prompt":"",
#             "multi_modal_data":{"audio":(waveform,sampling_rate)},
#         },
#         "decoder_prompt":{
#             "prompt_token_ids

[... truncated for brevity ...]

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

## Issue #N/A: [Usage]: How can I quickly obtain the number of prompt tokens containing multimodal data?

**Link**: https://github.com/vllm-project/vllm/issues/16191
**State**: open
**Created**: 2025-04-07T14:45:08+00:00
**Comments**: 7
**Labels**: help wanted, usage, multi-modality

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How would you like to use vllm

The /tokenize API can only return the number of prompt tokens that contain text and multimodal placeholders, but cannot return the actual number of prompt tokens. @DarkLight1337 


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

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

## Issue #N/A: [RFC]: Multi-modality Support on vLLM

**Link**: https://github.com/vllm-project/vllm/issues/4194
**State**: open
**Created**: 2024-04-19T07:51:48+00:00
**Comments**: 98
**Labels**: RFC, multi-modality

### Description

**Active Projects (help wanted!):**
- [Core tasks](https://github.com/orgs/vllm-project/projects/8)
- [Model requests](https://github.com/orgs/vllm-project/projects/10)

**Update [11/18] - In the upcoming months, we will focus on performance optimization for multimodal models as part of vLLM V1 engine re-arch effort**

**P0** (We will definitely work on them):
- [ ] V1 re-arch for multimodal models - See high-level design ([Slides](https://docs.google.com/presentation/d/1e3CxQBV3JsfGp30SwyvS3eM_tW-ghOhJ9PAJGK6KR54/edit#slide=id.g31455c8bc1e_2_122), [Doc](https://docs.google.com/document/d/11_DFQTku6C2aV6ghK21P76ST6uAUVjMlEjs54prtb_g/edit?usp=sharing))
  - [ ] Core 
    - [x] [1/N] #9871
    - [x] [2/N] #10374 
    - [x] [3/N] #10570 
    - [x] [4/N] #10699
    - [x] [5/N] #11210
    - [x] [6/N] #12128 
    - [x] [7/N] Enable rest of single-modality LMMs on V1
      - [x] #11632 (Aria, BLIP-2, Chameleon, Fuyu)
      - [x] #14275
      - [x] #11685
      - [x] #11733
      - [x] #12069
 

[... truncated for brevity ...]

---

