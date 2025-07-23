# help_wanted - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 5
- Closed Issues: 25

### Label Distribution

- help wanted: 30 issues
- feature request: 13 issues
- good first issue: 12 issues
- stale: 7 issues
- bug: 6 issues
- multi-modality: 4 issues
- new-model: 2 issues
- performance: 2 issues
- documentation: 1 issues
- speculative-decoding: 1 issues

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

## Issue #N/A: [Feature]: Implement Priority Scheduling In V1 Engine

**Link**: https://github.com/vllm-project/vllm/issues/14002
**State**: closed
**Created**: 2025-02-28T01:33:35+00:00
**Closed**: 2025-06-23T03:18:09+00:00
**Comments**: 10
**Labels**: help wanted, feature request

### Description

### 🚀 The feature, motivation and pitch

In V0, we support request priority. I would like to see this in V1

cc @WoosukKwon 

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

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

## Issue #N/A: [New Model]: WisdomShell

**Link**: https://github.com/vllm-project/vllm/issues/11155
**State**: closed
**Created**: 2024-12-13T03:30:27+00:00
**Closed**: 2025-04-13T02:16:48+00:00
**Comments**: 8
**Labels**: help wanted, new-model, stale

### Description

### 🚀 The feature, motivation and pitch

Traceback (most recent call last):
  File "/usr/local/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/local/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/usr/local/lib/python3.10/site-packages/vllm/entrypoints/api_server.py", line 158, in <module>
    asyncio.run(run_server(args))
  File "/usr/local/lib/python3.10/asyncio/runners.py", line 44, in run
    return loop.run_until_complete(main)
  File "/usr/local/lib/python3.10/asyncio/base_events.py", line 649, in run_until_complete
    return future.result()
  File "/usr/local/lib/python3.10/site-packages/vllm/entrypoints/api_server.py", line 115, in run_server
    app = await init_app(args, llm_engine)
  File "/usr/local/lib/python3.10/site-packages/vllm/entrypoints/api_server.py", line 103, in init_app
    if llm_engine is not None else AsyncLLMEngine.from_engine_args(
  F

[... truncated for brevity ...]

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

## Issue #N/A: [New Model]: Google SigLip 2

**Link**: https://github.com/vllm-project/vllm/issues/13663
**State**: open
**Created**: 2025-02-21T09:40:40+00:00
**Comments**: 12
**Labels**: help wanted, good first issue, new-model

### Description

### The model to consider.

[Google SigLip 2](https://huggingface.co/collections/google/siglip2-67b5dcef38c175486e240107)

### The closest model vllm already supports.

https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/siglip.py
> """Implementation of SiglipVisionModel intended to be only used
within a vision language model."""

However, SigLip 2 can be very useful for zero-shot image classification and image-text retrieval

### What's your difficulty of supporting the model you want?

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: FlashAttention 3 support

**Link**: https://github.com/vllm-project/vllm/issues/6348
**State**: closed
**Created**: 2024-07-11T19:11:40+00:00
**Closed**: 2025-02-21T16:42:32+00:00
**Comments**: 17
**Labels**: help wanted, feature request

### Description

### 🚀 The feature, motivation and pitch

As you know, FA3 promises 1.5x~ improvements
https://github.com/Dao-AILab/flash-attention/commit/7ef24848cf2f855077cef88fe122775b727dcd74

### Alternatives

_No response_

### Additional context

_No response_

---

## Issue #N/A: [Performance] [Speculative decoding]: Support draft model on different tensor-parallel size than target model

**Link**: https://github.com/vllm-project/vllm/issues/4632
**State**: closed
**Created**: 2024-05-06T18:14:40+00:00
**Closed**: 2024-06-25T09:56:08+00:00
**Comments**: 5
**Labels**: help wanted, performance, speculative-decoding

### Description

## Overview
Speculative decoding allows a speedup for memory-bound LLMs by using a fast proposal method to propose tokens that are verified in a single forward pass by the larger LLM. Papers report 2-3x speedup for bs=1, in Anyscale's fork we see up to 2x speedup with a small draft model for bs=8 (30% for bs=16) (we can improve this! see https://github.com/vllm-project/vllm/issues/4630 if you want to help).

A key optimization for small models (68m/160m domain) is to use tensor-parallel degree 1, even if the target model is using tensor-parallel degree 4 or 8. In our fork, this reduces proposal time from 5ms/tok to 1.5ms/tok. This will allow a well-aligned 68m draft model to get 2x per-user throughput improvement on 70B target model.

Furthermore, a 1B/7B proposer model may ideally be placed on TP=2 or TP=4, while the larger model is placed on TP=8. vLLM should support these configuration so the community can use the configuration best for their draft model.

## Design suggestio

[... truncated for brevity ...]

---

## Issue #N/A: Publish wheels with pre-built CUDA binaries

**Link**: https://github.com/vllm-project/vllm/issues/139
**State**: closed
**Created**: 2023-06-05T00:07:35+00:00
**Closed**: 2023-08-25T04:38:06+00:00
**Comments**: 3
**Labels**: help wanted, installation

### Description

Currently, pip installing our package takes 5-10 minutes because our CUDA kernels are compiled on the user machine. For better UX, we should include pre-built CUDA binaries in our PyPI distribution, just like PyTorch and xformers.

---

## Issue #N/A: [Bug]: Missing Content Type returns 500 Internal Server Error instead of 415 Unsupported Media Type

**Link**: https://github.com/vllm-project/vllm/issues/11171
**State**: closed
**Created**: 2024-12-13T11:31:21+00:00
**Closed**: 2025-02-13T14:52:23+00:00
**Comments**: 1
**Labels**: bug, help wanted, good first issue

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Versions of relevant libraries:
[pip3] flashinfer==0.1.6+cu121torch2.4
[pip3] numpy==1.26.4
[pip3] nvidia-cublas-cu12==12.4.5.8
[pip3] nvidia-cuda-cupti-cu12==12.4.127
[pip3] nvidia-cuda-nvrtc-cu12==12.4.127
[pip3] nvidia-cuda-runtime-cu12==12.4.127
[pip3] nvidia-cudnn-cu12==9.1.0.70
[pip3] nvidia-cufft-cu12==11.2.1.3
[pip3] nvidia-curand-cu12==10.3.5.147
[pip3] nvidia-cusolver-cu12==11.6.1.9
[pip3] nvidia-cusparse-cu12==12.3.1.170
[pip3] nvidia-ml-py==12.560.30
[pip3] nvidia-nccl-cu12==2.21.5
[pip3] nvidia-nvjitlink-cu12==12.4.127
[pip3] nvidia-nvtx-cu12==12.4.127
[pip3] pyzmq==26.2.0
[pip3] torch==2.5.1
[pip3] torchvision==0.20.1
[pip3] transformers==4.46.2
[pip3] triton==3.1.0
[conda] Could not collect
ROCM Version: Could not collect
Neuron SDK Version: N/A
vLLM Version: 0.6.4.post1

I don't see the point in sharing hardware specifications for a sim

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

## Issue #N/A: Support `tools` and `tool_choice` parameter in OpenAI compatible service 

**Link**: https://github.com/vllm-project/vllm/issues/1869
**State**: closed
**Created**: 2023-11-30T19:57:08+00:00
**Closed**: 2024-06-03T23:25:30+00:00
**Comments**: 7
**Labels**: help wanted, good first issue, feature request

### Description

Also aliased as `functions` and `function_call` in deprecated parameters.

https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools

After #1756 is merged (thanks @Tostino!), it should be straightforward to add this as a core parameter to OpenAI compatible service. This will help unlock client libraries using similar interface. Do note that the underlying model need to support function calling (e.g. OpenHermes) and prompt engineering might be needed. 

Also see @dongxiaolong's example here: https://github.com/vllm-project/vllm/pull/1756#issuecomment-1827064922


---

## Issue #N/A: [Feature]: Support block manager v2 for chunked prefill

**Link**: https://github.com/vllm-project/vllm/issues/7371
**State**: closed
**Created**: 2024-08-09T16:51:57+00:00
**Closed**: 2024-08-09T23:48:50+00:00
**Comments**: 3
**Labels**: help wanted, feature request

### Description

### 🚀 The feature, motivation and pitch

Chunked prefill currently doesn't work with block manager v2. Add `use_v2_block_manager=True` to https://github.com/vllm-project/vllm/blob/main/tests/basic_correctness/test_chunked_prefill.py#L48 to reproduce the error:

```
    def append_token_ids(self, block_index: int, token_ids: List[int]) -> None:
>       block = self._blocks[block_index]
E       IndexError: list index out of range
```

### Alternatives

Use block manager v1

### Additional context

cc @rkooo567 @cadedaniel 

---

## Issue #N/A: [V1][Bug]: Consider sampler in memory profiling

**Link**: https://github.com/vllm-project/vllm/issues/13507
**State**: closed
**Created**: 2025-02-19T02:36:15+00:00
**Closed**: 2025-02-22T08:08:30+00:00
**Comments**: 0
**Labels**: bug, help wanted

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```

</details>


### 🐛 Describe the bug

Currently, V1 model runner does not consider (or run) sampler during the initial memory profiling. This should be fixed 1) for more accurate memory profiling and 2) to warm up the sampler before processing any real requests.

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: When apply continue_final_message for OpenAI server, the `"echo":false` is ignored.

**Link**: https://github.com/vllm-project/vllm/issues/10111
**State**: closed
**Created**: 2024-11-07T07:30:08+00:00
**Closed**: 2024-11-21T16:24:33+00:00
**Comments**: 1
**Labels**: bug, help wanted, good first issue

### Description

### Your current environment


vLLM Version: 0.6.3.post2.dev256+g4be3a451
<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...                                                                                                                                                                       INFO 11-06 09:39:21 importing.py:15] Triton not installed or not compatible; certain GPU-related functions will not be available.                                                                           PyTorch version: 2.4.0+cpu                                                                                                                                                                                  
Is debug build: False                              
CUDA used to build PyTorch: None                   
ROCM used to build PyTorch: N/A                                                                       
                    

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Support for Running Classification Task in Online Server

**Link**: https://github.com/vllm-project/vllm/issues/13567
**State**: closed
**Created**: 2025-02-19T20:57:19+00:00
**Closed**: 2025-05-11T07:57:08+00:00
**Comments**: 7
**Labels**: help wanted, good first issue, feature request

### Description

### 🚀 The feature, motivation and pitch

I would like it to be easy to stand up models for sequence classification using the vllm online inference pattern. Currently this is available for offline inference but it would be nice to expose this server in kubernetes similar to how we host OpenAI compatible servers.

### Alternatives

We could train a causal lm where we treat special tokens as the classification labels. We could then take the softmaxed logprobs for those 2 tokens to threshold. However this is going to require slightly more code on the client side.

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

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

## Issue #N/A: [Bug]: When reading the content from the configuration file specified by the --config parameter, the parameter type was not considered.

**Link**: https://github.com/vllm-project/vllm/issues/9499
**State**: closed
**Created**: 2024-10-18T10:22:37+00:00
**Closed**: 2024-10-27T17:46:42+00:00
**Comments**: 3
**Labels**: bug, help wanted, good first issue

### Description

### Your current environment

<details>
This bug is environment-independent.
</details>


### Model Input Dumps

_No response_

### 🐛 Describe the bug

The code in the method '_load_config_file' of vllm/vllm/utils.py which reads the config file from the parameter --config has a bug.

```python
# only expecting a flat dictionary of atomic types
processed_args: List[str] = []

config: Dict[str, Union[int, str]] = {}
try:
    with open(file_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
except Exception as ex:
    logger.error(
        "Unable to read the config file at %s. \
        Make sure path is correct", file_path)
    raise ex

for key, value in config.items():
    processed_args.append('--' + key)
    processed_args.append(str(value))

return processed_args
```
The code here simply spans the key-value pairs in the config file. So, if I want to store a 'store_true' parameter like '--trust-remote-code', I cannot put it in the config

[... truncated for brevity ...]

---

## Issue #N/A: Performance issue when loading lora modules

**Link**: https://github.com/vllm-project/vllm/issues/3219
**State**: closed
**Created**: 2024-03-06T03:55:08+00:00
**Closed**: 2024-11-30T02:02:10+00:00
**Comments**: 5
**Labels**: help wanted, stale

### Description

I compared two ways to launch the server.

The model is vicuna-7b, and GPU is 2 \* A30.

and the 1st way is

```
python -m vllm.entrypoints.openai.api_server \
            --model /data/models/vicuna-7b-v1.5/ \
            --tensor-parallel-size 2  --gpu-memory-utilization 0.9 --enforce-eager --disable-log-requests
```

The 2nd way is:

```
python -m vllm.entrypoints.openai.api_server \
            --model /data/models/vicuna-7b-v1.5/ \
            --max-loras 16 --tensor-parallel-size 2  --max-lora-rank 64 --gpu-memory-utilization 0.9 \
            --enable-lora --enforce-eager --disable-log-requests --lora-modules lora1=/root/path1/  lora2=/root/path2/ ...
```

In both tests, I send the same request, which sets the model as `/data/models/vicuna-7b-v1.5/`.

But the performance differs a lot.

![image](https://uploads.linear.app/342cff15-f40f-4cf7-8bee-343d25adb534/0421c1bd-2196-4601-80e3-62d2f9769277/83c5b379-57ba-4acd-a1ce-a9004ddf41bf?signature=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e

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

## Issue #N/A: [Feature]: Improve Logging for Error Messages

**Link**: https://github.com/vllm-project/vllm/issues/14083
**State**: open
**Created**: 2025-03-01T17:14:04+00:00
**Comments**: 4
**Labels**: help wanted, good first issue, feature request

### Description

### 🚀 The feature, motivation and pitch

Improve logging on VLLM V1 for common errors in initialization.

For example:
- not enough memory to fit the model
- not enough kv cache space to fit the model
- ...

Currently we have decently logging of the exceptions that arise. It would be better however if we could explicitly catch these issues and return clearer error messages

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [help wanted]: why cmake 3.31 breaks vllm and how to fix it 

**Link**: https://github.com/vllm-project/vllm/issues/10189
**State**: closed
**Created**: 2024-11-10T01:15:52+00:00
**Closed**: 2024-11-12T23:06:49+00:00
**Comments**: 2
**Labels**: help wanted, misc

### Description

### Anything you want to discuss about vllm.

we need to figure it out and revert https://github.com/vllm-project/vllm/pull/10188 in the end.

the recent cmake 3.31 release breaks the release pipeline.
this successful release https://buildkite.com/vllm/release/builds/1745#019311b9-49d5-4f13-8064-df14308bd9ae uses cmake 3.30
and this failed release https://buildkite.com/vllm/release/builds/1746#01931327-c49b-4e52-8c0b-95fb95140ea4 uses cmake 3.31, and we get `CMake Error: Could not find CMAKE_ROOT !!! .`

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

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

## Issue #N/A: [Feature]: Support for predicted outputs

**Link**: https://github.com/vllm-project/vllm/issues/10137
**State**: closed
**Created**: 2024-11-08T01:01:42+00:00
**Closed**: 2025-05-09T02:10:30+00:00
**Comments**: 4
**Labels**: help wanted, feature request, stale

### Description

### 🚀 The feature, motivation and pitch

https://platform.openai.com/docs/guides/latency-optimization#use-predicted-outputs

Reminds me on:
https://github.com/FasterDecoding/REST
https://arxiv.org/html/2311.08252v2

### Alternatives

_No response_

### Additional context

I could give it a try to implement it based on ngram speculation 

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Performance]: Prefix-caching aware scheduling

**Link**: https://github.com/vllm-project/vllm/issues/7883
**State**: closed
**Created**: 2024-08-26T21:30:38+00:00
**Closed**: 2024-12-20T02:18:05+00:00
**Comments**: 8
**Labels**: help wanted, performance

### Description

### Proposal to improve performance

The current execution flow with prefix caching is as follows:
1. Scheduler takes the next prefill sequence:
    a. Calculate how many blocks it needs.
    b. Check whether we have sufficient number of blocks in the block manager.
    c. If so, determine the number of tokens to be prefilled in this batch (it is equal to the prompt length without chunked prefill, or at maximum the chunked size otherwise).
    d. Update the batch token budget by subtracting the tokens to be prefilled.
    e. Allocate all (regardless how many tokens to prefill in this batch) blocks.
    f. Match allocated block IDs with prefix cache, and list them in `computed_block_nums`.
2. Prepare input:
    a. Get the number of tokens to prefill for this sequence in this batch.
    b. Setup input token IDs and positions.
    c. If `computed_block_nums` is not none, then remove the cached tokens from input tokens, and adjust input positions, query length and context leng

[... truncated for brevity ...]

---

## Issue #N/A: [CI][Contribution Welcomed] Conditional Testing

**Link**: https://github.com/vllm-project/vllm/issues/4569
**State**: closed
**Created**: 2024-05-02T23:15:11+00:00
**Closed**: 2024-10-28T03:06:34+00:00
**Comments**: 12
**Labels**: help wanted, good first issue, stale

### Description

### Anything you want to discuss about vllm.

Currently we run all CI tests matrix on every single commit in pull requests. The CI cost of the vLLM has been doubling each week as we add more tests and receiving many PRs from the community. 

A good first step would be to only run some tests when relevant code is changed. For example, do not run unit/integration tests when docs or examples are changed. 

---

## Issue #N/A: [Feature]: Implement Concurrent Partial Prefills In V1 Engine

**Link**: https://github.com/vllm-project/vllm/issues/14003
**State**: open
**Created**: 2025-02-28T01:34:31+00:00
**Comments**: 7
**Labels**: help wanted, feature request

### Description

### 🚀 The feature, motivation and pitch

In V0, we support concurrent partial prefills to avoid TTFT latency with long requests. Implement it in V1

cc @WoosukKwon 

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

