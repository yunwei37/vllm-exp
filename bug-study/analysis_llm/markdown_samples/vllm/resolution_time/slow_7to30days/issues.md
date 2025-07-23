# slow_7to30days - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- bug: 11 issues
- usage: 5 issues
- feature request: 4 issues
- RFC: 3 issues
- new-model: 3 issues
- good first issue: 2 issues
- quantization: 1 issues
- misc: 1 issues
- v1: 1 issues
- multi-modality: 1 issues

---

## Issue #N/A: [Bug]: failed to test tests/lora/test_layers.py::test_embeddings[True-512-cuda:1-1]

**Link**: https://github.com/vllm-project/vllm/issues/9794
**State**: closed
**Created**: 2024-10-29T11:19:41+00:00
**Closed**: 2024-11-12T03:10:16+00:00
**Comments**: 4
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
WARNING 10-29 04:15:30 _custom_ops.py:19] Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.12.7 (main, Oct  1 2024, 08:52:12) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-105-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A100-SXM4-40GB
GPU 1: NVIDIA A100-SXM4-40GB

Nvidia driver version: 535.183.06
cuDNN version: Could not collect
HIP run

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

## Issue #N/A: [RFC]: Build `vllm-flash-attn` from source

**Link**: https://github.com/vllm-project/vllm/issues/8002
**State**: closed
**Created**: 2024-08-29T16:09:02+00:00
**Closed**: 2024-09-21T06:27:12+00:00
**Comments**: 3
**Labels**: RFC

### Description

### Motivation.

To use a custom version of PyTorch in vLLM, `vllm-flash-attn` needs to be built with the same version. The easiest way to achieve that is by building it from source during the vLLM build.

### Proposed Change.

We propose 3 different ways of building `vllm-flash-attn` from source: absorbing the package completely, building it as a CMake dependency, or running a nested `pip install`. Currently, alternative 2 is preferred, but we'd like to get feedback on that. I will update this RFC once we decide on an approach.

More details here: https://docs.google.com/document/d/1njmz8NPT3am5gNcjbjzZG1BN-v8wIxWq6vb5QoctuZ0/edit?usp=sharing

### Feedback Period.

_No response_

### CC List.

@WoosukKwon @youkaichao @tlrmchlsmth @bnellnm 

### Any Other Things.

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/late

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: TimeoutError and EngineDeadError in vLLM: RPC Call to execute_model Timed Out and EngineCore Failure

**Link**: https://github.com/vllm-project/vllm/issues/17965
**State**: closed
**Created**: 2025-05-11T16:29:17+00:00
**Closed**: 2025-05-23T02:22:12+00:00
**Comments**: 36
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
INFO 05-12 00:28:31 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: TencentOS Server 3.2 (Final) (x86_64)
GCC version: (GCC) 11.2.1 20210728 (Red Hat 11.2.1-1)
Clang version: 16.0.6 (Red Hat 16.0.6-2.module+el8.8.0+557+454507bd)
CMake version: version 3.28.0
Libc version: glibc-2.28

Python version: 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.4.241-1-tlinux4-0017.7-x86_64-with-glibc2.28
Is CUDA available: True
CUDA runtime version: 12.8.61
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H20
GPU 1: NVIDIA H20
GPU 2: NVIDIA H20
GPU 3: NVIDIA H20
GPU 4: NVIDIA H20
GPU 5: NVIDIA H20
GPU 6: NVIDIA H20
GPU

[... truncated for brevity ...]

---

## Issue #N/A: How does this compare to MQA (multi-query attention)?

**Link**: https://github.com/vllm-project/vllm/issues/169
**State**: closed
**Created**: 2023-06-20T21:11:29+00:00
**Closed**: 2023-07-16T21:57:12+00:00
**Comments**: 5
**Labels**: new-model

### Description

https://arxiv.org/abs/1911.02150

For example, StarCoder uses MQA to speed up inference. How does PagedAttention compare to Multi-Query Attention? Are they compatible?

---

## Issue #N/A: run new qwen 7b v1.1 results error?

**Link**: https://github.com/vllm-project/vllm/issues/1192
**State**: closed
**Created**: 2023-09-27T07:57:14+00:00
**Closed**: 2023-10-07T01:37:58+00:00
**Comments**: 7

### Description

python -m vllm.entrypoints.api_server  --model  /***/Qwen-7B-Chat --swap-space 16   --disable-log-requests --host 192.168.19.14 --port 10860 --max-num-seqs 256   --trust-remote-code --tensor-parallel-size 2  --dtype=half

It turned out to be full of exclamation marks!!!
![image](https://github.com/vllm-project/vllm/assets/40717349/b3140269-d8e0-4ed2-ac69-8afd9d2292c9)


---

## Issue #N/A: [Usage]: How to terminate vllm completely?

**Link**: https://github.com/vllm-project/vllm/issues/17273
**State**: closed
**Created**: 2025-04-28T01:19:40+00:00
**Closed**: 2025-05-05T13:41:59+00:00
**Comments**: 1
**Labels**: usage

### Description

### Your current environment

```text
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: CentOS Linux 8 (x86_64)
GCC version: (GCC) 10.5.0
Clang version: Could not collect
CMake version: version 3.20.2
Libc version: glibc-2.29

Python version: 3.10.0 (default, Mar  3 2022, 09:58:08) [GCC 7.5.0] (64-bit runtime)
Python platform: Linux-4.18.0-348.7.1.el8_5.x86_64-x86_64-with-glibc2.29
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA RTX A6000
GPU 1: NVIDIA RTX A6000
GPU 2: NVIDIA RTX A6000
GPU 3: NVIDIA RTX A6000
GPU 4: NVIDIA RTX A6000
GPU 5: NVIDIA RTX A6000
GPU 6: NVIDIA RTX A6000
GPU 7: NVIDIA RTX A6000

Nvidia driver version: 550.135
cuDNN version: Probably one of the following:
/usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudnn.so.8.9.7
/usr/local/cuda-12.2/targets/x86_64-linux/lib

[... truncated for brevity ...]

---

## Issue #N/A: Memory usage decreases as batch size increases

**Link**: https://github.com/vllm-project/vllm/issues/606
**State**: closed
**Created**: 2023-07-27T22:57:59+00:00
**Closed**: 2023-08-07T22:34:52+00:00
**Comments**: 4

### Description

Hi all,

I am running OPT-6.7B [https://huggingface.co/facebook/opt-6.7b] on an A100 GPU with 80GB.

The 'gpu_memory_utilization' is 0.9 (as default) and I am using `torch.cuda.memory_allocated` to get the GPU memory that's allocated.

For input length and output length of 40 and 156, respectively, this is the allocated memory (GB) I see across batch sizes:
Batch | Allocated memory
2        | 72.8 GB
4        | 72.68 GB 
8        | 72.68 GB
16      | 72.55 GB
32      | 72.18 GB
64      | 71.68 GB
128    | 70.55 GB
256    | 68.18 GB
512    | 63.68 GB
For smaller batch sizes, the allocated memory is around 80 * 0.9 as expected, but it becomes smaller as the batch size increases.

Is there a reason to allocate less memory for larger batch sizes? Is the unallocated memory used for some other purposes?
Following the discussion in other issues, the allocated memory for engine includes both the model for inference and KV cache.
With the allocate memory numbers above, does

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: How to start vLLM on a particular GPU?

**Link**: https://github.com/vllm-project/vllm/issues/4981
**State**: closed
**Created**: 2024-05-22T12:41:56+00:00
**Closed**: 2024-06-13T23:06:51+00:00
**Comments**: 9
**Labels**: usage

### Description

### Your current environment

```
Collecting environment information...
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0 
Clang version: Could not collect
CMake version: version 3.29.3
Libc version: glibc-2.31

Python version: 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-1056-azure-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 11.8.89
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA A100 80GB PCIe
GPU 1: NVIDIA A100 80GB PCIe

Nvidia driver version: 545.23.08
cuDNN version: Probably one of the following:
/usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudnn.so.8.7.0
/usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.7.0
/usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.7

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Reimplement and separate beam search on top of vLLM core

**Link**: https://github.com/vllm-project/vllm/issues/8306
**State**: closed
**Created**: 2024-09-09T20:17:13+00:00
**Closed**: 2024-10-07T05:47:05+00:00
**Comments**: 21
**Labels**: RFC

### Description

### Motivation.

A rework of https://github.com/vllm-project/vllm/issues/6226 

After discussing further with the community, we find that the common use case for beam search is: 
1. throughput oriented
2. mainly offline batch inference
3. use one beam search parameter for all the prompts in the batch

After discussing with many contributors, we find:

because beam search is a **search** algorithm, it conflicts with all the rest **sampling** algorithm. As a result, many features in vllm already directly assert beam search is not used, e.g.

https://github.com/vllm-project/vllm/blob/6e36f4fa6ce64619b9ea94c88a157f5783a63a65/vllm/spec_decode/batch_expansion.py#L303-L305

https://github.com/vllm-project/vllm/blob/6e36f4fa6ce64619b9ea94c88a157f5783a63a65/vllm/engine/output_processor/multi_step.py#L100-L103

**keeping beam-search as-is in the codebase, will not benefit current beam search user, as no optimization will target at better beam search performance. What's worse, very

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Qwen2_5_VL-3B :When running the multi-modal model, encountered multiple critical issues related to sequence length and context window limitations.

**Link**: https://github.com/vllm-project/vllm/issues/12940
**State**: closed
**Created**: 2025-02-08T06:34:26+00:00
**Closed**: 2025-03-07T13:46:48+00:00
**Comments**: 5
**Labels**: bug

### Description

### Your current environment


Qwen2_5_VL-3B

[<!-- Failed to upload "企业微信截图_17389938499377.png" -->](url)

### 🐛 Describe the bug

The model throws three main warnings/errors:
Image rescaling issues
Token sequence length exceeding maximum limit
Insufficient context length for multi-modal embeddings

![Image](https://github.com/user-attachments/assets/6634d3e6-fe7c-41dd-aa19-bda346cc780f)

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [RFC]: Adopt mergify for auto-labeling PRs

**Link**: https://github.com/vllm-project/vllm/issues/9192
**State**: closed
**Created**: 2024-10-09T13:32:20+00:00
**Closed**: 2024-10-28T16:38:11+00:00
**Comments**: 4
**Labels**: RFC

### Description

### Motivation.

vLLM is a very active project with a large and busy queue of PRs. Additional usage of github labels would assist with narrowing down which PRs to look at given a person's interests, as well as the state of the PR.

### Proposed Change.

Adopt mergify to perform automated labeling of PRs. https://docs.mergify.com/

While github also provides an [action for PR labeling](https://github.com/actions/labeler), it only supports labeling based on the branch and the files changed. Mergify supports labeling based on more criteria, such as whether a branch has conflicts with `main`.

Configuration would go into `.github/mergify.yml`. An example entry to auto-label PRs that touch files in the `docs/` directory:

```yaml
- name: label-documentation
  description: Automatically apply documentation label
  conditions:
    - or:
      - files~=^[^/]+\.md$
      - files~=^CONTRIBUTING/
      - files~=^docs/
  actions:
    label:
      add:
        - documentation
```

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: About '--chat-template' parameters for model google/paligemma2-3b-ft-docci-448

**Link**: https://github.com/vllm-project/vllm/issues/11471
**State**: closed
**Created**: 2024-12-24T21:23:14+00:00
**Closed**: 2025-01-04T09:49:42+00:00
**Comments**: 4
**Labels**: usage

### Description

### Your current environment

I use [template_llava.jinja](https://github.com/vllm-project/vllm/blob/v0.6.5/examples/template_llava.jinja) to launch the model google/paligemma2-3b-ft-docci-448. Despite working, I wonder 1) how to decide on a template for a specific model and 2) whether my setting is correct for the model google/paligemma2-3b-ft-docci-448?
```text
vllm serve google/paligemma2-3b-ft-docci-448 --chat-template template_llava.jinja  --host 0.0.0.0  --port 8001 --enforce-eager --dtype auto
```
on a 

### How would you like to use vllm

I want to run inference of a [specific model](put link here). I don't know how to integrate it with vllm.


### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: vllm:0.7.1 启动MiniCPM-o-2_6报错

**Link**: https://github.com/vllm-project/vllm/issues/12820
**State**: closed
**Created**: 2025-02-06T10:02:43+00:00
**Closed**: 2025-02-28T06:56:33+00:00
**Comments**: 1
**Labels**: bug

### Description

### Your current environment

vLLM API server version 0.7.1
INFO 02-06 18:01:41 api_server.py:839] args: Namespace(subparser='serve', model_tag='/xiaobaogong/ai/model/MiniCPM-o-2_6/', config='', host=None, port=8000, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, lora_modules=None, prompt_adapters=None, chat_template=None, chat_template_content_format='auto', response_role='assistant', ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_cert_reqs=0, root_path=None, middleware=[], return_tokens_as_token_ids=False, disable_frontend_multiprocessing=False, enable_request_id_headers=False, enable_auto_tool_choice=False, enable_reasoning=False, reasoning_parser=None, tool_call_parser=None, tool_parser_plugin='', model='/xiaobaogong/ai/model/MiniCPM-o-2_6/', task='auto', tokenizer=None, skip_tokenizer_init=False, revision=None, code_revision=None, tokenizer_revision=None, tokenizer_mode='auto', trus

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

## Issue #N/A: [Misc]: [V1] prompt logprobs + chunked prefill can result in `EngineCore` partial prefill output

**Link**: https://github.com/vllm-project/vllm/issues/14239
**State**: closed
**Created**: 2025-03-04T22:43:32+00:00
**Closed**: 2025-03-24T16:42:01+00:00
**Comments**: 1
**Labels**: misc

### Description

See https://github.com/vllm-project/vllm/blob/4f5b059f146adeecd153fa781cf21863ed6679d8/vllm/v1/engine/output_processor.py#L277

Prompt logprobs + chunked prefill can result in engine core returning an output for a partial prefill (in order to send back partial prompt logprobs.) This breaks the invariant that process_outputs is only operating on engine core outputs associated with non-partial completions. Currently this is handled by having `is_prefilling` in `OutputProcessor` check for new decoded tokens, indicating that the completion is not partial.

A follow-up PR should aggregate partial prompt logprobs in the EngineCore.

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: Output state configuration of vision encoder In VLM

**Link**: https://github.com/vllm-project/vllm/issues/9186
**State**: closed
**Created**: 2024-10-09T08:50:00+00:00
**Closed**: 2024-10-23T11:27:38+00:00
**Comments**: 0
**Labels**: feature request

### Description

### Anything you want to discuss about vllm.

When siglip or clip acts as a multimodal vision encoder,  there will have several cases:
- The output state of an intermediate layer is used without layer normalization
- The output state of the last layer is used without layer normalization
- The output state of the last layer is used with layer normalization

For example, In the `LLaVA-Next` code implementation, `post_layernorm` is not used.

#8106 #8155

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Decrease the default size of swap space

**Link**: https://github.com/vllm-project/vllm/issues/69
**State**: closed
**Created**: 2023-05-04T09:54:55+00:00
**Closed**: 2023-05-24T01:22:27+00:00
**Comments**: 1

### Description

The current default swap space size (20 GiB per GPU) is a bit too large. It can lead to OOM especially for the machine with multiple GPUs.

---

## Issue #N/A: [Bug]: vllm v1: RuntimeError: Cannot re-initialize CUDA in forked subprocess

**Link**: https://github.com/vllm-project/vllm/issues/12754
**State**: closed
**Created**: 2025-02-04T23:27:14+00:00
**Closed**: 2025-02-13T18:30:01+00:00
**Comments**: 3
**Labels**: bug, v1

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 02-04 23:23:46 __init__.py:186] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.31.4
Libc version: glibc-2.35

Python version: 3.12.8 (main, Jan 14 2025, 22:49:14) [Clang 19.1.6 ] (64-bit runtime)
Python platform: Linux-5.15.0-122-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA H200
Nvidia driver version: 550.90.12
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.9.7
/usr/lib/x86_64-linux-gnu/li

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Pipeline parallelism support for qwen model

**Link**: https://github.com/vllm-project/vllm/issues/6471
**State**: closed
**Created**: 2024-07-16T11:32:20+00:00
**Closed**: 2024-08-01T19:41:07+00:00
**Comments**: 6
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Pipeline parallelism is only supported for the following architectures: ['AquilaModel', 'AquilaForCausalLM', 'InternLMForCausalLM', 'LlamaForCausalLM', 'LLaMAForCausalLM', 'MistralForCausalLM', 'Phi3ForCausalLM', 'GPT2LMHeadModel'].

### Alternatives

_No response_

### Additional context

_No response_

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

## Issue #N/A: [Feature]: ChatCompletionRequest get default value from generation_config.json

**Link**: https://github.com/vllm-project/vllm/issues/10758
**State**: closed
**Created**: 2024-11-29T03:14:25+00:00
**Closed**: 2024-12-19T10:50:40+00:00
**Comments**: 8
**Labels**: good first issue, feature request

### Description

### 🚀 The feature, motivation and pitch

temperature/top_k/top_p These values ​​will affect the model output，The default value should be read from generation_config.json if the user does not set it.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: VLLM get stucks with Qwen VL 7B

**Link**: https://github.com/vllm-project/vllm/issues/11899
**State**: closed
**Created**: 2025-01-09T14:40:25+00:00
**Closed**: 2025-01-20T14:29:50+00:00
**Comments**: 6
**Labels**: bug

### Description

### Your current environment

I'm using ["v0.6.5"] of VLLM.
When I try to launch  Qwen VL 7B with 100% of the GPU (24GB VRAM) it's ok.
Then even if the model is only 4GB when I reduce to a little bit less the launch of VLLM is getting stuck by printing an endless: 'INFO: 127.0.0.6:XXX - "GET /metrics HTTP/1.1" 200 OK'
I'm confused because I know that I have enough space for the model.

```
      vllm serve Qwen/Qwen2-VL-7B-Instruct-AWQ --trust-remote-code --enable-chunked-prefill --max_model_len 4096 --quantization awq_marlin --gpu_memory_utilization=0.8 --max-num-batched-tokens 4097 --kv-cache-dtype fp8_e4m3

```

### Model Input Dumps

_No response_

### 🐛 Describe the bug

INFO 01-09 06:23:59 api_server.py:651] vLLM API server version 0.6.5
INFO 01-09 06:23:59 api_server.py:652] args: Namespace(subparser='serve', model_tag='Qwen/Qwen2-VL-7B-Instruct-AWQ', config='', host=None, port=8000, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['*'], all

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: minicpmv2_6 OOM

**Link**: https://github.com/vllm-project/vllm/issues/7856
**State**: closed
**Created**: 2024-08-26T03:20:31+00:00
**Closed**: 2024-09-02T04:28:00+00:00
**Comments**: 7
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: CentOS Linux 7 (Core) (x86_64)
GCC version: (GCC) 9.3.1 20200408 (Red Hat 9.3.1-2)
Clang version: Could not collect
CMake version: version 3.29.2
Libc version: glibc-2.17

Python version: 3.9.19 | packaged by conda-forge | (main, Mar 20 2024, 12:50:21)  [GCC 12.3.0] (64-bit runtime)
Python platform: Linux-4.18.0-147.mt20200626.413.el8_1.x86_64-x86_64-with-glibc2.17
Is CUDA available: True
CUDA runtime version: 11.8.89
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA L40
Nvidia driver version: 535.129.03
cuDNN version: Probably one of the following:
/usr/lib64/libcudnn.so.8.9.2
/usr/lib64/libcudnn_adv_infer.so.8.9.2
/usr/lib64/libcudnn_adv_train.so.8.9.2
/usr/lib64/libcudnn_cnn_infer.so.8.9.2
/usr/lib64/libcud

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: how to use openai compatible api to run GGUF model?

**Link**: https://github.com/vllm-project/vllm/issues/8401
**State**: closed
**Created**: 2024-09-12T06:26:12+00:00
**Closed**: 2024-09-19T19:15:56+00:00
**Comments**: 5
**Labels**: usage

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How would you like to use vllm

I want to run inference of a [specific model](put link here). I don't know how to integrate it with vllm.


### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: Running Tensor Parallel on TPUs on Ray Cluster

**Link**: https://github.com/vllm-project/vllm/issues/12058
**State**: closed
**Created**: 2025-01-14T21:32:12+00:00
**Closed**: 2025-01-24T05:41:50+00:00
**Comments**: 9
**Labels**: usage, tpu, ray

### Description

### Your current environment

```text
The output of `python collect_env.py`
The output of `python collect_env.py`
(test_hf_qwen pid=17527, ip=10.130.4.26) Environment Information:
(test_hf_qwen pid=17527, ip=10.130.4.26) Collecting environment information...
(test_hf_qwen pid=17527, ip=10.130.4.26) PyTorch version: 2.6.0.dev20241126+cpu
(test_hf_qwen pid=17527, ip=10.130.4.26) Is debug build: False
(test_hf_qwen pid=17527, ip=10.130.4.26) CUDA used to build PyTorch: None
(test_hf_qwen pid=17527, ip=10.130.4.26) ROCM used to build PyTorch: N/A
(test_hf_qwen pid=17527, ip=10.130.4.26) 
(test_hf_qwen pid=17527, ip=10.130.4.26) OS: Ubuntu 22.04.4 LTS (x86_64)
(test_hf_qwen pid=17527, ip=10.130.4.26) GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
(test_hf_qwen pid=17527, ip=10.130.4.26) Clang version: 14.0.0-1ubuntu1.1
(test_hf_qwen pid=17527, ip=10.130.4.26) CMake version: version 3.31.2
(test_hf_qwen pid=17527, ip=10.130.4.26) Libc version: glibc-2.35
(test_hf_qwen pid=17527, ip=10.13

[... truncated for brevity ...]

---

## Issue #N/A: [New Model]: Support BAAI/bge-reranker-v2-gemma model

**Link**: https://github.com/vllm-project/vllm/issues/19673
**State**: closed
**Created**: 2025-06-16T04:19:03+00:00
**Closed**: 2025-07-07T14:46:05+00:00
**Comments**: 8
**Labels**: new-model

### Description

### The model to consider.

https://huggingface.co/BAAI/bge-reranker-v2-gemma

### The closest model vllm already supports.

BAAI/bge-reranker-v2-m3
google/gemma-2b

### What's your difficulty of supporting the model you want?

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: 'dict' object has no attribute 'is_kv_transfer_instance'

**Link**: https://github.com/vllm-project/vllm/issues/19259
**State**: closed
**Created**: 2025-06-06T07:36:09+00:00
**Closed**: 2025-06-14T19:32:08+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
Your output of `python collect_env.py` here
```

</details>


### 🐛 Describe the bug

- run vllm container
```
docker run -d -it --rm --privileged --entrypoint /bin/bash --network host --name poolv1-mbl-test-2  --shm-size 512g --gpus all   -v /:/disc       vllm/vllm-openai:v0.9.0.1

docker exec -it poolv1-mbl-test-2 bash
pip install lmcache
```

- start vllm by lmcache example.

The following python script is copied from `examples/others/lmcache/cpu_offload_lmcache.py` and did some minor changes to run model locally.
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file demonstrates the example usage of cpu offloading
with LMCache in vLLM v1 or v0.

Usage:

    Specify vLLM version

    -v v0 : Use LMCacheConnector
            model = mistralai/Mistral-7B-Instruct-v0.2
            (Includes enab

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Help, RuntimeError: CUDA error: no kernel image is available for execution on the device

**Link**: https://github.com/vllm-project/vllm/issues/18835
**State**: closed
**Created**: 2025-05-28T11:59:00+00:00
**Closed**: 2025-06-26T17:16:43+00:00
**Comments**: 5
**Labels**: bug

### Description

### Your current environment


ERROR 05-28 19:38:44 [dump_input.py:68] Dumping input data
--- Logging error ---
Traceback (most recent call last):
  File "/root/miniconda3/envs/vllm-qwen3/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 207, in execute_model
    return self.model_executor.execute_model(scheduler_output)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/envs/vllm-qwen3/lib/python3.12/site-packages/vllm/v1/executor/abstract.py", line 86, in execute_model
    output = self.collective_rpc("execute_model",
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/envs/vllm-qwen3/lib/python3.12/site-packages/vllm/executor/uniproc_executor.py", line 56, in collective_rpc
    answer = run_method(self.driver_worker, method, args, kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/envs/vllm-qwen3/lib/python3.12/site-packages/vllm/utils.py", line 2605, in run_method
  

[... truncated for brevity ...]

---

## Issue #N/A: CUDA error: an illegal memory acces with Falcon 40B

**Link**: https://github.com/vllm-project/vllm/issues/767
**State**: closed
**Created**: 2023-08-16T00:11:11+00:00
**Closed**: 2023-09-10T08:39:04+00:00
**Comments**: 15
**Labels**: bug

### Description

Hi,
I am testing different models with vllm. I see 
```CUDA error: an illegal memory access``` when I use falcon 40 b. The code I use is 
```
llm = LLM(model=ckpt_dir,tensor_parallel_size=4,trust_remote_code=True,gpu_memory_utilization=0.8)
sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=300)
results = llm.generate(prompts, sampling_params)
```
I am using an A100 with 4 GPUs. Please let me know if you have any questions

---

