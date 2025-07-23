# steady_state_6months - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 3
- Closed Issues: 27

### Label Distribution

- bug: 15 issues
- stale: 10 issues
- usage: 6 issues
- new-model: 3 issues
- performance: 2 issues
- rocm: 2 issues
- installation: 2 issues
- feature request: 1 issues
- documentation: 1 issues
- structured-output: 1 issues

---

## Issue #N/A: [Bug]: The random seed behavior when loading a model in vLLM is confusing.

**Link**: https://github.com/vllm-project/vllm/issues/11953
**State**: closed
**Created**: 2025-01-11T07:38:52+00:00
**Closed**: 2025-02-10T15:26:51+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.11.10 (main, Sep  7 2024, 18:35:41) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.8.0-50-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 4070 Ti
Nvidia driver version: 550.142
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:            

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: File Access Error When Using RunAI Model Streamer with S3 in VLLM

**Link**: https://github.com/vllm-project/vllm/issues/12311
**State**: closed
**Created**: 2025-01-22T09:21:16+00:00
**Closed**: 2025-01-24T03:06:08+00:00
**Comments**: 7
**Labels**: usage

### Description

### Your current environment

```text
I am encountering a persistent issue when attempting to serve a model from an S3 bucket using the vllm serve command with the --load-format runai_streamer option. Despite having proper access to the S3 bucket and all required files being present, the process fails with a "File access error." Below are the details of the issue:

Command Used:
vllm serve s3://hip-general/benchmark-model-loading/ --load-format runai_streamer

Error Message:
Exception: Could not send runai_request to libstreamer due to: b'File access error'

Environment Details:
VLLM version: 0.6.6
Python version: 3.12
RunAI Model Streamer version: 0.11.2
S3 Region: us-west-2


Files in S3 Bucket:
config.json
generation_config.json
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json
special_tokens_map.json
tokenizer.json
tokenizer_config.json
```


### my deployment file is 
api

[... truncated for brevity ...]

---

## Issue #N/A: [New Model]: Request for supporting microsoft/phi-4  Model

**Link**: https://github.com/vllm-project/vllm/issues/12358
**State**: closed
**Created**: 2025-01-23T14:22:04+00:00
**Closed**: 2025-01-24T03:13:11+00:00
**Comments**: 2
**Labels**: new-model

### Description

### The model to consider.

https://huggingface.co/microsoft/phi-4

### The closest model vllm already supports.

https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/phi3.py

### What's your difficulty of supporting the model you want?

Do not much understand the steps to add the model

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Question]: vllm是否乐意支持基于其他人工智能框架的模型，如Mindspore，PaddlePaddle

**Link**: https://github.com/vllm-project/vllm/issues/11505
**State**: closed
**Created**: 2024-12-26T08:05:06+00:00
**Closed**: 2024-12-26T08:05:18+00:00
**Comments**: 0
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

你好！目前vllm支持的模型大多是pytorch实现的模型，想问一下vllm是否愿意支持基于其他人工智能框架的模型呢？现在paddlepaddle的一些模型也已经集成到了huggingface上

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: how to input messages as multi-message (a batch) instead of just one

**Link**: https://github.com/vllm-project/vllm/issues/12234
**State**: closed
**Created**: 2025-01-21T02:58:50+00:00
**Closed**: 2025-01-21T19:13:41+00:00
**Comments**: 2
**Labels**: usage

### Description

### Your current environment

currently i could input a message and call vllm api.
the message could be like:
messages = [
    {"role": "system", "content": "In the following sentence, please give some suggestions to improve word usage. Please give the results with the JSON format of {“original word”: [“suggestion 1”, “suggestion 2”]}. The 'original word' should include all words that can be improved in the sentence, directly extracted from the sentence itself, and the suggestions should be ranked in order of the degree of improvement, from the most effective to the least."},
    {"role": "user", "content": "In conclusion, the professor pointed out the inconsistencies between the reading and the listening passages and explained why the arguments in the speech are more reliable."},
]

but if i want to input a batch size > 1 messages, like:
messages = [[
    {"role": "system", "content": "In the following sentence, please give some suggestions to improve word usage. Please give the resul

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: Huge prompts impact other parallel generations

**Link**: https://github.com/vllm-project/vllm/issues/11893
**State**: closed
**Created**: 2025-01-09T11:42:20+00:00
**Closed**: 2025-05-11T02:12:49+00:00
**Comments**: 4
**Labels**: performance, stale

### Description

### Proposal to improve performance

_No response_

### Report of performance regression

_No response_

### Misc discussion on performance

Hello, 
  I am running VLLM OpenAI compatible server in the environment below and it works really well in general. However, I have an issue when a huge prompt comes, the other already running generations from the same VLLM server get extremely slowed down until the generation starts for the huge prompt. 

Can I do anything about this behavior? 

### Your current environment (if you think it is necessary)

PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.10.230-223.885.amzn2.x86_64-x86_64

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: AMD GPU docker image build No matching distribution found for torch==2.6.0.dev20241113+rocm6.2

**Link**: https://github.com/vllm-project/vllm/issues/12178
**State**: closed
**Created**: 2025-01-17T23:36:10+00:00
**Closed**: 2025-03-12T05:50:14+00:00
**Comments**: 2
**Labels**: bug, rocm

### Description

### Your current environment

Archlinux 13th Gen Intel(R) Core(TM) i9-13900HX environment to build the docker image

### Model Input Dumps

_No response_

### 🐛 Describe the bug

Trying to build the AMD GPU docker image:
```
git checkout v0.6.6.post1
DOCKER_BUILDKIT=1 docker build -f Dockerfile.rocm -t substratusai/vllm-rocm:v0.6.6.post1 .
```

Results in following error:

```
1.147 Looking in indexes: https://pypi.org/simple, https://download.pytorch.org/whl/nightly/rocm6.2
1.717 ERROR: Could not find a version that satisfies the requirement torch==2.6.0.dev20241113+rocm6.2 (from versions: 1.7.1, 1.8.0, 1.8.1, 1.9.0, 1.9.1, 1.10.0, 1.10.1, 1.10.2, 1.11.0, 1.12.0, 1.12.1, 1.13.0, 1.13.1, 2.0.0, 2.0.1, 2.1.0, 2.1.1, 2.1.2, 2.2.0, 2.2.1, 2.2.2, 2.3.0, 2.3.1, 2.4.0, 2.4.1, 2.5.0, 2.5.1, 2.6.0.dev20241119+rocm6.2, 2.6.0.dev20241120+rocm6.2, 2.6.0.dev20241121+rocm6.2, 2.6.0.dev20241122+rocm6.2)
2.135 ERROR: No matching distribution found for torch==2.6.0.dev20241113+rocm6.2
------
Dockerfil

[... truncated for brevity ...]

---

## Issue #N/A: [Installation][build][docker]: rocm Dockerfile pinned to stale python torch nightly wheel builds

**Link**: https://github.com/vllm-project/vllm/issues/12066
**State**: closed
**Created**: 2025-01-15T05:01:48+00:00
**Closed**: 2025-01-17T04:15:05+00:00
**Comments**: 3
**Labels**: installation, rocm

### Description

### How you are installing vllm

https://github.com/vllm-project/vllm/blob/0794e7446efca1fd7b8ea1cde96777897660cdea/Dockerfile.rocm#L48-L58

Python packages for `torch==2.6.0.dev20241113+rocm6.2` and `torchvision==0.20.0.dev20241113+rocm6.2` are no longer available due to them being outside of the build retention window

Wheel Index:
https://download.pytorch.org/whl/nightly/torch/

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [New Model]: command-r7b

**Link**: https://github.com/vllm-project/vllm/issues/11650
**State**: closed
**Created**: 2024-12-31T07:28:21+00:00
**Closed**: 2025-01-02T06:46:55+00:00
**Comments**: 2
**Labels**: new-model

### Description

### The model to consider.

https://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024

### The closest model vllm already supports.

I don‘t know，but i had installe the newest transformers and newest vllm,and I had to see the history of Cohere2ForCausalLM,but it still error  after i tried again

### What's your difficulty of supporting the model you want?

ValueError: Model architectures ['CohereForCausalLM'] are not supported for now. Supported architectures: ['AquilaModel', 'AquilaForCausalLM', 'BaiChuanForCausalLM', 'BaichuanForCausalLM', 'BloomForCausalLM', 'ChatGLMModel', 'ChatGLMForConditionalGeneration', 'DeciLMForCausalLM', 'DeepseekForCausalLM', 'FalconForCausalLM', 'GemmaForCausalLM', 'GPT2LMHeadModel', 'GPTBigCodeForCausalLM', 'GPTJForCausalLM', 'GPTNeoXForCausalLM', 'InternLMForCausalLM', 'InternLM2ForCausalLM', 'LlamaForCausalLM', 'LLaMAForCausalLM', 'MistralForCausalLM', 'MixtralForCausalLM', 'QuantMixtralForCausalLM', 'MptForCausalLM', 'MPTForCausalLM', 'OLMoForCausalL

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: meta/lLlama-3.1-8B-Instruct with vllm class very slow in comparision to other models

**Link**: https://github.com/vllm-project/vllm/issues/12047
**State**: closed
**Created**: 2025-01-14T16:16:45+00:00
**Closed**: 2025-01-16T15:30:08+00:00
**Comments**: 1
**Labels**: usage

### Description

### Your current environment

Running 2 Nvidia A30 GPUs. Environment works perfectly fine for non-llama models.



### How would you like to use vllm

I initialize models based on the following snipped:
```
llm = LLM(model=args.llm_identifier)
sampling_params = llm.get_default_sampling_params()
sampling_params.max_tokens = 1024

# Generate texts from the prompts
outputs = llm.generate(prompts, sampling_params)
```

When using this code with [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) it is significantly faster than running [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). Also the output for the Llama Model is much longer and does not really refer to the prompt.

These are my prompt templates:
```
template = """
<|system|>: {system_prompt}
<|user|>: {user_prompt}
<|assistant|>:
"""

system_prompt = """
You are a helpful assistant. Your task is to extract information from the given text into a markdown table.
- You MUST only Out

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Error while importing vllm since v0.6.6

**Link**: https://github.com/vllm-project/vllm/issues/11683
**State**: closed
**Created**: 2025-01-02T10:58:59+00:00
**Closed**: 2025-01-06T06:46:43+00:00
**Comments**: 18
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
>>> import vllm
Warning: Your installation of OpenCV appears to be broken: module 'cv2.dnn' has no attribute 'DictValue'.Please follow the instructions at https://github.com/opencv/opencv-python/issues/884 to correct your environment. The import of cv2 has been skipped.
WARNING 01-02 10:53:16 cuda.py:32] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python3.10/dist-packages/vllm/__init__.py", line 3, in <module>
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
  File "/usr/local/lib/python3.10/dist-packages/vllm/engine/arg_utils.py", line 11, in <m

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: How can I check which operators are used by vLLM-Llama-2-7b-hf?

**Link**: https://github.com/vllm-project/vllm/issues/12293
**State**: closed
**Created**: 2025-01-22T02:00:53+00:00
**Closed**: 2025-02-03T02:03:01+00:00
**Comments**: 9
**Labels**: usage

### Description

### Your current environment

Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.26.0
Libc version: glibc-2.35

Python version: 3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-125-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 3090
Nvidia driver version: 550.120
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                        46 bits physical, 48 bits virtual
Byte Order:      

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Cutlass 2:4 Sparsity + FP8/Int8 Quant RuntimeError: Error Internal

**Link**: https://github.com/vllm-project/vllm/issues/11763
**State**: closed
**Created**: 2025-01-06T08:02:19+00:00
**Closed**: 2025-04-10T02:11:10+00:00
**Comments**: 9
**Labels**: bug, stale

### Description

### Your current environment
```
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Alibaba Group Enterprise Linux Server 7.2 (Paladin) (x86_64)
GCC version: (GCC) 10.2.1 20200825 (Alibaba 10.2.1-3 2.17)
Clang version: Could not collect
CMake version: version 3.26.4
Libc version: glibc-2.32

Python version: 3.10.13 (main, Sep 11 2023, 13:44:35) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.10.134-16.3.al8.x86_64-x86_64-with-glibc2.32
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H20
GPU 1: NVIDIA H20
GPU 2: NVIDIA H20
GPU 3: NVIDIA H20

Nvidia driver version: 535.183.06
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: init_mm_limits_per_prompt not been called when using V1 + TensorSplit + Qwen2VL

**Link**: https://github.com/vllm-project/vllm/issues/12245
**State**: closed
**Created**: 2025-01-21T05:50:38+00:00
**Closed**: 2025-01-21T10:09:40+00:00
**Comments**: 1
**Labels**: bug

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

OS: Debian GNU/Linux 11 (bullseye) (x86_64)
GCC version: (Debian 10.2.1-6) 10.2.1 20210110
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.31

Python version: 3.10.16 (main, Jan  5 2025, 05:32:43) [Clang 19.1.6 ] (64-bit runtime)
Python platform: Linux-5.10.135.bsk.6-amd64-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA H20
GPU 1: NVIDIA H20

Nvidia driver version: 535.161.08
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bi

[... truncated for brevity ...]

---

## Issue #N/A: [Doc]: Performance/Optimization Page doesn't mention Pipeline Parallel Size

**Link**: https://github.com/vllm-project/vllm/issues/12012
**State**: closed
**Created**: 2025-01-13T15:36:07+00:00
**Closed**: 2025-03-01T05:43:56+00:00
**Comments**: 0
**Labels**: documentation

### Description

### 📚 The doc issue

In the Page
https://github.com/vllm-project/vllm/blob/main/docs/source/performance/optimization.md

One of the recommended options includes the following:

```
Increase tensor_parallel_size. This approach shards model weights, so each GPU has more memory available for KV cache.
```

This document does not mention increasing `pipeline_parallel_size` which would also result in the model being sharded across more GPUs so their is more memory available for KV cache.

### Suggest a potential alternative/fix

Increase `tensor_parallel_size` or `pipeline_parallel_size` (if using Multi-Node Multi-GPU). This approach shards model weights, so each GPU has more memory available for KV cache.

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [New Model]: Jinja template for  Llama3.3

**Link**: https://github.com/vllm-project/vllm/issues/11854
**State**: closed
**Created**: 2025-01-08T15:38:53+00:00
**Closed**: 2025-05-10T02:06:44+00:00
**Comments**: 6
**Labels**: new-model, stale

### Description

### The model to consider.

https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct

### The closest model vllm already supports.

vllm already supports Llama3.3 deployment but I am wondering on the chat template for it.

### What's your difficulty of supporting the model you want?

Looking for chat template examples for Llama3.3 and wondering the closest or best recommended from https://github.com/vllm-project/vllm/tree/main/examples ?

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: vLLM LoRA Crash when using Dynamic Loading

**Link**: https://github.com/vllm-project/vllm/issues/11702
**State**: closed
**Created**: 2025-01-03T02:49:49+00:00
**Closed**: 2025-01-10T07:56:37+00:00
**Comments**: 7
**Labels**: bug

### Description

### Your current environment

```
root@mistral-7b-lora-7946cc6459-jqx4h:/vllm-workspace# python3 collect_env.py 
Collecting environment information...
PyTorch version: 2.5.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.12.7 (main, Oct  1 2024, 08:52:12) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.6.41-amd64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA H100 80GB HBM3
Nvidia driver version: 535.86.10
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                  

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Run Pixtral-Large-Instruct-2411 raised a error Attempted to assign 1 x 2074 = 2074 multimodal tokens to 2040 placeholders

**Link**: https://github.com/vllm-project/vllm/issues/11792
**State**: closed
**Created**: 2025-01-07T05:49:21+00:00
**Closed**: 2025-07-11T02:16:55+00:00
**Comments**: 3
**Labels**: bug, stale

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

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 10.5.0-1ubuntu1~20.04) 10.5.0
Clang version: Could not collect
CMake version: version 3.31.2
Libc version: glibc-2.31

Python version: 3.12.8 (main, Dec  4 2024, 08:54:13) [GCC 9.4.0] (64-bit runtime)
Python platform: Linux-5.14.0-284.11.1.el9_2.x86_64-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A800 80GB PCIe
GPU 1: NVIDIA A800 80GB PCIe
GPU 2: NVIDIA A800 80GB PCIe
GPU 3: NVIDIA A800 80GB PCIe
GPU 4: NVIDIA A800 80GB PCIe
GPU 5: NVIDIA A800 80GB PCIe
GPU 6: NVIDIA A800 80GB PCIe
GPU 7: NVIDIA A800 80GB PCIe

Nvidia driver version: 535

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf8 in position 0: invalid start byte

**Link**: https://github.com/vllm-project/vllm/issues/12390
**State**: open
**Created**: 2025-01-24T07:08:05+00:00
**Comments**: 3
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

Collecting environment information...
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 24.04 LTS (x86_64)
GCC version: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.39

Python version: 3.9.20 (main, Oct  3 2024, 07:27:41)  [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.153.1-microsoft-standard-WSL2-x86_64-with-glibc2.39
Is CUDA available: True
CUDA runtime version: 12.0.140
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 2060
Nvidia driver version: 556.12
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:    

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Phi-3-small-8k cannot be served for vllm >= 0.6.5

**Link**: https://github.com/vllm-project/vllm/issues/12124
**State**: closed
**Created**: 2025-01-16T16:50:23+00:00
**Closed**: 2025-02-06T18:01:45+00:00
**Comments**: 10
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov  6 2024, 20:22:13) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.8.0-1021-aws-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A10G
Nvidia driver version: 550.127.05
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                        48 bits physical, 48 bits virtual
B

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: How to run vllm with regression task, just like classify task

**Link**: https://github.com/vllm-project/vllm/issues/12379
**State**: closed
**Created**: 2025-01-24T02:19:51+00:00
**Closed**: 2025-05-25T02:29:11+00:00
**Comments**: 3
**Labels**: usage, stale

### Description

### Your current environment

I have trained qwen2.5 with regression task,  below is inference code : 

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("qwen3_score_0123", trust_remote_code=True)

model = AutoModelForSequenceClassification.from_pretrained(
    "qwen3_score_0123",
    num_labels=1,
    problem_type="regression",
    trust_remote_code=True
).to(torch.bfloat16)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

text = """
hello
"""

inputs = tokenizer(
    text,
    truncation=True,
    padding='max_length',
    max_length=512,
    return_tensors="pt"
)

inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

predicted_score = outputs.logits.item() 



### How would you like to use vllm

I want to inference the qwen llm regression model with vllm, need help ,thanks


### Befo

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: MQLLMEgine Error on Apple Silicon M4 Pro

**Link**: https://github.com/vllm-project/vllm/issues/11863
**State**: closed
**Created**: 2025-01-08T19:14:23+00:00
**Closed**: 2025-01-08T22:16:31+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```

</details>


### Model Input Dumps

_No response_

### 🐛 Describe the bug

I receive the following error when calling VLLM which is stop on MacOS on my M4 Pro Mac mini with 64GB RAM. The same error occurs on my M1 Ultra Mac Studio. I have followed the installing steps for macOS correctly, including reinstalling Xcode command line tools. The errors happens instantly when calling the server, and occurs with any model I have tried. 

INFO 01-08 18:58:57 logger-py:37] Received request cmpl-e4776fdb878c4500912374ad23cd2785-0: prompt:
'User: Why is the sky blue?\nAssistant:',
params: SamplingParams (n=1,
presence_penalty=0.0,
frequency_penalty=0.0,
repetition_penalty=1.0,
=-1-
min_p=0.0, seed=None,
=None,
stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False,
max tokens=100,
guided_decoding=None),

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: CUDA initialization error with vLLM 0.5.4 and PyTorch 2.4.0+cu121

**Link**: https://github.com/vllm-project/vllm/issues/12189
**State**: closed
**Created**: 2025-01-19T12:06:45+00:00
**Closed**: 2025-06-22T02:15:00+00:00
**Comments**: 4
**Labels**: bug, stale

### Description

### Your current environment
```
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.2 LTS (x86_64)
GCC version: (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0
Clang version: Could not collect
CMake version: version 3.31.4
Libc version: glibc-2.35

Python version: 3.11.11 (main, Dec 11 2024, 16:28:39) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.10.134-008.7.kangaroo.al8.x86_64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.1.105
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

Nvidia driver version: 535.54.03
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.1
/usr/lib/x86_64-linux-gnu/libcudnn_adv_

[... truncated for brevity ...]

---

## Issue #N/A: vllm build failure on IBM ppc64le

**Link**: https://github.com/vllm-project/vllm/issues/11616
**State**: closed
**Created**: 2024-12-30T07:24:08+00:00
**Closed**: 2025-01-08T05:05:39+00:00
**Comments**: 1
**Labels**: installation

### Description

### Your current environment
IBM powerpc64le.
RHEL 9/ubi9
vllm: Built from source main branch.

### How you are installing vllm

```sh
docker build -t vllm:latest -f Dockerfile.ppc64le .
```
Error:
```

51.54     cargo:rerun-if-env-changed=PKG_CONFIG_SYSROOT_DIR
51.54 
51.54 
51.54     Could not find openssl via pkg-config:
51.54 
51.54     pkg-config exited with status code 1
51.54     > PKG_CONFIG_ALLOW_SYSTEM_CFLAGS=1 pkg-config --libs --cflags openssl
51.54 
51.54     The system library `openssl` required by crate `openssl-sys` was not found.
51.54     The file `openssl.pc` needs to be installed and the PKG_CONFIG_PATH environment variable must contain its parent directory.
51.54     The PKG_CONFIG_PATH environment variable is not set.
51.54 
51.54     HINT: if you have installed the library, try setting PKG_CONFIG_PATH to the directory containing `openssl.pc`.
51.55 
51.55 
51.55     cargo:warning=Could not find directory of OpenSSL installation, and th

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Close feature gaps when using xgrammar for structured output

**Link**: https://github.com/vllm-project/vllm/issues/12131
**State**: open
**Created**: 2025-01-16T21:10:50+00:00
**Comments**: 2
**Labels**: bug, structured-output

### Description

### 🐛 Describe the bug

As of v0.6.5, we use xgrammar as the default backend for structured output. However, not all ways of expressing output requirements are supported. This issue is for tracking the list of known cases needed to be resolved for making xgrammar the default in all cases.

Fallback cases can be found here: https://github.com/vllm-project/vllm/blob/d06e824006d1ba4b92871347738ce1b89f658499/vllm/model_executor/guided_decoding/__init__.py#L40-L76

- [ ] non-x86 architectures
- [ ] regex 
  - related: https://github.com/mlc-ai/xgrammar/pull/144
  - https://github.com/mlc-ai/xgrammar/issues/175
- [ ] choice \
  - https://github.com/vllm-project/vllm/pull/12632
- [ ] jsonschema support is incomplete
  - https://github.com/mlc-ai/xgrammar/issues/160
- [ ] lark grammars

---

## Issue #N/A: [Usage]: How do I set default temperature for openai compatible server?

**Link**: https://github.com/vllm-project/vllm/issues/11861
**State**: closed
**Created**: 2025-01-08T17:46:11+00:00
**Closed**: 2025-01-15T14:52:36+00:00
**Comments**: 6
**Labels**: usage

### Description

### Your current environment

```text
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.12.8 (main, Dec  4 2024, 08:54:12) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.4.0-177-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 4090
Nvidia driver version: 535.113.01
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:                      43 bits physical,

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: preemptmode recompute

**Link**: https://github.com/vllm-project/vllm/issues/11805
**State**: closed
**Created**: 2025-01-07T07:51:56+00:00
**Closed**: 2025-05-08T02:10:47+00:00
**Comments**: 2
**Labels**: bug, stale

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
                                                                                                            
OS: Ubuntu 22.04.4 LTS (x86_64)                                                                             
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0                                   

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: RuntimeError: CUDA error: device-side assert triggered CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.

**Link**: https://github.com/vllm-project/vllm/issues/11931
**State**: closed
**Created**: 2025-01-10T14:00:43+00:00
**Closed**: 2025-01-11T05:29:28+00:00
**Comments**: 6
**Labels**: bug

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

OS: Ubuntu 24.04.1 LTS (x86_64)
GCC version: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
Clang version: Could not collect
CMake version: version 3.31.2
Libc version: glibc-2.39

Python version: 3.10.16 (main, Dec  4 2024, 08:53:38) [GCC 13.2.0] (64-bit runtime)
Python platform: Linux-5.4.0-204-generic-x86_64-with-glibc2.39
Is CUDA available: True
CUDA runtime version: 12.2.91
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA A10
GPU 1: NVIDIA A10
GPU 2: NVIDIA A10
GPU 3: NVIDIA A10

Nvidia driver version: 525.147.05
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: CPU Offloading errors (Worker.__init__() got an unexpected keyword argument 'kv_cache_dtype')

**Link**: https://github.com/vllm-project/vllm/issues/11986
**State**: closed
**Created**: 2025-01-13T05:41:04+00:00
**Closed**: 2025-05-24T02:07:52+00:00
**Comments**: 6
**Labels**: bug, stale

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

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.12.8 | packaged by Anaconda, Inc. | (main, Dec 11 2024, 16:31:09) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.8.0-50-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.6.85
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 4090
Nvidia driver version: 550.120
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):   

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: V1 vs V0 with multi-steps

**Link**: https://github.com/vllm-project/vllm/issues/11649
**State**: open
**Created**: 2024-12-31T06:45:17+00:00
**Comments**: 2
**Labels**: performance, stale

### Description

### Proposal to improve performance

N/A

### Report of performance regression

For V1:
python -m vllm.entrypoints.openai.api_server --model NousResearch/Meta-Llama-3.1-8B-Instruct --tensor-parallel-size 8 --max-num-seqs 32 --max-model-len 4096 --disable-sliding-window --return-tokens-as-token-ids --port 8080 --enable-prefix-caching --enable-chunked-prefill --disable-log-requests --disable-log-stats

For V0:
python -m vllm.entrypoints.openai.api_server --model NousResearch/Meta-Llama-3.1-8B-Instruct --tensor-parallel-size 8 --max-num-seqs 32 --max-model-len 4096 --num-scheduler-steps 32 --multi-step-stream-outputs --disable-sliding-window --return-tokens-as-token-ids --port 8080 --enable-prefix-caching --enable-chunked-prefill --disable-log-requests --disable-log-stats

run the following script:
time curl -XPOST -s http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' -H 'Authorization: Bearer 1234' -d '{"model": "NousResearch/Meta-Llama-3.1-8B-Inst

[... truncated for brevity ...]

---

