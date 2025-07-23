# no_engagement_0 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 7
- Closed Issues: 23

### Label Distribution

- bug: 13 issues
- stale: 6 issues
- usage: 5 issues
- feature request: 3 issues
- new-model: 1 issues
- RFC: 1 issues
- performance: 1 issues
- documentation: 1 issues

---

## Issue #N/A: [Feature]: try to gracefully destroy process group in `vllm serve` on handling Ctrl+C (prior to processes termination)

**Link**: https://github.com/vllm-project/vllm/issues/19196
**State**: open
**Created**: 2025-06-05T09:16:45+00:00
**Comments**: 0
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Otherwise I get 
```
[rank0]:[W604 11:18:57.117195760 ProcessGroupNCCL.cpp:1476] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
```

which seems harmless, but it would be better to not have this warning if possible

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: inference Bug

**Link**: https://github.com/vllm-project/vllm/issues/1130
**State**: closed
**Created**: 2023-09-21T12:22:33+00:00
**Closed**: 2024-03-08T12:22:04+00:00
**Comments**: 4

### Description

问题描述：
使用vllm进行gpt2推理，输入prompt为空会触发以下error，之后任何请求都不会出结果，除非重启服务；不仅是这个场景，其他情况下触发这个error也会出现相同情况

Problem Description:
When using vllm for gpt2 inference, if the input prompt is empty, it triggers the following error, and thereafter no request will yield results unless the service is restarted. This situation occurs not only in this scenario but also in other cases where this error is triggered.

error:

ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "/home/ubuntu/anaconda3/envs/lyra/lib/python3.9/site-packages/uvicorn/protocols/http/h11_impl.py", line 407, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
  File "/home/ubuntu/anaconda3/envs/lyra/lib/python3.9/site-packages/uvicorn/middleware/proxy_headers.py", line 78, in __call__
    return await self.app(scope, receive, send)
  File "/home/ubuntu/anaconda3/envs/lyra/lib/python3.9/site-packages/fastapi/applications.py", line 270, in __call__
    await super().__call_

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: API Returns Only Single Result Despite n=8 Parameter Setting

**Link**: https://github.com/vllm-project/vllm/issues/17173
**State**: closed
**Created**: 2025-04-25T08:41:53+00:00
**Closed**: 2025-04-25T12:47:46+00:00
**Comments**: 8
**Labels**: bug

### Description

### Your current environment

Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 24.04.1 LTS (x86_64)
GCC version: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
Clang version: Could not collect
CMake version: version 3.28.3
Libc version: glibc-2.39

Python version: 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.8.0-57-generic-x86_64-with-glibc2.39
Is CUDA available: True
CUDA runtime version: 12.6.85
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H20
GPU 1: NVIDIA H20
GPU 2: NVIDIA H20
GPU 3: NVIDIA H20
GPU 4: NVIDIA H20
GPU 5: NVIDIA H20
GPU 6: NVIDIA H20
GPU 7: NVIDIA H20

Nvidia driver version: 565.57.01
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-

[... truncated for brevity ...]

---

## Issue #N/A: Error: max retries exceeded: unexpected EOF

**Link**: https://github.com/vllm-project/vllm/issues/2187
**State**: closed
**Created**: 2023-12-19T00:08:59+00:00
**Closed**: 2023-12-21T03:51:07+00:00
**Comments**: 5

### Description

pulling manifest 
pulling bdb11b0699e0...  38% ▕██████████████████████████████████████████████████                                                                                   ▏  10 GB/ 26 GB  2.1 MB/s   2h11m
Error: max retries exceeded: unexpected EOF

---

## Issue #N/A: [Bug]: OLMoE produces incorrect output with TP>1

**Link**: https://github.com/vllm-project/vllm/issues/8747
**State**: closed
**Created**: 2024-09-23T22:42:56+00:00
**Closed**: 2025-01-24T01:58:52+00:00
**Comments**: 3
**Labels**: bug, stale

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

We can perform evaluations for GSM8k with lm-eval to see the issue. Please use `pip install lm-eval==0.4.3`

TP=1
```
VLLM_WORKER_MULTIPROC_METHOD=spawn lm_eval --model vllm --model_args pretrained=allenai/OLMoE-1B-7B-0924-Instruct,tensor_parallel_size=1 --tasks gsm8k --num_fewshot 5 --batch_size auto

vllm (pretrained=allenai/OLMoE-1B-7B-0924-Instruct,tensor_parallel_size=1), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.3457|±  |0.0131|
|     |       |strict-match    |     5|exact_match|↑  |0.3313|±  |0.0130|
```

TP

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Jamba-1.5-mini doesn't run on A100 with 70GB available memory

**Link**: https://github.com/vllm-project/vllm/issues/7992
**State**: closed
**Created**: 2024-08-29T10:31:50+00:00
**Closed**: 2024-08-31T07:19:12+00:00
**Comments**: 3
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
WARNING 08-29 10:29:48 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: Could not collect
CMake version: version 3.29.3
Libc version: glibc-2.31

Python version: 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-1068-azure-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and c

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Why AWQ quantization does not support expert parallelism?

**Link**: https://github.com/vllm-project/vllm/issues/15760
**State**: open
**Created**: 2025-03-30T03:18:28+00:00
**Comments**: 1
**Labels**: usage, stale

### Description

### Your current environment

PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.26.0
Libc version: glibc-2.35

Python version: 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.2.0-39-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.99
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A40
GPU 1: NVIDIA A40

Nvidia driver version: 560.35.03


### How would you like to use vllm

I want to run Deepseek-V3-AWQ with expert parallelism, but I get a error saying " Expert fused Parallelism is not supported for Marlin MoE method". I want to know whether awq can achieve expert parallelism under the current vllm architecture.
"


### Before submitting a new issu

[... truncated for brevity ...]

---

## Issue #N/A: Understanding about LLM class from vllm

**Link**: https://github.com/vllm-project/vllm/issues/2380
**State**: closed
**Created**: 2024-01-08T10:26:43+00:00
**Closed**: 2024-04-03T15:43:34+00:00
**Comments**: 1

### Description

is LLM class from vllm is asynchronous by nature ?
why am i asking this from the [slides](https://docs.google.com/presentation/d/1QL-XPFXiFpDBh86DbEegFXBXFXjix4v032GhShbKf3s/edit#slide=id.g24ad94a0065_0_84) on the first meetup  it has mentioned that llm is synchronous rather api_server and openai_server are asynchronous ?

if that is true ,how to call the llm model asynchronously ?

correct me if i am wrong !
TIA

---

## Issue #N/A: Where does the default number 43328 of KV cache come from and How can I change it?

**Link**: https://github.com/vllm-project/vllm/issues/11391
**State**: closed
**Created**: 2024-12-21T06:11:52+00:00
**Closed**: 2025-04-21T02:10:11+00:00
**Comments**: 2
**Labels**: usage, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```
Not an technical issue, not related to environment.

### How would you like to use vllm

I have encountered "The model's max seq len (56000) is larger than the maximum number of tokens that can be stored in KV cache (43328)" numerous times. Although it can be solved by setting a smaller --max-model-len parameter, it's actually an issue when you really want to set a large --max-model-len for a large context. What makes it more complicated is that the KV cache number changes automatically when we set different --max-model-len. My question is: 1) can we change the size of KV cache? 2) how? 3) Anyway for us user to manage the KV cache issue more directly?

Thanks
George


### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Could not `pip install vllm` inside dockerfile after certain commit in `main` branch

**Link**: https://github.com/vllm-project/vllm/issues/9226
**State**: closed
**Created**: 2024-10-10T05:47:12+00:00
**Closed**: 2024-10-11T19:57:40+00:00
**Comments**: 6
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.4.0
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.26.4
Libc version: glibc-2.35

Python version: 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.25-051525-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA GeForce RTX 4090
GPU 1: NVIDIA GeForce RTX 4090

Nvidia driver version: 545.23.08
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                    x86_64
CPU op-mode(s)

[... truncated for brevity ...]

---

## Issue #N/A: H800 support

**Link**: https://github.com/vllm-project/vllm/issues/1143
**State**: closed
**Created**: 2023-09-22T09:30:51+00:00
**Closed**: 2024-03-08T12:24:17+00:00
**Comments**: 2

### Description

does vllm compatible with the gpu of H800

---

## Issue #N/A: [Usage]: The results of the model depend on the number of GPUs.

**Link**: https://github.com/vllm-project/vllm/issues/7645
**State**: closed
**Created**: 2024-08-19T05:08:51+00:00
**Closed**: 2024-12-19T02:04:30+00:00
**Comments**: 4
**Labels**: usage, stale

### Description

### Your current environment

**package version**
- vllm: 0.2.6
- python: 3.9

**Phenomenon**
- The output of the same model is different when using only one gpu and two.
- the same environment
- the same arguments

### How would you like to use vllm

I would like to get the same result even if the number of gpu is used differently when referring using vLLM.
The code I used is as follows.

```python
payload = json.dumps({
  "model_name": model_name,
  "prompt": output_ex['model_input'],
  "max_tokens": 1024,
  "stream": False,
  "top_p": 0.9,
  "temperature": 0.01,
  "presence_penalty": 1.0,
  "stop_token_ids": [2],
  "best_of": 1
})

headers = {
'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)
```
<br><br>

- when using only one gpu
  - output: "" (=empty)
- when using twon gpu
  - output: "No answer found."

<br><br>

Has anyone experienced a phenomenon like me where the output comes ou

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Speculative Decoding + FlashInfer + benchmark_serving.py TransferEncodingError ISSUE

**Link**: https://github.com/vllm-project/vllm/issues/6885
**State**: closed
**Created**: 2024-07-29T04:41:40+00:00
**Closed**: 2024-08-05T15:05:06+00:00
**Comments**: 3
**Labels**: bug

### Description

### Your current environment

Collecting environment information...
PyTorch version: 2.3.1+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.1
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-72-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.5.40
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A100-SXM4-80GB
Nvidia driver version: 535.183.01
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.1.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.1.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9.1.0
/usr/lib/x86_64-linux-gnu/libcudnn_engines_precompiled.so.9.1.0
/usr/lib/x86_64-linux-gnu/libcudnn_engin

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: vLLM DP=2 didn't speed up the training as low batch size.

**Link**: https://github.com/vllm-project/vllm/issues/17129
**State**: closed
**Created**: 2025-04-24T17:54:11+00:00
**Closed**: 2025-04-24T19:02:17+00:00
**Comments**: 5
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Hi team,

First of all, thanks for the recent efforts, especially to @qgallouedec, for supporting the new `data_parallel_size` feature in vLLM. I tested the `vllm-serve-dp` branch with `data_parallel_size=2`, and confirmed that it launches two processes for rollouts as expected. Great work!

However, **the speedup of having `data_parallel_size=2` isn't quite as significant as I hoped**. In my previous setup using a single GPU, I was achieving around 4000–6000 toks/s generation. With `data_parallel_size=2`, this drops to only 1000–1200 tokens/s per process, which results in an overall slower or comparable throughput.

It seems that the GPUs may be underutilized, possibly waiting on input (prompts/questions) to arrive. I suspect the issue could be mitigated by **allowing larger batch sizes for generation in the vLLM server, while keeping a smaller batch size for gradient calculations to avoid OOM errors.**

I’m reporting this primarily for visibil

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: AMD Instinct MI210 + vllm fail to run deepseek-r1-awq model, any solutions please? Is there any other deepseek-r1-671b models that can run succesfully on AMD Instinct MI210 + vllm? Thanks!

**Link**: https://github.com/vllm-project/vllm/issues/16386
**State**: open
**Created**: 2025-04-10T03:42:15+00:00
**Comments**: 1
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 04-10 03:25:29 [__init__.py:207] Automatically detected platform rocm.
Collecting environment information...
PyTorch version: 2.7.0a0+git6c0e746
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.3.42133-1b9c17779

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.1 24491 1e0fda770a2079fbd71e4b70974d74f62fd3af10)
CMake version: version 3.31.4
Libc version: glibc-2.35

Python version: 3.12.9 (main, Feb  5 2025, 08:49:00) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-136-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Instinct MI210 (gfx90a:sramecc+:xnack-)
Nvidia driver version: Could not collect
cuDNN ver

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Failed to run docker vllm-cpu-env arm docker on MacOS

**Link**: https://github.com/vllm-project/vllm/issues/11266
**State**: closed
**Created**: 2024-12-17T18:51:42+00:00
**Closed**: 2024-12-18T16:05:04+00:00
**Comments**: 4
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

After building Docker Images with [Dockerfile.arm](https://docs.vllm.ai/en/latest/getting_started/arm-installation.html), it built successfully but when attempts to run `docker run -it \
             --rm \
             --network=host \
             vllm-cpu-env --device="cpu" --disable_async_output_proc --enforce-eager --model=Qwen/Qwen2.5-1.5B-Instruct --dtype=float16`. it gets error in :
`File "/usr/local/lib/python3.10/dist-packages/vllm/utils.py", line 1639, in resolve_obj_by_qualname
    module_name, obj_name = qualname.rsplit(".", 1)
`
I am running on MacStudio Ultra and env is collected by building `Dockerfile.arm` file by executing `docker build -f Dockerfile.arm -t vllm-cpu-env --shm-size=4g .`

### Before submitting a new issue...

- 

[... truncated for brevity ...]

---

## Issue #N/A: [New Model]: CohereForCausalLM Support request 

**Link**: https://github.com/vllm-project/vllm/issues/3546
**State**: closed
**Created**: 2024-03-21T06:11:59+00:00
**Closed**: 2024-03-21T06:17:40+00:00
**Comments**: 1
**Labels**: new-model

### Description

### The model to consider.

ValueError: Model architectures ['CohereForCausalLM'] are not supported for now

![1711001492046](https://github.com/vllm-project/vllm/assets/7098003/c9f431ec-bb69-4903-9bdd-0073fef03ade)


### The closest model vllm already supports.

_No response_

### What's your difficulty of supporting the model you want?

_No response_

---

## Issue #N/A: Running multiple cards in parallel is slower(nearly twice) than a single card

**Link**: https://github.com/vllm-project/vllm/issues/935
**State**: closed
**Created**: 2023-09-03T02:40:06+00:00
**Closed**: 2024-03-08T11:13:14+00:00
**Comments**: 2

### Description

Hi, when I am running four A100 with parameter tensor_parallel_size is 4 in parallel, I found that the speed  is slower(nearly twice) than a single card. can you explain what causes this and how to solve it. Thank you.

---

## Issue #N/A: [Bug]: Model architectures ['LlavaForConditionalGeneration'] are not supported for now.

**Link**: https://github.com/vllm-project/vllm/issues/9377
**State**: closed
**Created**: 2024-10-15T15:26:44+00:00
**Closed**: 2024-10-16T08:47:29+00:00
**Comments**: 11
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
WARNING 10-15 15:24:09 cuda.py:22] You are using a deprecated `pynvml` package. Please install `nvidia-ml-py` instead, and make sure to uninstall `pynvml`. When both of them are installed, `pynvml` will take precedence and cause errors. See https://pypi.org/project/pynvml for more information.
Warning: Your installation of OpenCV appears to be broken: module 'cv2.dnn' has no attribute 'DictValue'.Please follow the instructions at https://github.com/opencv/opencv-python/issues/884 to correct your environment. The import of cv2 has been skipped.
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.2
Libc version: glibc-2.35

Python version: 3.10.12 (main, Jul 29 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Qwen2.5-Math-RM-72B Online Inference Fails

**Link**: https://github.com/vllm-project/vllm/issues/11446
**State**: closed
**Created**: 2024-12-24T03:05:09+00:00
**Closed**: 2024-12-24T09:54:31+00:00
**Comments**: 5
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

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: 14.0.0-1ubuntu1.1
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov  6 2024, 20:22:13) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.5.0-1025-gcp-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
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

Nvidia driver version: 550.90.07
c

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Persistent Errors with vllm serve on Neuron Device: Model architectures ['LlamaForCausalLM'] failed to be inspected. 

**Link**: https://github.com/vllm-project/vllm/issues/10932
**State**: closed
**Created**: 2024-12-05T18:49:34+00:00
**Closed**: 2024-12-09T21:53:25+00:00
**Comments**: 7
**Labels**: usage

### Description

### Your current environment

Hello vLLM Development Team,
I am encountering persistent issues when trying to run the ```vllm serve``` command for the ```meta-llama/Llama-3.2-1B``` model on an AWS EC2 inf2 instance with the Neuron AMI. Despite following all the recommended installation and upgrade steps, and adjusting the numpy versions as per the guidelines, the issue persists.

I already referred the issues I could find such as:

https://github.com/vllm-project/vllm/issues/9624
https://github.com/vllm-project/vllm/issues/9713
https://github.com/vllm-project/vllm/issues/9624

Here is the way I installed the vllm under the instruction guideline through:
[](https://docs.vllm.ai/en/latest/getting_started/neuron-installation.html)
<img width="1145" alt="image" src="https://github.com/user-attachments/assets/66a1bb4b-31f7-44b7-a14a-9b69bd2e719a">

I already tried to reinstall or upgrade the vllm under the instruction above many times, also tried to set the numpy versions. Stil

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: layer-wise kv cache offloading to enable larger batches

**Link**: https://github.com/vllm-project/vllm/issues/15123
**State**: open
**Created**: 2025-03-19T11:04:38+00:00
**Comments**: 3
**Labels**: RFC, stale

### Description

### Motivation.

I tested on some large models like qwen-32B on H100.

There are totally 64 layers. 

The compute cost for each layer is about 470 μs, and the transfer of the kv cache tensor for a layer is 10 ms.

If we offload the gpu kv cache to cpu, and load it back ahead of 32 layers, we can enable double batches.

Is there anyone doing the same thing? 

I draw a picture with 6 layers and 2 blocks share the same gpu cache.
![Image](https://github.com/user-attachments/assets/0f871ced-0b0c-4b05-be48-8bfce2a619c9)

### Proposed Change.

The kv cache manager and the attention layer

### Feedback Period.

_No response_

### CC List.

_No response_

### Any Other Things.

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: new bug after loosening type check on `llava_onevision.py`

**Link**: https://github.com/vllm-project/vllm/issues/15078
**State**: closed
**Created**: 2025-03-19T03:19:23+00:00
**Closed**: 2025-03-20T11:24:46+00:00
**Comments**: 24
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.5.1+cu118
Is debug build: False
CUDA used to build PyTorch: 11.8
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
Clang version: Could not collect
CMake version: version 3.16.3
Libc version: glibc-2.31

Python version: 3.12.9 | packaged by conda-forge | (main, Mar  4 2025, 22:48:41) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-5.4.0-146-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.0.140
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA GeForce RTX 4090
GPU 1: NVIDIA GeForce RTX 4090
GPU 2: NVIDIA GeForce RTX 4090
GPU 3: NVIDIA GeForce RTX 4090
GPU 4: NVIDIA GeForce RTX 4090
GPU 5: NVIDIA GeForce RTX 4090
GPU 6: NVIDIA GeForce RTX 4090
GPU 7: NVIDIA GeForce RTX 4090

Nvidia driver version: 52

[... truncated for brevity ...]

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

## Issue #N/A: [Performance]:

**Link**: https://github.com/vllm-project/vllm/issues/20368
**State**: closed
**Created**: 2025-07-02T10:33:39+00:00
**Closed**: 2025-07-04T09:50:43+00:00
**Comments**: 0
**Labels**: performance

### Description

### Proposal to improve performance

![Image](https://github.com/user-attachments/assets/7c7d91d1-cf2f-4f91-bee7-1fdfcbd0c15b)
I've encountered a phenomenon where when running the DeepSeek V2 Lite Chat MoE model with vLLM's v1 engine, increasing the number of activated experts boosts prefill processing speed by 100%-300%, while decoder speed remains unchanged. Since more activated experts increase computational parameters, speed should decrease – yet it improves. What causes this? I'm not using Expert Parallelism (EP). What mechanisms in vLLM's MoE handling could explain this behavior?

### Report of performance regression

_No response_

### Misc discussion on performance

_No response_

### Your current environment (if you think it is necessary)

```text
The output of `python collect_env.py`
```


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://d

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: In multimodal inference, is it possible to cache textual content and only load images each time to optimize inference efficiency

**Link**: https://github.com/vllm-project/vllm/issues/15608
**State**: open
**Created**: 2025-03-27T08:30:23+00:00
**Comments**: 2
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

In multimodal inference, is it possible to cache textual content and only load images each time to optimize inference efficiency

### Alternatives

_No response_

### Additional context

In multimodal inference, is it possible to cache textual content and only load images each time to optimize inference efficiency

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: Curl failed: Received HTTP/0.9 when not allowed

**Link**: https://github.com/vllm-project/vllm/issues/20119
**State**: open
**Created**: 2025-06-26T08:37:53+00:00
**Comments**: 3
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

**Issue Description:**
In a data-parallel scenario, when the --data-parallel-rpc-port argument value conflicts with the vLLM HTTP service port, the service starts without any error logs despite the port collision.

**Actual Behavior:**
When clients send requests, they only receive the error message: "Received HTTP/0.9 when not allowed".

**Expected Behavior:**
The vLLM service should either:

Detect the port conflict during startup and fail with a clear error message, or
Dynamically handle the conflict by selecting an alternative port.

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently aske

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: AttributeError: 'Llama_Nemotron_Nano_VL_Config' object has no attribute 'hidden_size'. Did you mean: 'vit_hidden_size'?

**Link**: https://github.com/vllm-project/vllm/issues/19360
**State**: open
**Created**: 2025-06-09T11:41:59+00:00
**Comments**: 4
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
OS                           : Debian GNU/Linux 12 (bookworm) (x86_64)
GCC version                  : (Debian 12.2.0-14) 12.2.0
Clang version                : Could not collect
CMake version                : Could not collect
Libc version                 : glibc-2.36

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
Python version               : 3.12.6 (main, Sep 27 2024, 06:10:12) [GCC 12.2.0] (64-bit runtime)
Python platform              : Linux-4.4.0-x86_64-with-glibc2.36

=========

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Does vLLM support co-hosting multiple models on single server?

**Link**: https://github.com/vllm-project/vllm/issues/11822
**State**: closed
**Created**: 2025-01-08T00:41:24+00:00
**Closed**: 2025-01-21T19:34:21+00:00
**Comments**: 2
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

## Issue #N/A: [Doc]: Supported Hardware for Quantization Kernels

**Link**: https://github.com/vllm-project/vllm/issues/6979
**State**: closed
**Created**: 2024-07-31T07:16:05+00:00
**Closed**: 2024-07-31T15:33:53+00:00
**Comments**: 1
**Labels**: documentation

### Description

### 📚 The doc issue

I'm confused what "the quantization method is supported" mean?  Ampere arch doesn't support FP8, according to Nvidia. So does this mean the FP8 operation is supported on A100/A800 GPU?  Or just we can conver the weight parameters form FP16 to FP8? 

### Suggest a potential alternative/fix

_No response_

---

