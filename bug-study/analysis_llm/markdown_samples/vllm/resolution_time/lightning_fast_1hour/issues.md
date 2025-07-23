# lightning_fast_1hour - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- bug: 13 issues
- usage: 7 issues
- feature request: 3 issues
- documentation: 2 issues

---

## Issue #N/A: [Bug]: OpenAI API Completions and Chat API inconsistency

**Link**: https://github.com/vllm-project/vllm/issues/6699
**State**: closed
**Created**: 2024-07-23T18:18:16+00:00
**Closed**: 2024-07-23T18:35:41+00:00
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

Python version: 3.10.12 (main, Mar 22 2024, 16:50:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-113-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 4090
Nvidia driver version: 550.90.07
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Address sizes:                      46 bits physical, 48 bits virtual
By

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Set dtype for VLLM using YAML

**Link**: https://github.com/vllm-project/vllm/issues/3503
**State**: closed
**Created**: 2024-03-19T17:55:06+00:00
**Closed**: 2024-03-19T18:04:06+00:00
**Comments**: 3
**Labels**: usage

### Description

### Your current environment

Collecting environment information...
PyTorch version: 2.2.1+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.19.0-1010-nvidia-lowlatency-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA GeForce RTX 2080 Ti
GPU 1: NVIDIA GeForce RTX 2080 Ti
GPU 2: NVIDIA GeForce RTX 2080 Ti
GPU 3: NVIDIA GeForce RTX 2080 Ti
GPU 4: NVIDIA GeForce RTX 2080 Ti
GPU 5: NVIDIA GeForce RTX 2080 Ti
GPU 6: NVIDIA GeForce RTX 2080 Ti
GPU 7: NVIDIA GeForce RTX 2080 Ti

Nvidia driver version: 550.54.14
cuDNN version: Co

[... truncated for brevity ...]

---

## Issue #N/A: `RuntimeError: Cannot re-initialize CUDA in forked subprocess` when initializing vLLM with tensor parallelism

**Link**: https://github.com/vllm-project/vllm/issues/14535
**State**: closed
**Created**: 2025-03-10T03:36:50+00:00
**Closed**: 2025-03-10T03:43:14+00:00
**Comments**: 4
**Labels**: bug

### Description

### Description
When attempting to initialize a vLLM instance with a custom model and tensor parallelism, I encounter a `RuntimeError: Cannot re-initialize CUDA in forked subprocess` error across multiple worker processes. The error suggests that CUDA cannot be re-initialized in a forked subprocess and recommends using the `'spawn'` start method for multiprocessing. This issue prevents the model from loading successfully.

The error occurs consistently when running the provided code snippet on a system with CUDA-enabled GPUs and tensor parallelism set to 8.

### Steps to Reproduce
1. Install the dependencies as listed in the [environment](#environment) section.
2. Run the following Python code:
   ```python
   import vllm
   model_name = '/QwQ-32B/'
   llm = vllm.LLM(model_name, tensor_parallel_size=8)
   ```
3. Observe the error output in the logs.

### Expected Behavior
The vLLM instance should initialize successfully, with all worker processes starting and the model loading without 

[... truncated for brevity ...]

---

## Issue #N/A: ValueError: Bfloat16 is only supported on GPUs with compute capability of at least 8.0. Your Tesla V100-SXM2-32GB GPU has compute capability 7.0.

**Link**: https://github.com/vllm-project/vllm/issues/946
**State**: closed
**Created**: 2023-09-05T01:22:12+00:00
**Closed**: 2023-09-05T01:39:42+00:00
**Comments**: 2

### Description

Does any one know about this issue?


<img width="1206" alt="Screenshot 2023-09-05 at 11 21 03 am" src="https://github.com/vllm-project/vllm/assets/29119972/a59c401a-399e-4c44-9a4c-72cb7fa24ded">


---

## Issue #N/A: [Usage]: How to replace vllm model's weights after engine is started up?

**Link**: https://github.com/vllm-project/vllm/issues/16586
**State**: closed
**Created**: 2025-04-14T10:45:48+00:00
**Closed**: 2025-04-14T11:39:39+00:00
**Comments**: 9
**Labels**: usage

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How would you like to use vllm

I want to run inference of a [specific model](put link here). I don't know how to integrate it with vllm.


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: load Qwen2.5-14B-1M error

**Link**: https://github.com/vllm-project/vllm/issues/19408
**State**: closed
**Created**: 2025-06-10T08:46:23+00:00
**Closed**: 2025-06-10T08:58:08+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

torch                                    2.6.0
torchaudio                               2.6.0
torchvision                              0.21.0
tqdm                                     4.67.1
transformers                             4.51.3
triton                                   3.2.0
typer                                    0.16.0
typing_extensions                        4.14.0
typing-inspection                        0.4.1
urllib3                                  2.4.0
uvicorn                                  0.34.3
uvloop                                   0.21.0
virtualenv                               20.31.2
vllm                                     0.8.5.post1
watchfiles                               1.0.5
websockets                               15.0.1
wheel                                    0.45.1
wrapt                                    1.17.2
xformers                   

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: What are the correct parameters for offline beam search inference in vllm ?

**Link**: https://github.com/vllm-project/vllm/issues/11297
**State**: closed
**Created**: 2024-12-18T10:16:51+00:00
**Closed**: 2024-12-18T10:43:59+00:00
**Comments**: 3
**Labels**: usage

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How would you like to use vllm

As beam search api was changed recently and `use_beam_search` was removed from SamplingParams, I'm not sure which is the way to trigger beam search (without sampling) in vllm offline inference. Currently, I'm adopting the following codes:
```
from vllm import LLM, SamplingParams
beam_size=4
sampling_params = SamplingParams(temperature=1.0,n=1,top_k=-1,top_p=1,seed=42,max_tokens=128,best_of=beam_size)
outputs = vllm_model.chat(messages=prompts, sampling_params=sampling_params, use_tqdm=True)
```

Is this the correct way to trigger beam search ?

vllm version: v0.6.4.post1

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: When performing inference with vLLM, it keeps getting stuck at 0%.

**Link**: https://github.com/vllm-project/vllm/issues/16301
**State**: closed
**Created**: 2025-04-09T02:27:49+00:00
**Closed**: 2025-04-09T02:28:51+00:00
**Comments**: 0
**Labels**: usage

### Description

### Your current environment

NFO 04-09 02:05:52 [loader.py:447] Loading weights took 33.15 seconds
INFO 04-09 02:05:53 [gpu_model_runner.py:1273] Model loading took 61.0374 GiB and 33.525153 seconds
INFO 04-09 02:06:09 [backends.py:416] Using cache directory: /root/.cache/vllm/torch_compile_cache/e8c79b34a0/rank_0_0 for vLLM's torch.compile
INFO 04-09 02:06:09 [backends.py:426] Dynamo bytecode transform time: 16.61 s
INFO 04-09 02:06:10 [backends.py:115] Directly load the compiled graph for shape None from the cache
INFO 04-09 02:06:25 [monitor.py:33] torch.compile takes 16.61 s in total
INFO 04-09 02:06:28 [kv_cache_utils.py:578] GPU KV cache size: 16,064 tokens
INFO 04-09 02:06:28 [kv_cache_utils.py:581] Maximum concurrency for 10,000 tokens per request: 1.61x
INFO 04-09 02:07:01 [gpu_model_runner.py:1608] Graph capturing finished in 33 secs, took 3.17 GiB
INFO 04-09 02:07:01 [core.py:162] init engine (profile, create kv cache, warmup model) took 68.69 seconds
Processed prompts:   0

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: 

**Link**: https://github.com/vllm-project/vllm/issues/3485
**State**: closed
**Created**: 2024-03-19T01:44:56+00:00
**Closed**: 2024-03-19T01:48:44+00:00
**Comments**: 1
**Labels**: bug

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### 🐛 Describe the bug

python -m vllm.entrypoints.openai.api_server --model /root/autodl-tmp/models/Qwen1.5-14B-Chat-GPTQ-Int8 --max-model-len 2000  --trust-remote-code

when request api, some errors return:

{"object":"error","message":"The model `Qwen1.5-14B-Chat-GPTQ-Int8` does not exist.","type":"NotFoundError","param":null,"code":404}

---

## Issue #N/A: [Feature]: Support Int8 dtype for storing weights - currently uses FP16 wasting 50% of VRAM

**Link**: https://github.com/vllm-project/vllm/issues/4031
**State**: closed
**Created**: 2024-04-12T07:56:27+00:00
**Closed**: 2024-04-12T08:15:30+00:00
**Comments**: 1
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Could you please add Int8 as a supported dtype? Currently when using Int8 models such as https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GPTQ-Int8 with xformers instead of FlashAttention, the weights are stored as FP16 taking double the VRAM.

### Alternatives

_No response_

### Additional context

_No response_

---

## Issue #N/A: [Bug]: Gemma model is giving empty responses with new version of docker image vllm-openai:v.8.5

**Link**: https://github.com/vllm-project/vllm/issues/17718
**State**: closed
**Created**: 2025-05-06T13:12:44+00:00
**Closed**: 2025-05-06T13:58:35+00:00
**Comments**: 2
**Labels**: bug

### Description

### Current environment

Kubernetes Cluster on Azure with A100 GPUs

### Bug

Hello team,

After upgrading the Docker image from vllm-openai:v0.8.4 to v0.8.5, I observed one issue when running the google/gemma-3-27b-it model ([Hugging Face Model Link](https://huggingface.co/google/gemma-3-27b-it)).

The model successfully returns metadata (e.g., finish reason, token usage), but the content field in the response is consistently an empty string. No changes were made to the Kubernetes deployment manifest apart from the image version bump.

When reverting to v0.8.4, the model responds correctly with expected text completions, confirming that the issue is specific to the new image version.

Steps to Reproduce:

1. Deploy vllm-openai:v0.8.5 with the gemma-3-27b-it model.

2. Send a chat completion request.

3. Observe that the content field is empty in the response.


---

## Issue #N/A: [Bug]: With docker "ValueError: No supported config format found in"

**Link**: https://github.com/vllm-project/vllm/issues/13451
**State**: closed
**Created**: 2025-02-18T03:23:50+00:00
**Closed**: 2025-02-18T04:16:53+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

Official latest docker image

### 🐛 Describe the bug


I run a docker container with the commande line:

"docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-v0.1"

A few hours ago, everything was working fine. However, when I try to run a model using the Docker image now, I encounter the following error:

"ValueError: No supported config format found in"

Below, you’ll find the full error message that appears when I run the command provided on the vLLM website.


docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-v0.1
INFO 02-17 19:14:31 __init__.py:190] Automatically detected platfo

[... truncated for brevity ...]

---

## Issue #N/A: flash_attn is installed, but "flash_attn is not found. Using xformers backend."

**Link**: https://github.com/vllm-project/vllm/issues/3306
**State**: closed
**Created**: 2024-03-11T02:46:04+00:00
**Closed**: 2024-03-11T03:42:36+00:00
**Comments**: 3

### Description

```bash  
 >>> flash_attn is not found. Using xformers backend.
```
but flash_attn has been added into the vllm wheel 
```bash 
adding 'vllm/thirdparty_files/flash_attn/ops/triton/rotary.py'
adding 'vllm/thirdparty_files/flash_attn/ops/triton/__pycache__/__init__.cpython-310.pyc'
adding 'vllm/thirdparty_files/flash_attn/ops/triton/__pycache__/cross_entropy.cpython-310.pyc'
adding 'vllm/thirdparty_files/flash_attn/ops/triton/__pycache__/k_activations.cpython-310.pyc'
adding 'vllm/thirdparty_files/flash_attn/ops/triton/__pycache__/layer_norm.cpython-310.pyc'
adding 'vllm/thirdparty_files/flash_attn/ops/triton/__pycache__/linear.cpython-310.pyc'
adding 'vllm/thirdparty_files/flash_attn/ops/triton/__pycache__/mlp.cpython-310.pyc'
adding 'vllm/thirdparty_files/flash_attn/ops/triton/__pycache__/rotary.cpython-310.pyc'
adding 'vllm/thirdparty_files/flash_attn/utils/__init__.py'
adding 'vllm/thirdparty_files/flash_attn/utils/benchmark.py'
adding 'vllm/thirdparty_files/flash_attn

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Issues with Applying LoRA in vllm on a T4 GPU

**Link**: https://github.com/vllm-project/vllm/issues/5198
**State**: closed
**Created**: 2024-06-02T16:03:20+00:00
**Closed**: 2024-06-02T16:08:39+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

I am currently using a T4 instance on Google Colaboratory.

```text
Collecting environment information...
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: 14.0.0-1ubuntu1.1
CMake version: version 3.27.9
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.1.85+-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.2.140
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: Tesla T4
Nvidia driver version: 535.104.05
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.6
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.9.6
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.9.6
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.s

[... truncated for brevity ...]

---

## Issue #N/A: Does the Mixtral implementation follow the official code?

**Link**: https://github.com/vllm-project/vllm/issues/2023
**State**: closed
**Created**: 2023-12-11T16:03:24+00:00
**Closed**: 2023-12-11T16:25:46+00:00
**Comments**: 1

### Description

Hi,
Does your Mixtral implementation follow the newly released official Mixtral code?

https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
Thanks for creating this great project!

---

## Issue #N/A: TypeError: 'builtins.safe_open' object is not iterable

**Link**: https://github.com/vllm-project/vllm/issues/1733
**State**: closed
**Created**: 2023-11-20T23:56:44+00:00
**Closed**: 2023-11-21T00:01:16+00:00
**Comments**: 1

### Description

```
/vllm/vllm/model_executor/weight_utils.py", line 243, in hf_model_weights_iterator
    for name in f:
TypeError: 'builtins.safe_open' object is not iterable
```

I am using the main branch build.
Triggers when using /openai/api_server.py and loading the model weights for the engine object (AsyncLLMEngine)

Traceback:

```
Traceback (most recent call last):
  File "runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "api_server.py", line 644, in <module>
    engine = AsyncLLMEngine.from_engine_args(engine_args)
  File "async_llm_engine.py", line 486, in from_engine_args
    engine = cls(parallel_config.worker_use_ray,
  File "async_llm_engine.py", line 269, in __init__
    self.engine = self._init_engine(*args, **kwargs)
  File "async_llm_engine.py", line 305, in _init_engine
    return engine_class(*args, **kwargs)
  File "llm_engine.py", line 

[... truncated for brevity ...]

---

## Issue #N/A: [Doc]: vllm部署Gemma-3-27b问题

**Link**: https://github.com/vllm-project/vllm/issues/16378
**State**: closed
**Created**: 2025-04-10T02:34:32+00:00
**Closed**: 2025-04-10T02:40:24+00:00
**Comments**: 1
**Labels**: documentation

### Description

### 📚 The doc issue

我的启动命令是
(torch) root@vvbbovkdctrbyrag-wind-b6df56d5-r96vk:/data/coding# vllm serve Gemma-3-27b --tensor-parallel-size 4 --max-model-len 65536 --max-num-batched-tokens 16384 --port 30041  --trust-remote-code  --served-model-name gemma3-27b  --max-num-seqs 64 --enable-chunked-prefill --limit-mm-per-prompt image=50,video=2 --api-key k7YgF9RwP4qXmTnV2LsJ3HdO5zIc6AeB0Uv1lKpN8Q
INFO 04-10 08:57:28 [__init__.py:256] Automatically detected platform cuda.
INFO 04-10 08:57:30 [api_server.py:972] vLLM API server version 0.8.0rc3.dev5+g5eeabc2a.d20250318
INFO 04-10 08:57:30 [api_server.py:973] args: Namespace(subparser='serve', model_tag='Gemma-3-27b', config='', host=None, port=30041, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key='k7YgF9RwP4qXmTnV2LsJ3HdO5zIc6AeB0Uv1lKpN8Q', lora_modules=None, prompt_adapters=None, chat_template=None, chat_template_content_format='auto', response_role='assistant

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Kernel died while waiting for execute reply in Kaggle TPU VM v3-8 (2024-08-22)

**Link**: https://github.com/vllm-project/vllm/issues/8352
**State**: closed
**Created**: 2024-09-11T04:50:40+00:00
**Closed**: 2024-09-11T05:05:10+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
2024-09-11 04:42:09.993145: F external/local_xla/xla/stream_executor/tpu/tpu_executor_init_fns.inc:25] TpuExecutor_AllocateStream not available in this library.
bash: line 2:  1424 Aborted                 (core dumped) python collect_env.py
---------------------------------------------------------------------------
CalledProcessError                        Traceback (most recent call last)
Cell In[6], line 1
----> 1 get_ipython().run_cell_magic('bash', '', '#wget [https://raw.githubusercontent.com/vllm-project/vllm/main/collect_env.py](https://raw.githubusercontent.com/vllm-project/vllm/main/collect_env.py%3C/span%3E%3Cspan) class="ansi-yellow-bg ansi-bold" style="color:rgb(175,95,0)">\npython collect_env.py\n')

File /usr/local/lib/python3.10/site-packages/IPython/core/interactiveshell.py:2541, in InteractiveShell.run_cell_magic(self, magic_name, line, cell)
   2539 wi

[... truncated for brevity ...]

---

## Issue #N/A: ImportError: libcudart.so.12: cannot open shared object file: No such file or directory

**Link**: https://github.com/vllm-project/vllm/issues/1716
**State**: closed
**Created**: 2023-11-19T07:18:47+00:00
**Closed**: 2023-11-19T07:48:59+00:00
**Comments**: 3

### Description

My code work well yesterday but now it is not working today since the latest update (v.0.2.2)!
My code:
```python
from langchain.llms import VLLM
llm = VLLM(
    model=GENERATE_MODEL_NAME,
    trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=max_new_tokens,
    top_k=10,
    top_p=0.95,
    temperature=0.8,
    # dtype="half",
    vllm_kwargs={"quantization": "awq"}
)
```
The error:
```
ImportError: libcudart.so.12: cannot open shared object file: No such file or directory

During handling of the above exception, another exception occurred:

ImportError                               Traceback (most recent call last)
[/usr/local/lib/python3.10/dist-packages/langchain/llms/vllm.py](https://localhost:8080/#) in validate_environment(cls, values)
     79             from vllm import LLM as VLLModel
     80         except ImportError:
---> 81             raise ImportError(
     82                 "Could not import vllm python package. "
     83

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: support for Llama 4 family models

**Link**: https://github.com/vllm-project/vllm/issues/16345
**State**: closed
**Created**: 2025-04-09T13:10:16+00:00
**Closed**: 2025-04-09T13:12:47+00:00
**Comments**: 1
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Llama 4 is out there. hence, please add support for llama 4 models too

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: Support Qwen2.5-VL-7B Instruct

**Link**: https://github.com/vllm-project/vllm/issues/12825
**State**: closed
**Created**: 2025-02-06T12:01:41+00:00
**Closed**: 2025-02-06T12:27:21+00:00
**Comments**: 2
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Qwen2.5-VL-7B Instruct

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: Local model returns empty text with vLLM despite matching upstream HF files

**Link**: https://github.com/vllm-project/vllm/issues/20685
**State**: closed
**Created**: 2025-07-09T13:16:14+00:00
**Closed**: 2025-07-09T13:49:58+00:00
**Comments**: 2
**Labels**: usage

### Description

### Your current environment

```text
INFO 07-09 20:53:46 [__init__.py:240] Automatically detected platform rocm.
Collecting environment information...
/usr/local/lib/python3.10/dist-packages/_distutils_hack/__init__.py:30: UserWarning: Setuptools is replacing distutils. Support for replacing an already imported distutils is deprecated. In the future, this condition will fail. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
PyTorch version: 2.4.1
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.3.25211

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: 15.0.0
CMake version: version 3.29.0
Libc version: glibc-2.35

Python version: 3.10.12 (main, May 27 2025, 17:12:29) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-4.18.0-348.el8.x86_64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODU

[... truncated for brevity ...]

---

## Issue #N/A: [Doc]: APIConnectionError with OpenAI

**Link**: https://github.com/vllm-project/vllm/issues/15518
**State**: closed
**Created**: 2025-03-26T03:04:55+00:00
**Closed**: 2025-03-26T03:36:10+00:00
**Comments**: 1
**Labels**: documentation

### Description

### 📚 The doc issue

Hello,

While going through the documentation's QuickStart section (https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server), I haven't been able to resolve the `APIConnectionError: Connection error.` with the OpenAI API.

The very same code in the documentation, except the `openai_api_key` where I input my API key:
```
import os
os.environ['HF_HOME'] = '~/scratch/LLM/cache/'
import torch
torch.cuda.empty_cache()

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "my own API key I got from OpenAI"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
completion = client.completions.create(model="Qwen/Qwen2.5-1.5B-Instruct",
                                      prompt="San Francisco is a")
print("Completion result:", completion)
```
The error:
```
--------------------------------------------------------------

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Build error, nvcc error   : 'ptxas' died due to signal 11 (Invalid memory reference)

**Link**: https://github.com/vllm-project/vllm/issues/15452
**State**: closed
**Created**: 2025-03-25T08:58:06+00:00
**Closed**: 2025-03-25T09:40:51+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

L20, CUDA 12.6, PyTorch 2.6.0

### 🐛 Describe the bug

```bash
FAILED: vllm-flash-attn/CMakeFiles/_vllm_fa3_C.dir/hopper/instantiations/flash_fwd_hdimall_fp16_paged_split_sm90.cu.o
/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler -DFLASHATTENTION_DISABLE_BACKWARD -DFLASHATTENTION_DISABLE_DROPOUT -DFLASHATTENTION_DISABLE_PYBIND -DFLASHATTENTION_DISABLE_UNEVEN_K -DFLASHATTENTION_VARLEN_ONLY -DPy_LIMITED_API=3 -DTORCH_EXTENSION_NAME=_vllm_fa3_C -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE -D_vllm_fa3_C_EXPORTS -I/workspace/dev/vipshop/vllm/.deps/vllm-flash-attn-src/csrc -I/workspace/dev/vipshop/vllm/.deps/vllm-flash-attn-src/hopper -I/workspace/dev/vipshop/vllm/.deps/vllm-flash-attn-src/csrc/common -I/workspace/dev/vipshop/vllm/.deps/vllm-flash-attn-src/csrc/cutlass/include -isystem /usr/include/python3.12 -isystem /usr/local/lib/python3.12/dist-packages/torch/include -isystem /usr/local/lib/python3.12/dist-packages/t

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Does vllm support inflight batch?

**Link**: https://github.com/vllm-project/vllm/issues/14536
**State**: closed
**Created**: 2025-03-10T03:45:47+00:00
**Closed**: 2025-03-10T03:52:12+00:00
**Comments**: 1
**Labels**: usage

### Description

### Your current environment


### How would you like to use vllm

Does vllm support inflight batch?
trtllm supports it but I can't find any information on vllm documentation
Could some kind person explain it?
Thank you so much in advance

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: ValueError: Model architectures ['Phi3VForCausalLM'] are not supported for now.

**Link**: https://github.com/vllm-project/vllm/issues/5864
**State**: closed
**Created**: 2024-06-26T11:05:33+00:00
**Closed**: 2024-06-26T11:25:59+00:00
**Comments**: 3
**Labels**: bug

### Description

### Your current environment

```text
Collecting environment information...
PyTorch version: N/A
Is debug build: N/A
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: Could not collect
CMake version: version 3.16.3
Libc version: glibc-2.31

Python version: 3.10.14 | packaged by conda-forge | (main, Mar 20 2024, 12:45:18) [GCC 12.3.0] (64-bit runtime)
Python platform: Linux-5.15.0-1063-aws-x86_64-with-glibc2.31
Is CUDA available: N/A
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: GPU 0: Tesla T4
Nvidia driver version: 535.183.01
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: N/A

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Byte Order:                         Little Endian
Address sizes:

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vllm run Qwen2-Audio-7B-Instruct raise openai.InternalServerError: Error code: 500

**Link**: https://github.com/vllm-project/vllm/issues/16525
**State**: closed
**Created**: 2025-04-12T04:16:29+00:00
**Closed**: 2025-04-12T04:37:19+00:00
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


### 🐛 Describe the bug

vllm==0.8.2
transformers==4.51.0

1、vllm serve：
VLLM_AUDIO_FETCH_TIMEOUT=360000 CUDA_VISIBLE_DEVICES=1 VLLM_LOGGING_LEVEL=DEBUG  vllm serve Qwen2-Audio-7B-Instruct --max-model-len 4096  --port 8000 --served-model-name qwen2-audio-7b-instruct

2、python code：
 openai_api_base="http://localhost:8000/v1"
AUDIO_FILE=”sample-9s.wav“
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


with open(AUDIO_FILE, "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode("utf-8")
    audio_data_url = f"data:audio/wav;base64,{audio_base64}"

chat_completion_from_base64 = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: how to use vllm serve start BertForSequenceClassification model

**Link**: https://github.com/vllm-project/vllm/issues/16176
**State**: closed
**Created**: 2025-04-07T09:26:14+00:00
**Closed**: 2025-04-07T10:01:49+00:00
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


### 🐛 Describe the bug

I trained an emotion classification model using BERTModel。Added a classification header to classify。when i start vllm serve ，What task type should I specify？
and how to curl the server?can provide a example

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: How to use vllm to run Qwen2-VL-72B?

**Link**: https://github.com/vllm-project/vllm/issues/10153
**State**: closed
**Created**: 2024-11-08T11:31:42+00:00
**Closed**: 2024-11-08T12:30:22+00:00
**Comments**: 5
**Labels**: usage

### Description

### Your current environment

```
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/path/Qwen2-VL-72B-Instruct"

llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10, "video": 10},
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=256,
    stop_token_ids=[],
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "porsche.jpg",
                "min_pixels": 224 * 224,
                "max_pixels": 1280 * 28 * 28,
            },
            {"type": "text", "text": "What is the text in the illustrate?"},
        ],
    },
]
# For video input, you can pass following values instead:
# "type": "video",
# "video": "<video URL>",

proc

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]:

**Link**: https://github.com/vllm-project/vllm/issues/18888
**State**: closed
**Created**: 2025-05-29T07:49:01+00:00
**Closed**: 2025-05-29T08:31:27+00:00
**Comments**: 1
**Labels**: bug

### Description

### Your current environment

FlashMLA V1 with FP8 KV cache not yet supported!

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NCCL_SOCKET_IFNAME=bond0 \
GLOO_SOCKET_IFNAME=bond0 \
VLLM_USE_V1=1 \
VLLM_USE_MODELSCOPE=true \
vllm serve /data/models/huggingface.co/deepseek-ai/DeepSeek-R1/DeepSeek-R1-Hzz1 \
--served-model-name deepseek-r1 \
--gpu-memory-utilization 0.8 \
--tensor-parallel-size 8  \
--trust-remote-code \
--enable-chunked-prefill \
--port 8000 \
--kv-cache-dtype fp8 \
--enable-expert-parallel
```

result:

![Image](https://github.com/user-attachments/assets/44ca8e77-1ce2-4b4c-b80b-9bda20e1ca35)

GPU DEVICE h800 x 8 x 140GB
cuda drive version 550.127.08
vllm version 0.8.5 

pip list 

```
Package                                  Version
---------------------------------------- --------------------
accelerate                               0.34.0
aiofiles                                 23.2.1
aiohappyeyeballs                         2.4.0
aiohttp                                  

[... truncated for brevity ...]

---

