# high_discussion_21to50 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 11
- Closed Issues: 19

### Label Distribution

- bug: 19 issues
- feature request: 3 issues
- installation: 3 issues
- unstale: 3 issues
- stale: 3 issues
- RFC: 1 issues
- new-model: 1 issues
- usage: 1 issues
- performance: 1 issues
- speculative-decoding: 1 issues

---

## Issue #N/A: [Feature]: Support for RTX 5090 (CUDA 12.8)

**Link**: https://github.com/vllm-project/vllm/issues/13306
**State**: closed
**Created**: 2025-02-14T20:41:08+00:00
**Closed**: 2025-06-26T17:16:55+00:00
**Comments**: 21
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Currently only nightlies from torch targeting 12.8 support blackwell such as the rtx 5090.
I tried using VLLM with a rtx 5090 and no dice. Vanilla vllm installation ends in:
```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
If you want to use the NVIDIA GeForce RTX 5090 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
Thanks

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: RuntimeError on RTX 5090: "no kernel image is available for execution on the device

**Link**: https://github.com/vllm-project/vllm/issues/16901
**State**: closed
**Created**: 2025-04-21T04:26:48+00:00
**Closed**: 2025-06-26T17:16:10+00:00
**Comments**: 28
**Labels**: installation

### Description

### Your current environment

### Describe the bug

When running 

[vLLM.log](https://github.com/user-attachments/files/19829093/vLLM.log)

 with a NVIDIA RTX 5090 GPU, I encountered the following error:

RuntimeError: CUDA error: no kernel image is available for execution on the device

From the logs, it seems that PyTorch does not support the compute capability of the RTX 5090 (sm_120):

### To Reproduce

1. Use RTX 5090 GPU
2. Install vLLM with Docker or system Python environment
3. Launch the vLLM OpenAI API server
4. Engine fails to start due to CUDA kernel compatibility issue

### Environment

- **GPU**: NVIDIA GeForce RTX 5090
- **CUDA Driver Version**: 12.8
- **CUDA Toolkit**: 12.8.93
- **NVIDIA Driver**: 570.124.06
- **PyTorch Version**: 2.x (installed via pip)
- **vLLM Version**: Latest (from PyPI)
- **Python Version**: 3.10
- **OS**: Ubuntu 22.04

### Additional Context

It seems that the RTX 5090 uses a new compute capability (`sm_120`), which is currently not supported in 

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: KV cache offloading

**Link**: https://github.com/vllm-project/vllm/issues/19854
**State**: open
**Created**: 2025-06-19T10:52:43+00:00
**Comments**: 21
**Labels**: RFC

### Description

### Motivation.

Currently, in vLLM v1 there is no in-house solution for offloading KV cache data from the GPU memory to other medium (in particular, CPU memory).
There is a proposed RFC (#16144) and respective PRs (#13377 and #17653) that try to address that.
The approach they take is somewhat similar to the way offloading was implemented in V0:
1. On the scheduler side, extend the core GPU allocator (KVCacheManager) to support CPU offloading
2. On the worker side, add a synchronous call to handle the actual CPU<->GPU transfer in the `execute_model` function.

In this RFC I propose an alternative approach which supports the following requirements:
* **Async saving** of new KV data from GPU to cache. GPU memory will not be freed until save is completed (similar to NixlConnector)
* **Async loading** of KV data from the cache to GPU. Requests waiting for cache load won't be scheduled until load is completed (similar to NixlConnector)
* Support **pluggable backends** for cache (CPU backen

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vLLM v0.6.1 Instability issue under load.

**Link**: https://github.com/vllm-project/vllm/issues/8219
**State**: closed
**Created**: 2024-09-06T00:36:46+00:00
**Closed**: 2024-09-13T14:58:53+00:00
**Comments**: 23
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.31

Python version: 3.10.14 (main, Apr  6 2024, 18:45:05) [GCC 9.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-25-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA H100 80GB HBM3
GPU 1: NVIDIA H100 80GB HBM3
GPU 2: NVIDIA H100 80GB HBM3
GPU 3: NVIDIA H100 80GB HBM3

Nvidia driver version: 535.86.10
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: 

[... truncated for brevity ...]

---

## Issue #N/A: Data parallel inference

**Link**: https://github.com/vllm-project/vllm/issues/1237
**State**: closed
**Created**: 2023-09-30T23:24:38+00:00
**Closed**: 2024-09-13T17:00:25+00:00
**Comments**: 29
**Labels**: feature request

### Description

Is there a recommended way to run data parallel inference (i.e. a copy of the model on each GPU)? It's possible by hacking CUDA_VISIBLE_DEVICES, but I was wondering if there's a cleaner method.
```python
def worker(worker_idx):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_idx)
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(model="facebook/opt-125m")
    outputs = llm.generate(prompts, sampling_params)


if __name__ == "__main__":
    
    with multiprocessing.Pool(4) as pool:
        pool.map(worker, range(4))
```

---

## Issue #N/A: Whisper support

**Link**: https://github.com/vllm-project/vllm/issues/180
**State**: closed
**Created**: 2023-06-21T07:06:07+00:00
**Closed**: 2025-01-03T08:39:21+00:00
**Comments**: 40
**Labels**: new-model

### Description

Is support for Whisper on the roadmap? Something like https://github.com/ggerganov/whisper.cpp would be great.

---

## Issue #N/A: [Bug]: Multi-modal inference too slow

**Link**: https://github.com/vllm-project/vllm/issues/16626
**State**: open
**Created**: 2025-04-15T02:58:23+00:00
**Comments**: 25
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 04-15 02:56:47 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: Could not collect
CMake version: version 3.16.3
Libc version: glibc-2.31

Python version: 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 13:27:36) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-1086-azure-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.6.85
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A100-SXM4-80GB
GPU 1: NVIDIA A100-SXM4-80GB
GPU 2: NVIDIA A100-SXM4-80GB
GPU 3: NVIDIA A100-SXM4-80GB
GPU 4: NVIDIA A100-SXM4-80GB
GPU 5: NVIDIA A100-SXM4-80GB
GPU 6: NVIDIA A100-SXM

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: [vllm-openvino]: ValueError: `use_cache` was set to `True` but the loaded model only supports `use_cache=False`. 

**Link**: https://github.com/vllm-project/vllm/issues/6473
**State**: closed
**Created**: 2024-07-16T12:05:32+00:00
**Closed**: 2024-07-19T02:04:07+00:00
**Comments**: 21
**Labels**: bug

### Description

### Your current environment

```text
The output of `python collect_env.py`

(vllm-openvino) yongshuai_wang@cpu-10-48-1-249:~/models$ python collect_env.py 
Collecting environment information...
WARNING 07-16 19:50:52 _custom_ops.py:14] Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
/home/yongshuai_wang/miniconda3/envs/vllm-openvino/lib/python3.10/site-packages/vllm/usage/usage_lib.py:19: RuntimeWarning: Failed to read commit hash:
No module named 'vllm.commit_id'
  from vllm.version import __version__ as VLLM_VERSION
PyTorch version: 2.3.1+cpu
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.0
Libc version: glibc-2.35

Python version: 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-94-generic-x86_64

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: Kimi-VL-A3B failed to be deployed using vllm mirroring

**Link**: https://github.com/vllm-project/vllm/issues/16715
**State**: closed
**Created**: 2025-04-16T10:11:58+00:00
**Closed**: 2025-04-17T12:53:32+00:00
**Comments**: 28
**Labels**: installation

### Description

### Your current environment

model：Kimi-VL-A3B-Thinking
image：vllm-openai：latest    
vllm version:0.8.4

1.docker pull vllm/vllm-openai
里面的vllm version:0.8.4
2.docker run --gpus all -v /mnt/data1/LargeLanguageModels/qwen:/model --ipc=host --network=host  --name kimi-vl -it --entrypoint vllm/vllm-openai ：latest  bash
3. 在容器中如下命令启动大模型

> CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \

    --port 3000 \
    --served-model-name kimi-vl \
    --trust-remote-code \
    --model /models/Kimi-VL-A3B-Thinking/Kimi-VL-A3B-Thinking \
    --tensor-parallel-size 1 \
    --max-num-batched-tokens 131072 \
    --max-model-len 131072 \
    --max-num-seqs 512 \
    --limit-mm-per-prompt image=256 \
    --disable-mm-preprocessor-cache

出现报错

> Traceback (most recent call last):
  File "/usr/local/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/local/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: LLaMa 3.1 8B/70B/405B all behave poorly and differently using completions API as compared to good chat API

**Link**: https://github.com/vllm-project/vllm/issues/7382
**State**: open
**Created**: 2024-08-10T01:47:36+00:00
**Comments**: 22
**Labels**: bug, unstale

### Description

### Your current environment

Docker latest 0.5.4

```
docker pull vllm/vllm-openai:latest
docker run -d --restart=always \
    --runtime=nvidia \
    --gpus '"device=0"' \
    --shm-size=10.24gb \
    -p 5000:5000 \
        -e NCCL_IGNORE_DISABLED_P2P=1 \
    -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
    -e VLLM_NCCL_SO_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib/libnccl.so.2 \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -u `id -u`:`id -g` \
    -v "${HOME}"/.cache:$HOME/.cache/ \
    -v "${HOME}"/.cache/huggingface:$HOME/.cache/huggingface \
    -v "${HOME}"/.cache/huggingface/hub:$HOME/.cache/huggingface/hub \
    -v "${HOME}"/.config:$HOME/.config/   -v "${HOME}"/.triton:$HOME/.triton/  \
    --network host \
    --name llama31_8b \
    vllm/vllm-openai:latest \
        --port=5000 \
        --host=0.0.0.0 \
        --model=meta-llama/Meta-Llama-3.1-8B-Instruct \
        --seed 1234 \
        --ten

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: DRY Sampling

**Link**: https://github.com/vllm-project/vllm/issues/8581
**State**: open
**Created**: 2024-09-18T23:05:42+00:00
**Comments**: 21
**Labels**: feature request, unstale

### Description

### 🚀 The feature, motivation and pitch

DRY is a sampler that completely mitigates repetitions. This is especially important for small models which tend to slop in large contexts. Here's an explanation of DRY from the author himself https://github.com/oobabooga/text-generation-webui/pull/5677
Along with oobabooga, koboldcpp also has an implementation of DRY which according to author IIRC is better than oobabooga's
DRY has been a completely game changer for me and from what I have seen several others. It completely removes need for other samplers like top_p, top_k, repetition_penalty. It is recommended to be used with min_p and produces great coherent results.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked 

[... truncated for brevity ...]

---

## Issue #N/A: ValueError: Model architectures ['Qwen2ForCausalLM'] failed to be inspected. Please check the logs for more details.

**Link**: https://github.com/vllm-project/vllm/issues/13216
**State**: open
**Created**: 2025-02-13T09:42:39+00:00
**Comments**: 24
**Labels**: usage

### Description

### Your current environment

ValueError: Model architectures ['Qwen2ForCausalLM'] failed to be inspected. Please check the logs for more details. 在使用
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B模型时，报了这个错误。

### How would you like to use vllm

I want to run inference of a [specific model](put link here). I don't know how to integrate it with vllm.


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: TRACKING ISSUE: `AsyncEngineDeadError`

**Link**: https://github.com/vllm-project/vllm/issues/5901
**State**: open
**Created**: 2024-06-27T11:49:38+00:00
**Comments**: 23
**Labels**: bug, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### 🐛 Describe the bug

Recently, we have seen reports of `AsyncEngineDeadError`, including:

- [ ] #5060
- [x] #2000
- [x] #3310
- [x] #3839
- [x] #4000
- [x] #4135
- [x] #4293
- [x] #5443
- [x] #5732
- [x] #5822
- [x] #6190 
- [x] #6208
- [x] #6361
- [x] #6421
- [ ] #6614
- [x] #6790
- [x] #6969
- [x] #7356

If you see something like the following, please report here:

```bash
2024-06-25 12:27:29.905   File "/usr/local/lib/python3.10/dist-packages/vllm/entrypoints/openai/api_server.py", line 84, in health
2024-06-25 12:27:29.905     await openai_serving_chat.engine.check_health()
2024-06-25 12:27:29.905   File "/usr/local/lib/python3.10/dist-packages/vllm/engine/async_llm_engine.py", line 839, in check_health
2024-06-25 12:27:29.905     raise AsyncEngineDeadError("Background loop is stopped.")
2024-06-25 12:27:29.905 vllm.engine.async_llm_engine.AsyncEngineDeadError:

[... truncated for brevity ...]

---

## Issue #N/A: ray OOM in tensor parallel

**Link**: https://github.com/vllm-project/vllm/issues/322
**State**: closed
**Created**: 2023-06-30T09:27:50+00:00
**Closed**: 2024-03-20T12:35:08+00:00
**Comments**: 27
**Labels**: bug

### Description

In my case , I can deploy the vllm service on single GPU. but when I use multi gpu, I meet the ray OOM error. Could you please help solve this problem?
my model is yahma/llama-7b-hf
my transformers version is 4.28.0
my cuda version is 11.4


--------
2023-06-30 09:24:53,455 WARNING utils.py:593 -- Detecting docker specified CPUs. In previous versions of Ray, CPU detection in containers was incorrect. Please ensure that Ray has enough CPUs allocated. As a temporary workaround to revert to the prior behavior, set `RAY_USE_MULTIPROCESSING_CPU_COUNT=1` as an env var before starting Ray. Set the env var: `RAY_DISABLE_DOCKER_CPU_WARNING=1` to mute this warning.
2023-06-30 09:24:53,459 WARNING services.py:1826 -- WARNING: The object store is using /tmp instead of /dev/shm because /dev/shm has only 67108864 bytes available. This will harm performance! You may be able to free up space by deleting files in /dev/shm. If you are inside a Docker container, you can increase /dev/shm size by 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: serve Llama-3.2-11B-Vision-Instruct with 2 A10 oom

**Link**: https://github.com/vllm-project/vllm/issues/10034
**State**: closed
**Created**: 2024-11-05T11:34:17+00:00
**Closed**: 2024-11-06T03:45:11+00:00
**Comments**: 22
**Labels**: bug

### Description

### Your current environment

docker image vllm/vllm-openai:v0.6.2 and vllm/vllm-openai:v0.6.3
command：docker run --runtime nvidia --gpus '"device=0,1"' -d -v /data/model/llama:/data/model/llama -p 8001:8000 vllm/vllm-openai:v0.6.2 --model /data/model/llama --max-model-len 1024 --served_model_name Llama-3.2-11B-Vision-Instruct --tensor-parallel-size 2 --gpu_memory_utilization 0.7

I tried v0.6.2 and v0.6.3，both not work，only half of the gpu memory is occupied

nvidia-smi output：
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.42.06              Driver Version: 555.42.06      CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |  

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12

**Link**: https://github.com/vllm-project/vllm/issues/10300
**State**: open
**Created**: 2024-11-13T16:27:59+00:00
**Comments**: 23
**Labels**: bug

### Description

### Your current environment

i cannot execute collect_env.py  because of this error.

in my another environment: torch is 2.4.0 and the version of vllm is `0.6.3.post1` which works fine.

### Model Input Dumps

_No response_

### 🐛 Describe the bug

following installation guide: https://docs.vllm.ai/en/stable/getting_started/installation.html#install-the-latest-code

`vllm version: 0.6.3.post2.dev386+g0b8bb86b`

however, it forces the installation of torch to be `2.5.1`

which causes the error :

> Traceback (most recent call last):
>   File "/home/ubuntu/vllm/collect_env.py", line 15, in <module>
>     from vllm.envs import environment_variables
>   File "/home/ubuntu/vllm/vllm/__init__.py", line 3, in <module>
>     from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
>   File "/home/ubuntu/vllm/vllm/engine/arg_utils.py", line 8, in <module>
>     import torch
>   File "/opt/conda/envs/vllmsource/lib/python3.11/site-packages/torch/__init__.py", li

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Speculative decoding inconsistency for Qwen-Coder-32B

**Link**: https://github.com/vllm-project/vllm/issues/10913
**State**: open
**Created**: 2024-12-05T03:49:40+00:00
**Comments**: 23
**Labels**: bug, unstale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Output of `python collect_env.py` was not collected because I do not have wget within my docker container. I can attempt a docker cp command to put the collect_env.py within my container later, but haven't yet.
```

</details>


### Model Input Dumps

Running VLLM with docker. Speculative decoding for the Qwen-coder-32B using the 0.5B model does not work. Note that all the Qwen models described are from the official Qwen AWQ repos on Huggingface.

Here is the relevant section of docker-compose.yml:

command: >
      --model /app/models/Qwen2.5-Coder-32B-Instruct-AWQ
      --tensor_parallel_size 2
      --max-model-len 23568
      --enable-auto-tool-choice
      --tool-call-parser hermes
      --speculative_model="/app/models/Qwen2.5-Coder-0.5B-Instruct-AWQ"
      --num_speculative_tokens=5

However, curiously, the 7B model DOES work.

command: >
      --mode

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Disagreement and misalignment between supported models in documentation and actual testing

**Link**: https://github.com/vllm-project/vllm/issues/15779
**State**: closed
**Created**: 2025-03-30T21:10:24+00:00
**Closed**: 2025-04-07T06:09:23+00:00
**Comments**: 26
**Labels**: bug

### Description

### Your current environment

Tested under VLLM ==0.7.3 and 0.8.2


### 🐛 Describe the bug

I am using this model (after quantizing it to 4 bits):
**nvidia/Llama-3_3-Nemotron-Super-49B-v1**
https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1
and according to the documentation here:
https://docs.vllm.ai/en/latest/models/supported_models.html
(This is the summary of the related section in the above website:
`To determine whether a given model is natively supported, you can check the config.json file inside the HF repository. If the "architectures" field contains a model architecture listed below, then it should be natively supported.`
)
The **nvidia/Llama-3_3-Nemotron-Super-49B-v1** model architecture according to the HF is **DeciLMForCausalLM**
and according to the documentation above, **DeciLMForCausalLM** is listed as one of the supported architectures hence it should be natively supported. However loading the above model create these issues:

![Image](https://github.com/use

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Docker vLLM 0.9.1 CUDA error: an illegal memory access, sampled_token_ids.tolist()

**Link**: https://github.com/vllm-project/vllm/issues/19483
**State**: open
**Created**: 2025-06-11T09:17:05+00:00
**Comments**: 21
**Labels**: bug

### Description

### Your current environment

Docker on 4 x A100 SMX.
BTW: vLLM 0.8.4 worked stable with same setup.
0.9.01 was already unstable (restarted few time a day), now even more.

```
services:
  vllm-qwen25-72b:
    image: vllm/vllm-openai:v0.9.1
    container_name: vllm-qwen25-72b
    environment:
     ...
      - HF_TOKEN=$HF_TOKEN
      - VLLM_NO_USAGE_STATS=1
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0', '1', '2', '3']
              capabilities: [ gpu ]
    network_mode: host
    volumes:
      - /mnt/sda/huggingface:/root/.cache/huggingface
      - .:/opt/vllm
    command:
      - --port=8000
      - --disable-log-requests
      - --model=Qwen/Qwen2.5-72B-Instruct
      # - --served-model-name=Qwen/Qwen2.5-72B-Instruct
      # - --max-model-len=32768
      - --tensor-parallel-size=4
      - --gpu-memory-utilization=0.90
      - --swap-space=5
    restart: unless-stopped

```

### 🐛 Descri

[... truncated for brevity ...]

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

## Issue #N/A: [Bug]: Loading mistral-7B-instruct-v03 KeyError: 'layers.0.attention.wk.weight'

**Link**: https://github.com/vllm-project/vllm/issues/4989
**State**: closed
**Created**: 2024-05-22T18:52:58+00:00
**Closed**: 2024-05-24T13:38:02+00:00
**Comments**: 25
**Labels**: bug

### Description

### Your current environment

PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Rocky Linux 8.8 (Green Obsidian) (x86_64)
GCC version: (GCC) 8.5.0 20210514 (Red Hat 8.5.0-20)
Clang version: Could not collect
CMake version: version 3.29.2
Libc version: glibc-2.28

Python version: 3.9.13 (main, Oct 13 2022, 21:15:33)  [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-4.18.0-513.9.1.el8_9.x86_64-x86_64-with-glibc2.28
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA H100 PCIe
Nvidia driver version: 535.129.03
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              256
On-line CPU(s) list: 0-255
Thread(s) per core:  2


[... truncated for brevity ...]

---

## Issue #N/A: [Bug]:vllm从0.7.0开始版本部署Qwen2_vl服务存在内存(不是GPU显存)泄漏问题

**Link**: https://github.com/vllm-project/vllm/issues/15597
**State**: closed
**Created**: 2025-03-27T04:45:50+00:00
**Closed**: 2025-03-27T07:13:46+00:00
**Comments**: 23
**Labels**: bug

### Description

### Your current environment

<details>
<summary>vllm从0.7.0开始版本部署Qwen2_vl服务存在内存(不是GPU显存)泄漏问题</summary>

```text


```

</details>


### 🐛 Describe the bug

使用0.7.0版本的vllm部署Qwen2_vl模型服务时，对服务进行请求后，服务相关进程内存不会释放，最终打爆服务器内存，导致服务停止，测试发现0.6.6版本无此问题，0.7.0及以上版本均有此问题。

<img width="625" alt="Image" src="https://github.com/user-attachments/assets/e588c75e-efb7-4e49-872b-4dadba95e06c" />

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: BitsandBytes quantization is not working as expected

**Link**: https://github.com/vllm-project/vllm/issues/5569
**State**: closed
**Created**: 2024-06-15T14:20:10+00:00
**Closed**: 2024-07-27T02:08:24+00:00
**Comments**: 32
**Labels**: bug

### Description

### Your current environment

```text
$ python collect_env.py
Collecting environment information...
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.29.5
Libc version: glibc-2.35

Python version: 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-105-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: Tesla T4
Nvidia driver version: 550.54.15
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.7.6.5
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.9.7
/usr/lib/x86_64-linu

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Llama3.2 tool calling OpenAI API not working

**Link**: https://github.com/vllm-project/vllm/issues/9991
**State**: closed
**Created**: 2024-11-04T11:04:00+00:00
**Closed**: 2024-11-14T04:14:36+00:00
**Comments**: 21
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...                                                                                                                                                                                                                                                           
WARNING 11-04 11:59:16 cuda.py:81] Detected different devices in the system:                                                                                                                                                                                                                    
WARNING 11-04 11:59:16 cuda.py:81] NVIDIA A100 80GB PCIe                                                                                                                                                                                                                                        
WARNING 11-04 11:5

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: VLLM on ARM machine with GH200

**Link**: https://github.com/vllm-project/vllm/issues/10459
**State**: open
**Created**: 2024-11-19T16:57:34+00:00
**Comments**: 28
**Labels**: installation

### Description

### Your current environment

(I can not run collect_env since it requires VLLM installed)

```text
$ pip freeze
certifi==2022.12.7
charset-normalizer==2.1.1
filelock==3.16.1
fsspec==2024.10.0
idna==3.4
Jinja2==3.1.4
MarkupSafe==3.0.2
mpmath==1.3.0
networkx==3.4.2
numpy==2.1.3
pillow==10.2.0
pynvml==11.5.3
requests==2.28.1
sympy==1.13.1
torch==2.5.1
typing_extensions==4.12.2
urllib3==1.26.13

$ lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 22.04.4 LTS
Release:        22.04
Codename:       jammy
```

I have an ARM CPU and a NVIDIA GH200 Driver Version: 550.90.07 CUDA Version: 12.4.

### How you are installing vllm

```sh
pip install torch numpy
pip install vllm
```

I get this error:
```sh
pip install vllm
Collecting vllm
  Using cached vllm-0.6.4.post1.tar.gz (3.1 MB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... error
  error: subprocess-exited-with-error
  
  

[... truncated for brevity ...]

---

## Issue #N/A: Running out of memory loading 7B AWQ quantized models with 12GB vram

**Link**: https://github.com/vllm-project/vllm/issues/1234
**State**: closed
**Created**: 2023-09-30T18:09:45+00:00
**Closed**: 2024-12-01T02:16:11+00:00
**Comments**: 25
**Labels**: performance, stale

### Description

Hi, 

i am trying to make use of the AWQ quantization to try to load 7B LLama based models onto my RTX 3060 with 12 GB.
This fails OOM for models like https://huggingface.co/TheBloke/leo-hessianai-7B-AWQ .
I was able to load https://huggingface.co/TheBloke/tulu-7B-AWQ with its 2k seq length taking up 11.2GB of my ram.

My expectation was that these 7B models with AWQ quantization with GEMM would need for inference around ~ 3.5 gB to load.

I tried to load the models from within my app using vLLM as a lib and following Brokes instructions with
```
python -m vllm.entrypoints.api_server --model TheBloke/tulu-7B-AWQ --quantization awq
```

Do I miss something here?

Thx,
Manuel

---

## Issue #N/A: [SpecDecode] Support EAGLE in V1

**Link**: https://github.com/vllm-project/vllm/issues/15901
**State**: open
**Created**: 2025-04-01T19:45:13+00:00
**Comments**: 21
**Labels**: speculative-decoding, v1

### Description

- [x] 1. Correctly initializing and loading the EAGLE draft model
- [x] 2. Consider the lookahead slots in the KV cache manager
- [x] 3. Cache `draft_probs` inside the model runner and correctly feed it to the rejection sampler in the next step (temporarily workaround: #16899)
- [x] 4. Handle the edge cases like when the draft model generates beyond `max_pos_embeddings`
- [ ] 5. Handle the seeds correctly
- [ ] 6. Do E2E correctness and performance tests
- [x] 7. Support prefix caching. Eagle requires special handling because Eagle's i-th KV cache is coupled with the i+1-th token ID. (@LiuXiaoxuanPKU)
- [ ] 8. Properly handle the sampling parameters that are not (currently) compatible with spec decoding (e.g., min_p).
- [x] 9. Use CUDA graphs for draft model. (@luyuzhe111)
- [x] 10. Support Eagle 3 (https://github.com/vllm-project/vllm/pull/16937)

_Originally posted by @WoosukKwon in https://github.com/vllm-project/vllm/issues/15729#issuecomment-2765192455_
            

---

## Issue #N/A: [Bug]: qwen2.5-vl-72b oom in 4 A100 in 0.8.3

**Link**: https://github.com/vllm-project/vllm/issues/16570
**State**: closed
**Created**: 2025-04-14T06:25:47+00:00
**Closed**: 2025-04-16T10:29:50+00:00
**Comments**: 37
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
- Platform: Linux-5.15.0-56-generic-x86_64-with-glibc2.35
- Python version: 3.12.3
- PyTorch version: 2.6.0+cu124 (GPU)
- Transformers version: 4.50.3
- Datasets version: 3.2.0
- Accelerate version: 1.2.1
- PEFT version: 0.15.0
- TRL version: 0.9.6
- GPU type: NVIDIA A100-PCIE-40GB
- GPU number: 8
- GPU memory: 39.38GB
- vLLM version: 0.8.3
```

</details>


### 🐛 Describe the bug

VLLM_WORKER_MULTIPROC_METHOD=spawn python -m vllm.entrypoints.openai.api_server \
--dtype auto \
--port 8009 \
--trust-remote-code \
--served-model-name qwen2vl \
--model /home/ps/data/pretrained_model/Qwen/Qwen2.5-VL-72B-Instruct/ \
--tensor-parallel-size 4 \
--gpu_memory_utilization 0.95 \
--max_num_seqs 2 \
--max_model_len 8192 \
--mm_processor_kwargs '{"max_pixels":1280, "min_pixels":256}'

above is my running code. I find that it caused 36GB memory each card, but i don't know the later process and v

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Qwen2.5 VL Internal Server Error

**Link**: https://github.com/vllm-project/vllm/issues/13655
**State**: closed
**Created**: 2025-02-21T08:07:01+00:00
**Closed**: 2025-07-01T02:58:45+00:00
**Comments**: 22
**Labels**: bug

### Description

### Your current environment

I used the official docker image v0.7.2 and reinstalled vllm with commit d0a7a2769d92619afdcdc3b91c78098eaa9e38c0 and trainsformers 4.49.0.

<details>
<summary>The output of `python collect_env.py`</summary>

```text
/usr/local/lib/python3.12/dist-packages/vllm/__init__.py:5: RuntimeWarning: Failed to read commit hash:
No module named 'vllm._version'
  from .version import __version__, __version_tuple__  # isort:skip
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
Python platform: Linux-5.10.0-32-amd64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: DeepSeek-Coder-V2-Instruct-AWQ    assert self.quant_method is not None

**Link**: https://github.com/vllm-project/vllm/issues/7494
**State**: closed
**Created**: 2024-08-14T01:25:56+00:00
**Closed**: 2025-05-07T02:10:19+00:00
**Comments**: 21
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
```text
ollecting environment information...
PyTorch version: 2.3.1+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Debian GNU/Linux 11 (bullseye) (x86_64)
GCC version: (Debian 10.2.1-6) 10.2.1 20210110
Clang version: Could not collect
CMake version: version 3.30.2
Libc version: glibc-2.31

Python version: 3.9.2 (default, Feb 28 2021, 17:03:44)  [GCC 10.2.1 20210110] (64-bit runtime)
Python platform: Linux-5.4.143.bsk.8-amd64-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA L40
GPU 1: NVIDIA L40
GPU 2: NVIDIA L40
GPU 3: NVIDIA L40
GPU 4: NVIDIA L40
GPU 5: NVIDIA L40
GPU 6: NVIDIA L40
GPU 7: NVIDIA L40

Nvidia driver version: Could not collect
cuDNN version: Probably one of the following:
/usr/li

[... truncated for brevity ...]

---

