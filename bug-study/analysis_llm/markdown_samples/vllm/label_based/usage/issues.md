# usage - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 6
- Closed Issues: 24

### Label Distribution

- usage: 30 issues
- stale: 10 issues

---

## Issue #N/A: [Usage]: Distributed inference not supported with OpenVINO?

**Link**: https://github.com/vllm-project/vllm/issues/14933
**State**: open
**Created**: 2025-03-17T07:06:59+00:00
**Comments**: 3
**Labels**: usage, stale

### Description

### How would you like to use vllm

The [installation page for OpenVINO](https://docs.vllm.ai/en/latest/getting_started/installation/ai_accelerator.html?device=openvino) mentions using the environment variable "VLLM_OPENVINO_DEVICE to specify which device utilize for the inference. If there are multiple GPUs in the system, additional indexes can be used to choose the proper one (e.g, VLLM_OPENVINO_DEVICE=GPU.1). If the value is not specified, CPU device is used by default."

So is it not possible to use multiple GPUs or GPU + CPU for running inference on OpenVINO backend?

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: OpenAI API for Phi-3-vision-128k-instruct 

**Link**: https://github.com/vllm-project/vllm/issues/7068
**State**: closed
**Created**: 2024-08-02T06:55:10+00:00
**Closed**: 2024-08-02T08:05:13+00:00
**Comments**: 4
**Labels**: usage

### Description


```text
BadRequestError: Error code: 400 - {'object': 'error', 'message': 'Attempted to assign 1 x 2509 = 2509 image tokens to 0 placeholders', 'type': 'BadRequestError', 'param': None, 'code': 400}
```
calling using following function:
```python
def prepare_prompts(self, prompts, images):
        messages = []
        #re.sub(r"<\|.*?\|>", "", )
        for i in range(len(prompts)):
            if i % 2 == 0:
                content = [
                    {
                        "type": "text",
                        "text": prompts[i]
                    }
                ]
                if images[i]:
                    img_byte_arr = io.BytesIO()
                    images[i].save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    image_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
                    content.append(
                        {
                            "type": "image_u

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method

**Link**: https://github.com/vllm-project/vllm/issues/8392
**State**: closed
**Created**: 2024-09-12T01:54:55+00:00
**Closed**: 2024-09-12T03:35:34+00:00
**Comments**: 3
**Labels**: usage

### Description

### Your current environment

I used the same service deployment command, but when I upgraded from 0.5.5 to 0.6.1 today, the deployment went wrong

![20240912-095248](https://github.com/user-attachments/assets/278b91a6-6a35-4c0c-b956-f20c5e09997e)


### How would you like to use vllm

I want to run inference of a [specific model](put link here). I don't know how to integrate it with vllm.


### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: what is enforce_eager

**Link**: https://github.com/vllm-project/vllm/issues/4449
**State**: closed
**Created**: 2024-04-29T07:14:26+00:00
**Closed**: 2024-05-01T13:38:09+00:00
**Comments**: 5
**Labels**: usage

### Description

### Your current environment

vllm 0.4.0
cuda 12.1
2*v100-16G
qwen1.5 Moe

### How would you like to use vllm

what is enforce_eager?
and when it's enabled, will the inference become slower?

---

## Issue #N/A: [Usage]: Failed to get global TPU topology.

**Link**: https://github.com/vllm-project/vllm/issues/16243
**State**: open
**Created**: 2025-04-08T07:49:20+00:00
**Comments**: 1
**Labels**: usage, stale

### Description

### Your current environment

PyTorch version: 2.8.0
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Debian GNU/Linux 11 (bullseye) (x86_64)
GCC version: (Debian 10.2.1-6) 10.2.1 20210110
Clang version: Could not collect
CMake version: version 4.0.0
Libc version: glibc-2.31

Python version: 3.10.16 (main, Jan 14 2025, 05:27:07) [GCC 10.2.1 20210110] (64-bit runtime)
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

CPU:
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Byte Order:                      Little Endian
Address sizes:                   52 bits physical, 57 bits virtual
CPU(s):                          44
On-line CPU(s) list:   

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: how to redirect save logs to local file.

**Link**: https://github.com/vllm-project/vllm/issues/16319
**State**: open
**Created**: 2025-04-09T06:31:49+00:00
**Comments**: 2
**Labels**: usage, stale

### Description

### Your current environment

i am using docker 0.8.2 to run the model, output of collect_env.py
```text
The output of `Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.31.6
Libc version: glibc-2.35

Python version: 3.12.9 (main, Feb  5 2025, 08:49:00) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-78-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: Tesla T4
Nvidia driver version: 550.54.15
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Address sizes:                   

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: How can I use temperature correctly for Qwen2-VL?

**Link**: https://github.com/vllm-project/vllm/issues/13322
**State**: closed
**Created**: 2025-02-15T07:49:44+00:00
**Closed**: 2025-02-17T09:21:11+00:00
**Comments**: 1
**Labels**: usage

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How would you like to use vllm

I want to run inference of Qwen2-VL-2B. The code is:
```
# Qwen2-VL
def init_qwen2_vl(model_name_or_path: str, **kwargs):
    from vllm import LLM
    try:
        from qwen_vl_utils import process_vision_info
    except ModuleNotFoundError:
        print('WARNING: `qwen-vl-utils` not installed, input images will not '
              'be automatically resized. You can enable this functionality by '
              '`pip install qwen-vl-utils`.')
        process_vision_info = None

    model_name = model_name_or_path

    llm = LLM(
        model=model_name,
        device=kwargs['device'], 
        max_model_len=kwargs.get("max_context_len", 4096 if process_vision_info is not None else 32768),  
        enable_prefix_caching=True,
        enforce_eager=True,
        disable_mm_preprocessor_cache=kwargs.get("disable_mm_preprocessor_cache", True),
    )
    stop_token_ids = 

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: how to acquire logits in vllm

**Link**: https://github.com/vllm-project/vllm/issues/8762
**State**: closed
**Created**: 2024-09-24T06:42:51+00:00
**Closed**: 2025-01-24T01:58:47+00:00
**Comments**: 4
**Labels**: usage, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How would you like to use vllm

I want to acquire logits when I run benchmark_throughput.py to do the softmax optimization, but the output in vllm doesn't have logits, how can I acquire it.

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: 请问如何用vllm进行多机部署？

**Link**: https://github.com/vllm-project/vllm/issues/12765
**State**: closed
**Created**: 2025-02-05T03:24:26+00:00
**Closed**: 2025-02-05T03:25:00+00:00
**Comments**: 0
**Labels**: usage

### Description

### Your current environment

Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-130-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.3.107
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

Nvidia driver version: 550.142
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.4.0
/usr/l

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: How to use a Python script to start a FastAPI service for internvl2-8b with vllm, instead of using the terminal command vllm serve ./internvl2-1b/ --tensor-parallel-size 1 --trust-remote-code? Is there any sample code for this?

**Link**: https://github.com/vllm-project/vllm/issues/10953
**State**: closed
**Created**: 2024-12-06T14:35:00+00:00
**Closed**: 2024-12-06T15:30:54+00:00
**Comments**: 1
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

## Issue #N/A: [Usage]: FP8 online quantization weight synchronization

**Link**: https://github.com/vllm-project/vllm/issues/17272
**State**: open
**Created**: 2025-04-27T22:32:04+00:00
**Comments**: 0
**Labels**: usage

### Description

### Your current environment

If we run a vllm instance with ```quantization='fp8'```, how should we update its weight ?

I encountered an issue: 
```
assert param.size() == loaded_weight.size(), (
[rank10]: AssertionError: Attempted to load weight (torch.Size([3840, 1280])) into parameter (torch.Size([1280, 3840]))
```
this is due to the training models' type is bfloat16.  Can we only enable FP8 training if we want to use FP8 inference? 




### How would you like to use vllm

I want to run inference of a [specific model](put link here). I don't know how to integrate it with vllm.


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: KV Cache Warning for `gemma2`

**Link**: https://github.com/vllm-project/vllm/issues/7404
**State**: closed
**Created**: 2024-08-11T22:33:24+00:00
**Closed**: 2024-12-10T02:07:51+00:00
**Comments**: 3
**Labels**: usage, stale

### Description

### Your current environment

I get the following warning running a quantized version of `gemma2`, when I have not quantized the kv cache:

```bash
WARNING 08-11 22:31:50 gemma2.py:399] Some weights are not initialized from checkpoints: {'model.layers.2.self_attn.attn.v_scale', 'model.layers.13.self_attn.attn.v_scale', 'model.layers.14.self_attn.attn.v_scale', 'model.layers.16.self_attn.attn.k_scale', 'model.layers.16.self_attn.attn.v_scale', 'model.layers.19.self_attn.attn.v_scale', 'model.layers.21.self_attn.attn.v_scale', 'model.layers.2.self_attn.attn.k_scale', 'model.layers.3.self_attn.attn.k_scale', 'model.layers.20.self_attn.attn.v_scale', 'model.layers.24.self_attn.attn.k_scale', 'model.layers.6.self_attn.attn.k_scale', 'model.layers.6.self_attn.attn.v_scale', 'model.layers.12.self_attn.attn.k_scale', 'model.layers.15.self_attn.attn.v_scale', 'model.layers.18.self_attn.attn.k_scale', 'model.layers.14.self_attn.attn.k_scale', 'model.layers.9.self_attn.attn.k_scale', 'model.

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: When using with Peft-loaded model, got error: PreTrainedTokenizerFast has no attribute lower

**Link**: https://github.com/vllm-project/vllm/issues/17620
**State**: closed
**Created**: 2025-05-04T00:48:02+00:00
**Closed**: 2025-05-06T05:18:48+00:00
**Comments**: 2
**Labels**: usage

### Description

### Your current environment

```text
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.10.12 (main, Jan 17 2025, 14:35:34) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-113-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA RTX 6000 Ada Generation
GPU 1: NVIDIA RTX 6000 Ada Generation
GPU 2: NVIDIA RTX 6000 Ada Generation
GPU 3: NVIDIA RTX 6000 Ada Generation

Nvidia driver version: 560.28.03
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]:RuntimeError: Triton Error [CUDA]: device kernel image is invalid

**Link**: https://github.com/vllm-project/vllm/issues/18580
**State**: closed
**Created**: 2025-05-23T02:08:47+00:00
**Closed**: 2025-05-27T00:55:54+00:00
**Comments**: 3
**Labels**: usage

### Description

### Your current environment

```text
(vllm) llm@aitt:/data_a/llm$ python collect_env.py 
INFO 05-23 10:07:58 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
==============================
        System Info
==============================
OS                           : Ubuntu 20.04.4 LTS (x86_64)
GCC version                  : (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version                : Could not collect
CMake version                : version 3.16.3
Libc version                 : glibc-2.31

==============================
       PyTorch Info
==============================
PyTorch version              : 2.6.0+cu118
Is debug build               : False
CUDA used to build PyTorch   : 11.8
ROCM used to build PyTorch   : N/A

==============================
      Python Environment
==============================
Python version               : 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27) [GCC 11.2.0] (64-bit runtime

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Given sufficient GPU memory, which is better: starting a single vLLM instance or starting multiple instances for load balancing?

**Link**: https://github.com/vllm-project/vllm/issues/13442
**State**: closed
**Created**: 2025-02-18T01:02:44+00:00
**Closed**: 2025-02-18T19:37:07+00:00
**Comments**: 1
**Labels**: usage

### Description

### Your current environment

```txt
A100-80G×8
```

### How would you like to use vllm

Given sufficient GPU memory, which is better: starting a single vLLM instance or starting multiple instances for load balancing?

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: DOCKER - Getting OOM while running `meta-llama/Llama-3.2-11B-Vision-Instruct`

**Link**: https://github.com/vllm-project/vllm/issues/8903
**State**: closed
**Created**: 2024-09-27T12:26:27+00:00
**Closed**: 2024-09-27T12:32:19+00:00
**Comments**: 2
**Labels**: usage

### Description

### Your current environment

I'm trying to run `meta-llama/Llama-3.2-11B-Vision-Instruct` using vLLM docker:

**GPU Server specifications:**

- GPU Count: 4
- GPU Type: A100 - 80GB

**vLLM Docker run command:**
```bash
docker run  --gpus all \
    -v /data/hf_cache/ \
    --env "HUGGING_FACE_HUB_TOKEN=<token>" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.2-11B-Vision-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --download_dir /data/vllm_cache \
    --enforce-eager
```

----
**Following is the issue which I'm facing:**
```bash
VllmWorkerProcess pid=214) ERROR 09-27 05:20:38 multiproc_worker_utils.py:233] Exception in worker VllmWorkerProcess while processing method determine_num_available_blocks: CUDA out of memory. Tried to allocate 19.63 GiB. GPU 3 has a total capacity of 79.15 GiB of which 17.73 GiB is free. Process 78729 has 61.41 GiB memory in use. Of the allocated memory 56.

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Using AsyncLLMEngine with asyncio.run

**Link**: https://github.com/vllm-project/vllm/issues/3996
**State**: closed
**Created**: 2024-04-11T06:31:52+00:00
**Closed**: 2024-06-19T20:57:14+00:00
**Comments**: 9
**Labels**: usage

### Description

### Your current environment

```text
Collecting environment information...
PyTorch version: 2.1.2+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-102-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA GeForce RTX 3060
GPU 1: NVIDIA RTX A6000

Nvidia driver version: 550.54.15
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

## Issue #N/A: [Usage]: intent is added for guided generation

**Link**: https://github.com/vllm-project/vllm/issues/19107
**State**: open
**Created**: 2025-06-03T21:33:32+00:00
**Comments**: 0
**Labels**: usage

### Description

### Your current environment

`{\n"product":\n"pixel",\n"rating":\n3\n}` 

response_format + guided generation will add \n. how can we avoid this intent=2 for guided generation? and only force the model to generate via dense json.dumps default.  

### How would you like to use vllm

- use guided generation without intent

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: Tokenizer or Not?

**Link**: https://github.com/vllm-project/vllm/issues/13415
**State**: closed
**Created**: 2025-02-17T17:45:11+00:00
**Closed**: 2025-02-17T21:21:34+00:00
**Comments**: 3
**Labels**: usage

### Description

### Your current environment

```
INFO 02-17 17:43:56 __init__.py:190] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-130-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A100 80GB PCIe
GPU 1: NVIDIA A100 80GB PCIe

Nvidia driver version: 535.216.03
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):             

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Get first token latency 

**Link**: https://github.com/vllm-project/vllm/issues/8471
**State**: closed
**Created**: 2024-09-13T17:38:49+00:00
**Closed**: 2025-01-17T01:57:44+00:00
**Comments**: 3
**Labels**: usage, stale

### Description



Is there a way to get the first token latency? benchmarks/benchmark_latency.py provides the latency of processing a single batch of requests but I am interested in getting first token latency


### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: Can vLLM profile capture the python process of api_server?

**Link**: https://github.com/vllm-project/vllm/issues/10684
**State**: closed
**Created**: 2024-11-27T01:50:38+00:00
**Closed**: 2024-11-27T03:24:13+00:00
**Comments**: 2
**Labels**: usage

### Description

### Your current environment

```text
The output of `python collect_env.py`
```
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
Python platform: Linux-5.10.134-16.3.al8.x86_64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H20
GPU 1: NVIDIA H20
GPU 2: NVIDIA H20
GPU 3: NVIDIA H20

Nvidia driver version: 550.54.15
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit


[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: If I want to test in a concurrent environment using llava, say request/s=6, how can I do it?

**Link**: https://github.com/vllm-project/vllm/issues/12033
**State**: closed
**Created**: 2025-01-14T11:11:19+00:00
**Closed**: 2025-01-14T11:58:35+00:00
**Comments**: 1
**Labels**: usage

### Description

### Your current environment

No samples of concurrent use of Llava were found.

### How would you like to use vllm

I want to run inference of a [specific model](put link here). I don't know how to integrate it with vllm.


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: 怎么修改python -m vllm.entrypoints.openai.api_server的提示词

**Link**: https://github.com/vllm-project/vllm/issues/10220
**State**: closed
**Created**: 2024-11-11T11:43:36+00:00
**Closed**: 2025-03-12T02:02:44+00:00
**Comments**: 6
**Labels**: usage, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```
python3.10.12

### How would you like to use vllm

I want to run inference of a [specific model](put link here). I don't know how to integrate it with vllm.
![108FCB7D-FBB6-429b-9B79-EA00E0A9AFDB](https://github.com/user-attachments/assets/a750281d-eed9-4036-8b22-361193adeb6c)
我希望修改系统提示词，不论是在代码里、模型文件里或者命令行里，怎么改变这个提示词呢

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: Pass multiple LoRA modules through YAML config

**Link**: https://github.com/vllm-project/vllm/issues/9655
**State**: closed
**Created**: 2024-10-24T10:39:28+00:00
**Closed**: 2025-01-23T05:45:42+00:00
**Comments**: 11
**Labels**: usage, stale

### Description

### How would you like to use vllm

I would like to pass multiple LoRA modules to the vLLM engine, but currently I'm receiving error while parsing the `lora_modules` property.

The `LoRAParserAction` class receives a `Sequence[str]` in case you want to use multiple LoRA modules.

I have a YAML config file in which I declare the vLLM engine arguments, like this:
```
model: ai-models/Meta-Llama-3.1-8B-Instruct-rev-5206a32
tokenizer_mode: auto
dtype: half
lora_modules: "ai-models/adv_perizia_exp7_run6=ai-models/adv_perizia_exp7_run6"
max_num_batched_tokens: 32768
max_num_seqs: 192
gpu_memory_utilization: 0.95
tensor_parallel_size: <RAY_LLM_NUM_WORKERS>
max_model_len: 32768
```

In that way (`name=path` for the LoRA module), all works and I'm able to perform inference with LoRA (I set `enable_lora` argument later in the code, not in the YAML file).
Now I would like to pass multiple `lora_modules`, but I'm receiving parsing error in every different ways I tried:

`lora

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: How to offload some layers to CPU？

**Link**: https://github.com/vllm-project/vllm/issues/3931
**State**: closed
**Created**: 2024-04-09T09:10:35+00:00
**Closed**: 2024-11-28T02:06:43+00:00
**Comments**: 7
**Labels**: usage, stale

### Description

### Your current environment

None

### How would you like to use vllm

I want to load qwen2-14B-chat using VLLM, but I only have 1 RTX4090(24G). 
Can vllm offload some layers to cpu and others to gpu?
As I know, the transformers-accelerate and llama.cpp can do it. But I want to use the multilora switch function in VLLM.

---

## Issue #N/A: [Usage]:

**Link**: https://github.com/vllm-project/vllm/issues/15390
**State**: closed
**Created**: 2025-03-24T11:45:12+00:00
**Closed**: 2025-03-24T14:23:55+00:00
**Comments**: 1
**Labels**: usage

### Description

### Your current environment

Hello everyone:

I have 3 x 2080 8G, 1 x 2080Ti (22G VRAM), 1 x 3080 (20G),  1 x RTX5070 Ti,
how do I use vllm to load the QwQ-32B-AWQ large language model with these graphics cards? 

Thank you



### How would you like to use vllm

I want to run inference of a [specific model](put link here). I don't know how to integrate it with vllm.


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: 支持qwen3的版本下，如何使用beam search限制生成

**Link**: https://github.com/vllm-project/vllm/issues/19421
**State**: open
**Created**: 2025-06-10T12:04:32+00:00
**Comments**: 0
**Labels**: usage

### Description

### Your current environment

vllm:v0.8.5


### How would you like to use vllm

支持qwen3的版本下，vllm.beam_search并没有可以配置logits_processors的地方


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: Does larger max_num_batched_tokens use more VRAM?

**Link**: https://github.com/vllm-project/vllm/issues/13604
**State**: closed
**Created**: 2025-02-20T09:57:01+00:00
**Closed**: 2025-02-20T11:28:58+00:00
**Comments**: 0
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

## Issue #N/A: [Usage]: 125m parameter model is also showing CUDA: Out of memory error in a Nvidia16GB 4060 

**Link**: https://github.com/vllm-project/vllm/issues/8136
**State**: closed
**Created**: 2024-09-03T23:43:52+00:00
**Closed**: 2024-09-08T23:42:03+00:00
**Comments**: 14
**Labels**: usage

### Description




### How would you like to use vllm

Even for a smaller model like "facebook/opt-125m" when I am trying to do multiprocessing(even with batch size of 2) on a single 16GB Nvidia 4060, I am encountering CUDA: OUT OF MEMORY ERROR. When I am running the same model sequentially, I am able to run it fine. Can you explain this?

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: RuntimeError: Failed to infer device type (Intel Iris Xe Graphics)

**Link**: https://github.com/vllm-project/vllm/issues/8863
**State**: closed
**Created**: 2024-09-26T18:59:17+00:00
**Closed**: 2025-04-24T02:08:29+00:00
**Comments**: 10
**Labels**: usage, stale

### Description

### Your current environment

```text
Collecting environment information...
WARNING 09-26 20:43:46 _custom_ops.py:18] Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
INFO 09-26 20:43:46 importing.py:10] Triton not installed; certain GPU-related functions will not be available.
C:\Users\sasha\vllm\vllm\vllm\connections.py:8: RuntimeWarning: Failed to read commit hash:
No module named 'vllm.commit_id'
  from vllm.version import __version__ as VLLM_VERSION
PyTorch version: 2.4.0+cpu
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Microsoft Windows 10 Enterprise
GCC version: Could not collect
Clang version: Could not collect
CMake version: Could not collect
Libc version: N/A

Python version: 3.10.14 | packaged by Anaconda, Inc. | (main, May  6 2024, 19:44:50) [MSC v.1916 64 bit (AMD64)] (64-bit runtime)
Python platform: Windows-10-10.0.19045-SP0
Is CUDA available: False
CUDA runtime versi

[... truncated for brevity ...]

---

