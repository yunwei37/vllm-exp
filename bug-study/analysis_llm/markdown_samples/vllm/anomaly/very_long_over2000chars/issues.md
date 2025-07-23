# very_long_over2000chars - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 9
- Closed Issues: 21

### Label Distribution

- bug: 21 issues
- stale: 7 issues
- installation: 2 issues
- good first issue: 1 issues
- feature request: 1 issues
- ray: 1 issues
- rocm: 1 issues
- RFC: 1 issues
- new-model: 1 issues

---

## Issue #N/A: [Bug]: A800 GPU set VLLM_USE_V1=1 ValueError: No available memory for the cache blocks

**Link**: https://github.com/vllm-project/vllm/issues/17431
**State**: open
**Created**: 2025-04-30T02:52:40+00:00
**Comments**: 4
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 04-30 10:44:38 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
/home/python_vllm_env_085/lib/python3.11/site-packages/_distutils_hack/__init__.py:30: UserWarning: Setuptools is replacing distutils. Support for replacing an already imported distutils is deprecated. In the future, this condition will fail. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: BigCloud Enterprise Linux For Euler 21.10 LTS (x86_64)
GCC version: (GCC) 7.3.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.28

Python version: 3.11.9 (main, Jan 14 2025, 14:39:54) [GCC 4.8.5 20150623 (Red Hat 4.8.5-44)] (64-bit runtime)
Python platform

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: VLLM 0.5.1 with LLaVA 1.6 exceptions

**Link**: https://github.com/vllm-project/vllm/issues/6322
**State**: closed
**Created**: 2024-07-11T06:11:25+00:00
**Closed**: 2024-07-11T17:21:12+00:00
**Comments**: 1
**Labels**: bug

### Description

### Your current environment

See https://github.com/vllm-project/vllm/issues/6176


### 🐛 Describe the bug

I have lots of image, where the service throws exception and after that must be restarted, because it stucks in exception mode, even for images, that worked before.
Example image below.

```
curl 'https://ai1.dev.init/multimodal-llava/v1/chat/completions' -k -H 'Content-Type: application/json' -d @- <<EOF
{
    "model": "llava-hf/llava-v1.6-mistral-7b-hf",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,$(base64 -w 0 /opt/initai_copilot/data/images/image2015-7-20_16_17_54.png)"
                    }
                },
                {
                    "type": "text",
                    "text": "Was ist in dem Bild?"
                }
            ]
        }
    ],
    "t

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Cannot use FlashAttention-2 backend because the vllm_flash_attn package is not found. But I have installed vllm-flash-attn.

**Link**: https://github.com/vllm-project/vllm/issues/7112
**State**: closed
**Created**: 2024-08-03T15:07:46+00:00
**Closed**: 2024-09-16T13:54:15+00:00
**Comments**: 6
**Labels**: bug

### Description

### Your current environment

```text
Collecting environment information...
/app/apps/anaconda3/envs/vllm_040p1/lib/python3.9/site-packages/requests/__init__.py:102: RequestsDependencyWarning: urllib3 (1.26.16) or char
det (5.2.0)/charset_normalizer (2.0.4) doesn't match a supported version!
  warnings.warn("urllib3 ({}) or chardet ({})/charset_normalizer ({}) doesn't match a supported "
WARNING 08-03 23:03:20 _custom_ops.py:15] Failed to import from vllm._C with ImportError('libcudart.so.12: cannot open shared object file: No 
such file or directory')
PyTorch version: 2.4.0+cu118
Is debug build: False
CUDA used to build PyTorch: 11.8
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.31

Python version: 3.9.12 (main, Apr  5 2022, 06:56:58)  [GCC 7.5.0] (64-bit runtime)
Python platform: Linux-5.4.0-148-generic-x86_64

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Output the JSON for the response payload when VLLM_LOGGING_LEVEL=DEBUG

**Link**: https://github.com/vllm-project/vllm/issues/15571
**State**: closed
**Created**: 2025-03-26T20:19:33+00:00
**Closed**: 2025-03-27T17:49:40+00:00
**Comments**: 5
**Labels**: good first issue, feature request

### Description

### 🚀 The feature, motivation and pitch

It's difficult to debug VLLM requests and responses when we as developers can't see the JSON request and response payloads in debug mode. This is especially important when using VLLM as an inference server with MCP or an agentic framework.  Differences in how the models format their function-calling and tool-invocation responses can cause problems, so we need to be able to see exactly what the response was.

When VLLM_LOGGING_LEVEL=DEBUG I am currently not able to see this information. For example, here's what I see in the logs.  The request payload is there even with the logging level set to INFO, so that's good.  However, the response payload is not there even if we set the logging level to DEBUG.  All we get back is `"POST /v1/chat/completions HTTP/1.1" 200 OK`:

```
INFO 03-05 19:47:39 logger.py:37] Received request chatcmpl-xxxx: prompt: '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nEnvironment: ipython\nCutting Knowledge 

[... truncated for brevity ...]

---

## Issue #N/A: Failed when load a merged Mistral 8x7b model

**Link**: https://github.com/vllm-project/vllm/issues/2386
**State**: closed
**Created**: 2024-01-09T06:09:22+00:00
**Closed**: 2024-11-30T02:03:06+00:00
**Comments**: 7
**Labels**: stale

### Description

  I merged a mistal 8x7b model with the lora adapter, and I save the .pt with torch.save(model.state_dict(), 'path_to_model.pt')

However, when I use vllm to inference on the new merged model, I failed with this:

```
File "/home/zhh/miniconda3/envs/vllm/lib/python3.9/site-packages/vllm/entrypoints/llm.py", line 93, in __init__
    self.llm_engine = LLMEngine.from_engine_args(engine_args)
  File "/home/zhh/miniconda3/envs/vllm/lib/python3.9/site-packages/vllm/engine/llm_engine.py", line 246, in from_engine_args
    engine = cls(*engine_configs,
  File "/home/zhh/miniconda3/envs/vllm/lib/python3.9/site-packages/vllm/engine/llm_engine.py", line 107, in __init__
    self._init_workers_ray(placement_group)
  File "/home/zhh/miniconda3/envs/vllm/lib/python3.9/site-packages/vllm/engine/llm_engine.py", line 194, in _init_workers_ray
    self._run_workers(
  File "/home/zhh/miniconda3/envs/vllm/lib/python3.9/site-packages/vllm/engine/llm_engine.py", line 750, in _run_workers
    

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: client socket has timed out while trying to connect to GPU node, when initializing DeepSeek R1 in ray vllm serving

**Link**: https://github.com/vllm-project/vllm/issues/15744
**State**: open
**Created**: 2025-03-29T07:02:08+00:00
**Comments**: 6
**Labels**: bug, ray

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```

</details>


### 🐛 Describe the bug

I construct a ray cluster and try to deploy several DeepSeek-R1 replicas. pipeline-parallel-size: 3
Often, the model initialization would fail (not every time, could turn succeed after several retries)
I'm on ray[serve]==2.44.0, vllm==0.8.2. This issue starts ever since 0.7.0. I've verified that it works well on 0.6.6.post1, but every version after that would possible to trigger below error msg when initializing multi-node models, (in our case is R1)

Error:

```
:job_id:02000000
:actor_name:ServeReplica:DS-R1:vllmDeployment
INFO 2025-03-29 06:17:53,742 DS-R1_vllmDeployment a3e34qi7 -- Starting with engine args: AsyncEngineArgs(model='DeepSeek-R1', served_model_name=None, tokenizer='DeepSeek-R1', hf_config_path=None, task='auto', skip_tokenizer_init=False, tokenizer_mode='auto', trust_remote_code=

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: [V1 Engine] GLM4-1V video processing fails with token count mismatch: "Attempted to assign X multimodal tokens to Y placeholders"

**Link**: https://github.com/vllm-project/vllm/issues/20742
**State**: open
**Created**: 2025-07-10T07:38:16+00:00
**Comments**: 3
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

+ cuda==12.8
+ pip3 install vllm==0.9.2 

export CUDA_VISIBLE_DEVICES="0"
export server_ip=0.0.0.0
export server_port=8000
export model_path=/workspace/data/GLM-4.1V-9B-Thinking
export server_name=glm-41v-9b
export VLLM_USE_V1="1"

python3 -m vllm.entrypoints.openai.api_server --model ${model_path} \
--max-num-seqs=32 \
--tensor-parallel-size=1 \
--gpu-memory-utilization=0.8 \
--no-enable-chunked-prefill \
--limit-mm-per-prompt video=1 \
--enable-prefix-caching \
--served-model-name=${server_name} 

</details>


### 🐛 Describe the bug

### Description

**Problem Summary:**
When processing video inputs with GLM4-1V model using vLLM V1 engine, the system crashes with a ValueError indicating a mismatch between the number of multimodal tokens generated by the vision encoder and the number of placeholder tokens reserved during preprocessing.

**Error Message:**
```
ValueError: Attemp

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: AMD GPU tp>1 模型上线卡住

**Link**: https://github.com/vllm-project/vllm/issues/10150
**State**: closed
**Created**: 2024-11-08T09:47:36+00:00
**Closed**: 2025-03-11T02:03:37+00:00
**Comments**: 2
**Labels**: bug, rocm, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.4.0+rocm6.1
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.1.40091-a8dbc0c19

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.4
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-124-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: Radeon RX 7900 XTX (gfx1100)
Nvidia driver version: Could not collect
cuDNN version: Could not collect
HIP runtime version: 6.1.40091
MIOpen runtime version: 3.1.0
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-b

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: MiniCPMV Raises UnboundLocalError When Image Placeholder is Omitted

**Link**: https://github.com/vllm-project/vllm/issues/8990
**State**: closed
**Created**: 2024-10-01T06:16:15+00:00
**Closed**: 2024-10-01T09:52:45+00:00
**Comments**: 0
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

OS: Red Hat Enterprise Linux release 8.9 (Ootpa) (x86_64)
GCC version: (GCC) 8.5.0 20210514 (Red Hat 8.5.0-20)
Clang version: Could not collect
CMake version: version 3.30.2
Libc version: glibc-2.28

Python version: 3.10.4 (main, Mar 31 2022, 08:41:55) [GCC 7.5.0] (64-bit runtime)
Python platform: Linux-4.18.0-513.11.1.el8_9.x86_64-x86_64-with-glibc2.28
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A100-SXM4-80GB
Nvidia driver version: 535.54.03
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:      

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: InternVl2-8B-AWQ gives error when trying to run with vllm-openai cuda 11.8 docker image

**Link**: https://github.com/vllm-project/vllm/issues/8736
**State**: closed
**Created**: 2024-09-23T10:16:32+00:00
**Closed**: 2024-09-23T10:28:46+00:00
**Comments**: 1
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

</details>


### Model Input Dumps

_No response_

### 🐛 Describe the bug

Process SpawnProcess-1:
Traceback (most recent call last):
  File "/usr/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/usr/lib/python3.12/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 319, in run_mp_engine
    engine = MQLLMEngine.from_engine_args(engine_args=engine_args,
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 113, in from_engine_args
    return cls(
           ^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 69, in __init__
    self.engine = LLMEngine

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: tool_calls and None types.

**Link**: https://github.com/vllm-project/vllm/issues/16678
**State**: closed
**Created**: 2025-04-15T18:51:52+00:00
**Closed**: 2025-04-22T15:40:25+00:00
**Comments**: 0
**Labels**: RFC

### Description

### Motivation.

### Summary
Exploring tool_calls in VLLM.  Seems to be a need for additional handling of non-iterable None type responses in the tool_calls field.  

### Motivation

I've been testing with Google's new ADK (backed by LiteLLM) against a VLLM hosted Qwen 2.5 Instruct model and the Hermes parser.  (Using Kserve).  

https://github.com/google/adk-python/blob/290058eb05211ef531b1752c6290da3f365e4e73/src/google/adk/models/lite_llm.py#L194
ADK explicitly returns a None in the tool_calls field.  
` tool_calls=tool_calls or None,`

From what I understand, some models and hosting do expect this None type, however VLLM does not.  This means that on the second message of the chat, you get the following:  
`ERROR - fast_api.py:616 - Error in event_generator: litellm.BadRequestError: OpenAIException - 'NoneType' object is not iterable`

This seems to bring us to 
https://github.com/vllm-project/vllm/blob/54a66e5fee4a1ea62f1e4c79a078b20668e408c6/vllm/entrypoints/chat_utils.py#L1072



[... truncated for brevity ...]

---

## Issue #N/A: [Bug]:  "address already in use" while deploying pipeline parallel

**Link**: https://github.com/vllm-project/vllm/issues/9556
**State**: closed
**Created**: 2024-10-21T15:38:41+00:00
**Closed**: 2024-10-21T16:13:52+00:00
**Comments**: 2
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

OS: Ubuntu 24.04.1 LTS (x86_64)
GCC version: (Ubuntu 13.2.0-23ubuntu4) 13.2.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.39

Python version: 3.12.3 (main, Sep 11 2024, 14:17:37) [GCC 13.2.0] (64-bit runtime)
Is CUDA available: True
CUDA runtime version: 12.6.77
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA L40S
GPU 1: NVIDIA L40S
GPU 2: NVIDIA L40S
GPU 3: NVIDIA L40S
GPU 4: NVIDIA L40S
GPU 5: NVIDIA L40S
GPU 6: NVIDIA L40S
GPU 7: NVIDIA L40S

Nvidia driver version: 560.35.03
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):    

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: cc1plus: error: invalid feature modifier in ‘-march=armv8.2-a+dotprod+fp16’

**Link**: https://github.com/vllm-project/vllm/issues/17906
**State**: open
**Created**: 2025-05-09T14:23:47+00:00
**Comments**: 3
**Labels**: installation

### Description

### Your current environment

```text
The output of `python collect_env.py`
```

  -- CPU extension compile flags: -fopenmp;-DVLLM_CPU_EXTENSION;-march=armv8.2-a+dotprod+fp16
      -- Enabling C extension.
      -- Configuring done (3.5s)
      -- Generating done (0.0s)
      -- Build files have been written to: /tmp/pip-install-b261gxij/vllm_e99d65c51e5c4cea948384f47668e6c3/build/temp.linux-aarch64-cpython-310
      [1/9] Building CXX object CMakeFiles/_C.dir/csrc/cpu/activation.cpp.o
      FAILED: CMakeFiles/_C.dir/csrc/cpu/activation.cpp.o
      /usr/bin/c++ -DPy_LIMITED_API=3 -DTORCH_EXTENSION_NAME=_C -DUSE_C10D_GLOO -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE -D_C_EXPORTS -I/tmp/pip-install-b261gxij/vllm_e99d65c51e5c4cea948384f47668e6c3/csrc -isystem /home/ma-user/anaconda3/envs/py310/include/python3.10 -isystem /tmp/pip-build-env-mwllr031/overlay/lib/python3.10/site-packages/torch/include -isystem /tmp/pip-build-env-mwllr031/overlay/lib/python3.10/site-packages/torch/include/tor

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Unable to serve Qwen2-audio in V1

**Link**: https://github.com/vllm-project/vllm/issues/12168
**State**: closed
**Created**: 2025-01-17T14:30:27+00:00
**Closed**: 2025-01-19T03:16:35+00:00
**Comments**: 6
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 01-17 22:19:48 __init__.py:179] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.5.1+cu118
Is debug build: False
CUDA used to build PyTorch: 11.8
ROCM used to build PyTorch: N/A

OS: Ubuntu 18.04.6 LTS (x86_64)
GCC version: (Ubuntu 10.3.0-1ubuntu1~18.04~1) 10.3.0
Clang version: Could not collect
CMake version: version 3.31.2
Libc version: glibc-2.27

Python version: 3.12.8 | packaged by Anaconda, Inc. | (main, Dec 11 2024, 16:31:09) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.4.0-169-generic-x86_64-with-glibc2.27
Is CUDA available: True
CUDA runtime version: 11.7.99
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A100-SXM4-80GB
GPU 1: NVIDIA A100-SXM4-80GB
GPU 2: NVIDIA A100-SXM4-80GB
GPU 3: NVIDIA A100-SXM4-80GB

Nvidia driver version: 535.129.03
cuDNN version: Probably one of the following

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: ERROR hermes_tool_parser.py:108] Error in extracting tool call from response.

**Link**: https://github.com/vllm-project/vllm/issues/10831
**State**: closed
**Created**: 2024-12-02T11:55:32+00:00
**Closed**: 2025-04-04T02:05:36+00:00
**Comments**: 7
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
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.12.7 (main, Oct  1 2024, 08:52:12) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-125-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA RTX A6000
GPU 1: NVIDIA RTX A6000
GPU 2: NVIDIA RTX A6000
GPU 3: NVIDIA RTX A6000

Nvidia driver version: 535.183.01
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Arc

[... truncated for brevity ...]

---

## Issue #N/A: Runtime Error When Running the Throughput Benchmarks WIth 2000 Requests (Invalid Token Generated) (Model: OPT-2.7B, Dataset: SharedGPT)

**Link**: https://github.com/vllm-project/vllm/issues/2310
**State**: closed
**Created**: 2023-12-31T03:30:09+00:00
**Closed**: 2024-03-28T13:31:24+00:00
**Comments**: 0

### Description

When running the throughput benchmark on the OPT-2.7B model on the SharedGPT dataset (2000 requests), a None token is generated, incurring a runtime error within the de-tokenizer. The ID of the problematic token is 50265 while the vocab size of the OPT tokenizer is only 50265. 

Reproduce the bug:
```bash
python benchmark_throughput.py --backend vllm --dataset ../ShareGPT_V3_unfiltered_cleaned_split.json --model facebook/opt-2.7b --num-prompts 2000
``` 

The logging output:
```bash
Namespace(backend='vllm', dataset='../ShareGPT_V3_unfiltered_cleaned_split.json', dtype='auto', enforce_eager=False, hf_max_batch_size=None, input_len=None, max_model_len=None, model='facebook/opt-2.7b', n=1, num_prompts=2000, output_len=None, quantization=None, seed=0, tensor_parallel_size=1, tokenizer='facebook/opt-2.7b', trust_remote_code=False, use_beam_search=False)
INFO 12-30 22:13:01 llm_engine.py:73] Initializing an LLM engine with config: model='facebook/opt-2.7b', tokenizer='facebook/opt-

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]:  the issue of "cuda out of memory" arises

**Link**: https://github.com/vllm-project/vllm/issues/15182
**State**: open
**Created**: 2025-03-20T03:31:44+00:00
**Comments**: 1
**Labels**: bug

### Description

### Your current environment

### current environment
Use 8 A800 80 GPU cards, report the following error: Deepseek is using the Q4 gguf model.
### command: 
python3 api_server.py --host 0.0.0.0 --port 7803 --model /data/models/DeepSeek-R1-Q2/ --served-model-name deepseek-r1 --max-model-len 8192 --enable-reasoning --reasoning-parser deepseek_r1 --gpu-memory-utilization 0.9 --tensor-parallel-size 8 --trust-remote-code

(VllmWorker rank=0 pid=3013) INFO 03-20 03:14:59 [shm_broadcast.py:258] vLLM message queue communication handle: Handle(local_reader_ranks=[1, 2, 3, 4, 5, 6, 7], buffer_handle=(7, 4194304, 6, 'psm_5030d6c2'), local_subscribe_addr='ipc:///tmp/54feae1f-3d13-452d-9e9e-1bf1fffe271e', remote_subscribe_addr=None, remote_addr_ipv6=False)
(VllmWorker rank=1 pid=3024) INFO 03-20 03:14:59 [parallel_state.py:967] rank 1 in world size 8 is assigned as DP rank 0, PP rank 0, TP rank 1
(VllmWorker rank=6 pid=3085) INFO 03-20 03:14:59 [parallel_state.py:967] rank 6 in world size 8 is ass

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vllm deploy qwen1.5-14b/qwen2-7b+medusa, RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x5120 and 4096x4096)

**Link**: https://github.com/vllm-project/vllm/issues/8613
**State**: closed
**Created**: 2024-09-19T02:57:41+00:00
**Closed**: 2024-09-19T03:52:25+00:00
**Comments**: 1
**Labels**: bug

### Description

### Your current environment
vllm=0.6.1

### Model Input Dumps

CUDA_VISIBLE_DEVICES=7 python3 -m vllm.entrypoints.openai.api_server --port 8010 \
  --served-model-name qwen2-7b \
  --model /mnt/user/deploy/qwen15_14b_finetuning_chatbot_v1_0914_deploy --dtype auto -tp 1 \
  --max-model-len 4096 --gpu-memory-utilization 0.9 \
  --max-num-seqs 1 \
  --speculative-model /mnt/user/deploy/qwen15_14b_finetuning_chatbot_v1_0914_deploy/medusa \
  --speculative-draft-tensor-parallel-size 1 \
  --num-speculative-tokens 3 \
  --speculative-disable-by-batch-size 3 \
  --use-v2-block-manager \
  --spec-decoding-acceptance-method typical_acceptance_sampler

### 🐛 Describe the bug

ERROR 09-19 10:53:38 async_llm_engine.py:63] Engine background task failed
ERROR 09-19 10:53:38 async_llm_engine.py:63] Traceback (most recent call last):
ERROR 09-19 10:53:38 async_llm_engine.py:63]   File "/opt/conda/envs/vllm/lib/python3.10/site-packages/vllm/engine/async_llm_engine.py", line 53, in

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Incorrect kernel selected when multiple GPUs

**Link**: https://github.com/vllm-project/vllm/issues/19741
**State**: open
**Created**: 2025-06-17T11:19:39+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
INFO 06-17 14:15:03 [__init__.py:243] Automatically detected platform cuda.
WARNING 06-17 14:15:03 [cuda.py:435] Detected different devices in the system: NVIDIA GeForce RTX 5090, NVIDIA GeForce RTX 4090, NVIDIA GeForce RTX 3090. Please make sure to set `CUDA_DEVICE_ORDER=PCI_BUS_ID` to avoid unexpected behavior.
Collecting environment information...
/home/unat/.venv/lib/python3.12/site-packages/torch/cuda/__init__.py:287: UserWarning:
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
If you want to use the NVIDIA GeForce RTX 5090 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

  warnings.warn(
==============================
        System Info
============================

[... truncated for brevity ...]

---

## Issue #N/A: Confusing problems generated by  Qwen14b-int4

**Link**: https://github.com/vllm-project/vllm/issues/1774
**State**: closed
**Created**: 2023-11-24T08:28:55+00:00
**Closed**: 2024-03-25T09:42:02+00:00
**Comments**: 1

### Description

Confusing problems generated by qwen14b-int
- If max-tokens is not added, it will stop after a few words.
just like this:

```
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "来一首宋词吧:",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=1024)


and this result:
INFO 11-24 16:22:09 llm_engine.py:72] Initializing an LLM engine with config: model='/home/incar/newdata2/tms/llm/QWenData/Qwen-14B-Chat-Int4', tokenizer='/home/incar/newdata2/tms/llm/QWenData/Qwen-14B-Chat-Int4', tokenizer_mode=auto, revision=v1.1.8, tokenizer_revision=None, trust_remote_code=True, dtype=torch.float16, max_seq_len=2048, download_dir=None, load_format=auto, tensor_parallel_size=1, quantization=gptq, seed=0)
WARNING 11-24 16:22:10 tokenizer.py:66] Using a slow tokenizer. This might cause a significant slowdown. Consider using a fast tokenizer instead.

[... truncated for brevity ...]

---

## Issue #N/A: [New Model]: Qwen2.5-VL

**Link**: https://github.com/vllm-project/vllm/issues/13715
**State**: closed
**Created**: 2025-02-23T04:36:34+00:00
**Closed**: 2025-02-23T04:44:32+00:00
**Comments**: 0
**Labels**: new-model

### Description

### The model to consider.

It looks like vllm doesn't currently support Qwen 2.5-VL?

ray.exceptions.RayTaskError(ValueError): ray::WorkerDict.actor_rollout_generate_sequences() (pid=67366, ip=192.168.128.5, actor_id=2681de9b19487eb6c57dea4401000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x7ee6341e7dc0>)
  File "/mnt/2050data/wentao.zhang/MultiModalMath/verl/single_controller/ray/base.py", line 399, in func
    return getattr(self.worker_dict[key], name)(*args, **kwargs)
  File "/mnt/2050data/wentao.zhang/MultiModalMath/verl/single_controller/base/decorator.py", line 404, in inner
    return func(*args, **kwargs)
  File "/mnt/2050data/wentao.zhang/MultiModalMath/verl/workers/fsdp_workers.py", line 463, in generate_sequences
    with self.rollout_sharding_manager:
  File "/mnt/2050data/wentao.zhang/MultiModalMath/verl/workers/sharding_manager/fsdp_vllm.py", line 83, in __enter__
    load_dtensor_weights(
  File "/mnt/2050data/wentao.zhang/MultiModalMath/verl/third_

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Error while running Deepseek-R1: vllm.engine.async_llm_engine.AsyncEngineDeadError: Task finished unexpectedly.

**Link**: https://github.com/vllm-project/vllm/issues/13676
**State**: closed
**Created**: 2025-02-21T16:25:09+00:00
**Closed**: 2025-06-25T02:16:28+00:00
**Comments**: 2
**Labels**: bug, stale

### Description

### Your current environment

<details>
Was working 10minutes ago, seems like after vllm crash whole GPUs are stuck ?

```
Traceback (most recent call last):
  File "/root/collect_env.py", line 17, in <module>
    from vllm.envs import environment_variables
  File "/usr/local/lib/python3.10/dist-packages/vllm/__init__.py", line 7, in <module>
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
  File "/usr/local/lib/python3.10/dist-packages/vllm/engine/arg_utils.py", line 20, in <module>
    from vllm.executor.executor_base import ExecutorBase
  File "/usr/local/lib/python3.10/dist-packages/vllm/executor/executor_base.py", line 15, in <module>
    from vllm.platforms import current_platform
  File "<frozen importlib._bootstrap>", line 1075, in _handle_fromlist
  File "/usr/local/lib/python3.10/dist-packages/vllm/platforms/__init__.py", line 222, in __getattr__
    _current_platform = resolve_obj_by_qualname(
  File "/usr/local/lib/python3.10/dist-packages/vllm/utils.py", 

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: Docker novice installation help (urgent)

**Link**: https://github.com/vllm-project/vllm/issues/12549
**State**: closed
**Created**: 2025-01-29T13:17:11+00:00
**Closed**: 2025-01-30T17:36:22+00:00
**Comments**: 1
**Labels**: installation

### Description

### Your current environment

I used this command "DOCKER_BUILDKIT=1 docker build . --target vllm-openai-base --tag vllm/vllm-openai" to build the vllm image, but at one step the build became unusually slow

`root@iZwz9av7dpqr38k3rph9ziZ:~/vllm# DOCKER_BUILDKIT=1 docker build . --target vllm-openai-base --tag vllm/vllm-openai --build-arg torch_cuda_arch_list=""
[+] Building 418.7s (28/37)                                                                                  docker:default
 => [internal] load build definition from Dockerfile                                                                   0.0s
 => => transferring dockerfile: 12.57kB                                                                                0.0s
 => WARN: FromAsCasing: 'as' and 'FROM' keywords' casing do not match (line 141)                                       0.0s
 => [internal] load metadata for docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04                                        0.3s
 => [internal] l

[... truncated for brevity ...]

---

## Issue #N/A: [BUG]: Streaming `logprob` & `echo` combo.

**Link**: https://github.com/vllm-project/vllm/issues/2703
**State**: closed
**Created**: 2024-02-01T04:54:55+00:00
**Closed**: 2024-04-11T22:15:52+00:00
**Comments**: 2

### Description

I'm trying to start writing a `logprob` & `echo` support for chat request. 

Unfortunately, running test like #1992 when `echo` is setted as `true` server doesn't respond.

Seeing furtherer I checked that the **bug** begging in #2449 (sha: dd7e8f5f643167e3f13045cf75cbead54cb2ccfe).
Previous commit #2463 (sha: d2a68364c473a3167a1c2b90f947bb611322a867) worked ok.

## LOG:
```
vllm-openai-main  | INFO 02-01 04:31:38 async_llm_engine.py:385] Received request cmpl-dc7fb40d1b534a879768966f3dc50d39: prompt: None, prefix_pos: None,sampling params: SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=[], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=True, max_tokens=20, logprobs=0, prompt_logprobs=0, skip_special_tokens=True, spaces_between_special_tokens=True), prompt token ids: [2, 12375, 351, 5, 232, 6

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Guided decoding only generating single character during inference with finetuned model

**Link**: https://github.com/vllm-project/vllm/issues/13448
**State**: closed
**Created**: 2025-02-18T02:58:39+00:00
**Closed**: 2025-06-20T02:13:26+00:00
**Comments**: 2
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
RuntimeWarning: Failed to read commit hash:
No module named 'vllm._version'
  from vllm.version import __version__ as VLLM_VERSION
Collecting environment information...
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: Could not collect
CMake version: version 3.26.3
Libc version: glibc-2.31

Python version: 3.10.15 (main, Dec  2 2024, 18:21:11) [GCC 9.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-1035-aws-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.2.140
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A10G
GPU 1: NVIDIA A10G
GPU 2: NVIDIA A10G
GPU 3: NVIDIA A10G

Nvidia driver version: 535.183.01
cuDNN version: Probably one of the following:
/usr/local/cuda-

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Extreme low throughput when using pipeline parallelism when Batch Size(running req) is small

**Link**: https://github.com/vllm-project/vllm/issues/9176
**State**: open
**Created**: 2024-10-09T02:46:05+00:00
**Comments**: 11
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.29.5
Libc version: glibc-2.35

Python version: 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-78-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.2.140
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
cuDNN version: Could not collect
HIP run

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: When qwen3-reranker-0.6B is loaded using Tesla T4, the CPU memory continues to grow until the system crashes

**Link**: https://github.com/vllm-project/vllm/issues/20658
**State**: open
**Created**: 2025-07-09T03:15:57+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

The graphics card is Tesla T4, and NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0 
[pip-requirements-all.txt](https://github.com/user-attachments/files/21133901/pip-requirements-all.txt)

### 🐛 Describe the bug

The machine is a single card machine with 32G memory. When executed using the following code, the memory grows until it crashes,：
[https://github.com/QwenLM/Qwen3-Embedding/blob/main/examples/qwen3_reranker_vllm.py](url)
I think it's the distributed parameter: distributed_executor_backend='ray'.
This doesn't happen when I load it like this：
```
self.lm = LLM(
model=model_name_or_path,
#   tensor_parallel_size=number_of_gpu,
max_model_len=self.max_length,
#   enable_prefix_caching=True,
#   distributed_executor_backend='ray',
enforce_eager=True,
Gpu_memory_utilization = 0.5,
dtype=kwargs.get('dtype', 'float16'),
)
```
enable_prefix_caching is not used, presumably because triton is not yet compatible with T4


```
import logg

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Whisper not working on 0.9.2 docker image

**Link**: https://github.com/vllm-project/vllm/issues/20671
**State**: open
**Created**: 2025-07-09T09:16:41+00:00
**Comments**: 1
**Labels**: bug

### Description

### Your current environment

Docker image 0.9.2 on NVidia L40S.
(The docker image has to be modified, because librosa dependency is missing.)

```
services:
  vllm-whisper-large-v3:
    # Must modify image for <= v0.9.2
    # ImportError: Please install vllm[audio] for audio support
    # image: vllm/vllm-openai:v0.9.2
    image: vllm/vllm-openai-audio:v0.9.2
    build:
      context: .
    container_name: vllm-whisper-large-v3
    environment:
      - HF_TOKEN=$HF_TOKEN
      - VLLM_NO_USAGE_STATS=1
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['2']
              capabilities: [ gpu ]
    network_mode: host
    volumes:
      - /mnt/sda/huggingface:/root/.cache/huggingface
      - .:/opt/vllm
    command:
      - --port=8006
      - --disable-log-requests
      - --model=openai/whisper-large-v3
      - --gpu-memory-utilization=0.40
      - --swap-space=5
    restart: unless-stopped
```

```


[... truncated for brevity ...]

---

## Issue #N/A: [Bug]:  Have the same bug with Issue #11762, using vllm>=0.7.2

**Link**: https://github.com/vllm-project/vllm/issues/13969
**State**: closed
**Created**: 2025-02-27T14:59:01+00:00
**Closed**: 2025-03-20T11:14:28+00:00
**Comments**: 8
**Labels**: bug

### Description

### Your current environment


vllm = 0.7.2 or 0.7.3



### 🐛 Describe the bug

Exactly the same bug on Qwen2.5 VL with issue [#11732](https://github.com/vllm-project/vllm/issues/11762)
```python
ERROR 02-27 18:50:20 engine.py:389] RuntimeError: Failed to apply Qwen2_5_VLProcessor on data={'text': '<|image_pad|><|image_pad|><|image_pad|><|image_pad|><|video_pad|>', 'images': [<PIL.Image.Image image mode=RGB size=3584x3584 at 0x791D887EC2B0>, <PIL.Image.Image image mode=RGB si
ze=3584x3584 at 0x791D887EC2B0>, <PIL.Image.Image image mode=RGB size=3584x3584 at 0x791D887EC2B0>, <PIL.Image.Image image mode=RGB size=3584x3584 at 0x791D887EC2B0>], 'videos': [array([[[[0., 0., 0.],                                                                                   
ERROR 02-27 18:50:20 engine.py:389]          [0., 0., 0.],                                                                                                                                                                                

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Build/Install Issues with pip install -e .

**Link**: https://github.com/vllm-project/vllm/issues/5071
**State**: closed
**Created**: 2024-05-27T16:34:19+00:00
**Closed**: 2024-11-25T02:06:01+00:00
**Comments**: 5
**Labels**: bug, stale

### Description

### Your current environment

```text
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

Python version: 3.8.10 (default, Nov 22 2023, 10:22:35)  [GCC 9.4.0] (64-bit runtime)
Python platform: Linux-5.4.0-182-generic-x86_64-with-glibc2.29
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA L40
GPU 1: NVIDIA A100 80GB PCIe

Nvidia driver version: 550.54.15
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Byte Order:                         Little

[... truncated for brevity ...]

---

