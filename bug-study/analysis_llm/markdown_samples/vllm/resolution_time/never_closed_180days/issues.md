# never_closed_180days - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 30
- Closed Issues: 0

### Label Distribution

- bug: 16 issues
- stale: 12 issues
- unstale: 11 issues
- feature request: 9 issues
- structured-output: 4 issues
- performance: 2 issues
- tool-calling: 2 issues
- RFC: 2 issues
- ray: 2 issues
- usage: 1 issues

---

## Issue #N/A: [Feature]: Support Multiple Tasks Per Model

**Link**: https://github.com/vllm-project/vllm/issues/11905
**State**: open
**Created**: 2025-01-09T18:22:13+00:00
**Comments**: 15
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Requesting this for **V1** #11862 

The idea is pretty simple, it would be nice to be able to, e.g., get generations and embeddings out of a single model. An example use case is when you have a LoRA for generation and a LoRA for embedding on top of the same base model. Deploying two vLLM servers is really inefficient for accomplishing this. 

### Alternatives

A lesser feature would be one task per LoRA, but it's better to be general if possible.

### Additional context

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Performance]: 1P1D Disaggregation performance

**Link**: https://github.com/vllm-project/vllm/issues/11345
**State**: open
**Created**: 2024-12-19T18:37:57+00:00
**Comments**: 11
**Labels**: performance

### Description

### Proposal to improve performance

I try to reproduce the P&D 1P1D benchmark to compare performance with chunked prefill  https://github.com/vllm-project/vllm/blob/main/benchmarks/disagg_benchmarks/disagg_performance_benchmark.sh. TTFL is higher than what I expected. Because the overhead benchmark only shows ~20-30ms level. What's more, seems ITL is also much higher than chunked prefill. 

- GPU device: 2* L40S.
- Model: Qwen/Qwen2.5-7B-Instruct
- Parameters: gpu-memory-utilization 0.6 + kv_buffer_size 10e9 
- dataset input 1024 output 50.

/cc @KuntaiDu 


### Report of performance regression

![image](https://github.com/user-attachments/assets/2c5ec50f-1e5b-48c6-aca2-ab0be42935ed)

### Misc discussion on performance

_No response_

### Your current environment (if you think it is necessary)

```text
The output of `python collect_env.py`
```


### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the ch

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Inconsistent Responses with VLLM When Batch Size > 1 even temperature = 0

**Link**: https://github.com/vllm-project/vllm/issues/5898
**State**: open
**Created**: 2024-06-27T09:34:57+00:00
**Comments**: 36
**Labels**: bug, unstale

### Description

### 🐛 Describe the bug

Test Environment:
- Hardware: A100 80GB GPU
- Model: Llama3-8b 
**- Parameters: temperature = 0, max_tokens = 1024, max_num_seqs = 256, seed=1**
- I make OpenAI-Compatilbe Server using python -m vllm.entrypoints.openai.api_server  ...

Test Method:
- First, requests were sent one at a time to verify if the same prompt consistently produces the same response.
- Second, more than one requests with same prompt at a time were sent  to verify if the same prompt consistently produces the same response.

Discovered Issue:
**- Consistency with Single Request: When the batch size is 1, the same prompt consistently produces the same response.**
**- Inconsistency with Multiple Requests: When the batch size increases to more than 1, the responses for the same prompt become inconsistent.**
- When I set vllm server with parameter max_num_seqs = 1, the result is all same

I think this suggests that the issue arises from the way the Batch Scheduler processes mu

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Consider parallel_tool_calls parameter at the API level

**Link**: https://github.com/vllm-project/vllm/issues/9451
**State**: open
**Created**: 2024-10-17T07:41:26+00:00
**Comments**: 19
**Labels**: feature request, tool-calling

### Description

### 🚀 The feature, motivation and pitch

Currently, there is a [parallel_tool_calls](https://github.com/vllm-project/vllm/blob/18b296fdb2248e8a65bf005e7193ebd523b875b6/vllm/entrypoints/openai/protocol.py#L177) field that is part of the `ChatCompletionRequest` pydantic class. However, this field is only there for being compatible with OpenAI's API.

In other words, it's not being used at all according to the documentation or the code:

```
# NOTE this will be ignored by VLLM -- the model determines the behavior
parallel_tool_calls: Optional[bool] = False
```

Would it be possible to consider implementing the logic behind this field for different model families. For instance, in the case of llama3.1-8b-insturct, tool calling works, but the model ends up returning three tool calls instead of one by one.
This makes me lose compatibility with frameworks like LangGraph.

Here's an example request and response:

**Request**
```
{
  "messages": [
    {
      "content": "You 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]:  error: Segmentation fault(SIGSEGV received at time)

**Link**: https://github.com/vllm-project/vllm/issues/6918
**State**: open
**Created**: 2024-07-29T21:52:17+00:00
**Comments**: 13
**Labels**: bug, stale

### Description

### Your current environment

Collecting environment information...
PyTorch version: N/A
Is debug build: N/A
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.10.12 (main, Jun 11 2023, 05:26:28) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-25-generic-x86_64-with-glibc2.35
Is CUDA available: N/A
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration:
GPU 0: NVIDIA H100 80GB HBM3
GPU 1: NVIDIA H100 80GB HBM3
GPU 2: NVIDIA H100 80GB HBM3
GPU 3: NVIDIA H100 80GB HBM3
GPU 4: NVIDIA H100 80GB HBM3
GPU 5: NVIDIA H100 80GB HBM3
GPU 6: NVIDIA H100 80GB HBM3
GPU 7: NVIDIA H100 80GB HBM3

Nvidia driver version: 535.86.10
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: who to run cluster withou docker

**Link**: https://github.com/vllm-project/vllm/issues/12053
**State**: open
**Created**: 2025-01-14T19:24:08+00:00
**Comments**: 2
**Labels**: usage, unstale

### Description

### Your current environment

if i cannot use docker ,how to run vllm on multiple nodes ?



### How would you like to use vllm

if i cannot use docker ,how to run vllm on multiple nodes ?



### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: Llama3.3 Tool calling support or a Geneneric and extensible llama tool calling support

**Link**: https://github.com/vllm-project/vllm/issues/11799
**State**: open
**Created**: 2025-01-07T07:01:45+00:00
**Comments**: 1
**Labels**: feature request, unstale, tool-calling

### Description

### 🚀 The feature, motivation and pitch

We have customer moving from llama3.1/3.2 to 3.3 and further when available

### Alternatives

Not yet explored

### Additional context

A generic way where we can use use tool calling support against llms instead of using specific params like 
--tool-call-parser llama3_json  /instead of --tool-call-parser <whatever model supports> as an external reference via chat template or so ?

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: load and save kv cache from disk

**Link**: https://github.com/vllm-project/vllm/issues/10611
**State**: open
**Created**: 2024-11-25T02:17:06+00:00
**Comments**: 6
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

For prefix cache, cache hits can significantly reduce FTT. However, kv cache occupies a large amount of storage space, and the space in CPU memory and GPU video memory is very expensive and limited, resulting in limited prefix cache and decreased hit probability. By caching the kv cache on disk/SSD, the kv-cache can be reused, greatly improving the hit rate.



### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: Guided Decoding Schema Cache Store

**Link**: https://github.com/vllm-project/vllm/issues/8902
**State**: open
**Created**: 2024-09-27T12:07:20+00:00
**Comments**: 3
**Labels**: feature request, structured-output, unstale

### Description

### 🚀 The feature, motivation and pitch

# Problem

I am currently working with structured outputs and experimented a little with VLLM + Outlines. Since our JSON Schemas can get quite complex the generation of the FSM can take around 2 Minutes per Schema. It would be great to have a feature where you can provide a Schema-Store to save your generated schemas over time in a local file and reload them when you restart your deployment. Ideally this would be implemented as flag in the vllm serve arguments:

https://docs.vllm.ai/en/latest/models/engine_args.html

# Current Implementation
I assume that this is currently not supported and the code to not recompute the schema is handled with the @cache() decorator here:
![Screenshot 2024-09-27 134948](https://github.com/user-attachments/assets/4d6480a8-5a79-40ab-8b5c-6023b0551233)


### Alternatives

Alternative solution would probably be to create custom python code to handle this for my use-case and use the VLLM python function

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: I want to integrate vllm into LLaMA-Factory, a transformers-based LLM training framework. However, I encountered two bugs: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method & RuntimeError: NCCL error: invalid usage (run with NCCL_DEBUG=WARN for details)

**Link**: https://github.com/vllm-project/vllm/issues/9469
**State**: open
**Created**: 2024-10-17T15:17:07+00:00
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

OS: Ubuntu 20.04.5 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.31

Python version: 3.10.0 (default, Mar  3 2022, 09:58:08) [GCC 7.5.0] (64-bit runtime)
Python platform: Linux-5.15.0-50-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.1.66
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

Nvidia driver version: 530.30.02
cuDNN version: Could not collect
HIP

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: [Performance] 100% performance drop using multiple lora vs no lora(qwen-chat model)

**Link**: https://github.com/vllm-project/vllm/issues/9496
**State**: open
**Created**: 2024-10-18T08:14:35+00:00
**Comments**: 14
**Labels**: bug, unstale

### Description

### Your current environment

[Performance] 100% performance drop using multiple lora vs no lora(qwen-chat model)
gpu: 4 * T4
vllm version： v0.5.4
model：qwenhalf-14b-chat



### Model Input Dumps

_No response_

### 🐛 Describe the bug

[Performance] 100% performance drop using multiple lora vs no lora(qwen-chat model)
gpu: 4*T4
2k prompt,500 output tokens, in multiple lora model infer by vllm cost 32s
when 2k prompt,500 output tokens, in sinle model infer by vllm cost 16s;
why is there such a speed difference? the reason？

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: vllm.engine.async_llm_engine.AsyncEngineDeadError: Background loop has errored already.

**Link**: https://github.com/vllm-project/vllm/issues/5060
**State**: open
**Created**: 2024-05-26T22:44:41+00:00
**Comments**: 45
**Labels**: bug, stale

### Description

### Your current environment

docker image: vllm/vllm-openai:0.4.2
Model: https://huggingface.co/alpindale/c4ai-command-r-plus-GPTQ
GPUs: RTX8000 * 2

### 🐛 Describe the bug

The model works fine until the following error is raised. 
-------------------------------------------------------


INFO 05-26 22:28:18 async_llm_engine.py:529] Received request cmpl-10dff83cb4b6422ba8c64213942a7e46: prompt: '<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>"Question: Is Korea the name of a Nation?\nGuideline: No explanation.\nFormat: {"Answer": "<your yes/no answer>"}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>', sampling_params: SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=['---'], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=4096, min_tokens=0, log

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vLLM CPU mode broken Unable to get JIT kernel for brgemm

**Link**: https://github.com/vllm-project/vllm/issues/10478
**State**: open
**Created**: 2024-11-20T06:39:21+00:00
**Comments**: 14
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.5.1+cpu
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0
Clang version: Could not collect
CMake version: version 3.31.0
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov  6 2024, 20:22:13) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.8.0-1018-gcp-x86_64-with-glibc2.35
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
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:               

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: Transformers 4.45.1 slows down `outlines` guided decoding

**Link**: https://github.com/vllm-project/vllm/issues/9032
**State**: open
**Created**: 2024-10-02T22:28:16+00:00
**Comments**: 5
**Labels**: performance, structured-output

### Description

### Report of performance regression

I noticed that guided decoding was a bit slower on newer builds of vllm, but couldn't track down a commit that caused a performance regression. Instead it looks like upgrading transformers from `4.44.2` to `4.45.1` causes the issue.

I ran a small artillery test with requests using guided decoding, using the code from commit `4f1ba0844`. This is the last commit before `mllama` support was added, so it's the last point where vllm will work with both transformers versions `4.44.2` and `4.45.1`. VLLM was run with 1xA100 gpu, using model `mistralai/Mistral-7B-Instruct-v0.2`

The results with `4.44.2` installed:
```
http.codes.200: ................................................................ 240
http.downloaded_bytes: ......................................................... 91928
http.request_rate: ............................................................. 3/sec
http.requests: ..........................................................

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: LoRA support for qwen2-vl Models

**Link**: https://github.com/vllm-project/vllm/issues/11255
**State**: open
**Created**: 2024-12-17T06:41:58+00:00
**Comments**: 16
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

I fine-tuned a qwen2-vl-7b model using llama factory, deployed it with AsyncLLMEngine, and loaded the LoRA adapter using lora_request. However, the inference results are significantly worse compared to the merged model.

<img width="822" alt="image" src="https://github.com/user-attachments/assets/29bb01c1-2507-4fab-972f-ac4245d884a3" />


It would be great if we can have the support for LoRA for multimodal models as our team wants to use multiple LoRAs and merging the LoRA adapters to original model weights is not feasible for us. We are short on time for this project and as far as I can tell no other framework supports LoRA in this way. Also we need outlines for structured generation so vLLM (being the most user friendly, stable and mature framework ) is our best bet now. Can we get a timeline when will this be supported ? Also are there any workarounds possible until this feature is officially supported ?

Thank you for your adaptation.

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: PaliGemma2 not working with OpenAI Docker serve

**Link**: https://github.com/vllm-project/vllm/issues/12052
**State**: open
**Created**: 2025-01-14T19:22:48+00:00
**Comments**: 19
**Labels**: bug, unstale

### Description

### Your current environment

Just using Docker image 0.6.6post1


### Model Input Dumps

_No response_

### 🐛 Describe the bug

Just try to run https://huggingface.co/google/paligemma2-3b-pt-896 using Docker vllm image. My docker compose follows:

```
services:
  app:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./cache:/root/.cache/huggingface
    environment:
      - HUGGING_FACE_HUB_TOKEN=hf_
    ipc: host
    command:
      - --host
      - 0.0.0.0
      - --model
      - google/paligemma2-3b-pt-896
      - --limit-mm-per-prompt
      - 'image=1'
      - --trust-remote-code
      - --max-model-len
      - "8192"

```

It does NOT work, the issue is the same reported in the pull request here: https://github.com/vllm-project/vllm/pull/11142#issuecomment-2541342321
and is:
`ValueError: As of transformers v4.44, default chat template is no longer allowed, so you must provide a chat template if th

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vllm.core.block.interfaces.BlockAllocator.NoFreeBlocksError to old Mistral Model

**Link**: https://github.com/vllm-project/vllm/issues/11168
**State**: open
**Created**: 2024-12-13T09:06:59+00:00
**Comments**: 4
**Labels**: bug, unstale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.4.0
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.26.4
Libc version: glibc-2.35

Python version: 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.25-051525-generic-x86_64-with-glibc2.35
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
Address sizes:                   39 bits physical, 48 bits virtual
Byte Or

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: ValueError: could not broadcast input array from shape (513,) into shape (512,)

**Link**: https://github.com/vllm-project/vllm/issues/8432
**State**: open
**Created**: 2024-09-12T23:40:44+00:00
**Comments**: 9
**Labels**: bug, stale

### Description

### Your current environment

Collecting environment information...
/home/miniconda3/envs/vllm/lib/python3.12/site-packages/torch/cuda/init.py:128: UserWarning: CUDA initialization: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 2: out of memory (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
return torch._C._cuda_getDeviceCount() > 0
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.12.4 | packaged by Anaconda, Inc. | (main, Jun 18 2024, 15:12:24) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.153.1-microsoft-standard-WSL2-x86_64-with-glibc2.35
Is CUDA available: False
CUDA runtime version

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Error with structured output inference after upgrade 0.6.2->0.6.3

**Link**: https://github.com/vllm-project/vllm/issues/9462
**State**: open
**Created**: 2024-10-17T12:35:39+00:00
**Comments**: 2
**Labels**: bug, structured-output, unstale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
ollecting environment information...
/opt/conda/lib/python3.11/site-packages/vllm/connections.py:8: RuntimeWarning: Failed to read commit hash:
No module named 'vllm._version'
  from vllm.version import __version__ as VLLM_VERSION
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.11.8 | packaged by conda-forge | (main, Feb 16 2024, 20:53:32) [GCC 12.3.0] (64-bit runtime)
Python platform: Linux-6.1.109-118.189.amzn2023.x86_64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A10G
Nvidia drive

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: unsloth/Llama-3.3-70B-Instruct-bnb-4bit can't work on vllm 0.6.4.post1

**Link**: https://github.com/vllm-project/vllm/issues/11010
**State**: open
**Created**: 2024-12-09T08:56:39+00:00
**Comments**: 7
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

I'm using following command to load Llama3.3 on one GPU:
```
CUDA_VISIBLE_DEVICES=1  python3 -m vllm.entrypoints.openai.api_server --model unsloth/Llama-3.3-70B-Instruct-bnb-4bit            --host 0.0.0.0 --port 8081  --seed 42 --trust-remote-code --enable-chunked-prefill            --tensor-parallel-size 1 --max-model-len 1024
```

Got error:
```
Loading safetensors checkpoint shards:   0% Completed | 0/8 [00:00<?, ?it/s]
ERROR 12-09 00:49:29 engine.py:366] 'layers.0.mlp.down_proj.weight.absmax'
ERROR 12-09 00:49:29 engine.py:366] Traceback (most recent call last):
ERROR 12-09 00:49:29 engine.py:366]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 357, in run_mp_engine
ERROR 12-09 00:49:29 engine

[... truncated for brevity ...]

---

## Issue #N/A: ExLlamaV2: exl2 support

**Link**: https://github.com/vllm-project/vllm/issues/3203
**State**: open
**Created**: 2024-03-05T14:54:03+00:00
**Comments**: 38
**Labels**: feature request, stale

### Description

If is possible ExLlamaV2 is a very fast and good library to Run [LLM](https://mlabonne.github.io/blog/posts/ExLlamaV2_The_Fastest_Library_to_Run%C2%A0LLMs.html)

[ExLlamaV2 Repo](https://github.com/turboderp/exllamav2)

---

## Issue #N/A: vLLM's V1 Engine Architecture

**Link**: https://github.com/vllm-project/vllm/issues/8779
**State**: open
**Created**: 2024-09-24T18:25:22+00:00
**Comments**: 14
**Labels**: RFC, keep-open

### Description

This issues describes the high level directions that "create LLM Engine V1". We want the design to be as transparent as possible and created this issue to track progress and solicit feedback. 

Goal:
* The new engine will be simple and performant. We found the first iteration of the engine to be simple, the multistep engine to be performant, but we want best of the both worlds. For it to be performat, we want to **minimize GPU idle time**. 
* The new architecture will be extensible and modular. We found the current codebase becoming difficult to extend and add new features (both production and experimental features) due to the hard tangling of different features. In the new design, features should be compatible with each other.
* Tech debts will be cleaned up. We will remove optimizations that compromise code readability. We will also redo ad-hoc implementations to support certain features/models. 

Non-goals, the following are important but orthogonal:
* Optimize GPU time/kern

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Encoder/decoder models & feature compatibility

**Link**: https://github.com/vllm-project/vllm/issues/7366
**State**: open
**Created**: 2024-08-09T15:03:54+00:00
**Comments**: 17
**Labels**: RFC, unstale

### Description

## Motivation <a href="#user-content-motivation" id="motivation">#</a>

There is significant interest in vLLM supporting encoder/decoder models. Issues #187  and #180 , for example, request encoder/decoder model support. As a result encoder/decoder support was recently introduced to vLLM via the following three PRs:

* #4837 
* #4888 
* #4942 

These three PRs make encoder/decoder model inference possible; however, they leave more to be desired in terms of (1) parity between vLLM's decoder-only & encoder/decoder request processing pipelines with respect to feature support, and (2) the number of encoder/decoder models which are supported.

The ask for the vLLM community is to contribute PRs which help bring vLLM encoder/decoder functionality to a similar level of maturity as that of vLLM's decoder-only functionality.

## Proposed changes <a href="#user-content-proposed-changes" id="proposed-changes">#</a>

The support matrix below summarizes which encoder/decoder models ha

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Support Inflight quantization: load as 8bit quantization.

**Link**: https://github.com/vllm-project/vllm/issues/11655
**State**: open
**Created**: 2024-12-31T08:42:16+00:00
**Comments**: 20
**Labels**: feature request, unstale

### Description

### 🚀 The feature, motivation and pitch

VLLM supports [4bit inflight quantification](https://docs.vllm.ai/en/stable/quantization/bnb.html#inflight-quantization-load-as-4bit-quantization), but does not support 8bit, 8bit speed is faster than 4bit, request support for support.


### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: Multi-Node Online Inference on TPUs Failing

**Link**: https://github.com/vllm-project/vllm/issues/12179
**State**: open
**Created**: 2025-01-17T23:38:35+00:00
**Comments**: 5
**Labels**: bug, tpu, ray, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
root@t1v-n-4d36f9a1-w-0:/workspace/vllm# python collect_env.py
INFO 01-17 23:21:42 __init__.py:179] Automatically detected platform tpu.
Collecting environment information...
PyTorch version: 2.6.0.dev20241126+cpu
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Debian GNU/Linux 11 (bullseye) (x86_64)
GCC version: (Debian 10.2.1-6) 10.2.1 20210110
Clang version: Could not collect
CMake version: version 3.31.4
Libc version: glibc-2.31

Python version: 3.10.15 (main, Oct 17 2024, 02:58:23) [GCC 10.2.1 20210110] (64-bit runtime)
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

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: stuck at  "generating GPU P2P access cache in /home/luban/.cache/vllm/gpu_p2p_access_cache_for_0,1.json"

**Link**: https://github.com/vllm-project/vllm/issues/8735
**State**: open
**Created**: 2024-09-23T09:59:54+00:00
**Comments**: 10
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
python collect_env.py 
Collecting environment information...
2024-09-23 17:57:46.577274: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-09-23 17:57:46.594737: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-09-23 17:57:46.616458: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-09-23 17:57:46.622847: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to regi

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Provide pre-built CPU docker image

**Link**: https://github.com/vllm-project/vllm/issues/10919
**State**: open
**Created**: 2024-12-05T06:41:08+00:00
**Comments**: 4
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

Hi thanks for the lib! Currently it seems that https://docs.vllm.ai/en/v0.6.1/getting_started/cpu-installation.html requires build the docker image by oneself, thus it would be great to have a prebuilt one (probably by CI).

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: Very slow guided decoding with Outlines backend since v0.6.5

**Link**: https://github.com/vllm-project/vllm/issues/12005
**State**: open
**Created**: 2025-01-13T11:06:36+00:00
**Comments**: 7
**Labels**: bug, structured-output

### Description

### Your current environment

<details>
<summary>Output of `pip list`</summary>
Package                           Version
--------------------------------- -------------
absl-py                           2.1.0
accelerate                        1.1.1
aiofiles                          23.2.1
aiohappyeyeballs                  2.4.4
aiohttp                           3.11.9
aiohttp-cors                      0.7.0
aiosignal                         1.3.1
airportsdata                      20241001
annotated-types                   0.7.0
anyio                             4.6.2.post1
astor                             0.8.1
attrs                             24.2.0
bert-score                        0.3.13
bitsandbytes                      0.44.1
blake3                            1.0.1
cachetools                        5.5.0
certifi                           2024.8.30
charset-normalizer                3.4.0
chex                              0.1.87
click                       

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Ray memory leak

**Link**: https://github.com/vllm-project/vllm/issues/4241
**State**: open
**Created**: 2024-04-21T14:06:32+00:00
**Comments**: 14
**Labels**: bug, ray, stale

### Description

### Your current environment

```text
PyTorch version: 2.1.2+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: Could not collect
CMake version: version 3.29.2
Libc version: glibc-2.31

Python version: 3.11.3 (main, Apr 19 2024, 17:22:27) [GCC 9.4.0] (64-bit runtime)
Python platform: Linux-5.4.0-177-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 10.1.243
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA A40
GPU 1: NVIDIA A40
GPU 2: NVIDIA A40

Nvidia driver version: 535.161.08
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Byte Order:                         Little Endian
Address sizes:          

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

