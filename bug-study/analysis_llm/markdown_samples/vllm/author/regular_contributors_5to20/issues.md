# regular_contributors_5to20 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 5
- Closed Issues: 25

### Label Distribution

- bug: 15 issues
- stale: 8 issues
- usage: 5 issues
- feature request: 3 issues
- performance: 3 issues
- new-model: 1 issues
- structured-output: 1 issues
- help wanted: 1 issues
- good first issue: 1 issues
- misc: 1 issues

---

## Issue #N/A: [Bug]: AttributeError: 'Qwen2_5OmniConfig' object has no attribute 'num_attention_heads'

**Link**: https://github.com/vllm-project/vllm/issues/16645
**State**: open
**Created**: 2025-04-15T07:48:19+00:00
**Comments**: 2
**Labels**: bug, stale

### Description

### Your current environment

just look here:
[https://github.com/huggingface/transformers/issues/37515#issuecomment-2804126324](url)

### 🐛 Describe the bug

`System Info
root@445d74596699:/vllm-workspace# transformers-cli env

Copy-and-paste the text below in your GitHub issue and FILL OUT the two last points.

transformers version: 4.52.0.dev0
Platform: Linux-5.15.0-43-generic-x86_64-with-glibc2.35
Python version: 3.12.9
Huggingface_hub version: 0.30.2
Safetensors version: 0.5.3
Accelerate version: 1.5.2
Accelerate config: not found
DeepSpeed version: not installed
PyTorch version (GPU?): 2.6.0+cu124 (True)
Tensorflow version (GPU?): not installed (NA)
Flax version (CPU?/GPU?/TPU?): not installed (NA)
Jax version: not installed
JaxLib version: not installed
Using distributed or parallel set-up in script?:
Using GPU in script?:
GPU type: NVIDIA L20
`(base) root@node15:/disk2/Qwen2.5-Omni-7B# more docker-compose.yml
#version: '3.3'
services:

vllm
vllm-openai:
image: vllm/vllm-openai:

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: FlashMLA V1 with FP8 KV cache not yet supported!

**Link**: https://github.com/vllm-project/vllm/issues/18887
**State**: open
**Created**: 2025-05-29T07:48:59+00:00
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

## Issue #N/A: [Bug]: vllm v0.6.2 is crashed on multiple GPU

**Link**: https://github.com/vllm-project/vllm/issues/9225
**State**: closed
**Created**: 2024-10-10T05:46:02+00:00
**Closed**: 2024-10-10T16:08:36+00:00
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

INFO:     Started server process [11912]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
ERROR:    [Errno 98] error while attempting to bind on address ('0.0.0.0', 8080): address already in use
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.

My command to start vllm:
```
python3 -m vllm.entrypoints.openai.api_server --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
        --host 0.0.0.0 --port 8080  --seed 42 --trust-remote-code --disable-frontend-multiprocessing \
        --enable-chunked-prefill --tensor-parallel-size 2 --max-model-len 98304 >> "$LOG_FILE" 2>&1 &
```

If I change tensor-parallel-size from 2 to 1, no such issue.

docker image in use i

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: How to optimize long text input token？

**Link**: https://github.com/vllm-project/vllm/issues/13154
**State**: closed
**Created**: 2025-02-12T11:58:41+00:00
**Closed**: 2025-02-17T09:43:14+00:00
**Comments**: 1
**Labels**: usage

### Description

### Your current environment

The content of each group of conversations is stored as historical information, and the historical information + new questions are used as new input to ask questions. However, a problem will arise. As historical information accumulates, the time required for the first token increases. Is there any optimization solution for this?

for example:

 ```
history:  [{'role': 'user', 'content': '介绍下爱因斯坦'}]
history:  [{'role': 'user', 'content': '介绍下爱因斯坦'}, {'role': 'assistant', 'content': '爱因斯坦是著名的理论物理学家，最著名的是他的相对论.'}]
history:  [{'role': 'user', 'content': '介绍下爱因斯坦'}, {'role': 'assistant', 'content': '爱因斯坦是著名的理论物理学家，最著名的是他的相对论.'}, {'role': 'user', 'content': '你是谁?'}]
 ```


### How would you like to use vllm

I want to run inference of a [specific model](put link here). I don't know how to integrate it with vllm.


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corne

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: error `is not a multimodal model` when serving `Qwen/Qwen3-8B` connected to `gr.load_chat(...)`

**Link**: https://github.com/vllm-project/vllm/issues/19144
**State**: open
**Created**: 2025-06-04T13:41:26+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

recent vllm nightly 0.9.0

### 🐛 Describe the bug

In combination with this problem (related to "text_encoded") https://github.com/gradio-app/gradio/issues/11331

I get the error when using `gr.load_chat(...)` to send extremely long prompt and then extremely short prompt into `vllm serve`-d `Qwen/Qwen3-8B` model. Not sure if the problem is in Gradio or in vllm impl of OpenAI server
 
```

ERROR 06-04 11:39:38 [serving_chat.py:199] Error in preprocessing prompt inputs                                                                                                                                                           
ERROR 06-04 11:39:38 [serving_chat.py:199] Traceback (most recent call last):                                                                                                                                                             
ERROR 06-04 11:39:38 [serving_chat.py:199]   File "/mnt/fs/venv_cu126_py312/lib/python3.12/site-packages/vll

[... truncated for brevity ...]

---

## Issue #N/A: [New Model]: NV-Embed-v2

**Link**: https://github.com/vllm-project/vllm/issues/9868
**State**: closed
**Created**: 2024-10-31T05:07:05+00:00
**Closed**: 2025-07-13T02:15:56+00:00
**Comments**: 5
**Labels**: new-model, stale

### Description

### The model to consider.

https://huggingface.co/nvidia/NV-Embed-v2

### The closest model vllm already supports.

_No response_

### What's your difficulty of supporting the model you want?

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: v1 engine with full cuda graph option outputs garbage

**Link**: https://github.com/vllm-project/vllm/issues/18533
**State**: closed
**Created**: 2025-05-22T07:19:36+00:00
**Closed**: 2025-06-04T08:10:24+00:00
**Comments**: 4
**Labels**: bug

### Description

### Your current environment

x

### 🐛 Describe the bug

refers to https://github.com/vllm-project/vllm/issues/18520

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: Adopt Colossal Inference Features (55% speedup over vLLM)

**Link**: https://github.com/vllm-project/vllm/issues/5085
**State**: closed
**Created**: 2024-05-28T10:12:41+00:00
**Closed**: 2024-11-27T02:07:59+00:00
**Comments**: 2
**Labels**: feature request, stale

### Description

### 🚀 The feature, motivation and pitch

ColossalAI has been able to demonstrate an impressive speedup over vLLM in multi-GPU inference. With TP=2, batch size 64, input len 512, output len 256 - a 55% speedup can be observed. I believe vLLM could see a speedup if it was to adopt a more performant batched prefilling.

![image](https://github.com/vllm-project/vllm/assets/27340033/ad30d755-8683-4ecb-b1b7-52ee6216b6bd)

For reference, here is the continuous batching feature:

![image](https://github.com/vllm-project/vllm/assets/27340033/a5026956-d087-4a5e-b0aa-d7a4c19f73d9)


### Alternatives

_No response_

### Additional context

Blog post: https://hpc-ai.com/blog/colossal-inference
Source code: https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/inference

---

## Issue #N/A: [Bug]:

**Link**: https://github.com/vllm-project/vllm/issues/18889
**State**: closed
**Created**: 2025-05-29T07:49:03+00:00
**Closed**: 2025-05-29T08:31:41+00:00
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

## Issue #N/A: [Bug]: Compiling FSM index high memory && subprocess OOM

**Link**: https://github.com/vllm-project/vllm/issues/7332
**State**: closed
**Created**: 2024-08-09T03:10:36+00:00
**Closed**: 2025-06-27T19:59:20+00:00
**Comments**: 7
**Labels**: bug, structured-output

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

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.30.1
Libc version: glibc-2.35

Python version: 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-3.10.0-1160.71.1.el7.x86_64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA GeForce RTX 3090
GPU 1: NVIDIA GeForce RTX 3090

Nvidia driver version: 535.129.03
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                    x86_64
CPU

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: load llama 70B more than 10min， is that right？

**Link**: https://github.com/vllm-project/vllm/issues/10702
**State**: closed
**Created**: 2024-11-27T09:57:48+00:00
**Closed**: 2024-12-04T10:27:19+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

image: vllm/vllm-openai:latest

0.6.4.post1

H100 8GPU


### Model Input Dumps
```
INFO 11-27 02:01:37 api_server.py:585] vLLM API server version 0.6.4.post1
INFO 11-27 02:01:37 api_server.py:586] args: Namespace(subparser='serve', model_tag='/models/Meta-Llama-3.1-70B-Instruct', config='', host=None, port=8000, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, lora_modules=None, prompt_adapters=None, chat_template=None, response_role='assistant', ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_cert_reqs=0, root_path=None, middleware=[], return_tokens_as_token_ids=False, disable_frontend_multiprocessing=False, enable_auto_tool_choice=False, tool_call_parser=None, tool_parser_plugin='', model='/models/Meta-Llama-3.1-70B-Instruct', task='auto', tokenizer=None, skip_tokenizer_init=False, revision=None, code_revision=None, tokenizer_revision=None, tokeni

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vllm-cpu docker gguf: AttributeError: '_OpNamespace' '_C' object has no attribute 'ggml_dequantize'

**Link**: https://github.com/vllm-project/vllm/issues/8500
**State**: closed
**Created**: 2024-09-16T04:21:46+00:00
**Closed**: 2024-09-17T01:26:22+00:00
**Comments**: 3
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
INFO 09-16 04:16:36 importing.py:10] Triton not installed; certain GPU-related functions will not be available.
PyTorch version: 2.4.0+cpu
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0
Clang version: Could not collect
CMake version: version 3.30.3
Libc version: glibc-2.35

Python version: 3.10.12 (main, Jul 29 2024, 16:56:48) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.8.0-1015-aws-x86_64-with-glibc2.35
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
CPU op-mode

[... truncated for brevity ...]

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

## Issue #N/A: [Bug]: AttributeError: 'Int8Params' object has no attribute 'bnb_shard_offsets', It seems that vllm's bnb prequantification support for cls models is not yet complete.

**Link**: https://github.com/vllm-project/vllm/issues/11807
**State**: closed
**Created**: 2025-01-07T12:59:06+00:00
**Closed**: 2025-06-01T02:14:40+00:00
**Comments**: 5
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



 It seems that vllm's bnb prequantification support for cls models is not yet complete.

This problem occurs only in the score layer of any bnb format cls model.


```python
outputs = llm.classify(
    all_prompts,
    use_tqdm=True,
)

```








```python
Processed prompts:   0%|          | 0/6 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-11-d3f2b71f59e1> in <cell line: 7>()
      5 
      6 all_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in data["input_ids"].values]
----> 7 outputs = llm.classify(
      8     all_prompt

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: How to get "num_gpu_blocks" in V1？

**Link**: https://github.com/vllm-project/vllm/issues/15538
**State**: open
**Created**: 2025-03-26T09:40:16+00:00
**Comments**: 9
**Labels**: help wanted, good first issue, usage

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How would you like to use vllm

In V0, I can get "num_gpu_blocks" through "llm.llm_engine.cache_config.num_gpu_blocks". 

But in V1, LLM and EngineCore are in different processes. How can I get "num_gpu_blocks"?

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: mllama AssertionError during kv cache profiling

**Link**: https://github.com/vllm-project/vllm/issues/13929
**State**: closed
**Created**: 2025-02-26T22:19:48+00:00
**Closed**: 2025-04-07T04:07:16+00:00
**Comments**: 4
**Labels**: bug

### Description

### Your current environment

Repro command below.

### 🐛 Describe the bug

Attempting to serve `meta-llama/Llama-3.2-11B-Vision-Instruct` with recent vLLM (>=v0.7.3), results in the error below during the execution of `determine_num_available_blocks()` during bootup

```
$ vllm serve meta-llama/Llama-3.2-11B-Vision-Instruct --max-num-seqs 8
```

```
Traceback (most recent call last):
  File "/opt/vllm/lib64/python3.12/site-packages/vllm/engine/multiprocessing/engine.py", line 400, in run_mp_engine
    engine = MQLLMEngine.from_engine_args(engine_args=engine_args,
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/vllm/lib64/python3.12/site-packages/vllm/engine/multiprocessing/engine.py", line 125, in from_engine_args
    return cls(ipc_path=ipc_path,
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/vllm/lib64/python3.12/site-packages/vllm/engine/multiprocessing/engine.py", line 77, in __init__
    self.engine = LLMEngine(*args, **kwargs)
                  ^^

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vllm0.6.3.post1  7B model can not use cmd vllm.entrypoints.openai.api_server on wsl

**Link**: https://github.com/vllm-project/vllm/issues/10116
**State**: closed
**Created**: 2024-11-07T10:24:57+00:00
**Closed**: 2024-11-16T14:16:00+00:00
**Comments**: 43
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.4.0
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 24.04.1 LTS (x86_64)
GCC version: (Ubuntu 13.2.0-23ubuntu4) 13.2.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.39

Python version: 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 13:27:36) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.153.1-microsoft-standard-WSL2-x86_64-with-glibc2.39
Is CUDA available: True
CUDA runtime version: 12.6.77
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 4090
Nvidia driver version: 560.94
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.5.1
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.5.1
/usr/lib/x86_64-linux-gnu/libcudnn_cn

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: INT4 quantisation does not lead to any observable throughput increase 

**Link**: https://github.com/vllm-project/vllm/issues/8006
**State**: closed
**Created**: 2024-08-29T16:57:41+00:00
**Closed**: 2024-08-29T19:48:37+00:00
**Comments**: 3
**Labels**: performance

### Description

### Proposal to improve performance

_No response_

### Report of performance regression

I have been using vLLM with prefix caching to optimise inference in cases where majority of operations are pre-fills with large shared prefix. Specially, most of the prompts are 130 tokens in size with 90% of it is a shared system prompt. 
The decode is only phase is only one token.
There benchmark is a 100000 prompts (`formatted_prompts` below) executed via generate:
```python
from outlines import models, generate
llm = LLM("meta-llama/Meta-Llama-3-8B-Instruct", enable_prefix_caching=True)
sampling_params = SamplingParams(temperature=0.5, top_p=0.2, max_tokens=1)
model = models.VLLM(llm)
generator = generate.choice(model, ["yes", "no"])
predictions = generator(formatted_prompts, sampling_params=sampling_params)
``` 

When I use INT4 quantised model `neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w4a16` I observe no speed-up in inference. 
Since my use cases highly leverages prefix ca

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Different default value for temperature in SamplingParams and ChatCompletionRequest

**Link**: https://github.com/vllm-project/vllm/issues/10930
**State**: closed
**Created**: 2024-12-05T14:18:25+00:00
**Closed**: 2024-12-16T08:15:41+00:00
**Comments**: 3
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

The default value for `temperature` in `SamplingParams` is `1.0`. https://github.com/vllm-project/vllm/blob/571da8fc431ec36427ee1034a7779b23229b015e/vllm/sampling_params.py#L176

The default value for `temperature` in `ChatCompletionRequest` is `0.7`. https://github.com/vllm-project/vllm/blob/571da8fc431ec36427ee1034a7779b23229b015e/vllm/entrypoints/openai/protocol.py#L173

This can lead to inconsistencies between online and offline inference results.

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Speed between gptq w4a16 and awq w4a16?

**Link**: https://github.com/vllm-project/vllm/issues/1853
**State**: closed
**Created**: 2023-11-30T07:20:55+00:00
**Closed**: 2024-04-04T07:42:55+00:00
**Comments**: 10

### Description

Hi, I am wondering the implementation of gptq w4a16(exllama) and awq w4a16(llm-awq), which is faster?

It seems the mathematical computation is similar between the two, so can these two share the same copy of cuda function?

Hoping for your reply, thank you

---

## Issue #N/A: [Usage]: Can I get the streaming output when using offline inference?

**Link**: https://github.com/vllm-project/vllm/issues/5862
**State**: closed
**Created**: 2024-06-26T09:08:45+00:00
**Closed**: 2024-07-02T01:34:42+00:00
**Comments**: 4
**Labels**: usage

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How would you like to use vllm

I want to get the streaming output when using offline inference. But I can't find the 'stream' switch.


---

## Issue #N/A: [Usage]: Parameters for improving throughput of deepseek v3 

**Link**: https://github.com/vllm-project/vllm/issues/11600
**State**: closed
**Created**: 2024-12-29T09:02:28+00:00
**Closed**: 2025-05-06T02:09:21+00:00
**Comments**: 5
**Labels**: usage, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```
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
Python platform: Linux-5.15.0-130-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA H200
GPU 1: NVIDIA H200
GPU 2: NVIDIA H200
GPU 3: NVIDIA H200
GPU 4: NVIDIA H200
GPU 5: NVIDIA H200
GPU 6: NVIDIA H200
GPU 7: NVIDIA H200

Nvidia driver version: 550.54.15
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNP

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: CI - Split up "Models Test" and "Vision Language Models Test"

**Link**: https://github.com/vllm-project/vllm/issues/7439
**State**: closed
**Created**: 2024-08-12T20:01:31+00:00
**Closed**: 2024-10-29T17:40:25+00:00
**Comments**: 6
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Takes 1 hour+ on CI compared to others, which take <~30 min. Thus, ends up being a bottleneck

So, should be split up similar to kernels

CC: @khluu 


---

## Issue #N/A: [Performance]: Unified flashattn kernel not outperforming current one

**Link**: https://github.com/vllm-project/vllm/issues/10707
**State**: closed
**Created**: 2024-11-27T11:09:32+00:00
**Closed**: 2025-03-20T17:16:43+00:00
**Comments**: 2
**Labels**: performance, stale

### Description

### Proposal to improve performance

While working on https://github.com/vllm-project/vllm/pull/9291/, I experimented with unifying prefills and decodes processing in a single forward call (through the `flash_attn_varlen_func` API), while currently we separate the two by "splitting" the flattened 1d tokens tensor (size n_prefills+n_decodes). 
The unification is meaningful when chunked prefill is enabled, as it will allow mixed prefill-decodes batches to be scheduled. 

Following the change, @sroy745 found no speedup in his benchmarks with the new version using a single kernel call, which is quite baffling.

I believe we should give the fused version another try in a separate PR, investigating the causes of the unexpected slowdown, as in theory this should be a low-hanging fruit in terms of performance optimization.

The plan would be to rebase the changes introduced prior to this commit https://github.com/vllm-project/vllm/pull/9291/commits/2a9d8f1e48646eb79431c72b608bb2f4453266

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Breaking Down Single Process into Asynchronous Tokenization, Model Inference, and Detokenization for Enhanced GPU Utilization

**Link**: https://github.com/vllm-project/vllm/issues/8295
**State**: closed
**Created**: 2024-09-09T12:23:22+00:00
**Closed**: 2024-09-09T23:34:45+00:00
**Comments**: 1
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Feature Proposal:
I would like to request an optimization feature where tokenization, model inference, and detokenization are performed asynchronously in separate processes, leading to a significant improvement in GPU utilization. This setup would enable parallel execution of these tasks, minimizing idle GPU time between the phases of the pipeline and increasing overall throughput.

Motivation:
Currently, these three stages (tokenization, inference, detokenization) are typically handled sequentially, which results in underutilization of the GPU during the tokenization and detokenization phases. By separating these stages into asynchronous, tri-process collaboration, the GPU could be used more efficiently, especially for large models where tokenization and detokenization overhead becomes non-negligible.

Pitch:
Implementing this feature could greatly enhance the performance of vLLM for high-throughput applications, leading to faster infere

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: sample_params

**Link**: https://github.com/vllm-project/vllm/issues/3773
**State**: closed
**Created**: 2024-04-01T10:06:02+00:00
**Closed**: 2024-04-02T06:25:24+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### 🐛 Describe the bug

When I start openai server with `openai = = 1.12.0` i run into some problems when I want to pass in some unique sample Param, and it prompts me the error.
But the old version of `openai=0.27.8` wouldn't have such a problem. Will vLLM consider adapting the server's incoming parameters to the new version of openai in the future?
``````---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[1], line 12
      5 openai_api_base = "http://localhost:8000/v1" # "http://39.98.81.39:6005/v1"
      7 client = OpenAI(
      8     api_key=openai_api_key,
      9     base_url=openai_api_base,
     10 )
---> 12 chat_response = client.chat.completions.create(
     13     # model="/root/autodl-tmp/model/Qwen1___5-72B-Chat-GPTQ-Int4",
     14     model="Qwen1.5-14B-Chat",
     15  

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: page attention v2

**Link**: https://github.com/vllm-project/vllm/issues/3929
**State**: closed
**Created**: 2024-04-09T07:40:04+00:00
**Closed**: 2024-04-10T06:27:48+00:00
**Comments**: 1
**Labels**: misc

### Description

### Anything you want to discuss about vllm.

Can VLLM's page attention v2 be understood as incorporating the implementation of flash decoding

---

## Issue #N/A: [Usage]: Failure to Init Qwen2.5-VL-7B-Instruct with inflight bnb quantization

**Link**: https://github.com/vllm-project/vllm/issues/12899
**State**: closed
**Created**: 2025-02-07T13:20:47+00:00
**Closed**: 2025-02-07T13:27:31+00:00
**Comments**: 0
**Labels**: usage

### Description

### Your current environment

```text
docker vllm-openai:v0.7.2 with latest transformers installed
```


### How would you like to use vllm

Hi and I'm trying to launch qwen2.5-vl-7b-instruct in bnb inflight quanization but got error
```
 (AssertionError: param_data.shape == loaded_weight.shape) 
```

I was able to run this model at full precision with docker. Below is how I init the full precision one:
```
sudo docker run --runtime nvidia --gpus '"device=0,1"' --ipc=host -p 18434:8000 \
   -v hf_cache:/root/.cache/huggingface -d \
   --name qwen2.5-vl-7b \
   --entrypoint "python3" qwen-vl-fixed \ # I installed new transformer and commited into a new image. 
   -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-VL-7B-Instruct \
   --tensor-parallel-size 2 --trust-remote-code --max-model-len 18000 --dtype half
```

When I added `--quantization bitsandbytes --load-format bitsandbytes` into the docker command, the launch of the model in bnb 4bit inflight quantization failed.

Bel

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Drop use of pickle where possible

**Link**: https://github.com/vllm-project/vllm/issues/12055
**State**: closed
**Created**: 2025-01-14T20:57:29+00:00
**Closed**: 2025-05-15T02:09:21+00:00
**Comments**: 2
**Labels**: bug, stale

### Description


### 🐛 Describe the bug

vLLM uses pickle for serialization and sometimes also sends serialized objects over a local zeromq unix socket. Using pickle and any sort of network communication is a known dangerous combination, as it's an easy way to open a vulnerability to remote code execution on a host when a host deserializes pickled data.

There have already been some changes to use [msgpack](https://github.com/msgpack/msgpack-python) instead. This issue is open to track the conversion away from using pickle where possible.

Thank you to @avilum who responsibly reported this as a security report. We discussed it and concluded we did not see any path to exploit at this time. However, it is still an important weakness that should be addressed to improve vLLM security.

---

## Issue #N/A: [feature on nm-vllm] Sparse Inference with weight only int8 quant

**Link**: https://github.com/vllm-project/vllm/issues/3307
**State**: closed
**Created**: 2024-03-11T03:39:31+00:00
**Closed**: 2024-11-29T02:07:54+00:00
**Comments**: 2
**Labels**: stale

### Description

Can sparsity and quantization be used simultaneously to further improve inference speed? Do you have any plans in this regard? Looking forward to your reply @robertgshaw2-neuralmagic 

---

