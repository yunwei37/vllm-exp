# author_association_NONE - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 8
- Closed Issues: 22

### Label Distribution

- bug: 16 issues
- stale: 11 issues
- usage: 6 issues
- installation: 2 issues
- performance: 1 issues
- help wanted: 1 issues
- new-model: 1 issues
- feature request: 1 issues
- documentation: 1 issues

---

## Issue #N/A: [Bug]: Vllm api server does not receive supported parameter `truncate_prompt_tokens`

**Link**: https://github.com/vllm-project/vllm/issues/6890
**State**: closed
**Created**: 2024-07-29T05:56:58+00:00
**Closed**: 2024-12-01T02:14:32+00:00
**Comments**: 16
**Labels**: bug, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### 🐛 Describe the bug

I used the openai compatible server deployed with vllm:
```bash
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct--host 127.0.0.1 --port 8077 --enforce-eager --gpu-memory-utilization 0.8 --swap-space 32
```
When I send a request with the following snippet (openai client):
```python
openai_api_key="EMPTY"
openai_api_base="http://localhost:8077/v1"
from openai import OpenAI
client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
)
models = client.models.list()
model = models.data[0].id
client.chat.completions.create(
    messages=[
            {
            "role": "user",
            "content": "How are you today?"
            },
        ],
        model=model,
        max_tokens=128,
        temperature=0.0,
        seed=42,
        extra_body=dict(
            truncate_prompt_toke

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: qwen2.5 function calling,ChatLanguageModel is ok, but in StreamingChatLanguageModel,the logger report error

**Link**: https://github.com/vllm-project/vllm/issues/9184
**State**: closed
**Created**: 2024-10-09T07:34:57+00:00
**Closed**: 2025-02-07T01:59:36+00:00
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

![image](https://github.com/user-attachments/assets/039317df-453c-47e2-a449-297df451503b)


### 🐛 Describe the bug

qwen2.5 function calling,ChatLanguageModel is ok, but in StreamingChatLanguageModel,the logger report error

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: How to improve concurrent processing capacity

**Link**: https://github.com/vllm-project/vllm/issues/14513
**State**: closed
**Created**: 2025-03-09T08:47:52+00:00
**Closed**: 2025-03-10T03:55:45+00:00
**Comments**: 2
**Labels**: usage

### Description

### Your current environment

vllm version: 0.6.1.post2

When I was testing the performance at 200 concurrent users, I found that vLLM can handle up to 100 requests at most.
![Image](https://github.com/user-attachments/assets/941c2fcf-30fd-4e67-a74f-46a43cdfa571)
After each request is processed, a new request will be added, but the maximum number of requests is 100.

Below is my startup script.
![Image](https://github.com/user-attachments/assets/8f9857b9-81e5-401b-891d-8cae195733ba)

So，my question is how to increase the number of concurrent connections from 100 to 200 or more？

Thank you。

### How would you like to use vllm

I want to increase the number of concurrent connections from 100 to 200 or more.

---

## Issue #N/A: [Bug]: RuntimeError in gptq_marlin_24_gemm

**Link**: https://github.com/vllm-project/vllm/issues/8654
**State**: closed
**Created**: 2024-09-20T07:32:12+00:00
**Closed**: 2025-05-17T02:09:39+00:00
**Comments**: 9
**Labels**: bug, stale

### Description

### Your current environment

python 3.8
L20*4
vllm 0.5.4

### Model Input Dumps

_No response_

### 🐛 Describe the bug

$python -m vllm.entrypoints.api_server --model='/mntfn/yanyi/Qwen2-7B-Instruct_24_w4a16/stage_quantization' --max-model-len=16000 --tensor-parallel-size=4 --use-v2-block-manager  --enable-prefix-caching 

[rank0]:   File "/opt/conda/lib/python3.8/site-packages/vllm/engine/async_llm_engine.py", line 735, in from_engine_args
[rank0]:     engine = cls(
[rank0]:   File "/opt/conda/lib/python3.8/site-packages/vllm/engine/async_llm_engine.py", line 631, in __init__
[rank0]:     self.engine = self._init_engine(*args, **kwargs)
[rank0]:   File "/opt/conda/lib/python3.8/site-packages/vllm/engine/async_llm_engine.py", line 830, in _init_engine
[rank0]:     return engine_class(*args, **kwargs)
[rank0]:   File "/opt/conda/lib/python3.8/site-packages/vllm/engine/async_llm_engine.py", line 267, in __init__
[rank0]:     super().__init__(*args, **kwargs)
[rank0

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: how to create envs.py file for build on CPU machine?

**Link**: https://github.com/vllm-project/vllm/issues/12649
**State**: closed
**Created**: 2025-02-01T15:57:48+00:00
**Closed**: 2025-05-07T16:34:34+00:00
**Comments**: 3
**Labels**: installation, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```
Here's the output of collect_env.py file:

WARNING 02-01 09:54:47 _custom_ops.py:20] Failed to import from vllm._C with ImportError('libcudart.so.12: cannot open shared object file: No such file or directory')
Collecting environment information...
PyTorch version: 2.5.1+cpu
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Ubuntu 24.04.1 LTS (x86_64)
GCC version: (Ubuntu 12.3.0-17ubuntu1) 12.3.0
Clang version: Could not collect
CMake version: version 3.31.4
Libc version: glibc-2.39

Python version: 3.12.3 (main, Jan 17 2025, 18:03:48) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.39
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: No CUDA
Nvidia driver version: No CUDA
cuDNN version: No CUDA
HIP runtime version: N/A
MIOpen runtime version:

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Accessing stat_logger from AsyncLLMEngine

**Link**: https://github.com/vllm-project/vllm/issues/3736
**State**: closed
**Created**: 2024-03-29T22:26:35+00:00
**Closed**: 2024-11-28T02:07:16+00:00
**Comments**: 2
**Labels**: usage, stale

### Description

### Your current environment

```text
PyTorch version: 2.1.2+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: 14.0.0-1ubuntu1.1
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-1051-aws-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.3.107
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A10G
GPU 1: NVIDIA A10G
GPU 2: NVIDIA A10G
GPU 3: NVIDIA A10G
GPU 4: NVIDIA A10G
GPU 5: NVIDIA A10G
GPU 6: NVIDIA A10G
GPU 7: NVIDIA A10G

Nvidia driver version: 535.104.12
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.0.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.0.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9.0.0
/

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: the ttft and latency for each request calculated by benchmark_serving.py seems abnormal

**Link**: https://github.com/vllm-project/vllm/issues/4252
**State**: closed
**Created**: 2024-04-22T03:53:26+00:00
**Closed**: 2024-04-23T04:23:33+00:00
**Comments**: 3
**Labels**: bug

### Description

### Your current environment

```text
The output of `python collect_env.py`
```
Collecting environment information...
PyTorch version: 2.2.1+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.27.6
Libc version: glibc-2.35

Python version: 3.10.12 (main, Jun 11 2023, 05:26:28) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-52-shopee-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.2.140
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A30
GPU 1: NVIDIA A30

Nvidia driver version: 535.104.12
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.5
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.9.5
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.9.5
/usr/lib/x86_64-linux-

[... truncated for brevity ...]

---

## Issue #N/A: First tpot/itl is too long?

**Link**: https://github.com/vllm-project/vllm/issues/15106
**State**: open
**Created**: 2025-03-19T07:52:03+00:00
**Comments**: 4
**Labels**: performance

### Description

I discovered a somewhat strange (or is it reasonable?) phenomenon: For example, I started a vllm OpenAI server with chunked prefill=True and max_num_batched_tokens=2k, then sent an input of 5k tokens using the /chat endpoint with stream=True. The server first returned a token that includes the role but with an empty content, taking 0.9 seconds (on my hardware and model, the prefill time for 2k tokens is 0.9 seconds), which is counted as ttft. Then, it returned a second token with non-empty content, taking 1.3 seconds (on my hardware and model, the prefill time for 3k tokens is 1.3 seconds), which is counted as the first tpot/itl. Next, it returned a third token with non-empty content, taking 0.02 seconds (on my hardware and model, the decode time is 0.02 seconds), and all subsequent tokens took 0.02 seconds each. This is inconsistent with the tpot/itl I usually observe in benchmarks (which are all 0.02 seconds), so I traced it down to the implementation of the chat interface:

https://

[... truncated for brevity ...]

---

## Issue #N/A: [New Model]: meta-llama/Llama-Guard-3-1B

**Link**: https://github.com/vllm-project/vllm/issues/9294
**State**: closed
**Created**: 2024-10-11T18:26:49+00:00
**Closed**: 2024-10-24T05:05:50+00:00
**Comments**: 5
**Labels**: help wanted, new-model

### Description

### The model to consider.

meta-llama/Llama-Guard-3-1B

### The closest model vllm already supports.

meta-llama/Llama-Guard-3-8B

### What's your difficulty of supporting the model you want?

Currently the model runs, but its outputs are completely random, so the same prompt can be safe or unsafe at any point. Setting the temperature to 0.0 makes EVERY prompt return safe.

My hunch is the issue comes from the model pruning:

Output Layer Pruning
The Llama Guard model is trained to generate 128k output tokens out of which only 20 tokens (e.g. safe, unsafe, S, 1,...) are used. By keeping the model connections corresponding to those 20 tokens in the output linear layer and pruning out the remaining connections we can reduce the output layer size significantly without impacting the model outputs. Using output layer pruning, we reduced the output layer size from 262.6M parameters (2048x128k) to 40.96k parameters (2048x20), giving us a total savings of 131.3MB with 4-bit quantized wei

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: kv_cache incorrect with cuda graph batch inputs vllm0.6.3

**Link**: https://github.com/vllm-project/vllm/issues/20426
**State**: open
**Created**: 2025-07-03T08:32:47+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

vllm0.6.3  about 2024.OCT

### 🐛 Describe the bug

Due to some project constraints, I'm currently still using vLLM 0.6.3 and don't have time to upgrade for now. I've noticed that when CUDA Graph is enabled, the output becomes incorrect for some batches in multi-batch inputs. I constructed a batch with identical inputs, so theoretically, the outputs of each batch should be the same.

In my testing:
    Single-batch with CUDA Graph: output is correct
    Multi-batch without CUDA Graph (eager mode): output is also correct
    Multi-batch with CUDA Graph: after attention computation, the hidden_states across batches start to differ 
        unexpectedly
    Interestingly, when I forcibly set the kv_cache values to 1, the hidden_states become identical again after 
        attention.

Has anyone encountered this issue before? Are there any related issues or discussions about this? Thanks!

### Before submitting a new issue...

- [x] Make sure you already search

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vllm run bge-m3 error

**Link**: https://github.com/vllm-project/vllm/issues/17877
**State**: open
**Created**: 2025-05-09T03:48:20+00:00
**Comments**: 1
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

Deploy  local BGE-M3 using VLLM and start the command as follows：
python3 -m vllm.entrypoints.openai.api_server --served-model-name embed --model /app/bge-m3 --gpu-memory-utilization 0.8 --trust-remote-code --port 8080 --task embed 。

Service error: huggingface_hub.errors.HFValidationError: Repo id must be in the from 'repo_name' or 'namespace/repo_name': '/app/bge_m3'. usee 'repo_type' argument if needed。 
（This model can run and start normally without using VLLM for testing）

environment： 
VLLM 0.7.3
transformers 4.51.3
torch 2.5.1


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked

[... truncated for brevity ...]

---

## Issue #N/A: [Question] Usage with Audio Models?

**Link**: https://github.com/vllm-project/vllm/issues/2546
**State**: closed
**Created**: 2024-01-22T09:31:46+00:00
**Closed**: 2024-04-04T08:04:30+00:00
**Comments**: 1

### Description

I did some initial search but couldn't find any frameworks that provide the same functionality as vLLM for Audio models. Currently I'm running a Flask server for running a Distil-Whisper model , any leads for a more performant server will be great. Thanks.

---

## Issue #N/A: [Question] Is it possible to crop the KV cache with the currently supported operations?

**Link**: https://github.com/vllm-project/vllm/issues/720
**State**: closed
**Created**: 2023-08-09T17:11:09+00:00
**Closed**: 2024-03-08T10:37:07+00:00
**Comments**: 1

### Description

Title. 

---

## Issue #N/A: [Bug]: out-of-bound in attention.cu

**Link**: https://github.com/vllm-project/vllm/issues/9136
**State**: closed
**Created**: 2024-10-07T22:22:48+00:00
**Closed**: 2025-02-06T01:59:33+00:00
**Comments**: 2
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
I didn’t run the code, just manually reviewed it on my MacBook.
```

</details>


### Model Input Dumps

_No response_

### 🐛 Describe the bug

https://github.com/vllm-project/vllm/blob/c0d9a98d0c7182b73c2e7f88508e690a186bf0e3/csrc/rocm/attention.cu#L199-L225
https://github.com/vllm-project/vllm/blob/c0d9a98d0c7182b73c2e7f88508e690a186bf0e3/csrc/rocm/attention.cu#L264
https://github.com/vllm-project/vllm/blob/c0d9a98d0c7182b73c2e7f88508e690a186bf0e3/csrc/rocm/attention.cu#L352-L359
https://github.com/vllm-project/vllm/blob/c0d9a98d0c7182b73c2e7f88508e690a186bf0e3/csrc/rocm/attention.cu#L914-L917
https://github.com/vllm-project/vllm/blob/c0d9a98d0c7182b73c2e7f88508e690a186bf0e3/csrc/rocm/attention.cu#L960-L973

So, for `alibi_slopes[wg_start_head_idx + qhead_idx]` in line 358:
- `wg_start_head_idx = blockIdx.z * GQA_RATIO`
- `blockIdx.z = num_kv_heads`
- `GQA_RATIO 

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: How to use vllm serve for batch inference?

**Link**: https://github.com/vllm-project/vllm/issues/16494
**State**: closed
**Created**: 2025-04-11T16:03:51+00:00
**Closed**: 2025-04-11T22:17:25+00:00
**Comments**: 4
**Labels**: usage

### Description

### Your current environment

```Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.26.0
Libc version: glibc-2.35

Python version: 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-1045-azure-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A100 80GB PCIe
GPU 1: NVIDIA A100 80GB PCIe

Nvidia driver version: 535.86.10
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.9.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.9.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8.9.0


[... truncated for brevity ...]

---

## Issue #N/A: "attn_bias is not correctly aligned" on A100 for MPT-30B

**Link**: https://github.com/vllm-project/vllm/issues/795
**State**: closed
**Created**: 2023-08-18T21:59:05+00:00
**Closed**: 2023-08-23T08:44:23+00:00
**Comments**: 3
**Labels**: bug

### Description

Hello,

I saw a similar issue to this for MPT30B0-chat on H100, but I see the same error on A100 80Gb. Using vllm 0.1.3. Is there any workaround to fix this currently? It does happen for random prompt, so not straightforward to understand where it's coming from:

    96 │   │   │   │   prompt_template = PromptTemplate(input_variables=["text"] │
│    97 │   │   │   │   answer_chain = LLMChain(llm=self.llm , prompt=prompt_temp │
│    98 │   │   │   │                                                             │
│ ❱  99 │   │   │   │   response = answer_chain.run(query)                        │
│   100 │   │   │                                                                 │
│   101 │   │   │   else:                                                         │
│   102                                                                           │
│                                                                                 │
│ /root/miniconda3/envs/py311/lib/python3.11/site-pac

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Unable to use tensor parallel

**Link**: https://github.com/vllm-project/vllm/issues/20303
**State**: open
**Created**: 2025-07-01T06:18:54+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
Collecting environment information...
PyTorch version: 2.7.0+cu126
Is debug build: False
CUDA used to build PyTorch: 12.6
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (conda-forge gcc 13.3.0-2) 13.3.0
Clang version: Could not collect
CMake version: version 3.16.3
Libc version: glibc-2.31

Python version: 3.12.9 | packaged by conda-forge | (main, Mar  4 2025, 22:48:41) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-5.4.0-144-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.6.68
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: Tesla V100-SXM2-32GB
GPU 1: Tesla V100-SXM2-32GB
GPU 2: Tesla V100-SXM2-32GB
GPU 3: Tesla V100-SXM2-32GB
GPU 4: Tesla V100-SXM2-32GB
GPU 5: Tesla V100-SXM2-32GB
GPU 6: Tesla V100-SXM2-32GB
GPU 7: Tesla V100-SXM2-32GB

Nvidia driver version: 560.35.03
cuDNN versi

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: error when the concurrency reaches 10 with gptq4bits

**Link**: https://github.com/vllm-project/vllm/issues/9216
**State**: closed
**Created**: 2024-10-10T03:52:15+00:00
**Closed**: 2025-02-09T02:01:31+00:00
**Comments**: 2
**Labels**: bug, stale

### Description

### Your current environment



vllm 6.0



### Model Input Dumps

_No response_

### 🐛 Describe the bug



```Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/protocols/http/httptools_impl.py", line 411, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/middleware/proxy_headers.py", line 69, in __call__
    return await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/applications.py", line 1054, in __call__
    await super().__call__(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/applications.py", line 113, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/errors.py", line 187, in __call__
    raise exc
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/errors.py", line 165, in 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Can't create non-root user using vllm/vllm-openai:v0.8.1 as a base image

**Link**: https://github.com/vllm-project/vllm/issues/15359
**State**: closed
**Created**: 2025-03-23T15:02:23+00:00
**Closed**: 2025-03-24T12:53:12+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

U used to create a non-root user Docker image for vLLM and the following version was working fine up to v0.7.3:

``` 
FROM vllm/vllm-openai:v0.7.3

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_CACHE=/api/models
ENV HF_HOME=/api/models

RUN mkdir -p /api/models/

# RUN chmod +x /api/entrypoint.sh
RUN chmod 777 -R /api \
  && umask 000

EXPOSE 8000

# Set user and group
ARG user=appuser
ARG group=appuser
ARG uid=1000
ARG gid=1000
RUN groupadd -g ${gid} ${group}
RUN useradd -u ${uid} -g ${group} -s /bin/sh -m ${user}
    
RUN chown ${user}:${group} /api

# Switch to user
USER ${uid}:${gid}
```

Today was trying to make the same with v0.8.1 and got permissions errors like this:
`bash: /opt/venv/bin/vllm: /opt/venv/bin/python3: bad interpreter: Permission denied`

To my understanding, in addition, `/opt/venv/bin/python3` is actually a symbolic link pointing to `/root/.local/share/uv/python/cpython-3.12.9-linux-x86_64-gnu/bin/python3.12`

And even Dockefile modification th

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: how to use cpu offload gb with v1 engine my 4080 only has 16gb vram i want to use 64 of my system rams of gigabytes

**Link**: https://github.com/vllm-project/vllm/issues/16538
**State**: open
**Created**: 2025-04-12T16:54:10+00:00
**Comments**: 1
**Labels**: usage, stale

### Description

### Your current environment

<details>```text
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 24.10 (x86_64)
GCC version: (Ubuntu 14.2.0-4ubuntu2) 14.2.0
Clang version: 19.1.1 (1ubuntu1)
CMake version: version 3.31.6
Libc version: glibc-2.40

Python version: 3.12.7 (main, Feb  4 2025, 14:46:03) [GCC 14.2.0] (64-bit runtime)
Python platform: Linux-6.11.0-19-generic-x86_64-with-glibc2.40
Is CUDA available: True
CUDA runtime version: 12.6.77
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 4080
Nvidia driver version: 565.57.01
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                       32-bit, 64-bit
Address sizes:                        48 bits physical, 48 bits virtual
Byte Order:                           Little Endian
CPU(s): 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Decode n tokens gives different output for first seq position compared to decode 1 token

**Link**: https://github.com/vllm-project/vllm/issues/8783
**State**: closed
**Created**: 2024-09-24T22:00:05+00:00
**Closed**: 2025-01-24T01:58:43+00:00
**Comments**: 2
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
CMake version: version 3.27.6
Libc version: glibc-2.35

Python version: 3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.4.0-176-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.2.140
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

Nvidia driver version: 550.54.15
cuDNN version: Probably one of the followin

[... truncated for brevity ...]

---

## Issue #N/A: [Feature][Improvement]: Benchmarking with random conversation lengths

**Link**: https://github.com/vllm-project/vllm/issues/17780
**State**: open
**Created**: 2025-05-07T10:01:05+00:00
**Comments**: 2
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

### Background
I'm trying to figure out how many users my vLLM server can handle and how increases in number of users and average request rate per user over time effect the important metrics (ttft, tpot, itl). 
In this research, I'm assuming the users will create conversation of varying lengths.

I've started this research using the ShareGPT dataset, and I found that the number of users I was able to host for was a good amount above my expectations.

I started looking into how sampling was implemented for ShareGPT, and found that each SampleRequest is generated using only the two first turns of the conversation. (In ShareGPT many of the conversations are multi turn conversation).

### Feature
I would suggest a feature that makes the benchmark user able to control a maximum random length of conversations. Such that when sampling occurs, it isn't only 2 first turns of the conversation, but perhaps the first 6 turns.

### Why
Sampling only the firs

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Guided choice not working as expected

**Link**: https://github.com/vllm-project/vllm/issues/12225
**State**: open
**Created**: 2025-01-20T16:02:03+00:00
**Comments**: 11
**Labels**: usage

### Description

### Your current environment

```
PyTorch version: 2.5.1


Versions of relevant libraries:
[pip3] mypy-extensions==1.0.0
[pip3] numpy==1.26.4
[pip3] nvidia-ml-py==12.560.30
[pip3] onnxruntime==1.20.1
[pip3] pyzmq==26.2.0
[pip3] sentence-transformers==3.3.1
[pip3] torch==2.5.1
[pip3] torchvision==0.20.1
[pip3] transformers==4.48.0
[conda] Could not collect
vLLM Version: 0.6.6.post1
vLLM Build Flags:
```

### How would you like to use vllm

I am using a hosted vLLM model for inference. I would like to be able to use the `guided_choice` param to constrain my outputs. I was using the example provided on the vLLM documentation [here](https://docs.vllm.ai/en/latest/features/structured_outputs.html). 


This is my current setup: 


```python
from openai import OpenAI

api_key = "<api_key>"
hosted_url = "<hosted_url>"

client = OpenAI(
    base_url=hosted_url,
    api_key=api_key
)

completion = client.chat.completions.create(
    model="<model>",
    messages=[
        {"role": "user", "conte

[... truncated for brevity ...]

---

## Issue #N/A: [Doc]: Has the offline chat inference function been updated?

**Link**: https://github.com/vllm-project/vllm/issues/7623
**State**: closed
**Created**: 2024-08-17T09:00:17+00:00
**Closed**: 2024-09-14T03:10:21+00:00
**Comments**: 1
**Labels**: documentation

### Description

### 📚 The doc issue

I did see the official documentation contains the offline inference chat function. But I still get the Attribute Error where LLM object does not have chat() attribute. Has this been updated in the latest package?

### Suggest a potential alternative/fix

_No response_

---

## Issue #N/A: [Usage]: Can multimodal models, such as qwen2.5vl, use the PD separation feature?

**Link**: https://github.com/vllm-project/vllm/issues/19213
**State**: open
**Created**: 2025-06-05T12:56:17+00:00
**Comments**: 0
**Labels**: usage

### Description

### Your current environment

Can multimodal models, such as qwen2.5vl, use the PD separation feature?




---

## Issue #N/A: [Bug]: Automatic Prefix caching not working while hitting same request multiple times

**Link**: https://github.com/vllm-project/vllm/issues/5420
**State**: closed
**Created**: 2024-06-11T13:14:14+00:00
**Closed**: 2024-06-11T13:19:34+00:00
**Comments**: 2
**Labels**: bug

### Description

### Your current environment

PyTorch version: 2.1.2+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.6 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: Could not collect
CMake version: version 3.29.3
Libc version: glibc-2.31

Python version: 3.9.16 (main, Oct 26 2023, 03:04:46)  [GCC 9.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-1048-aws-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.1.105
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A10G
Nvidia driver version: 535.104.12
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Byte Order:                         Little Endian
Address sizes:                      48 bits physical, 48 bits virtual
CPU

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: AttributeError: '_OpNamespace' '_C' object has no attribute 'gptq_marlin_repack'

**Link**: https://github.com/vllm-project/vllm/issues/12267
**State**: closed
**Created**: 2025-01-21T13:55:04+00:00
**Closed**: 2025-05-22T02:11:02+00:00
**Comments**: 4
**Labels**: bug, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.5.1+cpu
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Red Hat Enterprise Linux 9.5 (Plow) (x86_64)
GCC version: (conda-forge gcc 11.4.0-13) 11.4.0
Clang version: Could not collect
CMake version: version 3.28.4
Libc version: glibc-2.34

Python version: 3.12.8 | packaged by conda-forge | (main, Dec  5 2024, 14:24:40) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-5.14.0-503.21.1.el9_5.x86_64-x86_64-with-glibc2.34
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
Address sizes:                        46 bits

[... truncated for brevity ...]

---

## Issue #N/A: [Installation]: Compile and Install from source

**Link**: https://github.com/vllm-project/vllm/issues/4313
**State**: closed
**Created**: 2024-04-24T02:05:22+00:00
**Closed**: 2024-11-29T02:06:27+00:00
**Comments**: 13
**Labels**: installation, stale

### Description

### Your current environment

```text
PyTorch version: 2.2.1+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.3 LTS (x86_64)
GCC version: (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Clang version: Could not collect
CMake version: version 3.29.2
Libc version: glibc-2.31

Python version: 3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.4.0-155-generic-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.2.91
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

Nvidia driver version: 535.54.03
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: Tru

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: enqueue.cc:1556 NCCL WARN Cuda failure 700 'an illegal memory access was encountered'

**Link**: https://github.com/vllm-project/vllm/issues/19890
**State**: closed
**Created**: 2025-06-20T02:39:23+00:00
**Closed**: 2025-06-20T02:47:00+00:00
**Comments**: 1
**Labels**: bug

### Description

### Your current environment

INFO 06-20 10:37:01 [__init__.py:243] Automatically detected platform cuda.
Collecting environment information...
uv is set
==============================
        System Info
==============================
OS                           : Ubuntu 24.04.2 LTS (x86_64)
GCC version                  : (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
Clang version                : Could not collect
CMake version                : version 4.0.3
Libc version                 : glibc-2.39

==============================
       PyTorch Info
==============================
PyTorch version              : 2.7.1+cu128
Is debug build               : False
CUDA used to build PyTorch   : 12.8
ROCM used to build PyTorch   : N/A

==============================
      Python Environment
==============================
Python version               : 3.11.13 (main, Jun  4 2025, 17:37:17) [Clang 20.1.4 ] (64-bit runtime)
Python platform              : Linux-6.11.0-26-generic-x86_64-with-glibc2.39

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Obvious hang caused by Custom All Reduce OP（Valuable Debug Info Obtained）

**Link**: https://github.com/vllm-project/vllm/issues/8410
**State**: closed
**Created**: 2024-09-12T12:19:12+00:00
**Closed**: 2024-09-24T08:08:15+00:00
**Comments**: 6
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
                                                                                    

[... truncated for brevity ...]

---

