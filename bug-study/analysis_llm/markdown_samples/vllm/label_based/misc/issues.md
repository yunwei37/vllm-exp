# misc - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 29

### Label Distribution

- misc: 30 issues
- stale: 11 issues
- RFC: 1 issues
- question: 1 issues
- good first issue: 1 issues
- v1: 1 issues

---

## Issue #N/A: [RFC]: Interface and Abstraction for Distributed Inference Environment

**Link**: https://github.com/vllm-project/vllm/issues/3587
**State**: closed
**Created**: 2024-03-23T23:41:40+00:00
**Closed**: 2024-06-14T01:00:32+00:00
**Comments**: 18
**Labels**: RFC, misc

### Description

This RFC describes a proposal for interfaces and abstractions for distributed inference environments. I plan to solicit discussions for a week (until March 31st) before I begin to actually refactor the code.

# Motivation

The current distributed inference environment in `vllm` is quite tangled, and we often see deadlocks and hangs (see https://github.com/vllm-project/vllm/issues/3455 , https://github.com/vllm-project/vllm/issues/2770 , https://github.com/vllm-project/vllm/issues/3559 , to name a few). The problem becomes prominent when we try to upgrade to pytorch 2.2.0 (see https://github.com/vllm-project/vllm/pull/3442 , https://github.com/vllm-project/vllm/pull/3442 ), because `pytorch 2.2.0` upgrades from `nccl==2.18.1` to `2.19.3` (see https://pypi.org/pypi/torch/2.1.2/json and https://pypi.org/pypi/torch/2.2.0/json to compare the dependency), and `nccl==2.19.3` breaks `vllm` due to increased memory cost during cudagraph capture (from 10MB per graph to 100MB per graph, adds u

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: Remove max_tokens field for chat completion requests when not supported anymore by the OpenAI client

**Link**: https://github.com/vllm-project/vllm/issues/9845
**State**: closed
**Created**: 2024-10-30T16:35:40+00:00
**Closed**: 2025-02-28T02:01:57+00:00
**Comments**: 2
**Labels**: misc, stale

### Description

With the introduction of the `o1` model series, OpenAI deprecated the `max_tokens` field in favor of the new `max_completion_tokens` field for the [chat completion API](https://platform.openai.com/docs/api-reference/chat/create).

This change is active since the [v1.45.0](https://github.com/openai/openai-python/compare/v1.44.1...v1.45.0) version of the OpenAI client.

https://github.com/vllm-project/vllm/pull/9837 added the support for the new `max_completion_tokens` in vLLM while deprecating the `max_tokens` field. However, both fields are supported and cohabit during the deprecation period.

When the OpenAI client definitely drops the `max_tokens` field, this change must also be reflected in the vLLM frontend.

This ticket is to keep track of this task. Relevant parts of the code to be updated are commented with `TODO(#9845)`.




### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom r

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: When using lossy optimization, how to explain that the loss caused by optimization is within the acceptable range?

**Link**: https://github.com/vllm-project/vllm/issues/14128
**State**: closed
**Created**: 2025-03-03T09:30:10+00:00
**Closed**: 2025-07-11T02:16:13+00:00
**Comments**: 8
**Labels**: misc, stale

### Description

### Anything you want to discuss about vllm.

I’ve noticed that with each version upgrade of vllm, there seems to be some degree of precision loss. How do you determine whether these losses are within an acceptable range?

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Misc]: Add sanity test on Python 3.8

**Link**: https://github.com/vllm-project/vllm/issues/4099
**State**: closed
**Created**: 2024-04-15T22:42:17+00:00
**Closed**: 2024-10-30T02:35:42+00:00
**Comments**: 3
**Labels**: misc, stale

### Description

### Anything you want to discuss about vllm.

Currently, CI only runs on Python 3.9. For Python > 3.9, there's no problem with this approach because Python guarantees the backward compatibility, but Python 3.8 (which is still the version that's supported) is not properly tested and sometimes caused issue like https://github.com/vllm-project/vllm/pull/4092#issuecomment-2057867553

To solve this issue, we can have a simple sanity check test on Python 3.8.

---

## Issue #N/A: [Misc]:  Question about Grouped-query attention (GQA)

**Link**: https://github.com/vllm-project/vllm/issues/13222
**State**: closed
**Created**: 2025-02-13T13:14:37+00:00
**Closed**: 2025-02-14T16:24:47+00:00
**Comments**: 1
**Labels**: misc

### Description

### Implementation of Grouped-query attention (GQA)

Hello:) I was wondering whether [Grouped-query attention](https://arxiv.org/pdf/2305.13245#:~:text=Multi%2Dquery%20attention%20shares%20single,head%20and%20multi%2Dquery%20attention) (GQA) is implemented in vLLM. I see that Llama3 models come with this feature in their [architecture](https://arxiv.org/pdf/2407.21783), and they are available through vLLM. Are they using GQA in the backend?

Thanks a lot and sorry for the inconveniences

### Before submitting a new issue...

- [ ] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Misc] [CI]: Flaky test failure in `test_chatglm3_lora`

**Link**: https://github.com/vllm-project/vllm/issues/3947
**State**: closed
**Created**: 2024-04-09T19:22:33+00:00
**Closed**: 2024-04-09T20:33:37+00:00
**Comments**: 2
**Labels**: misc

### Description

* [main CI failed](https://buildkite.com/vllm/ci/builds/4362#018ec42c-e40a-4cbd-943a-e2e4b6e11f53) after https://github.com/vllm-project/vllm/pull/3837 was merged
* tracking in an issue in case we need to ~~fix-forward/revert/or maybe the test is flaky~~ @youkaichao confirmed the test is flaky

```
    def test_chatglm3_lora(chatglm3_lora_files):
        llm = vllm.LLM(MODEL_PATH,
                       max_model_len=1024,
                       enable_lora=True,
                       max_loras=4,
                       max_lora_rank=64,
                       trust_remote_code=True)

        expected_lora_output = [
            "SELECT count(*) FROM singer",
            "SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France'",  # noqa: E501
            "SELECT name ,  country ,  age FROM singer ORDER BY age",
        ]

        output1 = do_sample(llm, chatglm3_lora_files, lora_id=1)
        for i in range(len(expected_lora_output)):
      

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: Wondering why we checkout from a specific commit of Triton

**Link**: https://github.com/vllm-project/vllm/issues/11838
**State**: closed
**Created**: 2025-01-08T09:46:30+00:00
**Closed**: 2025-05-09T02:10:04+00:00
**Comments**: 2
**Labels**: misc, stale

### Description

### Anything you want to discuss about vllm.

I reviewed the Dockerfile and documentation and noticed that we are building Triton from the commit `e192dba224c673671ae70f73842fc693ca279a45`. Is there a specific reason for using this commit? I ask because, based on my experience, Triton's kernel performance on AMD GPUs is better with vlllm Docker at that commit compared to the newest release.

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Misc]: vLLM performs consistently poor as compared to HF TGI when tested with the DeepSeek Coder Model

**Link**: https://github.com/vllm-project/vllm/issues/4030
**State**: closed
**Created**: 2024-04-12T06:55:22+00:00
**Closed**: 2024-05-08T09:35:29+00:00
**Comments**: 20
**Labels**: misc

### Description

### Anything you want to discuss about vllm.

Hello Folks, 

We are using the Deep Seek Coder model for code completions and chat completions. I did try to run the benchmark scripts for that model both for vLLM and TGI and I see that vLLM metrics are consistently poorer as compared to TGI. 

Could you please review and comment on the setup ?

Bring up the servers:
```
MODEL="deepseek-ai/deepseek-coder-1.3b-instruct"
```

```
python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL} \
    --swap-space 16 \
    --disable-log-requests
```

```
    (TGI backend)
    ./launch_tgi_server.sh ${MODEL} 8192
```

On the client side, run:
```
python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model ${MODEL} \
    --dataset-name sharegpt \
    --dataset-path /home/anindya/ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 10 \
    --num-prompts 1000 \
    --save-result
```

```
python benchmarks/benchmark_serving.py \
  

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: vLLM logger disables other existing loggers by default

**Link**: https://github.com/vllm-project/vllm/issues/5803
**State**: closed
**Created**: 2024-06-24T22:59:42+00:00
**Closed**: 2024-08-19T22:11:59+00:00
**Comments**: 1
**Labels**: misc

### Description

### Anything you want to discuss about vllm.

## Issue
The current default behavior of the logger in vLLM is to disable all other existing loggers. This can prevent logs from being outputted from other code that is defined/imported before vLLM is imported.

## Details
The default logging config defined [here](https://github.com/vllm-project/vllm/blob/1744cc99ba9bdefea8f3f798cf51ed650b81a98e/vllm/logger.py#L22-L46) does not include `disable_existing_loggers=False`. When using logging.dictConfig() to configure logging, this value is set to True by default for backwards compatibility. Unless this is the intended behavior, I believe this key should be added to the configuration dictionary.

Happy to add this small change if maintainers agree with this. Thank you!

---

## Issue #N/A: [Misc]:  Improving VLLM KVCACHE Transfer Efficiency with NCCL P2P Communication

**Link**: https://github.com/vllm-project/vllm/issues/7370
**State**: closed
**Created**: 2024-08-09T15:56:06+00:00
**Closed**: 2024-09-09T07:06:15+00:00
**Comments**: 2
**Labels**: misc

### Description

### Anything you want to discuss about vllm.

I hope to utilize the NCCL point-to-point communication protocol (P2P) to transfer the VLLM KVCACHE from the prefill node to the decode node for decoupled inference. Since the KVCACHE in VLLM is stored as a list of tensors, I need to send approximately 16,384 block slices every time I transmit 128 blocks. The non-contiguous distribution of these slices in GPU memory leads to low efficiency in the cyclic transmission, preventing optimal utilization of the communication bandwidth.

Therefore, I am considering concatenating these slices into a single large tensor for transmission. On the receiving end, the node would split this large tensor and write the data back to the corresponding positions based on the slice indices. However, this process is quite time-consuming due to the involvement of up to 16,384 slices. I would like to know if CUDA operations can be utilized to parallelize this process in order to improve performance.

Additional

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: How to call the paged_attention_v2 on my own q and kv caches?

**Link**: https://github.com/vllm-project/vllm/issues/3585
**State**: closed
**Created**: 2024-03-23T15:39:13+00:00
**Closed**: 2024-03-23T23:30:29+00:00
**Comments**: 0
**Labels**: misc

### Description

### Anything you want to discuss about vllm.

Hi, I am trying to use the `paged_attention_v2` function on my own data, qkv.

However, I find it is not giving the correct result. I test with the following script:

```
from typing import Optional
import argparse
import random
import time

import torch
from flash_attn import flash_attn_func
from vllm._C import ops, cache_ops
from vllm.utils import create_kv_caches_with_random

NUM_BLOCKS = 1024
BLOCK_SIZE = 32
PARTITION_SIZE = 512

torch.set_default_device('cuda:0')
torch.set_default_dtype(torch.float16)

def expand_heads(tensor, num_heads=32, num_heads_kv=8):
    assert tensor.dim() == 3
    _, length, dim_head = tensor.shape
    num_group = num_heads // num_heads_kv
    tensor = tensor.view((num_heads_kv, 1, length, dim_head))
    tensor = tensor.expand((num_heads_kv, num_group, length, dim_head)).reshape((num_heads, length, dim_head))
    return tensor


def make_qkv(len_k, num_head, num_head_kv, head_d

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: LoRA request with Multi GPU does not provide correct responses with num_scheduler_steps config

**Link**: https://github.com/vllm-project/vllm/issues/12487
**State**: closed
**Created**: 2025-01-27T20:41:23+00:00
**Closed**: 2025-02-01T05:05:12+00:00
**Comments**: 1
**Labels**: misc

### Description

### Anything you want to discuss about vllm.

Hello All,

We are encountering a strange issue with our LoRA adapter, when running in multi-GPU setup.

Context:
Base model: Mistral Nemo 12B (https://huggingface.co/nvidia/Mistral-NeMo-12B-Instruct)
Adapter Rank: 8

Vllm Model.json
```json
{
    "model": "/model-store/backbone/Mistral-Nemo-Base",
    "disable_log_requests": "true",
    "gpu_memory_utilization": 0.85,
    "max_model_len": 16000,
    "tensor_parallel_size": 2,
    "distributed_executor_backend": "ray",
    "enable_lora": "true",
    "max_lora_rank": 8,
    "max_loras": 4,
    "trust_remote_code": "true"
}
```

Multi-lora.json
```json
{
    "t2f": "/model-store/backbone/loras/Mistral-Nemo-Base-t2f-lora"
}
```

Now, when we add the num_scheduler_steps configuration to the model.json, 

```json
 "num_scheduler_steps": 8,
```

Now the adapter responds with correct response when we don't have 'num_scheduler_steps' in the multi-GPU setup, but when we add this configuration, we do

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: How are quantized models loaded compared to non-quantized models?

**Link**: https://github.com/vllm-project/vllm/issues/8632
**State**: closed
**Created**: 2024-09-19T11:53:50+00:00
**Closed**: 2024-09-20T16:58:28+00:00
**Comments**: 3
**Labels**: misc

### Description


Hi,

I am trying to research MoE layer memory optimizations and I am using vLLM to do so. I have some custom logging code in the initializers/model code of the Mixtral model. When I load a quantized model, no logging code is executed. Simple print statements in the `MixtralModel.__init__` are not printed to screen. Is this on purpose? Where are the MoE kernels getting executed? 

Thanks for any help, I have been stuck on this for a while.

For reference, I have tried to use the https://huggingface.co/TheBloke/mixtral-8x7b-v0.1-AWQ and I have quantized my own models with autoAWQ and bitsandbytes and the same behavior occurs.


---

## Issue #N/A: Remove EOS token before passing the tokenized input to model

**Link**: https://github.com/vllm-project/vllm/issues/4814
**State**: closed
**Created**: 2024-05-14T17:36:47+00:00
**Closed**: 2024-05-29T00:15:36+00:00
**Comments**: 0
**Labels**: misc

### Description



How to remove eos token id before passing the input tokens to model. I'm trying for fine-tuned mistral model. Just because there is an eos token id at the end of sentence, model generates the results for a different input which is similar to original input

---

## Issue #N/A: computation of prompt_logprobs

**Link**: https://github.com/vllm-project/vllm/issues/2848
**State**: closed
**Created**: 2024-02-13T08:09:58+00:00
**Closed**: 2024-06-03T03:02:12+00:00
**Comments**: 1
**Labels**: question, misc

### Description

What are the distinctions between the computation of **prompt_logprobs** in input tokens and **logprobs** in output tokens?

---

## Issue #N/A: Upgrade to numpy >= 2.0.0

**Link**: https://github.com/vllm-project/vllm/issues/6570
**State**: closed
**Created**: 2024-07-19T09:44:44+00:00
**Closed**: 2025-04-10T06:52:10+00:00
**Comments**: 7
**Labels**: misc

### Description

Hi 👋 

Would it be possible to upgrade the dependency of vLLM to numpy to remove the pinning [here](https://github.com/vllm-project/vllm/blob/main/requirements-common.txt#L5)?

We are running into a dependency conflict issue in our dependency chain and would appreciate the move to support more recent numpy versions.

Thanks!

---

## Issue #N/A: [Misc]: Min thread limitation inconsistency for gptq_marlin

**Link**: https://github.com/vllm-project/vllm/issues/6244
**State**: closed
**Created**: 2024-07-09T05:49:18+00:00
**Closed**: 2024-11-25T02:05:00+00:00
**Comments**: 4
**Labels**: misc, stale

### Description

### Anything you want to discuss about vllm.

For gptq_marlin, `min_thread_n=64 min_thread_k=64` is required in [https://github.com/vllm-project/vllm/blob/70c232f85a9e83421a4d9ca95e6384364271f2bc/csrc/quantization/gptq_marlin/gptq_marlin.cuh#L22-L23](url), while `min_thread_n=64 min_thread_k=128` is required in [https://github.com/vllm-project/vllm/blob/70c232f85a9e83421a4d9ca95e6384364271f2bc/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L21-L22](url). Why the limitation is different?

---

## Issue #N/A: [Misc]: Very High GPU RX/TX using vllm

**Link**: https://github.com/vllm-project/vllm/issues/11760
**State**: closed
**Created**: 2025-01-06T06:17:32+00:00
**Closed**: 2025-05-15T02:09:29+00:00
**Comments**: 10
**Labels**: misc, stale

### Description

### Anything you want to discuss about vllm.

I found there are very big size of data transfer to GPU when making a request with 10K tokens. 
VLLM result a very high TTFT compare to Ollama.

I dont think it is a normal data size of 10K tokens. 

vllm version: v0.6.4.post1
There is how I run vllm
`
vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ --pipeline-parallel-size 2 --enable-auto-tool-choice --tool-call-parser hermes --gpu-memory-utilization 0.9 --max_model_len 32000  --max-num-seqs 5 --kv-cache-dtype fp8_e4m3
`

There is my GPU receiving data (over 10GiB/s RX)
![image](https://github.com/user-attachments/assets/87de1545-f7c6-4375-af9e-f99e51ac45f5)



### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Misc] [ROCm]: Build from source failure with Arch/gcc14 with ROCm 6.3

**Link**: https://github.com/vllm-project/vllm/issues/13777
**State**: open
**Created**: 2025-02-24T17:52:19+00:00
**Comments**: 5
**Labels**: misc

### Description

### Anything you want to discuss about vllm.

Hi team!

Been trying to build vllm from source for ROCm 6.3 for gfx1100 on Arch/gcc14 following the instructions from the official documentation. Kept running into a compile error on the hipify step during the build:-

Excerpt from error -
```
...

In file included from <built-in>:1:
In file included from /opt/rocm/lib/llvm/lib/clang/18/include/__clang_hip_runtime_wrapper.h:145:
In file included from /opt/rocm/lib/llvm/lib/clang/18/include/cuda_wrappers/algorithm:55:
In file included from /usr/lib64/gcc/x86_64-pc-linux-gnu/14.2.1/../../../../include/c++/14.2.1/algorithm:61:
/usr/lib64/gcc/x86_64-pc-linux-gnu/14.2.1/../../../../include/c++/14.2.1/bits/stl_algo.h:3626:7: error: reference to __host__ function '__glibcxx_assert_fail' in __host__ __device__ function
 3626 |       __glibcxx_assert(!(__hi < __lo));
      |       ^
/usr/lib64/gcc/x86_64-pc-linux-gnu/14.2.1/../../../../include/c++/14.2.1/x86_64-pc-linux-gnu/bits/c++config.h:614:12:

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: need "first good issue"

**Link**: https://github.com/vllm-project/vllm/issues/4437
**State**: closed
**Created**: 2024-04-28T14:49:54+00:00
**Closed**: 2024-05-31T12:13:00+00:00
**Comments**: 10
**Labels**: misc

### Description

### Anything you want to discuss about vllm.

As a beginner, there are too many issues and PRs, and I find it hard to start contributing.

Could anyone please add `good first issue` label to some issues? So that beginner like me can get started quickly.

Thanks!

---

## Issue #N/A: [Misc]: benchmark_serving with image input

**Link**: https://github.com/vllm-project/vllm/issues/8205
**State**: closed
**Created**: 2024-09-05T18:43:02+00:00
**Closed**: 2024-09-17T07:34:28+00:00
**Comments**: 3
**Labels**: misc

### Description

### Anything you want to discuss about vllm.

Can the current benchmark_serving.py be used with Multimodal LLM (llava) and image input? The existing code send the request in the following format in backend_request_func.py, Is it possible to make it support image input?
 
```
        payload = {
            "model": request_func_input.model,
            "messages": [
                {
                    "role": "user",
                    "content": request_func_input.prompt,
                },
            ],
            "temperature": 0.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }
```


### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page]

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: Question about adding more kv cache blocks

**Link**: https://github.com/vllm-project/vllm/issues/5308
**State**: closed
**Created**: 2024-06-06T05:47:00+00:00
**Closed**: 2024-06-06T14:48:50+00:00
**Comments**: 1
**Labels**: misc

### Description

### Anything you want to discuss about vllm.

Is it theoretically possible to increase the total amount of kv cache blocks by adding new GPU resources, that only for kv cache allocation, without using tensor parallel or other parallel?

---

## Issue #N/A: [Misc]: Can we remove `vllm/entrypoints/api_server.py`?

**Link**: https://github.com/vllm-project/vllm/issues/3852
**State**: closed
**Created**: 2024-04-04T12:58:49+00:00
**Closed**: 2024-11-28T02:06:58+00:00
**Comments**: 12
**Labels**: misc, stale

### Description

### Anything you want to discuss about vllm.

While gardening GitHub issues I've seen many issues where the user is using `-m vllm.entrypoints.api_server`, which indicates that users are either not aware of ignoring the note at the top of the file:

https://github.com/vllm-project/vllm/blob/b7782002e1da25de77e0b1890ff8b72dd4df917c/vllm/entrypoints/api_server.py#L1-L7

Can we remove it to avoid future confusion?

---

## Issue #N/A: [Misc]: Server Does Not Follow Scheduler Policy

**Link**: https://github.com/vllm-project/vllm/issues/4563
**State**: closed
**Created**: 2024-05-02T17:47:32+00:00
**Closed**: 2024-05-04T16:23:24+00:00
**Comments**: 1
**Labels**: misc

### Description

### Anything you want to discuss about vllm.

I was testing out vLLM on Colab and notices something weird. It seems from the code that vLLM is using first come first serve order policy:

https://github.com/vllm-project/vllm/blob/7038e8b80303bf6128acbe508dec910183a1be56/vllm/core/scheduler.py#L729
https://github.com/vllm-project/vllm/blob/7038e8b80303bf6128acbe508dec910183a1be56/vllm/core/policy.py#L29-L36

However, When I was running the OpenAI compatible vLLM server, I sent in orders in sequence and found the server to not follow the first come first serve policy. Instead, they seemed random? Here is a example jupyter notebook replicating the issue:
https://colab.research.google.com/drive/1mMPTZiKJoQEsvjBjNUGttsbp9L1F9zXm?usp=sharing

Is there some optimization I missed that optimized the order of inputs? I am a bit confused on what controls the server's output order. Any advice would be appreciated, thanks!

---

## Issue #N/A: [Misc]: Cross-attention QKV computation is inefficient

**Link**: https://github.com/vllm-project/vllm/issues/7397
**State**: closed
**Created**: 2024-08-10T16:43:36+00:00
**Closed**: 2024-12-12T02:06:53+00:00
**Comments**: 3
**Labels**: misc, stale

### Description

This issue is not in response to a performance regression.

The method of performing cross-attention QKV computations introduced in #4942 could be improved. Because this issue relates to cross-attention, it only impacts encoder/decoder models, not decoder-only models.

For context, `QKVParallelLinear` computes QKV from the previous decoder layer's hidden state output, i.e. only a single input. The problem is that cross attention requires QKV to be computed from two inputs: Q must be computed from the previous decoder layer's hidden state output, and KV must be computed from the encoder's output hidden states. Additionally,
* During prefill phase, both Q and KV must be computed
* During decode phase, only Q is computed because the encoder sequence is static so there are no new encoder KVs

The current, inefficient workaround for cross-attention is to construct a `QKVParallelLinear` layer & apply it at most 2 times in a given run of the cross-attention `forward()` method: once to

[... truncated for brevity ...]

---

## Issue #N/A: [V1][Help Wanted] Porting missing sampling parameters to V1

**Link**: https://github.com/vllm-project/vllm/issues/13058
**State**: closed
**Created**: 2025-02-10T23:13:42+00:00
**Closed**: 2025-03-20T14:15:16+00:00
**Comments**: 10
**Labels**: good first issue, misc, v1

### Description

### Anything you want to discuss about vllm.

To switch the engine from V0 to V1, we need to comprehensively support the sampling parameters in https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py

While most of the key parameters are already supported, some of them are missing:

TODO (help wanted):
- [x] `n` (parallel sampling) #10980  @afeldman-nm 
- [x] `guided_decoding` (structured decoding) #12388  @aarnphm 
- [x] `logit_bias` #13079 @houseroad 
- [x] `min_p` #13191 @AoyuQC
- [ ] `bad_words` (originally implemented via logits processor) #13376 @22quinn 
- [x] `allowed_token_ids` (originally implemented via logits processor) #13210 @houseroad 

Parameters that will not be supported in V1:
* best_of
* logits_processors


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequentl

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: Best practice for accelerating and deploying Llava series & Phi3-Vision using vLLM

**Link**: https://github.com/vllm-project/vllm/issues/6084
**State**: closed
**Created**: 2024-07-03T02:14:01+00:00
**Closed**: 2024-07-09T04:46:07+00:00
**Comments**: 0
**Labels**: misc

### Description

### Anything you want to discuss about vllm.

VLLM is a great inference acceleration framework for LLM and multimodal LLM!!!

The ms-swift LLM toolbox has integrated vLLM to accelerate and deploy multimodal models (currently including the Llava series and Phi3-Vision).

Best practice can be found at:

English: https://github.com/modelscope/swift/blob/main/docs/source_en/Multi-Modal/vllm-inference-acceleration.md

中文：https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/vLLM%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E6%96%87%E6%A1%A3.md

---

## Issue #N/A: [Misc]: Can vLLM go out of memory during the decode phase if too many tokens are generated?

**Link**: https://github.com/vllm-project/vllm/issues/8419
**State**: closed
**Created**: 2024-09-12T14:46:09+00:00
**Closed**: 2025-01-12T02:05:57+00:00
**Comments**: 3
**Labels**: misc, stale

### Description

### Anything you want to discuss about vllm.

Can vLLM go out of memory during the decode phase if too many tokens are generated?

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Misc]: vLLM v0.6.0  CUDA 12 missing wheel file

**Link**: https://github.com/vllm-project/vllm/issues/8362
**State**: closed
**Created**: 2024-09-11T09:10:52+00:00
**Closed**: 2025-01-11T01:59:44+00:00
**Comments**: 3
**Labels**: misc, stale

### Description

### Anything you want to discuss about vllm.

I'd like to understand why the most recent release is omitting the CUDA 12 wheel package built?


### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Misc]: How to set num-scheduler-steps

**Link**: https://github.com/vllm-project/vllm/issues/9158
**State**: closed
**Created**: 2024-10-08T13:23:11+00:00
**Closed**: 2025-02-07T01:59:39+00:00
**Comments**: 6
**Labels**: misc, stale

### Description

### Anything you want to discuss about vllm.

Recently **num-scheduler-steps** was introduced to "set the maximum number of forward steps per scheduler call". Is there any documentation on what this exactly means?
Also some guidance would on how to set this value would be much appreciated. For example, if I host a 70B model on 2x A100 with 80GB, does this narrow down the range of values I should consider?

Thanks to all the amazing vllm contributers for making this great peace of software! 🏎

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

