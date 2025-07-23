# performance - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 11
- Closed Issues: 19

### Label Distribution

- performance: 30 issues
- stale: 15 issues
- help wanted: 1 issues
- feature request: 1 issues

---

## Issue #N/A: [Performance]: test speculative decode accuracy

**Link**: https://github.com/vllm-project/vllm/issues/9609
**State**: closed
**Created**: 2024-10-23T07:40:46+00:00
**Closed**: 2024-10-25T09:18:03+00:00
**Comments**: 1
**Labels**: performance

### Description

### Proposal to improve performance

I use lm-evaluation-harness to test vllm accuracy
1.when don't enable spec decode,I got some result below
num_concurrent=1
![image](https://github.com/user-attachments/assets/dfa6ef55-216e-4460-9ef4-d387e0ce460e)

num_concurrent=8
![image](https://github.com/user-attachments/assets/505d051f-f119-4275-a5d4-5683b74be398)

num_concurrent=16
![image](https://github.com/user-attachments/assets/87e7c9c6-f2de-43de-8a20-96f82c4c9c7c)

num_concurrent=32
![image](https://github.com/user-attachments/assets/312e2703-cfc8-42c7-9751-22a0b1aba21d)


2.when enable spec decode,I got some result below
num_concurrent=1
![image](https://github.com/user-attachments/assets/6681a17f-3bc7-4d52-b0e5-5451a40dfcf4)

num_concurrent=8
![image](https://github.com/user-attachments/assets/4a1878a8-2da7-475e-9ecd-8400a6fc0620)

num_concurrent=16
![image](https://github.com/user-attachments/assets/fc9ca925-a57c-4c6f-9a07-1fa056e67d66)

num_concurrent=32
![i

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: UVA vs UVM for CPU offloading on v0.8.4+

**Link**: https://github.com/vllm-project/vllm/issues/17062
**State**: open
**Created**: 2025-04-23T15:58:29+00:00
**Comments**: 1
**Labels**: performance

### Description

### Proposal to improve performance

Referencing the recent implementation on https://github.com/vllm-project/vllm/pull/15354 (v0.8.4+) for CPU offloading

@youkaichao, is there any specific reason to pick UVA (`cudaHostAlloc`) over UVM `cudaMallocManaged()`? 

1. UVM goes further than UVA to manage data automatically, often using page-faulting hardware to migrate pages on demand. On systems like the GH200, this has potentially additional benefits such as hardware orchestrated frequency based migration. 
2. A key benefit of Unified Memory is simplifying the heterogeneous computing memory model by eliminating the need for deep copies when accessing structured data in GPU kernels. [Source](https://developer.nvidia.com/blog/unified-memory-in-cuda-6/#unified_memory_or_unified_virtual_addressing)
3. On several discussion threads, the larger access sizes of CPU offloading makes UVM seems to be the better approach compared to UVA [Source](https://forums.developer.nvidia.com/t/page-fault-profi

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]:Why do the prefill and decoding need to be executed twice for the same task?

**Link**: https://github.com/vllm-project/vllm/issues/12266
**State**: closed
**Created**: 2025-01-21T13:19:51+00:00
**Closed**: 2025-01-22T05:44:54+00:00
**Comments**: 3
**Labels**: performance

### Description

### Proposal to improve performance





### Report of performance regression

_No response_

### Misc discussion on performance

Hello, when I start the serving service using vllm serve and conduct tests using the benchmark_serving.py script, I captured the kernel pipeline of the CUDA backend through the nsight system. I found out why the prefill and decoding stages of the same task are executed twice?

![Image](https://github.com/user-attachments/assets/19c57ea0-bb61-49a5-ad3e-e2e4a678b845)

At the same time, my commands are as follows:
* serving:
```
vllm serve data/llama-3-8b-instruct \
        --swap-space 16 \
        --disable-log-requests \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.9 \
        --dtype bfloat16
        --enforce-eager
```
* client:
```
python3 vllm/benchmarks/benchmark_serving.py \
        --backend vllm \
        --model data/llama-3-8b-instruct \
        --profile \
        --dataset-name random \
        --random-input-len 2048 \
 

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: why hf is better than vllm when using benchmark throughput

**Link**: https://github.com/vllm-project/vllm/issues/4702
**State**: closed
**Created**: 2024-05-09T06:32:31+00:00
**Closed**: 2024-11-21T03:01:34+00:00
**Comments**: 6
**Labels**: performance, stale

### Description

When I run benchmark on H800,  the results are confusing. Why hf is better than vllm? Is anything wrong when I run the script?

```
python benchmark_throughput.py --input-len 128 --model /home/jiekong/.cache/modelscope/hub/AI-ModelScope/opt-125 --output-len 128 --max-num-batched-tokens 2048 --trust-remote-code
```
Throughput: 59.50 requests/s, 15231.62 tokens/s

![image](https://github.com/vllm-project/vllm/assets/12995855/92d2d824-da47-43f2-aa59-78ff44ad0cd9)

```
python benchmark_throughput.py --input-len 128 --model /home/jiekong/.cache/modelscope/hub/AI-ModelScope/opt-125 --output-len 128 --backend hf --hf-max-batch-size 256
```
Throughput: 108.34 requests/s, 27736.31 tokens/s

![image](https://github.com/vllm-project/vllm/assets/12995855/ce316880-4b7d-408d-9189-25a15731691e)


---

## Issue #N/A: [Performance]: use Python array to replace Python list for zero-copy tensor creation

**Link**: https://github.com/vllm-project/vllm/issues/6879
**State**: closed
**Created**: 2024-07-28T23:58:48+00:00
**Closed**: 2024-12-01T02:14:38+00:00
**Comments**: 19
**Labels**: performance, stale

### Description

### Proposal to improve performance

For flexibility, lots of code in vLLM uses Python list.

The memory layout for a Python list of `[1, 2, 3, 4, 5]`, is:

```
----
PyObject pointer --> PyLong(1)
----
PyObject pointer --> PyLong(2)
----
PyObject pointer --> PyLong(3)
----
PyObject pointer --> PyLong(4)
----
PyObject pointer --> PyLong(5)
----
```

This is because a Python list can hold arbitrary Python object.

When we use `torch.tensor([1, 2, 3, 4, 5], dtype=torch.int, device="cuda")`, there's two copy operation happening:

1. PyTorch has to collect all the data from scattered memory into a continuous memory area, i.e. a CPU memory segment holding `1, 2, 3, 4, 5` consecutively (40 bytes)
2. PyTorch launches an operation to copy the CPU memory to GPU memory, wraps it into a GPU tensor

There is a better alternative in Python, called `array.array`. It is very similar to `vector` type in `C++`, which can hold variable length data with the same type. Since the me

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: Prefix-caching aware scheduling

**Link**: https://github.com/vllm-project/vllm/issues/7883
**State**: closed
**Created**: 2024-08-26T21:30:38+00:00
**Closed**: 2024-12-20T02:18:05+00:00
**Comments**: 8
**Labels**: help wanted, performance

### Description

### Proposal to improve performance

The current execution flow with prefix caching is as follows:
1. Scheduler takes the next prefill sequence:
    a. Calculate how many blocks it needs.
    b. Check whether we have sufficient number of blocks in the block manager.
    c. If so, determine the number of tokens to be prefilled in this batch (it is equal to the prompt length without chunked prefill, or at maximum the chunked size otherwise).
    d. Update the batch token budget by subtracting the tokens to be prefilled.
    e. Allocate all (regardless how many tokens to prefill in this batch) blocks.
    f. Match allocated block IDs with prefix cache, and list them in `computed_block_nums`.
2. Prepare input:
    a. Get the number of tokens to prefill for this sequence in this batch.
    b. Setup input token IDs and positions.
    c. If `computed_block_nums` is not none, then remove the cached tokens from input tokens, and adjust input positions, query length and context leng

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: Inefficient prefill attention compared to HuggingFace

**Link**: https://github.com/vllm-project/vllm/issues/20174
**State**: open
**Created**: 2025-06-27T08:41:58+00:00
**Comments**: 0
**Labels**: performance

### Description

### Proposal to improve performance

_No response_

### Report of performance regression

While benchmarking vLLM for offline inference against HuggingFace Transformers, I observed that the prefill attention in vLLM is significantly slower under certain conditions.

With input_len=128 and max_num_seqs=1 on GPT2 model, vLLM defaults to using FlashAttention. In this setup, vLLM invokes two separate kernels (`flash::prepare_varlen_num_blocks_kernel` and `cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90...>>`). This results in a total latency of ~9μs (without additional kernel launch overhead).
In comparison, HuggingFace Transformers uses `pytorch_flash::flash_fwd_kernel` and completes the same computation in ~6μs.

To reproduce the result, you can run the `benchmark_throughput.py` script in vLLM repo with the configuration above.

### Misc discussion on performance

_No response_

### Your current environment (if you think it is necessary)

```text
The output of 

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: TTFT increases linearly with the number of batched tokens

**Link**: https://github.com/vllm-project/vllm/issues/8086
**State**: closed
**Created**: 2024-09-02T13:19:25+00:00
**Closed**: 2025-01-14T01:57:12+00:00
**Comments**: 4
**Labels**: performance, stale

### Description

### Proposal to improve performance

I have observed that TTFT increases linearly with a total number of batched tokens.
For example, given 100k batch 
- TTFT is around 2min when an average prompt+completion length is 200
- TTFT is around 10min (increase 5X) when an average prompt+completion length is 2000 (increase 10x)

This has been observed for several LLama3 model and the following parameters 
```
enable_prefix_caching=True, block_size=3
max_num_batched_tokens=16000
max_model_len=16000
use_v2_block_manager=True
```

I would of course expect Requests Per Second (or similar metrics) to increase with increase in prompt+completion length, but why this happens so dramatically with TTFT (5X with 10X increase in prompt length)?
Would be helpful for clarification and suggestions on resolution (e.g. adjusting continuous batching).
Thanks in advance!

### Report of performance regression

_No response_

### Misc discussion on performance

_No response_

### Your c

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: 单次请求速度30t/s ，并发请求只有1.5t/s

**Link**: https://github.com/vllm-project/vllm/issues/17568
**State**: closed
**Created**: 2025-05-02T00:12:30+00:00
**Closed**: 2025-05-03T03:41:18+00:00
**Comments**: 24
**Labels**: performance

### Description

### Proposal to improve performance

使用8卡4090部署deepseek 32B模型，单次请求推理速度在30t/s，但是当并发请求waiting reqs队列有排队数据的时候，推理速度只有个位
使用的启动命令:vllm serve llm_model/ds_32B/ --served-model-name deepseek --api-key 12345  --disable-log-requests --trust-remote-code --tensor-parallel-size 8 --max-model-len 36000 --gpu_memory_utilization 0.7 --max-num-seqs 128 --max-num-batched-tokens 4096  --enforce-eager

### Report of performance regression

我的情况和这个问题情况相似，我是用的vllm0.8.5，v1 #16444 

<!-- Failed to upload "info.PNG" -->

### Misc discussion on performance

_No response_

### Your current environment (if you think it is necessary)

```text
```text
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0] (64-bit runt

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: VLLM 请求数量过多时太慢

**Link**: https://github.com/vllm-project/vllm/issues/9474
**State**: closed
**Created**: 2024-10-17T20:29:57+00:00
**Closed**: 2025-02-16T02:03:35+00:00
**Comments**: 2
**Labels**: performance, stale

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How would you like to use vllm

我正在使用一张A100 部署的72B量化模型 这是启动脚本 
python -m vllm.entrypoints.openai.api_server --host 0.0.0.0  --max-model-len 9000 --served-model-name chat-yzq --model /workspace/chat-v1-Int4 --enforce-eager  --tensor-parallel-size 1 --gpu-memory-utilization 0.85

当1天有1万次请求时 回复会变得非常缓慢 有什么办法吗

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Performance]: Llava runs with small batch size and # of GPU blocks

**Link**: https://github.com/vllm-project/vllm/issues/6623
**State**: closed
**Created**: 2024-07-21T16:14:21+00:00
**Closed**: 2024-07-23T02:07:37+00:00
**Comments**: 2
**Labels**: performance

### Description

### Misc discussion on performance

I was running `llava-hf/llava-1.5-7b-hf` vs. `meta-llama/Meta-Llama-3-8B-Instruct` on vLLM 0.5.2 and noticed that Llava 7B runs with a significantly smaller batch size overall -- Llama 3 8B would hit the maximum batch size 256, whereas Llava 7B would remain in the 70~80 range. I do notice that Llava 7B begins with much less GPU blocks allocated (# GPU blocks: 3631, # CPU blocks: 512) compared to LLama 3 8B (# GPU blocks: 13078, # CPU blocks: 2048), which probably explains the batch size.

I wanted to understand whether this difference (existence and magnitude) is expected and the causes. I can think of some reasons that contribute to this:
- Parameters of the vision tower and multimodal projector
  - Less than half a billion parameters
- Activations of the vision tower and multimodal projector
  - They can't be *that* big, can they? I believe they can also be deallocated after generating the image embeddings.
- Image tokens inserted into the

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: V0 and V1 give the same throughput number

**Link**: https://github.com/vllm-project/vllm/issues/15253
**State**: open
**Created**: 2025-03-20T22:15:28+00:00
**Comments**: 1
**Labels**: performance, stale

### Description

### Proposal to improve performance

_No response_

### Report of performance regression

_No response_

### Misc discussion on performance

I constructed an experiment to assess the impact of preemption on inference throughput in V0 and V1. In this experiment, I intentionally designed the workload to exceed GPU memory capacity by setting the number of prompts to 100 and the output length to 4096. This scenario is intended to induce memory overflow and trigger preemption.
I executed the benchmark_throughput.py script on a single GPU node with both V0 and V1, but the resulting throughput numbers were surprisingly similar. 


```bash
    MODEL_NAME="NousResearch/Hermes-3-Llama-3.1-8B"
    NUM_PROMPTS=100
    DATASET_NAME="sonnet"
    DATASET_PATH="benchmarks/sonnet.txt"

    numactl --cpunodebind=1 --membind=1 python3 benchmarks/benchmark_throughput.py \
      --model "${MODEL_NAME}" \
      --dataset-name "${DATASET_NAME}" \
      --dataset-path "${DATASET_PATH}" \
      --output-len 40

[... truncated for brevity ...]

---

## Issue #N/A: Does Tensor Parallelism Ignore GPU Memory When Applied?

**Link**: https://github.com/vllm-project/vllm/issues/13141
**State**: closed
**Created**: 2025-02-12T08:58:29+00:00
**Closed**: 2025-06-13T02:13:14+00:00
**Comments**: 2
**Labels**: performance, stale

### Description

Hi~

I understand that Tensor Parallelism can be applied at the head level or by splitting the heads.
Currently, in vLLM, it seems that the decision to use either v1 or v2 is made when calling the paged_attention kernel.
I am curious whether this decision is made without considering the GPU memory(especially, shared memory) information.

```python
# NOTE(woosuk): We use a simple heuristic to decide whether to use
# PagedAttention V1 or V2. If the number of partitions is 1, we use
# V1 to avoid the overhead of reduction. Also, if the number of
# sequences or heads is large, we use V1 since there is enough work
# to parallelize.
# TODO(woosuk): Tune this heuristic.
# For context len > 8192, use V2 kernel to avoid shared memory shortage.
use_v1 = (max_seq_len <= 8192
          and (max_num_partitions == 1 or num_seqs * num_heads > 512))
```
Can most cases be covered with only the above condition?

---

## Issue #N/A: [Performance]: Added request take too much time, and the model will not run untill all the request are added into the cache

**Link**: https://github.com/vllm-project/vllm/issues/13259
**State**: open
**Created**: 2025-02-14T04:06:28+00:00
**Comments**: 8
**Labels**: performance

### Description

### Proposal to improve performance

```
INFO 02-14 11:57:32 engine.py:275] Added request chatcmpl-1af15bd86d5f413683cd727e1028852c.                                                                                                                                                                              
INFO 02-14 11:57:32 engine.py:275] Added request chatcmpl-b4e5eba8d8d144a0813ffb6e378ee784.                                                                                                                                                                              
INFO 02-14 11:57:32 engine.py:275] Added request chatcmpl-1ca0f490ea104efc9884777815e51618.                                                                                                                                                                              
INFO 02-14 11:57:33 engine.py:275] Added request chatcmpl-984040d9c3cf424984a719970de484f5.                                                                      

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: does vllm support tensor data for requests？

**Link**: https://github.com/vllm-project/vllm/issues/20898
**State**: open
**Created**: 2025-07-14T03:10:39+00:00
**Comments**: 3
**Labels**: performance

### Description

### Proposal to improve performance

For my current project, I need to accelerate image-text inference. I want to cache images locally first, and then directly send the tensor values when calling the API. However, it seems that image data can only be sent in Base64 format. Is there any way to send tensor-type data instead?

### Report of performance regression

_No response_

### Misc discussion on performance

_No response_

### Your current environment (if you think it is necessary)

```text
The output of `python collect_env.py`
```


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Performance]: Throughput and Latency degradation with a  single LoRA adapter on A100 40 GB

**Link**: https://github.com/vllm-project/vllm/issues/10062
**State**: closed
**Created**: 2024-11-06T00:42:03+00:00
**Closed**: 2025-05-28T02:13:20+00:00
**Comments**: 16
**Labels**: performance, stale

### Description

### Proposal to improve performance

_No response_

### Report of performance regression

_No response_

### Misc discussion on performance



---

**Setup Summary for vLLM Benchmarking with Llama-2 Model:**

- **Hardware**: A100 40 GB (a2-highgpu-2g) on Google Kubernetes Engine (GKE)
- **Model**: `meta-llama/Llama-2-7b-hf`
- **GPU Count**: 1
- **Experiments**:
  - **Experiment 1**: Requests using the base model `meta-llama/Llama-2-7b-hf`.
  - **Experiment 2**: vLLM deployed with LoRA adapter `vineetsharma/qlora-adapter-Llama-2-7b-hf-TweetSumm` (size 160 MB).
  - **Experiment 3**: vLLM deployed with LoRA adapter `xtuner/Llama-2-7b-qlora-moss-003-sft` (size 640 MB).
  
  For all three experiments, we used the same input prompt (ShareGPT) and observed a similar output length.

**Settings**:
- **Eager Mode**: Not enabled.
- **Max GPU Utilization**: Default at 90%.

**Benchmark Metrics**:
We measured:
  - **Latency per output token**
  - **Throughput** (output tokens

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: Very low generation throughput on CPU

**Link**: https://github.com/vllm-project/vllm/issues/12153
**State**: closed
**Created**: 2025-01-17T08:10:25+00:00
**Closed**: 2025-05-19T02:13:22+00:00
**Comments**: 4
**Labels**: performance, stale

### Description

### Proposal to improve performance

_No response_

### Report of performance regression

_No response_

### Misc discussion on performance

I am deploying vLLM API server with `ibm-granite/granite-3.1-8b-instruct` model on an Ubuntu server with only CPUs available.

I noticed that the average generation throughput is as low as 0.1 token/s as shown below in the logs, plus it took 10 mins from "Added request" to actually generation (which was spent for prompt processing I believe?) 
```
INFO 01-17 07:46:18 engine.py:270] Added request chatcmpl-522a81bb1b6d4e6196db0786acf51046.
WARNING 01-17 07:57:05 _logger.py:72] Pin memory is not supported on CPU.
INFO 01-17 07:57:05 metrics.py:467] Avg prompt throughput: 0.1 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
INFO 01-17 07:57:22 metrics.py:467] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.1 tokens/s, Running: 1 r

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: yarn degrades the performance of qwen3

**Link**: https://github.com/vllm-project/vllm/issues/18728
**State**: closed
**Created**: 2025-05-26T18:32:46+00:00
**Closed**: 2025-06-05T14:58:13+00:00
**Comments**: 1
**Labels**: performance

### Description

### Proposal to improve performance

`vllm version == 0.8.5.post1`

without yarn
```bash
vllm serve Qwen/Qwen3-32B   \
 --trust-remote-code --gpu_memory_utilization 0.95 --tensor-parallel-size 2 \
--quantization bitsandbytes --load_format bitsandbytes --enforce_eager \
--max-model-len 32768
```

with yarn
```bash
vllm serve Qwen/Qwen3-32B   \
--trust-remote-code --gpu_memory_utilization 0.95 --tensor-parallel-size 2 \
--quantization bitsandbytes --load_format bitsandbytes --enforce_eager \
--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
--max-model-len 131072
```

I have some tests on my end for its agentic capabilities based on qwen3 and I have some solid findings that enabling yarn to extend window context does degrade the performace, with around 15-20% performance drop. 

do u also encounter the same findings ? any suggestion about this drop ?



### Report of performance regression

_No response_

### Misc discussion on performance

_No

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: Performance Bottleneck in Mooncake PD Disaggregation: tensorhash() and safetensor_save() Overhead

**Link**: https://github.com/vllm-project/vllm/issues/20009
**State**: open
**Created**: 2025-06-24T08:14:19+00:00
**Comments**: 2
**Labels**: performance

### Description

### Proposal to improve performance

Hi team,

I've been conducting performance tests on vllm PD Disaggregation using mooncake_store_connector, and found that the most time-consuming parts are not the actual put() operations, but rather:
- [tensorhash()](https://github.com/vllm-project/vllm/blob/b6553be1bc75f046b00046a4ad7576364d03c835/vllm/distributed/kv_transfer/kv_connector/mooncake_store_connector.py#L198)
- [safetensor_save()](https://github.com/vllm-project/vllm/blob/b6553be1bc75f046b00046a4ad7576364d03c835/vllm/distributed/kv_transfer/kv_lookup_buffer/mooncake_store.py#L131)

Based on profiling traces, these two steps dominate the runtime during PD disaggregation, more than the actual storage or network transmission:
![Image](https://github.com/user-attachments/assets/320e80c2-976e-4ff5-9fd4-ff65ecf3ba83)

**Observations:**

tensorhash() seems to repeatedly compute SHA256 hashes over possibly large tensors.
safetensor_save() is used per tensor and appears to serialize, which is 

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: The Unstable Performance Difference between CUDA and PyTorch

**Link**: https://github.com/vllm-project/vllm/issues/18884
**State**: open
**Created**: 2025-05-29T06:52:56+00:00
**Comments**: 2
**Labels**: performance

### Description

### Proposal to improve performance

_No response_

### Report of performance regression

_No response_

### Misc discussion on performance

I have encountered such a problem：I implemented a custom CUDA operator for matrix multiplication and compared its time performance with PyTorch’s einsum method. In a standalone Python test script, the execution time of the CUDA operator was significantly less than that of the einsum method. The code and results for the test time are as follows:
import pytest
import torch
import time
from vllm._custom_ops import decode_matrix as decode_matrix_cuda
from vllm.platforms import current_platform

def decode_matrix_torch(
        a_sm: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
        q: torch.Tensor,  # [NUM_HEADS,NUM_TOKENS, HEAD_SIZE]
        k_cache: torch.Tensor,  # [NUM_HEADS,NUM_TOKENS, HEAD_SIZE]
        window_factors: torch.Tensor,  # [NUM_HEADS,1, 1]  
):
    a_sm = torch.einsum('hmd,hnd->hmn', q, k_cache) * (k_cache.shape[-1] ** -0.

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: 5x slower throught with openAI client/server than native one

**Link**: https://github.com/vllm-project/vllm/issues/7935
**State**: closed
**Created**: 2024-08-28T02:44:31+00:00
**Closed**: 2024-10-28T19:10:50+00:00
**Comments**: 21
**Labels**: performance

### Description

### Proposal to improve performance

I've been trying to write a reliable benchmark to be used with vllm, and I discovered that when I use the openAI client it can't scale. If I try to use 50 concurrent clients the gpu load goes down to 5% and the throughput is extremely slow. The more clients I add the worst things get. With a single client there is no problem.

I then used the same benchmark switching to the [vllm native client/server](https://docs.vllm.ai/en/latest/getting_started/examples/api_client.html) and I'm getting a 60-70% gpu util and 5x higher throughput.

I checked that I had the same `SamplingParams` reported by the server in both cases.

In parallel with those I was using https://github.com/grafana/k6 against both uses cases - with openAI entrypoints and with the native entrypoint - I can confirm that the server isn't the problem - in both cases I get high gpu util with k6 client and high throughput.

I thought that perhaps streaming was the cause but disablin

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: Where is the cache stored when vLLM reads the model's checkpoint for the first time? I hope it can be saved so that the model loads quickly every time.

**Link**: https://github.com/vllm-project/vllm/issues/19398
**State**: open
**Created**: 2025-06-10T04:52:04+00:00
**Comments**: 6
**Labels**: performance

### Description

### Proposal to improve performance

_No response_

### Report of performance regression

_No response_

### Misc discussion on performance

_No response_

### Your current environment (if you think it is necessary)

```text
The output of `python collect_env.py`
```


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Performance]: only 0.4 tokens/s when running 2 or more request

**Link**: https://github.com/vllm-project/vllm/issues/15018
**State**: open
**Created**: 2025-03-18T09:17:19+00:00
**Comments**: 4
**Labels**: performance, stale

### Description

### Proposal to improve performance

_No response_

### Report of performance regression

_No response_

### Misc discussion on performance

I was tring to run DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf with 7900 XTX（24G) and it works,I have got 19.3 tokens/s when it run with one request.However，the throughput was only 0.4 tokens/s   when it running two or more requsets.The GPU KV cache usage  is enought,is there any parameters i have to set?
INFO 03-18 08:46:00 [__init__.py:256] Automatically detected platform rocm.
INFO 03-18 08:46:01 [api_server.py:912] vLLM API server version 0.7.4.dev442+gfd8e055f
INFO 03-18 08:46:01 [api_server.py:913] args: Namespace(subparser='serve', model_tag='/app/model/DeepSeek-R1-Distill-Qwen-32B-GGUF/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf', config='', host='0.0.0.0', port=8199, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, lora_modules=None, prompt_adapters=None, chat_

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: Slowdown compared to Gradio

**Link**: https://github.com/vllm-project/vllm/issues/8866
**State**: closed
**Created**: 2024-09-26T20:35:42+00:00
**Closed**: 2025-01-26T01:59:49+00:00
**Comments**: 3
**Labels**: performance, stale

### Description

### Proposal to improve performance

vLLM is amazingly fast
However, when running below prompt, with meta-llama/Meta-Llama-3-8B-Instruct, Gradio takes ~4sec per prompt (one by one) while vLLM takes ~12sec by def. When setting --quantization fp8 times reduced to ~8s
Overall vLLM is much faster since it allows to process in parallel while Gradio doesn't
Tested with AWS L4, Gradio 4.43.0
What am I missing?

`prompt = """You are a knowledgeable, efficient, and direct Al assistant. Provide concise answers up to 100 words`
`without explainations or extra notes, focusing on the key information needed. Answer in question: answer JSON format`
`**User:**I like the color red. Our website is www.nba.com. My age is 18.`
`**Assistant:**Great. Write 3 things for me to answer.`
`**User:**What is our website? What is my age? What kind of drink do I like to drink?`
`**Assistant**:`
`"""`

### Report of performance regression

_No response_

### Misc discussion on performance

_No r

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: Cannot use FlashAttention-2 backend for Volta and Turing GPUs.

**Link**: https://github.com/vllm-project/vllm/issues/10592
**State**: closed
**Created**: 2024-11-23T11:49:39+00:00
**Closed**: 2025-03-24T02:06:43+00:00
**Comments**: 2
**Labels**: performance, stale

### Description

### Proposal to improve performance

![image](https://github.com/user-attachments/assets/c70acb43-596a-490c-8409-8e90d180d0fc)
I would like to ask if I cannot use FlashAttention because my gpu is v100.

### Report of performance regression

_No response_

### Misc discussion on performance

_No response_

### Your current environment (if you think it is necessary)

```text
The output of `python collect_env.py`
```


### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Performance]: Why AWQ model‘s performance issue on A100&H100

**Link**: https://github.com/vllm-project/vllm/issues/15809
**State**: open
**Created**: 2025-03-31T10:13:02+00:00
**Comments**: 1
**Labels**: performance, stale

### Description



### Misc discussion on performance

I am using 0.8.3 version of vllm,driver 570.124.06, 
this command to serve to depoly AWQ model casperhansen/llama-3.3-70b-instruct-awq （GEMM） on single H100PCIE & single A100 PCIE

python -m vllm.entrypoints.openai.api_server --model casperhansen/llama-3.3-70b-instruct-awq --max-num-seqs=256 --max-model-len=4096 --max-num-batched-tokens=4096 --tensor-parallel-size=1 --block-size=128 --host=0.0.0.0 --port=8000 --gpu-memory-utilization=0.9  --trust-remote-code
 
We run the test with 2048 input and output, on batch size 1,2,4,8,32,64, and we find H100 just little better than A00 about 10-30% on TTFT and TPOT almost all batch size.

However on GPTQ model (w4a16). the perofromance is very different. H100 is 2 times better than A100. 

So my question is what is going on with AWQ quantized model? Why AWQ model on H100 is not 2time better than A100 as GPTQ model, they both Q4A16, should have similar performance?




### Before submitting a new issue...

- 

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: poor performance in pipeline parallesm when batch-size is large

**Link**: https://github.com/vllm-project/vllm/issues/15330
**State**: open
**Created**: 2025-03-22T12:46:07+00:00
**Comments**: 5
**Labels**: performance, stale

### Description

### Proposal to improve performance

In the case where the lengths of the sent requests are the same, pipeline parallelism should have fewer bubbles, which also means that pipeline parallelism should have a higher throughput than tensor parallelism. However, when I issue requests with a batch size of 400 and a sequence length of 2048, the throughput of the Decode stage in tensor parallelism is nearly three times higher than that in pipeline parallelism.

![Image](https://github.com/user-attachments/assets/c40d96ec-3c99-41c6-85ad-4f7bca36fdae)

### Report of performance regression

You can use the following script to reproduce the phenomenon that the performance of the Decode stage in pipeline parallelism is very poor as I mentioned. I sent 400 requests from the client to the started server, and the request configuration is that the input length is 2048 and the maximum output length is 1000.

`nsys profile -o report.nsys-rep-pp-4-batch-micro-batch-100-python --trace-fork-before-exec=tru

[... truncated for brevity ...]

---

## Issue #N/A: [Performance]: INFO 09-11 12:41:50 spec_decode_worker.py:790] SpecDecodeWorker scoring_time_ms is slow

**Link**: https://github.com/vllm-project/vllm/issues/8370
**State**: closed
**Created**: 2024-09-11T12:45:29+00:00
**Closed**: 2025-01-13T02:03:15+00:00
**Comments**: 5
**Labels**: performance, stale

### Description

### Proposal to improve performance

_No response_

### Report of performance regression

INFO 09-11 12:41:50 spec_decode_worker.py:790] SpecDecodeWorker stage times: average_time_per_proposal_tok_ms=4.96 scoring_time_ms=54.92 verification_time_ms=1.20

The proportion of scoretime in decde is too large. The draft model only requires 5ms for each decode, but it takes 50ms for each score calculation.



### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Performance]: vllm0.6.5加载GLM4-9B-Chat，动态加载lora，输入长文本时推理性能下降较多

**Link**: https://github.com/vllm-project/vllm/issues/11317
**State**: closed
**Created**: 2024-12-19T03:37:08+00:00
**Closed**: 2025-03-21T08:44:54+00:00
**Comments**: 14
**Labels**: performance, stale

### Description

### Proposal to improve performance

_No response_

### Report of performance regression

### A800，单卡处理单条请求
1. **vllm0.6.5不加载lora**
（1）启动：
CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server --model /Work/....../glm-4-9b-chat/ --trust-remote-code
（2）请求：
response = client.chat.completions.create(
        model='/Work/....../glm-4-9b-chat/',
        messages=messages,
        n=1,
        temperature=0,
        extra_body={"stop_token_ids": [151329, 151336, 151338]},
        max_tokens=2048,
        stream=True)

2. **vllm0.6.5动态加载lora**
【lora模型使用llama_factory框架训练】
（1）启动：
CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server --model /Work/....../glm-4-9b-chat/ --enable-lora --max-loras 10 --lora-modules summary=/Work/....../sft_1218/ --trust-remote-code --max-lora-rank 64
（2）请求：
response = client.chat.completions.create(
        model='summary',
        messages=messages,
        n=1,
        temperature=0,
        extra_body={"stop_t

[... truncated for brevity ...]

---

## Issue #N/A: [FEATURE] Implement Dynamic SplitFuse

**Link**: https://github.com/vllm-project/vllm/issues/1562
**State**: closed
**Created**: 2023-11-04T14:06:52+00:00
**Closed**: 2024-07-26T10:25:27+00:00
**Comments**: 7
**Labels**: performance, feature request

### Description

Dear vLLM maintainers @WoosukKwon and @zhuohan123 (@Yard1),

DeepSpeed has released its serving framework which claims to be faster than vLLM. The main speedup comes from [Dynamic SplitFuse](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen#b-dynamic-splitfuse-) which is a technique that does the following:

- Long prompts are decomposed into much smaller chunks and scheduled across multiple forward passes (iterations) with only the final pass performing any generation.

- Short prompts will be composed to exactly fill a target token budget. Even short prompts may be decomposed to ensure the budget is precisely met and the forward sizes are well-aligned.

Code: https://github.com/microsoft/DeepSpeed-MII
Background: https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen

Llama 13B (1x A100-80GB):
![image](https://github.com/vllm-project/vllm/assets/27340033/cc7842b8-e234-482d-8550-d38d39d94473)

Llama 70B (4x A100x80GB with TP):

[... truncated for brevity ...]

---

