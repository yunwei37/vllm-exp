# moderate_impact_3to10 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 10
- Closed Issues: 20

### Label Distribution

- inactive: 13 issues
- good first issue: 3 issues
- high priority: 3 issues
- feature: 3 issues
- enhancement: 2 issues
- help wanted: 2 issues
- MLLM: 2 issues
- collaboration: 2 issues
- flashinfer: 1 issues
- performance: 1 issues

---

## Issue #N/A: [Feature]: Benchmarking H200

**Link**: https://github.com/sgl-project/sglang/issues/2450
**State**: open
**Created**: 2024-12-11T14:11:42+00:00
**Comments**: 6
**Labels**: good first issue, high priority

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

#  Research Questions

- Explore the tradeoffs of increasing the **number of chips** with more memory, H200, versus increasing the parallel inference **world size** when using less HBM GPUs, H100 (see [[Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)](https://arxiv.org/abs/2211.05102)). Reduce as much as possible **price/generation** at **scale.**
- How can we leverage H200 **extra HBM** for efficient KV cache management?  Test long context window.
- Measure the implications of faster GPU **memory bandwidth** while executing **parallel inference**.

# Models of Interest

- **Llama 3.3 70B**
- **Llama 3.1 405B**
- **DeepSeek Models:** Testing latest sglang `0.4` [data pa

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support Deepseek's DeepGemm MoE

**Link**: https://github.com/sgl-project/sglang/issues/3881
**State**: closed
**Created**: 2025-02-26T08:45:49+00:00
**Closed**: 2025-05-03T00:18:09+00:00
**Comments**: 7
**Labels**: inactive

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Will the MoE operator adopt the DeepGemm operator open sourced by Deepseek?

### Related resources

_No response_

---

## Issue #N/A: [Proposal] SGLang Support Distributed Cache in PD Disaggregation

**Link**: https://github.com/sgl-project/sglang/issues/7761
**State**: open
**Created**: 2025-07-04T02:13:04+00:00
**Comments**: 11
**Labels**: high priority

### Description

# [Proposal] SGLang Supports Distributed Cache in PD Disaggregation
## Table of Contents

1. [Summary](#)
2. [Current Cache Architecture](#)
3. [Pain Points](#)
4. [Proposed Improvements](#)
5. [Maybe More](#)

# Summary
Distributed cache lays a solid foundation for efficient cache reuse in multi-turn dialogue scenarios within distributed environments, serving as a prerequisite for cache-aware routing strategies.

Traditional homogeneous GPU clusters are split into three independent resource pools, realizing the separation of compute and cache. By decoupling global resources with KVCache at the center, we enable “compute-bandwidth-storage” to be optimized independently during large model inference. This aims to address pain points in long-context processing and high-concurrency workloads.  

# Current Cache Architecture

The cache class implementation resides in `python/sglang/srt/mem_cache`, comprising two major structures:  
1. Classes based on `BasePrefixCache` (e.g., chunked cache*

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] disable-req-waiting

**Link**: https://github.com/sgl-project/sglang/issues/5446
**State**: closed
**Created**: 2025-04-16T05:45:06+00:00
**Closed**: 2025-07-09T00:20:20+00:00
**Comments**: 8
**Labels**: inactive

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

TTFT is also an important online indicator.In sglang, I find some badcases:
when vram is not enough for the coming req, the req must wait for a while in waiting_queue, then ttft could be bad as user see(including waiting time in waiting queue)
so I want to fuse my some work about it in upstream. if we disable-req-waiting, when vram is not enough for the coming req, the scheduler could return 403 to server and user or router could try again at the service level.

Which parts may be modified:
1. in scheduler.py, we need add some free-vram check in "handle_generate_request" and if vram is not enough, just return aborted status to tokenizer
2. in tokenizer.py and open_ai/adapter.py , we need to support return this kind

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Cache-aware Data Parallel Router

**Link**: https://github.com/sgl-project/sglang/issues/1732
**State**: closed
**Created**: 2024-10-20T18:38:23+00:00
**Closed**: 2024-12-20T00:16:49+00:00
**Comments**: 1
**Labels**: inactive, feature

### Description

### Motivation

See more context in the [design doc](https://docs.google.com/document/d/1cCqK3dh7ZR_rUPkcZT2cr0kLnAxv6_Sd-P1q37-3RNQ/edit?usp=sharing)

The doc is still work in progress. Please expect active changes

### Related resources

_No response_

---

## Issue #N/A: RuntimeError in llava image encoding

**Link**: https://github.com/sgl-project/sglang/issues/273
**State**: closed
**Created**: 2024-03-10T16:27:59+00:00
**Closed**: 2024-07-25T06:33:30+00:00
**Comments**: 6
**Labels**: inactive

### Description

When running llava 1.6 mistral 7b, i get this error:
```
RuntimeError in llava image encoding: The expanded size of the tensor (0) must match the existing size (2438) at non-singleton dimension 0.  Target sizes: [0, 4096].  Tensor sizes: [2438, 4096]
torch.Size([2758, 4096])
0 -1
```

Note the sizes `2438` and `2758` changes 
This error happens randomly and is not specific to data.
Removing image input removes this error too.

---

## Issue #N/A: [Feature] Tokenizer endpoint in server mode

**Link**: https://github.com/sgl-project/sglang/issues/5653
**State**: closed
**Created**: 2025-04-23T02:55:53+00:00
**Closed**: 2025-06-24T00:19:46+00:00
**Comments**: 2
**Labels**: inactive

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Using Server mode to generate Rollout in Agentic RL training is a very necessary and natural approach. However, the design of Agent Scaffold typically only considers compatibility with OpenAI compatible API interface, making it difficult to collect token IDs at the Agent Scaffold level—information that is essential for training. Additionally, current design couples tokenization with the inference model, which indicates it's a logically sound idea to let inference engine handle tokenization. 
Thus, a `tokenize` and endpoint is needed.

### Related resources

Maybe refer to vllm's `tokenize` endpoint. https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#tokenizer-api

---

## Issue #N/A: [Feature] Integration into Dynamo Planner

**Link**: https://github.com/sgl-project/sglang/issues/6163
**State**: open
**Created**: 2025-05-09T20:48:46+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Dynamo Planner is a dynamo services which can monitor the state of the inference system, and perform scaling up/down prefill/decode workers based on kv cache load and prefill queue sizes. For now it supports the aggregated/disaggregated VLLM worker, but not yet SGLang.

By looking at the code, Dynamo Planner has a well abstracted interfaces which would collect the metrics from different inference framework backends. The server side is implemented in metrics_aggregator.rs, and the client side will use its Python bindings to publish the metrics. The key part in the current VLLM implementation is below:
```
self.metrics_publisher.publish(
                            metrics.request_active_slots,
                      

[... truncated for brevity ...]

---

## Issue #N/A: Task 001: Introduce Worker Abstraction

**Link**: https://github.com/sgl-project/sglang/issues/7534
**State**: closed
**Created**: 2025-06-25T20:11:12+00:00
**Closed**: 2025-07-12T03:21:17+00:00
**Comments**: 2

### Description

# Task 001: Introduce Worker Abstraction

## Summary
Replace string-based worker URLs with a proper Worker trait that provides type safety, health tracking, and load monitoring capabilities throughout the router.

## Problem Statement
Currently, workers are represented as simple URL strings (`Vec<String>`), which leads to several issues:
- No way to track worker health status
- Load tracking requires separate data structures
- No type distinction between regular, prefill, and decode workers
- Health checking logic is scattered and inconsistent
- Difficult to add worker-specific metadata

## Proposed Solution

### 1. Worker Trait Definition
Create a trait that encapsulates all worker functionality:

```rust
// src/core/worker.rs
pub trait Worker: Send + Sync + Clone {
    fn url(&self) -> &str;
    fn worker_type(&self) -> WorkerType;
    fn is_healthy(&self) -> bool;
    async fn check_health(&self) -> Result<(), WorkerError>;
    fn load(&self) -> Arc<AtomicUsize>;
    fn update_healt

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Add alternative names for pp-size, tp-size, and dp-size

**Link**: https://github.com/sgl-project/sglang/issues/893
**State**: closed
**Created**: 2024-08-02T12:37:26+00:00
**Closed**: 2024-08-05T18:13:04+00:00
**Comments**: 3
**Labels**: good first issue

### Description

### Motivation

Probably a bit immature but I just had a meeting with a coworker to explain the options for sglang. I really don't want to say `pp-size` at work again......

Can we modify the options to include pipeline-parallel-size, tensor-parallel-size, and data-parallel-size like vLLM do? We can still keep the old ones for back compatibility.

### Related resources

https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py

---

## Issue #N/A: run python3 test_httpserver_llava.py get ValueError: 64002 is not in list

**Link**: https://github.com/sgl-project/sglang/issues/413
**State**: closed
**Created**: 2024-05-08T11:35:48+00:00
**Closed**: 2024-07-30T01:03:13+00:00
**Comments**: 3
**Labels**: inactive

### Description

run python3 test_httpserver_llava.py
offset = input_ids.index(self.config.image_token_index)
ValueError: 64002 is not in list

def test_streaming(args):
    url = f"{args.host}:{args.port}"
    response = requests.post(
        url + "/generate",
        json={
            'text' : 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions. USER: <im_start><image><im_end> description the video indetail \n Assistant:', 
            # "text": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: Describe this picture <|im_start|> <|im_end|>\n ASSISTANT:",
            "image_data": "examples/image1.webp",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 128,
            },
            "stream": True,
        },
   

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] fp4 flashinfer moe error in latest blackwell docker image

**Link**: https://github.com/sgl-project/sglang/issues/7768
**State**: open
**Created**: 2025-07-04T06:59:23+00:00
**Comments**: 4

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

When trying to run Deepseek R1 FP4 on B200, after the first CUDA graph capture, it reports a very long error message. I have pasted a snippet here `/sgl-workspace/projects/sglang/python312/lib/python3.12/site-packages/flashinfer/data/csrc/nv_internal/tensorrt_llm/cutlass_instantiations/gemm_grouped/cutlass_kernel_file_3.generated.cu(15): e

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support AMD GPU via PyTorch for ROCm

**Link**: https://github.com/sgl-project/sglang/issues/1419
**State**: closed
**Created**: 2024-09-14T05:55:31+00:00
**Closed**: 2024-09-19T11:01:59+00:00
**Comments**: 1
**Labels**: enhancement

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Enable SGLang on AMD GPUs !

### Related resources

_No response_

---

## Issue #N/A: [Feature] Support more multi-modal input for VLM

**Link**: https://github.com/sgl-project/sglang/issues/5964
**State**: open
**Created**: 2025-05-02T02:28:40+00:00
**Comments**: 4
**Labels**: good first issue, help wanted, feature, MLLM

### Description

### Motivation

The current endpoint only supports image data input, limiting its flexibility for diverse VLM use cases. We need additional input formats, particularly for RL applications:
(Could be split into multiple PRs)

- [x] Pre-computed Image Embeddings
- [ ] Pixel Values
- [ ] Pixel Value Range Parameters (min_pixel/max_pixel) for qwen-vl

Welcome to propose more.

#### Benefits

1. Enhanced flexibility for RL workflows
2. Reduced preprocessing overhead
3. Better integration with existing pipelines

---

## Issue #N/A: [Bug] pydantic validation errors for ChatCompletion

**Link**: https://github.com/sgl-project/sglang/issues/3637
**State**: closed
**Created**: 2025-02-17T12:53:53+00:00
**Closed**: 2025-03-06T15:15:36+00:00
**Comments**: 14

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

when use autogen with qwen2.5

messages=[SystemMessage(content='You are a helpful AI assistant. Solve tasks using your tools. Reply with TERMINATE when the task has been completed.', type='SystemMessage'), UserMessage(content='shanghai weather', source='user', type='UserMessage')], 
client.chat.completions.create(
                        m

[... truncated for brevity ...]

---

## Issue #N/A: sgl-kernel for aarch64

**Link**: https://github.com/sgl-project/sglang/issues/3769
**State**: closed
**Created**: 2025-02-21T17:19:00+00:00
**Closed**: 2025-05-12T00:20:28+00:00
**Comments**: 3
**Labels**: help wanted, inactive

### Description

Hello,

Thank you very much for your great work on SGLang!

I was wondering if it would be possible to release wheels for `sgl-kernel` for aarch64 (the one on pypi right now only supports x86_64). Alternatively, it would be very helpful if you could provide instructions on how to build `sgl-kernel` from source as well!

---

## Issue #N/A: [Feature] QwQ support

**Link**: https://github.com/sgl-project/sglang/issues/2237
**State**: closed
**Created**: 2024-11-28T08:42:43+00:00
**Closed**: 2024-12-01T10:27:32+00:00
**Comments**: 4
**Labels**: enhancement, feature

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

ref https://qwenlm.github.io/blog/qwq-32b-preview/

### Related resources

_No response_

---

## Issue #N/A: [Feature] Adding flashinfer's cuDNN backend kernel for DSR1 prefill

**Link**: https://github.com/sgl-project/sglang/issues/7842
**State**: open
**Created**: 2025-07-08T05:12:26+00:00
**Comments**: 0

### Description

https://github.com/sgl-project/sglang/pull/7841 [WIP]

---

## Issue #N/A: [Bug] Only one Worker active when using sglang_router.launch_server on a single machine with multiple GPUs

**Link**: https://github.com/sgl-project/sglang/issues/5528
**State**: open
**Created**: 2025-04-18T10:42:36+00:00
**Comments**: 11

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

On a single machine equipped with 4× NVIDIA L20Y (80GB) GPUs, when launching SGLang using the built-in router via:
`python -m sglang_router.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --dp 4
`
and sending multiple concurrent chat completion requests using multi-threading, only one worker appears to be actively handl

[... truncated for brevity ...]

---

## Issue #N/A: After enabling tensor parallelism (tp-size=2), there is no response

**Link**: https://github.com/sgl-project/sglang/issues/150
**State**: closed
**Created**: 2024-02-06T12:51:16+00:00
**Closed**: 2024-07-25T06:32:47+00:00
**Comments**: 6
**Labels**: inactive

### Description

my command is:
```shell
CUDA_VISIBLE_DEVICES="2,4" python -m sglang.launch_server --model-path  ./Yi-34B-Chat --trust-remote-code --port 30000 --tp-size 2 
``` 
when I run the demo code, **there is nothing returned.** 

```python
from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint

@function
def multi_turn_question(s, question_1, question_2):
    s += system("You are a helpful assistant.")
    s += user(question_1)
    s += assistant(gen("answer_1", max_tokens=256))
    s += user(question_2)
    s += assistant(gen("answer_2", max_tokens=256))

set_default_backend(RuntimeEndpoint("http://localhost:30000"))

state = multi_turn_question.run(
    question_1="What is the capital of the United States?",
    question_2="List two local attractions.",
)

for m in state.messages():
    print(m["role"], ":", m["content"])
```

**But when I  remove  "--tp-size 2 " in the command ,which means the model is only in 1 GPU , it wor

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] [DeepSeek-R1/V3] The description of --kv-cache-dtype in the documentation and the code is inconsistent.

**Link**: https://github.com/sgl-project/sglang/issues/3995
**State**: closed
**Created**: 2025-03-02T11:10:25+00:00
**Closed**: 2025-06-16T00:20:38+00:00
**Comments**: 5
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

In the documentation, the description for --kv-cache-dtype is: "we should not run it with any quantization arguments like --quantization fp8 --kv-cache-dtype fp8_e5m2." However, in the code implementation, if --kv-cache-dtype is not set, it defaults to using bfloat16. Is the documentation incorrect, or is there an issue with the code?

![I

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] DP + TP support

**Link**: https://github.com/sgl-project/sglang/issues/4765
**State**: closed
**Created**: 2025-03-25T14:14:27+00:00
**Closed**: 2025-05-26T00:19:55+00:00
**Comments**: 1
**Labels**: inactive

### Description

Hi all,

I'm going through the rank allocation part of the codebase. But I'm really confused about the setting of dp_size and tp_size. Support dp attention is not enabled, and we should have dp_size * tp_size = total_gpus_num.

From the code of `DataParallelController`, we have:
```
for dp_rank in range(server_args.dp_size):
            ...
            thread = threading.Thread(
                target=self.launch_tensor_parallel_group,
                args=(server_args, tmp_port_args, base_gpu_id, dp_rank),
            )
           ...
```
which indicates each `node` will have `dp_size` dp workers and each dp worker has a tensor parallel group initialized as follows:
```
def launch_tensor_parallel_group():
       ...
        # Launch tensor parallel scheduler processes
        scheduler_pipe_readers = []
        tp_size_per_node = server_args.tp_size // server_args.nnodes
        tp_rank_range = range(
            tp_size_per_node * server_args.node_rank,
            tp_size_per_node *

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Do we have any plan for supporting MiniCPM-V 2.6?

**Link**: https://github.com/sgl-project/sglang/issues/2461
**State**: closed
**Created**: 2024-12-12T03:25:08+00:00
**Closed**: 2025-01-18T22:17:00+00:00
**Comments**: 12
**Labels**: collaboration

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Do we have any plan for supporting MiniCPM-V 2.6?

To my experience this 8B model has better performance than other 7B vlm models

### Related resources

https://github.com/OpenBMB/MiniCPM-V
https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/minicpmv.py

---

## Issue #N/A: [Bug] Why isn't the enable_thinking parameter in sglang working as expected?

**Link**: https://github.com/sgl-project/sglang/issues/6529
**State**: open
**Created**: 2025-05-22T12:00:59+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

My sglang version is 0.4.6-post4.


The Docker-compose startup command for sglang is:

```yaml
version: '3.9'

services:
  Sglang_server:
    image: docker.1ms.run/lmsysorg/sglang:v0.4.6.post4-cu124
    shm_size: '10gb'
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=6,7
    volumes:
      - /data/sdv1/model:/model
    po

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] attention backend default choice

**Link**: https://github.com/sgl-project/sglang/issues/5064
**State**: closed
**Created**: 2025-04-04T08:13:51+00:00
**Closed**: 2025-05-21T09:29:52+00:00
**Comments**: 2
**Labels**: high priority, collaboration, flashinfer, performance, MLLM, deepseek

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

The standards we choose prioritize **performance first**, ease of use second (such as interface and installation), while also considering compatibility (such as older arch). Therefore, if in the future, the performance of different backends changes, we will still choose **the best performing one**.

1. NVIDIA

```
sm75 -> Triton
sm80, sm86, sm89 -> FlashInfer
sm90 -> FA3 (Llama, Qwen, Gemma), FlashInfer (Others)
sm100 -> FlashInfer

MLA
sm90 -> FA3 (DeepSeek)
sm100 -> FlashInfer (DeepSeek)

Other options
FlashMLA, cuDNN etc
```

SGLang will install the JIT version of FlashInfer on PyPI for a better user installation experience. Alternatively, the whl size limit of FlashInfer can be increased on PyPI. cc @yzh119 

F

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] _pickle.UnpicklingError: invalid load key, '\x11'.

**Link**: https://github.com/sgl-project/sglang/issues/5713
**State**: open
**Created**: 2025-04-24T12:26:27+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

_pickle.UnpicklingError: invalid load key, '\x11'.

### Reproduction

[2025-04-21 22:11:12] TokenizerManager hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tokenizer_manager.py", line 1169, in print_exception_wrapper
    await func()
  File "/sgl-workspace/sglang/python/sglang/

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Qwen2-VL-7B IndexError

**Link**: https://github.com/sgl-project/sglang/issues/2181
**State**: closed
**Created**: 2024-11-25T17:02:41+00:00
**Closed**: 2025-01-31T00:16:28+00:00
**Comments**: 4
**Labels**: bug, inactive

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

Occasionally, we will see a random "IndexError" which crashes sglang when serving Qwen2-VL-7B models. The crash is usually such that sglang will livelock, so the process will not exit, but no new requests will be servable. 

I have tried to rerun the requests again in a local interactive environment, but I cannot get an exact repro case 

[... truncated for brevity ...]

---

## Issue #N/A: [Notice] If someone can not run `examples/usage/llava/srt_llava_next_test.py` and meet the `rpc` error or `connection reset by peer` error.

**Link**: https://github.com/sgl-project/sglang/issues/494
**State**: closed
**Created**: 2024-06-01T11:43:44+00:00
**Closed**: 2024-08-01T01:08:51+00:00
**Comments**: 1
**Labels**: inactive

### Description

I found a workaround to fix it. 

By adding a custom port here:
```python
runtime = sgl.Runtime(
    model_path="lmms-lab/llama3-llava-next-8b",
    tokenizer_path="lmms-lab/llama3-llava-next-8b-tokenizer",
    port=8000,
)
```

---

## Issue #N/A: [Feature] Make random-range-ratio give symmetric distribution around --input-length (parity with vllm)

**Link**: https://github.com/sgl-project/sglang/issues/7253
**State**: open
**Created**: 2025-06-17T00:00:59+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Feature suggestion / request to change the way --random-range-ratio is used, as done in the vllm codebase 
[Fix range_ratio Bug in RandomDataset #16126](https://github.com/vllm-project/vllm/pull/16126)
 
There's another recent change at [[Bugfix] Fixed prompt length for random dataset](https://github.com/vllm-project/vllm/pull/17408/files#top) which may also be useful.

Some backstory: the syntax of --random-range-ratio looks identical in sglang and vllm, but the ranges in token lengths are quite different: 

sglang => [input_len * random_ratio, input_len]
vllm => [input_len * (1 - random_ratio), input_len * (1 + random_ratio)]

With a default of zero, this leads to sglang averaging half the input tokens for the sa

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] OpenAI API won't accept tool call result

**Link**: https://github.com/sgl-project/sglang/issues/5708
**State**: closed
**Created**: 2025-04-24T10:05:04+00:00
**Closed**: 2025-06-24T00:19:45+00:00
**Comments**: 1
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

https://github.com/sgl-project/sglang/blob/c998d04b46920f06d945fbef9023884a768723fc/python/sglang/srt/openai_api/protocol.py#L317

I notice that the `ChatCompletionRequest` is using `List[ChatCompletionMessageParam]` as messages, which do not support tool call. There is another `ChatMessage` supporting tool call but it's not used here. Any

[... truncated for brevity ...]

---

