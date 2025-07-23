# regular_contributors_5to20 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 4
- Closed Issues: 26

### Label Distribution

- inactive: 13 issues
- help wanted: 3 issues
- deepseek: 2 issues
- good first issue: 1 issues

---

## Issue #N/A: [Bug] perf drop when flashinfer update from 0.1.6 to 0.2.x

**Link**: https://github.com/sgl-project/sglang/issues/4642
**State**: closed
**Created**: 2025-03-21T02:38:40+00:00
**Closed**: 2025-05-22T00:19:06+00:00
**Comments**: 3
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

https://github.com/flashinfer-ai/flashinfer/issues/960

### Reproduction

https://github.com/flashinfer-ai/flashinfer/issues/960

### Environment

Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC ver

[... truncated for brevity ...]

---

## Issue #N/A: physical_expert_id error

**Link**: https://github.com/sgl-project/sglang/issues/6269
**State**: closed
**Created**: 2025-05-13T13:03:05+00:00
**Closed**: 2025-05-14T02:30:48+00:00
**Comments**: 1

### Description

branch: **deepseel_ep**

env: 2 node H20 (device each)

cmd :
python -m sglang.launch_server \
    --model-path /data2/deepseek-ai/DeepSeek-R1 \
    --port 2233 \
    --trust-remote-code \
    --dist-init-addr 10.93.75.30:12345 --nnodes 2 --node-rank $1 \
    --tp-size 16 \
    --enable-ep-moe \
    --mem-fraction-static 0.7 \
    --disable-radix-cache

error info : 
![Image](https://github.com/user-attachments/assets/90ca809d-fde5-4b90-8cf0-de24f47d8f0d)

question :
I want to use EP without MLA DP. Are there any potential configuration mistakes?  and i add  "--enable-dp-attention" the error info is same.

---

## Issue #N/A: [Bug] Fix #3161 break on ROCm

**Link**: https://github.com/sgl-project/sglang/issues/3289
**State**: closed
**Created**: 2025-02-04T13:22:32+00:00
**Closed**: 2025-02-05T18:41:31+00:00
**Comments**: 3

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

#3161 breaks AMD build, to fix.

### Reproduction

AMD CI!

### Environment

AMD ROCm

---

## Issue #N/A: Torch.compile Performance Tracking

**Link**: https://github.com/sgl-project/sglang/issues/1008
**State**: closed
**Created**: 2024-08-09T20:10:44+00:00
**Closed**: 2024-11-26T00:24:28+00:00
**Comments**: 2
**Labels**: inactive

### Description

torch.compile can accelerate small batch sizes for llama-3 8B. However,  it is sometimes slower for large batch size or tensor parallelism. We use this issue to track the performance and potential fixes.

## Instructions and results
```bash
# Benchmark llama-3-8B (TP=1, bs=1) with cuda graph
# Decode.  median latency: 0.00737 s, median throughput:    135.64 token/s
python3 -m sglang.bench_latency --model meta-llama/Meta-Llama-3-8B --batch-size 1 --input 128 --output 8

# Benchmark llama-3-8B (TP=1, bs=1) with torch.compile
# Decode.  median latency: 0.00642 s, median throughput:    155.67 token/s
python3 -m sglang.bench_latency --model meta-llama/Meta-Llama-3-8B --batch-size 1 --input 128 --output 8 --enable-torch-compile


# Benchmark llama-3-8B (TP=1, bs=128) with cuda graph
# Decode.  median latency: 0.01184 s, median throughput:  10815.07 token/s
python3 -m sglang.bench_latency --model meta-llama/Meta-Llama-3-8B --batch-size 128 --input 128 --output 8

# Benchmark 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] After deploying for a period of time (2 days), the speed slows down and the memory usage increases

**Link**: https://github.com/sgl-project/sglang/issues/2395
**State**: closed
**Created**: 2024-12-08T09:23:26+00:00
**Closed**: 2025-02-08T00:16:01+00:00
**Comments**: 3
**Labels**: inactive

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

After deploying for a period of time (2 days), the speed slows down and the memory usage increases
model is llava-onevision

### Reproduction

just like sglang official example, deploy server

### Environment

python: 3.9
sglang:0.3.0
cuda:12.1
torch:2.4.0
flashinfer:0.1.6+cu121torch2.4

---

## Issue #N/A: [Bug] The first request with "regex" is too slow

**Link**: https://github.com/sgl-project/sglang/issues/2420
**State**: closed
**Created**: 2024-12-09T10:17:04+00:00
**Closed**: 2025-02-16T00:18:18+00:00
**Comments**: 2
**Labels**: inactive

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

When I deployed a Qwen2-7B model on the L40, using a regex request, the response time after the first request exceeded 1 hour. Is this response time related to the length of the "regex"? Is there any way to speed up the process?

### Reproduction

Command
`CUDA_VISIBLE_DEVICES=4,5 python3 -m sglang.launch_server --model-path /data/ljl/dat

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] qwq-32b does not support concurrent requests.

**Link**: https://github.com/sgl-project/sglang/issues/4305
**State**: closed
**Created**: 2025-03-11T08:42:51+00:00
**Closed**: 2025-05-18T00:20:52+00:00
**Comments**: 4
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When I use the vllm benchmark to evaluate the performance of the QWQ-32B model deployed with sglang, I found that the model fails to process requests properly under high-concurrency conditions and keeps generating tokens continuously.

Specifically, when I set the max_concurrency of the vllm benchmark to 16, the following occurs at the ver

[... truncated for brevity ...]

---

## Issue #N/A: Task 004: Port Observability Module

**Link**: https://github.com/sgl-project/sglang/issues/7537
**State**: open
**Created**: 2025-06-25T20:16:15+00:00
**Comments**: 1

### Description

# Task 004: Port Observability Module

## Summary
Port the centralized observability module from existing implementation to consolidate metrics collection and improve monitoring capabilities across the router.

## Motivation
Currently:
- Metrics are scattered throughout the codebase
- No consistent metric naming convention
- Difficult to add new metrics
- No centralized configuration for observability

## Implementation Plan

### 1. Create Observability Module Structure
```rust
// src/infrastructure/observability/mod.rs
pub mod metrics;
pub mod logging;

use metrics_exporter_prometheus::PrometheusBuilder;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub struct ObservabilityConfig {
    pub metrics: MetricsConfig,
    pub logging: LoggingConfig,
}

pub struct MetricsConfig {
    pub enabled: bool,
    pub port: u16,
    pub host: String,
    pub buckets: Vec<f64>,
}

pub fn init_observability(config: ObservabilityConfig) -> Result<(), ObservabilityError> {
 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] mookcake error

**Link**: https://github.com/sgl-project/sglang/issues/6854
**State**: closed
**Created**: 2025-06-04T02:26:55+00:00
**Closed**: 2025-06-04T06:51:11+00:00
**Comments**: 1

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

**question**
mooncake library report  some error when using PD without DeepEP for deepseekR1 in main branch 

**prefill node launch command**
  ```
  SGL_ENABLE_JIT_DEEPGEMM=1  nohup python3 -m sglang.launch_server  --mem-fraction-static 0.85 --model-path ${model_path} \
     --disaggregation-mode prefill --host ${node_ip} --trust-remote-c

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Add Support for Structured Output Format

**Link**: https://github.com/sgl-project/sglang/issues/2696
**State**: closed
**Created**: 2025-01-01T16:55:53+00:00
**Closed**: 2025-01-02T18:43:40+00:00
**Comments**: 4

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

In many use cases, users need to extract or analyze data from the output generated by this tool/project. 

### Related resources

https://docs.vllm.ai/en/latest/usage/structured_outputs.html#experimental-automatic-parsing-openai-api

---

## Issue #N/A: [Bug] too many processes 

**Link**: https://github.com/sgl-project/sglang/issues/1382
**State**: closed
**Created**: 2024-09-11T03:12:46+00:00
**Closed**: 2024-11-14T19:26:58+00:00
**Comments**: 1

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

app.py 
```python
def set_qwen_runtime():
    runtime = sgl.Runtime(
        model_path="/root/luka/llm/cloth-attri-qwen-vl-chat-int4/checkpoint-6038-gptq-int4",
        tokenizer_path="/root/luka/llm/cloth-attri-qwen-vl-chat-int4/checkpoint-6038-gptq-int4",
        disable_cuda_graph=True,
        mem_fraction_static=0.90

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] stop_str of qwen2-vl template should be a tuple not a str

**Link**: https://github.com/sgl-project/sglang/issues/1832
**State**: closed
**Created**: 2024-10-29T07:42:03+00:00
**Closed**: 2024-10-29T20:32:35+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

https://github.com/sgl-project/sglang/blob/5e6c32657e384b023faf03d79e06f7727feedb7c/python/sglang/lang/chat_template.py#L147

### Reproduction

-

### Environment

-

---

## Issue #N/A: [Error]Input length (160062 tokens) exceeds the maximum allowed length (59862 tokens).

**Link**: https://github.com/sgl-project/sglang/issues/4048
**State**: closed
**Created**: 2025-03-04T03:59:50+00:00
**Closed**: 2025-05-05T00:20:08+00:00
**Comments**: 12
**Labels**: help wanted, inactive, deepseek

### Description

Hi, I am trying to use `sglang` to deploy a DeepSeek R1 serving program. The deployment command is as follows:

```shell
python -m sglang.launch_server \
--model-path  /models/deepseek-ai/deepseek-r1 \
--host 0.0.0.0 \
--port 8100 \
--tensor-parallel-size 8 \
--mem-fraction-static 0.9 \
--trust-remote-code \
--context-length 163840 \
--chunked-prefill-size 4096 \
--served-model-name DeepSeek-R1-Sglang-160k
```

Although I set the `--context-length` parameter to 160k and the serving program starts successfully, an error occurs when I send a request with content of length 160k. The error message is as follows:

```shell
if self.sampling_params.max_new_tokens > 0:
TypeError: '>' not supported between instances of 'NoneType' and 'int'
[2025-03-03 06:49:28 TP5] Input length (160062 tokens) exceeds the maximum allowed length (59862 tokens). Use a shorter input or enable --allow-auto-truncate.
[2025-03-03 06:49:28 TP1] Input length (160062 tokens) exceeds the maximum allowed length (59862 tok

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] deepseek v3 60 tokens/sec on deepseek API vs. 13 tokens/sec on sglang

**Link**: https://github.com/sgl-project/sglang/issues/3196
**State**: closed
**Created**: 2025-01-28T18:40:18+00:00
**Closed**: 2025-02-15T01:21:30+00:00
**Comments**: 29
**Labels**: help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

The PR for AMD + sglang and NVIDIA + sglang was that it was "fully" supported, but it seems something is off by the speed.  A single sequence runs at only order 13 tokens/sec for long generation with TTFT order 2 seconds.  This is consistent with vLLM as well.  True for either 8*MI300X or 8*H200 or 2*8*H200.

For only 37B parameters + 14B MOE parameters, this seems way too slow.  Also, deepseek API (before it started to break down) was order 60 tokens/sec early on and they advertise 60 tokens/sec.  This is more aligned with the parameters active.

What is missing from truly fully suppporting deepseek V3 and R1?  Can these features be enumerated and added in a roadmap?

### Related resources

_No response_

---

## Issue #N/A: Switch to non gated models

**Link**: https://github.com/sgl-project/sglang/issues/387
**State**: closed
**Created**: 2024-04-23T16:05:54+00:00
**Closed**: 2024-07-25T06:33:28+00:00
**Comments**: 2
**Labels**: inactive

### Description

When running
```
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```
I get
```
Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/resolve/main/config.json.
Access to model meta-llama/Llama-2-7b-chat-hf is restricted. You must be authenticated to access it.
```
I assume it's the same for most people. How about switching it to a non gated model, e.g. Mistral or a quantized 8B Llama 3?

---

## Issue #N/A: [Bug] Windows installation failure: “Filename too long” error when building wheel via pip

**Link**: https://github.com/sgl-project/sglang/issues/5644
**State**: closed
**Created**: 2025-04-22T17:06:38+00:00
**Closed**: 2025-06-22T00:21:55+00:00
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

When installing `sgl-kernel` directly from GitHub via pip, the wheel build fails with numerous `Filename too long` errors originating from submodules (especially FlashInfer). There is also a warning about an undefined extra `[srt]`:

https://gist.github.com/celsowm/a66016889d5c57030c4bbedc1978ec6f


### Reproduction

**Steps to Reproduce**

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] sglang-router should perform extra status check on workers upon startup in addition to port reachability

**Link**: https://github.com/sgl-project/sglang/issues/4208
**State**: closed
**Created**: 2025-03-08T12:09:22+00:00
**Closed**: 2025-05-10T00:18:08+00:00
**Comments**: 1
**Labels**: inactive

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

When Co-launch Router and Runtimes (with cuda-graph and torch-compile), worker start-up can take longer than 300s, result in health check timeout. 

Reproduce: start the router and server with `python3 -m sglang_router.launch_server  --enable-torch-compile`, I run with a Mistral 8*7B here. Then get error:

```
SingleProcess AUTOTUNE benchmarking takes 1.3427 seconds and 8.8429 seconds precompiling
AUTOTUNE mm(4x4096, 4096x8)
  mm 0.0134 ms 100.0%
  triton_mm_99 0.0134 ms 99.8% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, .......
SingleProcess AUTOTUNE benchmarking takes 1.3672 seconds and 8.3651 seconds precompiling
[Router (Rust)] 2025-03-08 02:33:09 - INFO - Worker http://0.0.0.0:31000 health

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Partial Deserialization in rust router implementation.

**Link**: https://github.com/sgl-project/sglang/issues/7298
**State**: open
**Created**: 2025-06-18T04:32:58+00:00
**Comments**: 2

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation
In the current SGLang router implementation (written in Rust), we support:
- Regular routing strategies: cache-aware, random, and round-robin
- Prefill-decode (PD) disaggregated routing: random and power-of-two (Po2) based

Previously, incoming requests were deserialized from raw bytes into dictionaries (maps) to extract minimal fields (e.g., stream). However, with the addition of PD routing requirements, fields like bootstrap_port and bootstrap_room need to be injected into the request object. As a result, the router now deserializes the full request into a fully typed struct.

This shift raises performance concerns regarding deserialization overhead, especially under high QPS.

### Goal
Evaluate and implement an o

[... truncated for brevity ...]

---

## Issue #N/A: ../aten/src/ATen/native/cuda/Indexing.cu:1236: indexSelectSmallIndex: block: [40,0,0], thread: [26,0,0] Assertion `srcIndex < srcSelectDimSize` failed.

**Link**: https://github.com/sgl-project/sglang/issues/473
**State**: closed
**Created**: 2024-05-25T01:53:51+00:00
**Closed**: 2024-08-13T01:05:23+00:00
**Comments**: 4
**Labels**: inactive

### Description

```
python -m sglang.launch_server --model-path liuhaotian/llava-v1.6-34b --tokenizer-path liuhaotian/llava-v1.6-34b-tokenizer --port=30020 --host="0.0.0.0" --tp-size=1 --random-seed=1234 --context-length=4096 &> 34b.log &
```

client:
```
pload = {'text': '<|im_start|>system\nAnswer the questions.<|im_end|>user<image>\nGive detailed information.<|im_end|>', 'sampling_params': {'max_new_tokens': 1024, 'temperature': 0.0, 'top_p': 1.0, 'presence_penalty': 0.14000000000000012, 'frequency_penalty': 2, 'stop': ['<|im_end|>']}, 'stream': False}
```

The pload also has the image in bytes form, e.g.:
```
data:image/png;base64,iVBORw0KGgoAAAANSU...
```

For this image:

![bigben](https://github.com/sgl-project/sglang/assets/2249614/7ecd8dd4-f934-4d87-87bc-60d98c5ea587)


client code:
```
        response = requests.post(
            url,
            json=pload,
            stream=False,
        )
```
stream False or True doesn't help.  url is just `'http://xxx.xxx.xx

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] --quantization w8a8_int8 report RuntimeError: mat_a must be a 2D tensor

**Link**: https://github.com/sgl-project/sglang/issues/5806
**State**: closed
**Created**: 2025-04-28T04:06:02+00:00
**Closed**: 2025-07-06T00:22:06+00:00
**Comments**: 3
**Labels**: inactive

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

1)sglang[all] version 0.4.5.* 0.4.6.* is error
python3 -m sglang.launch_server --model-path   RedHatAI/Qwen2.5-VL-7B-Instruct-quantized.w8a8   --host 0.0.0.0 --port 9080 --mem-fraction-static 0.8  --chat-template=qwen2-vl   --trust-remote-code --chunked-prefill-size 4096 --enable-torch-compile --quantization w8a8_int8

2)sglang[all] versio

[... truncated for brevity ...]

---

## Issue #N/A: OOM Error on A40 GPU [Jupyter notebook]

**Link**: https://github.com/sgl-project/sglang/issues/86
**State**: closed
**Created**: 2024-01-23T12:03:01+00:00
**Closed**: 2024-07-25T06:32:01+00:00
**Comments**: 6
**Labels**: inactive

### Description

I was trying the following code sample (adapted from the discussion in https://github.com/sgl-project/sglang/issues/81) - 

```
import sglang as sgl
from sglang import function, gen, set_default_backend, Runtime

@sgl.function
def tool_use(s, question):
    s += "To answer this question: " + question + ", "
    s += "I need to use a " + sgl.gen("tool", choices=["calculator", "web browser"]) + ". "
    if s["tool"] == "calculator":
        s += "The math expression is" + sgl.gen("expression")
    elif s["tool"] == "web browser":
        s += "The website url is" + sgl.gen("url")

runtime = Runtime(model_path='Model_Saves/teknium--OpenHermes-2.5-Mistral-7B')
set_default_backend(runtime)

driver_tool_use()
```
   
I firstly got the same error as described here: https://github.com/sgl-project/sglang/issues/41#issuecomment-1899347676
I then followed Solution 2 from this [comment](https://github.com/sgl-project/sglang/issues/41#issuecomment-1899354400) and the error dis

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] How to profile the performance of multi-machine Decode instances without starting Prefill instances?

**Link**: https://github.com/sgl-project/sglang/issues/6707
**State**: open
**Created**: 2025-05-28T12:11:05+00:00
**Comments**: 12

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

I hope to only profile the performance of Decode instances. I noticed that after starting a Decode instance, it will undergo warmup, and this warmup does not require the participation of Prefill instances. Can I start Decode instances in the way described in https://github.com/sgl-project/sglang/issues/6017 for starting multi-machine Decode instances and modify the warmup process of the Decode instances to achieve my goal?

### Related resources

_No response_

---

## Issue #N/A: [Bug] sglang-router failure

**Link**: https://github.com/sgl-project/sglang/issues/2361
**State**: closed
**Created**: 2024-12-05T09:41:25+00:00
**Closed**: 2025-04-02T00:18:13+00:00
**Comments**: 7
**Labels**: inactive

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

```bash 
python3 -m sglang_router.launch_server --quantization fp8 --enable-overlap-schedule   --model $LOCAL_PATH  --attention-backend $ATTENTION_BACKEND --stream-interval  $STREAM_INTERVAL --max-prefill-tokens $MODEL_LEN  --root-path $ROOT_PATH --trust-remote-code  --mem-frac $MEM_FRAC --tp $TP_SIZE --dp $DP_SIZE --kv-cache-dtype $KV_CA

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] The Reasoning Parser doesn't consider the situation that there's no `</think>` tag in the output.

**Link**: https://github.com/sgl-project/sglang/issues/4711
**State**: closed
**Created**: 2025-03-24T06:34:22+00:00
**Closed**: 2025-04-08T04:49:11+00:00
**Comments**: 8

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hi team, 

I encountered a situation that the output of `QWQ-32B` model doesn't output any `<think>` and `</think>` tag, especially when working with `--grammar-backend xgrammer` which should treat the response content as `.content` instead of `.reasoning_content`, while SGLang currently do the opposite. 

### Reproduction

the server is l

[... truncated for brevity ...]

---

## Issue #N/A: Chinese Regex BUG in req.jump_forward_map.jump_forward_byte

**Link**: https://github.com/sgl-project/sglang/issues/549
**State**: closed
**Created**: 2024-06-15T03:01:26+00:00
**Closed**: 2024-06-16T13:45:05+00:00
**Comments**: 1

### Description

reproduce code:
```
@sgl.function
def fabric_gen(s):
    s+=sgl.user("用JSON数组列出一种衣服领形")
    r= '\\["(娃娃领|高领|海军领|斜领|连帽|翻领|一字领)"\\]'
    s += sgl.assistant(sgl.gen("json_output", max_tokens=256, regex=r))

if __name__ == "__main__":
    set_runtime() 
    print(fabric_gen.run().text())
```

Here is full error information

```
Traceback (most recent call last):
  File "/root/luka/sglang/python/sglang/srt/managers/controller/tp_worker.py", line 199, in exposed_step
    self.forward_step()
  File "/root/anaconda3/envs/sglang/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/luka/sglang/python/sglang/srt/managers/controller/tp_worker.py", line 229, in forward_step
    self.forward_decode_batch(self.running_batch)
  File "/root/luka/sglang/python/sglang/srt/managers/controller/tp_worker.py", line 562, in forward_decode_batch
    jump_forward_reqs = bat

[... truncated for brevity ...]

---

## Issue #N/A: [BUG] some problems with HiRadixCache

**Link**: https://github.com/sgl-project/sglang/issues/5499
**State**: closed
**Created**: 2025-04-17T13:45:53+00:00
**Closed**: 2025-04-21T18:46:49+00:00
**Comments**: 10

### Description

@xiezhq-hermann 
The background of the problem we described: 
We use HiRadixCache in the scenario of PD separation, write_back strategy. The local radix tree will send update events when nodes are added and deleted in rank 0, and the global radix tree will be adjusted according to the update events. When the request comes, we first match according to the global radix tree, and decide to choose P nodes and D nodes according to the number of prefix matches and load. We found that the number of matches in the global tree is sometimes much larger than the number of matches in the local number under the premise of distinguishing between instances. It looks like the host indices is not matched.
In the process of troubleshooting the problem, we encountered the following problems:

## 1、`pending_nodes` is not used
https://github.com/sgl-project/sglang/blob/8f783c1943af25e5bbccff628ba4385579b044e1/python/sglang/srt/mem_cache/hiradix_cache.py#L141-L179

`pending_nodes` is not used, this will cau

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] generate till it reaches model's context length

**Link**: https://github.com/sgl-project/sglang/issues/4511
**State**: closed
**Created**: 2025-03-17T11:30:10+00:00
**Closed**: 2025-05-01T17:24:47+00:00
**Comments**: 15
**Labels**: good first issue, help wanted

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

By default, sglang uses 128 tokens as the max_new_tokens even if None is given. Is there a way to specify that the model generated till it reaches it max model length? I have a different sequences of generation using sglang frontend language, so I cant keep track of the input lengths etc. I cant use one single max_new_tokens for different generations. If I put the max_new_tokens hardcoded, the input length is being restricted.

Let me know if I can help implemenatation of such a feature, it would be very helpful. I am thinking of an implementation that takes the current input token length and mas token length to calculate max_new_tokens?
Please let me know if this is valid concern.

### Related resources

_No respo

[... truncated for brevity ...]

---

## Issue #N/A: 为啥这里是异步拷贝，然后后面直接使用了？没有显示同步？

**Link**: https://github.com/sgl-project/sglang/issues/2211
**State**: closed
**Created**: 2024-11-27T06:16:40+00:00
**Closed**: 2024-11-27T08:21:05+00:00
**Comments**: 3

### Description

https://github.com/sgl-project/sglang/blob/0b46b951ae088dd22fe980acc7d855947ce2537f/python/sglang/srt/managers/schedule_batch.py#L982

 new_indices = torch.tensor(keep_indices, dtype=torch.int32).to(
          self.device, non_blocking=True
   )
 self.req_pool_indices = self.req_pool_indices[new_indices]

直接使用new_indices？

---

## Issue #N/A: [Bug] Flashinferv0.2.2.post1 shows Unsupported max_mma_kv: 0 error on L40 , when deploying Deepseek-V2-Lite-chat with --enable-flashinfer-mla

**Link**: https://github.com/sgl-project/sglang/issues/4196
**State**: closed
**Created**: 2025-03-08T05:53:08+00:00
**Closed**: 2025-03-08T18:02:25+00:00
**Comments**: 1
**Labels**: deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

    return self.forward_extend(
    o, _ = self.prefill_wrapper_ragged.forward_return_lse(
    return self.run_return_lse(q, k, v)
    self._cached_module.ragged_run(*run_args)
    ragged_run_func(
    return self._op(*args, **(kwargs or {}))
    return func(*args, **kwargs)
    return self._op(*args, **(kwargs or {}))
RuntimeError: Error 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] current `select_generate_worker` holds Mutex lock for large logical code scopes

**Link**: https://github.com/sgl-project/sglang/issues/6996
**State**: open
**Created**: 2025-06-09T09:35:51+00:00
**Comments**: 1

### Description

I have realized that the method `select_generate_worker` in `router.rs` holds some Mutex's locks for more than desirable code chunks. For this reason, it would be desirable to refactor the code to make sure locks are hold only for the necessary time being. 

Moreover, there are a few TODO's regarding that same method that should be addressed.

---

