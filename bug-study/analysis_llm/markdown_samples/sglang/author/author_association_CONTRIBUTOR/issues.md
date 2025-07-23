# author_association_CONTRIBUTOR - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 5
- Closed Issues: 25

### Label Distribution

- inactive: 5 issues
- good first issue: 4 issues
- help wanted: 2 issues
- high priority: 2 issues
- feature: 1 issues
- enhancement: 1 issues

---

## Issue #N/A: The `choices` normalised logprobs calculation returns poor results due to bias for longer-token options

**Link**: https://github.com/sgl-project/sglang/issues/523
**State**: closed
**Created**: 2024-06-10T12:56:23+00:00
**Closed**: 2024-08-05T10:27:50+00:00
**Comments**: 10

### Description

## Problem
I've noticed that the `gen(choices=[...])` functionality sometimes performs poorly, even for simple tasks. This is due to a flawed normalised logprobs calculation. The calculation biases options that comprise more tokens, where the latter tokens are highly predictable given the prior tokens.

## Reproducible Example
This is most easily seen in choices with token overlap, so I've constructed a contrived example that illustrates this. The outputs are generated with [llama 3 8B instruct](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF), which should breeze through this task under normal circumstances.
```python
import sglang as sgl
import textwrap

# Define answer choices with overlapping substrings and tokenised forms
# assumes llama 3 8B tokeniser
choices_and_tokenised_forms = [
    ("organ", ["organ"]),
    ("organism", ["organ", "ism"]),
    ("organisation", ["organisation"]),
    ("organelle", ["org", "ane", "lle"]),
    ("organometallic",

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] DeepError assert self.num_experts % self.tp_size == 0

**Link**: https://github.com/sgl-project/sglang/issues/6913
**State**: closed
**Created**: 2025-06-06T06:13:02+00:00
**Closed**: 2025-06-06T06:33:11+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

![Image](https://github.com/user-attachments/assets/28e0ae9a-4116-4533-a9e5-7d8bfad97eb4)

this refactor will make deepep assert error, @ch-wan 

### Reproduction

python3 -m sglang.launch_server --model-path=/models/DeepSeek-R1-BF16 --tp-size 32 --attention-backend flashinfer --trust-remote-code --nnodes 4 --node-rank 0 --host 0.0.0.0 --p

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] head_dim 96 not supported

**Link**: https://github.com/sgl-project/sglang/issues/1159
**State**: closed
**Created**: 2024-08-20T06:18:03+00:00
**Closed**: 2024-09-11T16:48:11+00:00
**Comments**: 7

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

Unable to load [vonjack/Phi-3-mini-4k-instruct-LLaMAfied](https://huggingface.co/vonjack/Phi-3-mini-4k-instruct-LLaMAfied)

Stacktrace:
```
server_args=ServerArgs(model_path='vonjack/Phi-3-mini-4k-instruct-LLaMAfied', tokenizer_path='vonjack/Phi-3-mini-4k-instruct-LLaMAfied', tokenizer_mode='auto', skip_tokenizer_init=False, 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] support /v1/completions suffix parameter for completion

**Link**: https://github.com/sgl-project/sglang/issues/3429
**State**: closed
**Created**: 2025-02-09T14:30:57+00:00
**Closed**: 2025-04-14T00:19:34+00:00
**Comments**: 7
**Labels**: inactive, feature

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

parameter suffix is not supported in sglang's openapi  v1/completions yet. but it's necessary for code completion.
can I support this?

### Related resources

_No response_

---

## Issue #N/A: [Bug] bench_speculative.py got error

**Link**: https://github.com/sgl-project/sglang/issues/4536
**State**: closed
**Created**: 2025-03-18T04:30:41+00:00
**Closed**: 2025-03-19T01:43:09+00:00
**Comments**: 4

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

run `bench_speculative.py` scripts 
```
python3 bench_speculative.py --model-path DeepSeek-R1 --speculative-draft-model-path deepseek-r1-nextn --tp-size 8 --trust-remote-code --batch-size 16 --steps 2 --topk 1  --num_draft_tokens 2 4 8 --context-len 2048 --mem-fraction-static 0.9 --enable-flashinfer-mla
```

got error:

```
[2025-03-18 03:

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Add SageMaker Support

**Link**: https://github.com/sgl-project/sglang/issues/3739
**State**: closed
**Created**: 2025-02-21T03:13:58+00:00
**Closed**: 2025-02-21T22:49:50+00:00
**Comments**: 3

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Hi.

SageMaker Endpoints will listen for invocations at `/invocations` and for health checks at `/health` ([ref](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html#your-algorithms-inference-code-container-response)). On modifying the server to listen on these endpoints, sglang can be used for SageMaker endpoints for hosting models. Is it possible to add support for this?

Thanks

### Related resources

_No response_

---

## Issue #N/A: [Bug] Inaccurate or Inconsistent Output in Qwen2.5-VL Multi-Image Testing with sglang

**Link**: https://github.com/sgl-project/sglang/issues/4123
**State**: closed
**Created**: 2025-03-06T03:59:30+00:00
**Closed**: 2025-03-06T16:06:04+00:00
**Comments**: 3

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When testing multi-image input with `sglang` using the Qwen2.5-VL model, the output descriptions of the image contents are inaccurate or inconsistent. Compared to the output from directly calling the same model with `transformers`, the results from `sglang` show significant differences in describing the commonalities between multiple image

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support Deepseek-vl2-tiny model, in which mla is disabled

**Link**: https://github.com/sgl-project/sglang/issues/5537
**State**: closed
**Created**: 2025-04-18T14:07:50+00:00
**Closed**: 2025-04-30T08:45:22+00:00
**Comments**: 2

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

According to https://github.com/sgl-project/sglang/issues/2653, the Deepseek-vl2 models are supported, but not all models in the series are supported as I used. Deepseek-vl2's models series is composed of three variants: DeepSeek-VL2-Tiny, DeepSeek-VL2-Small and DeepSeek-VL2, with 1.0B, 2.8B and 4.5B activated parameters respectively. The tiny models has different model structure with small and normal model, which MLA (Multi-Head Latent Attention) is disabled. And if using DeepseekV2ForCasualLLM as a language model, the qk_nope_head_dim and qk_rope_head_dim  added as qk_head_dim  will cause a ZeroDivisionError in the later sampling var calculation for **-0.5 operation (https://github.com/sgl-project/sglang/blob/bfa

[... truncated for brevity ...]

---

## Issue #N/A: [BUG] Flashinfer 0.0.3 compat with Sglang

**Link**: https://github.com/sgl-project/sglang/issues/283
**State**: closed
**Created**: 2024-03-12T00:33:39+00:00
**Closed**: 2024-03-12T13:45:59+00:00
**Comments**: 4

### Description

Using flashinfer 0.0.3 requires one line change #282 but there is a compat issue where same model runs fine on 0.0.2 but under 0.0.3 throws an infinite loop of the following on sglang:

```
Exception in ModelRpcClient:
Traceback (most recent call last):
  File "/root/miniconda3/lib/python3.11/site-packages/sglang/srt/managers/router/model_rpc.py", line 184, in exposed_step
    self.forward_step()
  File "/root/miniconda3/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/lib/python3.11/site-packages/sglang/srt/managers/router/model_rpc.py", line 211, in forward_step
    self.forward_decode_batch(self.running_batch)
  File "/root/miniconda3/lib/python3.11/site-packages/sglang/srt/managers/router/model_rpc.py", line 505, in forward_decode_batch
    next_token_ids, _ = batch.sample(logits)
                        ^^^^^^^^^^^^^^^^^^^^
  File "/root/

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Missing tool call id if tool call index >0 in streaming tool call output.

**Link**: https://github.com/sgl-project/sglang/issues/7048
**State**: closed
**Created**: 2025-06-10T10:03:45+00:00
**Closed**: 2025-06-11T02:27:30+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

In streaming function call result, the tool call id will be "None" if index > 0.
here is the streaming return with 2 tool calls, we can see "index=1 id=None"
```
[ChoiceDeltaToolCall(index=0, id='call_UQzAnUrYQreNoTmmHGnNyA', function=ChoiceDeltaToolCallFunction(arguments='', name='get_current_temperature'), type='function')]
[ChoiceDeltaT

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Improve transmission performance

**Link**: https://github.com/sgl-project/sglang/issues/6189
**State**: closed
**Created**: 2025-05-11T05:05:08+00:00
**Closed**: 2025-07-11T00:20:22+00:00
**Comments**: 1
**Labels**: inactive

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

When transmitting tensors via zmq from tokenizer manager (where processors are called) to scheduler, recv_pyobj leads to severe performance decrease. Maybe it is because of the weak performance of pickle

The picture shows the comparison of returning all medias in one mm_item and in respective mm_items in the processor. As the number of medias increases, this problem becomes prominent.

<img width="2166" alt="Image" src="https://github.com/user-attachments/assets/6f59be35-5d2c-4476-9b4b-25ca429cafb4" />

### Related resources

_No response_

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

## Issue #N/A: [Bug] Master node didn't detect worker node not functional, requests hang until timeout

**Link**: https://github.com/sgl-project/sglang/issues/4780
**State**: closed
**Created**: 2025-03-26T04:24:15+00:00
**Closed**: 2025-04-23T12:50:53+00:00
**Comments**: 8

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Context: 

The issue was identified after six days of stable operation.

Behavior:

From master node (aka. node rank 0) everything looks normal, API server still responding requests to `/health` API endpoint, but `/health_generate` will get 503 because of timeout: 

<img width="1730" alt="Image" src="https://github.com/user-attachments/ass

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Poor generation quality with Qwen2-57B-A14B-Instruct model

**Link**: https://github.com/sgl-project/sglang/issues/943
**State**: closed
**Created**: 2024-08-06T02:23:13+00:00
**Closed**: 2024-08-06T07:07:49+00:00
**Comments**: 3

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

We noticed poor generation quality with the [Qwen2-57B-A14B-Instruct](https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct) model (compared with vLLM). Here are some examples:

- prompt: `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n请介绍一下杭州<|im_end|>\n<|im_start|>assistant\n<|im_start|>assistant\n`
- sampling params:
  - temperature: 0.7
  - top_k: 20
  - top_p: 0.8
  - repetition_penalty: 1.05
- vLLM outputs: `对不起，我之前误解了你的问题。作为一个AI语言模型，我没有个人经验或主观意见。以下是关于杭州市的一些基本信息：\n\n杭州市是中国浙江省的省会城市，位于中国东南部，是长江三角洲地区的重要城市之一。杭 州以其美丽的西湖而闻名于世，也是中国历史文化名城和重

[... truncated for brevity ...]

---

## Issue #N/A:  typo in  sglang/srt/server.py  Runtime  log_evel

**Link**: https://github.com/sgl-project/sglang/issues/478
**State**: closed
**Created**: 2024-05-26T19:55:11+00:00
**Closed**: 2024-07-01T17:46:20+00:00
**Comments**: 1

### Description

There is a typo in  sglang/srt/server.py  Runtime

**log_evel** should be **log_level**

`class Runtime:
    def __init__(
        self,
        log_evel: str = "error",
        model_overide_args: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        """See the arguments in server_args.py::ServerArgs"""
        self.server_args = ServerArgs(*args, log_level=log_evel, **kwargs)
`

https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server.py#L254
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server.py#L260


This typo causes an error when using  fastchat.serve.sglang_worker
sglang.srt.server_args.ServerArgs() got multiple values for keyword argument 'log_level'

> python3 -m fastchat.serve.sglang_worker --model-path models/llava-v1.5-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --multimodal
2024-05-26 14:51:19 | ERROR | stderr | Traceback (most recent call last):
2024-05-26 14:51:19 | ERROR | stderr |   File "/

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support `abort_request` in the router

**Link**: https://github.com/sgl-project/sglang/issues/6531
**State**: open
**Created**: 2025-05-22T16:22:35+00:00
**Comments**: 1
**Labels**: good first issue

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Hi,

I'd like to use the new `abort_request` API: https://github.com/sgl-project/sglang/blob/58f10679e1850fdc86046057c23bac5193156de9/python/sglang/srt/entrypoints/http_server.py#L549

However, I'm using multiple SGLang servers behind a router, and currently the router doesn't seem to support `abort_request`: https://github.com/sgl-project/sglang/blob/main/sgl-router/src/server.rs

I'm wondering if it makes sense to also support `abort_request` in the router. Thanks!

### Related resources

_No response_

---

## Issue #N/A: [Bug] Inference with RadixAttention，but output weirdly

**Link**: https://github.com/sgl-project/sglang/issues/1959
**State**: closed
**Created**: 2024-11-08T09:32:46+00:00
**Closed**: 2024-11-08T09:33:17+00:00
**Comments**: 0

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

I'm trying to adjust my own LLM inference code, I have replaced the self-attention with RadixAttention, and some necessary components. But I found that the output is weird:

```python
# prompt: 

"which city is the capital of China?"

# output: 

"""
ijingBeijing is the capital of china.B. Beijing is the capital of china.
Which 

[... truncated for brevity ...]

---

## Issue #N/A: [Kernel] cuDNN attention backend

**Link**: https://github.com/sgl-project/sglang/issues/2272
**State**: open
**Created**: 2024-11-30T06:36:16+00:00
**Comments**: 3
**Labels**: enhancement, good first issue, help wanted, high priority, inactive

### Description

cuDNN provides very fast attention implementation and it is well maintained by NVIDIA. We would like to add a new attention backend based on cudnn.  

## Steps
1. Learn this cudnn paged attention python api. https://github.com/NVIDIA/cudnn-frontend/blob/v1.8.0/samples/python/52_scaled_dot_product_attention_with_paged_caches.ipynb
2. Add a new attention backend "cudnn" here https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/attention
3. We should be able to use it with `python3 -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --attention-backend cudnn`

---

## Issue #N/A: Continues batch technical for different length prompt

**Link**: https://github.com/sgl-project/sglang/issues/74
**State**: closed
**Created**: 2024-01-22T08:45:37+00:00
**Closed**: 2024-01-23T04:40:05+00:00
**Comments**: 4

### Description

Suppose model's max-model-length is 8096, and there is two requests, one prompt length is 8, another is 10. And how concat them, pad them to 8096 or pad them to 10 or truncate them to 8.
I feel you will pad them to 16. But I don't seek the code. Can you tell me the location?
Thanks.

---

## Issue #N/A: [Bug] Docker image (>=0.4.6.post1) suboptimal performance on H100 multi-node due to Incompatible NCCL

**Link**: https://github.com/sgl-project/sglang/issues/5980
**State**: closed
**Created**: 2025-05-02T14:59:44+00:00
**Closed**: 2025-05-03T15:10:02+00:00
**Comments**: 3

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

In `0.4.6.post1` and `0.4.6.post2` docker images, `nvidia-nccl-cu12` package is manually upgraded to 2.26.2: 

![Image](https://github.com/user-attachments/assets/c2a0343a-9a6a-4ee3-96b5-9e52c6e147de)

ref: https://hub.docker.com/layers/lmsysorg/sglang/v0.4.6.post2-cu124/images/sha256-b889741508e27fadd4000c70cb6b3f4612640a48a969ef0bc0fec41

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] mtp support dp-attention

**Link**: https://github.com/sgl-project/sglang/issues/6080
**State**: closed
**Created**: 2025-05-07T07:58:04+00:00
**Closed**: 2025-06-17T07:33:29+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

mtp support dp-attention

### Related resources

_No response_

---

## Issue #N/A: [Bug] Executing Qwen2.5-Omni-7B on SGLang 0.4.4 post2: AttributeError: 'Qwen2_5OmniConfig' object has no attribute 'hidden_size'

**Link**: https://github.com/sgl-project/sglang/issues/4862
**State**: closed
**Created**: 2025-03-28T15:00:39+00:00
**Closed**: 2025-05-30T08:43:41+00:00
**Comments**: 4
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hi,

When I try to launch Omni 7B on the [SGLang runtime](https://github.com/didier-durand/llms-in-clouds/blob/main/docs/sglang.md) (working already fine for multiple other Qwen models that I tested), I get the error mentioned here above:

`AttributeError: 'Qwen2_5OmniConfig' object has no attribute 'hidden_size'`

I understand that it's p

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] min_p_sampling_from_probs() related crashes

**Link**: https://github.com/sgl-project/sglang/issues/3201
**State**: closed
**Created**: 2025-01-29T02:46:51+00:00
**Closed**: 2025-02-02T19:50:45+00:00
**Comments**: 2

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

sglang srt crashes with log:
```
[2025-01-28 17:58:17 TP0] TpModelWorkerClient hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/tp_worker_overlap_thread.py", line 109, in forward_thread_func
    self.forward_thread_func_()
  File "/usr/local/lib/python3.10/dist-packages/torch/uti

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] DeepEP Low Latency failed on 2 node (8*H20)

**Link**: https://github.com/sgl-project/sglang/issues/5186
**State**: closed
**Created**: 2025-04-09T07:28:09+00:00
**Closed**: 2025-04-10T02:05:04+00:00
**Comments**: 3

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

We observe successful single-node (8×H20) deployment of DeepSeek-R1 with SGLang+DeepEP (deep-ep-mode=auto), but encounter failures in 2-node configuration with the following error:

```python
[2025-04-09 10:35:31 TP2] Scheduler hit an exception: Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/sglang/srt/m

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] support `gather` instead of `all_gather` when gathering the logits

**Link**: https://github.com/sgl-project/sglang/issues/3365
**State**: open
**Created**: 2025-02-07T07:14:12+00:00
**Comments**: 8
**Labels**: good first issue, help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

We noticed that in the `_get_logits` function of vllm, `gather` instead of `all_gather` will be used under certain conditions (the main condition is that for non-tpu devices):
Code link:

- [logits = tensor_model_parallel_gather(logits)](https://github.com/vllm-project/vllm/blob/6e1fc61f0fb90c37f0d4a1a8f76235a6e4e1103c/vllm/model_executor/layers/logits_processor.py#L101C22-L101C50)

- [condition of whether using `all_gather` or `gather`](https://github.com/vllm-project/vllm/blob/6e1fc61f0fb90c37f0d4a1a8f76235a6e4e1103c/vllm/model_executor/layers/logits_processor.py#L53-L57)

The change from using `all_gather` to `gather` is initially added in this PR for your reference: https://github.com/vllm-project/vllm/pull/222

[... truncated for brevity ...]

---

## Issue #N/A: Mistral model no longer loads following PR#101

**Link**: https://github.com/sgl-project/sglang/issues/107
**State**: closed
**Created**: 2024-01-26T15:06:54+00:00
**Closed**: 2024-01-26T17:38:45+00:00
**Comments**: 2

### Description

The `get_model_cls_by_arch_name` introduced in [Dynamic model class loading PR](https://github.com/sgl-project/sglang/pull/101) removes the hard-coded mapping between `MistralForCausalLM` and `LlamaForCausalLM` causing issues trying to local host [Mistral-7b](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) model as of sglang version 0.1.9. I have tested that adding the following simple `models/mistral.py` file allows hosting the mistral-7b model.

```python
from sglang.srt.models.llama2 import LlamaForCausalLM


class MistralForCausalLM(LlamaForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


EntryClass = MistralForCausalLM
```

---

## Issue #N/A: [Feature] Support encoder models for flash_infer backend

**Link**: https://github.com/sgl-project/sglang/issues/6050
**State**: open
**Created**: 2025-05-06T09:41:33+00:00
**Comments**: 1
**Labels**: good first issue

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

              @DavidBao03 hi, currently only support encoder model with torch_native attn backend and triton attn backend. Other attn backend is not supported yet.

_Originally posted by @woodx9 in https://github.com/sgl-project/sglang/issues/4887#issuecomment-2847365703_

### Related resources

_No response_

---

## Issue #N/A: How create a new branch?

**Link**: https://github.com/sgl-project/sglang/issues/69
**State**: closed
**Created**: 2024-01-21T12:27:58+00:00
**Closed**: 2024-01-21T13:02:51+00:00
**Comments**: 2

### Description

I just fix some bugs and support a new model. I hope create a new branch.

---

## Issue #N/A: [RFC] Remote KV Connector for SGLang Global Cache Reuse and PD 

**Link**: https://github.com/sgl-project/sglang/issues/7746
**State**: open
**Created**: 2025-07-03T12:57:07+00:00
**Comments**: 19
**Labels**: high priority

### Description

co-authors: @yizhang2077 


# \[RFC\] Remote KV Connector for SGLang Global Cache Reuse

## 1. Abstract

This RFC proposes a Remote KVCache Connector System to enable global KV cache reuse **across SGLang nodes**, solving redundant computation problems in multi-turn conversation scenarios and achieving global compute-to-storage conversion. The system introduces a connector abstraction layer that allows nodes to store and retrieve KV cache data from external storage, enabling prefix-based cache matching and reuse across distributed inference workers.

Key benefits include:

• **Global KV Cache Reuse**: Reducing redundant computation via cross-node global KV cache sharing through Global Prefix Index Management capabilities, achieving ~50% TTFT  reduction in Qwen-32B 4TP multi-turn dialogues

• **Flexible Storage Backend**: Supporting diverse storage backends via a universal KVConnector interface; enabling direct RDMA-based HBM KV Cache access for high-throughput data transfer

• **Seamle

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Overlap mode scheduler doesn't work for bench_serving with given request rate

**Link**: https://github.com/sgl-project/sglang/issues/2312
**State**: closed
**Created**: 2024-12-02T09:07:54+00:00
**Closed**: 2024-12-02T13:10:30+00:00
**Comments**: 1

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

When I tried using a overlapped scheduler to serve my model and use sglang.bench_serving to evluate performance with a given request rate, the scheduler got stuck and the reporting are as follows：
`Exception in thread Thread-3 (forward_thread_func):
Traceback (most recent call last):
  File "/state/partition/ykchen/conda/envs/sglang/lib

[... truncated for brevity ...]

---

