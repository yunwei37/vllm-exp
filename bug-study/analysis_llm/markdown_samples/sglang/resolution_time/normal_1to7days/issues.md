# normal_1to7days - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- good first issue: 3 issues
- help wanted: 2 issues
- high priority: 2 issues
- bug: 1 issues
- collaboration: 1 issues
- speculative-decoding: 1 issues
- enhancement: 1 issues
- await-response: 1 issues

---

## Issue #N/A: [Bug] start_profile interface makes server crash

**Link**: https://github.com/sgl-project/sglang/issues/6514
**State**: closed
**Created**: 2025-05-22T03:27:26+00:00
**Closed**: 2025-05-26T05:36:04+00:00
**Comments**: 1
**Labels**: bug

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I user `curl --location --request POST 'http://localhost:30000/start_profile' \
--header 'Content-Type: application/json' \
--data-raw '{
    "output_dir": "/models/profile",
    "num_steps": 10, 
    "activities": ["CPU", "GPU", "MEM", "CUDA_PROFILER"]
}'
`
to get the nsys profiling result, but server will crash when results are saved.

t

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Error while serving deepseek-ai/DeepSeek-V2-Lite in NVIDIA A40-48Q

**Link**: https://github.com/sgl-project/sglang/issues/3451
**State**: closed
**Created**: 2025-02-10T04:38:38+00:00
**Closed**: 2025-02-14T05:58:25+00:00
**Comments**: 6

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Sglang successfully deploys `deepseek-ai/DeepSeek-V2-Lite `, but fails to serve request. 

`Deployment and Error While Serving`

```
root@vultr:~# python3 -m sglang.launch_server --model-path $MODEL_ID --port 8000 --trust-remote-code                        
[2025-02-09 20:33:04] server_args=ServerArgs(model_path='deepseek-ai/DeepSeek-V2-Li

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Document is different with source of --reasoning-parser value

**Link**: https://github.com/sgl-project/sglang/issues/6023
**State**: closed
**Created**: 2025-05-05T08:39:15+00:00
**Closed**: 2025-05-11T15:22:47+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Document is different with source of --reasoning-parser value。

In document, https://docs.sglang.ai/backend/separate_reasoning.html,

```
Currently, SGLang supports the following reasoning models:

[DeepSeek R1 series](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d): The reasoning content is wrapped wit

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Using hiradix_cache with dp attention cause hang

**Link**: https://github.com/sgl-project/sglang/issues/7158
**State**: closed
**Created**: 2025-06-13T11:29:18+00:00
**Closed**: 2025-06-20T03:34:37+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I tried hicache with this using dp-attention, and the latest `0.4.7` version of sglang

However, sometimes server stuck before fask request brfore server fire up, sometimes stuck when benchmarking。

with print something in writing check function of `hiradix_cache.py`，I found that the server stuck before `all_reduce` function. I think, with

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Cannot run Janus Pro

**Link**: https://github.com/sgl-project/sglang/issues/4534
**State**: closed
**Created**: 2025-03-18T03:38:48+00:00
**Closed**: 2025-03-23T05:10:08+00:00
**Comments**: 3

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I've noticed there's a recently merged PR to support Janus Pro models #3203 

SGLang version: 0.4.4.post1

```
Mar 18 11:23:41 systemd[1]: Started SGLang Router Serve.
Mar 18 11:23:44 sgl-router-janus-pro-7b[931930]: INFO 03-18 11:23:44 __init__.py:190] Automatically detected platform cuda.
Mar 18 11:23:46 sgl-router-janus-pro-7b[931930]: 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Cannot run bitsandbytes llama models 

**Link**: https://github.com/sgl-project/sglang/issues/2600
**State**: closed
**Created**: 2024-12-26T15:51:48+00:00
**Closed**: 2024-12-29T06:21:40+00:00
**Comments**: 3
**Labels**: good first issue, help wanted

### Description

The issue is the same as https://github.com/sgl-project/sglang/issues/2556, but for llama models. We should be able to fix with a similar approach.

The following command crashes.
```
python3 -m sglang.bench_one_batch --model unsloth/llama-3-8b-bnb-4bit --load-format bitsandbytes
```

```
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
[rank0]: Traceback (most recent call last):
[rank0]:   File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
[rank0]:     return _run_code(code, main_globals, None,
[rank0]:   File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
[rank0]:     exec(code, run_globals)
[rank0]:   File "/root/sglang/python/sglang/bench_one_batch.py", line 470, in <module>
[rank0]:     main(server_args, bench_args)
[rank0]:   File "/root/sglang/python/sglang/bench_one_batch.py", line 434, in main
[rank0]:     work_func(server_args, port_args, bench_args, 0)
[rank0]:   File "/root/sglang/python/sglang/bench_on

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] cannot load prequantized model with scalar weight scale

**Link**: https://github.com/sgl-project/sglang/issues/4594
**State**: closed
**Created**: 2025-03-19T20:44:35+00:00
**Closed**: 2025-03-22T07:47:54+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Right now after loading the model and converting the weight scale to channel wise, there's an implicit assumption that the weight scale tensors in model weight is 1-D tensor. This is not the case for modelopt-quantized FP8 in fp8 cutlass supported hardware, since QKVParalleLinear will go through a requantization to the same scale.

### Rep

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Qwen3: Incorrect response field (reasoning_content instead of content) when enable_thinking=false with streaming enabled

**Link**: https://github.com/sgl-project/sglang/issues/5874
**State**: closed
**Created**: 2025-04-29T06:46:18+00:00
**Closed**: 2025-05-01T02:44:38+00:00
**Comments**: 5

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When enable_thinking=False and stream=True, the API incorrectly returns the response in the reasoning_content field rather than the expected content field.



### Reproduction

- Request

`
curl http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model": "qwen3-32b-fp8",
        "messages": [
  

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Message to guide using <=0.3.2 for data parallel is not shown when --dp is set

**Link**: https://github.com/sgl-project/sglang/issues/1617
**State**: closed
**Created**: 2024-10-09T07:40:22+00:00
**Closed**: 2024-10-11T14:37:50+00:00
**Comments**: 1

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

More details:
https://github.com/sgl-project/sglang/commit/048685430d4c46fd5bc150675b0df49fc6a681d3#r147740163

### Reproduction

setting `--dp 2`

### Environment

sglang 0.3.3

---

## Issue #N/A: no response running python -m sglang.launch_server --model-path NousResearch/Llama-2-7b-chat-hf --port 30000

**Link**: https://github.com/sgl-project/sglang/issues/100
**State**: closed
**Created**: 2024-01-25T15:04:48+00:00
**Closed**: 2024-01-30T14:30:56+00:00
**Comments**: 2

### Description

when I try to use `sglang` locally according to README.md:
``` sh
python -m sglang.launch_server --model-path NousResearch/Llama-2-7b-chat-hf --port 30000
```
(I use NousResearch/Llama-2-7b-chat-hf because my access of meta-llama is pending)
however, I receive no response and no log print. when I run the python script:
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

prin

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Streaming tool use is mixing in regular text with returned tools

**Link**: https://github.com/sgl-project/sglang/issues/3387
**State**: closed
**Created**: 2025-02-08T03:08:51+00:00
**Closed**: 2025-02-13T01:42:32+00:00
**Comments**: 3

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When streaming a request that includes `tools`, outputs look like this:


(sglang `v0.4.2.post3-cu124`)

```
(venv) > $ curl -N http://localhost:3001/v1/chat/completions \
  -H "Authorization: Bearer test-api-key" \
  -d '{"n":1,"model":"meta-llama/llama-3.1-8b-instruct/fp-16","tools":[{"type":"function","function":{"strict": true,"name":"

[... truncated for brevity ...]

---

## Issue #N/A: Cannot Execute Runtime Directly in Docker, with local install

**Link**: https://github.com/sgl-project/sglang/issues/274
**State**: closed
**Created**: 2024-03-11T00:37:46+00:00
**Closed**: 2024-03-13T05:08:32+00:00
**Comments**: 5

### Description

I'm running the runtime directly, like so:

```
SGLANG_PORT, additional_ports = handle_port_init(30000, None, 1)
RUNTIME = sgl.Runtime(
    model_path=model_path,
    port=SGLANG_PORT,
    additional_ports=additional_ports,
    model_mode=[] if os.environ.get("DISABLE_FLASH_INFER") == "yes" else ["flashinfer"],
)
print(f"Initialized SGLang runtime: {RUNTIME.url}")
```

But after upgrading from 0.1.12 to latest commit I get this error:

```
Process Process-1:1:
router init state: Traceback (most recent call last):
  File "/sglang/python/sglang/srt/managers/router/manager.py", line 68, in start_router_process
    model_client = ModelRpcClient(server_args, port_args)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/sglang/python/sglang/srt/managers/router/model_rpc.py", line 612, in __init__
    self.model_server.exposed_init_model(0, server_args, port_args)
  File "/sglang/python/sglang/srt/managers/router/model_rpc.py", line 62, in exposed_init_m

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]  'check_marlin_supported' is not defined when launching GPTQ model in 0.4.5

**Link**: https://github.com/sgl-project/sglang/issues/5366
**State**: closed
**Created**: 2025-04-14T06:56:00+00:00
**Closed**: 2025-04-18T09:46:07+00:00
**Comments**: 4

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When launching a GPTQ model using SGLang version 0.4.5, I encountered a runtime error:

```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/tiger/.pyenv/versions/3.11.2/lib/python3.11/site-packages/sglang/launch_server.py", line 14, in

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] support merge_state in sgl-kernel

**Link**: https://github.com/sgl-project/sglang/issues/5361
**State**: closed
**Created**: 2025-04-14T00:56:57+00:00
**Closed**: 2025-04-15T04:32:18+00:00
**Comments**: 3
**Labels**: high priority, collaboration, speculative-decoding

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

I have talked to @deftruth, and he will support it in the sgl-kernel today

### Related resources

_No response_

---

## Issue #N/A: [Bug] hiradix_cache encountered an exception while executing self.dec_lock_ref

**Link**: https://github.com/sgl-project/sglang/issues/5410
**State**: closed
**Created**: 2025-04-15T07:35:20+00:00
**Closed**: 2025-04-17T07:16:04+00:00
**Comments**: 1

### Description

https://github.com/sgl-project/sglang/blob/8aab7fdb21e86b87a3141951ce498246fd4aae73/python/sglang/srt/mem_cache/hiradix_cache.py#L120

hit an exception in hiradix_cache.py:120:
```
[2025-04-15 07:02:26 TP1] Scheduler hit an exception: Traceback (most recent call last):
  File "/mnt/yscfs/yscode/sglang/python/sglang/srt/managers/scheduler.py", line 1975, in run_scheduler_process
    scheduler.event_loop_overlap()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
  File "/mnt/yscfs/yscode/sglang/python/sglang/srt/managers/scheduler.py", line 561, in event_loop_overlap
    batch = self.get_next_batch_to_run()
  File "/mnt/yscfs/yscode/sglang/python/sglang/srt/managers/scheduler.py", line 1171, in get_next_batch_to_run
    new_batch = self.get_new_batch_prefill()
  File "/mnt/yscfs/yscode/sglang/python/sglang/srt/managers/scheduler.py", line 1207, in get_new_batch_prefill
    self.tree_cache.writing_c

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Use Embedding/Generation Model to get its Generation/Emebedding

**Link**: https://github.com/sgl-project/sglang/issues/1200
**State**: closed
**Created**: 2024-08-25T01:03:40+00:00
**Closed**: 2024-08-27T18:20:30+00:00
**Comments**: 1

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Currently, SGLang supports getting generation content (chat completion) from generative models and embedding from embedding models. But theoretically, we can get embedding/generation from both embedding/generation models.

Something should be stressed that even we can do this, it's not usefully in practice. 

> The key differences between generation and embedding models primarily stem from their post-training specialization, leading to a loss of some capabilities, akin to catastrophic forgetting. Embedding models focus on compressing information into a fixed-dimensional vector space, discouraging long-term predictions, while generation models aim to reduce uncertainty in the probability space, addressing both c

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] DeepSeekR1 under high QPS TBO Error:

**Link**: https://github.com/sgl-project/sglang/issues/7707
**State**: closed
**Created**: 2025-07-02T04:19:06+00:00
**Closed**: 2025-07-03T08:17:30+00:00
**Comments**: 5

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

```
[2025-07-02 11:36:23 DP5 TP5] Decode batch. #running-req: 1, #token: 2765, token usage: 0.01, pre-allocated usage: 0.00, #retracted-req: 0, cuda graph: True, gen throughput (token/s): 15.02, #queue-req: 0
[2025-07-02 11:36:23 DP7 TP7] Decode batch. #running-req: 1, #token: 6135, token usage: 0.02, pre-allocated usage: 0.00, #retracted-

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Fatal Python error: Bus error when loading safetensors check point shards.

**Link**: https://github.com/sgl-project/sglang/issues/6083
**State**: closed
**Created**: 2025-05-07T10:00:19+00:00
**Closed**: 2025-05-11T12:04:52+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I tried to deploy Qwen3-32B and encountered this issue. Full log as below.
```
(sglang) administrator@dayuanaiserver:~$ python -m sglang.launch_server --model-path /mnt/raid0disk0/models/Qwen3-32B/ -
-host 0.0.0.0 --port 6666 --served-model-name Qwen3-32B --mem-fraction-static 0.7 --max-running-requests 20 --tp 2
INFO 05-07 14:42:34 [impor

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] low_latency mode on multi nodes

**Link**: https://github.com/sgl-project/sglang/issues/5376
**State**: closed
**Created**: 2025-04-14T11:51:38+00:00
**Closed**: 2025-04-16T07:55:40+00:00
**Comments**: 10

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

Experiencing crashes when deploying DeepSeek-V3 in low latency mode across multiple nodes.

### Reproduction

## node1
```
python python/sglang/launch_server.py --model-path=./models/DeepSeek-V3/ --trust-remote-code --tp=16 --dp=16 --ep=16 --enable-dp-attention --enable-deepep-moe --deepep-mode=low_latency --cuda-graph-max-bs=128 --random-

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] How to accelerate constrained decoding when regex needs to change with input?

**Link**: https://github.com/sgl-project/sglang/issues/2168
**State**: closed
**Created**: 2024-11-25T03:36:49+00:00
**Closed**: 2024-12-01T11:06:06+00:00
**Comments**: 1

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

In some practical application scenarios, the regex needs to change with the input, and the speed of constrained decoding using Compressed FSM will be significantly slower than that of unconstrained decoding due to each time you need to compile. How do we support the constrained decoding acceleration requirement in unfixed regex scenarios? thanks~

### Related resources

_No response_

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

## Issue #N/A: OpenAI speculative execution

**Link**: https://github.com/sgl-project/sglang/issues/44
**State**: closed
**Created**: 2024-01-18T18:09:31+00:00
**Closed**: 2024-01-25T10:10:02+00:00
**Comments**: 3
**Labels**: enhancement, high priority

### Description

The current frontend using OpenAI will invoke multiple calls for the example below:
```
@sgl.function
def example(s):
  s += "Construct a character."
  s += "Name: " + gen("name") + " Birthday: " + gen("birthday") + " Job: " + gen("job")
```
We can optimize this to send less number of calls to save money:
1. Gen longer in the first gen call, and skip the later if the first gen did the right thing.
2. Allow using OpenAI's n=10 keyword argument to sample multiple completions when forked. We can also provide the interface `example.run(n=10)`.

---

## Issue #N/A: Async support

**Link**: https://github.com/sgl-project/sglang/issues/29
**State**: closed
**Created**: 2024-01-17T23:44:02+00:00
**Closed**: 2024-01-21T23:17:31+00:00
**Comments**: 0
**Labels**: good first issue

### Description

No description provided.

---

## Issue #N/A: [Bug] install from source cannot start

**Link**: https://github.com/sgl-project/sglang/issues/2527
**State**: closed
**Created**: 2024-12-19T18:20:23+00:00
**Closed**: 2024-12-20T18:39:23+00:00
**Comments**: 0

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

1. create a virtual environment, install from pip. start sglang.launch_server and it works fine.
2. create another virtual environment, install from source. start sglang.launch_server get error. Error is attached as below:
 30000 --host 0.0.0.0
[2024-12-20 02:00:36] server_args=ServerArgs(model_path='models/Qwen2.5-0.5B-Instruct', token

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Vision LM accuracy test

**Link**: https://github.com/sgl-project/sglang/issues/3141
**State**: closed
**Created**: 2025-01-26T06:24:30+00:00
**Closed**: 2025-02-01T18:07:14+00:00
**Comments**: 0

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

In sglang, LLMs have accuracy tests with Hugging Face models:

https://github.com/sgl-project/sglang/blob/main/test/srt/models/test_generation_models.py

We need similar one for VLM also.

### Related resources

_No response_

---

## Issue #N/A: [Feature] New models Gemma 3

**Link**: https://github.com/sgl-project/sglang/issues/4332
**State**: closed
**Created**: 2025-03-12T07:38:08+00:00
**Closed**: 2025-03-17T05:28:55+00:00
**Comments**: 3
**Labels**: good first issue, help wanted

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Gemma 3 has a large, 128K context window, multilingual support in over 140 languages, and is available in more sizes than previous versions. Gemma 3 models are well-suited for a variety of text generation and image understanding tasks, including question answering, summarization, and reasoning.

Inputs and outputs
Input:

Text string, such as a question, a prompt, or a document to be summarized
Images, normalized to 896 x 896 resolution and encoded to 256 tokens each
Total input context of 128K tokens for the 4B, 12B, and 27B sizes, and 32K tokens for the 1B size
Output:

Generated text in response to the input, such as an answer to a question, analysis of image content, or a summary of a document
Total output cont

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Inference speed difference between sglang and vllm is smaller than advertised

**Link**: https://github.com/sgl-project/sglang/issues/998
**State**: closed
**Created**: 2024-08-09T08:07:39+00:00
**Closed**: 2024-08-15T16:26:10+00:00
**Comments**: 8
**Labels**: await-response

### Description

### Motivation

I compared the inference speed of two large model inference frameworks. I found that sglang is only about 30% faster than vllm, which is much lower than the claimed 3.8 times speedup.

Below are my environment details, prompt, and inference results.
my environment：
gpu:4090*1
cuda:12.4
Python:3.11.0
vllm:0.5.3
sglang:0.2.7

launch command:
`python -m vllm.entrypoints.openai.api_server --model /home/modeldata/Qwen2-1.5B-Instruct --port 8899`
`python -m sglang.launch_server --model-path /home/modeldata/Qwen2-1.5B-Instruct --host 0.0.0.0 --port 30000`

prompt:
`请根据用户反馈，仔细思考标准答案构成要素，并改写出5句答案\n你是直播真人问答客服，为避免客服回答的答案重复度过高，请你逐句思考并改写问题的答案。\n****************\n#样例\n用户问题：声音好好听\n参考的问答对：["问题: 声音好好听, 答案: 谢谢宝宝的夸奖，喜欢主播的可以点个关注", "问题: 你是真人吗, 答案: 什么，你说我是不是真人", "问题: 没有红包吗, 答案: 红包左上角都会安排的", "问题: 拍啦, 答案: 好的感谢支持咱家玉米"]\n输出格式：["感谢你的夸赞支持", "你的夸赞是我前进的动力", "收到你的夸奖，心情美美哒", "夸奖收到，谢谢宝宝的热情", "你的夸奖我收到了，谢谢"]\n\n****************\n#规则（必须严格遵循）\n1、你的答案必须仿照改写用户觉得满意的答案\n2、你的答案绝对不能按照用户不满意答案的写法。

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] unrecognized arguments: --enable_metrics

**Link**: https://github.com/sgl-project/sglang/issues/5032
**State**: closed
**Created**: 2025-04-03T10:52:49+00:00
**Closed**: 2025-04-06T06:42:01+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

INFO 04-03 17:50:56 __init__.py:190] Automatically detected platform cuda.
usage: launch_server.py [-h] --model-path MODEL_PATH [--tokenizer-path TOKENIZER_PATH] [--host HOST] [--port PORT] [--tokenizer-mode {auto,slow}] [--skip-tokenizer-init]
                        [--load-format {auto,pt,safetensors,npcache,dummy,sharded_state,gguf,bit

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] potential correctness with triton-attention-num-kv-splits > 1

**Link**: https://github.com/sgl-project/sglang/issues/2465
**State**: closed
**Created**: 2024-12-12T09:10:12+00:00
**Closed**: 2024-12-14T08:50:56+00:00
**Comments**: 5

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

On H200, seemingly we hit a bit issue on correctness.
Would you please help to confirm?
cc @ispobock 

### Reproduction

```
python3 -m sglang.bench_one_batch --model meta-llama/Llama-3.1-8B-Instruct --tp 8 --batch-size 1 --input 1024 --output 2048 --correctness-test --attention-backend triton --triton-attention-num-kv-splits 2
```




[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Cannot batch generate or n>1 in sampling_params when input_embeds enabled

**Link**: https://github.com/sgl-project/sglang/issues/7807
**State**: closed
**Created**: 2025-07-06T13:42:01+00:00
**Closed**: 2025-07-08T21:00:43+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I tried to get multi generation outputs via a single post to the server, but I find that if input_embeds is used, the generation will fail and the error message is weired.

In the [document](https://docs.sglang.ai/backend/sampling_params.html), it is stated that `input_embeds` can be a `List[List[List[float]]]` item, but it turns out that 

[... truncated for brevity ...]

---

