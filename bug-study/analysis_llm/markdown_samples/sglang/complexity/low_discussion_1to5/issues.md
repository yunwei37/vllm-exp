# low_discussion_1to5 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 4
- Closed Issues: 26

### Label Distribution

- inactive: 12 issues
- feature: 3 issues
- good first issue: 2 issues
- help wanted: 2 issues
- high priority: 1 issues
- MLLM: 1 issues
- enhancement: 1 issues
- lora: 1 issues

---

## Issue #N/A: QVQ Prefill stage slow

**Link**: https://github.com/sgl-project/sglang/issues/2961
**State**: closed
**Created**: 2025-01-18T08:03:33+00:00
**Closed**: 2025-01-26T09:13:00+00:00
**Comments**: 4

### Description

Using int4 QVQ 72b model. https://huggingface.co/kosbu/QVQ-72B-Preview-AWQ 
basic config: 4 2080ti 22G tp=4

```python3 -m sglang.launch_server --model-path /root/model/QVQ-72B-Preview-AWQ --host 0.0.0.0 --port 30000 --tp 4 --mem-fraction-static 0.7 ```

<img width="909" alt="Image" src="https://github.com/user-attachments/assets/97af8f64-029b-4d4a-8e84-38dd3ede0340" />


As you may see, the prefilling stage take 20s.

What i can do to optimize the speed?
Or do i have option to turn off prefilling, when performing only one request?

---

## Issue #N/A: [Bug] Failed to create router: Timeout 300s waiting for workers to become healthy

**Link**: https://github.com/sgl-project/sglang/issues/2778
**State**: closed
**Created**: 2025-01-07T22:23:43+00:00
**Closed**: 2025-01-20T22:50:41+00:00
**Comments**: 3

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

Any time I try some slower to start options with the new router it times out at 300 sec - e.g. try `--enable-torch-compile`, which can easily take 10-15min to start with all its tune attempts.

How can this timeout be overridden to be made higher by the user when needed?

### Reproduction

python -m sglang_router.launch_server --enable-t

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] flashinfer.jit: Loading JIT ops: cascade

**Link**: https://github.com/sgl-project/sglang/issues/5034
**State**: closed
**Created**: 2025-04-03T12:01:39+00:00
**Closed**: 2025-06-06T00:19:20+00:00
**Comments**: 2
**Labels**: inactive

### Description


### Describe the bug

Discovered an interesting bug: when using the offline engine, if the amount of generated data is small, for example, just a few entries, it seems the loading of this cascade operator is not triggered. However, once the loading of this operator is triggered, it throws an error in my current version.
```bash
2025-04-03 19:59:12,086 - INFO - flashinfer.jit: Loading JIT ops: cascade
2025-04-03 19:59:12,938 - INFO - flashinfer.jit: Loading JIT ops: cascade
[2025-04-03 20:06:34 DP1 TP0] Watchdog timeout (self.watchdog_timeout=300)
[2025-04-03 20:06:34 DP1 TP0] self.cur_batch.batch_size()=1, self.cur_batch.reqs=[Req(rid=71d702268ab8458b98bc40e7667f80ec, input_ids=[3838, 374, 279, 6722, 315, 9625, 30], output_ids=[])], self.token_to_kv_pool_allocator.available_size()=2468326, self.tree_cache.evictable_size()=0, 
[2025-04-03 20:06:34 DP0 TP0] Watchdog timeout (self.watchdog_timeout=300)
[2025-04-03 20:06:34 DP0 TP0] self.cur_batch.batch_size()=362, self.cur_batch.reqs=[Re

[... truncated for brevity ...]

---

## Issue #N/A: how to fix "CUDA_HOME environment variable is not set" in docker

**Link**: https://github.com/sgl-project/sglang/issues/3952
**State**: closed
**Created**: 2025-02-28T07:25:02+00:00
**Closed**: 2025-03-25T03:29:01+00:00
**Comments**: 4

### Description

install sglang with pip in docker, according to the document(https://docs.sglang.ai/start/install.html)，when i run sglang, there is an error "CUDA_HOME environment variable is not set"
i need help
need to install CUDA in Docker？


---

## Issue #N/A: [Bug] OpenAI Endpoint '/v1/batches': `error: Object of type ChoiceLogprobs is not JSON serializable`

**Link**: https://github.com/sgl-project/sglang/issues/3895
**State**: closed
**Created**: 2025-02-26T20:03:36+00:00
**Closed**: 2025-03-13T05:04:30+00:00
**Comments**: 2

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

First of all, thanks for this amazing framework!
When processing a batch via the OpenAI-compatible endpoint 'v1/batches' , and one requests the output of the logprobs, the server outputs the following error:
```shell
[2025-02-26 20:13:21 TP0] Prefill batch. #new-seq: 1, #new-token: 1, #cached-token: 35, cache hit rate: 44.87%, token usage:

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] self.token_to_kv_pool.available_size(): AttributeError: 'Scheduler' object has no attribute 'token_to_kv_pool' with Qwen/QwQ-32B on SGLang v0.4.3.post3

**Link**: https://github.com/sgl-project/sglang/issues/4127
**State**: closed
**Created**: 2025-03-06T06:34:49+00:00
**Closed**: 2025-03-06T16:04:20+00:00
**Comments**: 3

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hi,
try to launch the just released Qwen/QwQ-32B  on very last version of SGLang 0.4.3.post3

See final message below: AttributeError: 'Scheduler' object has no attribute 'token_to_kv_pool'  in check_memory self.token_to_kv_pool.available_size()

```

### starting SGLang ...
sgl start command: python3.12 -m sglang.launch_server   --model Q

[... truncated for brevity ...]

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

## Issue #N/A: no batch run when using openai's format for calling.

**Link**: https://github.com/sgl-project/sglang/issues/404
**State**: closed
**Created**: 2024-04-30T01:45:19+00:00
**Closed**: 2024-07-25T06:33:23+00:00
**Comments**: 4
**Labels**: inactive

### Description

I just use this command to start the server 
`CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path LLMs/Qwen-14B-Chat --port 30000 --trust-remote-code --stream-interval 1 --enable-flashinfer --schedule-conservativeness 50`
and using the following code to test the concurrent capability.

It can only generate code with ~10tokens/s whereas the vllm can be ~30tokens/s. it seems the call method does not support batch inferencing. the logs show as below:
![image](https://github.com/sgl-project/sglang/assets/24971464/f9304ec5-7a3d-4b58-9748-efe08c75fb5f)
there is always 1 `runnning_req`. 

The question is should we do it myself to support the batch inferencing when API calling or is something wrong with my setup? 
BTW, I also tried the `batching` example from the README, and it works fine and running faster then I expected!!!

Thank you so much ahead.

**SCRIPTS**
```python
def run(ds):
    winner = "a" if "_" not in ds["winner"] else ds["winner"].split("_")[1]


[... truncated for brevity ...]

---

## Issue #N/A: Can multiple services be deployed simultaneously?

**Link**: https://github.com/sgl-project/sglang/issues/2916
**State**: closed
**Created**: 2025-01-16T09:57:42+00:00
**Closed**: 2025-01-21T22:12:15+00:00
**Comments**: 1

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Can multiple services be deployed simultaneously, similar to the FastChat project?

### Related resources

_No response_

---

## Issue #N/A: Offline Generation

**Link**: https://github.com/sgl-project/sglang/issues/24
**State**: closed
**Created**: 2024-01-17T21:10:27+00:00
**Closed**: 2024-01-17T21:29:02+00:00
**Comments**: 1

### Description

Hi, is it possible to do offline Generation similar to the vllm batch Inference where the model is not served?

Like
```
Llm = sglang("path/to/llm")
```

---

## Issue #N/A: [Feature] support W8A8(FP8) and KV Cache FP8 for DeepSeek V2

**Link**: https://github.com/sgl-project/sglang/issues/1156
**State**: closed
**Created**: 2024-08-19T17:21:17+00:00
**Closed**: 2024-09-01T09:51:32+00:00
**Comments**: 3
**Labels**: feature

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

As titled. Make DeepSeek V2 MLA Faster!

### Related resources

_No response_

---

## Issue #N/A: [Feature] Is AWQ W4Afp8 supported?

**Link**: https://github.com/sgl-project/sglang/issues/1964
**State**: closed
**Created**: 2024-11-08T21:29:15+00:00
**Closed**: 2025-01-10T00:17:05+00:00
**Comments**: 4
**Labels**: inactive

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

AWQ with INT4 weights and fp8 activations / KV cache works fairly well with Llama-3 models, and is a useful quantization technique for high-throughput regime. Is this quantization format supported by SGLang?


### Related resources

https://github.com/NVIDIA/TensorRT-LLM/blob/b7868dd1bd1186840e3755b97ea3d3a73ddd76c5/examples/falcon/README.md?plain=1#L311

---

## Issue #N/A: [Bug] AWQ scalar type error

**Link**: https://github.com/sgl-project/sglang/issues/3780
**State**: closed
**Created**: 2025-02-22T04:37:08+00:00
**Closed**: 2025-03-03T06:06:20+00:00
**Comments**: 5
**Labels**: help wanted

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When I run the Deepseek-R1-AWQ, I met a scalar type bug same as pr #3450 . @hnyls2002 
```
Loading safetensors checkpoint shards:  97% Completed | 72/74 [00:44<00:01,  1.55it/s]
Loading safetensors checkpoint shards:  99% Completed | 73/74 [00:45<00:00,  1.72it/s]
[2025-02-22 12:19:24 TP3] Scheduler hit an exception: Traceback (most recent

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] support aarch64 natively

**Link**: https://github.com/sgl-project/sglang/issues/5222
**State**: closed
**Created**: 2025-04-10T08:04:32+00:00
**Closed**: 2025-06-10T00:19:36+00:00
**Comments**: 2
**Labels**: high priority, inactive

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

We support sglang on jetson and gh200 but this type of things must be changed:
if os.path.exists("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12"):
    ctypes.CDLL(
        "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12",
        mode=ctypes.RTLD_GLOBAL,
    )

because we have to fix by sed or git diff...

### Related resources

pypi:
https://pypi.jetson-ai-lab.dev/jp6/cu128
https://pypi.jetson-ai-lab.dev/sbsa/cu128

---

## Issue #N/A: [Bug] Server crash when Input length exceeds the maximum allowed length

**Link**: https://github.com/sgl-project/sglang/issues/3910
**State**: closed
**Created**: 2025-02-27T06:03:57+00:00
**Closed**: 2025-06-01T00:24:09+00:00
**Comments**: 3
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

**Description** 
---
Server crashs when receives prompt longer than max.

![Image](https://github.com/user-attachments/assets/85f231aa-0571-4ac6-b728-a436c2e410a5)


### Reproduction

python3 -m sglang.launch_server --model-path /root/.cache/huggingface/models/DeepSeek-R1 --tp 16 --dist-init-addr  10.239.14.57:20000 --nnodes 2 --node-rank 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Stuck at NCCL initialization when TP>1

**Link**: https://github.com/sgl-project/sglang/issues/3666
**State**: closed
**Created**: 2025-02-18T09:44:24+00:00
**Closed**: 2025-04-21T00:19:33+00:00
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

Many thanks for this great work! When using TP>1, it will stuck at NCCL initialization:


>  INFO 02-18 09:33:49 __init__.py:190] Automatically detected platform cuda.                                                                   
[2025-02-18 09:33:55] server_args=ServerArgs(model_path='meta-llama/Llama-3.1-8B-Instruct', tokenizer_path

[... truncated for brevity ...]

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

## Issue #N/A: [Feature] Due to GIL issues, the overlap mode doesn't actually always bring benefits?

**Link**: https://github.com/sgl-project/sglang/issues/2573
**State**: closed
**Created**: 2024-12-25T13:31:50+00:00
**Closed**: 2025-02-24T00:17:23+00:00
**Comments**: 1
**Labels**: inactive

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Workaround python GIL (Work in progress)：
Idea 1: Try python 3.13 which can remove GIL
Idea 2: Use multiple processes, but need make the serialization very fast
In actual testing, I found that overlap does not necessarily bring benefits. I think this may be related to the GIL, since the current version is implemented with multi-threading. I'm wondering if this is expected? And under what circumstances would the GIL issue become more severe? @merrymercy 

### Related resources

_No response_

---

## Issue #N/A: OOM CUDA error on 8 * L4 machine when launching sglang server

**Link**: https://github.com/sgl-project/sglang/issues/445
**State**: closed
**Created**: 2024-05-15T21:02:07+00:00
**Closed**: 2024-07-25T06:33:39+00:00
**Comments**: 5
**Labels**: inactive

### Description

Hey!

I m trying launching a sglang server with [OpenBioLLM 70b](https://huggingface.co/aaditya/Llama3-OpenBioLLM-70B) with the command `python -m sglang.launch_server --model-path ~/Llama3-OpenBioLLM-70B-Instruct --port 30000` but I got on the 2 issues:

1. It errors out with OOM CUDA, I tried playing around with all possible memory arguments but still have the issue, for e.g running `python -m sglang.launch_server --model-path ~/Llama3-OpenBioLLM-70B-Instruct --port 30000 --mem-fraction-static 0.9 --tp 8 --disable-disk-cache` errors out, I tried decreasing the mem-fraction-static or try different values with tp but still fails, here is the error 
```
`Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
server started on [0.0.0.0]:10014
server started on [0.0.0.0]:10017
server started on [0.0.0.0]:10018
server started on [0.0.0.0]:10016
server started on [0.0.0.0]:10019
server started on [0.0.0.0]:10015
ser

[... truncated for brevity ...]

---

## Issue #N/A: setting mem-fraction-static to a lower value causes error

**Link**: https://github.com/sgl-project/sglang/issues/165
**State**: closed
**Created**: 2024-02-07T22:25:25+00:00
**Closed**: 2024-07-25T06:33:36+00:00
**Comments**: 4
**Labels**: inactive

### Description

With no change, I run out of memory (A100 w/ 24GB). Setting it to anything other than the default causes the following error:

```
Exception in ModelRpcClient:
Traceback (most recent call last):
  File "/workspace/sglang/python/sglang/srt/managers/router/model_rpc.py", line 170, in exposed_step
    self.forward_step()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/workspace/sglang/python/sglang/srt/managers/router/model_rpc.py", line 185, in forward_step
    self.forward_fill_batch(new_batch)
  File "/workspace/sglang/python/sglang/srt/managers/router/model_rpc.py", line 387, in forward_fill_batch
    batch.prepare_for_extend(
  File "/workspace/sglang/python/sglang/srt/managers/router/infer_batch.py", line 203, in prepare_for_extend
    req_pool_indices_cpu = req_pool_indices.cpu().numpy()
AttributeError: 'NoneType' object has no attribute 'cpu'
```

For reference I 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]  shared memory (/dev/shm) error when using FA3 with meta-llama/Llama-3.1-70B-Instruct on a multi-GPU host

**Link**: https://github.com/sgl-project/sglang/issues/5096
**State**: closed
**Created**: 2025-04-06T03:23:48+00:00
**Closed**: 2025-04-16T19:32:31+00:00
**Comments**: 2

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hello, 

I am trying to run `meta-llama/Llama-3.1-70B-Instruct` with SGLang's FA3  setting and running into /dev/shm error, when running on a multi-gpu host (8xH100).

Would anyone have seen the issue? 

Error log

```
ERROR 2025-04-05T00:01:19.509514570Z ./entrypoint.sh: line 48: 28 Killed python3 -m sglang.launch_server --host 0.0.0.0 --

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] embedding model failed with `--enable-metrics`

**Link**: https://github.com/sgl-project/sglang/issues/2800
**State**: closed
**Created**: 2025-01-08T20:27:18+00:00
**Closed**: 2025-01-22T01:06:46+00:00
**Comments**: 5

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

When I launch the latest SGLang server with e5-mistral-7b-instruct with `--enable-metrics`, the server crashes.

Docker command:
```
docker run -itd   --gpus \"device=0\"  \
 --shm-size 10g   \
 -v /raid/models:/models   \
 --ulimit nofile=65535:65535   \
 --network host   \
 --name sglang-latest-e5-metrics-7   \
 lmsys

[... truncated for brevity ...]

---

## Issue #N/A: Large Discrepancy in Speedup Between SGLang + Eagle and Eagle Repo Code

**Link**: https://github.com/sgl-project/sglang/issues/5502
**State**: open
**Created**: 2025-04-17T14:37:54+00:00
**Comments**: 4

### Description

### 📌 Background

According to the [SGLang speculative decoding documentation](https://docs.sglang.ai/backend/speculative_decoding.html), the speedup of `SGLang + Eagle3` compared to vanilla `SGLang` is reported to be approximately **2.5×** on `LLaMA3.1-Instruct-8B` using the `MT-Bench` evaluation.

![Image](https://github.com/user-attachments/assets/61bebbde-c7fe-4786-ab6e-2e721ab8edd7)

However, the **Eagle3 paper** reports a significantly higher speedup — around **4.4×** — on the same setup (`LLaMA3.1-Instruct-8B`, `MT-Bench`).

![Image](https://github.com/user-attachments/assets/0a12dbbb-d907-4ebb-8f7c-60fd06079487)

---

### 🧪 My Reproduction Results

To further verify, I ran tests using `DeepSeek R1 Distilled LLaMA 8B` on the `AIME` dataset. I observed the following:

| Model & Task | Framework Used | Measured Speedup |
|--------------|----------------|------------------|
| AIME + DeepSeek R1 LLaMA 8B | SGLang + Eagle3 | ~1.5× |
| AIME + DeepSeek R1 LLaMA 8B | Eagle (official rep

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] deepseek-r1 with 4*A100 got error

**Link**: https://github.com/sgl-project/sglang/issues/3491
**State**: closed
**Created**: 2025-02-11T12:50:36+00:00
**Closed**: 2025-02-11T13:08:38+00:00
**Comments**: 5

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I got an error like this. How can I fix it ?
I convert Deepseek-R1 to DeepSeek-R1-BF16 with this [script](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/fp8_cast_bf16.py)
It seems can not deploy with tp=32.

ValueError: Weight output_partition_size = 576 is not divisible by weight quantization block_n = 128.

### Reproducti

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support unified paging in multi-lora serving

**Link**: https://github.com/sgl-project/sglang/issues/3647
**State**: open
**Created**: 2025-02-17T19:14:47+00:00
**Comments**: 5
**Labels**: enhancement, inactive, feature, lora

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Currently, SGL doesn't support the unified paging feature proposed by S-LoRA. However, this feature is important for memory management in multi-LoRA serving.

### Related resources

_No response_

---

## Issue #N/A: [Misc] Use monotonic time for interval measurement

**Link**: https://github.com/sgl-project/sglang/issues/6177
**State**: closed
**Created**: 2025-05-10T19:40:56+00:00
**Closed**: 2025-05-26T21:02:11+00:00
**Comments**: 2

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Currently, in most places, SGL measures time interval using wall clock (`time.time()`), which is not recommended as it does not guarantee monotonicity (e.g., due to NTP sync). More details can be read here: [PEP 418](https://peps.python.org/pep-0418/#rationale).

**Examples in SGLang**

1. In benchmark code such as [bench_one_batch.py#L378](https://github.com/sgl-project/sglang/blob/9d8ec2e67e36117ac6da0c82e597d6dbf587d578/python/sglang/bench_one_batch.py#L378),
`time.perf_counter` should be used instead for its monotonicity guarantee and higher resolution. 

2. In inferencing code such as [RadixCache](https://github.com/sgl-project/sglang/blob/9d8ec2e67e36117ac6da0c82e597d6dbf587d578/python/sglang/srt/mem_cache/ra

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] pydantic_core._pydantic_core.ValidationError: 1 validation error for ChatCompletionResponseChoice

**Link**: https://github.com/sgl-project/sglang/issues/6674
**State**: closed
**Created**: 2025-05-27T16:29:04+00:00
**Closed**: 2025-05-28T07:39:48+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I'm trying to P/D disaggregating using SGLang and meet the error below:
```
[2025-05-27 21:19:46] Prefill batch. #new-seq: 1, #new-token: 1, #cached-token: 25, token usage: 0.00, #running-req: 0, #unbootstrapped-req: 0, #queue-req: 0, #transferring-req: 0 
2025-05-27 21:19:46,103 - INFO - flashinfer.jit: Loading JIT ops: cascade
2025-05-27

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Requests with logprobs throws Internal server 500 error

**Link**: https://github.com/sgl-project/sglang/issues/5984
**State**: closed
**Created**: 2025-05-02T17:47:14+00:00
**Closed**: 2025-07-05T00:18:56+00:00
**Comments**: 2
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

When I am trying to dump logprobs, I am getting `ValueError: Out of range float values are not JSON compliant`

Kindly help / Suggest a work around.





### Reproduction

**Request** 

POST /v1/completions

{'model': 'meta.llama3-70b-instruct-v1:0', 'prompt': 'prompt_string', 'stream': False, 'temperature': 0.01, 'top_p': 0.3, 'max_tokens

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Docker image v0.4.4.post3-cu125 is labeled as CUDA 12.5 (cu125), but it actually contains CUDA 12.4.

**Link**: https://github.com/sgl-project/sglang/issues/4952
**State**: closed
**Created**: 2025-03-31T17:45:08+00:00
**Closed**: 2025-06-01T00:24:08+00:00
**Comments**: 2
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

docker run -it --entrypoint bash lmsysorg/sglang:v0.4.4.post3-cu125                                                                                       

root@0928983058f3:/sgl-workspace# apt list --installed | grep nccl

WARNING: apt does not have a stable CLI interface. Use with caution in scripts.

libnccl-dev/now 2.21.5-1+cuda12.4 am

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

