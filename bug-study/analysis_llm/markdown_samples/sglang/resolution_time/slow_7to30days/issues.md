# slow_7to30days - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- bug: 2 issues
- high priority: 2 issues
- lora: 1 issues
- deepseek: 1 issues
- flashinfer: 1 issues
- good first issue: 1 issues

---

## Issue #N/A: [Feature] do sample = False?

**Link**: https://github.com/sgl-project/sglang/issues/2508
**State**: closed
**Created**: 2024-12-18T08:53:46+00:00
**Closed**: 2024-12-26T18:53:11+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

I am testing the sglang with qwen2, but the output seems to be unstable even if I have set sampling params;

```
temp = 0,
top_k = 1
```

But I got inconsistent outputs, while naive HF implementation always gives me same output when set do_sample = False,
is there such feature in SGL?  Is this due to some non-deterministic Cuda operations? Thanks

### Related resources

_No response_

---

## Issue #N/A: Question: Does sglang support prefix cache for multimodal models?

**Link**: https://github.com/sgl-project/sglang/issues/1870
**State**: closed
**Created**: 2024-11-01T11:03:05+00:00
**Closed**: 2024-11-14T19:08:19+00:00
**Comments**: 4

### Description

I noticed that the cache hit rate remains 0.0% no matter how much turns the conversation has. Does sglang support prefix cache for multimodal models?

---

## Issue #N/A: [Feature] Mooncake CPP (Chunked Pipeline Parallelism)

**Link**: https://github.com/sgl-project/sglang/issues/4842
**State**: closed
**Created**: 2025-03-28T03:52:49+00:00
**Closed**: 2025-04-26T08:33:44+00:00
**Comments**: 4

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Hi team, 

I found that the sequence parallelism (Meta CP based on https://www.arxiv.org/pdf/2411.01783) is in the roadmap (https://github.com/sgl-project/sglang/issues/4042). For long-context inference, I'm wondering do you have plans to support **chunked pipeline parallelism** from Mooncake or **sequence pipeline parallelism** from Mnemosyne? 
If there are no plans, could you please leave some comments about your thoughts on the technology to combine chunked prefill + pp for long-context prefilling, compared with Context Parallelism? Really appreciate your attention~

### Related resources

_No response_

---

## Issue #N/A: Why DP EP Use Two Different MLA kernel in Prefill phase

**Link**: https://github.com/sgl-project/sglang/issues/6301
**State**: closed
**Created**: 2025-05-14T16:22:46+00:00
**Closed**: 2025-05-28T08:12:25+00:00
**Comments**: 0

### Description

<img width="1275" alt="Image" src="https://github.com/user-attachments/assets/f8eb14a5-c42c-4edd-83fe-efbf55be3bdc" />

<img width="1275" alt="Image" src="https://github.com/user-attachments/assets/f2453597-9c39-489d-92d4-1a3b791d517d" />

I modified the DeepSeek model to have 2 layers for single - machine testing. When running 8DP and 8EP, I noticed that the MLA part of two Prefills used two different Kernels. I don't understand why two different Kernels are used here. As shown in the figure, the Kernel called by MLA in the first Prefill is "void flashinfer::PrefillWithKVCacheKernel", and the Kernel called by MLA in the second Prefill is "void flashinfer::mla::hopper::BatchMLAPageAttentionHopperKernel".


---

## Issue #N/A: [OAI Server Refactor] [ChatCompletions & Completions] Remove batch requests, refine validation and streaming

**Link**: https://github.com/sgl-project/sglang/issues/7108
**State**: closed
**Created**: 2025-06-12T00:31:22+00:00
**Closed**: 2025-06-21T17:45:36+00:00
**Comments**: 4

### Description

**Points:** 1-2 days

**Description:** Remove the batch requests process, refine request validation logic and streaming generator structure.

**Deliverables:**

- [x] Finish tasks listed below
- [x] Add UTs if available

---

## Issue #N/A: [Feature] Graceful handling of non-existing lora_path in inference request

**Link**: https://github.com/sgl-project/sglang/issues/7447
**State**: closed
**Created**: 2025-06-22T21:36:01+00:00
**Closed**: 2025-07-03T03:59:17+00:00
**Comments**: 1
**Labels**: lora

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Creating an issue to track this TODO for myself (or anyone else who wants to help):

Currently when users call SGLang with a non-existing lora_path, SGLang server/engine would crash due to failed assertions in `prepare_lora_batch`. This is unideal as it imposes unnecessary burden for server owner to validate request params before they are passed to the SGLang backend.

Ideally, SGLang should have gracefully handled the exception and respond 4xx errors without crashing the server.

### Related resources

_No response_

---

## Issue #N/A: [Bug] LLaVa-next does not work for single image processing

**Link**: https://github.com/sgl-project/sglang/issues/1506
**State**: closed
**Created**: 2024-09-24T21:52:55+00:00
**Closed**: 2024-10-06T22:42:54+00:00
**Comments**: 1

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

INFO 09-24 16:44:39 weight_utils.py:236] Using model weights format ['*.safetensors']
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:00<00:01,  2.63it/s]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:01<00:01,  1.34it/s]
Load

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Multimodal models with prefix caching

**Link**: https://github.com/sgl-project/sglang/issues/6552
**State**: closed
**Created**: 2025-05-23T08:41:11+00:00
**Closed**: 2025-06-16T08:50:54+00:00
**Comments**: 0

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

I know currently SGLang Multimodal models are not to work with prefix caching （Qwen2.5 VL）。

So:
1.What's the main reason for this ?
2.What should I do if I want to develop with the source code to support this?
I have gone though the tokenize manager and radix cache match func source code but I can not find the problem.Can someone help me and point the right direction?(which module/folder/python file or any toher directions)

### Related resources

_No response_

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

## Issue #N/A: [Bug] Use torch.inference_mode instead of torch.no_grad

**Link**: https://github.com/sgl-project/sglang/issues/4366
**State**: closed
**Created**: 2025-03-13T06:32:51+00:00
**Closed**: 2025-04-04T05:18:52+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

We found that `torch.no_grad` triggers the `AutogradXXX` backend for certain operators. Should we replace it with `inference_mode` instead, or keep supporting with `torch<1.9`?

### Reproduction

Example: `python/sglang/srt/mem_cache/memory_pool.py:144(def free_group_end(self):)`

- Result with torch.no_grad(): `NotImplementedError: Could 

[... truncated for brevity ...]

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

## Issue #N/A: [Bug] Incorrect Memory Allocation on CUDA:0 by Non-Zero CUDA Processes in TP/DP

**Link**: https://github.com/sgl-project/sglang/issues/5732
**State**: closed
**Created**: 2025-04-25T04:42:53+00:00
**Closed**: 2025-05-09T00:52:27+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

A potential memory leak has been identified related to the handling and broadcasting of multimodal data in distributed setups. The issue seems to originate from the interaction between the `BaseMultiModalProcessor` logic and the `broadcast_pyobj` utility function used by the `Scheduler`.

![Image](https://github.com/user-attachments/assets

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] [PD] TransferEngine fault auto-recovery: allows Prefill node failures without requiring a restart of Decode.

**Link**: https://github.com/sgl-project/sglang/issues/6215
**State**: closed
**Created**: 2025-05-12T07:23:10+00:00
**Closed**: 2025-05-31T14:07:13+00:00
**Comments**: 2

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Currently, if a Prefill node fails, the Decode side may hang during data transfer if cached information about the failed node is still in use. A mechanism is needed to notify Decode to re-establish the connection with a healthy Prefill node.

CC @ByronHsu @ShangmingCai 
### Related resources

_No response_

---

## Issue #N/A: [Bug] libcudart.so.12: cannot open shared object file: No such file or directory

**Link**: https://github.com/sgl-project/sglang/issues/2584
**State**: closed
**Created**: 2024-12-26T02:31:18+00:00
**Closed**: 2025-01-24T04:35:16+00:00
**Comments**: 9

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

There are no CUDA-related libraries in the rocm environment, but the SGLANG 0.4.1 version will report an error, while the 0.4.0 and earlier versions will not

**error info:** 
ImportError: [address=0.0.0.0:39501, pid=13418] libcudart.so.12: cannot open shared object file: No such file or directory
2024-12-26 10:14:09,664 xinference.api

[... truncated for brevity ...]

---

## Issue #N/A: [Usage] What's the best practice of deploying DeepSeekV3 using sglang?

**Link**: https://github.com/sgl-project/sglang/issues/4409
**State**: closed
**Created**: 2025-03-14T03:02:37+00:00
**Closed**: 2025-03-26T08:09:34+00:00
**Comments**: 1
**Labels**: deepseek

### Description

I want to run inference of [DeepSeekV3](https://huggingface.co/deepseek-ai/DeepSeek-V3) on multi-node GPU clusters. I have followed the [sglang deepseek guide](https://docs.sglang.ai/references/deepseek.html) to setup the distributed serving environment with the official sglang-v0.4.3.post4-cu124 docker image.

More specifically:
```
# node 1
docker run --gpus all \
    --shm-size 32g \
    --network=host \
    --privileged \
    -v xxx:xxx \
    --name sglang_multinode1 \
    -it \
    --ipc=host \
    xxx

# node 2
docker run --gpus all \
    --shm-size 32g \
    --network=host \
    --privileged \
    -v xxx:xxx \
    --name sglang_multinode2 \
    -it \
    --ipc=host \
    xxx
```
to create docker containers

```
# node 1
NCCL_IB_GID_INDEX=3 \
NCCL_DEBUG=TRACE \
NCCL_DEBUG=INFO \
python3 -m sglang.launch_server \
    --model-path xxx \
    --tp 16 --dist-init-addr xxx:20000 \
    --nnodes 2 \
    --node-rank 0 \
    --trust-remote-code \
    --host 0.0.0.0 --port 40000 2>&1 | tee 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] TypeError: argument 'ids': 'NoneType' object cannot be interpreted as an integer

**Link**: https://github.com/sgl-project/sglang/issues/737
**State**: closed
**Created**: 2024-07-26T07:26:44+00:00
**Closed**: 2024-08-16T07:02:10+00:00
**Comments**: 3
**Labels**: bug

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

```
Exception in ModelTpServer:
Traceback (most recent call last):
  File ".../sglang-0.2.0/lib/python3.11/site-packages/sglang/srt/managers/controller/tp_worker.py", line 186, in exposed_step
    self.forward_step()
  File ".../sglang-0.2.0/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File ".../sglang-0.2.0/lib/python3.11/site-packages/sglang/srt/managers/controller/tp_worker.py", line 216, in forward_step
    self.forward_decode_batch(self.running

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] microsoft/Phi-4-multimodal-instruct

**Link**: https://github.com/sgl-project/sglang/issues/5972
**State**: closed
**Created**: 2025-05-02T10:25:30+00:00
**Closed**: 2025-05-25T18:47:51+00:00
**Comments**: 6

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

 Model architectures ['Phi4MMForCausalLM'] are not supported for now. When can we expect support for this model. As Phi4-mini-instruct is already supported, this can be en extension to it.

### Related resources

_No response_

---

## Issue #N/A: [Bug] KeyError when running nvidia/Llama-3.1-70B-Instruct-FP8

**Link**: https://github.com/sgl-project/sglang/issues/5095
**State**: closed
**Created**: 2025-04-06T03:13:15+00:00
**Closed**: 2025-04-16T19:34:46+00:00
**Comments**: 7
**Labels**: high priority

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hello, 

I am currently trying to run `nvidia/Llama-3.1-70B-Instruct-FP8` with SGLang and running into Key Error: `KeyError: 'model.layers.78.mlp.gate_up_proj.input_scale'
ERROR 2025-04-04T06:24:51.493646621Z param = params_dict[name]`. Would anyone have insights on resolving the error, or recommend a variant of Llama-3.1-70B FP8 quantized

[... truncated for brevity ...]

---

## Issue #N/A: Customizing Sampling Behavior

**Link**: https://github.com/sgl-project/sglang/issues/89
**State**: closed
**Created**: 2024-01-24T00:06:48+00:00
**Closed**: 2024-02-06T18:25:19+00:00
**Comments**: 1

### Description

Hi, is there an interface to specify logits processors as in vLLM? 

If possible, could you specify how we can customize the sampling behavior during generation?

---

## Issue #N/A: [Bug] T4 not work

**Link**: https://github.com/sgl-project/sglang/issues/1058
**State**: closed
**Created**: 2024-08-12T14:46:18+00:00
**Closed**: 2024-08-28T13:05:35+00:00
**Comments**: 8
**Labels**: bug, flashinfer

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.

### Describe the bug

T4 not work w/o FlashInfer ref https://github.com/flashinfer-ai/flashinfer/issues/421

```
CUDA Error: no kernel image is available for execution on the device (209) /tmp/build-via-sdist-iemil769/flashinfer-0.1.4+cu121torch2.4/include/flashinfer/attention/handler.cuh: line 169 at function cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, num_threads, smem_size)
CUDA Er

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] sgl-router enters infinite panic loop when all workers die under active load

**Link**: https://github.com/sgl-project/sglang/issues/7028
**State**: closed
**Created**: 2025-06-10T05:34:36+00:00
**Closed**: 2025-06-26T05:58:47+00:00
**Comments**: 2

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Background: DP Attention is not stable yet. During high concurrency, illegal memory access is more likely to occur, leading to crashes. Therefore, when one worker fails, there is a certain probability it will trigger a cascading failure that brings down all workers.

To deal with worker failure, I have a sidecar container infinite looping 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] make page size greater than one compatible with EAGLE

**Link**: https://github.com/sgl-project/sglang/issues/4652
**State**: closed
**Created**: 2025-03-21T10:06:33+00:00
**Closed**: 2025-04-16T06:43:14+00:00
**Comments**: 1
**Labels**: high priority

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

### Related resources

_No response_

---

## Issue #N/A: [Bug] Failure to Dispatch Head Dimension 80 in sglang with Specific Configurations

**Link**: https://github.com/sgl-project/sglang/issues/1109
**State**: closed
**Created**: 2024-08-15T08:05:19+00:00
**Closed**: 2024-09-11T17:35:32+00:00
**Comments**: 6

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

## Issue Description:
When running sglang with hidden_dim set to 80, the following exceptions are encountered under different configurations:

### With enable_cuda_graph set to True:
```bash
Exception: Capture cuda graph failed: BatchDecodeWithPagedKVCachePyTorchWrapper::BeginForward(at::Tensor, at::Tensor, at::Tensor, at::Tensor, uns

[... truncated for brevity ...]

---

## Issue #N/A: ImportError: cannot import name 'pin_program'

**Link**: https://github.com/sgl-project/sglang/issues/532
**State**: closed
**Created**: 2024-06-11T22:07:22+00:00
**Closed**: 2024-06-30T06:43:18+00:00
**Comments**: 0

### Description

**Description**:
When running the 'test_multi_function' test in test_tracing.py, an ImportError occurs, indicating that the name 'pin_program' cannot be imported from the sglang.lang.interpreter module. It appears that 'pin_program' should be changed to 'cache_program'.

**Suggested Fix**:
Change 'pin_program' to 'cache_program' in the sglang.lang.interpreter module.
<img width="572" alt="8adb262c7acec342962c96fd9bd1d91" src="https://github.com/sgl-project/sglang/assets/157339885/f3874cc1-6670-4db9-8341-3d2b4f051090">




---

## Issue #N/A: [Bug] it didn't work when using tp on RTX 3090

**Link**: https://github.com/sgl-project/sglang/issues/1343
**State**: closed
**Created**: 2024-09-06T09:47:12+00:00
**Closed**: 2024-09-22T12:05:49+00:00
**Comments**: 4

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/dfs/data/github/sglang/python/sglang/launch_server.py", line 19, in <modul

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Logit Bias Not Working

**Link**: https://github.com/sgl-project/sglang/issues/6749
**State**: closed
**Created**: 2025-05-29T20:18:34+00:00
**Closed**: 2025-06-10T22:39:27+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

We're finding that the logit bias isn't working as expected, which we are using for our use cases.

Does SGL support this?

### Related resources

https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api

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

## Issue #N/A: is it time to rerun the benchmarks?

**Link**: https://github.com/sgl-project/sglang/issues/1639
**State**: closed
**Created**: 2024-10-12T00:55:49+00:00
**Closed**: 2024-11-01T04:10:14+00:00
**Comments**: 19

### Description

Hi SGLang team,

I have just tried SGLang for the first time - and it was probably one of the easiest projects to setup and launch - it literally took me a few minutes to go from 0 to serving - awesome!!! and thank you for making it so easy on the user.

I have just benchmarked vllm=0.6.2 vs sglang=0.3.2 on 2 H100s w/ 8b llama3 and tp=2 and I get vllm slightly faster than sglang performance, yet [the benchmark section](https://github.com/sgl-project/sglang?tab=readme-ov-file#benchmark-and-performance) shows a very different picture. Would it be possible to re-benchmark and tell me if I am missing on some optimization flags to see the results you get - I'm just checking the baseline at the moment - so no quantization and such. Will get there a bit later. FWIW, I have just benchmarked and vllm had a massive throughput speed up made in v0.6.2 over its v0.5 https://x.com/StasBekman/status/1844886291378470966 - which is probably why the benchmark on your site needs a refresher.

Thank

[... truncated for brevity ...]

---

## Issue #N/A: [Feature Request] Enable working with Azure-OpenAI API (openai.AzureOpenAI())

**Link**: https://github.com/sgl-project/sglang/issues/56
**State**: closed
**Created**: 2024-01-19T14:44:21+00:00
**Closed**: 2024-02-12T09:07:47+00:00
**Comments**: 6
**Labels**: good first issue

### Description

Sglang looks great to me, but at my work, we use the Azure-OpenAI API. I don't see how to access this with sglang.

It would need two inputs in addition to the API-key, because at minimum I need to create the client like this:

```python
client = openai.AzureOpenAI(
    api_key="<your-api-key>",
    base_url="https://<your-project-name>.openai.azure.com/openai",
    api_version="<your-api-version>",  # for example "2023-05-15"
)
```

Also, for some reason the models are called "gpt-35-turbo" instead of "gpt-3.5-turbo" (missing dot); and I believe that you can call your models whatever you want. This should be supported, too.

If this already works somehow, I would appreciate an explicit mention in the `README.md`. 

---

## Issue #N/A: [Bug] Incompatible with outlines>=0.1.0

**Link**: https://github.com/sgl-project/sglang/issues/1930
**State**: closed
**Created**: 2024-11-05T22:17:05+00:00
**Closed**: 2024-11-30T07:20:51+00:00
**Comments**: 5

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

When launching a server, I get a no module found error for outlines.fsm.regex. After checking the releases of the outlines library, I found that this module was removed in the 0.1.0 release. 

### Reproduction

python -m sglang.launch_server --model-path Meta-Llama-3-8B-Instruct \
--port 30000 --host 0.0.0.0

Error: No module named 'out

[... truncated for brevity ...]

---

