# steady_state_6months - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- inactive: 12 issues
- help wanted: 8 issues
- documentation: 4 issues
- high priority: 3 issues
- good first issue: 3 issues
- router: 2 issues
- feature: 1 issues
- RLHF: 1 issues
- amd: 1 issues
- enhancement: 1 issues

---

## Issue #N/A: [Feature] Clear PAT_TOKEN in CI

**Link**: https://github.com/sgl-project/sglang/issues/2659
**State**: closed
**Created**: 2024-12-30T07:44:56+00:00
**Closed**: 2025-03-01T00:18:50+00:00
**Comments**: 1
**Labels**: documentation, inactive

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

![image](https://github.com/user-attachments/assets/d62f4957-2802-4068-9c16-fbcaee2584f4)

@shuaills Would you like to take this? Pretty easy.

### Related resources

_No response_

---

## Issue #N/A: [Bug] https://docs.sglang.ai/references/benchmark_and_profiling.html  The --model-path parameter is incorrect; it should be --model

**Link**: https://github.com/sgl-project/sglang/issues/2884
**State**: closed
**Created**: 2025-01-14T08:02:51+00:00
**Closed**: 2025-01-14T09:14:21+00:00
**Comments**: 1

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

The --model-path parameter is incorrect; it should be --model
<img width="606" alt="屏幕截图 2025-01-14 160032" src="https://github.com/user-attachments/assets/ebd67b28-047d-4b4a-9309-8f47370ca126" />


### Reproduction

python -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct


### Environment

Use a normal environment.

---

## Issue #N/A: [Bug] Deepseek v3 doesn't work on mi300x 

**Link**: https://github.com/sgl-project/sglang/issues/2595
**State**: closed
**Created**: 2024-12-26T13:03:50+00:00
**Closed**: 2025-01-09T04:09:06+00:00
**Comments**: 5

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

After getting last source code of sglang I'm not able to run it.

### Reproduction

python3 -m sglang.launch_server --model DeepSeek-V3 --tp 8 --trust-remote-code

WARNING 12-26 13:00:41 rocm.py:17] `fork` method is not supported by ROCm. VLLM_WORKER_MULTIPROC_METHOD is overridden to `spawn` instead.
Traceback (most recent call last):


[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Qwen2-VL-7B with sglang Performance Degradation

**Link**: https://github.com/sgl-project/sglang/issues/3041
**State**: closed
**Created**: 2025-01-22T05:39:22+00:00
**Closed**: 2025-01-26T02:38:07+00:00
**Comments**: 14
**Labels**: high priority

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

As #2112 mentioned, Qwen2-VL with sglang Performance is bad. 
So I tested in ChartQA_TEST dataset with sglang and vllm, and the score is really different.
**(I also test mme bench and MMMU dataset, in the reply below.)**

This is sglang.

![Image](https://github.com/user-attachments/assets/afeb9586-d74d-4b97-8d9c-cc2bd93f37c5)

and this is

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]  add_worker API no response

**Link**: https://github.com/sgl-project/sglang/issues/2728
**State**: closed
**Created**: 2025-01-04T07:56:26+00:00
**Closed**: 2025-03-22T00:17:07+00:00
**Comments**: 2
**Labels**: inactive

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

'test_2_add_and_remove_worker' failed , can not use add_worker and remove_worker, such as ' curl -X POST "127.0.0.1:30000/remove_worker?url=http://127.0.0.1:31000" ', there is no any response or logging even in debug mode, 

### Reproduction

 python -m sglang_router.launch_server --model-path /mnt/140/llama3/Meta-Llama-3-8B-Instruct --dp-

[... truncated for brevity ...]

---

## Issue #N/A: Some question about layernom in MLA code

**Link**: https://github.com/sgl-project/sglang/issues/3072
**State**: closed
**Created**: 2025-01-23T07:03:32+00:00
**Closed**: 2025-01-23T13:28:26+00:00
**Comments**: 2
**Labels**: help wanted

### Description

Hi，I am confused that there is a layer normalization between the down-sample and up-sample of Q. However, this layer normalization is not shown in the DeepSeek v2 paper.

Here is the code of sglang

![Image](https://github.com/user-attachments/assets/6ab58ca0-f722-4447-9041-e54fd6a86b37)

Here is the formulate in paper

![Image](https://github.com/user-attachments/assets/78722b31-4015-4fbd-9064-fd8e66dc1caa)

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

## Issue #N/A: [Bug] JSONResponse fails if the probability distribution is very spiky.

**Link**: https://github.com/sgl-project/sglang/issues/2955
**State**: closed
**Created**: 2025-01-17T19:09:47+00:00
**Closed**: 2025-01-31T09:04:05+00:00
**Comments**: 3
**Labels**: help wanted

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

The JSONResponse in SGLang will fail if reported logprobs are -inf. This happens for example if I ask for logprobs > 1 and the probaiblity distribution is very spiky at a single value. 

### Reproduction

It fails for 

```
{'id': 'bf8e6d63938f470cac2a3d770c77f9aa',
 'object': 'chat.completion',
 'created': 1737140530,
 'model': 'meta-llam

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Benchmark results

**Link**: https://github.com/sgl-project/sglang/issues/2782
**State**: closed
**Created**: 2025-01-08T05:42:44+00:00
**Closed**: 2025-01-08T05:43:40+00:00
**Comments**: 0

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

I see many benchmark scripts, and I was wondering if there are aggregated results vs VLLM for different models/input lengths/output lengths so that I dont have to rerun them all.

### Related resources

_No response_

---

## Issue #N/A: [Feature] docs: Improve documentation on how to use EAGLE speculative docoding

**Link**: https://github.com/sgl-project/sglang/issues/3077
**State**: closed
**Created**: 2025-01-23T10:06:08+00:00
**Closed**: 2025-05-24T15:47:25+00:00
**Comments**: 6
**Labels**: documentation, good first issue

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

The recent addition of EAGLE speculative decoding in [here](https://github.com/SafeAILab/EAGLE/pull/173) is powerful. Thank you for creating and maintaining such a useful tool! The existing codebase gives insufficient examples of how it can be used (e.g for Llama3 models, for example) together with `docker compose`. It would be great if another file like https://github.com/sgl-project/sglang/blob/main/docker/compose.yaml can be added to illustrate how the feature can be used in docker environments. Thanks for looking into this issue!

---

## Issue #N/A: [Feature] Add progress bar in `Engine.generate` method

**Link**: https://github.com/sgl-project/sglang/issues/2994
**State**: closed
**Created**: 2025-01-20T03:00:38+00:00
**Closed**: 2025-01-21T19:22:16+00:00
**Comments**: 3

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

The current state of the `generate` method during the generation process is unknown, which makes it difficult to estimate the completion time during large-scale data inference. Therefore, it is hoped that a progress bar can be added to this method (this feature is supported within vllm).

### Related resources

_No response_

---

## Issue #N/A: [Feature] Rewrite docs for LLama 405B and ModelSpace

**Link**: https://github.com/sgl-project/sglang/issues/2743
**State**: closed
**Created**: 2025-01-06T03:00:14+00:00
**Closed**: 2025-05-16T02:58:35+00:00
**Comments**: 3
**Labels**: documentation, good first issue

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

https://sgl-project.github.io/backend/server_arguments.html#use-models-from-modelscope

https://sgl-project.github.io/backend/server_arguments.html#example-run-llama-3-1-405b

These two docs have been out of date for long. We need to move it under `docs/reference` as two separate markdown and verify the content.

### Related resources

No such.

---

## Issue #N/A: [Feature] Proposal: Releasing SGLang memory when idle

**Link**: https://github.com/sgl-project/sglang/issues/2583
**State**: closed
**Created**: 2024-12-26T02:23:14+00:00
**Closed**: 2025-03-01T00:18:51+00:00
**Comments**: 13
**Labels**: high priority, inactive, feature

### Description

### Proposal 1: Release KV cache when engine is idle

When using SGLang for generation in a training pipeline (such as PPO), at the phase of running HuggingFace model forward/backward, SGLang currently needs to take a lot of memory even though it does not use it. It would be great to make SGLang use as little memory as possible when it is idle.

Example usage cases:
* Suppose we run OpenRLHF on 8xH100, the currently we may allocate 4xH100 for vllm/SGLang and another 4xH100 for HF model (thanks @zhaochenyang20 for providing this usage scenario).
	* If we make SGLang use little memory when idle, then we can run the same experiment on half number of GPUs (4xH100) by putting those SGLang engines on the same GPUs as HF models.
* Suppose we run PPO on 1xH100 for a 7B model with Adam offloading (thanks @zhaochenyang20 for providing this usage scenario). Then policy (7Bx2) + critic (7Bx2) + ref (7Bx2) + reward (7Bx2) already takes 56B. The current SGLang needs 7Bx2 for weights and some 

[... truncated for brevity ...]

---

## Issue #N/A: [Usage] Some questions about the parameter --chunked-prefill-size

**Link**: https://github.com/sgl-project/sglang/issues/2814
**State**: closed
**Created**: 2025-01-09T11:49:19+00:00
**Closed**: 2025-01-09T12:01:26+00:00
**Comments**: 0

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

```
def budget_state(self):
        if self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0:
            return AddReqResult.NO_TOKEN

        if self.rem_input_tokens <= 0 or (
            self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0
        ):
            return AddReqResult.OTHER

        return AddReqRes

[... truncated for brevity ...]

---

## Issue #N/A: [torch.compile] Large cache size limit

**Link**: https://github.com/sgl-project/sglang/issues/2604
**State**: closed
**Created**: 2024-12-26T18:48:27+00:00
**Closed**: 2025-02-25T00:17:05+00:00
**Comments**: 4
**Labels**: inactive

### Description

### Describe the bug

https://github.com/sgl-project/sglang/blob/2125898af5224464f5b5999e32a6cc93f442199c/python/sglang/srt/model_executor/cuda_graph_runner.py#L103-L105

From torch 2.5 version, we should not need such a large cache size limit. Is it possible for someone to double check and remove the override?

### Reproduction

NA

### Environment

NA

---

## Issue #N/A: [Feature] Multinode docker container

**Link**: https://github.com/sgl-project/sglang/issues/2817
**State**: closed
**Created**: 2025-01-09T14:27:08+00:00
**Closed**: 2025-02-10T05:52:34+00:00
**Comments**: 3

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

I am encountering an issue where InfiniBand is not being fully utilized during multi-node deployment of DeepSeek v3. Upon investigation, I discovered that the current base Docker image being used is https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda-dl-base, which explicitly states in its description that it does not support multi-node configurations.

I attempted to switch to an alternative base image, https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch, but so far, I have not been successful in resolving the issue. Once I achieve a working solution, I will share the corresponding Dockerfile.

In the meantime, I would like to inquire if you are aware of a suitable base image that could replac

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Reasoning model API support

**Link**: https://github.com/sgl-project/sglang/issues/3043
**State**: closed
**Created**: 2025-01-22T06:24:07+00:00
**Closed**: 2025-03-06T06:30:24+00:00
**Comments**: 6
**Labels**: help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

In order to better support reasoning models, such as DeepSeek-R1, etc., the API needs to support the **reasoning_effort** parameter. In addition, it is recommended to add **reasoning_content** to the output field mentioned in [reasoning_model](https://api-docs.deepseek.com/zh-cn/guides/reasoning_model) , used to display step information of reasoning thinking.
Similar to the dialogue completion interface parameters provided by openai. The parameter reasoning_effort support o1 model: "constrains effort on reasoning for [reasoning models](https://platform.openai.com/docs/guides/reasoning). Currently supported values are low, medium, and high. Reducing reasoning effort can result in faster responses and fewer tokens us

[... truncated for brevity ...]

---

## Issue #N/A: Do not use tools param in stream request!

**Link**: https://github.com/sgl-project/sglang/issues/2810
**State**: closed
**Created**: 2025-01-09T08:40:45+00:00
**Closed**: 2025-03-23T00:19:21+00:00
**Comments**: 2
**Labels**: help wanted, inactive

### Description

https://github.com/sgl-project/sglang/blob/b5fb4ef58a6bbe6c105d533b69e8e8bc2bf4fc3c/python/sglang/srt/openai_api/adapter.py#L882

If you give a tools param in your request and set stream=True, then the output format will be changed by the server and you will get nothing by `for` grammar (no error will be raised), because the two processing are complete different in the client:
```
stream -> received with generator of chunks: generater -> async for chunk in result:
non-stream-> received with a fixed result chunk -> use it direct
```

So, I think if the server does not support stream with tools, then it will be better to return a http error than changing the return method so that the developers can know what should  be done or not.

---

## Issue #N/A: [Feature] Rewrite the SRT Backend docs

**Link**: https://github.com/sgl-project/sglang/issues/2660
**State**: closed
**Created**: 2024-12-30T07:49:17+00:00
**Closed**: 2025-05-24T21:27:16+00:00
**Comments**: 3
**Labels**: documentation, good first issue, help wanted, RLHF

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

This doc has been outdated for a long time:

https://sgl-project.github.io/backend/backend.html#backend-sglang-runtime-srt

1. Only keep an explanation for server arguments and give the link to sampling parameters.
2. Add essential explanation for server arguments. Remember to add these kinds of arguments. https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models
3. A group of parameters have ##, ### is not allowed.
4. Use Models From ModelScope and Run Llama 3.1 405B move to reference, and potentially adds docs for deepseek.
5. change main readme.md.


### Related resources

No such.

---

## Issue #N/A: [Bug] Regex isn't precluding parentheticals. And maybe more.

**Link**: https://github.com/sgl-project/sglang/issues/2957
**State**: closed
**Created**: 2025-01-17T22:03:25+00:00
**Closed**: 2025-03-30T00:19:33+00:00
**Comments**: 19
**Labels**: help wanted, inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I don't think the regex is working correctly as I'd expect the result below to exclude parentheses. I get that it's changing the distribution a lot, but something else seems to be going on that is letting through ')'.

The messages turned into a prompt ends with:
```
<start_of_turn>model
Hey<end_of_turn>
<start_of_turn>user
What are all th

[... truncated for brevity ...]

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

## Issue #N/A: [Bug] How to load weight with torchao

**Link**: https://github.com/sgl-project/sglang/issues/2721
**State**: closed
**Created**: 2025-01-03T07:27:11+00:00
**Closed**: 2025-03-24T00:18:34+00:00
**Comments**: 13
**Labels**: inactive

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

I load 160B weight with 4*L40 GPU
python3 -m sglang.launch_server --model-path 160B_32 --tp-size 4 --trust-remote-code --disable-cuda-graph --torchao-config int8wo
but I got CUDA OOM error
What method can be used to load this model with 4 gpus, or can the torchao loading model be saved locally?

### Reproduction

python3 -m sglang.launc

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] KeyError: 'lm_head.weight' when loading quantized llama 3.2 3B and 1B models

**Link**: https://github.com/sgl-project/sglang/issues/2935
**State**: closed
**Created**: 2025-01-17T06:46:51+00:00
**Closed**: 2025-02-24T04:04:16+00:00
**Comments**: 12

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

The issue arises when I try to load quantized models of llama 3.2 models of sizes 3B and 1B models. This doesnot happen with llama 3.1 8B model. When I launch the quantized model "neuralmagic/Llama-3.2-1B-Instruct-quantized.w8a8" using sglang docker, the following error is raised. The same model is loaded properly in VLLM.

```
[2025-01-16

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] The performance of v0.4.1 on AMD GPU is lower than v0.4.0

**Link**: https://github.com/sgl-project/sglang/issues/2675
**State**: closed
**Created**: 2024-12-31T03:56:50+00:00
**Closed**: 2025-03-02T00:18:46+00:00
**Comments**: 2
**Labels**: inactive, amd

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

We found that the performance test results on the latest sglang v0.4.1 version were lower than v0.4.0. The following are the test results
![v0 4 0](https://github.com/user-attachments/assets/0aa783d9-2f4a-4c7a-8670-9b6203428ba1)
![v0 4 1](https://github.com/user-attachments/assets/3410e457-34ff-4992-85fd-fea88f8cc027)

By com

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] finish_reason is not right when Qwen call a tool

**Link**: https://github.com/sgl-project/sglang/issues/2877
**State**: closed
**Created**: 2025-01-14T03:06:37+00:00
**Closed**: 2025-05-13T00:19:06+00:00
**Comments**: 7
**Labels**: help wanted, inactive

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

{
    "completion": {
        "created": 1736822678,
        "usage": {
            "completion_tokens": 75,
            "prompt_tokens": 43,
            "total_tokens": 118
        },
        "model": "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
        "id": "a82af6309caf48a0994c77acbedbc846",
        "choices": [
            {
   

[... truncated for brevity ...]

---

## Issue #N/A: Can router support --api-key parameter

**Link**: https://github.com/sgl-project/sglang/issues/3031
**State**: closed
**Created**: 2025-01-21T10:02:21+00:00
**Closed**: 2025-01-24T04:30:32+00:00
**Comments**: 4
**Labels**: router

### Description

When I add an api key to the worker, the router cannot access it

---

## Issue #N/A: [Bug] Bug of top_logprobs for the first chunk

**Link**: https://github.com/sgl-project/sglang/issues/2825
**State**: closed
**Created**: 2025-01-10T03:39:33+00:00
**Closed**: 2025-03-23T00:19:19+00:00
**Comments**: 3
**Labels**: help wanted, inactive

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

Somtimes I got this chunk in the first chunk where the content contains two token and the top_logprobs of the second token is not right.

```
# chunk 1:
{
  "id": "cdb8a0104327465c85455ea8ad0580fa",
  "choices": [
    {
      "delta": {
        "content": "Hello!",
        "function_call": null,
        "refusal": null

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] FlashInfer new version integration

**Link**: https://github.com/sgl-project/sglang/issues/2620
**State**: closed
**Created**: 2024-12-27T18:14:29+00:00
**Closed**: 2025-03-11T00:17:39+00:00
**Comments**: 3
**Labels**: enhancement, high priority, inactive, flashinfer

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

### Related resources

_No response_

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

## Issue #N/A: [Feature] Support service discovery on Kubernetes in router

**Link**: https://github.com/sgl-project/sglang/issues/3073
**State**: closed
**Created**: 2025-01-23T07:08:03+00:00
**Closed**: 2025-03-26T00:17:50+00:00
**Comments**: 3
**Labels**: inactive, router

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

This feature proposes adding Kubernetes service discovery support to the router component. Service discovery will enable the router to dynamically identify and connect to backend services running in a Kubernetes cluster. This is particularly useful for distributed systems where backend instances may scale up or down dynamically.

## UI/UX

```bash
# New approach
python -m sglang_router.launch_router --worker-service-on-k8s default/sglang-svc
# Static approach
python -m sglang_router.launch_router --worker-urls http://worker_url_1 http://worker_url_2
```

## Pseudo code

```py
# Load Kubernetes configuration (e.g., from kubeconfig or in-cluster config)
load_kube_config()

# Initialize Kubernetes API client
api_clien

[... truncated for brevity ...]

---

