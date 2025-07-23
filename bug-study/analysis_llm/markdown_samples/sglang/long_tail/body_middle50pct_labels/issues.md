# body_middle50pct_labels - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 9
- Closed Issues: 21

### Label Distribution

- inactive: 11 issues
- deepseek: 6 issues
- enhancement: 5 issues
- good first issue: 5 issues
- collaboration: 5 issues
- high priority: 4 issues
- amd: 4 issues
- feature: 4 issues
- await-response: 4 issues
- documentation: 3 issues

---

## Issue #N/A: Instruction for Running DeepSeek with PD, EP, and MTP

**Link**: https://github.com/sgl-project/sglang/issues/7998
**State**: open
**Created**: 2025-07-13T18:21:14+00:00
**Comments**: 3
**Labels**: deepseek

### Description

# Using Main Branch

## Environment Preparation
Use SGLang and DeepEP on master is sufficient. Also remember to upgrade Mooncake. It will be better to create customized expert distribution data for MTP (follow the related instructions in #6017)


## xP + 2D, max_running_requests=32, draft_token_num=3

### Command for decode
```
SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=10000000 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 MC_TE_METRIC=true python3 -m sglang.launch_server --model-path /mnt/shared-fs/models/deepseek-ai/DeepSeek-V3-0324 --disaggregation-ib-device mlx5_1 --disaggregation-mode decode --dist-init-addr 10.0.7.67:5757 --tp-size 16 --dp-size 16 --enable-dp-attention --decode-log-interval 1 --enable-deepep-moe --page-size 64 --host 0.0.0.0 --trust-remote-code --moe-dense-tp-size 1 --enable-dp-lm-head --disable-radix-cache --watchdog-timeout 1000000 --deepep-mode low_latency --mem-fraction-static 0.8 --max-running-requests 32 --cont

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] support NVRTC for DeepGEMM

**Link**: https://github.com/sgl-project/sglang/issues/5313
**State**: closed
**Created**: 2025-04-12T06:17:24+00:00
**Closed**: 2025-05-13T08:45:22+00:00
**Comments**: 0
**Labels**: high priority, deepseek

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

### Related resources

_No response_

---

## Issue #N/A: [Bug] sglang crashed when using /health_generate

**Link**: https://github.com/sgl-project/sglang/issues/3695
**State**: closed
**Created**: 2025-02-19T10:45:57+00:00
**Closed**: 2025-05-26T00:19:58+00:00
**Comments**: 4
**Labels**: inactive, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When deploying DeepSeek-V3/DeepSeek-R1 on two 8xH20 nodes, configuring a health probe (using health_generate) causes the service to crash after approximately 40 minutes.

the crash log:

```2025-02-19 11:45:37 TP0] Prefill batch. #new-seq: 1, #new-token: 1, #cached-token: 0, cache hit rate: 0.00%, token usage: 0.00, #running-req: 0, #queue

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Per-request random seed

**Link**: https://github.com/sgl-project/sglang/issues/1335
**State**: closed
**Created**: 2024-09-05T13:16:11+00:00
**Closed**: 2024-12-14T00:17:29+00:00
**Comments**: 8
**Labels**: enhancement, inactive

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

I believe there is an option for fixing the random seed for the backend, but I think there isn't a feature for per-request random seeds.

### Related resources

_No response_

---

## Issue #N/A: lora speed

**Link**: https://github.com/sgl-project/sglang/issues/2559
**State**: closed
**Created**: 2024-12-23T14:28:48+00:00
**Closed**: 2025-02-22T00:16:12+00:00
**Comments**: 2
**Labels**: enhancement, inactive

### Description

I measured the speed of starting multiple loras using sglang and vllm. Why is vllm faster than sglang? What acceleration method is sglang? I haven’t enabled it yet?
Graphics card 4090
sglang sever：
python -m sglang.launch_server --model-path /mnt/models/source/model/qwen2_5-7b-instruct/Qwen2___5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tp-size 1 \
  --mem-fraction-static 0.9 \
  --served-model-name "Qwen2.5-7B-Instruct" \
  --chunked-prefill-size 4096 \
  --disable-cuda-graph \
  --disable-radix-cache \
  --show-time-cost \
  --enable-torch-compile \
  --schedule-conservativeness 0.03 \
  --schedule-policy fcfs \
  --lora-paths lora0=“” lora_batch="" \
  --max-loras-per-batch 32 \
  --dtype bfloat16

vllm sever
python -m vllm.entrypoints.openai.api_server --model /mnt/models/source/model/qwen2_5-7b-instruct/Qwen2___5-7B-Instruct \
   --port 8899 \
   --served-model-name Qwen2.5-7B-Instruct \
   --enable-lora \
   --lora-moduleslora0=“” lora_batch=""

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Set outlines and xgrammar as addtional dependency

**Link**: https://github.com/sgl-project/sglang/issues/2549
**State**: closed
**Created**: 2024-12-23T02:35:28+00:00
**Closed**: 2025-02-22T00:16:13+00:00
**Comments**: 4
**Labels**: enhancement, inactive, grammar-backend

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

I am trying to integrate SGLang and vllm into OpenRLHF. For the grammar backend, could we set it as additional requirements, i.e. import it when we use it? Like:

```python

def __init__():
    if use_constrained_decoding:
        if grammar_backend == "xgrammar":
            import xgrammar
            xgrammar.function()
        if grammar_backend == "outlines":
            import outlines
            outlines.function()
```

This to avoid the version conflicts with vllm.

### Related resources

No such.

---

## Issue #N/A: [Bug] Tried to run DeepSeek V3 by amd instructions

**Link**: https://github.com/sgl-project/sglang/issues/3200
**State**: closed
**Created**: 2025-01-28T22:33:58+00:00
**Closed**: 2025-04-03T00:17:38+00:00
**Comments**: 4
**Labels**: documentation, help wanted, inactive, amd

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I tried to use [AMD instruction](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html) but i have an error.

### Reproduction

After running in a container
```
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --port 30000 --tp 8

[... truncated for brevity ...]

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

## Issue #N/A: [Feature] Add docs for pass in token ids directly

**Link**: https://github.com/sgl-project/sglang/issues/2661
**State**: open
**Created**: 2024-12-30T07:51:00+00:00
**Comments**: 10
**Labels**: documentation, good first issue, RLHF

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

In most of RLHF frameworks, the prompts are pre-tokenized when data processing, so they can directly pass in token ids to the sglang engine rather than the prompts. So we should add docs on how to do this and how to get tokens directly.

### Related resources

No such.

---

## Issue #N/A: [Feature] Support TRI-ML/prismatic-vlms

**Link**: https://github.com/sgl-project/sglang/issues/1129
**State**: open
**Created**: 2024-08-16T18:15:10+00:00
**Comments**: 2
**Labels**: good first issue, feature, new-model

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

I'm trying to speed up inference for new VLM models on huggingface: https://huggingface.co/TRI-ML/prismatic-vlms/tree/main. I'm wondering if there are additional documentation on how to adapt new models? 

### Related resources

The model I'm trying to adapt is detailed here: https://arxiv.org/pdf/2402.07865. 

---

## Issue #N/A: [Feature] Extend CustomLogitProcessor to Support input_ids in call Method

**Link**: https://github.com/sgl-project/sglang/issues/3524
**State**: closed
**Created**: 2025-02-12T12:48:38+00:00
**Closed**: 2025-06-25T00:20:02+00:00
**Comments**: 6
**Labels**: inactive, feature

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Thanks @hongpeng-guo for PR #2396. After reviewing your work, I'd like to propose an enhancement to the `CustomLogitProcessor`. Specifically, I suggest modifying its `__call__` method to accept `input_ids` as an additional parameter—similar to the implementation in Huggingface (see this [doc](https://huggingface.co/docs/transformers.js/en/api/generation/logits_process#module_generation/logits_process.LogitsProcessor)). This change would allow constraints to be applied conditionally based on the entire history of input tokens, enabling more flexible and context-aware processing.

Thank you for considering this feature request!

### Related resources

[Huggingface LogitsProcessor.](https://huggingface.co/docs/transfo

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] (Willing to PR) Avoid KV cache occupying GPU memory when not used

**Link**: https://github.com/sgl-project/sglang/issues/2542
**State**: closed
**Created**: 2024-12-22T09:07:26+00:00
**Closed**: 2025-03-16T14:34:36+00:00
**Comments**: 43
**Labels**: high priority, collaboration, inactive, feature

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Hi thank you for the library! The use case is that, when doing online PPO, I hope to use SGLang to generate llm completions, and then use RL to do gradient descent on those completions.

The problem is, to do this on a single GPU, the timeline is "SGLang generate - Torch backward - repeat it". Thus, when torch doing backprop, I hope SGLang can free its KV cache memory consumption, otherwise torch will not have enough memory.

Thanks for any suggestions!

### Related resources

_No response_

---

## Issue #N/A: [Bug] tp-size=2，model launch error

**Link**: https://github.com/sgl-project/sglang/issues/1945
**State**: closed
**Created**: 2024-11-07T06:14:03+00:00
**Closed**: 2025-01-29T00:16:26+00:00
**Comments**: 5
**Labels**: await-response, inactive, amd

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

tp-size=2, model launch is frozen.

### Reproduction

 python3 -m sglang.launch_server --model-path  /root/.xinference/cache/qwen2_5-instruct-gptq-7b-Int8/ --port 30000 --mem-fraction-static  0.8 --tp-size 2 --kv-cache-dtype int8 --attention-backend triton --sampling-backend pytorch --enable-torch-compile

### Environment

amd gpu RTX 7900

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] AttributeError: module 'vllm._custom_ops' has no attribute 'silu_and_mul'

**Link**: https://github.com/sgl-project/sglang/issues/3392
**State**: closed
**Created**: 2025-02-08T06:32:39+00:00
**Closed**: 2025-05-03T00:18:16+00:00
**Comments**: 5
**Labels**: inactive, amd, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hell folks,

 I'm attempting to deploy DeepSeek-R1 with SGLang on an AMD MI300X, but I'm encountering compatibility issues. 
Could someone please help me troubleshoot these issues?


### Reproduction

1. build and install **triton 3.0.0** from source
2. build and install **vllm v0.7.2** from source
3. build and install **sglang** (rev 0a6f

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] GPU inference on AMD Ryzen AI (370HX-890M) iGPU + NPU

**Link**: https://github.com/sgl-project/sglang/issues/3823
**State**: closed
**Created**: 2025-02-24T15:49:18+00:00
**Closed**: 2025-04-26T00:17:55+00:00
**Comments**: 2
**Labels**: inactive, feature, amd

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Ryzen AI devices have been out since mid 2024 yet there's no end user friendly local inference engine that can use the iGPU or the NPU for inference. Some people seem to be able to make it working using hacks but it's still a hit or miss and you need to build your own custom room and hip packages to it to kind of work. 

### Related resources

_No response_

---

## Issue #N/A: Instruction for Running DeepSeek with Large-scale PD and EP

**Link**: https://github.com/sgl-project/sglang/issues/6017
**State**: open
**Created**: 2025-05-05T04:48:15+00:00
**Comments**: 504
**Labels**: collaboration, deepseek

### Description

## Using main branch

~~NOTE: The feature is already on main, but the performance still needs some improvements on main branch.~~ will be good after a few already opened PRs - PR 6680, 6727, 6728

~~NOTE: I will try other config like 4 node for P and 9 node for D later.~~ updated

### Environment Preparation

Use SGLang and DeepEP on master is sufficient. Also remember to upgrade Mooncake.

### 4P + 9D experiments

Start server
where DeepEP config can be tuned by https://github.com/sgl-project/sglang/pull/6742

```python
# prefill nodes
MC_TE_METRIC=true SGLANG_TBO_DEBUG=1 python3 -m sglang.launch_server --model-path /dev/shm/DeepSeek-V3-0324 --disaggregation-ib-device mlx5_1 --disaggregation-mode prefill --dist-init-addr 10.5.55.3:5757 --nnodes 4 --node-rank 0 --tp-size 32 --dp-size 32 --enable-dp-attention --decode-log-interval 1 --enable-deepep-moe --page-size 1 --host 0.0.0.0 --trust-remote-code --moe-dense-tp-size 1 --enable-dp-lm-head --disable-radix-cache --watchdog-timeout 1000

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Benchmarking Performance on General Devices

**Link**: https://github.com/sgl-project/sglang/issues/2488
**State**: closed
**Created**: 2024-12-16T08:01:21+00:00
**Closed**: 2025-05-11T00:20:28+00:00
**Comments**: 4
**Labels**: enhancement, collaboration, inactive

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

We need to benchmark the speed performance of SGLang on various devices, at least different types of GPUs. This could give users a standard of the engine and whether their engines are working appropriately.

### Related resources

No such.

---

## Issue #N/A: [PD] Support Multi-Process for TokenizerManager

**Link**: https://github.com/sgl-project/sglang/issues/6553
**State**: open
**Created**: 2025-05-23T09:19:04+00:00
**Comments**: 0
**Labels**: enhancement, collaboration, deepseek

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.
The diagram below briefly outlines the process of a request from input to output：[Detailed Documentation]

<img width="789" alt="Image" src="https://github.com/user-attachments/assets/2e95df40-c9bd-4078-80da-77098881e62e" />


The TokenizerManager is responsible for three main tasks:

1.  Receiving r

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] add disable_custom_all_reduce

**Link**: https://github.com/sgl-project/sglang/issues/1118
**State**: closed
**Created**: 2024-08-16T04:59:48+00:00
**Closed**: 2024-08-21T04:53:40+00:00
**Comments**: 7
**Labels**: await-response

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Sometimes, we need to turn off Custom allreduce. 
Please  support disable_custom_all_reduce.


### Related resources

_No response_

---

## Issue #N/A: [Bug] process not terminated after PM2 is kill

**Link**: https://github.com/sgl-project/sglang/issues/680
**State**: closed
**Created**: 2024-07-21T00:11:31+00:00
**Closed**: 2024-08-01T10:51:40+00:00
**Comments**: 4
**Labels**: await-response

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.

### Describe the bug

I use pm2 to run the server and it appears the python process is still running after the pm2 is killed, the GPUs were still occupied. How do I properly terminate the process?

### Reproduction

pm2 start /usr/bin/python --name sglang-launch-server -- -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000

### Environment

```Shell
N/A
```


---

## Issue #N/A: Accuracy degrading in concurrent scenario

**Link**: https://github.com/sgl-project/sglang/issues/1203
**State**: closed
**Created**: 2024-08-25T03:00:16+00:00
**Closed**: 2024-09-22T12:51:30+00:00
**Comments**: 2
**Labels**: await-response

### Description

Hi, I have tested that when the concurrency is 1, the accuracy is expected. However, when concurrency increases, accuracy degrades. I have checked that no decoding oom happened. From the log, there also seems to have no exception.

The model is qwen2-7b-awq.

---

## Issue #N/A: [Feature] Customized mapping for LoRA weight names

**Link**: https://github.com/sgl-project/sglang/issues/6608
**State**: open
**Created**: 2025-05-26T04:08:39+00:00
**Comments**: 0
**Labels**: low-priority, lora

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

The current LoRA impl in SGL maps LoRA weight to modules by (layer index, op_type) tuple, where op_type operation looks like `qkv_proj`, `o_proj`, `gate_up`, etc. This works fine for most standard cases, however, there are some limitations:
1. For models where there are more than one attention stacks (e.g., VLM), there could be multiple modules with the same (layer index, op_type), e.g., one from vision tower, the other from the language model. Currently SGL cannot handle such cases correctly and would usually fail during loading due to incorrect mapping.
2. Users cannot enable/disable application of LoRA at module-level, e.g., if user only wants to apply LoRA at language model but not vision (common); or when user

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] optimize group gemm

**Link**: https://github.com/sgl-project/sglang/issues/3323
**State**: closed
**Created**: 2025-02-05T22:56:43+00:00
**Closed**: 2025-02-20T08:26:59+00:00
**Comments**: 1
**Labels**: high priority, performance, lora

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Rewrite the  Grouped GEMM used by LoRA with cuBLAS 12.5 in sgl-kernel for improved speed.

https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/
https://github.com/zhihu/ZhiLight/blob/main/src/nn/linear/gemm_grouped.cpp

### Related resources

_No response_

---

## Issue #N/A: [Feature] Cutlass kernels for LoRA

**Link**: https://github.com/sgl-project/sglang/issues/7910
**State**: open
**Created**: 2025-07-09T21:43:29+00:00
**Comments**: 0
**Labels**: lora

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Creating an issue to track the work for supporting a CUTLASS / CUTE kernel for LoRA to see if there is any perf gain comparing with the current Triton one.

Dependency: this task should happen after #7809 as the FlashInfer deprecation is expected to change / simplify the kernel interface.

(cc @Fridge003 @Ying1123 )

### Related resources

_No response_

---

## Issue #N/A: [Bug] FA3 + EAGLE2: speculative_token_map not supported

**Link**: https://github.com/sgl-project/sglang/issues/6863
**State**: closed
**Created**: 2025-06-04T08:14:37+00:00
**Closed**: 2025-06-27T21:00:23+00:00
**Comments**: 0
**Labels**: speculative-decoding

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

We conducted a large-scale load test with speculative decoding using EAGLE2. According to our results, enabling speculative_token_map effectively reduces overhead for larger batch sizes. However, this feature only works with the flashinfer backend, while the FA3 backend does not support it.

Here is our graph demonstrating this:

+FR is ru

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

## Issue #N/A: [Bug] Error when running Qwen2 EAGLE spec decoding with the official OFFLINE inference example

**Link**: https://github.com/sgl-project/sglang/issues/7263
**State**: open
**Created**: 2025-06-17T05:37:21+00:00
**Comments**: 1
**Labels**: speculative-decoding

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hi, I am running eagle offline speculative decoding example (examples/runtime/engine/offline_batch_inference_eagle.py), and I encountered an error as shown below. Specifically, I modified the target model to Qwen/Qwen2-7B-Instruct and draft model to yuhuili/EAGLE-Qwen2-7B-Instruct.
I did notice there was a previous [issue](https://github.c

[... truncated for brevity ...]

---

## Issue #N/A: Qwen2.5 VL sglang's output much worse than transformers

**Link**: https://github.com/sgl-project/sglang/issues/3746
**State**: closed
**Created**: 2025-02-21T06:38:34+00:00
**Closed**: 2025-05-16T06:24:46+00:00
**Comments**: 17
**Labels**: MLLM

### Description

I tried serving qwen2.5 vl 72B using sglang on a node with 4*A40 GPUs.
The image I used is the official sglang:v0.4.3.post2-cu125
The command:
```bash
python3 -m sglang.launch_server \
  --tp $NUM_SHARD \
  --mem-fraction-static 0.99 \
  --disable-cuda-graph \
  --model-path /model/Qwen2.5-VL-72B-Instruct \
  --host 0.0.0.0 \
  --port 23333
```

I tested  using an internal image classification dataset, the results were much worse than when using transformers, acc droped from 87% to 80%.
And I tried another image2code task, the rendered images were much worse, too.

---

## Issue #N/A: [Track] VLM accuracy in MMMU benchmark

**Link**: https://github.com/sgl-project/sglang/issues/4456
**State**: closed
**Created**: 2025-03-15T17:09:50+00:00
**Closed**: 2025-04-25T07:23:54+00:00
**Comments**: 5
**Labels**: good first issue, MLLM

### Description

This issue keeps track of all vlm models accuracy in MMMU benchmark. Keep updating

``` python
python benchmark/mmmu/bench_sglang.py
python benchmark/mmmu/bench_hf.py --model-path model

```

| | sglang | hf |
|--|--|--|
| Qwen2-VL-7B-Instruct |  0.485 | 0.255 |
| Qwen2.5-VL-7B-Instruct | 0.477 | 0.242 |
| MiniCPM-V-2_6 |  0.426 |  |
| MiniCPM-O-2_6 | 0.481| 0.49 |
| Deepseek-vl2 | 0.496 | 0.499|
|Deepseek-vl2-small | 0.464 | 0.453|
|Deepseek-vl2-tiny | 0.382 | 0.369|
| Deepseek-Janus-Pro-7B| | |
| Llava + Llama| | |
| Llava + qwen| | |
| Llava + Mistral| | |
| Mlama | | |
| Gemma-3-it-4B| 0.409 | 0.403 |
| InternVL2.5-38B | 0.61 | |



---

## Issue #N/A: [Feature] Benchmark with audio input

**Link**: https://github.com/sgl-project/sglang/issues/8072
**State**: open
**Created**: 2025-07-15T22:28:51+00:00
**Comments**: 1
**Labels**: good first issue, help wanted, MLLM

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

We need scripts to bench audio input for supported MLLM like minicpmo and gemma3n.

### Related resources

https://github.com/vllm-project/vllm/issues/16354

---

