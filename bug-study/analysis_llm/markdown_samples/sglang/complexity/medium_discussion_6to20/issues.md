# medium_discussion_6to20 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 2
- Closed Issues: 28

### Label Distribution

- inactive: 14 issues
- await-response: 3 issues
- help wanted: 1 issues
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

## Issue #N/A: [Feature] Does Sglang has Speculative MoE feature mentioned in the paper? https://arxiv.org/pdf/2503.04398

**Link**: https://github.com/sgl-project/sglang/issues/5906
**State**: closed
**Created**: 2025-04-30T05:06:11+00:00
**Closed**: 2025-07-01T00:22:53+00:00
**Comments**: 8
**Labels**: inactive

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

The paper mentioned that they build this feature into SGLang and hence wonder if it is True? as it does not seem to have related PR. 

### Related resources

_No response_

---

## Issue #N/A: [Bug] disable_flashinfer didn't take effect

**Link**: https://github.com/sgl-project/sglang/issues/945
**State**: closed
**Created**: 2024-08-06T06:46:53+00:00
**Closed**: 2024-08-06T09:35:48+00:00
**Comments**: 10
**Labels**: await-response

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

I tried pip install flashinfer, but got error from
```
 File "/usr/local/python/lib/python3.10/site-packages/sglang/srt/model_executor/model_runner.py", line 28, in <module>
    from flashinfer import (
ModuleNotFoundError: No module named 'flashinfer'
```

According to readme, i tried to set disable_flashinfer when init the runtime, my code is like

```
self.runtime = sgl.Runtime(
            model_path = self.loader.target_path,
            tp_size = envs.TENSOR_PARALLEL_SIZE,
            trust_remote_code = True,
            max_num_reqs = 40,
            d

[... truncated for brevity ...]

---

## Issue #N/A: Add SGLang usage examples

**Link**: https://github.com/sgl-project/sglang/issues/166
**State**: closed
**Created**: 2024-02-08T08:51:53+00:00
**Closed**: 2024-09-08T01:13:00+00:00
**Comments**: 8
**Labels**: inactive

### Description

List some good use cases of SGLang here:
- [SELF-DISCOVER: Large Language Models Self-Compose Reasoning Structures](https://arxiv.org/pdf/2402.03620.pdf)
- [Tractable Control for Autoregressive Language Generation](https://starai.cs.ucla.edu/papers/ZhangICML23.pdf)

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

## Issue #N/A: [Bug] Unrecognized keys in `rope_scaling` for 'rope_type'='yarn': {'original_max_position_embeddings'}

**Link**: https://github.com/sgl-project/sglang/issues/2943
**State**: closed
**Created**: 2025-01-17T12:12:19+00:00
**Closed**: 2025-04-22T16:45:03+00:00
**Comments**: 12
**Labels**: help wanted

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I am using Qwen2.5-72B which suppots positional extrapolation by Yarn through adding config(copied from https://huggingface.co/Qwen/Qwen2.5-72B-Instruct):
```json
{
  "rope_scaling": {
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "type": "yarn"
  }
}

```
However, this seems not supported by sglang, when I specify 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] No performance gain after using hierarchical cache

**Link**: https://github.com/sgl-project/sglang/issues/7059
**State**: closed
**Created**: 2025-06-10T16:56:33+00:00
**Closed**: 2025-06-18T20:25:14+00:00
**Comments**: 6

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hi, @xiezhq-hermann , thanks for your great effort in hierarchical cache. However, in my experiements, I do not observe any performance gain when this feature is enabled.

I am using Llama 3.1 8B model, and various sequence lengths, on a 4xH100 96GB NVLINK node. 

The results shown in this [PR](https://github.com/sgl-project/sglang/pull/40

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] attention dp + attention tp for deepseek v3

**Link**: https://github.com/sgl-project/sglang/issues/3750
**State**: closed
**Created**: 2025-02-21T08:47:02+00:00
**Closed**: 2025-04-12T02:26:03+00:00
**Comments**: 10

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

In the deepseek v3 technical report, it is mentioned "The attention part employs TP4 with SP, combined with DP80". In SGLang, it seems that currently, dp_size must equal tp_size for the system to run.

![Image](https://github.com/user-attachments/assets/b3bbfb18-1433-4d72-b45f-71593370de2d)

 Could you please suggest how to design a solution that would allow for a configuration similar to tp=8, dp=2, and attention_tp=4?

### Related resources

_No response_

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

## Issue #N/A: Deploying DeepSeek R1 on 2*H100 2*8*80G achieves only 19 tokens/s inference speed, while vllm deployment reaches 30 tokens/s. Is this normal?

**Link**: https://github.com/sgl-project/sglang/issues/3656
**State**: closed
**Created**: 2025-02-18T06:14:47+00:00
**Closed**: 2025-04-29T00:18:48+00:00
**Comments**: 6
**Labels**: inactive

### Description

"When deploying DeepSeek R1 on 2*H100 2*8*80G, the inference speed is only 19 tokens/s, while vllm deployment reaches 30 tokens/s. Is this normal? My inference command for node 1 is:

docker run --gpus all --rm --network=host -v /data/models:/data/models -it --env "NCCL_IB_DISABLE=0" --env "NCCL_IB_HCA=mlx5" --env "NCCL_IB_GID_INDEX=3" --env "NCCL_SOCKET_IFNAME=bond0" --env "GLOO_SOCKET_IFNAME=bond0" --env "NCCL_DEBUG=INFO" --ipc=host sglang:v0 python3 -m sglang.launch_server --model-path /data/models/DeepSeek-R1 --tp 16 --dist-init-addr 10.163.34.152:20000 --nnodes 2 --node-rank 0 --trust-remote-code --host 0.0.0.0 --port 8000

I didn't use --enable-dp-attention, as it throws an error whenever I try to use it. Is 19 tokens/s the correct output speed for DeepSeek-R1 using the SLang framework?"

---

## Issue #N/A: [Bug] [CI regression] TestEpMoEFP8

**Link**: https://github.com/sgl-project/sglang/issues/7586
**State**: open
**Created**: 2025-06-27T05:49:07+00:00
**Comments**: 6

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

CI unit-test-backend-2-gpu seems to be broken since the past few days (if not longer).

Looking at the log, it seems to be watchdog timeout at TestEpMoEFP8. The non-quantized version TestEpMoE seems to be working fine.

### Reproduction

Sample failure: https://github.com/sgl-project/sglang/actions/runs/15916204833/job/44895747274

### Env

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Segmentation fault: address not mapped to object at address 0x17

**Link**: https://github.com/sgl-project/sglang/issues/5013
**State**: closed
**Created**: 2025-04-03T01:14:57+00:00
**Closed**: 2025-06-18T00:19:37+00:00
**Comments**: 7
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I now have two single-machine eight-card A100 servers, and I use 0.4.4 to deploy deepseek-r1-channel-int8 and get an error，I now have two single-machine eight-card A100 servers. I use 0.4.4 to deploy deepseek-r1-channel-int8 and get an error. The image used is lmsysorg/sglang:v0.4.4.post1-cu124；

**master log**:
```
[2025-04-02 15:25:19 DP

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

## Issue #N/A: [Bug] --dp-size issue with AMD 8xMI300X and Llama 3.1 70B

**Link**: https://github.com/sgl-project/sglang/issues/3890
**State**: closed
**Created**: 2025-02-26T12:26:17+00:00
**Closed**: 2025-05-12T00:20:26+00:00
**Comments**: 10
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Using --dp-size 4 --tp 2 on 8xMI300X does not work. Is this an MI300X issue or an issue with how I'm passing in sizes?

Error:
```

    _TP = init_model_parallel_group(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/sgl-workspace/sglang/python/sglang/srt/distributed/parallel_state.py", line 890, in init_model_parallel_group
    return Group

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] SGLang Router design discussion

**Link**: https://github.com/sgl-project/sglang/issues/2389
**State**: closed
**Created**: 2024-12-07T11:53:13+00:00
**Closed**: 2025-04-06T00:19:37+00:00
**Comments**: 7
**Labels**: enhancement, inactive

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

@ispobock  and I had a brief discussion about the current design and implementation of the SGLang Router.

I think the main concerns currently are as follows, before large-scale deployment.

- The current Router is stateful, which means I cannot deploy the Router like scaling a stateless service.
May we consider storing the state of the Router in services like Redis, DB, or etcd here?

- The current Router is at the cluster level. Although there are replicas, when the master fails, a replica can be used.
Imagine a real deployment scenario, such as one used by actual customers, where the deployment requires simultaneous use of AWS, GCP, and Oracle. The data centers are distributed across the Western US, Cent

[... truncated for brevity ...]

---

## Issue #N/A: Different output format with OpenAI Client

**Link**: https://github.com/sgl-project/sglang/issues/6158
**State**: closed
**Created**: 2025-05-09T13:11:01+00:00
**Closed**: 2025-07-10T00:20:11+00:00
**Comments**: 10
**Labels**: inactive

### Description

Hello,

It seems there’s a difference in the output format between the OpenAI client and the SGLang OpenAI client for batch requests. This is causing issues when integrating with [`lmm_evals`](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/4ca1a52b55ac4d057329bb1dde092ce68b60256e/lmms_eval/models/batch_gpt4.py#L162).

Is this difference expected, or should we consider making changes on our end to be consistent with the official OpenAI client?

**Official OpenAI Client:** 

The OpenAI client gives output in the following format for a batch request:

> {"id": "batch_req_wnaDys", "custom_id": "request-2", "response": {"status_code": 200, "request_id": "req_c187b3", "body": {"id": "chatcmpl-9758Iw", "object": "chat.completion", "created": 1711475054, "model": "gpt-4o-mini", "choices": [{"index": 0, "message": {"role": "assistant", "content": "2 + 2 equals 4."}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 24, "completion_tokens": 15, "total_tokens": 39}, "system_fingerprint": 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] DeepSeek-Coder-V2-Instruct-FP8 on 8xA100

**Link**: https://github.com/sgl-project/sglang/issues/989
**State**: closed
**Created**: 2024-08-08T08:43:21+00:00
**Closed**: 2024-09-22T12:58:29+00:00
**Comments**: 9

### Description

### Motivation

VLLM has announced their support for running llama3.1-405b-fp8 on 8xA100. This is the [blog](https://blog.vllm.ai/2024/07/23/llama31.html)

Does sglang support running DeepSeek-Coder-V2-Instruct-FP8 on 8xA100?

### Related resources

_No response_

---

## Issue #N/A: [Bug] Multinode cannot be started on runpod

**Link**: https://github.com/sgl-project/sglang/issues/958
**State**: closed
**Created**: 2024-08-07T04:10:19+00:00
**Closed**: 2024-11-14T23:46:49+00:00
**Comments**: 6

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

Hi, I try to run a multi-node following the instructions for 2 machine with 8x4090 each. But it just stopped at "Init nccl begin" without further progress.
<img width="1523" alt="image" src="https://github.com/user-attachments/assets/3ed8af47-d512-42de-9222-33bb545ca05c">


### Reproduction

Node 1:
GLOO_SOCKET_IFNAME=eth0 pm2 start /root/miniconda3/envs/sgl/lib/python3.11/site-packages/sglang/launch_server.py --name sgl -- --model-path NousResearch/Meta-Llama-3.1-8B-Instruct --nccl-init-addr xxx:xxx --nnodes 2 --node-rank 0 --tp-size 16 --schedule-conservativeness 0.01 -

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] router adds add_worker_url, remove_worker_url api

**Link**: https://github.com/sgl-project/sglang/issues/2343
**State**: closed
**Created**: 2024-12-04T06:39:15+00:00
**Closed**: 2025-02-03T00:17:06+00:00
**Comments**: 7
**Labels**: inactive

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Thank you very much to the sglang team for open-sourcing such a useful inference framework.

We are currently deploying Llama3 70b on 3 X 8 X H100s. Run the command on each server separately:
```
python -m sglang_router.launch_server --model-path NousResearch/Hermes-3-Llama-3.1-70B-FP8 --port 6006 --dp-size 4 --tp 2 --router-eviction-interval 300 --context-length 20000
```
Then the three exposed http://127.0.0.1:6006 services are randomly routed by konga. Compared with directly using the --dp command, the cache rate is increased from 25% to 35%.

Why not use --worker-urls?

The current problem is that Rust routing currently supports cross-machine routing, but since workers cannot be added or delete

[... truncated for brevity ...]

---

## Issue #N/A: [DeepseekR1]How ragged prefill manage kv_cache?

**Link**: https://github.com/sgl-project/sglang/issues/3849
**State**: closed
**Created**: 2025-02-25T11:06:13+00:00
**Closed**: 2025-06-14T00:18:52+00:00
**Comments**: 10
**Labels**: inactive

### Description

I'm investigating the chunked prefill method in DeepSeek V3/R1. The code shows that it uses self.prefill_wrapper_ragged.forward_return_lse for both prefill and chunked prefill operations. However, I haven't been able to locate where the KV cache is provided in the code. Could you help me identify this part of the implementation?

<img width="710" alt="Image" src="https://github.com/user-attachments/assets/d98701eb-2d35-4f39-bb6e-d4dde25a571c" />

<img width="582" alt="Image" src="https://github.com/user-attachments/assets/e5684ecf-2239-42ef-a66b-e40de1452e33" />


---

## Issue #N/A: [Bug] out of resource: shared memory when serving RedHatAI/Qwen3-30B-A3B-FP8-dynamic

**Link**: https://github.com/sgl-project/sglang/issues/5995
**State**: open
**Created**: 2025-05-03T09:19:18+00:00
**Comments**: 6

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When running sglang with [Qwen3-30B-A3B-FP8-dynamic](https://huggingface.co/RedHatAI/Qwen3-30B-A3B-FP8-dynamic) quantized model, single request use curl is fine, but benchmark with only 2 concurrency will cause sglang exit with "triton out of resource: shared memory".

Sglang exit with error "triton out of resource: shared memory, Required

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]  DeepSeek server crushed while using sglang.bench_serving

**Link**: https://github.com/sgl-project/sglang/issues/4161
**State**: closed
**Created**: 2025-03-07T05:04:18+00:00
**Closed**: 2025-03-10T07:33:20+00:00
**Comments**: 8

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

I run a DeepSeek R1 server  on 2*8*H800  but it crush while i test the throuput by python3 -m sglang.bench_serving

full log:
nohup: ignoring input
/root/miniconda3/envs/deepseekr1/lib/python3.12/site-packages/transformers/models/auto/image_processing_auto.py:590: FutureWarning: The image_processor_class argument is deprecated and will be 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Unable to run deepseek-v2

**Link**: https://github.com/sgl-project/sglang/issues/1065
**State**: closed
**Created**: 2024-08-13T00:34:06+00:00
**Closed**: 2024-08-13T22:23:17+00:00
**Comments**: 7

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.

### Describe the bug

When I run deepseek-v2-lite with sglang, an error accured:
`  File "/opt/tiger/sglang/python/sglang/srt/managers/tp_worker.py", line 452, in forward_prefill_batch
    output = self.model_runner.forward(batch, ForwardMode.EXTEND)
  File "/opt/tiger/sglang/python/sglang/srt/model_executor/model_runner.py", line 397, in forward
    return self.forward_extend(batch)
  File "/home/tiger/.local/li

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]  main pd version Exception: Failed to encode tensor map: 700

**Link**: https://github.com/sgl-project/sglang/issues/6590
**State**: closed
**Created**: 2025-05-25T11:48:43+00:00
**Closed**: 2025-07-05T05:28:57+00:00
**Comments**: 15

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

```
[2025-05-25 19:45:03] [ERROR] Scheduler hit an exception: Traceback (most recent call last):
  File "/usr/local/src/sglang/python/sglang/srt/managers/scheduler.py", line 2346, in run_scheduler_process
    scheduler.event_loop_overlap_disagg_decode()
  File "/usr/local/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 116, 

[... truncated for brevity ...]

---

## Issue #N/A: [RFC] sm75 EOL

**Link**: https://github.com/sgl-project/sglang/issues/6006
**State**: closed
**Created**: 2025-05-04T06:12:20+00:00
**Closed**: 2025-07-17T00:21:12+00:00
**Comments**: 8
**Labels**: inactive

### Description

The SGLang team plans to deprecate support for sm75 in v0.5. If you’re still using SGLang for large-scale inference acceleration on sm75 devices in production, please let us know so we can defer this deprecation beyond v0.5.

---

## Issue #N/A: [Bug] Tensor model parallel group is not initialized when deploying Qwen3-30B-A3B-AWQ

**Link**: https://github.com/sgl-project/sglang/issues/6000
**State**: closed
**Created**: 2025-05-04T01:08:45+00:00
**Closed**: 2025-07-16T00:20:45+00:00
**Comments**: 18
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hi, SGLang Team,

I am using it to deploy an AWQ quantized model of Qwen3-30B-A3B: swift/Qwen3-30B-A3B-AWQ from modelscope. but encounter the following issue:

```bash
 File "/home/a/sglang/python/sglang/srt/managers/scheduler.py", line 2215, in run_scheduler_process
    scheduler = Scheduler(server_args, port_args, gpu_id, tp_rank, pp_ran

[... truncated for brevity ...]

---

## Issue #N/A: [Docs] EAGLE3 docs

**Link**: https://github.com/sgl-project/sglang/issues/4580
**State**: closed
**Created**: 2025-03-19T10:44:37+00:00
**Closed**: 2025-05-13T15:21:50+00:00
**Comments**: 11

### Description

As discussed [here](https://github.com/sgl-project/sglang/pull/4247) we want to update the docs for speculative decoding to incooperate the new parts on EAGLE3.
@zhaochenyang20 I will take this issue.

---

## Issue #N/A: [Bug] GGUF tokenizations issues

**Link**: https://github.com/sgl-project/sglang/issues/3427
**State**: closed
**Created**: 2025-02-09T13:10:43+00:00
**Closed**: 2025-05-02T00:18:43+00:00
**Comments**: 12
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Special tokens and BOS/EOS tokens are not used, even when they are required in metadata, chat template components that usually take 1 token are treated as multiple tokens,  leading to output quality degradation. Instead of  `<｜User｜>`  we get `<`, `｜`, `User`, `｜`,` >`. 

Originally discussed here https://github.com/sgl-project/sglang/issu

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed!

**Link**: https://github.com/sgl-project/sglang/issues/3385
**State**: closed
**Created**: 2025-02-08T02:09:21+00:00
**Closed**: 2025-02-10T20:49:34+00:00
**Comments**: 7

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I was chatting with @zhaochenyang20 and @shuaills about this issue in slack, [the Slack thread can be found here](https://sgl-fru7574.slack.com/archives/C064NB2TAP9/p1738960365156639).

> **@shuaills asked me to raise this issue.**

When running `meta-llama/Llama-3.2-1B` and performing an inference, the engine throws the following exceptio

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Launching a server with `--enable-torch-compile` produce torch dynamo error

**Link**: https://github.com/sgl-project/sglang/issues/1923
**State**: closed
**Created**: 2024-11-05T08:55:55+00:00
**Closed**: 2025-01-15T00:16:33+00:00
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

I'm using SGLang, but running into some issues when I launch a server with `--enable-torch-compile`. This issue does not occur without `--enable-torch-compile`. One strange thing is that this problem did not occur in previous versions (v0.3.1 or v0.3.2), but this error seems to occur starting from v0.3.4.
```bash
  File "/usr/l

[... truncated for brevity ...]

---

