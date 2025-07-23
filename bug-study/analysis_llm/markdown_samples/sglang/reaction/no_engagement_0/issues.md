# no_engagement_0 - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 6
- Closed Issues: 24

### Label Distribution

- inactive: 5 issues
- bug: 4 issues
- good first issue: 3 issues
- help wanted: 2 issues
- high priority: 2 issues
- RLHF: 1 issues
- deepseek: 1 issues
- speculative-decoding: 1 issues
- documentation: 1 issues

---

## Issue #N/A: [Bug] Memory leak problem in performance stress testing in PD scenario

**Link**: https://github.com/sgl-project/sglang/issues/7256
**State**: open
**Created**: 2025-06-17T02:56:55+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When I deploy a deepep+1p1d service and use batch-size 1500 to stress test the performance of isl-1k/osl-8k, a memory leak problem occurs on the P node. How can I solve this problem?
```
I0616 06:29:14.501163 1115123 transfer_engine.cpp:387] [Metrics] Transfer Engine Throughput: 113.07 MB/s (over last 5s)
[2025-06-16 06:29:15 DP1 TP1] Sche

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] ImportError: libcuda.so.1: cannot open shared object file: No such file or directory

**Link**: https://github.com/sgl-project/sglang/issues/4778
**State**: closed
**Created**: 2025-03-26T03:20:04+00:00
**Closed**: 2025-04-17T09:37:00+00:00
**Comments**: 6

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

There are no CUDA-related libraries in the rocm environment, but the SGLANG 0.4.4.post1 version will report an error,，The following error message is from having the sgl_kernel module installed. python -m sglang.check_env also reports an error. If you don't install sgl_kernel, an error will be reported: Failed to import from custom_ar with 

[... truncated for brevity ...]

---

## Issue #N/A: ModuleNotFoundError: No module named 'sgl_pdlb'

**Link**: https://github.com/sgl-project/sglang/issues/7350
**State**: closed
**Created**: 2025-06-19T09:12:45+00:00
**Closed**: 2025-06-19T09:34:52+00:00
**Comments**: 1

### Description

python3 -m sglang.srt.disaggregation.mini_lb --rust-lb 
How to use the rust lb?

---

## Issue #N/A: [Bug] PD + DP detects memory leak on decode side

**Link**: https://github.com/sgl-project/sglang/issues/6258
**State**: closed
**Created**: 2025-05-13T07:43:38+00:00
**Closed**: 2025-05-14T07:01:17+00:00
**Comments**: 3

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

```
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 2282, in run_scheduler_process                 
    scheduler.event_loop_normal_disagg_decode()                                                                             
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 116, 

[... truncated for brevity ...]

---

## Issue #N/A: Does sglang do automatic batching?

**Link**: https://github.com/sgl-project/sglang/issues/444
**State**: closed
**Created**: 2024-05-15T20:25:42+00:00
**Closed**: 2024-07-18T16:30:25+00:00
**Comments**: 2

### Description

If I hit an sglang server in parallel with 100 requests, will it automatically batch the requests to do as many in parallel as possible?

---

## Issue #N/A: [Help wanted] CANN'T capture GPU activities using `nsight system`

**Link**: https://github.com/sgl-project/sglang/issues/3049
**State**: closed
**Created**: 2025-01-22T11:13:42+00:00
**Closed**: 2025-02-05T09:07:38+00:00
**Comments**: 20

### Description

I use the following codes and commands to generate timeline using nsight system.
```
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
python offline.py
```

```
import sglang as sgl

if __name__ == '__main__':
    model = "/data/models/Llama-2-7b-hf"
    llm = sgl.Engine(model_path=model, watchdog_timeout=30000)
    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 3}

    prompts = [
        "Hello, What is your name? My name is Jim."
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}", flush=True)
```

But I can't find any GPU activities, can any one help?  It seems the nvtx annotations are also abnormal.

![Image](https://github.com/user-attachments/assets/60c876e8-b254-42b0-901b-65e8c7e3cfe3)

Here is my system config:

- GPU: A100
- NVIDIA-SMI 535.161.08             Driver Version: 535.161.08   CUDA Version: 12.5
- N

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support EBNF in xgrammar

**Link**: https://github.com/sgl-project/sglang/issues/2376
**State**: closed
**Created**: 2024-12-06T12:07:00+00:00
**Closed**: 2025-05-26T00:02:55+00:00
**Comments**: 2
**Labels**: good first issue, help wanted

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

xgrammar supports EBNF. We would like to integrate this feature into SGLang.

We can add a new parameter called `ebnf` in sampling_params.py and treat it similar to regex and JSON.


### Related resources

https://xgrammar.mlc.ai/docs/how_to/ebnf_guided_generation.html
https://github.com/sgl-project/sglang/blob/f5b2a3aa67efb10918965b9f3555ff24ef971902/python/sglang/srt/sampling/sampling_params.py#L36-L38
https://github.com/sgl-project/sglang/blob/main/test/srt/test_json_constrained.py

---

## Issue #N/A: [Bug] update_weights_from_tensor raise EOFError when TP>1

**Link**: https://github.com/sgl-project/sglang/issues/3726
**State**: closed
**Created**: 2025-02-20T07:57:02+00:00
**Closed**: 2025-02-24T17:12:54+00:00
**Comments**: 8
**Labels**: RLHF

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

 An EOFError error was raised when using `update_weights_from_tensor` at TP>4, it seens the data deserialize before the full data received.

Python error trace info:
```
Traceback (most recent call last):                                                                                                                        
  File "/usr/lib

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] run chatglm3-6b report error

**Link**: https://github.com/sgl-project/sglang/issues/735
**State**: closed
**Created**: 2024-07-26T06:12:47+00:00
**Closed**: 2024-07-27T09:44:47+00:00
**Comments**: 3

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

WARNING 07-26 14:06:20 interfaces.py:131] The model (<class 'sglang.srt.models.chatglm.ChatGLMForCausalLM'>) contains all LoRA-specific attributes, but does not set `supports_lora=True`.
Loading pt checkpoint shards:   0% Completed | 0/7 [00:00<?, ?it/s]
Loading pt checkpoint shards:  14% Completed | 1/7 [00:00<00:04,  1.28it/s]
Loading pt checkpoint shards:  29% Completed | 2/7 [00:01<00:04,  1.01it/s]
Loading pt checkpoint shards:  43% Completed | 3/7 [00:03<00:04,  1.05s/it]
Loading pt checkpoint shards:  57% Completed | 4/7 [00:04<00:03,  1.06s/it]
Loading pt check

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Server crashes after loading (Mixtral 8x7b) on L4

**Link**: https://github.com/sgl-project/sglang/issues/1191
**State**: closed
**Created**: 2024-08-23T11:08:48+00:00
**Closed**: 2024-11-04T01:13:36+00:00
**Comments**: 10
**Labels**: bug, inactive

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

Model fully loads, server runs and then instantly crashes

```
server_args=ServerArgs(model_path='/local_disk0/mistralai/Mixtral-8x7B-Instruct-v0.1', tokenizer_path='/local_disk0/mistralai/Mixtral-8x7B-Instruct-v0.1', tokenizer_mode='auto', skip_tokenizer_init=False, load_format='auto', dtype='auto', trust_remote_code=False, context_len

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] [CI regression] TestVILAServer.test_video_chat_completion

**Link**: https://github.com/sgl-project/sglang/issues/7587
**State**: closed
**Created**: 2025-06-27T05:51:45+00:00
**Closed**: 2025-06-28T04:13:46+00:00
**Comments**: 3

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

CI TestVILAServer.test_video_chat_completion has been broken since the past few days (if not longer). Creating an issue for tracking.

Sample run: https://github.com/sgl-project/sglang/actions/runs/15916204833/job/44895747345 

### Reproduction

N/A

### Environment

N/A

---

## Issue #N/A: [Feature] Add return hidden state in the native API

**Link**: https://github.com/sgl-project/sglang/issues/3461
**State**: closed
**Created**: 2025-02-10T06:26:45+00:00
**Closed**: 2025-02-27T06:06:55+00:00
**Comments**: 4
**Labels**: good first issue, help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

JM is submitting a feature to get a hidden state. We can add examples at the beginning of the test file `test/srt/test_hidden_states.py` right now. Later rewrite this API and add it in the docs.

Try to add a native API instead of adding a parameter and relaunching the engine.

If anyone is interested in this, could reach out to me and try to get in touch.

<img width="635" alt="Image" src="https://github.com/user-attachments/assets/32d66df2-a86b-408f-a02f-b2cb289e012e" />

### Related resources

https://github.com/sgl-project/sglang/pull/3364

---

## Issue #N/A: [Bug]  [0.4.5.post3] accuracy loss when both --speculative-algorithm NEXTN and  Shared experts fusion optimization are enabled.

**Link**: https://github.com/sgl-project/sglang/issues/5702
**State**: closed
**Created**: 2025-04-24T07:09:23+00:00
**Closed**: 2025-04-24T17:21:24+00:00
**Comments**: 3
**Labels**: bug, high priority

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

accuracy loss when both --speculative-algorithm NEXTN and  Shared experts fusion optimization are enabled.

### Reproduction

sglang                            0.4.5.post3

server start cmd:
python3 -m sglang.launch_server --model-path $deepseek_R1_MODEL_PATH --tp 8 - --disable-radix-cache --mem-fraction-static 0.85 --attention-backend fla

[... truncated for brevity ...]

---

## Issue #N/A: Do you support frontend-language inference for Llava-OneVision ?

**Link**: https://github.com/sgl-project/sglang/issues/1302
**State**: closed
**Created**: 2024-09-02T12:39:44+00:00
**Closed**: 2024-09-02T12:42:04+00:00
**Comments**: 0

### Description

No description provided.

---

## Issue #N/A: [Bug] DeepSeek R1 on the latest main branch sglang has some output issue

**Link**: https://github.com/sgl-project/sglang/issues/7599
**State**: open
**Created**: 2025-06-27T11:02:22+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Running DeepSeeK-R1-0528 with the latest main branch sglang will cause the first token in ReasonContent is None and Content is '', then the final usage is not in the final stop response but the following response with the 'choices': [], Why?
```
[2025-06-27 18:40:33 pd_handlers.py:423 INFO] Received chunk data: {'id': '99b4b631c29845af9d23

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Unexpected Single-Token Prefill Behavior in 8-bit Quantized Model under Single-GPU Testing

**Link**: https://github.com/sgl-project/sglang/issues/7138
**State**: open
**Created**: 2025-06-13T01:29:43+00:00
**Comments**: 0

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

When I tried to test the dpsk_r1_w8a8 model with single-GPU pruning, I noticed that the profiler captured during the prefill phase only showed an extension of one token length. The total time taken for a complete prefill was as long as a single decode step, which is unexpected. Please help investigate the cause of this issue. Here are my s

[... truncated for brevity ...]

---

## Issue #N/A: Refactor the openai speculative execution module in interpreter

**Link**: https://github.com/sgl-project/sglang/issues/452
**State**: closed
**Created**: 2024-05-19T05:20:52+00:00
**Closed**: 2024-05-21T15:16:49+00:00
**Comments**: 0

### Description

Mainly, make the `_execute_gen` simpler by moving out the speculative execution part as a new function.
https://github.com/sgl-project/sglang/blob/5b647543c141a6b21307f3fbc679d2a0a9231c41/python/sglang/lang/interpreter.py#L424

---

## Issue #N/A: [Bug] Using MLA with Lk >= 576 report out of resource: shared memory ERROR

**Link**: https://github.com/sgl-project/sglang/issues/2847
**State**: closed
**Created**: 2025-01-13T02:04:45+00:00
**Closed**: 2025-03-23T00:19:20+00:00
**Comments**: 2
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

We are trying to using SGLang for our new trained model, using MLA as the attention inspired by Deepseek-v2-lite, and MiniCPM3. Our model is very small, seems no reason to trigger the memory issues on A10, but still got 
"triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 106496, Hardware limit: 101376. Reduci

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]DeepSeek-R1 Process hangs after NCCL initialization in multi-server distributed inference setup

**Link**: https://github.com/sgl-project/sglang/issues/3516
**State**: closed
**Created**: 2025-02-12T08:37:39+00:00
**Closed**: 2025-02-19T12:42:30+00:00
**Comments**: 13
**Labels**: deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

### Environment and Setup
I am trying to run the DeepSeek-R1 671B model on three servers, each equipped with 8 A800 GPUs
I have 3 servers, each with 8 * A800 GPUs. I'm trying to create 5 nodes across these three servers using Docker overlay network for distributed inference, with each node using 4 GPUs. I've confirmed that all nodes can pi

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] This modeling file requires the following packages that were not found in your environment: datamodel_code_generator. Run `pip install datamodel_code_generator`

**Link**: https://github.com/sgl-project/sglang/issues/1398
**State**: closed
**Created**: 2024-09-12T03:28:35+00:00
**Closed**: 2024-09-22T11:49:36+00:00
**Comments**: 1

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

When attempting to run openBmb/MiniCPM 3.0 using Docker, I encountered an issue related to missing dependencies. The error log indicates that certain dependencies required for the project are not installed. Could you please provide guidance on how to resolve this issue, or update the Docker environment to include the necessary de

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support EAGLE-3 for speculative decoding on DeepSeek model

**Link**: https://github.com/sgl-project/sglang/issues/6268
**State**: open
**Created**: 2025-05-13T12:41:36+00:00
**Comments**: 7
**Labels**: speculative-decoding

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

EAGLE-3 appears to provide a higher speculative decoding acceptance rate to improve output throughput. 
For specific code generation scenarios, we found that focusing on multi-step losses when training the draft model can effectively improve the acceptance rate.

### Related resources

_No response_

---

## Issue #N/A: [Bug] AssertionError when launch server with pipeline model parallism

**Link**: https://github.com/sgl-project/sglang/issues/6831
**State**: open
**Created**: 2025-06-03T08:35:23+00:00
**Comments**: 0

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

command
`python3 -m sglang.launch_server --model  /models/Meta-Llama-3.1-8B-Instruct --pp-size 4 --port 8000`
error info
`[2025-06-03 08:20:40 PP0] Scheduler hit an exception: Traceback (most recent call last):
  File "/usr/local/python3/lib/python3.12/site-packages/sglang/srt/managers/scheduler.py", line 2269, in run_scheduler_process
   

[... truncated for brevity ...]

---

## Issue #N/A: multimodal can not use the choices?

**Link**: https://github.com/sgl-project/sglang/issues/1971
**State**: closed
**Created**: 2024-11-09T06:02:18+00:00
**Closed**: 2024-11-10T16:11:40+00:00
**Comments**: 1

### Description

qwen2-vl can not  use sgl.gen("answer", choices=["yes", "no"]))?
![image](https://github.com/user-attachments/assets/baadede2-414c-439b-aa6c-571e6127cec0)


---

## Issue #N/A: [Bug] Sg-kernel and sglang build from source

**Link**: https://github.com/sgl-project/sglang/issues/3970
**State**: closed
**Created**: 2025-02-28T20:54:26+00:00
**Closed**: 2025-05-01T00:21:12+00:00
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

For aarch64 it generates wheels but != platform
```bash
creating build/bdist.linux-aarch64/wheel/sgl_kernel-0.0.3.post6.dist-info/WHEEL
creating '/opt/sglang/sgl-kernel/wheels/sgl_kernel-0.0.3.post6-cp39-abi3-manylinux2014_x86_64.whl' and adding 'build/bdist.linux-aarch64/wheel' to it

```

### Reproduction

```bash
#!/usr/bin/env bash
set

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] google/gemma-3 fails to launch when attention_backend is torch_native

**Link**: https://github.com/sgl-project/sglang/issues/6044
**State**: closed
**Created**: 2025-05-06T06:32:32+00:00
**Closed**: 2025-07-06T00:22:07+00:00
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

I was trying to launch google/gemma-3-1b-it with torch_native as attention backend. However, it failed and the error message is:

```
[2025-05-06 06:19:33] TpModelWorkerClient hit an exception: Traceback (most recent call last):
  File "/home/chenlixiang/sglang/python/sglang/srt/managers/tp_worker_overlap_thread.py", line 118, in forward_t

[... truncated for brevity ...]

---

## Issue #N/A: How to quantify the Qwen2.5-VL-3B model?

**Link**: https://github.com/sgl-project/sglang/issues/4619
**State**: closed
**Created**: 2025-03-20T08:39:30+00:00
**Closed**: 2025-05-20T00:19:49+00:00
**Comments**: 3
**Labels**: inactive

### Description

Hi，big guys!
Recently I wanted to quantize a Qwen2.5-VL-3B model and deploy it locally, I tried to use sglang (https://docs.sglang.ai/backend/quantization.html) to quantize the model but it failed with the following error:

![Image](https://github.com/user-attachments/assets/89688b70-de2e-403a-a046-9c0be8eb7337)

sglang seems to be able to quantize chat models (Qwen2.5-3B) only? I was able to successfully quantize the Qwen2.5-3B model
I would like to ask you guys if there is any way to quantize the Qwen2.5-VL-3B model? (Is there any way to quantify a similar non-chat model like Qwen-2.5-vl? (Models like ASR))

---

## Issue #N/A: [Bug] Llama-4-Scout OOM with image requests

**Link**: https://github.com/sgl-project/sglang/issues/6933
**State**: open
**Created**: 2025-06-06T22:15:48+00:00
**Comments**: 6
**Labels**: bug, high priority

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Llama-4-Scout-17B-16E-Instruct would raise CUDA OOM error during our image benchmark.

### Reproduction

Server start command:
```
python3 -m sglang.launch_server --model meta-llama/Llama-4-Scout-17B-16E-Instruct --tp-size=4 --host=0.0.0.0 --mem-fraction-static=0.95 --context-length=196608 --enable-multimodal --tool-call-parser=pythonic --

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] A100 PCIE torch compile error

**Link**: https://github.com/sgl-project/sglang/issues/1301
**State**: closed
**Created**: 2024-09-02T11:24:36+00:00
**Closed**: 2024-09-02T23:18:49+00:00
**Comments**: 3
**Labels**: bug

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

```
[11:19:46 TP0] Decode batch. #running-req: 36, #token: 14473, token usage: 0.03, gen throughput (token/s): 2283.73, #queue-req: 0
../aten/src/ATen/native/cuda/MultinomialKernel.cu:112: binarySearchForMultinomial: block: [0,31,0], thread: [0,0,0] Assertion `cumdist[size - 1] > static_cast<scalar_t>(0)` failed.
../aten/src/ATen/native

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Step-by-Step Guide to Use SGLang on NVIDIA Jetson Orin platform

**Link**: https://github.com/sgl-project/sglang/issues/3182
**State**: closed
**Created**: 2025-01-27T16:45:59+00:00
**Closed**: 2025-02-21T12:45:13+00:00
**Comments**: 9
**Labels**: documentation, good first issue

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Hello Sglang team,

Great inference engine! 

Just FYI, I was able to successfully run SGLang on the NVIDIA Jetson AGX Orin Developer Kit. 

For more details, please check here: https://github.com/shahizat/SGLang-Jetson



### Related resources

_No response_

---

## Issue #N/A: [Feature] How does sglang perform on diffusion models

**Link**: https://github.com/sgl-project/sglang/issues/738
**State**: closed
**Created**: 2024-07-26T09:19:59+00:00
**Closed**: 2024-07-26T18:04:44+00:00
**Comments**: 2

### Description

### Motivation

I wonder if supporting sglang is helpful for the diffusion model, such as stable diffusion.

### Related resources

_No response_

---

