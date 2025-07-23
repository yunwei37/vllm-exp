# author_association_NONE - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 6
- Closed Issues: 24

### Label Distribution

- inactive: 13 issues
- help wanted: 2 issues
- blackwell: 1 issues
- research: 1 issues
- deepseek: 1 issues

---

## Issue #N/A: [Bug] An error occurs when handling requests when  deploying a model with PD disaggregation

**Link**: https://github.com/sgl-project/sglang/issues/6691
**State**: open
**Created**: 2025-05-28T06:50:00+00:00
**Comments**: 11

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

the prefill error:
```
[2025-05-28 06:26:01] The server is fired up and ready to roll!
[2025-05-28 06:29:57] INFO:     127.0.0.1:46780 - "GET /v1/models HTTP/1.1" 200 OK
[2025-05-28 06:30:04] INFO:     127.0.0.1:46782 - "GET /v1/models HTTP/1.1" 200 OK
[2025-05-28 06:30:10] INFO:     127.0.0.1:42176 - "GET /v1/models HTTP/1.1" 200 OK
[2025

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] request timeout with multi-gpu model

**Link**: https://github.com/sgl-project/sglang/issues/3358
**State**: closed
**Created**: 2025-02-07T03:01:34+00:00
**Closed**: 2025-02-07T03:42:38+00:00
**Comments**: 4

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

i launch a qwen2.5-32B-instruct model on two L20 gpus:

`docker run -it --rm --runtime=nvidia --gpus '"device=0,1"' --ipc=host --network=host --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 -v /nfs/hf_models:/models --env NCCL_P2P_DISABLE=1 lmsysorg/sglang:v0.4.2.post2-cu124-srt python3 -m sglang.launch_server --model-path /mode

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] torch.OutOfMemoryError: CUDA out of memory for 16GB Vram while trying to inference gemma-3-12b-IT

**Link**: https://github.com/sgl-project/sglang/issues/5576
**State**: closed
**Created**: 2025-04-20T14:11:09+00:00
**Closed**: 2025-06-21T00:19:44+00:00
**Comments**: 11
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Trying to run gemma-3-12b-It on my system that has 16 GB vram. I tried with multiple context-length. With context length 7k, it doesn't work. 

I used a docker image to run the command. 

output:

```
podman run --gpus all     -p 30000:30000     -v ~/.cache/huggingface:/root/.cache/huggingface     --env "HF_TOKEN=*************************"

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Seeing random output with nvidia/Llama-3.1-Nemotron-70B-Reward

**Link**: https://github.com/sgl-project/sglang/issues/1931
**State**: closed
**Created**: 2024-11-05T23:03:46+00:00
**Closed**: 2024-11-14T18:50:43+00:00
**Comments**: 1

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

I am trying to run the basic setup example with the nvidia/Llama-3.1-Nemotron-70B-Reward on a machine with 8 A6000s, but the observed output is random.

Generated output:
```
{"text":"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"","meta_info":{"prompt_tokens":6,"completion_tokens":16,"completion_tokens_wo_jump_forward":16,"cached_tokens":1,"finish_

[... truncated for brevity ...]

---

## Issue #N/A: What is the relationship between ModelRunner and the model(deepseek.py,llama.py..etc)?

**Link**: https://github.com/sgl-project/sglang/issues/5453
**State**: closed
**Created**: 2025-04-16T09:14:55+00:00
**Closed**: 2025-06-16T00:20:41+00:00
**Comments**: 1
**Labels**: inactive

### Description

Who can help me answer this question?  I haven't found the calling relationship.

---

## Issue #N/A: logprobs for each token when using local server

**Link**: https://github.com/sgl-project/sglang/issues/232
**State**: closed
**Created**: 2024-02-25T19:45:07+00:00
**Closed**: 2024-07-09T07:35:59+00:00
**Comments**: 1

### Description

I have a local server running and setup my script with the text_qa.run_batch example.

Is there a way to get the logprobs for each token like in the OpenAI API?

---

## Issue #N/A: [Bug] flashinfer_python with minimum required version 0.2.5 is not installed

**Link**: https://github.com/sgl-project/sglang/issues/6160
**State**: closed
**Created**: 2025-05-09T17:19:44+00:00
**Closed**: 2025-06-11T15:25:36+00:00
**Comments**: 6
**Labels**: blackwell

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I am trying to serve gemma3 27b-it on RTX 5090 using sglang blackwell image. However, I'm getting this error:
```bash
Traceback (most recent call last):
  File "/opt/conda/lib/python3.11/importlib/metadata/__init__.py", line 563, in from_name
    return next(cls.discover(name=name))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
StopIteration

D

[... truncated for brevity ...]

---

## Issue #N/A: Contradictory suggestions: Not enough memory. Please try to increase --mem-fraction-static

**Link**: https://github.com/sgl-project/sglang/issues/322
**State**: closed
**Created**: 2024-03-22T12:23:46+00:00
**Closed**: 2024-07-25T06:33:37+00:00
**Comments**: 5
**Labels**: inactive

### Description

**Q: Should I increase or decrease `--mem-fraction-static`?** (and what is the minimum and maximum value allowed?)

Looking in the source code (`python/sglang/srt/managers/router/model_runner.py`) I would believe that increasing the value would alleviate the memory requirements but I might be interpreting it wrong. Just wanted to inform that there is a mismatch between the advice given in documentation and the advice given in the actual code.

**Description of the problem:**

I am trying to launch Mistral-7B-Instruct-v0.2 (using sglang==0.1.13):

`python -m sglang.launch_server --model-path /llm_path/hf_model_mistral_7B_Instruct_v0_2 --port 30000`

but I have memory issues. At the end it is suggested to increase `--mem-fraction-static`. 

However, in the documentation (https://github.com/sgl-project/sglang) the opposite advice is given:

> If you see out-of-memory errors during serving, please try to reduce the memory usage of the KV cache pool by setting a smaller value 

[... truncated for brevity ...]

---

## Issue #N/A: [bug] logits processor race condition in overlap mode

**Link**: https://github.com/sgl-project/sglang/issues/8056
**State**: open
**Created**: 2025-07-15T09:14:14+00:00
**Comments**: 1

### Description

[One](https://github.com/sgl-project/sglang/blob/4a8837950abb7a39d5b890b8f4ee21bd9ded959d/test/srt/test_srt_endpoint.py#L452) of the logit processor tests is disabled because we have race condition on output_ids variable.
```python
        """
        NOTE: This feature has a race condition bug.
        This line https://github.com/sgl-project/sglang/blob/ef8ec07b2ce4c70c2a33ec5acda4ce529bc3cda4/test/srt/test_srt_endpoint.py#L395-L396 can be accessed by two concurrent threads at the same time. The access order is not guaranteed.
        In sglang, we use two python threads to overlap the GPU computation and CPU scheduling.
        Thread 1 (the CPU scheduling thread) will update the `param_dict["__req__"].output_ids`.
        Thread 2 (the GPU computation thread) will call `DeterministicStatefulLogitProcessor` because sampling is considered as GPU computation.
        We can fix this by moving the call of DeterministicStatefulLogitProcessor to the CPU scheduling thread.
        """
```

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] QwQ-32B The backend reported an error

**Link**: https://github.com/sgl-project/sglang/issues/4251
**State**: closed
**Created**: 2025-03-10T05:30:12+00:00
**Closed**: 2025-05-10T00:18:05+00:00
**Comments**: 2
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

![Image](https://github.com/user-attachments/assets/874ba0e8-d0fa-40b1-9ab8-4e5a1fda7497)

![Image](https://github.com/user-attachments/assets/e172dee8-684b-48ad-a087-04195653c463)

The sglang backend reported an error and the answer was not completed yet.

The same model has no problem with ollama and vllm

### Reproduction

python -m sgl

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] sanic Custom Server example  support openai  stream  api ?

**Link**: https://github.com/sgl-project/sglang/issues/1655
**State**: closed
**Created**: 2024-10-13T05:15:56+00:00
**Closed**: 2024-10-13T16:17:17+00:00
**Comments**: 3

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Custom Server example  surport openai  stream  api ?

### Related resources

Custom Server example  surport openai  stream  api ?

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

## Issue #N/A: Cuda graph supported bs in DP attention

**Link**: https://github.com/sgl-project/sglang/issues/5527
**State**: open
**Created**: 2025-04-18T10:10:19+00:00
**Comments**: 7

### Description

I have been testing and recording the output throughput of SGLang on 2*8 H100 GPUs, and I've observed a significant regression in output throughput for long outputs in the `enable-dp-attention` scenarios following this [PR](https://github.com/sgl-project/sglang/pull/4390). Through debugging and profiling with Nsight Systems, I confirmed that the performance degradation is caused by the CUDA graph not being properly launched.

See the [code](https://github.com/sgl-project/sglang/blob/00391f58a3c2b13b00dc0a0a983a5c7fab399883/python/sglang/srt/model_executor/cuda_graph_runner.py#L280)
```
if self.enable_dp_attention:
    total_global_tokens = sum(forward_batch.global_num_tokens_cpu)

    is_bs_supported = forward_batch.can_run_dp_cuda_graph and (
        total_global_tokens in self.graphs
        if self.disable_padding
        else total_global_tokens <= self.max_bs
    )
```
After `enable-dp-attention`, `total_global_tokens` equals the sum of tokens across all DP ranks. For example, dur

[... truncated for brevity ...]

---

## Issue #N/A: LLaVA-v1.6 RuntimeError in llava image encoding

**Link**: https://github.com/sgl-project/sglang/issues/409
**State**: closed
**Created**: 2024-05-04T16:02:03+00:00
**Closed**: 2024-07-25T06:33:29+00:00
**Comments**: 1
**Labels**: inactive

### Description

There still seems to be a bug in the newer LLaVA-v1.6 version where, for some images, the model only generates one or two tokens. The problem seems to be related to some kind of attributes of the images themselves, as changing the textual input has no influence. Furthermore, all 3 v1.6 models (7b, 13b, and 34b) have problems with the same images. Moreover, the 1.5 version works perfectly fine with the same inputs. This bug appears for around 5% of my images.
I'm on the sglang 0.1.14 and vllm 0.3.3. The issue seems to be related to #273, however i do not use regex for generation. The server casts the following runtime error when llava is not able to process the image:
`
RuntimeError in llava image encoding: The expanded size of the tensor (0) must match the existing size (2438) at non-singleton dimension 0.  Target sizes: [0, 4096].  Tensor sizes: [2438, 4096]
torch.Size([10194, 4096])
0 -1`

---

## Issue #N/A: [Bug] TypeError: moe_fused_gate() takes 5 positional arguments but 7 were given

**Link**: https://github.com/sgl-project/sglang/issues/5612
**State**: closed
**Created**: 2025-04-22T02:47:03+00:00
**Closed**: 2025-04-28T02:36:48+00:00
**Comments**: 3

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

When I run the following script on the latest code of the dual node H20, the following error appears

`python3 -m sglang.launch_server --model-path /media/nvme/deepseek/DeepSeek-V3-0324 --tensor-parallel-size 16 --trust-remote-code --dist-init-addr ****  --enable-metrics --port 50000 --nnodes 2 --node-rank 0`

[2025-04-22 10:37:30 DP0 TP0]

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] SGLang Tool Calling for Qwen2.5 models returns empty ChatCompletionMessage content

**Link**: https://github.com/sgl-project/sglang/issues/3797
**State**: closed
**Created**: 2025-02-24T01:41:21+00:00
**Closed**: 2025-05-06T00:18:59+00:00
**Comments**: 9
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

So the story is I deployed a Qwen2.5-72B-Instruct model with SGLang on 8 cards, been testing the tool calling agents with LangGraph. The agents do not output any content when it decides to call a tool, therefore it just decides to randomly call tools until it suddenly stops. The behavior has led to very poor agentic performance.

On vLLM s

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] 0.4.6-post4-cu124， run deepseekv30324 error

**Link**: https://github.com/sgl-project/sglang/issues/6418
**State**: open
**Created**: 2025-05-19T08:04:57+00:00
**Comments**: 4

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

sglang:v0.4.4 is OK, but v0.4.6 is not.

dockerimages:   lmsysorg/sglang:v0.4.6.post4-cu124
docker run --name sglang -v /export:/export -d --gpus all --shm-size 128g -p 80:80 --privileged --entrypoint "sleep" 65be730e0d41 infinity

sglang:
python3 -m sglang.launch_server --quantization fp8 --kv-cache-dtype fp8_e5m2 --model /export/model/de

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] In prefill phase the number of sequences is low

**Link**: https://github.com/sgl-project/sglang/issues/6901
**State**: open
**Created**: 2025-06-05T15:13:36+00:00
**Comments**: 1

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I want to decrease the ttft on my server. I have set the context-length to 163840 which should be there anyway because it's in the `config.json` of deepseek v3. 
I also set the chunked-prefill-size to 32768.
For testing I'm sending 1024 request each has 5000 tokens, and the prefill never goes above 15000
```
sglang  | [2025-06-05 15:08:56 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] AssertionError: res=<Response [503]> Process was always killed automactically

**Link**: https://github.com/sgl-project/sglang/issues/4094
**State**: closed
**Created**: 2025-03-05T11:35:03+00:00
**Closed**: 2025-05-25T00:21:16+00:00
**Comments**: 5
**Labels**: inactive

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

I successfully ran the serve  using command after installing SGLang. But after about 2 minutes, the process is always killed automatically.

### Reproduction

I depoly Qwen2.5-7B-Instruct using the below command

```
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server --model-path /data/models/Qwen2.5/Qwen2.5-7B-Instruct --api-key base -

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Add Model Hooks for Accessing and Customizing Model Activations

**Link**: https://github.com/sgl-project/sglang/issues/3266
**State**: closed
**Created**: 2025-02-03T05:44:46+00:00
**Closed**: 2025-04-05T00:17:32+00:00
**Comments**: 4
**Labels**: inactive, research

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

## Description
It would be beneficial to introduce model hooks that allow users to access and modify model activations. This feature would enable greater flexibility for tasks such as visualization, debugging, and custom processing of intermediate representations.

## Use case
* Extract intermediate outputs for interpretability analysis, such as [LogitLens-style investigations](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens).
* Expose internal activations, enabling users to cache activations and implement functions to edit, remove, or replace them dynamically during inference, for example [representation engineering](https://github.com/andyzoujm/representation-engineering).

While

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Enabling Speculative Decoding causes DeepSeek R1 generate no output if the input text is long

**Link**: https://github.com/sgl-project/sglang/issues/5734
**State**: closed
**Created**: 2025-04-25T05:17:05+00:00
**Closed**: 2025-04-25T10:00:43+00:00
**Comments**: 2

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When:
- 1. The Speculative Decoding is enabled, for exmaple:
```bash
    --log-requests \
    --log-requests-level 2 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-draft /public/home/deepseek/tests/cuda_llm/deepseek_r1/DeepSeek-

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] how to combine with ray.data

**Link**: https://github.com/sgl-project/sglang/issues/1987
**State**: closed
**Created**: 2024-11-10T15:08:00+00:00
**Closed**: 2025-01-15T11:29:34+00:00
**Comments**: 6
**Labels**: help wanted

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

ray.data.Dataset.map_batches(**args) is ok with vllm, i replace vllm.LLm with sglang.Engine, it's wrong, how to make it? Because ray.data is easy to manage data. Looking forward to you can help me

Running 0: 0 bundle [00:00, ? bundle/s]2024-11-10 23:01:22,492  ERROR streaming_executor_state.py:456 -- An exception was raised from a task 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] deepseek v3 2 nodes h100 segmentation fault

**Link**: https://github.com/sgl-project/sglang/issues/3283
**State**: closed
**Created**: 2025-02-04T06:43:27+00:00
**Closed**: 2025-02-04T07:40:25+00:00
**Comments**: 6
**Labels**: help wanted, deepseek

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

hello.
I run on 2 nodes of 8 x h100 using   lmsysorg/sglang:v0.4.2.post1-cu125 image

```
python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1 --tp 16 --dist-init-addr 172.16.1.68:5000 --nnodes 2 --node-rank 1 --trust-remote-code --quantization fp8 --kv-cache-dtype fp8_e5m2
```
I start a benchmark
```
 python3 -m sglang.ben

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] nvcc fatal   : Unknown option '-generate-dependencies-with-compile'

**Link**: https://github.com/sgl-project/sglang/issues/4120
**State**: closed
**Created**: 2025-03-06T03:19:09+00:00
**Closed**: 2025-05-06T00:18:56+00:00
**Comments**: 1
**Labels**: inactive

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

when use ”python3 -m sglang.launch_server --model /home/ydkj/lx/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --host 0.0.0.0 --port 8123 --max-total-tokens 8192 --tensor-parallel-size 2 “ Startup Command.  An error occurred, as follows:

nvcc fatal   : Unknown option '-generate-dependencies-with-compile'
ninja: build stopped: subcommand failed.

### Related resources

_No response_

---

## Issue #N/A: [Bug] when dp=2  on two nodes,  cannot get   content from  /metrics   on another node

**Link**: https://github.com/sgl-project/sglang/issues/7305
**State**: open
**Created**: 2025-06-18T07:50:49+00:00
**Comments**: 2

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

Using sglang to deploy a large model with tp=16, dp=2 across two servers, so two requests will run on dp0-7 and dp8-15 respectively.

Then, executing curl -X GET http://0.0.0.0:40000/metrics -H "Authorization: Bearer  xxxx" on servers A and B:

Server A (master node) has metrics, but Server B does not.
Both A and B have enable_metrics in t

[... truncated for brevity ...]

---

## Issue #N/A: Dependency conflict with LLaVA

**Link**: https://github.com/sgl-project/sglang/issues/464
**State**: closed
**Created**: 2024-05-23T07:43:08+00:00
**Closed**: 2024-07-26T01:02:23+00:00
**Comments**: 1
**Labels**: inactive

### Description

# Issue
Cannot run a finetuned LLaVA model with sglang==0.1.16, running `CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path llava-lora-34b-faceshape-ft/ --tokenizer-path liuhaotian/llava-v1.6-34b-tokenizer --port 30000 --tp 4`
throws: 
FileNotFoundError: [Errno 2] No such file or directory: '/home/iverkh/.triton/cache/a95dd9872513f57ade076cce4b51d3f0/_fwd_kernel_stage2.json.tmp.pid_33358_98246'

# Reproduction
After cloning LLaVA from https://github.com/haotian-liu/LLaVA, run `pip install -e . ` as instructed in the README
Run pip install sglang[all]
Run `CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path <fine tuned llava> --tokenizer-path liuhaotian/llava-v1.6-34b-tokenizer --port 30000 --tp 4`

# Possible cause
sglang has a dependency of vllm>0.4.2 which requires torch==2.3.0. LLaVA's official repository has a dependency of torch==2.1.2,
see here: https://github.com/haotian-liu/LLaVA/blob/main/pyproject.toml

After downgrad

[... truncated for brevity ...]

---

## Issue #N/A: next-N does not work on H20

**Link**: https://github.com/sgl-project/sglang/issues/5513
**State**: closed
**Created**: 2025-04-18T01:28:05+00:00
**Closed**: 2025-04-18T08:09:01+00:00
**Comments**: 4

### Description

I tried different next-N paras but inference speed(one batch) on one H20 server was always about 35tokens/s with or without next-N.

sglang script:
python -m sglang.launch_server --served-model-name=ds-r1-671b --model-path=$models/DeepSeek-R1 \
--enable-p2p-check --reasoning-parser=deepseek-r1 --trust-remote-code --host=0.0.0.0 --port=38001 \
--mem-fraction-static=0.9 \
--tp=8 --max-total-tokens=65000 --max-running-requests=64 \
--enable-ep-moe \
--disable-radix-cache \
--speculative-algorithm EAGLE3 \
--speculative-draft-model-path $moddels/DeepSeek-R1-NextN \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4

check-env:
Python: 3.12.9 | packaged by Anaconda, Inc. | (main, Feb  6 2025, 18:56:27) [GCC 11.2.0]
CUDA available: True
GPU 0,1,2,3,4,5,6,7: NVIDIA H20
GPU 0,1,2,3,4,5,6,7 Compute Capability: 9.0
CUDA_HOME: /usr/local/cuda-12.8
NVCC: Cuda compilation tools, release 12.8, V12.8.93
CUDA Driver Version: 570.124.06
PyTorch: 2.5.1+cu124
sglang

[... truncated for brevity ...]

---

## Issue #N/A: The performance stress test results of EAGLE-3 are not good

**Link**: https://github.com/sgl-project/sglang/issues/5274
**State**: closed
**Created**: 2025-04-11T05:29:08+00:00
**Closed**: 2025-04-17T09:37:35+00:00
**Comments**: 5

### Description

I used EvalScope to stress test the inference performance of SGLang. A very positive aspect is that SGLang outperforms vLLM. However, when I tried to enable the EAGLE-3 optimization, the throughput dropped sharply.

My environment is an NVIDIA A6000, and the stress test command is as follows:
```
docker run --rm --net=host \
--mount type=bind,source=/data,target=/data \
registry.bingosoft.net/bingomatrix/modelscope:ubuntu22.04-cuda12.1.0-py310-torch2.4.0-tf2.16.1-1.18.1-vllm-evalscope \
evalscope perf \
  --parallel 20 \
  --model /data/models/Meta-Llama-3.1-8B-Instruct \
  --url http://127.0.0.1:30000/v1/chat/completions \
  --api openai \
  --dataset random \
  --min-tokens 128 \
  --max-tokens 128 \
  --prefix-length 64 \
  --min-prompt-length 1024 \
  --max-prompt-length 2048 \
  --number 100 \
  --tokenizer-path Qwen/Qwen2.5-0.5B-Instruct \
  --debug
```

Start SGLang without EAGLE-3:
```
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v /data/:/data \
    -

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Cannot run official test case for precision evaluation with hf engine

**Link**: https://github.com/sgl-project/sglang/issues/5606
**State**: closed
**Created**: 2025-04-21T17:08:06+00:00
**Closed**: 2025-06-21T00:19:42+00:00
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

Hi team! I tried to test the precision of sgl engine (and verlengine) with hf engine, but I cannot pass the assertations in `test/srt/test_verl_engine.py`. @ocss884  @zhaochenyang20 

```text
File "/root/sglang/python/sglang/test/runners.py", line 773, in check_close_model_outputs
    assert torch.all(abs(hf_logprobs - srt_logprobs) < pref

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] gemma-3-27b-it-bnb-4bit crash 

**Link**: https://github.com/sgl-project/sglang/issues/4897
**State**: closed
**Created**: 2025-03-29T19:01:15+00:00
**Closed**: 2025-06-13T00:19:52+00:00
**Comments**: 6
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug


1、When loading a 27B 4-bit quantized model, why does it exhaust the 24GB of gpu memory?
2、Why did the program crash? Is it because the gpu memory was exhausted?
<img width="1400" alt="Image" src="https://github.com/user-attachments/assets/48732e57-a966-4f92-951b-3fd637da3f1b" />
[2025-03-29 18:56:25 TP0] Scheduler hit an exception: Traceb

[... truncated for brevity ...]

---

