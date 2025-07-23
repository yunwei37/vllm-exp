# head_top5_labels - issues

**Total Issues**: 48
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 7
- Closed Issues: 41

### Label Distribution

- inactive: 17 issues
- good first issue: 17 issues
- help wanted: 16 issues
- high priority: 16 issues
- bug: 11 issues
- documentation: 4 issues
- collaboration: 3 issues
- MLLM: 3 issues
- performance: 2 issues
- lora: 1 issues

---

## Issue #N/A: how to use fp8 for inference on h20?

**Link**: https://github.com/sgl-project/sglang/issues/3568
**State**: closed
**Created**: 2025-02-14T05:59:47+00:00
**Closed**: 2025-04-16T00:18:29+00:00
**Comments**: 8
**Labels**: inactive

### Description

I have a problem, that is, the model I am currently deploying is: neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic.
The graphics card is h20.

I would like to ask how to infer the fp8 capability of this graphics card?
Currently, sglang is used for deployment. The deployment instructions are as follows:

python -m sglang.launch_server --model-path neuralmagic/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic --port 30000 --host 0.0.0.0 --tp 2 

What I want to know is that my command has enabled fp8 for inference operations? If not, can you tell me how to do it? Thanks

---

## Issue #N/A: [Feature] Add a hash for each new release

**Link**: https://github.com/sgl-project/sglang/issues/3923
**State**: closed
**Created**: 2025-02-27T08:39:19+00:00
**Closed**: 2025-04-30T00:18:49+00:00
**Comments**: 3
**Labels**: inactive

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

## Summary
Implement SHA-256 hash verification for all package releases to enhance security for users installing from mirror sites.

## Description
Users who download packages from mirror sites instead of the official PyPI repository need a reliable way to verify package integrity. Adding SHA-256 hashes for each release would provide a standard method to confirm packages haven't been tampered with or corrupted.

## Implementation
- Generate SHA-256 hashes automatically as part of the CI/CD pipeline
- Include hashes in package metadata files
- Make hashes accessible through the official website
- Update documentation to explain the verification process

## Benefits
- Enhanced security for users with limited access to official repos

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Qwen-gme embedding model: cannot get fused embedding from text+image, and image input format may be incorrect

**Link**: https://github.com/sgl-project/sglang/issues/5498
**State**: closed
**Created**: 2025-04-17T13:09:28+00:00
**Closed**: 2025-07-11T00:20:24+00:00
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

Thanks for the great work!

While using the /v1/embeddings endpoint with the gme-qwen2-vl model, I encountered two issues:

**1. Incorrect handling of image input**
According to the docs, the image input is passed like this:
payload = {
    "model": "gme-qwen2-vl",
    "input": [
        {"type": "text", "text": text_input},
        {"type

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] HuggingFace and SGLang inference don't match

**Link**: https://github.com/sgl-project/sglang/issues/2671
**State**: closed
**Created**: 2024-12-30T22:54:09+00:00
**Closed**: 2025-05-03T00:18:08+00:00
**Comments**: 9
**Labels**: bug, inactive, lora

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

The accuracy of the model is degraded due to inconsistent outputs from SGLang. While HF and vLLM produce consistent results such as "A" or "B," SGLang occasionally outputs responses like "I can't process that request." or "A." / "B." This inconsistency impacts overall accuracy.

### Reproduction

What command or script did yo

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Auto-truncation still uses full context length instead of (context_length - max_tokens)

**Link**: https://github.com/sgl-project/sglang/issues/5409
**State**: open
**Created**: 2025-04-15T07:33:29+00:00
**Comments**: 1
**Labels**: good first issue, inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I'm experiencing an issue where prompt auto-truncation doesn't properly account for max_tokens when using the HTTP server, even with allow_auto_truncate=True enabled. This persists after the changes in https://github.com/sgl-project/sglang/pull/4919.

### Reproduction

1.  python -m sglang.launch_server --model-path NousResearch/Hermes-3-L

[... truncated for brevity ...]

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

## Issue #N/A: [Feature] SGLang Support for TileLang

**Link**: https://github.com/sgl-project/sglang/issues/4221
**State**: closed
**Created**: 2025-03-09T05:34:49+00:00
**Closed**: 2025-05-27T00:18:53+00:00
**Comments**: 10
**Labels**: help wanted, high priority, inactive

### Description

We recently came across an interesting project: [TileLang](https://github.com/tile-ai/tilelang). It appears to offer significant advantages over Triton in many cases while maintaining a clean dataflow and simple syntax.

Do we have any plans to support a TileLang backend in SGLang?

For instance, TileLang has demonstrated up to **5x speedup** over Triton’s Flash MLA implementations on H100, with a kernel implementation of just **80 lines of code (see document:** https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_mla). Given these promising results, it would be valuable to explore its potential integration.

Would love to hear thoughts on this!


---

## Issue #N/A: [Feature] Allow arbitrary logit processors

**Link**: https://github.com/sgl-project/sglang/issues/1036
**State**: closed
**Created**: 2024-08-11T19:34:38+00:00
**Closed**: 2024-10-21T01:13:28+00:00
**Comments**: 2
**Labels**: inactive

### Description

### Motivation

There's some great projects out there that modify logits, mostly for guided decoding or novel sampling techniques. Supporting every single one of them will cause too much bloat and distraction, but if SGLang were to allow arbitrary logit processors then the community can plug and play their own processors.

For example, I would have interest in using [https://github.com/noamgat/lm-format-enforcer](lm format enforcer) because it allows for optional JSON fields and recursive classes (unlike outlines). The API of lm format enforcer is also clean and simple and it is simple to make custom parsers for other formats than JSON (e.g. SQL).

One way I would imagine the API to work is:

```python
def my_logits_processor(inputs: list[int], logits: torch.Tensor) -> torch.Tensor:
   ...


@sgl.function
def character_gen(s, name):
    s += name + " is a character in Harry Potter. Please fill in the following information about this character.\n"
    s += sgl.gen("outpu

[... truncated for brevity ...]

---

## Issue #N/A: aglang

**Link**: https://github.com/sgl-project/sglang/issues/299
**State**: closed
**Created**: 2024-03-14T09:10:13+00:00
**Closed**: 2024-07-25T06:32:43+00:00
**Comments**: 1
**Labels**: inactive

### Description

I test yi-vl-6B with `srt_example_yi_vl.py`
get error:
```
AttributeError: 'TokenizerManager' object has no attribute 'executor
```

---

## Issue #N/A: [Bug] Got error with awq_marlin quantization args.

**Link**: https://github.com/sgl-project/sglang/issues/1792
**State**: closed
**Created**: 2024-10-25T10:21:16+00:00
**Closed**: 2024-12-26T00:16:32+00:00
**Comments**: 2
**Labels**: inactive

### Description

### Checklist

- [x]  I have searched related issues but cannot get the expected help.

- [x]  The bug has not been fixed in the latest version.

- [x]  Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

- [x]  If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.

 

- [x] Please use English, otherwise it will be closed.

### Describe the bug

I used the AutoAWQ tool to quantize [Deepseek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2) model . The quantization script is as follows, resulting in a quantized network. I expect to obtain a model in awq_marlin quantization format.
```
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


mo

[... truncated for brevity ...]

---

## Issue #N/A: [RFC] Bi-weekly release

**Link**: https://github.com/sgl-project/sglang/issues/7332
**State**: open
**Created**: 2025-06-18T23:17:05+00:00
**Comments**: 0
**Labels**: high priority, collaboration

### Description

After thorough internal discussions, the SGLang team has decided to standardize the release cycle as follows:

- A new version will be released every two weeks under normal circumstances (e.g., v0.4.8, v0.4.9).

- If urgent issues or high-priority features arise between regular releases, we may publish a patch release or an additional stable version as needed.

- Bi-weekly releases will typically occur around the middle and end of each month.

- Each release will aim to include a set of planned features, usually discussed and finalized by the SGLang team in advance.



---

## Issue #N/A: [Bug] sglang 0.4.4.post2 Latency greatly increases when tp=1 and dp > 1

**Link**: https://github.com/sgl-project/sglang/issues/5962
**State**: closed
**Created**: 2025-05-02T00:30:20+00:00
**Closed**: 2025-05-11T01:58:01+00:00
**Comments**: 3
**Labels**: high priority

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Noticed that after bumping sglang to 0.4.4.post2+, when setting dp > 1, the latency would increase 10+ times, the more dp we set, the more latency would increase. The issue is not found in sglang version <= 0.4.4.post1.

Data size: 4k per prompt
Endpoint: v1/completions

With max_token=1, latency for tp1 dp1 is ~40ms, but latency for tp1 d

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Accuracy test of VLM

**Link**: https://github.com/sgl-project/sglang/issues/3142
**State**: open
**Created**: 2025-01-26T06:25:40+00:00
**Comments**: 7
**Labels**: good first issue, help wanted, high priority, MLLM

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

In sglang, LLMs have accuracy tests with Hugging Face models:

https://github.com/sgl-project/sglang/blob/main/test/srt/models/test_generation_models.py

https://github.com/sgl-project/sglang/blob/main/test/srt/test_nightly_math_eval.py

We need similar one for VLM also.

### Related resources

_No response_

---

## Issue #N/A: [Bug] PD Failed to register memory on H200

**Link**: https://github.com/sgl-project/sglang/issues/6753
**State**: open
**Created**: 2025-05-29T23:27:04+00:00
**Comments**: 2
**Labels**: bug, high priority

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

```
root@nccl-test-host-1:/diagnostic# python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --disaggregation-mode prefill --disaggregation-ib-device mlx5_0
Cuda graph is disabled for prefill server
[2025-05-29 23:22:47] server_args=ServerArgs(model_path='meta-llama/Meta-Llama-3-8B-Instruct', tokenizer_path='meta

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] add more CIs for VLM

**Link**: https://github.com/sgl-project/sglang/issues/5249
**State**: open
**Created**: 2025-04-10T18:44:02+00:00
**Comments**: 7
**Labels**: high priority, MLLM

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
https://huggingface.co/google/gemma-3-27b-it
https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct

### Related resources

_No response_

---

## Issue #N/A: 🚧  RFC: Redesign Batch Processing as an Offline Workflow

**Link**: https://github.com/sgl-project/sglang/issues/7427
**State**: open
**Created**: 2025-06-21T18:23:45+00:00
**Comments**: 1
**Labels**: high priority

### Description

### **Summary**
This RFC proposes removing the existing `/v1/batches` and `/v1/files` endpoints from the main OpenAI-compatible server and replacing them with a standalone offline batch processing service.

> **Note:** As part of the ongoing OpenAI API refactor, the batch support has already been removed from the main server. This RFC serves to document the rationale and formalize the replacement plan.


---

### Problem

#### 7.1 Fundamental Issues with the Current Batch API (#7068 )

The current design for online batch processing is flawed and not production-safe. Key issues include:

- **Server Stability Risk**: Uploading and processing thousands of requests at once can overwhelm online API servers.
- **Timing Constraints**: Difficult to enforce `completion_window` in a real-time environment.
- **Resource Contention**: Batch jobs run alongside latency-sensitive requests without proper isolation.
- **Architecture Mismatch**: Batch workloads are inherently asynchronous/offline, confli

[... truncated for brevity ...]

---

## Issue #N/A: Why are there a group of processes concentrated on a single GPU?

**Link**: https://github.com/sgl-project/sglang/issues/3942
**State**: closed
**Created**: 2025-02-28T03:50:01+00:00
**Closed**: 2025-05-01T00:21:11+00:00
**Comments**: 2
**Labels**: high priority, inactive

### Description

I deployed DeepSeek - R1 on a 8*H20-96G server using the following command.

```
python3 -m sglang.launch_server --model-path DeepSeek-R1 --tp 8 --trust-remote-code --mem-fraction-static 0.9 --host 0.0.0.0 --port 50050 --max-running-requests 128 --context-length 32768 --enable-flashinfer-mla --attention-backend flashinfer
```

However, when using the following command to initiate a request on the H20 server, eight processes will be concentrated on GPU0, as shown in the following screenshot.

```
curl -k -X 'POST' \
    'http://localhost:50050/v1/chat/completions' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
"model": "DeepSeek-R1",
"messages": [{"role": "user", "content": "Hello，Who are you?"}],
"stream": false
}'
```

<img width="1280" alt="Image" src="https://github.com/user-attachments/assets/e44e0c0e-52d2-4c4d-8a58-6224da273e9d" />

Is this normal? Is there any way to distribute these processes across all GPUs to prevent GPU0 from running

[... truncated for brevity ...]

---

## Issue #N/A: TTFT latency for long context (16K) is very high around 15 seconds for llama3.1 70b model. (same or worse than vLLM)

**Link**: https://github.com/sgl-project/sglang/issues/922
**State**: closed
**Created**: 2024-08-04T23:14:23+00:00
**Closed**: 2024-10-09T01:10:58+00:00
**Comments**: 12
**Labels**: high priority, inactive, performance

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

### Describe the bug

I am experimenting with SGLang and vLLM for long context(16K) RAG application which requires real time responses.
I am using single Nvidia A6000 48GB GPU and llaam3.1 70b awq 4 bit model.

Currently I am seeing Time for first token latency is around 15 seconds which is very high.
Experimented with parameters like --chunked-prefill-size , --mem-frac etc

can you please suggest what are the parameters I need to mainly focus on to get the optimal TTFT for long context ?

### Reproduction

na

### Environment

```Shell
na
```


---

## Issue #N/A: [Feature] integrate FlashMLA

**Link**: https://github.com/sgl-project/sglang/issues/4384
**State**: closed
**Created**: 2025-03-13T10:43:57+00:00
**Closed**: 2025-03-25T04:14:02+00:00
**Comments**: 1
**Labels**: high priority

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Since SGLang now supports page sizes greater than 1, we should integrate FlashMLA https://github.com/deepseek-ai/FlashMLA.

### Related resources

_No response_

---

## Issue #N/A: [Bug] Qwen2-VL-7B with sglang has significant numerical calculation errors compared to HF Transformers

**Link**: https://github.com/sgl-project/sglang/issues/3106
**State**: closed
**Created**: 2025-01-24T11:32:46+00:00
**Closed**: 2025-01-28T06:04:43+00:00
**Comments**: 9
**Labels**: high priority

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

In practice, we found that sglang Qwen2-VL model has numerical calculation errors compared to HF Transformers model in both Qwen2VisionTransformer and Qwen2Model parts.
Our input image has 720 tokens input to Vit encoding, and the lowest embedded cosine similarity in the output is 0.1775. In addition, we directly feed the Vit output and te

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]Support Qwen2_5...etc tools calling by OpenAI API

**Link**: https://github.com/sgl-project/sglang/issues/1912
**State**: closed
**Created**: 2024-11-04T02:32:41+00:00
**Closed**: 2025-02-12T02:12:07+00:00
**Comments**: 3
**Labels**: good first issue

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Tools calling are becoming mainstream. If you can adapt some updated OpenAI APIs, I would be very grateful.

### Related resources

_No response_

---

## Issue #N/A: [Feature] beat torch compile

**Link**: https://github.com/sgl-project/sglang/issues/4748
**State**: closed
**Created**: 2025-03-25T06:18:28+00:00
**Closed**: 2025-05-26T16:55:12+00:00
**Comments**: 7
**Labels**: good first issue, help wanted, high priority, collaboration, performance

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled

Last year and in the first few months of this year, a significant part of my work focused on removing vLLM dependency. Many reliable teammates joined in this process, and we successfully removed the vLLM dependency on the NVIDIA platform for SGLang. Next, I will co-lead progress on beat torch compile. Past experience shows that torch compile is effective - we just need to write some simple torch ops and let torch compile handle the rest. However, in actual production serving, it is not as smooth as expected - for example, slow startup even with cache enabled, compatibility issues when upgrading torch versions leading to previous features breaking in new versions. We need to profile, benchmark, rewrite th

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] support constrained decoding benchmark

**Link**: https://github.com/sgl-project/sglang/issues/2399
**State**: closed
**Created**: 2024-12-08T10:56:36+00:00
**Closed**: 2025-05-29T21:48:23+00:00
**Comments**: 1
**Labels**: good first issue, help wanted

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

as titled
used for outlines, xgrammar and etc
ref https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving_guided.py

### Related resources

_No response_

---

## Issue #N/A: [Bug] RecursionError: maximum recursion depth exceeded while calling a Python object

**Link**: https://github.com/sgl-project/sglang/issues/4779
**State**: closed
**Created**: 2025-03-26T04:14:50+00:00
**Closed**: 2025-03-26T15:59:08+00:00
**Comments**: 5
**Labels**: good first issue

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug


I confronted this issue today by using docker-latest and docker-dev when using QWQ-32B  but no issue in QWQ-AWQ model. 



Following is my error log

```
  File "/usr/local/lib/python3.10/dist-packages/psutil/__init__.py", line 1277, in send_signal
    self._send_signal(sig)
  File "/usr/local/lib/python3.10/dist-packages/psutil/__init__.

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] GGUF support

**Link**: https://github.com/sgl-project/sglang/issues/1616
**State**: closed
**Created**: 2024-10-09T05:45:17+00:00
**Closed**: 2024-12-01T10:51:57+00:00
**Comments**: 26
**Labels**: good first issue

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Hi! Since .gguf format is already supported by vLLM, is it be possible to add support for it in SGLang server?

### Related resources

_No response_

---

## Issue #N/A: Loading Chat Template in a more flexible way?

**Link**: https://github.com/sgl-project/sglang/issues/376
**State**: closed
**Created**: 2024-04-21T12:50:17+00:00
**Closed**: 2024-07-25T06:33:13+00:00
**Comments**: 1
**Labels**: good first issue, inactive

### Description

The Chat models like [codellama-instruct](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf/blob/main/tokenizer_config.json), [qwen](https://modelscope.cn/models/qwen/Qwen1.5-14B-Chat/file/view/master?fileName=tokenizer_config.json&status=1) all have a `chat_template` field in the JSON which defines the chat template of the model. But I notice it seems that sglang currently hard-coded the chat-template in the [.py](https://github.com/sgl-project/sglang/blob/1bf1cf195302fdff14a4321eb8a17831f5c2fc11/python/sglang/lang/chat_template.py#L79) file. Would it be more flexible to load the default chat template from the tokenizer_config file if provided? It seems [vllm](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/serving_chat.py#L335) did in this way.

---

## Issue #N/A: Support Yi-VL-6B/34B

**Link**: https://github.com/sgl-project/sglang/issues/91
**State**: closed
**Created**: 2024-01-24T03:49:46+00:00
**Closed**: 2024-02-01T21:38:25+00:00
**Comments**: 7
**Labels**: good first issue

### Description

The Yi-VL adopts llava but with silightly different in weights and inference. see [disscusion](https://huggingface.co/01-ai/Yi-VL-34B/discussions/3)

hf repo:
https://huggingface.co/01-ai/Yi-VL-6B
https://huggingface.co/01-ai/Yi-VL-34B

---

## Issue #N/A: [Docs]  Improve DPSK docs in dark mode

**Link**: https://github.com/sgl-project/sglang/issues/3908
**State**: closed
**Created**: 2025-02-27T05:00:48+00:00
**Closed**: 2025-02-27T08:13:05+00:00
**Comments**: 2
**Labels**: documentation, good first issue, help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

<img width="1393" alt="Image" src="https://github.com/user-attachments/assets/39d60ef8-c7fa-42e0-9961-5bd9c082209f" />

I use html to write this docs and it looks bad. So could someone fix it here?

https://github.com/sgl-project/sglang/blob/main/docs/references/deepseek.md

### Related resources

_No response_

---

## Issue #N/A: [Feature] Change contribution guide

**Link**: https://github.com/sgl-project/sglang/issues/2662
**State**: closed
**Created**: 2024-12-30T07:53:12+00:00
**Closed**: 2025-04-29T16:22:21+00:00
**Comments**: 3
**Labels**: documentation, good first issue

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

https://sgl-project.github.io/references/contributor_guide.html

This has been outdated for long. We need to add guide on:

1. How to run docs CI, build it locally, compile it and clean the output and make PR.
2. How to do unit tests locally and add unit tests to CI.
3. How to write elegant unit test following other tests.
4. How to pre-commit.

### Related resources

_No response_

---

## Issue #N/A: [Feature] Parallelism Experiments on AIMO and LIMO

**Link**: https://github.com/sgl-project/sglang/issues/3615
**State**: closed
**Created**: 2025-02-16T19:11:32+00:00
**Closed**: 2025-02-20T19:11:38+00:00
**Comments**: 6
**Labels**: documentation, good first issue, help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Can anyone help test @Simon V’s branch? It’s pretty complete, but we’d like to run some parallel experiments 

https://github.com/sgl-project/sglang/pull/3532

Feel free to submit a PR reporting the results of the parallel experiments, including std, var, etc. Thanks!

### Related resources

_No response_

---

## Issue #N/A: Any benchmarks comparing with TGI?

**Link**: https://github.com/sgl-project/sglang/issues/3188
**State**: closed
**Created**: 2025-01-27T22:14:34+00:00
**Closed**: 2025-01-30T17:42:07+00:00
**Comments**: 2
**Labels**: help wanted

### Description

As the tittle says, is there any benchmark comparing with TGI (https://github.com/huggingface/text-generation-inference)? I see some results comparing directly with vLLM, but would love to see also a direct comparison against TGI, as in the last release the got a good performance improvement, thanks for the info in advance!

---

## Issue #N/A: [Feature] Use xgrammar as default grammar backend to aviod I/O errors while using Outlines in a multi-node setting

**Link**: https://github.com/sgl-project/sglang/issues/3383
**State**: closed
**Created**: 2025-02-07T23:11:12+00:00
**Closed**: 2025-05-26T21:08:02+00:00
**Comments**: 4
**Labels**: good first issue, help wanted, grammar-backend

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

related issues:
#3375 
related discussiton:
[#vllm 4193](https://github.com/vllm-project/vllm/issues/4193)
related pr:
https://github.com/sgl-project/sglang/pull/3379

### Related resources

xGrammar stores its cache in RAM instead of disk, avoiding file system conflicts.
Cache size is small (typically <0.5MB per schema), meaning it doesn't require persistent disk storage.
xGrammar is thread-safe, ensuring it can run across multiple Slurm nodes without concurrency issues.

---

## Issue #N/A: [Feature] Support ipv6 in SGLang

**Link**: https://github.com/sgl-project/sglang/issues/3263
**State**: closed
**Created**: 2025-02-02T19:37:24+00:00
**Closed**: 2025-05-15T00:49:46+00:00
**Comments**: 4
**Labels**: good first issue, help wanted

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

@shuaills 

https://github.com/sgl-project/sglang/issues/2892#issuecomment-2629436443

### Related resources

_No response_

---

## Issue #N/A: [Bug] RuntimeError: RMSNorm failed with error code invalid configuration argument

**Link**: https://github.com/sgl-project/sglang/issues/3304
**State**: closed
**Created**: 2025-02-05T02:25:13+00:00
**Closed**: 2025-05-11T15:17:16+00:00
**Comments**: 22
**Labels**: good first issue, help wanted

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Hi, I am using the main branch of SGLang, and downloading Mixtral-8x22B from huggingface. 

CUDA: 12.4
2 nodes, each has 4 H100 96GB.

I am deploying the server using:
```
python -m sglang.launch_server --model-path Mixtral-8x22B-v0.1 --tp 8 --dist-init-addr xxx:5000 --nnodes 2 --node-rank 0 --trust-remote-code --disable-cuda-graph
python 

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

## Issue #N/A: [Bug] fix gemma-2-2b-it-FP8 accuracy

**Link**: https://github.com/sgl-project/sglang/issues/4324
**State**: closed
**Created**: 2025-03-12T01:27:58+00:00
**Closed**: 2025-05-21T09:30:43+00:00
**Comments**: 8
**Labels**: bug, good first issue, help wanted, high priority, quant

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

The accuracy of `neuralmagic/gemma-2-2b-it-FP8` drops from 0.62 to 0.52 in the main branch. It was detected by our nightly CI run. We need to fix this.

```
neuralmagic/gemma-2-2b-it-FP8 | 0.512 | 0.6
```
https://github.com/sgl-project/sglang/actions/runs/13800885290

### Reproduction

N/A

### Environment

N/A

---

## Issue #N/A: [Feature] Support DeepSeek Janus Models

**Link**: https://github.com/sgl-project/sglang/issues/3195
**State**: closed
**Created**: 2025-01-28T18:37:47+00:00
**Closed**: 2025-04-30T00:18:51+00:00
**Comments**: 4
**Labels**: help wanted, inactive, MLLM

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Docker is a valuable tool for the management of dependencies. Indeed, it can simplify the running of Janus Models to a single command:  
```bash
docker run -it --rm \
  -p 8000:8000 \
  -d \
  -v huggingface:/root/.cache/huggingface \
  -w /app \
  --gpus all \
  --name janus \
  -e MODEL_NAME=deepseek-ai/Janus-Pro-7B \
  julianfl0w/janus:latest
```

Make sure it's working by navigating in your browser to  
[http://localhost:8000/webui](http://localhost:8000/webui)

and by running
```bash
docker logs janus
```

This keeps all the Torch dependencies contained within the image, meaning the user doesn't have to adjust their base installations to run models like these. 

Note: You will have to install NVIDIA Container 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Running multi-node offline engine inference ( via SLURM)

**Link**: https://github.com/sgl-project/sglang/issues/2561
**State**: closed
**Created**: 2024-12-23T15:24:49+00:00
**Closed**: 2025-01-31T23:58:27+00:00
**Comments**: 39
**Labels**: help wanted, collaboration, feature

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

A lot of academic institutions only allow access to larger node clusters via SLURM and it is not immediately clear how would I reuse the code to run Llama 405B BF16 on 2 nodes (by starting a server) to perform offline inference

### Related resources

_No response_

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

## Issue #N/A: [Bug] use Eagle with speculative-num-steps=1

**Link**: https://github.com/sgl-project/sglang/issues/3762
**State**: closed
**Created**: 2025-02-21T13:32:09+00:00
**Closed**: 2025-04-24T00:18:22+00:00
**Comments**: 3
**Labels**: bug, inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

When I attempted to use the Triton backend for Eagle to launch the Qwen-7B model, the process failed.
```
Traceback (most recent call last):
  File "/data/csl/project/sglang/python/sglang/srt/managers/scheduler.py", line 1827, in run_scheduler_process
    scheduler.event_loop_normal()
  File "/data/csl/miniconda3/envs/sglang/lib/python3.10

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Llama4 OOM with 400k input request

**Link**: https://github.com/sgl-project/sglang/issues/5212
**State**: closed
**Created**: 2025-04-10T00:22:53+00:00
**Closed**: 2025-04-11T08:24:15+00:00
**Comments**: 0
**Labels**: bug, high priority

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I started a server on 8xH100 with `meta-llama/Llama-4-Scout-17B-16E-Instruct` with the following command:

```
python sglang.launch_server --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
--port 8080 \
--tp-size 8 \
--chat-template llama-4 \
--attention-backend=fa3 \
--mem-fraction-static=0.8 \
--context-length 1000000 
```

Then sent a

[... truncated for brevity ...]

---

## Issue #N/A: upgrade setuptools and wheel if you found "torch module not found" when installing

**Link**: https://github.com/sgl-project/sglang/issues/2554
**State**: closed
**Created**: 2024-12-23T04:02:50+00:00
**Closed**: 2025-01-30T17:37:49+00:00
**Comments**: 7
**Labels**: bug

### Description

I encountered an issue while installing `sglang`. After upgrading pip (`pip install --upgrade pip`), I ran:

```bash
pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/
```

But it failed with the error:  
`ModuleNotFoundError: No module named 'torch'`.

I found on the Flash Attention GitHub that running this solved the issue:  
```bash
python -m pip install --upgrade pip wheel setuptools
```

It worked for me, so sharing in case someone faces the same problem! I don't know what the exact reason is though as the error itself was pretty strange. 

---

## Issue #N/A: [Bug] OOM for concurrent long requests

**Link**: https://github.com/sgl-project/sglang/issues/1030
**State**: closed
**Created**: 2024-08-11T10:51:49+00:00
**Closed**: 2024-09-22T13:00:44+00:00
**Comments**: 8
**Labels**: bug

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.

### Describe the bug

I am trying to benchmark inference of llama3-8b with long requests, I send **20** concurrent requests each with length of **1k tokens** and I set the **stream to True** and **max_tokens to 1024.** 


This is how I start the server:
`python -m sglang.launch_server --model-path NousResearch/Meta-Llama-3-8B-Instruct  --host 0.0.0.0  --port 8000 --context-length 4096 --dtype bfloat16  --chat-temp

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Tensor shape is wrong when cudagraph+enable_dp_attention

**Link**: https://github.com/sgl-project/sglang/issues/7951
**State**: open
**Created**: 2025-07-11T10:58:55+00:00
**Comments**: 5
**Labels**: bug, high priority

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

I tried to run DSR1 fp4 model on 8xB200, but found that some issue when I opened cudagraph and attndp, the input tensor dimension for each MoE layer is padded to global bs. For example, I take global bs 4096 and attention dp 8, which each rank should have 512 reqs for decode and the input tensor M dimension should be 512 for local rank. 
B

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] 0.0.0.0 host not supported

**Link**: https://github.com/sgl-project/sglang/issues/4935
**State**: closed
**Created**: 2025-03-30T22:04:01+00:00
**Closed**: 2025-05-30T08:43:39+00:00
**Comments**: 3
**Labels**: bug, inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

if we set the host as 0.0.0.0, the warmup function will fail.

<b>Connection to 0.0.0.0 failed.</b></p>\n</blockquote>\n\n<p id="sysmsg">The system returned: <i>(111) Connection refused</I>

_wait_and_warmup -> res = requests.get(url + "/get_model_info", timeout=5, headers=headers)

can we add a fix here to replace 0.0.0.0 with 127.0.0.1 f

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] device-side assert triggered when using run_batch

**Link**: https://github.com/sgl-project/sglang/issues/1279
**State**: closed
**Created**: 2024-09-01T01:51:32+00:00
**Closed**: 2024-09-03T13:02:03+00:00
**Comments**: 6
**Labels**: bug

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

The following error is raised when ever i run run_batch:

```
../aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [1193,0,0], thread: [124,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
../aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [1193,0,0], thread: [125,0,

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]NCCL error if enable the cuda graph

**Link**: https://github.com/sgl-project/sglang/issues/3538
**State**: closed
**Created**: 2025-02-13T06:38:16+00:00
**Closed**: 2025-02-19T14:35:47+00:00
**Comments**: 4
**Labels**: bug, high priority

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

<img width="1663" alt="Image" src="https://github.com/user-attachments/assets/e3b396cc-4771-474d-8843-d43d8d5dbf90" />

If I don't disable cuda graph, I will get the error shown in the picture when the cuda graph is being inited. If i use the official docker image, i will not get the error. The only difference of the environment with the d

[... truncated for brevity ...]

---

