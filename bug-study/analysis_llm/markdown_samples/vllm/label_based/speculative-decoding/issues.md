# speculative-decoding - issues

**Total Issues**: 14
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 4
- Closed Issues: 10

### Label Distribution

- speculative-decoding: 14 issues
- bug: 7 issues
- stale: 3 issues
- performance: 3 issues
- v1: 2 issues
- feature request: 2 issues
- help wanted: 2 issues
- quantization: 1 issues
- RFC: 1 issues
- structured-output: 1 issues

---

## Issue #N/A: [Bug]: Eagle3 in vLLM v0.9.0 has no acceleration effect.

**Link**: https://github.com/vllm-project/vllm/issues/18946
**State**: closed
**Created**: 2025-05-30T08:44:50+00:00
**Closed**: 2025-06-01T02:58:35+00:00
**Comments**: 5
**Labels**: bug, speculative-decoding

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
==============================
        System Info
==============================
OS                           : Ubuntu 22.04.3 LTS (x86_64)
GCC version                  : (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version                : Could not collect
CMake version                : version 3.22.1
Libc version                 : glibc-2.35

==============================
       PyTorch Info
==============================
PyTorch version              : 2.7.0+cu126
Is debug build               : False
CUDA used to build PyTorch   : 12.6
ROCM used to build PyTorch   : N/A

==============================
      Python Environment
==============================
Python version               : 3.10.8 (main, Nov 24 2022, 14:13:03) [GCC 11.2.0] (64-bit runtime)
Python platform              : Linux-5.15.0-88-generic-x86_64-with-glibc2.35

==============================
       

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: SpecDecoding metrics showing with disabled spec decoding

**Link**: https://github.com/vllm-project/vllm/issues/15958
**State**: closed
**Created**: 2025-04-02T18:23:33+00:00
**Closed**: 2025-04-04T15:52:42+00:00
**Comments**: 1
**Labels**: bug, speculative-decoding, v1

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
WARNING:root:libtpu.so and TPU device found. Setting PJRT_DEVICE=TPU.
INFO 04-02 18:23:13 [__init__.py:239] Automatically detected platform tpu.
Collecting environment information...
PyTorch version: 2.8.0
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.31.6
Libc version: glibc-2.35

Python version: 3.11.11 (main, Feb 12 2025, 14:51:05) [Clang 19.1.6 ] (64-bit runtime)
Python platform: Linux-6.8.0-1015-gcp-x86_64-with-glibc2.35
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: No CUDA
Nvidia driver version: No CUDA
cuDNN version: No CUDA
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:   

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Speculative decoding does not work

**Link**: https://github.com/vllm-project/vllm/issues/12323
**State**: closed
**Created**: 2025-01-22T17:15:23+00:00
**Closed**: 2025-05-24T02:07:47+00:00
**Comments**: 3
**Labels**: bug, speculative-decoding, stale

### Description

Here is a script:
```
docker run --gpus '"device=0,1"' --rm -d --net host \
    --name vllm \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /home/thinclient/llm-server/weights:/mnt/weights \
    --env "HUGGING_FACE_HUB_TOKEN=<HF_TOKEN>" \
    --env "TORCH_USE_CUDA_DSA=1" \
    --env "CUDA_LAUNCH_BLOCKING=1" \
    --env "VLLM_RPC_TIMEOUT=100000" \
    --shm-size=15g \
    --ipc host \
    vllm/vllm-openai:latest \
    --model jakiAJK/DeepSeek-R1-Distill-Qwen-7B_GPTQ-int4 \
    --speculative-model Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4 \
    --num-speculative-tokens 5 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.98 \
    --max_model_len 1000 \
    --enable-prefix-caching \

docker logs -f vllm
```
And logs:
```
INFO 01-22 09:01:26 api_server.py:651] vLLM API server version 0.6.5
INFO 01-22 09:01:26 api_server.py:652] args: Namespace(host=None, port=8000, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Disabling Marlin by setting --quantization gptq doesn't work when using a draft model

**Link**: https://github.com/vllm-project/vllm/issues/8784
**State**: closed
**Created**: 2024-09-24T22:51:58+00:00
**Closed**: 2025-01-24T01:58:42+00:00
**Comments**: 2
**Labels**: bug, quantization, speculative-decoding, stale

### Description

### Your current environment

.

### Model Input Dumps

_No response_

### 🐛 Describe the bug

It seems that setting --quantization gptq only disables the marlin for the main model. 

Maybe this can be fixed by adding a --quantization-draft-model setting or forcing the draft model to gptq when main model is forced.

```
INFO 09-24 15:46:11 gptq_marlin.py:112] Detected that the model can run with gptq_marlin, **however you specified quantization=gptq explicitly, so forcing gptq. Use quantization=gptq_marlin for faster inference**
WARNING 09-24 15:46:11 config.py:335] gptq quantization is not fully optimized yet. The speed can be slower than non-quantized models.
INFO 09-24 15:46:11 config.py:904] Defaulting to use mp for distributed inference
**INFO 09-24 15:46:11 gptq_marlin.py:108] The model is convertible to gptq_marlin during runtime. Using gptq_marlin kernel.**
```

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked th

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Speculative decoding server: `ValueError: could not broadcast input array from shape (513,) into shape (512,)`

**Link**: https://github.com/vllm-project/vllm/issues/5563
**State**: closed
**Created**: 2024-06-14T22:54:56+00:00
**Closed**: 2024-09-10T23:02:43+00:00
**Comments**: 18
**Labels**: bug, speculative-decoding

### Description

### Your current environment

```text
Collecting environment information...
PyTorch version: 2.3.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-112-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 11.5.119
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA H100 80GB HBM3
GPU 1: NVIDIA H100 80GB HBM3
GPU 2: NVIDIA H100 80GB HBM3
GPU 3: NVIDIA H100 80GB HBM3
GPU 4: NVIDIA H100 80GB HBM3
GPU 5: NVIDIA H100 80GB HBM3
GPU 6: NVIDIA H100 80GB HBM3
GPU 7: NVIDIA H100 80GB HBM3

Nvidia driver version: 535.161.08
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen run

[... truncated for brevity ...]

---

## Issue #N/A: [Performance] [Speculative decoding] Speed up autoregressive proposal methods by making sampler CPU serialization optional

**Link**: https://github.com/vllm-project/vllm/issues/5561
**State**: closed
**Created**: 2024-06-14T22:24:19+00:00
**Closed**: 2024-07-18T05:21:30+00:00
**Comments**: 6
**Labels**: performance, speculative-decoding

### Description

## Background
Speculative decoding leverages the ability to cheaply generate proposals, and cheaply verify them to achieve speedup for memory-bound inference. Different methods of speculative decoding explore the frontier between cost of proposal, alignment with the target model, and cost of verification.

For example, Medusa produces very cheap proposals, but the quality of the proposals are strictly less than Eagle because the heads do not have access to the previous proposals. Eagle on the other hand pays more for the proposals by sampling autoregressively instead of 1-shot, but it brings the benefit of higher-quality proposals.

At the end of the day, what the user cares about will dictate which speculative technique is used. vLLM's job is to provide them with the option for best speedup for their use case.

Draft-model, EAGLE, and MLPSpeculator rely on autoregressive proposals. This means their top-1 proposals are higher-quality than Medusa, which gives vLLM an ITL reductio

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] [Spec decode]: Combine chunked prefill with speculative decoding

**Link**: https://github.com/vllm-project/vllm/issues/5016
**State**: closed
**Created**: 2024-05-23T20:54:22+00:00
**Closed**: 2024-12-24T17:08:27+00:00
**Comments**: 5
**Labels**: feature request, speculative-decoding, stale

### Description

### 🚀 The feature, motivation and pitch

Speculative decoding can achieve 50%+ latency reduction, but in vLLM it can suffer from the throughput-optimized default scheduling strategy where prefills are prioritized eagerly. Chunked prefill is a recent work in vLLM which optimizes this by spreading out the prefill work over many different decode batches. We can combine chunked prefill with speculative decoding's dynamic speculation length to get the best of both worlds.

This is a complex task that requires some design, if you're interested please reach out.

### Alternatives

_No response_

### Additional context

cc @LiuXiaoxuanPKU @comaniac @rkooo567 

---

## Issue #N/A: [Help wanted] [Spec decode]: Increase acceptance rate via Medusa's typical acceptance

**Link**: https://github.com/vllm-project/vllm/issues/5015
**State**: closed
**Created**: 2024-05-23T20:50:19+00:00
**Closed**: 2024-06-18T02:29:10+00:00
**Comments**: 1
**Labels**: feature request, speculative-decoding

### Description

### 🚀 The feature, motivation and pitch

Speculative decoding allows emitting multiple tokens per sequence by speculating future tokens, scoring their likelihood using the LLM, and then accepting each speculative token based on its likelihood. This process is laid out in the following diagram:
![Screenshot 2024-05-23 at 1 45 16 PM](https://github.com/vllm-project/vllm/assets/950914/52d21c58-1a0e-4a8f-b1f8-4abe79651588)

The problem with rejection sampling is that it holds a very high bar for quality: it is lossless and guarantees the distribution of the target model, even if it means rejecting plausible speculative tokens.

This issue is a request to implement Medusa's typical acceptance routing in vLLM. Typical acceptance trades off output quality to increase the acceptance rate. See "Choice of threshold in typical acceptance" in the [Medusa blogpost](https://sites.google.com/view/medusa-llm) for more information.

vLLM users should be able to toggle between different acceptanc

[... truncated for brevity ...]

---

## Issue #N/A: [Performance] [Speculative decoding]: Support draft model on different tensor-parallel size than target model

**Link**: https://github.com/vllm-project/vllm/issues/4632
**State**: closed
**Created**: 2024-05-06T18:14:40+00:00
**Closed**: 2024-06-25T09:56:08+00:00
**Comments**: 5
**Labels**: help wanted, performance, speculative-decoding

### Description

## Overview
Speculative decoding allows a speedup for memory-bound LLMs by using a fast proposal method to propose tokens that are verified in a single forward pass by the larger LLM. Papers report 2-3x speedup for bs=1, in Anyscale's fork we see up to 2x speedup with a small draft model for bs=8 (30% for bs=16) (we can improve this! see https://github.com/vllm-project/vllm/issues/4630 if you want to help).

A key optimization for small models (68m/160m domain) is to use tensor-parallel degree 1, even if the target model is using tensor-parallel degree 4 or 8. In our fork, this reduces proposal time from 5ms/tok to 1.5ms/tok. This will allow a well-aligned 68m draft model to get 2x per-user throughput improvement on 70B target model.

Furthermore, a 1B/7B proposer model may ideally be placed on TP=2 or TP=4, while the larger model is placed on TP=8. vLLM should support these configuration so the community can use the configuration best for their draft model.

## Design suggestio

[... truncated for brevity ...]

---

## Issue #N/A: [Speculative decoding] [Help wanted] [Performance] Optimize draft-model speculative decoding

**Link**: https://github.com/vllm-project/vllm/issues/4630
**State**: closed
**Created**: 2024-05-06T17:16:50+00:00
**Closed**: 2024-08-05T18:05:31+00:00
**Comments**: 28
**Labels**: help wanted, performance, speculative-decoding

### Description

### Proposal to improve performance

With the end-to-end correctness tests merged in https://github.com/vllm-project/vllm/pull/3951, now we will optimize the implementation to get ~50% speedup on 70B model with temperature 1.0.

### Work required:
P0/P1 -- priority
(Small/Medium/Large) -- relative size estimate

* Optimizing proposal time
  - [x] P0 (Large) Reduce draft model control-plane communication from O(num_steps) to O(1)
  - [x] P0 (Medium) Support draft model on different tensor-parallel-size than target model https://github.com/vllm-project/vllm/issues/4632
* Optimizations for scoring time
  - [x] P0 (Medium) Re-enable bonus tokens to increase % accepted tokens https://github.com/vllm-project/vllm/issues/4212
  - [ ] P1 (Large) Replace CPU-based batch expansion with multi-query attention kernel call
  - [ ] P1 (Medium) Automate speculative decoding https://github.com/vllm-project/vllm/issues/4565
* Optimizations for both proposal and scoring time https://github

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Enabling Suffix Decoding, LSTM Speculator, Sequence Parallelism from Arctic Inference

**Link**: https://github.com/vllm-project/vllm/issues/18037
**State**: open
**Created**: 2025-05-13T01:25:02+00:00
**Comments**: 5
**Labels**: RFC, speculative-decoding

### Description

### Motivation.

Snowflake AI Research has recently released several optimizations like Suffix Decoding, LSTM Speculation, Sequence Parallelism, SwiftKV etc, improving TTFT, TPOT and throughput for vLLM via a plugin called Arctic Inference (repo: https://github.com/snowflakedb/arcticinference).

**Performance Improvements**
- 4x faster generation with [Suffix Decoding](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/) for SWEBench
- 2.4x faster generation with [LSTM Speculator](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/) for ShareGPT
- 2.8x faster coding with [LSTM Speculator](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/) for HumanEval
- 2x higher throughput with [SwiftKV](https://www.snowflake.com/en/engineering-blog/swiftkv-llm-compute-reduction/)
- 1.4x throughput than TP=8, but same TTFT as TP=8 with [Arctic Ulysses](https://www.snowflake.com/en/engineering-blog/uly

[... truncated for brevity ...]

---

## Issue #N/A: [SpecDecode] Support EAGLE in V1

**Link**: https://github.com/vllm-project/vllm/issues/15901
**State**: open
**Created**: 2025-04-01T19:45:13+00:00
**Comments**: 21
**Labels**: speculative-decoding, v1

### Description

- [x] 1. Correctly initializing and loading the EAGLE draft model
- [x] 2. Consider the lookahead slots in the KV cache manager
- [x] 3. Cache `draft_probs` inside the model runner and correctly feed it to the rejection sampler in the next step (temporarily workaround: #16899)
- [x] 4. Handle the edge cases like when the draft model generates beyond `max_pos_embeddings`
- [ ] 5. Handle the seeds correctly
- [ ] 6. Do E2E correctness and performance tests
- [x] 7. Support prefix caching. Eagle requires special handling because Eagle's i-th KV cache is coupled with the i+1-th token ID. (@LiuXiaoxuanPKU)
- [ ] 8. Properly handle the sampling parameters that are not (currently) compatible with spec decoding (e.g., min_p).
- [x] 9. Use CUDA graphs for draft model. (@luyuzhe111)
- [x] 10. Support Eagle 3 (https://github.com/vllm-project/vllm/pull/16937)

_Originally posted by @WoosukKwon in https://github.com/vllm-project/vllm/issues/15729#issuecomment-2765192455_
            

---

## Issue #N/A: [Bug]: Speculative decoding + guided decoding not working

**Link**: https://github.com/vllm-project/vllm/issues/10442
**State**: open
**Created**: 2024-11-19T06:13:22+00:00
**Comments**: 4
**Labels**: bug, structured-output, speculative-decoding

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```

</details>


### Model Input Dumps

_No response_

### 🐛 Describe the bug

When using speculative decoding plus guided decoding (outlines), the output is truncated to like 5 tokens on return. I am using ngram speculation and extracting company names from a document:
```

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date


class CompanyNameEntry(BaseModel):
    company_name: str
    #index: int
    #duration: str

class CompanyNamesList(BaseModel):
    company_names: List[CompanyNameEntry]



completion = client.chat.completions.create(
              model="model",
              messages=[
                {"role": "user", "content": document_instruction}
              ],
              extra_body={
                "guided_json": CompanyNamesList.schema(),


[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Persistent OutOfMemoryError error when using speculative decoding 

**Link**: https://github.com/vllm-project/vllm/issues/8073
**State**: open
**Created**: 2024-09-02T06:41:55+00:00
**Comments**: 4
**Labels**: bug, speculative-decoding

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.3.1+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.11.0rc1 (main, Aug 12 2022, 10:02:14) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-1067-aws-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A10G
Nvidia driver version: 535.161.07
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.9.7
/usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8.9.

[... truncated for brevity ...]

---

