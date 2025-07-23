# quantization - issues

**Total Issues**: 12
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 12

### Label Distribution

- quantization: 12 issues
- bug: 5 issues
- stale: 3 issues
- feature request: 2 issues
- good first issue: 1 issues
- speculative-decoding: 1 issues
- RFC: 1 issues
- misc: 1 issues
- help wanted: 1 issues
- performance: 1 issues

---

## Issue #N/A: [Feature]: Support FP8 Marlin MoE for CompressedTensorsW8A8Fp8MoEMethod

**Link**: https://github.com/vllm-project/vllm/issues/18008
**State**: closed
**Created**: 2025-05-12T18:02:12+00:00
**Closed**: 2025-05-20T11:58:40+00:00
**Comments**: 4
**Labels**: good first issue, feature request, quantization

### Description

### 🚀 The feature, motivation and pitch

Like what was added in https://github.com/vllm-project/vllm/pull/16850 for enabling marlin in fp8.py MoE layers, we should enable FP8 Marlin MoE for compressed tensors models to support users wanting to run them on older hardware.

Basically you want to take the changes in fp8.py's moe method (https://github.com/vllm-project/vllm/pull/16850/files#diff-5511bfcc9c53f7d96517ad43e4087f6777bef21302da983f42cafae40a866644) and apply them to `CompressedTensorsW8A8Fp8MoEMethod`

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

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

## Issue #N/A: [RFC]: Int8 Activation Quantization

**Link**: https://github.com/vllm-project/vllm/issues/3975
**State**: closed
**Created**: 2024-04-10T17:20:09+00:00
**Closed**: 2024-09-30T18:10:25+00:00
**Comments**: 3
**Labels**: quantization, RFC, misc

### Description

# Summary
* We (engineering at @neuralmagic) are working on support for int8 quantized activations.
* This RFC is proposing an _incremental_ approach to quantization, where the initial support for quantization will make _minimal_ and _local_ changes to the PyTorch model definitions.  We propose swapping out Linear and Attention modules with their quantized counterparts without modifying the graphs around them. The upside to this will be quicker support for quantized models. The downside is that we will be quantizing the activations on the fly prior to computation.
* To reduce the additional data movement from quantizing the activations on the fly, the activations will need to remain quantized throughout the graph, requiring more extensive and nonlocal modifications to the model definitions. We will be working on abstractions for the quantized model definitions to make adding support for new models as easy as possible. 
* Activation quantization will introduce additional elementwise

[... truncated for brevity ...]

---

## Issue #N/A: Implement 60% faster context processing for AWQ

**Link**: https://github.com/vllm-project/vllm/issues/2551
**State**: closed
**Created**: 2024-01-22T16:51:07+00:00
**Closed**: 2024-01-30T21:48:51+00:00
**Comments**: 1
**Labels**: quantization

### Description

After some experimentation, I found that dequantizing and running FP16 matmul is faster in cases where `batch_size * n_tokens >= 1024`. This should help with throughput.

https://github.com/casper-hansen/AutoAWQ/pull/316

---

## Issue #N/A: Mixtral Quantization Issues

**Link**: https://github.com/vllm-project/vllm/issues/2543
**State**: closed
**Created**: 2024-01-22T07:46:07+00:00
**Closed**: 2024-04-04T12:36:41+00:00
**Comments**: 5
**Labels**: bug, quantization

### Description

I'm currently working with quantized versions of Mixtral 8x7B provided by TheBloke, and I load them with vLLM. I'm currently with these issues:
`TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ` can be well loaded, but even if the temperature has been fixed to 0, the model gives different outputs on the same prompt. The lack of deterministic is not found on traditional models.
`TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ` keeps outputting nothing (is mentioned in huggingface discussions [here](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ/discussions/3)
Is there anyone having faced and resolved such a problem? I know it may not be directly related to vLLM. And is there anyone having tested a quantized Mixtral model with vLLM well? Great thx.

---

## Issue #N/A: GPTQ does not support bfloat16

**Link**: https://github.com/vllm-project/vllm/issues/2149
**State**: closed
**Created**: 2023-12-17T06:06:30+00:00
**Closed**: 2024-11-30T02:03:24+00:00
**Comments**: 6
**Labels**: help wanted, feature request, quantization, stale

### Description

Currently, our GPTQ kernels only support the float16 precision.

---

## Issue #N/A: GPTQ models don't support CUDA graph

**Link**: https://github.com/vllm-project/vllm/issues/2147
**State**: closed
**Created**: 2023-12-17T05:54:30+00:00
**Closed**: 2024-01-03T17:52:30+00:00
**Comments**: 4
**Labels**: bug, quantization

### Description

Got the following error while running `python examples/llm_engine_example.py --model TheBloke/Mixtral-8x7B-v0.1-GPTQ --dtype half`:
```
  File "/home/wskwon/workspace/vllm/vllm/model_executor/layers/sampler.py", line 396, in _random_sample
    random_samples = torch.multinomial(probs,
RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
```
The error didn't appear when the `--enforce-eager` flag was set.

*AWQ models did not raise errors.

I guess this is somehow related to exllama v2 kernels.

---

## Issue #N/A: error when inferencing Mixtral AWQ

**Link**: https://github.com/vllm-project/vllm/issues/2074
**State**: closed
**Created**: 2023-12-13T02:33:06+00:00
**Closed**: 2023-12-18T18:56:13+00:00
**Comments**: 30
**Labels**: quantization

### Description

When I try to run a AsyncEngine with ybelkada/Mixtral-8x7B-Instruct-v0.1-AWQ
I get Traceback (most recent call last):
  File "/home/marco/Scrivania/TESI/serving/vllm_server.py", line 91, in <module>
    engine = AsyncLLMEngine.from_engine_args(engine_args)
  File "/home/marco/miniconda3/envs/serving/lib/python3.10/site-packages/vllm/engine/async_llm_engine.py", line 495, in from_engine_args
    engine = cls(parallel_config.worker_use_ray,
  File "/home/marco/miniconda3/envs/serving/lib/python3.10/site-packages/vllm/engine/async_llm_engine.py", line 269, in __init__
    self.engine = self._init_engine(*args, **kwargs)
  File "/home/marco/miniconda3/envs/serving/lib/python3.10/site-packages/vllm/engine/async_llm_engine.py", line 314, in _init_engine
    return engine_class(*args, **kwargs)
  File "/home/marco/miniconda3/envs/serving/lib/python3.10/site-packages/vllm/engine/llm_engine.py", line 107, in __init__
    self._init_workers_ray(placement_group)
  File "/home/marco/mi

[... truncated for brevity ...]

---

## Issue #N/A: bug of opt awq model

**Link**: https://github.com/vllm-project/vllm/issues/1703
**State**: closed
**Created**: 2023-11-17T11:21:14+00:00
**Closed**: 2023-11-19T01:56:49+00:00
**Comments**: 2
**Labels**: bug, quantization

### Description

Hi @zhuohan123, I found two bugs of opt awq model in the latest code because of the code refactor. 
1. in https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/opt.py#L215, some opt model may not use quantized linear in project_in/project_out, 
2. in https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/opt.py#L254, project_in/project_out have two return value

---

## Issue #N/A: load_weights KeyError with quantized GPTBigCodeForCausalLM

**Link**: https://github.com/vllm-project/vllm/issues/1682
**State**: closed
**Created**: 2023-11-16T05:58:24+00:00
**Closed**: 2023-12-13T19:08:11+00:00
**Comments**: 7
**Labels**: bug, quantization

### Description

I trying load awq quantized [bigcode/octocoder](https://huggingface.co/bigcode/octocoder)
(GPTBigCodeForCausalLM) model wth vLLM.

**Environ**
- docker image based nvcr.io/nvidia/pytorch:23.08-py3
- CUDA 12.2.1
- pytorch 2.1.0a0+29c30b1
- transformers==4.35.0
- autoawq==0.1.6
- vllm local build from github sourece

**repro**
- quantize bigcode/octocoder with AutoAWQ
- load model with Transformer and inferencing
It works fine.

- clone `refactor-quantization` branch from [PR1622](https://github.com/vllm-project/vllm/pull/1622)
- try load model

```
Initializing an LLM engine with config: model='/usr/local/model/llm', tokenizer='/usr/local/model/llm', tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=4096, download_dir=None, load_format=auto, tensor_parallel_size=2, quantization=awq, seed=0)
```
- RayWorker dead with Error

```
ray.exceptions.RayTaskError(KeyError): ray::RayWorker.execute_method

[... truncated for brevity ...]

---

## Issue #N/A: Quantization for V100

**Link**: https://github.com/vllm-project/vllm/issues/1345
**State**: closed
**Created**: 2023-10-13T16:44:17+00:00
**Closed**: 2024-12-01T02:16:03+00:00
**Comments**: 12
**Labels**: quantization, stale

### Description

Similar to #1252 , do we have any plans for supporting V100. For now I can see that the place need to be modified is ldmatrix instruction and m16n8k16, as an example we may need to load the matrix manually and perform the mma in a smaller size, for example, maybe we need something similar to these
```c++
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
          // Manually loading each fragment, ldmatrix only available on sm_75 and after
          __asm__ __volatile__(
              "ld.shared.b16 %0, [%4];\n"
              "ld.shared.b16 %1, [%4 + 2];\n"
              "ld.shared.b16 %2, [%4 + 4];\n"
              "ld.shared.b16 %3, [%4 + 6];\n"
              : "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[0]), 
                "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[1]), 
                "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[2]), 
                "=r"(((unsigned *)(B_shared_warp + (ax1_0 * 8)))[3])
              : "r"(addr)
          );
#else


[... truncated for brevity ...]

---

## Issue #N/A: Huge latency increase with AWQ models at medium context lengths

**Link**: https://github.com/vllm-project/vllm/issues/1242
**State**: closed
**Created**: 2023-10-01T17:10:40+00:00
**Closed**: 2024-03-20T12:47:49+00:00
**Comments**: 1
**Labels**: performance, quantization

### Description

Using an awq quantized model from thebloke (TheBloke/manticore-13b-chat-pyg-AWQ), generation is fine and starts after a few seconds with only a few sentences in the context window, but anything more than three or four makes it take ~30 seconds to start generation. Tried different awq models, and the issue doesn't happen with unquantized models. Is this normal/something to do with the way it handles awq models?

---

