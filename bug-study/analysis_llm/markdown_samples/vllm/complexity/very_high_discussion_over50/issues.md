# very_high_discussion_over50 - issues

**Total Issues**: 15
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 7
- Closed Issues: 8

### Label Distribution

- stale: 3 issues
- feature request: 2 issues
- bug: 2 issues
- structured-output: 1 issues
- misc: 1 issues
- usage: 1 issues
- documentation: 1 issues
- v1: 1 issues
- performance: 1 issues
- RFC: 1 issues

---

## Issue #N/A: [Feature]: Support gemma3 architecture

**Link**: https://github.com/vllm-project/vllm/issues/14696
**State**: closed
**Created**: 2025-03-12T18:21:45+00:00
**Closed**: 2025-03-13T03:11:13+00:00
**Comments**: 56
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

I am using vLLM for hosting of LLMs/SLMs and with the recent release of Gemma 3, I would love to have it supported in vLLM. Google has stated Gemma 3 has day 1 support from HF Transformers, so it should (hopefully) be relatively simple to integrate into vLLM. 

Currently, when attempting to load google/gemma-3-12b-it, the following error is given:

```
ERROR 03-12 18:19:00 engine.py:400] ValueError: The checkpoint you are trying to load has model type `gemma3` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.
ERROR 03-12 18:19:00 engine.py:400]
ERROR 03-12 18:19:00 engine.py:400] You can update Transformers with the command `pip install --upgrade transformers`. If this does not work, and the checkpoint is very new, then there may not be a release version that supports this model yet. In this case, you can get the most up-to-date co

[... truncated for brevity ...]

---

## Issue #N/A: [Model] Meta Llama 3.1 Know Issues & FAQ

**Link**: https://github.com/vllm-project/vllm/issues/6689
**State**: closed
**Created**: 2024-07-23T15:19:50+00:00
**Closed**: 2024-09-04T06:02:40+00:00
**Comments**: 85

### Description

## Please checkout [Announcing Llama 3.1 Support in vLLM](https://blog.vllm.ai/2024/07/23/llama31.html) ##

* Chunked prefill is turned on for all Llama 3.1 models. However, it is currently incompatible with prefix caching, sliding window, and multi-lora. In order to use those features, you can set `--enable-chunked-prefill=false` then optionally combine it with `--max-model-len=4096` if turning it out cause OOM. You can change the length for the context window you desired. 
* Rope scaling `if rope_scaling is not None and rope_scaling["type"] not in, KeyError: 'type'.`
    * Please update to [v0.5.3.post1](https://github.com/vllm-project/vllm/releases/tag/v0.5.3.post1) which included a fix. 
* Rope scaling `ValueError: 'rope_scaling' must be a dictionary with two fields, 'type' and 'factor', got {'factor': 8.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0, 'original_max_position_embeddings': 8192, 'rope_type': 'llama3'}`
  * Please upgrade transformers to 4.43.1 (`pip install 

[... truncated for brevity ...]

---

## Issue #N/A: [Misc]: Throughput/Latency for guided_json with ~100% GPU cache utilization

**Link**: https://github.com/vllm-project/vllm/issues/3567
**State**: closed
**Created**: 2024-03-22T12:11:32+00:00
**Closed**: 2025-05-17T02:09:48+00:00
**Comments**: 67
**Labels**: structured-output, misc, stale

### Description

### Anything you want to discuss about vllm.

Hi,

I am running some benchmarks on the `vllm.entrypoints.openai.api_server` measuring latency and throughput with different number of concurrent requests.

Specs:
- H100 80GB
- qwen-1.5-14B-chat

I am sending 1000 requests with random prompts of token length 512. These are the results I get (see attached image):


**Guided_json**
- ~100 running requests
- ~70 generation tokens per second
- ~1700 ms median token time

**Non-guided_json**
- ~100 running requests
- ~800 generation tokens per second
- ~75 ms median token time (TPOT)

At 10 concurrent request (GPU utlization << 100%

Non-guided_json: ~20 ms median token time
guided_json: ~ 160 ms median token time


Currently the application I am building heavily relies on guided_json, however, to put it in an online setting I would like to ask 1) are the numbers I experience sensible and 2) what can be done to improve performance in the guided_json paradigm?

I am

[... truncated for brevity ...]

---

## Issue #N/A: ImportError: /ramyapra/vllm/vllm/_C.cpython-310-x86_64-linux-gnu.so: undefined symbol:

**Link**: https://github.com/vllm-project/vllm/issues/2747
**State**: closed
**Created**: 2024-02-04T15:52:34+00:00
**Closed**: 2024-03-25T04:44:55+00:00
**Comments**: 64

### Description

I'm trying to run vllm and lm-eval-harness. I'm using vllm 0.2.5. After I'm done installing both, if I try importing vllm I get the following error:
`
 File "/ramyapra/lm-evaluation-harness/lm_eval/models/__init__.py", line 7, in <module>
    from . import vllm_causallms
  File "/ramyapra/lm-evaluation-harness/lm_eval/models/vllm_causallms.py", line 16, in <module>
    from vllm import LLM, SamplingParams
  File "/ramyapra/vllm/vllm/__init__.py", line 3, in <module>
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
  File "/ramyapra/vllm/vllm/engine/arg_utils.py", line 6, in <module>
    from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
  File "/ramyapra/vllm/vllm/config.py", line 9, in <module>
    from vllm.utils import get_cpu_memory, is_hip
  File "/ramyapra/vllm/vllm/utils.py", line 8, in <module>
    from vllm._C import cuda_utils
ImportError: /ramyapra/vllm/vllm/_C.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN2at4_ops19empt

[... truncated for brevity ...]

---

## Issue #N/A: ARM aarch-64 server build failed (host OS: Ubuntu22.04.3) 

**Link**: https://github.com/vllm-project/vllm/issues/2021
**State**: closed
**Created**: 2023-12-11T14:37:44+00:00
**Closed**: 2024-09-22T19:47:55+00:00
**Comments**: 54

### Description

do as: https://docs.vllm.ai/en/latest/getting_started/installation.html 
1. docker run --gpus all -it --rm --ipc=host nvcr.io/nvidia/pytorch:23.10-py3
2. git clone https://github.com/vllm-project/vllm.git
3. cd vllm
4. pip install -e .

here is the details in side the docker instance:
root@f8c2e06fbf8b:/mnt/vllm# pip install -e .
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Obtaining file:///mnt/vllm
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... error
  error: subprocess-exited-with-error
  
  × Getting requirements to build editable did not run successfully.
  │ exit code: 1
  ╰─> [22 lines of output]
      /tmp/pip-build-env-4xoxai9j/overlay/local/lib/python3.10/dist-packages/torch/nn/modules/transformer.py:20: UserWarning: Failed to initialize NumPy: No module named 'numpy' (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.c

[... truncated for brevity ...]

---

## Issue #N/A: Does vllm support the Mac/Metal/MPS? 

**Link**: https://github.com/vllm-project/vllm/issues/1441
**State**: closed
**Created**: 2023-10-21T00:52:39+00:00
**Closed**: 2023-10-22T08:01:00+00:00
**Comments**: 110

### Description

I ran into the error when pip install vllm in Mac: 
    RuntimeError: Cannot find CUDA_HOME. CUDA must be available to build the package. 



---

## Issue #N/A: Support Multiple Models

**Link**: https://github.com/vllm-project/vllm/issues/299
**State**: closed
**Created**: 2023-06-28T19:14:50+00:00
**Closed**: 2024-09-04T04:24:59+00:00
**Comments**: 89
**Labels**: feature request

### Description

- Allow user to specify multiple models to download when loading server
- Allow user to switch between models 
- Allow user to load multiple models on the cluster (nice to have)


---

## Issue #N/A: CUDA error: out of memory

**Link**: https://github.com/vllm-project/vllm/issues/188
**State**: closed
**Created**: 2023-06-21T13:50:20+00:00
**Closed**: 2023-06-27T14:50:09+00:00
**Comments**: 54
**Labels**: bug

### Description

I successfully installed vLLM in WSL2, when I was trying to run the sample code, I got error info like this:

```
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="/mnt/d/github/text-generation-webui/models/facebook_opt-125m")

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

INFO 06-21 21:40:02 llm_engine.py:59] Initializing an LLM engine with config: model='/mnt/d/github/text-generation-webui/models/facebook_opt-125m', dtype=torch.float16, use_dummy_weights=False, download_dir=None, use_np_weights=False, tensor_parallel_size=1, seed=0)
INFO 06-21 21:40:12 llm_engine.p

[... truncated for brevity ...]

---

## Issue #N/A: [Usage] Qwen3 Usage Guide

**Link**: https://github.com/vllm-project/vllm/issues/17327
**State**: open
**Created**: 2025-04-28T22:05:06+00:00
**Comments**: 88
**Labels**: usage

### Description

vLLM v0.8.4 and higher natively supports all Qwen3 and Qwen3MoE models. Example command:
* `vllm serve Qwen/... --enable-reasoning --reasoning-parser deepseek_r1` 
    * All models should work with the command as above. You can test the reasoning parser with the following example script: https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_with_reasoning_streaming.py
    * Some MoE models might not be divisible by TP 8. Either lower your TP size or use `--enable-expert-parallel`. 


* If you are seeing the following error when running fp8 dense models, you are running on vLLM v0.8.4. Please upgrade to v0.8.5.
```
File ".../vllm/model_executor/parameter.py", line 149, in load_qkv_weight
    param_data = param_data.narrow(self.output_dim, shard_offset,
IndexError: start out of range (expected to be in range of [-18, 18], but got 2048)
```

* If you are seeing the following error when running MoE models with fp8, you are running with too much tenso

[... truncated for brevity ...]

---

## Issue #N/A: [Doc]: Steps to run vLLM on your RTX5080 or 5090!

**Link**: https://github.com/vllm-project/vllm/issues/14452
**State**: open
**Created**: 2025-03-07T18:12:24+00:00
**Comments**: 119
**Labels**: documentation

### Description

### 📚 The doc issue

Let's take a look at the steps required to run vLLM on your RTX5080/5090! 

1. **Initial Setup:** To start with, we need a container that has CUDA 12.8 and PyTorch 2.6 so that we have nvcc that can compile for Blackwell. 

```
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                                -it nvcr.io/nvidia/pytorch:25.02-py3 /bin/bash
```

2. **Clone vLLM Repository:** Let's clone top of tree vLLM. If you have an existing clone or working directory, ensure that you are at or above the commit [ed6ea06](https://github.com/vllm-project/vllm/commit/ed6ea06577ec06f0b3a9ac921b55ef254f19d923) in your clone. 

```
git clone https://github.com/vllm-project/vllm.git && cd vllm
```

3. **Build vLLM in the container:** Now, we start building vLLM. Please note here that we can't use precompiled vLLM because `vllm-project/vllm` has not moved to the required torch and CUDA versions yet. So, we leverage the torch and CUDA versions th

[... truncated for brevity ...]

---

## Issue #N/A: [V1] Feedback Thread

**Link**: https://github.com/vllm-project/vllm/issues/12568
**State**: open
**Created**: 2025-01-30T02:46:45+00:00
**Comments**: 92
**Labels**: v1

### Description

Please leave comments here about your usage of V1, does it work? does it not work? which feature do you need in order to adopt it? any bugs? 

For bug report, please file it separately and link the issue here. 

For in depth discussion, please feel free to join #sig-v1 in the vLLM Slack workspace. 


---

## Issue #N/A: [Performance]: decoding speed on long context

**Link**: https://github.com/vllm-project/vllm/issues/11286
**State**: open
**Created**: 2024-12-18T06:59:09+00:00
**Comments**: 53
**Labels**: performance, stale

### Description

### Proposal to improve performance

In our experiments, we found that the decoding speed of vLLM decreases dramatically when the length of the prompt becomes longer. 
We fixed the batchsize=90 the decoding speed is 5364 tokens/s when the length of the prompt is within 100, 5500 tokens/s when 100 to 200, and decreases to 782 when 4000 to 8000, and decreases to 273 when greater than 8000.

<html xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:x="urn:schemas-microsoft-com:office:excel" xmlns="http://www.w3.org/TR/REC-html40">
<head>

<meta name=Generator content="Microsoft Excel">
<!--[if !mso]>
<style>
v\:* {behavior:url(#default#VML);}
o\:* {behavior:url(#default#VML);}
x\:* {behavior:url(#default#VML);}
.shape {behavior:url(#default#VML);}
</style>
<![endif]-->

</head>
<body>
<!--StartFragment-->

prompt length | 0-100 | 100-200 | 200-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 | 8000+
-- | -- | -- | -- | -

[... truncated for brevity ...]

---

## Issue #N/A: Llama3.2 Vision Model: Guides and Issues

**Link**: https://github.com/vllm-project/vllm/issues/8826
**State**: open
**Created**: 2024-09-25T22:50:46+00:00
**Comments**: 54
**Labels**: stale

### Description

Running the server (using the vLLM CLI or our [docker image](https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html)):
* `vllm serve meta-llama/Llama-3.2-11B-Vision-Instruct --enforce-eager --max-num-seqs 16`
* `vllm serve meta-llama/Llama-3.2-90B-Vision-Instruct --enforce-eager --max-num-seqs 32 --tensor-parallel-size 8`

Currently:
* Only one leading image is supported. Support for multiple images and interleaving images are work in progress.
* Text only inference is supported.
* Only NVIDIA GPUs are supported.
* *Performance is acceptable but to be optimized!* We aim at first release to be functionality correct. We will work on making it fast 🏎️ 

**Please see the [next steps](https://github.com/vllm-project/vllm/issues/8826#issuecomment-2379960574) for better supporting this model on vLLM.**

cc @heheda12345 @ywang96 

---

## Issue #N/A: [RFC]: Multi-modality Support on vLLM

**Link**: https://github.com/vllm-project/vllm/issues/4194
**State**: open
**Created**: 2024-04-19T07:51:48+00:00
**Comments**: 98
**Labels**: RFC, multi-modality

### Description

**Active Projects (help wanted!):**
- [Core tasks](https://github.com/orgs/vllm-project/projects/8)
- [Model requests](https://github.com/orgs/vllm-project/projects/10)

**Update [11/18] - In the upcoming months, we will focus on performance optimization for multimodal models as part of vLLM V1 engine re-arch effort**

**P0** (We will definitely work on them):
- [ ] V1 re-arch for multimodal models - See high-level design ([Slides](https://docs.google.com/presentation/d/1e3CxQBV3JsfGp30SwyvS3eM_tW-ghOhJ9PAJGK6KR54/edit#slide=id.g31455c8bc1e_2_122), [Doc](https://docs.google.com/document/d/11_DFQTku6C2aV6ghK21P76ST6uAUVjMlEjs54prtb_g/edit?usp=sharing))
  - [ ] Core 
    - [x] [1/N] #9871
    - [x] [2/N] #10374 
    - [x] [3/N] #10570 
    - [x] [4/N] #10699
    - [x] [5/N] #11210
    - [x] [6/N] #12128 
    - [x] [7/N] Enable rest of single-modality LMMs on V1
      - [x] #11632 (Aria, BLIP-2, Chameleon, Fuyu)
      - [x] #14275
      - [x] #11685
      - [x] #11733
      - [x] #12069
 

[... truncated for brevity ...]

---

## Issue #N/A: Recent vLLMs ask for too much memory: ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization` when initializing the engine.

**Link**: https://github.com/vllm-project/vllm/issues/2248
**State**: open
**Created**: 2023-12-24T02:47:42+00:00
**Comments**: 53
**Labels**: bug, unstale

### Description

Since vLLM 0.2.5, we can't even run llama-2 70B 4bit AWQ on 4*A10G anymore, have to use old vLLM.  Similar problems even trying to be two 7b models on 80B A100.

For small models, like 7b with 4k tokens, vLLM fails for "cache blocks" even though alot more memory is left.

E.g.  building docker image with cuda 11.8 and vllm 0.2.5 or 0.2.6 and running like:

```
port=5001
tokens=8192
docker run -d \
    --runtime=nvidia \
    --gpus '"device=1"' \
    --shm-size=10.24gb \
    -p $port:$port \
    --entrypoint /h2ogpt_conda/vllm_env/bin/python3.10 \
    -e NCCL_IGNORE_DISABLED_P2P=1 \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -u `id -u`:`id -g` \
    -v "${HOME}"/.cache:/workspace/.cache \
    --network host \
    gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0 -m vllm.entrypoints.openai.api_server \
        --port=$port \
        --host=0.0.0.0 \
        --model=defog/sqlcoder2 \
        --seed 1234 \
        --trust-remote-code \
	--ma

[... truncated for brevity ...]

---

