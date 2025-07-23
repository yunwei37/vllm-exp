# keep-open - issues

**Total Issues**: 15
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 11
- Closed Issues: 4

### Label Distribution

- keep-open: 15 issues
- RFC: 7 issues
- feature request: 4 issues
- bug: 2 issues
- new-model: 1 issues

---

## Issue #N/A: [Bug]: Internal Server Error when hosting Alibaba-NLP/gte-Qwen2-7B-instruct

**Link**: https://github.com/vllm-project/vllm/issues/5827
**State**: closed
**Created**: 2024-06-25T14:40:21+00:00
**Closed**: 2024-11-15T04:23:11+00:00
**Comments**: 5
**Labels**: bug, keep-open

### Description

### Your current environment

Using latest available docker image: vllm/vllm-openai:v0.5.0.post1


### 🐛 Describe the bug

I am getting as response "Internal Server Error" when calling the /v1/embeddings endpoint of the Kubernetes-deployed version of the model x. I am using the following json request as body:
```json
{
  "model": "/mnt/models/",
  "input": [
    "test"
  ],
  "user": "user"
}
```

For reference, here is the log of the vLLM container:
```
INFO 06-25 14:21:47 api_server.py:177] vLLM API server version 0.5.0.post1
INFO 06-25 14:21:47 api_server.py:178] args: Namespace(host=None, port=8080, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, lora_modules=None, chat_template=None, response_role='assistant', ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_cert_reqs=0, root_path=None, middleware=[], model='/mnt/models/', tokenizer=None, skip_tokenizer_init=False, revis

[... truncated for brevity ...]

---

## Issue #N/A: [New Model]: DeepSeek VL

**Link**: https://github.com/vllm-project/vllm/issues/4982
**State**: closed
**Created**: 2024-05-22T12:54:25+00:00
**Closed**: 2025-02-25T19:04:23+00:00
**Comments**: 7
**Labels**: new-model, keep-open

### Description

### The model to consider.

https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat

### The closest model vllm already supports.

Llava

### What's your difficulty of supporting the model you want?



---

## Issue #N/A: Conda Forge Package

**Link**: https://github.com/vllm-project/vllm/issues/3126
**State**: closed
**Created**: 2024-02-29T22:11:12+00:00
**Closed**: 2025-06-19T08:00:29+00:00
**Comments**: 5
**Labels**: keep-open

### Description

Any plans to release vllm via conda-forge? It's generally much easier to manage CUDA versions, etc. with conda.

Happy to take a stab at a feedstock if there's interest but would prefer if one of the project owners could be a maintainer.

---

## Issue #N/A: Feature request: Expert parallel for MoE architectures

**Link**: https://github.com/vllm-project/vllm/issues/2405
**State**: closed
**Created**: 2024-01-10T13:20:22+00:00
**Closed**: 2025-03-11T13:55:51+00:00
**Comments**: 6
**Labels**: feature request, keep-open

### Description

Can we implement the expert parallel strategy for MoE to fully exploit the sparse activation property? Ideally, MoE should only use compute at the order of active parameters, but the current implementation uses the same compute as a dense model.

Expert parallelism is very similar to data parallelism across multiple GPUs, the only difference is that the experts are on separate GPUs and the tokens are permuted during MoE layer forward pass, as shown in the figure below.

I can help implement the MoE layer, but I'm curious how to implement data parallel with vLLM?

![Diagram](https://github.com/vllm-project/vllm/assets/26354659/c04c1e05-30c6-458e-b791-67562fabc76f)
(Diagram from [FastMoE](https://github.com/laekov/fastmoe))

---

## Issue #N/A: Migrating from `yapf` to `ruff format`

**Link**: https://github.com/vllm-project/vllm/issues/17657
**State**: open
**Created**: 2025-05-05T13:59:22+00:00
**Comments**: 0
**Labels**: RFC, keep-open

### Description

### Motivation.

We would like to transition vLLM from `yapf` to `ruff format`. This will give us:
- Increased line length
- Better formatting style which is more effectively enforced by tooling rather than by maintainers
- Fewer formatting tools fighting eachother

### Proposed Change.

We plan to make this change gradually using the following process.

If we are converting directory `x`:
- In `x`, we add a local `pyproject.toml` which:
  - overrides `ruff`'s line length to 88 (it's own default)
  - removes the deprecated type ignores
  - enables `isort` in ruff
  - enables formatting of code in docstrings (good for the API docs)
- We add `x` to the list of files to run `ruff-format` on in `.pre-commit-config.yaml`
- We add `x` to the list of ignores in the `yapf` and `isort` config in the root `pyproject.toml`

Here is the list of PRs used to make the transition:

- [x] https://github.com/vllm-project/vllm/pull/17656
- [x] https://github.com/vllm-project/vllm/pull/18068
- [x] https:/

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Refactor the logic in tool parser manager and reasoning parser manager

**Link**: https://github.com/vllm-project/vllm/issues/15658
**State**: open
**Created**: 2025-03-28T00:55:57+00:00
**Comments**: 3
**Labels**: feature request, keep-open

### Description

### 🚀 The feature, motivation and pitch

https://github.com/vllm-project/vllm/pull/14428#discussion_r2015661446 / https://github.com/vllm-project/vllm/pull/14428#discussion_r2015662511

The implementation of the tool parser manager and reasoning parser manager could be optimized.

/cc @aarnphm 

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [RFC]: Drop support for prompt adapter

**Link**: https://github.com/vllm-project/vllm/issues/13981
**State**: open
**Created**: 2025-02-27T17:58:49+00:00
**Comments**: 11
**Labels**: RFC, keep-open

### Description

### Motivation.

For code cleanup, we plan to drop the support for prompt adapter. Please let us know if you are using this feature.

### Proposed Change.

Dropping the prompt adapter and relevant code.

### Feedback Period.

2 weeks.

### CC List.

_No response_

### Any Other Things.

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [RFC]: Initial support for multi-model models using cross attention in V1

**Link**: https://github.com/vllm-project/vllm/issues/12761
**State**: open
**Created**: 2025-02-05T03:18:54+00:00
**Comments**: 4
**Labels**: RFC, keep-open

### Description

### Motivation.

The goal of this RFC is to propose a simple initial design to support multi-modal models that use cross attention for the V1 architecture.  Whisper is a prime example of such a model.  The design aims to be as simple as possible and easily replaceable without disrupting other ongoing V1 work.  Currently in V1, the only encoder/decoder models that are supported are ones that do not use cross attention.  These models use the `EncoderCacheManager` to communicate the outputs of the encode to the decoder.  Multi-modal models that use cross attention need a separate KV cache for the encoder portion of the model.  This cross attention KV cache has to be populated by the encoder and is used by the decoder's cross attention layers (as read only).  The cross attention KV cache is separate from the existing decoder KV cache and has to be managed separately.
        
### Non-goals
Since we are focusing on Whisper for the initial design, there are certain features/optimizations tha

[... truncated for brevity ...]

---

## Issue #N/A: vLLM's V1 Engine Architecture

**Link**: https://github.com/vllm-project/vllm/issues/8779
**State**: open
**Created**: 2024-09-24T18:25:22+00:00
**Comments**: 14
**Labels**: RFC, keep-open

### Description

This issues describes the high level directions that "create LLM Engine V1". We want the design to be as transparent as possible and created this issue to track progress and solicit feedback. 

Goal:
* The new engine will be simple and performant. We found the first iteration of the engine to be simple, the multistep engine to be performant, but we want best of the both worlds. For it to be performat, we want to **minimize GPU idle time**. 
* The new architecture will be extensible and modular. We found the current codebase becoming difficult to extend and add new features (both production and experimental features) due to the hard tangling of different features. In the new design, features should be compatible with each other.
* Tech debts will be cleaned up. We will remove optimizations that compromise code readability. We will also redo ad-hoc implementations to support certain features/models. 

Non-goals, the following are important but orthogonal:
* Optimize GPU time/kern

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Performance Roadmap 

**Link**: https://github.com/vllm-project/vllm/issues/6801
**State**: open
**Created**: 2024-07-25T21:32:34+00:00
**Comments**: 37
**Labels**: RFC, keep-open

### Description

### Anything you want to discuss about vllm.

This is a meta RFC tracking some of the performance enhancement works we are prioritizing. 

- [ ]  https://github.com/vllm-project/vllm/issues/6797
- [x] https://github.com/vllm-project/vllm/issues/6556
- [x] https://github.com/vllm-project/vllm/issues/6378
- [x] #6854
- [x] https://github.com/vllm-project/vllm/issues/6913

---

## Issue #N/A: [Feature]: Publish container images to additional registry (qhcr or quay.io)

**Link**: https://github.com/vllm-project/vllm/issues/6678
**State**: open
**Created**: 2024-07-23T07:42:19+00:00
**Comments**: 9
**Labels**: feature request, keep-open

### Description

### 🚀 The feature, motivation and pitch

Currently release artifacts in the form of OCI / container images are only published to DockerHub (https://hub.docker.com/r/vllm/vllm)

I like to suggest publishing the images to another registry for redundancy. The [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry) is an obvious choice, but also quay.io might be worth a look. In the end this is more or less a build once, retag and a push to another registry.

### Alternatives

_No response_

### Additional context

_No response_

---

## Issue #N/A: [RFC]: Support sparse KV cache framework

**Link**: https://github.com/vllm-project/vllm/issues/5751
**State**: open
**Created**: 2024-06-21T20:21:39+00:00
**Comments**: 17
**Labels**: RFC, keep-open

### Description

### Motivation

For current large model inference, KV cache occupies a significant portion of GPU memory, so reducing the size of KV cache is an important direction for improvement. Recently, several papers have approached this issue from different angles, detailed comparison in the table, including:

- FastDecode: This method offloads all computation of KV cache to the CPU. The computation and storage of KV cache occurs on CPU.

- Compression methods based on quantization (GEAR, Mixed Precision): By applying various quantization techniques, the size of individual token KV caches is reduced without decreasing the number of tokens stored in the KV cache. This method may also result in corresponding residual and outlier matrices, which need to be stored in memory but not in the KV cache. It may also involve quantizing unimportant token KV caches to reduce the memory footprint of the KV cache.

- Partial KV cache eviction (H2O, SnapKV, LESS, Adaptive Compression, Scissorhands, Dyn

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: prefix-caching: inconsistent completions

**Link**: https://github.com/vllm-project/vllm/issues/5543
**State**: open
**Created**: 2024-06-14T14:18:02+00:00
**Comments**: 22
**Labels**: bug, keep-open

### Description

### Your current environment

```text
vLLM version 0.5.0.post1
```




### 🐛 Describe the bug

Hi,

Seems that there is a dirty cache issue with `--enable-prefix-caching`.  We noticed it as we saw internal eval scores significantly degrade when running with `--enable-prefix-caching` and here I'll show how to reproduce it with a short snippet.

Running 2 vLLM servers with:

without prefix caching:
```bash
python -m vllm.entrypoints.openai.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 8001
```
and another with prefix caching:
```bash
python -m vllm.entrypoints.openai.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 8002 --enable-prefix-caching
```

Then running this snippet:
```python
import string 
import random

import openai

vllms = {
    "no-prefix-caching": "http://localhost:8001/v1",
    "with-prefix-caching": "http://localhost:8002/v1",
}

random.seed(0)
prompts = []
for i in range(16):
    prompts.append(''.join

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Add control panel support for vLLM

**Link**: https://github.com/vllm-project/vllm/issues/4873
**State**: open
**Created**: 2024-05-17T02:20:50+00:00
**Comments**: 12
**Labels**: RFC, keep-open

### Description

### Motivation.

The Fastchat-vLLM operational model offers significant advantages in deploying large language models (LLMs) for product services. [1](https://blog.vllm.ai/2023/06/20/vllm.html)

The controller architecture in Fastchat is particularly beneficial for LLM deployment, owing to its loosely coupled design with the vLLM backend. This allows for:

* Autoscaling: The vLLM backend can join and exit the cluster freely, enabling dynamic scaling capabilities.

* Rolling Updates: The introduction of new models with distinct names allows the cluster to gradually update models, a process known as rolling updates.

* Centralized Access: Users are relieved from the burden of tagging different URLs or IPs for various models; they simply send their requests to the controller, which then manages the rest, including dispatching requests to the appropriate backend based on the model name and ensuring effective load balancing.

However, the challenge for Fastchat lies in managing 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Build and publish Neuron docker image

**Link**: https://github.com/vllm-project/vllm/issues/4838
**State**: open
**Created**: 2024-05-15T15:27:17+00:00
**Comments**: 4
**Labels**: feature request, keep-open

### Description

### 🚀 The feature, motivation and pitch

It seems like the current docker images don't support Neuron (Inferentia).
It would be very helpful if there was a tested, managed Neuron docker image to use.
While at the same subject, it would be even better if some documentation would be added on running vLlm Neuron using containers.

### Alternatives

DJL?

### Additional context

_No response_

---

