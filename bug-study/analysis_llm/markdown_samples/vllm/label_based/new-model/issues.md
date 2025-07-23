# new-model - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 29

### Label Distribution

- new-model: 30 issues
- stale: 7 issues

---

## Issue #N/A: [New Model]: pfnet/plamo-2-8b

**Link**: https://github.com/vllm-project/vllm/issues/14214
**State**: closed
**Created**: 2025-03-04T14:59:42+00:00
**Closed**: 2025-07-11T02:16:10+00:00
**Comments**: 3
**Labels**: new-model, stale

### Description

### The model to consider.

Please add support for PFN's plamo-2-8b https://huggingface.co/pfnet/plamo-2-8b

### The closest model vllm already supports.

_No response_

### What's your difficulty of supporting the model you want?

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [New Model]: Llama4 Support

**Link**: https://github.com/vllm-project/vllm/issues/16106
**State**: closed
**Created**: 2025-04-05T21:39:19+00:00
**Closed**: 2025-04-06T04:32:40+00:00
**Comments**: 3
**Labels**: new-model

### Description

### 🚀 The feature, motivation and pitch

Meta released 2 Variants:

Llama 4 Scout:
A high-performing small model with 17B activated parameters across 16 experts. Extremely fast, natively multimodal, supports a 10M+ token context window, and runs on a single GPU.

Llama 4 Maverick:
A top-tier multimodal model outperforming GPT-4o and Gemini 2.0 Flash, with performance on par with DeepSeek V3 at half the active parameters. ELO 1417 on LMArena and runs on a single host.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [New Model]: LLaVA-NeXT-Video support

**Link**: https://github.com/vllm-project/vllm/issues/5124
**State**: closed
**Created**: 2024-05-30T03:22:17+00:00
**Closed**: 2024-09-11T05:21:37+00:00
**Comments**: 4
**Labels**: new-model

### Description

### The model to consider.

The llava-next-video project has already been released, and the test results are quite good. Are there any plans to support this project?
`https://github.com/LLaVA-VL/LLaVA-NeXT/blob/inference/docs/LLaVA-NeXT-Video.md`
Currently, Hugging Face does not support this model.

### The closest model vllm already supports.

_No response_

### What's your difficulty of supporting the model you want?

_No response_

---

## Issue #N/A: Support for Starcoder

**Link**: https://github.com/vllm-project/vllm/issues/170
**State**: closed
**Created**: 2023-06-20T23:24:39+00:00
**Closed**: 2023-06-22T18:00:45+00:00
**Comments**: 6
**Labels**: new-model

### Description

Does this work with Starcoder? The readme lists gpt-2 which is starcoder base architecture, has anyone tried it yet?

---

## Issue #N/A: supporting superhot models?

**Link**: https://github.com/vllm-project/vllm/issues/388
**State**: closed
**Created**: 2023-07-07T09:22:25+00:00
**Closed**: 2024-03-08T11:35:51+00:00
**Comments**: 3
**Labels**: new-model

### Description

specifically: 
- https://huggingface.co/kaiokendev/superhot-30b-8k-no-rlhf-test

i've confirmed that i can load the model in vllm and successfully generate completions but the context length is still only 2048. 


---

## Issue #N/A: [New Model]: Supporting DBRX from Databricks

**Link**: https://github.com/vllm-project/vllm/issues/3658
**State**: closed
**Created**: 2024-03-27T13:00:35+00:00
**Closed**: 2024-03-27T20:01:47+00:00
**Comments**: 1
**Labels**: new-model

### Description

### The model to consider.

Databricks has released [DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm), which consists of 2 models

- [dbrx](https://huggingface.co/databricks/dbrx-base)
- [dbrx-instruct](https://huggingface.co/databricks/dbrx-instruct)

It's a 132B parameter MoE model. Might be useful.

### The closest model vllm already supports.

_No response_

### What's your difficulty of supporting the model you want?

It seems that they have a custom script in their files, might need custom implementation on that regard.

---

## Issue #N/A: can model  Qwen/Qwen-VL-Chat work well?

**Link**: https://github.com/vllm-project/vllm/issues/962
**State**: closed
**Created**: 2023-09-06T10:18:59+00:00
**Closed**: 2024-09-05T12:48:12+00:00
**Comments**: 11
**Labels**: new-model

### Description

when i use Qwen/Qwen-VL-Chat  I do not know why!

throw a error 

`Traceback (most recent call last):
  File "test.py", line 20, in <module>
    model = LLM(model=model_path, tokenizer=model_path,tokenizer_mode='slow',tensor_parallel_size=1,trust_remote_code=True)
  File "/usr/local/miniconda3/lib/python3.8/site-packages/vllm/entrypoints/llm.py", line 66, in __init__
    self.llm_engine = LLMEngine.from_engine_args(engine_args)
  File "/usr/local/miniconda3/lib/python3.8/site-packages/vllm/engine/llm_engine.py", line 220, in from_engine_args
    engine = cls(*engine_configs,
  File "/usr/local/miniconda3/lib/python3.8/site-packages/vllm/engine/llm_engine.py", line 101, in __init__
    self._init_workers(distributed_init_method)
  File "/usr/local/miniconda3/lib/python3.8/site-packages/vllm/engine/llm_engine.py", line 133, in _init_workers
    self._run_workers(
  File "/usr/local/miniconda3/lib/python3.8/site-packages/vllm/engine/llm_engine.py", line 470, in _run_workers

[... truncated for brevity ...]

---

## Issue #N/A: [New Model]:Is MiniCPM-V-2_6 supported?

**Link**: https://github.com/vllm-project/vllm/issues/7267
**State**: closed
**Created**: 2024-08-07T15:03:56+00:00
**Closed**: 2024-08-08T14:02:43+00:00
**Comments**: 1
**Labels**: new-model

### Description

### The model to consider.

MiniCPM-V-2_6

### The closest model vllm already supports.

_No response_

### What's your difficulty of supporting the model you want?

_No response_

---

## Issue #N/A: [New Model]: HuggingFaceTB/SmolVLM2-2.2B-Instruct

**Link**: https://github.com/vllm-project/vllm/issues/15541
**State**: closed
**Created**: 2025-03-26T11:40:29+00:00
**Closed**: 2025-04-09T02:12:18+00:00
**Comments**: 5
**Labels**: new-model

### Description

### The model to consider.

HuggingFaceTB/SmolVLM2-2.2B-Instruct

### The closest model vllm already supports.

HuggingFaceTB/SmolVLM-256M-Instruct

### What's your difficulty of supporting the model you want?

Current error from vLLM: SmolVLMForConditionalGeneration has no vLLM implementation and the Transformers implementation is not compatible with vLLM. Try setting VLLM_USE_V1=0.

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [New Model]: allenai/Molmo-7B-0-0924 VisionLM

**Link**: https://github.com/vllm-project/vllm/issues/8808
**State**: closed
**Created**: 2024-09-25T16:34:48+00:00
**Closed**: 2024-10-14T14:56:25+00:00
**Comments**: 17
**Labels**: new-model

### Description

### The model to consider.

https://huggingface.co/allenai/Molmo-7B-O-0924
https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19

### The closest model vllm already supports.

Existing Olmo Models by AllenAi: `OLMoForCausalLM` and `OLMoEForCausalLM` are supported.

### What's your difficulty of supporting the model you want?

Molmo is a vision LM, so unlike the previous Olmo models by Allen AI, this model includes vision.

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [New Model]: janus pro support

**Link**: https://github.com/vllm-project/vllm/issues/12512
**State**: closed
**Created**: 2025-01-28T14:40:59+00:00
**Closed**: 2025-02-01T01:27:58+00:00
**Comments**: 1
**Labels**: new-model

### Description

### 🚀 The feature, motivation and pitch

kindly add jenus pro support for multi model

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [New Model]: Support for SFR-Embedding-Code-2B_R embbeding model

**Link**: https://github.com/vllm-project/vllm/issues/15362
**State**: open
**Created**: 2025-03-23T18:22:23+00:00
**Comments**: 2
**Labels**: new-model

### Description

### The model to consider.

Please add support for that  SFR-Embedding-Code-2B_R embedding model that use CodeXEmbedModel2B architecture. this embedding model is one of the best in Code Information Retrieval.

link to model https://huggingface.co/Salesforce/SFR-Embedding-Code-2B_R
link to Code Information Retrieval benchmark: https://archersama.github.io/coir/


### The closest model vllm already supports.

_No response_

### What's your difficulty of supporting the model you want?

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [New Model]: Could you please help me to support google/madlad400-3b-mt translator model in vLLM?

**Link**: https://github.com/vllm-project/vllm/issues/7930
**State**: closed
**Created**: 2024-08-27T23:32:49+00:00
**Closed**: 2024-12-28T01:59:06+00:00
**Comments**: 2
**Labels**: new-model, stale

### Description

### The model to consider.

https://huggingface.co/google/madlad400-3b-mt

### The closest model vllm already supports.

I was unable to find a model compatible with the Google model I want to implement for language translations.

### What's your difficulty of supporting the model you want?

The architecture required to mount the model is T5ForConditionalGeneration and for Tokenizer is T5Tokenizer. This architecture is not listed among those supported in vLLM.

The code I have implemented to mount the model and tokenizer is:
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
import torch

model = T5ForConditionalGeneration.from_pretrained(os.path.join(BASE_PATH,MODEL_NAME), torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = T5Tokenizer.from_pretrained(os.path.join(BASE_PATH,MODEL_NAME))

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [d

[... truncated for brevity ...]

---

## Issue #N/A: [New Model]: Phi-2 support for LoRA

**Link**: https://github.com/vllm-project/vllm/issues/3562
**State**: closed
**Created**: 2024-03-21T23:15:15+00:00
**Closed**: 2024-05-21T05:24:18+00:00
**Comments**: 0
**Labels**: new-model

### Description

### The model to consider.

Microsoft/Phi-2 with LoRA

### The closest model vllm already supports.

_No response_

### What's your difficulty of supporting the model you want?

_No response_

---

## Issue #N/A: [New Model]: https://huggingface.co/jinaai/jina-clip-v1

**Link**: https://github.com/vllm-project/vllm/issues/10197
**State**: closed
**Created**: 2024-11-10T13:04:36+00:00
**Closed**: 2025-03-31T02:09:17+00:00
**Comments**: 2
**Labels**: new-model, stale

### Description

### The model to consider.

https://huggingface.co/jinaai/jina-clip-v1

### The closest model vllm already supports.

CLIP

### What's your difficulty of supporting the model you want?

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: DeepSeekCoderV2

**Link**: https://github.com/vllm-project/vllm/issues/5763
**State**: closed
**Created**: 2024-06-22T16:15:34+00:00
**Closed**: 2024-06-28T20:24:58+00:00
**Comments**: 1
**Labels**: new-model

### Description

### The model to consider.

https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct

### The closest model vllm already supports.

DeepSeek MoE

### What's your difficulty of supporting the model you want?

Inexperienced in porting to vLLM

---

## Issue #N/A: feature request: support mpt-30b

**Link**: https://github.com/vllm-project/vllm/issues/332
**State**: closed
**Created**: 2023-07-02T09:16:18+00:00
**Closed**: 2023-07-03T23:47:56+00:00
**Comments**: 0
**Labels**: new-model

### Description

[MPT-30b]( https://huggingface.co/mosaicml/mpt-30b), the lastest model from Mosaic is setting benchmarks for being the current best single GPU LLM outthere.

Would be really cool to see mpt-30b & mpt-30b-instruct support by vLLM



---

## Issue #N/A: [New Model]: New models Gemma 3

**Link**: https://github.com/vllm-project/vllm/issues/14663
**State**: closed
**Created**: 2025-03-12T07:51:08+00:00
**Closed**: 2025-03-12T15:36:34+00:00
**Comments**: 1
**Labels**: new-model

### Description

### The model to consider.

Gemma 3 has a large, 128K context window, multilingual support in over 140 languages, and is available in more sizes than previous versions. Gemma 3 models are well-suited for a variety of text generation and image understanding tasks, including question answering, summarization, and reasoning.

Inputs and outputs
Input:

Text string, such as a question, a prompt, or a document to be summarized
Images, normalized to 896 x 896 resolution and encoded to 256 tokens each
Total input context of 128K tokens for the 4B, 12B, and 27B sizes, and 32K tokens for the 1B size
Output:

Generated text in response to the input, such as an answer to a question, analysis of image content, or a summary of a document
Total output context of 8192 tokens

https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d

https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf

### The closest model vllm already supports.

Gemma 2

### What's your dif

[... truncated for brevity ...]

---

## Issue #N/A: [New Model]: command-r7b

**Link**: https://github.com/vllm-project/vllm/issues/11650
**State**: closed
**Created**: 2024-12-31T07:28:21+00:00
**Closed**: 2025-01-02T06:46:55+00:00
**Comments**: 2
**Labels**: new-model

### Description

### The model to consider.

https://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024

### The closest model vllm already supports.

I don‘t know，but i had installe the newest transformers and newest vllm,and I had to see the history of Cohere2ForCausalLM,but it still error  after i tried again

### What's your difficulty of supporting the model you want?

ValueError: Model architectures ['CohereForCausalLM'] are not supported for now. Supported architectures: ['AquilaModel', 'AquilaForCausalLM', 'BaiChuanForCausalLM', 'BaichuanForCausalLM', 'BloomForCausalLM', 'ChatGLMModel', 'ChatGLMForConditionalGeneration', 'DeciLMForCausalLM', 'DeepseekForCausalLM', 'FalconForCausalLM', 'GemmaForCausalLM', 'GPT2LMHeadModel', 'GPTBigCodeForCausalLM', 'GPTJForCausalLM', 'GPTNeoXForCausalLM', 'InternLMForCausalLM', 'InternLM2ForCausalLM', 'LlamaForCausalLM', 'LLaMAForCausalLM', 'MistralForCausalLM', 'MixtralForCausalLM', 'QuantMixtralForCausalLM', 'MptForCausalLM', 'MPTForCausalLM', 'OLMoForCausalL

[... truncated for brevity ...]

---

## Issue #N/A: Gemma2 models from google

**Link**: https://github.com/vllm-project/vllm/issues/5953
**State**: closed
**Created**: 2024-06-28T09:16:34+00:00
**Closed**: 2024-06-28T09:53:51+00:00
**Comments**: 3
**Labels**: new-model

### Description

### The model to consider.

https://huggingface.co/google/gemma-2-27b-it 

### The closest model vllm already supports.

gemma

### What's your difficulty of supporting the model you want?

_No response_

---

## Issue #N/A: [New Model]: add GOT-OCR2

**Link**: https://github.com/vllm-project/vllm/issues/13862
**State**: closed
**Created**: 2025-02-26T02:31:43+00:00
**Closed**: 2025-06-27T02:15:30+00:00
**Comments**: 2
**Labels**: new-model, stale

### Description

### 🚀 The feature, motivation and pitch

Add an OCR model, the transformer has been added, and the effect is still quite good.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [New Model]: Babel Model, Open Multilingual Large Language Models Serving Over 90% of Global Speakers

**Link**: https://github.com/vllm-project/vllm/issues/14484
**State**: closed
**Created**: 2025-03-08T09:39:23+00:00
**Closed**: 2025-07-08T02:14:06+00:00
**Comments**: 3
**Labels**: new-model, stale

### Description

### 🚀 The feature, motivation and pitch

 Alibaba DAMO team released A multilingual LLM supporting 25 languages.

Model: https://huggingface.co/Tower-Babel/Babel-83B-Chat
can we add support to this

### Alternatives

_No response_

### Additional context

collection 
https://huggingface.co/collections/Tower-Babel/babel-67c172157372d4d6c4b4c6d5

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [New Model]: Support for allenai/OLMoE-1B-7B-0924

**Link**: https://github.com/vllm-project/vllm/issues/8170
**State**: closed
**Created**: 2024-09-04T21:55:46+00:00
**Closed**: 2025-01-23T01:58:36+00:00
**Comments**: 4
**Labels**: new-model, stale

### Description

### The model to consider.

allenai/OLMoE-1B-7B-0924

### The closest model vllm already supports.

_No response_

### What's your difficulty of supporting the model you want?

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [New Model]: dunsloth/DeepSeek-R1-GGUF

**Link**: https://github.com/vllm-project/vllm/issues/13877
**State**: closed
**Created**: 2025-02-26T07:11:16+00:00
**Closed**: 2025-06-27T02:15:20+00:00
**Comments**: 2
**Labels**: new-model, stale

### Description

### The model to consider.

when I run dunsloth/DeepSeek-R1-GGUF model, it raise error  GGUF model with architecture deepseek2 is not supported yet

### The closest model vllm already supports.

_No response_

### What's your difficulty of supporting the model you want?

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Support Nemotron

**Link**: https://github.com/vllm-project/vllm/issues/1686
**State**: closed
**Created**: 2023-11-16T16:48:30+00:00
**Closed**: 2024-03-13T13:10:42+00:00
**Comments**: 1
**Labels**: new-model

### Description

Can we get support for Nemotron ?  https://huggingface.co/nvidia/nemotron-3-8b-base-4k

---

## Issue #N/A: [New Model]: ValueError: Model architectures ['PhiMoEForCausalLM'] are not supported for now

**Link**: https://github.com/vllm-project/vllm/issues/7731
**State**: closed
**Created**: 2024-08-21T10:13:26+00:00
**Closed**: 2024-08-30T19:42:58+00:00
**Comments**: 1
**Labels**: new-model

### Description

### The model to consider.

PhiMoEForCausalLM

### The closest model vllm already supports.

Phi3ForCausalLM

### What's your difficulty of supporting the model you want?

_No response_

---

## Issue #N/A: [New Model]: jinaai/jina-reranker-v2-base-multilingual

**Link**: https://github.com/vllm-project/vllm/issues/15222
**State**: closed
**Created**: 2025-03-20T14:41:08+00:00
**Closed**: 2025-04-01T15:32:27+00:00
**Comments**: 1
**Labels**: new-model

### Description

### The model to consider.

https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual




### The closest model vllm already supports.

XLMRobertaForSequenceClassification  

### What's your difficulty of supporting the model you want?

The jinaai's XLMRoberta implementation is different from vllm's current implementation. 
When I try to using vllm to load jina-reranker-v2-base-multilingual . Exception occurred as following:

```python
[rank0]:   File "/home/xiayubin/.local/share/virtualenvs/test_project-krbMYW6A/lib/python3.11/site-packages/vllm/model_executor/models/roberta.py", line 224, in load_weights
[rank0]:     self.roberta.load_weights(bert_weights)
[rank0]:   File "/home/xiayubin/.local/share/virtualenvs/test_project-krbMYW6A/lib/python3.11/site-packages/vllm/model_executor/models/bert.py", line 381, in load_weights
[rank0]:     param = params_dict[name]
[rank0]:             ~~~~~~~~~~~^^^^^^
[rank0]: KeyError: 'emb_ln.weight'
Loading safetensors checkpoint shards:   0

[... truncated for brevity ...]

---

## Issue #N/A: [New Model]: Cohere2 (Command R7B)

**Link**: https://github.com/vllm-project/vllm/issues/11181
**State**: closed
**Created**: 2024-12-13T20:07:52+00:00
**Closed**: 2024-12-16T11:10:57+00:00
**Comments**: 1
**Labels**: new-model

### Description

### The model to consider.

https://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024

### The closest model vllm already supports.

Likely either the original Cohere (for. obvious reasons) or Gemma2 (as it also has a funky SWA architecture)

### What's your difficulty of supporting the model you want?

It uses SWA, but this can likely be ditched to get MVP inference working ala how gemma 2 was done
For some reason every 4th layer uses global attention _without_ positional embeddings? Not sure how or why that one works tbh

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [New Model]: Phi-3-medium-128k-instruct support

**Link**: https://github.com/vllm-project/vllm/issues/4953
**State**: closed
**Created**: 2024-05-21T15:58:44+00:00
**Closed**: 2024-05-31T21:30:34+00:00
**Comments**: 8
**Labels**: new-model

### Description

### The model to consider.

https://huggingface.co/microsoft/Phi-3-medium-128k-instruct

### The closest model vllm already supports.

The older phi model (including phi-3-mini) has been supported

### What's your difficulty of supporting the model you want?

I run into the following error on a 4*A6000 server:
```
[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/hu381/miniconda3/envs/py310/lib/python3.10/runpy.py", line 196, in _run_module_as_main
[rank0]:     return _run_code(code, main_globals, None,
[rank0]:   File "/home/hu381/miniconda3/envs/py310/lib/python3.10/runpy.py", line 86, in _run_code
[rank0]:     exec(code, run_globals)
[rank0]:   File "/home/hu381/miniconda3/envs/py310/lib/python3.10/site-packages/vllm/entrypoints/openai/api_server.py", line 168, in <module>
[rank0]:     engine = AsyncLLMEngine.from_engine_args(
[rank0]:   File "/home/hu381/miniconda3/envs/py310/lib/python3.10/site-packages/vllm/engine/async_llm_engine.py", line 366, in from_

[... truncated for brevity ...]

---

## Issue #N/A: [New Model]: tiiuae/falcon-11B

**Link**: https://github.com/vllm-project/vllm/issues/5010
**State**: closed
**Created**: 2024-05-23T14:33:35+00:00
**Closed**: 2024-05-27T23:41:44+00:00
**Comments**: 0
**Labels**: new-model

### Description

### The model to consider.

https://huggingface.co/tiiuae/falcon-11B

### The closest model vllm already supports.

tiiuae/falcon-7b
tiiuae/falcon-40b

### What's your difficulty of supporting the model you want?

### 🚀 The feature, motivation and pitch

[Falcon-11B](https://huggingface.co/tiiuae/falcon-11B) is trained on multilingual data. There is a lot of potential to serve this model where these languages are preferred. Functional, working inference in fp16 would be a great addition in my opinion.

### Additional context

The main architectural changes between the two configurations of the Falcon model are:

1. New Decoder Architecture:
   - Falcon-7B has `new_decoder_architecture: false`, which means it uses the original or a previous version of the decoder architecture.
   - Falcon-11B specifies `new_decoder_architecture: true`, indicating a newer version of the decoder architecture.

2.  Number of Attention Heads:
   - Falcon-7B uses `num_attention_heads:

[... truncated for brevity ...]

---

