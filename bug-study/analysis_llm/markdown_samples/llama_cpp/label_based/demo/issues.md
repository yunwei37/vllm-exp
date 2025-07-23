# demo - issues

**Total Issues**: 2
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 2

### Label Distribution

- demo: 2 issues
- stale: 2 issues
- documentation: 1 issues
- enhancement: 1 issues
- server/webui: 1 issues

---

## Issue #N/A: Working Fine-Tune Example?

**Link**: https://github.com/ggml-org/llama.cpp/issues/6361
**State**: closed
**Created**: 2024-03-28T06:44:04+00:00
**Closed**: 2024-06-21T01:07:11+00:00
**Comments**: 5
**Labels**: documentation, demo, stale

### Description

I am trying to find a working example of fine-tuning. 

- If I run the example from `https://github.com/ggerganov/llama.cpp/tree/master/examples/finetune`, the script can't find the model.
here is the error
```
main: seed: 1711608198
main: model base = 'open-llama-3b-v2-q8_0.gguf'
llama_model_load: error loading model: llama_model_loader: failed to load model from open-llama-3b-v2-q8_0.gguf

llama_load_model_from_file: failed to load model
llama_new_context_with_model: model cannot be NULL
Segmentation fault: 11
```

- If I try to use a model in `/models` folder such as 
```
./finetune \
        --model-base ./models/ggml-vocab-llama.gguf \
        --checkpoint-in  chk-lora-open-llama-3b-v2-q8_0-shakespeare-LATEST.gguf \
        --checkpoint-out chk-lora-open-llama-3b-v2-q8_0-shakespeare-ITERATION.gguf \
        --lora-out lora-open-llama-3b-v2-q8_0-shakespeare-ITERATION.bin \
        --train-data "shakespeare.txt" \
        --save-every 10 \
        --threads 6 

[... truncated for brevity ...]

---

## Issue #N/A: Server: add function calling API

**Link**: https://github.com/ggml-org/llama.cpp/issues/5588
**State**: closed
**Created**: 2024-02-19T13:47:28+00:00
**Closed**: 2024-06-16T01:07:14+00:00
**Comments**: 10
**Labels**: enhancement, demo, server/webui, stale

### Description

# Motivation

This subject is already brought up in https://github.com/ggerganov/llama.cpp/issues/4216 , but my initial research failed.

Recently, I discovered a new line of model designed specifically for this usage: https://github.com/MeetKai/functionary

This model can decide whether to call functions (and which function to be called) in a given context. The chat template looks like this:

```
{#v2.2#}
{% for message in messages %}
  {% if message['role'] == 'user' or message['role'] == 'system' %}
    {{ '<|from|>' + message['role'] + '\n<|recipient|>all\n<|content|>' + message['content'] + '\n' }}
  {% elif message['role'] == 'tool' %}
    {{ '<|from|>' + message['name'] + '\n<|recipient|>all\n<|content|>' + message['content'] + '\n' }}
  {% else %}
    {% set contain_content='no'%}
    {% if message['content'] is not none %}
      {{ '<|from|>assistant\n<|recipient|>all\n<|content|>' + message['content'] }}
      {% set contain_content='yes'%}
    {% endif %}

[... truncated for brevity ...]

---

