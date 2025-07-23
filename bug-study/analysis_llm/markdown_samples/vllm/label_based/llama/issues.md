# llama - issues

**Total Issues**: 2
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 1

### Label Distribution

- llama: 2 issues
- bug: 1 issues
- torch.compile: 1 issues
- feature request: 1 issues

---

## Issue #N/A: [Bug]: Illegal memory access on llama4 maverick

**Link**: https://github.com/vllm-project/vllm/issues/19631
**State**: closed
**Created**: 2025-06-13T22:33:29+00:00
**Closed**: 2025-07-07T17:10:56+00:00
**Comments**: 9
**Labels**: bug, torch.compile, llama

### Description

### Your current environment

PyTorch 2.7.0, vLLM main branch built from source.

### 🐛 Describe the bug

Repro:
```py
vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 --tensor-parallel-size 8 --max-num-batched-tokens 40000 --max-model-len 8192 --max-num-seqs 128 --gpu-memory-utilization 0.8
```
gives a CUDA Illegal Memory Access, as well as some errors:
```
ERROR 06-13 15:32:09 [core.py:515] EngineCore failed to start.
ERROR 06-13 15:32:09 [core.py:515] Traceback (most recent call last):
ERROR 06-13 15:32:09 [core.py:515]   File "/home/rzou/dev/stable0/vllm-stable0/vllm/v1/engine/core.py", line 506, in run_engine_core
ERROR 06-13 15:32:09 [core.py:515]     engine_core = EngineCoreProc(*args, **kwargs)
ERROR 06-13 15:32:09 [core.py:515]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 06-13 15:32:09 [core.py:515]   File "/home/rzou/dev/stable0/vllm-stable0/vllm/v1/engine/core.py", line 390, in __init__
ERROR 06-13 15:32:09 [core.py:515]     super().__init__(vllm_conf

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vLLM does not serve text-only version of Llama4

**Link**: https://github.com/vllm-project/vllm/issues/18022
**State**: open
**Created**: 2025-05-12T20:23:48+00:00
**Comments**: 1
**Labels**: feature request, llama

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
Not related
```

</details>


### 🐛 Describe the bug

Hi all! 
I am trying to serve a text-only version of Llama 4 Scout (17B-16E) using vLLM. This model requires the Llama4ForCausalLM architecture. However, it seems that vLLM currently expects only the multimodal Llama 4.

Although the Llama4ForCausalLM class is implemented in vllm/model_executor/models/llama4.py, it is not registered in the _TEXT_GENERATION_MODELS dictionary in vllm/model_executor/models/registry.py. After manually adding an entry for Llama4ForCausalLM, I was able to serve the model successfully.

This looks like an oversight or a missing feature, and might be considered a bug.

For the reference, the text-only version of Llama4 was loaded and saved with AutoModelForCausalLM with the model config updated accordingly. 
```
model_config = AutoConfig.from_pretrained(config["model"]["path"], trust_remote_c

[... truncated for brevity ...]

---

