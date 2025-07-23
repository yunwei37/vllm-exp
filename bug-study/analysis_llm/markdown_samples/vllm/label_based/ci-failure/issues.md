# ci-failure - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 4
- Closed Issues: 26

### Label Distribution

- ci-failure: 30 issues
- bug: 14 issues

---

## Issue #N/A: [Bug][Failing Test]: Distributed Tests (A100) - distributed/test_ca_buffer_sharing.py

**Link**: https://github.com/vllm-project/vllm/issues/18589
**State**: open
**Created**: 2025-05-23T04:35:17+00:00
**Comments**: 1
**Labels**: bug, ci-failure

### Description

### Your current environment

N/A

### 🐛 Describe the bug

https://buildkite.com/vllm/ci/builds/20544/steps?jid=0196f845-c352-4203-a55f-efb442b65c7d

cc @robertgshaw2-redhat 

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [CI Failure]: lint-and-deploy: unexpected HTTP status 500

**Link**: https://github.com/vllm-project/vllm/issues/19574
**State**: closed
**Created**: 2025-06-12T18:52:35+00:00
**Closed**: 2025-06-13T14:37:47+00:00
**Comments**: 2
**Labels**: ci-failure

### Description

### Name of failing test

lint-and-deploy

### Basic information

- [ ] Flaky test
- [ ] Can reproduce locally
- [ ] Caused by external libraries (e.g. bug in `transformers`)

### 🧪 Describe the failing test

https://github.com/vllm-project/vllm/actions/runs/15618574945/job/43997475275?pr=19573

https://github.com/vllm-project/vllm/actions/runs/15618287178/job/43996526081?pr=19572

```bash
Run helm/chart-testing-action@0d28d3144d3a25ea2cc349d6e59901c4ff469b3b
Run sigstore/cosign-installer@dc72c7d5c4d10cd6bcb8cf6e3fd625a9e5e537da
Run #!/bin/bash
INFO: Downloading bootstrap version 'v2.4.1' of cosign to verify version to be installed...
      https://github.com/sigstore/cosign/releases/download/v2.4.1/cosign-linux-amd64
INFO: bootstrap version successfully verified and matches requested version so nothing else to do
Run echo "$HOME/.cosign" >> $GITHUB_PATH
Run cd $GITHUB_ACTION_PATH \
Installing chart-testing v3.10.1...
Error: getting Rekor public keys: updating local metadata and target

[... truncated for brevity ...]

---

## Issue #N/A: [Bug][Failing Test] distributed tests (4 GPUS) - v1/test_async_llm_dp.py::test_load

**Link**: https://github.com/vllm-project/vllm/issues/18466
**State**: closed
**Created**: 2025-05-21T07:46:41+00:00
**Closed**: 2025-05-22T13:48:58+00:00
**Comments**: 6
**Labels**: bug, ci-failure

### Description

### Your current environment

Still failing on main as of commit 0c15c2e486

### 🐛 Describe the bug

Failing tests: https://buildkite.com/organizations/vllm/analytics/suites/ci-1/tests?branch=main&period=2days&query=test_async_llm_dp&commit=Search

```
FAILED v1/test_async_llm_dp.py::test_load[RequestOutputKind.DELTA] - vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.
FAILED v1/test_async_llm_dp.py::test_load[RequestOutputKind.FINAL_ONLY] - vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.
```

<details>

<summary>Logs</summary>

```
(EngineCore_0 pid=4396) (VllmWorker rank=1 pid=4418) WARNING 05-20 22:23:45 [fused_moe.py:682] Using default MoE config. Performance might be sub-optimal! Config file not found at /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/E=40,N=128,device_name=NVIDIA_L4.json
(EngineCore_1 pi

[... truncated for brevity ...]

---

## Issue #N/A: [Bug][Failing Test]  2-node-tests-4-gpus-in-total - distributed/test_pipeline_parallel.py::test_tp_*

**Link**: https://github.com/vllm-project/vllm/issues/18417
**State**: closed
**Created**: 2025-05-20T15:19:47+00:00
**Closed**: 2025-05-22T13:48:58+00:00
**Comments**: 4
**Labels**: bug, ci-failure

### Description

### Your current environment

Still failing on main as of commit 9609327fa4

### 🐛 Describe the bug

Failing test: https://buildkite.com/organizations/vllm/analytics/suites/ci-1/tests?branch=main&commit=Search&period=1day&query=test_tp_language_generation

```
FAILED distributed/test_pipeline_parallel.py::test_tp_language_generation[microsoft/Phi-3.5-MoE-instruct-parallel_setup26-ray-1-auto-test_options26]
```

<details>
<summary>Logs</summary>


```
[2025-05-20T05:24:25Z] (VllmWorker rank=0 pid=10229) WARNING 05-19 22:24:25 [fused_moe.py:682] Using default MoE config. Performance might be sub-optimal! Config file not found at /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/E=16,N=800,device_name=NVIDIA_L4.json
[2025-05-20T05:24:27Z] (VllmWorker rank=0 pid=10229) INFO 05-19 22:24:27 [monitor.py:33] torch.compile takes 10.50 s in total
[2025-05-20T05:24:28Z] (VllmWorker rank=0 pid=10229) ERROR 05-19 22:24:28 [multiproc_executor.py:522] WorkerProc hit

[... truncated for brevity ...]

---

## Issue #N/A: [CI Failure]: LM Eval Large Models - test_lm_eval_correctness.py

**Link**: https://github.com/vllm-project/vllm/issues/18766
**State**: closed
**Created**: 2025-05-27T15:53:35+00:00
**Closed**: 2025-05-28T08:59:41+00:00
**Comments**: 2
**Labels**: ci-failure

### Description

### Name of failing test

`test_lm_eval_correctness.py::test_lm_eval_correctness_param[config_filename4]`

### Basic information

- [ ] Flaky test
- [ ] Can reproduce locally
- [ ] Caused by external libraries (e.g. bug in `transformers`)

### 🧪 Describe the failing test

```
gsm8k | exact_match,strict-match: ground_truth=0.671 | measured=0.355
gsm8k | exact_match,flexible-extract: ground_truth=0.664 | measured=0.356
```

Full log: https://buildkite.com/vllm/ci/builds/20837/summary/annotations?sid=01970fe6-97da-4bd8-b139-30d20cf3912f

### 📝 History of failing test

Not sure

### CC List.

cc @robertgshaw2-redhat @mgoin 

---

## Issue #N/A: [Bug][Failing Test]: weight-loading-multiple-gpu-test -

**Link**: https://github.com/vllm-project/vllm/issues/18416
**State**: closed
**Created**: 2025-05-20T15:11:52+00:00
**Closed**: 2025-05-22T13:48:57+00:00
**Comments**: 5
**Labels**: bug, ci-failure

### Description

### Your current environment

Still failing on main as of commit bca55b556f

### 🐛 Describe the bug

Failing test: https://buildkite.com/organizations/vllm/analytics/suites/ci-1/tests/6873a23f-c2ec-8c01-9e20-bac3329482c0?tags=scm.branch%3Amain%2Cresult%3Afailed

```
FAILED weight_loading/test_weight_loading.py::test_weight_loading - RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {'EngineCore_0': 1}
```

<details>
<summary>Logs</summary>

```
[2025-05-20T10:46:52Z] (VllmWorker rank=0 pid=12189) INFO 05-20 03:46:52 [backends.py:172] Compiling a graph for general shape takes 20.71 s
[2025-05-20T10:46:52Z] (VllmWorker rank=0 pid=12189) DEBUG 05-20 03:46:52 [backends.py:512] Computation graph saved to /root/.cache/vllm/torch_compile_cache/07e0a984e7/rank_0_0/computation_graph.py
[2025-05-20T10:46:55Z] (VllmWorker rank=1 pid=12191) DEBUG 05-20 03:46:55 [wrapper.py:105] Dynamo transformed code saved to /root/.cache/vllm/torch_compile_cache/07e0a984

[... truncated for brevity ...]

---

## Issue #N/A: [Bug][Failing Test]: Multi-Modal Models 3 - models/multimodal/generation/test_common.py

**Link**: https://github.com/vllm-project/vllm/issues/18528
**State**: closed
**Created**: 2025-05-22T06:03:59+00:00
**Closed**: 2025-05-23T01:55:57+00:00
**Comments**: 7
**Labels**: bug, ci-failure

### Description

### Your current environment

N/A

### 🐛 Describe the bug

`models/multimodal/generation/test_common.py::test_single_image_models[gemma3-test_case91]` is failing on main. It is another illegal memory access error.

https://buildkite.com/vllm/ci/builds/20503/steps?jid=0196f626-d4d6-4af6-b10f-da8c3145ddfc

Stack:
```
[2025-05-22T05:33:18Z] ERROR 05-21 22:33:18 [dump_input.py:68] Dumping input data
--- Logging error ---
[2025-05-22T05:33:18Z] Traceback (most recent call last):
[2025-05-22T05:33:18Z]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 207, in execute_model
[2025-05-22T05:33:18Z]     return self.model_executor.execute_model(scheduler_output)
[2025-05-22T05:33:18Z]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[2025-05-22T05:33:18Z]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/abstract.py", line 86, in execute_model
[2025-05-22T05:33:18Z]     output = self.collective_rpc("execute_model",
[2025-05-22T05:33:18Z]   

[... truncated for brevity ...]

---

## Issue #N/A: [CI Failure]: Distributed Tests (2 GPUs) - v1/test_async_llm_dp.py::test_load

**Link**: https://github.com/vllm-project/vllm/issues/19731
**State**: closed
**Created**: 2025-06-17T07:53:51+00:00
**Closed**: 2025-06-17T20:59:30+00:00
**Comments**: 1
**Labels**: ci-failure

### Description

### Name of failing test

TP_SIZE=1 DP_SIZE=2 pytest -s -v "v1/test_async_llm_dp.py::test_load[ray-RequestOutputKind.DELTA]"

### Basic information

- [ ] Flaky test
- [x] Can reproduce locally
- [ ] Caused by external libraries (e.g. bug in `transformers`)

### 🧪 Describe the failing test

The error ends up looking like a triton bug with `AttributeError: module 'triton.language' has no attribute 'bfloat16'` reported, however very early in the logs you can see the following:
```
INFO 06-17 07:32:31 [utils.py:384] Creating placement groups for data parallel
(pid=3893316) INFO 06-17 07:32:33 [importing.py:27] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
(pid=3893316) INFO 06-17 07:32:33 [importing.py:47] Triton not installed or not compatible; certain GPU-related functions will not be available.
(pid=3893316) WARNING 06-17 07:32:33 [importing.py:59] Triton is not installed. Using dummy decorators. Install it via `pip install t

[... truncated for brevity ...]

---

## Issue #N/A: [CI Failure]: Plugin Tests (2 GPUs) - models/test_oot_registration.py

**Link**: https://github.com/vllm-project/vllm/issues/20148
**State**: closed
**Created**: 2025-06-26T20:12:49+00:00
**Closed**: 2025-06-27T03:21:05+00:00
**Comments**: 0
**Labels**: ci-failure

### Description

### Name of failing test

`models/test_oot_registration.py::test_oot_registration_embedding`

### Basic information

- [ ] Flaky test
- [x] Can reproduce locally
- [ ] Caused by external libraries (e.g. bug in `transformers`)

### 🧪 Describe the failing test

The `models/test_oot_registration.py::test_oot_registration_embedding` test seems to be failing in CI consistently with a context length OOM

https://buildkite.com/vllm/ci/builds/22737/steps/canvas?sid=0197acae-970a-43ee-9fef-108d8a58da0c#0197acae-98db-423d-8af9-eb4eb401f1b4/212-1320

```
[2025-06-26T16:27:15Z] ERROR 06-26 09:27:15 [core.py:519] ValueError: To serve at least one request with the models's max seq len (8192), (2.63 GiB KV cache is needed, which is larger than the available KV cache memory (1.64 GiB). Based on the available memory, the estimated maximum model length is 5088. Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.
```

### 📝 History of failing test

Not sure

[... truncated for brevity ...]

---

## Issue #N/A: [Bug][CI Failure] - VI Test - test_engine_core_client.py::test_kv_cache_events[True-tcp]

**Link**: https://github.com/vllm-project/vllm/issues/18708
**State**: closed
**Created**: 2025-05-26T11:38:43+00:00
**Closed**: 2025-06-04T12:57:32+00:00
**Comments**: 2
**Labels**: bug, ci-failure

### Description

### Your current environment

Flakey test for at least the past month: https://buildkite.com/organizations/vllm/analytics/suites/ci-1/tests/4abfbf0d-3a86-8a68-9ff3-0e0ab0fbb38b?period=28days&tags=scm.branch%3Amain%2Cresult%3Afailed

### 🐛 Describe the bug

Failing tests:

```
FAILED v1/engine/test_engine_core_client.py::test_kv_cache_events[True-tcp] - AssertionError: No message received
assert None is not None
```

<details>
<summary>Logs:</summary>

```
=================================== FAILURES ===================================
________________________ test_kv_cache_events[True-tcp] ________________________

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7fc027da70e0>
multiprocessing_mode = True
publisher_config = KVEventsConfig(enable_kv_cache_events=True, publisher='zmq', endpoint='tcp://*:51905', replay_endpoint='tcp://*:51906', buffer_steps=100, hwm=1000, max_queue_size=100000, topic='test')

    @pytest.mark.parametrize(
        "multiprocessing_mode,publisher_c

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Distributed Tests (4 GPUs) failing in main branch CI

**Link**: https://github.com/vllm-project/vllm/issues/20138
**State**: closed
**Created**: 2025-06-26T16:15:15+00:00
**Closed**: 2025-06-28T05:50:01+00:00
**Comments**: 7
**Labels**: bug, ci-failure

### Description

This is now consistently failing with CUDA OOM: https://buildkite.com/vllm/ci/builds/22221#01977f3a-71ea-41cb-bbeb-a43340a10124

I narrowed this down to https://github.com/vllm-project/vllm/pull/19572 which appears to have introduced the issue.



---

## Issue #N/A: [CI Failure]: Quantized Models Test - models/quantization/test_gguf.py::test_models[1-5-32-half-model0]

**Link**: https://github.com/vllm-project/vllm/issues/19458
**State**: open
**Created**: 2025-06-11T01:22:45+00:00
**Comments**: 1
**Labels**: ci-failure

### Description

### Name of failing test

`models/quantization/test_gguf.py::test_models[1-5-32-half-model0]`

### Basic information

- [x] Flaky test
- [ ] Can reproduce locally
- [ ] Caused by external libraries (e.g. bug in `transformers`)

### 🧪 Describe the failing test

This specific Llama 1B GGUF model test has been failing consistently in multiple PRs https://buildkite.com/vllm/ci/builds/21800/steps/waterfall?jid=01975af4-f581-4d43-a1e5-7175d960b2b7#01975af4-f581-4d43-a1e5-7175d960b2b7/212-6971

```

[2025-06-10T18:40:56Z] FAILED models/quantization/test_gguf.py::test_models[1-5-32-half-model0] - AssertionError: Test0:
[2025-06-10T18:40:56Z] Matched tokens:	[4897, 596, 4495, 13, 650, 4178, 44, 13656, 369]
[2025-06-10T18:40:56Z] original:	"That's correct. VLLM stands for Vision and Language Model, which is a type of large language model designed for both inference and serving. It's a"	{31541: Logprob(logprob=-1.6094070672988892, rank=1, decoded_token='ĠVision'), 28968: Logprob(logprob=-2.000031

[... truncated for brevity ...]

---

## Issue #N/A: [CI Failure]: Async Engine, Inputs, Utils, Worker Test: 'State' object has no attribute 'enable_server_load_tracking'

**Link**: https://github.com/vllm-project/vllm/issues/20842
**State**: closed
**Created**: 2025-07-11T20:38:10+00:00
**Closed**: 2025-07-12T01:57:25+00:00
**Comments**: 0
**Labels**: ci-failure

### Description

### Name of failing test

Async Engine, Inputs, Utils, Worker Test

### Basic information

- [ ] Flaky test
- [ ] Can reproduce locally
- [ ] Caused by external libraries (e.g. bug in `transformers`)

### 🧪 Describe the failing test

```bash
[2025-07-11T20:13:34Z] INFO 07-11 13:13:34 [async_llm_engine.py:222] Aborted request 85bac0a6a206462aadb2f9d86b92b5f6.
--
  | [2025-07-11T20:13:34Z] Task exception was never retrieved
  | [2025-07-11T20:13:34Z] future: <Task finished name='Task-456' coro=<listen_for_disconnect() done, defined at /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/utils.py:31> exception=AttributeError("'State' object has no attribute 'enable_server_load_tracking'")>
  | [2025-07-11T20:13:34Z] Traceback (most recent call last):
  | [2025-07-11T20:13:34Z]   File "/usr/local/lib/python3.12/dist-packages/starlette/datastructures.py", line 668, in __getattr__
  | [2025-07-11T20:13:34Z]     return self._state[key]
  | [2025-07-11T20:13:34Z]            ~~~~~~~~~~~^^^^

[... truncated for brevity ...]

---

## Issue #N/A: [CI Failure]: Speculative decoding tests - spec_decode/e2e/test_eagle_correctness.py

**Link**: https://github.com/vllm-project/vllm/issues/20214
**State**: closed
**Created**: 2025-06-28T16:36:12+00:00
**Closed**: 2025-06-29T02:31:39+00:00
**Comments**: 0
**Labels**: ci-failure

### Description

### Name of failing test

`spec_decode/e2e/test_eagle_correctness.py::test_llama3_eagle_e2e_greedy_correctness[1-1-32-test_llm_kwargs0-baseline_llm_kwargs0-per_test_common_llm_kwargs0-common_llm_kwargs0]`

### Basic information

- [ ] Flaky test
- [ ] Can reproduce locally
- [ ] Caused by external libraries (e.g. bug in `transformers`)

### 🧪 Describe the failing test

It doesn't fail locally but that might be because the OOM is specific to the L4 we use in CI

https://buildkite.com/vllm/ci/builds/22853/steps/canvas?jid=0197b520-e1dc-4ace-bfdc-f483b4dee76f
```
[2025-06-28T09:19:58Z] FAILED spec_decode/e2e/test_eagle_correctness.py::test_llama3_eagle_e2e_greedy_correctness[1-1-32-test_llm_kwargs0-baseline_llm_kwargs0-per_test_common_llm_kwargs0-common_llm_kwargs0] - torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 116.00 MiB. GPU 0 has a total capacity of 22.05 GiB of which 112.12 MiB is free. Including non-PyTorch memory, this process has 21.92 GiB memory in use. Of the al

[... truncated for brevity ...]

---

## Issue #N/A: [Bug][Failing Test] entrypoints-test - test_v1_v2_api_consistency_single_prompt_tokens

**Link**: https://github.com/vllm-project/vllm/issues/18418
**State**: closed
**Created**: 2025-05-20T15:30:23+00:00
**Closed**: 2025-05-22T13:48:58+00:00
**Comments**: 1
**Labels**: bug, ci-failure

### Description

### Your current environment

Still failing on main as of commit bca55b556f

### 🐛 Describe the bug

Failing tests: https://buildkite.com/organizations/vllm/analytics/suites/ci-1/tests?branch=main&period=2days&query=test_v1_v2_api_consistency_single_prompt_tokens&commit=Search

```
FAILED entrypoints/llm/test_generate.py::test_v1_v2_api_consistency_single_prompt_tokens[prompt_token_ids0] - vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.
FAILED entrypoints/llm/test_generate.py::test_v1_v2_api_consistency_single_prompt_tokens[prompt_token_ids1] - vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.
FAILED entrypoints/llm/test_generate.py::test_v1_v2_api_consistency_single_prompt_tokens[prompt_token_ids2] - vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.
FAILED entrypoints/llm/test

[... truncated for brevity ...]

---

## Issue #N/A: [CI Failure]: Spec Decoding - spec_decode/e2e/test_multistep_correctness.py

**Link**: https://github.com/vllm-project/vllm/issues/18954
**State**: closed
**Created**: 2025-05-30T11:27:47+00:00
**Closed**: 2025-06-16T23:43:08+00:00
**Comments**: 7
**Labels**: ci-failure

### Description

### Name of failing test

`test_spec_decode_e2e_greedy_correctness_tiny_model_large_bs_diff_output_len`

### Basic information

- [ ] Flaky test
- [x] Can reproduce locally
- [ ] Caused by external libraries (e.g. bug in `transformers`)

### 🧪 Describe the failing test

https://buildkite.com/vllm/ci/builds/21085/steps?jid=01971f59-be20-45f4-9e11-dcd3b1e67173

### 📝 History of failing test

Started failing since yesterday since today's nightly caught the failure

https://buildkite.com/organizations/vllm/analytics/suites/ci-1/tests/a8e2f4a9-a7cf-81ec-ba3e-3a1e8b022a79?period=1day&tags=scm.branch%3Amain

### CC List.

@mgoin @njhill 

---

## Issue #N/A: [Bug][Flaky]: V1 Test - v1/engine/test_engine_core_client.py

**Link**: https://github.com/vllm-project/vllm/issues/18604
**State**: closed
**Created**: 2025-05-23T10:42:19+00:00
**Closed**: 2025-06-04T12:58:08+00:00
**Comments**: 1
**Labels**: bug, ci-failure

### Description

### Your current environment

N/A

### 🐛 Describe the bug

The test `v1/engine/test_engine_core_client.py::test_kv_cache_events[True-tcp]` is flaky.

https://buildkite.com/organizations/vllm/analytics/suites/ci-1/tests/4abfbf0d-3a86-8a68-9ff3-0e0ab0fbb38b?period=7days&tags=scm.branch%3Amain

cc @robertgshaw2-redhat @njhill 





### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [CI Failure]: Basic Models Test - test_can_initialize[MiniMaxText01ForCausalLM]

**Link**: https://github.com/vllm-project/vllm/issues/20198
**State**: closed
**Created**: 2025-06-27T20:22:36+00:00
**Closed**: 2025-06-28T05:43:08+00:00
**Comments**: 1
**Labels**: ci-failure

### Description

### Name of failing test

`models/test_initialization.py::test_can_initialize[MiniMaxText01ForCausalLM]`

### Basic information

- [ ] Flaky test
- [x] Can reproduce locally
- [x] Caused by external libraries (e.g. bug in `transformers`)

### 🧪 Describe the failing test

It seems to be related to recent changes to Transformers for minimax

This is the error I see on `transformers==4.52.4` which matches the CI

```
[2025-06-27T18:12:39Z] FAILED models/test_initialization.py::test_can_initialize[MiniMaxText01ForCausalLM] - pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
[2025-06-27T18:12:39Z]   Value error, The checkpoint you are trying to load has model type `minimax` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.
[2025-06-27T18:12:39Z]
[2025-06-27T18:12:39Z] You can update Transformers with the command `pip install --upgrade transformer

[... truncated for brevity ...]

---

## Issue #N/A: [CI Failure]: Samplers Test - samplers/test_beam_search.py::test_beam_search_passes_multimodal_data

**Link**: https://github.com/vllm-project/vllm/issues/19736
**State**: closed
**Created**: 2025-06-17T09:23:35+00:00
**Closed**: 2025-06-18T22:48:30+00:00
**Comments**: 0
**Labels**: ci-failure

### Description

### Name of failing test

`samplers/test_beam_search.py::test_beam_search_passes_multimodal_data[False-2-64-half]`

### Basic information

- [ ] Flaky test
- [x] Can reproduce locally
- [ ] Caused by external libraries (e.g. bug in `transformers`)

### 🧪 Describe the failing test

It seems the issue is because we are now passing empty lists to _flatten_embeddings

```
FAILED samplers/test_beam_search.py::test_beam_search_passes_multimodal_data[False-2-64-half] - RuntimeError: torch.cat(): expected a non-empty list of Tensors
```

Full output:
```
pytest -s -v "samplers/test_beam_search.py::test_beam_search_passes_multimodal_data[False-2-64-half]"
INFO 06-17 09:19:56 [__init__.py:244] Automatically detected platform cuda.
/home/mgoin/venvs/vllm/lib/python3.12/site-packages/pytest_asyncio/plugin.py:208: PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset.
The event loop scope for asynchronous fixtures will default to the fixture caching scope.

[... truncated for brevity ...]

---

## Issue #N/A: [CI Failure]: LoRA TP Test (Distributed) - lora/test_llama_tp.py::test_tp2_serialize_and_deserialize_lora

**Link**: https://github.com/vllm-project/vllm/issues/20723
**State**: closed
**Created**: 2025-07-10T00:26:10+00:00
**Closed**: 2025-07-10T19:07:08+00:00
**Comments**: 2
**Labels**: ci-failure

### Description

### Name of failing test

`lora/test_llama_tp.py::test_tp2_serialize_and_deserialize_lora`

### Basic information

- [ ] Flaky test
- [x] Can reproduce locally
- [ ] Caused by external libraries (e.g. bug in `transformers`)

### 🧪 Describe the failing test

https://buildkite.com/vllm/ci/builds/23536/steps/canvas?sid=0197f0f3-a191-49c0-aef5-89d61c597808

```
[2025-07-09T23:17:11Z] (VllmWorker rank=0 pid=11292) WARNING 07-09 16:17:11 [tensorizer.py:226] Provided both tensorizer_dir and tensorizer_uri. Inferring tensorizer_dir from tensorizer_uri as the latter takes precedence.

[2025-07-09T23:17:11Z] (VllmWorker rank=0 pid=11292) ERROR 07-09 16:17:11 [multiproc_executor.py:487] Traceback (most recent call last):
[2025-07-09T23:17:11Z] (VllmWorker rank=0 pid=11292) ERROR 07-09 16:17:11 [multiproc_executor.py:487]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/multiproc_executor.py", line 461, in worker_main
[2025-07-09T23:17:11Z] (VllmWorker rank=0 pid=11292) ERROR 07-09

[... truncated for brevity ...]

---

## Issue #N/A: [Bug][Failing Test]: Distributed Comm Ops - distributed/test_shm_broadcast.py

**Link**: https://github.com/vllm-project/vllm/issues/18492
**State**: closed
**Created**: 2025-05-21T14:53:55+00:00
**Closed**: 2025-05-22T03:19:14+00:00
**Comments**: 5
**Labels**: bug, ci-failure

### Description

### Your current environment

pytest -v -x distributed/test_shm_broadcast.py

https://buildkite.com/vllm/ci/builds/20415#0196f100-f85c-4db6-8b50-72d3d5ade137/197-990


### 🐛 Describe the bug

See above

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [CI Failure]: Language Models Test (Extended Pooling)

**Link**: https://github.com/vllm-project/vllm/issues/20461
**State**: closed
**Created**: 2025-07-04T02:56:06+00:00
**Closed**: 2025-07-06T21:01:49+00:00
**Comments**: 9
**Labels**: ci-failure

### Description

### Name of failing test

See below

### Basic information

- [ ] Flaky test
- [ ] Can reproduce locally
- [ ] Caused by external libraries (e.g. bug in `transformers`)

### 🧪 Describe the failing test

Remaining failures:
```
FAILED models/language/pooling/test_scoring.py::test_cross_encoder_1_to_1[cross-encoder/ms-marco-MiniLM-L-6-v2] - assert 9.265625 == 1.0 ± 1.0e-02
  comparison failed
  Obtained: 9.265625
  Expected: 1.0 ± 1.0e-02
FAILED models/language/pooling/test_scoring.py::test_cross_encoder_1_to_N[cross-encoder/ms-marco-MiniLM-L-6-v2] - assert 9.265625 == 1.0 ± 1.0e-02
  comparison failed
  Obtained: 9.265625
  Expected: 1.0 ± 1.0e-02
FAILED models/language/pooling/test_scoring.py::test_cross_encoder_N_to_N[cross-encoder/ms-marco-MiniLM-L-6-v2] - assert 9.265625 == 1.0 ± 1.0e-02
  comparison failed
  Obtained: 9.265625
  Expected: 1.0 ± 1.0e-02
```

Fixed by #20168:
```
FAILED models/language/pooling/test_embedding.py::test_models[False-sentence-transformers/all-MiniLM-L12-

[... truncated for brevity ...]

---

## Issue #N/A: [CI Failure]: `deepseek_mtp_main_random_bf16` can't load, causes deepseek_mtp CI Failure.

**Link**: https://github.com/vllm-project/vllm/issues/20158
**State**: open
**Created**: 2025-06-27T01:42:24+00:00
**Comments**: 1
**Labels**: ci-failure

### Description

### Name of failing test

`https://github.com/vllm-project/vllm-ascend/actions/runs/15890661413/job/44812465270?pr=1128`

### Basic information

- [ ] Flaky test
- [x] Can reproduce locally
- [ ] Caused by external libraries (e.g. bug in `transformers`)

### 🧪 Describe the failing test

`Acsend 910b platform`

`pytest ./tests/e2e/long_term/spec_decode_v0/e2e/test_mtp_correctness.py`



### 📝 History of failing test

```bash
vllm-empty/vllm/model_executor/model_loader/default_loader.py:269: in load_weights
    loaded_weights = model.load_weights(
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = CustomDeepSeekMTP(
  (model): CustomDeepSeekMultiTokenPredictor(
    (layers): ModuleDict(
      (5): CustomDeepSeekMu...ogitsProcessor(vocab_size=129280, org_vocab_size=129280, scale=1.0, logits_as_input=False)
  )
  (sampler): Sampler()
)
weights = <generator object DefaultModelLoader.get_all_weights at 0xffff7dd9f060>

    def load_weights(self, weights:

[... truncated for brevity ...]

---

## Issue #N/A: [CI Failure]: entrypoints-test: test_streaming_response

**Link**: https://github.com/vllm-project/vllm/issues/20366
**State**: closed
**Created**: 2025-07-02T09:59:03+00:00
**Closed**: 2025-07-04T07:55:13+00:00
**Comments**: 3
**Labels**: ci-failure

### Description

### Name of failing test

entrypoints/openai/test_transcription_validation.py::test_streaming_response

### Basic information

- [x] Flaky test
- [ ] Can reproduce locally
- [ ] Caused by external libraries (e.g. bug in `transformers`)

### 🧪 Describe the failing test

Two sporadic failing tests:
entrypoints/openai/test_transcription_validation.py::test_streaming_response
entrypoints/openai/test_translation_validation.py::test_streaming_response

Failing with:
openai.APITimeoutError: Request timed out.

Example commits from main which fail:
https://github.com/vllm-project/vllm/commit/c05596f1a350f3d993c467959ed02492141c2527
https://github.com/vllm-project/vllm/commit/7da296be04933cfc29031f5bd1ba7cd28f376faa

There are more...

### 📝 History of failing test

The first commit in main with this error is:
https://github.com/vllm-project/vllm/commit/c05596f1a350f3d993c467959ed02492141c2527

However, looking at the code it seems unrelated. So maybe some other commit beforehand. My guess is a

[... truncated for brevity ...]

---

## Issue #N/A: [CI Failure]: Quantization Test - quantization/test_bitsandbytes.py::test_load_4bit_bnb_model

**Link**: https://github.com/vllm-project/vllm/issues/20767
**State**: open
**Created**: 2025-07-10T16:03:59+00:00
**Comments**: 1
**Labels**: ci-failure

### Description

### Name of failing test

`quantization/test_bitsandbytes.py::test_load_4bit_bnb_model[facebook/opt-125m-quantize opt model inflight]`

### Basic information

- [x] Flaky test
- [ ] Can reproduce locally
- [ ] Caused by external libraries (e.g. bug in `transformers`)

### 🧪 Describe the failing test

There are quite a few failing bnb tests https://buildkite.com/vllm/ci/builds/23559/steps/canvas?sid=0197f23d-c9e5-45de-b07a-5e290ae4a6ce

```

[2025-07-10T04:47:11Z] FAILED quantization/test_bitsandbytes.py::test_load_4bit_bnb_model[facebook/opt-125m-quantize opt model inflight] - AssertionError: function <function test_load_4bit_bnb_model at 0x7f4699a1cea0> failed when called with args () and kwargs {'hf_runner': <class 'tests.conftest.HfRunner'>, 'vllm_runner': <class 'tests.conftest.VllmRunner'>, 'example_prompts': ['vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.\n', 'Briefly describe the major milestones in the development of artificial intelligen

[... truncated for brevity ...]

---

## Issue #N/A: [Bug][Failing Test]: LoRA 2 - lora/test_lora_functions.py::test_lora_functions_sync

**Link**: https://github.com/vllm-project/vllm/issues/18498
**State**: closed
**Created**: 2025-05-21T16:35:09+00:00
**Closed**: 2025-05-22T04:48:54+00:00
**Comments**: 5
**Labels**: bug, ci-failure

### Description

### Your current environment

N/A

### 🐛 Describe the bug

https://buildkite.com/vllm/ci/builds/20460/steps?jid=0196f343-0fdb-4d91-80da-728e0fb8174c

Summary:
```
[2025-05-21T16:00:09Z] FAILED lora/test_lora_functions.py::test_lora_functions_sync[True] - Exception: Call to add_lora method failed: CUDA error: an illegal memory access was encountered
[2025-05-21T16:00:09Z] CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
[2025-05-21T16:00:09Z] For debugging consider passing CUDA_LAUNCH_BLOCKING=1
[2025-05-21T16:00:09Z] Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

Stack:
```
[2025-05-21T15:50:19Z] ERROR 05-21 08:50:19 [core.py:559] Invocation of add_lora method failed
[2025-05-21T15:50:19Z] ERROR 05-21 08:50:19 [core.py:559] Traceback (most recent call last):
[2025-05-21T15:50:19Z] ERROR 05-21 08:50:19 [core.py:559]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", lin

[... truncated for brevity ...]

---

## Issue #N/A: [Bug][Failing Test] - Quantization test - quantization/test_cpu_offload.py

**Link**: https://github.com/vllm-project/vllm/issues/18425
**State**: closed
**Created**: 2025-05-20T16:15:31+00:00
**Closed**: 2025-05-21T17:25:49+00:00
**Comments**: 4
**Labels**: bug, ci-failure

### Description

### Your current environment

Failing on main as of commit 9609327fa4

### 🐛 Describe the bug

Failing tests:

```
FAILED quantization/test_cpu_offload.py::test_cpu_offload_gptq - RuntimeError: Server exited unexpectedly.
FAILED quantization/test_cpu_offload.py::test_cpu_offload_awq - RuntimeError: Server exited unexpectedly.
FAILED quantization/test_cpu_offload.py::test_cpu_offload_compressed_tensors - AssertionError: Results for model='nm-testing/llama7b-one-shot-2_4-w4a16-marlin24-t' are not the same.
ref_args=[] ref_envs=None
compare_args=['--cpu-offload-gb', '1'] compare_envs=None
ref_result={'test': 'single_completion', 'text': ' ... ... . Today I', 'finish_reason': 'length', 'usage': CompletionUsage(completion_tokens=5, prompt_tokens=6, total_tokens=11, completion_tokens_details=None, prompt_tokens_details=None)}
compare_result={'test': 'single_completion', 'text': ' ... ... .\n I', 'finish_reason': 'length', 'usage': CompletionUsage(completion_tokens=5, prompt_tokens=6, total_t

[... truncated for brevity ...]

---

## Issue #N/A: [CI Failure]: Quantization Test - quantization/test_bitsandbytes.py::test_4bit_bnb_embedding_model

**Link**: https://github.com/vllm-project/vllm/issues/19964
**State**: closed
**Created**: 2025-06-23T04:53:19+00:00
**Closed**: 2025-06-23T13:30:57+00:00
**Comments**: 0
**Labels**: ci-failure

### Description

### Name of failing test

`quantization/test_bitsandbytes.py::test_4bit_bnb_embedding_model[half-intfloat/e5-mistral-7b-instruct-quantize embedding model inflight]`

### Basic information

- [ ] Flaky test
- [x] Can reproduce locally
- [ ] Caused by external libraries (e.g. bug in `transformers`)

### 🧪 Describe the failing test

```
pytest -s -v "quantization/test_bitsandbytes.py::test_4bit_bnb_embedding_model[half-intfloat/e5-mistral-7b-instruct-quantize embedding model inflight]"
INFO 06-23 04:48:10 [__init__.py:244] Automatically detected platform cuda.
/home/mgoin/venvs/vllm/lib/python3.12/site-packages/pytest_asyncio/plugin.py:208: PytestDeprecationWarning: The configuration option "asyncio_default_fixture_loop_scope" is unset.
The event loop scope for asynchronous fixtures will default to the fixture caching scope. Future versions of pytest-asyncio will default the loop scope for asynchronous fixtures to function scope. Set the default fixture loop scope explicitly in order to a

[... truncated for brevity ...]

---

## Issue #N/A: [Bug][Failing Test]: V1 - v1/entrypoints/llm/test_struct_output_generate.py

**Link**: https://github.com/vllm-project/vllm/issues/18525
**State**: closed
**Created**: 2025-05-22T04:41:07+00:00
**Closed**: 2025-05-22T13:48:59+00:00
**Comments**: 2
**Labels**: bug, ci-failure

### Description

### Your current environment

N/A

### 🐛 Describe the bug

`v1/entrypoints/llm/test_struct_output_generate.py::test_structured_output_with_reasoning_matrices` fails on main

e.g. https://buildkite.com/vllm/ci/builds/20477/steps?jid=0196f3e2-128a-409e-bafa-5d676afc9557

Stack:
```
[2025-05-21T18:59:44Z] ERROR 05-21 11:59:44 [core.py:493] EngineCore failed to start.

[2025-05-21T18:59:44Z] ERROR 05-21 11:59:44 [core.py:493] Traceback (most recent call last):

[2025-05-21T18:59:44Z] ERROR 05-21 11:59:44 [core.py:493]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 484, in run_engine_core

[2025-05-21T18:59:44Z] ERROR 05-21 11:59:44 [core.py:493]     engine_core = EngineCoreProc(*args, **kwargs)

[2025-05-21T18:59:44Z] ERROR 05-21 11:59:44 [core.py:493]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[2025-05-21T18:59:44Z] ERROR 05-21 11:59:44 [core.py:493]   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 383, in __init__

[2025

[... truncated for brevity ...]

---

## Issue #N/A: [Bug][Failing Test]: Samplers Test - samplers/test_seeded_generate.py

**Link**: https://github.com/vllm-project/vllm/issues/18656
**State**: closed
**Created**: 2025-05-24T10:22:59+00:00
**Closed**: 2025-05-24T15:25:21+00:00
**Comments**: 0
**Labels**: bug, ci-failure

### Description

### Your current environment

N/A

### 🐛 Describe the bug

`samplers/test_seeded_generate.py::test_random_sample_with_seed` has been failing on main since #17731

https://buildkite.com/organizations/vllm/analytics/suites/ci-1/tests/7615b2b4-ca19-80d3-ab9c-5b2395cd950a?period=7days&tags=scm.branch%3Amain

cc @shadeMe @mgoin @aarnphm 

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

