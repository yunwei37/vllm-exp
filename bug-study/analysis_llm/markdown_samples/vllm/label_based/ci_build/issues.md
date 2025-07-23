# ci_build - issues

**Total Issues**: 5
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 5
- Closed Issues: 0

### Label Distribution

- ci/build: 5 issues
- bug: 4 issues
- stale: 4 issues
- v1: 3 issues

---

## Issue #N/A: [Tracker] Nightly CI Test Failures

**Link**: https://github.com/vllm-project/vllm/issues/17405
**State**: open
**Created**: 2025-04-29T18:27:35+00:00
**Comments**: 0
**Labels**: ci/build

### Description

This is intended as umbrella issue tracking failures


---

## Issue #N/A: [Bug]: CI flake - v1/engine/test_async_llm.py::test_abort - assert has_unfinished_requests()

**Link**: https://github.com/vllm-project/vllm/issues/16054
**State**: open
**Created**: 2025-04-04T09:48:13+00:00
**Comments**: 1
**Labels**: bug, ci/build, stale, v1

### Description

### Your current environment

...

### 🐛 Describe the bug

main commit 51d7c6a2b23e100cd9e7d85b8e7c0eea656b331e

Seen in https://github.com/vllm-project/vllm/pull/15894

https://buildkite.com/organizations/vllm/pipelines/ci/builds/16742/jobs/0195f24d-e81a-46a3-ad08-6a51983d65d6/log


```
=================================== FAILURES ===================================
[2025-04-01T17:38:12Z] _ test_abort[engine_args0-Hello my name is Robert and-RequestOutputKind.DELTA] _
[2025-04-01T17:38:12Z]
[2025-04-01T17:38:12Z] monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7fd1fa052e70>
[2025-04-01T17:38:12Z] output_kind = <RequestOutputKind.DELTA: 1>
[2025-04-01T17:38:12Z] engine_args = AsyncEngineArgs(model='meta-llama/Llama-3.2-1B-Instruct', served_model_name=None, tokenizer='meta-llama/Llama-3.2-1B-I...additional_config=None, enable_reasoning=None, reasoning_parser=None, use_tqdm_on_load=True, disable_log_requests=True)
[2025-04-01T17:38:12Z] prompt = 'Hello my name is Robert and'
[

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: CI flake - v1/entrypoints/llm/test_struct_output_generate.py::test_structured_output - JSONDecodeError: Expecting value: line 1 column 1 (char 0)

**Link**: https://github.com/vllm-project/vllm/issues/16053
**State**: open
**Created**: 2025-04-04T09:46:16+00:00
**Comments**: 2
**Labels**: bug, ci/build, stale, v1

### Description

### Your current environment

...


### 🐛 Describe the bug

main commit 51d7c6a2b23e100cd9e7d85b8e7c0eea656b331e

Seen in https://github.com/vllm-project/vllm/pull/15894

https://buildkite.com/organizations/vllm/pipelines/ci/builds/16742/jobs/0195fc58-3d11-45b5-b76f-8e962cbda765/log

```
FAILED v1/entrypoints/llm/test_struct_output_generate.py::test_structured_output[Qwen/Qwen2.5-1.5B-Instruct-guidance:disable-any-whitespace-auto] - json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)

[2025-04-03T16:08:35Z] _ test_structured_output[Qwen/Qwen2.5-1.5B-Instruct-guidance:disable-any-whitespace-auto] _
[2025-04-03T16:08:35Z]
[2025-04-03T16:08:35Z] monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f318d89eb40>
[2025-04-03T16:08:35Z] sample_json_schema = {'properties': {'age': {'type': 'integer'}, 'name': {'type': 'string'}, 'skills': {'items': {'type': 'string'}, 'type'...ition'], 'type': 'object'}, 'type': 'array'}}, 'required': ['name', 'age', 'skills', 'work_

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: CI flake - v1/entrypoints/llm/test_struct_output_generate.py::test_structured_output bug Something isn't working ci/build v1

**Link**: https://github.com/vllm-project/vllm/issues/15944
**State**: open
**Created**: 2025-04-02T10:29:48+00:00
**Comments**: 2
**Labels**: bug, ci/build, stale

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```

</details>


### 🐛 Describe the bug

```

[2025-04-02T06:06:31Z] =================================== FAILURES ===================================
--
  | [2025-04-02T06:06:31Z] _ test_structured_output[mistralai/Ministral-8B-Instruct-2410-guidance:disable-any-whitespace-auto] _
  | [2025-04-02T06:06:31Z]
  | [2025-04-02T06:06:31Z] monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f6c3b4dcfb0>
  | [2025-04-02T06:06:31Z] sample_json_schema = {'properties': {'age': {'type': 'integer'}, 'name': {'type': 'string'}, 'skills': {'items': {'type': 'string'}, 'type'...ition'], 'type': 'object'}, 'type': 'array'}}, 'required': ['name', 'age', 'skills', 'work_history'], 'type': 'object'}
  | [2025-04-02T06:06:31Z] unsupported_json_schema = {'properties': {'email': {'pattern': '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$', 'type': 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: CI flake - v1/engine/test_llm_engine.py::test_parallel_sampling[True]

**Link**: https://github.com/vllm-project/vllm/issues/15855
**State**: open
**Created**: 2025-04-01T06:55:01+00:00
**Comments**: 4
**Labels**: bug, ci/build, stale, v1

### Description

### Your current environment

...


### 🐛 Describe the bug

Saw V1 test failing with this yesterday, went away with recheck:

```
[2025-03-31T17:33:47Z] _________________________ test_parallel_sampling[True] _________________________
[2025-03-31T17:33:47Z]
[2025-03-31T17:33:47Z] vllm_model = <tests.conftest.VllmRunner object at 0x7f0d875e06e0>
[2025-03-31T17:33:47Z] example_prompts = ['vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.\n', 'Briefly describe the majo...me.\n', 'Analyze the impact of the COVID-19 pandemic on global economic structures and future business models.\n', ...]
[2025-03-31T17:33:47Z]
[2025-03-31T17:33:47Z]     def test_parallel_sampling(vllm_model, example_prompts) -> None:
[2025-03-31T17:33:47Z]         """Test passes if parallel sampling `n>1` yields `n` unique completions.
[2025-03-31T17:33:47Z]
[2025-03-31T17:33:47Z]         Args:
[2025-03-31T17:33:47Z]           vllm_model: VllmRunner instance under test.
[2025-03-31T17:3

[... truncated for brevity ...]

---

