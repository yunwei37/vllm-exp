# testing - issues

**Total Issues**: 12
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 11

### Label Distribution

- testing: 12 issues
- good first issue: 6 issues
- enhancement: 3 issues
- help wanted: 3 issues
- roadmap: 2 issues
- stale: 2 issues
- performance: 1 issues
- model: 1 issues
- server/webui: 1 issues
- refactoring: 1 issues

---

## Issue #N/A: Move gguf fuzzers to the llama.cpp repository

**Link**: https://github.com/ggml-org/llama.cpp/issues/11514
**State**: open
**Created**: 2025-01-30T15:57:53+00:00
**Comments**: 5
**Labels**: enhancement, testing, roadmap

### Description

Fuzz testing of llama.cpp in OSS-Fuzz has been very valuable to detect leaks and security issues in the model loading code. Unfortunately, the build of the [current fuzzers](https://github.com/google/oss-fuzz/tree/master/projects/llamacpp) has been broken for a long time, and new code is not being tested.

We should move the fuzzers to this repository and ensure that they are maintained. More details: https://google.github.io/oss-fuzz/advanced-topics/ideal-integration/

@DavidKorczynski the current implementation seems to be Apache licensed, which would complicate moving the code here. Would it be possible to re-license it as MIT?

---

## Issue #N/A: ci : add Arm Cobalt 100 runners

**Link**: https://github.com/ggml-org/llama.cpp/issues/11275
**State**: closed
**Created**: 2025-01-17T09:17:03+00:00
**Closed**: 2025-02-22T11:09:50+00:00
**Comments**: 0
**Labels**: help wanted, good first issue, testing, roadmap

### Description

There are some new Github Actions runners "powered by the Cobalt 100-based processors":

https://github.blog/changelog/2025-01-16-linux-arm64-hosted-runners-now-available-for-free-in-public-repositories-public-preview/

Not sure what this processor is specifically, but it might have some Arm features that would be useful to exercise in the CI. We should look into more details and add workflows if it makes sense.

---

## Issue #N/A: ci : self-hosted runner issue

**Link**: https://github.com/ggml-org/llama.cpp/issues/7893
**State**: closed
**Created**: 2024-06-12T07:04:44+00:00
**Closed**: 2025-02-14T01:07:18+00:00
**Comments**: 5
**Labels**: testing, stale

### Description

Not sure what happened but Github shows hundreds of self-hosted runners here:

https://github.com/ggerganov/llama.cpp/actions/runners?tab=self-hosted

When I start the runner on the T4 node, it now just spams the following error:

```
invalid JIT response code: 422
    {"message":"Invalid Argument - Runner group 1 has reached max limit of 10000 runners.","documentation_url":"https://docs.github.com/rest/actions/self-hosted-runners#create-configuration-for-a-just-in-time-runner-for-a-repository","status":"422"}
ggml-ci:     ggml-runner-90970032-26062764448-pull_request_target-1718175354 triggered for workflow_name=Benchmark
invalid JIT response code: 422
    {"message":"Invalid Argument - Runner group 1 has reached max limit of 10000 runners.","documentation_url":"https://docs.github.com/rest/actions/self-hosted-runners#create-configuration-for-a-just-in-time-runner-for-a-repository","status":"422"}
ggml-ci:     ggml-runner-90970032-26062764706-pull_request_target-1718175355

[... truncated for brevity ...]

---

## Issue #N/A: Add some models in ggml-models HF repo

**Link**: https://github.com/ggml-org/llama.cpp/issues/6292
**State**: closed
**Created**: 2024-03-25T06:36:44+00:00
**Closed**: 2024-05-18T01:58:22+00:00
**Comments**: 5
**Labels**: enhancement, performance, model, testing, stale

### Description

### Motivation

In the context of:

- #6233

Need to add some models in the [GGML HF Repo](https://huggingface.co/ggml-org/models/tree/main):

- mixtral8x7B Q4 Q8 F16 in split format
- bert-bge-large F16
- llama7B 13B split Q4 F16


---

## Issue #N/A: server: add tests with `--split` and `--model-url`

**Link**: https://github.com/ggml-org/llama.cpp/issues/6223
**State**: closed
**Created**: 2024-03-22T07:13:46+00:00
**Closed**: 2024-03-23T17:07:01+00:00
**Comments**: 0
**Labels**: enhancement, testing, server/webui

### Description

since we added #6187 and #6192. Servers tests must be improved to support this feature

---

## Issue #N/A: ci : re-enable sanitizer builds when they work again

**Link**: https://github.com/ggml-org/llama.cpp/issues/6129
**State**: closed
**Created**: 2024-03-18T08:35:30+00:00
**Closed**: 2024-05-18T15:56:01+00:00
**Comments**: 1
**Labels**: good first issue, testing

### Description

Disabled temporary to avoid failure notifications https://github.com/ggerganov/llama.cpp/pull/6128

---

## Issue #N/A: llama : update the convert-llama2c-to-ggml example

**Link**: https://github.com/ggml-org/llama.cpp/issues/5608
**State**: closed
**Created**: 2024-02-20T09:50:31+00:00
**Closed**: 2024-03-22T18:49:07+00:00
**Comments**: 0
**Labels**: good first issue, testing, refactoring

### Description

The [convert-llama2c-to-ggml](https://github.com/ggerganov/llama.cpp/tree/master/examples/convert-llama2c-to-ggml) is mostly functional, but can use some maintenance efforts. It also needs an update to support the `n_head_kv` parameter, required for multi-query models (e.g. [stories260K](https://huggingface.co/karpathy/tinyllamas/blob/main/stories260K/readme.md)).

Here is quick'n'dirty patch to make it work with `stories260k` which uses `n_head = 8` and `n_head_kv = 4`:

```diff
diff --git a/examples/convert-llama2c-to-ggml/convert-llama2c-to-ggml.cpp b/examples/convert-llama2c-to-ggml/convert-llama2c-to-ggml.cpp
index 8209dcb6..4aab8552 100644
--- a/examples/convert-llama2c-to-ggml/convert-llama2c-to-ggml.cpp
+++ b/examples/convert-llama2c-to-ggml/convert-llama2c-to-ggml.cpp
@@ -162,8 +162,8 @@ static int checkpoint_init_weights(TransformerWeights *w, Config* p, FILE* f, bo
     if (fread(w->token_embedding_table, sizeof(float), p->vocab_size * p->dim, f) != static_cast<siz

[... truncated for brevity ...]

---

## Issue #N/A: Missing tokenizer tests

**Link**: https://github.com/ggml-org/llama.cpp/issues/3730
**State**: closed
**Created**: 2023-10-22T20:00:20+00:00
**Closed**: 2023-10-24T07:17:18+00:00
**Comments**: 6
**Labels**: help wanted, testing

### Description

AFAIU we are missing tokenizer tests for supported models like

* Baichuan
* Bloom
* GptNeoX
* Persimmon
* Refact
* Starcoder

It would be great if anyone would be helping out.

---

## Issue #N/A: ci : add Apple silicon (M1) macOS runners

**Link**: https://github.com/ggml-org/llama.cpp/issues/3469
**State**: closed
**Created**: 2023-10-04T11:21:05+00:00
**Closed**: 2024-12-23T14:43:36+00:00
**Comments**: 6
**Labels**: good first issue, testing

### Description

We should start running the CI on these runners too:

https://github.blog/changelog/2023-10-02-github-actions-apple-silicon-m1-macos-runners-are-now-available-in-public-beta/

Example from the `ggml` repo: https://github.com/ggerganov/ggml/pull/514

Need to do it for `llama.cpp` and `whisper.cpp`

---

## Issue #N/A: llama : add LoRA test to CI

**Link**: https://github.com/ggml-org/llama.cpp/issues/2634
**State**: closed
**Created**: 2023-08-16T15:57:07+00:00
**Closed**: 2023-08-27T07:03:28+00:00
**Comments**: 4
**Labels**: good first issue, testing

### Description

Add a simple test to [ci/run.sh](https://github.com/ggerganov/llama.cpp/tree/master/ci) to make sure LoRA functionality is OK

---

## Issue #N/A: llama : add test for saving/loading sessions to the CI

**Link**: https://github.com/ggml-org/llama.cpp/issues/2631
**State**: closed
**Created**: 2023-08-16T14:42:01+00:00
**Closed**: 2025-03-07T10:19:33+00:00
**Comments**: 3
**Labels**: good first issue, testing

### Description

See how the `save-load-state` example works:

https://github.com/ggerganov/llama.cpp/tree/master/examples/save-load-state

Add a simple test to [ci/run.sh](https://github.com/ggerganov/llama.cpp/tree/master/ci)

---

## Issue #N/A: Fix failing CI test using thread sanitizer

**Link**: https://github.com/ggml-org/llama.cpp/issues/582
**State**: closed
**Created**: 2023-03-28T17:16:53+00:00
**Closed**: 2023-04-02T07:18:54+00:00
**Comments**: 3
**Labels**: help wanted, high priority, testing

### Description

I cannot reproduce on my machines:

https://github.com/ggerganov/llama.cpp/actions/runs/4545676297/jobs/8013336777

If someone that can reproduce, please try to fix this

---

