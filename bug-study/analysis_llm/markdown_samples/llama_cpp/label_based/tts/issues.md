# tts - issues

**Total Issues**: 4
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 3

### Label Distribution

- tts: 4 issues
- good first issue: 2 issues
- model: 1 issues
- research 🔬: 1 issues
- stale: 1 issues

---

## Issue #N/A: tts : add support for Orpheus

**Link**: https://github.com/ggml-org/llama.cpp/issues/12476
**State**: open
**Created**: 2025-03-20T08:11:43+00:00
**Comments**: 5
**Labels**: good first issue, tts

### Description

HF: https://huggingface.co/collections/canopylabs/orpheus-tts-67d9ea3f6c05a941c06ad9d2

These TTS models seem suitable for supporting. To do that, we need to implement the SNAC audio codec: https://github.com/hubertsiuzdak/snac/

Sample implementation using Python-based inference of SNAC: https://github.com/isaiahbjork/orpheus-tts-local

Similar model support (OuteTTS): https://github.com/ggml-org/llama.cpp/pull/10784
Can be used as a reference how to implement this.

---

## Issue #N/A: csm : implement Sesame-based conversation example

**Link**: https://github.com/ggml-org/llama.cpp/issues/12392
**State**: closed
**Created**: 2025-03-14T14:49:46+00:00
**Closed**: 2025-05-14T01:07:48+00:00
**Comments**: 23
**Labels**: model, research 🔬, stale, tts

### Description

With the first Sesame CSM model [openly available](https://github.com/SesameAILabs/csm), we should implement a local example similar to their [online research demo](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice#demo). It seems that the released CSM model uses [Kyutai's Mimi](https://arxiv.org/abs/2410.00037) audio codec which we have to implement in a similar way as we did with the [WavTokenizer](https://github.com/ggml-org/llama.cpp/pull/10784). Next we can modify the [talk-llama](https://github.com/ggerganov/whisper.cpp/tree/master/examples/talk-llama) example to support audio generation with the CSM. This way we will be able to plug any LLM for the text response generation and use Sesame for speech input/output.

---

## Issue #N/A: llama-tts libc++abi: terminating due to uncaught exception of type std::out_of_range: vector
Aborted

**Link**: https://github.com/ggml-org/llama.cpp/issues/11749
**State**: closed
**Created**: 2025-02-08T05:33:49+00:00
**Closed**: 2025-02-21T15:56:06+00:00
**Comments**: 4
**Labels**: tts

### Description

```
./llama-tts -m $model -mv $voice -p "Hi i am Felix"

build: 4663 (c026ba3c) with clang version 19.1.5 for armv7a-unknown-linux-android24
llama_model_loader: loaded meta data with 38 key-value pairs and 272 tensors from /storage/7DE2-358B/ysf/models/smollm-135m-instruct-q8_0.gguf (version GGUF V3 (latest))       llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = SmolLM 135M
llama_model_loader: - kv   3:                       general.organization str              = HuggingFaceTB                                                                       llama_model_loader: - kv   4:                           general.finetune str              = instruct-add-basi

[... truncated for brevity ...]

---

## Issue #N/A: tts : add basic example for text-to-speech

**Link**: https://github.com/ggml-org/llama.cpp/issues/10173
**State**: closed
**Created**: 2024-11-04T18:53:25+00:00
**Closed**: 2024-12-18T17:27:22+00:00
**Comments**: 5
**Labels**: good first issue, tts

### Description

This new model seems suitable for integration: https://github.com/edwko/OuteTTS

We should add a very minimalistic example for generating audio with it. Ideally, we will implement the (audio tokens) -> (wav) from scratch.

---

