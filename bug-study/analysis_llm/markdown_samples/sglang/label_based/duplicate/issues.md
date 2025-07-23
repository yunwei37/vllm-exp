# duplicate - issues

**Total Issues**: 1
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 1

### Label Distribution

- duplicate: 1 issues
- feature: 1 issues

---

## Issue #N/A: [Feature] Respect max_completion_tokens

**Link**: https://github.com/sgl-project/sglang/issues/3531
**State**: closed
**Created**: 2025-02-12T17:52:38+00:00
**Closed**: 2025-02-13T19:23:21+00:00
**Comments**: 2
**Labels**: duplicate, feature

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Currently the OpenAI compatible API only respects the old `max_tokens` request argument. The updated spec introduces `max_completion_tokens`.

I can send a PR adding support for the new argument name and just change the code here:
https://github.com/sgl-project/sglang/blob/8616357a97c5f68eca194dfbeef0ae51943032ef/python/sglang/srt/openai_api/adapter.py#L512

to `request.max_completion_tokens or request.max_tokens`

### Related resources

_No response_

---

