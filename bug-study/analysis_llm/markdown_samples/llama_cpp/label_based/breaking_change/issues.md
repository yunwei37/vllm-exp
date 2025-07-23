# breaking_change - issues

**Total Issues**: 1
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 0

### Label Distribution

- enhancement: 1 issues
- good first issue: 1 issues
- breaking change: 1 issues

---

## Issue #N/A: GGUF endianness cannot be determined from GGUF itself

**Link**: https://github.com/ggml-org/llama.cpp/issues/3957
**State**: open
**Created**: 2023-11-05T14:00:47+00:00
**Comments**: 17
**Labels**: enhancement, good first issue, breaking change

### Description

As of the time of writing, the big-endian support that was added in https://github.com/ggerganov/llama.cpp/pull/3552 doesn't encode the endianness within the file itself: 

https://github.com/ggerganov/llama.cpp/blob/3d48f42efcd05381221654376e9f6f69d76af739/gguf-py/gguf/gguf.py#L689-L698

This means that there is no way to distinguish a big-endian GGUF file from a little-endian file, which may cause some degree of consternation in the future if these files get shared around 😅 

The cleanest solution would be to add the endianness to the header - ideally, it would be in the metadata, but the reading of the metadata is dependent on the endianness - but that would be a breaking change.

Given that, my suggestion would be to use `FUGG` as the header for big-endian files so that a little-endian executor won't attempt to read it at all unless it knows how to deal with it. The same can go the other way, as well (a big-endian executor won't attempt to read a little-endian executor).

---

