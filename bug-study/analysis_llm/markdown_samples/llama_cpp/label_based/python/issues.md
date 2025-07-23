# python - issues

**Total Issues**: 1
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 1

### Label Distribution

- documentation: 1 issues
- script: 1 issues
- python: 1 issues
- high severity: 1 issues

---

## Issue #N/A: Why is convert.py missing?

**Link**: https://github.com/ggml-org/llama.cpp/issues/7658
**State**: closed
**Created**: 2024-05-31T05:46:36+00:00
**Closed**: 2024-06-10T19:58:23+00:00
**Comments**: 15
**Labels**: documentation, script, python, high severity

### Description

### What happened?

Critical "non llama3" convert.py and change NOT in download of files.

ALSO:
It is unclear if "convert-hf-to-gguf.py"  supports what "convert.py" did . 

Does it convert llama, llama2, mistral or is "convert-legacy-llama.py" required?
Safetensor files are EVERYWHERE. (?)  [ RE: https://github.com/ggerganov/llama.cpp/pull/7430 ]

This critical action DID NOT OCCUR:
"Move convert.py to examples/convert-legacy-llama.py"

"examples/convert-legacy-llama.py" does not exist. 
(when downloading the zip files).

On another note why remove "convert.py" at all? 

-This breaks "bat files" and automation generation.
-This will break all colabs too.
-This will break any HF spaces that create GGUF files as well.
-This will create needless confusion.

If "convert-hf-to-gguf.py" (llama3) does everything convert.py did , just keep it as "convert.py" ?





### Name and Version

this is not applicable - core files missing.

### What operating system are you se

[... truncated for brevity ...]

---

