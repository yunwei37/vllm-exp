# script - issues

**Total Issues**: 3
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 3

### Label Distribution

- script: 3 issues
- documentation: 1 issues
- python: 1 issues
- high severity: 1 issues
- duplicate: 1 issues

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

## Issue #N/A: Update *-to-ggml.py scripts for new ggjt model format

**Link**: https://github.com/ggml-org/llama.cpp/issues/704
**State**: closed
**Created**: 2023-04-02T09:49:22+00:00
**Closed**: 2023-05-03T18:37:53+00:00
**Comments**: 1
**Labels**: script

### Description

See title, basically.

We should probably keep the option of generating the old formats.

Revert #690 when done.

Related: #545

---

## Issue #N/A: How to convert old ALPACA q4_0 model into ggjt format?

**Link**: https://github.com/ggml-org/llama.cpp/issues/701
**State**: closed
**Created**: 2023-04-02T08:29:38+00:00
**Closed**: 2023-04-04T19:32:30+00:00
**Comments**: 4
**Labels**: duplicate, script

### Description

I'm trying to use a python script, but it returns the following error:

d:\ALPACA2>python migrate-ggml-2023-03-30-pr613.py ggml-alpaca-7b-q4.bin ggml-alpaca-7b-q4-ggjt.bin
Traceback (most recent call last):
  File "d:\ALPACA2\migrate-ggml-2023-03-30-pr613.py", line 313, in <module>
    main()
  File "d:\ALPACA2\migrate-ggml-2023-03-30-pr613.py", line 274, in main
    tokens = read_tokens(fin, hparams)
  File "d:\ALPACA2\migrate-ggml-2023-03-30-pr613.py", line 135, in read_tokens
    word = fin.read(length)
ValueError: read length must be non-negative or -1


---

