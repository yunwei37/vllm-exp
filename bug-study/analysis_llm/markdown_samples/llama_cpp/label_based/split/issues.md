# split - issues

**Total Issues**: 9
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 3
- Closed Issues: 6

### Label Distribution

- split: 9 issues
- enhancement: 5 issues
- good first issue: 5 issues
- help wanted: 4 issues
- bug: 3 issues
- stale: 1 issues

---

## Issue #N/A: Probably best not to clobber files with `gguf-split -merge`

**Link**: https://github.com/ggml-org/llama.cpp/issues/6657
**State**: closed
**Created**: 2024-04-13T12:22:18+00:00
**Closed**: 2024-06-17T01:07:11+00:00
**Comments**: 6
**Labels**: enhancement, stale, split

### Description

To save idiots like me who just tried to use it without reading the command line options:

```
./gguf-split --merge dbrx-16x12b-instruct-q4_0-*-of-00010.gguf dbrx-16x12b-instruct-q4_0.gguf
```

```
-rw-r--r-- 1 juk juk          0 Apr 13 12:47 dbrx-16x12b-instruct-q4_0-00002-of-00010.gguf
```

:facepalm:

---

## Issue #N/A: Trying to split model with --split-max-size, but gguf-split ignores it

**Link**: https://github.com/ggml-org/llama.cpp/issues/6654
**State**: closed
**Created**: 2024-04-13T09:26:58+00:00
**Closed**: 2024-04-14T11:13:01+00:00
**Comments**: 2
**Labels**: bug, split

### Description


Latest version, ubuntu 2204, conda python=3.10.
Trying to split model with gguf-split, but something is going wrong 
```
(base) richard@richard-ProLiant-DL580-Gen9:~/Desktop/ramdisk/banana/llama.cpp$ ./gguf-split --split --split-max-size 4000M --dry-run /media/richard/5fbd0bfa-8253-4803-85eb-80a13218a927/grok-1-fp16-gguf/grok-1-Q5_K.gguf Q5_K/grok-1 
n_split: 1
split 00001: n_tensors = 2115, total_size = 214437M
gguf_split: 1 gguf split written with a total of 2115 tensors.
(base) richard@richard-ProLiant-DL580-Gen9:~/Desktop/ramdisk/banana/llama.cpp$ ./gguf-split --split --split-max-size 4G --dry-run /media/richard/5fbd0bfa-8253-4803-85eb-80a13218a927/grok-1-fp16-gguf/grok-1-Q5_K.gguf Q5_K/grok-1 
n_split: 17
split 00001: n_tensors = 128, total_size = 14609M
split 00002: n_tensors = 128, total_size = 13184M
split 00003: n_tensors = 128, total_size = 12648M
split 00004: n_tensors = 128, total_size = 12597M
split 00005: n_tensors = 128, total_size = 12648M
split 00006: n_tensors = 128,

[... truncated for brevity ...]

---

## Issue #N/A: Inconsistent size output of chunks with gguf-split

**Link**: https://github.com/ggml-org/llama.cpp/issues/6634
**State**: closed
**Created**: 2024-04-12T09:41:54+00:00
**Closed**: 2024-04-14T11:13:00+00:00
**Comments**: 2
**Labels**: bug, split

### Description

System:
Ubuntu server 22.04 LTS
5950X
64GB Ram 3200Mhz
2x Nvidia 3090

Steps to reproduce:

First I converted the model to FP16 GGUF with:
`./convert.py --outfile Karasu-Mixtral-8x22B-v0.1-fp16.gguf --outtype f16 lightblue_Karasu-Mixtral-8x22B-v0.1`

That worked just fine and I got:
![image](https://github.com/ggerganov/llama.cpp/assets/5622210/6da7e036-863c-44ef-9b38-eee0e746f406)

Then to quantize it to Q5_K_M:
`./quantize Karasu-Mixtral-8x22B-v0.1-fp16.gguf Karasu-Mixtral-8x22B-v0.1-Q5_K_M.gguf Q5_K_M`

That worked fine too:
![image](https://github.com/ggerganov/llama.cpp/assets/5622210/39401f0d-7f3d-4f18-b6b3-0f328ad34f12)


But when using gguf-split --split even though I'm using --split-max-tensors 128 the sizes of the chunks are inconsistent:

`./gguf-split --split --split-max-tensors 128 /nfs/models/Karasu-Mixtral-8x22B-v0.1-Q5_K_M.gguf /nfs/models/`

![image](https://github.com/ggerganov/llama.cpp/assets/5622210/f403478e-08f7-460b-8706-8b6b1c687111)
![

[... truncated for brevity ...]

---

## Issue #N/A: Re-quantization of a split gguf file produces "invalid split file"

**Link**: https://github.com/ggml-org/llama.cpp/issues/6548
**State**: closed
**Created**: 2024-04-08T17:09:20+00:00
**Closed**: 2024-04-25T10:29:36+00:00
**Comments**: 11
**Labels**: bug, good first issue, split

### Description

Hi, while testing #6491 branch, I downloaded a Q8_0 quant (split into 3 files) from `dranger003`, and re-quantized it to Q2_K_S to make it more digestible for my museum hardware:
```
./quantize --allow-requantize --imatrix ../models/ggml-c4ai-command-r-plus-104b-f16-imatrix.dat ../models/ggml-c4ai-command-r-plus-104b-q8_0-00001-of-00003.gguf ../models/command-r-plus-104b-Q2_K_S.gguf Q2_K_S 2
```

I only passed the first piece, but `./quantize` processed it correctly and produced a single file with the expected size. However, it probably did not update some metadata and `./main` still thinks the result is a split file:
```
./main -m ../models/command-r-plus-104b-Q2_K_S.gguf -t 15 --color -p "this is a test" -c 2048 -ngl 25 -ctk q8_0
...
llama_model_load: error loading model: invalid split file: ../models/command-r-plus-104b-Q2_K_S.gguf
llama_load_model_from_file: failed to load model
llama_init_from_gpt_params: error: failed to load model '../models/command-r-plus-104b-Q2_K_S

[... truncated for brevity ...]

---

## Issue #N/A: common: download from URL, improve parallel download progress status

**Link**: https://github.com/ggml-org/llama.cpp/issues/6537
**State**: open
**Created**: 2024-04-08T07:37:01+00:00
**Comments**: 6
**Labels**: enhancement, help wanted, good first issue, split

### Description

### Context

When downloading a sharded model, files are downloaded in parallel, it was added in:
- #6192

The progressions of each download conflict:
![image](https://github.com/ggerganov/llama.cpp/assets/5741141/d4937fc7-edf4-4920-ba63-dadf1c77b2d0)

Need to properly implement [CURLOPT_NOPROGRESS](https://curl.se/libcurl/c/CURLOPT_NOPROGRESS.html) for parallel download.

Example in #6515:

```shell
main --hf-repo ggml-org/models \
  --hf-file grok-1/grok-1-q4_0-00001-of-00009.gguf \
  --model   models/grok-1-q4_0-00001-of-00009.gguf \
  -ngl 64
   --prompt "I believe the meaning of life is"
```

---

## Issue #N/A: `gguf-split` add a default option to not include tensors data in first shard

**Link**: https://github.com/ggml-org/llama.cpp/issues/6463
**State**: closed
**Created**: 2024-04-03T16:16:12+00:00
**Closed**: 2024-05-04T16:56:23+00:00
**Comments**: 0
**Labels**: enhancement, help wanted, good first issue, split

### Description

### Motivation

be able to make a split where the first shard is very small and contains primarily the metadata so that it can be downloaded quickly and then start the download of the other shards without waiting for the first to finish

### Proposition
Add an option to not include tensor data in the first file. Maybe it should be enabled by default.
Should be well tested.

`ggml_alloc` should not be called as it will complain with `WARNING: Behavior may be unexpected when allocating 0 bytes for ggml_malloc!`

We can add extra meta data in the first file that describes all tensors in the shards for example

#### References
- #6404
- #6135
- #6187
- #6192
- #6343
- https://github.com/ggerganov/llama.cpp/pull/6343#issuecomment-2034990690
- https://github.com/ggerganov/llama.cpp/pull/6343#issuecomment-2035011205
- https://github.com/huggingface/huggingface.js/issues/604


---

## Issue #N/A: split: include the option in ./convert.py and quantize

**Link**: https://github.com/ggml-org/llama.cpp/issues/6260
**State**: open
**Created**: 2024-03-23T15:32:02+00:00
**Comments**: 9
**Labels**: enhancement, help wanted, good first issue, split

### Description

### Context

At the moment it is only possible to split after convertion or quantization. Mentionned by @Artefact2 in this `[comment](https://github.com/ggerganov/llama.cpp/pull/6135#issuecomment-2003942162)`:

> as an alternative, add the splitting logic directly to tools that produce ggufs, like convert.py and quantize.

### Proposition

Include split options in `convert*.py`, support splits in `quantize`

---

## Issue #N/A: split: allow --split-max-size option

**Link**: https://github.com/ggml-org/llama.cpp/issues/6259
**State**: open
**Created**: 2024-03-23T15:29:25+00:00
**Comments**: 1
**Labels**: enhancement, help wanted, good first issue, split

### Description

### Motivation

we support `--split-max-tensors` since:
- #6135

As mentionned by @Artefact2 in this [comment](https://github.com/ggerganov/llama.cpp/pull/6135#issuecomment-2003942162):
> allowing to split by file size would be more intuitive (and usually more appropriate since file size is usually the limiting factor, eg 4G for FAT or 50G for HF)

### Proposition:
Introduce `--split-max-size N(M|G)` split strategy to split files in file with a max size of N Megabytes or Gigabytes.
As it is not possible to have less than 1 tensor per GGUF, this size is a soft limit.

---

## Issue #N/A: gguf-split does not show as a compiled binary with other programs

**Link**: https://github.com/ggml-org/llama.cpp/issues/6257
**State**: closed
**Created**: 2024-03-23T13:55:52+00:00
**Closed**: 2024-03-23T16:18:14+00:00
**Comments**: 1
**Labels**: split

### Description

It is compiled and linked, but does not make it out of build/bin


---

