# invalid - issues

**Total Issues**: 11
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 11

### Label Distribution

- invalid: 11 issues
- need more info: 1 issues
- stale: 1 issues
- good first issue: 1 issues
- wontfix: 1 issues

---

## Issue #N/A: convert-hf-to-gguf.py  XVERSE-13B-256K  error

**Link**: https://github.com/ggml-org/llama.cpp/issues/6425
**State**: closed
**Created**: 2024-04-01T13:07:59+00:00
**Closed**: 2024-04-03T23:53:56+00:00
**Comments**: 4
**Labels**: invalid

### Description

Please include information about your system, the steps to reproduce the bug, and the version of llama.cpp that you are using. If possible, please provide a minimal code example that reproduces the bug.

Model: 
https://huggingface.co/xverse/XVERSE-13B-256K

python convert-hf-to-gguf.py /Volumes/FanData/models/XVERSE-13B-256K --outfile /Volumes/FanData/models/GGUF/xverse-13b-256k-f16.gguf --outtype f16

`
python convert-hf-to-gguf.py /Volumes/FanData/models/XVERSE-13B-256K --outfile /Volumes/FanData/models/GGUF/xverse-13b-256k-f16.gguf --outtype f16
Loading model: XVERSE-13B-256K
gguf: This GGUF file is for Little Endian only
Set model parameters
Set model tokenizer
gguf: Setting special token type bos to 2
gguf: Setting special token type eos to 3
gguf: Setting special token type pad to 1
Exporting model to '/Volumes/FanData/models/GGUF/xverse-13b-256k-f16.gguf'
gguf: loading model part 'pytorch_model-00001-of-00015.bin'
Traceback (most recent call last):
  File "/U

[... truncated for brevity ...]

---

## Issue #N/A: دل

**Link**: https://github.com/ggml-org/llama.cpp/issues/5375
**State**: closed
**Created**: 2024-02-06T21:22:59+00:00
**Closed**: 2024-02-06T21:54:25+00:00
**Comments**: 0
**Labels**: invalid

### Description

Please include information about your system, the steps to reproduce the bug, and the version of llama.cpp that you are using. If possible, please provide a minimal code example that reproduces the bug.

---

## Issue #N/A: Unable to convert bloom models

**Link**: https://github.com/ggml-org/llama.cpp/issues/4768
**State**: closed
**Created**: 2024-01-04T07:45:54+00:00
**Closed**: 2024-01-05T15:11:20+00:00
**Comments**: 10
**Labels**: invalid

### Description

When trying to convert bloom model downloaded from Huggingface (https://huggingface.co/bigscience/bloomz-1b7) using the following command
```shell
python3.10 convert.py /root/bloomz-1b7/
```
it outputs the following messages
```
Loading model file /root/bloomz-1b7/model.safetensors
Traceback (most recent call last):
  File "/root/workspace/llama.cpp/convert.py", line 1295, in <module>
    main()
  File "/root/workspace/llama.cpp/convert.py", line 1234, in main
    params = Params.load(model_plus)
  File "/root/workspace/llama.cpp/convert.py", line 318, in load
    params = Params.loadHFTransformerJson(model_plus.model, hf_config_path)
  File "/root/workspace/llama.cpp/convert.py", line 237, in loadHFTransformerJson
    raise Exception("failed to guess 'n_ctx'. This model is unknown or unsupported.\n"
Exception: failed to guess 'n_ctx'. This model is unknown or unsupported.
Suggestion: provide 'config.json' of the model in the same directory containing model files.
```

[... truncated for brevity ...]

---

## Issue #N/A: http://localhost:6800/jsonrpc

**Link**: https://github.com/ggml-org/llama.cpp/issues/3964
**State**: closed
**Created**: 2023-11-05T19:00:15+00:00
**Closed**: 2024-04-02T01:12:12+00:00
**Comments**: 1
**Labels**: invalid, need more info, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Please provide a detailed written description of what you were trying to do, and what you expected `llama.cpp` to do.

# Current Behavior

Please provide a detailed written description of what `llama.cpp` did, instead.

# Environment and Context

Please provide detailed informat

[... truncated for brevity ...]

---

## Issue #N/A: [User] covert.py thows KeyError: 'rms_norm_eps' on persimmon-8b-chat

**Link**: https://github.com/ggml-org/llama.cpp/issues/3344
**State**: closed
**Created**: 2023-09-26T16:44:23+00:00
**Closed**: 2023-11-07T22:51:56+00:00
**Comments**: 2
**Labels**: invalid

### Description

#python3 llama.cpp/convert.py persimmon-8b-chat --outfile persimmon-8b-chat.gguf --outtype q8_0
# Expected Behavior
produce a gguf of persimmon 8b at q8_0


# Current Behavior

Traceback (most recent call last):
  File "/mnt/c/Users/admin/src/llama.cpp/convert.py", line 1208, in <module>
    main()
  File "/mnt/c/Users/admin/src/llama.cpp/convert.py", line 1157, in main
    params = Params.load(model_plus)
  File "/mnt/c/Users/admin/src/llama.cpp/convert.py", line 288, in load
    params = Params.loadHFTransformerJson(model_plus.model, hf_config_path)
  File "/mnt/c/Users/admin/src/llama.cpp/convert.py", line 208, in loadHFTransformerJson
    f_norm_eps       = config["rms_norm_eps"]
KeyError: 'rms_norm_eps'
# Environment and Context

Please provide detailed information about your computer setup. This is important in case the issue is not reproducible except for under certain specific conditions.

Linux ubuntu 22.04

`$ lscpu`

Architecture:            x86_64


[... truncated for brevity ...]

---

## Issue #N/A: [User] faild to find n_mult number from range 256, with n_ff = 3072

**Link**: https://github.com/ggml-org/llama.cpp/issues/2241
**State**: closed
**Created**: 2023-07-16T10:42:50+00:00
**Closed**: 2023-07-17T03:39:24+00:00
**Comments**: 2
**Labels**: invalid

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

i'm not sure if i should
change this line to `for n_mult in range(3000, 1, -1):`

# Current Behavior
model tried to convert https://huggingface.co/symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli/tree/main

params: `n_vocab:250000 n_embd:768 n_head:12 n_layer:12`

[... truncated for brevity ...]

---

## Issue #N/A: [User] convert.py KeyError for redpajama chat 3B

**Link**: https://github.com/ggml-org/llama.cpp/issues/2114
**State**: closed
**Created**: 2023-07-05T16:53:30+00:00
**Closed**: 2023-11-07T22:55:01+00:00
**Comments**: 3
**Labels**: invalid

### Description

# Expected Behavior
I'm trying to convert the [redpajama 3b chat model](https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1) to a ggml model from the `pytorch_model.bin`.

I saw in [this discussion](https://github.com/ggerganov/llama.cpp/discussions/1394#discussion-5182815) that this is expected to work.


# Current Behavior
I get a failure running convert.py on a directory containing the `pytorch_model.bin` from HF as:
```
python3 convert.py models/RedPajama-INCITE-Chat-3B-v1/
```


# Environment and Context
I am on an m2 mba with 16gb of ram with macOS Ventura.

python version: 3.10.11
make: 3.81
g++: Apple clang version 14.0.0 (clang-1400.0.29.202)


# Failure Information (for bugs):
`convert.py` raises:
```
Loading model file models/RedPajama-INCITE-Chat-3B-v1/pytorch_model.bin
Traceback (most recent call last):
  File "/Users/lachlangray/local/llama.cpp/convert.py", line 1256, in <module>
    main()
  File "/Users/lachlangray/local/llama

[... truncated for brevity ...]

---

## Issue #N/A: Try Modular - Mojo

**Link**: https://github.com/ggml-org/llama.cpp/issues/1317
**State**: closed
**Created**: 2023-05-04T12:32:34+00:00
**Closed**: 2023-05-04T18:49:00+00:00
**Comments**: 2
**Labels**: invalid

### Description

https://www.modular.com/

---

## Issue #N/A: [User] Insert summary of your issue or enhancement..

**Link**: https://github.com/ggml-org/llama.cpp/issues/739
**State**: closed
**Created**: 2023-04-03T09:37:50+00:00
**Closed**: 2023-04-04T19:29:01+00:00
**Comments**: 1
**Labels**: invalid

### Description

Hello, is it possible to save the robot's response in a variable? to then read it in a request?
Example
Me : Hello how are you ?
Bot : I'm fine and you ?

Save result response in new_varifable for use this :
http://127.0.0.1:8888/?tts=new_variable

---

## Issue #N/A: How do i download the models? 

**Link**: https://github.com/ggml-org/llama.cpp/issues/650
**State**: closed
**Created**: 2023-03-31T11:55:15+00:00
**Closed**: 2023-03-31T13:40:17+00:00
**Comments**: 1
**Labels**: good first issue, invalid, wontfix

### Description

`65B 30B 13B 7B tokenizer_checklist.chk tokenizer.model`

This command in the readme.md file says to add the models into the models directory but the models arent even there in the directory.
Please let me know how to download the 7B model to run on my computer.
Thanks

---

## Issue #N/A: Error loading llama 65b 4bit model (HFv2) converted from .pt format

**Link**: https://github.com/ggml-org/llama.cpp/issues/538
**State**: closed
**Created**: 2023-03-26T20:26:54+00:00
**Closed**: 2023-03-27T05:22:28+00:00
**Comments**: 4
**Labels**: invalid

### Description

I used this command to get the converted model:

`python3 convert-gptq-to-ggml.py "path/to/llama-65b-4bit.pt" "path/to/tokenizer.model" "./models/ggml-llama-65b-q4_0.bin"`

I run it with this command:

`./main -m ./models/ggml-llama-65b-q4_0.bin -n 128`

And this is what I get at the end of the output:

```
llama_model_load: loading model part 1/8 from './models/ggml-llama-65b-q4_0.bin'
llama_model_load: llama_model_load: tensor 'tok_embeddings.weight' has wrong size in model file
llama_init_from_file: failed to load model
main: error: failed to load model './models/ggml-llama-65b-q4_0.bin'
```

P. S. Yes, I'm using the latest (or at least today's) version of this repo. While I'm at it, many thanks to ggerganov and everyone else involved! Great job.

---

