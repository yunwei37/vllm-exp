# documentation - issues

**Total Issues**: 24
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 5
- Closed Issues: 19

### Label Distribution

- documentation: 24 issues
- enhancement: 7 issues
- help wanted: 5 issues
- good first issue: 5 issues
- roadmap: 2 issues
- bug: 2 issues
- low severity: 2 issues
- stale: 2 issues
- server/webui: 2 issues
- model: 2 issues

---

## Issue #N/A: examples : add configuration presets

**Link**: https://github.com/ggml-org/llama.cpp/issues/10932
**State**: open
**Created**: 2024-12-21T09:10:47+00:00
**Comments**: 5
**Labels**: documentation, enhancement, help wanted, good first issue, examples

### Description

## Description

I was recently looking for ways to demonstrate some of the functionality of the `llama.cpp` examples and some of the commands can become very cumbersome. For example, here is what I use for the `llama.vim` FIM server:

```bash
llama-server \
    -m ./models/qwen2.5-7b-coder/ggml-model-q8_0.gguf \
    --log-file ./service-vim.log \
    --host 0.0.0.0 --port 8012 \
    --ctx-size 0 \
    --cache-reuse 256 \
    -ub 1024 -b 1024 -ngl 99 -fa -dt 0.1
```

It would be much cleaner if I could just run, for example:

```bash
llama-server --cfg-fim-7b
```

Or if I could turn this embedding server command into something simpler:

```bash
# llama-server \
#     --hf-repo ggml-org/bert-base-uncased \
#     --hf-file          bert-base-uncased-Q8_0.gguf \
#     --port 8033 -c 512 --embeddings --pooling mean

llama-server --cfg-embd-bert --port 8033
```

## Implementation

There is already an initial example of how we can create such configuration presets:

```bash
llama-tts --tts-ou

[... truncated for brevity ...]

---

## Issue #N/A: changelog : `llama-server` REST API

**Link**: https://github.com/ggml-org/llama.cpp/issues/9291
**State**: open
**Created**: 2024-09-03T06:56:11+00:00
**Comments**: 16
**Labels**: documentation, roadmap

### Description

# Overview

This is a list of changes to the public HTTP interface of the `llama-server` example. Collaborators are encouraged to edit this post in order to reflect important changes to the API that end up merged into the `master` branch.

If you are building a 3rd party project that relies on `llama-server`, it is recommended to follow this issue and check it carefully before upgrading to new versions.

See also:

- [Changelog for `libllama` API](https://github.com/ggerganov/llama.cpp/issues/9289)

## Recent API changes (most recent at the top)

| version | PR  | desc |
| ---     | --- | ---  |
| TBD.  | #13660 | Remove `/metrics` fields related to KV cache tokens and cells` |
| b5223 | #13174 | For chat competion, if last message is assistant, it will be a prefilled message |
| b4599 | #9639 | `/v1/chat/completions` now supports `tools` & `tool_choice` |
| TBD.  | #10974 | `/v1/completions` is now OAI-compat |
| TBD.  | #10783 | `logprobs` is now OAI-compat, default to pre-sampling p

[... truncated for brevity ...]

---

## Issue #N/A: changelog : `libllama` API

**Link**: https://github.com/ggml-org/llama.cpp/issues/9289
**State**: open
**Created**: 2024-09-03T06:48:45+00:00
**Comments**: 10
**Labels**: documentation, roadmap

### Description

# Overview

This is a list of changes to the public interface of the `llama` library. Collaborators are encouraged to edit this post in order to reflect important changes to the API that end up merged into the `master` branch.

If you are building a 3rd party project that relies on `libllama`, it is recommended to follow this issue and check it before upgrading to new versions.

See also:

- [Changelog for `llama-server` REST API](https://github.com/ggerganov/llama.cpp/issues/9291)

## Recent API changes (most recent at the top)

| version | PR  | desc |
| ---     | --- | ---  |
| TBD.  | #14363 | Update `llama_context_params` - add `bool kv_unified` |
| b5740 | #13037 | Update `llama_model_quantize_params` |
| b5870 | #14631 | Remove `enum llama_vocab_pre_type` |
| b5435 | #13653 | Remove `llama_kv_cache_view_*` API |
| b5429 | #13194 | Update `llama_context_params` - add `bool swa_full` |
| b5311 | #13284 | Update `llama_context_params` - remove `logits_all` + rearrange flags |
| b51

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Grammar readme seems incorrect

**Link**: https://github.com/ggml-org/llama.cpp/issues/7720
**State**: open
**Created**: 2024-06-03T20:48:22+00:00
**Comments**: 2
**Labels**: bug, documentation, low severity

### Description

### What happened?

[This bit on the grammar readme](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md#non-terminals-and-terminals) states:

> Non-terminal symbols (rule names) ... are required to be a dashed lowercase word, like `move`, `castle`, or `check-mate`.

However, the [`c.gbnf` defines a `dataType` rule](https://github.com/ggerganov/llama.cpp/blob/master/grammars/c.gbnf#L5) (which features a non-lower-case letter) and this grammar appears to be valid.

I'm not sure what the intended behavior is. I would be happy to update the README if uppercase variables are supported.

### Name and Version

N/A

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

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

## Issue #N/A: Refactor: Add CONTRIBUTING.md and/or update PR template with [no ci] tips

**Link**: https://github.com/ggml-org/llama.cpp/issues/7657
**State**: closed
**Created**: 2024-05-30T23:56:20+00:00
**Closed**: 2024-06-09T15:25:57+00:00
**Comments**: 4
**Labels**: documentation, enhancement, help wanted, devops, low severity

### Description

### Background Description

Discussion in https://github.com/ggerganov/llama.cpp/pull/7650 pointed out a need to add a CONTRIBUTING.md and maybe add a PR template to encourage contributors to add [no ci] tag to documentation only changes.

https://docs.github.com/en/actions/managing-workflow-runs/skipping-workflow-runs

(If anyone wants to tackle this, feel free to)

### Possible Refactor Approaches

Add info about

- doc only changes should have [no ci] in commit title to skip the unneeded CI checks.
- squash on merge with commit title format: "module : some commit title (`#1234`)"

---

## Issue #N/A: Working Fine-Tune Example?

**Link**: https://github.com/ggml-org/llama.cpp/issues/6361
**State**: closed
**Created**: 2024-03-28T06:44:04+00:00
**Closed**: 2024-06-21T01:07:11+00:00
**Comments**: 5
**Labels**: documentation, demo, stale

### Description

I am trying to find a working example of fine-tuning. 

- If I run the example from `https://github.com/ggerganov/llama.cpp/tree/master/examples/finetune`, the script can't find the model.
here is the error
```
main: seed: 1711608198
main: model base = 'open-llama-3b-v2-q8_0.gguf'
llama_model_load: error loading model: llama_model_loader: failed to load model from open-llama-3b-v2-q8_0.gguf

llama_load_model_from_file: failed to load model
llama_new_context_with_model: model cannot be NULL
Segmentation fault: 11
```

- If I try to use a model in `/models` folder such as 
```
./finetune \
        --model-base ./models/ggml-vocab-llama.gguf \
        --checkpoint-in  chk-lora-open-llama-3b-v2-q8_0-shakespeare-LATEST.gguf \
        --checkpoint-out chk-lora-open-llama-3b-v2-q8_0-shakespeare-ITERATION.gguf \
        --lora-out lora-open-llama-3b-v2-q8_0-shakespeare-ITERATION.bin \
        --train-data "shakespeare.txt" \
        --save-every 10 \
        --threads 6 

[... truncated for brevity ...]

---

## Issue #N/A: server: doc: document the `--defrag-thold` option

**Link**: https://github.com/ggml-org/llama.cpp/issues/6293
**State**: open
**Created**: 2024-03-25T06:40:20+00:00
**Comments**: 0
**Labels**: documentation, enhancement, help wanted, server/webui

### Description

### Context

The `--defrag-thold` has been added in:

- https://github.com/ggerganov/llama.cpp/pull/5941#issuecomment-1986947067

But it might be documented in the server README.md

---

## Issue #N/A: server: comment --threads option behavior

**Link**: https://github.com/ggml-org/llama.cpp/issues/6230
**State**: closed
**Created**: 2024-03-22T08:56:45+00:00
**Closed**: 2024-03-23T17:00:39+00:00
**Comments**: 4
**Labels**: documentation, enhancement, server/webui

### Description

As we are using batching, I am wondering what is the purpose of `--threads N` parameter in the `server`.

Should we remove it ?

---

## Issue #N/A: Confusion about the model versioning

**Link**: https://github.com/ggml-org/llama.cpp/issues/647
**State**: closed
**Created**: 2023-03-31T07:20:01+00:00
**Closed**: 2023-05-03T18:46:36+00:00
**Comments**: 21
**Labels**: documentation, enhancement

### Description

So back when project started, we had the first "unversioned" model format without the embedded tokens, with the magic 0x67676d6c (ggml).

Problem with that was that it didn't have any versioning support, so newer/older versions would just think "I don't know what this is, this is not a model file".

Then on this commit https://github.com/ggerganov/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4, adding the embedded the tokens we got a new versioned model format, with magic 0x67676d66 (ggmf), along with **versioning**, so it could now say "this is definitely a model file, but a wrong version" as shown here:
https://github.com/ggerganov/llama.cpp/blob/3bcc129ba881c99795e850b0a23707a4dfdabe9d/llama.h#L22

That was definitely a good move towards future proofing. Any breaking changes could just add +1 to that version and all would be fine and dandy for the next 4294967295 versions of the model format.

But then came this commit: https://github.com/ggerganov/llama.cpp/comm

[... truncated for brevity ...]

---

## Issue #N/A: Create clear instructions for downloading and converting the models

**Link**: https://github.com/ggml-org/llama.cpp/issues/644
**State**: closed
**Created**: 2023-03-31T02:23:32+00:00
**Closed**: 2023-05-03T18:43:16+00:00
**Comments**: 2
**Labels**: documentation, enhancement

### Description

Clear instructions are needed to allow new arrivals to download and convert the models, in spite of the multiple format versions (non quantised, quantised, various llama versions etc) .

I would suggest that each llama or alpaca etc print a version on startup, and that the conversions scripts have this in their name, and also that a program reading a file and figuring out what it is from a magic print the version read and the version expected even if it aborts.  

Edmund



---

## Issue #N/A: --help may show the wrong default values when used after other arguments

**Link**: https://github.com/ggml-org/llama.cpp/issues/573
**State**: closed
**Created**: 2023-03-28T13:26:10+00:00
**Closed**: 2023-04-02T02:41:14+00:00
**Comments**: 0
**Labels**: bug, documentation

### Description

For example, running `./main -b 512 --help` will show the help and say that 512 is the default batch size, which is wrong. This may lead to confusion.

---

## Issue #N/A: Help populating the examples README.md files

**Link**: https://github.com/ggml-org/llama.cpp/issues/518
**State**: closed
**Created**: 2023-03-26T07:25:05+00:00
**Closed**: 2023-07-28T19:21:19+00:00
**Comments**: 2
**Labels**: documentation, help wanted, good first issue

### Description

For now I just added empty README.md files:

- https://github.com/ggerganov/llama.cpp/tree/master/examples/main
- https://github.com/ggerganov/llama.cpp/tree/master/examples/quantize
- https://github.com/ggerganov/llama.cpp/tree/master/examples/perplexity
- https://github.com/ggerganov/llama.cpp/tree/master/examples/embedding
- etc.

It would be great to add usage instructions and various tips and tricks for better experience for each example.

Great task for initial contributions

---

## Issue #N/A: Change ./main help output to better reflect context size's affect on generation length

**Link**: https://github.com/ggml-org/llama.cpp/issues/449
**State**: closed
**Created**: 2023-03-24T01:38:43+00:00
**Closed**: 2023-07-28T19:40:24+00:00
**Comments**: 2
**Labels**: documentation, enhancement

### Description

### Discussed in https://github.com/ggerganov/llama.cpp/discussions/446

<div type='discussions-op-text'>

<sup>Originally posted by **cmp-nct** March 24, 2023</sup>
I've been testing alpaca 30B (-t 24 -n 2000 --temp 0.2 -b 32 --n_parts 1 --ignore-eos --instruct)
I've consistently have it "stop" after 300-400 tokens output (30-40 tokens input)
No error message, no crash and given the -n 2000 and the ignore-eos no reason to stop so early

I guess it would be useful if the program provides a verbose quit reason, though in my case I can't see any reason for it to stop before token max is reached.


I'm not sure if that's a bug to report or if I am missing something.</div>

---

## Issue #N/A: How to output text to a file?

**Link**: https://github.com/ggml-org/llama.cpp/issues/432
**State**: closed
**Created**: 2023-03-23T17:06:06+00:00
**Closed**: 2023-03-24T15:19:36+00:00
**Comments**: 2
**Labels**: documentation

### Description

I really, really tried hard to understand and modify the code but I am not an expert on C++ and so I find it a little bit difficult to change parts of this software. Is there a way to simply execute a command and get the output without all of that verbosity?

---

## Issue #N/A: [Documentation] C API examples

**Link**: https://github.com/ggml-org/llama.cpp/issues/384
**State**: closed
**Created**: 2023-03-22T08:08:14+00:00
**Closed**: 2023-06-16T18:58:42+00:00
**Comments**: 11
**Labels**: documentation

### Description

Hey!

There should be a simple example on how to use the new C API (like one that simply takes a hardcoded string and runs llama on it until \n or something like that).
Not sure the the `/examples/` directory is appropriate for this.

Thanks
Niansa

---

## Issue #N/A: Add proper instructions for using Alpaca models

**Link**: https://github.com/ggml-org/llama.cpp/issues/382
**State**: closed
**Created**: 2023-03-22T07:26:07+00:00
**Closed**: 2023-07-28T19:20:56+00:00
**Comments**: 22
**Labels**: documentation, help wanted, good first issue, high priority, 🦙.

### Description

So I am looking at https://github.com/antimatter15/alpaca.cpp and I see they are already running 30B Alpaca models, while we are struggling to run 7B due to the recent tokenizer updates.

I also see that the models are now even floating on Hugging Face - I guess license issues are no longer a problem?

We should add detailed instructions for obtaining the Alpaca models and a temporary explanation how to use the following script to make the models compatible with the latest `master`:

https://github.com/ggerganov/llama.cpp/issues/324#issuecomment-1476227818

The bigger issue is that people keep producing the old version of the `ggml` models instead of migrating to the latest `llama.cpp` changes. And therefore, we now need this extra conversion step. It's best to figure out the steps for generating the Alpaca models and generate them in the correct format.

**Edit: just don't post direct links to the models!**

---

## Issue #N/A: Invalid model error : too old, regenerate your model files!

**Link**: https://github.com/ggml-org/llama.cpp/issues/361
**State**: closed
**Created**: 2023-03-21T16:51:17+00:00
**Closed**: 2023-03-22T05:54:53+00:00
**Comments**: 14
**Labels**: documentation, model

### Description

Downloaded Alpaca 7B model successfully using the following command as mentioned in README.md:
`curl -o ./models/ggml-alpaca-7b-q4.bin -C - https://gateway.estuary.tech/gw/ipfs/QmUp1UGeQFDqJKvtjbSYPBiZZKRjLp8shVP9hT8ZB9Ynv1`

When I try to execute the command:
`main -m ./models/ggml-alpaca-7b-q4.bin --color -f ./prompts/alpaca.txt -ins`

This is the error output:
main: seed = 1679417098
llama_model_load: loading model from './models/ggml-alpaca-7b-q4.bin' - please wait ...
llama_model_load: invalid model file './models/ggml-alpaca-7b-q4.bin' (too old, regenerate your model files!)
main: failed to load model from './models/ggml-alpaca-7b-q4.bin'

How to fix this? Is the downloaded model corrupted and should I download it again? What is the SHA1 hash of the model so that I can verify that the downloaded model is corrupted or not?

---

## Issue #N/A: Create issue template for bug and enhancement issues

**Link**: https://github.com/ggml-org/llama.cpp/issues/239
**State**: closed
**Created**: 2023-03-17T13:38:57+00:00
**Closed**: 2023-03-21T17:50:50+00:00
**Comments**: 3
**Labels**: documentation, good first issue

### Description

The following is a proposed template for creating new issues. If people think the tone could be improved, I'd appreciate feedback!
___

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ ] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ ] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Please provide a detailed written description of what you were trying to do, and what you expected `lamma.cpp` to do.

# Curren

[... truncated for brevity ...]

---

## Issue #N/A: Document check sums of models so that we can confirm issues are not caused by bad downloads or conversion

**Link**: https://github.com/ggml-org/llama.cpp/issues/238
**State**: closed
**Created**: 2023-03-17T12:50:44+00:00
**Closed**: 2023-05-02T13:41:32+00:00
**Comments**: 9
**Labels**: documentation, model

### Description

Can someone please confirm the following md5 sums are correct?  I regenerated them with the latest code.

```
$ md5sum ./models/*/*.pth | sort -k 2,2
0804c42ca65584f50234a86d71e6916a  ./models/13B/consolidated.00.pth
016017be6040da87604f77703b92f2bc  ./models/13B/consolidated.01.pth
f856e9d99c30855d6ead4d00cc3a5573  ./models/30B/consolidated.00.pth
d9dbfbea61309dc1e087f5081e98331a  ./models/30B/consolidated.01.pth
2b2bed47912ceb828c0a37aac4b99073  ./models/30B/consolidated.02.pth
ea0405cdb5bc638fee12de614f729ebc  ./models/30B/consolidated.03.pth
9deae67e2e7b5ccfb2c738f390c00854  ./models/65B/consolidated.00.pth
0c4b00c30460c3818bd184ee949079ee  ./models/65B/consolidated.01.pth
847194df776dd38f8ae9ddcede8829a1  ./models/65B/consolidated.02.pth
3b6c8adcb5654fd36abab3206b46a0f1  ./models/65B/consolidated.03.pth
68d61d1242597ad92616ec31b8cb6b4c  ./models/65B/consolidated.04.pth
7f71259eaee2b906aa405d8edf39925f  ./models/65B/consolidated.05.pth
0574e26b6891ab2cb0df7340d773fe

[... truncated for brevity ...]

---

## Issue #N/A: Add the disk requirements

**Link**: https://github.com/ggml-org/llama.cpp/issues/195
**State**: closed
**Created**: 2023-03-16T03:23:50+00:00
**Closed**: 2023-03-16T11:54:44+00:00
**Comments**: 0
**Labels**: documentation, duplicate

### Description

Hi,

I found all the infos about the models:
https://cocktailpeanut.github.io/dalai/#/?id=_7b

You can put on readme the space requirements.

Thanks.

---

## Issue #N/A: How to build on windows?

**Link**: https://github.com/ggml-org/llama.cpp/issues/103
**State**: closed
**Created**: 2023-03-13T20:13:14+00:00
**Closed**: 2023-07-28T19:20:41+00:00
**Comments**: 22
**Labels**: documentation, good first issue, windows

### Description

Please give instructions. There is nothing in README but it says that it supports it 

---

## Issue #N/A: benchmarks?

**Link**: https://github.com/ggml-org/llama.cpp/issues/34
**State**: closed
**Created**: 2023-03-12T05:20:58+00:00
**Closed**: 2024-04-09T01:10:24+00:00
**Comments**: 57
**Labels**: documentation, question, stale

### Description

Where are the benchmarks for various hardware - eg. apple silicon 

---

## Issue #N/A: [Q] Memory Requirements for Different Model Sizes

**Link**: https://github.com/ggml-org/llama.cpp/issues/13
**State**: closed
**Created**: 2023-03-11T12:19:07+00:00
**Closed**: 2023-03-18T21:02:00+00:00
**Comments**: 18
**Labels**: documentation, question

### Description

No description provided.

---

