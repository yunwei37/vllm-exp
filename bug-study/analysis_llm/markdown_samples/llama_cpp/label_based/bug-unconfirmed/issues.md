# bug-unconfirmed - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 29

### Label Distribution

- bug-unconfirmed: 30 issues
- stale: 16 issues
- need more info: 3 issues
- high severity: 2 issues
- server/webui: 2 issues
- low severity: 1 issues

---

## Issue #N/A: b2447 (c47cf41) decreased output quality

**Link**: https://github.com/ggml-org/llama.cpp/issues/6571
**State**: closed
**Created**: 2024-04-09T18:50:49+00:00
**Closed**: 2024-05-24T13:29:29+00:00
**Comments**: 17
**Labels**: need more info, bug-unconfirmed

### Description

With identical seeds and options, b2447 (https://github.com/ggerganov/llama.cpp/commit/c47cf414efafb8f60596edc7edb5a2d68065e992) produces different output that seems lower in quality compared to b2446. Is it possible to preserve old output quality in new builds?

System: MacBook Pro w/ i5-1038NG7

---

## Issue #N/A: make process hangs if LLAMA_CUBLAS=1, at the line that includes the file scripts/get-flags.mk for the second time

**Link**: https://github.com/ggml-org/llama.cpp/issues/4575
**State**: closed
**Created**: 2023-12-21T21:18:17+00:00
**Closed**: 2024-04-02T01:10:16+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

I run make LLAMA_CUBLAS=1 and that process hangs. I used make --debug=f to figure out that make gets stuck at the line that includes get-flags.mk for the second time (it is already included a few lines before). 

# Environment and Context

+-----------------------

[... truncated for brevity ...]

---

## Issue #N/A: Bug:  Recent changes break Rocm compile on windows

**Link**: https://github.com/ggml-org/llama.cpp/issues/8612
**State**: closed
**Created**: 2024-07-21T08:53:56+00:00
**Closed**: 2024-07-21T14:39:23+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

It cannot compile after CUDA: MMQ code deduplication + iquant support on windows.
that's my guess that pr break compile.

### Name and Version

b3428 and Windows 11 rocm5.7.1

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
cmake --build . -j 99 --parallel 32 --config Release
[1/61] Linking CXX shared library bin\ggml.dll
FAILED: bin/ggml.dll ggml/src/ggml.lib
cmd.exe /C "cmd.exe /C "C:\Strawberry\c\bin\cmake.exe -E __create_def W:\git\llama.cpp\rocm_1100\ggml\src\CMakeFiles\ggml.dir\.\exports.def W:\git\llama.cpp\rocm_1100\ggml\src\CMakeFiles\ggml.dir\.\exports.def.objs --nm=C:\Strawberry\c\bin\nm.exe && cd W:\git\llama.cpp\rocm_1100" && C:\PROGRA~1\AMD\ROCm\5.7\bin\CLANG_~1.EXE -fuse-ld=lld-link -nostartfiles -nostdlib -O3 -DNDEBUG -D_DLL -D_MT -Xclang --dependent-lib=msvcrt  -Xlinker /DEF:ggml\src\CMakeFiles\ggml.dir\.\exports.def -shared -o bin\ggml.dll  -Xlinker /MANIFEST:EMBED -Xlinker /implib:ggml

[... truncated for brevity ...]

---

## Issue #N/A: Server: Multimodal Model Input Parameter No longer Exists

**Link**: https://github.com/ggml-org/llama.cpp/issues/7112
**State**: closed
**Created**: 2024-05-07T04:11:00+00:00
**Closed**: 2024-07-18T01:06:49+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale

### Description

I have noticed that when using the server that the --mmproj parameter for multimodal models has been disabled. Although it still remains in the README. Is there an alternative to --mmproj , I cannot seem to find one in the code. 

Any help on this would be great. 

Code to reproduce:
`./server -m ./ggml-model-q4_k.gguf --mmproj ./mmproj-model-f16.gguf -ngl 1`

Error:
`
error: unknown argument: --mmproj
usage: ./server [options]
`

---

## Issue #N/A: Misc. bug: Potential out of bound in rerank

**Link**: https://github.com/ggml-org/llama.cpp/issues/13549
**State**: open
**Created**: 2025-05-14T19:50:38+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

version: 5387 (3198405e)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu

### Operating systems

_No response_

### Which llama.cpp modules do you know to be affected?

_No response_

### Command line

```shell

```

### Problem description & steps to reproduce

[llama_context](https://github.com/ggml-org/llama.cpp/blob/f5170c1d7a66222ca7c75d2022fec3ed87257e0b/src/llama-context.cpp#L807) resize the rerank output to size 1 while [here](https://github.com/ggml-org/llama.cpp/blob/017f10b5fa630a013ec4f9936e410a60d4f460d5/examples/embedding/embedding.cpp#L69) we still normalize it as if we have full embedding vector. I found this problem happened randomly in python binding but cannot reproduce it in cpp. Not sure if it is a bug in cpp side.

### First Bad Commit

_No response_

### Relevant log output

```shell

```

---

## Issue #N/A: Misc. bug: Retrieval sample not decoding token successfully

**Link**: https://github.com/ggml-org/llama.cpp/issues/13102
**State**: closed
**Created**: 2025-04-24T22:26:15+00:00
**Closed**: 2025-06-08T01:08:06+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

version: 5184 (87616f06)
built with MSVC 19.41.34120.0 for x64

### Operating systems

Mac, Windows

### Which llama.cpp modules do you know to be affected?

Other (Please specify in the next section)

### Command line

```shell
llama-retrieval.exe --context-file <any_text_file> --chunk-size 1 -c 512 -t 8 -m bge-large-en-v1.5-f32.gguf
```

### Problem description & steps to reproduce

The sample failed to decode any tokens created from the text embeddings.

It looks like  we need to skip the kv-cache logic to look for an unused slot when pooling is active (which is true for the above model).

The following IF in llama-context.cpp is removed, causing us to go into this logic to search for an unused slot and hit the decoding spew.

        // non-causal masks do not use the KV cache
        if (hparams.causal_attn) {
            kv_self_update();

Just adding "if (!embd_pooling)" appears to fix the issue but I am not sure what it does to the original logic for the n

[... truncated for brevity ...]

---

## Issue #N/A: Since last update Mistral models doesn't works anymore

**Link**: https://github.com/ggml-org/llama.cpp/issues/7450
**State**: closed
**Created**: 2024-05-22T03:53:03+00:00
**Closed**: 2024-05-22T10:03:22+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

since this https://github.com/ggerganov/llama.cpp/tree/b2961
phi3-128k works better (if ctx <32k)
but mistral models are crazy, I tried 7bQ2 7bQ8, 70BQ2XS, none of them works anymore

```
Log start
main
main: build = 2961 (201cc11a)
main: built with cc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0 for x86_64-linux-gnu
main: seed  = 1716349593
llama_model_loader
llama_model_loader: loaded meta data with 24 key-value pairs and 291 tensors from models/mistral-7b-instruct-v0.2.Q2_K.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = mistralai_mistral-7b-instruct-v0.2
llama_model_loader: - kv   2:                       llama.context_length u32              = 32768
llama_model_loader: - kv   3:                     llama.

[... truncated for brevity ...]

---

## Issue #N/A: ggml_validate_row_data finding nan value for IQ4_NL

**Link**: https://github.com/ggml-org/llama.cpp/issues/7311
**State**: closed
**Created**: 2024-05-15T19:40:10+00:00
**Closed**: 2024-05-18T00:39:55+00:00
**Comments**: 8
**Labels**: bug-unconfirmed

### Description

Using b2854

Converted Hermes-2-Theta-Llama-3-8B to F32, then measured imatrix with https://gist.github.com/bartowski1182/b6ac44691e994344625687afe3263b3a

Upon quanting, all sizes work fine, except for IQ4_NL which produces this output:

```
load_imatrix: imatrix dataset='/training_data/calibration_data.txt'
load_imatrix: loaded 224 importance matrix entries from /models/Hermes-2-Theta-Llama-3-8B-GGUF/Hermes-2-Theta-Llama-3-8B.imatrix computed on 189 chunks
prepare_imatrix: have 224 importance matrix entries
main: build = 2854 (72c177c1)
main: built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
main: quantizing '/models/Hermes-2-Theta-Llama-3-8B-GGUF/Hermes-2-Theta-Llama-3-8B-f32.gguf' to '/models/Hermes-2-Theta-Llama-3-8B-GGUF/Hermes-2-Theta-Llama-3-8B-IQ4_NL.gguf' as IQ4_NL
llama_model_loader: loaded meta data with 23 key-value pairs and 291 tensors from /models/Hermes-2-Theta-Llama-3-8B-GGUF/Hermes-2-Theta-Llama-3-8B-f32.gguf (version GGUF V3 (late

[... truncated for brevity ...]

---

## Issue #N/A: Compile bug: tools build failing

**Link**: https://github.com/ggml-org/llama.cpp/issues/13614
**State**: closed
**Created**: 2025-05-18T14:26:43+00:00
**Closed**: 2025-07-02T01:07:53+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Git commit

Commit 6a2bc8b


### Operating systems

Windows

### GGML backends

CUDA

### Problem description & steps to reproduce

Environment: Windows 11 with CUDA toolkit installed.

Sorry, new to this. I tried searching if there was already a solution but couldn't find anything with my limited domain of knowledge.

I followed the guide to build llama.cpp with CUDA support which seems to worked as it built a few binaries that I can see in the bin/Release folder, but I noticed none of the tools were built. I.g. cli, server etc...

Also, my environment was missing CURL libraries, so I had to look it up and install a windows version. And issued the following to build this:

```
cmake -B build -DGGML_CUDA=ON -DCURL_LIBRARY=c:\Curl\lib\libcurl.a -DCURL_INCLUDE_DIR=c:\Curl\include
```

Reading up on the llama-server docs, I saw there was a way to build it so I tried it but I got this error:
```
common.lib(arg.obj) : error LNK2019: unresolved external symbol __imp_curl_slist_appe
nd re

[... truncated for brevity ...]

---

## Issue #N/A: Segmentation fault during inference on AMD gfx900 with codebooga-34b-v0.1.Q5_K_M.gguf

**Link**: https://github.com/ggml-org/llama.cpp/issues/6031
**State**: closed
**Created**: 2024-03-13T01:47:52+00:00
**Closed**: 2024-03-14T18:46:32+00:00
**Comments**: 15
**Labels**: bug-unconfirmed

### Description

Hi,

I compiled `llama.cpp` from git, todays master HEAD `commit 8030da7afea2d89f997aeadbd14183d399a017b9` on Fedora Rawhide (ROCm 6.0.x) like this:
```
CC=/usr/bin/clang CXX=/usr/bin/clang++ cmake .. -DLLAMA_HIPBLAS=ON -DAMDGPU_TARGETS=gfx900 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="--rocm-device-lib-path=/usr/lib/clang/17/amdgcn/bitcode"
make -j 16
```

Then I tried to run a prompt using the `codebooga-34b-v0.1.Q5_K_M.gguf` model which I got from here: https://huggingface.co/TheBloke/CodeBooga-34B-v0.1-GGUF

I kept the prompt simple and used the following command:
./main -t 10 -ngl 16 -m ~/models/codebooga-34b-v0.1.Q5_K_M.gguf --color -c 2048 --temp 0.7 --repeat_penalty 1.1 -n -1 -p "### Instruction: How do I get the length of a Vec in Rust?\n### Response:"

I have an AMD Instinct MI25 card with 16GB VRAM, according to `nvtop` with `-ngl 16` about half of it is used `8.219Gi/15.984`, so this does not seem to be an OOM issue.

The console output looks like this:
`

[... truncated for brevity ...]

---

## Issue #N/A: convert.py incompatible with most new models, including salesforce/codegen models

**Link**: https://github.com/ggml-org/llama.cpp/issues/6030
**State**: closed
**Created**: 2024-03-13T00:57:03+00:00
**Closed**: 2024-04-27T01:06:32+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale

### Description

Most newer models on huggingface are unable to be converted with convert.py, including all the Salesforce/codegen, Salesforce/codegen2, and Salesforce/codegen25 models. Examples of output of different salesforce models below

### codegen25-7b-multi
https://huggingface.co/Salesforce/codegen25-7b-multi
```
(pythonenv) raptor85@raptor1 /var/storage/llama/llama.cpp $ python convert.py huggingface/Salesforce/codegen25-7b-multi --outfile models/salesforce-codegen25-7b-multi.gguf --outtype f16
Loading model file huggingface/Salesforce/codegen25-7b-multi/pytorch_model-00001-of-00003.bin
Loading model file huggingface/Salesforce/codegen25-7b-multi/pytorch_model-00001-of-00003.bin
Loading model file huggingface/Salesforce/codegen25-7b-multi/pytorch_model-00002-of-00003.bin
Loading model file huggingface/Salesforce/codegen25-7b-multi/pytorch_model-00003-of-00003.bin
params = Params(n_vocab=51200, n_embd=4096, n_layer=32, n_ctx=2048, n_ff=11008, n_head=32, n_head_kv=32, n_experts=None, n

[... truncated for brevity ...]

---

## Issue #N/A: response dont see underscore 

**Link**: https://github.com/ggml-org/llama.cpp/issues/5335
**State**: closed
**Created**: 2024-02-05T08:43:11+00:00
**Closed**: 2024-05-04T01:06:43+00:00
**Comments**: 5
**Labels**: need more info, bug-unconfirmed, stale

### Description

Hello, when I use Llama.cpp as inference server, I found the result will have the following problem.

for example, if the expected result is a_b_c  but the result from Llama.cpp will be a_bc.   anyone know the reason? 

---

## Issue #N/A: quantize.exe Bug(s) --token-embedding-type / --output-tensor-type and  - Docu? Advanced Usage Context ?

**Link**: https://github.com/ggml-org/llama.cpp/issues/6776
**State**: closed
**Created**: 2024-04-20T01:43:18+00:00
**Closed**: 2024-04-30T02:05:16+00:00
**Comments**: 0
**Labels**: bug-unconfirmed

### Description

Windows 11. Use of quantize.exe - missing documentation?

I am trying to locate information on:
--include-weights tensor_name: use importance matrix for this/these tensor(s)
--exclude-weights tensor_name: use importance matrix for this/these tensor(s)

Specifically the format of "tensor_name(s)" to be used and/or file to be provided and used with these options.
Is it looking for a imatrix.dat or a file with "tensor name(s) : Q6_K" for example ?

I can see the output and names during execution - just need to know what format(s) that "--include-weights" is expecting/valid.
Not sure if this is a bug or not.

Same for this ( BUG? ) :
  --token-embedding-type ggml_type:
  --output-tensor-type ggml_type:

These do not seem work when using "Q8_0", "Q6_0" etc etc as in:
--token-embedding-type Q8_0
--token-embedding-type ggml_type_Q8_0
--token-embedding-type ggml_type:Q8_0

Example:

./quantize --output-tensor-type Q8_0 --token-embedding-type Q8_0 --imatrix imatrix.dat mo

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: Jinja not replacing `date_string`

**Link**: https://github.com/ggml-org/llama.cpp/issues/12729
**State**: closed
**Created**: 2025-04-03T05:13:42+00:00
**Closed**: 2025-05-15T01:39:52+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

```
$ ~/llama.cpp/build/bin/llama-cli --version
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA A800-SXM4-80GB MIG 7g.80gb, compute capability 8.0, VMM: yes
version: 5002 (2c3f8b85)
built with x86_64-conda-linux-gnu-cc (conda-forge gcc 11.4.0-13) 11.4.0 for x86_64-conda-linux-gnu
```

### Operating systems

Linux

### GGML backends

CUDA

### Hardware

AMD EPYC 7742 64-Core Processor + A800-SXM4-80GB

### Models

_No response_

### Problem description & steps to reproduce

Compile llama.cpp from source and run it with `~/llama.cpp/build/bin/llama-server -m /models/Llama-3.3-70B-Instruct-Q8_0.gguf --port 8000 -t 8 -ngl 81 -c 15360 --jinja`

### First Bad Commit

_No response_

### Relevant log output

```shell
$ ~/llama.cpp/build/bin/llama-server -m /models/Llama-3.3-70B-Instruct-Q8_0.gguf --port 8000 -t 8 -ngl 81 -c 15360 --jinja
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    n

[... truncated for brevity ...]

---

## Issue #N/A: Temperature slider not working

**Link**: https://github.com/ggml-org/llama.cpp/issues/6676
**State**: closed
**Created**: 2024-04-14T16:43:44+00:00
**Closed**: 2024-06-10T01:34:27+00:00
**Comments**: 6
**Labels**: server/webui, bug-unconfirmed, stale

### Description

Tried in version b2671. The temperature slider doesn't seem to do anything no matter the model, even when cranked all the way to 2 which should produce gibberish but the model behaves coherently instead. All other frontend params are set to default using the button on the top.

`.\bin\server.exe --model .\mistral-7b-instruct-v0.2.Q4_0.gguf --ctx-size 2048 --n-gpu-layers 99 --log-disable --path .\frontend`

The frontend folder is also updated to the latest as of today in examples/server/public.

---

## Issue #N/A: Bug: llama-cli templating does buf.resize(-1) if the model's template is not supported, causing crash

**Link**: https://github.com/ggml-org/llama.cpp/issues/8149
**State**: closed
**Created**: 2024-06-27T05:55:33+00:00
**Closed**: 2024-06-27T16:14:20+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

`common.cpp`'s `llama_chat_apply_template` says:

https://github.com/ggerganov/llama.cpp/blob/ac146628e47451c531a3c7e62e6a973a2bb467a0/common/common.cpp#L2630-L2637

`res` can be -1 (e.g. in the case that the model's Jinja template is not matched by any pattern in `llama_chat_apply_template_internal`?). When cast to `size_t` it becomes significantly bigger than `buf.size()` which leads to `buf.resize(-1)`. This is followed by a crash of `llama-cli` on my machine.

`llama-server` seems to fall back to `chatml` if no pattern matches the model's Jinja template:

https://github.com/ggerganov/llama.cpp/blob/ac146628e47451c531a3c7e62e6a973a2bb467a0/examples/server/server.cpp#L2600-L2605

Perhaps the same needs to be done for `llama-cli`, and perhaps `common.cpp`'s `llama_chat_apply_template` should be more defensive when `llama_chat_apply_template_internal` returns -1, rather than trying to resize `buf` to -1

### Name and Version

```
$ ./llama-cli --version


[... truncated for brevity ...]

---

## Issue #N/A: Bug: phi 3.5 mini produces garbage past 4096 context

**Link**: https://github.com/ggml-org/llama.cpp/issues/9127
**State**: closed
**Created**: 2024-08-22T01:16:01+00:00
**Closed**: 2024-10-25T01:28:13+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

Phi 3.5 mini doesn't produce <|end|> or <|endoftext|> when the context is set higher than 4096, just endless garbage tokens.  Possible rope scale issue?

### Name and Version

llama-server, recent compile

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

---

## Issue #N/A: server: Unable to Utilize Models Outside of 'ChatML' with OpenAI Library

**Link**: https://github.com/ggml-org/llama.cpp/issues/5921
**State**: closed
**Created**: 2024-03-07T11:31:07+00:00
**Closed**: 2024-04-21T01:06:36+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale

### Description

I'm unsure whether this is a limitation of the OpenAI library or a result of poor server management. However, after extensively testing various models using the latest server image in Docker with CUDA, I've come to a conclusion. It seems impossible to run a model that utilizes a chat template different from ChatML along with OpenAI library. All attempts resulted in failures in responses. This includes the model located at https://huggingface.co/mlabonne/AlphaMonarch-7B-GGUF, which I requested some time ago. I apologize if this isn't considered a bug, but I'm at a loss for what to do next. Thank you in advance.

---

## Issue #N/A: Misc. bug: Llama-Server is missing --Prompt-Cache from Llama-CLI

**Link**: https://github.com/ggml-org/llama.cpp/issues/12437
**State**: closed
**Created**: 2025-03-17T23:40:29+00:00
**Closed**: 2025-03-20T17:00:52+00:00
**Comments**: 2
**Labels**: bug-unconfirmed

### Description

### Name and Version

Llama-server is **missing** prompt caching from llama-cli.

Here are the parameters in Llama-cli:

```
--prompt-cache FNAME                    file to cache prompt state for faster startup (default: none)
--prompt-cache-all                      if specified, saves user input and generations to cache as well
--prompt-cache-ro                       if specified, uses the prompt cache but does not update it
```

Figured it might be a bug as, for the most part, I've seen feature parity across these binaries. May we please have this addressed?

Thank you all.

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell

```

### Problem description & steps to reproduce

See above 

### First Bad Commit

_No response_

### Relevant log output

```shell

```

---

## Issue #N/A: [server] phi-3 uses <|endoftext|> instead of <|end|> when applying chat template in /chat/completions

**Link**: https://github.com/ggml-org/llama.cpp/issues/7432
**State**: closed
**Created**: 2024-05-21T06:53:47+00:00
**Closed**: 2024-05-23T14:15:16+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

When using phi-3 without the option `--chat-template phi3`, the tokenization is incorrect. 

For example, if I **do** use `--chat-template phi3`, here is the log output when I send the message "hi":
```json
{
    "level": "VERB",
    "function": "update_slots",
    "line": 1954,
    "msg": "prompt tokenized",
    "id_slot": 0,
    "id_task": 1,
    "n_ctx": 8192,
    "n_keep": 0,
    "n_prompt_tokens": 7,
    "prompt_tokens": "<s><|system|><|end|><|user|> hi<|end|><|assistant|>"
}
```

actually the extra space after <|user|> is concerning, it should be a newline, but maybe that's just an artifact of how the log message is formatted.

But here's what happens when the `--chat-template phi3` is omitted:

```json
{
    "level": "VERB",
    "function": "update_slots",
    "line": 1954,
    "msg": "prompt tokenized",
    "id_slot": 0,
    "id_task": 0,
    "n_ctx": 8192,
    "n_keep": 0,
    "n_prompt_tokens": 11,
    "prompt_tokens": "<s><|system|><|endoftex

[... truncated for brevity ...]

---

## Issue #N/A: Prebuilt windows binaries (i.e. `server.exe`) have a dependency on `llama.dll` - can we remove that?

**Link**: https://github.com/ggml-org/llama.cpp/issues/4314
**State**: closed
**Created**: 2023-12-03T22:00:54+00:00
**Closed**: 2023-12-04T17:38:15+00:00
**Comments**: 4
**Labels**: bug-unconfirmed

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Current Behavior & Expected Behavior

I downloaded [llama-b1606-bin-win-noavx-x64.zip](https://github.com/ggerganov/llama.cpp/releases/download/b1606/llama-b1606-bin-win-noavx-x64.zip) from the [latest release](https://github.com/ggerganov/llama.cpp/releases/tag/b1606) and would like to

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: Gemma3 adapter gguf conversion fails

**Link**: https://github.com/ggml-org/llama.cpp/issues/12551
**State**: closed
**Created**: 2025-03-24T18:34:49+00:00
**Closed**: 2025-03-25T22:03:11+00:00
**Comments**: 6
**Labels**: bug-unconfirmed

### Description

### Name and Version


```bash
>llama-cli --version
version: 4948 (00d53800)
built with MSVC 19.43.34808.0 for x64

>pip list
Package                      Version
---------------------------- ---------------
absl-py                      0.11.0
accelerate                   0.27.2
addict                       2.4.0
aggdraw                      1.3.18.post0
aiohttp                      3.9.5
aiosignal                    1.3.1
annotated-types              0.7.0
ansicon                      1.89.0
antlr4-python3-runtime       4.9.3
anyio                        4.3.0
async-timeout                4.0.3
attrs                        23.2.0
beautifulsoup4               4.13.3
blend_modes                  2.1.0
blessed                      1.20.0
blind-watermark              0.4.4
blobfile                     3.0.0
Brotli                       1.1.0
bs4                          0.0.2
cachetools                   5.3.3
certifi                      2024.2.2
cffi                         1.16.0
chars

[... truncated for brevity ...]

---

## Issue #N/A: Error C2026 when building ggml-opencl.cpp with MSVC

**Link**: https://github.com/ggml-org/llama.cpp/issues/3973
**State**: closed
**Created**: 2023-11-06T21:22:11+00:00
**Closed**: 2024-04-02T01:12:09+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [x] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

MSVC can't handle long string litterals, so it throws out [Error C2026](https://learn.microsoft.com/en-us/cpp/error-messages/compiler-errors-1/compiler-error-c2026?view=msvc-170) when compiling the long strings that are constructed with the macro `MULTILINE_QUOTE( ... )` .

There is an ea

[... truncated for brevity ...]

---

## Issue #N/A: The output of the main service is inconsistent with that of the server service

**Link**: https://github.com/ggml-org/llama.cpp/issues/6569
**State**: closed
**Created**: 2024-04-09T15:40:35+00:00
**Closed**: 2024-05-27T01:06:36+00:00
**Comments**: 10
**Labels**: need more info, server/webui, bug-unconfirmed, stale

### Description

**When the same quantitative model is used for server service and main service, some specific words are answered differently. It seems that the input specific words are not received or received incorrectly.
For example, BYD, Tesla, Lexus and other car names have this problem, such as Geely, BMW, Audi and so on is normal.**
The specific problem is manifested in: When obtaining the word "BYD" in the server service, non-Chinese characters such as "ruit" are not obtained or obtained. As in the first example, when asked about BYD car, the reply only involved the car, and BYD was lost.
**Test results in the server**
********************************************************
**These are three examples of problems（BYD）**
********************************************************
{
  content: ' 汽车是一种交通工具，它通常由发动机，变速箱，底盘和底盘系统，悬挂系统，转向系统，车身和车轮等组成。汽车通常由汽油或柴油发动机提供动力，通过变速箱和传动系统来控制车辆行驶的速度和方向。汽车的设计和制造技术不断提高，汽车的功能也越来越强大。现在汽车已经不仅仅是一种交通工具，它已经成为人们日常生活中不可或缺的一部分，提供了各种便利。汽车在现代社会中的作用非常广泛，它可以满足人们的出行需求，同时也可以娱

[... truncated for brevity ...]

---

## Issue #N/A: Misc. bug: webui multimodal, image input is not supported by this server, server error 500

**Link**: https://github.com/ggml-org/llama.cpp/issues/13566
**State**: closed
**Created**: 2025-05-15T14:23:12+00:00
**Closed**: 2025-05-15T16:34:30+00:00
**Comments**: 12
**Labels**: bug-unconfirmed

### Description

### Name and Version

./build/bin/llama-server --version
version: 5394 (6c8b9150)
built with cc (GCC) 14.2.1 20250207 for x86_64-pc-linux-gnu

### Operating systems

Linux

### Which llama.cpp modules do you know to be affected?

llama-server

### Command line

```shell
./build/bin/llama-server -m ../../_models/SmolVLM-500M-Instruct-Q8_0.gguf
```

### Problem description & steps to reproduce

trying out new llama-server with multimodal vision recognition using the demo at https://github.com/ngxson/smolvlm-realtime-webcam (serving index.html with "python -m http.server" to access it with Chrome browser).

Run this command:

`./build/bin/llama-server -m ../../_models/SmolVLM-500M-Instruct-Q8_0.gguf`


Pressed "Start" button under the demo GUI.
Got this error:

```
got exception: {"code":500,"message":"image input is not supported by this server","type":"server_error"}
srv  log_server_r: request: POST /v1/chat/completions 127.0.0.1 500
```



### First Bad Commit

_No response_

### Relev

[... truncated for brevity ...]

---

## Issue #N/A: Something might be wrong with either llama.cpp or the Llama 3 GGUFs

**Link**: https://github.com/ggml-org/llama.cpp/issues/6914
**State**: closed
**Created**: 2024-04-25T22:09:50+00:00
**Closed**: 2024-05-11T10:51:02+00:00
**Comments**: 14
**Labels**: bug-unconfirmed

### Description

Try this query: "What is 3333+777?"

Yes, yes, LLMs are bad at math. That's not what I'm getting at. [Someone mentioned this on Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1cci5w6/quantizing_llama_3_8b_seems_more_harmful_compared/l169ovf/), and I have to agree that I'm seeing weird stuff too.

Let's get a baseline. Here is what [meta.ai](https://meta.ai) yields:

![image](https://github.com/ggerganov/llama.cpp/assets/726063/b4555bf8-aa7b-4156-a0c0-2fa3c6353110)

This is likely running on Llama 3 70B.

Here is what Groq yields:

![image](https://github.com/ggerganov/llama.cpp/assets/726063/4e8813e0-7cf3-4c7d-8109-b34c8523091c)

and at 8B:

![image](https://github.com/ggerganov/llama.cpp/assets/726063/12fbba62-c7b8-4eef-8454-0093ed0dc7ab)

Now, here's where things get weird. Using Open WebUI on top of Ollama, let's use llama.cpp to run the GGUFs of Llama 3.

First, 8B at fp16:

![image](https://github.com/ggerganov/llama.cpp/assets/726063/da103655-43b7-44cf

[... truncated for brevity ...]

---

## Issue #N/A: Eval bug: does llama.cpp support Intel AMX instruction? how to enable it

**Link**: https://github.com/ggml-org/llama.cpp/issues/12003
**State**: closed
**Created**: 2025-02-21T13:00:24+00:00
**Closed**: 2025-05-04T01:08:02+00:00
**Comments**: 18
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

llama-cli

### Operating systems

Linux

### GGML backends

AMX

### Hardware

XEON 8452Y + NV A40 

### Models

_No response_

### Problem description & steps to reproduce

as title

### First Bad Commit

_No response_

### Relevant log output

```shell
as title
```

---

## Issue #N/A: Misc. bug: convert_hf_to_gguf.py fails to convert the model of architecture T5ForConditionalGeneration

**Link**: https://github.com/ggml-org/llama.cpp/issues/12862
**State**: closed
**Created**: 2025-04-10T07:16:43+00:00
**Closed**: 2025-05-25T01:08:13+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale

### Description

### Name and Version

```
$ llama-cli --version
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = NVIDIA GeForce RTX 2060 (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 0 | matrix cores: KHR_coopmat
version: 0 (unknown)
built with FreeBSD clang version 19.1.7 (https://github.com/llvm/llvm-project.git llvmorg-19.1.7-0-gcd708029e0b2) for x86_64-unknown-freebsd14.2
```

Version: 5097

### Operating systems

BSD

### Which llama.cpp modules do you know to be affected?

_No response_

### Command line

```shell
convert_hf_to_gguf.py --outfile sage-v1.1.0.gguf ai-forever/sage-v1.1.0
```

### Problem description & steps to reproduce

1. Download https://huggingface.co/ai-forever/sage-v1.1.0
2. Run the above command

It fails:
```
...
WARNING:hf-to-gguf:Couldn't find context length in config.json, assuming default value of 512
INFO:hf-to-gguf:Set model tokenizer
Traceback (most recent call last):
  File "/usr/ports/misc/llama-cpp/work/llama.cpp-b5097/convert_

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] windows build 1833 opencl version

**Link**: https://github.com/ggml-org/llama.cpp/issues/4892
**State**: closed
**Created**: 2024-01-12T11:36:41+00:00
**Closed**: 2024-04-03T01:14:01+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, stale

### Description

Reproduce: ./main
Result: 
ggml_opencl:clGetPlatformIDs brbrbr error -1001 
ggml-opencl.cpp:965

---

## Issue #N/A: Build docker image llama.cpp:server-cuda: CMakeLists.txt missing

**Link**: https://github.com/ggml-org/llama.cpp/issues/10844
**State**: closed
**Created**: 2024-12-15T22:14:51+00:00
**Closed**: 2024-12-15T22:29:48+00:00
**Comments**: 1
**Labels**: bug-unconfirmed

### Description

### Git commit

docker build

### Operating systems

Linux

### GGML backends

CUDA

### Problem description & steps to reproduce

Jetson Linux 36.4 on Orin NX 16GB.
[NVidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed

The following command fails with an error:
`sudo docker build -t local/llama.cpp:server-cuda -f llama-server-cuda.Dockerfile .`

Error: 
`CMake Error: The source directory "/app" does not appear to contain CMakeLists.txt`

### First Bad Commit

_No response_

### Relevant log output

```shell
sudo docker build -t local/llama.cpp:server-cuda -f llama-server-cuda.Dockerfile .
[+] Building 112.9s (12/14)                                                                                                            docker:default
 => [internal] load build definition from llama-server-cuda.Dockerfile                                                                           0.0s
 => => transferr

[... truncated for brevity ...]

---

