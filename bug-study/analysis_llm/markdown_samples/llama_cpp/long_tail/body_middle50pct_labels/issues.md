# body_middle50pct_labels - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 3
- Closed Issues: 27

### Label Distribution

- bug-unconfirmed: 9 issues
- enhancement: 8 issues
- stale: 6 issues
- good first issue: 6 issues
- bug: 5 issues
- help wanted: 4 issues
- model: 4 issues
- need more info: 4 issues
- low severity: 3 issues
- high priority: 3 issues

---

## Issue #N/A: Bug: `-ngl` is missing from server docs for layer offload

**Link**: https://github.com/ggml-org/llama.cpp/issues/8605
**State**: closed
**Created**: 2024-07-20T20:45:02+00:00
**Closed**: 2024-09-05T01:07:00+00:00
**Comments**: 4
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

There's no mention of how to offload layers to gpu

### Name and Version

docs

### What operating system are you seeing the problem on?

Linux

### Relevant log output

_No response_

---

## Issue #N/A: Bug: There is an issue to execute llama-baby-llama.

**Link**: https://github.com/ggml-org/llama.cpp/issues/9478
**State**: closed
**Created**: 2024-09-14T02:21:32+00:00
**Closed**: 2024-10-01T08:33:00+00:00
**Comments**: 2
**Labels**: bug-unconfirmed, low severity

### Description

### What happened?

Whenever I go to execute llama-baby-llama, I get “ggml/src/ggml.c:6793: GGML_ASSERT(false && ‘backwards pass not implemented’) failed “ error.

### Name and Version

./llama-baby-llama -m ./models/Qwen-7B-Chat/Qwen-7B-Chat-Q4_0.gguf -p "I believe the meaning of life is" -n 128

### What operating system are you seeing the problem on?

Linux

### Relevant log output

_No response_

---

## Issue #N/A: Bug: Random inputs generated automatically in llama-cli

**Link**: https://github.com/ggml-org/llama.cpp/issues/9456
**State**: closed
**Created**: 2024-09-12T20:07:34+00:00
**Closed**: 2024-11-16T01:59:06+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, stale, low severity

### Description

### What happened?

I am running the cuurent model : 

 ./llama.cpp/llama-cli -m /home/piuser/Desktop/Abhrant/Meta-Llama-3-8B.Q4_K_S.gguf -n 512 --repeat_penalty 1.0 --color -i -r "User:" -f llama.cpp/prompts/chat-with-bob.txt
 
 When I do this, the cli starts and the conversation goes on normally. Sometimes, a random input is automatically taken even when I am not giving it. 
 For example:
<img width="923" alt="Screenshot 2024-09-13 at 1 35 13 AM" src="https://github.com/user-attachments/assets/8eccdfdd-e625-464b-9f6d-5d8bc85208f5">


I have added the question "what can you do? ". I have not added the input "I love you Bob." it automatically came up after the answer to "what can you do? " was generated. Any idea why? 

 

### Name and Version

version: 3733 (1b280614)
built with cc (Debian 12.2.0-14) 12.2.0 for aarch64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
[1726171239] == Running in interactive mode. =

[... truncated for brevity ...]

---

## Issue #N/A: Clean up server code

**Link**: https://github.com/ggml-org/llama.cpp/issues/5762
**State**: closed
**Created**: 2024-02-28T10:32:39+00:00
**Closed**: 2024-12-13T16:24:20+00:00
**Comments**: 3
**Labels**: enhancement, good first issue

### Description

## Motivation

As seen on https://github.com/ggerganov/llama.cpp/issues/4216 , one of the important task is to refactor / clean up the server code so that it's easier to maintain. However, without a detailed plan, personally I feel like it's unlikely to be archived.

This issue is created so that we can discuss about how to refactor or clean up the code.

The goal is to help existing and new contributors to easily find out where to work in the code base.

## Current architecture

The current server implementation has 2 thread: one for HTTP part and one for inference.

![image](https://github.com/ggerganov/llama.cpp/assets/7702203/6e44b6cc-04f0-465c-a3fb-dc5c4f13b8ae)

- The direction from HTTP ==> inference thread is done by `llama_server_queue.post(task)`
- The direction from inference ==> HTTP thread is done by `llama_server_response.send(result)`

## Ideas

Feel free to suggest any ideas that you find helpful (please keep in mind that we do not introduce new featu

[... truncated for brevity ...]

---

## Issue #N/A: Store KV cache of computed prompts to disk to avoid re-compute in follow-up runs

**Link**: https://github.com/ggml-org/llama.cpp/issues/64
**State**: closed
**Created**: 2023-03-12T21:55:25+00:00
**Closed**: 2023-04-29T02:57:37+00:00
**Comments**: 10
**Labels**: enhancement, help wanted, good first issue, high priority, 🦙.

### Description

Idea from: https://github.com/ggerganov/llama.cpp/issues/23#issuecomment-1465308592

We can add a `--cache_prompt` flag that if added will dump the computed KV caches of the prompt processing to the disk in a file with name produced by the hash of the prompt. Next time you run, it will first check if we have stored KV cache for this hash and load it straight from disk instead of computing it.

Great task for contributing to the project!

---

## Issue #N/A: [Feature Request] Ability to rewind model evaluation by a fixed number of tokens

**Link**: https://github.com/ggml-org/llama.cpp/issues/1281
**State**: closed
**Created**: 2023-05-02T15:00:02+00:00
**Closed**: 2023-05-05T01:52:30+00:00
**Comments**: 13
**Labels**: enhancement, good first issue, high priority

### Description

The recent additions of the state and session APIs have made it possible to implement caching for llama models which has greatly improved the responsiveness in many applications.

The current APIs howeve still leave something to be desired, specifically it would be very useful to be able to rewind / rollback an evaluated model by a fixed number of tokens so a single longer saved state could be used to restore any shorter state.

---

## Issue #N/A: Bug: Decoding special tokens in T5

**Link**: https://github.com/ggml-org/llama.cpp/issues/8938
**State**: closed
**Created**: 2024-08-08T16:32:39+00:00
**Closed**: 2024-08-09T16:53:10+00:00
**Comments**: 6
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

I have a T5/lora model trained to output some text separated by the `<extra_id_0>` special token (the tokenizer properly works after following instructions in #8872) .

When running the model using Huggingface's transformers/peft, it generates the expected output. However, when I use `llama-cli`, what happens instead is that the moment the first such token is reached, it's actually decoded into an `EOG` token instead of the extra token and generation is stopped.

I might be simply doing something wrong in using the library.

### Name and Version

version: 3549 (afd27f01)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu


### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

---

## Issue #N/A: Bug: not support langchain v0.3 to use tools

**Link**: https://github.com/ggml-org/llama.cpp/issues/10214
**State**: closed
**Created**: 2024-11-08T08:59:18+00:00
**Closed**: 2024-12-23T01:30:30+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, stale, high severity

### Description

### What happened?

request: request: POST /v1/chat/completions 192.168.139.86 500

request:
{
	"messages": [{
		"content": "98平米的房屋总价是多少",
		"role": "user"
	}],
	"model": "qwen-plus",
	"n": 1,
	"stream": false,
	"temperature": 0.7,
	"tools": [{
		"type": "function",
		"function": {
			"name": "magic_function",
			"description": "根据房屋面积，计算房屋价格。input 是房屋面积单位是平米，返回的结果是房屋价格，单位是元",
			"parameters": {
				"properties": {
					"input": {
						"type": "integer"
					}
				},
				"required": ["input"],
				"type": "object"
			}
		}
	}]
}

response:

{
	"error": {
		"code": 500,
		"message": "Unsupported param: tools",
		"type": "server_error"
	}
}


### Name and Version

(base) [root@localhost llama.cpp-master]# ./llama-cli --version
version: 0 (unknown)
built with cc (GCC) 9.3.1 20200408 (Red Hat 9.3.1-2) for x86_64-redhat-linux

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
request: POST /v1/chat/c

[... truncated for brevity ...]

---

## Issue #N/A: Bug: phi-3-mini-4k-it July update failing to load.

**Link**: https://github.com/ggml-org/llama.cpp/issues/8845
**State**: closed
**Created**: 2024-08-03T12:32:44+00:00
**Closed**: 2024-08-05T11:35:13+00:00
**Comments**: 3
**Labels**: bug-unconfirmed, high severity

### Description

### What happened?

i am trying to load the phi-3-mini july update model as usual but its giving me the following error:

```
llama_model_load: error loading model: error loading model hyperparameters: key not found in model: phi3.attention.sliding_window
llama_load_model_from_file: failed to load model
llama_init_from_gpt_params: error: failed to load model '.\models\me\phi-3-mini-4k-it-July-5\Phi-3.1-mini-4k-instruct-Q8_0_L.gguf'
main: error: unable to load model
```

Also, phi-2 and phi-3 original model still work! If its worth knowing, i have also downloaded the latest version of LM Studio, and its also unable to run this same model, throwing the same error.

### Name and Version

PS F:\ai3> .\llama.cpp\build\bin\Release\llama-cli.exe --version
version: 3505 (b72c20b8)
built with MSVC 19.40.33811.0 for x64

### What operating system are you seeing the problem on?

Windows

### Relevant log output

```shell
PS F:\ai3> .\llama.cpp\build\bin\Release\llama-cli

[... truncated for brevity ...]

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

## Issue #N/A: convert.py can not identify type of pytorch BF16 tensors

**Link**: https://github.com/ggml-org/llama.cpp/issues/2504
**State**: closed
**Created**: 2023-08-03T19:02:37+00:00
**Closed**: 2023-08-14T10:37:54+00:00
**Comments**: 6
**Labels**: bug, help wanted

### Description

The original model files have tensors stored in BF16 which have a wider numeric range than F16. The quality will suffer too much if the norm tensors are converted to lower precision, I believe this is the reason why they are always stored in F32 since there is no support for BF16 in ggml.

When I tried to list the tensor types in convert.py for experimentation purposes, I discovered that all BF16 tensors in pytorch and safetensors models are identified as F16, but are for some unknown reason correctly converted to F32. BF16 tensors in .pth model files are correctly identified.

Directly following this line:
https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/convert.py#L88

Insert this:
```
        if tensor.data_type == DT_F16:  print(name + " DT_F16")
        if tensor.data_type == DT_F32:  print(name + " DT_F32")
        if tensor.data_type == DT_BF16: print(name + " DT_BF16")

        if tensor.data_type == DT_F16:
            return D

[... truncated for brevity ...]

---

## Issue #N/A: llama : add Falcon LLM support

**Link**: https://github.com/ggml-org/llama.cpp/issues/1602
**State**: closed
**Created**: 2023-05-26T17:45:06+00:00
**Closed**: 2023-08-23T20:11:44+00:00
**Comments**: 210
**Labels**: help wanted, model

### Description

Falcon LLM 40b and 7b were just open sourced under a license which allows commercial use (~~with royalties for over $1 million revenue per year~~) and have are topping the Huggingface Open LLM leaderboard. It seems to be based on a modified gpt3 architecture. I’m wondering if support in llama.cpp would be considered.

https://huggingface.co/tiiuae/falcon-40b

---

## Issue #N/A: invalid model file './models/ggml-alpaca-7b-q4.bin' (too old, regenerate your model files!)

**Link**: https://github.com/ggml-org/llama.cpp/issues/329
**State**: closed
**Created**: 2023-03-20T14:56:00+00:00
**Closed**: 2023-03-20T15:32:21+00:00
**Comments**: 7
**Labels**: need more info, model

### Description

Hi, I have encounter the above problem when running the alpaca model. I download the model from the link "https://gateway.estuary.tech/gw/ipfs/QmQ1bf2BTnYxq73MFJWu1B7bQ2UD6qG7D7YDCxhTndVkPC" which is one of the three options from the readme. Should I download the model from somewhere else? 

---

## Issue #N/A: Will llama.cpp be able to use Phi-2 ?

**Link**: https://github.com/ggml-org/llama.cpp/issues/4437
**State**: closed
**Created**: 2023-12-13T12:02:56+00:00
**Closed**: 2023-12-18T17:27:49+00:00
**Comments**: 27
**Labels**: enhancement, good first issue, model

### Description

Surely we have to wait for a GGUF version, but in the meantime just curious about it

thanks

---

## Issue #N/A: Not having enough memory just causes a segfault or something

**Link**: https://github.com/ggml-org/llama.cpp/issues/257
**State**: closed
**Created**: 2023-03-18T07:28:43+00:00
**Closed**: 2023-05-06T18:03:16+00:00
**Comments**: 9
**Labels**: bug, duplicate, hardware, model

### Description

So. I'm trying to build with CMake on Windows 11 and the thing just stops after it's done loading the model.

![image](https://user-images.githubusercontent.com/4723091/226091364-64a488a7-ebb5-4c24-9dd0-1cb81378008d.png)

And apparently, this is a segfault.

![Screenshot_20230318_121935](https://user-images.githubusercontent.com/4723091/226091335-afbf2712-d2b8-4b88-9b44-6b6a43d78565.png)

Yay yay yyayy yyayay

this is a memory allocation failure it seems, from me not having enough memory. not like llama.cpp Tells Me That lmao, it just segfaults

(`ctx->mem_buffer` is nullptr which probably means the malloc just failed)

---

## Issue #N/A: Bug: Crash with GGML CUDA error when inferencing on llama-server

**Link**: https://github.com/ggml-org/llama.cpp/issues/8117
**State**: closed
**Created**: 2024-06-25T18:21:37+00:00
**Closed**: 2024-06-26T06:28:03+00:00
**Comments**: 9
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

llama-server is crashing repeatably with a GGML CUDA error on commit a818f30 and later.  d62e4aa and earlier work correctly.  I have not been able to reproduce this with llama-cli.

`/opt/llama.cpp-a818f30/bin/llama-server --host localhost --port 18443 --n-gpu-layers 81 --ctx-size 8192 --model meta-llama-3-70b-instruct-q4_k.gguf`

In addition to the log I posted, I also tried launching on a single GPU with only one GPU layer, but the result is the same. 
`CUDA_VISIBLE_DEVICES=0 /opt/llama.cpp-a818f30/bin/llama-server --host localhost --port 18443 --n-gpu-layers 1 --ctx-size 8192 --model meta-llama-3-70b-instruct-q4_k.gguf`

Even zero GPU layers will cause a crash.
`CUDA_VISIBLE_DEVICES=0 /opt/llama.cpp-a818f30/bin/llama-server --host localhost --port 18443 --n-gpu-layers 0 --ctx-size 8192 --model meta-llama-3-70b-instruct-q4_k.gguf`

This may be related to #8096 @JohannesGaessler 

### Name and Version

$ /opt/llama.cpp-a818f30/bin/llama-server --version


[... truncated for brevity ...]

---

## Issue #N/A: Bug: SYCL builds >= b4069 have half the context limit of previous builds

**Link**: https://github.com/ggml-org/llama.cpp/issues/10421
**State**: closed
**Created**: 2024-11-20T08:32:21+00:00
**Closed**: 2024-11-28T05:49:36+00:00
**Comments**: 5
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

```
found 1 SYCL devices:
|  |                   |                                       |       |Max    |        |Max  |Global |
    |
|  |                   |                                       |       |compute|Max work|sub  |mem    |
    |
|ID|        Device Type|                                   Name|Version|units  |group   |group|size   |       Driver version|
|--|-------------------|---------------------------------------|-------|-------|--------|-----|-------|---------------------|
| 0| [level_zero:gpu:0]|                Intel Arc A770 Graphics|    1.5|    512|    1024|   32| 16704M|            1.3.31093|
llama_kv_cache_init:      SYCL0 KV buffer size =  3937.50 MiB
llama_new_context_with_model: KV self size  = 3937.50 MiB, K (f16): 1968.75 MiB, V (f16): 1968.75 MiB
llama_new_context_with_model:  SYCL_Host  output buffer size =    27.82 MiB
ggml_backend_sycl_buffer_type_alloc_buffer: can't malloc 3278637056 Bytes memory on deviceggml_gallocr

[... truncated for brevity ...]

---

## Issue #N/A: Bug: Vulkan backend crash on model loading

**Link**: https://github.com/ggml-org/llama.cpp/issues/8828
**State**: closed
**Created**: 2024-08-02T13:32:45+00:00
**Closed**: 2024-08-18T13:58:08+00:00
**Comments**: 13
**Labels**: bug-unconfirmed, critical severity

### Description

### What happened?

I mainly use LLamaSharp C# bindings, after updating to v0.14.0 and releasing Vulkan backend, I decided to give it a try instead using CPU inference, but on loading model it crash with console output

```
WARNING: [Loader Message] Code 0 : windows_read_data_files_in_registry: Registry lookup failed to get layer manifest files.
WARNING: [Loader Message] Code 0 : Layer VK_LAYER_RENDERDOC_Capture uses API version 1.2 which is older than the application specified API version of 1.3. May cause issues.
llama_model_loader: loaded meta data with 25 key-value pairs and 327 tensors from C:\Models\Text\Index-1.9B-Character\Index-1.9B-Character-Q6_K.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = Index-1

[... truncated for brevity ...]

---

## Issue #N/A: Compile bug: Cann x86_64 not building

**Link**: https://github.com/ggml-org/llama.cpp/issues/12945
**State**: closed
**Created**: 2025-04-14T15:41:52+00:00
**Closed**: 2025-04-15T10:39:22+00:00
**Comments**: 3
**Labels**: bug, build, Ascend NPU

### Description

### Git commit

```
[ 13%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/llamafile/sgemm.cpp.o
/llama.cpp/ggml/src/ggml-cann/aclnn_ops.cpp: In function 'void ggml_cann_get_rows(ggml_backend_cann_context&, ggml_tensor*)':
/llama.cpp/ggml/src/ggml-cann/aclnn_ops.cpp:1786:49: error: 'float16_t' was not declared in this scope; did you mean 'float_t'?
 1786 |                 src0->data, ACL_FLOAT16, sizeof(float16_t), scale_ne, scale_nb,
      |                                                 ^~~~~~~~~
      |                                                 float_t
gmake[2]: *** [ggml/src/ggml-cann/CMakeFiles/ggml-cann.dir/build.make:90: ggml/src/ggml-cann/CMakeFiles/ggml-cann.dir/aclnn_ops.cpp.o] Error 1
gmake[2]: *** Waiting for unfinished jobs....
gmake[1]: *** [CMakeFiles/Makefile2:1790: ggml/src/ggml-cann/CMakeFiles/ggml-cann.dir/all] Error 2
gmake[1]: *** Waiting for unfinished jobs....
[ 13%] Linking CXX shared library ../../bin/libggml-cpu.so
[ 13%] Built target ggml

[... truncated for brevity ...]

---

## Issue #N/A: ggml : unified CMake build

**Link**: https://github.com/ggml-org/llama.cpp/issues/6913
**State**: open
**Created**: 2024-04-25T19:15:40+00:00
**Comments**: 4
**Labels**: enhancement, build, refactoring, roadmap

### Description

Currently the [ggml](https://github.com/ggerganov/ggml), [llama.cpp](https://github.com/ggerganov/llama.cpp) and [whisper.cpp](https://github.com/ggerganov/whisper.cpp) projects share the same source of the `ggml` library, but have different CMake scripts. The scripts are adapted to the specifics of the projects and are quite similar with each other - all of them build `ggml`. Still, there are differences due to manually rewriting them and applying changes from one repo to another

The goal in this task is to unify, deduplicate and streamline the build process of `ggml` with proper CMake scripts that are shared across the projects. This will simplify changes in the future and will also help other 3rd party projects that depend on `ggml`

More on this topic has been discussed in:

- https://github.com/ggerganov/llama.cpp/issues/5890
- https://github.com/ggerganov/ggml/pull/804

To achieve that, the `ggml`-related sources in `llama.cpp` and `whisper.cpp` would likely have to be 

[... truncated for brevity ...]

---

## Issue #N/A: Docker “--all-in-one” fails with ModuleNotFoundError: No module named ‘tqdm’

**Link**: https://github.com/ggml-org/llama.cpp/issues/289
**State**: closed
**Created**: 2023-03-19T10:51:52+00:00
**Closed**: 2023-03-20T08:24:13+00:00
**Comments**: 7
**Labels**: bug, duplicate, build

### Description

On Win 10
```
>  docker run -v /llama/models:/models ghcr.io/ggerganov/llama.cpp:full –all-in-one “/models/” 7B
Downloading model…
Traceback (most recent call last):
  File “/app/./download-pth.py”, line 3, in <module>
    from tqdm import tqdm
ModuleNotFoundError: No module named ‘tqdm’
```

---

## Issue #N/A: Feature Request: (webui) do not throw away message if there is error in stream

**Link**: https://github.com/ggml-org/llama.cpp/issues/13709
**State**: open
**Created**: 2025-05-22T15:00:03+00:00
**Comments**: 2
**Labels**: enhancement, good first issue, server/webui

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggml-org/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggml-org/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Currently, if the UI got an error while it's generating the text, it will throw away the generating message.

The most simple way to test is to Ctrl+C to kill the server while it's generating a response.

The expected behavior is to show a meaningful error like what they do on chatgpt

<img width="680" alt="Image" src="https://github.com/user-attachments/assets/a3734cef-3e47-4fda-b12b-231f74bdf43f" />

### Motivation

N/A

### Possible Implementation

_No response_

---

## Issue #N/A: Feature Request: allow setting jinja chat template from server webui

**Link**: https://github.com/ggml-org/llama.cpp/issues/11689
**State**: closed
**Created**: 2025-02-05T22:46:03+00:00
**Closed**: 2025-06-22T01:08:17+00:00
**Comments**: 5
**Labels**: enhancement, server/webui, stale

### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Allow setting jinja chat template from server webui. Should be the same way with change system message (via the Settings dialog)

### Motivation

N/A

### Possible Implementation

_No response_

---

## Issue #N/A: server: support control vectors

**Link**: https://github.com/ggml-org/llama.cpp/issues/6316
**State**: open
**Created**: 2024-03-26T07:25:43+00:00
**Comments**: 0
**Labels**: enhancement, good first issue, server/webui

### Description

### Motivation

It would be nice to support control vectors in the servers.


### Requirements
- Configure `gpt_params::control_vectors` from `common`
- Tests the feature using the framework

#### References
- A first attemp has been made here: #6289

---

## Issue #N/A: llama : support sliding window attention

**Link**: https://github.com/ggml-org/llama.cpp/issues/3377
**State**: closed
**Created**: 2023-09-28T12:12:40+00:00
**Closed**: 2024-11-01T01:21:36+00:00
**Comments**: 21
**Labels**: performance, stale

### Description

For more info, see: https://github.com/mistralai/mistral-src and references there in.

Also: https://arxiv.org/pdf/2310.06825v1.pdf

With #3228 it should be relatively easy to support this.

---

## Issue #N/A: Windows page fault disk i/o slow on first load

**Link**: https://github.com/ggml-org/llama.cpp/issues/705
**State**: closed
**Created**: 2023-04-02T10:04:24+00:00
**Closed**: 2024-04-11T01:07:14+00:00
**Comments**: 37
**Labels**: performance, windows, stale

### Description

Hello,

As of https://github.com/ggerganov/llama.cpp/pull/613 I have experienced significant regression in model loading speed (I'm on windows, compiled msvc llama.cpp, llama.cpp is located on HDD to prevent SSD wear in my case)

It takes roughly 15 minutes for model to load first time after each computer restart/hibernation, during this time my HDD usage is at 100% and my non-llama.cpp read/write operations are slowed down on my pc
![hdd](https://user-images.githubusercontent.com/76458234/229345728-b597023b-f7e3-4a8b-b550-3159863ba03d.png)

Before that, previous commits took 60 - 180 seconds at worst to load model first time, and after first loading occured, model loaded within 5 - 10 seconds on each program restart until pc reboot/hibernation

Before Commit:
![timings2](https://user-images.githubusercontent.com/76458234/229347345-2053d645-0f26-42ef-9f8e-5fc69ad04e1c.png)

After:
![timings1](https://user-images.githubusercontent.com/76458234/229345966-ee606c92-e7cb-42f6-8

[... truncated for brevity ...]

---

## Issue #N/A: [fixed]The last code build with memory fix running result is not good in my pc.

**Link**: https://github.com/ggml-org/llama.cpp/issues/462
**State**: closed
**Created**: 2023-03-24T14:22:06+00:00
**Closed**: 2023-03-27T00:13:38+00:00
**Comments**: 10
**Labels**: bug, performance

### Description

Be obviously slower with Q_1 30b model. And the memory usage become garbage...
(Linux 5.19 x64 Ubuntu base)

---

## Issue #N/A: Model runs but doesn't produce any output

**Link**: https://github.com/ggml-org/llama.cpp/issues/204
**State**: closed
**Created**: 2023-03-16T10:46:53+00:00
**Closed**: 2023-03-16T12:52:24+00:00
**Comments**: 5
**Labels**: need more info

### Description

I checked everything several times and quantized it, but both models do not output anything, in which mode I would not run them, the processor loads, but there is no output, no matter how long I wait
 input to the console also does not lead to anything

for ubuntu 22.04 8gb+15 swap (everything fits)


![Снимок экрана от 2023-03-16 11-42-21](https://user-images.githubusercontent.com/93709232/225592978-99f3c8a6-85a0-4606-a39d-6ddc1e334778.png)


---

## Issue #N/A: Error while converting to ggml.py format

**Link**: https://github.com/ggml-org/llama.cpp/issues/260
**State**: closed
**Created**: 2023-03-18T12:02:31+00:00
**Closed**: 2023-04-14T13:13:30+00:00
**Comments**: 1
**Labels**: need more info

### Description

After running the command: "python3 convert-pth-to-ggml.py /Users/tanish.shah/llama.cpp/models/7B/ 1"
Error with sentencepiece:

```
Traceback (most recent call last):
  File "/Users/tanish.shah/llama.cpp/convert-pth-to-ggml.py", line 75, in <module>
    tokenizer = sentencepiece.SentencePieceProcessor(fname_tokenizer)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/tanish.shah/llama.cpp/env/lib/python3.11/site-packages/sentencepiece/__init__.py", line 447, in Init
    self.Load(model_file=model_file, model_proto=model_proto)
  File "/Users/tanish.shah/llama.cpp/env/lib/python3.11/site-packages/sentencepiece/__init__.py", line 905, in Load
    return self.LoadFromFile(model_file)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/tanish.shah/llama.cpp/env/lib/python3.11/site-packages/sentencepiece/__init__.py", line 310, in LoadFromFile
    return _sentencepiece.SentencePieceProcessor_LoadFromFile(self, arg)
           ^^^^^^^^^^

[... truncated for brevity ...]

---

## Issue #N/A: Create json api service

**Link**: https://github.com/ggml-org/llama.cpp/issues/88
**State**: closed
**Created**: 2023-03-13T10:19:23+00:00
**Closed**: 2023-07-28T19:29:40+00:00
**Comments**: 8
**Labels**: need more info

### Description

so we can intergrate app/UI.

---

