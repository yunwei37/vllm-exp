# historical_1year_plus - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- inactive: 11 issues
- good first issue: 2 issues
- documentation: 1 issues
- bug: 1 issues
- high priority: 1 issues

---

## Issue #N/A: Is it possible to define the prompts for KV caching up-front?

**Link**: https://github.com/sgl-project/sglang/issues/401
**State**: closed
**Created**: 2024-04-29T08:40:04+00:00
**Closed**: 2024-07-25T06:33:23+00:00
**Comments**: 2
**Labels**: inactive

### Description

For a lot of use cases, there is already a pre-defined system + base prompt that is used.

Can we define the KV cache for these prompts up front manually? For example, if we are extracting information out of a provided context, the provided context prompt changes but the system + base prompt stays the same. Caching the context will make no sense as it is guaranteed to change on the next inference. 

---

## Issue #N/A: Accelerating Generation with Rollback using sglang?

**Link**: https://github.com/sgl-project/sglang/issues/73
**State**: closed
**Created**: 2024-01-22T04:40:25+00:00
**Closed**: 2024-01-22T04:56:40+00:00
**Comments**: 2

### Description

Hi team! My generation scenario involves rolling back and I was wondering how I could speed this up using sglang. 

In the first stage, I have an initial prompt, and I can obtain an output with sentences delimited by '\n'. 
Input: 
question.

Output:
sentence1 \n sentence2 \n sentence3 \n

In the second stage, I would like to rollback and generate on these prompts:
- question. sentence1 \n
- question. sentence1 \n sentence2 \n

Is it possible to reuse the KV caches in the first stage using sglang? Thanks for your help!

---

## Issue #N/A: llava http request hang when do set_default_backend(RuntimeEndpoint("http://ip:port"))

**Link**: https://github.com/sgl-project/sglang/issues/497
**State**: closed
**Created**: 2024-06-03T09:53:37+00:00
**Closed**: 2024-06-03T11:53:39+00:00
**Comments**: 0

### Description

I'm trying to start the server for llava-video-34b using 2 GPUs and I'm following code in [srt_example_llava_v.py](https://github.com/sgl-project/sglang/blob/main/examples/usage/llava_video/srt_example_llava_v.py). 
Everything is OK when I start the backend, and it can also do generation. But when I start a frontend python file using set_default_backend(RuntimeEndpoint("http://localhost:30000")), the program will always hang without any output information on backend and frontend terminal.
And I find the program stuck in this part
`res = http_request(
            self.base_url + "/get_model_info",
            auth_token=self.auth_token,
            api_key=self.api_key,
            verify=self.verify,
        )`

Dose anyone know what should I do to solve this issue? Thanks~
My backend runtime code is
`runtime = sgl.Runtime(
        model_path=args.model_path,
        tokenizer_path=tokenizer_path,
        port=cur_port,
        model_overide_args=model_overide_args,
  

[... truncated for brevity ...]

---

## Issue #N/A: Running LLaVA 1.5 4bit AWQ with SGLang

**Link**: https://github.com/sgl-project/sglang/issues/237
**State**: closed
**Created**: 2024-02-26T21:57:27+00:00
**Closed**: 2024-03-04T16:42:35+00:00
**Comments**: 2

### Description

👋 hi and thanks again for all the updates and improvements on this framework.

I've tried running SGLang with AWQ version of LLaVA and ran into the following error:

```console
$ python3 -m sglang.launch_server --model-path Shopify/llava-awq-test --tokenizer-path llava-hf/llava-1.5-7b-hf --host 0.0.0.0 --port 30000 --tp-size 1
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Rank 0: load weight begin.
quant_config: AWQConfig(weight_bits=4, group_size=128, zero_point=True)
/opt/conda/envs/sglang_awq/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_st

[... truncated for brevity ...]

---

## Issue #N/A: Parallelism with `run_batch` vs `fork`

**Link**: https://github.com/sgl-project/sglang/issues/295
**State**: closed
**Created**: 2024-03-13T20:35:36+00:00
**Closed**: 2024-04-07T10:09:39+00:00
**Comments**: 1
**Labels**: documentation

### Description

First of all, great work!

The frontend seems to support two kinds of parallel processing: batching and forking.

From the docs and paper, it is not entirely clear to me how they differ and how they are handled under the hood. Do they both launch separate threads that make requests to the server, which then does continuous batching? Or is there more to it?

From a practical standpoint, what are the considerations when both `run_batch` and `fork` are possible for the use case? Are there advantages/disadvantages besides fork being more flexible?

Is it safe to combine the two? Would the total number of threads be `num_threads * num_forks`?

---

## Issue #N/A: Setting Data Type from the CLI interface

**Link**: https://github.com/sgl-project/sglang/issues/325
**State**: closed
**Created**: 2024-03-22T23:22:13+00:00
**Closed**: 2024-07-25T06:32:54+00:00
**Comments**: 2
**Labels**: inactive

### Description

Is it possible to set the data type from the cli interface?

```
python -m sglang.launch_server --model-path llava-v1.6-34b.Q8_0.gguf --tokenizer-path liuhaotian/llava-v1.6-34b-tokenizer --port 8888 --host 0.0.0.0 --enable-flashinfer --dtype bfloat16
```

If not, it seems like a useful feature to add.

---

## Issue #N/A: Error under high concurrency. `sqlite3.DatabaseError: database disk image is malformed`

**Link**: https://github.com/sgl-project/sglang/issues/143
**State**: closed
**Created**: 2024-02-05T03:52:57+00:00
**Closed**: 2024-02-07T17:39:06+00:00
**Comments**: 5

### Description

The error stack is as follows:
```bash
File "/User/jay/sglang/python/sglang/api.py", line 37, in Runtime
  from sglang.srt.server import Runtime
File "/User/jay/sglang/python/sglang/srt/server.py", line 47, in <module>
  from sglang.srt.managers.router.manager import start_router_process
File "/User/jay/sglang/python/sglang/srt/managers/router/manager.py", line 8, in <module>
  from sglang.srt.managers.router.model_rpc import ModelRpcClient
File "/User/jay/sglang/python/sglang/srt/managers/router/model_rpc.py", line 15, in <module>
  from sglang.srt.constrained.fast_forward import FastForwardCache
File "/User/jay/sglang/python/sglang/srt/constrained/fast_forward.py", line 2, in <module>
  from sglang.srt.constrained.disk_cache import disk_cache
File "/User/jay/sglang/python/sglang/srt/constrained/disk_cache.py", line 13, in <module>
  memory = Cache(cache_dir, eviction_policy="none", cull_limit=0)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/

[... truncated for brevity ...]

---

## Issue #N/A: Potential Bug? Confusion about "need_vision" in llava implementation

**Link**: https://github.com/sgl-project/sglang/issues/341
**State**: closed
**Created**: 2024-04-01T00:27:04+00:00
**Closed**: 2024-07-25T06:33:00+00:00
**Comments**: 1
**Labels**: inactive

### Description

Thank you for the amazing work! I am trying to understand the specific implementation of llava. Specifically, what is the purpose of `need_vision`? 
My understanding is that currently, this value decides whether images are processed by the vision encoder, and added to the sequence of input_embeddings. 

```python
# Embed vision input
need_vision = (
(positions[input_metadata.extend_start_loc] < self.image_feature_len).cpu().numpy()
            )
# FIXME: We need to substract the length of the system prompt
```
However, I am struggling to understand why / when this condition should be set to False, and what's the rationale for using `self.image_feature_len`? In particular in Llava 1.6, the sequence length of an image would be dynamic based on the aspect ratio and size of the image whereas `self.image_feature_len` is only initialized and set once: 
`self.image_feature_len = int((self.image_size / self.patch_size) ** 2)` which seems to be just the number of tokens in a single "

[... truncated for brevity ...]

---

## Issue #N/A: OpenAI ChatCompletionRequest max_tokens defaults to None causing error

**Link**: https://github.com/sgl-project/sglang/issues/582
**State**: closed
**Created**: 2024-07-02T20:11:18+00:00
**Closed**: 2024-07-09T08:52:56+00:00
**Comments**: 2

### Description

I encountered a bug while using the sglang OpenAI API library. I have to specify max_tokens if using the `/v1/chat/completions` endpoint but not with the `/v1/completions` endpoint. I believe this is because of the default `max_tokens` value being set in `python/sglang/srt/openai_protocol.py`.


Default is 16 tokens https://github.com/sgl-project/sglang/blob/9380f50ff9cbc36afc1888c7a5b69f53c9a488f5/python/sglang/srt/openai_protocol.py#L31-L41


ChatCompletionRequest max_tokens defaults to None https://github.com/sgl-project/sglang/blob/9380f50ff9cbc36afc1888c7a5b69f53c9a488f5/python/sglang/srt/openai_protocol.py#L128-L137

Here's the end of the error:
```bash
  File "/usr/local/lib/python3.11/site-packages/sglang/srt/sampling_params.py", line 66, in verify
    if self.max_new_tokens < 0:
       ^^^^^^^^^^^^^^^^^^^^^^^
TypeError: '<' not supported between instances of 'NoneType' and 'int'
```
This can be fixed by setting a default or checking if max_tokens is None and han

[... truncated for brevity ...]

---

## Issue #N/A: vLLM import error

**Link**: https://github.com/sgl-project/sglang/issues/391
**State**: closed
**Created**: 2024-04-24T23:15:31+00:00
**Closed**: 2024-07-18T16:28:54+00:00
**Comments**: 7

### Description

I'm getting the following import error:

```
sgl ➜ export CUDA_VISIBLE_DEVICES=4; python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
Traceback (most recent call last):
  File "/home/jessy/.miniconda3/envs/sgl/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/jessy/.miniconda3/envs/sgl/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/jessy/projects/sglang/python/sglang/launch_server.py", line 3, in <module>
    from sglang.srt.server import ServerArgs, launch_server
  File "/home/jessy/projects/sglang/python/sglang/srt/server.py", line 56, in <module>
    from sglang.srt.managers.router.manager import start_router_process
  File "/home/jessy/projects/sglang/python/sglang/srt/managers/router/manager.py", line 9, in <module>
    from sglang.srt.managers.router.model_rpc import ModelRpcClient
  File "/home/jessy/projects/sglang/py

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Llava-v1.6-34B template is not updated.

**Link**: https://github.com/sgl-project/sglang/issues/285
**State**: closed
**Created**: 2024-03-12T02:44:23+00:00
**Closed**: 2024-07-25T06:32:39+00:00
**Comments**: 4
**Labels**: inactive

### Description

Reference to https://github.com/haotian-liu/LLaVA/blob/7440ec9ee37b0374c6b5548818e89878e38f3353/llava/serve/gradio_web_server.py#L176, the chat template used by llava-v1.6-34b is 'chatml_direct' which is not implement in current SGLANG.
The template 'chatml' is implemented, but totally different from 'chatml_direct'.

The bug leads to the different outputs between the gradio demo and sgl.function with sgl runtime.

Besides, the template structure and notation are totally different. I am not sure that I can transfer the chat template from llava to ChatTemplate correctly.

---

## Issue #N/A: LlaVa Usage with server option ValueError: ... not in list

**Link**: https://github.com/sgl-project/sglang/issues/41
**State**: closed
**Created**: 2024-01-18T16:38:30+00:00
**Closed**: 2024-01-18T23:43:01+00:00
**Comments**: 4

### Description

Hello, I tried to utilize `sglang` backend with LlaVa model utilizing the command that's provided in the README

I created a new environment and installed `sglang` with `pip install "sglang[all]"`.

```bash
python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.5-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --port 30000
```

```python
import requests
import json
# includes an image of a cat
path = "images/cat_2.jpeg"
text = "what is this?"
# checked https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/io_struct.py#L9 for input
data = {"text": text,
        "image_data": path}

headers = {'Content-Type': 'application/json'}

response = requests.post('http://localhost:30000/generate', json=data, headers=headers)

print(response.json())
```

it gets stuck in runtime(jupyter, does not raise the error) but in server I see the error

`ValueError: 32000 is not in list`

May I ask if there is something I'm doing wrong?


---

## Issue #N/A: Qwen 2 7B not working

**Link**: https://github.com/sgl-project/sglang/issues/522
**State**: closed
**Created**: 2024-06-10T11:39:41+00:00
**Closed**: 2024-06-10T18:29:21+00:00
**Comments**: 3

### Description

This appears on a fresh installation of sglang. Currently using docker container with the following packages

sglang==0.1.17
triton==2.3.0
transformers==4.41.2
torch==2.3.0
vllm==0.4.3
vllm-flash-attn==2.5.8.post2

nvcc --version

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Wed_Nov_22_10:17:15_PST_2023
Cuda compilation tools, release 12.3, V12.3.107
Build cuda_12.3.r12.3/compiler.33567101_0
```

```
CUDA_VISIBLE_DEVICES=2 python -m sglang.launch_server --model-path Qwen/Qwen2-7B-Instruct-GPTQ-Int8 --port 30000
/usr/local/lib/python3.10/dist-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
[gpu_id=

[... truncated for brevity ...]

---

## Issue #N/A: Async support

**Link**: https://github.com/sgl-project/sglang/issues/29
**State**: closed
**Created**: 2024-01-17T23:44:02+00:00
**Closed**: 2024-01-21T23:17:31+00:00
**Comments**: 0
**Labels**: good first issue

### Description

No description provided.

---

## Issue #N/A: /generate stuck and no response when serving the Mixtral AWQ

**Link**: https://github.com/sgl-project/sglang/issues/139
**State**: closed
**Created**: 2024-02-04T07:28:25+00:00
**Closed**: 2024-07-25T06:32:05+00:00
**Comments**: 2
**Labels**: inactive

### Description

Hi, I have been trying to launch Mixtral AWQ with 2 A10 GPUs. Here is my command:

```bash
python -m sglang.launch_server --model-path TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ --tp 2
```

The output appears to be correct with the following standard output:

```bash
Server started on [0.0.0.0]:10006
Server started on [0.0.0.0]:10007
Accepted ('127.0.0.1', 36290) with file descriptor 5
Accepted ('127.0.0.1', 37546) with file descriptor 5
Welcome ('127.0.0.1', 37546)
Welcome ('127.0.0.1', 36290)
Rank 0: Load weight begins.
Quant_config: AWQConfig(weight_bits=4, group_size=128, zero_point=True)
Rank 1: Load weight begins.
Quant_config: AWQConfig(weight_bits=4, group_size=128, zero_point=True)
INFO 02-04 07:17:50 weight_utils.py:164] Using model weights format ['*.safetensors']
INFO 02-04 07:17:50 weight_utils.py:164] Using model weights format ['*.safetensors']
Rank 0: Load weight ends.
Rank 1: Load weight ends.
Rank 1: Max_total_num_token=115621, max_prefill_num_toke

[... truncated for brevity ...]

---

## Issue #N/A: LLaVA-v1.6 RuntimeError in llava image encoding

**Link**: https://github.com/sgl-project/sglang/issues/409
**State**: closed
**Created**: 2024-05-04T16:02:03+00:00
**Closed**: 2024-07-25T06:33:29+00:00
**Comments**: 1
**Labels**: inactive

### Description

There still seems to be a bug in the newer LLaVA-v1.6 version where, for some images, the model only generates one or two tokens. The problem seems to be related to some kind of attributes of the images themselves, as changing the textual input has no influence. Furthermore, all 3 v1.6 models (7b, 13b, and 34b) have problems with the same images. Moreover, the 1.5 version works perfectly fine with the same inputs. This bug appears for around 5% of my images.
I'm on the sglang 0.1.14 and vllm 0.3.3. The issue seems to be related to #273, however i do not use regex for generation. The server casts the following runtime error when llava is not able to process the image:
`
RuntimeError in llava image encoding: The expanded size of the tensor (0) must match the existing size (2438) at non-singleton dimension 0.  Target sizes: [0, 4096].  Tensor sizes: [2438, 4096]
torch.Size([10194, 4096])
0 -1`

---

## Issue #N/A: Any plan  to support cascading  feature of flashinfer?

**Link**: https://github.com/sgl-project/sglang/issues/495
**State**: closed
**Created**: 2024-06-03T03:21:25+00:00
**Closed**: 2024-08-08T01:03:58+00:00
**Comments**: 5
**Labels**: inactive

### Description

No description provided.

---

## Issue #N/A: `model_override_args` with server

**Link**: https://github.com/sgl-project/sglang/issues/591
**State**: closed
**Created**: 2024-07-05T09:57:03+00:00
**Closed**: 2024-09-08T01:12:57+00:00
**Comments**: 2
**Labels**: good first issue, inactive

### Description

When using a server, one currently cannot use the `model_overide_args` which could be very useful, e.g. for rope scaling. 

This is currently the `sglang.launch_server.py`:

```py
import argparse

from sglang.srt.server import launch_server
from sglang.srt.server_args import ServerArgs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)

    launch_server(server_args, None)
```

The `model_overide_args` would be the third argument to `launch_server` defaulting to `None`. Adding a small cli parser that allows arbitrary model args would be great, e.g.

```bash
python -m sglang.launch_server --model_overide_args.rope_scaling.factor 2 --model_overide_args.rope_scaling.type linear
```

---

## Issue #N/A: llava-v1.6-vicuna-7b NoneType Object Error when handle_generate_request, maybe misspelled

**Link**: https://github.com/sgl-project/sglang/issues/131
**State**: closed
**Created**: 2024-02-02T17:27:23+00:00
**Closed**: 2024-02-02T19:57:05+00:00
**Comments**: 3

### Description

sglang version: `sglang==0.1.10`
torch version: `2.1.0+cu118`

When running the server via: 

`python3 -m sglang.launch_server --model-path ./llava-v1.6-vicuna-7b --tokenizer-path SurfaceData/llava-v1.6-vicuna-7b-processor --chat-template vicuna_v1.1 --port 30000`

Running the following test script :

```
@sgl.function
def pipeline(s, prompt, max_tokens):
    for p in prompt:
        if type(p) is str:
            s += p
        else:
            s += sgl.image(p) # p would be PIL.Image
    s += sgl.gen("response", max_tokens=max_tokens)

backend = RuntimeEndpoint(sgl_endpoint)
sgl.set_default_backend(backend)
pipeline.run(prompt, max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k, stream=True)
```

the request triggered `handle_generate_request` in `srt.managers.router.model_rpc` and assert
```
Exception in ModelRpcClient:
Traceback (most recent call last):
  File "/root/miniconda3/envs/sgl/lib/python3.10/site-packages/sglang/srt/managers/rou

[... truncated for brevity ...]

---

## Issue #N/A: Lora test

**Link**: https://github.com/sgl-project/sglang/issues/66
**State**: closed
**Created**: 2024-01-21T08:50:22+00:00
**Closed**: 2024-01-21T10:43:42+00:00
**Comments**: 3

### Description

Hi,

I saw that there is a lora dev branch. is it possible to test this already? Or is it still WIP. Asking as ive been asking for s-lora in vllm for like months now and s-lora being on your roadmap is very exciting.

Thanks!

---

## Issue #N/A: Supports the InternVL multimodal large model

**Link**: https://github.com/sgl-project/sglang/issues/328
**State**: closed
**Created**: 2024-03-24T02:33:14+00:00
**Closed**: 2024-09-22T14:22:31+00:00
**Comments**: 5

### Description

Can it support the InternVL multimodal large model, which currently ranks first in the MMMU open source ranking.
[https://github.com/OpenGVLab/InternVL/](https://github.com/OpenGVLab/InternVL/)
![WX20240324-102942@2x](https://github.com/sgl-project/sglang/assets/4583537/2416f85d-5231-4d8c-9255-b598385e6eaa)
[MMMU](https://mmmu-benchmark.github.io)


---

## Issue #N/A: Crash in `tokenize_fast_forward`

**Link**: https://github.com/sgl-project/sglang/issues/115
**State**: closed
**Created**: 2024-01-29T22:59:05+00:00
**Closed**: 2024-02-08T03:51:14+00:00
**Comments**: 2
**Labels**: bug

### Description

I was trying constraint decoding with Qwen but got crash at this line: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/router/infer_batch.py#L60

This is because the output type of Qwen `tokenizer.convert_ids_to_tokens(...)` is not `str` but `bytes`, so the following modification works:

```
if self.tokenizer.convert_ids_to_tokens(self.output_ids[0]).decode().startswith("▁"):
```

However, this is not a general solution, as not every tokenizer uses `▁` to represent the space. For example, the Falcon tokenizer uses `Ġ` to represent the space. A more general solution should be re-decoding so far output tokens with `tokenizer.decode(...)`. An alternative way is leveraging `tokenizer.decode(..., clean_up_tokenization_spaces=False)`. While this solution works with most tokenizers such as Falcon and Qwen, it doesn't work for Llama2:

```
llama2 = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
falcon = AutoTokenizer.from_pretrained("tiiua

[... truncated for brevity ...]

---

## Issue #N/A: How to use inside notebook?

**Link**: https://github.com/sgl-project/sglang/issues/38
**State**: closed
**Created**: 2024-01-18T14:11:32+00:00
**Closed**: 2024-01-19T18:38:29+00:00
**Comments**: 3

### Description

Im trying to use this on databricks inside the notebook that's running on top of a 8xA10 single node cluster, I'm initialising like:

```
from sglang import function, system, user, assistant, gen, set_default_backend, Runtime
runtime = Runtime("/local_disk0/mistralai/Mixtral-8x7B-Instruct-v0.1")
set_default_backend(runtime)
```

However I get this issue
```
router init state: Traceback (most recent call last):
  File "/local_disk0/.ephemeral_nfs/envs/pythonEnv-51dd0ee1-a396-4939-81a6-75e3afe59af5/lib/python3.10/site-packages/sglang/srt/managers/router/manager.py", line 68, in start_router_process
    model_client = ModelRpcClient(server_args, port_args)
  File "/local_disk0/.ephemeral_nfs/envs/pythonEnv-51dd0ee1-a396-4939-81a6-75e3afe59af5/lib/python3.10/site-packages/sglang/srt/managers/router/model_rpc.py", line 448, in __init__
    self.model_server.exposed_init_model(0, server_args, port_args)
  File "/local_disk0/.ephemeral_nfs/envs/pythonEnv-51dd0ee1-a396-4939-81a6

[... truncated for brevity ...]

---

## Issue #N/A: VLLM version

**Link**: https://github.com/sgl-project/sglang/issues/373
**State**: closed
**Created**: 2024-04-19T18:23:37+00:00
**Closed**: 2024-07-18T16:27:27+00:00
**Comments**: 4
**Labels**: high priority

### Description

`python -m sglang.launch_server --model-path Mistral-7B-Instruct-v0.2/` fails with 

```
router init state: Traceback (most recent call last):
  File ".venv/lib/python3.9/site-packages/sglang/srt/managers/router/manager.py", line 68, in start_router_process
    model_client = ModelRpcClient(server_args, port_args)
  File ".venv/lib/python3.9/site-packages/sglang/srt/managers/router/model_rpc.py", line 619, in __init__
    self.model_server.exposed_init_model(0, server_args, port_args)
  File ".venv/lib/python3.9/site-packages/sglang/srt/managers/router/model_rpc.py", line 70, in exposed_init_model
    self.model_runner = ModelRunner(
  File ".venv/lib/python3.9/site-packages/sglang/srt/managers/router/model_runner.py", line 287, in __init__
    self.load_model()
  File ".venv/lib/python3.9/site-packages/sglang/srt/managers/router/model_runner.py", line 296, in load_model
    model_class = get_model_cls_by_arch_name(architectures)
  File ".venv/lib/python3.9/site-packages/

[... truncated for brevity ...]

---

## Issue #N/A: run python3 test_httpserver_llava.py get ValueError: 64002 is not in list

**Link**: https://github.com/sgl-project/sglang/issues/413
**State**: closed
**Created**: 2024-05-08T11:35:48+00:00
**Closed**: 2024-07-30T01:03:13+00:00
**Comments**: 3
**Labels**: inactive

### Description

run python3 test_httpserver_llava.py
offset = input_ids.index(self.config.image_token_index)
ValueError: 64002 is not in list

def test_streaming(args):
    url = f"{args.host}:{args.port}"
    response = requests.post(
        url + "/generate",
        json={
            'text' : 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions. USER: <im_start><image><im_end> description the video indetail \n Assistant:', 
            # "text": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: Describe this picture <|im_start|> <|im_end|>\n ASSISTANT:",
            "image_data": "examples/image1.webp",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 128,
            },
            "stream": True,
        },
   

[... truncated for brevity ...]

---

## Issue #N/A: enable an installation without CUDA_HOME?

**Link**: https://github.com/sgl-project/sglang/issues/13
**State**: closed
**Created**: 2024-01-16T14:57:28+00:00
**Closed**: 2024-01-17T00:15:30+00:00
**Comments**: 3

### Description

Easier to install for users who just want to call LLM APIs.

---

## Issue #N/A: Outlines integration

**Link**: https://github.com/sgl-project/sglang/issues/43
**State**: closed
**Created**: 2024-01-18T17:40:02+00:00
**Closed**: 2024-02-09T06:07:59+00:00
**Comments**: 9

### Description

This package looks awesome!  I was wondering why you decided to copy Outlines' code instead of importing the FSMs directly from outlines? There are several improvements on the performance of guided generation in the pipeline and you will be missing out on those. By importing you get better as we get better :)

---

## Issue #N/A: Don't get API response when sending images

**Link**: https://github.com/sgl-project/sglang/issues/357
**State**: closed
**Created**: 2024-04-09T13:38:39+00:00
**Closed**: 2024-07-25T06:33:21+00:00
**Comments**: 2
**Labels**: inactive

### Description

I loaded Llava v1.6 34B on my server 
```
export DISABLE_NEST_ASYNCIO=True
model=liuhaotian/llava-v1.6-34b 
tokenizer=liuhaotian/llava-v1.6-34b-tokenizer 

CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server --model-path $model --tokenizer-path $tokenizer --port 30813 --tp 2
```
It works when I work with just with text, but when I send images I just don't get a response, this is what the server log shows:
```
$ ./start_sglang_server.sh 
/home/tom/.local/lib/python3.10/site-packages/transformers/models/llava/configuration_llava.py:104: FutureWarning: The `vocab_size` argument is deprecated and will be removed in v4.42, since it can be inferred from the `text_config`. Passing this argument has no effect
  warnings.warn(
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
server started on [0.0.0.0]:10007
server started on [0.0.0.0]:10008
Special tokens have been added in the vocabulary, make sure the a

[... truncated for brevity ...]

---

## Issue #N/A: Dependency conflict with LLaVA

**Link**: https://github.com/sgl-project/sglang/issues/464
**State**: closed
**Created**: 2024-05-23T07:43:08+00:00
**Closed**: 2024-07-26T01:02:23+00:00
**Comments**: 1
**Labels**: inactive

### Description

# Issue
Cannot run a finetuned LLaVA model with sglang==0.1.16, running `CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path llava-lora-34b-faceshape-ft/ --tokenizer-path liuhaotian/llava-v1.6-34b-tokenizer --port 30000 --tp 4`
throws: 
FileNotFoundError: [Errno 2] No such file or directory: '/home/iverkh/.triton/cache/a95dd9872513f57ade076cce4b51d3f0/_fwd_kernel_stage2.json.tmp.pid_33358_98246'

# Reproduction
After cloning LLaVA from https://github.com/haotian-liu/LLaVA, run `pip install -e . ` as instructed in the README
Run pip install sglang[all]
Run `CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server --model-path <fine tuned llava> --tokenizer-path liuhaotian/llava-v1.6-34b-tokenizer --port 30000 --tp 4`

# Possible cause
sglang has a dependency of vllm>0.4.2 which requires torch==2.3.0. LLaVA's official repository has a dependency of torch==2.1.2,
see here: https://github.com/haotian-liu/LLaVA/blob/main/pyproject.toml

After downgrad

[... truncated for brevity ...]

---

## Issue #N/A: ValueError: Can't patch loop of type <class 'uvloop.Loop'>

**Link**: https://github.com/sgl-project/sglang/issues/192
**State**: closed
**Created**: 2024-02-15T17:43:43+00:00
**Closed**: 2024-03-28T00:28:49+00:00
**Comments**: 2

### Description

Starting the server and doing inference used to work, but letting sglang select from choices caused it to hang. I updated from 0.1.11 to 0.1.12 and now the server doesn't start anymore:

```
model=liuhaotian/llava-v1.6-vicuna-7b
tokenizer=llava-hf/llava-1.5-7b-hf
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server --model-path $model --tokenizer-path $tokenizer --chat-template vicuna_v1.1 --port 30813 --tp 2
```


```
 % ./start_sglang_server.sh                                                                                                                                                                                                                                               [0/1436]
2024-02-15 17:38:56.694239: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
2024-02-15 17:38:56.727839: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions

[... truncated for brevity ...]

---

