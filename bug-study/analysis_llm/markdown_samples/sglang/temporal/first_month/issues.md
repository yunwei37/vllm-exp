# first_month - issues

**Total Issues**: 85
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 85

### Label Distribution

- inactive: 26 issues
- good first issue: 6 issues
- bug: 6 issues
- enhancement: 3 issues
- help wanted: 1 issues
- high priority: 1 issues
- collaboration: 1 issues

---

## Issue #N/A: Supporting api vendors of Mixtral

**Link**: https://github.com/sgl-project/sglang/issues/183
**State**: closed
**Created**: 2024-02-11T21:05:55+00:00
**Closed**: 2024-02-12T09:07:15+00:00
**Comments**: 1

### Description

Could this work with an OpenAI compatable vendor API like together or firework? I would like to use mixtral but would rather not host it. I tried but the openai implementation is hardcoded for openai model. 

---

## Issue #N/A: Port fused MoE Kernels

**Link**: https://github.com/sgl-project/sglang/issues/179
**State**: closed
**Created**: 2024-02-11T16:29:58+00:00
**Closed**: 2024-07-28T03:14:29+00:00
**Comments**: 1

### Description

There is a recent PR (https://github.com/vllm-project/vllm/pull/2542) in vLLM that introduced some fused kernels to accelerate mixtral MoE models.

We can bring it to our [code](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/mixtral.py) as well. 

---

## Issue #N/A: [model request] Camelidae/Sparsestral

**Link**: https://github.com/sgl-project/sglang/issues/176
**State**: closed
**Created**: 2024-02-09T22:49:30+00:00
**Closed**: 2024-07-25T06:32:07+00:00
**Comments**: 1
**Labels**: inactive

### Description

These are parameter efficient MoE models that claim to have performance better than Mixtral.

Camildae
https://github.com/wuhy68/Parameter-Efficient-MoE

Sparsestral
https://huggingface.co/serpdotai/sparsetral-16x7B-v2

Sparsestral has a vllm implementation in this form
https://github.com/serp-ai/vllm

---

## Issue #N/A: Incorrect token usage with jump forward

**Link**: https://github.com/sgl-project/sglang/issues/173
**State**: closed
**Created**: 2024-02-09T18:12:43+00:00
**Closed**: 2024-02-10T04:06:16+00:00
**Comments**: 2

### Description

Since jump forward breaks a decoding process to multiple ones, the number of prompt_tokens and completion_tokens are incorrect. Here is an example:

Request:

```python
regex = (r"""\{\n"""
    + r"""  "name": "[\w]{1,8}",\n"""
    + r"""  "description": "[\w\d\s]{1,64}"\n"""
    + r"""\}"""
)

response = requests.post(
    url + "/generate",
    json={
        "text": "Here is the info of France's capital: ",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 128,
            "regex": regex
        },
        "stream": True,
    },
    stream=True,
)
```

Streaming response by chunk:

```
Chunk (prompt 10, decode 1): {

Chunk (prompt 15, decode 1): {
  "name": "Paris

Chunk (prompt 22, decode 1): {
  "name": "Paris",
  "description": "Capital

Chunk (prompt 22, decode 2): {
  "name": "Paris",
  "description": "Capital city

Chunk (prompt 22, decode 10): {
  "name": "Paris",
  "description": "Capital city

[... truncated for brevity ...]

---

## Issue #N/A: OpenAI compatible API and JSON schema enforcing 

**Link**: https://github.com/sgl-project/sglang/issues/171
**State**: closed
**Created**: 2024-02-09T04:08:20+00:00
**Closed**: 2024-02-11T01:21:34+00:00
**Comments**: 2

### Description

Hi! I just wanted to ask if it is possible to use OpenAI compatible API in sglang with local models and to force a certain json schema. IIRC original openai api only supports `{ "type": "json_object" }` which allows generating any json objects. If currently it is not possible, introducing something like `_schema` as optional property may be a solution until a similar functionality appears in the openai specification. 

---

## Issue #N/A: Performance issue comparing sglang to vllm. 

**Link**: https://github.com/sgl-project/sglang/issues/169
**State**: closed
**Created**: 2024-02-08T23:26:03+00:00
**Closed**: 2024-02-11T14:00:42+00:00
**Comments**: 5

### Description

Hi there, Amazing work on the RadixAttention and json contained decoding. I am running into some unexcited performance issue comparing sglang and vllm. I use latest pip of vllm, and use git-clone-ed sglang as of today. 


here is my code to launch sglang 
`python -m sglang.launch_server --model-path NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO --port 30000 --tp 8`

Here is my code to launch v-llm 

python -m vllm.entrypoints.openai.api_server     --model NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO --tensor-parallel-size 8

Both running with the same Conda with  CUDA 12.1 environment, 8x a10g on aws. 
Here is the openai-compatible curl request

'curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant"},
            {"role": "user", "content": "You are a helpful AI ass

[... truncated for brevity ...]

---

## Issue #N/A: ModuleNotFoundError: No module named 'zmq'

**Link**: https://github.com/sgl-project/sglang/issues/167
**State**: closed
**Created**: 2024-02-08T09:58:25+00:00
**Closed**: 2024-02-11T13:51:39+00:00
**Comments**: 2

### Description

Hi!

Should zmq be on your dependencies list?

```
2024-02-08 09:56:23 | ERROR | stderr | Traceback (most recent call last):
2024-02-08 09:56:23 | ERROR | stderr |   File "/p/haicluster/llama/FastChat/fastchat/serve/sglang_worker.py", line 269, in <module>
2024-02-08 09:56:23 | ERROR | stderr |     runtime = sgl.Runtime(
2024-02-08 09:56:23 | ERROR | stderr |               ^^^^^^^^^^^^
2024-02-08 09:56:23 | ERROR | stderr |   File "/p/haicluster/llama/FastChat/sc_venv_2024/venv/lib/python3.11/site-packages/sglang/api.py", line 37, in Runtime
2024-02-08 09:56:23 | ERROR | stderr |     from sglang.srt.server import Runtime
2024-02-08 09:56:23 | ERROR | stderr |   File "/p/haicluster/llama/FastChat/sc_venv_2024/venv/lib/python3.11/site-packages/sglang/srt/server.py", line 30, in <module>
2024-02-08 09:56:23 | ERROR | stderr |     from sglang.srt.managers.detokenizer_manager import start_detokenizer_process
2024-02-08 09:56:23 | ERROR | stderr |   File "/p/haicluster/llama/Fas

[... truncated for brevity ...]

---

## Issue #N/A: Add SGLang usage examples

**Link**: https://github.com/sgl-project/sglang/issues/166
**State**: closed
**Created**: 2024-02-08T08:51:53+00:00
**Closed**: 2024-09-08T01:13:00+00:00
**Comments**: 8
**Labels**: inactive

### Description

List some good use cases of SGLang here:
- [SELF-DISCOVER: Large Language Models Self-Compose Reasoning Structures](https://arxiv.org/pdf/2402.03620.pdf)
- [Tractable Control for Autoregressive Language Generation](https://starai.cs.ucla.edu/papers/ZhangICML23.pdf)

---

## Issue #N/A: setting mem-fraction-static to a lower value causes error

**Link**: https://github.com/sgl-project/sglang/issues/165
**State**: closed
**Created**: 2024-02-07T22:25:25+00:00
**Closed**: 2024-07-25T06:33:36+00:00
**Comments**: 4
**Labels**: inactive

### Description

With no change, I run out of memory (A100 w/ 24GB). Setting it to anything other than the default causes the following error:

```
Exception in ModelRpcClient:
Traceback (most recent call last):
  File "/workspace/sglang/python/sglang/srt/managers/router/model_rpc.py", line 170, in exposed_step
    self.forward_step()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/workspace/sglang/python/sglang/srt/managers/router/model_rpc.py", line 185, in forward_step
    self.forward_fill_batch(new_batch)
  File "/workspace/sglang/python/sglang/srt/managers/router/model_rpc.py", line 387, in forward_fill_batch
    batch.prepare_for_extend(
  File "/workspace/sglang/python/sglang/srt/managers/router/infer_batch.py", line 203, in prepare_for_extend
    req_pool_indices_cpu = req_pool_indices.cpu().numpy()
AttributeError: 'NoneType' object has no attribute 'cpu'
```

For reference I 

[... truncated for brevity ...]

---

## Issue #N/A: RuntimeEndpoint doesn't work if endpoint requires Auth

**Link**: https://github.com/sgl-project/sglang/issues/163
**State**: closed
**Created**: 2024-02-07T19:42:23+00:00
**Closed**: 2024-02-08T07:14:12+00:00
**Comments**: 2

### Description

Hi, 

Came across an interesting use case today. I was trying to host the model on databricks. But call it locally. But databricks requires a personal access token to be passed in via the headers.

---

## Issue #N/A: offset = input_ids.index(self.config.image_token_index)

**Link**: https://github.com/sgl-project/sglang/issues/161
**State**: closed
**Created**: 2024-02-07T16:56:33+00:00
**Closed**: 2024-07-25T06:32:55+00:00
**Comments**: 3
**Labels**: inactive

### Description

anyone faced this problem before ? :

Exception in ModelRpcClient:
Traceback (most recent call last):
  File "/home/fjaadari/anaconda3/envs/llava/lib/python3.10/site-packages/sglang/srt/managers/router/model_rpc.py", line 161, in exposed_step
    self.handle_generate_request(recv_req)
  File "/home/fjaadari/anaconda3/envs/llava/lib/python3.10/site-packages/sglang/srt/managers/router/model_rpc.py", line 243, in handle_generate_request
    req.input_ids, req.image_offset = self.model_runner.model.pad_input_ids(
  File "/home/fjaadari/anaconda3/envs/llava/lib/python3.10/site-packages/sglang/srt/models/llava.py", line 63, in pad_input_ids
    offset = input_ids.index(self.config.image_token_index)
ValueError: 32000 is not in list



---

## Issue #N/A: initialise model with max_model_len

**Link**: https://github.com/sgl-project/sglang/issues/159
**State**: closed
**Created**: 2024-02-07T11:57:06+00:00
**Closed**: 2024-02-21T00:22:57+00:00
**Comments**: 2
**Labels**: good first issue

### Description

Similar to how vllm has the 'max_model_length' when starting the server. Can we have this here too?

This would help when trying to host models on smaller gpus. For example with vllm Mistral 7b with 32k context doesn't fit on a single 24GB GPU. Whereas with 8k context it does. 

---

## Issue #N/A: About quantization model inference

**Link**: https://github.com/sgl-project/sglang/issues/158
**State**: closed
**Created**: 2024-02-07T08:05:36+00:00
**Closed**: 2024-02-11T14:28:51+00:00
**Comments**: 1

### Description

Actually，I don't know how use quantize model inference in this project and vllm. Due my limited knowledge, they load quantize model and dequantize it for ops in vllm?

---

## Issue #N/A: Development Roadmap  (Deprecated)

**Link**: https://github.com/sgl-project/sglang/issues/157
**State**: closed
**Created**: 2024-02-07T07:13:40+00:00
**Closed**: 2024-07-17T02:23:05+00:00
**Comments**: 17

### Description

## Function Calling
- Frontend
    - Add `tools` argument in `sgl.gen`. See also guidance [tools](https://github.com/guidance-ai/guidance/blob/d1bbe1c698cbb201f89556d71193993e78c0686b/README.md?plain=1#L102)
- Backend
    - OpenAI: Translate to their function calling API (https://platform.openai.com/docs/guides/function-calling).
      - #573 
    - Local Models (SGLang)
        1. Use SGLang primitives (regex, select) and constrained decoding to implement a workflow 
        2. Directly use models that support function calling (e.g., Gorilla OpenFunctions, https://huggingface.co/jondurbin/bagel-dpo-7b-v0.4#prompting-strategies)
     - Local Models (OpenAI-compatible API)

## High-level Pythonic Interface
 - #39
 
## Inference Optimizations
- Speculative decoding for local models
- Speculative execution for OpenAI Chat API
  - #48 

## Structured Decoding
- Support parallel JSON decoding https://github.com/varunshenoy/super-json-mode/issues/8
- Support auto paralle

[... truncated for brevity ...]

---

## Issue #N/A: `RecursionError: maximum recursion depth exceeded while calling a Python object` when inferencing with long input

**Link**: https://github.com/sgl-project/sglang/issues/154
**State**: closed
**Created**: 2024-02-06T18:44:35+00:00
**Closed**: 2024-07-25T06:33:07+00:00
**Comments**: 9
**Labels**: inactive

### Description

Hi, I ran across this issue during inference
```bash
Exception in ModelRpcClient:
Traceback (most recent call last):
  File "/User/jay/miniconda3/envs/sglang/lib/python3.11/site-packages/sglang/srt/managers/router/model_rpc.py", line 168, in exposed_step
    self.forward_step()
  File "/User/jay/miniconda3/envs/sglang/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/User/jay/miniconda3/envs/sglang/lib/python3.11/site-packages/sglang/srt/managers/router/model_rpc.py", line 195, in forward_step
    self.forward_decode_batch(self.running_batch)
  File "/User/jay/miniconda3/envs/sglang/lib/python3.11/site-packages/sglang/srt/managers/router/model_rpc.py", line 460, in forward_decode_batch
    self.handle_finished_requests(batch)
  File "/User/jay/miniconda3/envs/sglang/lib/python3.11/site-packages/sglang/srt/managers/router/model_rpc.py", line 528, in handle_finish

[... truncated for brevity ...]

---

## Issue #N/A: unable to install uvloop in windows (dependency)

**Link**: https://github.com/sgl-project/sglang/issues/152
**State**: closed
**Created**: 2024-02-06T18:02:46+00:00
**Closed**: 2024-07-25T06:32:08+00:00
**Comments**: 3
**Labels**: inactive

### Description

Getting this error

pip install sglang[all]
Requirement already satisfied: sglang[all] in c:\users\keshav s\anaconda3\envs\llm-workspace\lib\site-packages (0.1.11)
Requirement already satisfied: requests in c:\users\keshav s\anaconda3\envs\llm-workspace\lib\site-packages (from sglang[all]) (2.31.0)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\keshav s\anaconda3\envs\llm-workspace\lib\site-packages (from requests->sglang[all]) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in c:\users\keshav s\anaconda3\envs\llm-workspace\lib\site-packages (from requests->sglang[all]) (3.6)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\keshav s\anaconda3\envs\llm-workspace\lib\site-packages (from requests->sglang[all]) (2.1.0)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\keshav s\anaconda3\envs\llm-workspace\lib\site-packages (from requests->sglang[all]) (2023.11.17)
Collecting anthropic (from sglang[all])
  Using cached anthropic-0.

[... truncated for brevity ...]

---

## Issue #N/A: regex support for openai models?

**Link**: https://github.com/sgl-project/sglang/issues/151
**State**: closed
**Created**: 2024-02-06T17:38:31+00:00
**Closed**: 2024-02-06T19:53:41+00:00
**Comments**: 2

### Description

Hi I was trying to use the regex constrained output with an OpenAI backend but it threw a warning saying regex is not supported for openai backend.

---

## Issue #N/A: After enabling tensor parallelism (tp-size=2), there is no response

**Link**: https://github.com/sgl-project/sglang/issues/150
**State**: closed
**Created**: 2024-02-06T12:51:16+00:00
**Closed**: 2024-07-25T06:32:47+00:00
**Comments**: 6
**Labels**: inactive

### Description

my command is:
```shell
CUDA_VISIBLE_DEVICES="2,4" python -m sglang.launch_server --model-path  ./Yi-34B-Chat --trust-remote-code --port 30000 --tp-size 2 
``` 
when I run the demo code, **there is nothing returned.** 

```python
from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint

@function
def multi_turn_question(s, question_1, question_2):
    s += system("You are a helpful assistant.")
    s += user(question_1)
    s += assistant(gen("answer_1", max_tokens=256))
    s += user(question_2)
    s += assistant(gen("answer_2", max_tokens=256))

set_default_backend(RuntimeEndpoint("http://localhost:30000"))

state = multi_turn_question.run(
    question_1="What is the capital of the United States?",
    question_2="List two local attractions.",
)

for m in state.messages():
    print(m["role"], ":", m["content"])
```

**But when I  remove  "--tp-size 2 " in the command ,which means the model is only in 1 GPU , it wor

[... truncated for brevity ...]

---

## Issue #N/A: 分类准确率下降，性能提升一倍，咋解决呢？

**Link**: https://github.com/sgl-project/sglang/issues/149
**State**: closed
**Created**: 2024-02-06T07:44:37+00:00
**Closed**: 2024-07-25T06:32:09+00:00
**Comments**: 3
**Labels**: inactive

### Description

qwen自带推理：
第一身份准确率：0.968421052631579
第二身份准确率：0.9368421052631579
一条数据平均耗时：0.9585727064233077

slang推理：
第一身份准确率：0.6631578947368421
第二身份准确率：0.7894736842105263
一条数据平均耗时：0.5773725032806396

---

## Issue #N/A: Add support for scalar values

**Link**: https://github.com/sgl-project/sglang/issues/145
**State**: closed
**Created**: 2024-02-05T18:12:49+00:00
**Closed**: 2024-02-05T23:47:19+00:00
**Comments**: 2

### Description

I often need an LLM to generate scalar values.
I look at the logprobs of True and False and do True - False.
Works very well for me:
```
f'{output_text}\n\nRate whether the text is well written.', '{"is_well_written_bool": '
```

Also the values are nicely distributed:
```
is_great_comment_bool: 0.72  
is_funny_bool_opt: -0.79  
neg_sounds_like_chatgpt_bool: -0.49  
neg_contains_placeholders_bool: -0.86  
neg_sounds_awkward_bool: -0.97 
 is_written_by_human: 0.94  
neg_sounds_robotic: -0.89
```

Could you add something like this into sglang?

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

## Issue #N/A: KV cache pool leak detected!

**Link**: https://github.com/sgl-project/sglang/issues/140
**State**: closed
**Created**: 2024-02-04T08:37:50+00:00
**Closed**: 2024-08-05T01:05:13+00:00
**Comments**: 9
**Labels**: inactive

### Description


python -m sglang.launch_server --model-path /root/autodl-tmp/Yi-6B-Chat --port 8000 --mem-fraction-static 0.9 --tokenizer-mode auto --tokenizer-path /root/autodl-tmp/Yi-6B-Chat --trust-remote-code

<img width="1439" alt="image" src="https://github.com/sgl-project/sglang/assets/4583537/dd6decd6-99a1-420e-8be2-b9a130d972fb">

python3 bench_throughput.py  --tokenizer /root/autodl-tmp/Yi-6B-Chat  --dataset /root/autodl-tmp/ShareGPT_V3_unfiltered_cleaned_split.json   --port 8000 --backend srt --trust-remote-code --num-prompts 2048

<img width="1512" alt="image" src="https://github.com/sgl-project/sglang/assets/4583537/5a47f9e6-e356-4269-859a-0e0e395e6d89">



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

## Issue #N/A: Error on json decoding with llava

**Link**: https://github.com/sgl-project/sglang/issues/138
**State**: closed
**Created**: 2024-02-03T23:12:32+00:00
**Closed**: 2024-02-04T02:46:17+00:00
**Comments**: 2

### Description

Encountered the following error when using the `regex` in call to `gen` method. Server doesn't work afterwards:

```console
~$ python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.5-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --port 30000
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Rank 0: load weight begin.
/opt/conda/envs/llava_test/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
INFO 02-03 23:02:55 weight_utils.py:164] Using model weights format 

[... truncated for brevity ...]

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

## Issue #N/A: [Bug] liuhaotian/llava-v1.6-mistral-7b doesn't load

**Link**: https://github.com/sgl-project/sglang/issues/128
**State**: closed
**Created**: 2024-01-31T23:55:51+00:00
**Closed**: 2024-07-25T06:32:21+00:00
**Comments**: 16
**Labels**: inactive

### Description

When trying to load the Mistral variant of LLaVa 1.6, I get an expected error:

```sh
python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.6-mistral-7b --chat-template vicuna_v1.1 --port 30000
```

```
ValueError: The checkpoint you are trying to load has model type `llava_mistral` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date
```

Transformers doesn't treat the LLaVa variants any differently, they all use the same config.  I think this *could* be easily fixed by adding a mapping from `llava_mistral` to the `LlavaConfig` in the config mapping.  

---

## Issue #N/A: [Bug] liuhaotian/llava-v1.6-vicuna-7b doesn't load

**Link**: https://github.com/sgl-project/sglang/issues/127
**State**: closed
**Created**: 2024-01-31T23:53:39+00:00
**Closed**: 2024-02-01T01:38:24+00:00
**Comments**: 2

### Description

With LLaVA 1.6 out I wanted to see how they run.  I think there's one more pre-deploy step needed for the vicuna base model variant.

When running the server via:

```sh
python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.6-vicuna-7b  --chat-template vicuna_v1.1 --port 30000
```

This fails because that variant isn't deployed with a huggingface pre-processor.  Someone will need to do the same HuggingFace conversion used to create [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf).  Maybe liuhaotian can do this so it's not on some other disjoint hf repo?

---

## Issue #N/A: Pydantic>2 causes issues loading models

**Link**: https://github.com/sgl-project/sglang/issues/126
**State**: closed
**Created**: 2024-01-31T17:31:27+00:00
**Closed**: 2024-07-25T06:32:13+00:00
**Comments**: 15
**Labels**: inactive

### Description

Hi,

Related to this issue,

By default, since the pydantic version is not pinned, greater than version 2 is used which is causing issues with loading.

However `pip install pydantic==1.10.14` resolves this issue and I can load models normally now.

---

## Issue #N/A: Support gptq  quantization

**Link**: https://github.com/sgl-project/sglang/issues/124
**State**: closed
**Created**: 2024-01-31T03:33:47+00:00
**Closed**: 2024-02-06T19:35:57+00:00
**Comments**: 2
**Labels**: good first issue

### Description

No description provided.

---

## Issue #N/A: How to use 4bit on LLava?

**Link**: https://github.com/sgl-project/sglang/issues/123
**State**: closed
**Created**: 2024-01-31T03:06:50+00:00
**Closed**: 2024-07-25T06:32:04+00:00
**Comments**: 1
**Labels**: inactive

### Description

I am using this code: "python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.5-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --chat-template vicuna_v1.1 --port 30000" , but I am not able to use any instruction for quantity 4-bit. 

can you tell me how to use 4-bit llava on sglang?

---

## Issue #N/A: Slow weight loading

**Link**: https://github.com/sgl-project/sglang/issues/122
**State**: closed
**Created**: 2024-01-30T14:46:10+00:00
**Closed**: 2024-07-25T06:33:34+00:00
**Comments**: 10
**Labels**: help wanted, inactive

### Description

Whenever I try to load the Mixtral models it takes very long and at the end instead of actually starting the server I get a similar error as the one here - https://github.com/sgl-project/sglang/issues/99
The same model save works in vLLM. 
My setup - 2xA100-80GBs

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

## Issue #N/A: Qwen-7B-Chat UnicodeDecodeError Stop working

**Link**: https://github.com/sgl-project/sglang/issues/111
**State**: closed
**Created**: 2024-01-27T04:42:07+00:00
**Closed**: 2024-01-30T15:43:52+00:00
**Comments**: 1
**Labels**: bug

### Description

I have found that when running the Qwen-7B-Chat model for inference, sometimes the following errors occur, which can cause the entire inference service to malfunction in the future. As shown in the following figure:

<img width="1298" alt="image" src="https://github.com/sgl-project/sglang/assets/4583537/53596022-c749-42d0-ad6b-919ad4bf3d5c">

<img width="1512" alt="image" src="https://github.com/sgl-project/sglang/assets/4583537/253015ab-3de3-40ca-a5b7-de9a980e4615">

Start script:
python -m sglang.launch_server --model-path /root/autodl-tmp/Qwen-7B-Chat --port 8000 --load-format safetensors --mem-fraction-static 0.9 --tokenizer-mode auto --trust-remote-code

Test script:
python3 bench_throughput.py  --tokenizer /root/autodl-tmp/Qwen-7B-Chat --dataset /root/autodl-tmp/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000  --port 8000 --backend srt --trust-remote-code


---

## Issue #N/A: Issue with running example with Mixtral AWQ

**Link**: https://github.com/sgl-project/sglang/issues/110
**State**: closed
**Created**: 2024-01-27T00:48:43+00:00
**Closed**: 2024-01-29T18:41:18+00:00
**Comments**: 3

### Description

Hey, 

Not sure if I'm doing something silly on my end, but I said I would submit an issue to get some help. Hope that's okay. 2x 3090 GPU. 

I am running with the following Dockerfile:

```
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

RUN apt update && apt dist-upgrade -y

RUN pip install --upgrade pip
RUN pip install "sglang[all]"

WORKDIR /

COPY . . 

EXPOSE 30000

CMD ["/bin/bash","-c","python -m sglang.launch_server --model-path /models/TheBloke_Nous-Hermes-2-Mixtral-8x7B-DPO-AWQ/ --tokenizer-path /models/TheBloke_Nous-Hermes-2-Mixtral-8x7B-DPO-AWQ/ --port 30000 --host 192.168.1.55 --tp-size 2"]
```

Here is the test example I have ran (also tried simple curl with same outcome):
```
from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint

@function
def multi_turn_question(s, question_1, question_2):
    s += system("You are a helpful assistant.")
    s += user(question_1)
    s += assistant(gen("answer_1"

[... truncated for brevity ...]

---

## Issue #N/A: OOM after flush_cache when flashinfer is enabled

**Link**: https://github.com/sgl-project/sglang/issues/109
**State**: closed
**Created**: 2024-01-27T00:35:40+00:00
**Closed**: 2024-02-16T20:20:00+00:00
**Comments**: 3

### Description

When benchmarking my serving workloads, I found the following pattern will constantly cause OOM error:

1. Launch a container with SRT and flashinfer enabled.
2. Benchmark with 800 requests.
3. Call `http://0.0.0.0:25000/flush_cache` and confirm the cache is flushed in the SRT log.
4. Benchmark with 800 requests again.

The above steps result in the following error in the middle of the second benchmarking:

```
CUDA Error: out of memory (2) /sglang/3rdparty/flashinfer/include/flashinfer/handler.cuh: line 100 at function cudaMallocAsync(&float_buffer_, sizeof(float) *
 tmp_size, stream_)
Exception in ModelRpcClient:
Traceback (most recent call last):
  File "/sglang/python/sglang/srt/managers/router/model_rpc.py", line 165, in exposed_step
    self.forward_step()
  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/sglang/python/sglang/srt/managers/router/model_rpc.py", line

[... truncated for brevity ...]

---

## Issue #N/A: Mistral model no longer loads following PR#101

**Link**: https://github.com/sgl-project/sglang/issues/107
**State**: closed
**Created**: 2024-01-26T15:06:54+00:00
**Closed**: 2024-01-26T17:38:45+00:00
**Comments**: 2

### Description

The `get_model_cls_by_arch_name` introduced in [Dynamic model class loading PR](https://github.com/sgl-project/sglang/pull/101) removes the hard-coded mapping between `MistralForCausalLM` and `LlamaForCausalLM` causing issues trying to local host [Mistral-7b](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) model as of sglang version 0.1.9. I have tested that adding the following simple `models/mistral.py` file allows hosting the mistral-7b model.

```python
from sglang.srt.models.llama2 import LlamaForCausalLM


class MistralForCausalLM(LlamaForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


EntryClass = MistralForCausalLM
```

---

## Issue #N/A: Implement prefix_cache

**Link**: https://github.com/sgl-project/sglang/issues/106
**State**: closed
**Created**: 2024-01-26T14:01:54+00:00
**Closed**: 2024-01-30T01:12:36+00:00
**Comments**: 1

### Description

Thanks so much for the work on this repo so far.

I think prefix caching could be very useful and I see that vLLM is also starting to support it for some architectures.

It looks like the [BaseBackend.prefix_cache](https://github.com/sgl-project/sglang/blob/81561f8e2d55d105aabbe0eab1b3b33f4fc04b0b/python/sglang/backend/base_backend.py#L19-L20) method still needs to be implemented:
```python
    def cache_prefix(self, prefix_str: str):
        pass
```

---

## Issue #N/A: kv cache pool leak detected when benchmark llama13B-awq using A40

**Link**: https://github.com/sgl-project/sglang/issues/105
**State**: closed
**Created**: 2024-01-26T03:39:00+00:00
**Closed**: 2024-07-25T06:32:06+00:00
**Comments**: 5
**Labels**: inactive

### Description

 UserWarning: Warning: available_size=35244, max_total_num_token=42308
KV cache pool leak detected!

---

## Issue #N/A: no response running python -m sglang.launch_server --model-path NousResearch/Llama-2-7b-chat-hf --port 30000

**Link**: https://github.com/sgl-project/sglang/issues/100
**State**: closed
**Created**: 2024-01-25T15:04:48+00:00
**Closed**: 2024-01-30T14:30:56+00:00
**Comments**: 2

### Description

when I try to use `sglang` locally according to README.md:
``` sh
python -m sglang.launch_server --model-path NousResearch/Llama-2-7b-chat-hf --port 30000
```
(I use NousResearch/Llama-2-7b-chat-hf because my access of meta-llama is pending)
however, I receive no response and no log print. when I run the python script:
```python
from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint

@function
def multi_turn_question(s, question_1, question_2):
    s += system("You are a helpful assistant.")
    s += user(question_1)
    s += assistant(gen("answer_1", max_tokens=256))
    s += user(question_2)
    s += assistant(gen("answer_2", max_tokens=256))

set_default_backend(RuntimeEndpoint("http://localhost:30000"))

state = multi_turn_question.run(
    question_1="What is the capital of the United States?",
    question_2="List two local attractions.",
)

for m in state.messages():
    print(m["role"], ":", m["content"])

prin

[... truncated for brevity ...]

---

## Issue #N/A: Fail to load TheBloke/tulu-2-dpo-70B-AWQ on A800*2: TimeoutError: result expired

**Link**: https://github.com/sgl-project/sglang/issues/99
**State**: closed
**Created**: 2024-01-25T11:35:16+00:00
**Closed**: 2024-07-25T06:32:11+00:00
**Comments**: 3
**Labels**: inactive

### Description

Thanks for your great work!
I am trying to load the AWQ model of Tulu-2-dpo-70B, here is my command line input:
```CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server --model-path TheBloke/tulu-2-dpo-70B-AWQ --tokenizer-path TheBloke/tulu-2-dpo-70B-AWQ --port 30000 --mem-fraction-static 0.5 --tp-size 2```
And it took me over 20 min to load the checkpoint into GPU memory, and I finally get the error:
```server started on [0.0.0.0]:10010
server started on [0.0.0.0]:10011
accepted ('127.0.0.1', 51884) with fd 6
welcome ('127.0.0.1', 51884)
accepted ('127.0.0.1', 40934) with fd 6
welcome ('127.0.0.1', 40934)
Rank 1: load weight begin.
quant_config: AWQConfig(weight_bits=4, group_size=128, zero_point=True)
Rank 0: load weight begin.
quant_config: AWQConfig(weight_bits=4, group_size=128, zero_point=True)
Rank 1: load weight end.
router init state: Traceback (most recent call last):
  File "/home/share/likai/sglang/python/sglang/srt/managers/router/manager.py", line 68, in

[... truncated for brevity ...]

---

## Issue #N/A: `TypeError: color must be int or single-element tuple` when processing a grayscale image with LLaVA

**Link**: https://github.com/sgl-project/sglang/issues/96
**State**: closed
**Created**: 2024-01-24T22:53:40+00:00
**Closed**: 2024-01-25T00:24:05+00:00
**Comments**: 2

### Description

When processing a grayscale image with different width and height the following error will occur.

```console
Exception in TokenizerManager:
Traceback (most recent call last):
  File "/home/gcpuser/sglang/python/sglang/srt/managers/tokenizer_manager.py", line 61, in get_pixel_values
    image = expand2square(
  File "/home/gcpuser/sglang/python/sglang/srt/mm_utils.py", line 167, in expand2square
    result = Image.new(pil_img.mode, (width, width), background_color)
  File "/opt/conda/envs/sglang_flashinfer/lib/python3.9/site-packages/PIL/Image.py", line 2941, in new
    return im._new(core.fill(mode, size, color))
TypeError: color must be int or single-element tuple
```

The error originates here and it fails because PIL won't the background consisting of 3 values to the new image with `L` mode.

So far I only encountered this for grayscale images, perhaps one way to solve it would be to convert these images to `RGB` before resizing them?

---

## Issue #N/A: how to use the finetuned mistral model for inference with sglang

**Link**: https://github.com/sgl-project/sglang/issues/94
**State**: closed
**Created**: 2024-01-24T16:14:41+00:00
**Closed**: 2024-01-30T14:25:14+00:00
**Comments**: 2

### Description

how to use the finetuned mistral model for inference with sglang. 
Please share the code for this

---

## Issue #N/A: Support Yi-VL-6B/34B

**Link**: https://github.com/sgl-project/sglang/issues/91
**State**: closed
**Created**: 2024-01-24T03:49:46+00:00
**Closed**: 2024-02-01T21:38:25+00:00
**Comments**: 7
**Labels**: good first issue

### Description

The Yi-VL adopts llava but with silightly different in weights and inference. see [disscusion](https://huggingface.co/01-ai/Yi-VL-34B/discussions/3)

hf repo:
https://huggingface.co/01-ai/Yi-VL-6B
https://huggingface.co/01-ai/Yi-VL-34B

---

## Issue #N/A: Customizing Sampling Behavior

**Link**: https://github.com/sgl-project/sglang/issues/89
**State**: closed
**Created**: 2024-01-24T00:06:48+00:00
**Closed**: 2024-02-06T18:25:19+00:00
**Comments**: 1

### Description

Hi, is there an interface to specify logits processors as in vLLM? 

If possible, could you specify how we can customize the sampling behavior during generation?

---

## Issue #N/A: 'Runtime' object has no attribute 'cache_prefix'

**Link**: https://github.com/sgl-project/sglang/issues/88
**State**: closed
**Created**: 2024-01-23T16:59:24+00:00
**Closed**: 2024-01-24T12:05:19+00:00
**Comments**: 9

### Description


`
runtime = sgl.Runtime(model_path="lmsys/vicuna-13b-v1.5")
`

```
Traceback (most recent call last):
  File "/root/test_sg.py", line 42, in <module>
    states = single_question.run_batch(
  File "/usr/local/lib/python3.10/dist-packages/sglang/lang/ir.py", line 178, in run_batch
    return run_program_batch(
  File "/usr/local/lib/python3.10/dist-packages/sglang/lang/interpreter.py", line 79, in run_program_batch
    pin_program(program, backend)
  File "/usr/local/lib/python3.10/dist-packages/sglang/lang/interpreter.py", line 132, in pin_program
    prefix_rid = backend.cache_prefix(prefix)
AttributeError: 'Runtime' object has no attribute 'cache_prefix'
```

How to fix this error?

(also how can I point it to existing cached models by say transformers
e.g. `~/.cache/huggingface/hub/models--lmsys--vicuna-13b-v1.5`)


---

## Issue #N/A: OOM Error on A40 GPU [Jupyter notebook]

**Link**: https://github.com/sgl-project/sglang/issues/86
**State**: closed
**Created**: 2024-01-23T12:03:01+00:00
**Closed**: 2024-07-25T06:32:01+00:00
**Comments**: 6
**Labels**: inactive

### Description

I was trying the following code sample (adapted from the discussion in https://github.com/sgl-project/sglang/issues/81) - 

```
import sglang as sgl
from sglang import function, gen, set_default_backend, Runtime

@sgl.function
def tool_use(s, question):
    s += "To answer this question: " + question + ", "
    s += "I need to use a " + sgl.gen("tool", choices=["calculator", "web browser"]) + ". "
    if s["tool"] == "calculator":
        s += "The math expression is" + sgl.gen("expression")
    elif s["tool"] == "web browser":
        s += "The website url is" + sgl.gen("url")

runtime = Runtime(model_path='Model_Saves/teknium--OpenHermes-2.5-Mistral-7B')
set_default_backend(runtime)

driver_tool_use()
```
   
I firstly got the same error as described here: https://github.com/sgl-project/sglang/issues/41#issuecomment-1899347676
I then followed Solution 2 from this [comment](https://github.com/sgl-project/sglang/issues/41#issuecomment-1899354400) and the error dis

[... truncated for brevity ...]

---

## Issue #N/A: Can the stop tokens be retained? 

**Link**: https://github.com/sgl-project/sglang/issues/85
**State**: closed
**Created**: 2024-01-23T06:39:30+00:00
**Closed**: 2024-07-25T06:32:03+00:00
**Comments**: 4
**Labels**: inactive

### Description

I set the stop token as `.`, but the output text will not include the stop token, so can the stop tokens be retained in the output?

request:
```bash
curl http://localhost:8081/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Once upon a time,",
    "sampling_params": {
      "max_new_tokens": 1024,
      "temperature": 0,
      "stop": ".",
    }
  }'
```

response:
```bash
{"text":" and in a land far, far away, there was a young girl named Lily","meta_info":{"prompt_tokens":6,"completion_tokens":18,"id":"fe8e78ca46ab4f4bb69842202201288c"}}
```

---

## Issue #N/A: Tutorial for Batch Decoding and Obtaining Log Probs

**Link**: https://github.com/sgl-project/sglang/issues/81
**State**: closed
**Created**: 2024-01-23T05:13:56+00:00
**Closed**: 2024-01-30T14:39:59+00:00
**Comments**: 25

### Description

Hi
Thanks for the great library
I have a usecase which I think will benefit a lot from Radix Attention. I need to obtain log probs for around a 100K sequences which can be binned into groups of 100 having a similar prefix like 'Wikipedia originated in' and having 100 different suffixes. I do not need to generate anything and I only need the log probs for the input. Is there a tutorial for such a usecase?

---

## Issue #N/A: ['LLaVA'] Error when trying to load a fine-tuned LLaVA model

**Link**: https://github.com/sgl-project/sglang/issues/79
**State**: closed
**Created**: 2024-01-22T22:24:11+00:00
**Closed**: 2024-01-23T02:15:56+00:00
**Comments**: 1

### Description

Hi, I encounter the following error when trying to load a fine-tuned LLaVA model:

```console
~$ python3 -m sglang.launch_server --model-path org/llava_1.5_13b_finetune --tokenizer-path llava-hf/llava-1.5-13b-hf --port 30000
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Process Process-1:
router init state: Traceback (most recent call last):
  File "/opt/conda/envs/llava_sglang/lib/python3.10/site-packages/sglang/srt/managers/router/manager.py", line 68, in start_router_process
    model_client = ModelRpcClient(server_args, port_args)
  File "/opt/conda/envs/llava_sglang/lib/python3.10/site-packages/sglang/srt/managers/router/model_rpc.py", line 448, in __init__
    self.model_server.exposed_init_model(0, server_args, port_args)
  File "/opt/conda/envs/llava_sglang/lib/python3.10/site-pa

[... truncated for brevity ...]

---

## Issue #N/A: Sketch guided constrained deciding for black box LLMs

**Link**: https://github.com/sgl-project/sglang/issues/78
**State**: closed
**Created**: 2024-01-22T19:07:11+00:00
**Closed**: 2024-07-25T06:32:00+00:00
**Comments**: 3
**Labels**: inactive

### Description

https://twitter.com/SaiboGeng/status/1749490603111387643?t=MRfYngCpJRB7zfZCpW43MA&s=19

Any thoughts about adding that?



---

## Issue #N/A: Not able to run AWQ Mixtral on 4xA10

**Link**: https://github.com/sgl-project/sglang/issues/77
**State**: closed
**Created**: 2024-01-22T18:23:22+00:00
**Closed**: 2024-02-22T14:58:11+00:00
**Comments**: 4
**Labels**: bug

### Description

Hi,

Im trying to run the AWQ version of Mixtral on 4xA10s. However im getting this error. Ive also tried with `--mem-frac 0.7` and still got the same error

Model I'm using : https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ

Command : `python -m sglang.launch_server --model-path /local_disk0/TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ/ --port 30000 --tp 4`

Code : 
```
from sglang import function, system, user, assistant, gen
import sglang as sgl

@function
def multi_turn_question(s, question_1, question_2):
    s += system("You are a helpful assistant.")
    s += user(question_1)
    s += assistant(gen("answer_1", max_tokens=256))
    s += user(question_2)
    s += assistant(gen("answer_2", max_tokens=256))

state = multi_turn_question.run(
    question_1="What is the capital of the United Kingdom?",
    question_2="List two local attractions.",
    temperature=0.7,
    stream=True,
)

for out in state.text_iter():
    print(out, end="", flush=T

[... truncated for brevity ...]

---

## Issue #N/A: Support for offline batch mode with local models?

**Link**: https://github.com/sgl-project/sglang/issues/76
**State**: closed
**Created**: 2024-01-22T13:02:31+00:00
**Closed**: 2024-01-23T13:23:51+00:00
**Comments**: 4

### Description

Hello, guys,
Any plans to support offline batch inference mode with local models, without spinning up an additional server? similar to [what is implemented in vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#offline-batched-inference). It would be way easier to use. 
Thanks! 

---

## Issue #N/A: Continues batch technical for different length prompt

**Link**: https://github.com/sgl-project/sglang/issues/74
**State**: closed
**Created**: 2024-01-22T08:45:37+00:00
**Closed**: 2024-01-23T04:40:05+00:00
**Comments**: 4

### Description

Suppose model's max-model-length is 8096, and there is two requests, one prompt length is 8, another is 10. And how concat them, pad them to 8096 or pad them to 10 or truncate them to 8.
I feel you will pad them to 16. But I don't seek the code. Can you tell me the location?
Thanks.

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

## Issue #N/A: Installing sglang[openai] does not have numpy requirement

**Link**: https://github.com/sgl-project/sglang/issues/70
**State**: closed
**Created**: 2024-01-21T21:59:38+00:00
**Closed**: 2024-01-21T22:56:42+00:00
**Comments**: 1

### Description

I tried installing sglang[openai], ran the code in the readme:
```python
import sglang as sgl
import snoop

@sgl.function
def tool_use(s, question):
    s += "To answer this question: " + question + ", "
    s += "I need to use a " + sgl.gen("tool", choices=["calculator", "web browser"]) + ". "
    if s["tool"] == "calculator":
        s += "The math expression is" + sgl.gen("expression")
    elif s["tool"] == "web browser":
        s += "The website url is" + sgl.gen("url")


@sgl.function
def tip_suggestion(s):
    s += (
        "Here are two tips for staying healthy: "
        "1. Balanced Diet. 2. Regular Exercise.\n\n"
    )

    forks = s.fork(2)
    for i, f in enumerate(forks):
        f += f"Now, expand tip {i+1} into a paragraph:\n"
        f += sgl.gen(f"detailed_tip", max_tokens=256, stop="\n\n")

    s += "Tip 1:" + forks[0]["detailed_tip"] + "\n"
    s += "Tip 2:" + forks[1]["detailed_tip"] + "\n"
    s += "In summary" + sgl.gen("summary")


[... truncated for brevity ...]

---

## Issue #N/A: How create a new branch?

**Link**: https://github.com/sgl-project/sglang/issues/69
**State**: closed
**Created**: 2024-01-21T12:27:58+00:00
**Closed**: 2024-01-21T13:02:51+00:00
**Comments**: 2

### Description

I just fix some bugs and support a new model. I hope create a new branch.

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

## Issue #N/A: Issue reproducing the example in Readme 

**Link**: https://github.com/sgl-project/sglang/issues/65
**State**: closed
**Created**: 2024-01-21T08:22:42+00:00
**Closed**: 2024-01-21T10:12:24+00:00
**Comments**: 1

### Description

```
from sglang import function, system, user, assistant, gen, set_default_backend, OpenAI

@function
def multi_turn_question(s, question_1, question_2):
    s += system("You are a helpful assistant.")
    s += user(question_1)
    s += assistant(gen("answer_1", max_tokens=256))
    s += user(question_2)
    s += assistant(gen("answer_2", max_tokens=256))

set_default_backend(OpenAI("gpt-3.5-turbo"))

state = multi_turn_question.run(
    question_1="What is the capital of the United States?",
    question_2="List two local attractions.",
)

for m in state.messages():
    print(m["role"], ":", m["content"])
```

```
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
[<ipython-input-3-ac9af75fa374>](https://localhost:8080/#) in <cell line: 11>()
      9     s += assistant(gen("answer_2", max_tokens=256))
     10 
---> 11 set_default_backend(OpenAI("gpt-3.5-turb

[... truncated for brevity ...]

---

## Issue #N/A: Issue when using choices option.  assert temperature <= 1e-5      TypeError: '<=' not supported between instances of 'NoneType' and 'float'

**Link**: https://github.com/sgl-project/sglang/issues/62
**State**: closed
**Created**: 2024-01-21T01:22:33+00:00
**Closed**: 2024-01-21T07:50:47+00:00
**Comments**: 5

### Description

I ran into the following error when running my code:

Code
```
import sglang as sgl

sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

@sgl.function
def test_function(s, test_prompt):
    s += "This is a test, I will provide a prompt and answer to the best of your abilities. Once you are done, end with the word 'END'" + "\n"
    s += "Q: " + test_prompt + "\n"
    s += "A: " + sgl.gen("car", max_tokens=100, stop='END') + "\n"
    s += "Q: Who would win batman or superman. You must choose ONE character in ONE sentence \n"  
    # s += "A: " + sgl.gen("hero", max_tokens=16, stop='END',)
    s += "A: " + sgl.gen("hero", choices=["batman", "superman"],)


output = test_function.run(test_prompt = "What is the fastest car?")

print(output["car"])
print(output["hero"])

print(output)
```

Error
```
Exception in thread Thread-7 (_thread_worker_func):
Traceback (most recent call last):
  File "/home/sr/anaconda3/envs/sglangONLY/lib/python3.11/

[... truncated for brevity ...]

---

## Issue #N/A: "WARNING:  Invalid HTTP request received" and latency SGLANG vs VLLM

**Link**: https://github.com/sgl-project/sglang/issues/61
**State**: closed
**Created**: 2024-01-20T18:50:25+00:00
**Closed**: 2024-01-30T15:26:55+00:00
**Comments**: 7
**Labels**: bug

### Description

Hi team,
I am using `sglang` with a local finetuned model (`basemodel_id = cognitivecomputations/dolphin-2.2.1-mistral-7b`). And running inference in a for loop.
GPU: 4090
batch_sz=1
tokens_in ~ 2000
tokens_out ~200

```
runtime = load_model(model_id)
for p in tqdm(prompts):
   resp = inference_sglang(p)

runtime.shutdown()
```
When the model is loaded I am getting:
`WARNING:  Invalid HTTP request received.`
which repeats itself until the code reaches the line `runtime.shutdown()`

1. Why am I getting this warning? 
2. does it affect inference time?
I ran the same prompts with `vllm` and inference times are very similar for `sglang` and `vllm`.
My prompts are single instruction (no multi-shot prompting as in your code examples):
```<s><[INST]{my_instruction}[/INST]```
is that a case, where sglang should show better performance than `vllm`?

Thank you

---

## Issue #N/A: sglang.launch_server raise "POST /v1/chat/completions HTTP/1.1" 404 Not Found

**Link**: https://github.com/sgl-project/sglang/issues/59
**State**: closed
**Created**: 2024-01-20T07:15:47+00:00
**Closed**: 2024-01-21T10:15:27+00:00
**Comments**: 5

### Description

No description provided.

---

## Issue #N/A: [Feature Request] Enable working with Azure-OpenAI API (openai.AzureOpenAI())

**Link**: https://github.com/sgl-project/sglang/issues/56
**State**: closed
**Created**: 2024-01-19T14:44:21+00:00
**Closed**: 2024-02-12T09:07:47+00:00
**Comments**: 6
**Labels**: good first issue

### Description

Sglang looks great to me, but at my work, we use the Azure-OpenAI API. I don't see how to access this with sglang.

It would need two inputs in addition to the API-key, because at minimum I need to create the client like this:

```python
client = openai.AzureOpenAI(
    api_key="<your-api-key>",
    base_url="https://<your-project-name>.openai.azure.com/openai",
    api_version="<your-api-version>",  # for example "2023-05-15"
)
```

Also, for some reason the models are called "gpt-35-turbo" instead of "gpt-3.5-turbo" (missing dot); and I believe that you can call your models whatever you want. This should be supported, too.

If this already works somehow, I would appreciate an explicit mention in the `README.md`. 

---

## Issue #N/A: `run_batch()` RuntimeError: Trying to create tensor with negative dimension

**Link**: https://github.com/sgl-project/sglang/issues/55
**State**: closed
**Created**: 2024-01-19T11:22:38+00:00
**Closed**: 2024-01-21T10:16:10+00:00
**Comments**: 2
**Labels**: bug

### Description

Hello, team!

Thanks for the excellent work.
When working batch inference, sometimes encountering server-side error that completely interrupts the process:
```
new fill batch. #seq: 7. #cached_token: 1541. #new_token: 4. #remaining_req: 0. #running_req: 0
Exception in ModelRpcClient:
Traceback (most recent call last):
  File "/opt/conda/envs/env/lib/python3.11/site-packages/sglang/srt/managers/router/model_rpc.py", line 130, in exposed_step
    self.forward_step()
  File "/opt/conda/envs/env/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/envs/env/lib/python3.11/site-packages/sglang/srt/managers/router/model_rpc.py", line 145, in forward_step
    self.forward_fill_batch(new_batch)
  File "/opt/conda/envs/env/lib/python3.11/site-packages/sglang/srt/managers/router/model_rpc.py", line 328, in forward_fill_batch
    logits, normalized_logprobs = self

[... truncated for brevity ...]

---

## Issue #N/A: Meaningless generated output with V100

**Link**: https://github.com/sgl-project/sglang/issues/54
**State**: closed
**Created**: 2024-01-19T09:32:33+00:00
**Closed**: 2024-01-22T00:53:39+00:00
**Comments**: 3
**Labels**: bug

### Description

Greate work. 
But When I run examples/quick_start/srt_example_complete.py with RuntimeEndpoint("http://localhost:30000") with Server and V100 32GB
```
python -m sglang.launch_server --model-path ~/model/Llama-2-7b-chat-hf/ --port 30000
```

Got result:
```
system : You are a helpful assistant.
{'role': 'system', 'content': 'You are a helpful assistant.'}
user : What is the capital of the United States?
{'role': 'user', 'content': 'What is the capital of the United States?'}
assistant : It Swiss made S that it has? dot system.You don't even NEED a computer just to use it. You can use it with or without alexa or Google to Wall Street stock screym to place home alarm system<– butwith or wit out the internetNo diversity, Way cool

We will pay for your answer leading to ROI (Return On Investment) generation.
{'role': 'assistant', 'content': "It Swiss made S that it has? dot system.You don't even NEED a computer just to use it. You can use it with or without alexa or Google to

[... truncated for brevity ...]

---

## Issue #N/A: Can SGL generate list of json?

**Link**: https://github.com/sgl-project/sglang/issues/53
**State**: closed
**Created**: 2024-01-19T08:04:28+00:00
**Closed**: 2024-07-25T06:31:59+00:00
**Comments**: 8
**Labels**: inactive

### Description

I want to generate the following format, that is, list of jsons:
[
{"name": "Alice", "age": 1},
{"name": "Bob", "age": 2},
]
The number of the objects in the list is random depending on the output of LLM. So can SGL support such format?

---

## Issue #N/A: Mixtral OutOfMemoryError with 2 GPUs

**Link**: https://github.com/sgl-project/sglang/issues/51
**State**: closed
**Created**: 2024-01-19T02:57:10+00:00
**Closed**: 2024-01-19T03:47:01+00:00
**Comments**: 4

### Description

I'm trying to run Mixtral (Mixtral Hermes) with two 48GB GPUs but it seems that sglang server is not using my second GPU.

`CUDA_VISIBLE_DEVICES="0,1" python -m sglang.launch_server --model-path /workspace/model --port 30000`

errors out with

```
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Process Process-1:
Traceback (most recent call last):
router init state: Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/sglang/srt/managers/router/manager.py", line 68, in start_router_process
    model_client = ModelRpcClient(server_args, port_args)
  File "/usr/local/lib/python3.10/dist-packages/sglang/srt/managers/router/model_rpc.py", line 448, in __init__
    self.model_server.exposed_init_model(0, server_args, port_args)
  File "/usr/local/lib/python3.10

[... truncated for brevity ...]

---

## Issue #N/A: OpenAI speculative execution

**Link**: https://github.com/sgl-project/sglang/issues/44
**State**: closed
**Created**: 2024-01-18T18:09:31+00:00
**Closed**: 2024-01-25T10:10:02+00:00
**Comments**: 3
**Labels**: enhancement, high priority

### Description

The current frontend using OpenAI will invoke multiple calls for the example below:
```
@sgl.function
def example(s):
  s += "Construct a character."
  s += "Name: " + gen("name") + " Birthday: " + gen("birthday") + " Job: " + gen("job")
```
We can optimize this to send less number of calls to save money:
1. Gen longer in the first gen call, and skip the later if the first gen did the right thing.
2. Allow using OpenAI's n=10 keyword argument to sample multiple completions when forked. We can also provide the interface `example.run(n=10)`.

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

## Issue #N/A: Custom chat template

**Link**: https://github.com/sgl-project/sglang/issues/40
**State**: closed
**Created**: 2024-01-18T15:37:27+00:00
**Closed**: 2024-01-18T22:02:09+00:00
**Comments**: 2

### Description

This would be useful when using a model like mistral-instruct, or any model that doesn't have a standardised template like chatml. Or for example using a finetuned model that uses a custom chat/instruct template.

---

## Issue #N/A: LLM integration with normal programming patterns or, a high level sglang interface

**Link**: https://github.com/sgl-project/sglang/issues/39
**State**: closed
**Created**: 2024-01-18T14:14:29+00:00
**Closed**: 2024-07-25T06:31:58+00:00
**Comments**: 3
**Labels**: enhancement, inactive

### Description

I posted a similar issue in outlines, but here goes:  we're building something complex and I think it would be helpful to have a marvin-like library that supports normal programming patterns with LLM's but also gives control over generation. This  would provide high level pythonic abstractions like typed functions dynamically compiling grammars for return pydantic structs that would also allow you to drop down to customize generation either within or around these functions. This could    be like high level mypy typed boundaries around sglang programs.

[Marvin](https://github.com/PrefectHQ/marvin) and [funcchain](https://github.com/shroominic/funcchain) do the high level (sort of), but you give up control. Marvin relies on json and/or function calling and is constrained to OAI models, funcchain uses dynamically compiled  Lllamacpp grammar   as well. 

Analogy would be Pytorch:triton::funcchain/equivalent:sglang

Aside from the funcchain-like feature, for my use case I'd love to s

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

## Issue #N/A: Triton support

**Link**: https://github.com/sgl-project/sglang/issues/35
**State**: closed
**Created**: 2024-01-18T06:31:20+00:00
**Closed**: 2024-11-01T05:54:44+00:00
**Comments**: 9

### Description

Hello, curious if we can already use sglang as a backend for NVIDIA's Triton Server.

Amazing work with the library btw, love it!

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

## Issue #N/A: [Feature Request] Optimized quantised kernels

**Link**: https://github.com/sgl-project/sglang/issues/28
**State**: closed
**Created**: 2024-01-17T23:14:04+00:00
**Closed**: 2024-03-13T02:10:13+00:00
**Comments**: 2

### Description

https://github.com/IST-DASLab/marlin

---

## Issue #N/A: Max Length.

**Link**: https://github.com/sgl-project/sglang/issues/27
**State**: closed
**Created**: 2024-01-17T21:26:39+00:00
**Closed**: 2024-07-25T06:31:56+00:00
**Comments**: 3
**Labels**: good first issue, inactive

### Description

Is there anyway to truncate text based on tokens? I really like that as a user I don't need to think about tokens. But to save memory I would like something like 
`s += left_trunc(inp, 500)` that keeps it reasonably sized.

---

## Issue #N/A: OpenAI Chat Completion Endpoint

**Link**: https://github.com/sgl-project/sglang/issues/26
**State**: closed
**Created**: 2024-01-17T21:25:15+00:00
**Closed**: 2024-01-19T07:43:10+00:00
**Comments**: 2

### Description

First of all, thank you for such a great framework and study! Do you plan to support `chat/completions` endpoint as well for the models utilizing their `chat_template` for completion during backend serving?

---

## Issue #N/A: Batching semantics?

**Link**: https://github.com/sgl-project/sglang/issues/25
**State**: closed
**Created**: 2024-01-17T21:24:06+00:00
**Closed**: 2024-01-17T23:19:30+00:00
**Comments**: 2

### Description

I'm curious about how careful I need to be with batching. If I batch together 50 calls of which which there are 5 unique prefixes (a,b,c,d,e), will it know to group those together. Or should I be careful about making that 5 different batches?

---

## Issue #N/A: Offline Generation

**Link**: https://github.com/sgl-project/sglang/issues/24
**State**: closed
**Created**: 2024-01-17T21:10:27+00:00
**Closed**: 2024-01-17T21:29:02+00:00
**Comments**: 1

### Description

Hi, is it possible to do offline Generation similar to the vllm batch Inference where the model is not served?

Like
```
Llm = sglang("path/to/llm")
```

---

## Issue #N/A: Metal support?

**Link**: https://github.com/sgl-project/sglang/issues/23
**State**: closed
**Created**: 2024-01-17T20:16:06+00:00
**Closed**: 2024-07-25T06:32:59+00:00
**Comments**: 4
**Labels**: inactive

### Description

Hey, when is planned the support for Metal backend? 

---

## Issue #N/A: [Quantization Support Request]: Exllamav2

**Link**: https://github.com/sgl-project/sglang/issues/22
**State**: closed
**Created**: 2024-01-17T19:37:50+00:00
**Closed**: 2024-07-25T06:32:53+00:00
**Comments**: 3
**Labels**: inactive

### Description

Exllamav2 is an excellent quantization method that would allow to use big models in consumer (~24Gb GPUs) thanks to fractional quantization methods. Would this be in the cards?

---

## Issue #N/A: [Feature Request] CFG in Backend Calls

**Link**: https://github.com/sgl-project/sglang/issues/21
**State**: closed
**Created**: 2024-01-17T18:50:56+00:00
**Closed**: 2024-07-25T06:31:57+00:00
**Comments**: 1
**Labels**: enhancement, inactive

### Description

Hello. In our use case, we would like to make calls to the backend with a "grammar" or "cfg" field similar to what outline implements in their vllm implementation:

https://github.com/outlines-dev/outlines/pull/517

Would this be in the cards? I see there is already a Lark CFG implementation, but it seems to be front end/python only. 

---

## Issue #N/A: Colab? 

**Link**: https://github.com/sgl-project/sglang/issues/14
**State**: closed
**Created**: 2024-01-16T20:08:21+00:00
**Closed**: 2024-07-25T06:32:37+00:00
**Comments**: 11
**Labels**: collaboration, inactive

### Description

Awesome project. We have a paper https://arxiv.org/abs/2310.14034 with really complicated KV caching that I would love to go back and implement in SGLang. 

I tried to get an example working in Colab for a demo, but I got kind of stuck getting the server running. 

This runs fine: 

!nohup python -m sglang.launch_server --model-path TheBloke/Mistral-7B-v0.1-AWQ --port 30000

But then when I run the following, 

```
%%script bash
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Say this is a test",
    "max_tokens": 16,
    "temperature": 0
  }'
```

I just get this. 

```
KV cache pool leak detected!
Warning: available_size=75821, max_total_num_token=75833
KV cache pool leak detected!
Warning: available_size=75821, max_total_num_token=75833
KV cache pool leak detected!
Warning: available_size=75821, max_total_num_token=75833
KV cache pool leak detected!
Warning: available_size=75821, max_total_num

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

## Issue #N/A: Typo: rename image_url to image_file

**Link**: https://github.com/sgl-project/sglang/issues/5
**State**: closed
**Created**: 2024-01-13T06:41:07+00:00
**Closed**: 2024-01-16T23:42:00+00:00
**Comments**: 2

### Description

https://github.com/sgl-project/sglang/blob/f652494df16ef9fa0fac998ddf63961aee0849d4/python/sglang/srt/utils.py#L212

---

