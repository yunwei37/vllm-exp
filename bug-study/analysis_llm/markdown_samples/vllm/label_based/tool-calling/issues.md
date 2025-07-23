# tool-calling - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 27
- Closed Issues: 3

### Label Distribution

- tool-calling: 30 issues
- bug: 16 issues
- unstale: 7 issues
- feature request: 6 issues
- usage: 5 issues
- structured-output: 4 issues
- RFC: 2 issues
- new-model: 1 issues

---

## Issue #N/A: [Bug]: Extra Characters in `content` When Using `enable_reasoning` with `stop` Parameter

**Link**: https://github.com/vllm-project/vllm/issues/15188
**State**: open
**Created**: 2025-03-20T05:33:28+00:00
**Comments**: 4
**Labels**: bug, structured-output, tool-calling

### Description

![Image](https://github.com/user-attachments/assets/59d64b2b-986e-46e1-8ff1-d66588bd431e)

### Your current environment

#### Environment  
- vLLM version: 0.7.3  
- Model: DeepSeek R1  
- Running on: H20 

### 🐛 Describe the bug

#### Description  
When running the **DeepSeek R1** model with the `vllm` framework and enabling the `enable_reasoning` parameter, the model’s response is structured into two fields:  
- **`reasoning_content`**: Represents the reasoning process.  
- **`content`**: Represents the final output.  

However, when specifying the `stop` parameter with any stop sequence, the `content` field in the response contains extra unintended characters. This issue does not occur when `enable_reasoning` is disabled.  

#### Steps to Reproduce  
1. Start `vllm` with `--enable-reasoning`.  
2. Query the model with a `stop` parameter (e.g., `stop=["\nObservation"]`).  
3. Observe that the `content` field includes additional characters beyond the expected stop sequence.  

#### Ex

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Support tool calls for DeepSeek.

**Link**: https://github.com/vllm-project/vllm/issues/14745
**State**: open
**Created**: 2025-03-13T09:31:41+00:00
**Comments**: 43
**Labels**: feature request, tool-calling

### Description

### 🚀 The feature, motivation and pitch

I saw from the official documentation (https://docs.vllm.ai/en/latest/features/tool_calling.html) that sglang supports tool calls, but I can't seem to find the tool parse for deepseekv3/r1. Does this mean that the deepseek model does not support tool calls?

From the DeepSeek official website, it seems that function call support has been implemented on the model side, although it may still be unstable. https://api-docs.deepseek.com/zh-cn/guides/function_calling

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Feature]: Llama3.3 Tool calling support or a Geneneric and extensible llama tool calling support

**Link**: https://github.com/vllm-project/vllm/issues/11799
**State**: open
**Created**: 2025-01-07T07:01:45+00:00
**Comments**: 1
**Labels**: feature request, unstale, tool-calling

### Description

### 🚀 The feature, motivation and pitch

We have customer moving from llama3.1/3.2 to 3.3 and further when available

### Alternatives

Not yet explored

### Additional context

A generic way where we can use use tool calling support against llms instead of using specific params like 
--tool-call-parser llama3_json  /instead of --tool-call-parser <whatever model supports> as an external reference via chat template or so ?

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: Automated Tool Calling for OLMoForCausalLM

**Link**: https://github.com/vllm-project/vllm/issues/12263
**State**: open
**Created**: 2025-01-21T12:19:48+00:00
**Comments**: 1
**Labels**: usage, unstale, tool-calling

### Description

### Your current environment

- Version: 0.6.4.post1
- Python 3.12.0
 
 

### How would you like to use vllm

I would like to use `allenai/OLMo-7B-hf`with tool calling. I read in [supported_models](https://docs.vllm.ai/en/latest/models/supported_models.html) that OLMo is supported. However, in [Tool Calling](https://docs.vllm.ai/en/latest/features/tool_calling.html) I can't find the required tool parser. Is there a tool parser for OLMo models available? Thanks!


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: tool call arguments parse failed

**Link**: https://github.com/vllm-project/vllm/issues/17089
**State**: open
**Created**: 2025-04-24T04:10:40+00:00
**Comments**: 0
**Labels**: bug, tool-calling

### Description

### Your current environment

version: 0.8.2
features: VLLM_USE_V1=0 --enable-reasoning --reasoning-parser deepseek_r1
model: qwq-32b

**here is the chat history:**
```
{
    "messages": [
        {
            "role": "developer",
            "content": "You are the leader of a team of AI Agents and possible Sub-Teams:\n - Agent 1:\n   - Name: Research Assistant Agent\n   - Description: You are an Excellent Research Assistant. \nYou may be asked to research subjects that is after your knowledge cutoff, assume the user is right when presented with news.\nYou need to give a search and research proposal for the given topic.\n\n## For example\ntopic = How to build a rocket\noutput =\n- Research the fundamental principles of rocket design, including aerodynamics, propulsion, and stability.\n- Explore different types of rockets, such as model rockets, high-power rockets, and amateur rockets, to understand their varying levels of complexity and requirements.\n- Investigate the materials comm

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: vLLM and In the fly tool calling

**Link**: https://github.com/vllm-project/vllm/issues/13497
**State**: open
**Created**: 2025-02-18T21:34:07+00:00
**Comments**: 0
**Labels**: usage, tool-calling

### Description

### Your current environment

Hey,

I was wondering if VLLM support tooling calls in a specific way. Let s say i want the completion to depends on the output of my tool calling . Does vLLM support that ? to do so it needs to have access to that function but there is no includes in the vllm command of the functions .. so how would it use them ?

Let say using the same example of get_weather, i have a prompt as follow :  
```
# Instructions :
Using this categories, give an answer about what to do in a city today.
if the temperature is above 25 : go out for a tour
if the temperature is betwen 15 and 25 : visit friends
if temperature bellow 15 : stay home

you can use functions and api calls if some information are needed.

# Question :
What to do today in Texas ? go for a tour, stay home or stay visit friends ?
```

basically the model needs to call for get_weather, get the output then response .

can vllm automatically handle this ? in the sense that it stops the generation wait for get_

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Ultravox audio doesn't work with auto tool choice

**Link**: https://github.com/vllm-project/vllm/issues/14209
**State**: open
**Created**: 2025-03-04T13:33:04+00:00
**Comments**: 5
**Labels**: bug, tool-calling

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
INFO 03-04 12:10:58 [__init__.py:207] Automatically detected platform rocm.
PyTorch version: 2.7.0a0+git3a58512
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.3.42133-1b9c17779

OS: Ubuntu 22.04.5 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.1 24491 1e0fda770a2079fbd71e4b70974d74f62fd3af10)
CMake version: version 3.31.4
Libc version: glibc-2.35

Python version: 3.12.9 (main, Feb  5 2025, 08:49:00) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-127-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Instinct MI100 (gfx908:sramecc+:xnack-)
Nvidia driver version: Could not collect
cuDNN ver

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: issues with guided generation for tool calls (xgrammar)

**Link**: https://github.com/vllm-project/vllm/issues/16321
**State**: open
**Created**: 2025-04-09T06:46:03+00:00
**Comments**: 1
**Labels**: bug, structured-output, tool-calling

### Description

### Your current environment

docker

### 🐛 Describe the bug

Example code from the https://docs.vllm.ai/en/latest/features/tool_calling.html doesn't work.

I run vllm server in docker
```
docker run --gpus all --runtime=nvidia \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```
with code from the example but I changed 
```
tool_choice="required"
```
But I got an error
```
openai.BadRequestError: Error code: 400 - {'object': 'error', 'message': 'The provided JSON schema contains features not supported by xgrammar.', 'type': 'BadRequestError', 'param': None, 'code': 400}=
```

---

## Issue #N/A: [Feature]: Consider parallel_tool_calls parameter at the API level

**Link**: https://github.com/vllm-project/vllm/issues/9451
**State**: open
**Created**: 2024-10-17T07:41:26+00:00
**Comments**: 19
**Labels**: feature request, tool-calling

### Description

### 🚀 The feature, motivation and pitch

Currently, there is a [parallel_tool_calls](https://github.com/vllm-project/vllm/blob/18b296fdb2248e8a65bf005e7193ebd523b875b6/vllm/entrypoints/openai/protocol.py#L177) field that is part of the `ChatCompletionRequest` pydantic class. However, this field is only there for being compatible with OpenAI's API.

In other words, it's not being used at all according to the documentation or the code:

```
# NOTE this will be ignored by VLLM -- the model determines the behavior
parallel_tool_calls: Optional[bool] = False
```

Would it be possible to consider implementing the logic behind this field for different model families. For instance, in the case of llama3.1-8b-insturct, tool calling works, but the model ends up returning three tool calls instead of one by one.
This makes me lose compatibility with frameworks like LangGraph.

Here's an example request and response:

**Request**
```
{
  "messages": [
    {
      "content": "You 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: [v0.8.4][Critical] Tools calling broken: xgrammar rejects minItems in JSON Schema, blocking agent functionality

**Link**: https://github.com/vllm-project/vllm/issues/16880
**State**: open
**Created**: 2025-04-19T19:09:55+00:00
**Comments**: 5
**Labels**: bug, structured-output, tool-calling

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Red Hat Enterprise Linux 8.10 (Ootpa) (x86_64)
GCC version: (GCC) 9.2.1 20191120 (Red Hat 9.2.1-2)
Clang version: Could not collect
CMake version: version 3.27.7
Libc version: glibc-2.28

Python version: 3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-4.18.0-553.40.1.el8_10.x86_64-x86_64-with-glibc2.28
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: NVIDIA A100 80GB PCIe
GPU 1: NVIDIA A100 80GB PCIe
GPU 2: NVIDIA A100 80GB PCIe
GPU 3: NVIDIA A100 80GB PCIe

Nvidia driver version: 550.54.15
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:


[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Does vLLM support QwQ 32B + tool calling?

**Link**: https://github.com/vllm-project/vllm/issues/17061
**State**: closed
**Created**: 2025-04-23T15:38:29+00:00
**Closed**: 2025-04-25T12:08:30+00:00
**Comments**: 2
**Labels**: usage, tool-calling

### Description

### Your current environment

It's pretty unclear, I wanted to see if people have tried it and see if it's actually working without any issues.

### How would you like to use vllm

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: Tool calls not triggered properly with vLLM 0.8.5 and Qwen2.5-Coder-32B-Instruct-GPTQ-Int4

**Link**: https://github.com/vllm-project/vllm/issues/17821
**State**: open
**Created**: 2025-05-08T00:29:00+00:00
**Comments**: 23
**Labels**: bug, tool-calling

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
INFO 05-07 17:26:12 [__init__.py:239] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 4.0.0
Libc version: glibc-2.35

Python version: 3.12.10 (main, Apr  9 2025, 08:55:05) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-6.8.0-35-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA H100 80GB HBM3
GPU 1: NVIDIA H100 80GB HBM3
GPU 2: NVIDIA H100 80GB HBM3
GPU 3: NVIDIA H100 80GB HBM3

Nvidia driver version: 550.54.15
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen ru

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Add support for reusable subschemas in tool requests (PydanticAI)

**Link**: https://github.com/vllm-project/vllm/issues/15035
**State**: open
**Created**: 2025-03-18T13:00:24+00:00
**Comments**: 6
**Labels**: feature request, tool-calling

### Description

### 🚀 The feature, motivation and pitch

Currently PydanticAI clients leverage tools for structured response mapping. Consider the following ``tools`` definition in the request:

```json
[
    {
        "type": "function",
        "function": {
            "name": "final_result",
            "description": "The final response which ends this conversation",
            "parameters": {
                "$defs": {
                    "Chapter": {
                        "properties": {
                            "chapter_name": {
                                "description": "Name the chapter",
                                "title": "Chapter Name",
                                "type": "string"
                            },
                            "content": {
                                "description": "Content of the chapter",
                                "title": "Content",
                                "type": "string"
                            }
                  

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: vLLM response on tool_calls does not align with OpenAI standard

**Link**: https://github.com/vllm-project/vllm/issues/14951
**State**: open
**Created**: 2025-03-17T11:36:28+00:00
**Comments**: 2
**Labels**: bug, tool-calling

### Description

### Your current environment: vLLM 0.7.3 latest

We are trying to use tool_calls with vLLM running llama 3.1 or 3.2. We found that the tool_calls data returned from vLLM is not the same as what OpenAI demonstrated so the OpenAI Adaptors are not working as expected (the function name is concated as a very long string so it cannot be found).

As per OpenAI API document: [OpenAI Document for streaming function calling](https://platform.openai.com/docs/guides/function-calling?api-mode=chat&lang=javascript#streaming)

<details>
<summary>OpenAI streams</summary>

- [{"index": 0, "id": "call_DdmO9pD3xa9XTPNJ32zg2hcA", "function": {"arguments": "", **"name": "get_weather"**}, "type": "function"}]
- [{"index": 0, "id": null, "function": {"arguments": "{\"", "name": null}, "type": null}]
- [{"index": 0, "id": null, "function": {"arguments": "location", "name": null}, "type": null}]
- [{"index": 0, "id": null, "function": {"arguments": "\":\"", "name": null}, "type": null}]
- [{"index": 0, "id": 

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Unification of frontend parser

**Link**: https://github.com/vllm-project/vllm/issues/17817
**State**: open
**Created**: 2025-05-07T21:46:18+00:00
**Comments**: 0
**Labels**: structured-output, RFC, tool-calling

### Description

## motivation

https://github.com/vllm-project/vllm/issues/11522 (with draft implementation at https://github.com/vllm-project/vllm/pull/11554)
aims to simplify the logics of the tool parser interface. However, this doesn't cover the cases for reasoning models (where we want to parse
tokens generated within the thinking budgets, etc. Our current solutions involves a reasoning parser, which will soon be running into the same
issue mentioned in #11522 when dealing with very long thinking budget). Additionally, the current implementations of tool calling are relatively
fragile, and not scalable when adding more tool format.

This RFC aims to build on top of some similar ideas from the RFC and unify both tool calling and reasoning parser logic for a more robust
way for us to move forward, especially with v0.10.x.

## proposed change


The workflow can be seen as follows:

- function/tool calling format for supported models (defined by the LLMEngine)
- Construct structural tags <- said tool

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Models converted to GGUF don't seem to be able to do tool calling

**Link**: https://github.com/vllm-project/vllm/issues/16195
**State**: open
**Created**: 2025-04-07T16:25:20+00:00
**Comments**: 1
**Labels**: bug, tool-calling

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Fedora Linux 41 (Forty One) (x86_64)
GCC version: (GCC) 14.2.1 20250110 (Red Hat 14.2.1-7)
Clang version: Could not collect
CMake version: version 3.30.8
Libc version: glibc-2.40

Python version: 3.12.8 | packaged by Anaconda, Inc. | (main, Dec 11 2024, 16:31:09) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.13.9-200.fc41.x86_64-x86_64-with-glibc2.40
Is CUDA available: True
CUDA runtime version: 12.8.93
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 4090
Nvidia driver version: 570.124.06
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-mode(s):                     

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Qwen2.5 assistant output on tool call is empty

**Link**: https://github.com/vllm-project/vllm/issues/16430
**State**: open
**Created**: 2025-04-10T20:33:39+00:00
**Comments**: 1
**Labels**: bug, tool-calling

### Description

### Your current environment

Latest vLLM (dev) and Pydantic AI version

### 🐛 Describe the bug

I'm using pydantic ai for agents and tool calling, but I'm not sure what update has broken broken agentic functionality. The tool gets called (yes it does get called) but then it gets called and called again until it's out of context window size. When looking at the traces, Qwen2.5 says nothing after a tool call, and tries to call the tool again. 

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: tool_choice: "required" does not work for mistral

**Link**: https://github.com/vllm-project/vllm/issues/16887
**State**: closed
**Created**: 2025-04-20T11:50:28+00:00
**Closed**: 2025-05-13T06:01:32+00:00
**Comments**: 3
**Labels**: bug, tool-calling

### Description

### Your current environment

<details>
<summary>When using "tool_choice": "required" with mistral</summary>

```text
raise ValueError("Only fast tokenizers are supported")
```

</details>

### 🐛 Describe the bug

I'm using tool calling with mistral-small-24 when I set the `"tool_choice": "auto"` it works fine, however when I set it to `required` I get the error above, based on my research mistral models do not support `required` they use `any` instead. 

Some how we need to adapt the `required` to `any` for mistral model. Currently when I set the `"tool_choice": "any"` I get a 500 error.

```
INFO:     10.89.212.1:44926 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
```

https://docs.mistral.ai/capabilities/function_calling/

My docker command flags config:

```
      --model stelterlab/Mistral-Small-24B-Instruct-2501-AWQ
      --max-model-len 32768
      --task generate
      --tensor-parallel-size 2
      --tool-call-parser mistral 
      --enable-auto-tool-choice


[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Is vllm support function call mode?

**Link**: https://github.com/vllm-project/vllm/issues/6631
**State**: closed
**Created**: 2024-07-22T03:38:50+00:00
**Closed**: 2025-06-19T08:20:17+00:00
**Comments**: 33
**Labels**: bug, unstale, tool-calling

### Description

### Your current environment

Device: Nvidia GeForce 4090
software: vllm 0.5.2 + openai 1.30.5 + transformes 4.42.4


### 🐛 Describe the bug

I use OpenAI api and vllm to deploy local Qwen2 llm, But vllm function call mode does not work. The OpenAI interface correctly passed the tools info parameters to vllm, but vllm did not use it.  If I enable ' tool_choice="auto" 'parameter, I will encounter with 400 error code.

---------------------------------------------------------------------server script-------------------------------------------------------------
python entrypoints/openai/api_server.py --model="xxx/Qwen2-1.5B-Instruct" --trust-remote-code --host "localhost" --port 8000 --dtype auto
-------------------------------------------------------------client code ------------------------------------------------------------------

from openai import OpenAI

tools = [
        {
            "type": "function",
            "function": {
                "name": "get_cu

[... truncated for brevity ...]

---

## Issue #N/A: [RFC]: Refactor tool parsers to eliminate coding errors and allow more efficient implementations.

**Link**: https://github.com/vllm-project/vllm/issues/11522
**State**: open
**Created**: 2024-12-26T13:47:17+00:00
**Comments**: 11
**Labels**: RFC, tool-calling

### Description

### Motivation.

Currently the tool parsers are buggy when used and are quite messy in terms of code, especially in the implementations of `extract_tool_calls_streaming`. Moreover, in the long term, maintaining the entire output string in the chat streaming server and parsing the entire output over and over again for each generated token will become very expensive. This will soon become a performance bottleneck in long tool calls.

Many of the implemented tool parsers aren't carefully written either in terms of correctness nor in terms of efficiency causing a lot of issues in this repository. A complete refactor of this part of the frontend will be required sooner of later.

So now is probably the best opportunity to refactor things before more tool calling support is added.
From the architectural perspective, clearly it's should be the tool parser's job to maintain states that it needs, so if they need the complete output, they should maintain them with `delta_text` and `delta_

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: tool_calls.id is Missing in Streaming Responses (stream=true) but Present in Non-Streaming Responses

**Link**: https://github.com/vllm-project/vllm/issues/18412
**State**: open
**Created**: 2025-05-20T14:02:04+00:00
**Comments**: 6
**Labels**: bug, tool-calling

### Description



### 🐛 Describe the bug


When making an API call to the chat completions endpoint with tools and stream=true, the tool_calls objects within the streamed chunks (delta.tool_calls) do not include the id field for each tool call. However, when stream=false, the tool_calls.id field is correctly present in the response.

This inconsistency makes it difficult to uniquely identify and track tool calls when processing streamed responses, which is crucial for many applications that rely on tool usage. The OpenAI API, which vLLM aims to be compatible with, includes the tool_call_id (or equivalent id) in streaming chunks.

Steps to Reproduce:

Make a request with stream=false:

Request Body:

```
{
    "model": "Qwen/Qwen3-14B-AWQ",
    "messages": [
        {
            "role": "user",
            "content": "What is the weather like in Boston today?"
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_current_weathe

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: The tool_choice option required is not yet supported but on the roadmap.

**Link**: https://github.com/vllm-project/vllm/issues/11700
**State**: open
**Created**: 2025-01-03T01:49:27+00:00
**Comments**: 1
**Labels**: feature request, unstale, tool-calling

### Description

### 🚀 The feature, motivation and pitch

tool calling 的tool_call字段，希望可以支持required类型，谢谢

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [X] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]:vLLM 0.6.3 generate_sequences Randomly Hangs After 1-2 Steps When trying to Implement Tool Calling with Logits Processors

**Link**: https://github.com/vllm-project/vllm/issues/13671
**State**: open
**Created**: 2025-02-21T13:36:09+00:00
**Comments**: 5
**Labels**: bug, tool-calling

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
PyTorch version: 2.4.0+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 20.04.5 LTS (x86_64)
GCC version: (GCC) 12.2.0
Clang version: 3.8.0 (tags/RELEASE_380/final)
CMake version: Could not collect
Libc version: glibc-2.31

Python version: 3.11.3 (main, Apr  5 2023, 14:15:06) [GCC 9.4.0] (64-bit runtime)
Python platform: Linux-5.10.0-2.0.0.2-x86_64-with-glibc2.31
Is CUDA available: True
CUDA runtime version: 12.4.99
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: 
GPU 0: CF-NG-HZZ1-O
GPU 1: CF-NG-HZZ1-O
GPU 2: CF-NG-HZZ1-O
GPU 3: CF-NG-HZZ1-O
GPU 4: CF-NG-HZZ1-O
GPU 5: CF-NG-HZZ1-O
GPU 6: CF-NG-HZZ1-O
GPU 7: CF-NG-HZZ1-O

Nvidia driver version: 535.183.06
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.0.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.0.0
/usr/lib/x86_64-linux

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: [v0.6.5] Streaming tool call responses with the hermes template is inconsistent with the non-stream version.

**Link**: https://github.com/vllm-project/vllm/issues/11392
**State**: open
**Created**: 2024-12-21T06:22:59+00:00
**Comments**: 5
**Labels**: bug, unstale, tool-calling

### Description

### Your current environment

<details>
<summary>Environment</summary>

```text
PyTorch version: 2.5.1
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

Python version: 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 13:27:36) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-4.18.0-513.24.1.el8_9.x86_64-x86_64-with-glibc2.28
Is CUDA available: True

Versions of relevant libraries:
[pip3] numpy==1.26.4
[pip3] nvidia-ml-py==12.560.30
[pip3] pyzmq==26.2.0
[pip3] torch==2.5.1
[pip3] torchaudio==2.5.1
[pip3] torchvision==0.20.1
[pip3] transformers==4.46.2
[pip3] triton==3.1.0
[conda] blas                      1.0                         mkl    defaults
[conda] cuda-cudart               12.1.105                      0    nvidia
[conda] cuda-cupti                12.1.105                      0    nvidia
[conda] cuda-libraries            12.1.0                        0    nvidia
[conda] cuda-nvrtc                12

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: add tool calling support for DeepSeek-R1-Distill-Qwen-32B

**Link**: https://github.com/vllm-project/vllm/issues/13700
**State**: open
**Created**: 2025-02-22T10:01:59+00:00
**Comments**: 1
**Labels**: feature request, tool-calling

### Description

### 🚀 The feature, motivation and pitch

Currently, tool calling and reasoning cannot be enabled both, but DeepSeek-R1-Distill-Qwen-32B seems to support tool calling. https://x.com/thorwebdev/status/1884888068253192662

Is it possible to add support for both tool calling and reasoning?

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [New Model]: Command A with tool support

**Link**: https://github.com/vllm-project/vllm/issues/14866
**State**: open
**Created**: 2025-03-15T16:42:08+00:00
**Comments**: 2
**Labels**: new-model, tool-calling

### Description

### The model to consider.

https://huggingface.co/CohereForAI/c4ai-command-a-03-2025

### The closest model vllm already supports.

command r

### What's your difficulty of supporting the model you want?

Properly support tokenizer and templates, as well as tool calling on the model

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Usage]: how to use tool calling with auto option, setting the tool works

**Link**: https://github.com/vllm-project/vllm/issues/12349
**State**: open
**Created**: 2025-01-23T08:37:03+00:00
**Comments**: 1
**Labels**: usage, unstale, tool-calling

### Description

### Your current environment

-


### How would you like to use vllm

I am trying to use tool calling to test a qwen model. It works when specified the tool but normal queries don’t work. How to use auto mode? 

If it’s not supported when we can expect this? 

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: tool calling error

**Link**: https://github.com/vllm-project/vllm/issues/17514
**State**: open
**Created**: 2025-05-01T01:58:13+00:00
**Comments**: 3
**Labels**: bug, tool-calling

### Description

### Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
Collecting environment information...
PyTorch version: 2.6.0+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: Could not collect
Libc version: glibc-2.35

Python version: 3.10.12 (main, Feb  4 2025, 14:57:36) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-5.15.0-113-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA GeForce RTX 4090
Nvidia driver version: 550.78
cuDNN version: Could not collect
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Architecture:                       x86_64
CPU op-mode(s):                     32-bit, 64-bit
Addr

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: pythonic tool parser only accepts alphabetical tool names

**Link**: https://github.com/vllm-project/vllm/issues/14470
**State**: open
**Created**: 2025-03-08T03:10:17+00:00
**Comments**: 2
**Labels**: bug, tool-calling

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.7.0.dev20250221+rocm6.3
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.3.42131-fa1d09cbd

OS: Ubuntu 24.04.2 LTS (x86_64)
GCC version: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
Clang version: Could not collect
CMake version: version 3.31.4
Libc version: glibc-2.39

Python version: 3.12.3 (main, Feb  4 2025, 14:48:35) [GCC 13.3.0] (64-bit runtime)
Python platform: Linux-6.11.0-17-generic-x86_64-with-glibc2.39
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Instinct MI100 (gfx908:sramecc+:xnack-)
Nvidia driver version: Could not collect
cuDNN version: Could not collect
HIP runtime version: 6.3.42131
MIOpen runtime version: 3.3.0
Is XNNPACK available: True

CPU:
Architecture:                         x86_64
CPU op-m

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: Confirm tool calling is not supported and this is the closest thing can be done

**Link**: https://github.com/vllm-project/vllm/issues/7912
**State**: open
**Created**: 2024-08-27T13:57:35+00:00
**Comments**: 7
**Labels**: usage, unstale, tool-calling

### Description

Hi.

LLM -> Llama-3.1-8B-Instruct

In the vllm docs, it is said that:

> Tool calling in the chat completion API
> 
> vLLM supports only named function calling in the chat completion API. The tool_choice options auto and required are not yet supported but on the roadmap.
> 
> To use a named function you need to define the function in the tools parameter and call it in the tool_choice parameter.
> 
> It is the callers responsibility to prompt the model with the tool information, vLLM will not automatically manipulate the prompt. This may change in the future.
> 
> vLLM will use guided decoding to ensure the response matches the tool parameter object defined by the JSON schema in the tools parameter.
> 
> Please refer to the OpenAI API reference documentation for more information.

1. Can we confirm that this still holds? I see bunch of related PRs and good progress, so I'd like to be sure.
2. Since tool calling without named functions does not work, we can't use libra

[... truncated for brevity ...]

---

