# quick_1to24hours - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- bug: 9 issues
- feature request: 3 issues
- usage: 3 issues
- documentation: 1 issues
- ci-failure: 1 issues
- misc: 1 issues

---

## Issue #N/A: [Bug]: Error Using V1 Engine with DeepSeek Llama 70B

**Link**: https://github.com/vllm-project/vllm/issues/12522
**State**: closed
**Created**: 2025-01-28T19:31:10+00:00
**Closed**: 2025-01-28T22:39:19+00:00
**Comments**: 3
**Labels**: bug

### Description

### Your current environment

Vllm 0.7.0
CUDA 12.6
Driver Version 560.94
torch 2.5.1
transformers 4.46.0



### Model Input Dumps

Traceback (most recent call last):
  File "/home/nd600/miniconda3/envs/vllm/bin/vllm", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/nd600/miniconda3/envs/vllm/lib/python3.12/site-packages/vllm/scripts.py", line 201, in main
    args.dispatch_function(args)
  File "/home/nd600/miniconda3/envs/vllm/lib/python3.12/site-packages/vllm/scripts.py", line 42, in serve
    uvloop.run(run_server(args))
  File "/home/nd600/miniconda3/envs/vllm/lib/python3.12/site-packages/uvloop/__init__.py", line 109, in run
    return __asyncio.run(
           ^^^^^^^^^^^^^^
  File "/home/nd600/miniconda3/envs/vllm/lib/python3.12/asyncio/runners.py", line 194, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "/home/nd600/miniconda3/envs/vllm/lib/python3.12/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complet

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: model_executor/test_model_load_with_params.py  fails with AttributeError

**Link**: https://github.com/vllm-project/vllm/issues/18757
**State**: closed
**Created**: 2025-05-27T10:01:18+00:00
**Closed**: 2025-05-28T05:42:56+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

Issue encountered on main branch tests.

### 🐛 Describe the bug

Test failing with below traceback:

```
vllm_runner = <class 'tests.conftest.VllmRunner'>

    @pytest.mark.skipif(current_platform.is_rocm(),
                        reason="Xformers backend is not supported on ROCm.")
    def test_model_loading_with_params(vllm_runner):
        """
        Test parameter weight loading with tp>1.
        """
        with vllm_runner(model_name=MODEL_NAME,
                         revision=REVISION,
                         dtype="float16",
                         max_model_len=MAX_MODEL_LEN) as vllm_model:
            output = vllm_model.encode("Write a short story about a robot that"
                                       " dreams for the first time.\n")
    
            model_config = vllm_model.model.llm_engine.model_config
            model_tokenizer = vllm_model.model.llm_engine.tokenizer
    
            # asserts on the bert model config file
      

[... truncated for brevity ...]

---

## Issue #N/A: [Doc]: Add list of commands for `vllm serve`

**Link**: https://github.com/vllm-project/vllm/issues/19859
**State**: closed
**Created**: 2025-06-19T12:45:31+00:00
**Closed**: 2025-06-20T05:06:18+00:00
**Comments**: 4
**Labels**: documentation

### Description

### 📚 The doc issue

In previous versions of documentation there was list of available arguments for `vllm serve` (e.g.  which was very conviniet, but I can't find them for `v0.9.0`.

In previous versions of the documentation, there was a list of available arguments for `vllm serve`, such as in the version [`v0.8.5`](https://docs.vllm.ai/en/v0.8.5.post1/serving/openai_compatible_server.html#cli-reference)). This was very convenient, but I am unable to find this information for version `v0.9.0` and above

### Suggest a potential alternative/fix

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [CI Failure]: lint-and-deploy: unexpected HTTP status 500

**Link**: https://github.com/vllm-project/vllm/issues/19574
**State**: closed
**Created**: 2025-06-12T18:52:35+00:00
**Closed**: 2025-06-13T14:37:47+00:00
**Comments**: 2
**Labels**: ci-failure

### Description

### Name of failing test

lint-and-deploy

### Basic information

- [ ] Flaky test
- [ ] Can reproduce locally
- [ ] Caused by external libraries (e.g. bug in `transformers`)

### 🧪 Describe the failing test

https://github.com/vllm-project/vllm/actions/runs/15618574945/job/43997475275?pr=19573

https://github.com/vllm-project/vllm/actions/runs/15618287178/job/43996526081?pr=19572

```bash
Run helm/chart-testing-action@0d28d3144d3a25ea2cc349d6e59901c4ff469b3b
Run sigstore/cosign-installer@dc72c7d5c4d10cd6bcb8cf6e3fd625a9e5e537da
Run #!/bin/bash
INFO: Downloading bootstrap version 'v2.4.1' of cosign to verify version to be installed...
      https://github.com/sigstore/cosign/releases/download/v2.4.1/cosign-linux-amd64
INFO: bootstrap version successfully verified and matches requested version so nothing else to do
Run echo "$HOME/.cosign" >> $GITHUB_PATH
Run cd $GITHUB_ACTION_PATH \
Installing chart-testing v3.10.1...
Error: getting Rekor public keys: updating local metadata and target

[... truncated for brevity ...]

---

## Issue #N/A: out of memory running mixtral gptq model in vllm 0.2.7

**Link**: https://github.com/vllm-project/vllm/issues/2413
**State**: closed
**Created**: 2024-01-11T03:57:46+00:00
**Closed**: 2024-01-11T13:34:33+00:00
**Comments**: 4

### Description


https://github.com/vllm-project/vllm/assets/39525455/c6787334-8a22-4dd4-838a-9fff1a1e0a38

The model downloaded from https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ and it works well in vllm 0.2.6 but run out of memory using 0.2.7.

---

## Issue #N/A: [V1]: Stuck at "Automatically detected platform cuda" when using V1 serving llava-next

**Link**: https://github.com/vllm-project/vllm/issues/12810
**State**: closed
**Created**: 2025-02-06T06:40:50+00:00
**Closed**: 2025-02-07T03:46:08+00:00
**Comments**: 7
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Debian GNU/Linux 12 (bookworm) (x86_64)
GCC version: (Debian 12.2.0-14) 12.2.0
Clang version: Could not collect
CMake version: version 3.25.1
Libc version: glibc-2.36

Python version: 3.11.2 (main, May  2 2024, 11:59:08) [GCC 12.2.0] (64-bit runtime)
Python platform: Linux-5.4.210.bsk.6-amd64-x86_64-with-glibc2.36
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: GPU 0: NVIDIA A100-SXM4-80GB
Nvidia driver version: 535.161.08
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.4.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.4.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9.4.0
/usr/lib/x86_64-linux-gnu/libcudnn_engines_precompil

[... truncated for brevity ...]

---

## Issue #N/A: why vllm==0.3.3 need to access google

**Link**: https://github.com/vllm-project/vllm/issues/3170
**State**: closed
**Created**: 2024-03-04T03:24:43+00:00
**Closed**: 2024-03-04T19:17:14+00:00
**Comments**: 4

### Description

![微信图片_20240304112403](https://github.com/vllm-project/vllm/assets/38678334/f21e1ec0-bd6f-4b26-aeee-d6e4e5822fc2)


---

## Issue #N/A: > **Bug**:Interesting finding: The official pip package v0.6.3 is broken. However, installing `https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl` fixes this issue. (`vLLM API server version 0.6.3.post2.dev139+g622b7ab9`)

**Link**: https://github.com/vllm-project/vllm/issues/9828
**State**: closed
**Created**: 2024-10-30T04:58:44+00:00
**Closed**: 2024-10-31T04:47:05+00:00
**Comments**: 3

### Description

              > Interesting finding: The official pip package v0.6.3 is broken. However, installing `https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl` fixes this issue. (`vLLM API server version 0.6.3.post2.dev139+g622b7ab9`)

@SinanAkkoyun What does python 3.10.15 should install, seemly I meet the same issue, thanks a lot!!

_Originally posted by @Wiselnn570 in https://github.com/vllm-project/vllm/issues/9732#issuecomment-2445843188_
            

---

## Issue #N/A: docker env cannot launch openai.server

**Link**: https://github.com/vllm-project/vllm/issues/442
**State**: closed
**Created**: 2023-07-12T09:11:14+00:00
**Closed**: 2023-07-12T12:00:52+00:00
**Comments**: 2

### Description

docker: `nvcr.io/nvidia/pytorch:22.12-py3`
command: `python -m vllm.entrypoints.openai.api_server --host [0.0.0.0](http://0.0.0.0/) --port 8080`
Error:
```
root@ip-172-31-1-200:/workspace# python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8080
Traceback (most recent call last):
  File "/usr/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/usr/local/lib/python3.8/dist-packages/vllm/entrypoints/openai/api_server.py", line 17, in <module>
    from fastchat.model.model_adapter import get_conversation_template
  File "/usr/local/lib/python3.8/dist-packages/fastchat/model/__init__.py", line 1, in <module>
    from fastchat.model.model_adapter import (
  File "/usr/local/lib/python3.8/dist-packages/fastchat/model/model_adapter.py", line 13, in <module>
    import accelerate
  File "/usr/local/lib/python3.8/dis

[... truncated for brevity ...]

---

## Issue #N/A: Model support: starcoder2

**Link**: https://github.com/vllm-project/vllm/issues/3165
**State**: closed
**Created**: 2024-03-03T09:25:18+00:00
**Closed**: 2024-03-03T22:37:55+00:00
**Comments**: 2

### Description

[bigcode/starcoder2-15b](https://huggingface.co/bigcode/starcoder2-15b)

---

## Issue #N/A: [Feature]: Make `tool_choice` parameter optional when `--enable-auto-tool-choice` is passed

**Link**: https://github.com/vllm-project/vllm/issues/12834
**State**: closed
**Created**: 2025-02-06T14:22:03+00:00
**Closed**: 2025-02-06T23:51:15+00:00
**Comments**: 1
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

When using vLLM with `--enable-auto-tool-choice`, the `tool_choice` parameter is currently required in the request. If not specified, a ValueError is raised stating "tool_choice must either be a named tool, 'auto', or 'none'".

This differs from OpenAI's API behavior where `tool_choice` is optional and defaults to "auto" when tools are provided ([source](https://platform.openai.com/docs/guides/function-calling#tool-choice)). For better API compatibility and developer experience, I propose making the `tool_choice` parameter optional in vLLM with a default value of "auto" when `--enable-auto-tool-choice` is passed.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions

[... truncated for brevity ...]

---

## Issue #N/A: How can i get metrics like {gpu_cache_usage, cpu_cache_usage, time_to_first_tokens, time_per_output_tokens, time_per_output_tokens} when using offline inference 

**Link**: https://github.com/vllm-project/vllm/issues/2935
**State**: closed
**Created**: 2024-02-20T13:41:10+00:00
**Closed**: 2024-02-21T03:47:14+00:00
**Comments**: 0

### Description

Is there any metris api to call.

For example : (Wrong )

    start = time.perf_counter()
    # FIXME(woosuk): Do not use internal method.
    llm._run_engine(use_tqdm=True)
    end = time.perf_counter()
    Stats = llm.llm_engine._get_stats()
    llm.llm_engine.stat_logger._log_prometheus(Stats)
    return end - start

Thanks a lot 

---

## Issue #N/A: [Bug]: ImportError: cannot import name 'get_scheduler_metadata' from 'vllm.vllm_flash_attn'

**Link**: https://github.com/vllm-project/vllm/issues/16813
**State**: closed
**Created**: 2025-04-18T04:20:40+00:00
**Closed**: 2025-04-18T13:44:25+00:00
**Comments**: 8
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text

ERROR 04-18 10:16:33 [core.py:390] Traceback (most recent call last):
ERROR 04-18 10:16:33 [core.py:390]   File "/home/user/vllm/vllm/v1/engine/core.py", line 381, in run_engine_core
ERROR 04-18 10:16:33 [core.py:390]     engine_core = EngineCoreProc(*args, **kwargs)
ERROR 04-18 10:16:33 [core.py:390]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-18 10:16:33 [core.py:390]   File "/home/user/vllm/vllm/v1/engine/core.py", line 323, in __init__
ERROR 04-18 10:16:33 [core.py:390]     super().__init__(vllm_config, executor_class, log_stats,
ERROR 04-18 10:16:33 [core.py:390]   File "/home/user/vllm/vllm/v1/engine/core.py", line 63, in __init__
ERROR 04-18 10:16:33 [core.py:390]     self.model_executor = executor_class(vllm_config)
ERROR 04-18 10:16:33 [core.py:390]                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-18 10:16:33 [core.py:390]   File "/home/user/v

[... truncated for brevity ...]

---

## Issue #N/A: 'MistralConfig' object has no attribute 'num_local_experts'

**Link**: https://github.com/vllm-project/vllm/issues/2016
**State**: closed
**Created**: 2023-12-11T10:16:33+00:00
**Closed**: 2023-12-11T18:50:34+00:00
**Comments**: 3

### Description

I have the Transformers from git and also installed vllm from git. I am trying to get the model up and running: https://huggingface.co/DiscoResearch/DiscoLM-mixtral-8x7b-v2


2023-12-11 10:12:18 | ERROR | stderr |   File "/project/vllm_git/vllm/worker/model_runner.py", line 36, in load_model
2023-12-11 10:12:18 | ERROR | stderr |     self.model = get_model(self.model_config)
2023-12-11 10:12:18 | ERROR | stderr |   File "/project/vllm_git/vllm/model_executor/model_loader.py", line 117, in get_model
2023-12-11 10:12:18 | ERROR | stderr |     model = model_class(model_config.hf_config, linear_method)
2023-12-11 10:12:18 | ERROR | stderr |   File "/project/vllm_git/vllm/model_executor/models/mixtral.py", line 469, in __init__
2023-12-11 10:12:18 | ERROR | stderr |     self.layers = nn.ModuleList([
2023-12-11 10:12:18 | ERROR | stderr |   File "/project/vllm_git/vllm/model_executor/models/mixtral.py", line 470, in <listcomp>
2023-12-11 10:12:18 | ERROR | stderr |     MixtralDecod

[... truncated for brevity ...]

---

## Issue #N/A: Error in newest dev ver: Worker.__init__() got an unexpected keyword argument 'cache_config'

**Link**: https://github.com/vllm-project/vllm/issues/2640
**State**: closed
**Created**: 2024-01-29T01:57:34+00:00
**Closed**: 2024-01-29T20:30:10+00:00
**Comments**: 5
**Labels**: bug

### Description

I build the newest master branch with #2279 commit.
And I run the following command
`python -m vllm.entrypoints.openai.api_server --model ./Mistral-7B-Instruct-v0.2-AWQ --quantization awq --dtype auto --host 0.0.0.0 --port 8081 --tensor-parallel-size 2`
I meet the error:
```

INFO 01-29 09:41:47 api_server.py:209] args: Namespace(host='0.0.0.0', port=8081, allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, served_model_name=None, chat_template=None, response_role='assistant', ssl_keyfile=None, ssl_certfile=None, root_path=None, middleware=[], model='./Mistral-7B-Instruct-v0.2-AWQ', tokenizer=None, revision=None, tokenizer_revision=None, tokenizer_mode='auto', trust_remote_code=False, download_dir=None, load_format='auto', dtype='auto', kv_cache_dtype='auto', max_model_len=None, worker_use_ray=False, pipeline_parallel_size=1, tensor_parallel_size=2, max_parallel_loading_workers=None, block_size=16, seed=0, swap_space=4, gpu_

[... truncated for brevity ...]

---

## Issue #N/A: [Feature]: Add a vllm help CLI command

**Link**: https://github.com/vllm-project/vllm/issues/13938
**State**: closed
**Created**: 2025-02-27T02:52:39+00:00
**Closed**: 2025-02-27T04:43:44+00:00
**Comments**: 1
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

It would be helpful to have a command like vllm --help or vllm <subcommand> --help which provides a list of the available CLI commands and provides a brief explanation of how to use each command.

From #13840, it is possible that vllm may have many new commands in the future, so having this functionality would greatly help discern each specific subcommand.

### Alternatives

_No response_

### Additional context

_No response_

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

## Issue #N/A: [Bug]: The performance for `Prefix Caching` is very un-stable for different requests !!!!

**Link**: https://github.com/vllm-project/vllm/issues/3918
**State**: closed
**Created**: 2024-04-08T12:17:19+00:00
**Closed**: 2024-04-09T06:08:21+00:00
**Comments**: 8
**Labels**: bug

### Description

### Your current environment

```text
PyTorch version: 2.1.2+cu121
Is debug build: False
CUDA used to build PyTorch: 12.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.27.6
Libc version: glibc-2.35

Python version: 3.10.12 (main, Jun 11 2023, 05:26:28) [GCC 11.4.0] (64-bit runtime)
Python platform: Linux-4.18.0-240.el8.x86_64-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.2.140
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA L20
GPU 1: NVIDIA L20
GPU 2: NVIDIA L20
GPU 3: NVIDIA L20
GPU 4: NVIDIA L20
GPU 5: NVIDIA L20
GPU 6: NVIDIA L20
GPU 7: NVIDIA L20

Nvidia driver version: 550.54.15
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.8.9.5
/usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.9.5
/usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.9

[... truncated for brevity ...]

---

## Issue #N/A: the syntax in serving_completion.py is not compatible in python3.8

**Link**: https://github.com/vllm-project/vllm/issues/2704
**State**: closed
**Created**: 2024-02-01T06:14:36+00:00
**Closed**: 2024-02-01T22:00:59+00:00
**Comments**: 0

### Description

in #2529 @simon-mo introduce some python typing syntax, which is not compatible in python3.8

like  TypeTokenIDs = list[int] in https://github.com/vllm-project/vllm/blob/93b38bea5dd03e1b140ca997dfaadef86f8f1855/vllm/entrypoints/openai/serving_completion.py#L22C1-L22C25

which should be TypeTokenIDs=List[int] in python3.8.

could you please fix it?

---

## Issue #N/A: [Bug]: Is vllm still support passing max_pixels and min_pixels for Qwen2VL?

**Link**: https://github.com/vllm-project/vllm/issues/13099
**State**: closed
**Created**: 2025-02-11T16:21:43+00:00
**Closed**: 2025-02-12T11:55:25+00:00
**Comments**: 7
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```

</details>
INFO 02-12 00:04:31 __init__.py:190] Automatically detected platform cuda.
Collecting environment information...
PyTorch version: 2.5.1+cu124
Is debug build: False
CUDA used to build PyTorch: 12.4
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.3 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.10.16 (main, Dec 11 2024, 16:24:50) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-43-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration:
GPU 0: NVIDIA A800 80GB PCIe
GPU 1: NVIDIA A800 80GB PCIe
GPU 2: NVIDIA A800 80GB PCIe
GPU 3: NVIDIA A800 80GB PCIe

Nvidia driver version: 535.86.05
cuDNN ve

[... truncated for brevity ...]

---

## Issue #N/A: [Usage]: vllm inferring gemma7B is still very slow

**Link**: https://github.com/vllm-project/vllm/issues/4964
**State**: closed
**Created**: 2024-05-22T02:07:37+00:00
**Closed**: 2024-05-22T07:13:44+00:00
**Comments**: 1
**Labels**: usage

### Description

### Your current environment

why  gemma7b using vllm inference is still slow? any params to set to improve? hope your help ~~


### How would you like to use vllm

I want to run inference of a [specific model](put link here). I don't know how to integrate it with vllm.


---

## Issue #N/A: [Misc]: page attention v2

**Link**: https://github.com/vllm-project/vllm/issues/3929
**State**: closed
**Created**: 2024-04-09T07:40:04+00:00
**Closed**: 2024-04-10T06:27:48+00:00
**Comments**: 1
**Labels**: misc

### Description

### Anything you want to discuss about vllm.

Can VLLM's page attention v2 be understood as incorporating the implementation of flash decoding

---

## Issue #N/A: [Usage]: openai.APIStatusError: Error code: 405 - {'detail': 'Method Not Allowed'}

**Link**: https://github.com/vllm-project/vllm/issues/7463
**State**: closed
**Created**: 2024-08-13T07:06:53+00:00
**Closed**: 2024-08-13T15:07:20+00:00
**Comments**: 3
**Labels**: usage

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How would you like to use vllm

I run `vllm serve /mnt/datastore/shared/model-fp8 --max-model-len 16384 --tensor-parallel-size 8 --gpu-memory-utilization 0.95 --served-model-name model-v2-405b-e4`

But then I get `openai.APIStatusError: Error code: 405 - {'detail': 'Method Not Allowed'}`

I get this only with the chat.completions api from oai lib. Text completion api works fine..

---

## Issue #N/A: [Feature]: vLLM support for Granite Rapids (GNR - Intel Xeon 6th Gen)

**Link**: https://github.com/vllm-project/vllm/issues/16407
**State**: closed
**Created**: 2025-04-10T11:04:06+00:00
**Closed**: 2025-04-10T14:54:51+00:00
**Comments**: 1
**Labels**: feature request

### Description

### 🚀 The feature, motivation and pitch

Intel Xeon Granite Rapids (GNR) is the 6th Generation Intel Xeon System. I was trying to run vLLM on bare metal which is not working giving an error "No platform detected, vLLM is running on UnspecifiedPlatform". This is a new hardware platform (x86) from Intel. Not sure when the support will be provided.

### Alternatives

We can host the model through some custom code and Flask as a temporary solution. This might not get the feature of adaptive batching and scalability required. Added to that multiple Flask instances will have to be locally load balanced using NGINX.

### Additional context

vllm serve mistralai/Mistral-7B-Instruct-v0.3
INFO 04-10 10:50:47 [__init__.py:243] No platform detected, vLLM is running on UnspecifiedPlatform
INFO 04-10 10:50:48 [api_server.py:981] vLLM API server version 0.8.2
INFO 04-10 10:50:48 [api_server.py:982] args: Namespace(subparser='serve', model_tag='mistralai/Mistral-7B-Instruct-v0.3', config='', host=None

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: Try-catch conditions are incorrect to import correct  ROCm Flash Attention Backend in Draft Model

**Link**: https://github.com/vllm-project/vllm/issues/9100
**State**: closed
**Created**: 2024-10-06T01:19:09+00:00
**Closed**: 2024-10-06T05:00:05+00:00
**Comments**: 0
**Labels**: bug

### Description

### Your current environment

<details>
<summary>The output of `python collect_env.py`</summary>

```text
Your output of `python collect_env.py` here
```

</details>


### Model Input Dumps

_No response_

### 🐛 Describe the bug

I found an issue running draft model speculative decoding on AMD platform, the issue arised from  `vllm/spec_decode/draft_model_runner.py`
```
try:
    from vllm.attention.backends.flash_attn import FlashAttentionMetadata # this is throwing ImportError rather than ModuleNotFoundError
except ModuleNotFoundError:
    # vllm_flash_attn is not installed, use the identical ROCm FA metadata
    from vllm.attention.backends.rocm_flash_attn import (
        ROCmFlashAttentionMetadata as FlashAttentionMetadata)
```

Within the try-catch block `ImportError` is thrown rather than `ModuleNotFoundError`

```
  File "/home/aac/apps/rocm611-0929/vllm-fix-spec-amd/vllm/engine/multiprocessing/engine.py", line 78, in __init__                          
    

[... truncated for brevity ...]

---

## Issue #N/A: ImportError: /opt/vllm/vllm/attention_ops.cpython-39-x86_64-linux-gnu.so: undefined symbol: _ZNK3c1010TensorImpl27throw_data_ptr_access_errorEv

**Link**: https://github.com/vllm-project/vllm/issues/1291
**State**: closed
**Created**: 2023-10-08T03:15:41+00:00
**Closed**: 2023-10-08T06:36:01+00:00
**Comments**: 8

### Description

Excuse me, why is this mistake?

![image](https://github.com/vllm-project/vllm/assets/22927505/d676ff76-2745-4a74-8cbd-b3ac248e9a38)


---

## Issue #N/A: pytest tests/models fail

**Link**: https://github.com/vllm-project/vllm/issues/2115
**State**: closed
**Created**: 2023-12-14T22:39:49+00:00
**Closed**: 2023-12-15T05:00:35+00:00
**Comments**: 3

### Description

Running the following command on an H100 computer

```
pytest tests/models/
```

gave me the following errors:

```
FAILED tests/models/test_models.py::test_models[128-half-mistralai/Mistral-7B-v0.1] - AssertionError: tensor model parallel group is already initialized
FAILED tests/models/test_models.py::test_models[128-half-tiiuae/falcon-7b] - AssertionError: tensor model parallel group is already initialized
FAILED tests/models/test_models.py::test_models[128-half-gpt2] - AssertionError: tensor model parallel group is already initialized
FAILED tests/models/test_models.py::test_models[128-half-bigcode/tiny_starcoder_py] - AssertionError: tensor model parallel group is already initialized
FAILED tests/models/test_models.py::test_models[128-half-EleutherAI/gpt-j-6b] - AssertionError: tensor model parallel group is already initialized
FAILED tests/models/test_models.py::test_models[128-half-EleutherAI/pythia-70m] - AssertionError: tensor model parallel group is already init

[... truncated for brevity ...]

---

## Issue #N/A: Error:  When using OpenAI-Compatible Server, the server is available but cannot be accessed from the same terminal.

**Link**: https://github.com/vllm-project/vllm/issues/1519
**State**: closed
**Created**: 2023-10-31T09:58:04+00:00
**Closed**: 2023-10-31T19:55:42+00:00
**Comments**: 6

### Description

I'm using this perfect framework to build up an on-air api server for my local LLM. Specifically, I had run this command in my linux terminal:
`python -m vllm.entrypoints.openai.api_server    --model /home/XXX/baichuan2    --trust-remote-code    --tensor-parallel-size 1    --host 10.201.1.181    --port 8000`
Note that /baichuan2 is a directory containing the fine-tuned model derived from baichuan-inc/Baichuan2-13B-Base.

And I get:
`INFO:     Started server process [448902]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://10.201.1.181:8000 (Press CTRL+C to quit)`

To test whether the server is ready or not, I used `telnet 10.201.1.181 8000` in another shell, which returned 
`Trying 10.201.1.181...
Connected to 10.201.1.181.
Escape character is '^]'.`
So I suppose the server is running properly.

But when I attempt to call this server via python scripts as documented :
`import openai

openai.api_key

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]: The content is empty after gemma3 is deployed on the T4 graphics card to send request inference

**Link**: https://github.com/vllm-project/vllm/issues/15610
**State**: closed
**Created**: 2025-03-27T08:56:58+00:00
**Closed**: 2025-03-28T00:44:44+00:00
**Comments**: 3
**Labels**: bug

### Description

The environment is two T4 cards and the operating system is centos7
### 🐛 Describe the bug

start command

nohup vllm serve /data1/model/LLM-Research/gemma-3-4b-it --tensor-parallel-size 2 --pipeline-parallel-size 1 --max-model-len 4096 --host 0.0.0.0 --dtype float16 --port 8000 --trust-remote-code --served-model-name gemma-3-4b-it --max-num-batched-tokens 4096 --gpu-memory-utilization 0.7 > vllm.log 2>&1 &

Here is the calling procedure
```python
from openai import OpenAI
import base64

# 配置 API
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# 打开本地图片文件，并将其转换为Base64编码的字符串
with open('1.jpg', 'rb') as file:
    image = "data:image/jpeg;base64," + base64.b64encode(file.read()).decode('utf-8')

# 使用客户端与模型进行交互，发送包含图片和文本的请求
chat_response = client.chat.completions.create(
    model="gemma-3-4b-it",  # 本地模型路径或Hugging Face ID
    messages=[{
        "role": "user",
        "content": [
         

[... truncated for brevity ...]

---

## Issue #N/A: model parallelism

**Link**: https://github.com/vllm-project/vllm/issues/238
**State**: closed
**Created**: 2023-06-25T12:53:25+00:00
**Closed**: 2023-06-25T17:07:08+00:00
**Comments**: 1

### Description

I would like to ask when the model can support parallelism inference?

---

## Issue #N/A: [Usage]: Whether to support quantized LoRA

**Link**: https://github.com/vllm-project/vllm/issues/20073
**State**: closed
**Created**: 2025-06-25T11:53:42+00:00
**Closed**: 2025-06-25T14:28:16+00:00
**Comments**: 2
**Labels**: usage

### Description

### Your current environment

```text
The output of `python collect_env.py`
```


### How would you like to use vllm

I want to know whether vllm supports quantized LoRA? 
Because I don't see any entry for providing `scale` in the lora expand/shrink kernel.


### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the [documentation page](https://docs.vllm.ai/en/latest/), which can answer lots of frequently asked questions.

---

