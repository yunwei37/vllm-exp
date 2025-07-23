# very_slow_over30days - issues

**Total Issues**: 30
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 30

### Label Distribution

- inactive: 25 issues
- help wanted: 3 issues
- bug: 2 issues
- router: 2 issues
- MLLM: 1 issues
- await-response: 1 issues
- good first issue: 1 issues
- grammar-backend: 1 issues
- enhancement: 1 issues
- feature: 1 issues

---

## Issue #N/A: [Feature] Support Video for Qwen VL

**Link**: https://github.com/sgl-project/sglang/issues/4940
**State**: closed
**Created**: 2025-03-31T05:26:45+00:00
**Closed**: 2025-06-03T00:19:54+00:00
**Comments**: 2
**Labels**: inactive

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

As Video usecase is becoming more popular, would it be possible to support Video for Qwen VL series?


### Related resources

_No response_

---

## Issue #N/A: Qwen2.5 VL sglang's output much worse than transformers

**Link**: https://github.com/sgl-project/sglang/issues/3746
**State**: closed
**Created**: 2025-02-21T06:38:34+00:00
**Closed**: 2025-05-16T06:24:46+00:00
**Comments**: 17
**Labels**: MLLM

### Description

I tried serving qwen2.5 vl 72B using sglang on a node with 4*A40 GPUs.
The image I used is the official sglang:v0.4.3.post2-cu125
The command:
```bash
python3 -m sglang.launch_server \
  --tp $NUM_SHARD \
  --mem-fraction-static 0.99 \
  --disable-cuda-graph \
  --model-path /model/Qwen2.5-VL-72B-Instruct \
  --host 0.0.0.0 \
  --port 23333
```

I tested  using an internal image classification dataset, the results were much worse than when using transformers, acc droped from 87% to 80%.
And I tried another image2code task, the rendered images were much worse, too.

---

## Issue #N/A: [Bug] Offline engine performance is not better than local server when running batch

**Link**: https://github.com/sgl-project/sglang/issues/1872
**State**: closed
**Created**: 2024-11-01T16:37:25+00:00
**Closed**: 2025-01-14T00:15:49+00:00
**Comments**: 3
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I'm running some benchmarks to test using the offline engine for batch processing Llama 405B ( `sglang.Engine.generate()` ) vs. spinning up a server and running the same batch of requests locally against that live SGLang server.


### Reproduction

### Local server batch benchmark:
- First, boot up a local server with `CUDA

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Can't run Qwen2-57B-A14B-Instruct-GPTQ-Int4

**Link**: https://github.com/sgl-project/sglang/issues/1100
**State**: closed
**Created**: 2024-08-14T12:39:32+00:00
**Closed**: 2024-09-22T12:41:49+00:00
**Comments**: 4
**Labels**: await-response

### Description

### Describe the bug

I can't start sglang with model qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4, below is the error ouput.
Does sglang support it now ?

python -m sglang.launch_server --quantization gptq --model-path qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4 --port 8000 --disable-flashinfer-sampling --disable-flashinfer --tp 2 --enable-p2p-check

server_args=ServerArgs(model_path='qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4', tokenizer_path='qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4', tokenizer_mode='auto', skip_tokenizer_init=False, load_format='auto', dtype='auto', trust_remote_code=False, context_length=None, quantization='gptq', served_model_name='qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4', chat_template=None, host='127.0.0.1', port=8000, additional_ports=[8001, 8002, 8003, 8004], mem_fraction_static=0.87, max_running_requests=None, max_num_reqs=None, max_total_tokens=None, chunked_prefill_size=None, max_prefill_tokens=16384, schedule_policy='lpm', schedule_conservativeness=1.0, tp_size=2, s

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Qwen2.5-VL-AWQ dose not support concurrent request

**Link**: https://github.com/sgl-project/sglang/issues/3882
**State**: closed
**Created**: 2025-02-26T09:09:25+00:00
**Closed**: 2025-04-28T00:19:25+00:00
**Comments**: 2
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Qwen2.5-VL-AWQ dose not support concurrent request

### Reproduction

shell commond:
```
modelPathOrName=$1
tpSize=$2
dpSize=$3
portService=$4
dockerName=$5
CUDA_VISIBLE_DEVICES=$6
image="lmsysorg/sglang:v0.4.0.post2-cu124"

echo "gpus=$CUDA_VISIBLE_DEVICES"
echo model:$modelPathOrName
echo TP:$tpSize
echo "DP:$dpSize"


[ -z "$modelPathOr

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] How to load weight with torchao

**Link**: https://github.com/sgl-project/sglang/issues/2721
**State**: closed
**Created**: 2025-01-03T07:27:11+00:00
**Closed**: 2025-03-24T00:18:34+00:00
**Comments**: 13
**Labels**: inactive

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

I load 160B weight with 4*L40 GPU
python3 -m sglang.launch_server --model-path 160B_32 --tp-size 4 --trust-remote-code --disable-cuda-graph --torchao-config int8wo
but I got CUDA OOM error
What method can be used to load this model with 4 gpus, or can the torchao loading model be saved locally?

### Reproduction

python3 -m sglang.launc

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

## Issue #N/A: [Bug]ImportError: undefined symbol: cuModuleGetFunction when using lmsysorg/sglang:v0.4.1.post7-cu124

**Link**: https://github.com/sgl-project/sglang/issues/3065
**State**: closed
**Created**: 2025-01-23T04:11:37+00:00
**Closed**: 2025-03-25T00:18:12+00:00
**Comments**: 7
**Labels**: help wanted, inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

**Description:**
While using the **lmsysorg/sglang:v0.4.1.post7-cu124** Docker image to launch the server, the following error occurred:

**Error Log:**
Thu Jan 23 11:55:50 2025[1,1]<stderr>:    scheduler.event_loop_overlap()
Thu Jan 23 11:55:50 2025[1,1]<stderr>:  File "/usr/local/lib/python3.10/dist-packages/torch/utils/_contextlib.py", 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] sglang-router curl get return without `content-type: application/json` in the header

**Link**: https://github.com/sgl-project/sglang/issues/3307
**State**: closed
**Created**: 2025-02-05T03:17:00+00:00
**Closed**: 2025-04-13T00:43:11+00:00
**Comments**: 4
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Thanks for this wonderful router. We are trying it to add several sglang workers to the router and then add the router to open webui for our staff. However, we found that there is a minor issue resulting in the open webui cannot add this router (http://router:30000/v1). 

Upon checking, it seems that the sglang router would return empty `c

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Does it support AMD Strix/Strix Halo APU (gfx1150/gfx1151 RDNA 3.5)?

**Link**: https://github.com/sgl-project/sglang/issues/5131
**State**: closed
**Created**: 2025-04-07T14:09:36+00:00
**Closed**: 2025-06-14T00:18:53+00:00
**Comments**: 2
**Labels**: inactive

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Does it support AMD Strix/Strix Halo APU (gfx1150/gfx1151 RDNA 3.5)?

### Related resources

_No response_

---

## Issue #N/A: [Bug] Logprobs overflow to -3.4e+38

**Link**: https://github.com/sgl-project/sglang/issues/4876
**State**: closed
**Created**: 2025-03-29T04:53:32+00:00
**Closed**: 2025-06-03T00:19:53+00:00
**Comments**: 4
**Labels**: bug, inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

![Image](https://github.com/user-attachments/assets/b3567c95-1206-4ef5-aeea-de21ee71f0d3)

logprobs overflow to the maximum negative value of fp32

### Reproduction

I'm using Qwen2.5-14B-Instruct

command:

```python
sampling_params = {
    "temperature": 0.9,
    "top_p": 0.9,
    "skip_special_tokens": False,
    "stop": "<|im_end|>",
}

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support service discovery on Kubernetes in router

**Link**: https://github.com/sgl-project/sglang/issues/3073
**State**: closed
**Created**: 2025-01-23T07:08:03+00:00
**Closed**: 2025-03-26T00:17:50+00:00
**Comments**: 3
**Labels**: inactive, router

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

This feature proposes adding Kubernetes service discovery support to the router component. Service discovery will enable the router to dynamically identify and connect to backend services running in a Kubernetes cluster. This is particularly useful for distributed systems where backend instances may scale up or down dynamically.

## UI/UX

```bash
# New approach
python -m sglang_router.launch_router --worker-service-on-k8s default/sglang-svc
# Static approach
python -m sglang_router.launch_router --worker-urls http://worker_url_1 http://worker_url_2
```

## Pseudo code

```py
# Load Kubernetes configuration (e.g., from kubeconfig or in-cluster config)
load_kube_config()

# Initialize Kubernetes API client
api_clien

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Qwen3 235B-A22B stalling on startup

**Link**: https://github.com/sgl-project/sglang/issues/5950
**State**: closed
**Created**: 2025-05-01T10:18:11+00:00
**Closed**: 2025-07-02T00:19:36+00:00
**Comments**: 2
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

Weights don't seem to download, stalls after this:
```
[2025-05-01 10:08:11 TP7] Detected fp8 checkpoint. Please note that the format is experimental and subject to change.
[2025-05-01 10:08:11 TP5] Detected fp8 checkpoint. Please note that the format is experimental and subject to change.
[2025-05-01 10:08:11 TP1] Detected fp8 checkpoint.

[... truncated for brevity ...]

---

## Issue #N/A: Using sglang without server

**Link**: https://github.com/sgl-project/sglang/issues/218
**State**: closed
**Created**: 2024-02-22T18:19:17+00:00
**Closed**: 2024-07-25T06:32:15+00:00
**Comments**: 2
**Labels**: inactive

### Description

I'd like to make batched requests using your inference engine without needing to start up a server. Is that possible? For me, the server is a lot of extra complexity to manage, when all I really want to do is something like:

```python
def run_inferences(images_and_prompts: list[tuple], batch_size) -> list[str]
   ...
```

---

## Issue #N/A: [Bug][minimal reproducible demo] High variability across batch inference runs

**Link**: https://github.com/sgl-project/sglang/issues/1729
**State**: closed
**Created**: 2024-10-20T14:07:03+00:00
**Closed**: 2025-02-25T00:17:03+00:00
**Comments**: 12
**Labels**: bug, inactive

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

## Background

This bug might be related to #1316.

When asking the model a block of questions it should answer with `yes` followed by a block of questions that should be answered by `no` a degradation in quality can be observed for some runs, when running the same data many times.

## Standard `lmsysorg/sglang:v0.3.3.post1

[... truncated for brevity ...]

---

## Issue #N/A: [Bug]  deploy sglang ,ds_r1 with two nodes, --dp 2,--tp16, can not find /metrics on the non_master node

**Link**: https://github.com/sgl-project/sglang/issues/5812
**State**: closed
**Created**: 2025-04-28T05:32:11+00:00
**Closed**: 2025-07-05T00:18:55+00:00
**Comments**: 3
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

hi, i deploy sglang ,ds_r1 with two nodes, --dp 2,--tp16
on the master and non master node,there are the same arguments,like port  40000, host 0.0.0.0 

while  i get response when  i curl http://0.0.0.0:40000/metrics on the master node,
i got {"detail":"Not Found"} ,the same cmd on the non master node

so ,how to use it correctly ?plz need

[... truncated for brevity ...]

---

## Issue #N/A: LLaVA model parallelism/fork bug

**Link**: https://github.com/sgl-project/sglang/issues/260
**State**: closed
**Created**: 2024-03-04T13:53:35+00:00
**Closed**: 2024-07-25T06:32:30+00:00
**Comments**: 1
**Labels**: inactive

### Description

Thanks for this wonderful work! I was trying to use parallel inference/fork in sglang for the llava 1.5 model. 

Here is my env:
```
torch 2.1.2+cu118
sglang: built from main branch (b0b722e)
```

Here is my code: 
```
"""
Usage: python3 srt_example_llava.py
"""
import sglang as sgl

@sgl.function
def image_qa(s, image_path, question):
    s += sgl.user(sgl.image(image_path)+question)
    forks = s.fork(2)
    forks+= sgl.assistant(sgl.gen("answer"))
    forks.join()


def single():
    state = image_qa.run(
        image_path="images/cat.png",
        question="What is this?",
        max_new_tokens=64)
    for out in state["answer"]:
        print(out, end="\n", flush=True)


def batch():
    states = image_qa.run_batch(
        [
            {"image_path": "images/cat.png", "question":"What is this?"},
            {"image_path": "images/dog.png", "question":"What is this?"},
        ],
        max_new_tokens=64,
        temperature=1.0,
    )

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Apply structured output sampling after reasoning steps in Reasoning models

**Link**: https://github.com/sgl-project/sglang/issues/4055
**State**: closed
**Created**: 2025-03-04T07:58:42+00:00
**Closed**: 2025-04-08T04:46:48+00:00
**Comments**: 10

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Only apply constrained sampling only in the answer for reasoning model. i.e. for DeepSeek R1 only enforce grammar inside after `</think>`
This would make Reasoning models more useful in agent workflow expecting structured output.

### Related resources

https://github.com/vllm-project/vllm/issues/12619
https://github.com/vllm-project/vllm/pull/12955

---

## Issue #N/A: [Feature] Scheduler priority

**Link**: https://github.com/sgl-project/sglang/issues/5603
**State**: closed
**Created**: 2025-04-21T15:02:26+00:00
**Closed**: 2025-06-21T00:19:40+00:00
**Comments**: 1
**Labels**: inactive

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

SGLang exposes both Batch and Chat Completions endpoints.
Is there a way for your scheduler to be priority-aware, so it prioritizes real time completions over batch completions?

This way we can keep our server near 100% utilization, falling back to batch requests when there aren't real-time requests.

### Related resources

_No response_

---

## Issue #N/A: Sequence of norm concatenation in the eh_proj input of deepseek_nextn

**Link**: https://github.com/sgl-project/sglang/issues/3798
**State**: closed
**Created**: 2025-02-24T02:42:55+00:00
**Closed**: 2025-04-26T00:17:51+00:00
**Comments**: 4
**Labels**: inactive

### Description

https://github.com/sgl-project/sglang/blob/4d2a88bdffe91168dfc73ef7e3bc9100ba96686b/python/sglang/srt/models/deepseek_nextn.py#L85-L90

As we can see the eh_proj gets the input of `[enorm,hnorm]`, which does not correspond to the description in the DeepseekV3 [paper.](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf)

![Image](https://github.com/user-attachments/assets/5b1c02a4-89fe-47c7-9687-531edbb70b0e)

I wonder why the order is swapped here.

---

## Issue #N/A: [Bug] can not exists server when raising exception on multi-node deploy

**Link**: https://github.com/sgl-project/sglang/issues/3444
**State**: closed
**Created**: 2025-02-10T01:45:25+00:00
**Closed**: 2025-04-12T00:17:44+00:00
**Comments**: 1
**Labels**: inactive

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

hi，
  I deploy deepseek-r1 on 2-H100 nodes by the doc bellow：
https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3#example-serving-with-two-h2008-nodes-and-docker

when I add param `--torch-compile-max-bs`， it causes the oom exception on one gpu
when I add param `--enable-dp-attention`，it causes another Segmentation except

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support LLaMA-3.2 finetuned with Sentence Transformers !

**Link**: https://github.com/sgl-project/sglang/issues/2131
**State**: closed
**Created**: 2024-11-23T03:04:50+00:00
**Closed**: 2025-01-23T00:16:13+00:00
**Comments**: 1
**Labels**: inactive

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Please support Sentence Tranformers embedding finetune for model likes LLaMA-3.2. Currently we got error:

**`python3 -m sglang.launch_server --model-path ./Embedding/LLaMA-3.2-1B-Constrastive-cp8600  --port=30000 --is-embedding --mem-fraction-static 0.1`**

**==> ERROR:**
`Unsupported architectures: LlamaModel
`
Thanks,
Steve


### Related resources

_No response_

---

## Issue #N/A: [Feature] Use xgrammar as default grammar backend to aviod I/O errors while using Outlines in a multi-node setting

**Link**: https://github.com/sgl-project/sglang/issues/3383
**State**: closed
**Created**: 2025-02-07T23:11:12+00:00
**Closed**: 2025-05-26T21:08:02+00:00
**Comments**: 4
**Labels**: good first issue, help wanted, grammar-backend

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

related issues:
#3375 
related discussiton:
[#vllm 4193](https://github.com/vllm-project/vllm/issues/4193)
related pr:
https://github.com/sgl-project/sglang/pull/3379

### Related resources

xGrammar stores its cache in RAM instead of disk, avoiding file system conflicts.
Cache size is small (typically <0.5MB per schema), meaning it doesn't require persistent disk storage.
xGrammar is thread-safe, ensuring it can run across multiple Slurm nodes without concurrency issues.

---

## Issue #N/A: [Feature] Can router support prometheus metrics

**Link**: https://github.com/sgl-project/sglang/issues/3393
**State**: closed
**Created**: 2025-02-08T06:42:46+00:00
**Closed**: 2025-04-28T00:19:29+00:00
**Comments**: 3
**Labels**: enhancement, inactive, feature, router

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

K8s is often used to deploy applications online. After the router module is introduced, related service indicator monitoring is also required. Therefore, similar to https://github.com/sgl-project/sglang/pull/1853 provided by the server, does it support the collection of monitoring indicators of the router?

### Related resources

_No response_

---

## Issue #N/A: Efficience of quantization

**Link**: https://github.com/sgl-project/sglang/issues/6221
**State**: closed
**Created**: 2025-05-12T08:38:03+00:00
**Closed**: 2025-07-12T00:19:44+00:00
**Comments**: 2
**Labels**: inactive

### Description

environment:
vllm: 0.8.4
sglang: 0.4.6.post2


**Serving two models**:
**Model 1 with AWQ, about 20 GB**:
python3 -m sglang.launch_server --model 
/data/qwen3_30b_a3b_awq/cognitivecomputations_Qwen3-30B-A3B-AWQ/  --trust-remote-code --quantization moe_wna16

**Model 2, the raw model, about 61 GB**:
python3 -m sglang.launch_server --model /data/qwen3_30b_a3b/Qwen_Qwen3-30B-A3B  --trust-remote-code

**Benchmark**:
python3 benchmark/gsm8k/bench_sglang.py --port 30000 --parallel 1400 --num-questions 1400

**Model 1, gsm8k, Qwen3_30B_A3B-AWQ, moe_wna16**:
Accuracy: 0.894
Invalid: 0.000
Latency: 77.718 s
Output throughput: 2089.969 token/s

**Model2, gsm8k, Qwen3_30B_A3B**:
Accuracy: 0.908
Invalid: 0.000
Latency: 50.131 s
Output throughput: 3084.839 token/s

The result shows that the accuracy is close, which is good. 
However, the throughput is much worse in the quantized version.

Any ideas why is this happening?



---

## Issue #N/A: [Feature] make the compilation of torch.compile faster

**Link**: https://github.com/sgl-project/sglang/issues/2303
**State**: closed
**Created**: 2024-12-01T12:57:17+00:00
**Closed**: 2025-01-31T00:16:29+00:00
**Comments**: 1
**Labels**: inactive

### Description

Currently, the compilation step of torch.compile is very slow. We can explore methods to reduce the compilation time.

---

## Issue #N/A: [Feature] Add support for Phi4

**Link**: https://github.com/sgl-project/sglang/issues/3090
**State**: closed
**Created**: 2025-01-23T23:48:28+00:00
**Closed**: 2025-03-26T00:17:51+00:00
**Comments**: 8
**Labels**: help wanted, inactive

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 2. Please use English, otherwise it will be closed.

### Motivation

Please add support for Phi4, it's very powerful, vllm has it already

### Related resources

_No response_

---

## Issue #N/A: Llama-3 regex generation can get stuck in infinite generation beyond max_tokens and crash server (reproduction example)

**Link**: https://github.com/sgl-project/sglang/issues/414
**State**: closed
**Created**: 2024-05-08T19:42:26+00:00
**Closed**: 2024-08-05T01:05:15+00:00
**Comments**: 5
**Labels**: inactive

### Description

Hey, I've just been trying to catch this bug for half a day...

I've done `pip install git+https://github.com/sgl-project/sglang.git@51104cd#subdirectory=python`, which is the commit where 0.1.14 was mentioned.

Launched server like this:
```
python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 42069 --host 0.0.0.0 --tp-size 1 --mem-fraction-static 0.85
```

When the script below is launched, the server will get stuck in an infinite generation loop, which is long beyond the specified `max_tokens=1024`. Then it will crash. In my app there was some CUDA device assertion error (although same problem), however, in the reproduced example below the error is `RecursionError: maximum recursion depth exceeded while calling a Python object`. This is the log of server: [logfile.txt](https://github.com/sgl-project/sglang/files/15253738/logfile.txt)

```
import sglang as sgl
import asyncio
import time

@sgl.function
def demo(s):
    s += sgl.syst

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Support for Evicting Specific KV Cache to Save GPU Memory

**Link**: https://github.com/sgl-project/sglang/issues/2510
**State**: closed
**Created**: 2024-12-18T11:54:28+00:00
**Closed**: 2025-02-17T00:17:55+00:00
**Comments**: 1
**Labels**: inactive

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

Hi, congratulations on the amazing work!

I’d like to know if there is currently a feature that allows evicting specific parts of the KV cache (i.e., KV cache of some tokens) to save GPU memory. This capability is becoming increasingly important for many use cases involving KV cache compression, such as in methods like StreamingLLM and H2O.

I noticed that a similar issue was previously raised, and it was addressed with the introduction of DoubleSparse.(#1347, #1459 ) While DoubleSparse does reduce the computational cost of attention, it doesn’t seem to explicitly support operations for evicting specific parts of the KV cache from GPU memory.

I’m curious if such functionality is achievable within the current

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] `logit_bias` may not work as expected with OpenAI API

**Link**: https://github.com/sgl-project/sglang/issues/6171
**State**: closed
**Created**: 2025-05-10T11:02:22+00:00
**Closed**: 2025-06-10T22:39:26+00:00
**Comments**: 4

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

I’m running QwQ-32B on my sglang server without any modifications.
However, I don’t want the model to perform function calling (i.e., output the `<tool_call>` and `</tool_call>` tokens).
When I try to use `logit_bias` with the OpenAI API to prevent this, it doesn’t produce the expected results.

### Reproduction

Here’s a test case to illu

[... truncated for brevity ...]

---

