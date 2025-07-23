# unable-reproduce - issues

**Total Issues**: 4
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 4

### Label Distribution

- unable-reproduce: 4 issues
- await-response: 2 issues

---

## Issue #N/A: [Bug] tp == 2 model gibberish

**Link**: https://github.com/sgl-project/sglang/issues/2354
**State**: closed
**Created**: 2024-12-04T22:38:41+00:00
**Closed**: 2025-01-05T04:49:50+00:00
**Comments**: 2
**Labels**: await-response, unable-reproduce

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

I've been having issues with tensor parallelism tp=2 on various Llama models. The model outputs gibberish with tp=2 but performs fine without it.

### Reproduction

#### With tensor parallelism
Terminal 1
```python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000 --host 0.0.0.0 --tp 2```


[... truncated for brevity ...]

---

## Issue #N/A: [Bug] RuntimeError in ModelTpServer

**Link**: https://github.com/sgl-project/sglang/issues/1323
**State**: closed
**Created**: 2024-09-04T08:05:29+00:00
**Closed**: 2024-09-15T15:35:24+00:00
**Comments**: 20
**Labels**: unable-reproduce

### Description

### Checklist

- [ ] 1. I have searched related issues but cannot get the expected help.
- [ ] 2. The bug has not been fixed in the latest version.
- [ ] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [ ] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [ ] 5. Please use English, otherwise it will be closed.

### Describe the bug

benchmark serving got error when --num-prompts=5
num-prompts=1 and num-prompts=3 work fine, failed when num-prompts=5
```
Exception in ModelTpServer:
Traceback (most recent call last):
  File "/data1/nfs15/nfs/bigdata/zhanglei/conda/envs/sglang-0.2.13/lib/python3.10/site-packages/sglang/srt/managers/tp_worker.py", line 218, 

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] I set `--host 0.0.0.0`, but it can't be called on another server

**Link**: https://github.com/sgl-project/sglang/issues/1121
**State**: closed
**Created**: 2024-08-16T07:57:36+00:00
**Closed**: 2024-09-22T12:45:26+00:00
**Comments**: 1
**Labels**: unable-reproduce

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 5. Please use English, otherwise it will be closed.

### Describe the bug

I use the commond:
`python -m sglang.launch_server --model-path /ldata/llms/Meta-Llama-3.1-70B-Instruct --host 0.0.0.0 --port 30000 --tp 4 --mem-fraction-static 0.95 --context-length 4096 --max-total-tokens 4096
`, I was able to call the model successfully when I ran the script on this server, but calling the model on another s

[... truncated for brevity ...]

---

## Issue #N/A: [Bug] Always Watch Dog TimeOut

**Link**: https://github.com/sgl-project/sglang/issues/1093
**State**: closed
**Created**: 2024-08-14T09:55:55+00:00
**Closed**: 2024-09-23T03:55:28+00:00
**Comments**: 5
**Labels**: await-response, unable-reproduce

### Description

### Checklist

- [X] 1. I have searched related issues but cannot get the expected help.
- [X] 2. The bug has not been fixed in the latest version.
- [X] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [X] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.

### Describe the bug

I frequently encounter Watch Dog TimeOut errors when deploying Mistral-123B using 8x A800 80G, which causes the service to stop. This issue occurs whether I send a single request or multiple requests. Below are my startup command and logs.

Command：
python -m sglang.launch_server --model-path /Mistral-Large-Instruct-2/ --host 0.0.0.0 --port 9997 --disable-cuda-graph --schedule-conservativeness

[... truncated for brevity ...]

---

