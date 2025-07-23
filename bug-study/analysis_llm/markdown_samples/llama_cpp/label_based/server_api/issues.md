# server_api - issues

**Total Issues**: 4
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 3

### Label Distribution

- server/api: 4 issues
- server: 3 issues
- enhancement: 2 issues
- good first issue: 2 issues
- roadmap: 1 issues
- bug-unconfirmed: 1 issues
- high severity: 1 issues

---

## Issue #N/A: server : add support for multiple responses

**Link**: https://github.com/ggml-org/llama.cpp/issues/11142
**State**: open
**Created**: 2025-01-08T16:11:24+00:00
**Comments**: 2
**Labels**: server/api, server, roadmap

### Description

It would be very useful to add multi-response support per slot so that a single request would be able to generate `n` independent completions. This functionality is useful in different situations - for example, a FIM completion can provide multiple alternative suggestions at a smaller or equal compute cost compared to running them sequentially.

I think this can be implemented by adding multiple sequence id per slot (instead of having just one like we currently do). However, I am not sure how yet much complexity would be introduced to support this.

---

## Issue #N/A: Feature Request: support `"encoding_format": "base64"` in the `*/embeddings` endpoints

**Link**: https://github.com/ggml-org/llama.cpp/issues/10887
**State**: closed
**Created**: 2024-12-18T10:50:45+00:00
**Closed**: 2024-12-24T20:33:05+00:00
**Comments**: 0
**Labels**: enhancement, good first issue, server/api, server

### Description

### Prerequisites

- [X] I am running the latest code. Mention the version if possible as well.
- [X] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [X] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [X] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

The OpenAI embeddings API supports returning the embeddings in `base64` format:

https://platform.openai.com/docs/api-reference/embeddings/create#embeddings-create-encoding_format

We should implement this option in the server and enable it both for the `/v1/embeddings` and `/embeddings` endpoints.

### Motivation

Reduce JSON payload and increase OAI compatibility.

### Possible Implementation

_No response_

---

## Issue #N/A: server : temperature sampling is not working

**Link**: https://github.com/ggml-org/llama.cpp/issues/9842
**State**: closed
**Created**: 2024-10-11T07:38:07+00:00
**Closed**: 2024-10-11T07:41:07+00:00
**Comments**: 1
**Labels**: bug-unconfirmed, server/api, server, high severity

### Description

### What happened?

Using 1000000000000000 temperature does not affect model's response.
```python
import httpx

# Define the URL and the headers
url = 'http://localhost:8080/completion'
headers = {
    'Content-Type': 'application/json'
}

# Define the JSON payload with properly escaped newlines
data = {
    "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\nHow can I assist you today?<|im_end|>\n<|im_start|>user\nImplement fibbonaci in Python<|im_end|>\n<|im_start|>assistant\n",
    "n_predict": 128,
    "temperature": 1000000,
}

# Send the POST request using httpx with no timeout
response = httpx.post(url, json=data, headers=headers, timeout=None)

# Print the response from the server
print(response.json())
```

### Name and Version

d5cb86844f26f600c48bf3643738ea68138f961d

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

---

## Issue #N/A: server : support content array in OAI chat API

**Link**: https://github.com/ggml-org/llama.cpp/issues/8367
**State**: closed
**Created**: 2024-07-08T10:11:06+00:00
**Closed**: 2024-07-12T11:48:16+00:00
**Comments**: 0
**Labels**: enhancement, good first issue, server/api

### Description

According to the OpenAI API, the `"content"` field of user messages can be both `string` and `array`: https://platform.openai.com/docs/api-reference/chat/create

![image](https://github.com/ggerganov/llama.cpp/assets/1991296/62d6dc27-ca65-4eeb-80c0-5c134dbdcfb4)

So we should support requests such as:

```json
{
  "role": "user",
  "content": [ { "type": "text", "text": "tell me a joke" } ]
}
```

and

```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "msg part 0" },
    { "type": "text", "text": "msg part 1" },
    ...
    { "type": "text", "text": "msg part N" }
  ]
}
```

---

