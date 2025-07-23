# grammar-backend - issues

**Total Issues**: 5
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 5

### Label Distribution

- grammar-backend: 5 issues
- inactive: 3 issues
- good first issue: 1 issues
- help wanted: 1 issues
- enhancement: 1 issues

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

## Issue #N/A: [Feature] Set outlines and xgrammar as addtional dependency

**Link**: https://github.com/sgl-project/sglang/issues/2549
**State**: closed
**Created**: 2024-12-23T02:35:28+00:00
**Closed**: 2025-02-22T00:16:13+00:00
**Comments**: 4
**Labels**: enhancement, inactive, grammar-backend

### Description

### Checklist

- [X] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [X] 2. Please use English, otherwise it will be closed.

### Motivation

I am trying to integrate SGLang and vllm into OpenRLHF. For the grammar backend, could we set it as additional requirements, i.e. import it when we use it? Like:

```python

def __init__():
    if use_constrained_decoding:
        if grammar_backend == "xgrammar":
            import xgrammar
            xgrammar.function()
        if grammar_backend == "outlines":
            import outlines
            outlines.function()
```

This to avoid the version conflicts with vllm.

### Related resources

No such.

---

## Issue #N/A: [BUG] Problems with jump forward decoding

**Link**: https://github.com/sgl-project/sglang/issues/2045
**State**: closed
**Created**: 2024-11-15T14:33:07+00:00
**Closed**: 2025-01-24T00:16:13+00:00
**Comments**: 2
**Labels**: inactive, grammar-backend

### Description

There are still some issues with jump forward decoding for both backends (outlines and xgrammar). The outputs w/ jump forward are different from the outputs w/o jump forward. I tested the first 10 examples in https://github.com/sgl-project/sglang/tree/main/benchmark/json_schema and found the following issues.

## Issues with Outlines 
There is an extra space " " before the colon ":" for each key in the json. You can compare the outputs below.

### The outputs of outlines w/ jumpforward
```
{"ssid" : "OfficeNetSecure", "securityProtocol" : "WPA2-Enterprise", "bandwidth" : "1300 Mbps on the 5 GHz band"}
{"/" : {"device" : "string", "mount_point" : "string", "file_system_type" : "string", "options" : "string", "dump" : "0", "pass" : "1"}}
{"campaignID" : "CAMP123456", "productID" : "PROD7891011", "startDate" : "2023-06-01", "endDate" : "2023-06-30", "discountDetails" : "15% off on all purchases"}
{"reservationID" : "AH-158394", "guestName" : "Alexander Hamilton", "reservationTim

[... truncated for brevity ...]

---

## Issue #N/A: [BUG] Jump forward w/ outlines backend slightly changes the decoding results

**Link**: https://github.com/sgl-project/sglang/issues/2025
**State**: closed
**Created**: 2024-11-13T22:44:40+00:00
**Closed**: 2025-01-15T00:16:35+00:00
**Comments**: 1
**Labels**: inactive, grammar-backend

### Description

## Observation
For a json schema, outlines w/o jump forward and outlines w/ jump forward give slightly different results. We want to understand whether it is a bug or it is expected.

outlines w/ jumpforward
`{ "name" : "Paris" , "population" : 2 }`

outlines w/o jumpforward
`{ "name": "Paris", "population": 2140000 }`

xgrammar w/ jumpforward
`{"name": "Paris", "population": 2140000}`

xgrammar w/o jumpforward
`{"name": "Paris", "population": 2140000}`

## Reproduce
```
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B
```

```
import json
import requests

json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

# JSON
response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Here is the information of the capital of Fra

[... truncated for brevity ...]

---

## Issue #N/A: [BUG] xgrammar does not follow the constraint

**Link**: https://github.com/sgl-project/sglang/issues/2017
**State**: closed
**Created**: 2024-11-12T17:30:13+00:00
**Closed**: 2024-11-28T05:46:05+00:00
**Comments**: 6
**Labels**: grammar-backend

### Description

xgrammar does not follow the integer constraint and generate a floating number for an integer filed.

## Schema
https://github.com/sgl-project/sglang/blob/1f4514601e4c6595bb7b79dd24347ad01fa1d119/test/srt/test_json_constrained.py#L30

## Error
```
======================================================================
FAIL: test_json_openai (__main__.TestJSONConstrained)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/root/sglang/test/srt/test_json_constrained.py", line 109, in test_json_openai
    assert isinstance(js_obj["population"], int), f"{js_obj=}"
AssertionError: js_obj={'name': 'Paris', 'population': 2.0}
```

## Reproduce:

branch: https://github.com/sgl-project/sglang/tree/xgrammar-fail

```
python3 test/srt/test_json_constrained.py TestJSONConstrained.test_json_openai
```


---

