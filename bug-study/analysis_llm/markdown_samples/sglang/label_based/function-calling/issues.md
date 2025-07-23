# function-calling - issues

**Total Issues**: 3
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 2
- Closed Issues: 1

### Label Distribution

- function-calling: 3 issues
- enhancement: 1 issues
- high priority: 1 issues
- feature: 1 issues

---

## Issue #N/A: [Bug] Function calling format issues with deepseek-ai/DeepSeek-R1-0528 on SGLang

**Link**: https://github.com/sgl-project/sglang/issues/7047
**State**: open
**Created**: 2025-06-10T10:00:37+00:00
**Comments**: 3
**Labels**: function-calling

### Description

### Checklist

- [x] 1. I have searched related issues but cannot get the expected help.
- [x] 2. The bug has not been fixed in the latest version.
- [x] 3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.
- [x] 4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 5. Please use English, otherwise it will be closed.

### Describe the bug

We have deployed deepseek-ai/DeepSeek-R”1-0528 on SGLang with following arguments:
--tool-call-parser deepseekv3 --chat-template /sgl-workspace/sglang/examples/chat_template/tool_chat_template_deepseekr1.jinja

The model provides following response for curl:
{"id":"80890216cd5e485b86061e2efa71724b","object":"chat.completion","created":1749

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] Tool Call Roadmap

**Link**: https://github.com/sgl-project/sglang/issues/6589
**State**: open
**Created**: 2025-05-25T10:29:03+00:00
**Comments**: 6
**Labels**: enhancement, high priority, feature, function-calling

### Description

### Checklist

- [ ] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

## Motivation

Add a list of issues need to resolve in tool call.

## Track for Tool Call Issues

### High Piority

Issues related to accuracy, consistency, and performance.

- [x] [Multiple Tool Call Support for MistralDetector and Qwen25Detector](https://github.com/sgl-project/sglang/issues/6589#issuecomment-2907987558)
#6597 

- [ ] [JSON Double Dumping Behavior](https://github.com/sgl-project/sglang/issues/6589#issuecomment-2907988051)

- [x] [`ToolCallItem.tool_index` not following OpenAI API](https://github.com/sgl-project/sglang/issues/6589#issuecomment-2907988438) 
#6715 
#6655 
#6678 

----

### Medium Priority

Issues that are not immediate, such as features still WIP, or needs refactor, or edge cases.

- [ ] [Tests for 

[... truncated for brevity ...]

---

## Issue #N/A: [Feature] support function call for Qwen3 models

**Link**: https://github.com/sgl-project/sglang/issues/6040
**State**: closed
**Created**: 2025-05-06T03:15:59+00:00
**Closed**: 2025-06-12T09:50:47+00:00
**Comments**: 11
**Labels**: function-calling

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

function call is a very important feature of the qwen3 model, and it is currently supported by some other frameworks (such as vllm, Ollama, etc.). We hope that sglang can also support it as soon as possible.

### Related resources

_No response_

---

