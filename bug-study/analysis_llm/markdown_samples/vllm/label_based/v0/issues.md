# v0 - issues

**Total Issues**: 1
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 0

### Label Distribution

- bug: 1 issues
- v0: 1 issues

---

## Issue #N/A: [Bug]: Bug in LRUEvictor: priority_queue and free_table desynchronization cause error

**Link**: https://github.com/vllm-project/vllm/issues/16825
**State**: open
**Created**: 2025-04-18T08:19:48+00:00
**Comments**: 6
**Labels**: bug, v0

### Description

### Your current environment

vllm 0.7.3

### 🐛 Describe the bug

### Your current environment
vllm 0.7.3
### 🐛 Describe the bug
We encountered a bug in the LRUEvictor implementation when running VLLM (version 0.7.3) with the --preemption-mode swap flag.
The issue arises due to desynchronization between self.priority_queue and self.free_table in the remove method.
<img width="1561" alt="Image" src="https://github.com/user-attachments/assets/4d048b91-f914-43b9-89e2-5b6daf0c2012" />
Add logging to evictor.py and prefix_caching_block.py to track block additions and removals.
<img width="1259" alt="Image" src="https://github.com/user-attachments/assets/e2b876a9-bff6-4004-b591-636d77eaa64d" />
<img width="589" alt="Image" src="https://github.com/user-attachments/assets/1a72b5d7-867c-49d9-a0af-cc3229a3ed47" />
<img width="644" alt="Image" src="https://github.com/user-attachments/assets/996e42c3-5899-4d33-8532-9193eb91c3f0" />
<img width="761" alt="Image" src="https://github.com/user-attachme

[... truncated for brevity ...]

---

