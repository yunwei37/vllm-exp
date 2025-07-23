# linkedin - issues

**Total Issues**: 1
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 1
- Closed Issues: 0

### Label Distribution

- high priority: 1 issues
- linkedin: 1 issues

---

## Issue #N/A: [Feature] Generative Score API

**Link**: https://github.com/sgl-project/sglang/issues/5973
**State**: open
**Created**: 2025-05-02T10:43:23+00:00
**Comments**: 6
**Labels**: high priority, linkedin

### Description

### Checklist

- [x] 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.
- [x] 2. Please use English, otherwise it will be closed.

### Motivation

Similar to the cross-encoder Score API proposed here: https://github.com/sgl-project/sglang/issues/5577

Goal is to score items "generatively" using decoder-only models.

E.g. "Given a user liked A, B, and C, will the user like this item? Please answer "yes" or "no." The item is: D"

### API
```
{
  "text_1": [
    "Given a user liked A, B, and C, will the user like this item? Please answer "yes" or "no." The item is:",
  ],  
"text_2": [
     "D",
     "E"
   ],
  "positiveToken": "yes",
  "negativeToken": "no"
}
```

Returns: 

```
{
  "scores": [
    0.874,
    0.231
  ]
}
```

### Related resources

Original idea comes from this paper: [Holistic Evaluation of Language Models](https://arxiv.org/pdf/2211.09110) w

[... truncated for brevity ...]

---

