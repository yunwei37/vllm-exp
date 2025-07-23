# devops - issues

**Total Issues**: 5
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 2
- Closed Issues: 3

### Label Distribution

- devops: 5 issues
- help wanted: 3 issues
- enhancement: 2 issues
- documentation: 1 issues
- low severity: 1 issues
- bug: 1 issues
- bug-unconfirmed: 1 issues

---

## Issue #N/A: CI: build-linux-cross failing

**Link**: https://github.com/ggml-org/llama.cpp/issues/13869
**State**: closed
**Created**: 2025-05-28T17:06:51+00:00
**Closed**: 2025-05-28T18:46:48+00:00
**Comments**: 3
**Labels**: devops

### Description

It looks like Ubuntu deleted all the non-x86 binaries?

---

## Issue #N/A: Refactor: Add CONTRIBUTING.md and/or update PR template with [no ci] tips

**Link**: https://github.com/ggml-org/llama.cpp/issues/7657
**State**: closed
**Created**: 2024-05-30T23:56:20+00:00
**Closed**: 2024-06-09T15:25:57+00:00
**Comments**: 4
**Labels**: documentation, enhancement, help wanted, devops, low severity

### Description

### Background Description

Discussion in https://github.com/ggerganov/llama.cpp/pull/7650 pointed out a need to add a CONTRIBUTING.md and maybe add a PR template to encourage contributors to add [no ci] tag to documentation only changes.

https://docs.github.com/en/actions/managing-workflow-runs/skipping-workflow-runs

(If anyone wants to tackle this, feel free to)

### Possible Refactor Approaches

Add info about

- doc only changes should have [no ci] in commit title to skip the unneeded CI checks.
- squash on merge with commit title format: "module : some commit title (`#1234`)"

---

## Issue #N/A: Is nix-publish-flake github actions broken?

**Link**: https://github.com/ggml-org/llama.cpp/issues/7536
**State**: open
**Created**: 2024-05-25T14:48:32+00:00
**Comments**: 0
**Labels**: bug, bug-unconfirmed, devops

### Description

Was writing up a status badge for the master branch when I found that `nix-publish-flake.yml` is showing up as failing.

When I looked at https://github.com/ggerganov/llama.cpp/actions/runs/8469455017 I found that this flake github action has never been working. What's going on with this and what should we do with it? Delete it?

- [![bench action status](https://github.com/ggerganov/llama.cpp/actions/workflows/bench.yml/badge.svg)](https://github.com/ggerganov/llama.cpp/actions/workflows/bench.yml)
- [![build action status](https://github.com/ggerganov/llama.cpp/actions/workflows/build.yml/badge.svg)](https://github.com/ggerganov/llama.cpp/actions/workflows/build.yml)
- [![close-issue action status](https://github.com/ggerganov/llama.cpp/actions/workflows/close-issue.yml/badge.svg)](https://github.com/ggerganov/llama.cpp/actions/workflows/close-issue.yml)
- [![code-coverage action status](https://github.com/ggerganov/llama.cpp/actions/workflows/code-coverage.yml/badge.svg)](htt

[... truncated for brevity ...]

---

## Issue #N/A: CI Docker Build Issue (intel public key needs updating?)

**Link**: https://github.com/ggml-org/llama.cpp/issues/7507
**State**: closed
**Created**: 2024-05-24T02:39:04+00:00
**Closed**: 2024-06-03T17:53:32+00:00
**Comments**: 16
**Labels**: help wanted, devops

### Description

Regarding the docker build issue in master branch it appears that https://repositories.intel.com/gpu/ubuntu is down with an access denied message

```
#7 4.701 Reading package lists...
#7 5.410 W: GPG error: https://repositories.intel.com/gpu/ubuntu jammy InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY 28DA432DAAC8BAEA
#7 5.410 E: The repository 'https://repositories.intel.com/gpu/ubuntu jammy InRelease' is not signed.
#7 ERROR: process "/bin/sh -c apt-get update &&     apt-get install -y git" did not complete successfully: exit code: 100
```

This causes this error

```
5.410 W: GPG error: https://repositories.intel.com/gpu/ubuntu jammy InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY 28DA432DAAC8BAEA
5.410 E: The repository 'https://repositories.intel.com/gpu/ubuntu jammy InRelease' is not signed.
------
main-intel.Dockerfile:6
--------------------
 

[... truncated for brevity ...]

---

## Issue #N/A: Should we add an autolabeler for PR?

**Link**: https://github.com/ggml-org/llama.cpp/issues/7174
**State**: open
**Created**: 2024-05-09T11:25:45+00:00
**Comments**: 2
**Labels**: enhancement, help wanted, devops

### Description

https://github.com/actions/labeler

This would possibly allow use to apply general labels based on what files was changed in the PR. E.g. these are the categories we should identify

* documentation only changes
* main program UI changes
* backend logic changes
* python tooling changes

etc... identifying the language and level of complexity (via file/folder) could also be helpful as different people have different capabilities and showing what files was changed and expected expertise level could help with directing attention to what PR is easy or harder to deal with.


---

