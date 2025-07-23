# nix - issues

**Total Issues**: 4
**Generated**: 2025-07-23 11:45:14

## Summary Statistics

- Open Issues: 0
- Closed Issues: 4

### Label Distribution

- nix: 4 issues
- stale: 4 issues
- enhancement: 1 issues

---

## Issue #N/A: nix: ci: fit into the new limits

**Link**: https://github.com/ggml-org/llama.cpp/issues/6346
**State**: closed
**Created**: 2024-03-27T13:19:02+00:00
**Closed**: 2024-05-17T01:06:32+00:00
**Comments**: 4
**Labels**: nix, stale

### Description

Most (all) of the nix-build jobs are being cancelled in progress since the quotas have changed. Adjust the workflows to fit in the new limits.

Context: since https://github.com/ggerganov/llama.cpp/pull/6243 the ci jobs are grouped by refs and cancelled together. The existing "Nix CI" job wasn't prepared for this for two reasons:

- It builds _many_ variants of `llama.cpp` in a single job.
- It only pushes the results to cachix after all of the builds have ended (not sure if it does the push in the "destructor" step after the cancellation).
- PRs from forks don't have access to the repo secrets so they don't push to cachix. However, it's plausible that these could make up the majority of all jobs?
- We're running pure nix-builds, meaning we can only cache store paths (results of complete and successful builds) not e.g. intermediate object files. This provides a strong guarantee that a passing CI means the build can be reproduced locally, but this also limits how much we can reus

[... truncated for brevity ...]

---

## Issue #N/A: nix: consider moving outputs to legacyPackages for lazier evaluation

**Link**: https://github.com/ggml-org/llama.cpp/issues/5681
**State**: closed
**Created**: 2024-02-23T11:17:41+00:00
**Closed**: 2024-05-05T01:06:43+00:00
**Comments**: 5
**Labels**: nix, stale

### Description

(This issue is for a conversation concerning specifically the `flake.nix`, open here for transparency)

### The proposal

The current approach to managing multiple variants of llama (BLAS, ROCm, CUDA) is to instantiate Nixpkgs several times in a [flake-parts](https://github.com/hercules-ci/flake-parts)  [module](https://github.com/ggerganov/llama.cpp/blob/15499eb94227401bdc8875da6eb85c15d37068f7/.devops/nix/nixpkgs-instances.nix), expose these instances [as arguments](https://github.com/ggerganov/llama.cpp/blob/15499eb94227401bdc8875da6eb85c15d37068f7/.devops/nix/nixpkgs-instances.nix#L9) for other flake-parts "`perSystem`" modules, and use them to populate the flake's `packages` and `checks`: https://github.com/ggerganov/llama.cpp/blob/15499eb94227401bdc8875da6eb85c15d37068f7/flake.nix#L150-L157

This means that running simple commands like `nix flake show` will trigger evaluating all of these several nixpkgs instances, tripling the evaluation cost.


We could instead remove 

[... truncated for brevity ...]

---

## Issue #N/A: CI: nix flake update: no permission to create a PR

**Link**: https://github.com/ggml-org/llama.cpp/issues/4851
**State**: closed
**Created**: 2024-01-10T01:52:06+00:00
**Closed**: 2024-03-18T07:56:23+00:00
**Comments**: 11
**Labels**: nix, stale

### Description

The weekly workflow for updating the nixpkgs revision pinned in the `flake.lock`, introduced in https://github.com/ggerganov/llama.cpp/pull/4709, fails with insufficient permissions: https://github.com/ggerganov/llama.cpp/actions/runs/7434790471/job/20229364773. The action is attempting to open a PR with the updated lock file. The action can be configured to use a separate token when creating the PR: https://github.com/ggerganov/llama.cpp/blob/c75ca5d96f902564cbbbdd7f5cade80d53c288bb/.github/workflows/nix-flake-update.yml#L22

Side-note: for the CI/nix-build workflow to also run automatically on these PRs a personal token would be required, https://github.com/DeterminateSystems/update-flake-lock#with-a-personal-authentication-token

@ggerganov: it's preferable to keep the action; can we use another token or do you know if there's another way to allow opening PRs from the specific workflow?

---

## Issue #N/A: gguf-py not in flake.nix?

**Link**: https://github.com/ggml-org/llama.cpp/issues/3824
**State**: closed
**Created**: 2023-10-28T04:32:14+00:00
**Closed**: 2024-04-02T01:12:48+00:00
**Comments**: 2
**Labels**: enhancement, nix, stale

### Description

When I try to run convert.py from the default x86_64-linux package installed from the Nix flake, it fails with the error `ModuleNotFoundError: No module named 'gguf'`. The gguf-py documentation from its subdirectory in this repo suggests to install it with pip, which really isn't ideal at all in a Nix/NixOS setup and also defeats the purpose of installing llama.cpp from the flake.nix file in the first place.  Am I just doing something wrong, or is this program missing from the flake outputs? In which case, I think it should be added. Thanks.

---

