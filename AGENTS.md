# Agent Instructions

## GitHub and PR Workflow

- The user's normal terminal has a working `gh` setup, but Codex runs in a sandbox where `gh` auth and network access may fail.
- Do not spend repeated attempts debugging `gh` authentication or GitHub network failures inside the sandbox.
- For PR requests, first inspect the worktree with `git status --short` and identify the intended scope.
- If the worktree has unrelated modified files, stage only the files that belong to the requested change. Do not use `git add -A` unless the user explicitly confirms the whole worktree is in scope.
- Prefer this workflow:
  1. Make or verify the requested local changes.
  2. Create a branch if needed.
  3. Commit only the intended files.
  4. Provide the exact `git push` and `gh pr create` commands for the user to run in their own terminal.
- If the user explicitly wants Codex to push or open the PR, request sandbox escalation for `git push` or `gh pr create` once. If that fails due to auth or network restrictions, stop and give the terminal commands instead of retrying repeatedly.
- Open PRs as drafts unless the user explicitly asks for ready-for-review.

## Repository Safety

- Assume uncommitted changes may be the user's work. Do not revert or overwrite them without explicit instruction.
- Keep documentation-only changes scoped to the relevant docs files.
- For notebook changes, avoid clearing or rewriting outputs unless the task explicitly requires it.
