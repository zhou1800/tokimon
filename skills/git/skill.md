---
name: git
description: Perform guarded git mutations in the current checkout. Use for explicit requests to create a branch, commit confirmed paths, or merge work back with a clean fast-forward result.
---

# Git

Use the current checkout.
This skill owns three mutating actions: `branch`, `commit`, and `merge`.
Base branch means the branch this checkout was started from and should merge back into.
Use confirmed context or local worktree metadata; ask if it is unclear.

## Gate

For every mutation:

1. Inspect the relevant status, `HEAD`, branch, commits, and file set.
2. Report the exact action, target, expected result, and blockers.
3. Ask for confirmation unless an owning skill explicitly waives it for bounded automated commits.
4. Re-check before mutation; ask again if relevant state changed.
5. Run the action and verify the result.

Keep unrelated local changes outside the confirmed action.

## Bounded Automated Commits

An owning skill may waive per-commit confirmation only when the user has explicitly requested that skill's automated loop and the commit is bounded to the current checkout and current task.

The waiver does not allow merging, force-pushing, staging unrelated files, changing repository history, or mutating another branch.

## `branch`

Use before the first commit when the checkout is detached.
Branching may keep uncommitted changes.

Name the branch by priority:

1. Base branch starts with a number: reuse that number, such as `25-add-a-skill`.
2. Work mentions an issue key: start with it, such as `ENG-789-add-a-skill`.
3. Otherwise: use a short work-derived slug, such as `add-git-skill`.

Use ASCII lowercase slugs after any numeric or issue prefix.
After confirmation: create the branch, then report status.

## `commit`

Requires a named branch.
If detached, propose `branch` first.

Inspect status, staged/unstaged summaries, untracked files, and diffs as needed.
Read `.github/copilot-commit-message-instructions.md` when present.
Report included paths, excluded local changes, and the exact commit message.

After confirmation: stage only confirmed paths, commit with the exact message, then report status and the new commit.

## `merge`

Merge back into the base branch with a fast-forward merge.

Start only when:
- The current branch has committed work.
- The current checkout is clean.
- The base branch worktree is clean.

Before asking for confirmation:
- Show the current branch, base branch, commits to add, changed files, and blockers.
- Check whether the base branch can fast-forward after the current branch is rebased onto it.

After confirmation:

1. Rebase the current branch onto the base branch.
2. If rebase conflicts happen, resolve them directly, then continue with `git add` and `git rebase --continue`.
   Ask only when the correct resolution is ambiguous or risky.
3. In the base branch worktree, run `git merge --ff-only <current-branch>`.
4. Report status and the new base branch commit.

## Blockers

Stop before mutation on an unknown base branch, missing branch, divergence, or failed git commands, or on dirty state when the selected action requires a clean checkout.
Report the smallest next decision needed.
