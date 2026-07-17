# Implementation Guide

For steps B5–B6, after approval. Goal: apply the approved plan safely in an **isolated git worktree** — so the user's current checkout and branch are never touched — and prove behavior is preserved with a single end-of-run verification.

## Pre-flight

Work in a dedicated worktree, not the user's checkout. This is the default so the fix run never switches the user's branch or disturbs their in-progress work.

- **Detect your starting point first.** The session may already be inside a linked worktree (created by `EnterWorktree`, another agent, or the user). Check before creating anything:
  ```bash
  git rev-parse --git-dir          # differs from --git-common-dir ⇒ you are in a linked worktree
  git rev-parse --git-common-dir
  ```
  Either way, record the **base branch** (`git branch --show-current`) and **base worktree path** (`git rev-parse --show-toplevel`) *before* creating the new worktree — these name where the fix branch forks from, and a PR from the fix branch must target the base branch. If HEAD is detached (`--show-current` prints nothing), record the repo's default branch as the base and note the fallback in the changelog.
  Being in a worktree already is fine: do **not** `cd` back to the main checkout or refuse — running `git worktree add` from inside a linked worktree is fully supported. Just place the new worktree as a sibling of the *current* worktree directory (not inside it, and not inside the main checkout).
- The baseline test result was already recorded in the approved plan (its **Baseline** line) when the plan was built — use that as the starting state; don't re-run it. If the plan says tests were already failing, you can only promise "no *new* breakage," not green-from-red. If the plan somehow lacks a baseline, record one in the user's checkout before creating the worktree.
- Note: a worktree does **not** require a clean working tree in the user's checkout — their uncommitted changes stay put and are irrelevant to the new worktree. So there's no need to block on a dirty tree here.
- Create the worktree on a fresh branch, in a sibling directory outside the repo so it doesn't get scanned as part of the repo. If the harness provides a worktree tool (e.g. `EnterWorktree`), prefer it; otherwise:
  ```bash
  git worktree add ../<repo>-leaner-fixes -b repo-leaner/fixes-<date>
  cd ../<repo>-leaner-fixes
  ```
  If that path exists, pick another (`-fixes-2`, etc.). All subsequent edits and verification happen **inside this worktree directory**.
- **Set up the worktree's environment** before verifying: a new worktree shares git history but not the virtualenv or installed deps. Recreate what's needed — e.g. `python3 -m venv .venv && .venv/bin/pip install -e .` (or the project's documented install/`requirements`), matching how the user's checkout is set up. Invoke everything through the venv's binaries (`.venv/bin/python`, `.venv/bin/pip`, …) — shell state like venv activation does not persist between tool calls. If you can't reproduce the environment, say so up front, since it limits end-of-run verification.
- Tell the user the worktree path and branch name so they know where the work lives and that their own checkout is untouched.

Carry the recorded base branch and base worktree path in the running changelog — the final report needs them.

## Editing order (matches the plan)

1. **Confirmed dead code & unused imports** — lowest risk. Remove only what analysis/report confirmed unused; re-verify a symbol is truly unreferenced before deleting (watch for dynamic use: entry points, plugin registries, framework hooks, `__all__`).
2. **Deduplication / consolidation** — move duplicated logic to one home, update all call sites, delete the copies. This is the highest-value work; do it carefully and update every import.
3. **Config/constant unification** — one source of truth; repoint references.
4. **File splits, moves, renames** — highest blast radius, do last. When moving code, update every import path. Prefer `git mv` so history is preserved.
5. **Add/refresh `CLAUDE.md`** if the plan calls for it.

While editing, keep a running changelog (file, what changed, why) for the final summary.

## Keeping the worktree coherent

The user chose "verify at end," so don't run the full suite after every step. But do fix obvious breakage immediately — if a move breaks an import, repoint it before continuing rather than accumulating errors that make the final failure hard to localize.

## Final verification (run once, at the end)

Discover the project's real commands rather than assuming, and run them through the worktree's venv:
- **Tests**: look in `pyproject.toml`/`tox.ini`/`Makefile`/CI config; common: `.venv/bin/python -m pytest -q`.
- **Lint**: `.venv/bin/ruff check .` if configured.
- **Types**: `.venv/bin/mypy <pkg>` or `pyright` if configured.
- **Import/smoke**: `.venv/bin/python -c "import <package>"` and, if there's a CLI/entry point, invoke `--help`.

Run them; capture output.

## Reporting back

- **Green**: summarize changes grouped by type (deleted, deduped, moved, added), show `git diff --stat`, and list the exact verification commands you ran. Tell the user the worktree path and branch name, the recorded base branch and base worktree path — stated as "a PR from this branch should target `<base branch>`" — that their original checkout was never switched or modified, and that nothing was committed/pushed. Mention they can review the branch and, when satisfied, merge it and remove the worktree with `git worktree remove <path>`.
- **Red**:
  - If the failure is a mechanical consequence of your own edit (moved import, renamed symbol), fix it and re-run — bounded to a couple of attempts.
  - If it's not cleanly attributable to your change or resists a quick fix, **stop**. Report exactly what failed, what you suspect, and options (revert the offending step, hand back for the user to inspect). Don't thrash or keep mutating in the dark.
  - Never describe a red run as done.

## Stop conditions (halt and ask)

- The worktree can't be created (e.g. not a git repo, or git unavailable) — don't fall back to editing the user's checkout in place; stop and ask.
- The worktree's environment can't be reproduced well enough to run verification — proceed only if the user accepts weaker verification.
- A fix would change public API or observable behavior and wasn't approved as behavior-sensitive.
- Deleting something you can't confirm is unused.
- Verification fails in a way you can't attribute to your own mechanical change.
- The user asked to commit/push/PR — you don't do that unprompted; confirm scope first.
