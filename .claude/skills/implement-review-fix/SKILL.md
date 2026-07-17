---
name: implement-review-fix
description: End-to-end feature delivery pipeline - implement the requested change, commit, push, open a PR, then have a fresh-context agent review the PR with /review, and finally have a second fresh-context agent address the review findings. Use this skill whenever the user asks to "implement and open a PR", "ship this change through review", "do the full implement/review/fix loop", or otherwise wants a change taken all the way from code to a reviewed, cleaned-up pull request.
---

# Implement → Review → Fix

Take a requested change from implementation to a reviewed and corrected pull request, using fresh-context subagents for the review and fix phases so each phase gets an unbiased look at the work.

## Why fresh context matters here

The agent that wrote the code is a poor reviewer of it — it already "knows" what the code is supposed to do and will read intent instead of reality. Dispatching the review to a subagent with no conversation history forces the review to work only from the PR diff, like a real teammate would. The same logic applies to the fix phase: a fresh agent addresses the review findings on their merits rather than defending earlier decisions.

## Phase 1: Implement

1. Plan before you edit. Sketch the implementation: which files change, what the discrete units of work are, and what depends on what. Follow the repository's existing conventions (style, test layout, commit format). Run the project's tests/linters if they exist and are fast enough to be practical.

   **While writing the plan, look for opportunities to dispatch subagents to work concurrently.** If the plan decomposes into units that don't depend on each other, launch one subagent per unit in a single message (multiple Agent calls in one block) so they run in parallel instead of doing the work serially yourself.

   Decompose on **disjoint file sets with no ordering dependency**. Good candidates: several modules each implementing the same pattern; code vs. docs/tests once the interface is settled; independent research or exploration. Bad candidates — do these inline instead:
   - one unit's output defines another's interface (the second can't start until the first lands),
   - two units would edit the same file (concurrent writers conflict),
   - the unit is small enough that a fresh agent's cold-start cost exceeds the work.

   Each concurrent subagent starts cold, so give it a self-contained prompt: repo path, the exact files it owns, the conventions to follow, and an explicit instruction not to touch files outside its set. When they finish, reconcile the results — read the combined diff and confirm the pieces fit — before moving on.

   Steps 2–4 stay serial: branch, commit, push, and open the PR once, after all the concurrent work has landed.
2. Create a feature branch if you're on the default branch:
   ```bash
   git checkout -b <short-kebab-description>
   ```
3. Stage and commit only the files you changed for this task (never `git add -A` blindly — check `git status` first):
   ```bash
   git add <files>
   git commit -m "<conventional summary of the change>"
   ```
4. Push and open the PR:
   ```bash
   git push -u origin <branch>
   gh pr create --title "<title>" --body "<what changed and why, plus how it was tested>"
   ```
   Capture the PR number/URL from the output — the next phases need it.
5. Check the PR for merge conflicts against the base branch:
   ```bash
   gh pr view <number> --json mergeable,mergeStateStatus
   ```
   (`mergeable` may report `UNKNOWN` while GitHub computes it — wait a few seconds and re-check.) If it reports `CONFLICTING`, dispatch a fresh-context subagent with the Task tool to resolve the conflict — like the review and fix phases, give it **only** the handoff below, not your implementation history, so it judges each conflict on what the code says rather than defending your side of it:

   ```
   You are resolving merge conflicts on a pull request branch. You did not write
   this code — work only from what you see in the repository.

   Repository: <repo path>
   Branch: <branch>
   Base branch: <base-branch>
   PR: <PR number / URL>

   Run:
     git fetch origin
     git merge origin/<base-branch>

   Resolve each conflicted file, preferring resolutions that preserve both the
   base branch's intent and this PR's change. Then commit the merge and push.

   If a conflict is one you cannot confidently reconcile — the two sides make
   contradictory changes to the same behavior, or resolving it would require
   design decisions beyond mechanical merging — run `git merge --abort` and
   reply describing the conflicting files and why they can't be auto-resolved.
   Do not guess.
   ```

   When the subagent finishes, re-check `mergeable`. If the subagent aborted because it couldn't reconcile a conflict, stop the pipeline and ask the user how to proceed. Do not continue to the review phase with an unresolved conflict.

If `gh` is not authenticated or push fails, stop and tell the user what's blocking; don't try to work around auth problems.

## Phase 2: Review (fresh-context subagent)

Dispatch a subagent with the Task tool. Give it **only** the information below — do not summarize your implementation, your reasoning, or the conversation for it. The point is that it sees the PR cold.

Subagent prompt template:

```
You are reviewing a pull request with no prior context — judge it purely on what you can see.

Repository: <repo path>
PR: <PR number / URL>

Invoke the /review skill on this PR.

Then, separately from /review, check whether the PR keeps the repository's
documentation honest. Read README.md (and CLAUDE.md, plus any docs/ page the
diff makes stale) and compare them against the diff: commands, flags, env vars,
setup steps, schema, module maps, and feature lists that the change adds,
renames, removes, or alters must be reflected there. A stale README is a real
finding — report it as an issue with the specific file, section, and the exact
wording that is now wrong. If the docs are already accurate, say so explicitly
in your summary rather than staying silent.

Then write your complete findings to <workspace>/review-findings.md, structured as:

## Summary
## Issues (each with severity: blocker / important / minor / nit, file, line, and a concrete description)
## Documentation (README/CLAUDE.md/docs accuracy: either the specific stale passages, or an explicit "docs are up to date")
## Suggestions (optional improvements, clearly separated from issues)

Be specific enough that another engineer could fix each issue without asking questions.
```

Wait for the subagent to finish and read `review-findings.md`. If the file wasn't written, extract the findings from the subagent's final message and write the file yourself before continuing.

## Phase 3: Fix (second fresh-context subagent)

Dispatch a second Task subagent — again with no conversation history, only the handoff below:

```
You are addressing code-review feedback on an open pull request. You did not write
this code and were not part of the review — work only from the materials below.

Repository: <repo path>
Branch: <branch>
PR: <PR number / URL>
Review findings: read <workspace>/review-findings.md

For each issue:
- Fix all blocker and important issues.
- Fix minor issues and nits when the fix is safe and local.
- Treat the Documentation section as work, not commentary: update README.md (and
  CLAUDE.md / docs/) so they match what the PR actually does.
- If you disagree with a finding or it would require a large design change, do NOT
  silently skip it — note it in your summary with your reasoning.

When done: run the project's tests if present, commit your fixes to <branch> with a
message referencing the review, and push. Reply with a summary listing each finding
and how it was resolved (fixed / skipped with reason).
```

## Phase 4: Report back

After the fix agent finishes:

1. Pull the branch locally and skim the diff of the fix commit(s) for anything obviously wrong.
2. Summarize for the user: PR link, review findings by severity, what was fixed, and anything the fix agent skipped (with its reasoning) so the user can make the final call.

Do not merge the PR — that decision belongs to the user.

## Notes

- Keep each phase's context isolation intact even if it feels inefficient. Passing your own opinions to the reviewer or fixer defeats the purpose of the skill.
- Phase 1's concurrent subagents serve a different purpose than the review/fix ones: throughput, not context isolation. Give them all the context they need to do their piece well — the isolation rule above applies only to the review and fix phases, and is not a reason to avoid fanning out during implementation.
- If the review finds zero issues, skip Phase 3 and tell the user the PR came back clean.
- If anything in this pipeline fails partway (push rejected, /review unavailable, subagent errors), report exactly which phase failed and leave the repo in a clean, committed state.
