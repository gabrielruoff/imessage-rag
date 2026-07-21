---
name: parallel-linear-project
description: Implement an entire Linear project by fanning out parallel coding agents, one per issue, each in its own git worktree opening a PR, with a fresh-context agent reviewing every PR before merge. Use whenever the user asks to "implement this Linear project", points you at a Linear project link or name and wants it built out, or asks you to act as project manager driving multiple issues to completion in parallel. Trigger even if they don't say the words "skill" or "worktree" — any request to build out a whole Linear project with parallel agents, per-issue PRs, and independent review belongs here.
---

# Parallel Linear Project Implementation

Drive a full Linear project to completion by acting as **project manager** over a fleet of parallel implementation and review agents. You own coordination, quality gating, and escalation — you do not personally implement every issue.

## Prerequisites

Before starting, confirm you have:
- Access to the Linear project (via Linear MCP/connector, or ask the user to paste issue details). If no Linear tool is connected, search for one and ask the user to connect it, or have them paste the issue list.
- A git repository with GitHub remote and permission to push branches / open PRs (via `gh` CLI or GitHub connector).
- The ability to spawn subagents. If subagents aren't available in this environment, tell the user this skill's parallel model can't run and offer to implement issues sequentially instead.

## Workflow

### 1. Load and analyze the project
Fetch every issue in the project. Build a dependency map: which issues are independent (parallelizable now) vs. blocked by others. Note the project is *optimized for parallel agents*, so expect most issues to be independent — but verify, don't assume. Confirm the plan with the user before fanning out, and flag anything ambiguous.

### 2. Fan out implementation agents (aggressively parallel)
Spawn one implementation agent per independent issue, as many at once as the environment allows. Each agent must:
- Create a dedicated **git worktree** for its issue (`git worktree add ../<repo>-<issue-id> -b <issue-id>-<slug>`), so agents never collide on the working tree or branch.
- Implement the issue fully within that worktree, including tests.
- Commit and push, then open a **GitHub PR** referencing the Linear issue (put the issue ID in the PR title/body so Linear links it).
- Report back: PR URL, branch, summary of changes, and any blockers discovered.

Do not merge from the implementation agent. Implementation and review are separate roles.

See `references/agent-prompts.md` for ready-to-use implementation and reviewer agent prompts.

### 3. Independent review — fresh context per PR
For each PR, spawn a **separate reviewer agent with fresh context** that did NOT write the code. The reviewer:
- Checks out the PR branch, reads the diff against the issue's acceptance criteria.
- Verifies correctness, tests pass, no scope creep, no obvious security/quality issues.
- Leaves review comments and returns a verdict: **approve**, **request changes**, or **escalate**.

Never let an agent review its own work. If a reviewer requests changes, route the PR back to an implementation agent (fresh or original) to address, then re-review before merge.

### 4. Merge and unblock
Merge only PRs that a fresh reviewer approved and whose CI is green. After merging an issue that others depended on, release the newly-unblocked issues into a new wave of implementation agents. Repeat waves until the project is complete.

### 5. Project-manager duties throughout
- Maintain a live status table: issue → agent state (todo / in-progress / in-review / changes-requested / merged) → PR link.
- Keep waves moving; don't let approved work sit unmerged or unblocked issues sit idle.
- **Alert the user immediately** on any critical issue or roadblock: merge conflicts you can't safely resolve, failing shared infrastructure, an issue that turns out to depend on a decision only the user can make, security concerns, or repeated review failures on the same PR. Surface these rather than guessing.
- Give the user periodic concise progress summaries (X of N merged, Y in review, blockers).

### 6. Wrap up
When all issues are merged, report final status: every issue, its PR, and merge state. Note anything deferred or any Linear issues you couldn't close, and confirm the project is complete.

## Guardrails
- Merging a PR and pushing to shared branches are consequential — follow the environment's confirmation norms and the user's stated autonomy level before merging.
- If the repo has no CI, tell the user; reviewer agents become the only quality gate and should be extra thorough.
- Keep worktrees clean: remove them (`git worktree remove`) after their branch is merged.
