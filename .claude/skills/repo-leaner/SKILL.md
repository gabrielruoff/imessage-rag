---
name: repo-leaner
description: Review a Python repo bloated by rapid or AI-generated "vibe coded" changes and produce a detailed severity-labeled report on eliminating bloat, deduplicating code, and restructuring for AI coding agents like Claude Code — then, on request, plan and implement every finding above informational severity. Review is report-only; fixes run only after an explicitly approved plan, in an isolated git worktree, verified at the end. Use when a user wants a repo audited, cleaned up, de-bloated, reorganized, made "Claude Code friendly", or wants the audit's findings fixed. Triggers on "review my repo", "my codebase is a mess", "too much bloat", "refactor plan", "optimize for Claude Code", "clean up after merging AI changes", "implement the fixes from the report", "apply the recommendations".
---

# Repo Leaner

Two phases for keeping a Python repo — typically one bloated by many merged AI-generated changes — lean and optimal for AI coding agents.

- **Phase A · Review** produces a severity-labeled report. **It never edits code.**
- **Phase B · Fix** turns that report into an approved plan and implements it. **It edits code, but only after explicit plan approval.**

Both phases stay **moderate**: behavior-preserving restructuring and deduplication, never a rearchitecture.

## Why fresh context matters

Each stage's worker runs in a **fresh-context subagent**: the agent that wrote the review report shouldn't build the plan (it would plan around its own findings' weaknesses instead of parsing them cold), the agent that built the plan shouldn't implement it (it would defend its plan rather than execute what was actually approved), and the implementer should see only the approved plan — not unapproved review chatter or whatever unrelated conversation preceded the skill. Isolation also keeps scope creep out: each worker can only act on what's in its handoff.

**Orchestration model.** The invoking session is a thin orchestrator: it decides the phase, dispatches one fresh subagent per stage, holds all user interaction (subagents can't talk to the user — the approval gate always lives with the orchestrator), and relays results. Files on disk are the only inter-stage handoff: `REPO_REVIEW.md` (review → plan), `REPO_FIX_PLAN.md` (plan → implement), plus the user's approval/amendments passed verbatim. Dispatch with the Agent tool as a fresh general-purpose agent — **never `subagent_type: "fork"`**, which inherits the full conversation and defeats the isolation.

## Which phase am I in?

- User wants an audit / review / report, or has no report yet → **start at Phase A**.
- User has a report and wants the findings fixed / actioned / implemented → **go to Phase B**. If no report is found, offer to run Phase A first.
- User says "review and fix" → run A, deliver the report, then ask before starting B. The approval gate in B still applies.

---

## Phase A — Review (report only)

### Operating rules

1. **Report only.** In this phase, never modify, move, or delete files. The single deliverable is `REPO_REVIEW.md` at the repo root (or a user-given path). This binds the review subagent absolutely. The one exception is the orchestrator's A0 pre-flight, which clears this skill's own artifacts from previous runs and touches nothing else.
2. **Moderate scope.** Recommend consolidation, extraction of duplicated logic, reorganization, dead-code removal, dependency pruning. Never framework swaps, public-API changes, or from-scratch rewrites. Every recommendation preserves behavior.
3. **Evidence over vibes.** Every finding cites file paths, line numbers, counts, or tool output.
4. **Label severity.** Every finding gets a severity from `references/severity-model.md`. Phase B keys off these labels.
5. **Prioritize.** Rank by impact-to-effort; the top of the report names the three things to do first.

### Workflow

**A0 — Pre-flight and dispatch (orchestrator).**

*Clear stale artifacts first.* Artifacts from a previous run are indistinguishable from fresh output once a subagent — which has no conversation history — reads them, so remove them before anything else runs. In scope: `REPO_REVIEW.md` and `REPO_FIX_PLAN.md` at the repo root, plus any user-given report path. A new review invalidates an old plan too, so both go. Nothing else is a candidate: never `README.md`, `CLAUDE.md`, or anything under `docs/` — a file named like a plan is not this skill's plan.

For each artifact found, look at it before deleting it: record its provenance (`git log -1 --format='%ad %s' -- <path>` if tracked, mtime if not) and confirm it's actually this skill's output (a `# Repo Review:` heading, or the plan structure from `references/plan-format.md`). If the contents contradict the filename, leave the file alone and say so instead of deleting it. Then delete the confirmed artifacts and report in one line — `Cleared stale REPO_REVIEW.md (last commit 2026-06-30, "Lean pass: …").` — and continue. This is pre-flight, not a finding; it needs no approval and doesn't end the turn.

*Then dispatch a fresh review subagent.* The review runs in a fresh-context subagent so it judges the repo purely from its contents. Give it **only** the handoff below — do not summarize the conversation, your suspicions about the repo, or the user's commentary.

Subagent prompt template:

```
You are auditing a Python repository for bloat with no prior context — judge it
purely from the repository's contents.

Repository: <repo path>
Instructions: read <repo path>/.claude/skills/repo-leaner/SKILL.md and follow
Phase A exactly — its operating rules, steps A1–A4, and the Phase A reference
files it lists. Do not modify, move, or delete any file; your single deliverable
is the report.
Write the report to: <report path, default <repo path>/REPO_REVIEW.md>

Reply with the report path and the top 3 highest-impact actions.
```

The subagent executes A1–A4:

**A1 — Inventory.** Map the tree (`find . -name '*.py'`, `wc -l` for size outliers), identify entry points and layout, read packaging config, note test layout, and scan for AI-generation tells in `references/vibe-code-smells.md`.

**A2 — Automated analysis.** Run the tooling in `references/analysis-toolkit.md` (dead code, duplication, complexity, dependency reality, lint-as-bloat-signal, import graph). Record raw counts — they anchor the report and let the user re-measure later. If a tool can't be installed, note it and fall back to manual inspection.

**A3 — Structural assessment.** Judge against `references/target-structure.md` and `references/claude-code-fitness.md`. For each gap: current → target → the moderate move → the payoff for AI-assisted work.

**A4 — Write the report.** Use the exact structure in `references/report-template.md`, labeling every finding with severity.

**A5 — Relay (orchestrator).** When the subagent finishes, confirm the report file exists — if it wasn't written, extract the report from the subagent's final message and write the file yourself before continuing. Then tell the user the path, give the top 3 actions in chat (2–4 sentences), and ask whether they want you to proceed to implementing the fixes (Phase B).

### What "lean and AI-optimized" means

- **Small, single-purpose files** (~400–500 line ceiling; propose concrete splits).
- **Obvious boundaries** — names predict contents.
- **One source of truth** — duplication is the #1 tax on AI-merged repos.
- **A real `CLAUDE.md`** — draft one in the report.
- **Delete before you add** — prefer removing dead code/deps over new layers.

---

## Phase B — Fix (edits code, after approval)

### Hard rules

1. **Approval gate.** Plan freely, but do not modify, move, or delete any file until the user explicitly approves. Presenting the plan ends your turn — wait for a clear yes. The orchestrator enforces this gate: the implement subagent is simply never dispatched before explicit approval. Clearing a previous run's `REPO_FIX_PLAN.md` at B2 is pre-flight rather than a fix, and is outside this gate.
2. **The report is input; never delete it.** Phase A's pre-flight and Phase B's are not symmetric. In Phase B the report is what the whole phase reads from, so the only stale artifact here is `REPO_FIX_PLAN.md`. Never remove or overwrite the report in this phase — including when Phase A just produced it in this same session.
3. **Report-driven.** Fix what the report identifies. Don't invent scope; note anything extra under "Out of scope / flagged."
4. **Severity floor.** Implement every finding above **Informational**. Skip Informational unless asked.
5. **Moderate & behavior-preserving.** If a fix would change behavior or public API, surface it in the plan as behavior-sensitive and get explicit sign-off; default to leaving it out.
6. **Isolated worktree, verify at end.** All work happens in a dedicated git worktree on a new branch, so the user's current checkout and branch are never switched or disturbed. Run the full verification suite once at the end. Never leave a broken intermediate state presented as done.

### Workflow

**B1 — Locate the report (orchestrator).** Check the user-given path, else `REPO_REVIEW.md`, else search for a Markdown file with the `# Repo Review:` heading. If none is found, don't guess — offer to run Phase A first and stop.

Unless Phase A produced the report earlier in this session, state its provenance when you first mention it — last commit date and subject, or mtime if untracked. A report whose findings already shipped will otherwise be planned against silently, since the plan subagent gets no history and can't tell. This is a line of information, not a gate; B3 is where the user decides.

**B2 — Clear the stale plan, then dispatch a fresh plan subagent (orchestrator).** First, if `REPO_FIX_PLAN.md` already exists, delete it using the same show-then-delete procedure as A0 (check provenance, confirm it's this skill's plan, report the removal in one line). The report stays — see hard rule 2.

The plan is then built by a fresh-context subagent that did not write the report and parses it cold. Give it **only** the handoff below — not the review conversation, your own reading of the report, or opinions about the findings.

Subagent prompt template:

```
You are turning a repo-audit report into an implementation plan, with no prior
context — you did not write the report. Parse it cold and plan from what it says.

Repository: <repo path>
Report: <report path>
Instructions: read <repo path>/.claude/skills/repo-leaner/SKILL.md — Phase B's
hard rules bind you — plus references/severity-model.md and
references/plan-format.md. Then:

1. Extract each discrete issue with its location, severity, and recommended fix.
   Use the report's labels; for older reports without labels, assign them from
   severity-model.md and mark your assignments in the plan. Filter to above
   Informational.
2. Establish the baseline: run the project's tests in the user's checkout
   (read-only) and record pass/fail — you can't claim preserved behavior if it
   started red. The working tree needn't be clean; fixes will run in a separate
   worktree.
3. Group fixes into ordered, low-risk-first steps, each with a stated
   verification, and write the plan to <repo path>/REPO_FIX_PLAN.md using the
   exact structure in references/plan-format.md.

Do not modify, move, or delete any file except writing REPO_FIX_PLAN.md.
Reply with the plan path.
```

**B3 — Present for approval (orchestrator).** Read `REPO_FIX_PLAN.md` (if the subagent didn't write it, extract the plan from its final message and write the file yourself). Present its contents in chat and stop: "Approve as-is, tell me what to change, or list which steps to include." Don't proceed on ambiguity. This ends the turn. If the user requests changes, update `REPO_FIX_PLAN.md` to match before dispatching implementation.

**B4 — Dispatch a fresh implement subagent (orchestrator, only after explicit approval).** The approved plan is executed by a third fresh-context subagent that wrote neither the report nor the plan. Hand it only:

```
You are implementing an approved repo-cleanup plan, with no prior context — you
did not write the report or the plan. Execute the plan as approved.

Repository: <repo path>
Approved plan: <plan path>
User's approval and amendments, verbatim: <what the user said>
Instructions: read <repo path>/.claude/skills/repo-leaner/SKILL.md — Phase B's
hard rules bind you — and follow references/implementation-guide.md exactly:
create an isolated worktree on a new branch (never switch the user's current
branch), set up its environment, edit in the guide's safe order, keep a running
changelog, fix obvious breakage as you go, and run the full verification once at
the end, inside the worktree. Never commit, push, merge, or PR.

Reply with: the worktree path and branch, the base branch and base worktree
path the new worktree was created from (per the guide's pre-flight detection),
the verification results (green/red, with output), your changelog, and
`git diff --stat`. If verification is red or
you hit one of the guide's stop conditions, report exactly what failed — never
describe a red run as done.
```

**B5 — Relay (orchestrator).** Report the outcome to the user per the implementation guide: green → changelog, `git diff --stat`, worktree path and branch, and the recorded base branch/base worktree the fix branch forks from; red → never call it done. On green, **offer** to commit, push, and open a PR against the recorded base branch — and only after an explicit yes, from inside the fix worktree, commit the changes, push the branch, and run `gh pr create --base <recorded base branch>`. Never commit, push, merge, or PR without that explicit yes.

### "Done" for Phase B

The branch holds behavior-preserving fixes for every above-informational finding you committed to, the full verification passes, and the user has a changelog plus reproducible commands — including the base branch a PR from the fix branch should target. Anything you couldn't safely fix is listed explicitly.

---

## Context-isolation notes

- Keep each stage's isolation intact even when it feels inefficient. Passing your summaries, reasoning, or the conversation into a subagent's handoff defeats the purpose of the skill — the templates above are the complete handoff.
- Never dispatch a stage as `subagent_type: "fork"`; forks inherit the full conversation context.
- The orchestrator owns every user interaction (report relay, approval gate, final report). Subagents never wait on the user; anything ambiguous in a handoff means fix the handoff, not the subagent.
- If a subagent errors partway, report which stage failed and what state is on disk — don't silently redo its work inline in the orchestrator's context.

## Reference files

Read each when you reach the relevant step — don't load them all upfront.

Phase A:
- `references/vibe-code-smells.md` — AI-generation and merge-scar patterns. (A1)
- `references/analysis-toolkit.md` — exact tool commands and how to read output. (A2)
- `references/target-structure.md` — the recommended lean layout. (A3)
- `references/claude-code-fitness.md` — Claude Code checklist + what a good `CLAUDE.md` contains. (A3)
- `references/report-template.md` — the exact report structure. (A4)

Shared / Phase B:
- `references/severity-model.md` — the severity vocabulary for both labeling and filtering. (A4, B2 plan subagent)
- `references/plan-format.md` — the approval-plan structure. (B2 plan subagent)
- `references/implementation-guide.md` — safe editing order, verification discovery, stop conditions. (B4 implement subagent)
