---
name: repo-leaner
description: Review a Python repo bloated by rapid or AI-generated "vibe coded" changes and produce a detailed severity-labeled report on eliminating bloat, deduplicating code, and restructuring for AI coding agents like Claude Code — then, on request, plan and implement every finding above informational severity. Review is report-only; fixes run only after an explicitly approved plan, in an isolated git worktree, verified at the end. Use when a user wants a repo audited, cleaned up, de-bloated, reorganized, made "Claude Code friendly", or wants the audit's findings fixed. Triggers on "review my repo", "my codebase is a mess", "too much bloat", "refactor plan", "optimize for Claude Code", "clean up after merging AI changes", "implement the fixes from the report", "apply the recommendations".
---

# Repo Leaner

Two phases for keeping a Python repo — typically one bloated by many merged AI-generated changes — lean and optimal for AI coding agents.

- **Phase A · Review** produces a severity-labeled report. **It never edits code.**
- **Phase B · Fix** turns that report into an approved plan and implements it. **It edits code, but only after explicit plan approval.**

Both phases stay **moderate**: behavior-preserving restructuring and deduplication, never a rearchitecture.

## Which phase am I in?

- User wants an audit / review / report, or has no report yet → **start at Phase A**.
- User has a report and wants the findings fixed / actioned / implemented → **go to Phase B**. If no report is found, offer to run Phase A first.
- User says "review and fix" → run A, deliver the report, then ask before starting B. The approval gate in B still applies.

---

## Phase A — Review (report only)

### Operating rules

1. **Report only.** In this phase, never modify, move, or delete files. The single deliverable is `REPO_REVIEW.md` at the repo root (or a user-given path).
2. **Moderate scope.** Recommend consolidation, extraction of duplicated logic, reorganization, dead-code removal, dependency pruning. Never framework swaps, public-API changes, or from-scratch rewrites. Every recommendation preserves behavior.
3. **Evidence over vibes.** Every finding cites file paths, line numbers, counts, or tool output.
4. **Label severity.** Every finding gets a severity from `references/severity-model.md`. Phase B keys off these labels.
5. **Prioritize.** Rank by impact-to-effort; the top of the report names the three things to do first.

### Workflow

**A1 — Inventory.** Map the tree (`find . -name '*.py'`, `wc -l` for size outliers), identify entry points and layout, read packaging config, note test layout, and scan for AI-generation tells in `references/vibe-code-smells.md`.

**A2 — Automated analysis.** Run the tooling in `references/analysis-toolkit.md` (dead code, duplication, complexity, dependency reality, lint-as-bloat-signal, import graph). Record raw counts — they anchor the report and let the user re-measure later. If a tool can't be installed, note it and fall back to manual inspection.

**A3 — Structural assessment.** Judge against `references/target-structure.md` and `references/claude-code-fitness.md`. For each gap: current → target → the moderate move → the payoff for AI-assisted work.

**A4 — Write the report.** Use the exact structure in `references/report-template.md`, labeling every finding with severity. Then tell the user the path, give the top 3 actions in chat (2–4 sentences), and ask whether they want you to proceed to implementing the fixes (Phase B).

### What "lean and AI-optimized" means

- **Small, single-purpose files** (~400–500 line ceiling; propose concrete splits).
- **Obvious boundaries** — names predict contents.
- **One source of truth** — duplication is the #1 tax on AI-merged repos.
- **A real `CLAUDE.md`** — draft one in the report.
- **Delete before you add** — prefer removing dead code/deps over new layers.

---

## Phase B — Fix (edits code, after approval)

### Hard rules

1. **Approval gate.** Plan freely, but do not modify, move, or delete any file until the user explicitly approves. Presenting the plan ends your turn — wait for a clear yes.
2. **Report-driven.** Fix what the report identifies. Don't invent scope; note anything extra under "Out of scope / flagged."
3. **Severity floor.** Implement every finding above **Informational**. Skip Informational unless asked.
4. **Moderate & behavior-preserving.** If a fix would change behavior or public API, surface it in the plan as behavior-sensitive and get explicit sign-off; default to leaving it out.
5. **Isolated worktree, verify at end.** All work happens in a dedicated git worktree on a new branch, so the user's current checkout and branch are never switched or disturbed. Run the full verification suite once at the end. Never leave a broken intermediate state presented as done.

### Workflow

**B1 — Locate & parse the report.** Check the user-given path, else `REPO_REVIEW.md`, else search for a Markdown file with the `# Repo Review:` heading. If none is found, don't guess — offer to run Phase A first and stop. Extract each discrete issue with its location, severity, and recommended fix.

**B2 — Severity.** Use the report's labels. For older reports without labels, assign them with `references/severity-model.md` and show your assignments in the plan. Filter to above Informational.

**B3 — Build the plan.** Follow `references/plan-format.md`. Establish the baseline first: record whether tests currently pass in the user's checkout — you can't claim preserved behavior if it started red. (The working tree needn't be clean; the fix runs in a separate worktree, so the user's uncommitted work is left alone.) Group fixes into ordered, low-risk-first steps, each with a stated verification.

**B4 — Get approval.** Present the plan and stop: "Approve as-is, tell me what to change, or list which steps to include." Don't proceed on ambiguity. This ends the turn.

**B5 — Implement (only after approval).** Per `references/implementation-guide.md`: create an isolated worktree on a new branch (never switch the user's current branch), set up its environment, edit in the guide's safe order, keep a running changelog. Fix obvious breakage as you go, but hold the full verification for the end.

**B6 — Verify & report back.** Run the full verification once, inside the worktree, then report per the guide: green → changelog, `git diff --stat`, worktree path and branch; red → never call it done. Never commit, push, merge, or PR unless the user asks.

### "Done" for Phase B

The branch holds behavior-preserving fixes for every above-informational finding you committed to, the full verification passes, and the user has a changelog plus reproducible commands. Anything you couldn't safely fix is listed explicitly.

---

## Reference files

Read each when you reach the relevant step — don't load them all upfront.

Phase A:
- `references/vibe-code-smells.md` — AI-generation and merge-scar patterns. (A1)
- `references/analysis-toolkit.md` — exact tool commands and how to read output. (A2)
- `references/target-structure.md` — the recommended lean layout. (A3)
- `references/claude-code-fitness.md` — Claude Code checklist + what a good `CLAUDE.md` contains. (A3)
- `references/report-template.md` — the exact report structure. (A4)

Shared / Phase B:
- `references/severity-model.md` — the severity vocabulary for both labeling and filtering. (A4, B2)
- `references/plan-format.md` — the approval-plan structure. (B3)
- `references/implementation-guide.md` — safe editing order, verification discovery, stop conditions. (B5–B6)
