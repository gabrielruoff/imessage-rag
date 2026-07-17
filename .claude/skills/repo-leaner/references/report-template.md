# Report Template

Write `REPO_REVIEW.md` using this exact structure. Fill every section with repo-specific, evidence-backed content. Keep prose tight; the report is a working document, not an essay. Every recommendation must be moderate (behavior-preserving) and cite evidence (path:line, counts, tool output).

**Label every finding in sections 3–6 with a severity** from `references/severity-model.md` (Critical / High / Medium / Low / Informational). The fix phase implements everything above Informational, so a missing or wrong label directly changes what gets built — assign deliberately, never leave findings unlabeled.

```markdown
# Repo Review: <repo name>

_Generated <date>. Scope: report-only, moderate refactors (restructure + dedupe). No code was changed._

## 1. Executive summary

<3–5 sentences: overall health, the dominant problem, and the single biggest lever. State the target end-state in one line.>

**Top 3 actions (do these first):**
1. <action> — <impact> / <effort> — <evidence pointer>
2. ...
3. ...

**Severity tally:** Critical <n> · High <n> · Medium <n> · Low <n> · Informational <n>

## 2. Metrics snapshot

| Metric | Current | Notes |
|---|---|---|
| Python files / total LOC | | |
| Largest files (LOC) | | split candidates |
| Unused symbols (vulture) | | |
| Duplicate blocks (pylint) | | |
| Unused deps (deptry) | | |
| Undeclared deps (deptry) | | |
| Avg complexity (radon) | | |
| Files >500 LOC | | |

_Re-run the commands in the appendix after acting to measure progress._

## 3. Bloat findings

Grouped by theme, sorted by impact. Each finding leads with its severity, then: what, where (path:line), why it matters, moderate fix. Format each as:

`**[SEVERITY]** <what> — \`<path:line>\` — <why it matters> — Fix: <moderate action>`

### 3.1 Duplication
### 3.2 Dead & speculative code
### 3.3 Over-engineering
### 3.4 Dependency bloat
### 3.5 Checked-in noise

## 4. Structural assessment

Current layout vs. target. Use a before/after tree sketch. For each move: current → target → the moderate step → payoff. Order low-risk-high-value first. Label each proposed change with its severity.

### 4.1 Proposed structure
### 4.2 Files to split (with the specific split)
### 4.3 Modules to consolidate / rename / relocate

## 5. Claude Code fitness

The scored checklist (pass / gap per item) from the fitness reference, then the gaps folded into fixes. Label each gap with its severity.

### 5.1 Checklist results
### 5.2 Proposed CLAUDE.md

<A ready-to-use CLAUDE.md in a fenced code block.>

## 6. Prioritized roadmap

An ordered list the user can execute as discrete, safe PRs. Each row: step, severity, why, risk, rough effort, and how to verify (which test/command). Sequence so each step leaves the repo green.

| # | Step | Severity | Why | Risk | Effort | Verify with |
|---|---|---|---|---|---|---|
```

## Filling guidance

- **Be specific.** "Consolidate the three `parse_config` copies in `a.py:12`, `b.py:88`, `cli.py:40` into `core/config.py`" — not "reduce duplication."
- **Preserve behavior.** If a recommendation could change behavior, say so and mark it out-of-scope for this moderate pass.
- **Make it verifiable.** Every roadmap step names the test or command that confirms it's safe.
- **Draft, don't dictate.** The `CLAUDE.md` block and proposed tree are drafts for the user to adopt or adjust.
