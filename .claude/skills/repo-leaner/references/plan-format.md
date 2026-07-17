# Approval Plan Format

The plan subagent writes this structure to `REPO_FIX_PLAN.md`; the orchestrator then presents the file's contents in chat and stops for approval. Keep it scannable — the user needs to say yes quickly and confidently.

```markdown
## Implementation plan for <repo>

**Source report:** <path>
**Worktree:** ../<repo>-leaner-fixes on branch repo-leaner/fixes-<date> _(isolated — your current checkout and branch stay untouched)_
**Scope:** all findings above informational — <N> issues (<c> critical, <h> high, <m> medium, <l> low)
_Severities come from the report's labels; if any were missing or you re-rated, note it here._
**Baseline:** tests currently <pass|fail|none found> in your checkout

### Findings to fix (by severity)

**Critical**
- [ ] <finding> — `<path:line>` — fix: <moderate, behavior-preserving action>

**High**
- [ ] ...

**Medium**
- [ ] ...

**Low**
- [ ] ...

### Ordered steps (low-risk first, each behavior-preserving)

1. **<step>** — touches `<files>` — verify: `<command/test>`
2. ...

### Behavior-sensitive items (need your explicit OK)
- <item> — why it could change behavior — recommended: include / leave out

### Out of scope / flagged (not fixing unless you say so)
- <thing the report didn't cover, or informational items>

### Verification at the end
I'll run: `<tests>`, `<lint>`, `<typecheck>`, `<import/smoke check>` once all steps are done.

---
**Approve as-is, tell me what to change, or list which steps to include.**
```

## Rules for the plan

- **Sequence low-risk first**: deletions of confirmed dead code and unused imports, then dedup/consolidation, then file splits/moves last (highest blast radius).
- **Bundle related edits** into one step so verification is meaningful, but keep steps small enough to reason about.
- **State verification per step** even though the full suite runs only at the end — it tells the user how each change will be checked and gives you a fallback if you need to sanity-check mid-way.
- **Never hide behavior risk.** Anything that could alter output, API, or side effects goes in the behavior-sensitive list with a recommendation, defaulting to "leave out."
- End with the explicit approval ask. When the orchestrator presents the plan, that ends its turn — implementation waits for a clear yes.
