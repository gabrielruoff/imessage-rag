# Severity Model

The shared vocabulary for both phases. In the **review phase**, label every finding with one of these levels. In the **fix phase**, implement everything rated **above Informational** (Critical, High, Medium, Low). Rate by *risk if left unfixed* × *how much it taxes future work*, not by how hard it is to fix.

If you're fixing from a report that predates severity labels (older repo-leaner output), assign levels yourself using this model and show the assignments in the plan so the user can correct them.

## Levels

**Critical** — actively harmful or likely to cause wrong behavior / data loss.
- Divergent duplicate logic where copies have silently drifted (a fix to one won't reach the others).
- Checked-in secrets (flag; do not reproduce the value).
- Circular imports that already cause runtime import errors.

**High** — strong correctness or maintainability hazard; big AI-merge tax.
- Exact duplicated logic in 2+ places (single-source-of-truth violation).
- God files mixing several unrelated responsibilities.
- Unused dependencies that pull heavy weight, or undeclared deps that break fresh installs.
- Dead code paths that mislead readers/agents into thinking they're live.

**Medium** — real friction, low correctness risk.
- Grab-bag `utils`/`misc` modules hiding domain logic.
- Files over ~500 lines that should split along a clear seam.
- Missing `CLAUDE.md`, or one that's stale/misleading.
- Inconsistent conventions with no enforced formatter/linter.

**Low** — minor cleanup, safe and cheap.
- A handful of unused imports/locals.
- Single-use trivial wrapper functions.
- Naming inconsistencies in a small area.
- Redundant constants defined twice.

**Informational** — observations, not action items. **Not implemented unless the user asks.**
- "Consider src-layout eventually."
- Style opinions with no functional impact.
- Nice-to-have future ideas marked speculative.

## Tie-breakers

- If a finding could change behavior when fixed, cap it at the level where you'd still want explicit sign-off, and mark it "behavior-sensitive" regardless of level.
- Duplication almost always rates High or Critical — it's the primary hazard in AI-merged repos.
- When unsure between two levels, pick the higher and let the user downgrade it during approval.

## Output

- In the **report**, each finding reads `**[SEVERITY]** <finding> — <location> — Fix: <action>`.
- In the **plan**, group by severity, Critical first: `SEVERITY — finding — location — fix`. The user can re-rate anything before approving.
