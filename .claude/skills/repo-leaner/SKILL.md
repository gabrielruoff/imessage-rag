---
name: repo-leaner
description: Thoroughly review a Python repository and produce a detailed report on how to eliminate bloat, deduplicate code, and restructure the codebase so it stays lean and optimized for AI agents like Claude Code. Use this whenever a user wants a repo audited, cleaned up, de-bloated, reorganized, made "Claude Code friendly", or asks how a messy/AI-generated Python project should be structured. Triggers on phrases like "review my repo", "my codebase is a mess", "too much bloat", "refactor plan", "optimize for Claude Code", "clean up after merging a bunch of AI changes", or "how should this repo be structured". This skill produces a REPORT ONLY — it never edits code — and recommends MODERATE, behavior-preserving refactors (restructure + dedupe), not a rearchitecture.
---

# Repo Leaner

Audit a Python repo — typically one bloated by many merged AI-generated changes — and write a thorough, actionable report on how to make it lean and optimal for AI coding agents (especially Claude Code). **You produce a report. You do not edit code.** All recommendations must be **moderate**: behavior-preserving restructuring and deduplication, never a rearchitecture.

## Operating rules

1. **Report only.** Never modify, move, or delete source files. Your single deliverable is a Markdown report written to `REPO_REVIEW.md` at the repo root (or a path the user gives). If asked to apply changes, decline and point back to the report — the user reviews and applies.
2. **Moderate scope.** Recommend consolidation, extraction of duplicated logic, file/module reorganization, dead-code removal, and dependency pruning. Do **not** recommend swapping frameworks, changing public APIs, rewriting working modules from scratch, or large paradigm shifts. Every recommendation must preserve observable behavior.
3. **Evidence over vibes.** Every finding cites concrete evidence: file paths, line numbers, function names, duplication counts, import graphs, or tool output. No hand-waving.
4. **Prioritize.** Rank findings by impact-to-effort. The user should be able to read the top of the report and know the three things worth doing first.

## Workflow

Follow these phases in order. Do the analysis, THEN write the report.

### Phase 1 — Inventory

Establish ground truth before judging anything.

- Map the tree: `find . -type f -name '*.py' | head -300`, and get sizes with `wc -l` on all Python files to find outliers (very large files are prime bloat/split candidates).
- Identify the entry points, package layout, and whether it uses `src/` layout, flat layout, or ad-hoc.
- Read packaging config (`pyproject.toml`, `setup.py`, `setup.cfg`, `requirements*.txt`, `Pipfile`, `poetry.lock`) to understand declared deps and tooling.
- Note test layout and coverage of critical modules.
- Check for AI-generation tells (see `references/vibe-code-smells.md`).

### Phase 2 — Automated analysis

Run available tooling to gather hard evidence. Install into a venv if needed; if a tool isn't available and can't be installed, note it and fall back to manual inspection. See `references/analysis-toolkit.md` for exact commands and how to read output.

Gather, at minimum:
- **Dead code / unused symbols** (e.g. `vulture`, `ruff --select F401,F811`).
- **Duplication** (e.g. `pylint --disable=all --enable=duplicate-code`, or structural comparison of similar function names).
- **Complexity & size hotspots** (e.g. `radon cc -s`, `radon raw`).
- **Dependency reality**: declared-but-unused and used-but-undeclared (e.g. `deptry`, `pip-check` / `pipdeptree`).
- **Lint noise & style drift** (e.g. `ruff check`), used as a *bloat signal*, not a style nitpick.
- **Import graph** to find tangles and cycles (e.g. `pydeps`, or `grep`-based import mapping).

Record raw counts — they anchor the report and let the user re-measure after acting.

### Phase 3 — Structural assessment

Judge the repo against the target structure in `references/target-structure.md` and the Claude Code compatibility criteria in `references/claude-code-fitness.md`. For each, note the gap between current and target and the moderate move that closes it.

Key axes:
- **Bloat**: duplicated logic, dead code, over-engineered abstractions used once, vendored copies, generated artifacts checked in, unused deps.
- **Structure**: module boundaries, file sizes, cohesion, naming, layering, test placement.
- **Claude Code fitness**: presence/quality of `CLAUDE.md`, discoverability, file sizes that fit an agent's working context, clear boundaries so an agent can locate and change one thing without reading everything, and stable conventions that survive future AI edits.

### Phase 4 — Write the report

Write `REPO_REVIEW.md` using the exact template in `references/report-template.md`. Then tell the user where it is and summarize the top 3 recommendations in chat (2–4 sentences, no restating the whole report).

## What "lean and AI-optimized" means

Keep these principles front of mind; they're the standard every recommendation serves:

- **Small, single-purpose files.** An agent edits one file confidently when it fits in a modest slice of context and does one thing. Flag files that mix concerns or exceed ~400–500 lines as split candidates (with a concrete proposed split).
- **Obvious boundaries.** Directory and module names should let an agent predict where code lives without a full read. Recommend renames/moves that make the map self-evident.
- **One source of truth.** Duplicated logic is the #1 tax on AI-merged repos — the agent changes one copy and misses the others. Consolidation is almost always the highest-value recommendation.
- **A real `CLAUDE.md`.** Recommend (and draft, in the report) a concise `CLAUDE.md` capturing build/test/run commands, conventions, architecture in a paragraph, and gotchas — so future agents don't rediscover them.
- **Delete before you add.** Prefer removing dead code, unused deps, and redundant abstraction over introducing new layers.

## Reference files

Read these as you reach the relevant phase — don't load them all upfront.

- `references/vibe-code-smells.md` — signatures of rapidly/AI-generated code and merge-scar patterns to hunt for. (Phase 1)
- `references/analysis-toolkit.md` — exact tool commands, install fallbacks, and how to interpret output. (Phase 2)
- `references/target-structure.md` — the recommended lean Python repo layout with rationale. (Phase 3)
- `references/claude-code-fitness.md` — the Claude Code compatibility checklist and what a good `CLAUDE.md` contains. (Phase 3)
- `references/report-template.md` — the exact report structure to fill in. (Phase 4)
