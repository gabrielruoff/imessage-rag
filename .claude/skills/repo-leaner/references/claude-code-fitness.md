# Claude Code Fitness

How well an AI coding agent can navigate and safely change the repo. Assess each item in step A3, mark pass/gap, and turn every gap into a moderate recommendation. Weight this equally with structural cleanliness.

## Why this matters

An agent works best when it can (a) locate the one place a change belongs without reading the whole repo, (b) fit the relevant file in a comfortable context slice, (c) rely on stated conventions instead of guessing, and (d) verify its change with a known command. A repo bloated by AI merges usually fails all four — duplicated logic means it edits the wrong copy, god files blow its context, and undocumented conventions make each edit a coin flip.

## Checklist

**Orientation**
- [ ] `CLAUDE.md` exists at root and is current (see below).
- [ ] `README.md` states purpose and quickstart.
- [ ] Build/test/run commands are discoverable and actually work.

**Discoverability**
- [ ] Directory/module names predict their contents; an agent can guess where a feature lives.
- [ ] No grab-bag `utils`/`misc` modules hiding domain logic.
- [ ] Public API is visible from package `__init__.py` without spelunking.

**Context-fit**
- [ ] Few/no files over ~500 lines (each is a split candidate).
- [ ] Functions mostly under ~50 lines.
- [ ] One concern per module — an agent can read one file to change one thing.

**Single source of truth**
- [ ] No duplicated logic (the biggest AI-merge hazard: agent edits one copy, misses others).
- [ ] Config/constants defined once.
- [ ] Fixtures/helpers defined once.

**Convention stability**
- [ ] Enforced formatter + linter (ruff) so future AI edits don't drift style.
- [ ] Consistent naming, error handling, and logging patterns.
- [ ] Type hints on public interfaces so an agent infers contracts.

**Verifiability**
- [ ] Tests exist for critical/most-changed modules and run with one command.
- [ ] CI or a documented local check the agent can invoke after editing.

## What a good CLAUDE.md contains

Recommend one if absent; if present, assess and suggest edits. Keep it concise — it loads into the agent's context every session, so bloat here is costly. Draft a proposed `CLAUDE.md` inside the report (in a fenced block) so the user can drop it in.

A strong `CLAUDE.md` covers:
- **Commands**: install, run, test, lint, typecheck — the exact invocations.
- **Architecture in a paragraph**: the domains, where each lives, how they connect. Enough that an agent knows where to look.
- **Conventions**: naming, error handling, logging, typing, formatting — what to imitate.
- **Boundaries & gotchas**: what not to touch, known sharp edges, non-obvious constraints.
- **Where things go**: "new domain logic → `src/<pkg>/<domain>/`; shared primitives → `core/`; tests mirror the package."

Keep it under ~1–2 pages. A `CLAUDE.md` that restates the whole codebase is itself bloat.

## Output

Produce a scored checklist in the report (pass / gap per item), then fold the gaps into the prioritized recommendations. The `CLAUDE.md` draft is a concrete deliverable inside the report.
