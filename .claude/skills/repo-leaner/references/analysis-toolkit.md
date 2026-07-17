# Analysis Toolkit

Exact commands for step A2. Prefer running tools over eyeballing — they give reproducible counts the user can re-check after acting. If a tool can't be installed (no network, restricted env), note it in the report and use the manual fallback.

## Setup

Work in an isolated env so you don't pollute the target repo:

```bash
python3 -m venv /tmp/leaner-venv
/tmp/leaner-venv/bin/pip install ruff vulture radon deptry pylint pipdeptree
```

Invoke every tool by its full path (`/tmp/leaner-venv/bin/<tool>`, as the commands below do) — shell state like venv activation does not persist between tool calls. If installs fail, each section below has a `grep`/`find` fallback.

## Size & hotspots

```bash
# Line counts, largest first — split candidates float to the top
find . -name '*.py' -not -path '*/.*' -not -path '*/venv/*' | xargs wc -l | sort -rn | head -30
```

```bash
/tmp/leaner-venv/bin/radon cc . -s -a   # cyclomatic complexity per function, ranked; -a gives average
/tmp/leaner-venv/bin/radon raw . -s     # LOC, comments, blanks per file
/tmp/leaner-venv/bin/radon mi . -s      # maintainability index; low = refactor candidate
```

Interpretation: functions rated D/E/F by radon and files >400–500 lines are prime moderate-refactor targets (propose a concrete split, not a rewrite).

## Dead code / unused symbols

```bash
/tmp/leaner-venv/bin/vulture . --min-confidence 80   # unused funcs/classes/vars/imports; confidence filters false positives
/tmp/leaner-venv/bin/ruff check . --select F401,F811,F841   # unused imports, redefinitions, unused locals
```

Fallback (unused imports, rough): grep each imported name and count references.

Note: `vulture` reports false positives for dynamically-referenced names (CLI entry points, plugin registries, framework hooks). Verify before recommending deletion; flag uncertain ones as "confirm then remove".

## Duplication

```bash
/tmp/leaner-venv/bin/pylint --disable=all --enable=duplicate-code --min-similarity-lines=6 <package_dir>
```

Fallback: sort function signatures and bodies, or grep verb-family names (see vibe-code-smells). Report duplication as: the duplicated logic, every location, and the single home it should move to.

## Dependencies

```bash
/tmp/leaner-venv/bin/deptry .      # DEP001 undeclared-but-used, DEP002 declared-but-unused, DEP004 misplaced
/tmp/leaner-venv/bin/pipdeptree    # full tree; spot heavy transitive weight and duplicate-purpose libs
```

Fallback: diff the set of top-level imports against declared deps in `pyproject.toml`/`requirements.txt`.

## Lint as a bloat signal

```bash
/tmp/leaner-venv/bin/ruff check . --statistics   # aggregate rule hits; high counts of a rule = systemic issue, not one-off
```

Use the *shape* of the output (which rules dominate) to describe systemic problems, not to list every violation.

## Import graph / cycles

Cheap cycle & tangle scan (primary — the output is directly readable):

```bash
grep -rn '^\s*\(from\|import\) ' --include='*.py' . | sed 's/:.*//' | sort | uniq -c | sort -rn | head
# and look for imports inside function bodies (a cycle-avoidance smell):
grep -rn '    \(from\|import\) ' --include='*.py' .
```

Optionally, generate a graph image as an artifact *for the user* (you can't read the SVG yourself):

```bash
/tmp/leaner-venv/bin/pip install pydeps && /tmp/leaner-venv/bin/pydeps <package_dir> --max-bacon=2 --noshow -o /tmp/deps.svg
```

## Recording results

For each tool, capture the headline numbers (e.g. "vulture: 37 unused symbols across 9 files", "deptry: 4 unused deps, 2 undeclared"). These go verbatim into the report's Metrics section so progress is measurable.
