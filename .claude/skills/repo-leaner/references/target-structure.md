# Target Lean Python Structure

The layout to measure the repo against in step A3. This is the *recommended* end state; because scope is moderate, recommend incremental moves toward it, not a big-bang reorg. Always adapt to the repo's real domain — don't impose folders it doesn't need.

## Recommended layout

```
repo/
├── CLAUDE.md                # agent orientation: commands, conventions, architecture, gotchas
├── README.md                # human orientation
├── pyproject.toml           # single source of deps, build, tool config (ruff, pytest, etc.)
├── .gitignore               # excludes caches, venvs, build artifacts, data
├── src/
│   └── <package>/           # src-layout: importable only when installed, avoids shadowing
│       ├── __init__.py      # thin; public API surface only
│       ├── <domain_a>/      # cohesive package per domain concern
│       │   ├── __init__.py
│       │   ├── models.py
│       │   ├── service.py
│       │   └── ...
│       ├── <domain_b>/
│       ├── core/            # shared primitives used across domains (config, errors, logging)
│       └── cli.py / app.py  # entry point(s), thin — delegate into packages
└── tests/
    ├── conftest.py          # shared fixtures, defined ONCE
    ├── test_<domain_a>.py
    └── test_<domain_b>.py
```

## Principles behind it

- **src-layout** prevents accidental imports of the uninstalled package and keeps the top level clean. If the repo is flat and small, note src-layout as a nice-to-have, not urgent.
- **One package, cohesive sub-packages by domain.** Group by *what it's about* (billing, auth, ingest), not by *what it is* (all models together, all views together) unless the framework demands otherwise. Domain grouping lets an agent change one feature in one folder.
- **`core/` for genuine cross-cutting primitives** — config, error types, logging setup. This is the *only* legitimate "shared" bucket. It is NOT a dumping ground: everything in it must be used by 2+ domains.
- **No `utils.py` / `helpers.py` / `misc.py`.** These attract unrelated code. Recommend redistributing their contents to the domain that owns each function, or into `core/` if truly cross-cutting.
- **Thin entry points.** `cli.py`/`app.py`/`main.py` parse input and delegate; business logic lives in packages. Fat entry points are a common vibe-code smell.
- **Thin `__init__.py`.** Export the public API, no logic. Deep logic in `__init__` hides it from agents scanning the tree.
- **Tests mirror the package.** One test module per domain module; fixtures defined once in `conftest.py`, never copy-pasted.
- **One config source.** `pyproject.toml` holds deps and all tool config. Flag scattered `setup.py` + `requirements.txt` + `setup.cfg` combos for consolidation.

## File-size guidance

- Aim for modules under ~400–500 lines and functions under ~50. These aren't hard rules — they're the point where an agent can no longer hold the file in a comfortable slice of context and starts making blind edits.
- When flagging an oversized file, always propose the *specific* split: which functions/classes move to which new module, and why the boundary is natural.

## How to apply

For each axis above, state: current state → target → the moderate move to get there → the payoff for AI-assisted work. Order moves so that low-risk, high-value ones (delete dead code, kill `utils.py`, consolidate config) come before riskier moves (reorganizing packages).
