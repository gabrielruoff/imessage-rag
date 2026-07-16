# Vibe-Code & Merge-Scar Smells

Signatures of rapidly generated or heavily AI-merged Python repos. Hunt for these in Phase 1 and cite concrete instances. Presence of a smell is a finding; a cluster of them in one file makes it a top refactor target.

## Duplication & drift (highest priority)

- **Parallel implementations**: two+ functions that do nearly the same thing under different names (`get_user`, `fetch_user`, `load_user`), often added by successive AI passes that didn't find the existing one. Grep for verb families: `get_/fetch_/load_/retrieve_`, `save_/write_/persist_/store_`.
- **Copy-pasted blocks** with a variable renamed. Duplication detectors catch these; also eyeball similarly-sized functions.
- **Divergent copies**: the same helper pasted into 3 modules, then edited in only one. Dangerous because behavior now differs silently.
- **Redundant config/constants** defined in multiple places.

## Dead & speculative code

- Unused functions, classes, imports, and whole modules never referenced.
- "Just in case" abstractions: a base class or factory with a single implementation; a plugin system with one plugin; config flags never read.
- Commented-out code blocks left as sediment.
- Unreachable branches, `if False:`, TODO stubs that `pass`.

## Over-engineering for the size

- Deep inheritance or layers of indirection wrapping trivial logic.
- Wrapper functions that only call one other function.
- Premature generalization: generic `**kwargs` pass-throughs, dynamic dispatch, metaclasses where a plain function would do.

## Structural scars

- **God files**: one module accreting unrelated responsibilities (utils.py that's 1,500 lines).
- **Grab-bag `utils`/`helpers`/`misc`** modules with no cohesion.
- Inconsistent layout: some code in `src/`, some flat, some in oddly-named dirs.
- Circular imports, or imports done inside functions to dodge cycles.
- Tests missing for the most-changed modules; or test files that duplicate large fixtures.

## Dependency bloat

- Declared dependencies never imported.
- Heavy libraries pulled in for one trivial call (e.g. `pandas` to read one CSV).
- Multiple libraries doing the same job (two HTTP clients, two date libs).
- Pinned/vendored copies of code that's also a dependency.

## Convention inconsistency (confuses future agents)

- Mixed naming (`camelCase` and `snake_case` in the same layer).
- Mixed error handling styles, mixed logging vs `print`.
- Multiple formatting styles indicating no enforced formatter.
- Docstring styles that vary file to file.

## Checked-in noise

- Generated artifacts, caches (`__pycache__`, `.pytest_cache`), build output, virtualenvs, or large data files committed to the repo.
- `.env` or secrets (flag as security note, don't reproduce contents).

## How to use

For each smell you confirm, record: the smell name, file(s)+line(s), a one-line why-it-matters, and the moderate fix. Roll these into the report's findings, grouped by theme and sorted by impact.
