"""argparse CLI for evalkit: manifest | run | check | report | p3 | recall* |
annotate-pilot | selfcheck.

Thin dispatcher: each submodule registers its own flags via add_parser and
sets func=<its command>. Run from the repo/worktree root:

    .venv/bin/python3 -m eval.evalkit <cmd> ...
"""

import argparse
import sys

from . import cases, checks, manifest, p3, pilot, recall, report, runner


def cmd_selfcheck(args):
    from . import recipes

    suites = cases.load_suites()  # validates schemas, ids, placeholders
    n_cases = sum(len(s["cases"]) for s in suites)
    missing = sorted(
        {
            c["gt"]["recipe"]
            for s in suites
            for c in s["cases"]
            if c.get("gt") and c["gt"]["recipe"] not in recipes.RECIPES
        }
    )
    if missing:
        print(f"selfcheck FAILED: unknown gt recipes: {', '.join(missing)}")
        return 1
    # cases.ALL_TOOLS is a hardcoded copy of the registry (kept import-light
    # for hermetic tests), so guard against drift from tool_schemas.py here.
    # ALL_TOOLS lists only the UNCONDITIONAL tools; the flag-gated tools appear in
    # TOOL_DEFINITIONS only when their flag is on, so subtract the ACTIVE gated
    # tools before diffing -- otherwise selfcheck false-fails whenever DRIFT/
    # CONFLICT_TRACE/PERSONA_SIM are enabled (e.g. the owner's own .env).
    import config
    import tool_schemas  # import-light: dict literals only

    schema_names = {d["name"] for d in tool_schemas.TOOL_DEFINITIONS}
    active_gated = {
        name
        for flag, names in tool_schemas.GATED_TOOLS.items()
        if getattr(config, flag, False)
        for name in names
    }
    drift = set(cases.ALL_TOOLS) ^ (schema_names - active_gated)
    if drift:
        print(
            "selfcheck FAILED: cases.ALL_TOOLS out of sync with "
            f"tool_schemas.TOOL_DEFINITIONS: {', '.join(sorted(drift))}"
        )
        return 1
    print(
        f"selfcheck ok: {len(suites)} suites, {n_cases} cases, "
        f"{len(recipes.RECIPES)} recipes registered, "
        f"{len(schema_names - active_gated)} unconditional tools in sync "
        f"with tool_schemas ({len(active_gated)} flag-gated active)"
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="eval.evalkit", description="analytics-eval framework (see eval/README.md)"
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    for mod in (manifest, runner, checks, report, p3, recall, pilot):
        mod.add_parser(subparsers)
    sc = subparsers.add_parser(
        "selfcheck", help="validate suite files, recipe registry, roster placeholders"
    )
    sc.set_defaults(func=cmd_selfcheck)
    # --- RES-70 (F4): eval-gate harness subcommands (label + gate) ----------
    # Additive registration -- the labeling module owns both `label` and
    # `gate`; kept out of the tuple loop above so this block stays a clean,
    # self-contained diff.
    from . import labeling

    labeling.add_parser(subparsers)
    # --- end RES-70 ---------------------------------------------------------
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
