"""argparse CLI for evalkit: manifest | run | check | report | p3 | selfcheck.

Thin dispatcher: each submodule registers its own flags via add_parser and
sets func=<its command>. Run from the repo/worktree root:

    .venv/bin/python3 -m eval.evalkit <cmd> ...
"""

import argparse
import sys

from . import cases, checks, manifest, p3, report, runner


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
    import tool_schemas  # import-light: dict literals only

    schema_names = {d["name"] for d in tool_schemas.TOOL_DEFINITIONS}
    drift = set(cases.ALL_TOOLS) ^ schema_names
    if drift:
        print(
            "selfcheck FAILED: cases.ALL_TOOLS out of sync with "
            f"tool_schemas.TOOL_DEFINITIONS: {', '.join(sorted(drift))}"
        )
        return 1
    print(
        f"selfcheck ok: {len(suites)} suites, {n_cases} cases, "
        f"{len(recipes.RECIPES)} recipes registered, "
        f"{len(schema_names)} tools in sync with tool_schemas"
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="eval.evalkit", description="analytics-eval framework (see eval/README.md)"
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    for mod in (manifest, runner, checks, report, p3):
        mod.add_parser(subparsers)
    sc = subparsers.add_parser(
        "selfcheck", help="validate suite files, recipe registry, roster placeholders"
    )
    sc.set_defaults(func=cmd_selfcheck)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
