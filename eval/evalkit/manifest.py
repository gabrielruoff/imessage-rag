"""P0: resolve the roster + compute ground truth into
eval/analytics_eval_manifest.json (plan §3.4, §4.3). Gitignored -- it carries
real names. Pure SQL + query_tools; never touches Ollama."""

import datetime
import functools
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from . import EVAL_DIR, MANIFEST_PATH


@functools.lru_cache(maxsize=1)
def main_checkout_db_hint():
    """The main checkout's imessage_analysis.db (worktrees hold a 0-byte
    stub), derived from git's common dir so no user path is hardcoded.
    Help/error text only -- never used to resolve the DB."""
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, cwd=EVAL_DIR, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        common = ""
    if not common:
        return "<main checkout>/imessage_analysis.db"
    return str(Path(common).parent / "imessage_analysis.db")


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_db_path(db_arg):
    """Resolve --db (env-before-import: sets TARGET_DB before config loads).
    Refuses (exit 2) when the DB is missing or a 0-byte worktree stub."""
    if db_arg:
        path = Path(db_arg).resolve()
        os.environ["TARGET_DB"] = str(path)
    else:
        import config  # first app import; reads .env/default

        path = Path(config.TARGET_DB)
    if not path.exists() or path.stat().st_size == 0:
        print(
            f"refusing to start: {path} is missing or 0 bytes (worktrees hold a stub).\n"
            f"Fix: pass --db {main_checkout_db_hint()}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return path


def fingerprint(db_path):
    con = sqlite3.connect(db_path)
    try:
        count, max_date = con.execute("SELECT COUNT(*), MAX(date) FROM messages").fetchone()
    finally:
        con.close()
    return {"message_count": count, "max_message_date": max_date, "db_path": str(db_path)}


def fingerprints_match(a, b):
    # db_path is informational (main checkout vs worktree alias same file).
    return (
        a["message_count"] == b["message_count"]
        and a["max_message_date"] == b["max_message_date"]
    )


def write_json_atomic(path, data):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, default=str)
    os.replace(tmp, path)


def load_manifest(path=None):
    with open(path or MANIFEST_PATH) as f:
        return json.load(f)


def _sidecar_path(suite_id):
    return EVAL_DIR / f"analytics_eval_manifest.{suite_id}.json"


def build_gt(suite_list, roster):
    from . import cases, recipes

    gt = {}
    for suite in suite_list:
        for case in suite["cases"]:
            if not case.get("gt"):
                continue
            spec = case["gt"]
            entry = {"recipe": spec["recipe"], "params_resolved": None, "values": {},
                     "raw": None, "status": "ok", "error": None}
            try:
                fn = recipes.RECIPES[spec["recipe"]]
                params = {
                    k: cases.resolve_ref(v, roster) for k, v in spec.get("params", {}).items()
                }
                entry["params_resolved"] = params
                out = fn(roster, **params)
                entry["values"] = out["values"]
                entry["raw"] = out.get("raw")
            except Exception as e:  # recorded, never aborts the build
                entry["status"] = "error"
                entry["error"] = f"{type(e).__name__}: {e}"
                print(f"  GT {case['id']}: ERROR {entry['error']}")
            gt[case["id"]] = entry
    return gt


def cmd_manifest(args):
    if args.merge:  # pure file I/O over the sidecars; needs no DB
        return _merge_sidecars()

    db_path = resolve_db_path(args.db)

    from . import cases
    from .roster import resolve_roster

    overrides = None
    if args.roster_overrides:
        with open(args.roster_overrides) as f:
            overrides = json.load(f)

    if args.suites and MANIFEST_PATH.exists() and overrides is None:
        roster = load_manifest()["roster"]
        print(f"roster loaded from {MANIFEST_PATH.name} (partial rebuild)")
    else:
        print("resolving roster:")
        roster = resolve_roster(db_path, overrides)

    fp = fingerprint(db_path)
    suite_list = cases.load_suites(suites=args.suites)
    gt = build_gt(suite_list, roster)
    doc = {
        "created_utc": utcnow(),
        "corpus_fingerprint": fp,
        "roster": roster,
        "gt": gt,
    }
    if args.suites:
        for suite_id in args.suites:
            part = dict(doc)
            part["gt"] = {
                cid: entry
                for cid, entry in gt.items()
                if cid.split(".")[0] == suite_id
            }
            write_json_atomic(_sidecar_path(suite_id), part)
            print(f"wrote {_sidecar_path(suite_id)} ({len(part['gt'])} GT entries)")
    else:
        write_json_atomic(MANIFEST_PATH, doc)
        errors = [cid for cid, e in gt.items() if e["status"] == "error"]
        print(
            f"wrote {MANIFEST_PATH} ({len(gt)} GT entries, "
            f"{len(errors)} errors{': ' + ', '.join(errors) if errors else ''})"
        )
    return 0


def _merge_sidecars():
    sidecars = sorted(EVAL_DIR.glob("analytics_eval_manifest.S*.json"))
    if not sidecars:
        print("no sidecar manifests to merge", file=sys.stderr)
        return 2
    if MANIFEST_PATH.exists():
        doc = load_manifest()
    else:
        with open(sidecars[0]) as f:
            doc = json.load(f)
        doc["gt"] = {}
    for path in sidecars:
        with open(path) as f:
            part = json.load(f)
        if not fingerprints_match(part["corpus_fingerprint"], doc["corpus_fingerprint"]):
            print(f"refusing to merge {path.name}: corpus fingerprint mismatch",
                  file=sys.stderr)
            return 2
        doc["gt"].update(part["gt"])
        print(f"merged {path.name} ({len(part['gt'])} GT entries)")
    doc["created_utc"] = utcnow()
    write_json_atomic(MANIFEST_PATH, doc)
    print(f"wrote {MANIFEST_PATH} ({len(doc['gt'])} GT entries)")
    return 0


def add_parser(subparsers):
    p = subparsers.add_parser(
        "manifest", help="P0: resolve roster + compute ground truth (no Ollama)"
    )
    p.add_argument("--db", help=f"analysis DB path (real corpus: {main_checkout_db_hint()})")
    p.add_argument(
        "--suite", action="append", dest="suites", metavar="Sx",
        help="scope to suite(s); writes a sidecar manifest per suite",
    )
    p.add_argument("--roster-overrides", help="JSON file of placeholder overrides")
    p.add_argument("--merge", action="store_true", help="fold sidecar manifests into the main one")
    p.set_defaults(func=cmd_manifest)
