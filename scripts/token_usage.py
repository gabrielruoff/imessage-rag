"""Log Claude Code token usage for this repo: `python3 scripts/token_usage.py [csv_path]`.

Reads the session transcripts Claude Code keeps in ~/.claude/projects/<this repo>/ (main sessions
and subagents, plus this repo's worktrees), sums the API `usage` blocks per (UTC day, user, model),
and merges them into a CSV (default scripts/token_usage.csv). Rows are keyed by (day, user, model),
where user is `git config user.email` (else user@host), so each person only rewrites their own
rows. The CSV is local: gitignored, not committed and not shared. Days still in the transcripts
are recomputed; days whose transcripts Claude Code has since cleaned up keep their old row, so
the CSV only grows.
"""

import csv
import getpass
import json
import re
import socket
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Claude Code names a project dir after the path with every non-alphanumeric turned into "-".
PROJECT_KEY = re.sub(r"[^A-Za-z0-9]", "-", str(REPO))
PROJECTS = Path.home() / ".claude" / "projects"
FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
HEADER = ["day", "user", "model", *FIELDS]


def whoami() -> str:
    try:
        email = subprocess.run(
            ["git", "config", "user.email"], capture_output=True, text=True, cwd=REPO
        ).stdout.strip()
    except OSError:
        email = ""
    return email or f"{getpass.getuser()}@{socket.gethostname()}"


def transcript_dirs() -> list[Path]:
    # This repo plus its worktrees (`<repo>/.claude/worktrees/<name>`); not sibling "<repo>-foo".
    worktrees = PROJECT_KEY + "--claude-worktrees-"
    return [
        d for d in PROJECTS.glob(PROJECT_KEY + "*")
        if d.name == PROJECT_KEY or d.name.startswith(worktrees)
    ]


def scan(user: str) -> dict[tuple[str, str, str], list[int]]:
    totals: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0] * len(FIELDS))
    seen: set[tuple[str, str]] = set()
    for d in transcript_dirs():
        for path in d.rglob("*.jsonl"):
            for line in path.open(encoding="utf-8", errors="replace"):
                try:
                    rec = json.loads(line)
                    msg = rec["message"]
                    usage, model, ts = msg["usage"], msg["model"], rec["timestamp"]
                except (ValueError, KeyError, TypeError):
                    continue
                if rec.get("type") != "assistant" or model == "<synthetic>":
                    continue
                # A streamed response is logged once per content block with the same usage.
                key = (str(msg.get("id")), str(rec.get("requestId")))
                if msg.get("id") and key in seen:
                    continue
                seen.add(key)
                row = totals[(ts[:10], user, model)]
                for i, f in enumerate(FIELDS):
                    row[i] += usage.get(f) or 0
    return totals


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "scripts" / "token_usage.csv"
    fresh = scan(whoami())
    # Keep every existing row; overwrite only the ones recomputed from live transcripts.
    rows: dict[tuple[str, str, str], list[int]] = {}
    if out.exists():
        with out.open(newline="") as fh:
            for r in csv.DictReader(fh):
                rows[(r["day"], r["user"], r["model"])] = [int(r[f]) for f in FIELDS]
    rows.update(fresh)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for k in sorted(rows):
            w.writerow([*k, *rows[k]])
    print(f"{out}: {len(rows)} rows ({len(fresh)} recomputed)")
    for i, f in enumerate(FIELDS):
        print(f"  {f}: {sum(v[i] for v in rows.values()):,}")


if __name__ == "__main__":
    main()
