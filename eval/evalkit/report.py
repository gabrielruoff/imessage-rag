"""P4: aggregate agent-written grades + deterministic checks into
scorecard.json / scorecard.md (plan §3.8, §4.6). Pure file I/O."""

import json
import sys

from . import RESULTS_DIR, cases
from .manifest import write_json_atomic


def _suite_of(grade):
    return grade.get("suite") or grade["id"].split(".")[0]


def grade_error(grade):
    """None when the agent-written grade is usable, else what's wrong with it.
    Malformed grades are skipped (their cases stay ungraded) rather than
    crashing the report -- graders are LLM agents writing files."""
    if not isinstance(grade, dict):
        return "not a JSON object"
    if not isinstance(grade.get("id"), str) or not grade["id"]:
        return "missing string id"
    axes = grade.get("axes")
    if not isinstance(axes, dict):
        return "missing axes object"
    for axis in cases.AXES:
        if axes.get(axis) not in (0, 1):
            return f"axes.{axis} must be 0 or 1"
    return None


def _prev_axis_means(prev):
    """{suite_id: {axis: mean}} from a previous scorecard (empty when none)."""
    if not prev:
        return {}
    return {
        sid: {ax: v["mean"] for ax, v in block["axes"].items()}
        for sid, block in prev.get("suites", {}).items()
    }


def _suite_axes(scored, targets, prev_means):
    """One suite's per-axis stats (mean vs target, delta vs previous run).
    Returns (axes_out, missed): missed is True when any scored axis with a
    target came in under it -- the overall_pass input."""
    axes_out, missed = {}, False
    for axis in cases.AXES:
        target = targets.get(axis)
        mean = (
            round(sum(g["axes"][axis] for g in scored) / len(scored), 3)
            if scored else None
        )
        passed = mean is not None and target is not None and mean >= target
        if scored and target is not None and not passed:
            missed = True
        prev_mean = prev_means.get(axis)
        axes_out[axis] = {
            "mean": mean,
            "target": target,
            "pass": passed if scored else None,
            "delta_vs_prev": (
                round(mean - prev_mean, 3)
                if mean is not None and prev_mean is not None
                else None
            ),
        }
    return axes_out, missed


def _regressions(prev, suites_out):
    """(newly_failing, newly_passing) case ids vs the previous scorecard."""
    prev_failing = {
        f["id"] for s in prev.get("suites", {}).values() for f in s["failing_cases"]
    }
    cur_failing = {f["id"] for s in suites_out.values() for f in s["failing_cases"]}
    return sorted(cur_failing - prev_failing), sorted(prev_failing - cur_failing)


def _suite_failing(scored, lever_histogram):
    """A suite's failing cases (any axis 0), accumulating their levers into
    the shared histogram."""
    failing = []
    for g in sorted(scored, key=lambda g: g["id"]):
        if any(g["axes"][a] == 0 for a in cases.AXES):
            failing.append({"id": g["id"], "axes": g["axes"], "levers": g.get("levers", [])})
            for lever in g.get("levers", []):
                lever_histogram[lever] = lever_histogram.get(lever, 0) + 1
    return failing


def build_scorecard(run_doc, suites, grades, p3_block=None, prev=None, transcript_ids=(),
                    malformed_grades=()):
    """Aggregate per plan §3.8. `grades` are §3.7 dicts; blocked cases are
    excluded from axis means; every axis mean is compared to the suite target.
    overall_pass requires every transcript graded -- a run where grading was
    skipped (or a grader crashed) must not read as a pass."""
    targets_by_suite = {s["suite"]: s["targets"] for s in suites}
    by_suite = {}
    for grade in grades:
        by_suite.setdefault(_suite_of(grade), []).append(grade)
    prev_means = _prev_axis_means(prev)

    suites_out = {}
    lever_histogram = {}
    blocked_cases = []
    overall_pass = True
    for sid in sorted(set(by_suite) | set(targets_by_suite)):
        suite_grades = by_suite.get(sid, [])
        scored = [g for g in suite_grades if not g.get("blocked")]
        blocked = [g for g in suite_grades if g.get("blocked")]
        blocked_cases.extend(g["id"] for g in blocked)
        axes_out, missed = _suite_axes(
            scored, targets_by_suite.get(sid, {}), prev_means.get(sid, {}))
        if missed:
            overall_pass = False
        suites_out[sid] = {
            "n": len(suite_grades),
            "graded": len(scored),
            "blocked": len(blocked),
            "axes": axes_out,
            "failing_cases": _suite_failing(scored, lever_histogram),
        }

    graded_ids = {g["id"] for g in grades}
    ungraded = sorted(set(transcript_ids) - graded_ids)
    scorecard = {
        "run_id": run_doc.get("run_id"),
        "compared_to": prev.get("run_id") if prev else None,
        "model": run_doc.get("model"),
        "config": run_doc.get("config"),
        "corpus_fingerprint": run_doc.get("corpus_fingerprint"),
        "suites": suites_out,
        "lever_histogram": dict(sorted(lever_histogram.items(), key=lambda kv: -kv[1])),
        "run_flags": run_doc.get("flags", []),
        "p3": p3_block or {},
        "overall_pass": (
            overall_pass and not run_doc.get("flags") and bool(grades)
            and not ungraded and not malformed_grades
        ),
        "ungraded": ungraded,
        "malformed_grades": sorted(malformed_grades),
        "blocked_cases": sorted(blocked_cases),
    }
    if prev:
        scorecard["newly_failing"], scorecard["newly_passing"] = _regressions(
            prev, suites_out)
    return scorecard


def _md_header(scorecard):
    lines = [f"# Scorecard -- run `{scorecard['run_id']}`", ""]
    lines.append(f"- model: `{scorecard['model']}`")
    if scorecard["compared_to"]:
        lines.append(f"- compared to: `{scorecard['compared_to']}`")
    fp = scorecard.get("corpus_fingerprint") or {}
    lines.append(
        f"- corpus: {fp.get('message_count')} messages through {fp.get('max_message_date')}"
    )
    lines.append(f"- overall: {'PASS' if scorecard['overall_pass'] else 'FAIL'}")
    return lines


def _md_suite_table(scorecard):
    lines = ["", "## Suites", ""]
    header = "| suite | graded | blocked |"
    sep = "|---|---|---|"
    for axis in cases.AXES:
        header += f" {axis} |"
        sep += "---|"
    lines += [header, sep]
    for sid, block in scorecard["suites"].items():
        row = f"| {sid} | {block['graded']}/{block['n']} | {block['blocked']} |"
        for axis in cases.AXES:
            a = block["axes"][axis]
            if a["mean"] is None:
                row += " - |"
            else:
                mark = "ok" if a["pass"] else "**MISS**"
                delta = f", {a['delta_vs_prev']:+.3f}" if a["delta_vs_prev"] is not None else ""
                row += f" {a['mean']:.2f} vs {a['target']} ({mark}{delta}) |"
        lines.append(row)
    return lines


def _md_levers(scorecard):
    lines = ["", "## Lever histogram", ""]
    if scorecard["lever_histogram"]:
        for lever, count in scorecard["lever_histogram"].items():
            lines.append(f"- `{lever}`: {count}")
    else:
        lines.append("(no failing cases)")
    return lines


def _md_failing_cases(scorecard, evidence_by_id):
    lines = ["", "## Failing cases", ""]
    any_failing = False
    for sid, block in scorecard["suites"].items():
        for failing in block["failing_cases"]:
            any_failing = True
            grade = evidence_by_id.get(failing["id"], {})
            axes = ", ".join(f"{a}={failing['axes'][a]}" for a in cases.AXES)
            lines.append(f"- **{failing['id']}** ({axes}; levers: "
                         f"{', '.join(failing['levers']) or '-'})")
            for ev in grade.get("evidence", []):
                lines.append(f"  - [{ev.get('axis')}] {ev.get('note')} ({ev.get('ref', '')})")
    if not any_failing:
        lines.append("(none)")
    return lines


def _md_footers(scorecard):
    lines = []
    if scorecard["run_flags"]:
        lines += ["", "## Run flags", ""]
        lines += [f"- {flag}" for flag in scorecard["run_flags"]]
    if scorecard["p3"]:
        lines += ["", "## P3 (generator evals)", "", "```json",
                  json.dumps(scorecard["p3"], indent=1), "```"]
    if scorecard.get("newly_failing") or scorecard.get("newly_passing"):
        lines += ["", "## vs previous run", ""]
        lines.append(f"- newly failing: {', '.join(scorecard['newly_failing']) or '(none)'}")
        lines.append(f"- newly passing: {', '.join(scorecard['newly_passing']) or '(none)'}")
    if scorecard["ungraded"]:
        lines += ["", f"**Ungraded transcripts (block overall pass):** "
                      f"{', '.join(scorecard['ungraded'])}"]
    if scorecard["malformed_grades"]:
        lines += ["", f"**Malformed grade files (skipped; re-run their graders):** "
                      f"{', '.join(scorecard['malformed_grades'])}"]
    if scorecard["blocked_cases"]:
        lines += ["", f"**Blocked (bug) cases, excluded from means:** "
                      f"{', '.join(scorecard['blocked_cases'])}"]
    return lines


def render_markdown(scorecard, grades):
    evidence_by_id = {g["id"]: g for g in grades}
    lines = (
        _md_header(scorecard)
        + _md_suite_table(scorecard)
        + _md_levers(scorecard)
        + _md_failing_cases(scorecard, evidence_by_id)
        + _md_footers(scorecard)
    )
    return "\n".join(lines) + "\n"


def _load_run_files(run_dir):
    run_path = run_dir / "run.json"
    run_doc = json.loads(run_path.read_text()) if run_path.exists() else {}
    grades, malformed = [], []
    for path in sorted(run_dir.glob("*.grade.json")):
        try:
            with open(path) as f:
                grade = json.load(f)
        except json.JSONDecodeError as e:
            grade, error = None, f"invalid JSON: {e}"
        else:
            error = grade_error(grade)
        if error:
            malformed.append(path.name)
            print(f"WARNING: skipping {path.name}: {error} -- re-run its grader",
                  file=sys.stderr)
            continue
        grades.append(grade)
    transcript_ids = [
        p.name[: -len(".json")]
        for p in run_dir.glob("S*.json")
        if not p.name.endswith((".checks.json", ".grade.json"))
    ]
    p3_block = {}
    for path in sorted(run_dir.glob("P3.*.checks.json")):
        with open(path) as f:
            p3_block[path.name[: -len(".checks.json")]] = json.load(f)
    return run_doc, grades, malformed, transcript_ids, p3_block


def cmd_report(args):
    run_dir = RESULTS_DIR / args.run
    if not run_dir.is_dir():
        print(f"no such run dir: {run_dir}", file=sys.stderr)
        return 2
    run_doc, grades, malformed, transcript_ids, p3_block = _load_run_files(run_dir)
    prev = None
    if args.compare:
        prev_path = RESULTS_DIR / args.compare / "scorecard.json"
        if not prev_path.exists():
            print(f"no scorecard.json in previous run {args.compare}", file=sys.stderr)
            return 2
        prev = json.loads(prev_path.read_text())

    suites = cases.load_suites()
    scorecard = build_scorecard(
        run_doc, suites, grades, p3_block=p3_block, prev=prev,
        transcript_ids=transcript_ids, malformed_grades=malformed,
    )
    write_json_atomic(run_dir / "scorecard.json", scorecard)
    (run_dir / "scorecard.md").write_text(render_markdown(scorecard, grades))
    print(f"wrote {run_dir / 'scorecard.json'} and scorecard.md")
    print(f"overall: {'PASS' if scorecard['overall_pass'] else 'FAIL'}")

    if args.strict:
        faithful = all(
            block["axes"]["faithfulness"]["mean"] == 1.0
            for block in scorecard["suites"].values()
            if block["graded"]
        )
        if not (scorecard["overall_pass"] and faithful):
            return 1
    return 0


def add_parser(subparsers):
    p = subparsers.add_parser(
        "report", help="P4: aggregate grades + checks into scorecard.json/.md"
    )
    p.add_argument("--run", required=True, metavar="RUN_ID")
    p.add_argument("--compare", metavar="PREV_RUN_ID")
    p.add_argument(
        "--strict", action="store_true",
        help="exit 1 unless every target met and faithfulness == 1.0",
    )
    p.set_defaults(func=cmd_report)
