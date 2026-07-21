"""The ``label`` and ``gate`` subcommands of ``python -m eval.evalkit``
(RES-70 / F4).

``label <task>``  -- samples candidate items (the task's ``sample_sql``),
presents each in the terminal, and appends the owner's label to the gitignored
``eval/labels/<task>.jsonl``. Append-only, resumable (already-labeled ids are
skipped), and Ctrl-C safe: every accepted label is flushed the instant it is
entered, mirroring annotate.py's commit-as-you-go ergonomics. This path is
run interactively by the repo owner -- agents never write to ``eval/labels/``.

``gate <task>``  -- runs the task's production classifier over the owner's
labeled items, computes kappa/precision vs the gold labels, prints a per-class
scorecard, and PERSISTS the verdict to ``ingest_state`` (source "owner",
overriding any shipped seed). Below the per-class sample floor a class can
never be marked passed.

Stdlib-only at import; config/db pulled in lazily so ``--db`` can set
TARGET_DB before config first loads (the evalkit env-before-import invariant).
"""

import json
import sys

from . import gates
from .manifest import main_checkout_db_hint, resolve_db_path, utcnow


# --- shared IO ---------------------------------------------------------------


def load_labels(labels_path):
    """Read a ``<task>.jsonl`` label file into a list of records (empty when
    the file does not exist yet)."""
    if not labels_path.exists():
        return []
    records = []
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sample_items(con, task_cfg, n, params=None):
    """Sample up to ``n`` candidate items via the task's ``sample_sql``
    (one bound ``?`` = the LIMIT). Returns [{id, text}]."""
    sql = task_cfg.get("sample_sql")
    if not sql:
        raise ValueError("task has no sample_sql (not a hand-label task)")
    rows = con.execute(sql, (n,) if params is None else params).fetchall()
    return [{"id": r["id"], "text": r["text"]} for r in rows]


# --- label -------------------------------------------------------------------


def run_label(con, task_name, task_cfg, labels_path, n, ask, oversample=4):
    """Core, hermetically testable label loop. ``ask(item, classes)`` returns
    a class string, ``None`` to skip the item, or the sentinel ``"__quit__"``
    to stop. Already-labeled ids are skipped (resumable); each accepted label
    is appended and flushed immediately (Ctrl-C safe). Returns the count added
    this run."""
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = {rec.get("id") for rec in load_labels(labels_path)}
    # Oversample so we can still gather n fresh labels after skipping the ones
    # already recorded from earlier runs.
    items = sample_items(con, task_cfg, max(n, n * oversample))
    classes = list(task_cfg["classes"])
    added = 0
    with open(labels_path, "a") as f:
        for item in items:
            if added >= n:
                break
            if item["id"] in done_ids:
                continue
            label = ask(item, classes)
            if label is None:
                continue
            if label == "__quit__":
                break
            if label not in classes:
                raise ValueError(f"ask returned unknown class {label!r}")
            rec = {
                "id": item["id"],
                "text": item["text"],
                "gold": label,
                "task": task_name,
                "labeled_at": utcnow(),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done_ids.add(item["id"])
            added += 1
    return added


def _interactive_ask(item, classes):
    print(f"\n--- {item['id']} " + "-" * 40)
    print(item["text"])
    menu = "  ".join(f"[{i}] {c}" for i, c in enumerate(classes))
    print(f"\nlabel: {menu}   [s]kip  [q]uit")
    while True:
        raw = input("> ").strip().lower()
        if raw in ("", "s", "skip"):
            return None
        if raw in ("q", "quit"):
            return "__quit__"
        if raw.isdigit() and int(raw) < len(classes):
            return classes[int(raw)]
        if raw in classes:
            return raw
        print("  (invalid -- enter a number, a class name, s to skip, q to quit)")


def cmd_label(args):
    task_cfg = gates.TASKS.get(args.task)
    if task_cfg is None:
        print(f"unknown task {args.task!r} (choose from {', '.join(sorted(gates.TASKS))})",
              file=sys.stderr)
        return 2
    if not task_cfg.get("sample_sql"):
        print(f"task {args.task!r} is not a hand-label task (no sample_sql)", file=sys.stderr)
        return 2
    db_path = resolve_db_path(args.db)
    import sqlite3

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    labels_path = gates.LABELS_DIR / f"{args.task}.jsonl"
    print(f"labeling {args.task} -> {labels_path} (append-only, Ctrl-C safe)")
    try:
        added = run_label(con, args.task, task_cfg, labels_path, args.n, _interactive_ask)
    except KeyboardInterrupt:
        print("\nInterrupted -- labels entered so far are saved; rerun to resume.")
        return 130
    finally:
        con.close()
    total = len(load_labels(labels_path))
    print(f"added {added} labels this run ({total} total in {labels_path.name})")
    return 0


# --- gate --------------------------------------------------------------------


def run_gate(con, task_name, task_cfg, classifier, labels_path, *,
             min_labels, min_positives, labeled_at=None):
    """Core, hermetically testable gate. Loads gold labels, runs ``classifier``
    (``classifier(con, items) -> {id: predicted_class}``) over the same items,
    computes the verdict, persists it, and returns (verdict, scorecard_lines).
    ``classifier`` is mocked in tests; in production it is the task's
    registered classifier path."""
    records = load_labels(labels_path)
    if not records:
        raise ValueError(f"no labels in {labels_path} -- run `label {task_name}` first")
    items = [{"id": r["id"], "text": r.get("text", "")} for r in records]
    predicted_by_id = classifier(con, items)
    gold, predicted = [], []
    for r in records:
        gold.append(r["gold"])
        predicted.append(predicted_by_id.get(r["id"]))
    verdict = gates.compute_verdict(
        task_cfg, gold, predicted,
        min_labels=min_labels, min_positives=min_positives,
        labeled_at=labeled_at or utcnow(), source="owner",
    )
    gates.persist_verdict(con, task_name, verdict)
    return verdict, scorecard_lines(task_name, task_cfg, verdict, min_labels, min_positives)


def scorecard_lines(task_name, task_cfg, verdict, min_labels, min_positives):
    """Per-class scorecard as printable lines."""
    metric = verdict["metric"]
    floor = task_cfg["floor"]
    guard = min_labels if metric == "kappa" else min_positives
    lines = [
        f"gate {task_name}: metric={metric} floor={floor} "
        f"min_sample={guard} n_labels={verdict['n_labels']}",
    ]
    passed = set(verdict["passed"])
    for c in task_cfg["classes"]:
        v = verdict["values"].get(c)
        if v is None:
            mark = "INSUFFICIENT/UNDEFINED"
        elif c in passed:
            mark = "PASS"
        else:
            mark = "MISS"
        vs = "n/a" if v is None else f"{v:.4f}"
        lines.append(f"  {c:<16} {metric}={vs}  {mark}")
    lines.append(f"  => cleared: {', '.join(verdict['passed']) or '(none)'}")
    return lines


def cmd_gate(args):
    task_cfg = gates.TASKS.get(args.task)
    if task_cfg is None:
        print(f"unknown task {args.task!r} (choose from {', '.join(sorted(gates.TASKS))})",
              file=sys.stderr)
        return 2
    if task_cfg["metric"] not in gates.LABEL_METRICS:
        print(f"task {args.task!r} has no hand-label gate "
              f"(metric {task_cfg['metric']!r} is scored elsewhere)", file=sys.stderr)
        return 2
    classifier = _resolve_classifier(task_cfg)
    if classifier is None:
        print(f"task {args.task!r} has no production classifier wired yet -- its "
              "consumer pipeline has not landed; nothing to gate.", file=sys.stderr)
        return 2
    db_path = resolve_db_path(args.db)
    import sqlite3

    import config

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    labels_path = gates.LABELS_DIR / f"{args.task}.jsonl"
    try:
        verdict, lines = run_gate(
            con, args.task, task_cfg, classifier, labels_path,
            min_labels=config.EVAL_GATE_MIN_LABELS,
            min_positives=config.EVAL_GATE_MIN_PRECISION_POSITIVES,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    finally:
        con.close()
    print("\n".join(lines))
    print(f"persisted verdict to ingest_state[{gates.state_key(args.task)}]")
    return 0


def _resolve_classifier(task_cfg):
    """The task's production classifier callable, or None when not yet wired.
    A registered ``classifier`` may be a callable or a dotted path string; the
    consumer pipeline registers it when it lands."""
    ref = task_cfg.get("classifier")
    if ref is None or callable(ref):
        return ref
    module_name, _, attr = ref.rpartition(".")
    import importlib

    return getattr(importlib.import_module(module_name), attr)


def add_parser(subparsers):
    lab = subparsers.add_parser(
        "label", help="F4: hand-label a task's sampled items (owner-run; Ollama-free)")
    lab.add_argument("task", choices=sorted(gates.TASKS))
    lab.add_argument("-n", type=int, default=25, help="items to label this run (default 25)")
    lab.add_argument("--db", help=f"analysis DB path (real corpus: {main_checkout_db_hint()})")
    lab.set_defaults(func=cmd_label)

    g = subparsers.add_parser(
        "gate", help="F4: score a task vs its labels + persist the verdict")
    g.add_argument("task", choices=sorted(gates.TASKS))
    g.add_argument("--db", help=f"analysis DB path (real corpus: {main_checkout_db_hint()})")
    g.set_defaults(func=cmd_gate)
