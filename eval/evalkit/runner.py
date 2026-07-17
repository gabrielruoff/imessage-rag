"""P1: the in-process sequential sweep (plan §4.4). Drives chat.run_turn
directly -- the mutated messages list IS the transcript; stdout is never
scraped. One Ollama lane, enforced by eval/results/.ollama.lock."""

import fcntl
import json
import os
import sys
import time

from . import LOCK_PATH, RESULTS_DIR
from .manifest import (
    fingerprint,
    fingerprints_match,
    load_manifest,
    main_checkout_db_hint,
    resolve_db_path,
    utcnow,
    write_json_atomic,
)

# Config knobs snapshotted into every transcript (plan §1).
CONFIG_KNOBS = (
    "CHAT_MODEL", "MAX_TOOL_ITERATIONS", "TOOL_RESULT_MAX_CHARS", "CHAT_TIMEOUT",
    "OLLAMA_HOST", "OLLAMA_PORT", "LOCAL_TZ", "SESSION_GAP_MINUTES",
    "INITIATION_GAP_HOURS", "DOUBLE_TEXT_MIN_MINUTES", "ADJACENCY_MINUTES",
    "LATE_NIGHT_START_HOUR", "LATE_NIGHT_END_HOUR", "DYAD_TERM_MIN_USES",
    "DYAD_TERM_MIN_LIFT",
)


class CaseTimeout(Exception):
    """Per-case wall budget exceeded (checked between round trips)."""


_lock_fd = None  # held open for the process's lifetime; the flock rides on it


def acquire_ollama_lock(run_id):
    """One sequential Ollama lane across runner and p3 (plan §1): flock on
    eval/results/.ollama.lock, exit 2 if another holder is alive. The kernel
    drops the flock when its holder dies, so there is no stale-lock state and
    no steal race; the JSON payload is diagnostics only."""
    global _lock_fd
    RESULTS_DIR.mkdir(exist_ok=True)
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        try:
            holder = json.loads(LOCK_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            holder = {}
        print(
            f"refusing to start: Ollama lane is locked by {holder or LOCK_PATH} "
            "(never run two Ollama-touching commands concurrently)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    payload = json.dumps({"pid": os.getpid(), "run_id": run_id, "started_utc": utcnow()})
    os.ftruncate(fd, 0)
    os.write(fd, payload.encode())
    _lock_fd = fd


def release_ollama_lock():
    """Closing the fd releases the flock; the (truncated) lock file stays --
    unlinking it would let a late-coming process lock a fresh inode while an
    older holder still runs against the unlinked one."""
    global _lock_fd
    if _lock_fd is None:
        return
    os.ftruncate(_lock_fd, 0)
    os.close(_lock_fd)
    _lock_fd = None


def config_snapshot():
    import config

    return {k: getattr(config, k) for k in CONFIG_KNOBS}


def _select_cases(args):
    from . import cases

    if not (args.all or args.suites or args.case_ids):
        print("nothing selected: pass --all, --suite Sx, or --case id", file=sys.stderr)
        raise SystemExit(2)
    suites = cases.load_suites(
        suites=args.suites or None, case_ids=None if args.all or args.suites else args.case_ids
    )
    selected = [c for s in suites for c in s["cases"]]
    if args.case_ids and (args.all or args.suites):
        selected = [c for c in selected if c["id"] in args.case_ids]
    return selected


def _resolve_turns(case, roster):
    from . import cases

    turns = case["turns"] if case.get("turns") else [case["prompt"]]
    return [cases.substitute_text(t, roster) for t in turns]


def _install_recorder(chat_mod):
    """Wrap chat.ollama_chat_stream (a module global resolved at call time by
    run_turn) to time round trips, record per-round tool calls, and enforce
    the per-case wall budget between round trips."""
    state = {"rounds": [], "turn": 1, "deadline": None}
    real = chat_mod.ollama_chat_stream

    def recording_stream(messages, tools=chat_mod.OLLAMA_TOOLS, show_model_header=True,
                         model=None):
        if state["deadline"] is not None and time.monotonic() > state["deadline"]:
            raise CaseTimeout("per-case wall budget exceeded")
        t0 = time.monotonic()
        msg = real(messages, tools=tools, show_model_header=show_model_header, model=model)
        state["rounds"].append({
            "turn": state["turn"],
            "duration_s": round(time.monotonic() - t0, 1),
            "tool_calls": [
                {"name": c["function"]["name"], "arguments": c["function"].get("arguments", {})}
                for c in msg.get("tool_calls") or []
            ],
        })
        return msg

    chat_mod.ollama_chat_stream = recording_stream
    return state


def _run_case(chat_mod, state, case, turns, model, case_timeout):
    """One case attempt. Returns (status, messages, turn_boundaries,
    final_reply). Transport errors escape for the caller's retry."""
    state["rounds"] = []
    state["deadline"] = time.monotonic() + case_timeout
    messages = [{"role": "system", "content": chat_mod._build_system_prompt()}]
    turn_boundaries = []
    reply = ""
    try:
        for i, turn in enumerate(turns, 1):
            state["turn"] = i
            messages.append({"role": "user", "content": turn})
            user_index = len(messages) - 1
            reply = chat_mod.run_turn(messages, show_model_header=False, model=model)
            turn_boundaries.append(
                {"turn": i, "user_index": user_index, "final_reply_index": len(messages) - 1}
            )
    except CaseTimeout:
        # Close the interrupted turn's boundary so checks.extract_tool_calls
        # attributes its tool calls to this turn, not the previous one.
        turn_boundaries.append(
            {"turn": i, "user_index": user_index, "final_reply_index": len(messages) - 1}
        )
        return "timeout", messages, turn_boundaries, reply
    return ("ok" if reply else "empty_reply"), messages, turn_boundaries, reply


def _write_transcript(run_dir, case_id, transcript):
    write_json_atomic(run_dir / f"{case_id}.json", transcript)
    lines = [
        f"# {case_id} status={transcript['status']} model={transcript['model']} "
        f"duration={transcript['duration_s']}s"
    ]
    for i, turn in enumerate(transcript["turns_resolved"], 1):
        lines.append(f"\n[turn {i}] you> {turn}")
        for rt in transcript["round_trips"]:
            if rt["turn"] != i:
                continue
            for call in rt["tool_calls"]:
                lines.append(f"  [tool call] {call['name']}({json.dumps(call['arguments'])})")
    lines.append(f"\n[final reply]\n{transcript['final_reply']}")
    (run_dir / f"{case_id}.txt").write_text("\n".join(lines) + "\n")


def cmd_run(args):
    # Env from flags BEFORE the first app import (config reads env at import;
    # load_dotenv never overrides pre-set vars).
    if args.max_iterations is not None:
        os.environ["MAX_TOOL_ITERATIONS"] = str(args.max_iterations)
    if args.tool_result_max_chars is not None:
        os.environ["TOOL_RESULT_MAX_CHARS"] = str(args.tool_result_max_chars)
    db_path = resolve_db_path(args.db)

    manifest = load_manifest()
    roster = manifest["roster"]
    selected = _select_cases(args)
    if not selected:
        print("no cases selected", file=sys.stderr)
        return 2

    if args.dry_run:
        for case in selected:
            try:
                turns = _resolve_turns(case, roster)
            except KeyError as e:
                print(f"{case['id']}: SKIP unresolvable placeholder ({e})")
                continue
            expected = [
                "/".join([te["name"], *te.get("alternates", [])])
                for te in (case.get("expect") or {}).get("tools", [])
            ]
            print(f"{case['id']}:")
            for t in turns:
                print(f"  prompt: {t}")
            print(f"  expect: {', '.join(expected) or '(none)'}")
        return 0

    fp = fingerprint(db_path)
    if not fingerprints_match(fp, manifest["corpus_fingerprint"]):
        if not args.force:
            print(
                "refusing to run: DB fingerprint does not match the manifest "
                "(re-run `manifest`, or pass --force)",
                file=sys.stderr,
            )
            return 2
        print("WARNING: fingerprint mismatch, forced (recorded in run.json)")

    import http.client

    import chat
    import ollama
    import progress  # imports config -- must stay below the env block above
    from rich.console import Console

    chat.console = Console(quiet=True)
    state = _install_recorder(chat)
    model = args.model or config_snapshot()["CHAT_MODEL"]

    run_dir = RESULTS_DIR / args.run_id
    acquire_ollama_lock(args.run_id)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        run_path = run_dir / "run.json"
        if run_path.exists():
            with open(run_path) as f:
                run_doc = json.load(f)
        else:
            run_doc = {
                "run_id": args.run_id, "label": args.label, "created_utc": utcnow(),
                "model": model, "config": config_snapshot(), "corpus_fingerprint": fp,
                "cases": {}, "flags": [],
            }
        if args.force and not fingerprints_match(fp, manifest["corpus_fingerprint"]):
            run_doc["flags"].append("fingerprint_mismatch_forced")

        exit_code = 0
        # The description stays generic: resolved prompts carry roster names, and
        # progress output is not the place to leak one.
        # log_every=1: ~71 cases at minutes apiece, so one plain-mode line per
        # case is the right cadence -- and the default stride of 200 would emit
        # none at all for a loop this small (the exact case an operator pipes to
        # a sweep log).
        with progress.track("Running cases", total=len(selected), log_every=1) as p:
            for case in selected:
                cid = case["id"]
                existing = run_dir / f"{cid}.json"
                explicit = bool(args.case_ids and cid in args.case_ids)
                if args.resume and not explicit and existing.exists():
                    with open(existing) as f:
                        if json.load(f).get("status") == "ok":
                            print(f"{cid}: resume skip (ok)")
                            p.advance()
                            continue
                try:
                    turns = _resolve_turns(case, roster)
                except KeyError as e:
                    print(f"{cid}: SKIP unresolvable placeholder ({e})")
                    run_doc["cases"][cid] = {"status": "setup_error", "duration_s": 0}
                    write_json_atomic(run_path, run_doc)
                    exit_code = 1
                    p.advance()
                    continue
                started = utcnow()
                t0 = time.monotonic()
                status = messages = boundaries = reply = None
                for attempt in (1, 2):
                    try:
                        status, messages, boundaries, reply = _run_case(
                            chat, state, case, turns, model, args.case_timeout
                        )
                        break
                    except (OSError, http.client.HTTPException, ollama.OllamaResponseError) as e:
                        print(f"{cid}: transport error ({e}), attempt {attempt}")
                        if attempt == 2:
                            status, messages, boundaries, reply = "transport_error", [], [], ""
                duration = round(time.monotonic() - t0, 1)
                transcript = {
                    "id": cid, "run_id": args.run_id, "status": status,
                    "prompt_resolved": turns[0] if not case.get("turns") else None,
                    "turns_resolved": turns, "model": model, "config": config_snapshot(),
                    "corpus_fingerprint": fp, "started_utc": started, "duration_s": duration,
                    "round_trips": state["rounds"], "turn_boundaries": boundaries,
                    "messages": messages, "final_reply": reply,
                }
                _write_transcript(run_dir, cid, transcript)
                run_doc["cases"][cid] = {"status": status, "duration_s": duration}
                write_json_atomic(run_path, run_doc)  # crash-resumable
                print(f"{cid}: {status} ({duration}s, {len(state['rounds'])} round trips)")
                if status != "ok":
                    exit_code = 1
                p.advance()
        return exit_code
    finally:
        release_ollama_lock()


def add_parser(subparsers):
    p = subparsers.add_parser("run", help="P1: sequential sweep through chat.run_turn (Ollama)")
    p.add_argument("--run-id", required=True)
    p.add_argument("--all", action="store_true")
    p.add_argument("--suite", action="append", dest="suites", metavar="Sx")
    p.add_argument("--case", action="append", dest="case_ids", metavar="id")
    p.add_argument("--db", help=f"analysis DB path (real corpus: {main_checkout_db_hint()})")
    p.add_argument("--model", help="chat model override (one model per run)")
    p.add_argument("--resume", action="store_true", help="skip cases already ok in this run dir")
    p.add_argument("--dry-run", action="store_true", help="print resolved prompts; no Ollama")
    p.add_argument("--force", action="store_true", help="run despite a fingerprint mismatch")
    p.add_argument("--max-iterations", type=int, help="sets MAX_TOOL_ITERATIONS for the run")
    p.add_argument(
        "--tool-result-max-chars", type=int, help="sets TOOL_RESULT_MAX_CHARS for the run"
    )
    p.add_argument("--case-timeout", type=float, default=1200, metavar="S")
    p.add_argument("--label", default=None)
    p.set_defaults(func=cmd_run)
