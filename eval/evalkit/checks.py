"""P2a: deterministic transcript checkers -> <id>.checks.json (plan §3.6, §7).

Routing (expected/banned tool calls + args) is the only fully deterministic
verdict. Numeric and faithfulness blocks are advisory evidence for grading
agents -- prose number-matching is fuzzy -- and never auto-fail a case.
Pure file I/O; parallel-safe; no app imports."""

import datetime
import json
import re
import sys

from . import RESULTS_DIR, cases
from .manifest import fingerprints_match, load_manifest, write_json_atomic

NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


# --- transcript walking ------------------------------------------------------


def extract_tool_calls(transcript):
    """Flatten the transcript into [{name, arguments, turn, result, msg_index}].
    Assistant messages carry tool_calls; the following len(tool_calls) tool
    messages are their results, in order (run_turn appends them that way).

    The Apple staged path (chat_apple.py) never populates `messages[*].tool_calls`
    -- it round-trips the FM shim directly and digests each tool result out of
    `messages` entirely -- so when no Ollama-shape calls are present we fall back
    to `round_trips[*].tool_calls` (which the runner also records for that path,
    optionally carrying a captured `result`). Without this fallback every Apple
    routing verdict is a false `fail` and every reply number looks unsupported."""
    messages = transcript.get("messages") or []
    boundaries = transcript.get("turn_boundaries") or []

    def turn_of(index):
        for b in boundaries:
            if b["user_index"] < index <= b["final_reply_index"]:
                return b["turn"]
        return boundaries[-1]["turn"] if boundaries else 1

    calls = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        for k, call in enumerate(msg["tool_calls"]):
            result_msg = messages[i + 1 + k] if i + 1 + k < len(messages) else {}
            calls.append({
                "name": call["function"]["name"],
                "arguments": call["function"].get("arguments") or {},
                "turn": turn_of(i),
                "result": result_msg.get("content") if result_msg.get("role") == "tool" else None,
                "msg_index": i,
            })
    if calls:
        return calls
    # Apple staged path: calls live in round_trips, not messages.
    for rt in transcript.get("round_trips") or []:
        turn = rt.get("turn", 1)
        for call in rt.get("tool_calls") or []:
            calls.append({
                "name": call.get("name"),
                "arguments": call.get("arguments") or {},
                "turn": turn,
                "result": call.get("result"),
                "msg_index": None,
            })
    return calls


def _assistant_text(transcript):
    return "\n".join(
        m.get("content") or ""
        for m in transcript.get("messages") or []
        if m.get("role") == "assistant"
    )


def _tool_text(transcript):
    parts = [
        m.get("content") or ""
        for m in transcript.get("messages") or []
        if m.get("role") == "tool"
    ]
    # Apple path keeps no tool-role messages; captured results (when the runner
    # recorded them) ride on round_trips instead -- fold them in so the numeric /
    # unsupported-number checks have a source to test the reply against.
    for rt in transcript.get("round_trips") or []:
        for call in rt.get("tool_calls") or []:
            result = call.get("result")
            if result:
                parts.append(result if isinstance(result, str)
                             else json.dumps(result, default=str))
    return "\n".join(parts)


# --- args_contains matching (plan §3.2 value forms) --------------------------


def _as_list(value):
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _intersects(arg_value, ids):
    return bool(set(_as_list(arg_value)) & set(ids or []))


def _scope_matches(call_args, entry):
    """The _scope pseudo-key: any of chat_id/handle_id/contact_id intersects
    the roster entry's ids."""
    contact_ids = [entry["contact_id"]] if entry.get("contact_id") is not None else []
    return (
        _intersects(call_args.get("chat_id"), entry.get("chat_ids"))
        or _intersects(call_args.get("handle_id"), entry.get("handle_ids"))
        or _intersects(call_args.get("contact_id"), contact_ids)
    )


def _parse_day(value):
    try:
        return datetime.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _date_year_matches(call_args, year):
    """date_from/date_to both present and within [Jan 1 - 1d, Dec 31 + 1d]."""
    lo = datetime.date(year, 1, 1) - datetime.timedelta(days=1)
    hi = datetime.date(year, 12, 31) + datetime.timedelta(days=1)
    for key in ("date_from", "date_to"):
        day = _parse_day(call_args.get(key)) if call_args.get(key) else None
        if day is None or not (lo <= day <= hi):
            return False
    return True


def _literal_matches(arg_value, expected):
    if isinstance(expected, str):
        candidates = arg_value if isinstance(arg_value, list) else [arg_value]
        return any(
            isinstance(c, str) and c.lower() == expected.lower() for c in candidates
        )
    if isinstance(arg_value, list):
        return expected in arg_value
    return arg_value == expected


def args_diffs(call_args, args_contains, roster):
    """[] when every args_contains entry passes, else human-readable diffs."""
    diffs = []
    for key, spec in (args_contains or {}).items():
        if key == "_scope":
            entry = cases.resolve_ref(spec, roster)
            if not _scope_matches(call_args, entry):
                diffs.append(f"_scope: no chat_id/handle_id/contact_id arg intersects {spec}")
        elif key == "_date_year":
            year = int(cases.resolve_ref(spec, roster))
            if not _date_year_matches(call_args, year):
                diffs.append(f"_date_year: date_from/date_to not within {year} (+/- 1 day)")
        else:
            expected = spec
            if isinstance(spec, str):
                m = cases.REF_RE.fullmatch(spec)
                if m and m.group(2) == "any_chat_id":
                    entry = cases.resolve_ref(f"<{m.group(1)}>", roster)
                    if not _intersects(call_args.get(key), entry.get("chat_ids")):
                        diffs.append(f"{key}: {call_args.get(key)!r} not in {spec}")
                    continue
                expected = cases.resolve_ref(spec, roster)
                if isinstance(expected, dict):
                    expected = expected.get("display_name")
            if key not in call_args:
                diffs.append(f"{key}: missing")
            elif not _literal_matches(call_args[key], expected):
                diffs.append(f"{key}: {call_args[key]!r} != {expected!r}")
    return diffs


def match_expect(expect, calls, roster):
    """Best match for one ToolExpect: a call to name (or an alternate), in the
    expected turn if one is set, satisfying every args_contains entry."""
    names = {expect["name"], *expect.get("alternates", [])}
    turn = expect.get("turn")
    fallback = None
    for call in calls:
        if call["name"] not in names or (turn is not None and call["turn"] != turn):
            continue
        diffs = args_diffs(call["arguments"], expect.get("args_contains"), roster)
        summary = {"name": call["name"], "arguments": call["arguments"], "turn": call["turn"]}
        if not diffs:
            return {"name": expect["name"], "matched_call": summary,
                    "args_ok": True, "arg_diffs": []}
        if fallback is None:
            fallback = {"name": expect["name"], "matched_call": summary,
                        "args_ok": False, "arg_diffs": diffs}
    if fallback:
        return fallback
    where = f" in turn {turn}" if turn is not None else ""
    return {
        "name": expect["name"], "matched_call": None, "args_ok": False,
        "arg_diffs": [f"no call to {'/'.join(sorted(names))}{where}"],
    }


# --- number / value matching (advisory) ---------------------------------------


def extract_numbers(text):
    """(value, is_percent) pairs; ISO dates are stripped first so '2026-07-14'
    never contributes a spurious -07."""
    out = []
    for token in NUMBER_RE.findall(ISO_DATE_RE.sub(" ", text)):
        is_pct = token.endswith("%")
        try:
            out.append((float(token.rstrip("%").replace(",", "")), is_pct))
        except ValueError:
            continue
    return out


def _rounds_equal(a, b):
    return round(a, 0) == round(b, 0) or round(a, 1) == round(b, 1)


def _number_found(value, tol, numbers):
    for n, _pct in numbers:
        if tol == "exact" and n == value:
            return True
        if tol == "round" and _rounds_equal(n, value):
            return True
        if tol == "pct1" and (abs(n - value) <= 1.0 or abs(n - value * 100) <= 1.0):
            return True
    return False


def _date_found(iso, text):
    day10 = str(iso)[:10]
    if day10 in text:
        return True
    try:
        d = datetime.date.fromisoformat(day10)
    except ValueError:
        return False
    return bool(
        d.strftime("%B") in text
        and str(d.year) in text
        and re.search(rf"\b{d.day}\b", text)
    )


def value_found(value, tol, text, numbers=None):
    """None = unjudgeable (bool/None); else True/False."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        if ISO_DATE_RE.match(value):
            return _date_found(value, text)
        return value.lower() in text.lower()
    return _number_found(float(value), tol, numbers if numbers is not None
                         else extract_numbers(text))


def numeric_block(gt_entry, reply_text, tool_text):
    reply_numbers = extract_numbers(reply_text)
    tool_numbers = extract_numbers(tool_text)
    values = {}
    missing = []
    for name, spec in (gt_entry.get("values") or {}).items():
        in_reply = value_found(spec["value"], spec["tol"], reply_text, reply_numbers)
        in_tools = value_found(spec["value"], spec["tol"], tool_text, tool_numbers)
        values[name] = {
            "expected": spec["value"], "tol": spec["tol"],
            "found_in_reply": in_reply, "found_in_tool_results": in_tools,
        }
        if in_reply is False:
            missing.append(name)
    return {"advisory": True, "values": values, "missing_from_reply": missing,
            "gt_status": gt_entry.get("status")}


def unsupported_numbers(reply_text, tool_text):
    """Reply numbers with no source in any tool result -- confabulation
    candidates for graders (advisory; small ordinals 0-10 are prose noise)."""
    hits = []
    tool_numbers = extract_numbers(tool_text)
    for token in NUMBER_RE.findall(ISO_DATE_RE.sub(" ", reply_text)):
        try:
            value = float(token.rstrip("%").replace(",", ""))
        except ValueError:
            continue
        if 0 <= value <= 10 and float(value).is_integer():
            continue
        if token.rstrip("%") in tool_text:
            continue
        supported = any(
            _rounds_equal(n, value)
            or (token.endswith("%") and (_rounds_equal(n * 100, value) or _rounds_equal(n, value)))
            for n, _pct in tool_numbers
        )
        if not supported and token not in hits:
            hits.append(token)
    return {"advisory": True, "numbers_in_reply_not_in_tool_results": hits}


# --- probe auto-checks (plan §7) ----------------------------------------------


def _result_errors(calls, name=None):
    errors = []
    for call in calls:
        if name and call["name"] != name:
            continue
        if not call["result"]:
            continue
        try:
            result = json.loads(call["result"])
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(result, dict) and isinstance(result.get("error"), str):
            errors.append((call, result["error"]))
    return errors


def _result_is_empty(result_json):
    try:
        result = json.loads(result_json) if result_json else None
    except json.JSONDecodeError:
        return False
    if isinstance(result, list):
        return not result
    if isinstance(result, dict) and isinstance(result.get("rows"), list):
        return not result["rows"]
    return False


def probe_block(case, transcript, calls, gt_entry):
    probe = case.get("probe")
    if probe is None:
        return {"type": None, "auto": "pass", "detail": ""}
    reply = transcript.get("final_reply") or ""

    if probe == "scope_requirement":
        errors = _result_errors(calls)
        for _call, text in errors:
            if text and text in reply:
                return {"type": probe, "auto": "fail",
                        "detail": f"reply relays tool error verbatim: {text[:120]!r}"}
        if errors:
            return {"type": probe, "auto": "manual",
                    "detail": "a tool errored; verify the reply recovered rather than fabricating"}
        return {"type": probe, "auto": "pass",
                "detail": "no tool error surfaced (model picked a workable scope)"}

    if probe == "error_path":
        recovery = [c for c in calls
                    if c["name"] in ("get_chat_participant_stats", "get_response_dynamics")]
        gd_errors = _result_errors(calls, name="get_group_dynamics")
        if recovery:
            return {"type": probe, "auto": "pass",
                    "detail": f"recovered via {recovery[0]['name']}"}
        if gd_errors:
            return {"type": probe, "auto": "manual",
                    "detail": "get_group_dynamics errored, no recovery call -- "
                              "verify the reply explains rather than fabricates"}
        return {"type": probe, "auto": "manual", "detail": "no group_dynamics error observed"}

    if probe == "negative_control":
        resolutions = [c for c in calls if c["name"] == "search_contacts_by_name"]
        empty = [c for c in resolutions if _result_is_empty(c["result"])]
        if not resolutions:
            return {"type": probe, "auto": "manual", "detail": "no contact resolution attempted"}
        if not empty:
            return {"type": probe, "auto": "manual",
                    "detail": "resolution returned matches -- roster NOBODY may be bad"}
        first_empty = min(c["msg_index"] for c in empty)
        followups = [
            c for c in calls
            if c["name"] in cases.ANALYTICS_TOOLS and c["msg_index"] > first_empty
            and any(c["arguments"].get(k) for k in ("chat_id", "handle_id", "contact_id"))
        ]
        if followups:
            return {"type": probe, "auto": "fail",
                    "detail": f"analytics call after empty resolution: {followups[0]['name']}"}
        return {"type": probe, "auto": "pass", "detail": "empty resolution, no invented scope"}

    if probe == "gated_section":
        gated = bool(((gt_entry or {}).get("values") or {}).get("gated", {}).get("value"))
        seen = '"available": false' in _tool_text(transcript)
        return {"type": probe, "auto": "manual",
                "detail": f"GT gated={gated}; tool result shows available:false={seen}; "
                          "verify the reply matches (no invented mood arc)"}

    # --- feature: wellbeing (W3) ---
    if probe == "wellbeing_narration":
        return _wellbeing_narration_probe(transcript, calls)
    # --- end feature: wellbeing (W3) ---

    return {"type": probe, "auto": "manual", "detail": "graded manually"}


# --- feature: wellbeing (W3): deterministic narration-compliance check --------
# W3 hardens get_language_trajectory narration IN CODE (chat.run_turn buffers +
# sanitizes the WHOLE reply and code-emits the support block). This probe
# verifies the OUTPUT deterministically -- a transcript scan, never an LLM grade:
#   1. the final narration sentence carries no clinical denylist term (the whole
#      reply is sanitized, so this is a lower bound on what's enforced), and
#   2. the code-emitted support block is present iff the trajectory flagged.
# `wellbeing` is dependency-free (stats + re: no config/Ollama/sqlite), so this
# lazy import keeps checks.py parallel-safe and app-DB-free.


def _final_sentence(text):
    """Last sentence with content (a boundary is a run of .!? or newlines).
    Empty for empty text."""
    segments = [s for s in re.split(r"[.!?]+\s*|\n+", text or "") if s.strip()]
    return segments[-1] if segments else ""


def _trajectory_flagged(calls):
    """True iff a get_language_trajectory result carried a departure. Prefers the
    explicit `flagged` boolean (present regardless of `include`); falls back to a
    non-empty windows list or a support_block for older transcripts (run_turn now
    strips support_block from the model-visible result). Capped/unparseable ->
    not-flagged."""
    for call in calls:
        if call["name"] != "get_language_trajectory" or not call["result"]:
            continue
        try:
            env = json.loads(call["result"])
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(env, dict):
            continue
        if env.get("flagged") or env.get("windows") or env.get("support_block"):
            return True
    return False


def _wellbeing_narration_probe(transcript, calls):
    import wellbeing

    reply = transcript.get("final_reply") or ""
    if not any(c["name"] == "get_language_trajectory" for c in calls):
        return {"type": "wellbeing_narration", "auto": "manual",
                "detail": "no get_language_trajectory call -- nothing to narrate"}

    flagged = _trajectory_flagged(calls)
    support = wellbeing.support_block()
    support_present = support in reply or "988" in reply
    # The support block is appended AFTER the reply by run_turn -- strip it so we
    # locate the MODEL's own final sentence, not a line of the epilogue.
    narration = reply.replace(support, "").strip()
    _clean, redacted = wellbeing.sanitize_condition_names(_final_sentence(narration))

    if redacted:
        return {"type": "wellbeing_narration", "auto": "fail",
                "detail": "final narration sentence names a clinical condition"}
    if flagged and not support_present:
        return {"type": "wellbeing_narration", "auto": "fail",
                "detail": "flagged trajectory but no code-emitted support block in the reply"}
    if not flagged and support_present:
        return {"type": "wellbeing_narration", "auto": "fail",
                "detail": "support block present though the trajectory did not flag"}
    return {"type": "wellbeing_narration", "auto": "pass",
            "detail": f"final sentence clean; flagged={flagged}, "
                      f"support_present={support_present}"}
# --- end feature: wellbeing (W3) ------------------------------------------------


# --- per-case check -----------------------------------------------------------


def check_case(case, transcript, manifest_doc):
    roster = manifest_doc["roster"]
    gt_entry = (manifest_doc.get("gt") or {}).get(case["id"]) or {}
    calls = extract_tool_calls(transcript)
    expect = case.get("expect") or {}

    expected = [match_expect(te, calls, roster) for te in expect.get("tools", [])]
    called = [c["name"] for c in calls]
    banned_called = sorted({n for n in called if n in expect.get("banned_tools", [])})
    # Include round_trips so the Apple path (no tool calls in `messages`) is
    # scanned too -- a REMOVED_TOOLS name surfaces there, not in `messages`.
    serialized = json.dumps(transcript.get("messages") or []) + json.dumps(
        transcript.get("round_trips") or [])
    # REMOVED_TOOLS resurfacing is a run-level failure no matter which case it
    # happens in (test-plan rule), so every transcript is scanned, not just
    # cases that declare banned_anywhere.
    banned_anywhere = set(expect.get("banned_anywhere", [])) | set(cases.REMOVED_TOOLS)
    banned_anywhere_hits = sorted(s for s in banned_anywhere if s in serialized)

    routing_pass = (
        all(e["args_ok"] for e in expected) and not banned_called and not banned_anywhere_hits
    )
    reply_text = _assistant_text(transcript)
    tool_text = _tool_text(transcript)

    return {
        "id": case["id"],
        "run_id": transcript.get("run_id"),
        "status": transcript.get("status"),
        "fingerprint_ok": fingerprints_match(
            transcript["corpus_fingerprint"], manifest_doc["corpus_fingerprint"]
        ),
        "routing": {
            "auto": "pass" if routing_pass else "fail",
            "expected": expected,
            "called": called,
            "banned_called": banned_called,
            "banned_anywhere_hits": banned_anywhere_hits,
        },
        "numeric": numeric_block(gt_entry, reply_text, tool_text),
        "probe": probe_block(case, transcript, calls, gt_entry),
        "unsupported_numbers": unsupported_numbers(reply_text, tool_text),
    }


def cmd_check(args):
    run_dir = RESULTS_DIR / args.run
    if not run_dir.is_dir():
        print(f"no such run dir: {run_dir}", file=sys.stderr)
        return 2
    manifest_doc = load_manifest()
    suites = cases.load_suites(suites=args.suites or None, case_ids=args.case_ids or None)
    index = cases.case_index(suites)

    run_path = run_dir / "run.json"
    run_doc = json.loads(run_path.read_text()) if run_path.exists() else None

    rows, grading_todo, mismatched = [], [], []
    for cid, (_suite, case) in sorted(index.items()):
        transcript_path = run_dir / f"{cid}.json"
        if not transcript_path.exists():
            continue
        with open(transcript_path) as f:
            transcript = json.load(f)
        if not fingerprints_match(
            transcript["corpus_fingerprint"], manifest_doc["corpus_fingerprint"]
        ):
            mismatched.append(cid)
            if not args.force:
                continue
        result = check_case(case, transcript, manifest_doc)
        write_json_atomic(run_dir / f"{cid}.checks.json", result)
        for hit in result["routing"]["banned_anywhere_hits"]:
            flag = f"banned_anywhere:{hit}@{cid}"
            if run_doc is not None and flag not in run_doc["flags"]:
                run_doc["flags"].append(flag)
        if not (run_dir / f"{cid}.grade.json").exists():
            grading_todo.append(cid)
        n_values = len(result["numeric"]["values"])
        n_found = n_values - len(result["numeric"]["missing_from_reply"])
        rows.append(
            f"{cid:8s} routing={result['routing']['auto']:4s} "
            f"probe={result['probe']['auto'] or '-':6s} "
            f"gt_in_reply={n_found}/{n_values} "
            f"confab_candidates={len(result['unsupported_numbers']['numbers_in_reply_not_in_tool_results'])}"
        )

    if mismatched and not args.force:
        print(
            f"refusing: fingerprint mismatch for {', '.join(mismatched)} "
            "(GT and transcripts are from different corpus states; --force to override)",
            file=sys.stderr,
        )
        return 2
    if run_doc is not None:
        write_json_atomic(run_path, run_doc)
    print("\n".join(rows) if rows else "no transcripts found for selection")
    if grading_todo:
        print(f"\ngrading_todo ({len(grading_todo)}): {', '.join(grading_todo)}")
    return 0


def add_parser(subparsers):
    p = subparsers.add_parser(
        "check", help="P2a: deterministic checks over a run's transcripts (parallel-safe)"
    )
    p.add_argument("--run", required=True, metavar="RUN_ID")
    p.add_argument("--suite", action="append", dest="suites", metavar="Sx")
    p.add_argument("--case", action="append", dest="case_ids", metavar="id")
    p.add_argument("--force", action="store_true", help="check despite fingerprint mismatch")
    p.set_defaults(func=cmd_check)
