"""Suite/case loading, validation, and placeholder resolution (plan §3.1-3.3,
§4.1). Import-light by design: no app modules, so hermetic tests and every
subcommand can import it before TARGET_DB & co. are set in the environment."""

import json
import re

from . import CASES_DIR

PLACEHOLDER_RE = re.compile(r"<([A-Z][A-Z0-9_]*)(?:\.([a-z_]+))?>")
# Full-reference form for gt.params / args_contains values: '<TOP1>' or
# '<TOP1>.chat_ids' (attribute OUTSIDE the brackets).
REF_RE = re.compile(r"<([A-Z][A-Z0-9_]*)>(?:\.([a-z_0-9]+))?")

PROBE_TYPES = (
    "scope_requirement",
    "error_path",
    "gated_section",
    "negative_control",
    "thin_data",
    "open_ended",
    # --- feature: wellbeing (W3) ---
    "wellbeing_narration",  # deterministic transcript scan in checks.probe_block
    # --- end feature: wellbeing (W3) ---
)
LEVERS = ("desc", "steer", "args", "shape", "size", "confab", "pipeline", "judge-batch")
AXES = ("routing", "accuracy", "faithfulness", "communication")

ANALYTICS_TOOLS = (
    "get_chat_summary_stats",
    "get_activity_patterns",
    "get_response_dynamics",
    "get_reaction_stats",
    "get_milestones",
    "get_word_trends",
    "get_group_dynamics",
    "get_chat_participant_stats",
    "get_word_frequency",
)
# Wave-1+ feature tools now shipped unconditionally in tool_schemas
# (persona / conflict / nudges / wellbeing). ALL_TOOLS must mirror the schema
# names exactly -- selfcheck fails on any drift -- so they are reconciled here.
# Fenced by feature so the PM can dedupe if a sibling worktree adds the same
# names. The flag-gated tools (DRIFT_ENABLED / CONFLICT_TRACE_ENABLED /
# PERSONA_SIM_ENABLED -- see tool_schemas.GATED_TOOLS) are intentionally absent
# here; selfcheck subtracts the active ones from the schema before diffing, so it
# stays flag-config-agnostic without this module importing config.
_FEATURE_TOOLS = (
    # --- feature: persona ---
    "get_voiceprint",
    "draft_reply",
    "check_style_match",
    # --- feature: conflict ---
    "find_conflicts",
    "get_conflict",
    "diagnose_conflict",
    "draft_apology",
    # --- feature: wellbeing ---
    "get_language_trajectory",
    # --- feature: nudges ---
    "get_loose_threads",
    "resolve_loose_thread",
    "mute_contact_from_nudges",
)
ALL_TOOLS = (
    "search_messages_hybrid",
    "search_conversations",
    "search_messages_fts",
    "search_messages_sql",
    "find_similar_conversations",
    "get_conversation",
    "get_message_context",
    "search_contacts_by_name",
    "list_chats",
    "search_contact_by_handle",
    "get_message_reactions",
    "get_full_messages",
) + ANALYTICS_TOOLS + _FEATURE_TOOLS
# Superseded tool names that must never resurface anywhere in a transcript.
# checks.py scans EVERY transcript for these (a hit is a run-level flag);
# a case's banned_anywhere list adds case-specific strings on top.
REMOVED_TOOLS = (
    "get_activity_histogram",
    "get_conversation_dynamics",
    "get_top_reacted_messages",
)

_EXPECT_KEYS = {"tools", "banned_tools", "banned_anywhere"}
_TOOL_EXPECT_KEYS = {"name", "alternates", "args_contains", "turn"}


def load_roster_defs(path=None):
    """Placeholder *definitions* (committed roster.json) -- name -> {kind,
    definition}. The resolved roster (real names/ids) lives in the manifest."""
    with open(path or CASES_DIR / "roster.json") as f:
        return json.load(f)["placeholders"]


def _iter_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)


def case_placeholders(case):
    """Every placeholder name referenced anywhere in the case."""
    names = set()
    sources = [case.get("prompt"), case.get("turns")]
    for expect in (case.get("expect") or {}).get("tools", []):
        sources.append(expect.get("args_contains"))
    if case.get("gt"):
        sources.append(case["gt"].get("params"))
    for text in _iter_strings([s for s in sources if s is not None]):
        for m in PLACEHOLDER_RE.finditer(text):
            names.add(m.group(1))
    return names


def validate_case(case, roster_defs, suite_id=""):
    cid = case.get("id")
    if not isinstance(cid, str) or not cid:
        raise ValueError(f"{suite_id}: case without a string id: {case!r}")

    def fail(msg):
        raise ValueError(f"{cid}: {msg}")

    has_prompt = isinstance(case.get("prompt"), str)
    has_turns = isinstance(case.get("turns"), list) and case["turns"]
    if has_prompt == bool(has_turns):
        fail("exactly one of prompt/turns must be set")
    if has_turns and not all(isinstance(t, str) for t in case["turns"]):
        fail("turns must be a list of strings")

    probe = case.get("probe")
    if probe is not None and probe not in PROBE_TYPES:
        fail(f"unknown probe {probe!r} (expected one of {PROBE_TYPES})")
    if case.get("auto_probe") is not None and not isinstance(case["auto_probe"], dict):
        fail("auto_probe must be a dict")
    if not isinstance(case.get("grader_notes"), str):
        fail("grader_notes (str) is required")

    expect = case.get("expect") or {}
    unknown = set(expect) - _EXPECT_KEYS
    if unknown:
        fail(f"unknown expect keys {sorted(unknown)}")
    for tool in expect.get("banned_tools", []):
        if tool not in ALL_TOOLS:
            fail(f"banned_tools entry {tool!r} is not a known tool")
    if not all(isinstance(s, str) for s in expect.get("banned_anywhere", [])):
        fail("banned_anywhere must be strings")
    for te in expect.get("tools", []):
        unknown = set(te) - _TOOL_EXPECT_KEYS
        if unknown:
            fail(f"unknown ToolExpect keys {sorted(unknown)}")
        names = [te.get("name")] + list(te.get("alternates", []))
        for name in names:
            if name not in ALL_TOOLS:
                fail(f"ToolExpect tool {name!r} is not a known tool")
        if not isinstance(te.get("args_contains", {}), dict):
            fail("args_contains must be a dict")
        if "turn" in te and not isinstance(te["turn"], int):
            fail("ToolExpect turn must be an int")

    gt = case.get("gt")
    if gt is not None:
        if not isinstance(gt.get("recipe"), str):
            fail("gt.recipe must be a string")
        if not isinstance(gt.get("params", {}), dict):
            fail("gt.params must be a dict")

    unknown = case_placeholders(case) - set(roster_defs)
    if unknown:
        fail(f"placeholders {sorted(unknown)} not defined in roster.json")


def validate_suite(suite, roster_defs, seen_ids):
    sid = suite.get("suite")
    if not isinstance(sid, str) or not sid:
        raise ValueError(f"suite file without a suite id: {suite.get('title')!r}")
    if not isinstance(suite.get("targets"), dict) or set(suite["targets"]) != set(AXES):
        raise ValueError(f"{sid}: targets must set all of {AXES}")
    for case in suite.get("cases", []):
        validate_case(case, roster_defs, suite_id=sid)
        if case["id"] in seen_ids:
            raise ValueError(f"duplicate case id {case['id']}")
        seen_ids.add(case["id"])


def load_suites(suites=None, case_ids=None, cases_dir=None):
    """Load + validate eval/cases/S*.json, optionally filtered to suite ids
    and/or case ids. Unknown filter values raise rather than silently
    matching nothing."""
    cases_dir = cases_dir or CASES_DIR
    roster_defs = load_roster_defs(cases_dir / "roster.json")
    seen_ids = set()
    loaded = []
    for path in sorted(cases_dir.glob("S*.json")):
        with open(path) as f:
            suite = json.load(f)
        validate_suite(suite, roster_defs, seen_ids)
        loaded.append(suite)
    if suites:
        unknown = set(suites) - {s["suite"] for s in loaded}
        if unknown:
            raise ValueError(f"unknown suite(s) {sorted(unknown)}")
        loaded = [s for s in loaded if s["suite"] in suites]
    if case_ids:
        unknown = set(case_ids) - seen_ids
        if unknown:
            raise ValueError(f"unknown case id(s) {sorted(unknown)}")
        for suite in loaded:
            suite["cases"] = [c for c in suite["cases"] if c["id"] in case_ids]
        loaded = [s for s in loaded if s["cases"]]
    return loaded


def case_index(suites):
    """{case_id: (suite, case)} over loaded suites."""
    return {c["id"]: (s, c) for s in suites for c in s["cases"]}


def _display(entry):
    if isinstance(entry, dict):
        return str(entry.get("display_name", ""))
    return str(entry)


def substitute_text(text, roster):
    """Replace every <NAME> with its display form (contact/group dicts ->
    display_name, scalars -> str). Unknown placeholders raise."""

    def repl(m):
        name, attr = m.group(1), m.group(2)
        if name not in roster:
            raise KeyError(f"placeholder <{name}> not in roster")
        entry = roster[name]
        if entry is None:
            raise KeyError(f"placeholder <{name}> is unresolved in the roster")
        if attr is not None:
            if not isinstance(entry, dict) or attr not in entry:
                raise KeyError(f"<{name}.{attr}>: no such attribute in roster entry")
            return str(entry[attr])
        return _display(entry)

    return PLACEHOLDER_RE.sub(repl, text)


def resolve_ref(value, roster):
    """Resolve a gt.params / args_contains value. A string that IS a
    placeholder ref returns the resolved object ('<TOP1>' -> the roster entry
    dict, '<TOP1>.chat_ids' -> that list, '<YEAR>' -> int); a string merely
    containing placeholders gets inline substitution; everything else passes
    through unchanged."""
    if not isinstance(value, str):
        return value
    m = REF_RE.fullmatch(value)
    if m:
        name, attr = m.group(1), m.group(2)
        if name not in roster:
            raise KeyError(f"placeholder <{name}> not in roster")
        entry = roster[name]
        if attr is None:
            return entry
        if not isinstance(entry, dict) or attr not in entry:
            raise KeyError(f"<{name}.{attr}>: no such attribute in roster entry")
        return entry[attr]
    if PLACEHOLDER_RE.search(value):
        return substitute_text(value, roster)
    return value
