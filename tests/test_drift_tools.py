"""Hermetic tests for the drift READ tools (RES-59 / D2): get_relationship_drift
and get_fadeout_watchlist, plus the DRIFT_ENABLED schema-registration gate.

No Ollama and no real chat.db -- the read tools run against a temporary SQLite
DB seeded with dyad_drift rows directly (the read layer never re-scores, so the
D1 pipeline is not involved). The registration test reloads tool_schemas + chat
with config.DRIFT_ENABLED toggled.

NOTE: tests/test_drift.py owns the ROOT pipeline (drift.py + dyad_drift build);
this file is the separate read-tool suite. Run from the repo root:
    python3 -m unittest tests.test_drift_tools
"""

import importlib
import json
import os
import sqlite3
import tempfile
import unittest

import chat
import config
import query_tools
import schema
import tool_schemas
from query_tools.drift import _COOLING_PHASES, _MAX_TRAIL_MONTHS, _SIGNALS

# dyad_drift insert column order (mirrors drift._INSERT_COLUMNS; kept local so
# this read-tool suite never imports the root drift.py twin).
_DRIFT_COLUMNS = (
    "contact_key", "label", "n_messages", "n_months", "first_month", "last_month",
    "drift_score", "phase", "phase_corroborated", "mandatory_complete",
    "confidence_tier",
    "volume_value", "volume_gate", "reciprocity_value", "reciprocity_gate",
    "latency_value", "latency_gate", "silence_value", "silence_gate",
    "valence_value", "valence_gate", "depth_value", "depth_gate",
    "initiation_value", "initiation_gate",
    "per_signal_gate_json", "coverage_json", "monthly_series_json",
)


def _insert_drift(con, contact_key, *, label="Alex Rivers", phase="cooling",
                  drift_score=0.5, mandatory_complete=1, phase_corroborated=1,
                  confidence_tier="medium", n_messages=500, n_months=12,
                  signal_values=None, signal_gates=None, monthly_series=None,
                  last_message_days=40):
    """Seed one dyad_drift row with sensible defaults + overrides."""
    values = {s: 0.0 for s in _SIGNALS}
    gates = {s: 0 for s in _SIGNALS}
    if signal_values:
        values.update(signal_values)
    if signal_gates:
        gates.update(signal_gates)
    if monthly_series is None:
        monthly_series = {"silence": [["2023-11", 0.7], ["2023-12", 0.85]]}
    present = [s for s in _SIGNALS if s in monthly_series]
    coverage = {
        "n_messages": n_messages, "n_months": n_months,
        "first_month": "2023-01", "last_month": "2023-12",
        "last_message_days": last_message_days, "both_active_recent": False,
        "signals_present": present,
        "signals_absent": [s for s in _SIGNALS if s not in monthly_series],
        "annotation": {"n_llm": 3, "n_distilled": 0,
                       "valence_llm_month_fraction": 1.0},
    }
    per_signal = {s: {"present": s in monthly_series,
                      "significant": bool(gates[s]), "drift_val": values[s],
                      "n_months": n_months} for s in _SIGNALS}
    row = {
        "contact_key": contact_key, "label": label, "n_messages": n_messages,
        "n_months": n_months, "first_month": "2023-01", "last_month": "2023-12",
        "drift_score": drift_score, "phase": phase,
        "phase_corroborated": phase_corroborated,
        "mandatory_complete": mandatory_complete,
        "confidence_tier": confidence_tier,
        "per_signal_gate_json": json.dumps(per_signal),
        "coverage_json": json.dumps(coverage),
        "monthly_series_json": json.dumps(monthly_series),
    }
    for s in _SIGNALS:
        row[f"{s}_value"] = values[s]
        row[f"{s}_gate"] = gates[s]
    placeholders = ",".join("?" * len(_DRIFT_COLUMNS))
    con.execute(
        f"INSERT INTO dyad_drift({','.join(_DRIFT_COLUMNS)}) "
        f"VALUES ({placeholders})", tuple(row[c] for c in _DRIFT_COLUMNS))


class _TempDBTest(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))
        self._saved = query_tools.DB_PATH
        query_tools.DB_PATH = self.db_path
        self.addCleanup(setattr, query_tools, "DB_PATH", self._saved)
        con = self._con()
        schema.create_schema(con)  # creates dyad_drift among the rest
        con.commit()

    def _con(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        self.addCleanup(con.close)
        return con

    def _seed_identity(self, con):
        """A contact-resolved handle (50 -> contact 2), a contact-less handle
        (51), a 1:1 chat for each, and a group chat -- the four input-mapping
        cases."""
        con.execute("INSERT INTO contacts(id, source_uniqueid, first_name, "
                    "last_name) VALUES (2, 'uid-2', 'Alex', 'Rivers')")
        con.execute("INSERT INTO handles(id, handle_identifier, contact_id, "
                    "service) VALUES (50, '+15550001111', 2, 'iMessage')")
        con.execute("INSERT INTO handles(id, handle_identifier, contact_id, "
                    "service) VALUES (51, '+15559998888', NULL, 'SMS')")
        con.execute("INSERT INTO chats(id, chat_identifier, is_group, service) "
                    "VALUES (1, 'chat1', 0, 'iMessage')")   # 1:1 -> contact:2
        con.execute("INSERT INTO chats(id, chat_identifier, is_group, service) "
                    "VALUES (2, 'group2', 1, 'iMessage')")  # group -> error
        con.execute("INSERT INTO chats(id, chat_identifier, is_group, service) "
                    "VALUES (3, 'chat3', 0, 'SMS')")        # 1:1 -> handle:51
        con.execute("INSERT INTO chat_participants(chat_id, handle_id) VALUES (1, 50)")
        con.execute("INSERT INTO chat_participants(chat_id, handle_id) VALUES (3, 51)")
        con.commit()


# --- input -> contact_key mapping -------------------------------------------


class InputMappingTest(_TempDBTest):
    def setUp(self):
        super().setUp()
        con = self._con()
        self._seed_identity(con)
        _insert_drift(con, "contact:2", label="Alex Rivers")
        _insert_drift(con, "handle:51", label="+15559998888")
        con.commit()

    def test_contact_id_maps_to_contact_key(self):
        out = query_tools.get_relationship_drift(contact_id=2)
        self.assertTrue(out["found"])
        self.assertEqual(out["contact_key"], "contact:2")

    def test_bare_handle_with_contact_resolves_to_contact_key(self):
        out = query_tools.get_relationship_drift(handle_id=50)
        self.assertEqual(out["contact_key"], "contact:2")  # 50 -> contact 2

    def test_bare_handle_without_contact_stays_handle_key(self):
        out = query_tools.get_relationship_drift(handle_id=51)
        self.assertEqual(out["contact_key"], "handle:51")

    def test_one_to_one_chat_resolves_to_counterparty(self):
        self.assertEqual(
            query_tools.get_relationship_drift(chat_id=1)["contact_key"], "contact:2")
        self.assertEqual(
            query_tools.get_relationship_drift(chat_id=3)["contact_key"], "handle:51")

    def test_group_chat_is_an_error(self):
        out = query_tools.get_relationship_drift(chat_id=2)
        self.assertIn("error", out)
        self.assertIn("group", out["error"].lower())

    def test_scalar_or_list_accepted(self):
        self.assertEqual(
            query_tools.get_relationship_drift(contact_id=[2])["contact_key"],
            "contact:2")

    def test_no_identifier_is_an_error(self):
        self.assertIn("error", query_tools.get_relationship_drift())

    def test_multiple_identifiers_is_an_error(self):
        out = query_tools.get_relationship_drift(contact_id=2, handle_id=50)
        self.assertIn("error", out)

    def test_unknown_handle_and_chat_error_not_raise(self):
        self.assertIn("error", query_tools.get_relationship_drift(handle_id=999))
        self.assertIn("error", query_tools.get_relationship_drift(chat_id=999))

    def test_missing_row_reports_not_found(self):
        # A resolvable key with no dyad_drift row (below inclusion gates).
        con = self._con()
        con.execute("INSERT INTO handles(id, handle_identifier, contact_id) "
                    "VALUES (77, '+15550000000', NULL)")
        con.commit()
        out = query_tools.get_relationship_drift(handle_id=77)
        self.assertFalse(out["found"])
        self.assertEqual(out["contact_key"], "handle:77")


# --- channel caveat ----------------------------------------------------------


class ChannelCaveatTest(_TempDBTest):
    # One numeric contact per declining phase, so each is readable via contact_id.
    _PHASE_CONTACT = {"cooling": 3, "fading": 4, "dormant": 5}

    def test_caveat_present_on_every_declining_phase(self):
        con = self._con()
        for phase, cid in self._PHASE_CONTACT.items():
            con.execute("INSERT INTO contacts(id, first_name) VALUES (?, ?)",
                        (cid, f"Person{phase}"))
            _insert_drift(con, f"contact:{cid}", label=f"Person {phase}",
                          phase=phase)
        con.commit()
        self.assertEqual(set(self._PHASE_CONTACT), set(_COOLING_PHASES))
        for phase, cid in self._PHASE_CONTACT.items():
            out = query_tools.get_relationship_drift(contact_id=cid)
            self.assertIn("channel_caveat", out, phase)
            cav = out["channel_caveat"]
            self.assertFalse(cav["confirmed_imessage_only"])
            self.assertIn("iMessage only", cav["ask"])
            self.assertIn(f"Person {phase}", cav["ask"])  # code-emitted name

    def test_caveat_absent_on_healthy_phases(self):
        con = self._con()
        con.execute("INSERT INTO contacts(id, first_name) VALUES (9, 'Sam')")
        _insert_drift(con, "contact:9", label="Sam", phase="steady")
        con.commit()
        out = query_tools.get_relationship_drift(contact_id=9)
        self.assertNotIn("channel_caveat", out)


# --- watchlist coverage partition -------------------------------------------


class WatchlistPartitionTest(_TempDBTest):
    def setUp(self):
        super().setUp()
        con = self._con()
        # A LOW-coverage row with the HIGHEST score: it must never be ranked
        # against the mandatory-complete tier.
        _insert_drift(con, "contact:3", label="High-but-thin", phase="fading",
                      drift_score=0.95, mandatory_complete=0)
        # Mandatory-complete rows: the rankable tier.
        _insert_drift(con, "contact:2", label="Alex", phase="cooling",
                      drift_score=0.40, mandatory_complete=1,
                      signal_values={"latency": 0.35, "volume": 0.20},
                      signal_gates={"latency": 1, "volume": 1})
        _insert_drift(con, "contact:4", label="Jo", phase="fading",
                      drift_score=0.60, mandatory_complete=1,
                      signal_values={"reciprocity": 0.5},
                      signal_gates={"reciprocity": 1})
        con.commit()

    def test_never_ranks_across_coverage(self):
        out = query_tools.get_fadeout_watchlist()
        ranked_keys = [r["contact_key"] for r in out["watchlist"]]
        insufficient_keys = [r["contact_key"] for r in out["insufficient_signal"]]
        # The 0.95 low-coverage dyad is NOT in the ranked tier despite the top score.
        self.assertNotIn("contact:3", ranked_keys)
        self.assertIn("contact:3", insufficient_keys)
        # The ranked tier is ordered by score WITHIN mandatory-complete only.
        self.assertEqual(ranked_keys, ["contact:4", "contact:2"])

    def test_rows_expose_signal_count_and_top_driver(self):
        out = query_tools.get_fadeout_watchlist()
        by_key = {r["contact_key"]: r for r in out["watchlist"]}
        alex = by_key["contact:2"]
        self.assertEqual(alex["signal_count"], 2)          # latency + volume gated
        self.assertEqual(alex["top_driver"]["signal"], "latency")  # max value
        self.assertIn("signal_count", by_key["contact:4"])

    def test_declining_rows_carry_channel_caveat(self):
        out = query_tools.get_fadeout_watchlist()
        for r in out["watchlist"] + out["insufficient_signal"]:
            self.assertIn("channel_caveat", r)  # all seeded phases are declining

    def test_phase_filter_narrows_both_buckets(self):
        out = query_tools.get_fadeout_watchlist(phase_filter="fading")
        self.assertEqual([r["contact_key"] for r in out["watchlist"]], ["contact:4"])
        self.assertEqual([r["contact_key"] for r in out["insufficient_signal"]],
                         ["contact:3"])

    def test_min_messages_floors_dyad_size(self):
        con = self._con()
        _insert_drift(con, "contact:8", label="Tiny", phase="cooling",
                      drift_score=0.5, mandatory_complete=1, n_messages=40)
        con.commit()
        out = query_tools.get_fadeout_watchlist(min_messages=100)
        self.assertNotIn("contact:8",
                         [r["contact_key"] for r in out["watchlist"]])


# --- compactness (no backstop truncation on a max-width fixture) -------------


class CompactnessTest(_TempDBTest):
    def _full_series(self):
        months = [f"{y:04d}-{m:02d}" for y in (2021, 2022, 2023)
                  for m in range(1, 13)]  # 36 months, D1's cap
        return {s: [[mo, round(0.1 + i * 0.001, 6)] for i, mo in enumerate(months)]
                for s in _SIGNALS}

    def test_relationship_drift_payload_never_hits_backstop(self):
        con = self._con()
        con.execute("INSERT INTO contacts(id, first_name) VALUES (2, 'Alex')")
        con.execute("INSERT INTO handles(id, handle_identifier, contact_id) "
                    "VALUES (50, '+1', 2)")
        _insert_drift(con, "contact:2", label="Alex", phase="fading",
                      drift_score=0.7, confidence_tier="high",
                      signal_values={s: 0.5 for s in _SIGNALS},
                      signal_gates={s: 1 for s in _SIGNALS},
                      monthly_series=self._full_series())
        con.commit()

        out = query_tools.get_relationship_drift(contact_id=2)
        result_json = json.dumps(out)
        # The chat.run_turn backstop must be a NO-OP on this payload.
        self.assertEqual(chat._cap_tool_result(out, result_json), result_json)
        self.assertLessEqual(len(result_json), config.TOOL_RESULT_MAX_CHARS)
        # Every signal's evidence trail was trimmed in-tool.
        self.assertEqual(len(out["signals"]), len(_SIGNALS))
        for sig in out["signals"]:
            self.assertLessEqual(len(sig["monthly"]), _MAX_TRAIL_MONTHS)
        # The narration template rides the pre-computed tier.
        self.assertEqual(out["narration"]["confidence_tier"], "high")


# --- no dyad_drift table -----------------------------------------------------


class NoTableTest(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(self.db_path) and os.remove(self.db_path))
        con = sqlite3.connect(self.db_path)
        con.execute("CREATE TABLE chats(id INTEGER PRIMARY KEY, is_group INTEGER)")
        con.commit()
        con.close()
        self._saved = query_tools.DB_PATH
        query_tools.DB_PATH = self.db_path
        self.addCleanup(setattr, query_tools, "DB_PATH", self._saved)

    def test_read_tools_degrade_without_table(self):
        self.assertIn("error", query_tools.get_relationship_drift(contact_id=1))
        out = query_tools.get_fadeout_watchlist()
        self.assertEqual(out["watchlist"], [])
        self.assertEqual(out["insufficient_signal"], [])


# --- DRIFT_ENABLED schema-registration gate ---------------------------------


class RegistrationGateTest(unittest.TestCase):
    """Both schemas are registered ONLY when DRIFT_ENABLED; otherwise the tools
    do not exist to the model. Reloads tool_schemas + chat with the flag toggled
    (the schema list and SYSTEM_PROMPT lines are import-time gated)."""

    DRIFT_NAMES = {"get_relationship_drift", "get_fadeout_watchlist"}

    def setUp(self):
        self._saved = config.DRIFT_ENABLED
        self.addCleanup(self._reload, self._saved)  # restore last

    @staticmethod
    def _reload(enabled):
        config.DRIFT_ENABLED = enabled
        importlib.reload(tool_schemas)
        importlib.reload(chat)

    def test_disabled_registers_neither_schema(self):
        self._reload(False)
        names = {t["name"] for t in tool_schemas.TOOL_DEFINITIONS}
        self.assertFalse(self.DRIFT_NAMES & names)
        # chat's derived dispatch table and Ollama schema list follow suit.
        self.assertFalse(self.DRIFT_NAMES & set(chat.TOOL_FUNCTIONS))
        ollama_names = {t["function"]["name"] for t in chat.OLLAMA_TOOLS}
        self.assertFalse(self.DRIFT_NAMES & ollama_names)
        self.assertNotIn("get_relationship_drift", chat.SYSTEM_PROMPT)

    def test_enabled_registers_both_schemas_and_prompt_lines(self):
        self._reload(True)
        names = {t["name"] for t in tool_schemas.TOOL_DEFINITIONS}
        self.assertTrue(self.DRIFT_NAMES <= names)
        self.assertTrue(self.DRIFT_NAMES <= set(chat.TOOL_FUNCTIONS))
        ollama_names = {t["function"]["name"] for t in chat.OLLAMA_TOOLS}
        self.assertTrue(self.DRIFT_NAMES <= ollama_names)
        # SYSTEM_PROMPT advertises the tools AND the load-bearing caveat rule.
        self.assertIn("get_relationship_drift", chat.SYSTEM_PROMPT)
        self.assertIn("get_fadeout_watchlist", chat.SYSTEM_PROMPT)
        self.assertIn("channel_caveat", chat.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
