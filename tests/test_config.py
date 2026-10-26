"""Hermetic tests for config.py -- the PARSING of the env knobs, not their
consumption (~40 `mock.patch.object(config, ...)` sites elsewhere already cover
consumption).

THE governing fact: every value in config.py resolves AT IMPORT. There are no
lazy accessors, so a `patch.dict(os.environ, ...)` applied AFTER import has no
effect whatsoever on the module constants. Two strategies are used here:

  1. PREFERRED -- call the pure helpers directly (`_think`,
     `_parse_json_dict_env`, `_nudge_kind_weights`, `_resolve`,
     `has_effort_levels`, `is_apple_model`) under a patched environment. Zero
     side effects, no reload, no bleed into sibling test modules.
  2. RELOAD (`_reloaded_config`) -- only where a module-level CONSTANT is
     genuinely the subject. The trap, copied from tests/test_nudges.py: patch
     `dotenv.load_dotenv` AT THE SOURCE, because reloading config re-runs its
     `from dotenv import load_dotenv` and would clobber a patch applied to
     config's own attribute -- and without the stub the test reads the
     DEVELOPER's .env and is not hermetic. Every reload restores the real
     environment and reloads again in a `finally`, or sibling test modules would
     see mutated values (both `unittest tests.test_config tests.test_nudges` and
     the reverse order must pass).

A second fact worth carrying, for anyone reaching for `mock.patch.object`: the
repo binds config two different ways, and only one of them is patchable on
`config` itself. Measured over the tree (app + tests, excluding .venv):

  * 22 modules do `from config import X`, which BINDS THE VALUE at their own
    import time -- 20 of them app modules. For those,
    `mock.patch.object(config, "X")` is a NO-OP; the patch must land on the
    CONSUMING module (e.g. `ollama.CHAT_THINK`, `chat.MAX_TOOL_ITERATIONS`).
  * 42 modules read `config.X` per call and ARE patchable on `config` directly.
    (5 modules do both, for different knobs.)

The counts move as the tree does; the two-shapes distinction does not.
"""

import contextlib
import importlib
import os
import pathlib
import re
import unittest

from unittest import mock

import config
import dotenv

# Every knob parsed with the STRICT form -- int/float(os.getenv("X", "default"))
# -- as of this commit: 42 of the 103 numeric knobs (the other 61 use the
# tolerant `os.getenv("X") or "default"`). The strict form dies at import on a
# set-but-EMPTY value, so this is a landmine inventory, not a style record; see
# EnvParsingTest.test_no_new_strict_form_knob_is_added for what it is asserted
# against and why the assertion is one-directional. Regenerate with:
#
#   python -c 'import re,pathlib;print(sorted(set(re.findall(
#     r"(?:int|float)\(os\.getenv\(\s*\"([A-Z][A-Z0-9_]*)\"\s*,",
#     pathlib.Path("config.py").read_text()))))'
_KNOWN_STRICT_KNOBS = frozenset({
    "ADJACENCY_MINUTES", "ANNOTATE_COMMIT_EVERY",
    "ANNOTATE_HYBRID_SEED_TARGET", "ANNOTATE_KNN_K", "ANNOTATE_SEED_TARGET",
    "ANNOTATE_TOP_DYADS", "ANNOTATE_WORKERS", "API_PORT",
    "APPLE_ANSWER_MAX_TOKENS", "APPLE_DIGEST_MAX_TOKENS",
    "APPLE_FILL_MAX_TOKENS", "APPLE_MAX_TOOL_ITERATIONS",
    "APPLE_NO_PROGRESS_LIMIT", "APPLE_ROUTE_MAX_TOKENS",
    "APPLE_TOOL_RESULT_MAX_CHARS", "CHAT_TIMEOUT", "CHUNK_GAP_MINUTES",
    "CHUNK_MAX_CHARS", "CHUNK_MIN_CHARS", "CHUNK_OVERLAP_MESSAGES",
    "DOUBLE_TEXT_MIN_MINUTES", "DYAD_TERM_MIN_LIFT", "DYAD_TERM_MIN_USES",
    "DYAD_TERM_TOP_K", "EMBED_BATCH_SIZE", "EMBED_COMMIT_EVERY",
    "EMBED_MAX_WORKERS", "INITIATION_GAP_HOURS", "LATE_NIGHT_END_HOUR",
    "LATE_NIGHT_START_HOUR", "MAX_TOOL_ITERATIONS", "OLLAMA_PORT",
    "OLLAMA_TIMEOUT", "PROGRESS_LOG_EVERY", "SESSION_GAP_MINUTES",
    "TOOL_RESULT_MAX_CHARS", "TOPIC_K", "TOPIC_KMEANS_ITERS",
    "TOPIC_LABEL_SAMPLES", "WRAPPED_SILENT_MIN_MESSAGES",
    "WRAPPED_SILENT_MONTHS", "WRAPPED_TOP_CONTACTS",
})


def _env_without(*names):
    """A `patch.dict` target holding the real environment MINUS `names`, so an
    "unset" assertion cannot be broken by whatever the developer exports."""
    env = {k: v for k, v in os.environ.items() if k not in names}
    return mock.patch.dict(os.environ, env, clear=True)


@contextlib.contextmanager
def _reloaded_config(**overrides):
    """Reload config.py with `overrides` applied to os.environ (a None value
    UNSETS the var) and .env loading stubbed at the dotenv source, then restore
    the environment and reload again so nothing leaks to sibling tests."""
    saved = dict(os.environ)
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        with mock.patch.object(dotenv, "load_dotenv", lambda *a, **k: False):
            importlib.reload(config)
        yield config
    finally:
        os.environ.clear()
        os.environ.update(saved)
        importlib.reload(config)  # re-reads the real .env, restores real values


class ThinkTriStateTest(unittest.TestCase):
    """config._think (config.py:71-78). `think` is TRI-STATE and the state has
    to survive to the request body: None OMITS the flag entirely (some models
    reject the option), False SENDS `think: false`, True forces a pass."""

    def test_an_unset_think_is_none_not_false(self):
        # assertIs, never assertFalse: assertFalse passes for BOTH None and
        # False and would hide exactly the bug this pins.
        with _env_without("CHAT_THINK"):
            self.assertIs(config._think("CHAT_THINK"), None)
        # ...and an explicit default is what an unset var falls back to.
        with _env_without("TOPIC_THINK"):
            self.assertIs(config._think("TOPIC_THINK", False), False)

    def test_an_empty_think_falls_back_to_the_inherited_default(self):
        # Set-but-empty and whitespace-only are both "unset" here, so
        # TOPIC_THINK= in a .env inherits CHAT_THINK rather than meaning false.
        for raw in ("", "   ", "\t\n"):
            with self.subTest(raw=raw):
                with mock.patch.dict(os.environ, {"TOPIC_THINK": raw}, clear=False):
                    self.assertIs(config._think("TOPIC_THINK", True), True)
                    self.assertIs(config._think("TOPIC_THINK"), None)

    def test_on_is_truthy_for_flags_but_false_for_think(self):
        # The highest-value asymmetry in the file: _think accepts only
        # ("1", "true", "yes") -- no "on" -- while every other bool in config.py
        # accepts "on" too. So one .env written in a consistent style gets two
        # different answers: DRIFT_ENABLED=on turns the feature ON, while
        # CHAT_THINK=on ACTIVELY SENDS think:false (it does not even omit the
        # flag). Both halves asserted together, on the real constants.
        with _reloaded_config(CHAT_THINK="on", DRIFT_ENABLED="on") as cfg:
            self.assertIs(cfg.DRIFT_ENABLED, True)
            self.assertIs(cfg.CHAT_THINK, False)
        # The words _think does accept, for contrast.
        for raw in ("1", "true", "yes", "TRUE", " Yes "):
            with self.subTest(raw=raw):
                with mock.patch.dict(os.environ, {"CHAT_THINK": raw}, clear=False):
                    self.assertIs(config._think("CHAT_THINK"), True)

    def test_an_unrecognized_value_sends_think_false_rather_than_omitting_it(self):
        # Anything non-empty and unrecognized collapses to False (send
        # think:false), NOT to None (omit) and NOT to the passed default.
        for raw in ("banana", "0", "false", "no", "off", "low"):
            with self.subTest(raw=raw):
                with mock.patch.dict(os.environ, {"CHAT_THINK": raw}, clear=False):
                    self.assertIs(config._think("CHAT_THINK"), False)
                    self.assertIs(config._think("CHAT_THINK", True), False)


class EnvParsingTest(unittest.TestCase):
    def test_a_whitespace_only_quality_is_not_the_default(self):
        # config.py:222 is (os.getenv(...) or "distilled").strip().lower(): the
        # `or` fires BEFORE the .strip(), so a whitespace-only value is TRUTHY,
        # survives the fallback, and strips down to "" -- not "distilled".
        with _reloaded_config(ANNOTATE_QUALITY="   ") as cfg:
            self.assertEqual(cfg.ANNOTATE_QUALITY, "")
            self.assertNotEqual(cfg.ANNOTATE_QUALITY, "distilled")
        # A set-but-EMPTY value is falsy and does reach the default.
        with _reloaded_config(ANNOTATE_QUALITY="") as cfg:
            self.assertEqual(cfg.ANNOTATE_QUALITY, "distilled")
        with _reloaded_config(ANNOTATE_QUALITY=None) as cfg:
            self.assertEqual(cfg.ANNOTATE_QUALITY, "distilled")

    def test_the_tolerant_int_form_reads_empty_as_unset(self):
        # int(os.getenv("X") or "default"): a set-but-empty var is falsy, so the
        # default is used and the import survives.
        with _reloaded_config(EMBED_DIM="", ANNOTATE_KNN_INDEX_CAP="") as cfg:
            self.assertEqual(cfg.EMBED_DIM, 0)
            self.assertEqual(cfg.ANNOTATE_KNN_INDEX_CAP, 800)
        with mock.patch.dict(os.environ, {"EMBED_DIM": ""}, clear=False):
            self.assertEqual(int(os.getenv("EMBED_DIM") or "0"), 0)

    def test_a_set_but_empty_strict_knob_kills_the_import(self):
        # int(os.getenv("X", "default")): a var that is SET BUT EMPTY returns ""
        # (the fallback never fires), int("") raises ValueError AT IMPORT, and
        # every CLI in the repo dies on `import config` -- a one-line .env typo
        # that takes the whole tool down with a traceback naming neither the
        # variable nor the file.
        #
        # Asserted by reloading the REAL module. An earlier version asserted
        # int(os.getenv("OLLAMA_PORT", "11434")) raises, which is a fact about
        # Python's int(), not about config.py, and could not fail for the
        # reason described above.
        for knob in ("OLLAMA_PORT", "CHUNK_MAX_CHARS", "TOPIC_K"):
            with self.subTest(knob=knob):
                with self.assertRaises(ValueError):
                    with _reloaded_config(**{knob: ""}):
                        pass
        # ...and the module survives it: _reloaded_config restores in a finally,
        # so a raising reload cannot leave siblings with a half-built config.
        self.assertEqual(config.OLLAMA_PORT, 11434)
        # The tolerant form over the same shape imports fine -- the two forms
        # genuinely differ, and which one a knob got is not cosmetic.
        with _reloaded_config(ANNOTATE_KNN_INDEX_CAP="") as cfg:
            self.assertIsInstance(cfg.ANNOTATE_KNN_INDEX_CAP, int)

    def test_no_new_strict_form_knob_is_added(self):
        # A SUBSET assertion, not equality, and the direction is the point:
        # migrating a knob strict -> tolerant is the fix for the defect above
        # and must not require editing this list, but ADDING a strict site
        # plants a new import-time landmine and should be a conscious edit.
        src = pathlib.Path(config.__file__).resolve().read_text()
        strict = set(re.findall(
            r'(?:int|float)\(os\.getenv\(\s*"([A-Z][A-Z0-9_]*)"\s*,', src))
        self.assertLessEqual(
            strict, _KNOWN_STRICT_KNOBS,
            "new strict-form knob(s): a set-but-empty value now kills the "
            "import. Prefer int(os.getenv(\"X\") or \"default\"), or add the "
            "name to _KNOWN_STRICT_KNOBS deliberately")
        # Both forms are genuinely live, so neither branch above is dead.
        tolerant = set(re.findall(
            r'(?:int|float)\(os\.getenv\(\s*"([A-Z][A-Z0-9_]*)"\s*\)\s*or\s*"', src))
        self.assertTrue(tolerant)
        self.assertEqual(strict & tolerant, set(), "a knob parsed both ways")

    def test_positive_int_env_reads_empty_as_unset_and_refuses_non_positive(self):
        with mock.patch.dict(os.environ, {"MCP_RESULT_MAX_CHARS": ""}, clear=False):
            self.assertEqual(config._positive_int_env("MCP_RESULT_MAX_CHARS", 40000), 40000)
        for bad in ("0", "-40000"):
            with self.subTest(value=bad), \
                    mock.patch.dict(os.environ, {"MCP_RESULT_MAX_CHARS": bad}, clear=False):
                with self.assertRaises(ValueError) as ctx:
                    config._positive_int_env("MCP_RESULT_MAX_CHARS", 40000)
                self.assertIn("MCP_RESULT_MAX_CHARS", str(ctx.exception))

    def test_parse_json_dict_env_raises_naming_the_variable(self):
        # config.py:496-509 fails LOUDLY at import on a malformed drift knob,
        # and the message names the var so the .env line is findable.
        with mock.patch.dict(os.environ, {"DRIFT_WEIGHTS": "{not json"}, clear=False):
            with self.assertRaises(ValueError) as ctx:
                config._parse_json_dict_env("DRIFT_WEIGHTS", "{}")
        self.assertIn("DRIFT_WEIGHTS", str(ctx.exception))
        # Valid JSON that is not an OBJECT is refused the same way.
        with mock.patch.dict(os.environ, {"DRIFT_PHASE_THRESHOLDS": "[1, 2]"}, clear=False):
            with self.assertRaises(ValueError) as ctx:
                config._parse_json_dict_env("DRIFT_PHASE_THRESHOLDS", "{}")
        self.assertIn("DRIFT_PHASE_THRESHOLDS", str(ctx.exception))
        # An unset var parses the shipped default string, which must be an object.
        with _env_without("DRIFT_WEIGHTS"):
            parsed = config._parse_json_dict_env(
                "DRIFT_WEIGHTS", config._DRIFT_WEIGHTS_DEFAULT)
        self.assertIsInstance(parsed, dict)
        self.assertIn("volume", parsed)

    def test_nudge_kind_weights_falls_back_to_defaults_on_malformed_json(self):
        # config.py:420-448 takes the OPPOSITE policy to _parse_json_dict_env on
        # purpose: a bad urgency weight must not take the whole import down.
        with _env_without("NUDGE_KIND_WEIGHTS"):
            defaults = config._nudge_kind_weights()
        for raw in ("{not json", "[1, 2]", '"a string"', "null", "7"):
            with self.subTest(raw=raw):
                with mock.patch.dict(os.environ, {"NUDGE_KIND_WEIGHTS": raw}, clear=False):
                    self.assertEqual(config._nudge_kind_weights(), defaults)
        # A whitespace-only value is stripped to empty and also means "unset".
        with mock.patch.dict(os.environ, {"NUDGE_KIND_WEIGHTS": "   "}, clear=False):
            self.assertEqual(config._nudge_kind_weights(), defaults)

    def test_nudge_kind_weights_merges_a_partial_override(self):
        with _env_without("NUDGE_KIND_WEIGHTS"):
            defaults = config._nudge_kind_weights()
        raw = '{"dormant_tie": 0.1, "unknown_kind": 2, "dropped_plan": "nope"}'
        with mock.patch.dict(os.environ, {"NUDGE_KIND_WEIGHTS": raw}, clear=False):
            merged = config._nudge_kind_weights()
        # Overridden, coerced to float.
        self.assertEqual(merged["dormant_tie"], 0.1)
        self.assertIsInstance(merged["dormant_tie"], float)
        # Untouched keys keep their defaults -- a partial override is partial.
        self.assertEqual(merged["unanswered_question"], defaults["unanswered_question"])
        self.assertEqual(merged["faded_initiator"], defaults["faded_initiator"])
        # An unknown kind is ADDED (harmless: nothing reads it)...
        self.assertEqual(merged["unknown_kind"], 2.0)
        # ...but a non-numeric value is skipped, leaving that kind's default.
        self.assertEqual(merged["dropped_plan"], defaults["dropped_plan"])


class SafetyDefaultTest(unittest.TestCase):
    def test_the_dv_screen_stays_on_for_an_unrecognized_value(self):
        # config.py:622-623 is INVERSE-default -- `not in ("0","off","false","no")`
        # -- so a typo, a nonsense word, or a value from another convention
        # leaves the DV screen ON. That is the correct direction for a safety
        # gate and the opposite of every other bool in the file; pinned here so
        # nobody "normalizes" it into the `in (...)` form used elsewhere.
        for raw in ("banana", "", "   ", "on", "true", "1", "yes", "ON"):
            with self.subTest(raw=raw):
                with _reloaded_config(CONFLICT_DV_SCREEN=raw) as cfg:
                    self.assertIs(cfg.CONFLICT_DV_SCREEN, True)
        # Only the four explicit off-words turn it off.
        for raw in ("0", "off", "false", "no", " OFF "):
            with self.subTest(raw=raw):
                with _reloaded_config(CONFLICT_DV_SCREEN=raw) as cfg:
                    self.assertIs(cfg.CONFLICT_DV_SCREEN, False)
        # Unset is on: the shipped default.
        with _reloaded_config(CONFLICT_DV_SCREEN=None) as cfg:
            self.assertIs(cfg.CONFLICT_DV_SCREEN, True)


class PredicateTest(unittest.TestCase):
    def test_has_effort_levels_is_true_only_for_the_named_family(self):
        # _EFFORT_LEVEL_MODELS is the small NAMED set; everything else is
        # assumed to merely toggle, so an effort string buys it a full pass.
        self.assertEqual(config._EFFORT_LEVEL_MODELS, ("gpt-oss",))
        for name in ("gpt-oss", "gpt-oss:20b", "GPT-OSS:120b", "ollama/gpt-oss"):
            with self.subTest(name=name):
                self.assertTrue(config.has_effort_levels(name))
        for name in ("qwen3.5:4b", "gemma4", "apple", "", "gptoss", None, 7):
            with self.subTest(name=name):
                self.assertFalse(config.has_effort_levels(name))

    def test_is_apple_model_matches_the_sentinel_and_its_prefix(self):
        self.assertTrue(config.is_apple_model("apple"))
        self.assertTrue(config.is_apple_model("apple:"))
        self.assertTrue(config.is_apple_model("apple:on-device"))
        # A near-miss is NOT the sentinel: no substring or case folding.
        for name in ("Apple", "apple-fm", "applefm", "my-apple", " apple", "apples"):
            with self.subTest(name=name):
                self.assertFalse(config.is_apple_model(name))

    def test_is_apple_model_is_false_for_a_non_string(self):
        # Every dispatch seam gates on this, so a None/unset *_MODEL must not
        # raise its way through ollama.py / embeddings.py / chat.py.
        for value in (None, 0, 7, [], {}, object()):
            with self.subTest(value=value):
                self.assertFalse(config.is_apple_model(value))


class DotenvAnchoringTest(unittest.TestCase):
    def test_dotenv_is_loaded_from_the_config_directory_only(self):
        # config.py:19-22. A BARE load_dotenv() walks UP the tree, so a checkout
        # under .claude/worktrees/<name>/ would silently inherit the parent
        # checkout's flags, models and DB paths. This repo literally runs from a
        # worktree, so the invariant is live. Record the call and pin the path.
        expected = pathlib.Path(config.__file__).resolve().parent / ".env"
        recorder = mock.Mock(return_value=False)
        saved = dict(os.environ)
        try:
            with mock.patch.object(dotenv, "load_dotenv", recorder):
                importlib.reload(config)
        finally:
            os.environ.clear()
            os.environ.update(saved)
            importlib.reload(config)
        self.assertEqual(recorder.call_count, 1)
        self.assertEqual(recorder.call_args, mock.call(expected))
        # Explicitly NOT the bare, tree-walking call.
        self.assertNotEqual(recorder.call_args, mock.call())

    def test_a_relative_db_path_anchors_to_the_config_directory(self):
        # config._resolve (config.py:25-27): relative values hang off config.py's
        # own directory, never the CWD, so every CLI opens the same DB whatever
        # it is launched from. An ABSOLUTE env value still wins.
        root = pathlib.Path(config.__file__).resolve().parent
        self.assertEqual(config._resolve("imessage_analysis.db"),
                         root / "imessage_analysis.db")
        self.assertEqual(config._resolve("source_data/chat.db"),
                         root / "source_data" / "chat.db")
        self.assertEqual(config._resolve("/tmp/elsewhere.db"),
                         pathlib.Path("/tmp/elsewhere.db"))
        # And the shipped constants really are anchored.
        self.assertEqual(config.TARGET_DB.parent, root)
        self.assertEqual(config.APP_DB.parent, root)


class EnvExampleParityTest(unittest.TestCase):
    def test_every_knob_is_documented_in_env_example(self):
        # Parity is EXACT in both directions today: a knob nobody can discover
        # is a knob nobody uses, and a documented line for a knob that no longer
        # exists is a lie that `make env` copies verbatim.
        #
        # The scanner must recognize the HELPER forms too. A naive
        # `os.getenv("NAME"` regex reports 9 false positives -- CHAT_THINK,
        # CONFLICT_THINK, NUDGES_THINK, TOPIC_THINK, WRAPPED_THINK,
        # DRIFT_WEIGHTS, DRIFT_PHASE_THRESHOLDS, MCP_RESULT_MAX_CHARS,
        # TOOL_RESULT_MAX_CHARS -- because those are read through _think(...)
        # / _parse_json_dict_env(...) / _positive_int_env(...) instead.
        root = pathlib.Path(config.__file__).resolve().parent
        src = (root / "config.py").read_text()
        declared = set(re.findall(
            r'(?:os\.getenv|_think|_parse_json_dict_env|_positive_int_env)\(\s*"([A-Z][A-Z0-9_]*)"', src))
        naive = set(re.findall(r'os\.getenv\(\s*"([A-Z][A-Z0-9_]*)"', src))
        self.assertEqual(
            sorted(declared - naive),
            ["CHAT_THINK", "CONFLICT_THINK", "DRIFT_PHASE_THRESHOLDS",
             "DRIFT_WEIGHTS", "HOSTED_LIMIT_RECOVER_AFTER", "HOSTED_MAX_PARALLEL",
             "MCP_RESULT_MAX_CHARS", "NUDGES_THINK",
             "TOOL_RESULT_MAX_CHARS", "TOPIC_THINK", "WRAPPED_THINK"])

        # .env.example is a bare commented inventory: `# VAR=default` lines,
        # every one of which must STAY commented (make env copies it verbatim).
        example = (root / ".env.example").read_text()
        documented = set(re.findall(r'^#\s*([A-Z][A-Z0-9_]*)=', example, re.M))
        self.assertEqual(sorted(declared - documented), [],
                         "knob(s) read by config.py but absent from .env.example")
        self.assertEqual(sorted(documented - declared), [],
                         "knob(s) documented in .env.example but not read by config.py")
        self.assertGreater(len(declared), 100)

    def test_no_env_example_line_is_left_uncommented(self):
        root = pathlib.Path(config.__file__).resolve().parent
        live = [ln for ln in (root / ".env.example").read_text().splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")]
        self.assertEqual(live, [], "make env copies verbatim: an uncommented "
                                   "line becomes a live override")


class HostedConfigTest(unittest.TestCase):
    def test_hosted_predicates_match_only_their_prefix(self):
        self.assertTrue(config.is_openrouter_model("openrouter/anthropic/claude-haiku-5.5"))
        self.assertTrue(config.is_voyage_model("voyage/voyage-4-large"))
        for name in ("voyage-4-large", "qwen3.5:4b", "apple", "", None, 3):
            with self.subTest(name=name):
                self.assertFalse(config.is_openrouter_model(name))
                self.assertFalse(config.is_voyage_model(name))
        self.assertFalse(config.is_openrouter_model("voyage/voyage-4-large"))
        self.assertFalse(config.is_voyage_model("openrouter/x"))

    def test_hosted_model_id_strips_only_the_backend_prefix(self):
        self.assertEqual(config.hosted_model_id("openrouter/anthropic/claude-haiku-5.5"),
                         "anthropic/claude-haiku-5.5")
        self.assertEqual(config.hosted_model_id("voyage/voyage-4-large"), "voyage-4-large")
        self.assertEqual(config.hosted_model_id("qwen3.5:4b"), "qwen3.5:4b")

    def test_hosted_budget_has_no_default_and_refuses_non_positive(self):
        for unset in (None, "", "  "):
            with self.subTest(value=unset), _reloaded_config(HOSTED_BUDGET_USD=unset) as cfg:
                self.assertIsNone(cfg.HOSTED_BUDGET_USD)
        with _reloaded_config(HOSTED_BUDGET_USD="2.5") as cfg:
            self.assertEqual(cfg.HOSTED_BUDGET_USD, 2.5)
        for bad in ("0", "-1", "nan", "inf", "-inf"):
            with self.subTest(value=bad), self.assertRaises(ValueError) as ctx:
                with _reloaded_config(HOSTED_BUDGET_USD=bad):
                    pass
            self.assertIn("HOSTED_BUDGET_USD", str(ctx.exception))

    def test_hosted_max_parallel_defaults_to_sequential_and_refuses_below_one(self):
        with _reloaded_config(HOSTED_MAX_PARALLEL=None) as cfg:
            self.assertEqual(cfg.HOSTED_MAX_PARALLEL, 1)
        with self.assertRaises(ValueError) as ctx:
            with _reloaded_config(HOSTED_MAX_PARALLEL="0"):
                pass
        self.assertIn("HOSTED_MAX_PARALLEL", str(ctx.exception))

    def test_voyage_embed_dim_defaults_to_1024_and_refuses_unsupported_values(self):
        with _reloaded_config(EMBED_MODEL="voyage/voyage-4-large", EMBED_DIM=None) as cfg:
            self.assertEqual(cfg.EMBED_DIM, 1024)
        with _reloaded_config(EMBED_MODEL="voyage/voyage-4-large", EMBED_DIM="512") as cfg:
            self.assertEqual(cfg.EMBED_DIM, 512)
        with self.assertRaises(ValueError) as ctx:
            with _reloaded_config(EMBED_MODEL="voyage/voyage-4-large", EMBED_DIM="768"):
                pass
        self.assertIn("EMBED_DIM", str(ctx.exception))
        # A local model keeps the old meaning: 0 = probe, any value = truncate.
        with _reloaded_config(EMBED_MODEL="nomic-embed-text", EMBED_DIM=None) as cfg:
            self.assertEqual(cfg.EMBED_DIM, 0)
        with _reloaded_config(EMBED_MODEL="nomic-embed-text", EMBED_DIM="768") as cfg:
            self.assertEqual(cfg.EMBED_DIM, 768)


if __name__ == "__main__":
    unittest.main()
