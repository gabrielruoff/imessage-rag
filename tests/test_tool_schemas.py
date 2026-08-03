"""Hermetic tests for tool_schemas.py: the shape of every TOOL_DEFINITIONS entry,
its parity with the query_tools function it names, its line in chat.py's
SYSTEM_PROMPT, and the flag-gated registration blocks.

tool_schemas.py:1-5 and chat.py:33-35 document a FOUR-way sync contract: a
query_tools function + its re-export + a schema here + a one-line summary in
chat.py's SYSTEM_PROMPT. Three legs already fail loudly (the re-export at import,
the schema-name -> function leg at chat.py's TOOL_FUNCTIONS getattr); the
SYSTEM_PROMPT leg is checked by nothing, so a new tool can ship invisible to the
model. That is what SystemPromptSyncTest exists for.

No Ollama, no real chat.db, no DB at all -- these read module-level data
structures and function signatures only. The gate tests use the reload dance
(mirroring tests/test_evalkit_selfcheck.py) because the registration blocks read
config flags AT IMPORT.
"""

import importlib
import inspect
import unittest

import chat
import query_tools
import tool_schemas

# Derived from GATED_TOOLS rather than hardcoded, so a fourth gated feature is
# picked up by the gate tests without editing them.
_FLAGS = tuple(tool_schemas.GATED_TOOLS)

# JSON Schema type names this repo's schemas are allowed to use.
_JSON_TYPES = ("string", "integer", "number", "boolean", "array", "object", "null")

# The ONE deliberate schema<-function asymmetry: exclude_chat_id is a real
# parameter of these four analytics functions but is intentionally NOT advertised
# to the model -- it is Wrapped's internal knob (wrapped.py:143,
# wrapped_sections.py:63,93). Pinned per-tool so a NEW omission has to be added
# here consciously instead of silently widening the exception.
_SCHEMA_OMISSIONS = {
    "get_activity_patterns": {"exclude_chat_id"},
    "get_reaction_stats": {"exclude_chat_id"},
    "get_word_trends": {"exclude_chat_id"},
    "get_word_frequency": {"exclude_chat_id"},
}


def _types_of(prop_schema):
    """A property's declared type(s) as a tuple -- a union list is legal here
    (e.g. _ID_OR_LIST_SCHEMA's ["integer", "array"])."""
    declared = prop_schema.get("type")
    return tuple(declared) if isinstance(declared, list) else (declared,)


def _matches_type(value, types):
    """JSON-Schema type check for a literal default. bool is handled before int
    because Python's bool subclasses int -- {"type": "integer", "default": False}
    is a real mistake this must catch, not an integer default."""
    if isinstance(value, bool):
        return "boolean" in types
    for name in types:
        if name == "integer" and isinstance(value, int):
            return True
        if name == "number" and isinstance(value, (int, float)):
            return True
        if name == "string" and isinstance(value, str):
            return True
        if name == "array" and isinstance(value, list):
            return True
        if name == "object" and isinstance(value, dict):
            return True
        if name == "null" and value is None:
            return True
    return False


def _named_params(func):
    """Parameter names of `func`, excluding *args/**kwargs -- a schema property
    must map to a real named parameter, never be absorbed by a catch-all."""
    return {
        name for name, param in inspect.signature(func).parameters.items()
        if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
    }


def _apply_flags(values):
    """Set the gate flags on the LIVE config module and rebuild the import-time
    gated state. tool_schemas rebuilds TOOL_DEFINITIONS + GATED_TOOLS; chat
    rebuilds SYSTEM_PROMPT / OLLAMA_TOOLS / TOOL_FUNCTIONS from it. Both are
    reloaded on restore too, or sibling test modules inherit a mutated module."""
    live_config = importlib.import_module("config")
    for flag, value in values.items():
        setattr(live_config, flag, value)
    importlib.reload(tool_schemas)
    importlib.reload(chat)


class ToolSchemaShapeTest(unittest.TestCase):
    """Structural contract for every registered schema. These run against
    whatever flag state the process is in -- a gated schema is held to the same
    shape as an unconditional one."""

    def test_every_definition_has_exactly_name_description_and_input_schema(self):
        for definition in tool_schemas.TOOL_DEFINITIONS:
            with self.subTest(tool=definition.get("name")):
                self.assertEqual(set(definition), {"name", "description", "input_schema"})
                self.assertIsInstance(definition["name"], str)
                self.assertTrue(definition["name"])
                self.assertIsInstance(definition["description"], str)
                self.assertTrue(definition["description"].strip())

    def test_every_tool_name_is_unique(self):
        names = [d["name"] for d in tool_schemas.TOOL_DEFINITIONS]
        self.assertEqual(len(names), len(set(names)), "duplicate tool name registered")

    def test_every_input_schema_is_an_object_with_properties(self):
        for definition in tool_schemas.TOOL_DEFINITIONS:
            with self.subTest(tool=definition["name"]):
                schema = definition["input_schema"]
                self.assertEqual(schema.get("type"), "object")
                self.assertIsInstance(schema.get("properties"), dict)
                self.assertLessEqual(set(schema), {"type", "properties", "required"})

    def test_every_required_name_is_present_in_properties(self):
        for definition in tool_schemas.TOOL_DEFINITIONS:
            schema = definition["input_schema"]
            for name in schema.get("required", []):
                with self.subTest(tool=definition["name"], param=name):
                    self.assertIn(name, schema["properties"])

    def test_every_property_declares_a_valid_json_schema_type(self):
        for definition in tool_schemas.TOOL_DEFINITIONS:
            for name, prop in definition["input_schema"]["properties"].items():
                with self.subTest(tool=definition["name"], param=name):
                    types = _types_of(prop)
                    self.assertTrue(types, "no type declared")
                    for declared in types:
                        self.assertIn(declared, _JSON_TYPES)

    def test_every_array_property_declares_its_items(self):
        # An untyped array is what makes Ollama/Apple schema conversion guess.
        for definition in tool_schemas.TOOL_DEFINITIONS:
            for name, prop in definition["input_schema"]["properties"].items():
                if "array" in _types_of(prop):
                    with self.subTest(tool=definition["name"], param=name):
                        self.assertIn("items", prop)
                        self.assertIn(prop["items"].get("type"), _JSON_TYPES)

    def test_every_enum_is_a_list_of_strings(self):
        for definition in tool_schemas.TOOL_DEFINITIONS:
            for name, prop in definition["input_schema"]["properties"].items():
                if "enum" in prop:
                    with self.subTest(tool=definition["name"], param=name):
                        self.assertIsInstance(prop["enum"], list)
                        self.assertTrue(prop["enum"])
                        for value in prop["enum"]:
                            self.assertIsInstance(value, str)

    def test_every_default_matches_its_declared_type(self):
        for definition in tool_schemas.TOOL_DEFINITIONS:
            for name, prop in definition["input_schema"]["properties"].items():
                if "default" not in prop:
                    continue
                with self.subTest(tool=definition["name"], param=name):
                    types = _types_of(prop)
                    self.assertTrue(
                        _matches_type(prop["default"], types),
                        f"default {prop['default']!r} is not a {types}",
                    )
                    if "enum" in prop:
                        self.assertIn(prop["default"], prop["enum"])


class SchemaFunctionParityTest(unittest.TestCase):
    """Schema <-> query_tools function parity: the model must never be told about
    an argument the function will reject, and a `required` param must genuinely
    have no Python default to fall back on."""

    def test_every_schema_name_resolves_to_a_query_tools_function(self):
        for definition in tool_schemas.TOOL_DEFINITIONS:
            with self.subTest(tool=definition["name"]):
                func = getattr(query_tools, definition["name"], None)
                self.assertTrue(callable(func), "not exported from query_tools")

    def test_every_schema_property_is_a_real_parameter_of_its_function(self):
        for definition in tool_schemas.TOOL_DEFINITIONS:
            params = _named_params(getattr(query_tools, definition["name"]))
            for name in definition["input_schema"]["properties"]:
                with self.subTest(tool=definition["name"], param=name):
                    self.assertIn(name, params)

    def test_every_required_property_has_no_python_default(self):
        for definition in tool_schemas.TOOL_DEFINITIONS:
            signature = inspect.signature(getattr(query_tools, definition["name"]))
            for name in definition["input_schema"].get("required", []):
                with self.subTest(tool=definition["name"], param=name):
                    self.assertIs(
                        signature.parameters[name].default, inspect.Parameter.empty,
                        "declared required but the function defaults it",
                    )

    def test_exclude_chat_id_is_the_only_deliberate_schema_omission(self):
        # Full symmetry is NOT asserted: the four Wrapped tools deliberately hide
        # exclude_chat_id from the model. Pinned exactly, so hiding another
        # parameter is a conscious edit here rather than a silent divergence.
        omissions = {}
        for definition in tool_schemas.TOOL_DEFINITIONS:
            params = _named_params(getattr(query_tools, definition["name"]))
            missing = params - set(definition["input_schema"]["properties"])
            if missing:
                omissions[definition["name"]] = missing
        self.assertEqual(omissions, _SCHEMA_OMISSIONS)


class SystemPromptSyncTest(unittest.TestCase):
    """The fourth leg of the sync contract. Nothing else checks it: a tool whose
    SYSTEM_PROMPT line is missing is registered, callable, and invisible -- the
    model is never told it exists."""

    def setUp(self):
        live_config = importlib.import_module("config")
        self._saved = {flag: getattr(live_config, flag) for flag in _FLAGS}
        self.addCleanup(_apply_flags, self._saved)

    def test_every_active_tool_name_appears_in_the_system_prompt(self):
        # Checked in BOTH flag states: the gated tools carry their own conditional
        # prompt blocks in chat.py, which the default (all-off) run never reaches.
        for enabled in (False, True):
            _apply_flags({flag: enabled for flag in _FLAGS})
            for definition in tool_schemas.TOOL_DEFINITIONS:
                with self.subTest(flags=enabled, tool=definition["name"]):
                    # The BULLET form, not a bare substring: several names also
                    # appear in the prompt's prose guidance ("call an analytics
                    # tool (get_activity_patterns, ...)"), so a plain `in` check
                    # would pass for a tool that lost its own catalog line.
                    self.assertIn(f"- {definition['name']} (", chat.SYSTEM_PROMPT)

    def test_draft_apology_ships_unconditionally(self):
        # tool_schemas.py:1012-1013: draft_apology has NO config gate on purpose --
        # the DV floor is enforced inside the function (query_tools.conflict_dv),
        # not by withholding the schema. Pinned so nobody "hardens" it into a gate.
        for names in tool_schemas.GATED_TOOLS.values():
            self.assertNotIn("draft_apology", names)
        _apply_flags({flag: False for flag in _FLAGS})
        self.assertIn("draft_apology", {d["name"] for d in tool_schemas.TOOL_DEFINITIONS})
        self.assertIn("- draft_apology (", chat.SYSTEM_PROMPT)


class GateDerivationTest(unittest.TestCase):
    """Registration gating, looped GENERICALLY over GATED_TOOLS -- the per-feature
    gate tests elsewhere (test_drift_tools, test_conflict, test_voiceprint) each
    check one flag by hand, so a fourth gated feature would inherit no coverage.
    Withholding the schema is the repo's only reliable off-switch, so an
    accidentally-unconditional tool is a boundary failure, not a cosmetic one."""

    def setUp(self):
        live_config = importlib.import_module("config")
        self._saved = {flag: getattr(live_config, flag) for flag in _FLAGS}
        self.addCleanup(_apply_flags, self._saved)
        self._gated = {flag: set(names) for flag, names in tool_schemas.GATED_TOOLS.items()}

    def test_each_gated_tool_is_absent_when_its_flag_is_off(self):
        _apply_flags({flag: False for flag in _FLAGS})
        names = {d["name"] for d in tool_schemas.TOOL_DEFINITIONS}
        for flag, gated in self._gated.items():
            with self.subTest(flag=flag):
                self.assertFalse(gated & names, "registered with its flag off")
        # chat.py derives its dispatch table from the same list, so nothing gated
        # is callable either.
        self.assertFalse(set().union(*self._gated.values()) & set(chat.TOOL_FUNCTIONS))

    def test_each_gated_tool_is_registered_when_its_flag_is_on(self):
        _apply_flags({flag: False for flag in _FLAGS})
        unconditional = {d["name"] for d in tool_schemas.TOOL_DEFINITIONS}
        for flag, gated in self._gated.items():
            # One flag at a time: proves each block reads ITS OWN flag rather than
            # riding another feature's.
            _apply_flags({f: (f == flag) for f in _FLAGS})
            names = [d["name"] for d in tool_schemas.TOOL_DEFINITIONS]
            with self.subTest(flag=flag):
                self.assertEqual(len(names), len(set(names)))
                self.assertEqual(set(names), unconditional | gated)
                self.assertTrue(gated <= set(chat.TOOL_FUNCTIONS))
        _apply_flags({flag: True for flag in _FLAGS})
        all_names = {d["name"] for d in tool_schemas.TOOL_DEFINITIONS}
        self.assertEqual(all_names, unconditional.union(*self._gated.values()))


if __name__ == "__main__":
    unittest.main()
