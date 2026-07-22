"""Hermetic tests for chat_apple's staged FM orchestration.

No shim, no Ollama, no DB, no network: chat_apple._respond (the single model
seam) is swapped for a SCRIPTED fake that distinguishes the four stages by their
(schema, stream) shape and records every prompt it sees, and the tool functions
are stubbed like tests/test_chat.py's DedupGuardTest. chat.console is swapped for
a non-terminal Console so the streamed answer renders once, quietly.

Covers the load-bearing behaviors, all of which regress silently:
  - the route/fill/execute/digest/answer staging (happy path)
  - an oversized result is DIGESTED, never fed raw into a prompt
  - the loop is bounded by MAX_TOOL_ITERATIONS and still answers
  - an unknown tool / off-catalog route action does not crash the turn
  - the compact catalog and schema converter cover all 21 tools

Run from the repo root:  python3 -m unittest tests.test_chat_apple
"""

import inspect
import io
import json
import unittest

from rich.console import Console

import applefm
import chat
import chat_apple
import config
import tool_schemas
import wellbeing


class ScriptedRespond:
    """A scripted stand-in for chat_apple._respond. It tells the stages apart by
    argument shape -- stream=True is ANSWER, a schema whose properties carry
    `action` is ROUTE, any other schema is FILL, and no schema is DIGEST -- and
    records every prompt so a test can assert what did (and did not) reach the
    model."""

    def __init__(self, route_actions=(), route_default="answer", fill_args=None,
                 digest_reply="a relevant fact", answer_chunks=("the answer",)):
        self.route_actions = list(route_actions)
        self.route_default = route_default
        self.fill_args = {} if fill_args is None else fill_args
        self.digest_reply = digest_reply
        self.answer_chunks = list(answer_chunks)
        self.prompts = []
        self.route_calls = self.fill_calls = self.digest_calls = self.answer_calls = 0

    def __call__(self, system, prompt, schema=None, stream=False, max_tokens=None):
        self.prompts.append(prompt)
        if stream:
            self.answer_calls += 1
            return iter(self.answer_chunks)
        if schema is not None and "action" in (schema.get("properties") or {}):
            self.route_calls += 1
            action = self.route_actions.pop(0) if self.route_actions else self.route_default
            return json.dumps({"action": action})
        if schema is not None:
            self.fill_calls += 1
            return json.dumps(self.fill_args)
        self.digest_calls += 1
        return self.digest_reply


class AppleChatTest(unittest.TestCase):
    """Base: quiet console, restore the _respond seam / _recorder hook after."""

    def setUp(self):
        self._saved_console = chat.console
        chat.console = Console(file=io.StringIO())  # non-terminal: renders once
        self.addCleanup(setattr, chat, "console", self._saved_console)

        self.addCleanup(setattr, chat_apple, "_respond", chat_apple._respond)
        self.addCleanup(setattr, chat_apple, "_recorder", chat_apple._recorder)

    def install(self, fake):
        chat_apple._respond = fake
        return fake

    def stub_tool(self, name, fn):
        """Swap one query_tools function in chat's registry for a fake, restoring
        it after (mirrors DedupGuardTest). The name must be a real tool so the
        route enum accepts it and _TOOL_SPECS has its schema."""
        saved = chat.TOOL_FUNCTIONS.get(name)
        chat.TOOL_FUNCTIONS[name] = fn
        self.addCleanup(lambda: chat.TOOL_FUNCTIONS.__setitem__(name, saved)
                        if saved is not None else chat.TOOL_FUNCTIONS.pop(name, None))


class HappyPathTest(AppleChatTest):
    def test_route_fill_execute_answer(self):
        executed = []

        def fake_tool(**kwargs):
            executed.append(kwargs)
            return [{"contact_id": 7, "name": "Alice"}]  # small -> used verbatim

        self.stub_tool("search_contacts_by_name", fake_tool)
        fake = self.install(ScriptedRespond(
            route_actions=["search_contacts_by_name", "answer"],
            fill_args={"name": "Alice"},
            answer_chunks=("Alice is ", "contact 7."),
        ))
        messages = [{"role": "user", "content": "who is Alice?"}]
        reply = chat_apple.run_turn(messages)

        self.assertEqual(reply, "Alice is contact 7.")
        self.assertEqual(messages[-1], {"role": "assistant", "content": reply})
        # Deterministic pre-resolution queries the name first (limit=5), then the
        # routed loop runs it again with the model-filled args.
        self.assertEqual(executed, [{"name": "Alice", "limit": 5}, {"name": "Alice"}])
        self.assertEqual(fake.route_calls, 2)  # a tool, then answer
        self.assertEqual(fake.fill_calls, 1)
        self.assertEqual(fake.answer_calls, 1)
        self.assertEqual(fake.digest_calls, 0)  # small result never needs a digest session
        # The gathered fact reaches the answer prompt (the last one recorded).
        self.assertIn("Alice", fake.prompts[-1])

    def test_recorder_hook_captures_executed_tools(self):
        self.stub_tool("list_chats", lambda **kw: [{"chat_id": 3}])
        recorded = []
        chat_apple._recorder = lambda name, args: recorded.append((name, args))
        self.install(ScriptedRespond(
            route_actions=["list_chats", "answer"], fill_args={"query": "fam"},
        ))
        chat_apple.run_turn([{"role": "user", "content": "which chat?"}])
        self.assertEqual(recorded, [("list_chats", {"query": "fam"})])


class DigestOverflowTest(AppleChatTest):
    def test_oversized_result_is_digested_not_fed_raw(self):
        # A result whose JSON is far over APPLE_TOOL_RESULT_MAX_CHARS: many rows.
        big_rows = [{"id": i, "text": "x" * 120} for i in range(400)]
        big_json = json.dumps(big_rows, default=str)
        self.assertGreater(len(big_json), config.APPLE_TOOL_RESULT_MAX_CHARS)
        self.stub_tool("search_messages_sql", lambda **kw: big_rows)

        fake = self.install(ScriptedRespond(
            route_actions=["search_messages_sql", "answer"],
            digest_reply="found 400 messages about x",
            answer_chunks=("done",),
        ))
        messages = [{"role": "user", "content": "how many x messages?"}]
        reply = chat_apple.run_turn(messages)

        self.assertEqual(reply, "done")
        self.assertGreater(fake.digest_calls, 0)  # digest sessions actually ran
        # The RAW result JSON never appears whole in ANY prompt -- only its
        # <=budget slices reach the digester, and only digested facts reach the
        # route/fill/answer prompts.
        for prompt in fake.prompts:
            self.assertNotIn(big_json, prompt)
        # The digested facts feed the answer prompt.
        self.assertIn("found 400 messages", fake.prompts[-1])

    def test_oversized_envelope_preserves_population_metadata(self):
        # An oversized search envelope: _slice_result walks `rows` only, so
        # total_matching/returned/order/truncated must be prepended verbatim as
        # a fact -- otherwise the answerer can't report the true total or that
        # the sample was capped (exactly the case where it matters most).
        envelope = {
            "rows": [{"id": i, "text": "x" * 120} for i in range(400)],
            "total_matching": 9999, "returned": 50,
            "order": "newest-first", "truncated": True,
        }
        big_json = json.dumps(envelope, default=str)
        self.assertGreater(len(big_json), config.APPLE_TOOL_RESULT_MAX_CHARS)
        self.stub_tool("search_messages_sql", lambda **kw: envelope)

        fake = self.install(ScriptedRespond(
            route_actions=["search_messages_sql", "answer"],
            digest_reply="a matching message",
            answer_chunks=("done",),
        ))
        chat_apple.run_turn([{"role": "user", "content": "how many total?"}])

        # The raw envelope JSON never reaches any prompt whole...
        for prompt in fake.prompts:
            self.assertNotIn(big_json, prompt)
        # ...but the population metadata does, verbatim, in the answer prompt.
        answer_prompt = fake.prompts[-1]
        self.assertIn("9999", answer_prompt)  # true total, not the 400-row sample
        self.assertIn("total_matching", answer_prompt)
        self.assertIn("truncated", answer_prompt)


class IterationBoundTest(AppleChatTest):
    def test_loop_stops_at_max_iterations_and_answers(self):
        self._saved_max = config.MAX_TOOL_ITERATIONS
        config.MAX_TOOL_ITERATIONS = 3
        self.addCleanup(setattr, config, "MAX_TOOL_ITERATIONS", self._saved_max)

        executed = []
        self.stub_tool("search_messages_fts", lambda **kw: executed.append(kw) or [{"id": 1}])
        # route_default (queue empty) -> always a tool: the model never says
        # "answer", so only the iteration budget stops the loop.
        fake = self.install(ScriptedRespond(
            route_default="search_messages_fts", fill_args={"query": "hi"},
            answer_chunks=("final",),
        ))
        reply = chat_apple.run_turn([{"role": "user", "content": "search hi"}])

        self.assertEqual(reply, "final")
        self.assertEqual(fake.route_calls, 3)  # exactly MAX_TOOL_ITERATIONS routes
        self.assertEqual(fake.answer_calls, 1)  # then it answers regardless
        self.assertEqual(len(executed), 1)  # the dedup guard collapses the repeats


class BadRouteTest(AppleChatTest):
    def test_off_catalog_route_action_answers_without_crashing(self):
        fake = self.install(ScriptedRespond(
            route_actions=["not_a_real_tool"], route_default="not_a_real_tool",
            answer_chunks=("fallback answer",),
        ))
        reply = chat_apple.run_turn([{"role": "user", "content": "hello"}])
        self.assertEqual(reply, "fallback answer")
        self.assertEqual(fake.route_calls, 1)  # off-catalog -> straight to answer
        self.assertEqual(fake.fill_calls, 0)

    def test_unknown_tool_function_becomes_a_digestible_error(self):
        # A real catalog name whose function is absent from the registry: the
        # route enum accepts it, but _execute finds no fn and must not crash.
        saved = chat.TOOL_FUNCTIONS.pop("get_milestones")
        self.addCleanup(chat.TOOL_FUNCTIONS.__setitem__, "get_milestones", saved)
        fake = self.install(ScriptedRespond(
            route_actions=["get_milestones", "answer"], answer_chunks=("ok",),
        ))
        reply = chat_apple.run_turn([{"role": "user", "content": "first message?"}])
        self.assertEqual(reply, "ok")
        self.assertEqual(fake.fill_calls, 1)  # it did try to fill + execute


class CatalogTest(unittest.TestCase):
    def test_catalog_covers_every_tool(self):
        names = [t["name"] for t in tool_schemas.TOOL_DEFINITIONS]
        # Derived, not a magic number: the catalog + route enum must stay in
        # lockstep with the live toolset (which grows as tools are added), so
        # this can never silently drift the way a hardcoded count did.
        self.assertEqual(len(chat_apple._TOOL_SPECS), len(names))
        self.assertEqual(set(chat_apple._TOOL_SPECS), set(names))
        for name in names:
            self.assertIn(name, chat_apple.CATALOG)
            self.assertTrue(chat_apple._TOOL_SPECS[name]["summary"])
        # The route enum offers every tool plus the terminal "answer".
        enum = chat_apple._ROUTE_SCHEMA["properties"]["action"]["enum"]
        self.assertEqual(set(enum), set(names) | {"answer"})

    def test_summary_is_the_first_sentence(self):
        # search_messages_sql's description opens with a "... range." sentence.
        spec = chat_apple._TOOL_SPECS["search_messages_sql"]
        self.assertEqual(spec["summary"], "Filter messages by chat, sender, or date range")


class SchemaConverterTest(unittest.TestCase):
    def test_union_types_collapse_to_a_single_supported_type(self):
        # The three collapse rules the plan pins down.
        self.assertEqual(chat_apple._pick_type(["integer", "array"]), "integer")
        self.assertEqual(chat_apple._pick_type(["string", "array"]), "string")
        self.assertEqual(chat_apple._pick_type(["string", "integer"]), "string")

    def test_every_input_schema_maps_into_the_supported_subset(self):
        supported_scalars = {"string", "integer", "number", "boolean"}
        for t in tool_schemas.TOOL_DEFINITIONS:
            simplified = chat_apple._simplify_schema(t["input_schema"])
            self.assertEqual(simplified["type"], "object")
            original_props = set((t["input_schema"].get("properties") or {}))
            # Every property maps (none is dropped): full coverage of the types
            # actually used across all 21 schemas.
            self.assertEqual(set(simplified["properties"]), original_props)
            for pname, pschema in simplified["properties"].items():
                # A union list must have been collapsed to a single string type.
                self.assertIsInstance(pschema["type"], str)
                self.assertIn(pschema["type"], supported_scalars | {"array"})
                if pschema["type"] == "array":
                    self.assertIn(pschema["items"]["type"], supported_scalars)
                if "enum" in pschema:
                    self.assertEqual(pschema["type"], "string")
                    self.assertTrue(all(isinstance(e, str) for e in pschema["enum"]))
            for req in simplified.get("required", []):
                self.assertIn(req, simplified["properties"])  # required stays kept-only

    def test_description_is_preserved_and_capped(self):
        long_desc = "x" * (chat_apple._ARG_DESC_MAX + 50)
        schema = {
            "type": "object",
            "properties": {
                "chat_id": {"type": ["integer", "array"], "items": {"type": "integer"},
                            "description": "  " + long_desc + "  "},
                "granularity": {"type": "string", "enum": ["day", "month"],
                                "description": "bucket size"},
                "bare": {"type": "integer"},
            },
        }
        out = chat_apple._simplify_schema(schema)["properties"]
        # kept, stripped, and capped to _ARG_DESC_MAX
        self.assertEqual(out["chat_id"]["description"], "x" * chat_apple._ARG_DESC_MAX)
        # survives on an enum-bearing string property too
        self.assertEqual(out["granularity"]["description"], "bucket size")
        # a property with no description gets no description key
        self.assertNotIn("description", out["bare"])

    def test_real_schemas_carry_descriptions_through(self):
        # The regression this fixes: the FILL model was blind to what each arg
        # meant. get_activity_patterns.include's valid sections live only in its
        # description; assert that guidance now reaches the simplified schema.
        spec = chat_apple._TOOL_SPECS["get_activity_patterns"]
        self.assertIn("heatmap", spec["schema"]["properties"]["include"]["description"])


class GroundArgsTest(unittest.TestCase):
    """The code guardrail behind ARG_SYSTEM's 'never invent an id or date' rule:
    id/date filters not grounded in the gathered facts (or question) are dropped
    before the tool runs."""

    def test_empty_scratchpad_drops_every_id(self):
        # First call of a turn: no facts -> filters stripped -> whole-corpus call.
        out = chat_apple._ground_args({"chat_id": 1, "handle_id": 42}, "how many messages?", "", [])
        self.assertEqual(out, {})

    def test_grounded_id_survives_fabricated_dropped(self):
        sp = ["[search_contacts_by_name] contact_id=168, chat_id=42 for Jill"]
        out = chat_apple._ground_args(
            {"chat_id": 42, "handle_id": 999}, "when did Jill and I first text?", "", sp)
        self.assertEqual(out, {"chat_id": 42})  # 42 grounded, 999 fabricated

    def test_word_boundary_not_substring(self):
        # The prototype bug: "1" in "1234" matched. A word-boundary test must not.
        out = chat_apple._ground_args({"chat_id": 1}, "q", "", ["chat with id=1234"])
        self.assertEqual(out, {})

    def test_list_valued_id_filters_to_grounded_members(self):
        sp = ["chat_id=42 and chat_id=43 in the family thread"]
        out = chat_apple._ground_args({"chat_id": [42, 999, 43]}, "q", "", sp)
        self.assertEqual(out, {"chat_id": [42, 43]})

    def test_date_year_grounding(self):
        out = chat_apple._ground_args(
            {"date_from": "2024-01-01", "date_to": "2019-05-01"},
            "what did we say in 2024?", "", [])
        self.assertEqual(out, {"date_from": "2024-01-01"})  # 2019 ungrounded

    def test_non_id_date_keys_pass_untouched(self):
        args = {"limit": 50, "granularity": "day", "include": ["heatmap"],
                "is_from_me": True, "query": "beach"}
        out = chat_apple._ground_args(dict(args), "q", "", [])
        self.assertEqual(out, args)

    def test_id_grounded_via_summary_or_question(self):
        # Grounding corpus includes the prior-turn summary and the question.
        out = chat_apple._ground_args({"contact_id": 7}, "tell me about contact 7", "", [])
        self.assertEqual(out, {"contact_id": 7})
        out = chat_apple._ground_args({"contact_id": 7}, "q", "earlier we found contact_id 7", [])
        self.assertEqual(out, {"contact_id": 7})


class ExtractNamesTest(unittest.TestCase):
    """The model-free capitalized-name scan feeding pre-resolution."""

    def test_drops_question_words_keeps_names(self):
        self.assertEqual(
            chat_apple._extract_candidate_names("When did Jill and I first text?"),
            ["Jill"])
        self.assertEqual(
            chat_apple._extract_candidate_names("Who is Jill Handrahan?"),
            ["Jill Handrahan"])

    def test_dedup_and_multiple_names(self):
        self.assertEqual(
            chat_apple._extract_candidate_names("Who replies faster, me or Kyle? And Kyle again"),
            ["Kyle"])

    def test_no_names_returns_empty(self):
        self.assertEqual(
            chat_apple._extract_candidate_names("what hour of the day do i text the most?"), [])

    def test_caps_candidate_count(self):
        q = "A B C D E F G"  # single letters are dropped (len < 2); use real tokens
        many = "Al Bo Cy Di El Fo Gu"
        self.assertLessEqual(len(chat_apple._extract_candidate_names(many)),
                             chat_apple._MAX_PRERESOLVE_NAMES)
        self.assertEqual(chat_apple._extract_candidate_names(q), [])


class PreresolveTest(AppleChatTest):
    """Deterministic name->id seeding runs before the loop, using only the public
    directory tools, and emits clearly-labeled id facts."""

    def _fake_chat(self, contacts_fn, chats_fn=None):
        class _C:
            TOOL_FUNCTIONS = {"search_contacts_by_name": contacts_fn}
            console = chat.console
        if chats_fn is not None:
            _C.TOOL_FUNCTIONS["list_chats"] = chats_fn
        return _C

    def test_seeds_labeled_ids_with_direct_chat(self):
        calls = {"contacts": [], "chats": []}

        def contacts_fn(**kw):
            calls["contacts"].append(kw)
            return [{"id": 168, "first_name": "Jill", "last_name": "Handrahan",
                     "handles": [{"id": 5}]}]

        def chats_fn(**kw):
            calls["chats"].append(kw)
            return [{"id": 42, "is_group": 0}, {"id": 99, "is_group": 1}]

        facts = chat_apple._preresolve_entities(
            self._fake_chat(contacts_fn, chats_fn), "When did Jill and I first text?")
        self.assertEqual(len(facts), 1)
        f = facts[0]
        self.assertIn("contact_id=168", f)
        self.assertIn("handle_id=[5]", f)
        self.assertIn("1:1 chat_id=[42]", f)   # only the is_group=0 chat
        self.assertNotIn("99", f)              # the group chat is excluded
        self.assertEqual(calls["contacts"], [{"name": "Jill", "limit": 5}])
        self.assertEqual(calls["chats"], [{"contact_id": 168, "limit": 50}])

    def test_no_name_no_tool_calls(self):
        called = []
        facts = chat_apple._preresolve_entities(
            self._fake_chat(lambda **kw: called.append(kw) or []),
            "what hour of the day do i text the most?")
        self.assertEqual(facts, [])
        self.assertEqual(called, [])  # never queried the directory

    def test_unresolved_name_is_skipped(self):
        facts = chat_apple._preresolve_entities(
            self._fake_chat(lambda **kw: []), "Who is Nobody Here?")
        self.assertEqual(facts, [])

    def test_tool_exception_degrades_to_no_seed(self):
        def boom(**kw):
            raise RuntimeError("db down")
        facts = chat_apple._preresolve_entities(self._fake_chat(boom), "Who is Jill?")
        self.assertEqual(facts, [])

    def test_seed_grounds_a_filled_chat_id(self):
        # End-to-end: the seeded chat_id survives _ground_args on the next fill.
        facts = chat_apple._preresolve_entities(
            self._fake_chat(
                lambda **kw: [{"id": 168, "first_name": "Jill", "handles": []}],
                lambda **kw: [{"id": 42, "is_group": 0}]),
            "Who is Jill?")
        out = chat_apple._ground_args({"chat_id": 42}, "Who is Jill?", "", facts)
        self.assertEqual(out, {"chat_id": 42})  # grounded by the seed


class CompactionTest(AppleChatTest):
    """The scratchpad is compacted back under _SCRATCHPAD_BUDGET so a later
    route/fill/answer prompt (which also carries the ~3K-char catalog) never
    overflows the fixed FM window."""

    def test_under_budget_is_untouched_without_a_model_call(self):
        called = []
        self.install(lambda *a, **k: called.append(1) or "x")
        sp = ["short fact", "another short fact"]
        self.assertEqual(chat_apple._compact("q", sp), sp)
        self.assertEqual(called, [])  # no digest call when already under budget

    def test_over_budget_compacts_via_one_digest(self):
        self.install(lambda system, prompt, schema=None, stream=False, max_tokens=None: "COMPACTED")
        sp = ["f" * 4000, "g" * 4000]  # 8000 > _SCRATCHPAD_BUDGET
        self.assertEqual(chat_apple._compact("q", sp), ["COMPACTED"])

    def test_empty_or_overflowing_digest_falls_back_to_recent_facts(self):
        # A digest that returns "nothing relevant" (or itself overflows) must not
        # wipe the scratchpad -- keep the most recent facts under budget instead.
        self.install(lambda *a, **k: "nothing relevant")
        out = chat_apple._compact("q", ["f" * 4000, "g" * 4000])
        self.assertTrue(out)
        self.assertLessEqual(sum(len(x) for x in out), chat_apple._SCRATCHPAD_BUDGET)


class WindowOverflowTest(AppleChatTest):
    """A stage prompt that overflows the fixed 4,096-token window must degrade,
    never raise applefm.ContextOverflowError out of run_turn -- the real-shim
    failure a hermetic fake can only reach by raising it deliberately."""

    def test_route_overflow_falls_through_to_answer(self):
        def fake(system, prompt, schema=None, stream=False, max_tokens=None):
            if stream:
                return iter(("degraded answer",))
            if schema is not None and "action" in (schema.get("properties") or {}):
                raise applefm.ContextOverflowError("window exceeded")
            return json.dumps({})

        self.install(fake)
        reply = chat_apple.run_turn([{"role": "user", "content": "hi"}])
        self.assertEqual(reply, "degraded answer")  # answered, did not crash

    def test_answer_overflow_retries_then_degrades(self):
        state = {"answer": 0}

        def fake(system, prompt, schema=None, stream=False, max_tokens=None):
            if stream:
                state["answer"] += 1
                if state["answer"] < 3:
                    def boom():
                        raise applefm.ContextOverflowError("still too big")
                        yield  # unreachable -- makes boom a generator
                    return boom()
                return iter(("final degraded",))
            return json.dumps({"action": "answer"})  # route -> answer immediately

        self.install(fake)
        reply = chat_apple.run_turn([{"role": "user", "content": "hi"}])
        self.assertEqual(reply, "final degraded")
        self.assertEqual(state["answer"], 3)  # two overflows, then a reply


class GenerationRejectedTest(AppleChatTest):
    """A per-request content rejection (applefm.GenerationRejectedError -- the
    permissive guardrail still occasionally refuses, or an unsupported-locale
    input) must degrade exactly like a window overflow: skipped/answered, never
    raised out of run_turn. Mirrors WindowOverflowTest with the sibling error."""

    def test_route_rejection_falls_through_to_answer(self):
        def fake(system, prompt, schema=None, stream=False, max_tokens=None):
            if stream:
                return iter(("degraded answer",))
            if schema is not None and "action" in (schema.get("properties") or {}):
                raise applefm.GenerationRejectedError("Apple FM rejected the request")
            return json.dumps({})

        self.install(fake)
        reply = chat_apple.run_turn([{"role": "user", "content": "hi"}])
        self.assertEqual(reply, "degraded answer")  # answered, did not crash

    def test_answer_rejection_retries_then_degrades(self):
        state = {"answer": 0}

        def fake(system, prompt, schema=None, stream=False, max_tokens=None):
            if stream:
                state["answer"] += 1
                if state["answer"] < 3:
                    def boom():
                        raise applefm.GenerationRejectedError("still refused")
                        yield  # unreachable -- makes boom a generator
                    return boom()
                return iter(("final degraded",))
            return json.dumps({"action": "answer"})  # route -> answer immediately

        self.install(fake)
        reply = chat_apple.run_turn([{"role": "user", "content": "hi"}])
        self.assertEqual(reply, "final degraded")
        self.assertEqual(state["answer"], 3)  # two rejections, then a reply


class WellbeingFloorTest(AppleChatTest):
    """The apple path enforces chat.run_turn's code-level wellbeing floor:
    get_language_trajectory arms a whole-reply condition-name sanitize, and its
    reserved _epilogue/support_block are stripped BEFORE digest (never narrated
    as data) and printed verbatim AFTER the reply. All of this lives in
    chat.run_turn on the Ollama path; the apple dispatch bypasses it, so
    chat_apple must re-enforce it (regressions here are silent + high-stakes)."""

    _SUPPORT = "SUPPORT_RESOURCES_SENTINEL: reach out to someone you trust"

    def _stub_wb_tool(self, name="get_language_trajectory", narratable=None):
        """Stub a tool that returns the reserved wellbeing envelope keys."""
        payload = {"scope": "self", "flagged": True,
                   "_epilogue": self._SUPPORT, "support_block": self._SUPPORT}
        if narratable:
            payload.update(narratable)
        self.stub_tool(name, lambda **kw: payload)

    def _console_text(self):
        return chat.console.file.getvalue()

    def test_epilogue_and_support_block_never_reach_a_prompt(self):
        # Both reserved keys are code-delivery directives, not data: they must be
        # stripped before the result is digested, so they never enter the
        # digester or the answer prompt (where the model could narrate them).
        self._stub_wb_tool()
        fake = self.install(ScriptedRespond(
            route_actions=["get_language_trajectory", "answer"],
            answer_chunks=("all clear",),
        ))
        chat_apple.run_turn([{"role": "user", "content": "how am I doing?"}])
        for prompt in fake.prompts:
            self.assertNotIn("SUPPORT_RESOURCES_SENTINEL", prompt)

    def test_reply_condition_names_are_sanitized(self):
        # The whole assembled answer is scrubbed once get_language_trajectory has
        # run -- a condition name anywhere in the reply is redacted at sentence
        # granularity, matching chat.ollama_chat_stream's floor.
        self._stub_wb_tool()
        raw = "Your messages show clear signs of depression lately."
        expected, was = wellbeing.sanitize_condition_names(raw)
        self.assertTrue(was)  # guard: the fixture really is redactable
        self.install(ScriptedRespond(
            route_actions=["get_language_trajectory", "answer"],
            answer_chunks=(raw,),
        ))
        reply = chat_apple.run_turn([{"role": "user", "content": "how am I?"}])
        self.assertNotIn("depression", reply)          # scrubbed in the return...
        self.assertNotIn("depression", self._console_text())  # ...and on screen
        self.assertIn(self._SUPPORT, reply)            # epilogue still folded in

    def test_epilogue_printed_after_reply_and_deduped(self):
        # Two tools return the SAME support block; it is printed + folded exactly
        # once (order-preserving de-dup), never once per tool.
        self._stub_wb_tool("get_language_trajectory")
        self._stub_wb_tool("find_conflicts")
        self.install(ScriptedRespond(
            route_actions=["get_language_trajectory", "find_conflicts", "answer"],
            answer_chunks=("here is the answer",),
        ))
        reply = chat_apple.run_turn([{"role": "user", "content": "check in"}])
        self.assertEqual(reply.count(self._SUPPORT), 1)     # folded once
        self.assertTrue(reply.startswith("here is the answer"))
        self.assertTrue(reply.endswith(self._SUPPORT))      # after the reply
        self.assertEqual(self._console_text().count(self._SUPPORT), 1)

    def test_epilogue_survives_a_fully_degraded_answer(self):
        # Every answer attempt overflows -> reply is "" -- the support block must
        # STILL print (resources must survive even a failed answer).
        self._stub_wb_tool()

        def fake(system, prompt, schema=None, stream=False, max_tokens=None):
            if stream:
                def boom():
                    raise applefm.ContextOverflowError("too big")
                    yield  # unreachable -- makes boom a generator
                return boom()
            if schema is not None and "action" in (schema.get("properties") or {}):
                # route: fire the wb tool once, then answer
                action = "get_language_trajectory" if not fake.routed else "answer"
                fake.routed = True
                return json.dumps({"action": action})
            return json.dumps({})  # fill / any non-stream

        fake.routed = False
        self.install(fake)
        reply = chat_apple.run_turn([{"role": "user", "content": "how am I?"}])
        self.assertEqual(reply, self._SUPPORT)  # empty answer, epilogue-only
        self.assertIn(self._SUPPORT, self._console_text())

    def test_non_wellbeing_turn_is_unchanged(self):
        # Regression: no get_language_trajectory -> no sanitize, no epilogue. A
        # condition word in the answer passes through untouched and the reply is
        # exactly the streamed text (no block appended).
        self.stub_tool("search_contacts_by_name", lambda **kw: [{"contact_id": 1}])
        raw = "This casually mentions depression and must not be redacted here."
        self.install(ScriptedRespond(
            route_actions=["search_contacts_by_name", "answer"],
            answer_chunks=(raw,),
        ))
        reply = chat_apple.run_turn([{"role": "user", "content": "who?"}])
        self.assertEqual(reply, raw)  # no sanitize, no epilogue folded in


class RunTurnSignatureTest(unittest.TestCase):
    def test_run_turn_mirrors_chat_run_turn(self):
        self.assertEqual(
            inspect.signature(chat_apple.run_turn),
            inspect.signature(chat.run_turn),
        )


if __name__ == "__main__":
    unittest.main()
