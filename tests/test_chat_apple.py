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
        # Apple path reads its OWN bound (APPLE_MAX_TOOL_ITERATIONS), not the
        # shared MAX_TOOL_ITERATIONS. Disable the no-progress backoff so the
        # iteration budget is what stops the loop.
        self.addCleanup(setattr, config, "APPLE_MAX_TOOL_ITERATIONS",
                        config.APPLE_MAX_TOOL_ITERATIONS)
        self.addCleanup(setattr, config, "APPLE_NO_PROGRESS_LIMIT",
                        config.APPLE_NO_PROGRESS_LIMIT)
        config.APPLE_MAX_TOOL_ITERATIONS = 3
        config.APPLE_NO_PROGRESS_LIMIT = 99  # never backs off in this test

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
        self.assertEqual(fake.route_calls, 3)  # exactly APPLE_MAX_TOOL_ITERATIONS routes
        self.assertEqual(fake.answer_calls, 1)  # then it answers regardless
        self.assertEqual(len(executed), 1)  # the dedup guard collapses the repeats


class BackoffAndIntentTest(AppleChatTest):
    def setUp(self):
        super().setUp()
        for k in ("APPLE_MAX_TOOL_ITERATIONS", "APPLE_NO_PROGRESS_LIMIT"):
            self.addCleanup(setattr, config, k, getattr(config, k))

    def test_no_progress_backoff_stops_early_without_polluting(self):
        # Every call is an exact repeat -> dud after the first; the loop must stop
        # after APPLE_NO_PROGRESS_LIMIT no-signal iterations, well short of the
        # bound, and never append a duplicate_call/error note to the scratchpad.
        config.APPLE_MAX_TOOL_ITERATIONS = 20
        config.APPLE_NO_PROGRESS_LIMIT = 2
        self.stub_tool("search_messages_fts", lambda **kw: [])  # always zero rows
        fake = self.install(ScriptedRespond(
            route_default="search_messages_fts", fill_args={"query": "hi"},
            answer_chunks=("done",),
        ))
        reply = chat_apple.run_turn([{"role": "user", "content": "search hi"}])
        self.assertEqual(reply, "done")
        # 2 no-signal iterations then break -- not the full 20.
        self.assertEqual(fake.route_calls, 2)
        # No dud fact reached the answer prompt.
        self.assertNotIn("duplicate_call", fake.prompts[-1])
        self.assertNotIn("error", fake.prompts[-1])

    def test_intent_forces_first_action_without_routing_it(self):
        config.APPLE_MAX_TOOL_ITERATIONS = 8
        config.APPLE_NO_PROGRESS_LIMIT = 2
        executed = []
        self.stub_tool("get_chat_summary_stats",
                       lambda **kw: executed.append(kw) or {"message_count": 42})
        # The router queue would say "answer" first -- but the intent rule forces
        # get_chat_summary_stats as iteration 0 BEFORE any _route call. Its result
        # is a structured aggregate, so _derive_summary_facts computes the answer
        # in code and ENDS the loop (no second route call) -- the derive-and-stop
        # contract that keeps the small model from confabulating over the result.
        fake = self.install(ScriptedRespond(
            route_actions=["answer"], fill_args={}, answer_chunks=("42 total",),
        ))
        reply = chat_apple.run_turn(
            [{"role": "user", "content": "how many total messages do I have?"}])
        self.assertEqual(reply, "42 total")
        self.assertEqual(len(executed), 1)          # the forced tool ran
        self.assertEqual(fake.route_calls, 0)       # derived fact ended the loop
        self.assertEqual(fake.fill_calls, 1)        # filled the forced tool
        # The code-derived fact (not the raw result) reached the answer prompt.
        self.assertIn("42 messages", fake.prompts[-1])
        self.assertIn("[derived]", fake.prompts[-1])

    def test_forced_action_retries_with_defaults_when_fill_arg_errors(self):
        # The FM arg-filler decorates the forced tool with a spurious filter that
        # errors it; the loop must retry the (correct) tool with defaults rather
        # than fall through to search/refusal.
        config.APPLE_MAX_TOOL_ITERATIONS = 8
        config.APPLE_NO_PROGRESS_LIMIT = 2
        calls = []

        def reaction_tool(**kw):
            calls.append(kw)
            if kw:  # spurious FILL arg (survives grounding) -> error
                return {"error": "bad filter"}
            return {"given": {"by_type": {"like": 6879, "laugh": 1998}}}
        self.stub_tool("get_reaction_stats", reaction_tool)
        # is_from_me is a shape arg _ground_args keeps (not an enum narrowing), so
        # it reaches the tool and errors -- exercising the retry-on-error path.
        fake = self.install(ScriptedRespond(
            route_actions=["answer"], fill_args={"is_from_me": True},
            answer_chunks=("You use Liked most.",)))
        chat_apple.run_turn(
            [{"role": "user", "content": "which tapback do I use the most?"}])
        # First call carried the bad arg; the retry ran with {} and succeeded.
        self.assertEqual(calls, [{"is_from_me": True}, {}])
        # The deriver over the defaults result reached the answer prompt.
        self.assertIn("Liked", fake.prompts[-1])

    def test_forced_retry_keeps_grounded_scope_drops_shape_args(self):
        # The retry sheds ONLY ungated shape args; a grounded contact_id is part of
        # the question and must survive, else the retry computes whole-corpus stats
        # and misattributes them to the named contact.
        config.APPLE_MAX_TOOL_ITERATIONS = 8
        config.APPLE_NO_PROGRESS_LIMIT = 2
        calls = []

        def reaction_tool(**kw):
            calls.append(kw)
            if "is_from_me" in kw:  # the spurious shape arg -> error
                return {"error": "bad arg"}
            return {"given": {"by_type": {"like": 6879}}}
        self.stub_tool("get_reaction_stats", reaction_tool)
        # contact_id 7 is grounded ("7" appears in the question) so it survives
        # _ground_args AND the retry; is_from_me is the sheddable shape arg.
        fake = self.install(ScriptedRespond(
            route_actions=["answer"], fill_args={"contact_id": 7, "is_from_me": True},
            answer_chunks=("Liked most.",)))
        chat_apple.run_turn(
            [{"role": "user",
              "content": "which tapback does contact 7 use the most?"}])
        self.assertEqual(calls, [{"contact_id": 7, "is_from_me": True},
                                 {"contact_id": 7}])
        self.assertIn("Liked", fake.prompts[-1])

    def test_forced_error_with_only_grounded_args_does_not_retry(self):
        # Every arg is grounded/asked-for -> nothing sheddable -> no retry at all
        # (a retry could only drop the scope and answer a different question).
        config.APPLE_MAX_TOOL_ITERATIONS = 8
        config.APPLE_NO_PROGRESS_LIMIT = 2
        calls = []
        self.stub_tool("get_reaction_stats",
                       lambda **kw: calls.append(kw) or {"error": "boom"})
        self.install(ScriptedRespond(
            route_actions=["answer"], fill_args={"contact_id": 7},
            answer_chunks=("couldn't find it",)))
        chat_apple.run_turn(
            [{"role": "user",
              "content": "which tapback does contact 7 use the most?"}])
        self.assertEqual(calls, [{"contact_id": 7}])

    def test_forced_contact_scoped_intent_fills_scope_deterministically(self):
        # A named person + a contact-scoped intent: the resolved contact_id is
        # injected in code (no FM fill for the scope), sidestepping the
        # contact_id/chat_id conflation.
        config.APPLE_MAX_TOOL_ITERATIONS = 8
        config.APPLE_NO_PROGRESS_LIMIT = 2
        self.stub_tool("search_contacts_by_name",
                       lambda **kw: [{"id": 168, "first_name": "Jill", "handles": []}])
        self.stub_tool("list_chats", lambda **kw: [{"id": 43, "is_group": 0}])
        executed = []
        self.stub_tool("get_milestones",
                       lambda **kw: executed.append(kw) or {"firsts": {"first": {"date": "2019"}}})
        fake = self.install(ScriptedRespond(
            route_actions=["answer"], answer_chunks=("2019, Jill first",)))
        chat_apple.run_turn(
            [{"role": "user", "content": "When did Jill and I first text?"}])
        # contact_id filled from pre-resolution, NOT a fabricated/ conflated id;
        # no model fill call was needed for the forced step.
        self.assertEqual(executed, [{"contact_id": 168}])
        self.assertEqual(fake.fill_calls, 0)

    def test_forced_contact_scoped_error_does_not_retry_without_scope(self):
        # A CODE-filled contact scope must not be dropped on error: retrying with {}
        # would recompute whole-corpus stats and misattribute them to the person.
        config.APPLE_MAX_TOOL_ITERATIONS = 8
        config.APPLE_NO_PROGRESS_LIMIT = 2
        self.stub_tool("search_contacts_by_name",
                       lambda **kw: [{"id": 168, "first_name": "Jill", "handles": []}])
        self.stub_tool("list_chats", lambda **kw: [{"id": 43, "is_group": 0}])
        calls = []
        self.stub_tool("get_milestones",
                       lambda **kw: calls.append(kw) or {"error": "boom"})
        self.install(ScriptedRespond(
            route_actions=["answer"], answer_chunks=("couldn't find it",)))
        chat_apple.run_turn(
            [{"role": "user", "content": "When did Jill and I first text?"}])
        # Exactly one call, WITH the scope -- no {} whole-corpus retry.
        self.assertEqual(calls, [{"contact_id": 168}])

    def test_forced_group_intent_fills_group_chat_id_from_preresolution(self):
        # A named GROUP + a group-scoped intent: list_chats resolves the group name
        # to its chat_id, which is injected in code (get_group_dynamics errors
        # without a group scope, so this is what stops the S6 confabulation).
        config.APPLE_MAX_TOOL_ITERATIONS = 8
        config.APPLE_NO_PROGRESS_LIMIT = 2
        self.stub_tool("search_contacts_by_name", lambda **kw: [])  # not a person
        self.stub_tool("list_chats", lambda **kw:
                       [{"id": 990, "display_name": "Real Tahoe", "is_group": 1}]
                       if kw.get("query") else [])
        executed = []
        self.stub_tool("get_group_dynamics", lambda **kw:
                       executed.append(kw) or {"members": [{"member": "Me", "share": 0.30}]})
        fake = self.install(ScriptedRespond(
            route_actions=["answer"], answer_chunks=("Me carries it.",)))
        chat_apple.run_turn([{"role": "user", "content": "Who carries Real Tahoe?"}])
        self.assertEqual(executed, [{"chat_id": 990}])  # group scope filled in code
        self.assertEqual(fake.fill_calls, 0)


class SuggestFirstActionTest(unittest.TestCase):
    def test_canonical_intents_map_to_tools(self):
        cases = {
            "how many total messages have I sent and received?": "get_chat_summary_stats",
            "what is my total number of texts?": "get_chat_summary_stats",
            "what hour of the day do I text the most?": "get_activity_patterns",
            "who is my busiest day with?": "get_activity_patterns",
            "when did Jill and I first text?": "get_milestones",
            "who texted first between me and Kyle?": "get_milestones",
            "what was our 10,000th message?": "get_milestones",
            "who replies faster, me or Kyle?": "get_response_dynamics",
            "how long does Teddy take to reply?": "get_response_dynamics",
            "does she leave me on read?": "get_response_dynamics",
            # Specific intents phrased with a generic-frequency opener must not be
            # stolen by the activity "how often do i" rule (ordering invariant).
            "how often do I ghost people?": "get_response_dynamics",
            "how often does she leave me on read?": "get_response_dynamics",
            "how often do we text?": "get_activity_patterns",
        }
        for q, tool in cases.items():
            self.assertEqual(chat_apple._suggest_first_action(q), tool, q)

    def test_non_matching_questions_return_none(self):
        for q in ["who is Jill Handrahan?", "find messages about the beach",
                  "what did Kyle say about his job?", ""]:
            self.assertIsNone(chat_apple._suggest_first_action(q), q)


class ProgressSignalTest(unittest.TestCase):
    def test_is_dud(self):
        self.assertTrue(chat_apple._is_dud({"duplicate_call": True, "note": "x"}))
        self.assertTrue(chat_apple._is_dud({"error": "boom"}))
        self.assertFalse(chat_apple._is_dud({"message_count": 5}))
        self.assertFalse(chat_apple._is_dud([{"id": 1}]))

    def test_has_signal(self):
        self.assertTrue(chat_apple._has_signal([{"id": 1}]))
        self.assertFalse(chat_apple._has_signal([]))
        self.assertTrue(chat_apple._has_signal({"rows": [1], "total_matching": 9}))
        self.assertFalse(chat_apple._has_signal({"rows": [], "total_matching": 0}))
        self.assertTrue(chat_apple._has_signal({"message_count": 0}))  # aggregate dict
        self.assertFalse(chat_apple._has_signal({"duplicate_call": True}))
        self.assertFalse(chat_apple._has_signal({"error": "x"}))


class DeriveSummaryFactsTest(unittest.TestCase):
    def test_heatmap_busiest_hour_and_day(self):
        # 7x24 zero matrix except a spike at Wednesday(2) 22:00 and a Friday load.
        counts = [[0] * 24 for _ in range(7)]
        counts[2][22] = 100          # Wednesday 22:00 -> busiest hour
        for d in range(7):
            counts[4][d] = 0
        counts[4] = [5] * 24         # Friday total 120 -> busiest weekday
        result = {"timezone": "America/New_York", "heatmap": {"counts": counts}}
        facts = chat_apple._derive_summary_facts("get_activity_patterns", result)
        self.assertEqual(len(facts), 1)
        self.assertIn("22:00 America/New_York", facts[0])
        self.assertIn("busiest weekday = Friday", facts[0])

    def test_ignores_other_tools_and_shapes(self):
        self.assertEqual(chat_apple._derive_summary_facts("get_milestones", {"x": 1}), [])
        self.assertEqual(
            chat_apple._derive_summary_facts("get_activity_patterns", {"heatmap": {}}), [])
        self.assertEqual(
            chat_apple._derive_summary_facts("get_activity_patterns",
                                             {"heatmap": {"counts": [[0] * 24] * 7}}), [])

    def test_response_dynamics_who_replies_faster(self):
        result = {"reply_latency": {"me": {"median_seconds": 16.7},
                                    "them": {"median_seconds": 18.3}}}
        facts = chat_apple._derive_summary_facts("get_response_dynamics", result)
        self.assertEqual(len(facts), 1)
        self.assertIn("you = 16.7s", facts[0])
        self.assertIn("you reply faster", facts[0])
        # reversed
        result2 = {"reply_latency": {"me": {"median_seconds": 40},
                                     "them": {"median_seconds": 5}}}
        self.assertIn("the other person replies faster",
                      chat_apple._derive_summary_facts("get_response_dynamics", result2)[0])

    def test_response_dynamics_missing_medians(self):
        self.assertEqual(
            chat_apple._derive_summary_facts("get_response_dynamics", {"reply_latency": {}}), [])

    def test_response_dynamics_ghosted_reads_ghosted_section(self):
        # only_me lives under "ghosted", not "sessions"; a ghost question must derive
        # the one-sided-session count, NOT fall through to a reply-latency verdict.
        result = {"ghosted": {"only_me": 3, "only_them": 1},
                  "sessions": {"count": 10, "median_messages": 6},
                  "reply_latency": {"me": {"median_seconds": 5},
                                    "them": {"median_seconds": 40}}}
        facts = chat_apple._derive_summary_facts(
            "get_response_dynamics", result, "who have I ghosted?")
        self.assertEqual(len(facts), 1)
        self.assertIn("3 conversation(s)", facts[0])
        self.assertIn("out of 10 sessions", facts[0])
        self.assertNotIn("reply faster", facts[0])   # not the latency default

    def test_response_dynamics_double_text(self):
        # A double-text question has its own branch; without it the deriver fell to
        # the reply-latency default and answered a different question.
        result = {"double_text": {"me": 50, "them": 10},
                  "reply_latency": {"me": {"median_seconds": 5},
                                    "them": {"median_seconds": 40}}}
        facts = chat_apple._derive_summary_facts(
            "get_response_dynamics", result, "do I double-text more than her?")
        self.assertEqual(len(facts), 1)
        self.assertIn("Double-texts", facts[0])
        self.assertIn("You double-text more", facts[0])
        self.assertNotIn("reply faster", facts[0])

    def test_response_dynamics_latency_verdict_gated_to_speed_questions(self):
        # A question that reached this tool but is not about reply speed must fall to
        # DIGEST ([]), not a spurious latency verdict...
        result = {"reply_latency": {"me": {"median_seconds": 16.7},
                                    "them": {"median_seconds": 18.3}}}
        self.assertEqual(chat_apple._derive_summary_facts(
            "get_response_dynamics", result, "what emoji do we use?"), [])
        # ...but a no-question call keeps the default verdict (production always
        # supplies a question; this guards the empty-question leniency).
        self.assertIn("you reply faster", chat_apple._derive_summary_facts(
            "get_response_dynamics", result)[0])

    def test_milestone_who_first_gated_on_question(self):
        result = {"firsts": {
            "first": {"date": "2022-05-21T06:37:50", "text": "Phone died"},
            "first_from_me": {"date": "2022-05-21T06:37:50"},
            "first_from_them": {"date": "2022-05-21T06:37:59"}}}
        # Fires for a first/who-first question ...
        facts = chat_apple._derive_summary_facts(
            "get_milestones", result, "When did Jill and I first text, and who texted first?")
        joined = " ".join(facts)
        self.assertIn("2022-05-21", joined)
        self.assertIn("You sent that very first message", joined)  # 06:37:50 < 06:37:59
        # ... but NOT for an unrelated milestone question (nth/anniversary).
        self.assertEqual(
            chat_apple._derive_summary_facts("get_milestones", result,
                                             "what was our 10,000th message?"), [])

    def test_milestone_last_message_uses_last_only(self):
        # The last-message branch must quote firsts.last and NEVER fall back to a
        # first_* receipt (which would present the oldest message as the newest).
        result = {"firsts": {
            "first_from_them": {"date": "2022-05-21T06:37:59", "sender": "Jill",
                                "text": "Phone died", "is_from_me": 0},
            "last": {"date": "2026-07-20T21:11:02", "sender": "Jill",
                     "text": "see you tomorrow!", "is_from_me": 0}}}
        facts = chat_apple._derive_summary_facts(
            "get_milestones", result, "what was the last message Jill sent me?")
        self.assertEqual(len(facts), 1)
        self.assertIn("see you tomorrow!", facts[0])
        self.assertNotIn("Phone died", facts[0])
        # No last at all -> [] (degrade to DIGEST), even with first_from_them present.
        no_last = {"firsts": {
            "first_from_them": {"date": "2022-05-21T06:37:59", "text": "Phone died"}}}
        self.assertEqual(chat_apple._derive_summary_facts(
            "get_milestones", no_last, "what was the last message she sent?"), [])

    def test_milestone_last_from_them_not_answered_with_my_message(self):
        # "Last thing THEY sent" when the most recent message is mine: the tool has
        # no last-inbound field, so degrade to DIGEST rather than present my own
        # message as theirs. A neutral "last message" ask still gets the fact.
        result = {"firsts": {
            "last": {"date": "2026-07-20T21:11:02", "sender": "Me",
                     "text": "on my way", "is_from_me": 1}}}
        self.assertEqual(chat_apple._derive_summary_facts(
            "get_milestones", result, "what's the last thing she sent me?"), [])
        facts = chat_apple._derive_summary_facts(
            "get_milestones", result, "what was the last message in our chat?")
        self.assertIn("on my way", facts[0])

    def test_response_dynamics_on_read_coverage_conditional(self):
        # Low coverage -> the honest caveat, with the pct; no measured figures.
        low = {"read_to_reply": {
            "pairs": 4, "median_deliberation_seconds": 30,
            "coverage": {"inbound_with_date_read": 5, "inbound_total": 100,
                         "outbound_with_date_read": 2, "outbound_total": 80}}}
        facts = chat_apple._derive_summary_facts(
            "get_response_dynamics", low, "do I leave people on read?")
        self.assertEqual(len(facts), 1)
        self.assertIn("only 5.0%", facts[0])
        self.assertIn("cannot be measured reliably", facts[0])
        # Adequate coverage -> a grounded partial answer (coverage + read->reply
        # median), not the blanket refusal -- but still no left-on-read count.
        high = {"read_to_reply": {
            "pairs": 62, "median_deliberation_seconds": 41,
            "coverage": {"inbound_with_date_read": 90, "inbound_total": 100,
                         "outbound_with_date_read": 2, "outbound_total": 80}}}
        facts = chat_apple._derive_summary_facts(
            "get_response_dynamics", high, "do I leave people on read?")
        self.assertEqual(len(facts), 1)
        self.assertIn("90.0%", facts[0])
        self.assertIn("41s", facts[0])
        self.assertNotIn("cannot be measured reliably", facts[0])
        self.assertIn("read-but-never-answered count is not available", facts[0])

    def test_response_dynamics_on_read_picks_side_from_question(self):
        # "Does she leave ME on read" is about THEIR receipts of my outbound --
        # the outbound coverage decides, and my read->reply median (which measures
        # my own replying) is not quoted for their side.
        result = {"read_to_reply": {
            "pairs": 62, "median_deliberation_seconds": 41,
            "coverage": {"inbound_with_date_read": 90, "inbound_total": 100,
                         "outbound_with_date_read": 70, "outbound_total": 80}}}
        facts = chat_apple._derive_summary_facts(
            "get_response_dynamics", result, "does she leave me on read?")
        self.assertEqual(len(facts), 1)
        self.assertIn("87.5%", facts[0])           # 70/80, the outbound side
        self.assertIn("your messages to them", facts[0])
        self.assertNotIn("41s", facts[0])          # my median: wrong side, omitted


class DeriveAggregateFactsTest(unittest.TestCase):
    """P2: every specialized aggregate routes through a deriver so the reply
    narrates a code-computed fact instead of confabulating over the raw result."""

    def d(self, name, result, q=""):
        return chat_apple._derive_summary_facts(name, result, q)

    def test_reaction_top_tapback_named_not_numbered(self):
        r = {"given": {"by_type": {"like": 6879, "love": 3469, "emphasize": 6342}}}
        facts = self.d("get_reaction_stats", r, "which tapback do I use the most?")
        self.assertIn("Liked", facts[0])       # the label, not "6879"
        self.assertIn("6879", facts[0])

    def test_reaction_hall_of_fame_quotes_the_real_message(self):
        r = {"hall_of_fame": [{"sender": "Matt", "text": "Cumdinger",
                               "date": "2026-06-18", "reaction_count": 3}]}
        facts = self.d("get_reaction_stats", r, "which message got the most reactions?")
        self.assertIn("Cumdinger", facts[0])
        self.assertIn("3 reactions", facts[0])

    def test_word_frequency_is_verbatim_from_rows(self):
        r = {"rows": [{"ngram": "want to", "count": 3071}, {"ngram": "going to", "count": 2988}],
             "ngram": 2, "messages_scanned": 100}
        facts = self.d("get_word_frequency", r, "two-word phrases I say most")
        self.assertIn("want to (3071)", facts[0])
        self.assertIn("phrases", facts[0])

    def test_word_trends_phrase_direction(self):
        r = {"phrase_trend": {"phrase": "sweet", "total_messages": 2647,
             "series": [{"period": "2022", "me": 300, "them": 25},
                        {"period": "2026", "me": 130, "them": 32}]}}
        facts = self.d("get_word_trends", r, "use of the word sweet over the years")
        self.assertIn("sweet", facts[0])
        self.assertIn("falling", facts[0])

    def test_group_top_member_and_gini(self):
        r = {"members": [{"member": "Me", "is_me": True, "share": 0.301}],
             "gini": {"gini": 0.20}}
        facts = self.d("get_group_dynamics", r, "who carries Real Tahoe?")
        self.assertIn("Me carries", facts[0])
        self.assertIn("fairly balanced", facts[0])

    def test_voiceprint_low_confidence_is_flagged(self):
        r = {"confidence": "low", "n_messages": 33,
             "features": {"lowercase_ratio": 0.9, "burst_size": 1.5}, "openers": [], "closers": []}
        facts = self.d("get_voiceprint", r, "voiceprint with Jenna")
        self.assertIn("low confidence", facts[0])
        self.assertIn("tentatively", facts[0])

    def test_participant_reactions_per_message_applies_floor(self):
        r = [{"participant": "Me", "messages_sent": 10, "reactions_received": 8},   # below floor
             {"participant": "Teddy", "messages_sent": 51, "reactions_received": 14},
             {"participant": "Owen", "messages_sent": 21, "reactions_received": 9}]
        facts = self.d("get_chat_participant_stats", r,
                       "who gets the most reactions per message among people with at least 20 messages?")
        self.assertIn("Owen", facts[0])        # 9/21=0.43 beats Teddy 14/51=0.27; Me excluded
        self.assertNotIn("Me", facts[0])

    def test_conflict_zero_says_none_and_does_not_manufacture(self):
        facts = self.d("find_conflicts", {"total_matching": 0, "rows": []},
                       "have Jenna and I ever had a fight?")
        self.assertIn("No tense stretches", facts[0])
        # The zero-conflict fact must be pure data -- no model-directive that could
        # leak into the user-facing reply (via _compose_no_fault_answer or a paraphrase).
        self.assertNotIn("manufacture", facts[0].lower())

    def test_zero_conflict_fault_answer_is_clean_no_leak(self):
        # A who-started question with zero conflicts: the composed reply must read
        # cleanly and never splice the internal negative fact / a directive.
        sp = self.d("find_conflicts", {"total_matching": 0, "rows": []},
                    "who started our fights?")
        out = chat_apple._compose_no_fault_answer(sp)
        self.assertIsNotNone(out)
        self.assertIn("couldn't find", out.lower())
        self.assertNotIn("manufacture", out.lower())
        self.assertNotIn("[derived]", out)
        self.assertNotIn("What the signals do show", out)

    def test_conflict_never_infers_fault(self):
        r = {"total_matching": 318, "rows": [{"first_date": "2026-05-27",
             "signals": ["latency_withdrawal"], "corroboration": "llm"}]}
        facts = self.d("find_conflicts", r, "who started more of the arguments?")
        self.assertIn("318", facts[0])
        self.assertIn("do NOT identify who was at fault", facts[0])

    def test_fault_question_answer_is_code_composed_no_blame(self):
        sp = ['[derived] 318 tense stretch(es) detected (deterministic text-only '
              'signals, not confirmed fights). The strongest: around 2026-05-27, '
              'signals: latency_withdrawal. These signals do NOT identify who was at '
              'fault or who started it.']
        out = chat_apple._compose_no_fault_answer(sp)
        self.assertIn("can't say who started", out.lower())
        self.assertIn("318 tense stretch", out)     # relays the count
        self.assertNotIn("[derived]", out)           # no scaffolding leaks
        # No conflict fact gathered -> falls back to the normal answer path.
        self.assertIsNone(chat_apple._compose_no_fault_answer(["nothing here"]))
        # The trigger fires on the fault phrasings, not on a neutral conflict ask.
        self.assertTrue(chat_apple._FAULT_QUESTION_RE.search("who started the fight?"))
        self.assertTrue(chat_apple._FAULT_QUESTION_RE.search("whose fault was it?"))
        self.assertFalse(chat_apple._FAULT_QUESTION_RE.search(
            "have we had any rough patches?"))

    def test_conflict_defers_for_apology_only(self):
        # Apology defers (draft_apology needs the conflict_id downstream); a
        # diagnosis ask grounds instead of deferring (the model won't reliably
        # chain to diagnose_conflict, so a grounded summary beats digest confab).
        r = {"total_matching": 318, "rows": [{"first_date": "2026-05-27", "signals": []}]}
        self.assertEqual(self.d("find_conflicts", r, "help me write an apology"), [])
        facts = self.d("find_conflicts", r, "break down the tensest stretch")
        self.assertTrue(facts and "318" in facts[0])

    def test_loose_threads_empty_says_none(self):
        self.assertIn("No open loose threads",
                      self.d("get_loose_threads", [], "anyone I owe a reply?")[0])

    def test_activity_late_night_names_top_contact(self):
        r = {"late_night": {"n": 54141, "start_hour": 0, "end_hour": 5, "share": 0.1547,
             "top_contacts": [{"contact": "Callie", "late_night": 610}]}}
        facts = self.d("get_activity_patterns", r, "how much do I text after midnight?")
        self.assertIn("54141", facts[0])
        self.assertIn("Callie", facts[0])

    def test_summary_stats_group_by_chat_leaderboard_derives(self):
        # group_by_chat=True returns a bare LIST -- the "who do I text most"
        # leaderboard must derive (top row by message_count), not fall through
        # to DIGEST/confabulation.
        r = [{"chat_id": 7, "message_count": 120, "from_me": 60, "from_them": 60},
             {"chat_id": 42, "message_count": 980, "from_me": 500, "from_them": 480}]
        facts = self.d("get_chat_summary_stats", r, "who do I text the most?")
        self.assertEqual(len(facts), 1)
        self.assertIn("chat_id=42", facts[0])
        self.assertIn("980", facts[0])
        # And an empty leaderboard degrades to DIGEST, not a crash.
        self.assertEqual(self.d("get_chat_summary_stats", [], "who do I text most?"), [])


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
        # Neutral wording: exercises the model-routed path, not an intent rule.
        reply = chat_apple.run_turn([{"role": "user", "content": "anything at all?"}])
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

    def test_invented_enum_filter_dropped_kept_when_asked(self):
        def g(a, q):
            return chat_apple._ground_args(a, q, "", [])
        # A hallucinated kind that empties the tool -> dropped; limit survives.
        self.assertEqual(g({"limit": 5, "kind": "loose_threads", "status": "open"},
                           "anyone I owe a reply to?"), {"limit": 5})
        # A spurious reaction_type / voiceprint subject -> dropped.
        self.assertEqual(g({"reaction_type": "laugh"}, "which tapback do I use most?"), {})
        self.assertEqual(g({"scope": "all", "subject": "them"},
                           "describe my texting style"), {"scope": "all"})
        # ... but a genuinely-asked narrowing survives.
        self.assertEqual(g({"reaction_type": "laugh"}, "how many laugh reactions?"),
                         {"reaction_type": "laugh"})
        # Shape args (ngram/is_from_me) are never grounded -- '2' won't match "two-word".
        self.assertEqual(g({"ngram": 2, "is_from_me": True}, "two-word phrases I say most"),
                         {"ngram": 2, "is_from_me": True})


class StripScratchpadTagsTest(unittest.TestCase):
    def test_strips_leaked_provenance_lines_only(self):
        t = ('**[resolved] "Jill" -> contact_id=168**\n'
             '[derived] 1 tense stretch detected.\n\nThe pattern was volume collapse.')
        out = chat_apple._strip_scratchpad_tags(t)
        self.assertNotIn("[resolved]", out)
        self.assertNotIn("[derived]", out)
        self.assertIn("volume collapse", out)   # ordinary prose kept

    def test_no_tags_is_a_noop(self):
        t = "You reply faster than Jill."
        self.assertEqual(chat_apple._strip_scratchpad_tags(t), t)


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

    def test_name_colliding_auxiliaries_are_positional(self):
        # Mid-sentence, a capitalized Will/May/Can is a NAME -- swallowing it
        # would also disarm the unresolved-name guard (S7.6 misattribution).
        self.assertEqual(
            chat_apple._extract_candidate_names("How often does Will text me?"), ["Will"])
        self.assertEqual(
            chat_apple._extract_candidate_names("What did May send over?"), ["May"])
        # Sentence-initial, the same word is an auxiliary and is trimmed.
        self.assertEqual(
            chat_apple._extract_candidate_names("Will I ever hear from Jenna again?"),
            ["Jenna"])
        self.assertEqual(
            chat_apple._extract_candidate_names("Can you check on Kyle?"), ["Kyle"])
        # After a trimmed sentence-initial auxiliary, the next token is a name.
        self.assertEqual(
            chat_apple._extract_candidate_names("Can Will text back?"), ["Will"])
        # A new sentence mid-question restarts sentence-initial trimming.
        self.assertEqual(
            chat_apple._extract_candidate_names("Tell me about Jill. Will she reply?"),
            ["Jill"])
        # Non-name auxiliaries stay unconditional stopwords everywhere.
        self.assertEqual(
            chat_apple._extract_candidate_names("Am I texting too much?"), [])

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

        facts, resolved = chat_apple._preresolve_entities(
            self._fake_chat(contacts_fn, chats_fn), "When did Jill and I first text?")
        self.assertEqual(len(facts), 1)
        f = facts[0]
        self.assertIn("contact_id=168", f)
        self.assertIn("handle_id=[5]", f)
        self.assertIn("1:1 chat_id=[42]", f)   # only the is_group=0 chat
        self.assertNotIn("99", f)              # the group chat is excluded
        self.assertEqual(calls["contacts"], [{"name": "Jill", "limit": 5}])
        self.assertEqual(calls["chats"], [{"contact_id": 168, "limit": 50}])
        # structured resolution carries the scope for deterministic filling
        self.assertEqual(resolved, [{"contact_id": 168, "chats": [42]}])
        self.assertEqual(chat_apple._primary_contact(resolved), 168)

    def test_no_name_no_tool_calls(self):
        called = []
        facts, resolved = chat_apple._preresolve_entities(
            self._fake_chat(lambda **kw: called.append(kw) or []),
            "what hour of the day do i text the most?")
        self.assertEqual(facts, [])
        self.assertEqual(resolved, [])
        self.assertEqual(called, [])  # never queried the directory

    def test_unresolved_name_is_skipped(self):
        facts, resolved = chat_apple._preresolve_entities(
            self._fake_chat(lambda **kw: []), "Who is Nobody Here?")
        self.assertEqual(facts, [])
        self.assertEqual(resolved, [])

    def test_tool_exception_degrades_to_no_seed(self):
        def boom(**kw):
            raise RuntimeError("db down")
        facts, resolved = chat_apple._preresolve_entities(
            self._fake_chat(boom), "Who is Jill?")
        self.assertEqual(facts, [])
        self.assertEqual(resolved, [])

    def test_seed_grounds_a_filled_chat_id(self):
        # End-to-end: the seeded chat_id survives _ground_args on the next fill.
        facts, _ = chat_apple._preresolve_entities(
            self._fake_chat(
                lambda **kw: [{"id": 168, "first_name": "Jill", "handles": []}],
                lambda **kw: [{"id": 42, "is_group": 0}]),
            "Who is Jill?")
        out = chat_apple._ground_args({"chat_id": 42}, "Who is Jill?", "", facts)
        self.assertEqual(out, {"chat_id": 42})  # grounded by the seed


class PrimaryContactTest(unittest.TestCase):
    def test_prefers_the_one_with_a_chat(self):
        # Ambiguous name (two matches) but only one has a real 1:1 chat.
        resolved = [{"contact_id": 230, "chats": []}, {"contact_id": 168, "chats": [43]}]
        self.assertEqual(chat_apple._primary_contact(resolved), 168)

    def test_sole_match_used_even_without_chat(self):
        self.assertEqual(
            chat_apple._primary_contact([{"contact_id": 7, "chats": []}]), 7)

    def test_ambiguous_returns_none(self):
        # Two contacts each with a chat -> can't pick deterministically.
        resolved = [{"contact_id": 1, "chats": [10]}, {"contact_id": 2, "chats": [20]}]
        self.assertIsNone(chat_apple._primary_contact(resolved))
        self.assertIsNone(chat_apple._primary_contact([]))


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
