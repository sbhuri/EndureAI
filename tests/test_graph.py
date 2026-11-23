import copy
import json
import unittest

from endureai.core import ROOT
from endureai.graph import GraphCoordinator, max_model_calls, stress_plan
from endureai.llm import OpenAIBackend, render_report

PAGES = [{"id": "FEMA-example", "url": "https://www.fema.gov/",
          "text": "Exercise planners should check accessible shelter capacity and backup power."}]
CLAIMS = [{"statement": PAGES[0]["text"], "citations": [{"page_id": PAGES[0]["id"], "quote": PAGES[0]["text"]}]}]


def load(name):
    return json.loads((ROOT / f"data/scenarios/{name}.json").read_text())


def alternative(*assignments):
    return {"label": "Synthetic candidate", "rationale": "Test allocation",
            "allocations": [{"shelter_id": key, "assigned": count, "open": opened}
                            for key, count, opened in assignments]}


class FixtureRetriever:
    name = "fixture"
    hash = "fixture"

    def run(self, scenario, original):
        return PAGES


class FakeBackend:
    live = False

    def __init__(self, proposals=(), *, evidence_status="sufficient", approval=None):
        self.proposals = iter(proposals)
        self.evidence_status = evidence_status
        self.approval = approval
        self.records = []

    def ask(self, role, _instructions, payload, _schema):
        self.records.append({"agent": role, "input": payload})
        if role == "Research":
            return {"summary": "Extractive fixture", "findings": CLAIMS if self.evidence_status == "sufficient" else [],
                    "limitations": [], "evidence_status": self.evidence_status}
        if role == "Planning":
            return {"alternatives": next(self.proposals), "limitations": []}
        if role == "Auditor":
            return {"acceptable": payload["checks"]["passes"], "explanation": "Mock audit", "concerns": [],
                    "evidence_status": "sufficient", "guidance_alignment": "supported", "evidence_review": CLAIMS}
        action = self.approval or ("accept_for_review" if payload["passing_branches"] else "revise")
        return {"action": action, "explanation": "Mock decision"}


class GraphTests(unittest.TestCase):
    def run_graph(self, scenario, backend, changes=None, **options):
        return GraphCoordinator(backend, retriever=FixtureRetriever(), **options).run(scenario, changes)

    def test_passing_original_never_calls_planner(self):
        scenario = load("balanced")
        backend = FakeBackend()
        result = self.run_graph(scenario, backend)
        self.assertEqual(result["status"], "CHECKS_PASSED")
        self.assertEqual([x["agent"] for x in backend.records], ["Research", "Auditor"])
        self.assertEqual(result["selected_plan"], scenario)
        self.assertTrue(result["human_review_required"])

    def test_stress_tests_existing_allocations_and_does_not_mutate_input(self):
        scenario = load("checkpoint_winter_plan")
        snapshot = copy.deepcopy(scenario)
        result = self.run_graph(scenario, FakeBackend([[]]), load("checkpoint_shelter_loss"))
        self.assertEqual(scenario, snapshot)
        self.assertEqual(result["base_audit"]["shortage"], 0)
        self.assertEqual(result["original"]["shortage"], 150)
        self.assertEqual(result["scenario"]["shelters"][0]["assigned"], 200)
        self.assertIn("UNAVAILABLE_OPEN", {f["code"] for f in result["original"]["findings"]})
        self.assertIsNone(result["selected_plan"])

    def test_invalid_or_actual_event_input_stops_before_tools(self):
        for scenario, expected in [
            ({**load("balanced"), "purpose": "active_emergency"}, "OPERATIONAL_USE"),
            ({**load("balanced"), "demand": True}, "INPUT_CLARIFICATION"),
            ({**load("balanced"), "notes": "Private records"}, "INPUT_CLARIFICATION"),
        ]:
            with self.subTest(expected=expected):
                backend = FakeBackend()
                result = self.run_graph(scenario, backend)
                self.assertEqual(result["escalation"]["code"], expected)
                self.assertEqual(backend.records, [])
                self.assertNotIn("Private records", render_report(result))

    def test_bad_perturbation_is_not_an_inventory_write(self):
        with self.assertRaisesRegex(ValueError, "existing synthetic shelters"):
            stress_plan(load("balanced"), {"shelters": {"invented": {"capacity": 999}}})
        with self.assertRaisesRegex(ValueError, "Change capacity"):
            stress_plan(load("balanced"), {"shelters": {"A": {"assigned": 100}}})

    def test_hard_violations_skip_llm_audit_and_cannot_be_accepted(self):
        bad = alternative(("A", 300, True), ("B", 100, True))
        backend = FakeBackend([[bad]], approval="accept_for_review")
        result = self.run_graph(load("winter_shortage"), backend)
        self.assertTrue(result["candidates"][0]["hard_violation"])
        self.assertIn("OVER_CAPACITY", result["candidates"][0]["rejection"])
        self.assertIsNone(result["selected_plan"])
        self.assertEqual([x["agent"] for x in backend.records].count("Auditor"), 1)

    def test_invalid_shelter_does_not_prevent_valid_sibling(self):
        bad = alternative(("Z", 250, True), ("B", 150, True))
        good = alternative(("A", 250, True), ("B", 150, True))
        result = self.run_graph(load("winter_shortage"), FakeBackend([[bad, good]]))
        self.assertEqual(result["status"], "REVISION_FOR_REVIEW")
        self.assertTrue(result["candidates"][0]["hard_violation"])
        self.assertTrue(result["candidates"][1]["passes"])
        self.assertEqual(result["candidates"][1]["score"]["components"],
                         {"feasibility": 40, "shortage_reduction": 25, "guidance_alignment": 20, "practicality": 7.5})

    def test_beam_width_depth_and_feedback(self):
        def partial(assigned):
            return alternative(("A", 250, True), ("B", assigned, True))

        backend = FakeBackend([
            [partial(148), partial(147), partial(146)],
            [partial(145)], [partial(144)],
            [partial(143)], [partial(142)],
        ])
        result = self.run_graph(load("winter_shortage"), backend)
        calls = [x["input"] for x in backend.records if x["agent"] == "Planning"]
        self.assertEqual([x["depth"] for x in calls], [1, 2, 2, 3, 3])
        self.assertEqual([x["parent"]["allocation"][1]["assigned"] for x in calls], [0, 148, 147, 145, 144])
        self.assertIn("DEMAND_MISMATCH", str(calls[1]["feedback"]))
        self.assertEqual(result["escalation"]["code"], "SEARCH_LIMIT")
        self.assertIsNone(result["selected_plan"])
        self.assertLessEqual(len(result["calls"]), max_model_calls(3))

    def test_early_revision_uses_prior_auditor_feedback(self):
        invalid = alternative(("A", 250, True), ("B", 150, False))
        partial = alternative(("A", 250, True), ("B", 149, True))
        full = alternative(("A", 250, True), ("B", 150, True))
        backend = FakeBackend([[invalid, partial], [full]])
        result = self.run_graph(load("winter_shortage"), backend)
        self.assertEqual(result["status"], "REVISION_FOR_REVIEW")
        self.assertEqual(result["depth"], 2)
        second = [x for x in backend.records if x["agent"] == "Planning"][1]
        self.assertIn("CLOSED_ALLOCATION", str(second["input"]["feedback"]))

    def test_graph_uses_sdk_structured_responses_and_records_citation_retry(self):
        import httpx
        from openai import OpenAI

        captured = []
        invalid_claims = copy.deepcopy(CLAIMS)
        invalid_claims[0]["citations"][0]["quote"] = "Text that is absent from the passage."
        repair = alternative(("A", 250, True), ("B", 150, True))
        fixture = FakeBackend([[repair]])

        def respond(request):
            data = json.loads(request.content)
            name = data["text"]["format"]["name"]
            payload = json.loads(data["input"])
            captured.append((name, payload, data["store"]))
            output = fixture.ask(name.title(), "", payload, None)
            if len(captured) == 1:
                output["findings"] = invalid_claims
            return httpx.Response(200, json={
                "id": "resp_test", "object": "response", "created_at": 1,
                "status": "completed", "model": "test-model",
                "output": [{"type": "message", "id": "msg_test", "role": "assistant",
                            "status": "completed", "content": [
                                {"type": "output_text", "text": json.dumps(output), "annotations": []}
                            ]}],
            })

        with OpenAI(api_key="test-only-placeholder",
                    http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
            backend = OpenAIBackend("test-model", client=client, max_calls=max_model_calls(3))
            result = self.run_graph(load("winter_shortage"), backend)
        self.assertEqual(result["status"], "REVISION_FOR_REVIEW")
        self.assertEqual([entry[0] for entry in captured],
                         ["research", "research", "auditor", "planning", "auditor", "coordinator"])
        self.assertIn("validation_feedback", captured[1][1])
        self.assertTrue(all(entry[2] is False for entry in captured))

    def test_global_working_state_is_fresh_for_each_run(self):
        backend = FakeBackend()
        graph = GraphCoordinator(backend, retriever=FixtureRetriever())
        graph.run({**load("balanced"), "purpose": "active_emergency"})
        result = graph.run(load("balanced"))
        self.assertIsNone(result["escalation"])
        self.assertEqual(result["depth"], 0)
        self.assertEqual(len(result["calls"]), 2)

    def test_conflicting_or_insufficient_guidance_stops_planning(self):
        for status in ("conflicting", "insufficient"):
            with self.subTest(status=status):
                backend = FakeBackend(evidence_status=status)
                result = self.run_graph(load("winter_shortage"), backend)
                self.assertEqual(result["escalation"]["code"], f"EVIDENCE_{status.upper()}")
                self.assertEqual(len(result["calls"]), 1)


if __name__ == "__main__":
    unittest.main()
