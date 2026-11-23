import unittest

from endureai.evaluate_checkpoints import FixtureBackend
from endureai.graph import GraphCoordinator
from endureai.metrics import audit_cases, claim_records, evaluate_case, summarize


class StaticRetriever:
    def run(self, scenario, original):
        return [{"id": "sample", "text": "Shelter capacity should account for the assumed demand."}]


class MetricsTests(unittest.TestCase):
    def test_denominators_and_live_groundedness_requires_review(self):
        graph = GraphCoordinator(FixtureBackend(), retriever=StaticRetriever())
        case = audit_cases()[0]
        result = graph.run(case["scenario"])
        row = evaluate_case(case, result, 0.5)
        metrics = summarize([row], {"passage_relevance": {"correct": 1, "total": 2, "rate": 0.5}})
        self.assertEqual(metrics["calculation_accuracy"]["rate"], 1)
        self.assertEqual(metrics["gap_detection_rate"]["total"], 2)
        self.assertEqual(metrics["retrieval_relevance"]["rate"], 0.5)
        self.assertEqual(metrics["latency"]["p95"], 0.5)
        self.assertIsNone(metrics["groundedness"]["rate"])
        self.assertTrue(all(claim["quote_verified"] for claim in row["claims"]))
        claim_ids = {claim["id"]: True for claim in row["claims"]}
        self.assertTrue(all(claim["support_review"] for claim in claim_records(result, claim_ids)))

    def test_invalid_input_is_scored_without_a_capacity_denominator(self):
        graph = GraphCoordinator(FixtureBackend(), retriever=StaticRetriever())
        case = audit_cases()[-2]
        row = evaluate_case(case, graph.run(case["scenario"]), 0.05)
        metrics = summarize([row], {"passage_relevance": {"correct": 0, "total": 0, "rate": None}})
        self.assertIsNone(metrics["calculation_accuracy"]["rate"])
        self.assertIsNone(metrics["constraint_compliance"]["rate"])
        self.assertEqual(metrics["escalation_accuracy"]["rate"], 1)
