"""Regression tests for evidence-grounded Research citation correction."""
import unittest
from endureai.llm import ResearchAgent, ModelFailure


PAGES = [{"id": "FEMA-p1", "text": "Shelter planning should account for accessible facilities and backup power."}]
EVIDENCE = {"capacity": PAGES}


class FakeBackend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def ask(self, role, instruction, payload, schema):
        self.calls.append((role, instruction, payload))
        return next(self.responses)


def research(quote, page_id="FEMA-p1"):
    return {"summary": "Guidance", "findings": [
        {"statement": "Plan for power.", "citations": [{"page_id": page_id, "quote": quote}]}
    ], "limitations": []}


class ResearchCitationTests(unittest.TestCase):
    def test_valid_citation_needs_no_retry(self):
        backend = FakeBackend([research("accessible facilities and backup power")])
        result, pages = ResearchAgent(backend).run({"id": "synthetic"}, EVIDENCE)
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(pages, PAGES)
        self.assertTrue(result["findings"])

    def test_mismatched_quote_is_corrected_once(self):
        backend = FakeBackend([
            research("nonexistent guidance quotation"),
            research("accessible facilities and backup power"),
        ])
        result, _ = ResearchAgent(backend).run({"id": "synthetic"}, EVIDENCE)
        self.assertEqual(len(backend.calls), 2)
        self.assertIn("validation_feedback", backend.calls[1][2])
        self.assertTrue(result["findings"])

    def test_unverified_retry_is_rejected(self):
        backend = FakeBackend([
            research("nonexistent guidance quotation"),
            research("another invented quotation"),
        ])
        with self.assertRaises(ModelFailure):
            ResearchAgent(backend).run({"id": "synthetic"}, EVIDENCE)
        self.assertEqual(len(backend.calls), 2)

    def test_empty_retry_is_not_accepted(self):
        backend = FakeBackend([
            research("nonexistent guidance quotation"),
            {"summary": "Insufficient evidence", "findings": [], "limitations": ["No support"]},
        ])
        with self.assertRaises(ModelFailure):
            ResearchAgent(backend).run({"id": "synthetic"}, EVIDENCE)


if __name__ == "__main__":
    unittest.main()
