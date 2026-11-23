"""Run Checkpoint 6 metrics with real local retrieval and a scripted model fixture."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
from platform import python_version
from time import perf_counter

from .core import PlanningAgent
from .graph import GraphCoordinator, allocation_key, max_model_calls
from .metrics import audit_cases, evaluate_case, retrieval_benchmark, summarize
from .rag import LexicalRetriever, SemanticRetriever


class FixtureBackend:
    """Deterministic exercise of the model boundary. No network calls to an LLM."""

    live = False

    def __init__(self):
        self.records = []
        self.planner = PlanningAgent()

    def ask(self, role, _instruction, payload, _schema):
        self.records.append({"agent": role})
        if role in {"Research", "Auditor"}:
            page = payload["pages"][0]
            excerpt = " ".join(page["text"].split()[:32])
            claim = {"statement": excerpt, "citations": [{"page_id": page["id"], "quote": excerpt}]}
            if role == "Research":
                return {"summary": "Scripted evidence fixture", "findings": [claim],
                        "limitations": [], "evidence_status": "sufficient"}
            return {"acceptable": payload["checks"]["passes"], "explanation": "Scripted audit fixture",
                    "concerns": [], "evidence_review": [claim], "evidence_status": "sufficient",
                    "guidance_alignment": "supported"}
        if role == "Coordinator":
            return {"action": "accept_for_review" if payload["passing_branches"] else "revise",
                    "explanation": "Scripted coordinator fixture"}

        scenario = payload["scenario"]
        parent = {**scenario, "shelters": payload["parent"]["allocation"]}
        candidates = [plan for plan in self.planner.propose(scenario) if allocation_key(plan) != allocation_key(parent)]
        candidates.sort(key=lambda plan: -sum(item["assigned"] for item in plan["shelters"]))
        alternatives = [self.proposal(plan) for plan in candidates[:3]]
        if scenario["id"] == "over_allocation" and payload["depth"] == 1 and alternatives:
            # Deliberately inject a defective first proposal to exercise the difference
            # between the linear and beam search under fixed agent outputs.
            rejected = json.loads(json.dumps(alternatives[0]))
            rejected["allocations"][0]["assigned"] = scenario["shelters"][0]["capacity"] + 1
            alternatives = [rejected, *alternatives[:2]]
        return {"alternatives": alternatives, "limitations": ["Scripted planning fixture"]}

    @staticmethod
    def proposal(plan):
        allocations = [
            {"shelter_id": item["id"], "open": item["open"], "assigned": item["assigned"]}
            for item in plan["shelters"]
        ]
        return {"label": "Synthetic reallocation", "rationale": "Fixture to exercise resource checks.",
                "allocations": allocations}


def evaluate_workflow(coordinator, cases, retrieval, *, fixture=False, reviews=None):
    rows, results = [], []
    for case in cases:
        start = perf_counter()
        result = coordinator.run(case["scenario"], case.get("perturbations"))
        elapsed = perf_counter() - start
        rows.append(evaluate_case(case, result, elapsed, reviews=reviews, fixture=fixture))
        results.append(result)
    return {"metrics": summarize(rows, retrieval), "cases": rows}, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/checkpoints.json"))
    args = parser.parse_args()
    started = perf_counter()
    semantic = SemanticRetriever()
    indexing_seconds = perf_counter() - started
    retrieval = {
        "semantic": retrieval_benchmark(semantic),
        "bm25": retrieval_benchmark(LexicalRetriever()),
        "bm25_matched_chunks": retrieval_benchmark(LexicalRetriever(chunks=semantic.chunks)),
    }
    outcomes = {}
    for strategy in ("beam", "linear"):
        graph = GraphCoordinator(FixtureBackend(), retriever=semantic, strategy=strategy)
        outcomes[strategy], _ = evaluate_workflow(graph, audit_cases(), retrieval["semantic"], fixture=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"mode": "real local MiniLM and Chroma retrieval, scripted agent outputs",
              "run_at": datetime.now(timezone.utc).isoformat(), "semantic_chunks": len(semantic.chunks),
              "corpus_sha256": semantic.hash, "indexing_seconds": indexing_seconds,
              "python_version": python_version(), "package_versions": {package: version(package)
                  for package in ("chromadb", "langgraph", "tiktoken", "openai")},
              "model_calls_per_audit_cap": max_model_calls(3), "retrieval": retrieval, "strategies": outcomes}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value["metrics"] for key, value in outcomes.items()}, indent=2))


if __name__ == "__main__":
    main()
