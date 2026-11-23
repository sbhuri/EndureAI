"""Run live model evaluation and record the Checkpoint 6 measurements."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .core import ROOT
from .evaluate_checkpoints import evaluate_workflow
from .graph import GraphCoordinator, max_model_calls
from .llm import OpenAIBackend, render_report
from .metrics import audit_cases, retrieval_benchmark
from .rag import SemanticRetriever


def main():
    maximum = max_model_calls(3)
    parser = argparse.ArgumentParser(description=f"Paid live evaluation, up to {maximum} model calls per scenario")
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--support-review", type=Path, help="JSON mapping from claim ID to human-reviewed support boolean")
    parser.add_argument("--rescore", type=Path, help="Existing live evaluation.json to rescore with --support-review, without API calls")
    args = parser.parse_args()
    cases = audit_cases()
    if not 1 <= args.limit <= len(cases):
        parser.error(f"limit must be 1–{len(cases)}")
    reviews = {}
    if args.support_review:
        reviews = json.loads(args.support_review.read_text())
        if not isinstance(reviews, dict) or any(type(value) is not bool for value in reviews.values()):
            parser.error("Support review must be a map of claim IDs to booleans.")
    if args.rescore:
        if not args.support_review:
            parser.error("--rescore requires --support-review.")
        from .metrics import summarize

        report = json.loads(args.rescore.read_text())
        if report.get("mode") != "llm-live":
            parser.error("--rescore expects live evaluation output.")
        for row in report["cases"]:
            for claim in row["claims"]:
                claim["support_review"] = reviews.get(claim["id"]) if claim["quote_verified"] else False
        report["metrics"] = summarize(report["cases"], report["retrieval"])
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Evidence:", args.output)
        return
    try:
        backend = OpenAIBackend(args.model, max_calls=args.limit * maximum)
    except ValueError as error:
        parser.error(str(error))
    indexed = SemanticRetriever()
    relevance = retrieval_benchmark(indexed)
    coordinator = GraphCoordinator(backend, retriever=indexed)
    args.output.mkdir(parents=True, exist_ok=False)
    evaluation, results = evaluate_workflow(coordinator, cases[:args.limit], relevance, reviews=reviews)
    for row, result in zip(evaluation["cases"], results, strict=True):
        name = row["id"]
        (args.output / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (args.output / f"{name}.md").write_text(render_report(result), encoding="utf-8")
        print(name, row["status"], "execution error:", bool(row["error"]), flush=True)
    report = {"mode": "llm-live", "requested_model": backend.model, "retrieval": relevance, **evaluation}
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Evidence:", args.output)
    if any(row["error"] for row in evaluation["cases"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
