import argparse
import json
from pathlib import Path


def read_json(path):
    with path.open("rb") as stream:
        raw = stream.read(128 * 1024 + 1)
    if len(raw) > 128 * 1024:
        raise ValueError("Structured input exceeds 128 KiB.")
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser(description="EndureAI synthetic shelter plan auditing")
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/audit"))
    parser.add_argument("--mode", choices=["llm", "baseline"], default="llm")
    parser.add_argument("--model", help="Responses API model; otherwise use OPENAI_MODEL")
    parser.add_argument("--max-rounds", type=int, default=3, help="Maximum graph search depth (1–3)")
    parser.add_argument("--max-candidates", type=int, default=32, help="Baseline only")
    parser.add_argument("--retrieval", choices=["semantic", "bm25"], default="semantic")
    parser.add_argument("--perturbations", type=Path, help="JSON changes to demand, power, or existing shelter conditions")
    args = parser.parse_args()
    try:
        scenario = read_json(args.scenario)
        if args.mode == "llm":
            from .graph import GraphCoordinator, max_model_calls
            from .llm import OpenAIBackend, render_report
            from .rag import LexicalRetriever

            changes = read_json(args.perturbations) if args.perturbations else None
            if args.perturbations and changes is None:
                raise ValueError("Supply a JSON object for perturbations.")
            budget = max_model_calls(args.max_rounds)
            backend = OpenAIBackend(model=args.model, max_calls=budget)
            retriever = LexicalRetriever() if args.retrieval == "bm25" else None
            result = GraphCoordinator(backend, retriever=retriever, max_rounds=args.max_rounds).run(scenario, changes)
        else:
            from .core import CoordinatorAgent, render_report

            if args.perturbations:
                parser.error("Use --mode llm to compare submitted and perturbed conditions.")
            result = CoordinatorAgent(max_candidates=args.max_candidates).run(scenario)
    except (ValueError, KeyError, OSError, ImportError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(render_report(result), encoding="utf-8")
    original = result.get("original")
    print(result["status"], f"shortage={original['shortage']}" if original else "", args.output.with_suffix(".md"))
    if result.get("error"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
