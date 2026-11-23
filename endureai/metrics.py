"""Checkpoint 6 development metrics and evidence records."""
from collections import Counter
import hashlib
import json
import statistics

from .core import AuditorAgent, ROOT
from .graph import deterministic_checks
from .llm import ModelFailure, verify_claims


def ratio(correct, total):
    return {"correct": correct, "total": total, "rate": correct / total if total else None}


def audit_cases():
    cases = json.loads((ROOT / "data/evaluation_cases.json").read_text())
    for case in cases:
        case["expected_escalation"] = case["expected_status"] == "REVIEW_REQUIRED"
    path = ROOT / "data/scenarios"
    winter = json.loads((path / "checkpoint_winter_plan.json").read_text())
    stress = json.loads((path / "checkpoint_shelter_loss.json").read_text())
    cases.extend([
        {"scenario": winter, "perturbations": stress, "expected_shortage": 150,
         "expected_codes": ["UNAVAILABLE_OPEN", "CAPACITY_SHORTAGE"], "expected_escalation": True},
        {"scenario": {**winter, "id": "missing_demand", "demand": None}, "expected_escalation": True},
        {"scenario": {**winter, "id": "active_emergency", "purpose": "active_emergency"},
         "expected_escalation": True},
    ])
    return cases


def claim_id(claim, pages):
    # Include the exact packet contents so a review cannot silently apply to
    # a later corpus version or a claim supported by different evidence.
    available = {page["id"]: page for page in pages}
    cited = [available.get(citation["page_id"]) for citation in claim["citations"]]
    encoded = json.dumps({"claim": claim, "passages": cited}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def claim_records(result, reviews, *, fixture=False):
    pages = [page for group in result.get("evidence", {}).values() for page in group]
    research = (result.get("research") or {}).get("findings", [])
    reviews_from_auditor = [result.get("original_review") or {}]
    reviews_from_auditor += [row.get("review") or {} for row in result["candidates"]]
    claims = [("Research", claim) for claim in research]
    claims += [
        ("Auditor", claim)
        for review in reviews_from_auditor for claim in review.get("evidence_review", [])
    ]
    records = []
    for role, claim in claims:
        key = claim_id(claim, pages)
        try:
            verify_claims([claim], pages)
            quote_verified = True
        except ModelFailure:
            quote_verified = False
        if not quote_verified:
            supported = False
        elif fixture:
            supported = any(claim["statement"] == citation["quote"] for citation in claim["citations"])
        else:
            supported = reviews.get(key)
        records.append({"id": key, "role": role, "claim": claim,
                        "quote_verified": quote_verified, "support_review": supported})
    return records


def evaluate_case(case, result, elapsed, *, reviews=None, fixture=False):
    original = result.get("original") or {}
    expected_codes = Counter(case.get("expected_codes", []))
    actual_codes = Counter(finding["code"] for finding in original.get("findings", []))
    proposals = result["candidates"]
    selected = result["selected_plan"]
    accepted_violation = bool(
        selected and not deterministic_checks(AuditorAgent(), selected, baseline=result["scenario"])["passes"]
    )
    return {
        "id": case["scenario"]["id"], "seconds": elapsed,
        "shortage_expected": case.get("expected_shortage"), "shortage_observed": original.get("shortage"),
        "gaps_expected": sum(expected_codes.values()), "gaps_detected": sum((expected_codes & actual_codes).values()),
        "unexpected_gaps": sum((actual_codes - expected_codes).values()), "actual_codes": sorted(actual_codes),
        "proposal_count": len(proposals),
        "within_resource_limits": sum(row.get("hard_violation") is False for row in proposals),
        "passing_alternatives": sum(row.get("passes") is True for row in proposals),
        "accepted_hard_violations": int(accepted_violation),
        "selected_shortage": 0 if selected is not None and not accepted_violation else None,
        "escalation_expected": case["expected_escalation"],
        "escalation_observed": result["escalation"] is not None, "escalation": result["escalation"],
        "calls": len(result["calls"]), "status": result["status"],
        "input_tokens": sum((call.get("usage") or {}).get("input_tokens", 0) for call in result["calls"]),
        "output_tokens": sum((call.get("usage") or {}).get("output_tokens", 0) for call in result["calls"]),
        "claims": claim_records(result, reviews or {}, fixture=fixture), "error": result.get("error"),
    }


def retrieval_benchmark(retriever):
    queries = json.loads((ROOT / "data/retrieval_benchmark.json").read_text())
    rows = []
    for case in queries:
        pages = retriever.retrieve(case["query"], hazard=case["hazard"])
        matches = []
        for page in pages:
            document = page["id"].split("-")[0] if "section" in page else "fnss"
            relevant_pages = set(case["relevant_pages"].get(document, []))
            covered = set(range(page["page"], page.get("page_end", page["page"]) + 1))
            matches.append({"id": page["id"], "document": document, "pages": sorted(covered),
                            "relevant": bool(covered & relevant_pages)})
        rows.append({"id": case["id"], "passages": matches, "hit_at_5": any(x["relevant"] for x in matches)})
    passages = [x for row in rows for x in row["passages"]]
    return {"retriever": retriever.name, "queries": rows,
            "passage_relevance": ratio(sum(x["relevant"] for x in passages), len(passages)),
            "precision_at_5": ratio(sum(x["relevant"] for x in passages), 5 * len(rows)),
            "hit_at_5": ratio(sum(row["hit_at_5"] for row in rows), len(rows))}


def summarize(rows, retrieval):
    with_gold_shortage = [row for row in rows if row["shortage_expected"] is not None]
    claims = [claim for row in rows for claim in row["claims"]]
    reviewed = [claim for claim in claims if claim["support_review"] is not None]
    seconds = sorted(row["seconds"] for row in rows)
    return {
        "calculation_accuracy": ratio(sum(row["shortage_expected"] == row["shortage_observed"]
                                         for row in with_gold_shortage), len(with_gold_shortage)),
        "constraint_compliance": ratio(sum(row["within_resource_limits"] for row in rows),
                                         sum(row["proposal_count"] for row in rows)),
        "retrieval_relevance": retrieval["passage_relevance"],
        "groundedness": {**ratio(sum(claim["support_review"] for claim in reviewed), len(reviewed)),
                         "unreviewed": len(claims) - len(reviewed),
                         "quote_provenance": ratio(sum(claim["quote_verified"] for claim in claims), len(claims))},
        "gap_detection_rate": ratio(sum(row["gaps_detected"] for row in rows), sum(row["gaps_expected"] for row in rows)),
        "escalation_accuracy": ratio(sum(row["escalation_expected"] == row["escalation_observed"]
                                         and not row["error"] for row in rows), len(rows)),
        "latency": {"unit": "seconds", "cases": len(seconds), "median": statistics.median(seconds),
                    "p95": seconds[(95 * len(seconds) + 99) // 100 - 1], "total": sum(seconds)},
        "accepted_hard_violations": sum(row["accepted_hard_violations"] for row in rows),
    }
