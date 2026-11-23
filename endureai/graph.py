"""LangGraph orchestration and bounded Tree of Thought for synthetic shelter audits."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import re
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .core import AuditorAgent as ConstraintAuditor, validate
from .llm import AuditorAgent, DECISION, ModelFailure, PlanningAgent, ResearchAgent, check_schema, materialize
from .rag import SemanticRetriever

BEAM_WIDTH = 2
MAX_DEPTH = 3
MIN_SCORE = 70
ALIGNMENT = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}
SAFETY_REASON = "Educational synthetic audit only. A human reviewer must assess every result."
ID_PATTERN = re.compile(r"[a-zA-Z0-9_-]{1,64}")
HAZARD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z -]{0,63}")
SCENARIO_FIELDS = {"id", "hazard", "synthetic", "demand", "power_required", "shelters", "purpose"}
SHELTER_FIELDS = {"id", "open", "assigned", "capacity", "available", "accessible", "backup_power"}
PERTURBATION_FIELDS = {"demand", "power_required", "shelters"}
SHELTER_CHANGE_FIELDS = {"capacity", "available", "accessible", "backup_power"}


class AuditInputError(ValueError):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def validate_audit_input(scenario):
    if isinstance(scenario, dict) and scenario.get("purpose", "simulation") != "simulation":
        raise AuditInputError("OPERATIONAL_USE", "EndureAI accepts synthetic planning exercises. Direct actual emergencies to local officials.")
    try:
        plan = validate(scenario, allow_unavailable_open=True)
    except ValueError as error:
        raise AuditInputError("INPUT_CLARIFICATION", str(error)) from None
    if set(plan) - SCENARIO_FIELDS or any(set(shelter) != SHELTER_FIELDS for shelter in plan["shelters"]):
        raise AuditInputError("INPUT_CLARIFICATION", "Supply only the documented structured scenario and shelter fields.")
    if not ID_PATTERN.fullmatch(plan["id"]) or not HAZARD_PATTERN.fullmatch(plan["hazard"]):
        raise AuditInputError("INPUT_CLARIFICATION", "Supply a short synthetic scenario ID and a hazard label.")
    if any(not ID_PATTERN.fullmatch(shelter["id"]) for shelter in plan["shelters"]):
        raise AuditInputError("INPUT_CLARIFICATION", "Supply short synthetic shelter identifiers.")
    return plan


def stress_plan(plan, changes):
    """Change assumed conditions while retaining the submitted plan's actions and assignments."""
    if not isinstance(changes, dict) or not changes or set(changes) - PERTURBATION_FIELDS:
        raise AuditInputError("INPUT_CLARIFICATION", "Perturbations must specify demand, power_required, or shelter conditions.")
    scenario = deepcopy(plan)
    updates = changes.get("shelters", {})
    inventory = {shelter["id"]: shelter for shelter in scenario["shelters"]}
    if not isinstance(updates, dict) or set(updates) - inventory.keys():
        raise AuditInputError("INPUT_CLARIFICATION", "Perturbations must reference existing synthetic shelters.")
    for identifier, change in updates.items():
        if not isinstance(change, dict) or not change or set(change) - SHELTER_CHANGE_FIELDS:
            raise AuditInputError("INPUT_CLARIFICATION", "Change capacity, availability, accessibility, or backup power only.")
        inventory[identifier].update(change)
    scenario.update({key: value for key, value in changes.items() if key != "shelters"})
    return validate_audit_input(scenario)


def max_model_calls(depth):
    # Two Research calls, an original audit, then Planning + at most three
    # Auditor calls per expanded parent, plus one Coordinator synthesis per layer.
    if type(depth) is not int or not 1 <= depth <= MAX_DEPTH:
        raise ValueError("max_rounds must be 1–3")
    return 3 + 4 * (1 + BEAM_WIDTH * (depth - 1)) + depth


class AuditState(TypedDict, total=False):
    submitted: dict
    perturbations: dict | None
    scenario: dict
    base_audit: dict
    original: dict
    evidence: dict
    research: dict
    original_review: dict
    depth: int
    frontier: list[dict]
    pending: list[dict]
    planning_rounds: list[dict]
    candidates: list[dict]
    feedback: list[dict]
    decisions: list[dict]
    passing: list[dict]
    selected_plan: dict | None
    status: str
    escalation: dict | None
    next_step: str
    error: str


def deterministic_checks(auditor, plan, *, baseline=None):
    checks = auditor.run(plan, {}, baseline=baseline, require_topic_evidence=False)
    for finding in checks["findings"]:
        finding["basis"] = "synthetic constraint"
    return checks


def score_candidate(original, plan, original_checks, checks, review):
    """Return points on a transparent 100-point rubric after hard constraints pass."""
    initial = original_checks["shortage"]
    reduction = max(0, initial - checks["shortage"]) / initial if initial else float(checks["shortage"] == 0)
    assigned = sum(shelter["assigned"] for shelter in plan["shelters"])
    coverage = assigned / original["demand"] if original["demand"] else 1.0
    changed = sum(
        (before["open"], before["assigned"]) != (after["open"], after["assigned"])
        for before, after in zip(original["shelters"], plan["shelters"], strict=True)
    )
    components = {
        "feasibility": round(40 * coverage, 4),
        "shortage_reduction": round(25 * reduction, 4),
        "guidance_alignment": 20 * ALIGNMENT[review["guidance_alignment"]],
        "practicality": round(15 * (1 - changed / len(plan["shelters"])), 4),
    }
    return {"points": round(sum(components.values()), 4), "components": components,
            "spaces_recovered": max(0, initial - checks["shortage"]), "changed_shelters": changed}


def allocation_key(plan):
    return tuple(sorted((x["id"], x["open"], x["assigned"]) for x in plan["shelters"]))


def branch_order(branch):
    return (-branch["score"]["points"], -branch["score"]["spaces_recovered"],
            branch["score"]["changed_shelters"], branch["branch"])


def escalation(code, detail):
    return {"escalation": {"code": code, "detail": detail}, "status": "REVIEW_REQUIRED", "next_step": "end"}


class GraphCoordinator:
    """Use per-invocation shared state for assumptions, evidence and bounded audit feedback."""

    def __init__(self, backend, *, retriever=None, max_rounds=MAX_DEPTH, strategy="beam"):
        max_model_calls(max_rounds)
        if strategy not in {"beam", "linear"}:
            raise ValueError("Unknown search strategy")
        self.backend = backend
        self.retriever = retriever
        self.max_depth = max_rounds
        self.strategy = strategy
        self.researcher = ResearchAgent(backend)
        self.planner = PlanningAgent(backend)
        self.auditor = AuditorAgent(backend)
        self.constraints = ConstraintAuditor()
        graph = StateGraph(AuditState)
        graph.add_node("Coordinator", self.prepare)
        graph.add_node("Research", self.research)
        graph.add_node("AuditSubmittedPlan", self.audit_original)
        graph.add_node("Planning", self.plan)
        graph.add_node("Auditor", self.audit_branches)
        graph.add_node("CoordinatorDecision", self.decide)
        graph.add_edge(START, "Coordinator")
        graph.add_conditional_edges("Coordinator", self.route, {"research": "Research", "end": END})
        graph.add_conditional_edges("Research", self.route, {"audit": "AuditSubmittedPlan", "end": END})
        graph.add_conditional_edges("AuditSubmittedPlan", self.route, {"plan": "Planning", "end": END})
        graph.add_conditional_edges("Planning", self.route, {"audit": "Auditor", "end": END})
        graph.add_conditional_edges("Auditor", self.route, {"decide": "CoordinatorDecision", "end": END})
        graph.add_conditional_edges("CoordinatorDecision", self.route, {"plan": "Planning", "end": END})
        self.workflow = graph.compile()

    @staticmethod
    def route(state):
        return state["next_step"]

    def prepare(self, state):
        try:
            submitted = validate_audit_input(state["submitted"])
            plan = submitted if state["perturbations"] is None else stress_plan(submitted, state["perturbations"])
        except AuditInputError as error:
            return escalation(error.code, str(error))
        base = deterministic_checks(self.constraints, submitted)
        original = deterministic_checks(self.constraints, plan)
        return {"scenario": plan, "base_audit": base, "original": original, "next_step": "research",
                "frontier": [{"branch": "root", "plan": plan, "checks": original}]}

    def research(self, state):
        try:
            if self.retriever is None:
                self.retriever = SemanticRetriever()
            pages = self.retriever.run(state["scenario"], state["original"])
        except Exception as error:
            return {**escalation("RETRIEVAL_FAILED", "Check whether the FEMA index and local embedding model are available."),
                    "error": f"Retrieval failed ({type(error).__name__})."}
        if not pages:
            return escalation("EVIDENCE_INSUFFICIENT", "No applicable FEMA passages were retrieved.")
        evidence = {"guidance": pages}
        try:
            research, _ = self.researcher.run(state["scenario"], evidence, graph=True)
        except ModelFailure as error:
            return {"evidence": evidence, "error": str(error),
                    **escalation("RESEARCH_UNVERIFIED", "A human reviewer must verify the research citations and relevance.")}
        outputs = {"evidence": evidence, "research": research}
        if research["evidence_status"] != "sufficient":
            return {**outputs, **self.evidence_escalation(research)}
        return {**outputs, "next_step": "audit"}

    @staticmethod
    def evidence_escalation(review):
        status = review["evidence_status"]
        return escalation(f"EVIDENCE_{status.upper()}", f"FEMA evidence was assessed as {status}. Confirm applicability and resolve open questions in human review.")

    def audit_original(self, state):
        plan = state["scenario"]
        try:
            review = self.auditor.review(plan, plan, state["original"], state["research"], state["evidence"]["guidance"])
        except ModelFailure as error:
            return {"error": str(error), **escalation("AUDIT_UNVERIFIED", "The original audit needs human verification.")}
        if review["evidence_status"] != "sufficient":
            return {"original_review": review, **self.evidence_escalation(review)}
        if state["original"]["passes"]:
            if review["acceptable"] and review["guidance_alignment"] == "supported":
                return {"original_review": review, "status": "CHECKS_PASSED", "selected_plan": plan, "next_step": "end"}
            return {"original_review": review, **escalation("AUDITOR_REJECTED", "The Auditor flagged concerns requiring human review.")}
        return {"original_review": review, "next_step": "plan",
                "feedback": [{"branch": "root", "checks": state["original"], "concerns": review["concerns"]}]}

    def plan(self, state):
        depth = state["depth"] + 1
        pending, rounds = [], []
        try:
            for parent in state["frontier"]:
                context = {"allocation": parent["plan"]["shelters"], "checks": parent["checks"]}
                proposals = self.planner.expand(state["scenario"], context, state["research"], state["feedback"], depth)
                rounds.append({"depth": depth, "parent": parent["branch"], "limitations": proposals["limitations"]})
                choices = proposals["alternatives"]
                if self.strategy == "linear":
                    choices = choices[:1]
                for index, proposal in enumerate(choices, 1):
                    pending.append({"branch": f"{parent['branch']}.{index}", "parent": parent["branch"],
                                    "depth": depth, "proposal": proposal})
        except ModelFailure as error:
            return {"error": str(error), **escalation("PLANNING_FAILED", "A bounded planning step failed. Review the unresolved findings.")}
        return {"depth": depth, "pending": pending, "planning_rounds": state["planning_rounds"] + rounds, "next_step": "audit"}

    def audit_branches(self, state):
        original = state["scenario"]
        seen = {allocation_key(original)}
        seen.update(allocation_key(row["plan"]) for row in state["candidates"] if "plan" in row)
        rows, frontier, passing, feedback = [], [], [], []
        for pending in state["pending"]:
            row = dict(pending)
            rows.append(row)
            try:
                plan = materialize(original, row["proposal"])
            except ModelFailure as error:
                row.update(rejection=str(error), hard_violation=True)
                continue
            checks = deterministic_checks(self.constraints, plan, baseline=original)
            row.update(plan=plan, checks=checks)
            key = allocation_key(plan)
            demand_exceeded = sum(x["assigned"] for x in plan["shelters"]) > plan["demand"]
            hard = [f["code"] for f in checks["findings"] if f["code"] not in {"CAPACITY_SHORTAGE", "DEMAND_MISMATCH"}]
            if demand_exceeded:
                hard.append("DEMAND_EXCEEDED")
            if hard:
                row.update(rejection=", ".join(hard), hard_violation=True)
                continue
            row["hard_violation"] = False
            if key in seen:
                row["rejection"] = "Duplicate allocation"
                continue
            seen.add(key)
            try:
                review = self.auditor.review(original, plan, checks, state["research"], state["evidence"]["guidance"])
            except ModelFailure as error:
                row["rejection"] = str(error)
                return {"candidates": state["candidates"] + rows, "error": str(error),
                        **escalation("AUDIT_UNVERIFIED", "A candidate's evidence needs human verification.")}
            row["review"] = review
            if review["evidence_status"] != "sufficient":
                return {"candidates": state["candidates"] + rows, **self.evidence_escalation(review)}
            row["score"] = score_candidate(original, plan, state["original"], checks, review)
            row["passes"] = checks["passes"] and review["acceptable"] and review["guidance_alignment"] == "supported"
            if row["passes"]:
                passing.append(row)
            elif row["score"]["points"] >= MIN_SCORE and review["guidance_alignment"] != "unsupported":
                frontier.append(row)
            else:
                row["rejection"] = "Below the continuation threshold or unsupported by guidance"
        for row in rows:
            feedback.append({"branch": row["branch"], "checks": row.get("checks"),
                             "rejection": row.get("rejection"), "concerns": row.get("review", {}).get("concerns", [])})
        return {"candidates": state["candidates"] + rows, "passing": sorted(passing, key=branch_order),
                "frontier": sorted(frontier, key=branch_order)[:BEAM_WIDTH],
                "feedback": state["feedback"] + feedback, "next_step": "decide"}

    def decide(self, state):
        passing = state["passing"]
        frontier = state["frontier"]
        if not passing and state["depth"] >= self.max_depth:
            return escalation("SEARCH_LIMIT", "The search reached its depth limit with unresolved findings.")
        if not passing and not frontier:
            return escalation("NO_FEASIBLE_ALTERNATIVE", "No branch remained eligible for refinement. Review the rejected or unresolved proposals.")
        try:
            decision = self.backend.ask("Coordinator",
                "Choose accept_for_review only if an independent passing candidate exists. Revise only "
                "when a branch and budget remain. Escalate unresolved serious concerns. All outcomes are synthetic.",
                {"depth":state["depth"], "maximum_depth":self.max_depth, "passing_branches":[x["branch"] for x in passing],
                 "continuation_branches":[x["branch"] for x in frontier], "feedback":state["feedback"]}, DECISION)
            check_schema(decision, DECISION)
        except ModelFailure as error:
            return {"error": str(error), **escalation("COORDINATOR_FAILED", "Human review is required to resolve the audit.")}
        updates = {"decisions": state["decisions"] + [decision]}
        if passing and decision["action"] == "accept_for_review":
            return {**updates, "selected_plan": passing[0]["plan"], "status": "REVISION_FOR_REVIEW", "next_step": "end"}
        if decision["action"] == "revise" and frontier and state["depth"] < self.max_depth and not passing:
            return {**updates, "next_step": "plan"}
        return {**updates, **escalation("COORDINATOR_ESCALATION", "The Coordinator did not clear a result for review. Inspect the recorded decision and audit checks.")}

    def run(self, scenario, perturbations=None):
        start = len(self.backend.records)
        started_at = datetime.now(timezone.utc).isoformat()
        state = self.workflow.invoke({
            "submitted": scenario, "perturbations": perturbations, "depth": 0,
            "planning_rounds": [], "candidates": [], "feedback": [], "decisions": [],
            "selected_plan": None, "escalation": None, "status": "REVIEW_REQUIRED",
        })
        state.pop("submitted", None)
        state.pop("next_step", None)
        for field in ("frontier", "pending", "passing"):
            state.pop(field, None)
        if "scenario" not in state:
            state["perturbations"] = None
        return {
            "scenario_id": state.get("scenario", {}).get("id", "invalid-input"),
            "mode": "llm-live" if self.backend.live else "llm-mocked-test",
            "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
            "scope": SAFETY_REASON, "human_review_required": True, "max_rounds": self.max_depth,
            "retrieval": getattr(self.retriever, "name", "chroma-minilm"),
            "corpus_sha256": getattr(self.retriever, "hash", None),
            "code_sha256": {path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sorted(Path(__file__).parent.glob("*.py"))},
            "strategy": self.strategy, "calls": deepcopy(self.backend.records[start:]), **state,
        }
