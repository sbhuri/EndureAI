# EndureAI deterministic baseline (historical)

A working, local four-agent prototype for auditing **synthetic shelter preparedness plans**. Adapted from the uploaded CMU Research Agent Starter Template. Requires Python 3.10+; the runtime and tests use only the standard library. No API key is needed and no model API is called.

## Run in this folder

```bash
python3 -m endureai.cli --mode baseline data/scenarios/winter_shortage.json --output evidence/demo
python3 -m unittest discover -s tests -v
python3 -m endureai.evaluate
```

The first command writes a readable Markdown audit and a detailed JSON trace. The included winter-storm plan has demand 400 and an open capacity of 250: the Auditor detects a 150-space shortage. A synthetic, available 150-space reserve shelter permits a revised allocation, subject to human review. `insufficient_inventory.json` demonstrates unresolved escalation.

Read `evidence/EVALUATION.md` for measurements from the actual run, `evidence/evaluation.json` for expected-versus-observed results and code hashes, and `evidence/test_run.txt` for the recorded test output. Re-running evaluation replaces those evaluation artifacts with a new timestamp and measured timings.

## Architecture

```mermaid
flowchart LR
 C[Coordinator: validate synthetic input] --> R[Research: retrieve FEMA pages]
 R --> A[Auditor: check original plan]
 A --> P[Planning: bounded shelter combinations]
 P --> V[Auditor: verify against original inventory]
 V --> H[Coordinator: proposed revision or escalation]
```

- **CoordinatorAgent:** validates structured input, maintains shared evidence and trace, limits candidates, and requires human review.
- **ResearchAgent:** BM25 keyword retrieval over 164 nonempty pages of the included 166-page FEMA FNSS guidance. Returns page IDs, source links, scores, source text, and extractive summaries using the starter's summarizer. This is extractive retrieval-augmented report generation, not neural/LLM RAG.
- **PlanningAgent:** tries subsets of eligible shelters in increasing subset size, allocating demand within each shelter's capacity. Stops at the first passing alternative or the configured limit (default 32). Never creates resources. This is bounded combinatorial search, not implemented LLM Tree-of-Thought reasoning.
- **AuditorAgent:** independently recomputes capacity, assignments, accessibility/power flags, evidence availability, and unchanged inventory for revisions. Detects over-allocation, closed-shelter assignments, demand mismatch, and unresolved shortages.

Agent roles are separate Python components in `endureai/core.py`. Shared state is retained in each output JSON; there is no cross-run agent memory. LangGraph, ChromaDB, Streamlit, embeddings, and LLM planning were discussed in the earlier design but are **not implemented** in this version. No claims about those systems appear in measured results.

## Inputs and assumptions

Use the JSON files in `data/scenarios/` as templates. Each scenario needs `synthetic: true`, an ID, a hazard label, integer demand, `power_required`, and 1–12 shelters. Each shelter needs a unique ID, integer capacity and assigned counts, and explicit boolean `open`, `available`, `backup_power`, and `accessible` values. Missing or contradictory required data is rejected, rather than silently filled in.

The existing plan is represented by shelter open/assignment fields. Available closed shelters are reserve inventory. The planner may change only open/assignment fields; facility characteristics and demand are immutable. Accessibility and power are simplified whole-facility flags; requiring power makes non-powered sites ineligible for this synthetic exercise. These are **prototype policy assumptions**, not a complete interpretation of FEMA guidance or applicable law. Shortage is demand minus eligible **currently open** capacity, floored at zero. The hazard label is descriptive and does not trigger a hazard-specific model.

## FEMA corpus and provenance

Source: *Guidance on Planning for Integration of Functional Needs Support Services in General Population Shelters*, November 2010, created for FEMA. The direct FEMA PDF returned HTTP 403 during research. The included PDF was downloaded from the [official New York DHSES mirror](https://www.dhses.ny.gov/system/files/documents/2021/10/fnss_guidance.pdf). The original URL, mirror URL, retrieval time, and PDF SHA-256 are recorded in `data/corpus.json`. Citations use 1-based PDF page numbers. This is a historical snapshot, not a claim that the guidance is the latest standard.

`python3 scripts/ingest_fema.py` rebuilds the local corpus from the included PDF if `pypdf` is installed (`python3 -m pip install pypdf`). Normal runs do not need it or network access. Retrieval queries cover capacity, accessibility, and backup power; the entire document is indexed. Source text is data, never executable instructions. The system does not let plan text override checks.

## Evaluation and limits

Ten authored synthetic regression scenarios compare expected finding-code sets, shortage, and final status. Separate unit tests cover input errors, state preservation, bounded search, missing evidence, real-page citations, unknown queries, altered inventory, and 144 small capacity/demand combinations. Three manually selected retrieval queries check whether at least one known relevant page appears in the top five.

These tests are development evidence, **not independent validation**, not a statistical generalization study, and not a measure of real-world safety. Code-level precision/recall treats each finding code per scenario once; it does not assess all natural-language claims. Search truncation can miss feasible plans. The first passing plan is not globally optimal. The PDF can be outdated, lexical retrieval can miss relevant passages, and the source does not justify universal numeric shelter thresholds. Empty evidence blocks passing results, but positive retrieval scores do not prove semantic sufficiency.

No free-form PDF plan ingestion, transport constraints, staffing, water/food inventory, geographic routing, medical triage, live conditions, LLM-generated alternatives, or operational deployment is implemented. Every result requires human review. `CHECKS_PASSED` means only that implemented synthetic checks passed. This tool is an educational prototype, not an emergency decision system.

## Files

- `starter_reference/`: original uploaded files, unchanged, for provenance.
- `STARTER_INSPECTION.md`: structure, reusable code, and adaptation decisions.
- `endureai/`: runnable four-agent implementation, CLI, and evaluation runner.
- `data/`: extracted FEMA corpus, synthetic plans, expected outcomes.
- `tests/`: automated checks; `scripts/`: fixture generation and corpus rebuilding.
- `evidence/`: measured results and per-scenario audit reports/traces.

No public repository has been created. The starter contains no LICENSE file; retain its attribution and confirm course redistribution terms before publishing the original course files. The included FEMA document contains third-party appendices and remains attributed to its source.
