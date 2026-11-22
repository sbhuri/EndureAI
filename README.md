# EndureAI

LLM-powered four-agent auditing of synthetic disaster shelter preparedness plans, grounded in a local FEMA guidance corpus.

**Development status:** API integration implemented and locally tested. Live LLM evaluation is pending API credentials. No live results are claimed. Earlier deterministic measurements are preserved under `evidence/` and documented in `BASELINE.md`.

## Quick start

Requires Python 3.10 or later. From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
export OPENAI_MODEL="your-supported-responses-model"
python3 scripts/run_secure.py data/scenarios/winter_shortage.json --output runs/demo
```

The secure launcher prompts for your API key with hidden input and holds it in the current process only. If `OPENAI_API_KEY` is already set, no prompt is needed. The model is configurable rather than silently pinned to the course sample's older `gpt-4.1-mini`. Choose a model available to your account that supports Responses structured outputs. Never put credentials in chat, source files, screenshots, or commits. No `.env` file is automatically loaded.

Alternatively, set `OPENAI_API_KEY` in your local environment and run:

```bash
python3 -m endureai.cli data/scenarios/winter_shortage.json --output runs/demo
python3 -m endureai.evaluate_live --limit 1
```

The default CLI mode uses live OpenAI API calls. Missing configuration is an error; there is no silent deterministic fallback. Each scenario uses at most 11 logical model calls over two rounds; the SDK permits one retry per call. API charges apply. Start with one scenario before running all ten (`--limit 10`).

## Four cooperating agents

1. **Research:** retrieves FEMA pages with BM25, then asks the LLM to synthesize findings with page citations and exact supporting quotations.
2. **Planning:** asks the LLM for up to three structured alternative allocations using only the supplied shelter inventory.
3. **Auditor:** independently asks the LLM to review guidance and candidate claims, alongside deterministic capacity, assignment, accessibility, power, and inventory checks.
4. **Coordinator:** asks the LLM whether to accept a passing candidate for review, request revision, or escalate. Code enforces the round bound and prevents acceptance without a passing audit.

The same configurable model serves all four roles in separate requests. Shared scenario state, candidate feedback, and model-call records provide per-run memory. No cross-run memory, LangGraph, ChromaDB, or free-form plan PDF ingestion is implemented. Multiple alternatives and bounded revision are implemented; hidden chain-of-thought is neither requested nor recorded.

The code checks that a cited quotation appears in a retrieved page. This validates provenance, **not semantic entailment**. LLM review is also fallible. All outputs require human review; no result certifies emergency readiness or FEMA compliance.

## Data and scope

Inputs are explicitly synthetic JSON scenarios in `data/scenarios/`. They cover shelter capacity, assignments, closure, availability, accessibility, and backup power. Hazard names are descriptive, not hazard-specific simulation models. Counts and boolean facility flags are supplied assumptions, not independently verified facts. Transport, staffing, medical triage, food, water, and live conditions remain out of scope.

`data/corpus.json` contains page-labelled text extracted from FEMA's November 2010 *Guidance on Planning for Integration of Functional Needs Support Services in General Population Shelters*. Provenance includes the original URL, official New York DHSES mirror URL, retrieval timestamp, and PDF hash. This historical source must not be presented as current legal advice. The original PDF and course ZIP reference files are retained in the local deliverable; the public development branch contains the runnable extracted corpus and attribution, without the full course reference archive.

## Tests and evidence

```bash
python3 -m unittest discover -s tests -v
```

Tests include independently checked arithmetic, a 144-combination allocation grid, LLM workflow test doubles, bounded revision, refusal/incomplete responses, fabricated citations, invented shelters, schema validation, call budgets, and rejection of unsafe approvals. Mock tests are **not** evidence of live model quality.

`evidence/llm_integration_tests.txt` records the local test run. `evidence/LIVE_STATUS.json` records that live evaluation has not yet run. Earlier `evidence/EVALUATION.md` and scenario traces are explicitly deterministic baseline results and must not be represented as LLM results.

Real runs write JSON traces and Markdown audits under `runs/`, ignored by Git until intentionally reviewed. Each call records its agent role, instructions, input, model ID, response ID, token usage, duration, output, and sanitized errors. Source code and corpus hashes accompany the run. `evaluate_live` records expected-versus-observed scenario status, execution failures, tokens, and rejected proposals. Semantic quality still requires independent human review; synthetic status matching alone is insufficient.

## Implementation map

- `endureai/llm.py`: four agents, Responses API adapter, schemas, citation and proposal validation.
- `endureai/core.py`: retrieval and independent constraint tools; deterministic baseline.
- `endureai/cli.py`: default live workflow and report output.
- `endureai/evaluate_live.py`: actual API evaluation runner.
- `tests/`: deterministic and mocked integration tests.
- `STARTER_INSPECTION.md`: original scaffold findings and attribution.

This adapts the CMU research-agent starter. Its extractive summarizer is reused with attribution. The solution ZIP's optional API-writer pattern informed the integration; EndureAI extends API use to the four agent roles. Do not claim the course reference implemented these disaster-specific components.
