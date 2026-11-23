# EndureAI: A Multi-Agent System for Disaster Preparedness Plan Auditing

EndureAI audits existing synthetic county shelter plans against their stated assumptions and public FEMA planning guidance. It tests what happens when those assumptions change, measures resulting gaps, and evaluates alternative allocations for human review. The prototype covers synthetic shelter capacity, availability, accessibility, and backup power.

## Run an audit

Use Python 3.10 or later. The prototype was tested with Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 scripts/run_secure.py data/scenarios/checkpoint_winter_plan.json \
  --perturbations data/scenarios/checkpoint_shelter_loss.json \
  --output runs/shelter-loss
```

Choose a Responses API model available to your account that supports structured outputs. The secure launcher prompts for the model and API key when they are missing. It holds the key in the current process. Use `python3 -m endureai.cli` directly when `OPENAI_MODEL` and `OPENAI_API_KEY` are set. A three-layer audit permits up to 26 logical API calls, including one Research correction attempt. The SDK may retry transport failures. Paid live evaluation begins with one scenario.

The example starts with 450 usable spaces for 400 people. The perturbation makes a 200-space shelter unavailable and retains the submitted allocation for audit. That leaves 250 usable spaces and a 150-space shortage. Perturbation JSON can change demand, the power requirement, or an existing shelter's availability, capacity, accessibility, or backup power. The input validator requires explicitly synthetic data and routes declared operational use to human intervention.

The default retriever uses the local Chroma index. It loads the embedding model and tokenizer on first use and can reuse their caches. Use `--retrieval bm25` for the original page-based retrieval, or `--mode baseline` for the original deterministic workflow. Live mode requires valid configuration and writes a Markdown report and JSON trace.

## Workflow

1. **Coordinator** validates the submitted allocation and records synthetic assumptions, perturbations, and baseline arithmetic in LangGraph state.
2. **Research** retrieves the five most similar FEMA passages after filtering for hazard applicability. It synthesizes guidance with exact quotations. A citation failure permits one correction using the same evidence.
3. **Auditor** reviews the submitted plan and its evidence first. A passing plan goes straight to human review. For a plan with gaps, **Planning** proposes up to three actions per parent using the existing resource inventory. The Auditor rejects hard violations before requesting model review of eligible candidates.
4. **Coordinator** keeps up to two eligible branches per layer over at most three layers. The score weights feasibility 40%, shortage reduction 25%, guidance alignment 20%, and practicality 15%. Partial candidates need at least 70 points to continue. The workflow records prior checks and rejected branches as feedback, then clears a passing alternative for human review or escalates the remaining gaps.

Safety escalations cover missing or contradictory inputs, insufficient or conflicting evidence, unverified citations, exhausted search, unavailable feasible alternatives, and attempts at operational use. An auditor verifies source IDs and exact quotes, and the workflow records the model's assessment of alignment. A human reviews whether the cited text substantively supports a live claim. All results are educational and require human review. Shared working memory lasts one audit. Persistent cross-audit memory is future scope.

## Public reference corpus

The corpus contains selected sections of FEMA's 2010 Functional Needs Support Services guidance, 2019 *Planning Considerations: Evacuation and Shelter-in-Place*, and 2021 *CPG 101 v3*. The extracts record titles, dates, original and government mirror URLs, PDF hashes, sections, PDF page ranges, and hazard tags. Chroma stores overlapping chunks of roughly 400–600 tokens using pretrained MiniLM embeddings and cosine distance. The sources are dated planning guidance. Shelter-specific facts and constraints come from the synthetic inputs.

`scripts/ingest_guidance.py` regenerates the newer extracts from the two cited PDFs using Poppler. The versioned JSON extracts are sufficient to build the index. Staffing, transport, supplies, medical triage, and live hazard feeds are outside the implemented shelter-allocation scope.

## Tests and evaluation

```bash
python3 -m unittest discover -s tests -v
python3 -m endureai.evaluate_checkpoints --output runs/checkpoint_metrics.json
python3 -m endureai.evaluate_live --limit 1
```

[Checkpoint results](evidence/CHECKPOINT_RESULTS.md) report the actual scripted-agent and local-embedding results, the matched-chunk and original BM25 comparisons, and all seven Checkpoint 6 metrics. [Machine-readable measurement](evidence/checkpoint_metrics.json) contains the cases and retrieval hits. The [prior baseline](BASELINE.md) remains reproducible.

The paid evaluator records model calls, usage, per-case metrics, and claim IDs. A human reviewer can map each claim ID to a JSON boolean for substantive support, then rescore the saved run without API calls:

```bash
python3 -m endureai.evaluate_live \
  --rescore runs/live-result/evaluation.json \
  --support-review reviews.json --output runs/reviewed
```

The prototype adapts the CMU research-agent starter, including its extractive summarizer. See [starter attribution](STARTER_INSPECTION.md) for the scaffold's contribution.
