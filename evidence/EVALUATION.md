# Measured evaluation

Run: 2026-11-21T23:04:54.910176+00:00
Mode: deterministic-extractive; Python 3.14.5

Scenario checks: 10/10
Finding-code precision: 1.0; recall: 1.0

| Scenario | Match | Status | Original shortage | Candidates | Seconds |
|---|---|---|---:|---:|---:|
| winter_shortage | True | REVISION_FOR_REVIEW | 150 | 4 | 0.070217 |
| insufficient_inventory | True | REVIEW_REQUIRED | 150 | 2 | 0.069381 |
| balanced | True | CHECKS_PASSED | 0 | 0 | 0.069239 |
| over_allocation | True | REVISION_FOR_REVIEW | 150 | 4 | 0.069238 |
| power_gap | True | REVIEW_REQUIRED | 400 | 2 | 0.075569 |
| access_gap | True | REVIEW_REQUIRED | 400 | 2 | 0.078042 |
| closed_allocation | True | REVISION_FOR_REVIEW | 400 | 4 | 0.083633 |
| zero_demand | True | CHECKS_PASSED | 0 | 0 | 0.074717 |
| surplus_assignment | True | REVISION_FOR_REVIEW | 0 | 2 | 0.097539 |
| power_not_required | True | CHECKS_PASSED | 0 | 0 | 0.079511 |

Known-page retrieval hit@5: 3/3

Authored synthetic regression cases, not independent real-world validation. No LLM, live retrieval, or operational accuracy evaluated.
