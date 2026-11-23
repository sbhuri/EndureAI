# Checkpoint evaluation

Measured 2026-11-23T07:15:51.274528+00:00 with Python 3.12.14. The offline run used pretrained MiniLM embeddings in ChromaDB and scripted outputs at the model boundary. The index contained 25 FEMA chunks and took 3.187 seconds to build in the measured process. The baseline branch passed 35 tests with its pinned SDK before implementation. The final suite passed 50 tests, including actual Responses SDK requests over mock HTTP.

| Checkpoint 6 metric | Bounded beam | Linear comparison | Measurement |
| --- | ---: | ---: | --- |
| Calculation accuracy | 11/11 | 11/11 | Exact shortage matches among the 11 cases with an expected numerical result. |
| Constraint compliance | 19/20 | 7/8 | Proposals within hard resource limits. A partial allocation can still have unresolved demand. |
| Retrieval relevance | 14/30 | 14/30 | Retrieved chunks overlapping manually labeled relevant PDF pages on six queries. |
| Groundedness | 41/41 | 29/29 | Scripted extractive claims whose statements equal verified source excerpts. Live substantive support requires human scoring. |
| Gap detection rate | 18/18 | 18/18 | Matched intentionally introduced planning weaknesses. Input escalations have no numerical gap label. |
| Escalation accuracy | 13/13 | 12/13 | Matches to expected explicit escalation across 13 cases. Routine human review is required for every output. |
| Latency | 0.133 s median, 0.250 s p95 | 0.137 s median, 0.146 s p95 | Per audit after index construction. Includes local retrieval and scripted agent execution. |

The beam and linear runs accepted zero hard-constraint violations. The scripted comparison intentionally placed a hard violation before a valid sibling in the over-allocation case. Beam search evaluated both siblings and cleared a repair for review. Linear search followed the first and escalated.

Beam search found 5 passing alternatives across 4 repaired cases with 53 scripted agent calls. Linear search found 3 passing alternatives across 3 repaired cases with 40 calls. These results characterize graph behavior under the fixture responses.

The winter perturbation retained the submitted assignments while removing a 200-space shelter from the usable pool. The auditable shortage changed from 0 to 150 spaces. The workflow detected both the unavailable open shelter and the shortage, then escalated the unresolved plan.

## Retrieval comparison

| Retriever | Queries with a relevant result in the top five | Returned passages matching labeled pages |
| --- | ---: | ---: |
| MiniLM and Chroma | 6/6 | 14/30 |
| BM25 on the same chunks | 6/6 | 10/30 |
| Original page-based BM25 | 1/6 | 2/30 |

The matched-chunk comparison holds the corpus, chunk boundaries, and hazard filtering fixed. The original BM25 corpus contains only the 2010 shelter guidance. Page overlap is a proxy for passage relevance. The small development benchmark includes two retrieval questions targeting the additional guides and one hurricane-specific example. Generalization and claim-level support need separate evaluation.

## Live evaluation

The live evaluator exited for missing model configuration before issuing a request. The workspace also had no OpenAI API key configured. Live model-call outcomes, costs, latency, and substantive groundedness remain unmeasured. The paid runner records case outputs and model usage when credentials are provided. Its rescore option applies independent human support judgments to saved claims.
