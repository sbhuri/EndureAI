# Starter inspection and adaptation

The uploaded `capstone_samples_starter.zip` contains 10 files (plus two directory entries): README, requirements, seven `research_agent` Python files, and one test file with two tests. It is an implemented generic research scaffold with extension TODO comments, not an empty template and not a disaster-auditing application.

| File | Existing function | EndureAI decision |
|---|---|---|
| `research_agent/cli.py` | argparse topic, source count, Markdown output | Adapted the CLI/report pattern to structured scenario input and JSON/Markdown audit output |
| `agent.py` | query generation, source collection, summarization, optional report writing | Replaced single research workflow with four explicit roles and a bounded verification loop |
| `models.py` | SearchResult, SourceDocument, ResearchReport dataclasses | Reused the source/report separation concept; EndureAI uses JSON-serializable dictionaries with page provenance |
| `summarizer.py` | tokenization, sentence scoring, extractive summaries, deduplication | Copied unchanged to `endureai/summarizer.py`; actively used by retrieval and report generation |
| `web_research.py` | Bing HTML scraping, paragraph fetch, URL deduplication | Replaced runtime web scraping with a reproducible local FEMA PDF corpus and BM25 page retrieval |
| `llm_writer.py` | optional OpenAI Responses report writer, enabled from environment | Preserved only in reference; not called by EndureAI and not evaluated |
| `__init__.py` | package marker | New package marker |
| `tests/test_agent.py` | two mocked tests for research and query limits | Preserved reference; new EndureAI tests cover domain constraints and retrieval |
| `requirements.txt` | requests, BeautifulSoup, openai, pytest lower bounds | Main prototype needs no external runtime packages; pypdf is optional for corpus rebuilding |
| `README.md` | setup, usage, enhancement ideas | Replaced with EndureAI instructions and explicit implemented/unimplemented scope |

## Important observations

The starter catches any source-fetch exception and stores the error message as source content, which could then be summarized as evidence. EndureAI instead uses an explicit curated local corpus and fails on corpus loading errors. Starter Bing retrieval depends on an HTML layout and has no official-source restriction. Starter citations are source-level; EndureAI preserves PDF page IDs. The starter README refers to a `.env.example` file, but none exists in the ZIP, and it does not load `.env` files itself. The original tests mock web/LLM behavior and cannot establish live retrieval or LLM quality. No original license file was present.

## What was actually reused

Only `summarizer.py` is directly copied into the executing package. The other original files remain unchanged in `starter_reference/`; their workflow and output patterns informed the replacement implementation. New functionality includes validation, page retrieval, synthetic fixture data, independent auditing, bounded allocation search, provenance, traces, tests, and measured evaluation. No claim is made that the original scaffold contained these components.

The original course archive is preserved only in the local deliverable, not redistributed in this public branch.
