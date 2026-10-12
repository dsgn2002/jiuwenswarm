# Research-proposal pipeline

Problem framing → literature review → potential directions → research proposal.

This package adds the front half of the auto-researcher loop that openjiuwen's
autoResearch (`openjiuwen.rsi.artifact_rsi.paper_opt.auto_research`, agent-core
commit `d0bd833` as pinned in `pyproject.toml`) does not have, and works around
the problems found when running its literature tools for real.

## Limitations of openjiuwen autoResearch for the auto-researcher paradigm

Found by reading the pinned source and by running its topic-survey tools on the
topic "evolution of activation functions in deep learning".

| # | Limitation | Evidence | Status here |
| --- | --- | --- | --- |
| 1 | **No problem-framing, direction or proposal stage.** The manager's modules are `topic_survey → experiment_design → code_implementation → experiment_execution → reflection → reporting`; task modes are only `create_new_paper` and `modify_paper`. The survey prompt mentions a later "Idea Generation" stage that does not exist. | `modules/manager/schemas.py` (`ModuleId`, `TaskMode`), `pipeline/transitions.py`, `modules/topic_survey/prompts/system.md` | **Added**: `ProblemFrame`, `DirectionMap` and `Proposal` stages in `pipeline.py` |
| 2 | **Free web search ships disabled.** `WebFreeSearchTool` enables DuckDuckGo/Bing only via env flags or `enabled_engines`; default is "all free search engines are disabled". | `harness/tools/web/free_search.py` (`_search_free`) | **Fixed**: `OpenJiuwenWebBackend` always passes `enabled_engines` |
| 3 | **Off-topic fallback results.** When DuckDuckGo returns an anti-bot challenge the tool falls back to Bing, which returned IRS tax forms for "GLU variants SwiGLU transformer feed-forward" and Windows activation pages for "activation function benchmark comparison". Nothing filters them. | live runs, 2026-10-11/12 | **Fixed**: every web result must pass `relevance_score`; live run rejected 61 results |
| 4 | **Web search is the only literature source.** No scholarly API, so recall depends on scraping general search engines. | `TopicSurveyToolsRail` registers only `free_search`, `fetch_webpage`, `download_survey_source` (+ optional paid search) | **Added**: `ArxivBackend` (arXiv export API, phrase-then-AND queries) |
| 5 | **Citations require a DOI.** `citation_exclusion_reasons` marks any source without a DOI as evidence-only; 13 of 14 papers in our survey were excluded, including arXiv papers. | `modules/topic_survey/citations.py` | **Fixed for arXiv**: DOIs `10.48550/arXiv.<id>` derived automatically. Pre-arXiv papers without DOIs (e.g. Nair & Hinton 2010) still cannot be cited by the openjiuwen host. |
| 6 | **Model access is API-key only.** Agents build a model from `API_KEY`/`API_BASE`; there is no way to run with an operator or another agent providing the reasoning. | `modules/topic_survey/agent.py` (`_build_model_from_config`) | **Added**: `ReasoningModel` seam with `OpenJiuwenReasoningModel` (wraps any openjiuwen `Model`) and `FileExchangeReasoningModel` (prompt/answer files, resumable) |
| 7 | **TLS verification off by default** for the survey model client (`verify_ssl=False`). | `modules/topic_survey/agent.py` | **Avoided**: this package's CLI builds its model with `verify_ssl=True` |
| 8 | **No grounding check between stages.** The survey agent is told not to invent sources, but nothing verifies that later stages cite only retrieved papers. | prompts only | **Added**: `grounding.py` rejects unknown source ids, unknown gap ids and proposals that do not cite the chosen direction's sources; the model gets one repair round with the exact errors |
| 9 | **Weak ranking.** Keyword relevance saturates (most kept results contain every query word), so one productive query can crowd out others. | live run: 12/12 papers scored 1.0 | **Mitigated**: `select_diverse` fills slots round-robin across queries; a topic filter against the framed key concepts drops papers that match one query but not the problem |
| 10 | **arXiv rate limits fail silently.** Bursts of API calls get HTTP 429; retrieval carried on with 2 web results and would have reviewed them as if complete. | live run 2026-10-12 | **Fixed**: `ArxivBackend` spaces requests ≥3 s and retries 429/503 with backoff; the pipeline stops (without caching) when a backend failed and fewer than half the requested papers came back |
| 11 | **No memory across research sessions.** autoResearch runs are independent; the context-engine compressors and their recall archive are scoped to one session. | `core/context_engine/processor/*`, `harness/tools/compression_recall.py` ("session-isolated") | **Added**: `ResearchLibrary` (below) |

Remaining gaps, not fixed here:

- arXiv-only scholarly search misses pre-arXiv and non-arXiv venues; in the live run the original GELU paper was not retrieved (a follow-up, SGELU, was), and one off-topic speaker-verification paper passed the filters.
- Relevance is lexical; an embedding or citation-graph signal would rank better.
- The pipeline is not yet registered as an `ARTIFACT:*` provider in `rsi/provider_factory.py`; it runs standalone and hands off through files (below).

## Research memory across sessions

Run one session per topic (multimodal agents and tool calling today, reasoning
tomorrow) against the same `--library`; each run builds on the earlier ones.

```bash
python -m jiuwenswarm.agents.harness.common.research_proposal --library ~/research-library \
    --run-dir runs/day1-agents --problem "What are the open problems in how multimodal agents call tools?"
python -m jiuwenswarm.agents.harness.common.research_proposal --library ~/research-library \
    --run-dir runs/day2-reasoning --problem "What open problems remain in LLM reasoning with tool use?"
python -m jiuwenswarm.agents.harness.common.research_proposal --library ~/research-library \
    --recall "verifying tool results during agent reasoning"     # inspect the memory
```

| When | What the library does |
| --- | --- |
| Before framing | `recall()` finds earlier sessions, their gaps and papers related to the new problem; they are added to the framing prompt and saved as `00_memory_recall.md` |
| Retrieval | `LibraryBackend` searches the library like any other backend; papers read before come back with their earlier notes (`prior_notes`), capped at 25% of the slots so old topics do not crowd out the new search |
| After the proposal | `ingest()` merges the corpus, per-paper notes, gaps and chosen direction; a paper read in two sessions lists both topics |

It stores compressed knowledge, not raw text: a paper card (metadata, summary,
findings, limitations, topics, sessions) and a session card (question,
synthesis, gaps, chosen direction). `library.json` is the source of truth;
`memory/*.md` holds one card per paper, session and gap.

### Which openjiuwen memory and compression components fit

| openjiuwen component | What it does | Fit for research memory |
| --- | --- | --- |
| Context-engine compressors (`MicroCompact`, `ToolResultWindow`, `ToolResultBudget`, `MessageSummaryOffloader`, `RoundLevel`/`Dialogue`/`CurrentRound` compressors, `FullCompactProcessor`) | Shrink one conversation's context: clear stale tool results, offload or summarise old rounds, full compaction when over budget | No: per-session context window management |
| Forked compressors (`SessionMemoryCompressor`, `RuleCompressionPipeline` for diff/HTML/JSON/log/search-result content) + `CompressionArchive` with BM25 `compression_recall` tool | Archive what was compressed and recall it later | No: archives and recall are session-isolated |
| **Lite memory** (`MemoryIndexManager`) | Markdown files indexed into SQLite FTS5 (trigram) plus optional sqlite-vec vectors, hybrid search | **Used**: `OpenJiuwenMemoryIndex` indexes the library's cards; pass an embedding config for hybrid search |
| `LongTermMemory`, `SemanticStore`, update checker | Vector-store memory with redundancy/conflict checks | Possible backend; needs an embedding model and vector store |
| `GraphMemory` | LLM entity/relation extraction into a graph | Future: paper → method → gap links; needs LLM extraction |
| `DreamingOrchestrator` | Idle-time periodic consolidation | Future: merge and re-summarise cards between sessions |
| External providers (mem0, JiuwenMemory, OpenViking, LakeBase, AgentArts) | Hosted memory services | Possible backends; external services |

Problems found in openjiuwen lite memory while building this, and the workaround here:

1. **Breaks with filelock 4.x.** openjiuwen pins `filelock>=3.25` with no upper bound; under 4.x every file read fails ("Cannot release a lock ... not held by this task") and nothing is indexed. Needs `filelock<4`; `refresh()` detects the empty index and the library falls back to keyword search.
2. **Full-text index wiped on every reopen.** `_fts_table_is_legacy()` looks for `tokenize=trigram` but the table is created with `tokenize='trigram'`, so each restart drops `chunks_fts` as legacy and never refills it. `refresh()` detects an empty FTS table and forces a reindex.
3. **Inverted keyword scores.** `bm25_rank_to_score` maps FTS5 ranks (negative, more negative = better) to `1/(1 - rank)`, so better matches score lower and the default `keywordMinScore` drops the best ones. The index keeps every hit and ranks by openjiuwen's result order.
4. **All-words matching.** Multi-word FTS queries rarely match a card from another topic. The index also queries each content word and fuses rankings (reciprocal rank fusion).
5. **No `sys_operation` gives an `UnboundLocalError`** rather than a clear error; the index registers a local `SysOperation`.

Two-day demo (2026-10-12, `examples/research_memory/`): day 1 (multimodal
agents, tool calling) stored 8 papers, 3 gaps and a proposal on verifying tool
outputs against visual evidence. Day 2 (reasoning) was framed with day 1's gaps
in its prompt, reused 2 day-1 papers with their notes alongside 6 new ones, and
proposed measuring whether reasoning steps depend on tool results, explicitly
the reasoning-side follow-up to day 1's gap G1. The library ended with 14
papers, 2 of them shared across both topics.

## Pipeline

| Stage | Who | Artifact in `run_dir` |
| --- | --- | --- |
| 1. Problem framing | model | `01_problem_frame.json` |
| 2a. Retrieval | search backends (code) | `02_corpus.json`, `sources/` (PDFs) |
| 2b. Literature review | model, grounded | `03_literature_review.json`, `research_summary.md` |
| 3. Directions | model, grounded | `04_directions.json`, `04_directions.md` |
| 4. Proposal | model, grounded | `05_proposal.json`, `proposal.md` |

Completed stages are loaded from disk, so a run can stop and continue.

## Usage

```bash
# An operator or another agent answers each stage through files:
python -m jiuwenswarm.agents.harness.common.research_proposal \
    --run-dir runs/act --problem "What open problem in Transformer activation design is worth researching next?"
# exits with status 3 and prints the prompt path; write the JSON answer to the
# printed response path, then run the same command again (no --problem needed).

# Or with an API model, configured the way openjiuwen expects:
API_KEY=... API_BASE=... MODEL_NAME=... MODEL_PROVIDER=OpenAI \
python -m jiuwenswarm.agents.harness.common.research_proposal --model openjiuwen --run-dir runs/act --problem "..."
```

In code, pass any openjiuwen `Model` (for example `ArtifactEngineRequest.model`):

```python
pipeline = ResearchProposalPipeline(
    OpenJiuwenReasoningModel(request.model),
    LiteratureRetriever(backends=[ArxivBackend(), OpenJiuwenWebBackend()]),
    run_dir,
)
result = await pipeline.run(raw_problem)
```

## Hand-off to the existing autoResearch loop

openjiuwen's experiment-design stage looks for files named `research_summary.md`
in `research_paths` and reads them first (`prefer_research_summary` in
`modules/experiment_design/agent.py`). Pass the pipeline's
`research_summary.md` and `proposal.md` as `research_paths` (and the proposal's
research question as `topic`) to `ManagerRuntime.arun` to continue into design,
code, execution and reporting.

## Live run (2026-10-12)

Problem: "What open problem in activation-function design for Transformers is
worth researching next, and how would we test it?" Seven queries; arXiv and
openjiuwen web search; 12 papers downloaded as PDFs, 61 results rejected; four
directions scored; recommended direction "natively sparse gated feed-forward
units" (gap: gated activations lose ReLU's natural sparsity). Every stage
passed its grounding check on the first attempt; the proposal cites seven
retrieved papers, all with DOIs. The text outputs are in
`examples/transformer_activations/` (downloaded PDFs are not committed).
