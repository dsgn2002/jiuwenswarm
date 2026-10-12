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

Remaining gaps, not fixed here:

- arXiv-only scholarly search misses pre-arXiv and non-arXiv venues; in the live run the original GELU paper was not retrieved (a follow-up, SGELU, was), and one off-topic speaker-verification paper passed the filters.
- Relevance is lexical; an embedding or citation-graph signal would rank better.
- The pipeline is not yet registered as an `ARTIFACT:*` provider in `rsi/provider_factory.py`; it runs standalone and hands off through files (below).

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
