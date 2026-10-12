from __future__ import annotations

import json
from pathlib import Path

import pytest

from jiuwenswarm.agents.harness.common.research_proposal import (
    AwaitingModelResponse,
    EmptyCorpus,
    FileExchangeReasoningModel,
    LiteratureRetriever,
    ResearchProposalPipeline,
    StageFailed,
)
from jiuwenswarm.agents.harness.common.research_proposal.schemas import (
    PaperRecord,
    ProblemFrame,
)

RAW_PROBLEM = (
    "Which activation functions should replace ReLU in Transformer feed-forward layers?"
)

FRAME = {
    "raw_problem": RAW_PROBLEM,
    "research_question": "Do gated activations beat smooth gates in Transformer FFNs at equal compute?",
    "motivation": "FFN activations dominate parameter count; the choice is made by habit.",
    "key_concepts": ["GLU", "GELU", "Transformer FFN"],
    "search_queries": ["GLU variants transformer", "GELU activation transformer"],
}

PAPERS = [
    PaperRecord(
        source_id="arxiv:2002.05202",
        title="GLU Variants Improve Transformer",
        url="https://arxiv.org/abs/2002.05202",
        authors=["Noam Shazeer"],
        year="2020",
        doi="10.48550/arXiv.2002.05202",
        abstract="GLU variants in the FFN improve quality over ReLU and GELU.",
        backend="arxiv",
        relevance=1.0,
    ),
    PaperRecord(
        source_id="arxiv:1606.08415",
        title="Gaussian Error Linear Units (GELUs)",
        url="https://arxiv.org/abs/1606.08415",
        authors=["Dan Hendrycks", "Kevin Gimpel"],
        year="2016",
        abstract="GELU weights inputs by value rather than gating by sign.",
        backend="arxiv",
        relevance=0.9,
    ),
]

REVIEW = {
    "synthesis": "Gated FFNs outperform smooth activations in the reported setups.",
    "notes": [
        {
            "source_id": "arxiv:2002.05202",
            "summary": "GLU variants in FFNs.",
            "relevance": 5,
        },
        {
            "source_id": "arxiv:1606.08415",
            "summary": "GELU definition and tests.",
            "relevance": 4,
        },
    ],
    "themes": [
        {
            "name": "Gating",
            "summary": "Multiplicative gates.",
            "source_ids": ["arxiv:2002.05202"],
        }
    ],
    "gaps": [
        {
            "id": "G1",
            "description": "No explanation of why GLUs help.",
            "evidence_source_ids": ["arxiv:2002.05202"],
        }
    ],
}

DIRECTIONS = {
    "directions": [
        {
            "id": "D1",
            "title": "Explain GLU gains",
            "hypothesis": "GLU gains come from multiplicative interactions, not the gate shape.",
            "rationale": "Shazeer reports gains without explanation.",
            "addresses_gap": "G1",
            "builds_on": ["arxiv:2002.05202"],
            "novelty": 4,
            "feasibility": 4,
            "impact": 3,
            "first_experiment": "Swap gate nonlinearities at fixed parameter count.",
        },
        {
            "id": "D2",
            "title": "Smooth gates at scale",
            "hypothesis": "GELU matches GLU when width is matched.",
            "rationale": "Compute-matched comparisons are missing.",
            "addresses_gap": "G1",
            "builds_on": ["arxiv:1606.08415"],
            "novelty": 2,
            "feasibility": 5,
            "impact": 2,
            "first_experiment": "Width-matched GELU vs SwiGLU.",
        },
    ],
    "recommended_id": "D1",
    "selection_rationale": "Highest combined score.",
}

PROPOSAL = {
    "title": "Why do gated linear units help Transformers?",
    "summary": "We test whether GLU gains [@arxiv:2002.05202] come from multiplicative structure.",
    "background": "GELU [@arxiv:1606.08415] smooths ReLU; GLU variants [@arxiv:2002.05202] improve FFNs.",
    "research_questions": ["Does the gate shape matter?"],
    "hypotheses": ["A linear gate recovers most of SwiGLU's gain."],
    "method": "Ablate gate nonlinearities at fixed parameters.",
    "evaluation": "Pre-training perplexity on C4; refuted if linear gates lose >50% of the gain.",
    "risks": ["Small-scale results may not transfer: repeat at two scales."],
    "milestones": ["Reproduce baseline", "Run ablations"],
}


class ScriptedModel:
    """Returns scripted answers per stage name and records every prompt."""

    def __init__(self, answers: dict[str, object]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, str]] = []

    async def complete(self, *, stage: str, system: str, prompt: str) -> str:
        self.calls.append((stage, prompt))
        answer = self.answers[stage]
        return answer if isinstance(answer, str) else json.dumps(answer)


class StaticBackend:
    name = "static"

    def __init__(self, papers: list[PaperRecord]) -> None:
        self.papers = papers

    async def search(self, query: str, max_results: int) -> list[PaperRecord]:
        return [p.model_copy(update={"matched_queries": [query]}) for p in self.papers]


def make_pipeline(model, run_dir: Path, papers=PAPERS) -> ResearchProposalPipeline:
    retriever = LiteratureRetriever(
        backends=[StaticBackend(papers)], downloader=None, pause_seconds=0
    )
    return ResearchProposalPipeline(model, retriever, run_dir)


ALL_ANSWERS = {
    "framing": FRAME,
    "literature_review": REVIEW,
    "directions": DIRECTIONS,
    "proposal": PROPOSAL,
}


async def test_full_run_writes_every_stage_artifact(tmp_path: Path):
    result = await make_pipeline(ScriptedModel(ALL_ANSWERS), tmp_path).run(RAW_PROBLEM)

    assert result.directions.recommended().id == "D1"
    for name in (
        "01_problem_frame.json",
        "02_corpus.json",
        "03_literature_review.json",
        "research_summary.md",
        "04_directions.json",
        "04_directions.md",
        "05_proposal.json",
        "proposal.md",
        "manifest.json",
    ):
        assert (tmp_path / name).is_file(), name
    proposal_md = result.proposal_path.read_text(encoding="utf-8")
    # Inline [@id] citations become numbered references, in order of first use.
    assert "GLU gains [1]" in proposal_md
    assert "GELU [2]" in proposal_md
    assert "1. Noam Shazeer (2020). *GLU Variants Improve Transformer*" in proposal_md
    assert "https://doi.org/10.48550/arXiv.2002.05202" in proposal_md
    summary = result.research_summary_path.read_text(encoding="utf-8")
    assert summary.startswith("# Literature review:")
    assert "**G1.**" in summary


async def test_ungrounded_answer_is_repaired_with_validator_errors(tmp_path: Path):
    bad = dict(
        PROPOSAL,
        background="Prior work [@arxiv:9999.99999] shows gains [@arxiv:2002.05202].",
    )
    model = ScriptedModel(
        {**ALL_ANSWERS, "proposal": bad, "proposal.repair1": PROPOSAL}
    )

    result = await make_pipeline(model, tmp_path).run(RAW_PROBLEM)

    assert result.proposal.title == PROPOSAL["title"]
    repair_stage, repair_prompt = model.calls[-1]
    assert repair_stage == "proposal.repair1"
    assert "unknown source_id 'arxiv:9999.99999'" in repair_prompt


async def test_stage_fails_when_repairs_are_exhausted(tmp_path: Path):
    bad_directions = json.loads(json.dumps(DIRECTIONS))
    bad_directions["directions"][0]["addresses_gap"] = "G9"
    model = ScriptedModel(
        {
            **ALL_ANSWERS,
            "directions": bad_directions,
            "directions.repair1": bad_directions,
        }
    )

    with pytest.raises(StageFailed) as failure:
        await make_pipeline(model, tmp_path).run(RAW_PROBLEM)

    assert failure.value.stage == "directions"
    assert any("unknown gap 'G9'" in e for e in failure.value.errors)
    assert not (tmp_path / "04_directions.json").exists()


async def test_review_must_cover_every_retrieved_source(tmp_path: Path):
    partial = dict(REVIEW, notes=REVIEW["notes"][:1])
    model = ScriptedModel(
        {
            **ALL_ANSWERS,
            "literature_review": partial,
            "literature_review.repair1": REVIEW,
        }
    )

    await make_pipeline(model, tmp_path).run(RAW_PROBLEM)

    repair_prompt = dict(model.calls)["literature_review.repair1"]
    assert "missing ['arxiv:1606.08415']" in repair_prompt


async def test_invalid_json_is_sent_back(tmp_path: Path):
    model = ScriptedModel(
        {
            **ALL_ANSWERS,
            "framing": "I think the question is interesting.",
            "framing.repair1": FRAME,
        }
    )

    result = await make_pipeline(model, tmp_path).run(RAW_PROBLEM)

    assert result.frame.research_question == FRAME["research_question"]
    assert "invalid JSON for ProblemFrame" in dict(model.calls)["framing.repair1"]


async def test_file_exchange_model_resumes_stage_by_stage(tmp_path: Path):
    exchange = tmp_path / "exchange"
    pipeline = make_pipeline(FileExchangeReasoningModel(exchange), tmp_path)

    for stage in ("framing", "literature_review", "directions", "proposal"):
        with pytest.raises(AwaitingModelResponse) as waiting:
            await pipeline.run(RAW_PROBLEM)
        assert waiting.value.stage == stage
        assert waiting.value.prompt_path.is_file()
        waiting.value.response_path.write_text(
            json.dumps(ALL_ANSWERS[stage]), encoding="utf-8"
        )

    result = await pipeline.run(RAW_PROBLEM)
    assert result.proposal_path.is_file()
    # Completed stages are loaded from disk, not asked again.
    assert ProblemFrame.model_validate_json(
        (tmp_path / "01_problem_frame.json").read_text()
    ).search_queries == [
        "GLU variants transformer",
        "GELU activation transformer",
    ]


async def test_empty_retrieval_stops_before_the_review(tmp_path: Path):
    model = ScriptedModel(ALL_ANSWERS)

    with pytest.raises(EmptyCorpus):
        await make_pipeline(model, tmp_path, papers=[]).run(RAW_PROBLEM)

    assert [stage for stage, _ in model.calls] == ["framing"]


async def test_small_corpus_after_backend_failure_stops_and_is_retried(tmp_path: Path):
    class FlakyBackend(StaticBackend):
        def __init__(self) -> None:
            super().__init__(PAPERS[:1])
            self.fail = True

        async def search(self, query: str, max_results: int):
            if self.fail:
                raise RuntimeError("HTTP 429")
            return await super().search(query, max_results)

    flaky = FlakyBackend()
    retriever = LiteratureRetriever(
        backends=[flaky, StaticBackend(PAPERS[:1])],
        downloader=None,
        pause_seconds=0,
        max_papers=6,
    )
    pipeline = ResearchProposalPipeline(ScriptedModel(ALL_ANSWERS), retriever, tmp_path)

    with pytest.raises(EmptyCorpus, match="after backend failures"):
        await pipeline.run(RAW_PROBLEM)
    assert not (
        tmp_path / "02_corpus.json"
    ).exists()  # nothing cached: the next run retries
