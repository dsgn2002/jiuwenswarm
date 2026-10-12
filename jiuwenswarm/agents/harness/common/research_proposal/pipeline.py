"""Problem framing → literature review → research directions → proposal.

openjiuwen's autoResearch manager starts at a topic survey and goes straight
to experiment design; it has no stage that frames the problem, compares
candidate directions, or writes a proposal. This pipeline supplies those
stages and keeps each hand-off checkable:

====================  ==========================  =================================
Stage                 Produced by                 Artifact (in ``run_dir``)
====================  ==========================  =================================
1. Problem framing    model                       ``01_problem_frame.json``
2a. Retrieval         search backends (code)      ``02_corpus.json``, ``sources/``
2b. Literature review model, grounded             ``03_literature_review.json``,
                                                  ``research_summary.md``
3. Directions         model, grounded             ``04_directions.json`` / ``.md``
4. Proposal           model, grounded             ``05_proposal.json``, ``proposal.md``
====================  ==========================  =================================

A stage whose artifact already exists is loaded instead of re-run, so a run
can stop (for example while ``FileExchangeReasoningModel`` waits for an
answer) and continue later. A model answer that fails schema validation or a
grounding check is sent back once with the exact errors.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from jiuwenswarm.agents.harness.common.research_proposal import (
    grounding,
    prompts,
    render,
)
from jiuwenswarm.agents.harness.common.research_proposal.literature import (
    LiteratureRetriever,
)
from jiuwenswarm.agents.harness.common.research_proposal.model import (
    ReasoningModel,
    extract_json_object,
)
from jiuwenswarm.agents.harness.common.research_proposal.schemas import (
    Corpus,
    DirectionMap,
    LiteratureReview,
    ProblemFrame,
    Proposal,
)

_LOGGER = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StageFailed(RuntimeError):
    def __init__(self, stage: str, errors: list[str]) -> None:
        super().__init__(f"stage {stage!r} failed validation: " + "; ".join(errors))
        self.stage = stage
        self.errors = errors


class EmptyCorpus(RuntimeError):
    """Retrieval found no usable source, so there is nothing to review."""


@dataclass
class PipelineResult:
    run_dir: Path
    frame: ProblemFrame
    corpus: Corpus
    review: LiteratureReview
    directions: DirectionMap
    proposal: Proposal

    @property
    def proposal_path(self) -> Path:
        return self.run_dir / "proposal.md"

    @property
    def research_summary_path(self) -> Path:
        return self.run_dir / "research_summary.md"


class ResearchProposalPipeline:
    def __init__(
        self,
        model: ReasoningModel,
        retriever: LiteratureRetriever,
        run_dir: str | Path,
        *,
        max_repairs: int = 1,
    ) -> None:
        self.model = model
        self.retriever = retriever
        self.run_dir = Path(run_dir)
        self.max_repairs = max_repairs

    # -- artifact helpers -------------------------------------------------------

    def _path(self, name: str) -> Path:
        return self.run_dir / name

    def _load(self, name: str, cls: type[T]) -> T | None:
        path = self._path(name)
        if not path.exists():
            return None
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def _save(self, name: str, value: BaseModel) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._path(name).write_text(value.model_dump_json(indent=1), encoding="utf-8")

    def _write_text(self, name: str, text: str) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._path(name).write_text(text, encoding="utf-8")

    # -- model stage runner -----------------------------------------------------

    async def _ask(
        self, stage: str, task: str, cls: type[T], check: Callable[[T], list[str]]
    ) -> T:
        prompt = task
        errors: list[str] = []
        for attempt in range(self.max_repairs + 1):
            name = stage if attempt == 0 else f"{stage}.repair{attempt}"
            answer = await self.model.complete(
                stage=name, system=prompts.SYSTEM, prompt=prompt
            )
            try:
                value = cls.model_validate(extract_json_object(answer))
            except (ValueError, ValidationError) as exc:
                errors = [f"invalid JSON for {cls.__name__}: {exc}"]
            else:
                errors = check(value)
                if not errors:
                    return value
            _LOGGER.warning(
                "stage %s attempt %d rejected: %s", stage, attempt + 1, errors
            )
            prompt = prompts.repair_prompt(task, errors)
        raise StageFailed(stage, errors)

    # -- stages -------------------------------------------------------------------

    async def frame_problem(self, raw_problem: str) -> ProblemFrame:
        cached = self._load("01_problem_frame.json", ProblemFrame)
        if cached is not None:
            return cached
        frame = await self._ask(
            "framing", prompts.framing_prompt(raw_problem), ProblemFrame, lambda _: []
        )
        self._save("01_problem_frame.json", frame)
        return frame

    async def retrieve(self, frame: ProblemFrame) -> Corpus:
        cached = self._load("02_corpus.json", Corpus)
        if cached is not None:
            return cached
        corpus = await self.retriever.retrieve(
            frame, self._path("sources"), self.run_dir
        )
        if not corpus.papers:
            raise EmptyCorpus(
                f"no source passed retrieval for {len(frame.search_queries)} queries"
            )
        self._save("02_corpus.json", corpus)
        return corpus

    async def review(self, frame: ProblemFrame, corpus: Corpus) -> LiteratureReview:
        cached = self._load("03_literature_review.json", LiteratureReview)
        if cached is None:
            cached = await self._ask(
                "literature_review",
                prompts.review_prompt(frame, corpus),
                LiteratureReview,
                lambda r: grounding.check_review(r, corpus),
            )
            self._save("03_literature_review.json", cached)
        self._write_text(
            "research_summary.md", render.render_review(frame, corpus, cached)
        )
        return cached

    async def directions(
        self, frame: ProblemFrame, review: LiteratureReview, corpus: Corpus
    ) -> DirectionMap:
        cached = self._load("04_directions.json", DirectionMap)
        if cached is None:
            cached = await self._ask(
                "directions",
                prompts.directions_prompt(frame, review, corpus),
                DirectionMap,
                lambda d: grounding.check_directions(d, review, corpus),
            )
            self._save("04_directions.json", cached)
        self._write_text("04_directions.md", render.render_directions(cached))
        return cached

    async def proposal(
        self,
        frame: ProblemFrame,
        review: LiteratureReview,
        directions: DirectionMap,
        corpus: Corpus,
    ) -> Proposal:
        cached = self._load("05_proposal.json", Proposal)
        if cached is None:
            cached = await self._ask(
                "proposal",
                prompts.proposal_prompt(frame, review, directions, corpus),
                Proposal,
                lambda p: grounding.check_proposal(p, directions, corpus),
            )
            self._save("05_proposal.json", cached)
        self._write_text("proposal.md", render.render_proposal(cached, corpus))
        return cached

    async def run(self, raw_problem: str) -> PipelineResult:
        frame = await self.frame_problem(raw_problem)
        corpus = await self.retrieve(frame)
        review = await self.review(frame, corpus)
        directions = await self.directions(frame, review, corpus)
        proposal = await self.proposal(frame, review, directions, corpus)
        manifest = {
            "raw_problem": raw_problem,
            "research_question": frame.research_question,
            "sources": len(corpus.papers),
            "rejected_search_results": corpus.rejected_results,
            "recommended_direction": directions.recommended_id,
            "artifacts": sorted(p.name for p in self.run_dir.iterdir() if p.is_file()),
        }
        self._write_text(
            "manifest.json", json.dumps(manifest, indent=1, ensure_ascii=False)
        )
        return PipelineResult(self.run_dir, frame, corpus, review, directions, proposal)


__all__ = ["EmptyCorpus", "PipelineResult", "ResearchProposalPipeline", "StageFailed"]
