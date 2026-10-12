"""Data contracts for the research-proposal pipeline.

Every stage hands the next one a validated pydantic model, and every model
that makes a claim about the literature names the ``source_id`` values it rests
on. ``grounding.py`` checks those ids against the retrieved corpus, so a stage
cannot cite a paper the pipeline never found.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

_CITATION_RE = re.compile(r"\[@([A-Za-z0-9:._/-]+)\]")


def citations_in(text: str) -> list[str]:
    """Return the ``[@source_id]`` citations that appear in ``text``, in order."""
    return _CITATION_RE.findall(text or "")


class _Strict(BaseModel):
    model_config = {"extra": "ignore"}


# -- Stage 1: problem framing -------------------------------------------------


class ProblemFrame(_Strict):
    """A raw research idea turned into something a literature search can use."""

    raw_problem: str = Field(min_length=1)
    research_question: str = Field(min_length=1)
    motivation: str = Field(min_length=1)
    key_concepts: list[str] = Field(min_length=1)
    in_scope: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(min_length=2, max_length=12)
    inclusion_criteria: list[str] = Field(default_factory=list)

    @field_validator("search_queries")
    @classmethod
    def _distinct_queries(cls, value: list[str]) -> list[str]:
        cleaned = [q.strip() for q in value if q.strip()]
        deduped = list(dict.fromkeys(cleaned))
        if len(deduped) < 2:
            raise ValueError("at least two distinct search queries are required")
        return deduped


# -- Stage 2: retrieval and literature review ----------------------------------


class PaperRecord(_Strict):
    """One retrieved source. Filled by search backends, never by the model."""

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    year: str | None = None
    venue: str | None = None
    doi: str | None = None
    pdf_url: str | None = None
    abstract: str = ""
    backend: str = ""
    matched_queries: list[str] = Field(default_factory=list)
    relevance: float = 0.0
    local_path: str | None = None
    retrieval_mode: Literal["downloaded_pdf", "downloaded_html", "metadata_only"] = (
        "metadata_only"
    )
    evidence_excerpt: str = ""
    # Notes written about this paper in earlier sessions (from the research library).
    prior_notes: str = ""


class Corpus(_Strict):
    problem: str
    papers: list[PaperRecord] = Field(default_factory=list)
    queries_run: list[str] = Field(default_factory=list)
    rejected_results: int = 0
    backend_errors: list[str] = Field(default_factory=list)

    def ids(self) -> set[str]:
        return {paper.source_id for paper in self.papers}

    def get(self, source_id: str) -> PaperRecord | None:
        return next((p for p in self.papers if p.source_id == source_id), None)


class PaperNote(_Strict):
    source_id: str
    summary: str = Field(min_length=1)
    key_findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    relevance: int = Field(ge=1, le=5)


class Theme(_Strict):
    name: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)


class Gap(_Strict):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    evidence_source_ids: list[str] = Field(min_length=1)


class LiteratureReview(_Strict):
    synthesis: str = Field(min_length=1)
    notes: list[PaperNote] = Field(min_length=1)
    themes: list[Theme] = Field(min_length=1)
    gaps: list[Gap] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_gap_ids(self) -> LiteratureReview:
        ids = [gap.id for gap in self.gaps]
        if len(ids) != len(set(ids)):
            raise ValueError("gap ids must be unique")
        return self


# -- Stage 3: potential directions --------------------------------------------


class Direction(_Strict):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    addresses_gap: str = Field(min_length=1)
    builds_on: list[str] = Field(min_length=1)
    novelty: int = Field(ge=1, le=5)
    feasibility: int = Field(ge=1, le=5)
    impact: int = Field(ge=1, le=5)
    risks: list[str] = Field(default_factory=list)
    first_experiment: str = Field(min_length=1)

    @property
    def score(self) -> float:
        """Equal-weight mean of the three 1-5 ratings."""
        return round((self.novelty + self.feasibility + self.impact) / 3, 2)


class DirectionMap(_Strict):
    directions: list[Direction] = Field(min_length=2)
    recommended_id: str
    selection_rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _recommended_exists(self) -> DirectionMap:
        ids = [d.id for d in self.directions]
        if len(ids) != len(set(ids)):
            raise ValueError("direction ids must be unique")
        if self.recommended_id not in ids:
            raise ValueError(
                f"recommended_id {self.recommended_id!r} is not one of {ids}"
            )
        return self

    def recommended(self) -> Direction:
        return next(d for d in self.directions if d.id == self.recommended_id)


# -- Stage 4: research proposal ------------------------------------------------


class Proposal(_Strict):
    """Prose fields cite sources inline as ``[@source_id]``."""

    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    background: str = Field(min_length=1)
    research_questions: list[str] = Field(min_length=1)
    hypotheses: list[str] = Field(min_length=1)
    method: str = Field(min_length=1)
    evaluation: str = Field(min_length=1)
    risks: list[str] = Field(default_factory=list)
    milestones: list[str] = Field(default_factory=list)

    def cited_ids(self) -> list[str]:
        parts = [self.summary, self.background, self.method, self.evaluation]
        parts += (
            self.research_questions + self.hypotheses + self.risks + self.milestones
        )
        return list(dict.fromkeys(cid for part in parts for cid in citations_in(part)))


__all__ = [
    "Corpus",
    "Direction",
    "DirectionMap",
    "Gap",
    "LiteratureReview",
    "PaperNote",
    "PaperRecord",
    "ProblemFrame",
    "Proposal",
    "Theme",
    "citations_in",
]
