"""Prompt templates for the four reasoning stages.

Each prompt asks for one JSON object and spells out its shape; grounding rules
match the checks in ``grounding.py`` so a well-behaved model passes first time.
"""

from __future__ import annotations

import json

from jiuwenswarm.agents.harness.common.research_proposal.schemas import (
    Corpus,
    DirectionMap,
    LiteratureReview,
    ProblemFrame,
)

SYSTEM = (
    "You are a careful research scientist working inside an automated research pipeline. "
    "Answer with exactly one JSON object and nothing else. Never invent papers, results, "
    "numbers or source ids: use only the evidence you are given."
)

FRAMING = """\
Turn the raw research problem below into a framed problem a literature search can use.

Raw problem:
{raw_problem}
{memory}
Return JSON with these keys:
- "raw_problem": the raw problem, copied verbatim
- "research_question": one precise, answerable question
- "motivation": 2-3 sentences on why it matters
- "key_concepts": list of the technical concepts involved
- "in_scope", "out_of_scope": lists that bound the study
- "search_queries": 4-8 distinct, specific queries for scholarly search, covering background,
  core methods, competing approaches, known limitations and recent work. Use technical
  vocabulary and paper-title style phrases; no site: operators.
- "inclusion_criteria": what makes a paper relevant
"""

REVIEW = """\
Write a grounded literature review for the framed problem, using ONLY the retrieved sources.

Research question: {research_question}
Key concepts: {key_concepts}

Retrieved sources (each has a source_id; abstracts and excerpts are the only evidence):
{evidence}

Return JSON with these keys:
- "synthesis": 1-2 paragraphs on the state of the field and how the sources relate
- "notes": one entry per source above, each {{"source_id", "summary", "key_findings": [...],
  "limitations": [...], "relevance": 1-5}}. Every source above must have exactly one note.
- "themes": 2-5 entries {{"name", "summary", "source_ids": [...]}} grouping the sources
- "gaps": 2-5 entries {{"id": "G1"..., "description", "evidence_source_ids": [...]}}: open problems
  or contradictions that the evidence itself shows, each tied to the sources that show it

Rules: use only source_id values listed above; state findings only if the abstract or excerpt
supports them; say "not stated" rather than guessing.
"""

DIRECTIONS = """\
Propose potential research directions that follow from the review.

Research question: {research_question}

Literature gaps:
{gaps}

Themes:
{themes}

Valid source_ids: {source_ids}

Return JSON with these keys:
- "directions": 3-5 entries, each {{"id": "D1"..., "title", "hypothesis" (testable),
  "rationale", "addresses_gap" (one gap id above), "builds_on" (source_ids it extends),
  "novelty", "feasibility", "impact" (integers 1-5, be critical), "risks": [...],
  "first_experiment" (the smallest experiment that would test the hypothesis)}}
- "recommended_id": the direction to develop into a proposal
- "selection_rationale": why it beats the others on the novelty/feasibility/impact trade-off

Rules: every direction must address a listed gap and build on listed source_ids only.
"""

PROPOSAL = """\
Write a research proposal for the chosen direction.

Research question: {research_question}

Chosen direction:
{direction}

Gap it addresses:
{gap}

Sources you may cite (cite inline as [@source_id], e.g. [@arxiv:2002.05202]):
{sources}

Return JSON with these keys:
- "title"
- "summary": one paragraph a reviewer can judge the proposal from
- "background": 2-3 paragraphs tracing the work this builds on, citing sources inline
- "research_questions": list
- "hypotheses": list of testable hypotheses
- "method": the approach, concrete enough to start work
- "evaluation": datasets/benchmarks, baselines, metrics and what result would refute the hypothesis
- "risks": list of risks with a mitigation each
- "milestones": ordered list of work packages

Rules: cite only the source_ids listed above; cite at least one source the chosen direction
builds on; do not invent results.
"""

REPAIR = """\
Your previous answer was rejected by the pipeline's validator:

{errors}

Return a corrected JSON object for the same task. The original task follows.

{task}
"""


def evidence_block(
    corpus: Corpus, *, abstract_chars: int = 1200, excerpt_chars: int = 2500
) -> str:
    blocks = []
    for paper in corpus.papers:
        authors = ", ".join(paper.authors[:6]) or "unknown"
        lines = [
            f"[{paper.source_id}] {paper.title}",
            f"  authors: {authors}; year: {paper.year or 'unknown'}; venue: {paper.venue or 'unknown'}",
            f"  abstract: {paper.abstract[:abstract_chars] or 'not available'}",
        ]
        if paper.prior_notes:
            lines.append(
                f"  notes from earlier sessions: {paper.prior_notes[:excerpt_chars]}"
            )
        if paper.evidence_excerpt:
            lines.append(
                f"  full-text excerpt: {paper.evidence_excerpt[:excerpt_chars]}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


MEMORY = """
Research memory (earlier sessions, their open gaps and papers already read). Build on it: link the
new question to related earlier gaps where they genuinely connect, and include search queries that
reach beyond papers already read. Do not assume earlier findings apply unless they clearly do.
{recall}
"""


def framing_prompt(raw_problem: str, recall: str = "") -> str:
    memory = MEMORY.format(recall=recall.strip()) if recall.strip() else ""
    return FRAMING.format(raw_problem=raw_problem.strip(), memory=memory)


def review_prompt(frame: ProblemFrame, corpus: Corpus) -> str:
    return REVIEW.format(
        research_question=frame.research_question,
        key_concepts=", ".join(frame.key_concepts),
        evidence=evidence_block(corpus),
    )


def directions_prompt(
    frame: ProblemFrame, review: LiteratureReview, corpus: Corpus
) -> str:
    return DIRECTIONS.format(
        research_question=frame.research_question,
        gaps=json.dumps(
            [g.model_dump() for g in review.gaps], indent=1, ensure_ascii=False
        ),
        themes=json.dumps(
            [t.model_dump() for t in review.themes], indent=1, ensure_ascii=False
        ),
        source_ids=", ".join(sorted(corpus.ids())),
    )


def proposal_prompt(
    frame: ProblemFrame,
    review: LiteratureReview,
    directions: DirectionMap,
    corpus: Corpus,
) -> str:
    chosen = directions.recommended()
    gap = next((g for g in review.gaps if g.id == chosen.addresses_gap), None)
    notes = {n.source_id: n for n in review.notes}
    sources = "\n".join(
        f"[@{p.source_id}] {p.title} ({p.year or 'n.d.'}): {notes[p.source_id].summary if p.source_id in notes else p.abstract[:300]}"
        for p in corpus.papers
    )
    return PROPOSAL.format(
        research_question=frame.research_question,
        direction=json.dumps(chosen.model_dump(), indent=1, ensure_ascii=False),
        gap=json.dumps(gap.model_dump(), indent=1, ensure_ascii=False)
        if gap
        else "not found",
        sources=sources,
    )


def repair_prompt(task: str, errors: list[str]) -> str:
    return REPAIR.format(errors="\n".join(f"- {e}" for e in errors), task=task)


__all__ = [
    "SYSTEM",
    "directions_prompt",
    "evidence_block",
    "framing_prompt",
    "proposal_prompt",
    "repair_prompt",
    "review_prompt",
]
