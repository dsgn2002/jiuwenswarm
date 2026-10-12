"""Markdown renderers for the pipeline's human-facing artifacts.

``render_review`` produces ``research_summary.md``: openjiuwen's
experiment-design stage looks for that file name in ``research_paths`` and
reads it first, so the proposal pipeline's output can seed the existing
design → code → execution → reporting loop.
"""

from __future__ import annotations

import re

from jiuwenswarm.agents.harness.common.research_proposal.schemas import (
    Corpus,
    DirectionMap,
    LiteratureReview,
    PaperRecord,
    ProblemFrame,
    Proposal,
    citations_in,
)


def _reference(paper: PaperRecord) -> str:
    authors = (
        ", ".join(paper.authors[:3]) + (" et al." if len(paper.authors) > 3 else "")
        or "Unknown authors"
    )
    doi = f" https://doi.org/{paper.doi}" if paper.doi else ""
    return f"{authors} ({paper.year or 'n.d.'}). *{paper.title}*. {paper.venue or ''}.{doi} <{paper.url}>"


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- (none)"


def render_review(frame: ProblemFrame, corpus: Corpus, review: LiteratureReview) -> str:
    notes = {n.source_id: n for n in review.notes}
    parts = [
        f"# Literature review: {frame.research_question}",
        "## Synthesis",
        review.synthesis,
        "## Themes",
    ]
    for theme in review.themes:
        parts.append(
            f"### {theme.name}\n\n{theme.summary}\n\nSources: {', '.join(theme.source_ids)}"
        )
    parts.append("## Gaps")
    parts += [
        f"- **{g.id}.** {g.description} (evidence: {', '.join(g.evidence_source_ids)})"
        for g in review.gaps
    ]
    parts.append("## Sources")
    for paper in corpus.papers:
        note = notes.get(paper.source_id)
        body = [f"### [{paper.source_id}] {paper.title}", _reference(paper)]
        if paper.local_path:
            body.append(f"Local copy: `{paper.local_path}` ({paper.retrieval_mode})")
        if note:
            body += [
                f"**Summary.** {note.summary}",
                f"**Key findings.**\n{_bullets(note.key_findings)}",
                f"**Limitations.**\n{_bullets(note.limitations)}",
                f"**Relevance:** {note.relevance}/5",
            ]
        parts.append("\n\n".join(body))
    parts.append(
        f"---\nQueries run: {len(corpus.queries_run)}. Sources kept: {len(corpus.papers)}. "
        f"Search results rejected as off-topic or below relevance threshold: {corpus.rejected_results}."
    )
    return "\n\n".join(parts) + "\n"


def render_directions(directions: DirectionMap) -> str:
    rows = [
        "| ID | Direction | Gap | Novelty | Feasibility | Impact | Score |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for d in sorted(directions.directions, key=lambda d: -d.score):
        mark = " (recommended)" if d.id == directions.recommended_id else ""
        rows.append(
            f"| {d.id} | {d.title}{mark} | {d.addresses_gap} | {d.novelty} | {d.feasibility} | {d.impact} | {d.score} |"
        )
    parts = [
        "# Potential research directions",
        "\n".join(rows),
        f"**Selection.** {directions.selection_rationale}",
    ]
    for d in directions.directions:
        parts.append(
            f"## {d.id}. {d.title}\n\n**Hypothesis.** {d.hypothesis}\n\n**Rationale.** {d.rationale}\n\n"
            f"**Builds on:** {', '.join(d.builds_on)}\n\n**First experiment.** {d.first_experiment}\n\n"
            f"**Risks.**\n{_bullets(d.risks)}"
        )
    return "\n\n".join(parts) + "\n"


def render_proposal(proposal: Proposal, corpus: Corpus) -> str:
    order = proposal.cited_ids()
    number = {sid: i + 1 for i, sid in enumerate(order)}

    def cite(text: str) -> str:
        return re.sub(
            r"\[@([A-Za-z0-9:._/-]+)\]",
            lambda m: f"[{number.get(m.group(1), '?')}]",
            text,
        )

    parts = [
        f"# {proposal.title}",
        "## Summary",
        cite(proposal.summary),
        "## Background",
        cite(proposal.background),
        "## Research questions",
        "\n".join(
            f"{i}. {cite(q)}" for i, q in enumerate(proposal.research_questions, 1)
        ),
        "## Hypotheses",
        "\n".join(f"{i}. {cite(h)}" for i, h in enumerate(proposal.hypotheses, 1)),
        "## Method",
        cite(proposal.method),
        "## Evaluation",
        cite(proposal.evaluation),
        "## Risks and mitigations",
        _bullets([cite(r) for r in proposal.risks]),
        "## Milestones",
        "\n".join(f"{i}. {cite(m)}" for i, m in enumerate(proposal.milestones, 1))
        or "- (none)",
        "## References",
        "\n".join(
            f"{number[sid]}. {_reference(corpus.get(sid))}"
            for sid in order
            if corpus.get(sid)
        ),
    ]
    return "\n\n".join(parts) + "\n"


__all__ = ["citations_in", "render_directions", "render_proposal", "render_review"]
