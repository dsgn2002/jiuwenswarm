"""Deterministic grounding checks between stages.

These checks are plain code, not model judgement: a stage answer that cites a
source the retriever never returned, or a gap the review never named, is
rejected and sent back to the model with the exact errors.
"""

from __future__ import annotations

from jiuwenswarm.agents.harness.common.research_proposal.schemas import (
    Corpus,
    DirectionMap,
    LiteratureReview,
    Proposal,
)


def _unknown(ids: list[str], known: set[str]) -> list[str]:
    return [i for i in dict.fromkeys(ids) if i not in known]


def check_review(review: LiteratureReview, corpus: Corpus) -> list[str]:
    known = corpus.ids()
    errors: list[str] = []
    note_ids = [n.source_id for n in review.notes]
    for sid in _unknown(note_ids, known):
        errors.append(f"note cites unknown source_id {sid!r}")
    duplicates = {i for i in note_ids if note_ids.count(i) > 1}
    if duplicates:
        errors.append(f"more than one note for {sorted(duplicates)}")
    for theme in review.themes:
        for sid in _unknown(theme.source_ids, known):
            errors.append(f"theme {theme.name!r} cites unknown source_id {sid!r}")
    for gap in review.gaps:
        for sid in _unknown(gap.evidence_source_ids, known):
            errors.append(f"gap {gap.id!r} cites unknown source_id {sid!r}")
    missing = sorted(known - set(note_ids))
    if missing:
        errors.append(f"every retrieved source needs a note; missing {missing}")
    return errors


def check_directions(
    directions: DirectionMap, review: LiteratureReview, corpus: Corpus
) -> list[str]:
    known = corpus.ids()
    gap_ids = {g.id for g in review.gaps}
    errors: list[str] = []
    for direction in directions.directions:
        for sid in _unknown(direction.builds_on, known):
            errors.append(
                f"direction {direction.id!r} builds on unknown source_id {sid!r}"
            )
        if direction.addresses_gap not in gap_ids:
            errors.append(
                f"direction {direction.id!r} addresses unknown gap {direction.addresses_gap!r}; "
                f"known gaps are {sorted(gap_ids)}"
            )
    return errors


def check_proposal(
    proposal: Proposal, directions: DirectionMap, corpus: Corpus
) -> list[str]:
    known = corpus.ids()
    cited = proposal.cited_ids()
    errors = [
        f"proposal cites unknown source_id {sid!r}" for sid in _unknown(cited, known)
    ]
    if not cited:
        errors.append("proposal must cite its evidence inline as [@source_id]")
    chosen = directions.recommended()
    if not set(chosen.builds_on) & set(cited):
        errors.append(
            f"proposal must cite at least one source the chosen direction {chosen.id!r} builds on: "
            f"{chosen.builds_on}"
        )
    return errors


__all__ = ["check_directions", "check_proposal", "check_review"]
