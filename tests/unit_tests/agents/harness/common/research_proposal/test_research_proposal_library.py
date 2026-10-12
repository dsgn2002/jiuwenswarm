from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from jiuwenswarm.agents.harness.common.research_proposal import (
    LiteratureRetriever,
    ResearchProposalPipeline,
)
from jiuwenswarm.agents.harness.common.research_proposal.library import (
    KeywordIndex,
    LibraryBackend,
    OpenJiuwenMemoryIndex,
    ResearchLibrary,
)
from jiuwenswarm.agents.harness.common.research_proposal.schemas import PaperRecord

REACT = PaperRecord(
    source_id="arxiv:2210.03629",
    title="ReAct: Synergizing Reasoning and Acting in Language Models",
    url="https://arxiv.org/abs/2210.03629",
    authors=["Shunyu Yao"],
    year="2022",
    abstract="Interleaves reasoning traces with tool-calling actions.",
    backend="arxiv",
    relevance=1.0,
)
MM_AGENT = PaperRecord(
    source_id="arxiv:2303.04671",
    title="Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models",
    url="https://arxiv.org/abs/2303.04671",
    year="2023",
    abstract="A multimodal agent that calls visual foundation model tools.",
    backend="arxiv",
    relevance=1.0,
)
COT = PaperRecord(
    source_id="arxiv:2201.11903",
    title="Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
    url="https://arxiv.org/abs/2201.11903",
    year="2022",
    abstract="Chain-of-thought prompting improves reasoning.",
    backend="arxiv",
    relevance=1.0,
)


def session_answers(
    papers: list[PaperRecord], question: str, gap: str
) -> dict[str, dict]:
    ids = [p.source_id for p in papers]
    return {
        "framing": {
            "raw_problem": question,
            "research_question": question,
            "motivation": "m",
            "key_concepts": ["tool calling", "multimodal agents", "reasoning"],
            "search_queries": [f"{question} methods", f"{question} limitations"],
        },
        "literature_review": {
            "synthesis": f"Synthesis about {question}.",
            "notes": [
                {
                    "source_id": sid,
                    "summary": f"Note on {sid}.",
                    "key_findings": [f"finding {sid}"],
                    "relevance": 4,
                }
                for sid in ids
            ],
            "themes": [{"name": "T", "summary": "s", "source_ids": ids}],
            "gaps": [{"id": "G1", "description": gap, "evidence_source_ids": ids[:1]}],
        },
        "directions": {
            "directions": [
                {
                    "id": f"D{i}",
                    "title": f"Direction {i}",
                    "hypothesis": "h",
                    "rationale": "r",
                    "addresses_gap": "G1",
                    "builds_on": ids[:1],
                    "novelty": 3,
                    "feasibility": 3,
                    "impact": 3,
                    "first_experiment": "e",
                }
                for i in (1, 2)
            ],
            "recommended_id": "D1",
            "selection_rationale": "s",
        },
        "proposal": {
            "title": f"Proposal on {question}",
            "summary": f"Builds on [@{ids[0]}].",
            "background": f"See [@{ids[0]}].",
            "research_questions": ["q"],
            "hypotheses": ["h"],
            "method": "m",
            "evaluation": "e",
        },
    }


class ScriptedModel:
    def __init__(self, answers: dict[str, dict]) -> None:
        self.answers = answers
        self.prompts: dict[str, str] = {}

    async def complete(self, *, stage: str, system: str, prompt: str) -> str:
        self.prompts[stage] = prompt
        answer = self.answers[stage]
        if stage == "literature_review":
            # The library can add papers from earlier sessions: note every source shown.
            shown = re.findall(r"^\[(\S+)\] ", prompt, flags=re.MULTILINE)
            answer = dict(
                answer,
                notes=[
                    {
                        "source_id": sid,
                        "summary": f"Note on {sid}.",
                        "key_findings": [f"finding {sid}"],
                        "relevance": 4,
                    }
                    for sid in shown
                ],
            )
        return json.dumps(answer)


class StaticBackend:
    name = "static"

    def __init__(self, papers: list[PaperRecord]) -> None:
        self.papers = papers

    async def search(self, query: str, max_results: int) -> list[PaperRecord]:
        return [p.model_copy(update={"matched_queries": [query]}) for p in self.papers]


async def run_session(
    library: ResearchLibrary, run_dir: Path, papers, question, gap
) -> ScriptedModel:
    model = ScriptedModel(session_answers(papers, question, gap))
    retriever = LiteratureRetriever(
        backends=[LibraryBackend(library), StaticBackend(papers)],
        downloader=None,
        pause_seconds=0,
        min_relevance=0.0,
        min_topic_relevance=0.0,
    )
    await ResearchProposalPipeline(model, retriever, run_dir, library=library).run(
        question
    )
    return model


async def test_second_session_recalls_first_and_reuses_its_papers(tmp_path: Path):
    library = ResearchLibrary(tmp_path / "library")

    await run_session(
        library,
        tmp_path / "day1-agents",
        [REACT, MM_AGENT],
        "multimodal agents tool calling",
        "Multimodal agents rarely verify the outputs of the tools they call.",
    )
    assert set(library.state.papers) == {REACT.source_id, MM_AGENT.source_id}
    assert [s.session_id for s in library.state.sessions] == ["day1-agents"]
    assert (library.cards_dir / "paper-arxiv_2210_03629.md").is_file()
    assert (library.cards_dir / "gap-day1_agents-G1.md").is_file()

    # Day 2: a new topic. ReAct is found again (from the web) and CoT is new.
    day2 = await run_session(
        ResearchLibrary(tmp_path / "library"),  # reopened from disk
        tmp_path / "day2-reasoning",
        [REACT, COT],
        "reasoning in tool calling agents",
        "Reasoning traces are not checked against tool results.",
    )

    framing = day2.prompts["framing"]
    assert "Research memory" in framing
    assert "Earlier session day1-agents" in framing
    assert "Multimodal agents rarely verify the outputs" in framing
    review = day2.prompts["literature_review"]
    assert (
        "notes from earlier sessions: (sessions: day1-agents) Note on arxiv:2210.03629."
        in review
    )
    assert (tmp_path / "day2-reasoning" / "00_memory_recall.md").is_file()

    reopened = ResearchLibrary(tmp_path / "library")
    react = reopened.state.papers[REACT.source_id]
    assert react.sessions == ["day1-agents", "day2-reasoning"]
    assert react.topics == [
        "multimodal agents tool calling",
        "reasoning in tool calling agents",
    ]
    assert react.key_findings == [
        "finding arxiv:2210.03629"
    ]  # accumulated, not duplicated
    assert len(reopened.state.papers) == 3
    # The day-1 multimodal paper was pulled into day 2 from the library alone.
    assert reopened.state.papers[MM_AGENT.source_id].sessions == [
        "day1-agents",
        "day2-reasoning",
    ]
    hits = await reopened.search("reasoning traces tool results")
    assert any(h.kind == "gap" and h.ref == "day2-reasoning/G1" for h in hits)


async def test_rerunning_a_session_replaces_it_instead_of_duplicating(tmp_path: Path):
    library = ResearchLibrary(tmp_path / "library")
    for _ in range(2):
        await run_session(
            library, tmp_path / "day1", [REACT], "tool calling", "gap text"
        )
    assert [s.session_id for s in library.state.sessions] == ["day1"]
    assert library.state.papers[REACT.source_id].sessions == ["day1"]


async def test_empty_library_adds_nothing_to_the_framing_prompt(tmp_path: Path):
    library = ResearchLibrary(tmp_path / "library")
    model = await run_session(
        library, tmp_path / "day1", [REACT], "tool calling", "gap"
    )
    assert "Research memory" not in model.prompts["framing"]


class BrokenIndex:
    name = "broken"

    async def refresh(self, cards_dir: Path) -> None:
        raise RuntimeError("index unavailable")

    async def search(self, query: str, limit: int):  # pragma: no cover - never reached
        return []


async def test_falls_back_to_keyword_index_when_openjiuwen_index_fails(tmp_path: Path):
    library = ResearchLibrary(tmp_path / "library", index=BrokenIndex())
    await run_session(
        library, tmp_path / "day1", [REACT], "tool calling", "gap about verification"
    )

    hits = await library.search("verification")

    assert isinstance(library._index, KeywordIndex)
    assert library.index_name == "keyword"
    assert [h.ref for h in hits if h.kind == "gap"] == ["day1/G1"]


async def test_openjiuwen_lite_memory_indexes_library_cards_across_reopens(
    tmp_path: Path,
):
    pytest.importorskip("openjiuwen.core.memory.lite.manager")
    library = ResearchLibrary.with_openjiuwen_index(tmp_path / "library")
    await run_session(
        library,
        tmp_path / "day1",
        [REACT, MM_AGENT],
        "tool calling",
        "Unverified visual tool outputs.",
    )

    # Open the same on-disk index three times, as separate days would. openjiuwen drops
    # its FTS table on every reopen (legacy-tokenizer check bug); refresh() must repair it.
    for day in range(3):
        index = OpenJiuwenMemoryIndex(library.root, agent_id=f"test_{tmp_path.name}")
        try:
            await index.refresh(library.cards_dir)
        except RuntimeError as exc:
            pytest.skip(f"openjiuwen lite memory unavailable here: {exc}")
        try:
            rows = await index.search("Synergizing Reasoning Acting", 5)
            partial = await index.search("reasoning about visual foundation tools", 5)
        finally:
            await index.close()
        assert rows and rows[0][0] == "paper-arxiv_2210_03629.md", f"reopen {day}"
        # Per-word fusion finds cards that do not contain every query word.
        assert "paper-arxiv_2303_04671.md" in [name for name, _, _ in partial], (
            f"reopen {day}"
        )
