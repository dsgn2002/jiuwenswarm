"""Persistent research memory shared across pipeline runs.

Each pipeline run is one *session* on one topic (say, multimodal agents and
tool calling today, reasoning tomorrow). The library keeps what every session
learned and feeds it into the next one:

* **Recall before framing** - ``recall()`` finds earlier sessions, gaps and
  papers related to the new problem and the framing prompt receives them, so
  tomorrow's question is framed knowing today's findings.
* **Search the library first** - ``LibraryBackend`` is a ``SearchBackend``:
  papers already read in earlier sessions come back with their notes, so they
  are reused (and cross-linked between topics) instead of re-discovered.
* **Ingest after the run** - ``ingest()`` merges the new corpus, per-paper
  notes, gaps and recommended direction into the library.

Storage is compressed, not raw: a paper is kept as a short card (metadata,
one-paragraph summary, findings, limitations, topics it served) and a session
as its question, synthesis, gaps and chosen direction - never full text.
``library.json`` is the source of truth; ``memory/*.md`` holds one markdown
card per paper, session and gap for search.

Search uses openjiuwen's lite memory (``MemoryIndexManager``: SQLite FTS5,
plus vectors when an embedding config is given) when it is installed and
working, and a built-in keyword index otherwise. The lite memory needs a
``filelock`` 3.x: with filelock 4.x its file reads fail and nothing is
indexed, which ``OpenJiuwenMemoryIndex`` detects and reports.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field

from jiuwenswarm.agents.harness.common.research_proposal.literature import (
    content_terms,
    relevance_score,
)
from jiuwenswarm.agents.harness.common.research_proposal.schemas import (
    Gap,
    PaperRecord,
)

if TYPE_CHECKING:
    from jiuwenswarm.agents.harness.common.research_proposal.pipeline import (
        PipelineResult,
    )

_LOGGER = logging.getLogger(__name__)


class LibraryPaper(PaperRecord):
    """A paper card: retrieval metadata plus the notes earlier sessions wrote."""

    summary: str = ""
    key_findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    sessions: list[str] = Field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""


class SessionRecord(BaseModel):
    session_id: str
    date: str
    raw_problem: str
    research_question: str
    key_concepts: list[str] = Field(default_factory=list)
    synthesis: str = ""
    gaps: list[Gap] = Field(default_factory=list)
    recommended_direction: str = ""
    recommended_hypothesis: str = ""
    proposal_title: str = ""
    paper_ids: list[str] = Field(default_factory=list)
    run_dir: str = ""


class LibraryState(BaseModel):
    papers: dict[str, LibraryPaper] = Field(default_factory=dict)
    sessions: list[SessionRecord] = Field(default_factory=list)


class LibraryHit(BaseModel):
    kind: str  # "paper", "session" or "gap"
    ref: str  # source_id, session_id or "<session_id>/<gap id>"
    score: float
    snippet: str = ""


# -- search indexes ----------------------------------------------------------------


class LibraryIndex(Protocol):
    name: str

    async def refresh(self, cards_dir: Path) -> None: ...

    async def search(self, query: str, limit: int) -> list[tuple[str, float, str]]:
        """Return ``(card file name, score, snippet)`` rows, best first."""


class KeywordIndex:
    """Dependency-free fallback: share of query words found in each card."""

    name = "keyword"

    def __init__(self) -> None:
        self._cards: dict[str, str] = {}

    async def refresh(self, cards_dir: Path) -> None:
        self._cards = {
            p.name: p.read_text(encoding="utf-8")
            for p in sorted(cards_dir.glob("*.md"))
        }

    async def search(self, query: str, limit: int) -> list[tuple[str, float, str]]:
        scored = [
            (name, relevance_score(query, text), text)
            for name, text in self._cards.items()
        ]
        ranked = sorted(
            (row for row in scored if row[1] > 0), key=lambda r: (-r[1], r[0])
        )
        return [
            (name, score, _first_lines(text)) for name, score, text in ranked[:limit]
        ]


class OpenJiuwenMemoryIndex:
    """openjiuwen lite memory (``MemoryIndexManager``) over the card directory.

    The cards directory is the workspace's ``memory`` node, so the manager
    indexes exactly the library's cards. ``embedding_config`` (an openjiuwen
    embedding config with an ``api_key``) turns on hybrid vector + keyword
    search; without it the manager uses FTS5 keyword search.
    """

    name = "openjiuwen_lite_memory"

    def __init__(
        self,
        root: Path,
        *,
        embedding_config: Any = None,
        agent_id: str = "research_library",
    ) -> None:
        self._root = root
        self._embedding_config = embedding_config
        self._agent_id = agent_id
        self._manager: Any = None

    async def _get_manager(self) -> Any:
        if self._manager is not None:
            return self._manager
        from openjiuwen.core.memory.lite.config import MemorySettings
        from openjiuwen.core.memory.lite.manager import (
            MemoryIndexManager,
            MemoryManagerParams,
        )
        from openjiuwen.core.runner import Runner
        from openjiuwen.core.sys_operation import (
            LocalWorkConfig,
            OperationMode,
            SysOperationCard,
        )
        from openjiuwen.harness.workspace.workspace import Workspace

        # The manager reads files through a SysOperation; without one it fails
        # with an UnboundLocalError instead of a clear message.
        sysop_id = f"{self._agent_id}_sysop"
        sysop = Runner.resource_mgr.get_sys_operation(sysop_id)
        if sysop is None:
            Runner.resource_mgr.add_sys_operation(
                SysOperationCard(
                    id=sysop_id,
                    mode=OperationMode.LOCAL,
                    work_config=LocalWorkConfig(shell_allowlist=None),
                )
            )
            sysop = Runner.resource_mgr.get_sys_operation(sysop_id)
        workspace = Workspace(root_path=str(self._root))
        manager = await MemoryIndexManager.get(
            MemoryManagerParams(
                agent_id=self._agent_id,
                workspace=workspace,
                settings=self._settings(MemorySettings),
                embedding_config=self._embedding_config,
                sys_operation=sysop,
            )
        )
        if manager is None:
            raise RuntimeError("openjiuwen MemoryIndexManager failed to initialise")
        self._manager = manager
        return manager

    @staticmethod
    def _settings(settings_cls: Any) -> Any:
        settings = settings_cls()
        settings.sources = ["memory"]  # library cards only, no chat-session transcripts
        settings.sync = {**settings.sync, "watch": False, "onSessionStart": False}
        # openjiuwen's bm25_rank_to_score maps SQLite FTS5 ranks (negative, more negative
        # = better) to 1/(1 - rank), so the best keyword matches get the LOWEST scores and
        # the default keywordMinScore=0.1 drops them. Keep every hit and rank by order.
        settings.query = {**settings.query, "min_score": 0.0, "keywordMinScore": 0.0}
        return settings

    async def refresh(self, cards_dir: Path) -> None:
        manager = await self._get_manager()
        # No file watcher without watchdog: tell the manager the cards changed.
        manager.dirty = True
        await manager.sync(reason="research_library")
        if (
            self._fts_rows(manager) == 0
            and int(manager.status().get("chunks") or 0) > 0
        ):
            # openjiuwen bug: _fts_table_is_legacy() looks for "tokenize=trigram" but the
            # table is created with tokenize='trigram', so every reopen drops chunks_fts as
            # "legacy" and, because meta already says ftsTrigram, never refills it. Keyword
            # search then finds nothing. Force a full reindex to repopulate it.
            await manager.sync(reason="research_library_fts_repair", force=True)
        indexed = int(manager.status().get("files") or 0)
        expected = len(list(cards_dir.glob("*.md")))
        if expected and indexed == 0:
            raise RuntimeError(
                f"openjiuwen lite memory indexed 0 of {expected} cards; with filelock>=4 its "
                "file reads fail ('Cannot release a lock ... not held by this task'), install filelock<4"
            )

    @staticmethod
    def _fts_rows(manager: Any) -> int:
        try:
            return int(
                manager.db.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
            )
        except Exception:  # noqa: BLE001 - schema differs: assume populated, let search decide
            return -1

    async def search(self, query: str, limit: int) -> list[tuple[str, float, str]]:
        manager = await self._get_manager()
        # openjiuwen's FTS query effectively requires every query word, so a multi-word
        # query rarely matches a card from another topic. Query the whole phrase and each
        # content word, and fuse the rankings (reciprocal rank fusion, k=10). Results
        # arrive best-first (ORDER BY rank); the keyword *scores* are inverted (see
        # _settings), so only the order is used. Several chunks of one card can match:
        # each card counts once per query, at its best rank.
        fused: dict[str, float] = {}
        snippets: dict[str, str] = {}
        for sub_query in [query, *content_terms(query)[:8]]:
            rows = await manager.search(
                sub_query, {"max_results": limit * 3, "min_score": 0.0}
            )
            seen: list[str] = []
            for row in rows:
                name = Path(row["path"]).name
                if name not in seen:
                    seen.append(name)
                    snippets.setdefault(name, row.get("snippet") or "")
            weight = (
                2.0 if sub_query == query else 1.0
            )  # a whole-phrase match counts double
            for rank, name in enumerate(seen):
                fused[name] = fused.get(name, 0.0) + weight / (10 + rank)
        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return [(name, round(score, 3), snippets[name]) for name, score in ranked]

    async def close(self) -> None:
        if self._manager is not None:
            await self._manager.close()
            self._manager = None


# -- the library ---------------------------------------------------------------------


def _first_lines(text: str, limit: int = 400) -> str:
    return " ".join(text.split())[:limit]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class ResearchLibrary:
    """Research memory rooted at ``root`` (``library.json`` + ``memory/`` cards)."""

    def __init__(self, root: str | Path, *, index: LibraryIndex | None = None) -> None:
        self.root = Path(root).resolve()
        self.cards_dir = self.root / "memory"
        self.state_path = self.root / "library.json"
        self.state = (
            LibraryState.model_validate_json(
                self.state_path.read_text(encoding="utf-8")
            )
            if self.state_path.exists()
            else LibraryState()
        )
        self._index = index
        self._index_ready = False

    @classmethod
    def with_openjiuwen_index(
        cls, root: str | Path, *, embedding_config: Any = None
    ) -> ResearchLibrary:
        resolved = Path(root).resolve()
        return cls(
            resolved,
            index=OpenJiuwenMemoryIndex(resolved, embedding_config=embedding_config),
        )

    # -- persistence -------------------------------------------------------------

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".library.", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(self.state.model_dump_json(indent=1))
        os.replace(tmp, self.state_path)
        self._write_cards()
        self._index_ready = False

    def _write_cards(self) -> None:
        self.cards_dir.mkdir(parents=True, exist_ok=True)
        wanted: dict[str, str] = {}
        for paper in self.state.papers.values():
            wanted[f"paper-{_slug(paper.source_id)}.md"] = render_paper_card(paper)
        for session in self.state.sessions:
            wanted[f"session-{_slug(session.session_id)}.md"] = render_session_card(
                session
            )
            for gap in session.gaps:
                wanted[f"gap-{_slug(session.session_id)}-{_slug(gap.id)}.md"] = (
                    render_gap_card(session, gap)
                )
        for name, text in wanted.items():
            path = self.cards_dir / name
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                path.write_text(text, encoding="utf-8")
        for stale in self.cards_dir.glob("*.md"):
            if stale.name not in wanted:
                stale.unlink()

    # -- ingest --------------------------------------------------------------------

    def ingest(
        self, result: PipelineResult, *, session_id: str | None = None
    ) -> SessionRecord:
        """Merge one finished pipeline run into the library and save it."""
        session_id = session_id or result.run_dir.name
        now = _now()
        notes = {note.source_id: note for note in result.review.notes}
        topic = result.frame.research_question
        for paper in result.corpus.papers:
            note = notes.get(paper.source_id)
            existing = self.state.papers.get(paper.source_id)
            data = paper.model_dump(
                exclude={
                    "evidence_excerpt",
                    "matched_queries",
                    "relevance",
                    "prior_notes",
                }
            )
            card = (
                LibraryPaper(**data)
                if existing is None
                else existing.model_copy(update={**data})
            )
            if note is not None:
                # Newer notes replace older ones; findings and limitations accumulate.
                card.summary = note.summary
                card.key_findings = list(
                    dict.fromkeys(card.key_findings + note.key_findings)
                )
                card.limitations = list(
                    dict.fromkeys(card.limitations + note.limitations)
                )
            card.topics = list(dict.fromkeys(card.topics + [topic]))
            card.sessions = list(dict.fromkeys(card.sessions + [session_id]))
            card.first_seen = card.first_seen or now
            card.last_seen = now
            card.backend = existing.backend if existing else paper.backend
            self.state.papers[paper.source_id] = card
        chosen = result.directions.recommended()
        record = SessionRecord(
            session_id=session_id,
            date=now,
            raw_problem=result.frame.raw_problem,
            research_question=topic,
            key_concepts=result.frame.key_concepts,
            synthesis=result.review.synthesis,
            gaps=result.review.gaps,
            recommended_direction=chosen.title,
            recommended_hypothesis=chosen.hypothesis,
            proposal_title=result.proposal.title,
            paper_ids=[p.source_id for p in result.corpus.papers],
            run_dir=str(result.run_dir),
        )
        self.state.sessions = [
            s for s in self.state.sessions if s.session_id != session_id
        ] + [record]
        self.save()
        return record

    # -- search and recall ----------------------------------------------------------

    async def _ready_index(self) -> LibraryIndex:
        if self._index is None:
            self._index = KeywordIndex()
        if not self._index_ready:
            self.cards_dir.mkdir(parents=True, exist_ok=True)
            try:
                await self._index.refresh(self.cards_dir)
            except Exception as exc:  # noqa: BLE001 - fall back rather than lose the memory
                _LOGGER.warning(
                    "%s unavailable (%s); using keyword index", self._index.name, exc
                )
                self._index = KeywordIndex()
                await self._index.refresh(self.cards_dir)
            self._index_ready = True
        return self._index

    @property
    def index_name(self) -> str:
        return self._index.name if self._index is not None else "keyword"

    async def search(self, query: str, limit: int = 10) -> list[LibraryHit]:
        index = await self._ready_index()
        hits = []
        for name, score, snippet in await index.search(query, limit):
            kind, _, rest = name[:-3].partition("-")
            ref = self._ref_for_card(kind, rest)
            if ref is not None and all(h.ref != ref or h.kind != kind for h in hits):
                hits.append(
                    LibraryHit(
                        kind=kind, ref=ref, score=round(score, 3), snippet=snippet
                    )
                )
        return hits

    def _ref_for_card(self, kind: str, slug: str) -> str | None:
        if kind == "paper":
            return next((sid for sid in self.state.papers if _slug(sid) == slug), None)
        for session in self.state.sessions:
            if kind == "session" and _slug(session.session_id) == slug:
                return session.session_id
            if kind == "gap":
                for gap in session.gaps:
                    if f"{_slug(session.session_id)}-{_slug(gap.id)}" == slug:
                        return f"{session.session_id}/{gap.id}"
        return None

    async def recall(self, query: str, *, limit: int = 8, max_sessions: int = 3) -> str:
        """Markdown block of the earlier sessions, gaps and papers related to ``query``.

        Sessions come from direct session/gap hits first, then from the sessions
        that recalled papers were read in, so a paper hit also brings back the
        context (question, gaps, chosen direction) it was read for.
        """
        if not self.state.sessions and not self.state.papers:
            return ""
        hits = await self.search(query, limit=limit * 2)
        sessions = {s.session_id: s for s in self.state.sessions}
        papers = [
            self.state.papers[h.ref]
            for h in hits
            if h.kind == "paper" and h.ref in self.state.papers
        ][:limit]
        ordered: list[str] = [
            h.ref.split("/", 1)[0] for h in hits if h.kind in ("session", "gap")
        ]
        ordered += [sid for paper in papers for sid in paper.sessions]
        chosen = [sid for sid in dict.fromkeys(ordered) if sid in sessions][
            :max_sessions
        ]
        lines: list[str] = []
        for session_id in chosen:
            s = sessions[session_id]
            lines.append(
                f"- Earlier session {s.session_id} ({s.date[:10]}): {s.research_question} "
                f"Recommended direction: {s.recommended_direction}."
            )
            lines += [f"  - Gap {g.id}: {g.description}" for g in s.gaps]
        for paper in papers:
            lines.append(
                f"- Paper [{paper.source_id}] {paper.title} ({paper.year or 'n.d.'}), "
                f"read in {', '.join(paper.sessions)}: {paper.summary or paper.abstract[:200]}"
            )
        return "\n".join(lines)

    def as_paper_record(self, source_id: str, query: str) -> PaperRecord | None:
        card = self.state.papers.get(source_id)
        if card is None:
            return None
        notes = " ".join(
            filter(
                None,
                [
                    card.summary,
                    *[f"Finding: {f}" for f in card.key_findings],
                    *[f"Limitation: {x}" for x in card.limitations],
                ],
            )
        )
        text = f"{card.title} {card.abstract} {notes}"
        return PaperRecord(
            **card.model_dump(
                include=set(PaperRecord.model_fields)
                - {"matched_queries", "relevance", "prior_notes", "evidence_excerpt"}
            ),
            matched_queries=[query],
            relevance=relevance_score(query, text),
            prior_notes=f"(sessions: {', '.join(card.sessions)}) {notes}".strip(),
        )


class LibraryBackend:
    """``SearchBackend`` over the library: earlier sessions' papers come back with their notes."""

    name = "library"

    def __init__(self, library: ResearchLibrary) -> None:
        self.library = library

    async def search(self, query: str, max_results: int) -> list[PaperRecord]:
        hits = await self.library.search(query, limit=max_results * 2)
        records = []
        for hit in hits:
            if hit.kind != "paper":
                continue
            record = self.library.as_paper_record(hit.ref, query)
            if record is not None:
                records.append(record.model_copy(update={"backend": self.name}))
        return records[:max_results]


# -- cards ----------------------------------------------------------------------------


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) or "- (none)"


def render_paper_card(paper: LibraryPaper) -> str:
    return (
        f"# {paper.title}\n\n"
        f"Source: {paper.source_id}. Authors: {', '.join(paper.authors[:6]) or 'unknown'}. "
        f"Year: {paper.year or 'n.d.'}. DOI: {paper.doi or 'none'}.\n\n"
        f"Topics: {'; '.join(paper.topics)}\n\n"
        f"Summary: {paper.summary or 'not reviewed yet'}\n\n"
        f"Abstract: {paper.abstract}\n\n"
        f"Key findings:\n{_bullets(paper.key_findings)}\n\n"
        f"Limitations:\n{_bullets(paper.limitations)}\n"
    )


def render_session_card(session: SessionRecord) -> str:
    return (
        f"# Research session {session.session_id}\n\n"
        f"Question: {session.research_question}\n\n"
        f"Key concepts: {', '.join(session.key_concepts)}\n\n"
        f"Synthesis: {session.synthesis}\n\n"
        f"Recommended direction: {session.recommended_direction}. {session.recommended_hypothesis}\n\n"
        f"Gaps:\n{_bullets([f'{g.id}: {g.description}' for g in session.gaps])}\n"
    )


def render_gap_card(session: SessionRecord, gap: Gap) -> str:
    return (
        f"# Open gap {gap.id} from session {session.session_id}\n\n"
        f"{gap.description}\n\n"
        f"Context: {session.research_question}\n\n"
        f"Evidence: {', '.join(gap.evidence_source_ids)}\n"
    )


def library_summary(library: ResearchLibrary) -> dict[str, Any]:
    return {
        "root": str(library.root),
        "papers": len(library.state.papers),
        "sessions": [s.session_id for s in library.state.sessions],
        "index": library.index_name,
    }


__all__ = [
    "KeywordIndex",
    "LibraryBackend",
    "LibraryHit",
    "LibraryPaper",
    "OpenJiuwenMemoryIndex",
    "ResearchLibrary",
    "SessionRecord",
    "library_summary",
    "render_gap_card",
    "render_paper_card",
    "render_session_card",
]
