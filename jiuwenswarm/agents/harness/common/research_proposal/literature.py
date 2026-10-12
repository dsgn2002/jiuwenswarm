"""Literature retrieval for the research-proposal pipeline.

Retrieval is deterministic code, not model output: backends return
``PaperRecord`` rows, the retriever filters, de-duplicates, ranks and
downloads them, and only then does a model read the evidence.

Three problems found when running openjiuwen's topic-survey tools are handled
here:

1. ``WebFreeSearchTool`` ships with every engine disabled unless configured;
   ``OpenJiuwenWebBackend`` always passes ``enabled_engines`` explicitly.
2. When DuckDuckGo answers with an anti-bot challenge the tool falls back to
   Bing, which returned unrelated pages (tax forms, OS activation help) for
   ML queries. Every web result must now pass ``relevance_score``.
3. openjiuwen's citation resolver marks sources without a DOI as
   evidence-only, which excluded arXiv papers. ``arxiv_doi`` derives the
   DataCite DOI (``10.48550/arXiv.<id>``) that arXiv assigns to every paper.

``ArxivBackend`` adds a scholarly API source, so retrieval no longer depends
only on scraping general web search engines.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from jiuwenswarm.agents.harness.common.research_proposal.schemas import (
    Corpus,
    PaperRecord,
    ProblemFrame,
)

_LOGGER = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "via",
        "vs",
        "what",
        "when",
        "which",
        "why",
        "with",
        "using",
        "based",
        "towards",
        "toward",
        "new",
        "study",
        "analysis",
        "approach",
        "approaches",
        "method",
        "methods",
        "paper",
        "arxiv",
        "pdf",
        "survey",
        "review",
    ]
)
SCHOLARLY_DOMAINS = (
    "arxiv.org",
    "openreview.net",
    "proceedings.mlr.press",
    "jmlr.org",
    "aclanthology.org",
    "neurips.cc",
    "thecvf.com",
    "semanticscholar.org",
    "doi.org",
    "dl.acm.org",
    "ieeexplore.ieee.org",
    "link.springer.com",
    "sciencedirect.com",
    "nature.com",
    "pmlr.org",
)
_ARXIV_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf|html)|huggingface\.co/papers)/(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)
_TITLE_NOISE_RE = re.compile(
    r"^(?:\[\d{4}\.\d{4,5}\]\s*|paper page\s*-\s*|\(?pdf\)?\s+)", re.IGNORECASE
)
_TITLE_SUFFIX_RE = re.compile(r"\s+[-|–]\s+[^-|–]{2,40}$")


def clean_title(title: str) -> str:
    """Drop search-engine decorations: '[2002.05202] X - arXiv.org' -> 'X'."""
    cleaned = _TITLE_NOISE_RE.sub("", (title or "").strip())
    cleaned = _TITLE_SUFFIX_RE.sub("", cleaned)
    return cleaned.strip() or (title or "").strip()


_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


def content_terms(text: str) -> list[str]:
    """Lower-cased content words of ``text`` (no stopwords, length > 2)."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return list(dict.fromkeys(w for w in words if len(w) > 2 and w not in _STOPWORDS))


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def relevance_score(query: str, text: str) -> float:
    """Share of the query's content words that occur in ``text`` (0-1)."""
    terms = list(dict.fromkeys(_singular(t) for t in content_terms(query)))
    if not terms:
        return 0.0
    haystack = {_singular(w) for w in re.findall(r"[a-z0-9]+", (text or "").lower())}
    return round(sum(1 for t in terms if t in haystack) / len(terms), 3)


def is_scholarly_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in SCHOLARLY_DOMAINS)


def arxiv_id(url: str) -> str | None:
    match = _ARXIV_ID_RE.search(url or "")
    return match.group(1) if match else None


def arxiv_doi(url: str) -> str | None:
    """DataCite DOI that arXiv assigns to every paper, derived from its URL."""
    identifier = arxiv_id(url)
    return f"10.48550/arXiv.{identifier}" if identifier else None


def title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())


def source_id_for(url: str, title: str) -> str:
    identifier = arxiv_id(url)
    if identifier:
        return f"arxiv:{identifier}"
    digest = hashlib.sha256((title_key(title) or url).encode("utf-8")).hexdigest()[:10]
    return f"web:{digest}"


class SearchBackend(Protocol):
    name: str

    async def search(self, query: str, max_results: int) -> list[PaperRecord]:
        """Return candidate papers for one query; never raise for empty results."""


# -- arXiv API -------------------------------------------------------------------


class ArxivBackend:
    """Search through the public arXiv export API (Atom feed)."""

    name = "arxiv"

    def __init__(
        self,
        *,
        endpoint: str = "https://export.arxiv.org/api/query",
        max_query_terms: int = 3,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
        min_interval: float = 3.0,
        retries: int = 3,
        backoff: float = 10.0,
    ) -> None:
        self._endpoint = endpoint
        self._max_query_terms = max_query_terms
        self._timeout = timeout
        self._client = client
        # arXiv asks API clients for at least 3 s between requests and answers
        # bursts with HTTP 429; space every request and back off on 429/503.
        self._min_interval = min_interval
        self._retries = retries
        self._backoff = backoff
        self._last_request = 0.0

    def build_query(self, query: str, max_terms: int | None = None) -> str:
        """AND of the first few content words, singularised.

        arXiv matches whole words and ANDs every clause, so long queries or
        plural forms ("functions") return almost nothing; three singular
        terms gave the most relevant results in manual comparison.
        """
        terms = [_singular(t) for t in content_terms(query)][
            : max_terms or self._max_query_terms
        ]
        return " AND ".join(f"all:{t}" for t in terms) if terms else f"all:{query}"

    async def _request(self, params: dict[str, Any]) -> httpx.Response:
        wait = self._last_request + self._min_interval - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request = time.monotonic()
        if self._client is not None:
            return await self._client.get(
                self._endpoint, params=params, timeout=self._timeout
            )
        async with httpx.AsyncClient(follow_redirects=True) as client:
            return await client.get(
                self._endpoint, params=params, timeout=self._timeout
            )

    async def _get(self, params: dict[str, Any]) -> str:
        for attempt in range(self._retries + 1):
            response = await self._request(params)
            if response.status_code not in (429, 503) or attempt == self._retries:
                break
            delay = self._backoff * 2**attempt
            _LOGGER.warning(
                "arXiv answered %s; retrying in %.0fs", response.status_code, delay
            )
            await asyncio.sleep(delay)
        response.raise_for_status()
        return response.text

    @staticmethod
    def phrase_query(query: str) -> str | None:
        """Exact-phrase query for short, title-like queries ("Gaussian error linear units").

        Three ANDed words of a name such as "gaussian AND error AND linear"
        match regression papers; the phrase finds the named method itself.
        """
        words = re.findall(r"[A-Za-z0-9]+", query)
        return f'all:"{" ".join(words).lower()}"' if 2 <= len(words) <= 5 else None

    async def search(self, query: str, max_results: int) -> list[PaperRecord]:
        params = {"max_results": max_results, "sortBy": "relevance"}
        attempts = [self.phrase_query(query), self.build_query(query)]
        if len(content_terms(query)) > 2:
            attempts.append(self.build_query(query, 2))
        records: list[PaperRecord] = []
        for search_query in filter(None, attempts):
            if len(records) >= max(2, max_results // 2):
                break
            # Each broader query only tops up: earlier, stricter hits keep their rank.
            seen = {r.source_id for r in records}
            found = parse_arxiv_feed(
                await self._get({**params, "search_query": search_query})
            )
            records += [r for r in found if r.source_id not in seen]
        return [replace_query(record, query) for record in records[:max_results]]

    async def lookup(self, ids: Iterable[str]) -> list[PaperRecord]:
        id_list = ",".join(dict.fromkeys(ids))
        if not id_list:
            return []
        return parse_arxiv_feed(
            await self._get({"id_list": id_list, "max_results": 100})
        )


def replace_query(record: PaperRecord, query: str) -> PaperRecord:
    text = f"{record.title} {record.abstract}"
    return record.model_copy(
        update={"matched_queries": [query], "relevance": relevance_score(query, text)}
    )


def parse_arxiv_feed(feed: str) -> list[PaperRecord]:
    root = ET.fromstring(feed)
    records: list[PaperRecord] = []
    for entry in root.findall(f"{_ATOM}entry"):
        abs_url = (entry.findtext(f"{_ATOM}id") or "").strip()
        identifier = arxiv_id(abs_url)
        title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
        if not identifier or not title:
            continue
        pdf_url = next(
            (
                link.get("href")
                for link in entry.findall(f"{_ATOM}link")
                if link.get("title") == "pdf"
            ),
            f"https://arxiv.org/pdf/{identifier}",
        )
        published = (entry.findtext(f"{_ATOM}published") or "")[:4] or None
        records.append(
            PaperRecord(
                source_id=f"arxiv:{identifier}",
                title=title,
                url=f"https://arxiv.org/abs/{identifier}",
                pdf_url=pdf_url,
                authors=[
                    " ".join((a.findtext(f"{_ATOM}name") or "").split())
                    for a in entry.findall(f"{_ATOM}author")
                    if a.findtext(f"{_ATOM}name")
                ],
                year=published,
                venue=(entry.findtext(f"{_ARXIV_NS}journal_ref") or "").strip()
                or "arXiv",
                doi=(entry.findtext(f"{_ARXIV_NS}doi") or "").strip()
                or f"10.48550/arXiv.{identifier}",
                abstract=" ".join((entry.findtext(f"{_ATOM}summary") or "").split()),
                backend="arxiv",
            )
        )
    return records


# -- openjiuwen web search -------------------------------------------------------

_RESULT_RE = re.compile(
    r"^\s*\d+\.\s+(?P<title>.+?)\s*\n\s*URL:\s*(?P<url>\S+)\s*(?:\n\s*Snippet:\s*(?P<snippet>.*))?",
    re.MULTILINE,
)


def parse_free_search_output(text: str) -> list[dict[str, str]]:
    """Parse the numbered text list that openjiuwen's ``free_search`` returns."""
    body = (text or "").split("Required next step:", 1)[0]
    return [
        {
            "title": m.group("title").strip(),
            "url": m.group("url").strip(),
            "snippet": (m.group("snippet") or "").strip(),
        }
        for m in _RESULT_RE.finditer(body)
    ]


class OpenJiuwenWebBackend:
    """openjiuwen ``WebFreeSearchTool`` with engines enabled and results filtered.

    A result is kept when at least ``min_relevance`` of the query's content
    words occur in its title and snippet, or ``scholarly_min_relevance`` for
    results on a scholarly domain.
    """

    name = "openjiuwen_web"

    def __init__(
        self,
        *,
        engines: tuple[str, ...] = ("duckduckgo", "bing"),
        proxy_url: str | None = None,
        min_relevance: float = 0.6,
        scholarly_min_relevance: float = 0.34,
        tool: Any = None,
    ) -> None:
        self._engines = engines
        self._proxy_url = proxy_url
        self.min_relevance = min_relevance
        self.scholarly_min_relevance = scholarly_min_relevance
        self._tool = tool
        self.rejected = 0

    def _get_tool(self) -> Any:
        if self._tool is None:
            from openjiuwen.harness.tools.web import WebFreeSearchTool

            self._tool = WebFreeSearchTool(
                language="en", proxy_url=self._proxy_url, enabled_engines=self._engines
            )
        return self._tool

    def accept(
        self, query: str, title: str, snippet: str, url: str
    ) -> tuple[bool, float]:
        score = relevance_score(query, f"{title} {snippet}")
        threshold = (
            self.scholarly_min_relevance
            if is_scholarly_url(url)
            else self.min_relevance
        )
        return score >= threshold, score

    async def search(self, query: str, max_results: int) -> list[PaperRecord]:
        output = await self._get_tool().invoke(
            {"query": query, "max_results": max_results}
        )
        text = output if isinstance(output, str) else str(output)
        if text.startswith("[ERROR]"):
            _LOGGER.warning(
                "openjiuwen free_search failed for %r: %s", query, text[:300]
            )
            return []
        records: list[PaperRecord] = []
        for row in parse_free_search_output(text):
            keep, score = self.accept(query, row["title"], row["snippet"], row["url"])
            if not keep:
                self.rejected += 1
                continue
            identifier = arxiv_id(row["url"])
            title = clean_title(row["title"])
            records.append(
                PaperRecord(
                    source_id=source_id_for(row["url"], title),
                    title=title,
                    url=f"https://arxiv.org/abs/{identifier}"
                    if identifier
                    else row["url"],
                    pdf_url=f"https://arxiv.org/pdf/{identifier}"
                    if identifier
                    else None,
                    doi=arxiv_doi(row["url"]),
                    abstract=row["snippet"],
                    backend=self.name,
                    matched_queries=[query],
                    relevance=score,
                )
            )
        return records


# -- download and evidence -------------------------------------------------------


@dataclass
class Downloader:
    """Save each selected source under ``sources_dir`` and keep a text excerpt."""

    timeout: float = 60.0
    max_bytes: int = 25_000_000
    excerpt_chars: int = 4000
    client: httpx.AsyncClient | None = None

    async def _fetch(self, url: str) -> httpx.Response:
        if self.client is not None:
            return await self.client.get(
                url, timeout=self.timeout, follow_redirects=True
            )
        async with httpx.AsyncClient(
            follow_redirects=True, headers={"User-Agent": "jiuwenswarm-research/0.1"}
        ) as client:
            return await client.get(url, timeout=self.timeout)

    async def download(
        self, paper: PaperRecord, sources_dir: Path, project_root: Path
    ) -> PaperRecord:
        url = paper.pdf_url or paper.url
        try:
            response = await self._fetch(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            _LOGGER.warning("download failed for %s: %s", url, exc)
            return paper
        content = response.content[: self.max_bytes]
        is_pdf = content[:5] == b"%PDF-"
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", paper.source_id)
        path = sources_dir / f"{stem}.{'pdf' if is_pdf else 'html'}"
        sources_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        excerpt = pdf_excerpt(path, self.excerpt_chars) if is_pdf else ""
        return paper.model_copy(
            update={
                "local_path": path.relative_to(project_root).as_posix(),
                "retrieval_mode": "downloaded_pdf" if is_pdf else "downloaded_html",
                "evidence_excerpt": excerpt,
            }
        )


def pdf_excerpt(path: Path, max_chars: int) -> str:
    """Abstract-to-introduction and conclusion text from a PDF, if pypdf is installed."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:12])
    except Exception as exc:  # noqa: BLE001 - malformed PDFs are common; keep the abstract
        _LOGGER.warning("could not read %s: %s", path, exc)
        return ""
    text = re.sub(r"\s+", " ", text)
    lower = text.lower()
    head = lower.find("abstract")
    head = max(head, 0)
    opening = text[head : head + max_chars // 2]
    tail = max(lower.rfind("conclusion"), lower.rfind("discussion"))
    closing = text[tail : tail + max_chars // 2] if tail > len(text) // 3 else ""
    return (opening + (" ... " + closing if closing else "")).strip()


# -- retrieval orchestration -----------------------------------------------------


@dataclass
class LiteratureRetriever:
    backends: list[SearchBackend]
    downloader: Downloader | None = field(default_factory=Downloader)
    max_papers: int = 12
    results_per_query: int = 8
    min_relevance: float = 0.34
    # Share of the framed problem's key-concept words a paper must mention.
    # Catches results that match one query's words but not the topic.
    min_topic_relevance: float = 0.2
    # At most this share of the slots goes to papers found only in the research
    # library: earlier sessions should inform a new topic, not crowd out its search.
    max_library_share: float = 0.25
    pause_seconds: float = 3.0

    async def retrieve(
        self, frame: ProblemFrame, sources_dir: Path, project_root: Path
    ) -> Corpus:
        merged: dict[str, PaperRecord] = {}
        # Per query, the papers it returned in the backends' own rank order.
        per_query: dict[str, list[tuple[str, str]]] = {
            q: [] for q in frame.search_queries
        }
        rejected = 0
        errors: list[str] = []
        for query in frame.search_queries:
            for backend in self.backends:
                try:
                    records = await backend.search(query, self.results_per_query)
                except Exception as exc:  # noqa: BLE001 - one failing backend must not stop retrieval
                    name = getattr(backend, "name", type(backend).__name__)
                    _LOGGER.warning("backend %s failed for %r: %s", name, query, exc)
                    errors.append(f"{name}: {query!r}: {exc}"[:300])
                    records = []
                for record in records:
                    if record.relevance < self.min_relevance:
                        rejected += 1
                        continue
                    _merge(merged, record)
                    per_query[query].append((record.source_id, title_key(record.title)))
                if self.pause_seconds:
                    await asyncio.sleep(self.pause_seconds)
        rejected += sum(getattr(b, "rejected", 0) for b in self.backends)
        by_title: dict[str, PaperRecord] = {}
        for record in sorted(
            merged.values(), key=lambda p: not p.source_id.startswith("arxiv:")
        ):
            # arXiv-keyed rows go first, so a web copy with the same title folds into them.
            _merge(by_title, record, key=title_key(record.title))
        topic = " ".join(frame.key_concepts)
        for key in [
            k
            for k, p in by_title.items()
            if relevance_score(topic, f"{p.title} {p.abstract}")
            < self.min_topic_relevance
        ]:
            del by_title[key]
            rejected += 1
        library_only = sorted(
            (k for k, p in by_title.items() if p.backend == "library"),
            key=lambda k: (
                -len(by_title[k].matched_queries),
                -by_title[k].relevance,
                k,
            ),
        )
        for key in library_only[
            max(1, int(self.max_papers * self.max_library_share)) :
        ]:
            del by_title[key]
        # A paper's final key is its title in by_title; arXiv rows may have been
        # renamed by the merge, so resolve through their source_id first.
        sid_to_key = {p.source_id: k for k, p in by_title.items()}
        ranked_keys = {
            q: [sid_to_key.get(sid, tkey) for sid, tkey in hits]
            for q, hits in per_query.items()
        }
        selected = await self._enrich_arxiv(
            select_diverse(by_title, ranked_keys, self.max_papers)
        )
        if self.downloader is not None:
            selected = [
                await self.downloader.download(p, sources_dir, project_root)
                for p in selected
            ]
        return Corpus(
            problem=frame.research_question,
            papers=selected,
            queries_run=list(frame.search_queries),
            rejected_results=rejected,
            backend_errors=errors,
        )

    async def _enrich_arxiv(self, papers: list[PaperRecord]) -> list[PaperRecord]:
        """Replace web-snippet metadata of arXiv papers with the arXiv API record."""
        arxiv = next((b for b in self.backends if isinstance(b, ArxivBackend)), None)
        wanted = [
            p.source_id.split(":", 1)[1]
            for p in papers
            if p.source_id.startswith("arxiv:")
            and p.backend not in ("arxiv", "library")
        ]
        if arxiv is None or not wanted:
            return papers
        try:
            found = {r.source_id: r for r in await arxiv.lookup(wanted)}
        except Exception as exc:  # noqa: BLE001 - enrichment is best effort
            _LOGGER.warning("arXiv metadata lookup failed: %s", exc)
            return papers
        enriched = []
        for paper in papers:
            api = found.get(paper.source_id)
            if api is None or paper.backend in ("arxiv", "library"):
                enriched.append(paper)
                continue
            enriched.append(
                api.model_copy(
                    update={
                        "matched_queries": paper.matched_queries,
                        "relevance": paper.relevance,
                        "backend": paper.backend,
                        "prior_notes": paper.prior_notes,
                    }
                )
            )
        return enriched


def select_diverse(
    papers: dict[str, PaperRecord], per_query: dict[str, list[str]], limit: int
) -> list[PaperRecord]:
    """Pick ``limit`` papers so that every query's facet is represented.

    The relevance score saturates (most kept results contain every query
    word), so sorting by it alone lets one productive query crowd out the
    rest. Papers matched by several queries come first; the remaining slots
    are filled round-robin across queries, each in its backends' rank order.
    """
    chosen: list[str] = [
        key
        for key, paper in sorted(
            papers.items(), key=lambda item: (-len(item[1].matched_queries), item[0])
        )
        if len(paper.matched_queries) > 1
    ][:limit]
    queues = [
        [k for k in dict.fromkeys(keys) if k in papers] for keys in per_query.values()
    ]
    while len(chosen) < limit and any(queues):
        for queue in queues:
            while queue and queue[0] in chosen:
                queue.pop(0)
            if queue and len(chosen) < limit:
                chosen.append(queue.pop(0))
    return [papers[key] for key in chosen]


def _merge(
    merged: dict[str, PaperRecord], record: PaperRecord, *, key: str | None = None
) -> None:
    if key is None:
        key = (
            record.source_id
            if record.source_id.startswith("arxiv:")
            else f"title:{title_key(record.title)}"
        )
    existing = merged.get(key)
    if existing is None:
        merged[key] = record
        return
    # Keep the richer metadata (the arXiv API row beats a web snippet) and pool the evidence of relevance.
    primary, other = (
        (record, existing)
        if record.backend == "arxiv" and existing.backend != "arxiv"
        else (existing, record)
    )
    merged[key] = primary.model_copy(
        update={
            "matched_queries": list(
                dict.fromkeys(primary.matched_queries + other.matched_queries)
            ),
            "relevance": max(primary.relevance, other.relevance),
            "abstract": primary.abstract or other.abstract,
            "prior_notes": primary.prior_notes or other.prior_notes,
        }
    )


__all__ = [
    "SCHOLARLY_DOMAINS",
    "ArxivBackend",
    "Downloader",
    "LiteratureRetriever",
    "OpenJiuwenWebBackend",
    "SearchBackend",
    "arxiv_doi",
    "arxiv_id",
    "clean_title",
    "content_terms",
    "is_scholarly_url",
    "parse_arxiv_feed",
    "parse_free_search_output",
    "pdf_excerpt",
    "relevance_score",
    "source_id_for",
]
