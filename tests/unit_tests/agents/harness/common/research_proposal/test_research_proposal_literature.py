from __future__ import annotations

from pathlib import Path

import httpx

from jiuwenswarm.agents.harness.common.research_proposal.literature import (
    ArxivBackend,
    LiteratureRetriever,
    OpenJiuwenWebBackend,
    arxiv_doi,
    clean_title,
    parse_free_search_output,
    relevance_score,
    select_diverse,
)
from jiuwenswarm.agents.harness.common.research_proposal.model import (
    extract_json_object,
)
from jiuwenswarm.agents.harness.common.research_proposal.schemas import (
    PaperRecord,
    ProblemFrame,
)

# Real failure mode seen with openjiuwen free_search: DuckDuckGo challenged the
# query and the Bing fallback returned tax forms for an ML query.
BING_JUNK = """Free search results (Bing) for: GLU variants SwiGLU transformer feed-forward
1. About Form 8829 , Expenses for Business Use of Your Home
   URL: https://www.irs.gov/forms-pubs/about-form-8829
   Snippet: Information about Form 8829, Expenses for Business Use of Your Home.
2. [2002.05202] GLU Variants Improve Transformer - arXiv.org
   URL: https://arxiv.org/abs/2002.05202
   Snippet: Variations on GLU are possible, using different nonlinear functions in place of sigmoid.

Required next step: before reformulating the query, fetch at least 2 relevant URLs.
Recommended fetch targets:
1. https://www.irs.gov/forms-pubs
"""


def _entry(identifier: str, title: str, summary: str) -> str:
    return f"""<entry>
    <id>http://arxiv.org/abs/{identifier}v1</id>
    <published>2020-02-12T00:00:00Z</published>
    <title>{title}</title>
    <summary>{summary}</summary>
    <author><name>Noam Shazeer</name></author>
    <link title="pdf" href="https://arxiv.org/pdf/{identifier}v1" rel="related" type="application/pdf"/>
  </entry>"""


def _feed(*entries: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">'
        + "".join(entries)
        + "</feed>"
    )


class FakeFreeSearchTool:
    def __init__(self, output: str) -> None:
        self.output = output
        self.inputs: list[dict] = []

    async def invoke(self, inputs: dict) -> str:
        self.inputs.append(inputs)
        return self.output


def test_relevance_score_ignores_stopwords_and_plurals():
    assert (
        relevance_score(
            "learnable activation functions", "A learnable activation function"
        )
        == 1.0
    )
    assert (
        relevance_score(
            "GLU variants transformer", "Form 8829 business use of your home"
        )
        == 0.0
    )


def test_arxiv_doi_and_title_cleanup():
    assert (
        arxiv_doi("https://arxiv.org/pdf/2002.05202v3") == "10.48550/arXiv.2002.05202"
    )
    assert (
        arxiv_doi("https://huggingface.co/papers/1710.05941")
        == "10.48550/arXiv.1710.05941"
    )
    assert arxiv_doi("https://proceedings.mlr.press/v15/glorot11a.pdf") is None
    assert (
        clean_title("[2002.05202] GLU Variants Improve Transformer - arXiv.org")
        == "GLU Variants Improve Transformer"
    )
    assert (
        clean_title("Paper page - Searching for Activation Functions")
        == "Searching for Activation Functions"
    )


async def test_web_backend_drops_off_topic_fallback_results_and_derives_doi():
    tool = FakeFreeSearchTool(BING_JUNK)
    backend = OpenJiuwenWebBackend(tool=tool)

    records = await backend.search("GLU variants SwiGLU transformer feed-forward", 8)

    assert [r.source_id for r in records] == ["arxiv:2002.05202"]
    assert records[0].title == "GLU Variants Improve Transformer"
    assert records[0].doi == "10.48550/arXiv.2002.05202"
    assert records[0].pdf_url == "https://arxiv.org/pdf/2002.05202"
    assert backend.rejected == 1
    assert (
        len(parse_free_search_output(BING_JUNK)) == 2
    )  # the "Recommended fetch targets" list is not a result


async def test_web_backend_returns_nothing_on_tool_error():
    backend = OpenJiuwenWebBackend(
        tool=FakeFreeSearchTool("[ERROR]: free search failed: all engines disabled")
    )
    assert await backend.search("anything at all", 5) == []


async def test_arxiv_backend_parses_feed_and_broadens_thin_results():
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        query = request.url.params.get("search_query", "")
        queries.append(query)
        if query.count("AND") == 2:
            return httpx.Response(200, text=_feed())
        return httpx.Response(
            200,
            text=_feed(
                _entry(
                    "2002.05202", "GLU Variants Improve\n Transformer", "GLU variants."
                )
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = ArxivBackend(client=client, min_interval=0)

    records = await backend.search("GLU variants improve transformer layers", 6)
    await client.aclose()

    # Exact phrase first, then progressively broader ANDs while results are thin.
    assert queries == [
        'all:"glu variants improve transformer layers"',
        "all:glu AND all:variant AND all:improve",
        "all:glu AND all:variant",
    ]
    assert len(records) == 1  # the same paper found twice is kept once
    paper = records[0]
    assert paper.source_id == "arxiv:2002.05202"
    assert paper.title == "GLU Variants Improve Transformer"
    assert paper.doi == "10.48550/arXiv.2002.05202"
    assert paper.year == "2020"
    assert paper.authors == ["Noam Shazeer"]
    assert paper.relevance > 0.5


async def test_retriever_merges_web_copy_into_arxiv_record_and_enriches(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("id_list"):
            return httpx.Response(
                200,
                text=_feed(
                    _entry(
                        "2002.05202",
                        "GLU Variants Improve Transformer",
                        "Full abstract.",
                    )
                ),
            )
        return httpx.Response(200, text=_feed())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    web = OpenJiuwenWebBackend(tool=FakeFreeSearchTool(BING_JUNK))
    retriever = LiteratureRetriever(
        backends=[ArxivBackend(client=client, min_interval=0), web],
        downloader=None,
        pause_seconds=0,
        min_relevance=0.3,
    )
    frame = ProblemFrame(
        raw_problem="x",
        research_question="Why do GLU variants help Transformers?",
        motivation="m",
        key_concepts=["GLU"],
        search_queries=[
            "GLU variants SwiGLU transformer feed-forward",
            "SwiGLU transformer feed-forward GLU",
        ],
    )

    corpus = await retriever.retrieve(frame, tmp_path / "sources", tmp_path)
    await client.aclose()

    assert [p.source_id for p in corpus.papers] == ["arxiv:2002.05202"]
    paper = corpus.papers[0]
    assert paper.abstract == "Full abstract."  # replaced by the arXiv API record
    assert paper.authors == ["Noam Shazeer"]
    assert len(paper.matched_queries) == 2
    assert corpus.rejected_results == 2  # one tax-form result per query


def test_extract_json_object_accepts_fences_and_prose():
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('Here you go: {"a": {"b": [1, 2]}} done') == {
        "a": {"b": [1, 2]}
    }


def test_select_diverse_covers_every_query_before_going_deep():
    def paper(key: str, queries: list[str]) -> PaperRecord:
        return PaperRecord(
            source_id=key,
            title=key,
            url=f"https://x/{key}",
            matched_queries=queries,
            relevance=1.0,
        )

    papers = {
        "a1": paper("a1", ["qa"]),
        "a2": paper("a2", ["qa"]),
        "a3": paper("a3", ["qa"]),
        "b1": paper("b1", ["qb"]),
        "both": paper("both", ["qa", "qb"]),
    }
    per_query = {"qa": ["a1", "a2", "a3", "both"], "qb": ["both", "b1"]}

    picked = [p.source_id for p in select_diverse(papers, per_query, 3)]

    # The paper both queries found comes first, then one per query in rank order.
    assert picked == ["both", "a1", "b1"]


async def test_arxiv_backend_backs_off_on_rate_limit():
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        queries.append(request.url.params["search_query"])
        if len(queries) == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(
            200,
            text=_feed(_entry("2002.05202", "GLU Variants Improve Transformer", "x")),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = ArxivBackend(client=client, min_interval=0, backoff=0)

    records = await backend.search("GLU variants", 4)
    await client.aclose()

    # The rate-limited request is retried as is before any broader query runs.
    assert queries[:2] == ['all:"glu variants"', 'all:"glu variants"']
    assert [r.source_id for r in records] == ["arxiv:2002.05202"]


async def test_retriever_records_backend_failures(tmp_path: Path):
    class Failing:
        name = "arxiv"

        async def search(self, query: str, max_results: int):
            raise httpx.HTTPStatusError(
                "429",
                request=httpx.Request("GET", "https://x"),
                response=httpx.Response(429),
            )

    retriever = LiteratureRetriever(
        backends=[Failing()], downloader=None, pause_seconds=0
    )
    frame = ProblemFrame(
        raw_problem="x",
        research_question="q",
        motivation="m",
        key_concepts=["k"],
        search_queries=["a b", "c d"],
    )

    corpus = await retriever.retrieve(frame, tmp_path, tmp_path)

    assert corpus.papers == []
    assert len(corpus.backend_errors) == 2
    assert corpus.backend_errors[0].startswith("arxiv: 'a b': 429")


async def test_library_only_papers_are_capped_and_keep_their_notes(tmp_path: Path):
    def paper(i: int, backend: str) -> PaperRecord:
        return PaperRecord(
            source_id=f"arxiv:2400.0000{i}",
            title=f"Tool reasoning paper {i} {backend}",
            url=f"https://arxiv.org/abs/2400.0000{i}",
            backend=backend,
            relevance=1.0,
            prior_notes="seen yesterday" if backend == "library" else "",
        )

    class Static:
        def __init__(self, name: str, papers: list[PaperRecord]) -> None:
            self.name, self.papers = name, papers

        async def search(self, query: str, max_results: int):
            return [
                p.model_copy(update={"matched_queries": [query]}) for p in self.papers
            ]

    library = Static("library", [paper(i, "library") for i in range(5)])
    fresh = Static("arxiv", [paper(i, "arxiv") for i in range(5, 9)])
    retriever = LiteratureRetriever(
        backends=[library, fresh],
        downloader=None,
        pause_seconds=0,
        max_papers=8,
        min_topic_relevance=0.0,
    )
    frame = ProblemFrame(
        raw_problem="x",
        research_question="q",
        motivation="m",
        key_concepts=["tool"],
        search_queries=["tool reasoning", "reasoning paper"],
    )

    corpus = await retriever.retrieve(frame, tmp_path, tmp_path)

    from_library = [p for p in corpus.papers if p.backend == "library"]
    assert len(from_library) == 2  # 25% of 8 slots
    assert all(p.prior_notes == "seen yesterday" for p in from_library)
    assert len([p for p in corpus.papers if p.backend == "arxiv"]) == 4
