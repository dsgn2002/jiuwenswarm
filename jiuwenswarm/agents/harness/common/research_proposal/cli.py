"""Command line entry point.

Examples::

    # Operator or coding agent acts as the model through files in RUN/exchange
    python -m jiuwenswarm.agents.harness.common.research_proposal \\
        --problem "Can activation functions be learned per layer cheaply?" --run-dir runs/act

    # Use an API model configured the way openjiuwen's autoResearch expects
    API_KEY=... API_BASE=... MODEL_NAME=... python -m ... --model openjiuwen --problem ...

With ``--model exchange`` the command exits with status 3 and prints the prompt
path each time a stage needs an answer; write the JSON answer next to it and
run the same command again.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from jiuwenswarm.agents.harness.common.research_proposal.literature import (
    ArxivBackend,
    Downloader,
    LiteratureRetriever,
    OpenJiuwenWebBackend,
)
from jiuwenswarm.agents.harness.common.research_proposal.model import (
    AwaitingModelResponse,
    FileExchangeReasoningModel,
    OpenJiuwenReasoningModel,
)
from jiuwenswarm.agents.harness.common.research_proposal.pipeline import (
    ResearchProposalPipeline,
)

EXIT_AWAITING = 3


def build_openjiuwen_model():
    """Build an openjiuwen ``Model`` from the environment, with TLS verification on."""
    from openjiuwen.core.foundation.llm import Model
    from openjiuwen.core.foundation.llm.schema.config import (
        ModelClientConfig,
        ModelRequestConfig,
    )

    api_key = os.getenv("API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "--model openjiuwen needs API_KEY (and usually API_BASE, MODEL_NAME, MODEL_PROVIDER)"
        )
    return Model(
        model_client_config=ModelClientConfig(
            client_provider=os.getenv("MODEL_PROVIDER", "OpenAI"),
            api_key=api_key,
            api_base=os.getenv("API_BASE", "https://api.openai.com/v1"),
            timeout=int(os.getenv("MODEL_TIMEOUT", "300")),
            verify_ssl=True,
        ),
        model_config=ModelRequestConfig(
            model_name=os.getenv("MODEL_NAME", "gpt-4.1-mini"), temperature=0.2
        ),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Problem framing -> literature review -> directions -> proposal"
    )
    parser.add_argument(
        "--problem", help="raw research problem; may be omitted on resume"
    )
    parser.add_argument(
        "--problem-file", type=Path, help="read the raw problem from a file"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--model", choices=("exchange", "openjiuwen"), default="exchange"
    )
    parser.add_argument(
        "--backends", default="arxiv,web", help="comma list of: arxiv, web"
    )
    parser.add_argument("--max-papers", type=int, default=12)
    parser.add_argument("--results-per-query", type=int, default=8)
    parser.add_argument(
        "--no-download", action="store_true", help="keep metadata and abstracts only"
    )
    return parser.parse_args(argv)


def _raw_problem(args: argparse.Namespace) -> str:
    stored = args.run_dir / "raw_problem.txt"
    if args.problem_file:
        text = args.problem_file.read_text(encoding="utf-8")
    elif args.problem:
        text = args.problem
    elif stored.exists():
        text = stored.read_text(encoding="utf-8")
    else:
        raise SystemExit("give --problem or --problem-file on the first run")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    stored.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw_problem = _raw_problem(args)
    backends = []
    for name in (b.strip() for b in args.backends.split(",") if b.strip()):
        if name == "arxiv":
            backends.append(ArxivBackend())
        elif name == "web":
            backends.append(
                OpenJiuwenWebBackend(proxy_url=os.getenv("HTTPS_PROXY") or None)
            )
        else:
            raise SystemExit(f"unknown backend {name!r}")
    model = (
        FileExchangeReasoningModel(args.run_dir / "exchange")
        if args.model == "exchange"
        else OpenJiuwenReasoningModel(build_openjiuwen_model())
    )
    retriever = LiteratureRetriever(
        backends=backends,
        downloader=None if args.no_download else Downloader(),
        max_papers=args.max_papers,
        results_per_query=args.results_per_query,
    )
    pipeline = ResearchProposalPipeline(model, retriever, args.run_dir)
    try:
        result = asyncio.run(pipeline.run(raw_problem))
    except AwaitingModelResponse as waiting:
        print(
            f"AWAITING {waiting.stage}\nprompt: {waiting.prompt_path}\nanswer: {waiting.response_path}"
        )
        return EXIT_AWAITING
    print(
        f"proposal: {result.proposal_path}\nliterature review: {result.research_summary_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
