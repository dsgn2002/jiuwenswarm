"""Research-proposal pipeline: problem framing → literature review → directions → proposal.

Fills the front half of the auto-researcher loop that openjiuwen's autoResearch
manager lacks (it starts at a topic survey and jumps to experiment design).
See ``README.md`` in this package for the limitations analysis and usage.
"""

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
    ReasoningModel,
)
from jiuwenswarm.agents.harness.common.research_proposal.pipeline import (
    EmptyCorpus,
    PipelineResult,
    ResearchProposalPipeline,
    StageFailed,
)

__all__ = [
    "ArxivBackend",
    "AwaitingModelResponse",
    "Downloader",
    "EmptyCorpus",
    "FileExchangeReasoningModel",
    "LiteratureRetriever",
    "OpenJiuwenReasoningModel",
    "OpenJiuwenWebBackend",
    "PipelineResult",
    "ReasoningModel",
    "ResearchProposalPipeline",
    "StageFailed",
]
