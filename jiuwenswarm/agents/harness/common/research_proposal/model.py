"""Reasoning-model seam for the research-proposal pipeline.

openjiuwen's autoResearch agents can only reach a model through an API client
built from ``API_KEY``/``API_BASE`` environment variables. This module keeps
the pipeline independent of how the reasoning is produced:

* ``OpenJiuwenReasoningModel`` wraps any openjiuwen ``Model`` (the same object
  ``ArtifactEngineRequest.model`` carries), so production runs reuse the
  configured provider.
* ``FileExchangeReasoningModel`` writes each stage's prompt to disk and reads
  the answer from a sibling file. An operator, or another agent such as a
  coding assistant, can then act as the model without any API key; the
  pipeline is resumable, so it simply runs again once the answer exists.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class AwaitingModelResponse(RuntimeError):
    """Raised by ``FileExchangeReasoningModel`` until the answer file exists."""

    def __init__(self, stage: str, prompt_path: Path, response_path: Path) -> None:
        super().__init__(
            f"stage {stage!r} is waiting for a model response: read {prompt_path} "
            f"and write the JSON answer to {response_path}, then run again"
        )
        self.stage = stage
        self.prompt_path = prompt_path
        self.response_path = response_path


@runtime_checkable
class ReasoningModel(Protocol):
    async def complete(self, *, stage: str, system: str, prompt: str) -> str:
        """Return the model's raw text answer for one stage prompt."""


class OpenJiuwenReasoningModel:
    """Adapter over ``openjiuwen.core.foundation.llm.model.Model.invoke``."""

    def __init__(
        self,
        model: Any,
        *,
        temperature: float | None = 0.2,
        max_tokens: int | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def complete(self, *, stage: str, system: str, prompt: str) -> str:
        message = await self._model.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        content = getattr(message, "content", message)
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content or "")


class FileExchangeReasoningModel:
    """Model that answers through files in ``exchange_dir``.

    For ``stage="directions"`` the prompt goes to ``directions.prompt.md`` and
    the answer is read from ``directions.response.json``. The pipeline names
    repair attempts ``directions.repair1`` and so on, so each gets its own pair.
    """

    def __init__(self, exchange_dir: str | Path) -> None:
        self.exchange_dir = Path(exchange_dir)

    def paths(self, stage: str) -> tuple[Path, Path]:
        return (
            self.exchange_dir / f"{stage}.prompt.md",
            self.exchange_dir / f"{stage}.response.json",
        )

    async def complete(self, *, stage: str, system: str, prompt: str) -> str:
        self.exchange_dir.mkdir(parents=True, exist_ok=True)
        prompt_path, response_path = self.paths(stage)
        body = f"# System\n\n{system}\n\n# Task\n\n{prompt}\n"
        if not prompt_path.exists() or prompt_path.read_text(encoding="utf-8") != body:
            prompt_path.write_text(body, encoding="utf-8")
        if not response_path.exists():
            raise AwaitingModelResponse(stage, prompt_path, response_path)
        return response_path.read_text(encoding="utf-8")


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in a model answer.

    Accepts a bare object, a fenced ```json block, or an object surrounded by
    prose. Raises ``ValueError`` when no object can be decoded.
    """
    candidates = [match.strip() for match in _FENCE_RE.findall(text or "")]
    candidates.append((text or "").strip())
    decoder = json.JSONDecoder()
    for candidate in candidates:
        start = candidate.find("{")
        while start != -1:
            try:
                value, _ = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                start = candidate.find("{", start + 1)
                continue
            if isinstance(value, dict):
                return value
            start = candidate.find("{", start + 1)
    raise ValueError("model answer contains no JSON object")


__all__ = [
    "AwaitingModelResponse",
    "FileExchangeReasoningModel",
    "OpenJiuwenReasoningModel",
    "ReasoningModel",
    "extract_json_object",
]
