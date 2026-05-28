"""
Base class for modality-specific understanding agents.

Each agent analyzes a single modality (clinical, transcriptomics, wsi,
methylation) and produces a textual summary that the Miner consumes for
cross-modal reasoning. Agents are not LangGraph nodes — they are Python
classes invoked from within the Miner node, optionally in parallel.

Following AFM2's understanding-agent pattern: each agent owns a
domain-specific prompt and translates raw features into biologically
meaningful narrative.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from src.orchestrator.llm import BaseLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent output schema
# ---------------------------------------------------------------------------


@dataclass
class AgentSummary:
    """Structured output of a modality agent."""

    modality: str
    summary: str  # 3-5 lines, natural language
    key_features: list[str]  # named biological features highlighted
    confidence: str  # "high" | "medium" | "low"
    concerns: list[str] = field(default_factory=list)
    raw_response: dict | None = None  # for debugging / execution log

    def to_prompt_block(self) -> str:
        """Render this summary for inclusion in the Miner's cross-modal prompt."""
        lines = [f"[{self.modality.upper()} — confidence: {self.confidence}]"]
        lines.append(self.summary.strip())
        if self.key_features:
            lines.append(f"Key features: {', '.join(self.key_features[:8])}")
        if self.concerns:
            lines.append(f"Concerns: {'; '.join(self.concerns[:3])}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class ModalityAgent(ABC):
    """
    Abstract base for modality understanding agents.

    Subclasses must define:
        modality:       one of MODALITY_KEYS
        SYSTEM_PROMPT:  class-level system prompt for the LLM
        _build_prompt: converts feature stats into a domain prompt

    Public API:
        analyze(features) -> AgentSummary
    """

    modality: str = ""
    SYSTEM_PROMPT: str = ""

    def __init__(self, llm: BaseLLMClient, metadata: dict | None = None):
        if not self.modality:
            raise ValueError(
                f"{type(self).__name__} must set the `modality` class attribute."
            )
        self.llm = llm
        self.metadata = metadata or {}
        self.columns: list[str] = self.metadata.get(f"{self.modality}_columns", [])

    # -- to be implemented by each subclass -------------------------------------

    @abstractmethod
    def _build_prompt(self, features: np.ndarray) -> str:
        """Turn the raw feature vector into a domain-specific user prompt."""
        ...

    # -- shared logic -----------------------------------------------------------

    def analyze(self, features: np.ndarray) -> AgentSummary:
        """
        Run the agent on a single patient's features for this modality.

        Returns an AgentSummary, falling back to a deterministic stub
        on LLM failure so the Miner always sees a complete summary set.
        """
        arr = np.asarray(features, dtype=np.float32).flatten()
        prompt = self._build_prompt(arr)

        try:
            response = self.llm.invoke_json(prompt, system=self.SYSTEM_PROMPT)
            return self._parse_response(response, arr)
        except Exception as e:
            logger.warning(f"[{self.modality} agent] LLM call failed: {e}. Using stub.")
            return self._stub_summary(arr, reason=f"LLM error: {e}")

    def _parse_response(self, response: dict, arr: np.ndarray) -> AgentSummary:
        """Validate and coerce the LLM JSON into an AgentSummary."""
        summary = str(response.get("summary", "")).strip()
        if not summary:
            return self._stub_summary(arr, reason="empty LLM summary")

        key_features = response.get("key_features") or []
        if not isinstance(key_features, list):
            key_features = [str(key_features)]
        key_features = [str(f) for f in key_features][:10]

        confidence = str(response.get("confidence", "medium")).lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"

        concerns = response.get("concerns") or []
        if not isinstance(concerns, list):
            concerns = [str(concerns)]
        concerns = [str(c) for c in concerns][:5]

        return AgentSummary(
            modality=self.modality,
            summary=summary,
            key_features=key_features,
            confidence=confidence,
            concerns=concerns,
            raw_response=response,
        )

    def _stub_summary(
        self, arr: np.ndarray, reason: str = "LLM unavailable"
    ) -> AgentSummary:
        """Deterministic fallback when the LLM call cannot be parsed."""
        if arr.size:
            stats_line = (
                f"Feature vector with {arr.size} dims, "
                f"{int(np.count_nonzero(arr))} non-zero, "
                f"range [{arr.min():.3f}, {arr.max():.3f}]."
            )
        else:
            stats_line = "No feature statistics available."
        return AgentSummary(
            modality=self.modality,
            summary=f"{stats_line} (Stub fallback: {reason}.)",
            key_features=[],
            confidence="low",
            concerns=[reason],
            raw_response=None,
        )

    # -- helpers for subclasses -------------------------------------------------

    def _topk_named(self, arr: np.ndarray, k: int = 8) -> list[tuple[str, float]]:
        """Return the top-k features by |value| with names from metadata."""
        if not self.columns or arr.size == 0:
            return []
        n = min(len(self.columns), arr.size)
        idx = np.argsort(np.abs(arr[:n]))[::-1][:k]
        return [(self.columns[i], float(arr[i])) for i in idx]

    def _active_named(
        self, arr: np.ndarray, max_n: int = 30
    ) -> list[tuple[str, float]]:
        """Return non-zero features with names (suited to one-hot clinical data)."""
        if not self.columns or arr.size == 0:
            return []
        n = min(len(self.columns), arr.size)
        return [(self.columns[i], float(arr[i])) for i in range(n) if arr[i] != 0][
            :max_n
        ]
