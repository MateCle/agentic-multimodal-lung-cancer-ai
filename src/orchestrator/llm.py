"""
LLM client wrapper for the orchestrator.
Supports OpenAI and Anthropic APIs, plus a mock mode for testing.

Configuration via environment variables:
    LLM_PROVIDER=openai|anthropic|mock
    LLM_MODEL=gpt-4o|claude-sonnet-4-20250514|...
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...

Usage:
    from src.orchestrator.llm import get_llm_client

    llm = get_llm_client()
    response = llm.invoke("Your prompt here", system="You are a biomedical expert.")
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    content: str
    model: str
    usage: dict  # {"input_tokens": ..., "output_tokens": ...}


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def invoke(self, prompt: str, system: str = "") -> LLMResponse:
        """Send a prompt and return the response."""
        ...

    @abstractmethod
    def invoke_json(self, prompt: str, system: str = "") -> dict:
        """Send a prompt and parse the response as JSON."""
        ...


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client for testing without API access.
    Returns deterministic responses based on keywords in the prompt.
    """

    def __init__(self, model: str = "mock"):
        self.model = model
        self._call_count = 0

    def invoke(self, prompt: str, system: str = "") -> LLMResponse:
        self._call_count += 1
        logger.debug(f"[MockLLM] Call #{self._call_count}, prompt length={len(prompt)}")

        # Generate contextual mock responses based on system + prompt keywords.
        # Pre-Verifier system prompt contains "guidance"; check it first so we
        # don't misroute it to the miner branch (which also sees "mining rules").
        if "guidance" in system.lower():
            content = self._mock_pre_verifier_response(prompt)
        elif "mining rules" in prompt.lower() or "reconstruct" in prompt.lower():
            content = self._mock_miner_response(prompt)
        elif "verify" in prompt.lower() or "plausib" in prompt.lower():
            content = self._mock_verifier_response(prompt)
        elif "summarize" in prompt.lower() or "analyze" in prompt.lower():
            content = self._mock_agent_response(prompt)
        else:
            content = "Mock LLM response. No API key configured."

        return LLMResponse(
            content=content,
            model=self.model,
            usage={
                "input_tokens": len(prompt) // 4,
                "output_tokens": len(content) // 4,
            },
        )

    def invoke_json(self, prompt: str, system: str = "") -> dict:
        response = self.invoke(prompt, system)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"raw_response": response.content}

    def _mock_pre_verifier_response(self, _prompt: str) -> str:
        return json.dumps(
            {
                "guidance": {
                    "transcriptomics": (
                        "Prioritise clinical staging and smoking history for retrieval; "
                        "these are the strongest predictors of gene expression patterns "
                        "in LUAD/LUSC."
                    ),
                    "wsi": (
                        "Focus on cohort (LUAD vs LUSC) and pathological stage; these "
                        "drive the gross morphological differences captured by slide-level "
                        "embeddings."
                    ),
                    "methylation": (
                        "Weight smoking history and age heavily; tobacco-related CpG "
                        "methylation patterns are well-established and actionable for "
                        "retrieval."
                    ),
                    "clinical": (
                        "Use transcriptomic pathway activity and histological type as "
                        "primary retrieval signals."
                    ),
                }
            }
        )

    def _mock_miner_response(self, _prompt: str) -> str:
        return json.dumps(
            {
                "rules": {
                    "transcriptomics": (
                        "Use morphological heterogeneity from WSI as proxy for "
                        "EMT and proliferation pathway activity. Weight neighbors "
                        "with similar staging and histological subtype higher."
                    ),
                    "wsi": (
                        "Infer gross tissue morphology from staging, tumor grade, "
                        "and histological subtype in clinical data. Prioritize "
                        "neighbors with matching LUAD/LUSC classification."
                    ),
                    "methylation": (
                        "Use age-at-diagnosis and smoking history as epigenetic "
                        "clock proxies. Leverage pathway-level expression scores "
                        "to infer promoter methylation patterns."
                    ),
                    "clinical": (
                        "Infer demographic and staging features from gene expression "
                        "patterns and tissue morphology characteristics."
                    ),
                }
            }
        )

    def _mock_verifier_response(self, _prompt: str) -> str:
        return json.dumps(
            {
                "plausible": True,
                "confidence": 0.75,
                "reasoning": (
                    "The generated features fall within expected biological ranges. "
                    "The gene expression profile is consistent with the patient's "
                    "clinical staging and morphological features."
                ),
                "concerns": [],
            }
        )

    def _mock_agent_response(self, _prompt: str) -> str:
        return json.dumps(
            {
                "summary": (
                    "Feature analysis: 63-dimensional clinical vector with mixed "
                    "categorical and continuous features. Notable patterns include "
                    "elevated values in staging-related dimensions."
                ),
                "key_features": [0, 5, 12, 23, 41],
                "domain_context": (
                    "Clinical features suggest advanced-stage lung cancer with "
                    "smoking-related etiology."
                ),
            }
        )


class OpenAIClient(BaseLLMClient):
    """OpenAI API client (GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.3,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )
        try:
            from openai import OpenAI

            # base_url enables pointing to a local vLLM server
            # instead of api.openai.com. Set OPENAI_BASE_URL to
            # http://localhost:8000/v1 when using vLLM on AI-LAB.
            base_url = os.getenv("OPENAI_BASE_URL", None)
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        except ImportError:
            raise ImportError("pip install openai")

    def invoke(self, prompt: str, system: str = "") -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=512,
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.model,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
        )

    def invoke_json(self, prompt: str, system: str = "") -> dict:
        json_system = (
            system + "\nRespond ONLY with valid JSON. No markdown, no explanation."
        )
        response = self.invoke(prompt, json_system)
        try:
            cleaned = response.content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(
                f"Failed to parse LLM JSON response: {response.content[:200]}"
            )
            return {"raw_response": response.content}


class AnthropicClient(BaseLLMClient):
    """Anthropic API client (Claude Sonnet, etc.)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        temperature: float = 0.3,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.temperature = temperature
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
            )
        try:
            import anthropic

            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("pip install anthropic")

    def invoke(self, prompt: str, system: str = "") -> LLMResponse:
        kwargs = {
            "model": self.model,
            "max_tokens": 2000,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)

        content = "".join(
            block.text for block in response.content if block.type == "text"
        )

        return LLMResponse(
            content=content,
            model=self.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    def invoke_json(self, prompt: str, system: str = "") -> dict:
        json_system = (
            system + "\nRespond ONLY with valid JSON. No markdown, no explanation."
        )
        response = self.invoke(prompt, json_system)
        try:
            cleaned = response.content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(
                f"Failed to parse LLM JSON response: {response.content[:200]}"
            )
            return {"raw_response": response.content}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "mock": MockLLMClient,
}


def get_llm_client(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
) -> BaseLLMClient:
    """
    Create an LLM client based on environment configuration.

    Priority:
        1. Explicit arguments
        2. Environment variables (LLM_PROVIDER, LLM_MODEL)
        3. Falls back to mock if no API key is found

    Returns:
        Configured LLM client ready for use.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "").lower()
    model = model or os.getenv("LLM_MODEL", "")

    # Auto-detect provider from available API keys
    if not provider:
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
            model = model or "gpt-4o"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
            model = model or "claude-sonnet-4-20250514"
        else:
            provider = "mock"
            model = model or "mock"
            logger.warning(
                "No LLM API key found. Using mock client. "
                "Set OPENAI_API_KEY or ANTHROPIC_API_KEY for real LLM calls."
            )

    client_cls = _PROVIDERS.get(provider)
    if client_cls is None:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Choose from: {list(_PROVIDERS.keys())}"
        )

    logger.info(f"LLM client: provider={provider}, model={model}")
    return client_cls(model=model, temperature=temperature)
