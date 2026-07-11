"""Vision backend abstraction — swappable multimodal model providers.

Every component that needs visual reasoning (the figure-analysis chat, the
panel detector) talks to a :class:`VisionBackend` instead of a concrete
model, so replacing Qwen2.5-VL with InternVL, GPT-4o, Claude, or a future
model requires changing only this module (or passing a different backend
instance).

This module also centralizes the provider plumbing shared across the
project: provider/model resolution from the environment, credential checks,
LangChain chat-model construction, and image/response (de)serialization.

Backends:
    * ``openai``    — ``gpt-4o`` by default (via ``langchain-openai``).
    * ``anthropic`` — ``claude-sonnet-5`` by default (via
      ``langchain-anthropic``). Anthropic retired ``claude-3-5-sonnet`` in
      October 2025; ``claude-sonnet-5`` is its vision-capable successor.
    * ``ollama``    — any *vision-capable* local model served by Ollama's
      OpenAI-compatible API (default ``qwen2.5vl``). Text-only models such
      as ``qwen3-coder`` cannot see images. Endpoint defaults to
      ``http://localhost:11434/v1`` (override with ``OLLAMA_BASE_URL``).
      No API key required.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from PIL import Image

logger = logging.getLogger(__name__)


class VLMProvider(str, Enum):
    """Supported vision-language model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


# Default multimodal model per provider (override with VLM_MODEL).
DEFAULT_MODELS: Dict[VLMProvider, str] = {
    VLMProvider.OPENAI: "gpt-4o",
    VLMProvider.ANTHROPIC: "claude-sonnet-5",
    VLMProvider.OLLAMA: "qwen2.5vl",
}

# Environment variable each provider needs for authentication.
# Local providers (Ollama) need no key and are absent from this map.
REQUIRED_ENV: Dict[VLMProvider, str] = {
    VLMProvider.OPENAI: "OPENAI_API_KEY",
    VLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
}

# Default endpoint of Ollama's OpenAI-compatible API.
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"

# USD per 1,000,000 tokens (input, output), keyed by a prefix of the model
# id. Local Ollama models are free. Used only for the cost report — update
# if provider prices change. Longest matching prefix wins.
MODEL_PRICES_PER_MTOK: Dict[str, Tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-opus": (5.00, 25.00),
    "claude-haiku": (1.00, 5.00),
}


def price_for_model(model: str) -> Tuple[float, float]:
    """Returns ``(input, output)`` USD per 1M tokens for a model id.

    Local models (served via Ollama) and unknown ids return ``(0, 0)``.
    """
    lowered = model.lower()
    best: Tuple[int, Tuple[float, float]] = (0, (0.0, 0.0))
    for prefix, price in MODEL_PRICES_PER_MTOK.items():
        if lowered.startswith(prefix) and len(prefix) > best[0]:
            best = (len(prefix), price)
    return best[1]


# ---------------------------------------------------------------------- #
# Provider plumbing (shared helpers)                                      #
# ---------------------------------------------------------------------- #


def resolve_provider(provider: Optional[Union[VLMProvider, str]]) -> VLMProvider:
    """Normalizes the provider argument, falling back to ``VLM_PROVIDER``.

    Args:
        provider: A :class:`VLMProvider`, its string value, or ``None``.

    Raises:
        ValueError: If the name is not a known provider.
    """
    if isinstance(provider, VLMProvider):
        return provider
    raw = (provider or os.getenv("VLM_PROVIDER") or "openai").strip().lower()
    try:
        return VLMProvider(raw)
    except ValueError:
        valid = ", ".join(p.value for p in VLMProvider)
        raise ValueError(
            f"Unknown VLM provider '{raw}'. Valid options: {valid}."
        ) from None


def resolve_model(provider: VLMProvider, model: Optional[str]) -> str:
    """Resolves the model name: argument > ``VLM_MODEL`` > provider default."""
    return (model or os.getenv("VLM_MODEL") or DEFAULT_MODELS[provider]).strip()


def ensure_credentials(provider: VLMProvider) -> None:
    """Raises early when the provider's API-key variable is missing.

    Raises:
        EnvironmentError: If a required key is not set. Local providers pass.
    """
    required_env = REQUIRED_ENV.get(provider)
    if required_env and not os.getenv(required_env):
        raise EnvironmentError(
            f"{required_env} is not set. Add it to your environment or .env "
            f"file to use the '{provider.value}' backend."
        )


def build_chat_model(
    provider: VLMProvider,
    model: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Instantiates the LangChain chat model for a provider.

    Imports are deferred so only the integration package for the chosen
    provider needs to be installed.

    Args:
        provider: The resolved provider.
        model: The model identifier.
        max_tokens: Response token budget.
        temperature: Sampling temperature — forwarded to OpenAI-compatible
            providers only; current Claude models reject non-default
            sampling parameters.
    """
    if provider in (VLMProvider.OPENAI, VLMProvider.OLLAMA):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - environment issue
            raise ImportError(
                f"langchain-openai is required for the '{provider.value}' "
                "backend. Install it with: pip install langchain-openai"
            ) from exc
        if provider is VLMProvider.OLLAMA:
            # Ollama serves an OpenAI-compatible API, so the same client is
            # reused with a local base URL. The API key is unused by Ollama
            # but must be non-empty for the OpenAI client. ``num_ctx`` raises
            # the context window (default 4096) so a two-image audit request
            # cannot overflow it; Ollama reads it from the request options.
            try:
                num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
            except ValueError:
                num_ctx = 8192
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=os.getenv("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL),
                api_key="ollama",
                extra_body={"options": {"num_ctx": num_ctx}},
            )
        return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:  # pragma: no cover - environment issue
        raise ImportError(
            "langchain-anthropic is required for the Anthropic backend. "
            "Install it with: pip install langchain-anthropic"
        ) from exc
    return ChatAnthropic(model=model, max_tokens=max_tokens)


def encode_image_png_b64(image: Image.Image) -> str:
    """Encodes a PIL image as base64 PNG, entirely in memory."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.standard_b64encode(buffer.getvalue()).decode("ascii")


def build_image_block(provider: VLMProvider, image_b64: str) -> Dict[str, Any]:
    """Builds the provider-appropriate multimodal image content block."""
    block: Dict[str, Any] = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
    }
    if provider is VLMProvider.OPENAI:
        # Request full-resolution tiling so GPT-4o inspects fine details
        # instead of a low-res thumbnail.
        block["image_url"]["detail"] = "high"
    return block


def response_text(response: BaseMessage) -> str:
    """Normalizes a LangChain response into plain text.

    Providers may return either a plain string or a list of typed content
    blocks; both shapes are handled.
    """
    content = response.content
    if isinstance(content, str):
        return content.strip()
    parts: List[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(part for part in parts if part).strip()


def response_usage(response: BaseMessage) -> Tuple[int, int]:
    """Extracts ``(input_tokens, output_tokens)`` from a LangChain response.

    Reads ``usage_metadata`` (newer langchain-core) and falls back to
    ``response_metadata['token_usage']``; returns ``(0, 0)`` when neither is
    present (e.g. some local runtimes do not report usage).
    """
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return (
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
        )
    meta = getattr(response, "response_metadata", None) or {}
    token_usage = meta.get("token_usage") or meta.get("usage") or {}
    if isinstance(token_usage, dict):
        return (
            int(token_usage.get("prompt_tokens", 0) or 0),
            int(token_usage.get("completion_tokens", 0) or 0),
        )
    return (0, 0)


# ---------------------------------------------------------------------- #
# Backend abstraction                                                     #
# ---------------------------------------------------------------------- #


class VisionBackend(ABC):
    """Abstract multimodal inference backend.

    A backend receives one image plus text prompts and returns the model's
    raw text response. It never crops, resizes, saves, or otherwise
    modifies images beyond transport encoding — image manipulation is the
    deterministic crop engine's job.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier, e.g. ``"ollama/qwen2.5vl"``."""

    def usage(self) -> Dict[str, float]:
        """Cumulative token usage and estimated USD cost for this backend.

        Base implementation reports zeros; real backends override.
        """
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    @abstractmethod
    def invoke(self, image: Image.Image, system_prompt: str, user_prompt: str) -> str:
        """Runs one multimodal request.

        Args:
            image: The image, sent exactly as given (callers control sizing;
                coordinate-returning callers must track any scaling they
                applied beforehand).
            system_prompt: The system/instruction prompt.
            user_prompt: The user-turn text accompanying the image.

        Returns:
            The model's raw text response.

        Raises:
            RuntimeError: When the request ultimately fails after retries.
        """

    def invoke_images(
        self,
        images: Sequence[Image.Image],
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Runs one request carrying several images (e.g. original + overlay).

        Backends that support multi-image inputs override this; the default
        signals lack of support so callers can degrade gracefully.

        Raises:
            NotImplementedError: When the backend is single-image only.
        """
        raise NotImplementedError(
            f"{self.name} does not support multi-image requests"
        )


class LangChainVisionBackend(VisionBackend):
    """Vision backend implemented over LangChain chat models.

    Covers OpenAI, Anthropic, and Ollama (OpenAI-compatible) providers with
    retry/backoff handling.

    Args:
        provider: Provider selector (``None`` reads ``VLM_PROVIDER``).
        model: Model identifier (``None`` reads ``VLM_MODEL`` / default).
        max_tokens: Response token budget.
        temperature: Sampling temperature (OpenAI-compatible providers only).
        request_retries: Retries after a failed request.
        retry_backoff_s: Base delay for exponential backoff.
    """

    def __init__(
        self,
        provider: Optional[Union[VLMProvider, str]] = None,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        request_retries: int = 2,
        retry_backoff_s: float = 2.0,
    ) -> None:
        self._provider = resolve_provider(provider)
        self._model_name = resolve_model(self._provider, model)
        ensure_credentials(self._provider)
        self._chat = build_chat_model(
            self._provider, self._model_name, max_tokens, temperature
        )
        self._request_retries = max(0, request_retries)
        self._retry_backoff_s = max(0.0, retry_backoff_s)
        self._price_in, self._price_out = price_for_model(self._model_name)
        self._calls = 0
        self._input_tokens = 0
        self._output_tokens = 0

    @property
    def name(self) -> str:
        return f"{self._provider.value}/{self._model_name}"

    def usage(self) -> Dict[str, float]:
        """Cumulative tokens and estimated USD cost since construction."""
        cost = (
            self._input_tokens * self._price_in
            + self._output_tokens * self._price_out
        ) / 1_000_000.0
        return {
            "calls": self._calls,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "cost_usd": cost,
        }

    @property
    def provider(self) -> VLMProvider:
        """The resolved provider."""
        return self._provider

    @property
    def model_name(self) -> str:
        """The resolved model identifier."""
        return self._model_name

    def invoke(self, image: Image.Image, system_prompt: str, user_prompt: str) -> str:
        return self.invoke_images([image], system_prompt, user_prompt)

    def invoke_images(
        self,
        images: Sequence[Image.Image],
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        content: List[Dict[str, Any]] = [
            build_image_block(self._provider, encode_image_png_b64(image))
            for image in images
        ]
        content.append({"type": "text", "text": user_prompt})
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
        ]
        last_error: Optional[Exception] = None
        attempts = self._request_retries + 1
        for attempt in range(attempts):
            try:
                response = self._chat.invoke(messages)
                self._calls += 1
                tokens_in, tokens_out = response_usage(response)
                self._input_tokens += tokens_in
                self._output_tokens += tokens_out
                return response_text(response)
            except Exception as exc:  # network / rate-limit / provider errors
                last_error = exc
                logger.warning(
                    "Vision request to %s failed (attempt %d/%d): %s",
                    self.name,
                    attempt + 1,
                    attempts,
                    exc,
                )
                if attempt < attempts - 1:
                    time.sleep(self._retry_backoff_s * (2**attempt))
        raise RuntimeError(
            f"Vision request to {self.name} failed after {attempts} attempt(s): "
            f"{last_error}"
        ) from last_error


def create_vision_backend(
    provider: Optional[Union[VLMProvider, str]] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> VisionBackend:
    """Factory for the default LangChain-based backend.

    Args:
        provider: Provider selector (``None`` reads the environment).
        model: Model identifier (``None`` reads the environment).
        **kwargs: Forwarded to :class:`LangChainVisionBackend`.
    """
    return LangChainVisionBackend(provider=provider, model=model, **kwargs)
