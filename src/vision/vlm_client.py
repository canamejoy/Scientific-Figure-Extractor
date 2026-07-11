"""Vision-language model client for targeted scientific figure analysis.

This module implements Stage 3 of the analysis pipeline: it packages the
isolated high-resolution figure crop together with its caption, the retrieved
textual context, and the user's analytical question into a single multimodal
request, and sends it to a VLM through the shared provider plumbing in
:mod:`src.vision.backend` (OpenAI / Anthropic / local Ollama).

The system prompt enforces a strict scientific-analysis protocol: pixel-level
inspection of axes, units, legends, and markers, plus mandatory
cross-referencing against the retrieved text to suppress hallucinations.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from PIL import Image

from src.vision.backend import (
    VLMProvider,
    build_chat_model,
    build_image_block,
    encode_image_png_b64,
    ensure_credentials,
    resolve_model,
    resolve_provider,
    response_text,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert scientific figure analyst embedded in a research-assistant
pipeline. You receive ONE isolated, high-resolution crop of a figure from a
scientific paper, its caption, and the passages of the paper that reference
it. Your job is rigorous, pixel-level visual analysis grounded in that text.

VISUAL INSPECTION PROTOCOL (perform in order, before answering):
1. AXES & SCALES — identify each axis title, its units, numeric range, tick
   spacing, and whether the scale is linear or logarithmic. If an axis is
   unlabeled or its units are missing, state that explicitly.
2. LEGEND & ENCODINGS — enumerate every data series with its exact color,
   marker shape, and line style BEFORE interpreting any of them. Never merge
   or confuse two visually similar series.
3. DATA READING — when asked for values, read them against the tick marks and
   gridlines. Report estimates with the precision the plot resolution
   actually supports (e.g. "≈0.83, plus or minus about 0.01") and never
   invent more significant digits than are visually justified.
4. STRUCTURE — for diagrams and architectures, trace the components and
   connections exactly as drawn (blocks, arrows, labels, groupings),
   following the layout's reading order.

CROSS-REFERENCING RULES:
- Compare every visual conclusion against the CAPTION and the PAPER CONTEXT
  passages provided with the question. When the text supports a claim, cite
  the passage (e.g. "context passage [2] states ...").
- If the text contradicts what you see, report BOTH readings and flag the
  discrepancy explicitly — never silently reconcile them.

ANTI-HALLUCINATION RULES:
- Describe only elements actually visible in the image. If something is
  illegible, occluded, cropped, or ambiguous, say so plainly.
- Never fabricate numeric values, series names, or units that appear in
  neither the image nor the provided text.
- If the question cannot be answered from the figure and context, state
  exactly what information is missing.

Answer in concise, well-structured Markdown.
"""


class FigureAnalyzer:
    """Sends multimodal figure-analysis requests to a VLM via LangChain.

    Args:
        provider: Backend to use (``VLMProvider`` value, its string name, or
            ``None`` to read the ``VLM_PROVIDER`` environment variable,
            defaulting to OpenAI).
        model: Model identifier; ``None`` reads ``VLM_MODEL`` and falls back
            to the provider default (``gpt-4o`` / ``claude-sonnet-5`` /
            ``qwen2.5vl``).
        max_tokens: Response token budget for the VLM.
        temperature: Sampling temperature — applied only to OpenAI-compatible
            backends. Recent Claude models reject non-default sampling
            parameters, so the Anthropic backend uses the API default.
        max_image_edge_px: Images larger than this on their longest edge are
            downscaled (LANCZOS) before base64 encoding, keeping payloads
            within provider limits while preserving legibility.
        request_retries: Number of retries after a failed request.
        retry_backoff_s: Base delay for exponential backoff between retries.

    Raises:
        ValueError: If the provider name is unknown.
        EnvironmentError: If the backend's API-key variable is not set.
        ImportError: If the required LangChain integration package is not
            installed.
    """

    def __init__(
        self,
        provider: Optional[Union[VLMProvider, str]] = None,
        model: Optional[str] = None,
        max_tokens: int = 1600,
        temperature: float = 0.0,
        max_image_edge_px: int = 2048,
        request_retries: int = 2,
        retry_backoff_s: float = 2.0,
    ) -> None:
        self._provider = resolve_provider(provider)
        self._model_name = resolve_model(self._provider, model)
        self._max_image_edge_px = max_image_edge_px
        self._request_retries = max(0, request_retries)
        self._retry_backoff_s = max(0.0, retry_backoff_s)

        ensure_credentials(self._provider)
        self._chat: BaseChatModel = build_chat_model(
            self._provider, self._model_name, max_tokens, temperature
        )
        logger.info(
            "FigureAnalyzer ready (provider=%s, model=%s)",
            self._provider.value,
            self._model_name,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    @property
    def provider(self) -> VLMProvider:
        """The active backend."""
        return self._provider

    @property
    def model_name(self) -> str:
        """The model identifier requests are sent to."""
        return self._model_name

    def analyze(
        self,
        image: Image.Image,
        caption: str,
        context_snippets: Sequence[str],
        user_query: str,
        figure_label: str = "Figure",
    ) -> str:
        """Runs one multimodal analysis request against the VLM.

        Args:
            image: The isolated high-resolution figure crop.
            caption: The figure's caption text.
            context_snippets: Retrieved passages from the paper that
                reference this figure, most relevant first.
            user_query: The user's analytical question.
            figure_label: Human-readable label (e.g. ``"Figure 3"``) used to
                anchor the prompt.

        Returns:
            The VLM's answer as Markdown text.

        Raises:
            ValueError: If the image is missing or the query is empty.
            RuntimeError: If every attempt (initial + retries) fails; the
                last underlying exception is chained.
        """
        if image is None:
            raise ValueError("An image is required for figure analysis.")
        if not user_query or not user_query.strip():
            raise ValueError("user_query must be a non-empty string.")

        messages = self._build_messages(
            image=image,
            caption=caption,
            context_snippets=context_snippets,
            user_query=user_query,
            figure_label=figure_label,
        )

        last_error: Optional[Exception] = None
        attempts = self._request_retries + 1
        for attempt in range(attempts):
            try:
                response = self._chat.invoke(messages)
                return response_text(response)
            except Exception as exc:  # network / rate-limit / provider errors
                last_error = exc
                logger.warning(
                    "VLM request failed (attempt %d/%d): %s",
                    attempt + 1,
                    attempts,
                    exc,
                )
                if attempt < attempts - 1:
                    time.sleep(self._retry_backoff_s * (2**attempt))

        raise RuntimeError(
            f"VLM analysis failed after {attempts} attempt(s): {last_error}"
        ) from last_error

    # ------------------------------------------------------------------ #
    # Payload construction                                                #
    # ------------------------------------------------------------------ #

    def _build_messages(
        self,
        image: Image.Image,
        caption: str,
        context_snippets: Sequence[str],
        user_query: str,
        figure_label: str,
    ) -> List[BaseMessage]:
        """Assembles the system + multimodal human message pair."""
        prepared = self._prepare_image(image)
        image_block: Dict[str, Any] = build_image_block(
            self._provider, encode_image_png_b64(prepared)
        )
        text_block: Dict[str, Any] = {
            "type": "text",
            "text": self._build_user_prompt(
                caption=caption,
                context_snippets=context_snippets,
                user_query=user_query,
                figure_label=figure_label,
            ),
        }
        return [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=[image_block, text_block]),
        ]

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """Downscales oversized crops for transport (LANCZOS).

        High-DPI crops of full-page figures can exceed provider payload
        limits without adding useful detail for the model.
        """
        longest = max(image.size)
        if longest <= self._max_image_edge_px:
            return image
        scale = self._max_image_edge_px / float(longest)
        new_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        logger.debug(
            "Downscaled figure image %sx%s -> %sx%s for transport",
            image.width,
            image.height,
            *new_size,
        )
        return image.resize(new_size, Image.Resampling.LANCZOS)

    @staticmethod
    def _build_user_prompt(
        caption: str,
        context_snippets: Sequence[str],
        user_query: str,
        figure_label: str,
    ) -> str:
        """Formats the textual half of the multimodal payload."""
        lines: List[str] = [
            f"FIGURE: {figure_label}",
            "",
            "CAPTION:",
            caption.strip() or "(no caption was extracted)",
            "",
            "PAPER CONTEXT (passages that reference this figure):",
        ]
        if context_snippets:
            for index, snippet in enumerate(context_snippets, start=1):
                lines.append(f"[{index}] {snippet.strip()}")
        else:
            lines.append("(no explicit in-text references were found)")
        lines += ["", "USER QUESTION:", user_query.strip()]
        return "\n".join(lines)
