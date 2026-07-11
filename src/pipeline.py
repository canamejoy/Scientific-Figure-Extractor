"""Central pipeline orchestration for the AI Scientific Figure Analyzer.

The :class:`Orchestrator` wires the three backend stages together:

1. :class:`~src.parsing.pdf_parser.ScientificPDFParser` — layout parsing and
   high-resolution figure cropping.
2. :class:`~src.rag.vector_db.MultiModalContextIndex` — binding each figure
   to its caption and the passages that reference it.
3. :class:`~src.vision.vlm_client.FigureAnalyzer` — targeted multimodal VLM
   querying.

The frontend (``src/api/app.py``) only ever talks to this class.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from src.extraction.framework import ExtractionConfig, FigureExtractionFramework
from src.models import ExtractedFigure, FigureContext, ParsedDocument
from src.parsing.pdf_parser import ScientificPDFParser
from src.rag.vector_db import MultiModalContextIndex
from src.vision.vlm_client import FigureAnalyzer

logger = logging.getLogger(__name__)


class Orchestrator:
    """End-to-end pipeline facade: process a PDF, then query its figures.

    Args:
        parser: Custom PDF parser; a default 300-DPI parser is created when
            omitted.
        index: Custom context index; a default in-memory index is created
            when omitted.
        analyzer: Custom VLM client. When omitted, the analyzer is created
            *lazily* on the first query so parsing and indexing work even
            before API credentials are configured.
        framework: Custom extraction framework. When omitted one is built
            around the parser; the ``PANEL_CROPPING`` environment variable
            (``deterministic`` default, ``markers-only``, or ``vlm-assisted``)
            selects the panel-crop strategy for document processing.
        context_passages_per_query: How many retrieved reference passages
            accompany each VLM query.
    """

    def __init__(
        self,
        parser: Optional[ScientificPDFParser] = None,
        index: Optional[MultiModalContextIndex] = None,
        analyzer: Optional[FigureAnalyzer] = None,
        framework: Optional[FigureExtractionFramework] = None,
        context_passages_per_query: int = 4,
    ) -> None:
        self._parser = parser or ScientificPDFParser()
        self._index = index or MultiModalContextIndex()
        self._analyzer = analyzer
        self._document: Optional[ParsedDocument] = None
        self._top_k = max(1, context_passages_per_query)
        if framework is None:
            cropping = (os.getenv("PANEL_CROPPING") or "deterministic").strip().lower()
            if cropping not in ("deterministic", "markers-only", "vlm-assisted"):
                cropping = "deterministic"
            framework = FigureExtractionFramework(
                config=ExtractionConfig(panel_cropping=cropping),
                parser=self._parser,
            )
        self._framework = framework

    # ------------------------------------------------------------------ #
    # Pipeline entry points                                               #
    # ------------------------------------------------------------------ #

    def process_document(self, pdf_path: str) -> None:
        """Parses a PDF, crops its figures, and builds the multimodal index.

        Replaces any previously processed document.

        Args:
            pdf_path: Filesystem path to the PDF.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If the PDF cannot be parsed at all.
        """
        logger.info("Processing document: %s", pdf_path)
        document = self._framework.extract(pdf_path)
        self._index.build(document)
        self._document = document
        logger.info(
            "Document ready — %d page(s), %d figure(s) indexed",
            len(document.pages),
            len(self._index),
        )

    def query_figure(self, figure_id: str, user_query: str) -> str:
        """Answers an analytical question about one extracted figure.

        Fetches the figure's multimodal bundle (high-res crop, caption, and
        the reference passages most relevant to the query) and forwards it to
        the VLM.

        Args:
            figure_id: The figure identifier (e.g. ``"figure-3"``).
            user_query: The user's analytical question.

        Returns:
            The VLM's Markdown answer.

        Raises:
            RuntimeError: If no document has been processed yet, or the VLM
                request ultimately fails.
            KeyError: If ``figure_id`` is not in the index.
            ValueError: If ``user_query`` is empty.
        """
        if self._document is None:
            raise RuntimeError(
                "No document has been processed yet — call process_document() first."
            )
        if not user_query or not user_query.strip():
            raise ValueError("user_query must be a non-empty string.")

        context = self._index.retrieve(figure_id, user_query, top_k=self._top_k)
        analyzer = self._get_analyzer()
        logger.info(
            "Querying %s (%s/%s) with %d context passage(s)",
            figure_id,
            analyzer.provider.value,
            analyzer.model_name,
            len(context.references),
        )
        return analyzer.analyze(
            image=context.figure.image,
            caption=context.figure.caption,
            context_snippets=context.references,
            user_query=user_query,
            figure_label=context.figure.label,
        )

    # ------------------------------------------------------------------ #
    # Introspection helpers (used by the UI)                              #
    # ------------------------------------------------------------------ #

    @property
    def figures(self) -> List[ExtractedFigure]:
        """Every indexed figure in document order (empty before processing)."""
        return [entry.figure for entry in self._index.list_figures()]

    @property
    def document(self) -> Optional[ParsedDocument]:
        """The currently processed document, if any."""
        return self._document

    def get_context(self, figure_id: str) -> FigureContext:
        """Returns a figure's stored context bundle without re-ranking.

        Args:
            figure_id: The figure identifier.

        Raises:
            KeyError: If the figure is not in the index.
        """
        return self._index.get(figure_id)

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _get_analyzer(self) -> FigureAnalyzer:
        """Creates the VLM client on first use (requires API credentials)."""
        if self._analyzer is None:
            self._analyzer = FigureAnalyzer()
        return self._analyzer
