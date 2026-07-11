"""Lightweight, in-memory multimodal context indexer.

For every extracted figure the index stores the cropped high-resolution
image, its caption, and its *local text context*: the passages of the paper
body that explicitly reference the figure ("as shown in Fig. 3", "Figure 3
compares ..."). At query time those passages are re-ranked against the user's
question with a simple lexical-overlap score.

The index is deliberately dependency-free — no embedding model, no external
vector store — so a whole document fits in process memory and the pipeline
works offline until the final VLM call. Reference passages are found with a
regex over the raw page text and expanded to sentence boundaries, which is
robust to the hard line breaks pdfplumber preserves in extracted text.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

from src.models import ExtractedFigure, FigureContext, ParsedDocument

logger = logging.getLogger(__name__)

_STOPWORDS: FrozenSet[str] = frozenset(
    {
        "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "is",
        "are", "was", "were", "this", "that", "these", "those", "it", "as",
        "at", "by", "with", "from", "be", "we", "our", "their", "its", "can",
        "what", "which", "how", "does", "do", "not",
    }
)


def _normalise(text: str) -> str:
    """Lower-cases and collapses every whitespace run into a single space."""
    return " ".join(text.lower().split())


def _tokenise(text: str) -> Set[str]:
    """Splits text into a set of lowercase alphanumeric tokens minus stopwords."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


class MultiModalContextIndex:
    """In-memory index binding each figure to its caption and text context.

    Args:
        context_chars: How far (in characters) a reference passage may extend
            on each side of a figure mention before being snapped to sentence
            boundaries.
        max_references: Maximum number of reference passages stored per
            figure.

    Raises:
        ValueError: If ``context_chars`` is unreasonably small.
    """

    def __init__(self, context_chars: int = 450, max_references: int = 8) -> None:
        if context_chars < 50:
            raise ValueError(f"context_chars must be >= 50, got {context_chars}.")
        if max_references < 1:
            raise ValueError(f"max_references must be >= 1, got {max_references}.")
        self._context_chars = context_chars
        self._max_references = max_references
        self._entries: Dict[str, FigureContext] = {}

    # ------------------------------------------------------------------ #
    # Building                                                            #
    # ------------------------------------------------------------------ #

    def build(self, document: ParsedDocument) -> None:
        """Indexes every figure of ``document``, replacing previous contents.

        For each figure this automatically extracts the paragraphs of the
        paper that explicitly reference it (excluding the caption itself).

        Args:
            document: The parsed document produced by the PDF parser.
        """
        self._entries.clear()
        for figure in document.figures:
            references = self._find_reference_passages(document, figure)
            self.add_figure(figure, references)
        logger.info(
            "Indexed %d figure(s) from '%s'", len(self._entries), document.source_path
        )

    def add_figure(self, figure: ExtractedFigure, references: Sequence[str]) -> None:
        """Adds (or replaces) a single figure entry.

        Args:
            figure: The extracted figure to index.
            references: Text passages that reference this figure.
        """
        if figure.figure_id in self._entries:
            logger.warning("Overwriting existing index entry for %s", figure.figure_id)
        self._entries[figure.figure_id] = FigureContext(
            figure=figure, references=list(references)
        )

    # ------------------------------------------------------------------ #
    # Lookup & retrieval                                                  #
    # ------------------------------------------------------------------ #

    def get(self, figure_id: str) -> FigureContext:
        """Returns the stored context bundle for a figure.

        Args:
            figure_id: The figure identifier, e.g. ``"figure-3"``.

        Raises:
            KeyError: If the figure is not in the index; the message lists
                every available id to make UI errors actionable.
        """
        try:
            return self._entries[figure_id]
        except KeyError:
            available = ", ".join(sorted(self._entries)) or "<empty index>"
            raise KeyError(
                f"Unknown figure id '{figure_id}'. Available: {available}"
            ) from None

    def retrieve(self, figure_id: str, query: str, top_k: int = 4) -> FigureContext:
        """Returns a figure's context with references re-ranked for a query.

        Passages are scored by lexical overlap with the query and the best
        ``top_k`` are kept, preserving document order among ties, so the VLM
        receives the most relevant textual grounding first.

        Args:
            figure_id: The figure identifier.
            query: The user's analytical question.
            top_k: Maximum number of passages to return.

        Returns:
            A new :class:`~src.models.FigureContext` with ranked references.

        Raises:
            KeyError: If the figure is not in the index.
        """
        entry = self.get(figure_id)
        if not entry.references or top_k <= 0:
            return entry
        ranked = self._rank_references(entry.references, query)[:top_k]
        return FigureContext(figure=entry.figure, references=ranked)

    def list_figures(self) -> List[FigureContext]:
        """Returns every indexed context bundle in insertion (document) order."""
        return list(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, figure_id: object) -> bool:
        return figure_id in self._entries

    # ------------------------------------------------------------------ #
    # Reference extraction internals                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _mention_pattern(number: str) -> "re.Pattern[str]":
        """Builds a regex matching in-text mentions of a figure number.

        Handles the common citation forms: "Figure 3", "Fig. 3", "figs. 2
        and 3", "Figures 1, 2 and 3". The optional leading list group lets
        the target number appear at any position within an enumeration.
        Panel numbers such as ``"2e"`` additionally match the parenthesized
        form authors actually write: "Fig. 2(e)" as well as "Fig. 2e".

        Args:
            number: The figure number as extracted from the caption
                (``"3"``, ``"2a"``, ``"4.1"``...).
        """
        parsed = re.fullmatch(r"(\d{1,3}(?:\.\d{1,2})?)([a-z])?", number.lower())
        if parsed and parsed.group(2):
            target = rf"{re.escape(parsed.group(1))}\s*\(?\s*{parsed.group(2)}\)?"
        else:
            target = re.escape(number)
        # A negative lookahead ends the match instead of \b because the
        # target may end on ")" (a non-word character, where \b never holds).
        return re.compile(
            rf"\bfig(?:ure)?s?\.?\s*"
            rf"(?:\d{{1,3}}(?:\.\d{{1,2}})?[a-z]?\s*(?:,|and|&)\s*)*"
            rf"{target}(?![a-zA-Z0-9])",
            re.IGNORECASE,
        )

    def _find_reference_passages(
        self, document: ParsedDocument, figure: ExtractedFigure
    ) -> List[str]:
        """Extracts the passages of the paper that reference a figure.

        The figure's own caption is excluded (it is stored separately on the
        figure), and overlapping windows around nearby mentions are
        de-duplicated. For subfigure panels (e.g. ``"2e"``) with no
        panel-specific mentions, the search falls back to mentions of the
        whole figure (``"2"``) so panels still receive textual grounding.

        Args:
            document: The parsed document to scan.
            figure: The figure whose mentions should be collected.

        Returns:
            Up to ``max_references`` whitespace-normalized passages.
        """
        caption_key = _normalise(figure.caption)[:60]
        passages = self._collect_passages(
            document, self._mention_pattern(figure.number), caption_key
        )

        if not passages:
            parsed = re.fullmatch(
                r"(\d{1,3}(?:\.\d{1,2})?)([a-z])", figure.number.lower()
            )
            if parsed:
                passages = self._collect_passages(
                    document, self._mention_pattern(parsed.group(1)), caption_key
                )

        if not passages:
            logger.debug("No in-text references found for %s", figure.figure_id)
        return passages

    def _collect_passages(
        self,
        document: ParsedDocument,
        pattern: "re.Pattern[str]",
        caption_key: str,
    ) -> List[str]:
        """Scans every page for mentions matching ``pattern``.

        Args:
            document: The parsed document to scan.
            pattern: The mention regex from :meth:`_mention_pattern`.
            caption_key: Normalized caption prefix used to drop the caption's
                own occurrence.

        Returns:
            Up to ``max_references`` whitespace-normalized passages.
        """
        passages: List[str] = []
        seen: List[str] = []

        for page in document.pages:
            text = page.text
            if not text:
                continue
            for match in pattern.finditer(text):
                # Skip the caption anchor itself: captions start on their own
                # line and the number is immediately followed by a separator.
                at_line_start = match.start() == 0 or text[match.start() - 1] == "\n"
                rest = text[match.end() : match.end() + 2].lstrip()
                if at_line_start and rest[:1] in {":", ".", "-", "–", "—", "|"}:
                    continue
                snippet = self._expand_to_sentences(text, match.start(), match.end())
                norm = _normalise(snippet)
                if not norm:
                    continue
                if caption_key and caption_key in norm:
                    continue  # this mention *is* the caption
                if any(norm in existing or existing in norm for existing in seen):
                    continue  # overlapping window around a nearby mention
                seen.append(norm)
                passages.append(snippet)
                if len(passages) >= self._max_references:
                    return passages
        return passages

    def _expand_to_sentences(self, text: str, start: int, end: int) -> str:
        """Expands a mention span to the surrounding sentence boundaries.

        A window of ``context_chars`` characters is opened on each side of
        the mention, then snapped inward to the closest sentence boundary so
        the passage starts and ends on whole sentences. Whitespace (including
        the hard line breaks of extracted PDF text) is normalized.

        Args:
            text: The full page text.
            start: Start offset of the regex mention match.
            end: End offset of the regex mention match.

        Returns:
            The whitespace-normalized passage containing the mention.
        """
        window_start = max(0, start - self._context_chars)
        window_end = min(len(text), end + self._context_chars)

        # Snap the left edge forward to just after the last sentence boundary
        # that precedes the mention inside the window.
        chunk_start = window_start
        for boundary in (". ", ".\n", "! ", "? ", "\n\n"):
            idx = text.rfind(boundary, window_start, start)
            if idx != -1:
                chunk_start = max(chunk_start, idx + len(boundary))

        # Snap the right edge backward to the first sentence end that follows
        # the mention inside the window.
        chunk_end = window_end
        for boundary in (". ", ".\n", "! ", "? "):
            idx = text.find(boundary, end, window_end)
            if idx != -1:
                chunk_end = min(chunk_end, idx + 1)

        return " ".join(text[chunk_start:chunk_end].split())

    @staticmethod
    def _rank_references(references: Sequence[str], query: str) -> List[str]:
        """Orders passages by lexical overlap with the query (best first).

        Ties keep their original document order, so with an empty or
        stopword-only query the passages come back unchanged.
        """
        query_tokens = _tokenise(query)
        if not query_tokens:
            return list(references)
        scored: List[Tuple[int, int, str]] = []
        for position, passage in enumerate(references):
            overlap = len(query_tokens & _tokenise(passage))
            scored.append((-overlap, position, passage))
        scored.sort()
        return [passage for _, _, passage in scored]
