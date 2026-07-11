"""Scientific PDF layout parsing and high-resolution figure extraction.

This module implements Stage 1 of the pipeline. It walks every page of a PDF
with ``pdfplumber``, locates figure captions with an anchor-based heuristic
("Figure 3:", "Fig. 2.", "FIGURE 4 —"), maps each caption to the graphical
region it describes, and renders that region into a high-resolution
``PIL.Image`` so that axis labels, tick marks, and legends stay legible for
downstream vision-language analysis.

Region detection strategy:

1. A vertical search band is opened between the caption and the nearest
   paragraph-like text line (short lines — axis tick labels — are ignored),
   constrained to the caption's column.
2. The figure region is the **union** of every raster image and vector
   drawing primitive inside the band. Multi-panel figures are often several
   separate embedded images and/or many vector primitives, so no single
   "best" object is representative.
3. When the band contains no graphics at all, a caption-relative fallback
   window is used.

Multi-panel figures are additionally split into per-panel subfigures
(``Figure 2(a)``, ``Figure 2(b)``, ...) by combining two signals:

* the panel marker texts ("(a)", "(b)", ...) rendered inside the figure, and
* a whitespace-gutter decomposition of the figure's actual content (words +
  graphics) into rows and cells,

so each panel crop is tight around its own content instead of relying on the
marker positions alone.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import pdfplumber
from PIL import Image

from src.models import (
    BoundingBox,
    DetectionMethod,
    ExtractedFigure,
    LayoutRegion,
    PageText,
    ParsedDocument,
    PixelBox,
)
from src.utils.captions import split_caption

logger = logging.getLogger(__name__)

# A text line as returned by pdfplumber's ``page.extract_text_lines()``.
_LineDict = Dict[str, Any]

# Whitespace gap (in PDF points) that separates two panels of a multi-panel
# figure. Intra-panel gaps (tick labels, stacked subplots sharing an axis)
# are typically below this; inter-panel gutters are typically above it.
_PANEL_GUTTER_PT: float = 8.0

# Caption anchor: "Figure 3:", "Fig. 12.", "FIGURE 2 -", "Figure 4a —",
# "Figure 3.2: ..." (book-style dotted numbering).
#
# The separator after the number is *required* so in-prose sentences such as
# "Figure 3 shows the results" are not mistaken for captions. Captions whose
# text starts on the following line (a bare "Figure 3" line) are not matched;
# in practice this layout is rare in scientific papers.
_CAPTION_ANCHOR: "re.Pattern[str]" = re.compile(
    r"^\s*(?:fig(?:ure)?\.?)\s*"
    r"(?P<number>\d{1,3}(?:\.\d{1,2})?[a-z]?)"
    r"\s*(?P<sep>[:.\-–—|])\s*\S",
    re.IGNORECASE,
)

# Panel markers rendered inside a figure: "(a)", "(b)", "a)", "(c". A bare
# letter with no parenthesis is NOT accepted — single letters also occur as
# axis labels and legend entries.
_PANEL_WORD: "re.Pattern[str]" = re.compile(
    r"^\(([a-h])\)?[.:]?$|^\(?([a-h])\)[.:]?$"
)

# Figure number inside caption prose (used by the layout-model path, where
# the caption text may start anywhere): "FIG. 3.", "Figure 4:", "Fig 2a".
_CAPTION_NUMBER: "re.Pattern[str]" = re.compile(
    r"\bfig(?:ure)?\.?\s*(?P<number>\d{1,3}(?:\.\d{1,2})?[a-z]?)",
    re.IGNORECASE,
)


def _panel_letter(text: str) -> Optional[str]:
    """Returns the panel letter if ``text`` is (or starts as) a panel marker.

    Accepts exact marker tokens ("(a)", "a)", "(a") and longer words that
    *begin* with a complete parenthesized marker — PDFs sometimes fuse the
    marker with adjacent panel text into one word ("(a)Total").
    """
    token = text.strip().lower()
    if not token:
        return None
    if len(token) <= 4:
        match = _PANEL_WORD.match(token)
        if match:
            return match.group(1) or match.group(2)
        return None
    prefixed = re.match(r"^\(([a-h])\)", token)
    return prefixed.group(1) if prefixed else None


class _CaptionMatch(NamedTuple):
    """A caption anchor located on a page, with its full paragraph text."""

    number: str
    label: str
    text: str
    bbox: BoundingBox


def _line_bbox(line: _LineDict) -> BoundingBox:
    """Builds a :class:`BoundingBox` from a pdfplumber text-line dict.

    Coordinates are cast to ``float`` because pdfplumber occasionally yields
    ``Decimal`` values depending on the underlying pdfminer objects.
    """
    return BoundingBox(
        x0=float(line["x0"]),
        top=float(line["top"]),
        x1=float(line["x1"]),
        bottom=float(line["bottom"]),
    )


class ScientificPDFParser:
    """Parses scientific PDFs into per-page text and high-resolution figure crops.

    The parser is stateless between :meth:`parse` calls and safe to reuse for
    multiple documents.

    Args:
        resolution: Target render resolution in DPI (minimum 72). The default
            of 300 DPI keeps small axis annotations legible for the VLM.
        max_render_edge_px: Safety cap on the longest rendered edge in pixels.
            The effective DPI is reduced for very large regions so that a
            single crop can never allocate an excessively large bitmap.
        bbox_padding_pt: Padding (in PDF points) added around detected regions
            so anti-aliased strokes at the border are not clipped.
        min_figure_width_pt: Minimum plausible figure width; smaller
            detections (icons, decorative rules) are rejected.
        min_figure_height_pt: Minimum plausible figure height.
        fallback_region_height_pt: Height of the synthetic caption-relative
            region used when no graphic object can be associated with a
            caption.
        use_layout_model: Whether to use the page-layout detection model
            (DocLayout-YOLO) as the *primary* figure locator. ``None`` reads
            the ``LAYOUT_DETECTION`` environment variable (on unless set to
            ``off``/``false``/``0``). When the model or its dependencies are
            unavailable, the parser silently falls back to caption-anchor
            heuristics.
        layout_detector: Injected detector instance (mainly for tests);
            built lazily from the environment when omitted.
        layout_detect_dpi: Resolution pages are rendered at for layout
            detection (final crops still render at ``resolution``).
        layout_min_confidence: Minimum confidence for a detected figure
            region to be used.

    Raises:
        ValueError: If ``resolution`` is below 72 DPI.
    """

    DEFAULT_RESOLUTION: int = 300

    def __init__(
        self,
        resolution: int = DEFAULT_RESOLUTION,
        max_render_edge_px: int = 4000,
        bbox_padding_pt: float = 6.0,
        min_figure_width_pt: float = 60.0,
        min_figure_height_pt: float = 40.0,
        fallback_region_height_pt: float = 320.0,
        use_layout_model: Optional[bool] = None,
        layout_detector: Optional[object] = None,
        layout_detect_dpi: int = 144,
        layout_min_confidence: float = 0.3,
    ) -> None:
        if resolution < 72:
            raise ValueError(f"resolution must be >= 72 DPI, got {resolution}.")
        self.resolution = resolution
        self.max_render_edge_px = max_render_edge_px
        self.bbox_padding_pt = bbox_padding_pt
        self.min_figure_width_pt = min_figure_width_pt
        self.min_figure_height_pt = min_figure_height_pt
        self.fallback_region_height_pt = fallback_region_height_pt
        if use_layout_model is None:
            use_layout_model = (os.getenv("LAYOUT_DETECTION") or "on").strip().lower() not in {
                "0",
                "off",
                "false",
                "no",
            }
        self.use_layout_model = use_layout_model
        self.layout_detect_dpi = layout_detect_dpi
        self.layout_min_confidence = layout_min_confidence
        self._layout_detector = layout_detector
        self._layout_detector_failed = False

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def parse(self, pdf_path: Union[str, Path]) -> ParsedDocument:
        """Parses a PDF into page texts and extracted figures.

        Args:
            pdf_path: Path to the PDF file on disk.

        Returns:
            A :class:`~src.models.ParsedDocument` holding every page's text
            (page order preserved) and all successfully cropped figures.

        Raises:
            FileNotFoundError: If ``pdf_path`` does not exist.
            RuntimeError: If the PDF cannot be opened or read at all.
                Page-level extraction failures are logged and skipped instead
                of aborting the whole document.
        """
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")

        pages: List[PageText] = []
        figures: List[ExtractedFigure] = []
        seen_ids: Set[str] = set()

        logger.info("Parsing '%s' at %d DPI", path.name, self.resolution)
        try:
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    try:
                        text = page.extract_text() or ""
                    except Exception:
                        logger.exception(
                            "Text extraction failed on page %d", page.page_number
                        )
                        text = ""
                    pages.append(PageText(page_number=page.page_number, text=text))

                    # Primary: visual layout detection on the rendered page
                    # (robust across journals/column layouts). Fallback:
                    # caption-anchor heuristics over PDF coordinates.
                    page_figures: List[ExtractedFigure] = []
                    detector = self._get_layout_detector()
                    if detector is not None:
                        try:
                            page_figures = self._extract_with_layout_model(
                                page, detector, seen_ids
                            )
                        except Exception:
                            logger.exception(
                                "Layout-model extraction failed on page %d",
                                page.page_number,
                            )
                    if not page_figures:
                        try:
                            page_figures = self._extract_page_figures(page, seen_ids)
                        except Exception:
                            logger.exception(
                                "Figure extraction failed on page %d",
                                page.page_number,
                            )
                    figures.extend(page_figures)

                    # Release pdfplumber's per-page object caches so large
                    # documents do not accumulate memory across pages.
                    try:
                        page.flush_cache()
                    except Exception:  # pragma: no cover - defensive only
                        pass
        except Exception as exc:
            raise RuntimeError(f"Unable to parse PDF '{path}': {exc}") from exc

        logger.info(
            "Parsed %d page(s) from '%s'; extracted %d figure(s)",
            len(pages),
            path.name,
            len(figures),
        )
        return ParsedDocument(source_path=str(path), pages=pages, figures=figures)

    # ------------------------------------------------------------------ #
    # Layout-model extraction (primary)                                   #
    # ------------------------------------------------------------------ #

    def _get_layout_detector(self) -> Optional[object]:
        """Returns the layout detector, building it lazily.

        Returns ``None`` when layout detection is disabled, its dependencies
        are missing, or a previous construction attempt failed — the caller
        then uses the heuristic path.
        """
        if not self.use_layout_model or self._layout_detector_failed:
            return None
        if self._layout_detector is not None:
            return self._layout_detector
        try:
            from src.parsing.layout_detector import LayoutFigureDetector

            self._layout_detector = LayoutFigureDetector(
                confidence=self.layout_min_confidence
            )
        except Exception as exc:
            logger.warning(
                "Layout model unavailable (%s) — using caption-anchor "
                "heuristics instead. Install with: pip install doclayout-yolo "
                "huggingface-hub",
                exc,
            )
            self._layout_detector_failed = True
            return None
        return self._layout_detector

    def _extract_with_layout_model(
        self,
        page: "pdfplumber.page.Page",
        detector: object,
        seen_ids: Set[str],
    ) -> List[ExtractedFigure]:
        """Extracts figures using visual layout detection on the page image.

        Pipeline: render the page → detect ``figure`` / ``figure_caption``
        regions → map boxes back to PDF points → read each caption's text
        from the PDF text layer → crop the figure at full resolution → run
        the usual panel splitting.

        Args:
            page: The pdfplumber page.
            detector: A detector exposing ``detect(image) -> List[LayoutRegion]``.
            seen_ids: Global figure-id registry.

        Returns:
            Figures (and panels) found on this page; empty when the detector
            reports nothing, which triggers the heuristic fallback.
        """
        rendered_page = page.to_image(resolution=self.layout_detect_dpi).original
        regions: List[LayoutRegion] = detector.detect(rendered_page)  # type: ignore[attr-defined]

        figure_regions = [
            r
            for r in regions
            if r.label == "figure" and r.confidence >= self.layout_min_confidence
        ]
        caption_regions = [r for r in regions if r.label == "figure_caption"]
        if not figure_regions:
            return []

        # Pixel (detect render) → PDF point mapping.
        scale = 72.0 / float(self.layout_detect_dpi)
        page_x0, page_top, page_x1, page_bottom = (float(v) for v in page.bbox)

        def to_points(box: PixelBox) -> BoundingBox:
            return BoundingBox(
                x0=page_x0 + box.x0 * scale,
                top=page_top + box.y0 * scale,
                x1=page_x0 + box.x1 * scale,
                bottom=page_top + box.y1 * scale,
            ).clamped(page_x0, page_top, page_x1, page_bottom)

        graphics = self._collect_graphics(page)
        figure_regions.sort(key=lambda r: (r.box.y0, r.box.x0))  # reading order
        extracted: List[ExtractedFigure] = []

        for index, figure_region in enumerate(figure_regions, start=1):
            region_pts = to_points(figure_region.box)
            caption_region = self._nearest_caption(figure_region.box, caption_regions)
            caption_text = (
                self._extract_caption_text(page, to_points(caption_region.box))
                if caption_region
                else ""
            )

            number_match = _CAPTION_NUMBER.search(caption_text)
            if number_match:
                number = number_match.group("number")
                label = f"Figure {number}"
            else:
                # No parseable number (missing/garbled caption): synthesize a
                # stable page-local identifier.
                number = f"p{page.page_number}-{index}"
                label = f"Figure {number}"
            figure_id = f"figure-{number.lower()}"
            if figure_id in seen_ids:
                logger.debug(
                    "Skipping duplicate layout-model figure %s on page %d",
                    figure_id,
                    page.page_number,
                )
                continue

            image, dpi, rendered = self._render_region(page, region_pts)
            if image is None:
                continue

            caption = caption_text or f"({label} — no caption text detected)"
            seen_ids.add(figure_id)
            extracted.append(
                ExtractedFigure(
                    figure_id=figure_id,
                    label=label,
                    number=number,
                    caption=caption,
                    page_number=page.page_number,
                    bbox=region_pts,
                    image=image,
                    detection_method="layout-model",
                    dpi=dpi,
                    render_bbox=rendered,
                )
            )
            logger.info(
                "Extracted %s via layout model (page %d, conf=%.2f, %dx%d px)",
                label,
                page.page_number,
                figure_region.confidence,
                image.width,
                image.height,
            )

            # Panel splitting reuses the standard machinery on the detected
            # region (geometric markers; the VLM fallback happens later in
            # the framework when this yields nothing).
            match = _CaptionMatch(
                number=number,
                label=label,
                text=caption,
                bbox=(
                    to_points(caption_region.box) if caption_region else region_pts
                ),
            )
            try:
                panels = self._split_subfigures(
                    page, region_pts, match, "layout-model", figure_id, graphics, seen_ids
                )
            except Exception:
                logger.exception("Sub-figure splitting failed for %s", figure_id)
                panels = []
            if panels:
                extracted.extend(panels)
                logger.info(
                    "Split %s into %d panel(s): %s",
                    label,
                    len(panels),
                    ", ".join(p.panel or "?" for p in panels),
                )
        return extracted

    @staticmethod
    def _nearest_caption(
        figure_box: PixelBox, caption_regions: List[LayoutRegion]
    ) -> Optional[LayoutRegion]:
        """Pairs a figure with its most plausible caption region.

        Candidates must overlap the figure horizontally; the winner minimizes
        the vertical gap, with a bias toward captions *below* the figure
        (the dominant convention in scientific journals).
        """
        best: Optional[LayoutRegion] = None
        best_score = float("inf")
        for candidate in caption_regions:
            overlap = min(figure_box.x1, candidate.box.x1) - max(
                figure_box.x0, candidate.box.x0
            )
            if overlap <= 0:
                continue
            if candidate.box.y0 >= figure_box.y1:
                gap = candidate.box.y0 - figure_box.y1  # below the figure
            elif candidate.box.y1 <= figure_box.y0:
                gap = (figure_box.y0 - candidate.box.y1) + 40  # above: biased
            else:
                gap = 0  # overlapping (caption inside the figure box)
            if gap < best_score:
                best_score = gap
                best = candidate
        return best

    def _extract_caption_text(
        self, page: "pdfplumber.page.Page", bbox: BoundingBox
    ) -> str:
        """Reads a caption's text from the PDF text layer within a region.

        Some PDFs use kerning so tight that pdfplumber's default word
        segmentation drops the inter-word spaces ("Theenergiesare...").
        Several ``x_tolerance`` values are tried and the variant recovering
        the most spaces wins — as long as it does not overshoot into
        shredded single letters ("T h e"), which a space-ratio ceiling
        filters out.
        """
        page_x0, page_top, page_x1, page_bottom = (float(v) for v in page.bbox)
        clamped = bbox.padded(2.0).clamped(page_x0, page_top, page_x1, page_bottom)
        if clamped.width <= 1 or clamped.height <= 1:
            return ""
        try:
            region = page.within_bbox(
                (clamped.x0, clamped.top, clamped.x1, clamped.bottom)
            )
            text = " ".join((region.extract_text() or "").split())
        except Exception:
            logger.exception("Caption text extraction failed")
            return ""

        if len(text) <= 40:
            return text
        candidates = [text]
        for tolerance in (1.5, 1.0, 0.7):
            try:
                variant = " ".join(
                    (region.extract_text(x_tolerance=tolerance) or "").split()
                )
            except Exception:  # pragma: no cover - defensive only
                continue
            if variant:
                candidates.append(variant)

        def space_ratio(candidate: str) -> float:
            return candidate.count(" ") / max(len(candidate), 1)

        # Normal English prose sits around a 0.12-0.20 space ratio; beyond
        # ~0.25 the tokenizer started splitting inside words.
        plausible = [c for c in candidates if space_ratio(c) <= 0.25] or candidates
        return max(plausible, key=lambda c: c.count(" "))

    # ------------------------------------------------------------------ #
    # Per-page extraction                                                 #
    # ------------------------------------------------------------------ #

    def _extract_page_figures(
        self, page: "pdfplumber.page.Page", seen_ids: Set[str]
    ) -> List[ExtractedFigure]:
        """Finds and crops every captioned figure on one page.

        Args:
            page: The pdfplumber page object.
            seen_ids: Figure ids already extracted earlier in the document;
                used to ignore duplicate caption matches (first match wins).

        Returns:
            The figures extracted from this page, possibly empty. Multi-panel
            figures contribute one whole-figure entry plus one entry per
            detected panel.
        """
        lines: List[_LineDict] = page.extract_text_lines() or []
        captions = self._find_captions(lines)
        if not captions:
            return []

        graphics = self._collect_graphics(page)
        extracted: List[ExtractedFigure] = []

        for match in captions:
            figure_id = f"figure-{match.number.lower()}"
            if figure_id in seen_ids:
                logger.debug(
                    "Skipping duplicate caption for %s on page %d",
                    figure_id,
                    page.page_number,
                )
                continue

            region, method = self._locate_figure_region(
                page, match.bbox, lines, graphics
            )
            image, dpi, rendered = self._render_region(page, region)
            if image is None:
                logger.warning(
                    "Could not render a crop for %s on page %d — skipping",
                    figure_id,
                    page.page_number,
                )
                continue

            seen_ids.add(figure_id)
            extracted.append(
                ExtractedFigure(
                    figure_id=figure_id,
                    label=match.label,
                    number=match.number,
                    caption=match.text,
                    page_number=page.page_number,
                    bbox=region,
                    image=image,
                    detection_method=method,
                    dpi=dpi,
                    render_bbox=rendered,
                )
            )
            logger.info(
                "Extracted %s (page %d, %s, %dx%d px @ %d DPI)",
                match.label,
                page.page_number,
                method,
                image.width,
                image.height,
                dpi,
            )

            # Multi-panel figures: split into per-panel subfigures when
            # "(a)", "(b)", ... markers are found inside the region.
            try:
                panels = self._split_subfigures(
                    page, region, match, method, figure_id, graphics, seen_ids
                )
            except Exception:
                logger.exception(
                    "Sub-figure splitting failed for %s — keeping the whole "
                    "figure only",
                    figure_id,
                )
                panels = []
            if panels:
                extracted.extend(panels)
                logger.info(
                    "Split %s into %d panel(s): %s",
                    match.label,
                    len(panels),
                    ", ".join(p.panel or "?" for p in panels),
                )
        return extracted

    # ------------------------------------------------------------------ #
    # Caption detection                                                   #
    # ------------------------------------------------------------------ #

    def _find_captions(self, lines: List[_LineDict]) -> List[_CaptionMatch]:
        """Detects caption anchors and gathers their full paragraph text.

        A caption starts at a line matching :data:`_CAPTION_ANCHOR` and
        continues over subsequent lines while (a) the vertical gap between
        consecutive lines stays below ~0.8 of a line height (i.e. the lines
        belong to the same paragraph) and (b) no new caption anchor begins.

        Args:
            lines: The output of ``page.extract_text_lines()``.

        Returns:
            All caption matches on the page, in reading order.
        """
        matches: List[_CaptionMatch] = []
        for idx, line in enumerate(lines):
            text = (line.get("text") or "").strip()
            anchor = _CAPTION_ANCHOR.match(text)
            if not anchor:
                continue

            number = anchor.group("number")
            caption_lines: List[_LineDict] = [line]
            prev = line
            for nxt in lines[idx + 1 :]:
                nxt_text = (nxt.get("text") or "").strip()
                if _CAPTION_ANCHOR.match(nxt_text):
                    break  # the next figure's caption starts here
                prev_box = _line_bbox(prev)
                nxt_box = _line_bbox(nxt)
                # Paragraph continuation check: consecutive caption lines are
                # tightly spaced; a gap larger than ~0.8 line heights means
                # the caption paragraph has ended.
                line_height = max(prev_box.height, 6.0)
                if nxt_box.top - prev_box.bottom > 0.8 * line_height:
                    break
                caption_lines.append(nxt)
                prev = nxt

            bbox = _line_bbox(caption_lines[0])
            for extra in caption_lines[1:]:
                bbox = bbox.union(_line_bbox(extra))
            caption_text = " ".join(
                (part.get("text") or "").strip() for part in caption_lines
            ).strip()

            matches.append(
                _CaptionMatch(
                    number=number,
                    label=f"Figure {number}",
                    text=caption_text,
                    bbox=bbox,
                )
            )
        return matches

    # ------------------------------------------------------------------ #
    # Region localization                                                 #
    # ------------------------------------------------------------------ #

    def _locate_figure_region(
        self,
        page: "pdfplumber.page.Page",
        caption_bbox: BoundingBox,
        lines: List[_LineDict],
        graphics: List[Tuple[BoundingBox, str]],
    ) -> Tuple[BoundingBox, DetectionMethod]:
        """Maps a caption to the graphical region it most plausibly describes.

        Args:
            page: The pdfplumber page.
            caption_bbox: Bounding box of the full caption paragraph.
            lines: All text lines on the page (used to bound search bands).
            graphics: Pre-collected graphic objects from
                :meth:`_collect_graphics`.

        Returns:
            The detected region and the detection method used.
        """
        page_x0, page_top, page_x1, page_bottom = (float(v) for v in page.bbox)

        # The caption's horizontal extent (slightly widened) approximates the
        # column the figure lives in. This is crucial for two-column papers,
        # where the page may contain unrelated graphics in the other column.
        column = BoundingBox(
            x0=max(page_x0, caption_bbox.x0 - 15.0),
            top=page_top,
            x1=min(page_x1, caption_bbox.x1 + 15.0),
            bottom=page_bottom,
        )

        # Captions usually sit below their figure, so search "above" first.
        for direction in ("above", "below"):
            band = self._content_band(page, caption_bbox, column, lines, direction)
            if band is None:
                continue
            detected = self._detect_in_band(page, band, graphics)
            if detected is not None:
                return detected

        # Fallback: no graphic object could be associated with this caption
        # (some PDFs draw figures via operators pdfplumber does not surface).
        # Use a fixed-height window above the caption, or below it when the
        # caption sits too close to the top of the page.
        top = max(page_top, caption_bbox.top - self.fallback_region_height_pt)
        fallback = BoundingBox(
            x0=column.x0,
            top=top,
            x1=column.x1,
            bottom=max(top + 1.0, caption_bbox.top - 2.0),
        )
        if fallback.height < self.min_figure_height_pt:
            bottom = min(
                page_bottom, caption_bbox.bottom + self.fallback_region_height_pt
            )
            fallback = BoundingBox(
                x0=column.x0,
                top=min(caption_bbox.bottom + 2.0, bottom - 1.0),
                x1=column.x1,
                bottom=bottom,
            )
        logger.debug(
            "Using caption-relative fallback region on page %d", page.page_number
        )
        return fallback, "caption-fallback"

    def _content_band(
        self,
        page: "pdfplumber.page.Page",
        caption_bbox: BoundingBox,
        column: BoundingBox,
        lines: List[_LineDict],
        direction: str,
    ) -> Optional[BoundingBox]:
        """Computes the vertical band where the figure could live.

        The band spans from the caption edge to the nearest *paragraph-like*
        text line in the given direction (or the page margin if none exists).
        Short text lines are ignored on purpose: axis tick labels, legend
        entries, and in-plot annotations are text too, and treating them as
        boundaries would truncate the figure. A line only counts as a
        boundary when it is at least half the caption's width and overlaps
        the caption column horizontally.

        Args:
            page: The pdfplumber page.
            caption_bbox: The caption paragraph's bounding box.
            column: Horizontal column band the figure is assumed to occupy.
            lines: All text lines on the page.
            direction: Either ``"above"`` or ``"below"`` the caption.

        Returns:
            The search band, or ``None`` if the available space is too small
            to contain a plausible figure.
        """
        page_x0, page_top, page_x1, page_bottom = (float(v) for v in page.bbox)
        min_paragraph_width = 0.5 * max(caption_bbox.width, 1.0)

        if direction == "above":
            boundary = page_top
            for line in lines:
                line_box = _line_bbox(line)
                if line_box.bottom > caption_bbox.top - 1.0:
                    continue  # not strictly above the caption
                if line_box.width < min_paragraph_width:
                    continue  # short line — likely part of the figure itself
                if line_box.horizontal_overlap(column) <= 0.0:
                    continue  # belongs to the other column of the layout
                boundary = max(boundary, line_box.bottom)
            band = BoundingBox(
                x0=column.x0, top=boundary, x1=column.x1, bottom=caption_bbox.top - 1.0
            )
        else:
            boundary = page_bottom
            for line in lines:
                line_box = _line_bbox(line)
                if line_box.top < caption_bbox.bottom + 1.0:
                    continue  # not strictly below the caption
                if line_box.width < min_paragraph_width:
                    continue
                if line_box.horizontal_overlap(column) <= 0.0:
                    continue
                boundary = min(boundary, line_box.top)
            band = BoundingBox(
                x0=column.x0,
                top=caption_bbox.bottom + 1.0,
                x1=column.x1,
                bottom=boundary,
            )

        if band.height < 25.0:
            return None
        return band

    def _detect_in_band(
        self,
        page: "pdfplumber.page.Page",
        band: BoundingBox,
        graphics: List[Tuple[BoundingBox, str]],
    ) -> Optional[Tuple[BoundingBox, DetectionMethod]]:
        """Looks for a figure-sized graphic inside a search band.

        The region is the union of *every* raster image and vector drawing
        primitive found in the band. Multi-panel figures are frequently
        composed of several separate embedded images (one per panel) and/or
        many vector primitives, so picking any single "best" object would
        crop only one panel; the union covers the whole figure. The band is
        already bounded by paragraph-like text and the caption, which keeps
        the union from swallowing surrounding prose.

        Args:
            page: The pdfplumber page (used for page-width heuristics).
            band: The vertical band to search within.
            graphics: Pre-collected ``(bbox, kind)`` pairs where ``kind`` is
                ``"image"`` or ``"vector"``.

        Returns:
            ``(region, method)`` when a plausible figure region is found,
            otherwise ``None``. The method is ``"embedded-image"`` when at
            least one raster contributed, else ``"vector-cluster"``.
        """
        page_x0 = float(page.bbox[0])
        page_x1 = float(page.bbox[2])
        page_width = page_x1 - page_x0

        candidates: List[BoundingBox] = []
        has_raster = False
        for box, kind in graphics:
            # An object belongs to the band when its vertical center falls
            # inside it and it shares horizontal span with the column band.
            if not (band.top <= box.vertical_center <= band.bottom):
                continue
            if box.horizontal_overlap(band) <= 0.0:
                continue
            if kind == "image":
                candidates.append(box)
                has_raster = True
            else:
                # Skip page-wide thin rules (header/footer/section separators)
                # which would otherwise stretch the union across the page.
                if box.width > 0.9 * page_width and box.height < 3.0:
                    continue
                candidates.append(box)

        if not candidates:
            return None

        union = candidates[0]
        for box in candidates[1:]:
            union = union.union(box)
        if (
            union.width < self.min_figure_width_pt
            or union.height < self.min_figure_height_pt
        ):
            return None

        clipped = union.clamped(page_x0, band.top, page_x1, band.bottom)
        return clipped, ("embedded-image" if has_raster else "vector-cluster")

    # ------------------------------------------------------------------ #
    # Sub-figure (panel) splitting                                        #
    # ------------------------------------------------------------------ #

    def _split_subfigures(
        self,
        page: "pdfplumber.page.Page",
        region: BoundingBox,
        match: _CaptionMatch,
        method: DetectionMethod,
        parent_id: str,
        graphics: List[Tuple[BoundingBox, str]],
        seen_ids: Set[str],
    ) -> List[ExtractedFigure]:
        """Splits a multi-panel figure into per-panel subfigures.

        The decomposition combines two independent signals:

        1. **Panel markers** — "(a)", "(b)", ... rendered as PDF text inside
           the region (matplotlib/TikZ figures expose them as words with
           coordinates).
        2. **Content clustering** — every word and graphic inside the region
           is clustered into rows (by merging vertical extents across
           whitespace gutters) and then into cells within each row (by
           merging horizontal extents). Cell boxes are tight around the
           actual panel content, so crops do not bleed into neighbours even
           when panels have uneven heights or offsets.

        Cells are then labeled with the marker they contain. Unlabeled cells
        between two labeled ones get interpolated letters when the alphabetic
        gap matches exactly (recovering panels whose marker text was missed);
        any remaining unlabeled cells (colorbars, shared axis strips) are
        merged into the preceding labeled cell. A cell containing several
        markers (panels separated by a gutter narrower than the threshold) is
        cut at the extra markers as a last resort.

        Finally each crop is *expanded* until every element it owns is fully
        contained (no truncated axes/legends/colorbars) and the panel set is
        *validated* for collisions — when panels overlap heavily the layout
        inference was wrong and no split is returned at all, keeping the
        intact whole figure instead of damaged subfigures.

        Figures whose panel labels are baked into a raster expose no marker
        words, so splitting silently yields nothing and the whole-figure crop
        remains the only entry.

        Args:
            page: The pdfplumber page.
            region: The whole figure's detected region.
            match: The caption match for the whole figure.
            method: The parent figure's detection method (inherited).
            parent_id: The parent figure's id, recorded on each panel.
            graphics: Pre-collected page graphics from
                :meth:`_collect_graphics`.
            seen_ids: Global id registry; panels colliding with existing ids
                are skipped.

        Returns:
            The per-panel subfigures (possibly empty).
        """
        # Search bounds are padded upward and sideways — but NOT downward:
        # the caption right below the region repeats the same "(a)", "(b)"
        # tokens in prose, and markers always sit at panel tops anyway.
        search_x0 = region.x0 - 12.0
        search_x1 = region.x1 + 12.0
        search_top = region.top - 12.0
        search_bottom = region.bottom - 2.0

        def _inside(box: BoundingBox) -> bool:
            center_x = (box.x0 + box.x1) / 2.0
            return (
                search_top <= box.vertical_center <= search_bottom
                and search_x0 <= center_x <= search_x1
            )

        word_boxes: List[Tuple[BoundingBox, Optional[str]]] = []
        for word in page.extract_words() or []:
            box = self._object_bbox(word)
            if box is None or not _inside(box):
                continue
            word_boxes.append((box, _panel_letter(word.get("text") or "")))

        letters_found = {letter for _, letter in word_boxes if letter}
        if len(letters_found) < 2 or "a" not in letters_found:
            return []
        if len(letters_found) > 12:
            logger.debug(
                "Implausible panel count (%d) for %s — not splitting",
                len(letters_found),
                parent_id,
            )
            return []

        elements: List[BoundingBox] = [box for box, _ in word_boxes]
        elements.extend(box for box, _ in graphics if _inside(box))

        # --- 1. rows: merge vertical extents across gutters --------------- #
        row_bands = self._merge_intervals(
            [(box.top, box.bottom) for box in elements], _PANEL_GUTTER_PT
        )

        # --- 2. cells: merge horizontal extents within each row ----------- #
        # Each cell keeps a tight union box of its member elements plus the
        # panel markers that landed inside it.
        cells: List[Tuple[BoundingBox, List[Tuple[str, BoundingBox]]]] = []
        for band_top, band_bottom in row_bands:
            row_elements = [
                box for box in elements if band_top <= box.vertical_center <= band_bottom
            ]
            if not row_elements:
                continue
            column_ranges = self._merge_intervals(
                [(box.x0, box.x1) for box in row_elements], _PANEL_GUTTER_PT
            )
            for cell_x0, cell_x1 in column_ranges:
                members = [
                    box
                    for box in row_elements
                    if cell_x0 <= (box.x0 + box.x1) / 2.0 <= cell_x1
                ]
                if not members:
                    continue
                tight = members[0]
                for box in members[1:]:
                    tight = tight.union(box)
                cells.append((tight, []))

        for box, letter in word_boxes:
            if not letter:
                continue
            center_x = (box.x0 + box.x1) / 2.0
            for tight, marks in cells:
                if (
                    tight.x0 <= center_x <= tight.x1
                    and tight.top <= box.vertical_center <= tight.bottom
                ):
                    marks.append((letter, box))
                    break

        # --- 3. resolve cells into (letter, bbox) panels ------------------- #
        resolved: List[Tuple[Optional[str], BoundingBox]] = []
        for tight, marks in cells:
            unique: Dict[str, BoundingBox] = {}
            for letter, box in sorted(marks, key=lambda kv: (kv[1].top, kv[1].x0)):
                unique.setdefault(letter, box)
            if len(unique) <= 1:
                resolved.append((next(iter(unique), None), tight))
                continue
            # Several panels merged into one cell — typical when the whole
            # figure is a single embedded raster (no internal gutters for
            # the content clustering to find). Cut the cell into a 2-D
            # marker grid: single-axis cuts would merge panels from
            # different rows (e.g. "(b)" with the "(e)" right below it).
            resolved.extend(self._cut_cell_by_markers(tight, unique))

        # The caption is ground truth for how many panels exist and which
        # letters they carry — use it to recover panels whose markers were
        # not detected.
        expected_letters = sorted(split_caption(match.text).keys())
        panels_by_letter = self._assign_letters(
            resolved, expected_letters if len(expected_letters) >= 2 else None
        )
        if len(panels_by_letter) < 2:
            return []

        # --- 4. expand crops & validate ------------------------------------ #
        # Grow each panel until every element it owns is fully contained
        # (nothing truncated), then verify the panels do not collide. When
        # validation fails the layout inference was wrong — prefer the intact
        # whole figure over damaged subfigures.
        validated = self._expand_and_validate(panels_by_letter, elements, region)
        if validated is None:
            logger.info(
                "Panel boundaries for %s are ambiguous — keeping the whole "
                "figure only",
                parent_id,
            )
            return []

        # --- 5. render each panel ------------------------------------------ #
        sub_captions = split_caption(match.text)
        panels: List[ExtractedFigure] = []
        for letter, panel_bbox in validated:
            clamped = panel_bbox.clamped(region.x0, region.top, region.x1, region.bottom)
            if clamped.width < 30.0 or clamped.height < 30.0:
                logger.debug(
                    "Panel (%s) of %s too small (%.0fx%.0f pt) — skipped",
                    letter,
                    parent_id,
                    clamped.width,
                    clamped.height,
                )
                continue
            figure_id = f"figure-{match.number.lower()}{letter}"
            if figure_id in seen_ids:
                continue
            image, dpi, rendered = self._render_region(page, clamped)
            if image is None:
                continue

            seen_ids.add(figure_id)
            panels.append(
                ExtractedFigure(
                    figure_id=figure_id,
                    label=f"Figure {match.number}({letter})",
                    number=f"{match.number.lower()}{letter}",
                    caption=sub_captions.get(letter) or match.text,
                    page_number=page.page_number,
                    bbox=clamped,
                    image=image,
                    detection_method=method,
                    dpi=dpi,
                    parent_id=parent_id,
                    panel=letter,
                    render_bbox=rendered,
                )
            )
        return panels

    @staticmethod
    def _expand_and_validate(
        panels: List[Tuple[str, BoundingBox]],
        elements: List[BoundingBox],
        region: BoundingBox,
        max_iterations: int = 10,
    ) -> Optional[List[Tuple[str, BoundingBox]]]:
        """Expands panel crops to completeness, then validates the result.

        **Expansion:** an element is *owned* by a panel when its center lies
        inside the panel's box. Each panel grows to the union of everything
        it owns, repeatedly — a union can pull new element centers inside
        (e.g. a legend or colorbar poking out of the tight cell) — until a
        fixed point is reached. This guarantees no owned axis, label, legend,
        colorbar, or annotation is truncated by the crop; a slightly large
        crop is preferred over one that cuts scientific information.

        **Validation:** after expansion, panels must remain essentially
        disjoint. When two *expanded* panels collide heavily (dense figures
        with narrow gutters make expansion overreach), the offending panels
        are reverted to their tight content boxes — which are disjoint by
        construction — rather than discarding the whole split. Only when the
        tight boxes themselves overlap heavily (the layout inference itself
        was wrong) is ``None`` returned so the caller keeps the intact whole
        figure: incorrect segmentation is worse than no segmentation.

        Args:
            panels: Reading-ordered ``(letter, tight_bbox)`` panels.
            elements: Every word/graphic box inside the figure region.
            region: The whole figure's region (expansion never exceeds it).
            max_iterations: Retained for API stability; expansion is a
                single ownership pass (iterating lets boxes chain into their
                neighbours in tightly packed figures).

        Returns:
            The expanded (or reverted) panels, or ``None`` when even the
            tight boxes conflict.
        """
        del max_iterations  # single-pass by design; see docstring

        def _overlap_ratio(first: BoundingBox, second: BoundingBox) -> float:
            overlap_w = min(first.x1, second.x1) - max(first.x0, second.x0)
            overlap_h = min(first.bottom, second.bottom) - max(first.top, second.top)
            if overlap_w <= 0.0 or overlap_h <= 0.0:
                return 0.0
            smaller = min(first.area, second.area)
            return (overlap_w * overlap_h) / smaller if smaller > 0.0 else 1.0

        # Single ownership pass: an element belongs to a panel when its
        # center lies inside the panel's ORIGINAL tight box. Iterating on the
        # grown box would let ownership chain across narrow gutters and
        # swallow neighbouring panels.
        expanded: List[Tuple[str, BoundingBox]] = []
        for letter, bbox in panels:
            grown = bbox
            for element in elements:
                center_x = (element.x0 + element.x1) / 2.0
                if (
                    bbox.x0 <= center_x <= bbox.x1
                    and bbox.top <= element.vertical_center <= bbox.bottom
                ):
                    grown = grown.union(element)
            expanded.append(
                (letter, grown.clamped(region.x0, region.top, region.x1, region.bottom))
            )

        # Collision handling: revert heavily-overlapping expanded panels to
        # their tight boxes instead of abandoning the split.
        reverted = list(expanded)
        for i in range(len(reverted)):
            for j in range(i + 1, len(reverted)):
                if _overlap_ratio(reverted[i][1], reverted[j][1]) > 0.35:
                    logger.debug(
                        "Expanded panels (%s) and (%s) collide — reverting "
                        "both to their tight content boxes",
                        reverted[i][0],
                        reverted[j][0],
                    )
                    reverted[i] = (reverted[i][0], panels[i][1])
                    reverted[j] = (reverted[j][0], panels[j][1])

        # Final gate: tight boxes conflicting means the layout inference
        # itself was wrong — keep the whole figure.
        for i in range(len(reverted)):
            for j in range(i + 1, len(reverted)):
                if _overlap_ratio(reverted[i][1], reverted[j][1]) > 0.35:
                    logger.info(
                        "Panels (%s) and (%s) overlap even at their tight "
                        "boxes — layout ambiguous",
                        reverted[i][0],
                        reverted[j][0],
                    )
                    return None
        return reverted

    @staticmethod
    def _cut_cell_by_markers(
        cell: BoundingBox, markers: Dict[str, BoundingBox]
    ) -> List[Tuple[Optional[str], BoundingBox]]:
        """Cuts an under-split content cell into a 2-D grid at its markers.

        Used when one content cell contains several panel markers — panels
        separated by gutters narrower than the merge threshold, or a single
        embedded raster covering the whole multi-panel figure. Markers sit
        at panel top-left corners, so:

        * markers are clustered into rows by their top edge (with a
          tolerance proportional to the cell height, so slightly offset
          markers of uneven panels stay in one row);
        * each row spans from its markers' top (first row snaps to the cell
          top) down to the next row's marker top;
        * within a row, each panel spans from its own marker to just before
          the next marker (first/last snap to the cell edges).

        Args:
            cell: The under-split cell's bounding box.
            markers: Panel letter → marker box, all inside the cell.

        Returns:
            Reading-ordered ``(letter, bbox)`` sub-cells.
        """
        ordered = sorted(markers.items(), key=lambda kv: (kv[1].top, kv[1].x0))
        # Fine enough that adjacent rows of dense multi-row figures are not
        # merged (which would fuse vertically neighbouring panels), while
        # still tolerating slightly offset markers of uneven panels.
        tolerance = max(18.0, 0.08 * max(cell.height, 1.0))
        rows: List[List[Tuple[str, BoundingBox]]] = []
        for letter, box in ordered:
            if rows and box.top - min(b.top for _, b in rows[-1]) <= tolerance:
                rows[-1].append((letter, box))
            else:
                rows.append([(letter, box)])
        row_tops = [min(box.top for _, box in row) for row in rows]

        cells: List[Tuple[Optional[str], BoundingBox]] = []
        for row_index, row in enumerate(rows):
            row_top = (
                cell.top
                if row_index == 0
                else max(cell.top, row_tops[row_index] - 2.0)
            )
            row_bottom = (
                cell.bottom
                if row_index == len(rows) - 1
                else min(cell.bottom, row_tops[row_index + 1] - 2.0)
            )
            row_sorted = sorted(row, key=lambda kv: kv[1].x0)
            for col_index, (letter, box) in enumerate(row_sorted):
                left = cell.x0 if col_index == 0 else max(cell.x0, box.x0 - 2.0)
                right = (
                    cell.x1
                    if col_index == len(row_sorted) - 1
                    else min(cell.x1, row_sorted[col_index + 1][1].x0 - 2.0)
                )
                cells.append(
                    (
                        letter,
                        BoundingBox(
                            x0=left, top=row_top, x1=right, bottom=row_bottom
                        ),
                    )
                )
        return cells

    @staticmethod
    def _merge_intervals(
        intervals: List[Tuple[float, float]], gap: float
    ) -> List[Tuple[float, float]]:
        """Merges 1-D intervals whose spacing is at most ``gap``.

        Args:
            intervals: ``(start, end)`` pairs in any order.
            gap: Maximum spacing between two intervals for them to merge.

        Returns:
            Non-overlapping merged intervals in ascending order.
        """
        if not intervals:
            return []
        ordered = sorted(intervals)
        merged: List[List[float]] = [list(ordered[0])]
        for start, end in ordered[1:]:
            if start - merged[-1][1] <= gap:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(start, end) for start, end in merged]

    @staticmethod
    def _assign_letters(
        resolved: List[Tuple[Optional[str], BoundingBox]],
        expected_letters: Optional[List[str]] = None,
    ) -> List[Tuple[str, BoundingBox]]:
        """Turns labeled/unlabeled content cells into a final panel list.

        Passes over the reading-ordered cells:

        1. **Interpolation** — a run of unlabeled cells strictly between two
           labeled ones receives interpolated letters when the alphabetic
           distance matches the run length exactly (recovers panels whose
           marker text was not detected).
        2. **Caption reconciliation** — when the caption's own panel list is
           known and its length matches the cell count, cells are assigned
           positionally (in reading order) provided no detected marker
           contradicts its position. This recovers leading/trailing panels
           whose markers were missed, which absorption would otherwise merge
           into a neighbour — the "two panels joined" failure mode.
        3. **Absorption** — remaining unlabeled cells (colorbars, shared axis
           strips, stray annotations) are merged into the preceding labeled
           cell (or the first following one, for leading cells).
        4. **Deduplication** — cells sharing a letter are unioned.

        Args:
            resolved: Reading-ordered ``(letter or None, bbox)`` cells.
            expected_letters: The panel letters the figure's caption
                describes, in order (``["a", "b", "c"]``), when known.

        Returns:
            ``(letter, bbox)`` panels in reading order.
        """
        cells = list(resolved)

        # 1. Interpolate letter gaps.
        labeled = [index for index, (letter, _) in enumerate(cells) if letter]
        for left, right in zip(labeled, labeled[1:]):
            run = right - left - 1
            if run <= 0:
                continue
            first, last = cells[left][0], cells[right][0]
            if first and last and ord(last) - ord(first) - 1 == run:
                for offset in range(1, run + 1):
                    cells[left + offset] = (
                        chr(ord(first) + offset),
                        cells[left + offset][1],
                    )

        # 2. Caption-guided positional reconciliation.
        if (
            expected_letters
            and len(cells) == len(expected_letters)
            and any(letter is None for letter, _ in cells)
            and all(
                letter is None or letter == expected_letters[index]
                for index, (letter, _) in enumerate(cells)
            )
        ):
            cells = [
                (expected_letters[index], box)
                for index, (_, box) in enumerate(cells)
            ]

        # 3. Absorb still-unlabeled cells into their labeled neighbour.
        merged: List[Tuple[str, BoundingBox]] = []
        pending: List[BoundingBox] = []
        for letter, box in cells:
            if letter is None:
                if merged:
                    prev_letter, prev_box = merged[-1]
                    merged[-1] = (prev_letter, prev_box.union(box))
                else:
                    pending.append(box)
                continue
            for waiting in pending:
                box = box.union(waiting)
            pending = []
            merged.append((letter, box))

        # 4. Union duplicate letters (e.g. one panel split across rows).
        final: Dict[str, BoundingBox] = {}
        order: List[str] = []
        for letter, box in merged:
            if letter in final:
                final[letter] = final[letter].union(box)
            else:
                final[letter] = box
                order.append(letter)
        return [(letter, final[letter]) for letter in order]

    # ------------------------------------------------------------------ #
    # Graphics collection & rendering                                     #
    # ------------------------------------------------------------------ #

    def _collect_graphics(
        self, page: "pdfplumber.page.Page"
    ) -> List[Tuple[BoundingBox, str]]:
        """Collects every graphic object on the page with a coarse kind tag.

        Returns:
            ``(bbox, kind)`` pairs where ``kind`` is ``"image"`` for embedded
            rasters and ``"vector"`` for drawing primitives.
        """
        graphics: List[Tuple[BoundingBox, str]] = []
        for obj in page.images or []:
            box = self._object_bbox(obj)
            if box is not None:
                graphics.append((box, "image"))
        for attr in ("rects", "lines", "curves"):
            for obj in getattr(page, attr, None) or []:
                box = self._object_bbox(obj)
                if box is not None:
                    graphics.append((box, "vector"))
        return graphics

    @staticmethod
    def _object_bbox(obj: Dict[str, Any]) -> Optional[BoundingBox]:
        """Safely builds a bounding box from a pdfplumber object dict.

        Returns:
            The box, or ``None`` if the object lacks usable coordinates.
        """
        try:
            return BoundingBox(
                x0=float(obj["x0"]),
                top=float(obj["top"]),
                x1=float(obj["x1"]),
                bottom=float(obj["bottom"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _render_region(
        self, page: "pdfplumber.page.Page", region: BoundingBox
    ) -> Tuple[Optional[Image.Image], int, Optional[BoundingBox]]:
        """Renders a page region into an in-memory high-resolution RGB image.

        Args:
            page: The pdfplumber page.
            region: The region to crop, in PDF point coordinates.

        Returns:
            ``(image, dpi, rendered_bbox)`` where ``image`` is ``None`` if
            rendering failed or the region was degenerate, ``dpi`` is the
            effective resolution actually used (it may be lower than the
            configured resolution for very large regions — see
            ``max_render_edge_px``), and ``rendered_bbox`` is the padded,
            page-clamped region the pixels actually correspond to — the
            anchor for exact pixel↔point coordinate mapping.
        """
        page_x0, page_top, page_x1, page_bottom = (float(v) for v in page.bbox)
        clamped = region.padded(self.bbox_padding_pt).clamped(
            page_x0, page_top, page_x1, page_bottom
        )
        if clamped.width < 4.0 or clamped.height < 4.0:
            logger.warning(
                "Degenerate crop region on page %d — skipping", page.page_number
            )
            return None, self.resolution, None

        # A region of W x H points renders to (W * dpi / 72) x (H * dpi / 72)
        # pixels (1 pt = 1/72 in). Cap the longest edge so a full-page region
        # cannot allocate a multi-hundred-megabyte bitmap.
        longest_pt = max(clamped.width, clamped.height)
        dpi = self.resolution
        if longest_pt * dpi / 72.0 > self.max_render_edge_px:
            dpi = max(72, int(self.max_render_edge_px * 72.0 / longest_pt))
            logger.debug(
                "Reducing render DPI to %d for a %.0f pt region on page %d",
                dpi,
                longest_pt,
                page.page_number,
            )

        try:
            cropped = page.crop((clamped.x0, clamped.top, clamped.x1, clamped.bottom))
            rendered = cropped.to_image(resolution=dpi).original.convert("RGB")
        except Exception:
            logger.exception("Failed to render crop on page %d", page.page_number)
            return None, dpi, None
        return rendered, dpi, clamped
