"""Typed data models shared across every stage of the pipeline.

All models are Pydantic v2 models so the pipeline stays fully validated and
serializable (except for the in-memory ``PIL.Image`` payloads, which are
allowed through ``arbitrary_types_allowed``).
"""

from __future__ import annotations

from typing import List, Literal, Optional

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

# How a figure's bounding box was determined on the page.
DetectionMethod = Literal[
    "layout-model", "embedded-image", "vector-cluster", "caption-fallback"
]


class BoundingBox(BaseModel):
    """Axis-aligned rectangle in PDF *point* space with a top-left origin.

    pdfplumber reports geometry with ``y`` increasing downwards, so
    ``top < bottom`` for every valid box. One PDF point equals 1/72 inch,
    which means a box of width ``w`` points renders to ``w * dpi / 72``
    pixels at a given resolution.

    Note:
        Derived operations such as :meth:`clamped` may produce a degenerate
        (zero or negative area) box when the operands do not overlap —
        callers should check :attr:`width`/:attr:`height` before using the
        result. This is intentional: an "empty intersection" is a valid
        geometric outcome, not a validation error.
    """

    x0: float = Field(..., description="Left edge in PDF points.")
    top: float = Field(..., description="Top edge in PDF points (smaller = higher on the page).")
    x1: float = Field(..., description="Right edge in PDF points.")
    bottom: float = Field(..., description="Bottom edge in PDF points.")

    @property
    def width(self) -> float:
        """Horizontal extent in points (may be negative for degenerate boxes)."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Vertical extent in points (may be negative for degenerate boxes)."""
        return self.bottom - self.top

    @property
    def area(self) -> float:
        """Non-negative area in square points."""
        return max(self.width, 0.0) * max(self.height, 0.0)

    @property
    def vertical_center(self) -> float:
        """The ``y`` coordinate halfway between ``top`` and ``bottom``."""
        return (self.top + self.bottom) / 2.0

    def padded(self, pad: float) -> "BoundingBox":
        """Returns a copy grown by ``pad`` points on every side.

        Args:
            pad: Amount of padding in PDF points (negative shrinks the box).

        Returns:
            A new, padded :class:`BoundingBox`.
        """
        return BoundingBox(
            x0=self.x0 - pad,
            top=self.top - pad,
            x1=self.x1 + pad,
            bottom=self.bottom + pad,
        )

    def clamped(self, x0: float, top: float, x1: float, bottom: float) -> "BoundingBox":
        """Returns the intersection of this box with the given bounds.

        Args:
            x0: Minimum allowed left edge.
            top: Minimum allowed top edge.
            x1: Maximum allowed right edge.
            bottom: Maximum allowed bottom edge.

        Returns:
            A new box confined to ``[x0, x1] × [top, bottom]``. May be
            degenerate if this box lies entirely outside the bounds.
        """
        return BoundingBox(
            x0=max(self.x0, x0),
            top=max(self.top, top),
            x1=min(self.x1, x1),
            bottom=min(self.bottom, bottom),
        )

    def union(self, other: "BoundingBox") -> "BoundingBox":
        """Returns the smallest box containing both this box and ``other``."""
        return BoundingBox(
            x0=min(self.x0, other.x0),
            top=min(self.top, other.top),
            x1=max(self.x1, other.x1),
            bottom=max(self.bottom, other.bottom),
        )

    def horizontal_overlap(self, other: "BoundingBox") -> float:
        """Length (in points) of the shared horizontal span with ``other``.

        Returns:
            ``0.0`` when the boxes do not overlap horizontally.
        """
        return max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))


class PixelBox(BaseModel):
    """Axis-aligned rectangle in *image pixel* space (origin top-left).

    Used for panel bounding boxes returned by vision models and consumed by
    the deterministic crop engine. Distinct from :class:`BoundingBox`, which
    lives in PDF point space.
    """

    x0: int = Field(..., ge=0)
    y0: int = Field(..., ge=0)
    x1: int = Field(..., ge=0)
    y1: int = Field(..., ge=0)

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_tuple(self) -> tuple:
        """The ``(left, upper, right, lower)`` tuple ``PIL.Image.crop`` expects."""
        return (self.x0, self.y0, self.x1, self.y1)

    def clamped(self, width: int, height: int) -> "PixelBox":
        """Returns a copy confined to an image of the given dimensions."""
        return PixelBox(
            x0=max(0, min(self.x0, width)),
            y0=max(0, min(self.y0, height)),
            x1=max(0, min(self.x1, width)),
            y1=max(0, min(self.y1, height)),
        )

    def padded(self, pad: int, width: int, height: int) -> "PixelBox":
        """Returns a copy grown by ``pad`` pixels, clamped to the image."""
        return PixelBox(
            x0=max(0, self.x0 - pad),
            y0=max(0, self.y0 - pad),
            x1=min(width, self.x1 + pad),
            y1=min(height, self.y1 + pad),
        )


class LayoutRegion(BaseModel):
    """One region found by a page-layout detection model.

    Attributes:
        label: Normalized region type (``"figure"``, ``"figure_caption"``,
            ``"table"``, ...).
        box: The region's extent in the rendered page image's pixel space.
        confidence: Detector confidence in [0, 1].
    """

    label: str
    box: PixelBox
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class CellIssue(BaseModel):
    """One problem a grid auditor reported for a numbered cell.

    Attributes:
        cell: 1-based badge index of the cell.
        problem: Problem tag — ``"two_panels_side_by_side"``,
            ``"two_panels_stacked"``, ``"more_than_two_panels"``,
            ``"fragment_of_panel"``, or ``"empty"``.
    """

    cell: int = Field(..., ge=1)
    problem: str = ""


class GridAudit(BaseModel):
    """A vision model's audit of a numbered segmentation grid.

    Attributes:
        cell_counts: Panels counted in each cell, badge order (1.0 = one
            complete panel; 0.5 = a cut fragment; 0 = empty).
        issues: Optional per-cell problem tags for cells whose count is not
            exactly 1 — orientation hints for the deterministic editor.
        confidence: The auditor's self-reported confidence in [0, 1].
    """

    cell_counts: List[float] = Field(default_factory=list)
    issues: List[CellIssue] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    def is_clean(self, min_confidence: float) -> bool:
        """True when every cell holds exactly one panel, confidently."""
        return (
            bool(self.cell_counts)
            and all(abs(count - 1.0) < 0.01 for count in self.cell_counts)
            and self.confidence >= min_confidence
        )


class PageText(BaseModel):
    """Full extracted text of a single PDF page.

    Attributes:
        page_number: 1-based page index, matching pdfplumber's convention.
        text: The page text as returned by ``page.extract_text()`` (may be
            empty for image-only pages).
    """

    page_number: int = Field(..., ge=1)
    text: str


class ExtractedFigure(BaseModel):
    """A figure detected in the document, with its high-resolution crop.

    Attributes:
        figure_id: Stable identifier derived from the caption number,
            e.g. ``"figure-3"`` or ``"figure-2a"``.
        label: Human-readable label, e.g. ``"Figure 3"``.
        number: Raw figure number extracted from the caption anchor
            (``"3"``, ``"2a"``, ``"4.1"``, ...).
        caption: The full caption paragraph, whitespace-normalized.
        page_number: 1-based page the figure was found on.
        bbox: The cropped region in PDF point coordinates (top-left origin).
        image: The rendered high-resolution crop as an RGB ``PIL.Image``.
        detection_method: How the region was located (embedded raster image,
            cluster of vector primitives, or the caption-relative fallback).
        dpi: The effective resolution the crop was rendered at.
        parent_id: For subfigure panels (e.g. ``"figure-2a"``), the id of the
            whole figure they were split from (``"figure-2"``); ``None`` for
            top-level figures.
        panel: The panel letter (``"a"``, ``"b"``, ...) for subfigures;
            ``None`` for top-level figures.
        render_bbox: The page region (PDF points) that was actually rendered
            into ``image`` — i.e. ``bbox`` after padding and page clamping.
            Enables exact pixel↔point coordinate mapping for VLM-detected
            panels. ``None`` on objects created before rendering metadata
            was tracked.
        origin: How the entry was produced — deterministic PDF geometry
            (``"geometric"``), deterministic raster gutter segmentation
            (``"raster"``), or vision-model panel detection (``"vlm"``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    figure_id: str
    label: str
    number: str
    caption: str
    page_number: int = Field(..., ge=1)
    bbox: BoundingBox
    image: Image.Image
    detection_method: DetectionMethod
    dpi: int = Field(..., ge=36)
    parent_id: Optional[str] = None
    panel: Optional[str] = None
    render_bbox: Optional[BoundingBox] = None
    origin: Literal["geometric", "raster", "vlm"] = "geometric"


class ParsedDocument(BaseModel):
    """The complete result of parsing one PDF.

    Attributes:
        source_path: Filesystem path of the parsed PDF.
        pages: Per-page extracted text, in page order.
        figures: Every successfully extracted figure, in document order.
    """

    source_path: str
    pages: List[PageText]
    figures: List[ExtractedFigure]

    @property
    def full_text(self) -> str:
        """The whole document text with pages joined by blank lines."""
        return "\n\n".join(page.text for page in self.pages)

    def page_text(self, page_number: int) -> str:
        """Returns the text of a specific page.

        Args:
            page_number: 1-based page index.

        Raises:
            KeyError: If the page does not exist in this document.
        """
        for page in self.pages:
            if page.page_number == page_number:
                return page.text
        raise KeyError(f"Page {page_number} not found in '{self.source_path}'.")


class FigureContext(BaseModel):
    """The multimodal bundle the RAG index stores for one figure.

    Attributes:
        figure: The extracted figure (image, caption, geometry).
        references: Passages from the paper body that explicitly reference
            this figure (excluding the caption itself), ordered by relevance
            when produced by a ranked retrieval.
    """

    figure: ExtractedFigure
    references: List[str] = Field(default_factory=list)
