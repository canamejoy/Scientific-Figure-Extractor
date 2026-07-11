"""The figure-extraction framework: PDF → figures → panels → dataset.

The framework follows a *robust-base* design: a downstream analysis agent
always receives the **complete figure** at high resolution plus a reliable
**panel→caption map** (from the caption text), and panel **crops** are an
optional, high-confidence extra. A correct whole figure beats a wrong crop,
so crops are only produced when they can be trusted.

Panel cropping runs in tiers, most-reliable first:

1. **PDF text markers** (in the parser) — when the "(a)", "(b)", ... labels
   exist as PDF text, panels are cropped at pixel-exact geometric boundaries.
   No model involved.
2. **Raster whitespace/line gutters** (deterministic) — the rendered figure
   is segmented along its whitespace gutters and frame lines; trusted only
   when the cell count matches the caption's panel count exactly.
3. **Vision-assisted grid audit** (opt-in, ``panel_cropping="vlm-assisted"``)
   — Python proposes candidate grids and a vision model *audits* them
   (per-cell panel counting); its report drives deterministic grid edits.
   The model never produces coordinates. Off by default: slow and, on small
   local models, unreliable on hard layouts.

When no tier can confidently split a figure, it is kept whole — the
panel→caption map is still exported, so no information is lost.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

from src.extraction.crop_engine import CropEngine
from src.extraction.exporter import DatasetExporter
from src.models import BoundingBox, ExtractedFigure, ParsedDocument, PixelBox
from src.parsing.pdf_parser import ScientificPDFParser
from src.utils.captions import split_caption
from src.vision.panel_detector import PanelDetector

logger = logging.getLogger(__name__)

# How panels are cropped. Deterministic tiers (markers + raster gutters) are
# always safe; the vision-assisted tier is opt-in.
PanelCropping = Literal["deterministic", "markers-only", "vlm-assisted"]


class ExtractionConfig(BaseModel):
    """Configuration for the extraction framework.

    Attributes:
        panel_cropping: Panel-crop strategy.

            * ``"deterministic"`` (default) — PDF text markers plus raster
              whitespace/line gutter segmentation. No model, fast, and only
              crops when confident (keeps the figure whole otherwise).
            * ``"markers-only"`` — the most conservative mode: crop only from
              PDF text markers, skip raster segmentation.
            * ``"vlm-assisted"`` — additionally run the vision grid-audit
              debate for figures the deterministic tiers could not split.
              Slow; use only with a capable model (ideally cloud). Trades
              speed for recall on hard layouts (montages, touching panels).
        provider: Vision provider override (``None`` → environment).
        model: Vision model override (``None`` → environment).
        crop_padding_px: Initial padding of every proposed crop.
        crop_max_expansion_px: Per-side expansion budget during validation.
        resolution_dpi: Parser render resolution.
        max_grid_attempts: Maximum candidate grids considered per figure
            (vlm-assisted only).
        verify_min_confidence: Minimum auditor confidence to accept a grid
            (vlm-assisted only).
        debate_model: Second vision model for the audit debate (e.g.
            ``"gemma3:12b"``); ``None`` reads ``VLM_DEBATE_MODEL``. With two
            models they alternate as grid auditor and a grid is accepted only
            when both sign it off; with one model it debates against itself.
        max_debate_rounds: Audit rounds per starting grid (vlm-assisted only).
    """

    panel_cropping: PanelCropping = "deterministic"
    provider: Optional[str] = None
    model: Optional[str] = None
    crop_padding_px: int = Field(12, ge=0)
    crop_max_expansion_px: int = Field(120, ge=0)
    resolution_dpi: int = Field(300, ge=72)
    max_grid_attempts: int = Field(4, ge=1)
    verify_min_confidence: float = Field(0.7, ge=0.0, le=1.0)
    debate_model: Optional[str] = None
    max_debate_rounds: int = Field(6, ge=1)

    @property
    def use_raster_gutters(self) -> bool:
        """Whether deterministic raster gutter segmentation runs."""
        return self.panel_cropping in ("deterministic", "vlm-assisted")

    @property
    def use_vlm(self) -> bool:
        """Whether the vision-assisted grid-audit debate runs."""
        return self.panel_cropping == "vlm-assisted"

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ExtractionConfig":
        """Loads the configuration from a JSON file.

        Args:
            path: Path to a JSON object matching this model's fields.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


class FigureExtractionFramework:
    """End-to-end extraction: parse, split, crop, caption, export.

    All collaborators are injectable for testing and replacement; defaults
    are constructed from ``config``.

    Args:
        config: Framework configuration (defaults applied when omitted).
        parser: PDF parser (figures, captions, marker panels).
        panel_detector: Vision detector for the ``"vlm-assisted"`` mode.
            Constructed lazily, so deterministic runs need no model
            credentials; pass an instance to inject a custom backend.
        crop_engine: Deterministic crop-validation engine.
    """

    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        parser: Optional[ScientificPDFParser] = None,
        panel_detector: Optional[PanelDetector] = None,
        crop_engine: Optional[CropEngine] = None,
    ) -> None:
        self.config = config or ExtractionConfig()
        self._parser = parser or ScientificPDFParser(
            resolution=self.config.resolution_dpi
        )
        self._panel_detector = panel_detector
        self._debate_detector: Optional[PanelDetector] = None
        self._debate_detector_failed = False
        self._crop_engine = crop_engine or CropEngine(
            padding_px=self.config.crop_padding_px,
            max_expansion_px=self.config.crop_max_expansion_px,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def extract(self, pdf_path: Union[str, Path]) -> ParsedDocument:
        """Extracts figures and panels from a PDF.

        Args:
            pdf_path: The PDF to process.

        Returns:
            The parsed document with panels attached according to the
            configured strategy.
        """
        document = self._parser.parse(pdf_path)

        if self.config.panel_cropping == "markers-only":
            return document  # PDF-marker panels only (done in the parser)

        parents = [f for f in document.figures if f.parent_id is None]
        children: Dict[str, List[ExtractedFigure]] = {}
        for figure in document.figures:
            if figure.parent_id is not None:
                children.setdefault(figure.parent_id, []).append(figure)

        augmented: List[ExtractedFigure] = []
        seen_ids = {figure.figure_id for figure in document.figures}
        for parent in parents:
            existing = children.get(parent.figure_id, [])
            expected_letters = sorted(split_caption(parent.caption).keys())
            # The caption is ground truth for the panel count: split further
            # when the marker tier found nothing, or fewer panels than the
            # caption describes (missed markers, merged panels).
            short = bool(expected_letters) and len(existing) < len(expected_letters)
            panels = existing
            if not existing or short:
                if existing:
                    logger.info(
                        "%s: markers found %d panel(s) but the caption "
                        "describes %d — trying raster%s segmentation",
                        parent.figure_id,
                        len(existing),
                        len(expected_letters),
                        "/vision" if self.config.use_vlm else "",
                    )
                trial_seen = seen_ids - {p.figure_id for p in existing}
                replacement: List[ExtractedFigure] = []
                # Tier 2 (deterministic): whitespace/line-gutter segmentation
                # of the figure raster — pixel-exact, only trusted when the
                # cell count matches the caption's panel count exactly.
                if self.config.use_raster_gutters:
                    replacement = self._raster_split(
                        parent, expected_letters, trial_seen
                    )
                # Tier 3 (vision-assisted, opt-in): Python proposes candidate
                # grids and the model AUDITS them (per-cell panel counting);
                # its report drives deterministic edits until clean. The
                # model never produces coordinates.
                if not replacement and self.config.use_vlm:
                    replacement, _ = self._verified_grid_panels(
                        parent, expected_letters, trial_seen
                    )
                if replacement:
                    panels = replacement
                    seen_ids = trial_seen | {p.figure_id for p in replacement}
            augmented.append(parent)
            augmented.extend(panels)

        self._log_vision_cost()
        return ParsedDocument(
            source_path=document.source_path,
            pages=document.pages,
            figures=augmented,
        )

    def _log_vision_cost(self) -> None:
        """Logs cumulative vision-model token usage and estimated USD cost.

        Sums usage across the panel detector and its debate partner, so a
        single run answers "what did this document cost". Local (Ollama)
        models report zero cost; cloud models report the real spend.
        """
        detectors = [
            detector
            for detector in (self._panel_detector, self._debate_detector)
            if detector is not None
        ]
        total_calls = 0
        total_in = 0
        total_out = 0
        total_cost = 0.0
        for detector in detectors:
            usage = detector.backend.usage()
            total_calls += int(usage["calls"])
            total_in += int(usage["input_tokens"])
            total_out += int(usage["output_tokens"])
            total_cost += float(usage["cost_usd"])
        if total_calls == 0:
            return
        logger.info(
            "Vision cost this run: %d call(s), %d input + %d output tokens, "
            "estimated $%.4f USD (local models are $0)",
            total_calls,
            total_in,
            total_out,
            total_cost,
        )

    def extract_to_dataset(
        self,
        pdf_path: Union[str, Path],
        output_dir: Union[str, Path],
        paper_name: Optional[str] = None,
    ) -> Path:
        """Extracts a PDF and writes the structured dataset to disk.

        Args:
            pdf_path: The PDF to process.
            output_dir: Root directory for the dataset.
            paper_name: Dataset folder name; defaults to the PDF's stem.

        Returns:
            The written paper directory.
        """
        document = self.extract(pdf_path)
        exporter = DatasetExporter(Path(output_dir))
        return exporter.export(document, paper_name=paper_name)

    # ------------------------------------------------------------------ #
    # Panel letter assignment                                             #
    # ------------------------------------------------------------------ #

    def _letters_for_cells(
        self,
        parent: ExtractedFigure,
        cells: list,
        expected_letters: List[str],
        overlay: Optional[object] = None,
    ) -> List[str]:
        """Determines each cell's panel letter, visually when possible.

        Reading order (the default) is wrong for figures labeled column by
        column, so when a vision backend is available the printed labels are
        READ from the figure instead (one cheap model call). Any unusable
        answer falls back to reading order.

        Args:
            parent: The complete figure.
            cells: The final panel cells, in reading order.
            expected_letters: Panel letters from the caption, in order.
            overlay: A pre-drawn numbered overlay of ``cells`` to reuse;
                drawn on demand when omitted.

        Returns:
            One letter per cell.
        """
        default = list(expected_letters)
        # In the deterministic default (no VLM assistance) reading order is
        # used directly — the correct panel→caption mapping is always
        # provided as metadata regardless of crop labels. Visual label
        # reading (for column-major layouts) is only worthwhile with a
        # capable model, so it is gated behind the opt-in VLM mode.
        if not self.config.use_vlm:
            return default
        detector = self._get_detector()
        if detector is None:
            return default
        try:
            from src.extraction.raster_splitter import draw_grid_overlay

            if overlay is None:
                overlay = draw_grid_overlay(parent.image, cells)
            mapping = detector.map_labels(
                parent.image, overlay, expected_letters, len(cells)
            )
        except NotImplementedError:
            return default
        except Exception:
            logger.exception("Label mapping failed for %s", parent.figure_id)
            return default
        if mapping:
            if mapping != default:
                logger.info(
                    "%s: visual label reading reordered the panels: %s",
                    parent.figure_id,
                    ", ".join(mapping),
                )
            return mapping
        return default

    # ------------------------------------------------------------------ #
    # Raster fallback (deterministic)                                     #
    # ------------------------------------------------------------------ #

    def _raster_split(
        self,
        parent: ExtractedFigure,
        expected_letters: List[str],
        seen_ids: set,
    ) -> List[ExtractedFigure]:
        """Splits a figure by whitespace gutters in its rendered image.

        Only trusted when the number of segmented cells matches the number
        of panels the caption describes exactly — then cells are labeled
        positionally in reading order. Any mismatch returns ``[]`` so the
        caller escalates to the vision model.

        Args:
            parent: The complete figure.
            expected_letters: Panel letters from the caption, in order.
            seen_ids: Figure ids already taken.

        Returns:
            One panel per expected letter, or ``[]``.
        """
        if len(expected_letters) < 2:
            return []
        try:
            from src.extraction.raster_splitter import split_by_gutters

            cells = split_by_gutters(parent.image)
        except Exception:
            logger.exception("Raster gutter segmentation failed for %s", parent.figure_id)
            return []
        if len(cells) != len(expected_letters):
            if cells:
                logger.info(
                    "%s: gutter segmentation found %d cell(s) but the "
                    "caption describes %d — not trusted",
                    parent.figure_id,
                    len(cells),
                    len(expected_letters),
                )
            return []

        segments: Dict[str, str] = split_caption(parent.caption)
        letters = self._letters_for_cells(parent, cells, expected_letters)
        panels: List[ExtractedFigure] = []
        for letter, cell in zip(letters, cells):
            figure_id = f"figure-{parent.number.lower()}{letter}"
            if figure_id in seen_ids:
                return []
            refined = self._crop_engine.refine(parent.image, cell)
            image = parent.image.crop(refined.as_tuple())
            if image.width < 20 or image.height < 20:
                return []
            panels.append(
                ExtractedFigure(
                    figure_id=figure_id,
                    label=f"Figure {parent.number}({letter})",
                    number=f"{parent.number.lower()}{letter}",
                    caption=segments.get(letter) or parent.caption,
                    page_number=parent.page_number,
                    bbox=self._pixel_to_points(parent, refined),
                    image=image,
                    detection_method=parent.detection_method,
                    dpi=parent.dpi,
                    parent_id=parent.figure_id,
                    panel=letter,
                    render_bbox=None,
                    origin="raster",
                )
            )
        panels.sort(key=lambda p: p.panel or "")
        logger.info(
            "%s: raster gutter segmentation split it into %d panel(s): %s",
            parent.figure_id,
            len(panels),
            ", ".join(panel.panel or "?" for panel in panels),
        )
        return panels

    # ------------------------------------------------------------------ #
    # Propose-verify grids (Python proposes, the model only judges)       #
    # ------------------------------------------------------------------ #

    def _verified_grid_panels(
        self,
        parent: ExtractedFigure,
        expected_letters: List[str],
        seen_ids: set,
    ) -> "Tuple[List[ExtractedFigure], bool]":
        """Splits a figure with a model-audited, Python-proposed grid.

        Flow (all coordinates stay in Python):

        1. The caption fixes the panel count.
        2. ``candidate_grids`` scans the image and generates deterministic
           tilings matching that count (gutter-based, band-based,
           aspect-estimated bands, uniform).
        3. Each candidate is drawn as a NUMBERED red grid on the figure and
           the vision model compares original vs. overlay, reporting how
           many panels each cell contains — auditing, a task models perform
           far more reliably than coordinate estimation.
        4. The first grid whose every cell counts exactly one panel is
           cropped (shrink-only refinement; there are no clean gutters to
           expand into on these figures).

        Args:
            parent: The complete figure.
            expected_letters: Panel letters from the caption, in order.
            seen_ids: Figure ids already taken.

        Returns:
            ``(panels, verifier_ran)`` — ``panels`` holds one entry per
            expected letter when a grid was approved (else empty), and
            ``verifier_ran`` reports whether the audit actually executed,
            so the caller can distinguish "rejected everything" (do NOT
            escalate to box estimates) from "could not run".
        """
        if len(expected_letters) < 2:
            return [], False
        detectors = self._debate_detectors()
        if not detectors:
            return [], False
        try:
            from src.extraction.raster_splitter import (
                candidate_grids,
                draw_grid_overlay,
                grid_cut_cost,
                refine_grid_with_audit,
                snap_grid,
            )

            candidates = candidate_grids(parent.image, len(expected_letters))
        except Exception:
            logger.exception("Grid proposal failed for %s", parent.figure_id)
            return [], False
        if not candidates:
            return [], False

        # Snap every candidate's internal lines to the nearest low-ink
        # valley (real panel boundaries), then rank by how much content the
        # lines cut through and discard hopeless ones — the verifier is
        # only as good as the best proposal it is shown first.
        scored = []
        for cells in candidates:
            cells = snap_grid(parent.image, cells)
            cost = grid_cut_cost(parent.image, cells)
            if cost > 0.4:
                logger.info(
                    "%s: grid candidate discarded (cut cost %.2f)",
                    parent.figure_id,
                    cost,
                )
                continue
            scored.append((cost, cells))
        if not scored:
            return [], False
        scored.sort(key=lambda item: item[0])
        logger.info(
            "%s: %d grid candidate(s), cut costs: %s",
            parent.figure_id,
            len(scored),
            ", ".join(f"{cost:.2f}" for cost, _ in scored),
        )

        # --- Debate loop: models alternate as auditor; Python edits. ------ #
        # Each round the current auditor counts panels per cell. A clean
        # report is one sign-off; two consecutive sign-offs (both models,
        # or the same model twice when solo) accept the grid immediately.
        # A dirty report becomes deterministic edits (valley splits/merges)
        # and roles rotate. If rounds run out without dual consensus, the
        # grid that earned at least one clean sign-off AND has the lowest
        # geometric cut cost is accepted — better than keeping the figure
        # whole when a model already approved a low-cost split.
        verifier_ran = False
        best_signed: Optional[tuple] = None  # (cut_cost, grid, overlay)
        starting_grids = [cells for _, cells in scored[:2]]
        for start_index, grid in enumerate(starting_grids, start=1):
            approvals = 0
            for round_index in range(self.config.max_debate_rounds):
                auditor = detectors[round_index % len(detectors)]
                overlay = draw_grid_overlay(parent.image, grid)
                try:
                    audit = auditor.audit_grid(
                        parent.image, overlay, expected_letters, len(grid)
                    )
                except NotImplementedError:
                    logger.debug("Backend cannot audit grids — skipping tier")
                    return [], verifier_ran
                except Exception as exc:
                    logger.warning(
                        "Grid audit failed for %s (%s) — skipping tier",
                        parent.figure_id,
                        exc,
                    )
                    return [], verifier_ran
                if audit is None:
                    logger.info(
                        "%s: unusable audit from %s — abandoning grid %d",
                        parent.figure_id,
                        auditor.backend.name,
                        start_index,
                    )
                    break
                verifier_ran = True

                if audit.is_clean(self.config.verify_min_confidence):
                    approvals += 1
                    cost = grid_cut_cost(parent.image, grid)
                    if best_signed is None or cost < best_signed[0]:
                        best_signed = (cost, list(grid), overlay)
                    logger.info(
                        "%s: %s signed off the grid (%d/%d sign-offs, "
                        "cut cost %.2f)",
                        parent.figure_id,
                        auditor.backend.name,
                        approvals,
                        len(detectors),
                        cost,
                    )
                    if approvals >= len(detectors):
                        panels = self._build_grid_panels(
                            parent, grid, expected_letters, seen_ids, overlay
                        )
                        if panels:
                            logger.info(
                                "%s: debate accepted grid %d after %d "
                                "round(s) — split into %d panel(s)",
                                parent.figure_id,
                                start_index,
                                round_index + 1,
                                len(panels),
                            )
                            return panels, True
                        break
                    continue  # same grid, next auditor judges

                approvals = 0
                issue_map = {issue.cell: issue.problem for issue in audit.issues}
                logger.info(
                    "%s: %s reports counts=%s issues=%s — editing the grid",
                    parent.figure_id,
                    auditor.backend.name,
                    [round(count, 1) for count in audit.cell_counts],
                    issue_map or "-",
                )
                edited = refine_grid_with_audit(
                    parent.image,
                    grid,
                    audit.cell_counts,
                    issue_map,
                    len(expected_letters),
                )
                if not edited:
                    logger.info(
                        "%s: audit not convertible into a valid %d-cell "
                        "grid — abandoning grid %d",
                        parent.figure_id,
                        len(expected_letters),
                        start_index,
                    )
                    break
                edited = snap_grid(parent.image, edited)
                if edited == grid:
                    break  # stalemate: same grid resubmitted
                grid = edited

        # Tiebreak: a single confident, low-cut-cost sign-off is enough.
        if best_signed is not None and best_signed[0] <= 0.15:
            cost, grid, overlay = best_signed
            panels = self._build_grid_panels(
                parent, grid, expected_letters, seen_ids, overlay
            )
            if panels:
                logger.info(
                    "%s: no dual consensus — accepting best single-signed "
                    "grid (cut cost %.2f) into %d panel(s)",
                    parent.figure_id,
                    cost,
                    len(panels),
                )
                return panels, True
        return [], verifier_ran

    def _build_grid_panels(
        self,
        parent: ExtractedFigure,
        cells: list,
        expected_letters: List[str],
        seen_ids: set,
        overlay: object,
    ) -> List[ExtractedFigure]:
        """Crops an accepted grid into panel figures (letters read visually)."""
        segments: Dict[str, str] = split_caption(parent.caption)
        letters = self._letters_for_cells(
            parent, cells, expected_letters, overlay=overlay
        )
        taken = set(seen_ids)
        panels: List[ExtractedFigure] = []
        for letter, cell in zip(letters, cells):
            figure_id = f"figure-{parent.number.lower()}{letter}"
            if figure_id in taken:
                return []
            refined = self._crop_engine.tighten(parent.image, cell)
            image = parent.image.crop(refined.as_tuple())
            if image.width < 20 or image.height < 20:
                return []
            taken.add(figure_id)
            panels.append(
                ExtractedFigure(
                    figure_id=figure_id,
                    label=f"Figure {parent.number}({letter})",
                    number=f"{parent.number.lower()}{letter}",
                    caption=segments.get(letter) or parent.caption,
                    page_number=parent.page_number,
                    bbox=self._pixel_to_points(parent, refined),
                    image=image,
                    detection_method=parent.detection_method,
                    dpi=parent.dpi,
                    parent_id=parent.figure_id,
                    panel=letter,
                    render_bbox=None,
                    origin="vlm",
                )
            )
        panels.sort(key=lambda p: p.panel or "")
        return panels

    # ------------------------------------------------------------------ #
    # Detector construction                                               #
    # ------------------------------------------------------------------ #

    def _get_detector(self) -> Optional[PanelDetector]:
        """Builds the panel detector lazily (needs model credentials)."""
        if self._panel_detector is None:
            try:
                from src.vision.backend import create_vision_backend

                self._panel_detector = PanelDetector(
                    backend=create_vision_backend(
                        provider=self.config.provider, model=self.config.model
                    ),
                    min_confidence=self.config.min_confidence,
                )
            except Exception as exc:
                logger.warning(
                    "Vision panel detector unavailable (%s) — geometric "
                    "results only",
                    exc,
                )
                return None
        return self._panel_detector

    def _debate_detectors(self) -> List[PanelDetector]:
        """Returns the auditors for the grid debate.

        Two detectors (the primary model plus ``debate_model`` /
        ``VLM_DEBATE_MODEL``) when configured and constructible — they then
        alternate roles each round and BOTH must sign a grid off; otherwise
        the primary detector debates against itself.
        """
        primary = self._get_detector()
        if primary is None:
            return []
        detectors = [primary]
        model = (self.config.debate_model or os.getenv("VLM_DEBATE_MODEL") or "").strip()
        if model and not self._debate_detector_failed:
            if self._debate_detector is None:
                try:
                    from src.vision.backend import create_vision_backend

                    self._debate_detector = PanelDetector(
                        backend=create_vision_backend(
                            provider=self.config.provider, model=model
                        ),
                        min_confidence=self.config.min_confidence,
                    )
                    logger.info("Debate auditor ready: %s", model)
                except Exception as exc:
                    logger.warning(
                        "Debate model '%s' unavailable (%s) — single-model "
                        "debate",
                        model,
                        exc,
                    )
                    self._debate_detector_failed = True
            if self._debate_detector is not None:
                detectors.append(self._debate_detector)
        return detectors

    @staticmethod
    def _pixel_to_points(parent: ExtractedFigure, box: PixelBox) -> BoundingBox:
        """Maps a pixel box in the parent's image back to PDF point space.

        Uses the parent's ``render_bbox`` (the exact page region its pixels
        correspond to) when available; otherwise falls back to the parent's
        nominal bbox, which is accurate up to the render padding.
        """
        anchor = parent.render_bbox or parent.bbox
        scale = 72.0 / float(parent.dpi)  # px → pt (1 pt = dpi/72 px)
        return BoundingBox(
            x0=anchor.x0 + box.x0 * scale,
            top=anchor.top + box.y0 * scale,
            x1=anchor.x0 + box.x1 * scale,
            bottom=anchor.top + box.y1 * scale,
        )
