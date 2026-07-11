"""Vision-model grid auditing for the opt-in ``vlm-assisted`` crop mode.

The input is always ONE complete, already-extracted figure image — never a
full PDF page. The vision model is used purely as a *verifier*: Python
proposes a candidate segmentation grid drawn on the figure, and the model
audits it (counts panels per numbered cell) or reads the printed panel
labels. The model never produces coordinates — every crop boundary is
computed deterministically. Any unusable response is treated as "cannot
confirm", so the caller keeps the whole figure. Incorrect segmentation is
worse than no segmentation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from PIL import Image

from src.models import CellIssue, GridAudit
from src.vision.backend import VisionBackend, create_vision_backend

logger = logging.getLogger(__name__)


_VERIFY_SYSTEM_PROMPT = """\
You are a strict auditor of panel-segmentation grids for multi-panel
scientific figures.

You receive TWO images: (1) the ORIGINAL figure, and (2) the SAME figure
with a proposed segmentation drawn on top as thick RED rectangles, each
carrying a red NUMBER badge (1, 2, 3, ...) at its top-left corner.

For EVERY numbered red cell, count how many panels it contains, comparing
against the original image:
- exactly one complete panel (its plot/image plus its axes, tick labels,
  and annotations) -> 1
- two or more panels (even partially) -> 2, 3, ...
- only a fragment of a panel that a red line cuts through -> 0.5
- empty (background only) -> 0

A correct segmentation yields 1 for every cell. Be skeptical: if a red line
crosses any plot area or an axis, the affected cells are NOT 1.

Answer with STRICT JSON only (no markdown fences, no prose):
{"cell_counts": [<one number per cell, in badge order>],
 "confidence": 0.0-1.0,
 "issues": [{"cell": <badge number>,
             "problem": "two_panels_side_by_side" | "two_panels_stacked" |
                        "more_than_two_panels" | "fragment_of_panel" |
                        "empty"}]}
Include one "issues" entry for every cell whose count is not exactly 1, and
none otherwise.
"""


_LABEL_MAP_SYSTEM_PROMPT = """\
You read the panel labels of a multi-panel scientific figure.

You receive TWO images: (1) the ORIGINAL figure, and (2) the SAME figure
with numbered RED cells drawn on top (a badge 1, 2, 3, ... at each cell's
top-left corner).

For EVERY numbered cell, in badge order, report the panel label letter
printed inside that cell in the original figure: "a" for a label "(a)",
"b" for "(b)", and so on. Figures may be labeled column by column instead
of row by row — report what is actually printed, never assume an order.
Use "?" when no label is visible in a cell.

Answer with STRICT JSON only (no markdown fences, no prose):
{"cell_labels": ["a", "e", ...]}
"""


class PanelDetector:
    """Audits Python-proposed segmentation grids with a vision backend.

    Args:
        backend: The vision backend to use; ``None`` builds the default one
            from the environment (``VLM_PROVIDER`` / ``VLM_MODEL``).
        min_confidence: Retained for API compatibility (unused by the audit
            path).
        max_transport_edge_px: Longest edge for single-image transport.
        max_audit_edge_px: Longest edge for TWO-image requests (audits and
            label reading). Local runtimes ship small default context
            windows (Ollama: 4096 tokens), and two 1600-px images overflow
            them — 1000 px keeps a two-image request comfortably inside
            while the numbered badges stay legible.
    """

    def __init__(
        self,
        backend: Optional[VisionBackend] = None,
        min_confidence: float = 0.6,
        max_transport_edge_px: int = 1600,
        max_audit_edge_px: int = 1000,
    ) -> None:
        self._backend = backend or create_vision_backend()
        self._min_confidence = min_confidence
        self._max_transport_edge_px = max_transport_edge_px
        self._max_audit_edge_px = max_audit_edge_px

    @property
    def backend(self) -> VisionBackend:
        """The underlying vision backend."""
        return self._backend

    def audit_grid(
        self,
        original: Image.Image,
        overlay: Image.Image,
        expected_letters: List[str],
        cell_count: int,
    ) -> Optional[GridAudit]:
        """Audits a drawn segmentation grid via per-cell panel counting.

        The model must report how many panels each numbered cell contains —
        a concrete counting task that is much harder to wave through than a
        global yes/no, and far more reliable than producing coordinates —
        plus a problem tag for every non-clean cell, which the deterministic
        grid editor uses as an orientation hint.
        Both images are downscaled identically for transport; no coordinate
        ever crosses the model boundary.

        Args:
            original: The complete figure at original resolution.
            overlay: The same figure with the NUMBERED candidate grid drawn
                in red (see ``draw_grid_overlay``).
            expected_letters: Panel letters the caption describes.
            cell_count: Number of cells in the candidate grid.

        Returns:
            The parsed audit, or ``None`` when the response is unusable
            (never raises for model misbehavior).

        Raises:
            NotImplementedError: When the backend cannot carry two images.
        """
        transport_original, _ = self._prepare_transport(
            original, self._max_audit_edge_px
        )
        transport_overlay, _ = self._prepare_transport(
            overlay, self._max_audit_edge_px
        )
        letters = ", ".join(f"({letter})" for letter in expected_letters)
        user_prompt = (
            "The first image is the original figure; the second is the same "
            f"figure with {cell_count} numbered RED cells proposing a "
            f"segmentation into the panels labeled {letters}. Report the "
            "panel count of every numbered cell as strict JSON."
        )
        raw = self._backend.invoke_images(
            [transport_original, transport_overlay],
            _VERIFY_SYSTEM_PROMPT,
            user_prompt,
        )

        payload = self._extract_json(raw)
        if payload is None:
            logger.debug("Grid audit returned unparsable output: %.200s", raw)
            return None
        try:
            confidence = max(0.0, min(1.0, float(payload.get("confidence", 1.0))))
        except (TypeError, ValueError):
            confidence = 0.0

        counts = payload.get("cell_counts")
        if not isinstance(counts, list):
            # Fallback for models that answer the older global form.
            if payload.get("coherent") is True:
                return GridAudit(
                    cell_counts=[1.0] * cell_count, confidence=confidence
                )
            return None
        try:
            values = [float(value) for value in counts]
        except (TypeError, ValueError):
            return None
        if len(values) != cell_count:
            logger.debug(
                "Grid audit reported %d count(s) for %d cell(s)",
                len(values),
                cell_count,
            )
            return None

        issues: List[CellIssue] = []
        for entry in payload.get("issues") or []:
            if not isinstance(entry, dict):
                continue
            try:
                cell = int(entry.get("cell", 0))
            except (TypeError, ValueError):
                continue
            if 1 <= cell <= cell_count:
                issues.append(
                    CellIssue(cell=cell, problem=str(entry.get("problem") or ""))
                )
        return GridAudit(cell_counts=values, issues=issues, confidence=confidence)

    def verify_grid(
        self,
        original: Image.Image,
        overlay: Image.Image,
        expected_letters: List[str],
        cell_count: int,
        min_confidence: float = 0.7,
    ) -> bool:
        """Convenience wrapper: True when :meth:`audit_grid` comes back clean."""
        audit = self.audit_grid(original, overlay, expected_letters, cell_count)
        return audit is not None and audit.is_clean(min_confidence)

    def map_labels(
        self,
        original: Image.Image,
        overlay: Image.Image,
        expected_letters: List[str],
        cell_count: int,
    ) -> Optional[List[str]]:
        """Reads which panel letter is printed inside each numbered cell.

        Journals sometimes label panels column by column, so the letter of a
        cell cannot be inferred from reading order alone — but *reading* a
        printed label is a trivial task for a vision model. The mapping is
        only trusted when it is an exact permutation of the expected
        letters.

        Args:
            original: The complete figure at original resolution.
            overlay: The same figure with the NUMBERED approved grid drawn
                in red (see ``draw_grid_overlay``).
            expected_letters: Panel letters the caption describes.
            cell_count: Number of cells in the grid.

        Returns:
            One letter per cell in badge order, or ``None`` when the answer
            is missing, malformed, or not a permutation of the expected
            letters (callers then fall back to reading order).

        Raises:
            NotImplementedError: When the backend cannot carry two images.
        """
        transport_original, _ = self._prepare_transport(
            original, self._max_audit_edge_px
        )
        transport_overlay, _ = self._prepare_transport(
            overlay, self._max_audit_edge_px
        )
        letters = ", ".join(f"({letter})" for letter in expected_letters)
        user_prompt = (
            f"The second image has {cell_count} numbered RED cells. The "
            f"panel labels present in this figure are {letters}. Report "
            "each cell's label letter as strict JSON."
        )
        try:
            raw = self._backend.invoke_images(
                [transport_original, transport_overlay],
                _LABEL_MAP_SYSTEM_PROMPT,
                user_prompt,
            )
        except NotImplementedError:
            raise
        except Exception as exc:
            logger.warning("Label mapping request failed: %s", exc)
            return None

        payload = self._extract_json(raw)
        if payload is None:
            return None
        labels = payload.get("cell_labels")
        if not isinstance(labels, list) or len(labels) != cell_count:
            return None
        normalized = [
            str(label).strip().lower().strip("()") for label in labels
        ]
        if sorted(normalized) != sorted(letter.lower() for letter in expected_letters):
            logger.debug(
                "Label mapping is not a permutation of the expected letters: %s",
                normalized,
            )
            return None
        return normalized

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _prepare_transport(
        self, image: Image.Image, max_edge: Optional[int] = None
    ) -> "tuple[Image.Image, float]":
        """Downscales for transport; returns the image and the scale-back factor.

        The original figure is never modified — the resized copy exists only
        for the network payload, and every returned coordinate is multiplied
        by ``scale_back`` to land in the original pixel space.

        Args:
            image: The image to prepare.
            max_edge: Longest-edge cap; defaults to the single-image size.
                Two-image requests pass the smaller audit size so the pair
                fits inside small local context windows.
        """
        limit = max_edge or self._max_transport_edge_px
        longest = max(image.size)
        if longest <= limit:
            return image, 1.0
        scale = limit / float(longest)
        transport = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        return transport, 1.0 / scale

    @staticmethod
    def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
        """Extracts the first JSON object from possibly-noisy model output."""
        text = raw.strip()
        # Strip markdown fences the model may add despite instructions.
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
