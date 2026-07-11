"""Deterministic pixel-space crop engine with validation and expansion.

The vision model only *proposes* panel bounding boxes; this module owns every
actual image operation — padding, boundary validation, bounded expansion, and
the final crop. Nothing here calls a model, and the source figure is never
resized or modified: crops are pixel-exact windows into the original image.

Validation criterion: a crop is valid when its border strips contain only
background pixels — i.e. no axis, label, legend, colorbar, or any other
graphical object touches the crop boundary. Dirty sides are expanded stepwise
(within a configurable budget) until clean, which mechanically guarantees
"nothing owned by the panel is truncated" for figures on a light background.
"""

from __future__ import annotations

import logging
from typing import Dict

from PIL import Image

from src.models import PixelBox

logger = logging.getLogger(__name__)


class CropEngine:
    """Validates, expands, and applies panel crops on figure images.

    Args:
        padding_px: Initial padding added around every proposed box — a
            slightly generous crop beats one that cuts scientific content.
        expand_step_px: How many pixels a dirty side grows per round.
        max_expansion_px: Maximum total expansion per side beyond the initial
            padded box. When the budget is exhausted the (possibly still
            imperfect) box is used as-is — the vision model is never
            re-invoked just to enlarge a crop.
        background_threshold: Grayscale level (0–255) at or above which a
            pixel counts as background. 245 tolerates JPEG noise and
            anti-aliasing on white paper backgrounds.
    """

    def __init__(
        self,
        padding_px: int = 12,
        expand_step_px: int = 8,
        max_expansion_px: int = 120,
        background_threshold: int = 245,
    ) -> None:
        if expand_step_px < 1:
            raise ValueError("expand_step_px must be >= 1.")
        self.padding_px = padding_px
        self.expand_step_px = expand_step_px
        self.max_expansion_px = max_expansion_px
        self.background_threshold = background_threshold

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def refine(self, figure: Image.Image, box: PixelBox) -> PixelBox:
        """Tightens a proposed box to its ink, then expands to clean borders.

        The tighten step first shrinks the box to the bounding box of the
        actual content inside it — vision models often return boxes with
        substantial slack, and slack on one side usually means truncation on
        another once panels are laid out in grids. The expansion step then
        grows any side whose border still cuts through content.

        Args:
            figure: The complete figure image (never modified).
            box: The proposed panel box, in ``figure``'s pixel space.

        Returns:
            The validated (or budget-capped) box, clamped to the image.
        """
        width, height = figure.size
        gray = figure.convert("L")
        current = box.padded(self.padding_px, width, height)

        # Tighten to the ink bounding box inside the proposal, then re-pad.
        region = gray.crop(current.as_tuple())
        ink = region.point(
            lambda v: 0 if v >= self.background_threshold else 255
        ).getbbox()
        if ink is not None:
            current = PixelBox(
                x0=current.x0 + ink[0],
                y0=current.y0 + ink[1],
                x1=current.x0 + ink[2],
                y1=current.y0 + ink[3],
            ).padded(self.padding_px, width, height)

        # Per-side expansion budgets, so one busy side cannot consume the
        # whole allowance of the others.
        budget: Dict[str, int] = {side: self.max_expansion_px for side in "ltrb"}
        x0, y0, x1, y1 = current.x0, current.y0, current.x1, current.y1

        changed = True
        while changed:
            changed = False
            # Left border strip: 1 px wide, spanning the crop's height.
            if x0 > 0 and budget["l"] > 0 and not self._strip_clean(gray, (x0, y0, x0 + 1, y1)):
                step = min(self.expand_step_px, budget["l"], x0)
                x0 -= step
                budget["l"] -= step
                changed = True
            if y0 > 0 and budget["t"] > 0 and not self._strip_clean(gray, (x0, y0, x1, y0 + 1)):
                step = min(self.expand_step_px, budget["t"], y0)
                y0 -= step
                budget["t"] -= step
                changed = True
            if x1 < width and budget["r"] > 0 and not self._strip_clean(gray, (x1 - 1, y0, x1, y1)):
                step = min(self.expand_step_px, budget["r"], width - x1)
                x1 += step
                budget["r"] -= step
                changed = True
            if y1 < height and budget["b"] > 0 and not self._strip_clean(gray, (x0, y1 - 1, x1, y1)):
                step = min(self.expand_step_px, budget["b"], height - y1)
                y1 += step
                budget["b"] -= step
                changed = True

        refined = PixelBox(x0=x0, y0=y0, x1=x1, y1=y1).clamped(width, height)
        if refined != box:
            logger.debug("Refined crop %s -> %s", box.as_tuple(), refined.as_tuple())
        return refined

    def tighten(self, figure: Image.Image, box: PixelBox) -> PixelBox:
        """Shrinks a box to the ink it contains — WITHOUT any expansion.

        For grid cells that tile a gutterless figure (adjacent flat-color
        panels, shared frames), border-cleanliness expansion would grow into
        the neighbouring panel; the safe refinement is shrink-only.

        Args:
            figure: The complete figure image (never modified).
            box: The proposed cell, in ``figure``'s pixel space.

        Returns:
            The ink bounding box inside ``box`` with a 2-px margin, or the
            clamped input when the cell is empty.
        """
        width, height = figure.size
        current = box.clamped(width, height)
        if current.width <= 0 or current.height <= 0:
            return current
        region = figure.convert("L").crop(current.as_tuple())
        ink = region.point(
            lambda v: 0 if v >= self.background_threshold else 255
        ).getbbox()
        if ink is None:
            return current
        return PixelBox(
            x0=current.x0 + ink[0],
            y0=current.y0 + ink[1],
            x1=current.x0 + ink[2],
            y1=current.y0 + ink[3],
        ).padded(2, width, height)

    def crop(self, figure: Image.Image, box: PixelBox) -> Image.Image:
        """Applies :meth:`refine` and returns the cropped image.

        The result preserves the source's resolution and pixels exactly —
        no resampling ever happens here.
        """
        refined = self.refine(figure, box)
        return figure.crop(refined.as_tuple())

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _strip_clean(self, gray: Image.Image, strip_box: tuple) -> bool:
        """True when every pixel in the strip is background-bright.

        ``getextrema`` on the 1-px strip is O(strip length) in C — cheap
        enough to call repeatedly during expansion.
        """
        left, upper, right, lower = strip_box
        if right <= left or lower <= upper:
            return True
        lo, _ = gray.crop(strip_box).getextrema()
        return lo >= self.background_threshold
