"""Deterministic raster panel segmentation via whitespace-gutter profiles.

For figures shipped as a single embedded bitmap, the PDF exposes no vector
objects to cluster — but the rendered image itself reveals the panel grid:
panels are separated by horizontal/vertical bands of background pixels
(gutters). This module finds those bands with projection profiles (pure PIL,
no ML, pixel-exact):

1. Threshold the figure into an ink mask.
2. Project onto the y-axis: rows containing ink form bands; gaps of
   background rows wider than the gutter threshold separate panel rows.
3. Within each row band, project onto the x-axis the same way to find the
   panel columns.
4. Tighten every cell to its actual ink bounding box.

Cells come out in reading order (top-to-bottom, left-to-right), ready to be
matched positionally against the panel letters the caption describes. When
the counts disagree the caller should treat the result as unreliable and
fall back to vision-model detection.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from src.models import PixelBox

logger = logging.getLogger(__name__)


def _ink_profile(mask: Image.Image, axis: str) -> List[float]:
    """Mean ink (0..255) per position along ``axis`` over the cross-section.

    Args:
        mask: An ink mask (255 = content, 0 = background).
        axis: ``"x"`` for a per-column profile, ``"y"`` for per-row.
    """
    if axis == "x":
        return list(mask.resize((mask.width, 1), Image.Resampling.BOX).getdata())
    return list(mask.resize((1, mask.height), Image.Resampling.BOX).getdata())


def find_separators(
    profile: List[float],
    length: int,
    min_run: int,
    gutter_max_ink: float = 6.0,
    gutter_min_width: int = 6,
    line_min_ink: float = 90.0,
    line_baseline_margin: float = 45.0,
    line_max_width_frac: float = 0.04,
) -> List[Tuple[int, int]]:
    """Splits a 1-D projection into content segments at gutters AND lines.

    Panel boundaries in scientific figures take two visual forms, and dense
    figures (arrow fields, framed plots) often have neither pure white nor
    pure black:

    * **Whitespace gutters** — a band where ink stays near zero (white
      margin between panels).
    * **Separator/frame lines** — a thin band where ink SPIKES well above the
      surrounding baseline (a black frame border or grid rule, darker than
      the sparse panel content on either side).

    Both are located and used as cut positions, so the same routine handles
    white-separated panels, framed plots, and dense spin-texture panels
    ringed by black borders.

    Args:
        profile: Per-position mean ink (0..255).
        length: ``len(profile)`` (the axis length in pixels).
        min_run: Minimum content-segment length to keep.
        gutter_max_ink: A position at/below this counts as background.
        gutter_min_width: Minimum width of a background band to be a gutter.
        line_min_ink: A separator line must be at least this dark.
        line_baseline_margin: ...and this much darker than the local median
            baseline, so ordinary dark content is not mistaken for a line.
        line_max_width_frac: Lines wider than this fraction of ``length`` are
            treated as content (flat-colour panels), not separators.

    Returns:
        Half-open ``(start, end)`` content segments in ascending order.
    """
    if length <= 0:
        return []
    cut = [False] * length

    # --- Whitespace gutters ------------------------------------------------ #
    index = 0
    while index < length:
        if profile[index] <= gutter_max_ink:
            end = index
            while end < length and profile[end] <= gutter_max_ink:
                end += 1
            if end - index >= gutter_min_width:
                for position in range(index, end):
                    cut[position] = True
            index = end
        else:
            index += 1

    # --- Separator / frame lines (local spikes above baseline) ------------- #
    ordered = sorted(profile)
    baseline = ordered[len(ordered) // 2]  # median ink of the region
    max_line_width = max(2, int(length * line_max_width_frac))
    index = 0
    while index < length:
        if (
            profile[index] >= line_min_ink
            and profile[index] >= baseline + line_baseline_margin
        ):
            end = index
            while (
                end < length
                and profile[end] >= line_min_ink
                and profile[end] >= baseline + line_baseline_margin
            ):
                end += 1
            if end - index <= max_line_width:
                for position in range(index, end):
                    cut[position] = True
            index = end
        else:
            index += 1

    # --- Content segments between cut regions ------------------------------ #
    segments: List[Tuple[int, int]] = []
    start = -1
    for position in range(length):
        if not cut[position]:
            if start < 0:
                start = position
        elif start >= 0:
            segments.append((start, position))
            start = -1
    if start >= 0:
        segments.append((start, length))
    return [seg for seg in segments if seg[1] - seg[0] >= min_run]


def _content_runs(
    profile: List[int],
    min_gap: int,
    min_run: int,
    ink_threshold: int = 2,
    solid_threshold: int = 250,
    max_solid_separator: int = 12,
) -> List[Tuple[int, int]]:
    """Finds runs of ink positions in a 1-D projection profile.

    Two kinds of separators split runs: *background* positions (whitespace
    gutters) and *thin fully-solid* runs — panels that touch often share a
    thin frame border, which shows up as a few consecutive positions of
    near-total ink spanning the whole cross-section. Wide solid regions
    (flat-color panels, colorbars) are content, not separators, which is why
    the solid rule is capped at ``max_solid_separator`` positions.

    Args:
        profile: Per-position mean ink values (0 = background, 255 = solid
            ink) as produced by a BOX-filter resize of the ink mask.
        min_gap: Background gaps shorter than this are treated as part of
            the surrounding content (intra-panel whitespace), so only real
            gutters split runs.
        min_run: Runs shorter than this are discarded (stray specks).
        ink_threshold: Minimum mean value for a position to count as ink —
            2/255 tolerates a thin 1-px line spanning ~1% of the axis.
        solid_threshold: Mean value at or above which a position counts as
            fully solid (a candidate separator line).
        max_solid_separator: Maximum width of a solid run to be treated as
            a shared border line rather than panel content. 12 px covers
            rules up to ~3 pt at 300 DPI while staying well below the width
            of flat-color panels and colorbars.

    Returns:
        Half-open ``(start, end)`` runs in ascending order.
    """
    values = list(profile)
    total = len(values)
    # Convert thin fully-solid runs (shared frame borders) into HARD gaps —
    # unlike ordinary narrow whitespace, these must never be merged back.
    hard = [False] * total
    index = 0
    while index < total:
        if values[index] >= solid_threshold:
            end = index
            while end < total and values[end] >= solid_threshold:
                end += 1
            if end - index <= max_solid_separator:
                for position in range(index, end):
                    values[position] = 0
                    hard[position] = True
            index = end
        else:
            index += 1

    runs: List[Tuple[int, int]] = []
    start: int = -1
    for index, value in enumerate(values):
        if value > ink_threshold:
            if start < 0:
                start = index
        elif start >= 0:
            runs.append((start, index))
            start = -1
    if start >= 0:
        runs.append((start, total))

    # Merge runs separated by less than a gutter — except across hard
    # separator lines.
    merged: List[Tuple[int, int]] = []
    for run in runs:
        if (
            merged
            and run[0] - merged[-1][1] < min_gap
            and not any(hard[merged[-1][1] : run[0]])
        ):
            merged[-1] = (merged[-1][0], run[1])
        else:
            merged.append(run)
    return [run for run in merged if run[1] - run[0] >= min_run]


def split_by_gutters(
    image: Image.Image,
    row_gutter_px: int = 12,
    col_gutter_px: int = 8,
    min_panel_px: int = 40,
    background_threshold: int = 245,
    max_cells: int = 16,
) -> List[PixelBox]:
    """Segments a figure image into panel cells along whitespace gutters.

    Args:
        image: The complete figure at its extracted resolution.
        row_gutter_px: Minimum height of a background band separating two
            panel rows (at 300 DPI, 12 px is about 3 pt).
        col_gutter_px: Minimum width of a background band separating two
            panels within a row (columns are often tighter than rows).
        min_panel_px: Minimum plausible panel edge; smaller runs/cells are
            discarded.
        background_threshold: Grayscale level at or above which a pixel is
            background.
        max_cells: Sanity cap — more cells than this means the segmentation
            latched onto texture, not panels, and ``[]`` is returned.

    Returns:
        Tightened cell boxes in reading order; empty when the image cannot
        be segmented into at least two plausible cells.
    """
    width, height = image.size
    if width < 2 * min_panel_px and height < 2 * min_panel_px:
        return []

    gray = image.convert("L")
    # Ink mask: 255 where there is content, 0 where background.
    mask = gray.point(lambda v: 0 if v >= background_threshold else 255)

    # y-projection: mean ink per row (BOX filter averages exactly).
    row_profile = list(mask.resize((1, height), Image.Resampling.BOX).getdata())
    row_runs = _content_runs(row_profile, min_gap=row_gutter_px, min_run=min_panel_px)
    if not row_runs:
        return []

    cells: List[PixelBox] = []
    for band_top, band_bottom in row_runs:
        band = mask.crop((0, band_top, width, band_bottom))
        col_profile = list(
            band.resize((width, 1), Image.Resampling.BOX).getdata()
        )
        col_runs = _content_runs(
            col_profile, min_gap=col_gutter_px, min_run=min_panel_px
        )
        for band_left, band_right in col_runs:
            # Tighten the cell to its actual ink extent (a cell's content
            # may not span the full row band height).
            cell_mask = mask.crop((band_left, band_top, band_right, band_bottom))
            ink_bbox = cell_mask.getbbox()
            if ink_bbox is None:
                continue
            cell = PixelBox(
                x0=band_left + ink_bbox[0],
                y0=band_top + ink_bbox[1],
                x1=band_left + ink_bbox[2],
                y1=band_top + ink_bbox[3],
            )
            if cell.width < min_panel_px or cell.height < min_panel_px:
                continue
            cells.append(cell)

    if len(cells) < 2 or len(cells) > max_cells:
        logger.debug(
            "Gutter segmentation yielded %d cell(s) — not usable", len(cells)
        )
        return []
    logger.debug("Gutter segmentation found %d cell(s)", len(cells))
    return cells


def row_bands(
    image: Image.Image,
    row_gutter_px: int = 10,
    min_band_px: int = 40,
    background_threshold: int = 245,
) -> List[Tuple[int, int]]:
    """Finds the horizontal content bands (panel rows) of a figure image.

    Args:
        image: The complete figure.
        row_gutter_px: Minimum background gap separating two bands.
        min_band_px: Minimum band height.
        background_threshold: Grayscale level at or above which a pixel is
            background.

    Returns:
        ``(top, bottom)`` bands in ascending order (possibly a single band).
    """
    width, height = image.size
    mask = image.convert("L").point(
        lambda v: 0 if v >= background_threshold else 255
    )
    profile = list(mask.resize((1, height), Image.Resampling.BOX).getdata())
    return _content_runs(profile, min_gap=row_gutter_px, min_run=min_band_px)


def xy_cut_cells(
    image: Image.Image,
    row_gutter_px: int = 10,
    col_gutter_px: int = 6,
    min_panel_px: int = 40,
    background_threshold: int = 245,
    max_depth: int = 4,
    max_cells: int = 16,
) -> List[PixelBox]:
    """Decomposes a figure by recursive XY-cuts along gutters/border lines.

    Unlike the one-level ``split_by_gutters``, this handles hierarchical
    layouts such as "one tall panel on the left, a 2x3 grid on the right":
    each region is split along whichever axis exposes a separator (row
    bands first, then columns), and its parts are decomposed recursively.
    The depth-first traversal order matches how such figures are labeled
    ((a) tall-left first, then the right block row by row).

    Args:
        image: The complete figure.
        row_gutter_px: Minimum horizontal separator height.
        col_gutter_px: Minimum vertical separator width.
        min_panel_px: Minimum leaf edge; smaller leaves are dropped.
        background_threshold: Grayscale level at or above which a pixel is
            background.
        max_depth: Recursion bound.
        max_cells: Sanity cap on the leaf count.

    Returns:
        Ink-tightened leaf boxes in label order; empty when decomposition
        yields fewer than 1 or more than ``max_cells`` leaves.
    """
    width, height = image.size
    mask = image.convert("L").point(
        lambda v: 0 if v >= background_threshold else 255
    )

    def region_runs(box: PixelBox, axis: str) -> List[Tuple[int, int]]:
        """Content runs of a sub-region along one axis, absolute coords.

        Uses :func:`find_separators`, which cuts at BOTH whitespace gutters
        and spike-shaped frame/grid lines — essential for dense figures
        whose panels are ringed by black borders (no white gutter exists).
        """
        sub = mask.crop(box.as_tuple())
        if axis == "y":
            profile = _ink_profile(sub, "y")
            runs = find_separators(
                profile, sub.height, min_run=min_panel_px, gutter_min_width=row_gutter_px
            )
            return [(box.y0 + start, box.y0 + end) for start, end in runs]
        profile = _ink_profile(sub, "x")
        runs = find_separators(
            profile, sub.width, min_run=min_panel_px, gutter_min_width=col_gutter_px
        )
        return [(box.x0 + start, box.x0 + end) for start, end in runs]

    def tightened(box: PixelBox) -> Optional[PixelBox]:
        ink = mask.crop(box.as_tuple()).getbbox()
        if ink is None:
            return None
        leaf = PixelBox(
            x0=box.x0 + ink[0],
            y0=box.y0 + ink[1],
            x1=box.x0 + ink[2],
            y1=box.y0 + ink[3],
        )
        if leaf.width < min_panel_px or leaf.height < min_panel_px:
            return None
        return leaf

    def recurse(box: PixelBox, depth: int) -> List[PixelBox]:
        if depth > 0:
            bands = region_runs(box, "y")
            if len(bands) > 1:
                leaves: List[PixelBox] = []
                for top, bottom in bands:
                    leaves.extend(
                        recurse(
                            PixelBox(x0=box.x0, y0=top, x1=box.x1, y1=bottom),
                            depth - 1,
                        )
                    )
                return leaves
            columns = region_runs(box, "x")
            if len(columns) > 1:
                leaves = []
                for left, right in columns:
                    leaves.extend(
                        recurse(
                            PixelBox(x0=left, y0=box.y0, x1=right, y1=box.y1),
                            depth - 1,
                        )
                    )
                return leaves
        leaf = tightened(box)
        return [leaf] if leaf is not None else []

    leaves = recurse(PixelBox(x0=0, y0=0, x1=width, y1=height), max_depth)
    if not leaves or len(leaves) > max_cells:
        return []
    return leaves


def complete_to_count(
    leaves: List[PixelBox],
    expected_count: int,
    target_aspect: float = 1.25,
) -> Optional[List[PixelBox]]:
    """Expands a leaf decomposition to exactly ``expected_count`` cells.

    When XY-cutting stops early (adjacent panels with no separator at all,
    e.g. touching flat-color maps), the merged leaf is conspicuously
    elongated. The extra cuts are allocated by exhaustive search over all
    per-leaf split counts, minimizing how far every resulting cell's aspect
    ratio strays from a typical panel's (``target_aspect``) — a greedy
    allocator ties on near-equal elongations and cuts the wrong leaf in
    half. Splits are applied uniformly along each leaf's long axis,
    preserving the traversal (label) order.

    Args:
        leaves: XY-cut leaves in label order.
        expected_count: Panel count the caption describes.
        target_aspect: Width/height ratio a plausible panel has.

    Returns:
        Exactly ``expected_count`` cells, or ``None`` when the leaf count
        already exceeds it (over-segmentation cannot be reconciled here).
    """
    if not leaves or len(leaves) > expected_count:
        return None
    if len(leaves) == expected_count:
        return list(leaves)

    def piece_cost(leaf: PixelBox, pieces: int) -> float:
        """Total implausibility of splitting ``leaf`` into ``pieces``."""
        width = max(1.0, float(leaf.width))
        height = max(1.0, float(leaf.height))
        # Splits run along the long axis, so each piece's aspect is:
        aspect = (width / pieces) / height if width >= height else width * pieces / height
        return pieces * abs(math.log(max(aspect, 1e-6) / target_aspect))

    total_leaves = len(leaves)
    max_pieces = expected_count - total_leaves + 1
    best_cost = math.inf
    best_counts: Optional[List[int]] = None

    def search(index: int, remaining: int, counts: List[int], cost: float) -> None:
        nonlocal best_cost, best_counts
        if cost >= best_cost:
            return  # prune
        if index == total_leaves - 1:
            if 1 <= remaining <= max_pieces:
                final = cost + piece_cost(leaves[index], remaining)
                if final < best_cost:
                    best_cost = final
                    best_counts = counts + [remaining]
            return
        leaves_after = total_leaves - 1 - index
        for pieces in range(1, min(max_pieces, remaining - leaves_after) + 1):
            search(
                index + 1,
                remaining - pieces,
                counts + [pieces],
                cost + piece_cost(leaves[index], pieces),
            )

    search(0, expected_count, [], 0.0)
    if best_counts is None:
        return None

    cells: List[PixelBox] = []
    for leaf, count in zip(leaves, best_counts):
        if count == 1:
            cells.append(leaf)
            continue
        if leaf.width >= leaf.height:
            for part in range(count):
                cells.append(
                    PixelBox(
                        x0=leaf.x0 + round(leaf.width * part / count),
                        y0=leaf.y0,
                        x1=leaf.x0 + round(leaf.width * (part + 1) / count),
                        y1=leaf.y1,
                    )
                )
        else:
            for part in range(count):
                cells.append(
                    PixelBox(
                        x0=leaf.x0,
                        y0=leaf.y0 + round(leaf.height * part / count),
                        x1=leaf.x1,
                        y1=leaf.y0 + round(leaf.height * (part + 1) / count),
                    )
                )
    return cells


def reading_order(cells: List[PixelBox]) -> List[PixelBox]:
    """Sorts panel cells into label (reading) order.

    Journals label panels row by row, left to right — except that a panel
    spanning several rows (a tall schematic on the left) is labeled at the
    position of the FIRST row it touches. Accordingly: normal-height cells
    are clustered into rows by their vertical center, each spanning cell
    (>1.5x the median height) is inserted into the first row it overlaps,
    and rows are emitted top-to-bottom with cells sorted by x.

    Args:
        cells: Panel cells in any order.

    Returns:
        The same cells in reading order.
    """
    if len(cells) <= 1:
        return list(cells)
    heights = sorted(cell.height for cell in cells)
    median_height = max(1, heights[len(heights) // 2])

    normal = [cell for cell in cells if cell.height <= 1.5 * median_height]
    spanning = [cell for cell in cells if cell.height > 1.5 * median_height]

    rows: List[List[PixelBox]] = []
    centers: List[float] = []
    for cell in sorted(normal, key=lambda c: (c.y0 + c.y1) / 2.0):
        center = (cell.y0 + cell.y1) / 2.0
        if rows and abs(center - centers[-1]) <= 0.5 * median_height:
            rows[-1].append(cell)
        else:
            rows.append([cell])
            centers.append(center)

    for cell in sorted(spanning, key=lambda c: c.y0):
        target = None
        for row in rows:
            row_top = min(c.y0 for c in row)
            row_bottom = max(c.y1 for c in row)
            if cell.y0 < row_bottom and cell.y1 > row_top:
                target = row
                break
        if target is None:
            rows.append([cell])
        else:
            target.append(cell)

    rows.sort(key=lambda row: min(c.y0 for c in row))
    return [cell for row in rows for cell in sorted(row, key=lambda c: c.x0)]


def _uniform_grid(width: int, height: int, rows: int, cols: int) -> List[PixelBox]:
    """Builds a uniform ``rows x cols`` tiling of the image, reading order."""
    cells: List[PixelBox] = []
    for row in range(rows):
        top = round(height * row / rows)
        bottom = round(height * (row + 1) / rows)
        for col in range(cols):
            cells.append(
                PixelBox(
                    x0=round(width * col / cols),
                    y0=top,
                    x1=round(width * (col + 1) / cols),
                    y1=bottom,
                )
            )
    return cells


def candidate_grids(
    image: Image.Image, expected_count: int, max_candidates: int = 6
) -> List[List[PixelBox]]:
    """Generates candidate panel tilings matching the caption's panel count.

    Candidates, most-informed first (all deterministic; a vision model then
    only has to VERIFY the drawn grid, never produce coordinates):

    1. Whitespace-gutter segmentation, strict and progressively relaxed —
       exact when gutters exist.
    2. Detected horizontal content bands with a uniform column split per
       band — handles rows separated by whitespace whose panels touch.
    3. Uniform ``rows x cols`` factorizations of the expected count,
       ordered by how square their cells would be — the shape of typical
       equal-sized panel layouts.

    Every candidate tiles the figure without overlaps and has exactly
    ``expected_count`` cells in reading order.

    Args:
        image: The complete figure.
        expected_count: Number of panels the caption describes.
        max_candidates: Cap on returned candidates (each may cost one
            vision-model verification call).

    Returns:
        Deduplicated candidate tilings, best-first.
    """
    if expected_count < 2:
        return []
    width, height = image.size
    signatures: set = set()
    candidates: List[List[PixelBox]] = []

    def push(cells: List[PixelBox]) -> None:
        if len(cells) != expected_count:
            return
        signature = tuple(
            (cell.x0 // 25, cell.y0 // 25, cell.x1 // 25, cell.y1 // 25)
            for cell in cells
        )
        if signature in signatures:
            return
        signatures.add(signature)
        candidates.append(cells)

    # 0. Recursive XY-cut, completed to the expected count by aspect-based
    #    uniform splits of elongated leaves — the most expressive proposal
    #    (handles "tall panel + grid" layouts and touching panels).
    for row_gutter, col_gutter, threshold in ((10, 6, 245), (6, 4, 250)):
        leaves = xy_cut_cells(
            image,
            row_gutter_px=row_gutter,
            col_gutter_px=col_gutter,
            background_threshold=threshold,
        )
        completed = complete_to_count(leaves, expected_count)
        if completed:
            push(reading_order(completed))

    # 1. One-level gutter segmentation, strict → relaxed.
    for row_gutter, col_gutter, threshold in ((12, 8, 245), (8, 5, 245), (5, 3, 250)):
        push(
            split_by_gutters(
                image,
                row_gutter_px=row_gutter,
                col_gutter_px=col_gutter,
                background_threshold=threshold,
            )
        )

    # 2. Detected row bands + uniform columns within each band.
    bands = row_bands(image)

    def band_cells(column_counts: List[int]) -> List[PixelBox]:
        cells: List[PixelBox] = []
        for (band_top, band_bottom), cols in zip(bands, column_counts):
            for col in range(cols):
                cells.append(
                    PixelBox(
                        x0=round(width * col / cols),
                        y0=band_top,
                        x1=round(width * (col + 1) / cols),
                        y1=band_bottom,
                    )
                )
        return cells

    if bands and expected_count % len(bands) == 0:
        push(band_cells([expected_count // len(bands)] * len(bands)))

    # 2b. Row bands with per-band column counts estimated from each band's
    # aspect ratio (panels are roughly square), adjusted until the total
    # matches the caption. This expresses irregular layouts such as one
    # wide panel on top of two rows of three (1+3+3).
    if bands and 2 <= len(bands) <= expected_count:
        def aspect(index: int) -> float:
            band_top, band_bottom = bands[index]
            return width / max(1.0, float(band_bottom - band_top))

        estimates = [
            max(1, min(expected_count, round(aspect(index))))
            for index in range(len(bands))
        ]
        for _ in range(4 * expected_count):  # bounded adjustment loop
            total = sum(estimates)
            if total == expected_count:
                break
            if total < expected_count:
                # Add a column where panels are still widest vs. estimate.
                index = max(
                    range(len(bands)), key=lambda k: aspect(k) / estimates[k]
                )
                estimates[index] += 1
            else:
                shrinkable = [k for k in range(len(bands)) if estimates[k] > 1]
                if not shrinkable:
                    break
                index = min(shrinkable, key=lambda k: aspect(k) / estimates[k])
                estimates[index] -= 1
        if sum(estimates) == expected_count:
            push(band_cells(estimates))

    # 3. Uniform factorizations, most plausible (squarest cells) first.
    factorizations = [
        (rows, expected_count // rows)
        for rows in range(1, expected_count + 1)
        if expected_count % rows == 0
    ]
    factorizations.sort(
        key=lambda rc: abs(math.log((width / rc[1]) / (height / rc[0])))
    )
    for rows, cols in factorizations:
        push(_uniform_grid(width, height, rows, cols))

    return candidates[:max_candidates]


def split_cell_at_valley(
    image: Image.Image,
    cell: PixelBox,
    vertical_cut: bool,
    background_threshold: int = 245,
) -> Tuple[PixelBox, PixelBox]:
    """Splits a cell in two at its interior ink minimum.

    Used by the audit-driven grid editor when a model reports that a cell
    contains two panels: the model only says WHERE the problem is; the cut
    position is found deterministically as the least-ink line in the cell's
    central region (falling back to the midpoint for flat-color content).

    Args:
        image: The complete figure.
        cell: The cell to split.
        vertical_cut: True to cut with a vertical line (panels side by
            side), False for a horizontal line (panels stacked).
        background_threshold: Grayscale level at or above which a pixel is
            background.

    Returns:
        The two sub-cells in reading order.
    """
    mask = image.convert("L").point(
        lambda v: 0 if v >= background_threshold else 255
    )
    strip = mask.crop(cell.as_tuple())
    extent = strip.width if vertical_cut else strip.height
    lo = max(1, int(extent * 0.2))
    hi = min(extent - 1, int(extent * 0.8))
    position = extent // 2
    if hi > lo:
        if vertical_cut:
            profile = list(strip.resize((strip.width, 1), Image.Resampling.BOX).getdata())
        else:
            profile = list(strip.resize((1, strip.height), Image.Resampling.BOX).getdata())
        interior = profile[lo:hi]
        minimum = min(interior)
        # Ties resolve toward the center (uniform panels split evenly).
        candidates = [lo + i for i, v in enumerate(interior) if v == minimum]
        position = min(candidates, key=lambda p: abs(p - extent // 2))
    if vertical_cut:
        cut = cell.x0 + position
        return (
            PixelBox(x0=cell.x0, y0=cell.y0, x1=cut, y1=cell.y1),
            PixelBox(x0=cut, y0=cell.y0, x1=cell.x1, y1=cell.y1),
        )
    cut = cell.y0 + position
    return (
        PixelBox(x0=cell.x0, y0=cell.y0, x1=cell.x1, y1=cut),
        PixelBox(x0=cell.x0, y0=cut, x1=cell.x1, y1=cell.y1),
    )


def _merge_fragment(edited: List[PixelBox], position: int) -> bool:
    """Merges cell ``position`` into its longest-shared-edge neighbour."""
    fragment = edited[position]
    best_index = -1
    best_shared = 0.0
    for index, other in enumerate(edited):
        if index == position:
            continue
        if abs(other.x1 - fragment.x0) <= 3 or abs(other.x0 - fragment.x1) <= 3:
            shared = min(other.y1, fragment.y1) - max(other.y0, fragment.y0)
        elif abs(other.y1 - fragment.y0) <= 3 or abs(other.y0 - fragment.y1) <= 3:
            shared = min(other.x1, fragment.x1) - max(other.x0, fragment.x0)
        else:
            continue
        if shared > best_shared:
            best_shared = shared
            best_index = index
    if best_index < 0:
        return False
    other = edited[best_index]
    edited[best_index] = PixelBox(
        x0=min(other.x0, fragment.x0),
        y0=min(other.y0, fragment.y0),
        x1=max(other.x1, fragment.x1),
        y1=max(other.y1, fragment.y1),
    )
    edited.pop(position)
    return True


def refine_grid_with_audit(
    image: Image.Image,
    cells: List[PixelBox],
    cell_counts: List[float],
    issue_map: dict,
    expected_count: int,
) -> Optional[List[PixelBox]]:
    """Applies an auditor's per-cell report as deterministic grid edits.

    Auditor counts are noisy and often do not sum to ``expected_count``
    (e.g. ``[1, 2, 0.5, 2, 1, 1, 1]`` for 7). Rather than demand exact
    arithmetic, this applies the reported edits — split cells counted ≥ 1.5
    at their ink valleys (orientation from the issue tag); merge cells
    counted ≤ 0.5 into a neighbour — and then RECONCILES the result to
    exactly ``expected_count`` deterministically: if too many cells, merge
    the smallest adjacent pairs; if too few, split the most elongated cell
    at its valley. The result always has ``expected_count`` cells, and the
    next audit round judges it. The models steer; Python never invents a
    coordinate.

    Args:
        image: The complete figure.
        cells: The audited grid, badge order.
        cell_counts: The auditor's per-cell panel counts.
        issue_map: 1-based badge index -> problem tag.
        expected_count: Panel count the caption describes.

    Returns:
        A grid of exactly ``expected_count`` plausible cells in reading
        order, or ``None`` only when no plausible reconciliation exists.
    """
    edited: List[PixelBox] = []
    merge_positions: List[int] = []
    for index, cell in enumerate(cells):
        count = cell_counts[index] if index < len(cell_counts) else 1.0
        problem = str(issue_map.get(index + 1, ""))
        if count >= 1.5:
            pieces = max(2, min(4, int(round(count))))
            if "side_by_side" in problem:
                vertical = True
            elif "stacked" in problem:
                vertical = False
            else:
                vertical = cell.width >= cell.height
            parts = [cell]
            while len(parts) < pieces:
                largest = max(parts, key=lambda c: c.width * c.height)
                parts.remove(largest)
                first, second = split_cell_at_valley(image, largest, vertical)
                parts.extend([first, second])
            edited.extend(reading_order(parts))
        elif count <= 0.5:
            merge_positions.append(len(edited))
            edited.append(cell)
        else:
            edited.append(cell)

    for position in sorted(merge_positions, reverse=True):
        _merge_fragment(edited, position)  # best-effort; reconciled below

    # --- Reconcile to exactly expected_count --------------------------------- #
    guard = 0
    while len(edited) > expected_count and guard < 4 * expected_count:
        guard += 1
        smallest = min(
            range(len(edited)), key=lambda k: edited[k].width * edited[k].height
        )
        if not _merge_fragment(edited, smallest):
            # No mergeable neighbour: drop the tiniest cell outright.
            edited.pop(smallest)
    while len(edited) < expected_count and guard < 8 * expected_count:
        guard += 1
        target = max(
            range(len(edited)),
            key=lambda k: max(
                edited[k].width / max(1.0, edited[k].height),
                edited[k].height / max(1.0, edited[k].width),
            )
            * (edited[k].width * edited[k].height),
        )
        cell = edited.pop(target)
        first, second = split_cell_at_valley(
            image, cell, vertical_cut=cell.width >= cell.height
        )
        edited.extend([first, second])

    if len(edited) != expected_count:
        return None
    if any(cell.width < 20 or cell.height < 20 for cell in edited):
        return None
    return reading_order(edited)


def snap_grid(
    image: Image.Image,
    cells: List[PixelBox],
    max_shift_frac: float = 0.3,
    background_threshold: int = 245,
    min_cell_px: int = 20,
) -> List[PixelBox]:
    """Snaps every internal grid line to the nearest low-ink valley.

    Candidate grids built from uniform assumptions land mid-panel whenever
    panels have unequal widths (axis labels, colorbars). Real panel
    boundaries are local minima of ink along the cut direction, so each
    shared internal boundary is moved — within a window proportional to the
    neighbouring cells — to the least-ink position, ties resolved toward
    the original location. Both neighbours are updated together, keeping
    the tiling consistent.

    Args:
        image: The complete figure.
        cells: The candidate tiling.
        max_shift_frac: Search window as a fraction of the smaller
            neighbouring cell's extent.
        background_threshold: Grayscale level at or above which a pixel is
            background.
        min_cell_px: A snap is rejected if it would shrink a cell below
            this.

    Returns:
        A new cell list with snapped boundaries (same order).
    """
    width, height = image.size
    mask = image.convert("L").point(
        lambda v: 0 if v >= background_threshold else 255
    )
    snapped = list(cells)

    def best_valley(lo: int, hi: int, origin: int, box: tuple, axis: str) -> int:
        """Least-ink position in [lo, hi); ties go to the nearest to origin."""
        strip = mask.crop(box)
        if strip.width <= 0 or strip.height <= 0:
            return origin
        size = (hi - lo, 1) if axis == "x" else (1, hi - lo)
        profile = list(strip.resize(size, Image.Resampling.BOX).getdata())
        minimum = min(profile)
        candidates = [lo + i for i, v in enumerate(profile) if v == minimum]
        return min(candidates, key=lambda p: abs(p - origin))

    for i in range(len(snapped)):
        for j in range(len(snapped)):
            if i == j:
                continue
            a, b = snapped[i], snapped[j]
            # Vertical boundary: a's right edge meets b's left edge.
            if abs(a.x1 - b.x0) <= 2:
                shared_y0 = max(a.y0, b.y0)
                shared_y1 = min(a.y1, b.y1)
                if shared_y1 - shared_y0 < 0.5 * min(a.height, b.height):
                    continue
                window = int(min(a.width, b.width) * max_shift_frac)
                if window < 4:
                    continue
                origin = a.x1
                lo = max(a.x0 + min_cell_px, origin - window)
                hi = min(b.x1 - min_cell_px, origin + window)
                if hi <= lo:
                    continue
                new_x = best_valley(
                    lo, hi, origin, (lo, shared_y0, hi, shared_y1), "x"
                )
                snapped[i] = PixelBox(x0=a.x0, y0=a.y0, x1=new_x, y1=a.y1)
                snapped[j] = PixelBox(x0=new_x, y0=b.y0, x1=b.x1, y1=b.y1)
            # Horizontal boundary: a's bottom edge meets b's top edge.
            elif abs(a.y1 - b.y0) <= 2:
                shared_x0 = max(a.x0, b.x0)
                shared_x1 = min(a.x1, b.x1)
                if shared_x1 - shared_x0 < 0.5 * min(a.width, b.width):
                    continue
                window = int(min(a.height, b.height) * max_shift_frac)
                if window < 4:
                    continue
                origin = a.y1
                lo = max(a.y0 + min_cell_px, origin - window)
                hi = min(b.y1 - min_cell_px, origin + window)
                if hi <= lo:
                    continue
                new_y = best_valley(
                    lo, hi, origin, (shared_x0, lo, shared_x1, hi), "y"
                )
                snapped[i] = PixelBox(x0=a.x0, y0=a.y0, x1=a.x1, y1=new_y)
                snapped[j] = PixelBox(x0=b.x0, y0=new_y, x1=b.x1, y1=b.y1)

    return snapped


def grid_cut_cost(
    image: Image.Image,
    cells: List[PixelBox],
    background_threshold: int = 245,
    chunk_px: int = 32,
) -> float:
    """Measures how much panel CONTENT a candidate grid's lines cut through.

    Every internal cell edge is sampled as a thin strip, subdivided into
    chunks. A chunk is a legitimate cut location when it is background
    (whitespace gutter, mean ink ≈ 0) or fully solid (a shared frame border
    or the junction of two flat-color panels, mean ink ≈ 255); anything in
    between means the line crosses real content (plot texture, curves,
    labels). The returned cost is the fraction of bad chunks — ~0 for a
    correct segmentation, high for grids that bisect panels.

    Used to rank candidates BEFORE any vision-model verification (and to
    discard hopeless ones), so an over-lenient verifier is shown the most
    plausible grid first.

    Args:
        image: The complete figure.
        cells: The candidate tiling.
        background_threshold: Grayscale level at or above which a pixel is
            background.
        chunk_px: Sampling granularity along each edge.

    Returns:
        Bad-chunk fraction in [0, 1]; 0.0 when the grid has no internal
        edges.
    """
    width, height = image.size
    mask = image.convert("L").point(
        lambda v: 0 if v >= background_threshold else 255
    )
    bad = 0
    total = 0

    def sample_edge(x0: int, y0: int, x1: int, y1: int) -> None:
        nonlocal bad, total
        strip_box = (max(0, x0), max(0, y0), min(width, x1), min(height, y1))
        if strip_box[2] <= strip_box[0] or strip_box[3] <= strip_box[1]:
            return
        strip = mask.crop(strip_box)
        horizontal = strip.width >= strip.height
        chunks = max(1, (strip.width if horizontal else strip.height) // chunk_px)
        size = (chunks, 1) if horizontal else (1, chunks)
        for mean in strip.resize(size, Image.Resampling.BOX).getdata():
            total += 1
            if 12 < mean < 243:
                bad += 1

    for cell in cells:
        if cell.x0 > 2:
            sample_edge(cell.x0 - 1, cell.y0, cell.x0 + 2, cell.y1)
        if cell.x1 < width - 2:
            sample_edge(cell.x1 - 2, cell.y0, cell.x1 + 1, cell.y1)
        if cell.y0 > 2:
            sample_edge(cell.x0, cell.y0 - 1, cell.x1, cell.y0 + 2)
        if cell.y1 < height - 2:
            sample_edge(cell.x0, cell.y1 - 2, cell.x1, cell.y1 + 1)

    return bad / total if total else 0.0


def draw_grid_overlay(
    image: Image.Image,
    cells: List[PixelBox],
    color: Tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """Draws a candidate tiling as thick, NUMBERED colored outlines.

    Each cell gets a visible index badge (1, 2, ...) so the vision verifier
    can report a per-cell panel count instead of a single global verdict.
    The source image is never modified.

    Args:
        image: The complete figure.
        cells: The candidate tiling, in reading order.
        color: Outline color (default red — rare in scientific plots'
            surroundings and easy for the verifier to reference).

    Returns:
        A new RGB image with the numbered grid drawn on top.
    """
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    line_width = max(4, min(image.size) // 150)
    font_size = max(28, min(image.size) // 25)
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:  # pragma: no cover - very old Pillow
        font = ImageFont.load_default()

    for index, cell in enumerate(cells, start=1):
        # Keep the outline inside the image so border cells stay visible.
        draw.rectangle(
            (
                max(cell.x0, line_width // 2),
                max(cell.y0, line_width // 2),
                min(cell.x1, image.width - 1 - line_width // 2),
                min(cell.y1, image.height - 1 - line_width // 2),
            ),
            outline=color,
            width=line_width,
        )
        # Index badge on a white patch at the cell's top-left corner.
        label = str(index)
        text_box = draw.textbbox((0, 0), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        badge_x = cell.x0 + line_width + 2
        badge_y = cell.y0 + line_width + 2
        draw.rectangle(
            (badge_x, badge_y, badge_x + text_w + 10, badge_y + text_h + 10),
            fill=(255, 255, 255),
        )
        draw.text(
            (badge_x + 5 - text_box[0], badge_y + 5 - text_box[1]),
            label,
            fill=color,
            font=font,
        )
    return overlay
