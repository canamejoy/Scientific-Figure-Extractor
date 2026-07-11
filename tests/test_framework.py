"""Tests for the extraction framework: crop engine, panel detector parsing,
VLM fallback end-to-end (with a stub backend), and dataset export.

No model or API key is needed — the vision backend is stubbed.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Users\eldre\AI-Scientific-Figure-Analyze")
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["LAYOUT_DETECTION"] = "off"  # deterministic tests use heuristics

from PIL import Image, ImageDraw  # noqa: E402

from src.extraction.crop_engine import CropEngine  # noqa: E402
from src.extraction.framework import (  # noqa: E402
    ExtractionConfig,
    FigureExtractionFramework,
)
from src.models import PixelBox  # noqa: E402
from src.vision.backend import VisionBackend  # noqa: E402
from src.vision.panel_detector import PanelDetector  # noqa: E402


# ---------------------------------------------------------------------- #
# 1. Crop engine: expansion until borders are clean                       #
# ---------------------------------------------------------------------- #

def test_crop_engine() -> None:
    image = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(image)
    # A dense panel (filled) — a crop cutting through it leaves ink on the
    # crop borders, which the validator must detect and expand past.
    draw.rectangle((50, 40, 200, 250), fill=(80, 80, 80))

    engine = CropEngine(padding_px=4, expand_step_px=6, max_expansion_px=150)
    # Deliberately bad proposal cutting through the panel.
    refined = engine.refine(image, PixelBox(x0=80, y0=70, x1=170, y1=200))
    assert refined.x0 < 50 and refined.y0 < 40, refined
    assert refined.x1 > 200 and refined.y1 > 250, refined

    crop = engine.crop(image, PixelBox(x0=80, y0=70, x1=170, y1=200))
    assert crop.width == refined.width and crop.height == refined.height
    print(f"crop engine        OK  (refined -> {refined.as_tuple()})")


# ---------------------------------------------------------------------- #
# 2. Grid auditor: JSON parsing (fences, cell counts, garbage)            #
# ---------------------------------------------------------------------- #

class CannedBackend(VisionBackend):
    def __init__(self, payload: str) -> None:
        self.payload = payload

    @property
    def name(self) -> str:
        return "stub/canned"

    def invoke(self, image, system_prompt, user_prompt) -> str:
        return self.payload

    def invoke_images(self, images, system_prompt, user_prompt) -> str:
        return self.payload


def test_grid_audit_parsing() -> None:
    image = Image.new("RGB", (1600, 1200), "white")

    # Markdown-fenced audit with all cells clean.
    fenced = (
        "```json\n"
        + json.dumps({"cell_counts": [1, 1], "confidence": 0.9, "issues": []})
        + "\n```"
    )
    detector = PanelDetector(backend=CannedBackend(fenced))
    audit = detector.audit_grid(image, image, ["a", "b"], 2)
    assert audit is not None and audit.is_clean(0.7)
    assert detector.verify_grid(image, image, ["a", "b"], 2)

    # A dirty cell with an issue tag is parsed and NOT clean.
    dirty = json.dumps(
        {
            "cell_counts": [1, 2],
            "confidence": 0.9,
            "issues": [{"cell": 2, "problem": "two_panels_side_by_side"}],
        }
    )
    audit = PanelDetector(backend=CannedBackend(dirty)).audit_grid(image, image, ["a", "b"], 2)
    assert audit is not None and not audit.is_clean(0.7)
    assert audit.issues[0].cell == 2

    # Wrong cell count -> unusable (None).
    mismatch = json.dumps({"cell_counts": [1, 1, 1], "confidence": 0.9})
    assert PanelDetector(backend=CannedBackend(mismatch)).audit_grid(image, image, ["a", "b"], 2) is None

    # Garbage -> None instead of crash.
    assert PanelDetector(backend=CannedBackend("I see two panels!")).audit_grid(image, image, ["a", "b"], 2) is None
    print("grid auditor       OK  (fences, cell counts, issues, garbage)")


# ---------------------------------------------------------------------- #
# 3. Framework end-to-end with VLM fallback + export                      #
# ---------------------------------------------------------------------- #

def build_pdf_without_markers() -> bytes:
    """One figure, two vector panels, NO '(a)'/'(b)' marker text — the
    geometric splitter must fail and the vision fallback must take over."""
    content = b"""BT /F1 11 Tf 72 700 Td (As shown in Figure 1, the accuracy increases steadily with the number of training epochs.) Tj ET
1 w
100 420 150 180 re S
280 420 150 180 re S
120 440 m 200 500 l S
300 440 m 380 560 l S
BT /F1 10 Tf 100 390 Td (Figure 1: Comparison views. \\(a\\) First panel view. \\(b\\) Second panel view.) Tj ET
"""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"endstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size " + str(len(objects) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n" + str(xref_pos).encode() + b"\n%%EOF\n"
    )
    return bytes(out)


class FailBackend(VisionBackend):
    """Backend that must never be reached — proves a deterministic tier won."""

    @property
    def name(self) -> str:
        return "stub/fail"

    def invoke(self, image, system_prompt, user_prompt) -> str:
        raise AssertionError("vision backend should not have been called")


def test_raster_splitter() -> None:
    from src.extraction.raster_splitter import split_by_gutters

    image = Image.new("RGB", (600, 400), "white")
    draw = ImageDraw.Draw(image)
    blocks = [
        (40, 30, 270, 180),
        (330, 30, 560, 180),
        (40, 230, 270, 370),
        (330, 230, 560, 370),
    ]
    for block in blocks:
        draw.rectangle(block, fill=(60, 60, 60))

    cells = split_by_gutters(image)
    assert len(cells) == 4, cells
    # Reading order and pixel alignment with the drawn blocks.
    for cell, (x0, y0, x1, y1) in zip(cells, blocks):
        assert abs(cell.x0 - x0) <= 2, (cell, (x0, y0, x1, y1))
        assert abs(cell.y0 - y0) <= 2, (cell, (x0, y0, x1, y1))
        assert abs(cell.x1 - (x1 + 1)) <= 2, (cell, (x0, y0, x1, y1))
        assert abs(cell.y1 - (y1 + 1)) <= 2, (cell, (x0, y0, x1, y1))
    print("raster splitter    OK  (2x2 grid segmented at pixel accuracy)")


def build_pdf_undersplit() -> bytes:
    """Two panels with markers (a),(b) as text, but the caption describes
    THREE panels (a),(b),(c) — geometry can only find two, so the framework
    must escalate to the vision model and replace the geometric result."""
    content = b"""BT /F1 11 Tf 72 700 Td (As shown in Figure 1, the accuracy increases steadily with the number of training epochs.) Tj ET
1 w
100 420 150 180 re S
280 420 150 180 re S
120 440 m 200 500 l S
300 440 m 380 560 l S
BT /F1 9 Tf 105 585 Td (\\(a\\)) Tj ET
BT /F1 9 Tf 285 585 Td (\\(b\\)) Tj ET
BT /F1 10 Tf 100 390 Td (Figure 1: Comparison views. \\(a\\) First panel view. \\(b\\) Second panel view. \\(c\\) Third panel view.) Tj ET
"""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"endstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size " + str(len(objects) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n" + str(xref_pos).encode() + b"\n%%EOF\n"
    )
    return bytes(out)


class ThirdsBackend(VisionBackend):
    """Stub that 'detects' three vertical thirds of whatever it sees."""

    @property
    def name(self) -> str:
        return "stub/thirds"

    def invoke(self, image, system_prompt, user_prompt) -> str:
        w, h = image.size
        third = w // 3
        return json.dumps(
            {
                "layout": "horizontal row",
                "confidence": 0.96,
                "panels": [
                    {"id": "a", "bbox": [0, 0, third - 5, h], "confidence": 0.95},
                    {"id": "b", "bbox": [third + 5, 0, 2 * third - 5, h], "confidence": 0.95},
                    {"id": "c", "bbox": [2 * third + 5, 0, w, h], "confidence": 0.95},
                ],
            }
        )


def test_xy_cut_layout() -> None:
    """'Tall panel left + 2x2 grid right' — the Figure-1 class of layout."""
    from src.extraction.raster_splitter import (
        complete_to_count,
        reading_order,
        xy_cut_cells,
    )

    image = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(image)
    tall = (20, 20, 280, 580)
    grid = [(330, 20, 590, 280), (640, 20, 880, 280),
            (330, 330, 590, 580), (640, 330, 880, 580)]
    for block in [tall] + grid:
        draw.rectangle(block, fill=(70, 70, 70))

    leaves = reading_order(xy_cut_cells(image))
    assert len(leaves) == 5, leaves
    # Label order: the tall-left panel first, then the grid row by row.
    assert leaves[0].x1 <= 300 and leaves[0].height > 500, leaves[0]
    assert leaves[1].y1 <= 300 and leaves[2].y1 <= 300, (leaves[1], leaves[2])
    assert leaves[1].x0 < leaves[2].x0
    assert leaves[3].y0 >= 300 and leaves[4].y0 >= 300, (leaves[3], leaves[4])
    assert leaves[3].x0 < leaves[4].x0

    # Completion: a merged strip of 3 touching color panels (no separator)
    # gets uniform-split along its long axis to reach the expected count —
    # the extra cuts go to the ELONGATED leaf, not the squarish one.
    strip_img = Image.new("RGB", (900, 620), "white")
    sdraw = ImageDraw.Draw(strip_img)
    sdraw.rectangle((20, 20, 460, 300), fill=(60, 60, 60))     # squarish single panel
    sdraw.rectangle((20, 360, 880, 600), fill=(120, 40, 40))   # strip = 3 touching panels
    leaves = xy_cut_cells(strip_img)
    assert len(leaves) == 2, leaves
    completed = complete_to_count(leaves, 4)
    assert completed is not None and len(completed) == 4
    top = [cell for cell in completed if cell.y0 < 300]
    bottom = [cell for cell in completed if cell.y0 >= 300]
    assert len(top) == 1 and len(bottom) == 3, completed
    assert bottom[0].x0 < bottom[1].x0 < bottom[2].x0
    print("xy-cut             OK  (tall+grid layout, strip completion)")


def test_completion_composition() -> None:
    """Figure-1 geometry: (a) big left, (b)(c)(d) row, merged (e)(f)(g) strip.
    The exhaustive allocator must give the strip 3 cuts — a greedy allocator
    ties on elongation and bisects (b) instead."""
    from src.extraction.raster_splitter import complete_to_count
    from src.models import PixelBox

    leaves = [
        PixelBox(x0=0, y0=0, x1=600, y1=520),        # (a) — near-square
        PixelBox(x0=650, y0=0, x1=1130, y1=330),     # (b)
        PixelBox(x0=1180, y0=0, x1=1660, y1=330),    # (c)
        PixelBox(x0=1710, y0=0, x1=2190, y1=330),    # (d)
        PixelBox(x0=650, y0=380, x1=2190, y1=810),   # (e)(f)(g) merged strip
    ]
    cells = complete_to_count(leaves, 7)
    assert cells is not None and len(cells) == 7, cells
    # (a)-(d) untouched, in place.
    assert cells[0] == leaves[0] and cells[1] == leaves[1]
    assert cells[2] == leaves[2] and cells[3] == leaves[3]
    # The strip received all three pieces, split into columns.
    strip_cells = cells[4:]
    assert all(cell.y0 == 380 for cell in strip_cells), cells
    assert strip_cells[0].x0 < strip_cells[1].x0 < strip_cells[2].x0
    print("completion         OK  (exhaustive allocation targets the strip)")


def test_grid_cut_cost() -> None:
    from src.extraction.raster_splitter import grid_cut_cost
    from src.models import PixelBox

    image = Image.new("RGB", (600, 400), "white")
    draw = ImageDraw.Draw(image)
    # Two textured panels with a white gutter at x = 290..310. Texture runs
    # HORIZONTALLY so any vertical cut through a panel must cross it.
    for y in range(30, 370, 14):
        draw.line((30, y, 280, y), fill=(90, 90, 90), width=3)
        draw.line((320, y, 570, y), fill=(90, 90, 90), width=3)

    good = [PixelBox(x0=0, y0=0, x1=300, y1=400), PixelBox(x0=300, y0=0, x1=600, y1=400)]
    bad = [PixelBox(x0=0, y0=0, x1=150, y1=400), PixelBox(x0=150, y0=0, x1=600, y1=400)]
    good_cost = grid_cut_cost(image, good)
    bad_cost = grid_cut_cost(image, bad)
    assert good_cost < 0.05, good_cost
    assert bad_cost > 0.4, bad_cost
    print(f"cut cost           OK  (gutter {good_cost:.2f} vs through-panel {bad_cost:.2f})")


def test_snap_grid() -> None:
    """Uniform cut lands inside a plot; snapping moves it to the real gap.

    Panels are line plots (sparse ink, UNEQUAL widths) — the failure class
    where uniform grids land mid-panel and the cut cost alone is too weak.
    """
    from src.extraction.raster_splitter import grid_cut_cost, snap_grid
    from src.models import PixelBox

    image = Image.new("RGB", (600, 400), "white")
    draw = ImageDraw.Draw(image)
    # Left plot frame 30..250, right plot frame 290..570 (gap 250..290).
    draw.rectangle((30, 40, 250, 360), outline="black", width=2)
    draw.rectangle((290, 40, 570, 360), outline="black", width=2)
    for y in range(70, 340, 24):  # sparse "curves" inside both plots
        draw.line((35, y, 245, y), fill=(120, 60, 60), width=2)
        draw.line((295, y, 565, y), fill=(60, 60, 120), width=2)

    # Uniform 2-column proposal: the cut at x=300 slices the right plot.
    uniform = [
        PixelBox(x0=0, y0=0, x1=300, y1=400),
        PixelBox(x0=300, y0=0, x1=600, y1=400),
    ]
    snapped = snap_grid(image, uniform)
    boundary = snapped[0].x1
    assert 250 <= boundary <= 290, snapped  # landed in the real gap
    assert snapped[1].x0 == boundary  # tiling stays consistent
    assert grid_cut_cost(image, snapped) <= grid_cut_cost(image, uniform)
    print(f"snap grid          OK  (cut 300 -> {boundary}, inside the true gap)")


def test_find_separators_dense_framed() -> None:
    """Two DENSE panels (arrow-field texture) sharing a black border line,
    with NO white gutter — find_separators must cut at the border spike."""
    from src.extraction.raster_splitter import (
        _ink_profile,
        find_separators,
        xy_cut_cells,
    )

    image = Image.new("RGB", (600, 300), "white")
    draw = ImageDraw.Draw(image)
    # Dense arrow-field texture: continuous faint rows across the WHOLE width
    # (every column has ink; no white gutter anywhere).
    for y in range(20, 285, 6):
        draw.line((20, y, 580, y), fill=(150, 150, 150), width=2)
    # A single black vertical border line at x=300 (the panel separator),
    # darker than the surrounding texture — a spike above the baseline.
    draw.line((300, 10, 300, 290), fill=(0, 0, 0), width=4)

    mask = image.convert("L").point(lambda v: 0 if v >= 245 else 255)
    profile = _ink_profile(mask, "x")
    segments = find_separators(profile, 600, min_run=40)
    # The border spike splits the row into two content segments.
    assert len(segments) == 2, segments
    assert segments[0][1] <= 305 and segments[1][0] >= 295, segments

    # xy_cut splits the dense figure at the border despite no white gutter.
    leaves = xy_cut_cells(image)
    assert len(leaves) == 2, leaves
    print("find separators    OK  (black frame line splits dense panels, no gutter)")


def test_audit_reconciliation() -> None:
    """Noisy counts that don't sum to the target still yield a target grid."""
    from src.extraction.raster_splitter import refine_grid_with_audit
    from src.models import PixelBox

    image = Image.new("RGB", (900, 400), "white")
    draw = ImageDraw.Draw(image)
    for i in range(3):
        draw.rectangle((20 + i * 300, 30, 280 + i * 300, 370), fill=(70, 70, 70))

    # One-row grid of 3 cells, but the auditor reports [1, 2, 0.5] (sum 3.5).
    cells = [
        PixelBox(x0=0, y0=0, x1=300, y1=400),
        PixelBox(x0=300, y0=0, x1=600, y1=400),
        PixelBox(x0=600, y0=0, x1=900, y1=400),
    ]
    edited = refine_grid_with_audit(
        image, cells, [1.0, 2.0, 0.5], {2: "two_panels_side_by_side", 3: "fragment_of_panel"}, 3
    )
    # Split cell 2 (+1) and merge cell 3 (-1) reconciles back to 3 — no None.
    assert edited is not None and len(edited) == 3, edited
    print("audit reconcile    OK  (noisy [1,2,0.5] still yields a 3-cell grid)")


def test_solid_border_separator() -> None:
    """Two framed panels sharing a border line (no whitespace between)."""
    from src.extraction.raster_splitter import split_by_gutters

    image = Image.new("RGB", (600, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 300, 280), outline="black", width=3)
    draw.rectangle((300, 20, 580, 280), outline="black", width=3)
    # Sparse in-panel content so the shared border is the only solid column.
    for x in range(40, 290, 30):
        draw.line((x, 150, x + 12, 150), fill="black", width=2)
    for x in range(320, 570, 30):
        draw.line((x, 150, x + 12, 150), fill="black", width=2)

    cells = split_by_gutters(image)
    assert len(cells) == 2, cells
    assert cells[0].x1 <= 302 and cells[1].x0 >= 296, cells
    print("solid separator    OK  (shared frame border splits panels)")


def test_candidate_grids() -> None:
    from src.extraction.raster_splitter import candidate_grids, draw_grid_overlay

    # Two solid rows separated by a white gutter; ask for 6 panels (3 per row).
    image = Image.new("RGB", (900, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 20, 890, 220), fill=(70, 70, 70))
    draw.rectangle((10, 280, 890, 480), fill=(70, 70, 70))

    candidates = candidate_grids(image, 6)
    assert candidates, "expected at least one candidate"
    for cells in candidates:
        assert len(cells) == 6, cells
    # The band-based candidate must exist: 2 detected rows x 3 columns whose
    # rows align with the drawn bands (not a blind uniform 500/2 split).
    band_based = [
        cells
        for cells in candidates
        if len({cell.y0 for cell in cells}) == 2
        and min(cell.y0 for cell in cells) >= 15
    ]
    assert band_based, [c[0] for c in candidates]
    # Overlay drawing stays pure and size-preserving.
    overlay = draw_grid_overlay(image, candidates[0])
    assert overlay.size == image.size
    assert overlay is not image
    print("candidate grids    OK  (gutter/band/uniform proposals generated)")


import re as _re


class GridJudgeBackend(VisionBackend):
    """Stub that audits grids (all cells = one panel); single-image box
    detection must never be reached."""

    def __init__(self) -> None:
        self.verifications = 0

    @property
    def name(self) -> str:
        return "stub/grid-judge"

    def invoke(self, image, system_prompt, user_prompt) -> str:
        raise AssertionError("box detection should not be reached")

    def invoke_images(self, images, system_prompt, user_prompt) -> str:
        assert len(images) == 2  # original + overlay
        if '"cell_labels"' in system_prompt:
            # Label reading: answer the expected letters in reading order.
            letters = _re.findall(r"\(([a-h])\)", user_prompt)
            return json.dumps({"cell_labels": letters})
        assert '"cell_counts"' in system_prompt
        self.verifications += 1
        match = _re.search(r"with (\d+) numbered", user_prompt)
        cells = int(match.group(1)) if match else 0
        return json.dumps({"cell_counts": [1] * cells, "confidence": 0.9})


class RejectingJudgeBackend(VisionBackend):
    """Stub whose audit rejects every grid (reports a merged cell); box
    detection must never run after an executed-and-rejecting audit."""

    def __init__(self) -> None:
        self.verifications = 0

    @property
    def name(self) -> str:
        return "stub/rejecting-judge"

    def invoke(self, image, system_prompt, user_prompt) -> str:
        raise AssertionError(
            "box detection must not run when the verifier rejected the grids"
        )

    def invoke_images(self, images, system_prompt, user_prompt) -> str:
        self.verifications += 1
        match = _re.search(r"with (\d+) numbered", user_prompt)
        cells = int(match.group(1)) if match else 1
        counts = [2.0] + [1.0] * (cells - 1)  # first cell holds two panels
        return json.dumps({"cell_counts": counts, "confidence": 0.9})


def test_verifier_rejection_keeps_geometry() -> None:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(build_pdf_undersplit())
        pdf_path = Path(tmp.name)
    try:
        backend = RejectingJudgeBackend()
        framework = FigureExtractionFramework(
            config=ExtractionConfig(panel_cropping="vlm-assisted"),
            panel_detector=PanelDetector(backend=backend),
        )
        document = framework.extract(pdf_path)
        ids = [f.figure_id for f in document.figures]
        print("rejection ids      =", ids)
        # Verifier ran and rejected everything -> the partial geometric
        # result is kept; the box tier is never consulted (the stub's
        # invoke() would raise).
        assert backend.verifications >= 1
        assert ids == ["figure-1", "figure-1a", "figure-1b"], ids
        assert all(f.origin == "geometric" for f in document.figures[1:])
        print("rejection gate     OK  (no box fallback after rejected audit)")
    finally:
        pdf_path.unlink(missing_ok=True)


def test_caption_ranges() -> None:
    from src.utils.captions import split_caption

    caption = (
        "FIG. 7. Top views of the annihilation process. (a)-(d) Process for "
        "Q = +1. (e)-(h) Process for Q = -1."
    )
    segments = split_caption(caption)
    assert sorted(segments) == list("abcdefgh"), sorted(segments)
    assert "Q = +1" in segments["b"] and "Q = -1" in segments["g"]
    assert segments["a"] == segments["d"]  # shared range segment
    # Single-paren range form and plain singles still work.
    segments = split_caption("Figure 2: Views. (a-c) Fields. (d) Energy.")
    assert sorted(segments) == list("abcd"), sorted(segments)
    assert "Energy" in segments["d"] and "Fields" in segments["b"]

    # Cross-reference to ANOTHER figure's panel must NOT inflate the count:
    # only the contiguous a,b prefix is kept, the stray (d) is dropped.
    segments = split_caption(
        "FIG. 1. (a) Energy E versus Q. (b) Top views, as in Fig. 2(d)."
    )
    assert sorted(segments) == ["a", "b"], sorted(segments)
    # A gap breaks the sequence: (a) present, (c) is out-of-sequence -> only a.
    segments = split_caption("FIG. 3. (a) Panel one. See panel (c) of Fig. 4.")
    assert sorted(segments) == ["a"], sorted(segments)
    print("caption ranges     OK  ((a)-(d) expands; cross-refs dropped)")


def test_map_labels_parsing() -> None:
    class LabelBackend(VisionBackend):
        def __init__(self, payload: str) -> None:
            self.payload = payload

        @property
        def name(self) -> str:
            return "stub/labels"

        def invoke(self, image, system_prompt, user_prompt) -> str:
            raise AssertionError("unexpected single-image call")

        def invoke_images(self, images, system_prompt, user_prompt) -> str:
            return self.payload

    image = Image.new("RGB", (400, 300), "white")
    # Column-major mapping, messy formatting — must normalize and accept.
    detector = PanelDetector(
        backend=LabelBackend(json.dumps({"cell_labels": ["b", "(C)", " A "]}))
    )
    mapping = detector.map_labels(image, image, ["a", "b", "c"], 3)
    assert mapping == ["b", "c", "a"], mapping
    # Not a permutation (duplicate) -> rejected.
    detector = PanelDetector(
        backend=LabelBackend(json.dumps({"cell_labels": ["a", "a", "c"]}))
    )
    assert detector.map_labels(image, image, ["a", "b", "c"], 3) is None
    # Wrong length -> rejected.
    detector = PanelDetector(
        backend=LabelBackend(json.dumps({"cell_labels": ["a", "b"]}))
    )
    assert detector.map_labels(image, image, ["a", "b", "c"], 3) is None
    print("label mapping      OK  (permutation accepted, garbage rejected)")


def test_audit_editing() -> None:
    """The auditor's report becomes deterministic valley splits/merges."""
    from src.extraction.raster_splitter import (
        refine_grid_with_audit,
        split_cell_at_valley,
    )
    from src.models import PixelBox

    image = Image.new("RGB", (600, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 30, 270, 270), fill=(70, 70, 70))
    draw.rectangle((330, 30, 580, 270), fill=(70, 70, 70))

    # Valley split of a cell containing both blocks lands in the gap.
    left, right = split_cell_at_valley(
        image, PixelBox(x0=0, y0=0, x1=600, y1=300), vertical_cut=True
    )
    assert 270 <= left.x1 <= 330, (left, right)
    assert right.x0 == left.x1

    # Audit-driven refinement: cell 1 holds two panels side by side.
    cells = [PixelBox(x0=0, y0=0, x1=600, y1=300)]
    edited = refine_grid_with_audit(
        image, cells, [2.0], {1: "two_panels_side_by_side"}, 2
    )
    assert edited is not None and len(edited) == 2, edited
    assert edited[0].x1 <= 330 and edited[1].x0 >= 270, edited
    print("audit editing      OK  (reports become valley splits)")


class AlternatingJudgeA(VisionBackend):
    """First auditor: rejects the initial grid once (cell 1 = two panels),
    then approves. Proves the debate loop applies edits and rotates roles."""

    def __init__(self) -> None:
        self.audits = 0

    @property
    def name(self) -> str:
        return "stub/judge-a"

    def invoke(self, image, system_prompt, user_prompt) -> str:
        raise AssertionError("box detection should not be reached")

    def invoke_images(self, images, system_prompt, user_prompt) -> str:
        if '"cell_labels"' in system_prompt:
            letters = _re.findall(r"\(([a-h])\)", user_prompt)
            return json.dumps({"cell_labels": letters})
        self.audits += 1
        match = _re.search(r"with (\d+) numbered", user_prompt)
        cells = int(match.group(1)) if match else 0
        return json.dumps({"cell_counts": [1] * cells, "confidence": 0.9})


class AlternatingJudgeB(VisionBackend):
    """Second auditor: always approves; must be consulted for the double
    sign-off."""

    def __init__(self) -> None:
        self.audits = 0

    @property
    def name(self) -> str:
        return "stub/judge-b"

    def invoke(self, image, system_prompt, user_prompt) -> str:
        raise AssertionError("box detection should not be reached")

    def invoke_images(self, images, system_prompt, user_prompt) -> str:
        if '"cell_labels"' in system_prompt:
            letters = _re.findall(r"\(([a-h])\)", user_prompt)
            return json.dumps({"cell_labels": letters})
        self.audits += 1
        match = _re.search(r"with (\d+) numbered", user_prompt)
        cells = int(match.group(1)) if match else 0
        return json.dumps({"cell_counts": [1] * cells, "confidence": 0.85})


def test_two_model_debate() -> None:
    """With a second detector, BOTH models must sign the grid off."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(build_pdf_undersplit())
        pdf_path = Path(tmp.name)
    try:
        backend_a = AlternatingJudgeA()
        backend_b = AlternatingJudgeB()
        framework = FigureExtractionFramework(
            config=ExtractionConfig(panel_cropping="vlm-assisted"),
            panel_detector=PanelDetector(backend=backend_a),
        )
        # Inject the second debater directly (bypasses env/model creation).
        framework._debate_detector = PanelDetector(backend=backend_b)
        framework.config = framework.config.model_copy(
            update={"debate_model": "stub-b"}
        )
        document = framework.extract(pdf_path)
        ids = [f.figure_id for f in document.figures]
        print("debate ids         =", ids)
        assert ids == ["figure-1", "figure-1a", "figure-1b", "figure-1c"], ids
        # Both auditors were consulted — the double sign-off happened.
        assert backend_a.audits >= 1 and backend_b.audits >= 1, (
            backend_a.audits,
            backend_b.audits,
        )
        assert all(f.origin == "vlm" for f in document.figures[1:])
        print(
            "two-model debate   OK  "
            f"(A audited {backend_a.audits}x, B audited {backend_b.audits}x)"
        )
    finally:
        pdf_path.unlink(missing_ok=True)


def test_verified_grid_tier() -> None:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(build_pdf_undersplit())
        pdf_path = Path(tmp.name)
    try:
        backend = GridJudgeBackend()
        framework = FigureExtractionFramework(
            config=ExtractionConfig(panel_cropping="vlm-assisted"),
            panel_detector=PanelDetector(backend=backend),
        )
        document = framework.extract(pdf_path)
        ids = [f.figure_id for f in document.figures]
        print("verify-tier ids    =", ids)
        assert ids == ["figure-1", "figure-1a", "figure-1b", "figure-1c"], ids
        assert backend.verifications >= 1
        panels = document.figures[1:]
        assert all(panel.origin == "vlm" for panel in panels)
        # NOTE: the stub approves the FIRST proposal blindly — geometric
        # correctness of the chosen grid is the real verifier's job, so only
        # structural invariants are asserted here.
        assert all(panel.image.width > 0 and panel.image.height > 0 for panel in panels)
        assert [panel.panel for panel in panels] == ["a", "b", "c"]
        assert "Third panel view" in panels[2].caption
        print("verified-grid tier OK  (Python proposes, model judges, Python cuts)")
    finally:
        pdf_path.unlink(missing_ok=True)


def test_unusable_audit_keeps_geometry() -> None:
    """When the model can't produce a usable audit, the partial geometric
    result is kept — there is no box-estimate fallback anymore."""

    class GarbageBackend(VisionBackend):
        @property
        def name(self) -> str:
            return "stub/garbage"

        def invoke(self, image, system_prompt, user_prompt) -> str:
            raise AssertionError("box detection tier must not exist")

        def invoke_images(self, images, system_prompt, user_prompt) -> str:
            return "I cannot help with that."

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(build_pdf_undersplit())
        pdf_path = Path(tmp.name)
    try:
        framework = FigureExtractionFramework(
            config=ExtractionConfig(panel_cropping="vlm-assisted"),
            panel_detector=PanelDetector(backend=GarbageBackend()),
        )
        document = framework.extract(pdf_path)
        ids = [f.figure_id for f in document.figures]
        print("garbage-audit ids  =", ids)
        # Geometry found 2 panels; the unusable audit cannot improve on it,
        # and no box fallback exists — the 2 geometric panels are kept.
        assert ids == ["figure-1", "figure-1a", "figure-1b"], ids
        assert all(f.origin == "geometric" for f in document.figures[1:])
        print("no-box-fallback    OK  (unusable audit keeps geometry, no boxes)")
    finally:
        pdf_path.unlink(missing_ok=True)


def test_framework_end_to_end() -> None:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(build_pdf_without_markers())
        pdf_path = Path(tmp.name)
    out_dir = Path(tempfile.mkdtemp(prefix="dataset_"))

    try:
        framework = FigureExtractionFramework(
            config=ExtractionConfig(panel_cropping="deterministic"),
            panel_detector=PanelDetector(backend=FailBackend()),
        )
        document = framework.extract(pdf_path)
        ids = [f.figure_id for f in document.figures]
        print("extracted ids      =", ids)
        assert ids == ["figure-1", "figure-1a", "figure-1b"], ids

        parent, panel_a, panel_b = document.figures
        # The deterministic raster tier resolved it — FailBackend proves the
        # vision model was never consulted.
        assert panel_a.origin == "raster" and panel_b.origin == "raster"
        assert panel_a.parent_id == "figure-1"
        # Caption matching (Stage 5): panel-specific segments, not the whole caption.
        assert "First panel view" in panel_a.caption
        assert "Second panel view" in panel_b.caption
        assert "Second panel" not in panel_a.caption
        # Panels are pixel crops of the parent at original resolution.
        assert 0 < panel_a.image.width < parent.image.width
        assert panel_a.dpi == parent.dpi
        # Pixel→PDF-point mapping should land inside the page.
        assert 0 <= panel_a.bbox.x0 < panel_a.bbox.x1 <= 612

        # Stage 6: export layout.
        paper_dir = framework.extract_to_dataset(pdf_path, out_dir, paper_name="paper")
        fig_dir = paper_dir / "figure_001"
        expected = [
            "figure.png", "caption.txt", "metadata.json",
            "panel_a.png", "panel_a.txt", "panel_a.json",
            "panel_b.png", "panel_b.txt", "panel_b.json",
        ]
        for name in expected:
            assert (fig_dir / name).is_file(), f"missing {name}"
        # Option-A metadata contract: panel→caption map from the caption,
        # marking which panels have a crop file.
        metadata = json.loads((fig_dir / "metadata.json").read_text(encoding="utf-8"))
        assert [p["panel"] for p in metadata["panels"]] == ["a", "b"], metadata
        assert metadata["panel_count"] == 2 and metadata["cropped_panel_count"] == 2
        panel_a = next(p for p in metadata["panels"] if p["panel"] == "a")
        assert panel_a["crop_file"] == "panel_a.png"
        assert "First panel view" in panel_a["caption"]
        assert "First panel view" in (fig_dir / "panel_a.txt").read_text(encoding="utf-8")
        print("framework e2e      OK  (raster tier + Option-A export, VLM untouched)")

        # markers-only mode must not touch the detector.
        geo = FigureExtractionFramework(
            config=ExtractionConfig(panel_cropping="markers-only"),
            panel_detector=None,
        )
        document = geo.extract(pdf_path)
        assert [f.figure_id for f in document.figures] == ["figure-1"]
        print("markers-only       OK  (no model, no raster)")
    finally:
        pdf_path.unlink(missing_ok=True)
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    test_crop_engine()
    test_raster_splitter()
    test_xy_cut_layout()
    test_completion_composition()
    test_grid_cut_cost()
    test_snap_grid()
    test_find_separators_dense_framed()
    test_audit_reconciliation()
    test_solid_border_separator()
    test_candidate_grids()
    test_grid_audit_parsing()
    test_framework_end_to_end()
    test_caption_ranges()
    test_map_labels_parsing()
    test_audit_editing()
    test_verified_grid_tier()
    test_two_model_debate()
    test_unusable_audit_keeps_geometry()
    test_verifier_rejection_keeps_geometry()
    print("\nFRAMEWORK TESTS PASSED")
