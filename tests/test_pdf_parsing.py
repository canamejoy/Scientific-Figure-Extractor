"""End-to-end smoke test for the AI Scientific Figure Analyzer pipeline.

Builds a minimal, valid PDF from raw bytes: one page with a body sentence
referencing Figure 1 and a THREE-PANEL vector figure that stresses the
splitter: markers "(a)" and "(c)" sit at different heights (uneven panels)
and the "(b)" marker is deliberately missing — the middle panel must be
recovered by letter interpolation. Verifies whole-figure union extraction,
content-based panel decomposition, caption splitting, and reference
retrieval with the panel→whole-figure fallback.
"""

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Users\eldre\AI-Scientific-Figure-Analyze")
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["LAYOUT_DETECTION"] = "off"  # deterministic geometric tests


def build_pdf() -> bytes:
    """Assembles a one-page PDF with correct xref offsets."""
    content = b"""BT /F1 11 Tf 72 700 Td (As shown in Figure 1, the accuracy increases steadily with the number of training epochs.) Tj ET
1 w
100 420 100 180 re S
230 420 100 180 re S
360 420 100 180 re S
110 440 m 180 500 l S
240 440 m 310 560 l S
370 440 m 440 520 l S
BT /F1 9 Tf 105 585 Td (\\(a\\)) Tj ET
BT /F1 9 Tf 365 570 Td (\\(c\\)) Tj ET
BT /F1 10 Tf 100 390 Td (Figure 1: Comparison views. \\(a\\) First panel view. \\(b\\) Second panel view. \\(c\\) Third panel view.) Tj ET
"""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(content)).encode()
        + b" >>\nstream\n"
        + content
        + b"endstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_pos = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_pos).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


def build_grid_pdf() -> bytes:
    """One figure drawn as a SINGLE solid block (like one embedded raster)
    with markers (a),(b) on row 1 and (c) on row 2. The content clustering
    finds one cell with three markers, forcing the 2-D marker-grid cut —
    a single-axis cut would merge (a) with (c) below it."""
    content = b"""BT /F1 11 Tf 72 700 Td (As shown in Figure 1, the accuracy increases steadily with the number of training epochs.) Tj ET
1 w
100 340 360 300 re S
110 500 m 200 620 l S
300 360 m 380 460 l S
BT /F1 9 Tf 105 610 Td (\\(a\\)) Tj ET
BT /F1 9 Tf 290 610 Td (\\(b\\)) Tj ET
BT /F1 9 Tf 105 470 Td (\\(c\\)) Tj ET
BT /F1 10 Tf 100 310 Td (Figure 1: Grid views. \\(a\\) A one. \\(b\\) B two. \\(c\\) C three.) Tj ET
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


def test_single_cell_grid() -> None:
    from src.parsing.pdf_parser import ScientificPDFParser

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(build_grid_pdf())
        pdf_path = Path(tmp.name)
    try:
        parser = ScientificPDFParser(use_layout_model=False)
        document = parser.parse(pdf_path)
        ids = [f.figure_id for f in document.figures]
        print("grid-cut ids       =", ids)
        assert ids == ["figure-1", "figure-1a", "figure-1b", "figure-1c"], ids
        _, a, b, c = document.figures

        # Row 1 panels must NOT extend into row 2 (the (b)+(e) bug class).
        assert a.bbox.bottom <= 318, a.bbox
        assert b.bbox.bottom <= 318, b.bbox
        assert c.bbox.top >= 308, c.bbox
        # Column cut inside row 1.
        assert a.bbox.x1 <= 292, a.bbox
        assert b.bbox.x0 >= 284, b.bbox
        # Row 2 panel spans the full figure width.
        assert c.bbox.width >= 350, c.bbox
        assert "A one" in a.caption and "B two" in b.caption and "C three" in c.caption
        print("grid cut           OK  (rows respected, no cross-row panels)")
    finally:
        pdf_path.unlink(missing_ok=True)


def build_pdf_trailing_missing_marker() -> bytes:
    """Three separated panels, but only markers (a) and (b) exist as text —
    (c)'s marker is missing. The caption still describes (a),(b),(c), so
    caption-guided reconciliation must recover the third panel instead of
    absorbing it into (b)."""
    content = b"""BT /F1 11 Tf 72 700 Td (As shown in Figure 1, the accuracy increases steadily with the number of training epochs.) Tj ET
1 w
100 420 100 180 re S
230 420 100 180 re S
360 420 100 180 re S
110 440 m 180 500 l S
240 440 m 310 560 l S
370 440 m 440 520 l S
BT /F1 9 Tf 105 585 Td (\\(a\\)) Tj ET
BT /F1 9 Tf 235 585 Td (\\(b\\)) Tj ET
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


def test_caption_reconciliation() -> None:
    from src.parsing.pdf_parser import ScientificPDFParser

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(build_pdf_trailing_missing_marker())
        pdf_path = Path(tmp.name)
    try:
        parser = ScientificPDFParser(use_layout_model=False)
        document = parser.parse(pdf_path)
        ids = [f.figure_id for f in document.figures]
        print("reconciliation ids =", ids)
        assert ids == ["figure-1", "figure-1a", "figure-1b", "figure-1c"], ids
        panel_c = document.figures[3]
        # The recovered panel is the THIRD rect, not merged into (b).
        assert panel_c.bbox.x0 >= 350, panel_c.bbox
        assert "Third panel view" in panel_c.caption
        panel_b = document.figures[2]
        assert panel_b.bbox.x1 <= 340, panel_b.bbox  # (b) did not swallow (c)
        print("reconciliation     OK  (missing trailing marker recovered from caption)")
    finally:
        pdf_path.unlink(missing_ok=True)


def main() -> None:
    from src.parsing.pdf_parser import ScientificPDFParser
    from src.rag.vector_db import MultiModalContextIndex
    from src.vision.vlm_client import FigureAnalyzer

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(build_pdf())
        pdf_path = Path(tmp.name)

    try:
        parser = ScientificPDFParser(use_layout_model=False)
        document = parser.parse(pdf_path)

        ids = [f.figure_id for f in document.figures]
        print("extracted ids      =", ids)
        assert ids == ["figure-1", "figure-1a", "figure-1b", "figure-1c"], ids

        parent, panel_a, panel_b, panel_c = document.figures

        # Whole figure must cover ALL panels (union, not largest-object).
        print(f"parent bbox        = {parent.bbox.x0:.0f}..{parent.bbox.x1:.0f} pt")
        assert parent.bbox.x0 <= 101 and parent.bbox.x1 >= 459, parent.bbox
        assert parent.parent_id is None and parent.panel is None

        # Panels are tight around their own content — no bleed into
        # neighbours even with uneven marker heights.
        for panel, lo, hi in ((panel_a, 95, 205), (panel_b, 225, 335), (panel_c, 355, 465)):
            print(
                f"{panel.figure_id:11s} bbox = {panel.bbox.x0:.0f}..{panel.bbox.x1:.0f}"
                f"  caption='{panel.caption}'"
            )
            assert lo <= panel.bbox.x0 and panel.bbox.x1 <= hi, (panel.figure_id, panel.bbox)
            assert panel.parent_id == "figure-1"

        # Interpolation recovered the middle panel whose marker was missing.
        assert panel_b.label == "Figure 1(b)" and panel_b.panel == "b"
        assert "Second panel view" in panel_b.caption
        assert "First panel view" in panel_a.caption
        assert "Third panel view" in panel_c.caption
        assert "Comparison views" in panel_a.caption  # shared intro retained
        assert "Second panel" not in panel_a.caption

        # Index: parent gets the body reference; panels have no
        # panel-specific mention, so they fall back to whole-figure mentions.
        index = MultiModalContextIndex()
        index.build(document)
        ctx_parent = index.retrieve("figure-1", "accuracy over epochs")
        ctx_b = index.retrieve("figure-1b", "accuracy over epochs")
        print("parent references  =", ctx_parent.references)
        print("panel-b references =", ctx_b.references)
        assert any("increases steadily" in r for r in ctx_parent.references)
        assert any("increases steadily" in r for r in ctx_b.references)
        for refs in (ctx_parent.references, ctx_b.references):
            assert all("Comparison views" not in r for r in refs), refs

        # Panel-specific mention pattern: "Fig. 2(e)" and "Fig. 2e" forms.
        pat = MultiModalContextIndex._mention_pattern("2e")
        assert pat.search("as seen in Fig. 2(e), the spins tilt")
        assert pat.search("as seen in Figure 2e, the spins tilt")
        assert not pat.search("as seen in Fig. 2(f), the spins tilt")

        # Prompt building stays pure (no API key needed).
        prompt = FigureAnalyzer._build_user_prompt(
            caption=panel_b.caption,
            context_snippets=ctx_b.references,
            user_query="Describe the trend in this panel.",
            figure_label=panel_b.label,
        )
        assert "Figure 1(b)" in prompt and "USER QUESTION:" in prompt

        print("\nSMOKE TEST PASSED")
    finally:
        pdf_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
    test_single_cell_grid()
    test_caption_reconciliation()
