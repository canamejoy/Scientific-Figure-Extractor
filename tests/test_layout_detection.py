"""Tests for the layout-model extraction path.

Part 1 uses a stub detector (no model) to verify the full mapping chain:
page render → regions → PDF points → caption text from the text layer →
figure number parsing → high-res crop → panel splitting.

Part 2 loads the REAL DocLayout-YOLO model (downloads weights on first run)
and checks it initializes and runs inference on a rendered page without
crashing — detection content on a synthetic page is not asserted.
"""

import sys
import tempfile
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(r"c:\Users\eldre\AI-Scientific-Figure-Analyze")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from test_pdf_parsing import build_pdf  # noqa: E402  (three-panel synthetic PDF)

from src.models import LayoutRegion, PixelBox  # noqa: E402
from src.parsing.pdf_parser import ScientificPDFParser  # noqa: E402


class StubLayoutDetector:
    """Returns hand-placed figure/caption regions for the synthetic page.

    The synthetic page is 612x792 pt; at the 144-DPI detection render this
    is 1224x1584 px (scale 2.0). The figure spans x 100–460 pt / top
    192–372 pt; the caption line sits near top 394–404 pt.
    """

    def detect(self, page_image) -> List[LayoutRegion]:
        return [
            LayoutRegion(
                label="figure",
                box=PixelBox(x0=195, y0=375, x1=925, y1=750),
                confidence=0.93,
            ),
            LayoutRegion(
                label="figure_caption",
                box=PixelBox(x0=195, y0=780, x1=1160, y1=820),
                confidence=0.91,
            ),
            # Noise the parser must ignore.
            LayoutRegion(
                label="plain_text",
                box=PixelBox(x0=140, y0=150, x1=1100, y1=200),
                confidence=0.99,
            ),
        ]


def test_layout_path_with_stub() -> None:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(build_pdf())
        pdf_path = Path(tmp.name)
    try:
        parser = ScientificPDFParser(
            use_layout_model=True, layout_detector=StubLayoutDetector()
        )
        document = parser.parse(pdf_path)
        ids = [f.figure_id for f in document.figures]
        print("layout-path ids    =", ids)
        assert ids == ["figure-1", "figure-1a", "figure-1b", "figure-1c"], ids

        parent = document.figures[0]
        assert parent.detection_method == "layout-model"
        # Caption read from the PDF text layer inside the detected region.
        assert parent.caption.startswith("Figure 1: Comparison views."), parent.caption
        assert parent.label == "Figure 1"
        # Panels still split geometrically inside the detected region.
        assert document.figures[2].panel == "b"
        assert "Second panel view" in document.figures[2].caption
        print("layout stub path   OK  (caption from text layer, panels split)")
    finally:
        pdf_path.unlink(missing_ok=True)


def test_real_model_loads() -> None:
    import pdfplumber

    from src.parsing.layout_detector import LayoutFigureDetector

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(build_pdf())
        pdf_path = Path(tmp.name)
    try:
        detector = LayoutFigureDetector()
        with pdfplumber.open(str(pdf_path)) as pdf:
            page_image = pdf.pages[0].to_image(resolution=144).original
        regions = detector.detect(page_image)
        print(
            "real model         OK  (loaded + inference ran; "
            f"{len(regions)} region(s) on synthetic page: "
            + (", ".join(sorted({r.label for r in regions})) or "none")
        )
    finally:
        pdf_path.unlink(missing_ok=True)


if __name__ == "__main__":
    test_layout_path_with_stub()
    test_real_model_loads()
    print("\nLAYOUT TESTS PASSED")
