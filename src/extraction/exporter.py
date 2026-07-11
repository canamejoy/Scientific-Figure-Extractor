"""Stage 6 — structured on-disk dataset export.

Produces, per figure::

    <output_dir>/<paper_name>/
        figure_001/
            figure.png          # the complete figure, original resolution
            caption.txt         # the full caption, verbatim
            metadata.json       # figure metadata + panel→caption map
            panel_a.png         # panel crop (only for confidently cropped panels)
            panel_a.txt         # panel caption
            panel_a.json        # panel metadata
            ...

**Output contract (robust-base design).** ``metadata.json`` always carries a
``panels`` list derived from the caption text — one entry per panel the
caption describes, each with its own caption segment — *whether or not* a
crop was produced. Panel crops are an optional, high-confidence extra: a
downstream analysis agent always receives the complete figure plus a
reliable panel→caption map, and uses crops only when they exist. A missing
crop never loses information; a wrong crop is never emitted.

Images are written losslessly (PNG) at their extracted resolution — no
resizing, no recompression beyond PNG encoding.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models import ExtractedFigure, ParsedDocument
from src.utils.captions import split_caption

logger = logging.getLogger(__name__)


def _safe_name(name: str) -> str:
    """Reduces an arbitrary string to a filesystem-safe directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "paper"


class DatasetExporter:
    """Writes an extracted document to a structured dataset directory.

    Args:
        output_dir: Root directory datasets are written under (created on
            demand).
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)

    def export(
        self, document: ParsedDocument, paper_name: Optional[str] = None
    ) -> Path:
        """Exports every figure (and its cropped panels) of a document.

        Args:
            document: The processed document; cropped panels are recognized
                by their ``parent_id``.
            paper_name: Dataset folder name; defaults to the PDF's stem.

        Returns:
            The path of the written paper directory.
        """
        name = _safe_name(paper_name or Path(document.source_path).stem)
        paper_dir = self.output_dir / name
        paper_dir.mkdir(parents=True, exist_ok=True)

        parents = [f for f in document.figures if f.parent_id is None]
        cropped: Dict[str, List[ExtractedFigure]] = {}
        for figure in document.figures:
            if figure.parent_id is not None:
                cropped.setdefault(figure.parent_id, []).append(figure)

        total_crops = 0
        for index, parent in enumerate(parents, start=1):
            figure_dir = paper_dir / f"figure_{index:03d}"
            figure_dir.mkdir(parents=True, exist_ok=True)

            parent.image.save(figure_dir / "figure.png", format="PNG")
            (figure_dir / "caption.txt").write_text(parent.caption, encoding="utf-8")

            crops = cropped.get(parent.figure_id, [])
            total_crops += len(crops)
            crops_by_letter = {crop.panel: crop for crop in crops if crop.panel}

            (figure_dir / "metadata.json").write_text(
                json.dumps(
                    self._figure_metadata(parent, crops_by_letter),
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            for crop in crops:
                letter = crop.panel or "x"
                crop.image.save(figure_dir / f"panel_{letter}.png", format="PNG")
                (figure_dir / f"panel_{letter}.txt").write_text(
                    crop.caption, encoding="utf-8"
                )
                (figure_dir / f"panel_{letter}.json").write_text(
                    json.dumps(
                        self._panel_metadata(crop), indent=2, ensure_ascii=False
                    ),
                    encoding="utf-8",
                )

            logger.info(
                "Exported %s (%d panel crop(s)) -> %s",
                parent.label,
                len(crops),
                figure_dir,
            )

        logger.info(
            "Dataset written: %s (%d figure(s), %d panel crop(s))",
            paper_dir,
            len(parents),
            total_crops,
        )
        return paper_dir

    # ------------------------------------------------------------------ #
    # Metadata serialization                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _figure_metadata(
        figure: ExtractedFigure,
        crops_by_letter: Dict[Optional[str], ExtractedFigure],
    ) -> Dict[str, Any]:
        """Figure metadata plus the caption-derived panel→caption map.

        The ``panels`` list is built from the caption text, so it is present
        and correct even for figures kept whole (no crop). Each entry marks
        whether a crop file exists for that panel.
        """
        segments = split_caption(figure.caption)
        panels: List[Dict[str, Any]] = []
        for letter in sorted(segments):
            crop = crops_by_letter.get(letter)
            panels.append(
                {
                    "panel": letter,
                    "label": f"{figure.label}({letter})",
                    "caption": segments[letter],
                    "crop_file": f"panel_{letter}.png" if crop is not None else None,
                    "crop_size_px": (
                        [crop.image.width, crop.image.height] if crop else None
                    ),
                }
            )
        return {
            "figure_id": figure.figure_id,
            "label": figure.label,
            "number": figure.number,
            "page_number": figure.page_number,
            "caption": figure.caption,
            "detection_method": figure.detection_method,
            "dpi": figure.dpi,
            "bbox_pdf_points": figure.bbox.model_dump(),
            "image_size_px": [figure.image.width, figure.image.height],
            "panel_count": len(panels),
            "cropped_panel_count": sum(1 for p in panels if p["crop_file"]),
            "panels": panels,
        }

    @staticmethod
    def _panel_metadata(panel: ExtractedFigure) -> Dict[str, Any]:
        """JSON-serializable metadata for one cropped panel."""
        return {
            "figure_id": panel.figure_id,
            "label": panel.label,
            "panel": panel.panel,
            "parent_id": panel.parent_id,
            "page_number": panel.page_number,
            "caption": panel.caption,
            "detection_method": panel.detection_method,
            "origin": panel.origin,
            "dpi": panel.dpi,
            "bbox_pdf_points": panel.bbox.model_dump(),
            "image_size_px": [panel.image.width, panel.image.height],
        }
