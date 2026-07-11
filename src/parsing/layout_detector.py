"""Page-level figure/caption localization with a document-layout model.

Wraps `DocLayout-YOLO <https://github.com/opendatalab/DocLayout-YOLO>`_
(the detector used by MinerU), trained on DocStructBench to recognize
``figure``, ``figure_caption``, ``table``, and other regions on *rendered
page images*. Detecting figures visually — the way a human skims a page —
is far more robust across journals (Nature, APS/REVTeX two-column, IEEE,
Elsevier, ...) than PDF-internal heuristics, because it does not depend on
how the PDF encodes its text lines or graphics.

The heavy dependencies (``doclayout-yolo`` → PyTorch, ``huggingface-hub``
for the ~40 MB weights, cached locally after the first download) are
imported lazily; when they are missing, :class:`LayoutFigureDetector`
raises ``ImportError`` and the PDF parser falls back to its caption-anchor
heuristics automatically.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Union

from PIL import Image

from src.models import LayoutRegion, PixelBox

logger = logging.getLogger(__name__)


class LayoutFigureDetector:
    """Detects figure and caption regions on rendered page images.

    The model is loaded once per instance and reused across pages and
    documents (inference is ~1 s/page on CPU).

    Args:
        model_path: Path to the YOLO weights. Defaults to the
            ``LAYOUT_MODEL_PATH`` environment variable, else the
            DocStructBench weights are downloaded from the Hugging Face Hub
            (cached under the user's HF cache directory).
        confidence: Minimum detection confidence to keep a region.
        image_size: Inference resolution (the model was trained at 1024).
        device: Torch device string (``"cpu"`` or ``"cuda"``).

    Raises:
        ImportError: If ``doclayout-yolo`` (or ``huggingface-hub`` when the
            weights must be downloaded) is not installed.
    """

    DEFAULT_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
    DEFAULT_WEIGHTS = "doclayout_yolo_docstructbench_imgsz1024.pt"

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        confidence: float = 0.25,
        image_size: int = 1024,
        device: str = "cpu",
    ) -> None:
        try:
            from doclayout_yolo import YOLOv10
        except ImportError as exc:  # pragma: no cover - environment issue
            raise ImportError(
                "doclayout-yolo is required for layout-model figure "
                "detection. Install it with: pip install doclayout-yolo "
                "huggingface-hub"
            ) from exc

        resolved = model_path or os.getenv("LAYOUT_MODEL_PATH")
        if not resolved:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:  # pragma: no cover - environment issue
                raise ImportError(
                    "huggingface-hub is required to download the layout "
                    "model weights. Install it with: pip install "
                    "huggingface-hub (or set LAYOUT_MODEL_PATH to local "
                    "weights)."
                ) from exc
            logger.info(
                "Fetching layout-model weights %s/%s (cached after first "
                "download)",
                self.DEFAULT_REPO,
                self.DEFAULT_WEIGHTS,
            )
            resolved = hf_hub_download(self.DEFAULT_REPO, self.DEFAULT_WEIGHTS)

        self._confidence = confidence
        self._image_size = image_size
        self._device = device
        self._model = YOLOv10(str(resolved))
        logger.info("Layout model ready (%s, device=%s)", resolved, device)

    def detect(self, page_image: Image.Image) -> List[LayoutRegion]:
        """Detects layout regions on one rendered page image.

        Args:
            page_image: The page rendered as a PIL image.

        Returns:
            All regions above the confidence threshold, with normalized
            lower-case labels (``"figure"``, ``"figure_caption"``, ...), in
            the image's pixel space.
        """
        results = self._model.predict(
            page_image.convert("RGB"),
            imgsz=self._image_size,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )
        regions: List[LayoutRegion] = []
        width, height = page_image.size
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = getattr(result, "names", None) or {}
            for index in range(len(boxes)):
                region = self._to_region(boxes, index, names, width, height)
                if region is not None:
                    regions.append(region)
        logger.debug(
            "Layout model found %d region(s): %s",
            len(regions),
            ", ".join(f"{r.label}({r.confidence:.2f})" for r in regions) or "-",
        )
        return regions

    @staticmethod
    def _to_region(
        boxes: Any, index: int, names: dict, width: int, height: int
    ) -> Optional[LayoutRegion]:
        """Converts one YOLO detection into a :class:`LayoutRegion`."""
        try:
            class_id = int(boxes.cls[index])
            confidence = float(boxes.conf[index])
            x0, y0, x1, y1 = (float(v) for v in boxes.xyxy[index])
        except (TypeError, ValueError, IndexError):  # pragma: no cover
            return None
        label = str(names.get(class_id, class_id)).strip().lower().replace(" ", "_")
        box = PixelBox(
            x0=max(0, int(x0)),
            y0=max(0, int(y0)),
            x1=max(0, int(x1)),
            y1=max(0, int(y1)),
        ).clamped(width, height)
        if box.width <= 0 or box.height <= 0:
            return None
        return LayoutRegion(label=label, box=box, confidence=confidence)
