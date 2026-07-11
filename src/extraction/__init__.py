"""Dataset-extraction framework: confidence-first figure/panel extraction.

Composes the pipeline into a reusable framework:

* :class:`~src.extraction.framework.FigureExtractionFramework` — PDF →
  figures → panels (deterministic tiers, optional vision assist) → dataset.
* :class:`~src.extraction.crop_engine.CropEngine` — pixel-space crop
  validation and bounded expansion.
* :class:`~src.extraction.exporter.DatasetExporter` — the structured
  on-disk dataset layout (figure + caption + panel→caption map + crops).
"""

from src.extraction.crop_engine import CropEngine
from src.extraction.exporter import DatasetExporter
from src.extraction.framework import ExtractionConfig, FigureExtractionFramework

__all__ = [
    "CropEngine",
    "DatasetExporter",
    "ExtractionConfig",
    "FigureExtractionFramework",
]
