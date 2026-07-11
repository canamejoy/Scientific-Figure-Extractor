"""Stage 3/4 — vision-language model access.

* :mod:`src.vision.backend` — swappable :class:`VisionBackend` providers.
* :mod:`src.vision.vlm_client` — :class:`FigureAnalyzer` analytical chat.
* :mod:`src.vision.panel_detector` — :class:`PanelDetector` layout analysis.
"""

from src.vision.backend import (
    LangChainVisionBackend,
    VisionBackend,
    VLMProvider,
    create_vision_backend,
)
from src.vision.panel_detector import PanelDetector
from src.vision.vlm_client import FigureAnalyzer

__all__ = [
    "FigureAnalyzer",
    "LangChainVisionBackend",
    "PanelDetector",
    "VisionBackend",
    "VLMProvider",
    "create_vision_backend",
]
