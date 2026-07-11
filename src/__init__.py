"""AI Scientific Figure Analyzer — core package.

The figure-extraction stage of a multi-agent paper-analysis pipeline. It
turns a scientific PDF into a structured dataset — the complete figure, its
full caption, a reliable panel→caption map, and panel crops when they can be
produced with confidence — ready to feed a downstream analysis agent.

Modules:

* ``src.parsing``    — figure/caption detection (DocLayout-YOLO + heuristics)
  and marker-based panel cropping.
* ``src.extraction`` — the confidence-first panel-cropping framework, crop
  engine, and dataset exporter (:class:`~src.extraction.framework.
  FigureExtractionFramework`, CLI).
* ``src.utils``      — deterministic caption → per-panel splitting.
* ``src.vision``     — swappable vision backends: grid auditing for the
  opt-in ``vlm-assisted`` crop mode, and the analytical chat.
* ``src.rag``        — in-memory figure + caption + reference index (chat).
* ``src.api``        — the interactive Streamlit dashboard.

``src.pipeline.Orchestrator`` wires the dashboard's stages together.
"""
