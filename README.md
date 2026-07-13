# Scientific Figure Extractor

**Turn a scientific PDF into a clean dataset of figures + captions — pure Python, no LLM, no API keys.**

This is the figure-extraction stage of a multi-agent paper-analysis pipeline. It detects every complete figure in a PDF, renders it at high resolution, pairs it with its caption, and writes a structured dataset — ready to hand to a downstream analysis agent. Extraction is fully deterministic: a local object-detection model (DocLayout-YOLO) locates figures visually, and everything else is plain Python.

---

## Key features

- **Robust figure detection** — every page is analyzed by [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) (the layout model used by MinerU), which locates `figure` and `figure_caption` regions *visually*, independent of how the PDF encodes its text — so two-column journals (APS/REVTeX, IEEE, Nature, Elsevier, …) work reliably.
- **Caption-anchor fallback** — when the layout model is unavailable (it pulls PyTorch), the parser falls back to a caption-anchor heuristic (`Figure 3:`, `Fig. 2.`, `FIGURE 4 —`) that maps each caption to the graphic region it describes. No dependency is mandatory.
- **High-resolution, lossless output** — each figure is rendered to a PNG at 300 DPI (configurable) at its original aspect ratio, with its full caption verbatim.
- **Structured dataset** — `figure_001.png` + `figure_001.txt` + `figure_001.json` per figure, plus an `index.json` — trivial for the next agent to consume.
- **CLI + optional Streamlit viewer**, fully typed (Pydantic v2), with tests that need no model or network.

> **No panels, no LLM.** This tool extracts *complete* figures and their captions. Panel-level reasoning ("what does panel (b) show?") is left to the downstream analysis agent, which reads the whole figure plus the full caption — a correct whole figure beats a mis-cropped panel.

---

## How it works

```
   PDF ──► render each page ──► DocLayout-YOLO ──► figure + caption regions
                                     │  (fallback: caption-anchor heuristic)
                                     ▼
              read caption text (PDF text layer)  ·  crop figure @ 300 DPI
                                     ▼
        dataset/<paper>/  figure_001.png · figure_001.txt · figure_001.json · index.json
```

---

## Quickstart

```bash
git clone https://github.com/<you>/AI-Scientific-Figure-Analyze.git
cd AI-Scientific-Figure-Analyze

python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (cmd)
.venv\Scripts\activate.bat

pip install -r requirements.txt
```

> Requires **Python 3.10+**. DocLayout-YOLO pulls PyTorch (~2 GB); if you can't install it, extraction still works via the caption-anchor heuristics. No API keys are ever required.

---

## Usage

### Command line

```bash
# Extract into ./dataset/<pdf-stem>/
python -m src.extraction.cli paper.pdf

# Custom output directory, name, and DPI
python -m src.extraction.cli paper.pdf -o out --name mypaper --dpi 400

# Force the caption-anchor heuristics (skip the layout model)
python -m src.extraction.cli paper.pdf --no-layout-model
```

Output — the **contract** a downstream agent consumes:

```
dataset/<paper>/
    figure_001.png      # complete figure, original resolution
    figure_001.txt      # full caption, verbatim
    figure_001.json     # figure metadata (below)
    figure_002.png
    ...
    index.json          # one entry per figure
```

`figure_NNN.json`:

```json
{
  "figure_id": "figure-3",
  "label": "Figure 3",
  "number": "3",
  "page_number": 3,
  "caption": "FIG. 3. Bimeron Hall angle as a function of damping ...",
  "detection_method": "layout-model",
  "dpi": 300,
  "bbox_pdf_points": {"x0": 52.0, "top": 88.0, "x1": 560.0, "bottom": 300.0},
  "image_size_px": [1079, 460],
  "files": {"image": "figure_003.png", "caption": "figure_003.txt"}
}
```

### Optional viewer

```bash
streamlit run src/api/app.py
```

Upload a PDF → see every extracted figure with its caption and detection metadata → click **Export dataset** to write it to disk.

### As a library

```python
from src.parsing.pdf_parser import ScientificPDFParser
from src.extraction.exporter import DatasetExporter

document = ScientificPDFParser().parse("paper.pdf")
for fig in document.figures:
    print(fig.label, "→", fig.caption[:80])

DatasetExporter("dataset").export(document, paper_name="mypaper")
```

---

## Configuration

The default needs no configuration. Optional environment variables (see `.env.example`):

| Variable            | Purpose                                                                    |
| ------------------- | -------------------------------------------------------------------------- |
| `LAYOUT_DETECTION`  | `on` (default) / `off` to force the caption-anchor heuristic detector      |
| `LAYOUT_MODEL_PATH` | Use local DocLayout-YOLO weights instead of downloading from the HF Hub    |

---

## Repository layout

```
├── src/
│   ├── models.py               # Domain models (BoundingBox, PixelBox, ExtractedFigure, ...)
│   ├── parsing/
│   │   ├── pdf_parser.py       # Figure/caption detection + high-res rendering
│   │   └── layout_detector.py  # DocLayout-YOLO wrapper
│   ├── extraction/
│   │   ├── exporter.py         # DatasetExporter (figure + caption + metadata + index)
│   │   └── cli.py              # python -m src.extraction.cli
│   ├── pipeline.py             # Thin facade for the viewer
│   └── api/app.py              # Streamlit viewer
├── tests/                      # pytest — no model or network required
├── requirements.txt · requirements-dev.txt · pyproject.toml
├── .env.example · .gitignore · LICENSE (MIT)
└── README.md
```

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers figure + caption extraction (heuristic path), the dataset export contract, and the layout-model mapping chain with a stubbed detector — no model or network required. One test exercises the real DocLayout-YOLO model when installed, and is skipped otherwise.

---

## Contributing

Issues and PRs welcome — use the [bug report](.github/ISSUE_TEMPLATE/bug_report.md) and [feature request](.github/ISSUE_TEMPLATE/feature_request.md) templates. Interesting areas: caption-above-figure layouts, multi-figure pages, and table extraction.

## License

Released under the [MIT License](LICENSE).
