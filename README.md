# 🔬 AI Scientific Figure Analyzer

**Extract figures from scientific PDFs, split them into panels, and pair each panel with its caption — clean, cropped, and ready to feed a downstream analysis agent.**

LLMs struggle to read scientific figures: feeding a whole page (or a downscaled document) destroys the spatial resolution needed to read axes, legends, and data markers, and strips away the local text that explains what the figure means. This project is the **figure-extraction stage** of a multi-agent paper-analysis pipeline. It turns a PDF into a structured dataset — the complete figure at high resolution, its full caption, a reliable **panel → caption map**, and panel crops when they can be produced with confidence — so a specialized analysis agent always receives correct, well-contextualized inputs.

---

## ✨ Key features

- **📄 Robust figure detection** — every page is analyzed by [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) (the layout model used by MinerU), which locates `figure` and `figure_caption` regions *visually*, independent of how the PDF encodes its text — so two-column journals (APS/REVTeX, IEEE, Nature, Elsevier, …) work reliably. A caption-anchor heuristic is the automatic fallback when the model is unavailable.
- **✂️ Confidence-first panel cropping** — panels are cropped deterministically from PDF text markers (`(a)`, `(b)`, …) and from raster whitespace/frame-line gutters. When a figure cannot be split with confidence, it is **kept whole** — a correct whole figure beats a wrong crop.
- **🧾 Reliable panel → caption map** — the caption is split into per-panel descriptions from text (always correct, no vision needed) and exported as metadata *whether or not* a crop exists. No panel information is ever lost.
- **📦 Structured dataset export** — `figure.png` + `caption.txt` + `metadata.json` + optional `panel_x.png/txt/json`, ready to hand to the next agent.
- **🧩 Optional vision assist** — a VLM (local Ollama, GPT-4o, or Claude) can *audit* candidate grids for hard layouts (montages, touching panels). The model never produces coordinates; all cropping is deterministic Python. Off by default.
- **🖥️ Streamlit dashboard** + **CLI** + fully-typed, tested codebase (Pydantic v2, Google-style docstrings, 23 tests).

---

## 🏗️ Architecture

```
                       ┌─────────────────────────┐
                       │  Scientific PDF (📄)     │
                       └────────────┬────────────┘
                                    ▼
   ┌────────────────────────────────────────────────────────────┐
   │ 1 · Figure detection          src/parsing                 │
   │     DocLayout-YOLO on rendered pages → figure + caption    │
   │     regions  (fallback: caption-anchor heuristics)         │
   └────────────────────────────┬───────────────────────────────┘
                                ▼
   ┌────────────────────────────────────────────────────────────┐
   │ 2 · Panel cropping (confidence-first)   src/extraction     │
   │     a. PDF text markers  "(a)" "(b)"  → pixel-exact crops  │
   │     b. raster whitespace / frame-line gutters             │
   │     c. (opt-in) VLM grid-audit debate for hard layouts    │
   │     …else keep the figure whole                            │
   └────────────────────────────┬───────────────────────────────┘
                                ▼
   ┌────────────────────────────────────────────────────────────┐
   │ 3 · Caption → panel map        src/utils/captions          │
   │     split "(a) … (b) … (c) …" into per-panel captions      │
   └────────────────────────────┬───────────────────────────────┘
                                ▼
   ┌────────────────────────────────────────────────────────────┐
   │ 4 · Dataset export             src/extraction/exporter     │
   │     figure.png · caption.txt · metadata.json · panels      │
   └────────────────────────────────────────────────────────────┘

   (Streamlit dashboard + analytical chat live in src/api & src/vision)
```

---

## 🚀 Quickstart

### 1. Clone & create a virtual environment

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
```

> Requires **Python 3.10+**.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> DocLayout-YOLO pulls PyTorch (~2 GB). If you cannot install it, the parser automatically falls back to caption-anchor heuristics — everything else still works.

### 3. Configure (optional)

The default (deterministic) extraction needs **no API keys and no configuration**. Copy the template if you want to change the crop strategy or enable the optional chat / vision assist:

```bash
cp .env.example .env
```

| Variable          | Purpose                                                                    |
| ----------------- | -------------------------------------------------------------------------- |
| `PANEL_CROPPING`  | `deterministic` (default), `markers-only`, or `vlm-assisted`               |
| `VLM_PROVIDER`    | `ollama` (local, no key), `openai`, or `anthropic` — chat & vlm-assisted   |
| `VLM_MODEL`       | Vision model id (e.g. `qwen2.5vl:7b`, `gpt-4o-mini`, `claude-sonnet-5`)     |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Cloud provider keys (API billing, separate from ChatGPT Plus / Claude Pro) |
| `LAYOUT_DETECTION`| `on` (default) / `off` to force the heuristic figure detector              |

---

## 📖 Usage

### Command line — build a dataset

```bash
# Deterministic (default): fast, no model, crops only when confident
python -m src.extraction.cli paper.pdf -o dataset

# Most conservative: crop panels only from PDF text markers
python -m src.extraction.cli paper.pdf -o dataset --cropping markers-only

# Vision-assisted grid audit for hard layouts (needs a capable model)
python -m src.extraction.cli paper.pdf -o dataset --cropping vlm-assisted \
    --provider ollama --model qwen2.5vl:7b
```

Output layout — the **contract** a downstream agent consumes:

```
dataset/<paper>/
    figure_001/
        figure.png          # complete figure, original resolution
        caption.txt         # full caption, verbatim
        metadata.json       # figure metadata + panel → caption map (below)
        panel_a.png         # panel crop — only for confidently cropped panels
        panel_a.txt         # panel caption
        panel_a.json        # panel metadata
        ...
```

`metadata.json` always carries the panel → caption map, **whether or not a crop was produced**:

```json
{
  "figure_id": "figure-3",
  "label": "Figure 3",
  "caption": "FIG. 3. (a) ...  (b) ...",
  "image_size_px": [1079, 460],
  "panel_count": 2,
  "cropped_panel_count": 2,
  "panels": [
    {"panel": "a", "label": "Figure 3(a)", "caption": "FIG. 3. (a) ...", "crop_file": "panel_a.png", "crop_size_px": [520, 460]},
    {"panel": "b", "label": "Figure 3(b)", "caption": "FIG. 3. (b) ...", "crop_file": "panel_b.png", "crop_size_px": [540, 460]}
  ]
}
```

A figure kept whole still lists its panels with `"crop_file": null` — the analysis agent gets the full figure plus the panel map, and uses crops only when they exist.

### Interactive dashboard

```bash
streamlit run src/api/app.py
```

Upload a PDF → the sidebar shows the extracted main figures → pick one to see it, its caption, its panel crops, and an analytical chat (needs a configured `VLM_PROVIDER`).

### As a library

```python
from src.extraction.framework import ExtractionConfig, FigureExtractionFramework

framework = FigureExtractionFramework(config=ExtractionConfig())  # deterministic
framework.extract_to_dataset("paper.pdf", "dataset")

# Or work with the objects directly:
document = framework.extract("paper.pdf")
for fig in document.figures:
    if fig.parent_id is None:
        print(fig.label, "→", fig.caption[:80])
```

---

## ⚙️ How panel cropping works (confidence-first)

Panels are produced by tiers, most-reliable first. The first tier that can split a figure *with confidence* wins; otherwise the figure is kept whole.

1. **PDF text markers.** When the `(a)`, `(b)`, … labels exist as PDF text inside the figure, panels are cut at pixel-exact boundaries from the marker geometry (2-D grid reconstruction, caption-guided so missing markers and cross-references are handled).
2. **Raster whitespace / frame-line gutters.** The rendered figure is segmented along its whitespace gutters **and** thin frame/grid lines (spikes in the ink projection — essential for dense figures ringed by black borders that have no white margin). Trusted only when the cell count matches the caption's panel count exactly.
3. **Vision-assisted grid audit** *(opt-in, `--cropping vlm-assisted`)*. Python proposes candidate grids (ranked by how little content each cut line crosses, snapped to real gutters), draws each numbered on the figure, and a VLM **audits** it — counting panels per cell. Its report drives deterministic edits (valley splits / merges) until the audit is clean. Two models can **debate** (alternate as auditor, both must sign off). The model never emits coordinates.

**Why confidence-first?** Across many real papers, a mis-cropped panel (half of two panels) actively misleads a downstream model, whereas the complete figure is always correct. So the framework prefers *correct-but-whole* over *split-but-wrong* — and the caption-derived panel map means no panel information is lost either way.

---

## 🗂️ Repository layout

```
├── src/
│   ├── models.py               # Domain models (BoundingBox, PixelBox, ExtractedFigure, …)
│   ├── parsing/
│   │   ├── pdf_parser.py       # Figures, captions, marker panels
│   │   └── layout_detector.py  # DocLayout-YOLO figure/caption detection
│   ├── extraction/
│   │   ├── framework.py        # FigureExtractionFramework + ExtractionConfig
│   │   ├── raster_splitter.py  # Deterministic gutter/line segmentation & grid ops
│   │   ├── crop_engine.py      # Crop validation & bounded expansion
│   │   ├── exporter.py         # DatasetExporter (Option-A output contract)
│   │   └── cli.py              # python -m src.extraction.cli
│   ├── vision/
│   │   ├── backend.py          # Swappable VisionBackend (OpenAI / Anthropic / Ollama) + cost meter
│   │   ├── panel_detector.py   # Grid auditing / label reading (vlm-assisted)
│   │   └── vlm_client.py       # FigureAnalyzer — analytical chat over a figure
│   ├── utils/captions.py       # Caption → per-panel split (contiguous-prefix, ranges)
│   ├── rag/vector_db.py        # In-memory figure + caption + reference index (chat)
│   ├── pipeline.py             # Orchestrator for the dashboard
│   └── api/app.py              # Streamlit dashboard
├── tests/                      # 23 tests (pytest) — no model or API key needed
├── requirements.txt · requirements-dev.txt · pyproject.toml
├── .env.example · .gitignore · LICENSE (MIT)
└── README.md
```

---

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite (23 tests) covers the deterministic splitter (gutters, frame lines, XY-cut, grid snapping, caption splitting), the crop engine, the dataset contract, and the vision-audit path with stubbed backends — no model or API key required.

---

## 🤖 Notes on the vision model

The default extraction is **fully deterministic** — no model runs. The vision model is only used for (a) the optional analytical chat and (b) the opt-in `vlm-assisted` crop mode. It's swappable behind one interface (`src/vision/backend.py`):

- **Ollama (local, free):** `ollama pull qwen2.5vl:7b` then set `VLM_PROVIDER=ollama`. Must be a *vision* model.
- **Cloud (OpenAI / Anthropic):** set the provider, model, and API key. This is a **pay-per-token API, separate from ChatGPT Plus / Claude Pro subscriptions.** The framework logs token usage and an estimated cost per document so you can measure spend.

Local 7B–12B models are adequate for the chat and for easy layouts; hard panel-segmentation judgments are where a larger/cloud model helps.

---

## 🤝 Contributing

Issues and PRs welcome — use the [bug report](.github/ISSUE_TEMPLATE/bug_report.md) and [feature request](.github/ISSUE_TEMPLATE/feature_request.md) templates. Interesting areas: raster-only montage handling, column-major panel labeling without a VLM, and table extraction.

## 📄 License

Released under the [MIT License](LICENSE).
