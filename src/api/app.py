"""Streamlit dashboard for the AI Scientific Figure Analyzer.

Run from the repository root with::

    streamlit run src/api/app.py

Workflow: upload a scientific PDF -> the backend crops every figure (and its
panels) at high resolution and indexes each one with its caption and in-text
references -> pick a main figure -> inspect its panels in the workspace ->
interrogate the figure (or one panel) through the analytical chat.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

# Make the project root importable so absolute "src.*" imports resolve when
# the app is launched via "streamlit run src/api/app.py".
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models import ExtractedFigure, FigureContext  # noqa: E402
from src.pipeline import Orchestrator  # noqa: E402

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Scientific Figure Analyzer", layout="wide")


# ---------------------------------------------------------------------- #
# Session-state helpers                                                   #
# ---------------------------------------------------------------------- #


def _figure_option_label(figure: ExtractedFigure) -> str:
    """The display string identifying a figure in the selection widgets."""
    return f"{figure.label} — page {figure.page_number}"


def _orchestrator() -> Orchestrator:
    """Returns the per-session pipeline orchestrator, creating it on demand."""
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = Orchestrator()
        st.session_state.chat_histories = {}
        st.session_state.processed_digest = None
    return st.session_state.orchestrator


def _check_api_key() -> None:
    """Warns early when the configured VLM backend has no API key.

    Local backends (``ollama``) need no key, so no warning is shown; the
    model itself must still be vision-capable (e.g. ``qwen2.5vl``).
    """
    provider = (os.getenv("VLM_PROVIDER") or "openai").strip().lower()
    if provider == "ollama":
        return
    required = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    if not os.getenv(required):
        st.warning(
            f"`{required}` is not set — figure chat will fail until you add "
            "it to your `.env` file (see `.env.example`)."
        )


def _process_upload(uploaded_file: "st.runtime.uploaded_file_manager.UploadedFile") -> None:
    """Runs the extraction pipeline on a newly uploaded PDF.

    The upload is hashed so re-runs of the Streamlit script (which happen on
    every widget interaction) do not re-process the same file.
    """
    payload = uploaded_file.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    if st.session_state.get("processed_digest") == digest:
        return  # this exact file was already processed in this session

    orchestrator = _orchestrator()
    tmp_path: Optional[Path] = None
    try:
        with st.spinner(
            "Parsing layout, cropping figures at high DPI, and indexing context…"
        ):
            # pdfplumber needs a real file path; write the upload to a
            # temporary file that is removed as soon as parsing finishes.
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(payload)
                tmp_path = Path(tmp.name)
            orchestrator.process_document(str(tmp_path))
    except Exception as exc:
        logger.exception("Failed to process uploaded PDF")
        st.error(f"Could not process this PDF: {exc}")
        return
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    st.session_state.processed_digest = digest
    st.session_state.chat_histories = {}
    # Drop selections carried over from a previous document.
    st.session_state.pop("figure_select", None)
    st.session_state.pop("chat_target", None)

    count = len(orchestrator.figures)
    if count:
        st.success(
            f"Extracted and indexed {count} figure(s). "
            "Pick one below to start the analysis."
        )
    else:
        st.warning(
            "No figures were detected in this document. Check the terminal "
            "log for details."
        )


# ---------------------------------------------------------------------- #
# Rendering                                                               #
# ---------------------------------------------------------------------- #


def _render_sidebar(parents: List[ExtractedFigure]) -> None:
    """Renders the gallery of MAIN figures in the sidebar.

    Panels are intentionally not listed here — they are shown in the main
    workspace under their figure. The sidebar is rendered before the main
    dropdown widget is instantiated, so writing to its ``figure_select``
    session key here takes effect on the same rerun.
    """
    with st.sidebar:
        st.header("Figure gallery")
        if not parents:
            st.caption("Upload a PDF to populate the gallery.")
            return
        for figure in parents:
            st.image(figure.image, width="stretch")
            st.markdown(f"**{figure.label}** · page {figure.page_number}")
            if st.button(
                f"Open {figure.label}",
                key=f"select-{figure.figure_id}",
                width="stretch",
            ):
                st.session_state["figure_select"] = _figure_option_label(figure)
                st.session_state.pop("chat_target", None)
            st.divider()


def _render_panels(panels: List[ExtractedFigure]) -> None:
    """Renders the panel breakdown of the selected figure as a grid.

    Each panel card shows its crop and caption, plus a button that targets
    the analytical chat at that specific panel.
    """
    st.subheader("Panels")
    if not panels:
        st.caption(
            "No individual panels were extracted for this figure — the chat "
            "below analyzes the complete figure."
        )
        return
    columns = st.columns(3, gap="medium")
    for index, panel in enumerate(panels):
        with columns[index % 3]:
            st.image(panel.image, width="stretch")
            st.markdown(f"**{panel.label}**")
            st.caption(panel.caption)
            if st.button(
                f"Chat about {panel.label}",
                key=f"target-{panel.figure_id}",
                width="stretch",
            ):
                st.session_state["chat_target"] = panel.figure_id


def _render_context_panel(context: FigureContext) -> None:
    """Shows the caption, retrieved references, and extraction metadata."""
    figure = context.figure
    with st.expander("Caption & retrieved paper context", expanded=False):
        st.markdown(f"**Caption** — {figure.caption}")
        if context.references:
            st.markdown("**In-text references:**")
            for index, passage in enumerate(context.references, start=1):
                st.markdown(f"{index}. {passage}")
        else:
            st.caption("No explicit in-text references to this figure were found.")
        st.caption(
            f"Detection: `{figure.detection_method}` · origin `{figure.origin}` · "
            f"rendered at {figure.dpi} DPI · crop "
            f"{figure.image.width}×{figure.image.height} px"
        )


def _render_chat(orchestrator: Orchestrator, figure: ExtractedFigure) -> None:
    """Renders the analytical chat bound to ``figure`` (figure or panel)."""
    histories: Dict[str, List[Dict[str, str]]] = st.session_state.setdefault(
        "chat_histories", {}
    )
    history = histories.setdefault(figure.figure_id, [])

    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(f"Ask an analytical question about {figure.label}…")
    if not prompt:
        return

    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Inspecting the figure with the VLM…"):
            try:
                answer = orchestrator.query_figure(figure.figure_id, prompt)
            except Exception as exc:
                logger.exception("VLM query failed")
                answer = f"Analysis failed: {exc}"
        st.markdown(answer)
    history.append({"role": "assistant", "content": answer})


# ---------------------------------------------------------------------- #
# Main page                                                               #
# ---------------------------------------------------------------------- #


def main() -> None:
    """Entry point executed on every Streamlit rerun."""
    orchestrator = _orchestrator()

    st.title("AI Scientific Figure Analyzer")
    _check_api_key()

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded is not None:
        _process_upload(uploaded)

    figures = orchestrator.figures
    parents = [figure for figure in figures if figure.parent_id is None]
    children: Dict[str, List[ExtractedFigure]] = {}
    for figure in figures:
        if figure.parent_id is not None:
            children.setdefault(figure.parent_id, []).append(figure)

    _render_sidebar(parents)

    if not parents:
        st.info(
            "Upload a paper to extract its figures. Each one is cropped at "
            "high resolution, split into panels, and bound to its caption "
            "and in-text references."
        )
        return

    # Only MAIN figures are selectable here; panels live below the figure.
    options = {_figure_option_label(figure): figure.figure_id for figure in parents}
    if st.session_state.get("figure_select") not in options:
        st.session_state.pop("figure_select", None)
    selected_label = st.selectbox(
        "Figure under analysis", list(options), key="figure_select"
    )
    selected_id = options[selected_label]

    context = orchestrator.get_context(selected_id)
    figure = context.figure
    panels = children.get(selected_id, [])

    # 1. The complete main figure.
    image_col, meta_col = st.columns([3, 2], gap="large")
    with image_col:
        st.image(
            figure.image,
            caption=f"{figure.label} (page {figure.page_number})",
            width="stretch",
        )
    with meta_col:
        st.subheader(figure.label)
        st.markdown(figure.caption)
        _render_context_panel(context)

    # 2. The cropped panels, right below the main figure.
    st.divider()
    _render_panels(panels)

    # 3. The analytical chat, targeting the figure or one of its panels.
    valid_targets = {selected_id} | {panel.figure_id for panel in panels}
    target_id = st.session_state.get("chat_target")
    if target_id not in valid_targets:
        target_id = selected_id
        st.session_state["chat_target"] = target_id
    target = orchestrator.get_context(target_id).figure

    st.divider()
    st.subheader("Analytical chat")
    caption_col, reset_col = st.columns([4, 1])
    with caption_col:
        st.caption(
            f"Chat target: **{target.label}**. Use the buttons under each "
            "panel to focus the chat on it."
        )
    with reset_col:
        if target_id != selected_id and st.button(
            "Whole figure", width="stretch"
        ):
            st.session_state["chat_target"] = selected_id
            target = figure
    _render_chat(orchestrator, target)


main()
