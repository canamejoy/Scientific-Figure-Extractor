"""Command-line interface for the figure-extraction framework.

Examples::

    # Deterministic extraction (default — no model, no credentials needed)
    python -m src.extraction.cli paper.pdf -o dataset

    # Most conservative: crop panels only from PDF text markers
    python -m src.extraction.cli paper.pdf -o dataset --cropping markers-only

    # Vision-assisted grid audit for hard layouts (needs a capable model)
    python -m src.extraction.cli paper.pdf -o dataset --cropping vlm-assisted \\
        --provider ollama --model qwen2.5vl:7b

    # From a JSON configuration file
    python -m src.extraction.cli paper.pdf -o dataset --config extraction.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.extraction.framework import ExtractionConfig, FigureExtractionFramework


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.extraction.cli",
        description=(
            "Extract figures and subfigure panels from a scientific PDF into "
            "a structured dataset."
        ),
    )
    parser.add_argument("pdf", type=Path, help="Path to the scientific PDF.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("dataset"),
        help="Dataset output directory (default: ./dataset).",
    )
    parser.add_argument(
        "--cropping",
        choices=["deterministic", "markers-only", "vlm-assisted"],
        default=None,
        help="Panel-crop strategy (default: deterministic).",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Vision provider: openai | anthropic | ollama (vlm-assisted only).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Vision model identifier (default: env / provider default).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON configuration file (CLI flags override its values).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Dataset folder name (default: the PDF's file stem).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser


def main(argv: list = None) -> int:
    """CLI entry point.

    Returns:
        Process exit code (0 on success).
    """
    args = build_arg_parser().parse_args(argv)
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    logger = logging.getLogger("extraction.cli")

    config = (
        ExtractionConfig.from_json(args.config) if args.config else ExtractionConfig()
    )
    # Explicit CLI flags override the configuration file.
    if args.cropping:
        config = config.model_copy(update={"panel_cropping": args.cropping})
    if args.provider:
        config = config.model_copy(update={"provider": args.provider})
    if args.model:
        config = config.model_copy(update={"model": args.model})

    framework = FigureExtractionFramework(config=config)
    try:
        paper_dir = framework.extract_to_dataset(
            args.pdf, args.output, paper_name=args.name
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    except Exception:
        logger.exception("Extraction failed")
        return 1

    logger.info("Done — dataset at %s", paper_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
