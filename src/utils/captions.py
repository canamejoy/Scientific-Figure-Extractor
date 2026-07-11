"""Caption utilities shared by the geometric splitter and the VLM framework.

Splitting a figure caption into per-panel descriptions is a deterministic
text problem, so it lives here — outside both the PDF parser and the vision
modules — and is reused by every panel-producing code path.
"""

from __future__ import annotations

import re
from typing import Dict

# Panel markers inside a caption's prose. Three forms, ranges first so the
# alternation prefers them: "(a)-(d) ...", "(a-d) ...", "(a) ...".
_CAPTION_TOKEN: "re.Pattern[str]" = re.compile(
    r"\(\s*([a-h])\s*\)\s*[-–—]\s*\(\s*([a-h])\s*\)"  # (a)-(d)
    r"|\(\s*([a-h])\s*[-–—]\s*([a-h])\s*\)"  # (a-d)
    r"|\(\s*([a-h])\s*\)",  # (a)
    re.IGNORECASE,
)


def split_caption(caption: str) -> Dict[str, str]:
    """Splits a caption into per-panel descriptions.

    Captions of multi-panel figures typically read "<intro> (a) ... (b) ...
    (c) ...". Each panel gets the shared intro (which carries the figure
    number and general context) plus its own segment, ending where the next
    panel marker begins — never the entire caption.

    Range markers — "(a)-(d) Annihilation process." — expand to every letter
    they cover, each receiving that shared segment; this also makes the set
    of keys the correct panel COUNT for figures described by ranges.

    **Only the contiguous alphabetical sequence starting at (a) is kept.**
    A figure's panels are always labeled a, b, c, ... without gaps, so a
    letter that breaks the run is a cross-reference to *another* figure's
    panel ("... as in Fig. 2(c) ...") — counting it would inflate the panel
    count and derail every downstream grid. The prefix a..a+k−1 present in
    the caption is the panel set; out-of-sequence letters are dropped.

    Args:
        caption: The full caption text.

    Returns:
        Mapping of panel letter to its caption; empty when the caption
        contains no panel markers (or none starting at "(a)").
    """
    tokens = list(_CAPTION_TOKEN.finditer(caption))
    if not tokens:
        return {}
    intro = caption[: tokens[0].start()].strip()
    result: Dict[str, str] = {}
    for index, token in enumerate(tokens):
        end = tokens[index + 1].start() if index + 1 < len(tokens) else len(caption)
        segment = caption[token.start() : end].strip(" ,;")
        text = f"{intro} {segment}".strip() if intro else segment
        first = (token.group(1) or token.group(3) or token.group(5)).lower()
        last = (token.group(2) or token.group(4) or first).lower()
        if ord(last) < ord(first):
            last = first
        for code in range(ord(first), ord(last) + 1):
            # First occurrence wins (repeated letters may reference back).
            result.setdefault(chr(code), text)

    # Keep only the contiguous prefix a, b, c, ... — drop cross-references.
    contiguous: Dict[str, str] = {}
    for offset, letter in enumerate(chr(ord("a") + i) for i in range(8)):
        if letter not in result:
            break
        contiguous[letter] = result[letter]
    return contiguous
