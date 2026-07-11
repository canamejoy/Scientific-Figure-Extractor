---
name: 🐛 Bug report
about: Report a reproducible problem with the figure extraction pipeline or the UI
title: "[Bug] "
labels: bug
assignees: ""
---

## Describe the bug

A clear, concise description of what went wrong.

## To reproduce

Steps to reproduce the behavior:

1. Upload PDF `...` (attach it if licensing allows, or link to the arXiv page)
2. Select figure `...`
3. Ask `...`
4. See error

## Expected behavior

What you expected to happen instead.

## Screenshots / logs

If applicable, add screenshots of the Streamlit UI and paste the relevant
terminal log output (the app logs every pipeline stage).

```text
<paste logs here>
```

## Environment

- OS: [e.g. Ubuntu 24.04, Windows 11, macOS 15]
- Python version: [e.g. 3.11.9]
- Package versions: output of `pip freeze | grep -Ei "streamlit|pdfplumber|pillow|langchain|pydantic"`
- VLM provider/model: [e.g. openai / gpt-4o]

## PDF characteristics (for extraction bugs)

- Single- or two-column layout?
- Are the figures raster images or vector plots (e.g. matplotlib)?
- Caption style: [e.g. "Figure 1:", "Fig. 1.", "FIGURE 1 —"]

## Additional context

Anything else that could help us diagnose the issue.
