"""Build a small fake 'paper' PDF with real vector charts, captions and body text.

Used by the tests and as a demo input when you don't have a paper at hand.
The true data behind every chart is written to a JSON file next to the PDF,
so extraction can be scored on it.

    python scripts/make_sample_paper.py data/sample_paper.pdf
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymupdf

CHARTS = [
    {
        "id": "1", "type": "bar",
        "caption": "Figure 1: Share of papers with figure alt text by venue (percent).",
        "x": ["CHI", "ASSETS", "UIST", "CSCW", "arXiv"], "y": [38.0, 61.0, 12.0, 9.0, 3.0],
        "xlabel": "Venue", "ylabel": "Papers with alt text (%)",
        "para": "As Figure 1 shows, alt text coverage is far higher at accessibility venues. ASSETS leads, "
                "while general preprint servers have almost no coverage. This gap motivates automated support for authors.",
    },
    {
        "id": "2", "type": "line",
        "caption": "Figure 2: Hallucinated numbers per 100 sentences across training steps.",
        "x": [0, 1000, 2000, 3000, 4000, 5000], "y": [14.2, 9.8, 6.1, 4.0, 3.6, 3.5],
        "xlabel": "Training step", "ylabel": "Hallucinations per 100 sent.",
        "para": "Figure 2 plots hallucination rate during training. Most of the improvement happens in the first "
                "3,000 steps, after which the curve flattens, suggesting diminishing returns from further training.",
    },
    {
        "id": "3", "type": "bar",
        "caption": "Figure 3: Median latency per figure for each pipeline stage (seconds).",
        "x": ["Detect", "Extract", "Retrieve", "Generate", "Verify"], "y": [0.4, 6.8, 0.3, 4.1, 1.2],
        "xlabel": "Pipeline stage", "ylabel": "Median latency (s)",
        "para": "Latency is dominated by the extraction stage (Figure 3), which reads the chart image with a "
                "vision-language model. Retrieval and detection are nearly free by comparison.",
    },
]

FILLER = ("Scholarly figures carry much of a paper's evidence, yet readers who use screen readers often cannot "
          "access them. We study automated description of charts and evaluate how often generated descriptions "
          "state values that do not appear in the underlying data. ") * 2


def chart_pdf(c) -> bytes:
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    if c["type"] == "bar":
        ax.bar(c["x"], c["y"], color="#4C72B0")
    else:
        ax.plot(c["x"], c["y"], marker="o", color="#4C72B0")
    ax.set_xlabel(c["xlabel"])
    ax.set_ylabel(c["ylabel"])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    return buf.getvalue()


def build(out: Path):
    doc = pymupdf.open()
    W, H = 612, 792
    page = doc.new_page(width=W, height=H)
    y = 60
    page.insert_textbox(pymupdf.Rect(60, y, W - 60, y + 40), "Measuring Hallucinations in Chart Descriptions",
                        fontsize=16, fontname="helv")
    y += 40
    page.insert_textbox(pymupdf.Rect(60, y, W - 60, y + 90), FILLER, fontsize=9.5, fontname="helv")
    y += 100
    for c in CHARTS:
        if y > H - 330:
            page = doc.new_page(width=W, height=H)
            y = 60
        src = pymupdf.open("pdf", chart_pdf(c))
        r = pymupdf.Rect(90, y, W - 90, y + 260)
        page.show_pdf_page(r, src, 0)
        y += 266
        page.insert_textbox(pymupdf.Rect(60, y, W - 60, y + 26), c["caption"], fontsize=9, fontname="helv")
        y += 34
        page.insert_textbox(pymupdf.Rect(60, y, W - 60, y + 60), c["para"] + " " + FILLER[:200], fontsize=9.5, fontname="helv")
        y += 74
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    truth = {c["id"]: {k: c[k] for k in ("type", "caption", "x", "y", "xlabel", "ylabel")} for c in CHARTS}
    out.with_suffix(".truth.json").write_text(json.dumps(truth, indent=2))
    return out


if __name__ == "__main__":
    p = build(Path(sys.argv[1] if len(sys.argv) > 1 else "data/sample_paper.pdf"))
    print(f"wrote {p} and {p.with_suffix('.truth.json')}")
