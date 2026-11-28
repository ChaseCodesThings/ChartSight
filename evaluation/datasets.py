"""Test-set loaders. Every loader yields EvalItem records with an image and ground truth.

VisText layout (after running the VisText repo's download_data.sh --images):
    <dir>/data_test.json      list of {"img_id", "caption_id", "datatable", "caption_L1", "caption_L2L3", ...}
    <dir>/images/<img_id>.png
A chart has several reference captions (one row per caption_id); we group them by img_id.

Paper test set (your own annotated real figures):
    <dir>/annotations.json    {"<figure_key>": {"pdf": "file.pdf", "figure_id": "fig2",
                                "x": [...], "y": [...], "caption": "...", "references": ["..."]}}
    <dir>/*.pdf
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from chartsight.numbers import find_numbers
from chartsight.schema import Axis, ChartData, Point, Series

_NUM_TOKEN = re.compile(r"^-?\d[\d,]*(?:\.\d+)?%?$")


@dataclass
class EvalItem:
    key: str
    image: bytes
    truth: ChartData
    truth_numbers: list[float]
    references: list[str] = field(default_factory=list)
    caption: Optional[str] = None
    paper_text: Optional[str] = None
    meta: dict = field(default_factory=dict)


def _is_num(tok: str) -> bool:
    return bool(_NUM_TOKEN.match(tok))


def _num(tok: str) -> float:
    return float(tok.replace(",", "").rstrip("%"))


def parse_datatable(s: str) -> tuple[str, list[tuple[str, float]], list[float]]:
    """Parse VisText's linearized table: '<title> <s> <x name> <y name> x1 y1 x2 y2 ...'.

    Labels can be several words, so pairs are read from the END: each y is a number, and its
    x is either the single number before it (years, steps) or the run of words before it.
    The first label may absorb the axis names; numbers are unaffected.
    Run `python -m evaluation.datasets --inspect DIR` to eyeball the parse on real rows.
    """
    title, _, body = s.partition("<s>")
    toks = body.split()
    pairs: list[tuple[str, float]] = []
    i = len(toks) - 1
    while i >= 1 and _is_num(toks[i]):
        y = _num(toks[i])
        i -= 1
        if _is_num(toks[i]):
            pairs.append((toks[i], y))
            i -= 1
        else:
            j = i
            while j - 1 >= 0 and not _is_num(toks[j - 1]):
                j -= 1
            pairs.append((" ".join(toks[j:i + 1]), y))
            i = j - 1
    pairs.reverse()
    numbers = [m.value for m in find_numbers(body)]
    return title.strip(), pairs, numbers


def chart_from_pairs(title: str, pairs: list[tuple[str, float]], numbers: list[float]) -> ChartData:
    ys = [y for _, y in pairs] or numbers
    pts = [Point(x=x, y=y) for x, y in pairs] if pairs else [Point(x="", y=v) for v in numbers]
    return ChartData(title=title or None, series=[Series(points=pts)],
                     y_axis=Axis(min=min(ys) if ys else None, max=max(ys) if ys else None))


def load_vistext(root: str | Path, split: str = "test", n: Optional[int] = None, seed: int = 0) -> list[EvalItem]:
    root = Path(root)
    fname = {"val": "validation"}.get(split, split)
    rows = json.loads((root / f"data_{fname}.json").read_text())
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(str(r["img_id"]), []).append(r)
    keys = sorted(groups, key=lambda k: int(k) if k.isdigit() else k)
    random.Random(seed).shuffle(keys)
    items = []
    for k in keys:
        img = root / "images" / f"{k}.png"
        if not img.exists():
            continue
        g = groups[k]
        title, pairs, numbers = parse_datatable(g[0]["datatable"])
        refs = [f"{r.get('caption_L1', '').strip()} {r.get('caption_L2L3', '').strip()}".strip() for r in g]
        items.append(EvalItem(key=f"vistext-{k}", image=img.read_bytes(), truth=chart_from_pairs(title, pairs, numbers),
                              truth_numbers=numbers, references=refs,
                              meta={"source": "vistext", "n_points": len(pairs)}))
        if n and len(items) >= n:
            break
    if not items:
        raise FileNotFoundError(f"no VisText charts with images found in {root} (need data_{fname}.json and images/)")
    return items


def load_papers(root: str | Path) -> list[EvalItem]:
    from chartsight.figures import extract_figures

    root = Path(root)
    ann = json.loads((root / "annotations.json").read_text())
    papers: dict[str, object] = {}
    items = []
    for key, a in ann.items():
        pdf = a["pdf"]
        if pdf not in papers:
            papers[pdf] = extract_figures((root / pdf).read_bytes())
        paper = papers[pdf]
        fig = next((f for f in paper.figures if f.figure_id == a["figure_id"]), None)
        if fig is None:
            print(f"warning: {key}: figure {a['figure_id']} not detected in {pdf}; found {[f.figure_id for f in paper.figures]}")
            continue
        pairs = [(str(x), float(y)) for x, y in zip(a["x"], a["y"])]
        numbers = [float(y) for y in a["y"]] + [float(x) for x in a["x"] if isinstance(x, (int, float))]
        items.append(EvalItem(key=key, image=fig.image, truth=chart_from_pairs(a.get("title", ""), pairs, numbers),
                              truth_numbers=numbers, references=a.get("references", []),
                              caption=a.get("caption") or fig.caption, paper_text=paper.text,
                              meta={"source": "paper", "pdf": pdf, "n_points": len(pairs)}))
    return items


def sample_paper_as_testset(pdf: str | Path) -> list[EvalItem]:
    """The generated sample paper, scored against its truth file (used by the smoke test)."""
    from chartsight.figures import extract_figures

    pdf = Path(pdf)
    truth = json.loads(pdf.with_suffix(".truth.json").read_text())
    paper = extract_figures(pdf.read_bytes())
    items = []
    for fid, t in truth.items():
        fig = next(f for f in paper.figures if f.figure_id == f"fig{fid}")
        pairs = [(str(x), float(y)) for x, y in zip(t["x"], t["y"])]
        numbers = [float(y) for y in t["y"]]
        items.append(EvalItem(key=f"sample-fig{fid}", image=fig.image, truth=chart_from_pairs("", pairs, numbers),
                              truth_numbers=numbers, caption=fig.caption, paper_text=paper.text,
                              meta={"source": "sample", "n_points": len(pairs)}))
    return items


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Check the VisText parser on a few real rows")
    ap.add_argument("--inspect", required=True, help="VisText data directory")
    ap.add_argument("-n", type=int, default=5)
    a = ap.parse_args()
    for it in load_vistext(a.inspect, n=a.n):
        print(f"\n{it.key}  points={it.meta['n_points']}")
        print("  parsed:", [(p.x, p.y) for p in it.truth.series[0].points][:12])
        print("  reference:", it.references[0][:200])
