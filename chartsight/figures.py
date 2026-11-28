"""Stage 2: find figures in a paper PDF and crop each one with its caption.

Papers draw charts two ways: as embedded raster images, or as vector graphics
(most matplotlib and ggplot figures). We use the caption as the anchor:
a block starting with "Figure N" or "Fig. N" marks a figure, and the figure is the
graphics region directly above it (or below it, for papers that caption on top).
Embedded images with no nearby caption are kept as a fallback.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pymupdf

CAPTION_RE = re.compile(r"^\s*(fig(?:ure)?\.?\s*\d+[a-z]?\s*[:.|])", re.I)  # "Figure 2:" or "Fig. 2." not "Figure 2 shows"


@dataclass
class Figure:
    figure_id: str
    page: int  # 1-based
    bbox: tuple[float, float, float, float]
    image: bytes  # PNG
    caption: Optional[str] = None
    source: str = "caption"  # caption | embedded


@dataclass
class Paper:
    text: str
    figures: list[Figure] = field(default_factory=list)
    pages: int = 0


def _graphics_rects(page) -> list[pymupdf.Rect]:
    rects = []
    for d in page.get_drawings():
        r = d.get("rect")
        # keep lines and shapes; skip page-sized frames and backgrounds
        if r is not None and (r.width > 2 or r.height > 2) and r.width < page.rect.width * 0.95:
            rects.append(pymupdf.Rect(r))
    for info in page.get_image_info():
        rects.append(pymupdf.Rect(info["bbox"]))
    return [r for r in rects if not r.is_empty]


def _union(rects) -> Optional[pymupdf.Rect]:
    out = None
    for r in rects:
        out = pymupdf.Rect(r) if out is None else out | r
    return out


def _render(page, rect: pymupdf.Rect, dpi: int) -> bytes:
    return page.get_pixmap(clip=rect, dpi=dpi).tobytes("png")


def extract_figures(pdf_bytes: bytes, dpi: int = 200, min_size: float = 60) -> Paper:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    text_parts, figures = [], []
    for pno, page in enumerate(doc):
        text_parts.append(page.get_text("text"))
        blocks = [b for b in page.get_text("blocks") if b[6] == 0]  # text blocks only
        graphics = _graphics_rects(page)
        used = []
        captions = [b for b in blocks if CAPTION_RE.match(b[4])]
        for b in captions:
            cap_rect = pymupdf.Rect(b[:4])
            caption = re.sub(r"\s+", " ", b[4]).strip()
            # graphics above the caption, down to the previous body-text block
            body_above = [pymupdf.Rect(t[:4]) for t in blocks
                          if t[3] <= cap_rect.y0 + 1 and t is not b and len(t[4].strip()) > 80
                          and t[2] - t[0] > page.rect.width * 0.3]
            ceiling = max((r.y1 for r in body_above), default=page.rect.y0)
            # same column as the caption (or a full-width figure)
            col = (min(cap_rect.x0, page.rect.width * 0.5), max(cap_rect.x1, page.rect.width * 0.5))
            above = [g for g in graphics if g.y1 <= cap_rect.y0 + 4 and g.y0 >= ceiling - 2
                     and g.x1 > col[0] - 20 and g.x0 < col[1] + page.rect.width * 0.5]
            region = _union(above)
            if region is None or region.width < min_size or region.height < min_size:
                below = [g for g in graphics if g.y0 >= cap_rect.y1 - 4 and g.y0 < cap_rect.y1 + page.rect.height * 0.5]
                region = _union(below)
            if region is None or region.width < min_size or region.height < min_size:
                continue
            # include tick labels and axis titles that sit just outside the drawn lines
            labels = [pymupdf.Rect(t[:4]) for t in blocks if t is not b and pymupdf.Rect(t[:4]).intersects(region + (-30, -14, 30, 14))
                      and len(t[4].strip()) < 80]
            region = _union([region, *labels]) + (-4, -4, 4, 4)
            region &= page.rect
            used.append(region)
            num = re.search(r"\d+[a-z]?", caption)
            fid = f"fig{num.group(0)}" if num else f"p{pno + 1}-{len(figures) + 1}"
            if any(f.figure_id == fid for f in figures):
                fid = f"{fid}-p{pno + 1}"
            figures.append(Figure(figure_id=fid, page=pno + 1, bbox=tuple(region), image=_render(page, region, dpi),
                                  caption=caption))
        # embedded images that no caption claimed
        for info in page.get_image_info():
            r = pymupdf.Rect(info["bbox"])
            if r.width < min_size * 2 or r.height < min_size * 2 or any(r.intersects(u) for u in used):
                continue
            figures.append(Figure(figure_id=f"p{pno + 1}-img{len(figures) + 1}", page=pno + 1, bbox=tuple(r),
                                  image=_render(page, r, dpi), caption=None, source="embedded"))
    return Paper(text="\n\n".join(text_parts), figures=figures, pages=len(doc))
