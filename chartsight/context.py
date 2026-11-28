"""Stage 4: retrieve the paper paragraphs that explain a figure.

Hybrid retrieval:
  1. exact references ("Figure 3", "Fig. 3") always come first
  2. then BM25 keyword ranking against the caption
  3. plus dense embeddings (sentence-transformers) when installed; scores are fused by rank
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("the a an of and or to in on for with by is are was were be this that these those as at from it its "
            "we our their which than then also can may into over under between".split())


def _tok(s: str) -> list[str]:
    return [t for t in _TOKEN.findall(s.lower()) if t not in _STOP]


def split_paragraphs(text: str, min_chars: int = 120, max_chars: int = 1200) -> list[str]:
    """Split extracted PDF text into paragraph-sized chunks."""
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join hyphenated line breaks
    blocks = [re.sub(r"\s*\n\s*", " ", b).strip() for b in re.split(r"\n\s*\n", text)]
    chunks, buf = [], ""

    def flush(s: str):
        while len(s) > max_chars:
            cut = s.rfind(". ", 0, max_chars) + 1 or max_chars
            chunks.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            chunks.append(s)

    for b in blocks:
        if not b:
            continue
        buf = f"{buf} {b}".strip()
        if len(buf) >= min_chars:
            flush(buf)
            buf = ""
    flush(buf)
    return chunks


def figure_number(caption: Optional[str]) -> Optional[str]:
    if not caption:
        return None
    m = re.match(r"\s*(?:fig(?:ure)?\.?)\s*(\d+[a-z]?)", caption, re.I)
    return m.group(1) if m else None


@lru_cache(maxsize=1)
def _embedder(name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    return SentenceTransformer(name)


class Retriever:
    def __init__(self, paragraphs: list[str], use_embeddings: bool = True,
                 embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.paragraphs = paragraphs
        self.bm25 = BM25Okapi([_tok(p) for p in paragraphs]) if paragraphs else None
        self.embed = _embedder(embed_model) if (use_embeddings and paragraphs) else None
        self.vecs = self.embed.encode(paragraphs, normalize_embeddings=True) if self.embed else None

    @property
    def mode(self) -> str:
        return "bm25+embeddings" if self.embed else "bm25"

    def for_figure(self, caption: Optional[str], k: int = 3) -> list[str]:
        if not self.paragraphs:
            return []
        num = figure_number(caption)
        picked: list[int] = []
        if num:
            ref = re.compile(rf"\b(?:fig(?:ure)?s?\.?)\s*{re.escape(num)}\b", re.I)
            picked = [i for i, p in enumerate(self.paragraphs)
                      if ref.search(p) and not (caption and p.strip().startswith(caption.strip()[:40]))]
        query = caption or ""
        if len(picked) < k and query:
            ranks: dict[int, float] = {}
            scores = self.bm25.get_scores(_tok(query))
            for r, i in enumerate(sorted(range(len(scores)), key=lambda i: -scores[i])[:20]):
                ranks[i] = ranks.get(i, 0) + 1 / (60 + r)
            if self.vecs is not None:
                q = self.embed.encode([query], normalize_embeddings=True)[0]
                sims = self.vecs @ q
                for r, i in enumerate(sorted(range(len(sims)), key=lambda i: -sims[i])[:20]):
                    ranks[i] = ranks.get(i, 0) + 1 / (60 + r)
            for i in sorted(ranks, key=lambda i: -ranks[i]):
                p = self.paragraphs[i]
                if i in picked or (caption and p.strip().startswith(caption.strip()[:40])):
                    continue
                picked.append(i)
                if len(picked) >= k:
                    break
        return [self.paragraphs[i] for i in picked[:k]]
