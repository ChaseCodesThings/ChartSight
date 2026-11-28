"""Find numbers in generated text and decide whether the chart's data supports them."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .schema import ChartData

_SCALE_WORDS = {
    "thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "mn": 1e6, "mil": 1e6,
    "billion": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12, "tn": 1e12,
}
_WORD_NUMS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twice": 2, "double": 2,
              "triple": 3, "half": 0.5, "quarter": 0.25}

_NUM_RE = re.compile(
    r"(?<![\w.])"
    r"(?P<sign>[-−–])?"
    r"(?P<cur>[$€£¥])?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?P<pct>\s?%|\s?percent\b)?"
    r"(?:\s?(?P<scale>thousand|million|billion|trillion|bn|mn|mil|tn|[kKmMbB])\b)?"
)


@dataclass
class Mention:
    raw: str
    value: float
    is_percent: bool
    start: int
    end: int
    kind: str = "value"  # value | year | ordinal | count_word


def find_numbers(text: str) -> list[Mention]:
    out: list[Mention] = []
    for m in _NUM_RE.finditer(text):
        raw = m.group(0).strip()
        num = float(m.group("num").replace(",", ""))
        if m.group("sign") and m.start("sign") == m.start():
            # treat "-" as a minus sign only when it is not a range dash like "2019-2021"
            prev = text[m.start() - 1] if m.start() > 0 else " "
            if not prev.isdigit():
                num = -num
        scale = m.group("scale")
        if scale:
            num *= _SCALE_WORDS.get(scale.lower(), 1)
        after = text[m.end():m.end() + 3].lower()
        kind = "value"
        if re.match(r"(st|nd|rd|th)\b", after):
            kind = "ordinal"
        elif not m.group("pct") and not scale and not m.group("cur") and "." not in m.group("num") \
                and "," not in m.group("num") and 1800 <= num <= 2100:
            kind = "year"
        out.append(Mention(raw=raw, value=num, is_percent=bool(m.group("pct")),
                           start=m.start(), end=m.end(), kind=kind))
    return out


def _close(a: float, b: float, rel: float, abs_tol: float) -> bool:
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), abs_tol)


class Support:
    """All the numbers a description may legitimately state about one chart."""

    def __init__(self, data: ChartData, rel_tol: float = 0.05, axis_tol: float = 0.02):
        self.data = data
        self.rel = rel_tol
        ys = [p.y for _, p in data.all_points()]
        self.ys = ys
        lo = data.y_axis.min if data.y_axis.min is not None else (min(ys) if ys else 0.0)
        hi = data.y_axis.max if data.y_axis.max is not None else (max(ys) if ys else 0.0)
        self.abs_tol = axis_tol * abs(hi - lo) if hi != lo else 1e-9

        self.values: list[float] = list(ys)
        for ax in (data.x_axis, data.y_axis):
            self.values += [v for v in (ax.min, ax.max) if v is not None]
        self.x_numbers: list[float] = []
        for _, p in data.all_points():
            if isinstance(p.x, (int, float)):
                self.x_numbers.append(float(p.x))
            else:
                self.x_numbers += [m.value for m in find_numbers(str(p.x))]
        for t in (data.title, data.x_axis.title, data.y_axis.title, data.notes):
            if t:
                self.x_numbers += [m.value for m in find_numbers(t)]

        # derived values a description may state when it makes a comparison, by kind
        self.diffs: list[float] = []
        self.pct_changes: list[float] = []
        self.ratios: list[float] = []
        self.aggregates: list[float] = []
        by_series = {}
        for name, p in data.all_points():
            by_series.setdefault(name, []).append(p.y)
        uniq = sorted(set(ys))
        if len(uniq) <= 80:
            for i, a in enumerate(uniq):
                for b in uniq[i + 1:]:
                    self.diffs.append(b - a)
                    if a != 0:
                        self.pct_changes.append((b - a) / abs(a) * 100)
                        self.ratios.append(b / a)
                    if b != 0:
                        self.pct_changes.append((a - b) / abs(b) * 100)
                        self.ratios.append(a / b)
        for vals in by_series.values():
            if vals:
                self.aggregates += [sum(vals), sum(vals) / len(vals)]
        self.counts = {len(data.series), len(data.all_points())} | {len(v) for v in by_series.values()}

    def check(self, m: Mention, sentence: str = "") -> Optional[str]:
        """Return how a number is supported, or None if it is not.

        Derived values (differences, percent changes, ratios, totals, means) count only when the
        sentence signals that comparison, and only within a tight tolerance; otherwise almost any
        invented number would match some difference between data values.
        """
        if m.kind in ("year", "ordinal"):
            # a year must appear among the x labels; ordinals ("2nd highest") are always allowed
            if m.kind == "ordinal" or any(_close(m.value, x, 0, 0.5) for x in self.x_numbers):
                return m.kind
            return None if self.x_numbers else "year"
        v = m.value
        if any(_close(v, x, self.rel, self.abs_tol) for x in self.values):
            return "data"
        if m.is_percent and any(_close(v, x * 100, self.rel, self.abs_tol * 100) for x in self.ys if abs(x) <= 1):
            return "data"
        if any(_close(v, x, 0, 1e-9) for x in self.x_numbers):
            return "label"
        s = sentence.lower()
        if v.is_integer() and int(v) in self.counts and _COUNT_CUE.search(s):
            return "count"
        tight = 0.02
        rounding = 0.5 if float(v).is_integer() else 0.05
        near = lambda xs: any(_close(abs(v), abs(d), tight, rounding) for d in xs)
        if _DIFF_CUE.search(s) and near(self.diffs):
            return "difference"
        if m.is_percent and _CHANGE_CUE.search(s) and near(self.pct_changes):
            return "percent_change"
        if _RATIO_CUE.search(s) and near(self.ratios):
            return "ratio"
        if _AGG_CUE.search(s) and near(self.aggregates):
            return "aggregate"
        return None


_DIFF_CUE = re.compile(r"\b(by|difference|differ|gap|more than|less than|higher than|lower than|greater than|"
                       r"fewer than|exceeds?|ahead|behind|above|below|apart|margin|points?)\b")
_CHANGE_CUE = re.compile(r"\b(increas\w*|decreas\w*|ris\w*|rose|fell|fall\w*|drop\w*|grow\w*|grew|growth|"
                         r"declin\w*|chang\w*|up|down|gain\w*|loss|lost|jump\w*|climb\w*|shr[iu]nk\w*)\b")
_RATIO_CUE = re.compile(r"\b(times|fold|twice|double|triple|half|ratio|x)\b")
_AGG_CUE = re.compile(r"\b(total|combined|sum|altogether|overall|average|mean|in all)\b")
_COUNT_CUE = re.compile(r"\b(bars?|points?|lines?|series|categories|groups?|years?|months?|items?|countries|"
                        r"venues?|models?|methods?|segments?|slices?|entries|values?|[a-z]+s)\b")
