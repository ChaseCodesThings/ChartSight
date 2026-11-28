"""Stage 6: check every sentence of the alt text against the extracted chart data.

Two kinds of checks run without any model call:
  * numbers: every number must match a data value, an axis bound, a label, a count,
    or a simple derived value (difference, ratio, percent change, total, mean)
  * extremes: a sentence naming the highest/lowest point must name the right one

Sentences that fail are sent back to the model once to be rewritten from the data.
If the rewrite still fails, the sentence is removed (or kept and flagged, if `drop=False`).
"""
from __future__ import annotations

import json
import re
from typing import Optional

from .llm import LLMClient
from .numbers import Support, find_numbers
from .schema import AltText, ChartData, SentenceCheck, Usage

_MAX_WORDS = r"\b(highest|largest|greatest|biggest|maximum|max|peak|peaks|peaked|most|top)\b"
_MIN_WORDS = r"\b(lowest|smallest|least|minimum|min|fewest|bottom)\b"


def _label(x) -> str:
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def check_extremes(sentence: str, data: ChartData) -> list[str]:
    """If a sentence says X is the highest (or lowest), X should be the argmax (or argmin) of its series."""
    issues = []
    s = sentence.lower()
    for words, pick, name in ((_MAX_WORDS, max, "highest"), (_MIN_WORDS, min, "lowest")):
        if not re.search(words, s):
            continue
        for series in data.series:
            if len(series.points) < 2:
                continue
            labels = {_label(p.x).lower(): p.y for p in series.points}
            mentioned = [lab for lab in labels if lab and len(lab) > 1 and re.search(rf"\b{re.escape(lab)}\b", s)]
            if len(mentioned) != 1:
                continue  # ambiguous or no label named: nothing to check
            best = pick(labels.values())
            if labels[mentioned[0]] != best:
                true_label = next(k for k, v in labels.items() if v == best)
                issues.append(f"says '{mentioned[0]}' is the {name}, but the data's {name} is '{true_label}'")
    return issues


def check_attribution(sentence: str, support: Support) -> list[str]:
    """If a sentence names exactly one label and states plain data values, one of them should be that label's value.
    Catches swapped values such as "UIST reaches 9%" when 9% belongs to CSCW."""
    s = sentence.lower()
    mentions = [m for m in find_numbers(sentence) if m.kind == "value" and support.check(m, sentence) == "data"]
    if not mentions:
        return []
    issues = []
    for series in support.data.series:
        labels = {_label(p.x).lower(): p.y for p in series.points if _label(p.x)}
        named = [lab for lab in labels if len(lab) > 1 and re.search(rf"\b{re.escape(lab)}\b", s)]
        if len(named) != 1 or len(labels) < 2:
            continue
        y = labels[named[0]]
        tol = max(support.rel * abs(y), support.abs_tol)
        def ok(m):
            v = m.value
            return abs(v - y) <= tol or (m.is_percent and abs(y) <= 1 and abs(v - y * 100) <= tol * 100)
        if not any(ok(m) for m in mentions) and not any(_close_axis(m.value, support) for m in mentions):
            issues.append(f"gives {', '.join(m.raw for m in mentions)} for '{named[0]}', whose value is {_label(y)}")
    return issues


def _close_axis(v: float, support: Support) -> bool:
    d = support.data
    return any(b is not None and abs(v - b) <= support.abs_tol for b in (d.y_axis.min, d.y_axis.max))


def check_sentence(level: str, text: str, support: Support) -> SentenceCheck:
    bad = [m.raw for m in find_numbers(text) if support.check(m, text) is None]
    issues = check_extremes(text, support.data) + check_attribution(text, support)
    if bad:
        issues.insert(0, "numbers not supported by the data: " + ", ".join(bad))
    has_numbers = bool(find_numbers(text))
    status = "unsupported" if (bad or issues) else ("verified" if has_numbers else "no_numbers")
    return SentenceCheck(level=level, text=text, status=status, unsupported_numbers=bad, issues=issues)


REPAIR_SYSTEM = (
    "You fix sentences in chart alt text so they are fully supported by the chart's data table. "
    "Rewrite each sentence to keep its purpose but use only values, labels and comparisons that the data supports. "
    "Round only when the original rounds. If a sentence cannot be fixed, return an empty string for it. "
    'Reply with JSON only: {"sentences": ["...", "..."]} in the same order.'
)


def verify(alt: AltText, data: ChartData, client: Optional[LLMClient] = None, rounds: int = 1,
           rel_tol: float = 0.05, axis_tol: float = 0.02, drop: bool = True) -> tuple[AltText, list[SentenceCheck], list[Usage]]:
    support = Support(data, rel_tol, axis_tol)
    checks = [check_sentence(level, text, support) for level, text in alt.sentences()]
    usages: list[Usage] = []

    for _ in range(rounds if client else 0):
        failing = [i for i, c in enumerate(checks) if c.status == "unsupported"]
        if not failing:
            break
        payload = {
            "data": data.model_dump(exclude_none=True),
            "sentences": [{"text": checks[i].text, "problems": checks[i].issues} for i in failing],
        }
        try:
            out, u = client.complete_json("repair", REPAIR_SYSTEM, json.dumps(payload, ensure_ascii=False))
            usages += u
            fixed = out.get("sentences", [])
        except Exception:
            break
        for i, new in zip(failing, fixed):
            new = (new or "").strip()
            if not new:
                continue
            c = check_sentence(checks[i].level, new, support)
            if c.status != "unsupported":
                c.status = "rewritten"
            c.original = checks[i].original or checks[i].text
            checks[i] = c

    for c in checks:
        if c.status == "unsupported" and drop:
            c.status = "removed"

    kept = [c for c in checks if c.status != "removed"]
    new_alt = AltText(
        summary=next((c.text for c in kept if c.level == "summary"), ""),
        construction=[c.text for c in kept if c.level == "construction"],
        statistics=[c.text for c in kept if c.level == "statistics"],
        trends=[c.text for c in kept if c.level == "trends"],
    )
    return new_alt, checks, usages
