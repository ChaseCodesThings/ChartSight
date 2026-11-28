"""Stage 5: write layered alt text.

Levels follow Lundgard and Satyanarayan's model of chart descriptions, as used by VisText:
  L1 construction (chart type, title, axes, ranges), L2 statistics (extremes, notable values,
  comparisons), L3 trends (patterns and the takeaway).
"""
from __future__ import annotations

import json
from typing import Optional

from .extract import data_for_prompt
from .llm import LLMClient
from .schema import AltText, ChartData, Usage

FEW_SHOT = """Example (from a data table, for style only):
{"summary": "Line chart showing global smartphone shipments peaking in 2016 and declining afterward.",
 "construction": ["This is a line chart titled Global smartphone shipments, 2012 to 2020.",
                  "The x-axis shows the year from 2012 to 2020; the y-axis shows shipments in millions of units, from 0 to 1,600."],
 "statistics": ["Shipments were lowest in 2012 at about 725 million units.",
                "They peaked in 2016 at about 1,470 million units."],
 "trends": ["Shipments roughly doubled from 2012 to 2016, then fell each year to about 1,290 million in 2020."]}"""

GENERATE_SYSTEM = f"""You write alt text that lets a blind or low-vision reader understand a chart in a research paper.
Write plain, concrete sentences that a screen reader can read aloud. Do not say "the image shows".
Use ONLY values, labels and comparisons supported by the chart data you are given. Round sensibly and say "about" when values are estimated.
Use the paper context only to explain what the chart means, never as a source of numbers.

Return JSON only, with these fields:
  "summary": one sentence with the chart type and its main takeaway.
  "construction": 1-2 sentences: chart type, title, what each axis shows, units and ranges.
  "statistics": 1-3 sentences: extremes, notable values, key comparisons.
  "trends": 1-2 sentences: the overall pattern and, if the context says so, why it matters.

{FEW_SHOT}"""

DIRECT_SYSTEM = """You write alt text that lets a blind or low-vision reader understand a chart image.
Return JSON only: {"summary": str, "construction": [str], "statistics": [str], "trends": [str]}
summary = one sentence with chart type and main takeaway; construction = chart type, title, axes, ranges;
statistics = extremes and notable values; trends = overall pattern."""


def _to_alt(obj: dict) -> AltText:
    def lst(v):
        if isinstance(v, str):
            return [v] if v.strip() else []
        return [str(x).strip() for x in (v or []) if str(x).strip()]

    return AltText(summary=str(obj.get("summary") or "").strip(), construction=lst(obj.get("construction")),
                   statistics=lst(obj.get("statistics")), trends=lst(obj.get("trends")))


def generate_alt_text(client: LLMClient, data: Optional[ChartData], image: Optional[bytes],
                      caption: Optional[str], context: list[str]) -> tuple[AltText, list[Usage]]:
    """With `data`, the model writes from the extracted table (and sees the image for layout).
    Without it (ablation), the model writes from the image alone."""
    parts = []
    if data is not None:
        parts.append("Chart data (extracted from the image):\n" + data_for_prompt(data))
    if caption:
        parts.append(f"Figure caption: {caption}")
    if context:
        parts.append("Paper paragraphs that discuss this figure:\n" + "\n---\n".join(context))
    parts.append("Write the layered alt text now.")
    obj, usages = client.complete_json("generate", GENERATE_SYSTEM, "\n\n".join(parts), [image] if image else None)
    return _to_alt(obj), usages


def direct_alt_text(client: LLMClient, image: bytes) -> tuple[AltText, list[Usage]]:
    """Naive baseline: one prompt, image in, alt text out."""
    obj, usages = client.complete_json("direct", DIRECT_SYSTEM, "Write alt text for this chart.", [image])
    return _to_alt(obj), usages
