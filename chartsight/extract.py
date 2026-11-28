"""Stage 3: chart image -> structured data (ChartData), using a vision-language model."""
from __future__ import annotations

import json
from typing import Optional

from pydantic import ValidationError

from .llm import LLMClient
from .schema import ChartData, Usage

EXTRACT_SYSTEM = """You extract the underlying data from a chart image so it can be described to a blind reader.
Read the chart carefully: title, axis titles, units, axis ranges, every series, and every data point.
For bar charts use each bar's category as x. For line and scatter charts read each marked point,
or for unmarked lines read the value at every labelled x tick. Estimate values from gridlines when they are not printed.
Use null for anything you cannot see. Do not invent series or points that are not in the image.
Set confidence to "low" if the chart is dense, blurry, or values had to be estimated roughly.

Reply with ONLY a JSON object of this shape:
{"chart_type": "bar|line|scatter|area|pie|histogram|box|other",
 "title": str|null,
 "x_axis": {"title": str|null, "unit": str|null, "scale": "linear|log|categorical|time|unknown", "min": num|null, "max": num|null},
 "y_axis": {"title": str|null, "unit": str|null, "scale": "linear|log|categorical|time|unknown", "min": num|null, "max": num|null},
 "series": [{"name": str|null, "points": [{"x": str|num, "y": num}]}],
 "confidence": "high|medium|low",
 "notes": str|null}"""


def _clean(obj: dict) -> dict:
    """Drop points whose y is missing or not numeric instead of failing the whole chart."""
    for s in obj.get("series") or []:
        pts = []
        for p in s.get("points") or []:
            y = p.get("y")
            if isinstance(y, str):
                try:
                    y = float(y.replace(",", "").replace("%", ""))
                except ValueError:
                    y = None
            if isinstance(y, (int, float)) and p.get("x") is not None:
                pts.append({"x": p["x"], "y": float(y)})
        s["points"] = pts
    for ax in ("x_axis", "y_axis"):
        a = obj.get(ax) or {}
        for k in ("min", "max"):
            if not isinstance(a.get(k), (int, float)):
                a[k] = None
        if a.get("scale") not in ("linear", "log", "categorical", "time", "unknown"):
            a["scale"] = "unknown"
        obj[ax] = a
    if obj.get("chart_type") not in ("bar", "line", "scatter", "area", "pie", "histogram", "box", "other"):
        obj["chart_type"] = "other"
    if obj.get("confidence") not in ("high", "medium", "low"):
        obj["confidence"] = "medium"
    return obj


def extract_chart(client: LLMClient, image: bytes, caption: Optional[str] = None) -> tuple[ChartData, list[Usage]]:
    prompt = "Extract the data from this chart."
    if caption:
        prompt += f"\nThe figure's caption (for context only; read values from the image): {caption}"
    obj, usages = client.complete_json("extract", EXTRACT_SYSTEM, prompt, [image])
    try:
        return ChartData.model_validate(_clean(obj)), usages
    except ValidationError as e:
        raise ValueError(f"extraction did not match the schema: {e}") from e


def data_for_prompt(data: ChartData) -> str:
    return json.dumps(data.model_dump(exclude_none=True), ensure_ascii=False)
