"""LLM-as-judge rubric scores. Validate it: score ~50 charts yourselves with the same rubric
(see evaluation/human_rating_sheet.csv from `run_eval.py --rating-sheet`) and report agreement
before trusting the judge on the rest."""
from __future__ import annotations

import json

from chartsight.llm import LLMClient
from chartsight.schema import AltText, ChartData

JUDGE_SYSTEM = """You grade alt text for a chart, from the point of view of a blind reader who cannot see it.
You are given the chart's TRUE data table and a candidate description. Score each criterion 1-5:
  accuracy: every stated value, label and comparison agrees with the data (5 = no errors)
  completeness: covers chart type, axes, extremes and the main trend (5 = nothing important missing)
  clarity: easy to follow when read aloud; no jargon or visual-only references (5 = very clear)
  trend: correctly conveys the overall pattern or takeaway (5 = precise and correct)
Reply with JSON only: {"accuracy": int, "completeness": int, "clarity": int, "trend": int, "reason": str}"""

CRITERIA = ("accuracy", "completeness", "clarity", "trend")


def judge(client: LLMClient, alt: AltText, truth: ChartData) -> tuple[dict, list]:
    prompt = json.dumps({"true_data": truth.model_dump(exclude_none=True), "description": alt.full_text()}, ensure_ascii=False)
    out, usage = client.complete_json("judge", JUDGE_SYSTEM, prompt)
    scores = {k: int(out.get(k, 0) or 0) for k in CRITERIA}
    scores["reason"] = str(out.get("reason", ""))[:300]
    return scores, usage
