"""Typed data passed between pipeline stages.

Every stage reads and writes these models, so each stage can be tested,
evaluated and swapped on its own.
"""
from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

ChartType = Literal["bar", "line", "scatter", "area", "pie", "histogram", "box", "other"]


class Axis(BaseModel):
    title: Optional[str] = None
    unit: Optional[str] = None
    scale: Literal["linear", "log", "categorical", "time", "unknown"] = "unknown"
    min: Optional[float] = None
    max: Optional[float] = None


class Point(BaseModel):
    x: Union[float, str]
    y: float


class Series(BaseModel):
    name: Optional[str] = None
    points: list[Point] = Field(default_factory=list)


class ChartData(BaseModel):
    """Structured reading of one chart. This is the ground truth later stages are checked against."""

    chart_type: ChartType = "other"
    title: Optional[str] = None
    x_axis: Axis = Field(default_factory=Axis)
    y_axis: Axis = Field(default_factory=Axis)
    series: list[Series] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    notes: Optional[str] = None

    def all_points(self) -> list[tuple[Optional[str], Point]]:
        return [(s.name, p) for s in self.series for p in s.points]

    def as_table(self) -> list[dict]:
        rows = []
        for s in self.series:
            for p in s.points:
                rows.append({"series": s.name or "", "x": p.x, "y": p.y})
        return rows


class AltText(BaseModel):
    """Layered description. Each level is a list of sentences so each can be verified on its own."""

    summary: str = ""
    construction: list[str] = Field(default_factory=list)  # Level 1
    statistics: list[str] = Field(default_factory=list)  # Level 2
    trends: list[str] = Field(default_factory=list)  # Level 3

    def sentences(self) -> list[tuple[str, str]]:
        out = [("summary", self.summary)] if self.summary else []
        for level in ("construction", "statistics", "trends"):
            out += [(level, s) for s in getattr(self, level)]
        return out

    def full_text(self) -> str:
        return " ".join(s for _, s in self.sentences())


class SentenceCheck(BaseModel):
    level: str
    text: str
    status: Literal["verified", "no_numbers", "rewritten", "unsupported", "removed"]
    unsupported_numbers: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    original: Optional[str] = None


class Usage(BaseModel):
    stage: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    cost_usd: Optional[float] = None


class ChartResult(BaseModel):
    figure_id: str
    caption: Optional[str] = None
    data: Optional[ChartData] = None
    context: list[str] = Field(default_factory=list)
    alt_text: AltText = Field(default_factory=AltText)
    checks: list[SentenceCheck] = Field(default_factory=list)
    usage: list[Usage] = Field(default_factory=list)
    error: Optional[str] = None

    @property
    def total_seconds(self) -> float:
        return sum(u.seconds for u in self.usage)

    @property
    def total_cost(self) -> Optional[float]:
        costs = [u.cost_usd for u in self.usage]
        return None if any(c is None for c in costs) else sum(costs)  # type: ignore[arg-type]
