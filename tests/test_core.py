import json

import pytest

from chartsight import llm
from chartsight.config import ModelConfig, PipelineConfig
from chartsight.numbers import Support, find_numbers
from chartsight.pipeline import describe_chart
from chartsight.schema import AltText, Axis, ChartData, Point, Series
from chartsight.verify import check_extremes, verify
from evaluation.datasets import parse_datatable
from evaluation.metrics import extraction_scores, hallucination_scores


def chart():
    return ChartData(chart_type="bar", title="Alt text coverage by venue",
                     y_axis=Axis(title="Papers with alt text", unit="%", min=0, max=70),
                     series=[Series(points=[Point(x="CHI", y=38), Point(x="ASSETS", y=61), Point(x="UIST", y=12),
                                            Point(x="CSCW", y=9), Point(x="arXiv", y=3)])])


# ---- number finding ---------------------------------------------------------
@pytest.mark.parametrize("text,values", [
    ("rose to 1,470 million units", [1.47e9]),
    ("about 45% in 2021", [45, 2021]),
    ("from $3.5 billion", [3.5e9]),
    ("between 2019-2021", [2019, 2021]),
    ("a drop of -12.5 points", [-12.5]),
    ("the 2nd highest", [2]),
    ("12k users", [12000]),
])
def test_find_numbers(text, values):
    assert [m.value for m in find_numbers(text)] == pytest.approx(values)


def test_year_and_ordinal_kinds():
    kinds = {m.raw: m.kind for m in find_numbers("In 2020, the 3rd bar is 2,020 units")}
    assert kinds["2020"] == "year" and kinds["3"] == "ordinal" and kinds["2,020"] == "value"


# ---- support ------------------------------------------------------------------
def test_support_accepts_data_derived_and_counts_rejects_inventions():
    sup = Support(chart())
    ok = lambda s: all(sup.check(m, s) for m in find_numbers(s))
    assert ok("ASSETS is highest at 61%")
    assert ok("ASSETS is about 60%")  # within 5%
    assert ok("ASSETS exceeds CHI by 23 points")  # difference
    assert ok("The chart compares 5 venues")  # count
    assert ok("The y-axis runs from 0 to 70")  # axis bounds
    assert not ok("UIST reaches 27%")  # invented
    assert not ok("CHI has 50% coverage")
    assert not ok("UIST reaches 26%")  # equals a difference (38-12), but no comparison is stated
    assert ok("CHI is about 3 times UIST")  # ratio 38/12 = 3.2, with a ratio cue


def test_attribution():
    from chartsight.verify import check_attribution
    sup = Support(chart())
    assert check_attribution("UIST has 12% coverage.", sup) == []
    assert check_attribution("UIST has 9% coverage.", sup)  # 9% is CSCW's value
    assert check_attribution("CHI (38%) is well ahead of UIST (12%).", sup) == []  # two labels: not checked


def test_extremes():
    assert check_extremes("ASSETS has the highest coverage", chart()) == []
    assert check_extremes("CHI has the highest coverage", chart())
    assert check_extremes("arXiv has the lowest coverage", chart()) == []
    assert check_extremes("UIST has the lowest coverage", chart())


# ---- verifier -----------------------------------------------------------------
def test_verify_without_model_removes_unsupported():
    alt = AltText(summary="Bar chart of alt text coverage by venue.",
                  statistics=["ASSETS is highest at 61%.", "UIST reaches 27%."])
    new, checks, _ = verify(alt, chart(), client=None)
    assert new.statistics == ["ASSETS is highest at 61%."]
    assert [c.status for c in checks] == ["no_numbers", "verified", "removed"]


def test_verify_repair_round(monkeypatch):
    def fake(stage, system, prompt, images):
        assert stage == "repair"
        assert "27%" in json.loads(prompt)["sentences"][0]["text"]
        return json.dumps({"sentences": ["UIST reaches 12%."]}), 100, 10
    monkeypatch.setattr(llm, "MOCK_HANDLER", fake)
    alt = AltText(statistics=["UIST reaches 27%."])
    new, checks, usage = verify(alt, chart(), client=llm.LLMClient(ModelConfig()))
    assert new.statistics == ["UIST reaches 12%."]
    assert checks[0].status == "rewritten" and checks[0].original == "UIST reaches 27%."
    assert usage and usage[0].stage == "repair"


# ---- full pipeline with a scripted model -----------------------------------------
def scripted(stage, system, prompt, images):
    if stage == "extract":
        return json.dumps({"chart_type": "bar", "title": "Coverage", "x_axis": {"title": "Venue"},
                           "y_axis": {"title": "%", "min": 0, "max": 70},
                           "series": [{"points": [{"x": "CHI", "y": 38}, {"x": "ASSETS", "y": "61"}, {"x": "UIST", "y": 12}]}]}), 50, 50
    if stage in ("generate", "direct"):
        return "```json\n" + json.dumps({"summary": "Bar chart of coverage.", "statistics": ["ASSETS leads with 61%.", "UIST has 44%."]}) + "\n```", 50, 50
    if stage == "repair":
        return json.dumps({"sentences": [""]}), 10, 5
    return "{}", 1, 1


@pytest.mark.parametrize("system,expect_44", [("full", False), ("no_verify", True), ("direct", True)])
def test_pipeline_ablations(monkeypatch, system, expect_44):
    monkeypatch.setattr(llm, "MOCK_HANDLER", scripted)
    res = describe_chart("t", b"\x89PNG\r\n\x1a\nfake", PipelineConfig.ablation(system, ModelConfig()))
    assert res.error is None
    assert ("44%" in res.alt_text.full_text()) == expect_44
    if system != "direct":
        assert res.data.series[0].points[1].y == 61.0  # string "61" coerced


def test_pipeline_reports_errors_instead_of_crashing(monkeypatch):
    monkeypatch.setattr(llm, "MOCK_HANDLER", lambda *a: ("not json at all", 1, 1))
    res = describe_chart("t", b"x", PipelineConfig.ablation("full", ModelConfig()))
    assert res.error and "JSON" in res.error


# ---- evaluation ------------------------------------------------------------------
def test_parse_datatable():
    title, pairs, nums = parse_datatable("Coverage by venue <s> Venue Share CHI 38 ASSETS 61 New Zealand 12.5")
    assert title == "Coverage by venue"
    assert pairs[-2:] == [("ASSETS", 61.0), ("New Zealand", 12.5)]
    assert pairs[0][1] == 38.0
    _, pairs2, _ = parse_datatable("Sales <s> Year Sales 2019 1,200 2020 950")
    assert pairs2 == [("2019", 1200.0), ("2020", 950.0)]


def test_metrics():
    truth = chart()
    pred = ChartData(series=[Series(points=[Point(x="CHI", y=37.5), Point(x="ASSETS", y=61), Point(x="X", y=50)])])
    s = extraction_scores(pred, truth)
    assert s["value_recall"] == pytest.approx(0.4) and s["value_precision"] == pytest.approx(0.6667)
    h = hallucination_scores(AltText(statistics=["ASSETS is 61%.", "UIST is 27%."]), truth)
    assert h["n_unsupported_numbers"] == 1 and h["unsupported_sentence_rate"] == 0.5
