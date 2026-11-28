from pathlib import Path

from chartsight.context import Retriever, figure_number, split_paragraphs
from chartsight.figures import extract_figures
from scripts.make_sample_paper import build


def test_figures_and_context(tmp_path: Path):
    pdf = build(tmp_path / "paper.pdf")
    paper = extract_figures(pdf.read_bytes())
    assert [f.figure_id for f in paper.figures] == ["fig1", "fig2", "fig3"]
    for f in paper.figures:
        assert f.image[:4] == b"\x89PNG" and f.caption.startswith(f"Figure {f.figure_id[3:]}:")
        x0, y0, x1, y1 = f.bbox
        assert x1 - x0 > 300 and y1 - y0 > 180  # the whole chart, not a fragment

    r = Retriever(split_paragraphs(paper.text), use_embeddings=False)
    ctx = r.for_figure(paper.figures[2].caption, k=2)
    assert "extraction stage (Figure 3)" in ctx[0]
    assert figure_number("Fig. 12b: something") == "12b"
