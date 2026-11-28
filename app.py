"""ChartSight web app.

    streamlit run app.py

Workflow: Upload a paper PDF or chart image -> Review detected figures -> Generate -> Verify -> Export.
Model settings come from .env (see .env.example) and can be changed in the sidebar.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from chartsight.config import ModelConfig, PipelineConfig, model_from_env
from chartsight.context import Retriever, split_paragraphs
from chartsight.figures import Figure, extract_figures
from chartsight.llm import LLMClient
from chartsight.numbers import Support
from chartsight.pipeline import describe_chart, flag_summary
from chartsight.schema import ChartData, ChartResult
from chartsight.verify import verify

st.set_page_config(page_title="ChartSight", page_icon="📊", layout="wide")

STATUS_BADGE = {
    "verified": ("Verified", "green"),
    "no_numbers": ("No numbers to check", "gray"),
    "rewritten": ("Rewritten to match data", "orange"),
    "unsupported": ("Unsupported (kept)", "red"),
    "removed": ("Removed: unsupported", "red"),
}

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    env = model_from_env()
    providers = ["anthropic", "openai", "mock"]
    provider = st.selectbox("Provider", providers, index=providers.index(env.provider) if env.provider in providers else 2,
                            help="'openai' also works with any OpenAI-compatible server (Ollama, vLLM) via the base URL.")
    model = st.text_input("Model", value=env.model or "")
    base_url = st.text_input("Base URL (optional)", value=env.base_url or "", help="e.g. http://localhost:11434/v1 for Ollama")
    st.subheader("Pipeline")
    use_context = st.checkbox("Use paper context (RAG)", value=True)
    use_verification = st.checkbox("Verify against extracted data", value=True)
    remove_unsupported = st.checkbox("Remove sentences that still fail", value=True,
                                     help="Off: keep them, marked as unsupported.")
    show_debug = st.checkbox("Show model usage and timing", value=True)

cfg = PipelineConfig(use_context=use_context, use_verification=use_verification,
                     model=ModelConfig(provider=provider, model=model or "mock-model", base_url=base_url or None,
                                       api_key=env.api_key, price_in_per_mtok=env.price_in_per_mtok,
                                       price_out_per_mtok=env.price_out_per_mtok))

# ---------------------------------------------------------------- 1. upload
st.title("ChartSight")
st.caption("Verified, layered alt text for charts in research papers. Generated descriptions are drafts: review before publishing.")
if provider == "mock":
    st.info("Running with the offline mock model: every chart gets the same canned output. Pick a real provider in the sidebar.")

up = st.file_uploader("Upload a paper (PDF) or a single chart image", type=["pdf", "png", "jpg", "jpeg", "webp"])
sample = Path(__file__).parent / "data" / "sample_paper.pdf"
if up is None and sample.exists() and st.button("Use the sample paper"):
    st.session_state["upload"] = ("sample_paper.pdf", sample.read_bytes())
if up is not None:
    st.session_state["upload"] = (up.name, up.getvalue())
if "upload" not in st.session_state:
    st.stop()

name, data = st.session_state["upload"]
key = f"{name}-{len(data)}"
if st.session_state.get("doc_key") != key:
    st.session_state.update(doc_key=key, results={})
    if name.lower().endswith(".pdf"):
        with st.spinner("Finding figures..."):
            paper = extract_figures(data)
        st.session_state["figures"] = paper.figures
        st.session_state["paragraphs"] = split_paragraphs(paper.text)
    else:
        st.session_state["figures"] = [Figure(figure_id="chart", page=1, bbox=(0, 0, 0, 0), image=data, caption=None, source="upload")]
        st.session_state["paragraphs"] = []

figures: list[Figure] = st.session_state["figures"]
results: dict[str, ChartResult] = st.session_state["results"]
if not figures:
    st.warning("No figures found. Try uploading the chart as an image instead.")
    st.stop()

# ---------------------------------------------------------------- 2. review figures
st.subheader(f"Figures found: {len(figures)}")
labels = [f"{f.figure_id} (p.{f.page})" + (f": {f.caption[:60]}..." if f.caption else "") for f in figures]
chosen = st.multiselect("Figures to describe", labels, default=labels)
selected = [f for f, l in zip(figures, labels) if l in chosen]

c1, c2 = st.columns([1, 3])
go = c1.button("Generate alt text", type="primary", disabled=not selected)
if go:
    retriever = Retriever(st.session_state["paragraphs"]) if (use_context and st.session_state["paragraphs"]) else None
    client = LLMClient(cfg.model)
    bar = st.progress(0.0, text="Starting...")
    for i, f in enumerate(selected):
        bar.progress(i / len(selected), text=f"Describing {f.figure_id}...")
        # verification here keeps failing sentences when the user asked to keep them
        run_cfg = PipelineConfig(**{**cfg.__dict__, "use_verification": False}) if use_verification and not remove_unsupported else cfg
        r = describe_chart(f.figure_id, f.image, run_cfg, caption=f.caption, retriever=retriever, client=client)
        if use_verification and not remove_unsupported and r.data is not None and not r.error:
            r.alt_text, r.checks, u = verify(r.alt_text, r.data, client, rounds=cfg.repair_rounds, drop=False)
            r.usage += u
        results[f.figure_id] = r
    bar.progress(1.0, text="Done")

# ---------------------------------------------------------------- 3-4. results
for f in figures:
    r = results.get(f.figure_id)
    if r is None:
        continue
    st.divider()
    left, right = st.columns([2, 3])
    with left:
        st.image(f.image, caption=f.caption or f.figure_id, use_container_width=True)
        if r.data is not None:
            with st.expander(f"Extracted data ({r.data.chart_type}, confidence: {r.data.confidence}) - edit to correct", expanded=False):
                edited = st.data_editor(r.data.as_table(), key=f"edit-{f.figure_id}", num_rows="dynamic", use_container_width=True)
                if st.button("Re-check with corrected data", key=f"recheck-{f.figure_id}"):
                    fixed = r.data.model_copy(deep=True)
                    series = {}
                    for row in edited:
                        try:
                            y = float(row["y"])
                        except (TypeError, ValueError):
                            continue
                        series.setdefault(row.get("series") or None, []).append({"x": row["x"], "y": y})
                    fixed.series = [{"name": k, "points": v} for k, v in series.items()]
                    r.data = ChartData.model_validate(fixed.model_dump())
                    r.alt_text, r.checks, u = verify(r.alt_text, r.data, LLMClient(cfg.model), rounds=1, drop=remove_unsupported)
                    r.usage += u
                    st.rerun()
    with right:
        st.markdown(f"### {f.figure_id}")
        if r.error:
            st.error(r.error)
            continue
        a = r.alt_text
        st.markdown(f"**Summary.** {a.summary}")
        tabs = st.tabs(["Construction (L1)", "Statistics (L2)", "Trends (L3)", "Data table"])
        for tab, items in zip(tabs[:3], (a.construction, a.statistics, a.trends)):
            with tab:
                for s in items:
                    st.markdown(f"- {s}")
        with tabs[3]:
            if r.data:
                st.dataframe(r.data.as_table(), use_container_width=True)
        if r.checks:
            counts = flag_summary(r.checks)
            st.markdown("**Verification:** " + " · ".join(f":{STATUS_BADGE[k][1]}[{STATUS_BADGE[k][0]}: {v}]" for k, v in counts.items()))
            with st.expander("Sentence-by-sentence checks"):
                for c in r.checks:
                    label, color = STATUS_BADGE[c.status]
                    st.markdown(f":{color}[**{label}**] {c.text}")
                    if c.original and c.original != c.text:
                        st.caption(f"Original: {c.original}")
                    for issue in c.issues:
                        st.caption(f"Issue: {issue}")
        if r.context:
            with st.expander(f"Paper context used ({len(r.context)} passages)"):
                for p in r.context:
                    st.caption(p[:500] + ("..." if len(p) > 500 else ""))
        text = a.full_text()
        st.text_area("Alt text (copy into your document)", text, key=f"alt-{f.figure_id}", height=120)
        st.download_button("Download as .txt", text, file_name=f"{f.figure_id}_alt_text.txt", key=f"dl-{f.figure_id}")
        if show_debug and r.usage:
            cost = r.total_cost
            st.caption(f"{len(r.usage)} model calls · {r.total_seconds:.1f}s · "
                       f"{sum(u.input_tokens for u in r.usage):,} in / {sum(u.output_tokens for u in r.usage):,} out tokens"
                       + (f" · ${cost:.4f}" if cost is not None else ""))

# ---------------------------------------------------------------- 5. export all
if results:
    st.divider()
    export = {fid: {"caption": r.caption, "alt_text": r.alt_text.model_dump(), "full_text": r.alt_text.full_text(),
                    "data": r.data.model_dump() if r.data else None,
                    "checks": [c.model_dump() for c in r.checks], "error": r.error}
              for fid, r in results.items()}
    st.download_button("Download all results (JSON)", json.dumps(export, indent=2), file_name="chartsight_results.json")
