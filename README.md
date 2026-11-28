# ChartSight

Verified, layered alt text for charts in research papers.

A user uploads a paper PDF. ChartSight finds each chart, extracts its data with a vision-language model,
retrieves the paragraphs that discuss it, writes a layered description (summary, construction, statistics,
trends), and then **checks every number and claim against the extracted data**, rewriting or removing
sentences the data does not support.

## Setup (10 minutes)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # then put your API key and model in .env
python -m pytest -q                                     # 20 tests, no API key needed
python scripts/make_sample_paper.py                     # demo paper with 3 real vector charts
streamlit run app.py
```

With no `.env`, everything runs on an offline **mock** model, so you can work on the UI and the
evaluation code without spending credits. The mock gives every chart the same canned answer.

## How the code is organized

| Stage | File | GenAI component |
|---|---|---|
| 2. Find figures in a PDF | `chartsight/figures.py` | (PyMuPDF; caption-anchored cropping of vector and raster figures) |
| 3. Chart image to data | `chartsight/extract.py` | Multimodal understanding + structured output (JSON, validated by Pydantic) |
| 4. Paper context | `chartsight/context.py` | RAG: figure references + BM25, plus embeddings if `sentence-transformers` is installed |
| 5. Layered alt text | `chartsight/generate.py` | Prompt engineering, few-shot, structured output |
| 6. Verification | `chartsight/verify.py`, `chartsight/numbers.py` | Hallucination detection + output verification, with an LLM repair round |
| Chain + ablation switches | `chartsight/pipeline.py`, `chartsight/config.py` | Multi-stage pipeline |
| Providers, tokens, cost | `chartsight/llm.py` | Multiple model comparison (Claude, OpenAI, or any OpenAI-compatible server like Ollama) |
| App | `app.py` | Streamlit UI |
| Evaluation | `evaluation/` | Datasets, metrics, LLM judge, runner |

**What the verifier checks** (no model call needed):
- every number must match a data value or axis bound (within 5%, or 2% of the y range), a label such as a year,
  or a count of items;
- differences, percent changes, ratios, totals and means are accepted only when the sentence states such a
  comparison ("by", "increased", "times", "total"), and only within 2%; otherwise almost any invented
  number would match *some* difference between data values;
- "X is the highest/lowest" must name the actual maximum/minimum;
- a sentence naming one label must give that label's value, not another one's.

Failing sentences go back to the model once to be rewritten from the data; if they still fail, they are removed.

## Evaluation

```bash
# offline smoke test
python -m evaluation.run_eval --dataset sample --systems direct,full

# VisText: baseline vs. ablations vs. full system
python -m evaluation.run_eval --dataset vistext --data-dir data/vistext/data --n 50 \
    --systems direct,no_extraction,no_verify,full --judge --rating-sheet

# your annotated real-paper figures (also tests the paper-context ablation)
python -m evaluation.run_eval --dataset papers --data-dir data/papers --systems no_context,full

# consistency
python -m evaluation.run_eval --dataset vistext --data-dir data/vistext/data --n 30 --systems full --repeats 3
```

Each run writes `results/<name>/`: `summary.md` (the table for your slides and report), `per_chart.csv`,
`examples.md` (failure cases for the error analysis), and with `--rating-sheet` a blinded
`human_rating_sheet.csv` plus `rating_key.csv`.

| System | Extraction | Context | Verification | What it tests |
|---|---|---|---|---|
| `direct` | no | no | no | naive baseline: "write alt text for this chart" |
| `no_extraction` | no (generation sees the image only) | yes | yes | does writing from a data table help? |
| `no_context` | yes | no | yes | does the paper text help? (papers set only; VisText has no paper) |
| `no_verify` | yes | yes | no | **the key ablation**: what does verification buy? |
| `full` | yes | yes | yes | the proposed system |

Metrics: `value_recall` / `value_precision` (extraction vs. true data), `unsupported_number_rate` and
`unsupported_sentence_rate` (judged against the **true** data, never the model's own extraction),
`verifier_precision` / `verifier_recall` (do the verifier's flags agree with the truth?), BLEU and ROUGE-L
(comparable to the published VisText baseline: BLEU 0.32, ROUGE-L 0.45 for full captions), LLM-judge rubric
scores, latency, tokens and cost.

### Getting VisText

The data is hosted at vis.csail.mit.edu. On your machine:

```bash
git clone https://github.com/mitvis/vistext data/vistext-repo
cd data/vistext-repo && bash download_data.sh --images && cd data && unzip -o tabular.zip && unzip -o images.zip
# then use --data-dir data/vistext-repo/data (needs data_test.json and images/)
python -m evaluation.datasets --inspect data/vistext-repo/data     # eyeball the table parser on 5 charts
```

Check the folder names after unzipping and adjust `--data-dir` if needed. **Tune prompts on the `val` split
(`--split val`) and report numbers only on `test`.**

### Real-paper test set

Put openly licensed PDFs (e.g., CC BY arXiv papers) in `data/papers/` and describe each chart in
`data/papers/annotations.json` (see `annotations.example.json`): the figure id the app shows (e.g. `fig3`)
and the true x and y values, read from the paper's tables or text, or carefully from the chart.

## Plan to the demo (11/30) and report (12/15)

| Dates | Both | Partner A (extraction + verification + eval) | Partner B (retrieval + generation + app) |
|---|---|---|---|
| Wed 11/18 to Fri 11/20 | Run setup, get an API key, read this README | Download VisText; run `--inspect`; 20 `val` charts with `full` and `direct`; fix extraction prompt failures | Try the app on 3 real papers; fix figure-detection misses; set up Ollama with an open vision model |
| Sat 11/21 to Tue 11/24 | Annotate ~15 real paper figures | **100 VisText test charts x 4 systems**: the first results table | Open model vs. API model on 50 charts; `papers` run with `no_context` vs. `full` |
| Wed 11/25 | **Feature freeze.** Results slide, demo script | | |
| 11/26 to 11/27 | Thanksgiving: buffer only | | |
| Sat 11/28 to Sun 11/29 | Rehearse; record a backup demo video in case the network fails | | |
| **Mon 11/30** | **Demo** | | |
| 12/1 to 12/15 | Scale to ~300 VisText + 60 to 100 paper figures; rate 50 charts by hand and report judge agreement; `--repeats 3` consistency; write the report | | |

**Demo story (5 minutes):** the problem (only about 7.5% of papers have figure alt text) → upload a real paper →
play one chart's description with your computer's screen reader (VoiceOver or NVDA) → open "sentence-by-sentence checks" and show a sentence the
verifier caught and fixed → the results table: `direct` vs. `no_verify` vs. `full`.
