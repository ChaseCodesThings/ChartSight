"""Run systems on a test set and write per-chart results plus one summary table.

Examples
  # offline smoke test on the generated sample paper (no API key needed)
  python -m evaluation.run_eval --dataset sample --systems direct,full

  # 50 VisText test charts, baseline vs. full pipeline vs. ablations
  python -m evaluation.run_eval --dataset vistext --data-dir path/to/vistext/data --n 50 \
      --systems direct,no_extraction,no_verify,full --judge

  # consistency: each chart 3 times
  python -m evaluation.run_eval --dataset vistext --data-dir ... --n 30 --systems full --repeats 3

Outputs go to results/<run name>/: per_chart.csv, summary.csv, summary.md, examples.md, config.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from chartsight.config import PipelineConfig, model_from_env
from chartsight.context import Retriever, split_paragraphs
from chartsight.llm import LLMClient
from chartsight.pipeline import describe_chart

from .datasets import EvalItem, load_papers, load_vistext, sample_paper_as_testset
from .metrics import caption_scores, extraction_scores, hallucination_scores, verifier_agreement

ROOT = Path(__file__).resolve().parent.parent


def load(args) -> list[EvalItem]:
    if args.dataset == "vistext":
        return load_vistext(args.data_dir, split=args.split, n=args.n, seed=args.seed)
    if args.dataset == "papers":
        items = load_papers(args.data_dir)
        return items[: args.n] if args.n else items
    if args.dataset == "sample":
        pdf = ROOT / "data" / "sample_paper.pdf"
        if not pdf.exists():
            from scripts.make_sample_paper import build
            build(pdf)
        return sample_paper_as_testset(pdf)
    raise SystemExit(f"unknown dataset {args.dataset}")


def run_one(item: EvalItem, system: str, rep: int, args, judge_client) -> dict:
    cfg = PipelineConfig.ablation(system)
    retriever = Retriever(split_paragraphs(item.paper_text), use_embeddings=not args.no_embeddings) \
        if (item.paper_text and cfg.use_context) else None
    res = describe_chart(item.key, item.image, cfg, caption=item.caption, retriever=retriever)
    row = {"system": system, "chart": item.key, "repeat": rep, "source": item.meta.get("source"),
           "n_true_points": item.meta.get("n_points"), "error": res.error or "",
           "seconds": round(res.total_seconds, 3),
           "input_tokens": sum(u.input_tokens for u in res.usage),
           "output_tokens": sum(u.output_tokens for u in res.usage),
           "cost_usd": res.total_cost, "model_calls": len(res.usage),
           "confidence": res.data.confidence if res.data else "",
           "text": res.alt_text.full_text()}
    if res.data is not None:
        row.update(extraction_scores(res.data, item.truth))
    h = hallucination_scores(res.alt_text, item.truth)
    row["bad_examples"] = " || ".join(h.pop("bad_examples"))
    row.update(h)
    if res.checks:
        row.update(verifier_agreement(res.checks, item.truth))
        row["n_removed"] = sum(c.status == "removed" for c in res.checks)
        row["n_rewritten"] = sum(c.status == "rewritten" for c in res.checks)
    row["_refs"] = item.references
    row["_extracted"] = [(p.x, p.y) for _, p in res.data.all_points()] if res.data else []
    if judge_client is not None and not res.error:
        from .judge import judge
        try:
            scores, usage = judge(judge_client, res.alt_text, item.truth)
            row.update({f"judge_{k}": v for k, v in scores.items()})
        except Exception as e:
            row["judge_error"] = str(e)
    return row


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.mean(xs), 4) if xs else None


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for system in dict.fromkeys(r["system"] for r in rows):
        rs = [r for r in rows if r["system"] == system]
        ok = [r for r in rs if not r["error"]]
        s = {"system": system, "charts": len({r["chart"] for r in rs}), "errors": len(rs) - len(ok)}
        for k in ("value_recall", "value_precision", "unsupported_number_rate", "unsupported_sentence_rate",
                  "n_sentences", "n_numbers", "seconds", "input_tokens", "output_tokens", "cost_usd",
                  "judge_accuracy", "judge_completeness", "judge_clarity", "judge_trend"):
            if any(k in r for r in ok):
                s[k] = _mean([r.get(k) for r in ok])
        if any("flag_tp" in r for r in ok):
            tp, fp, fn = (sum(r.get(k, 0) for r in ok) for k in ("flag_tp", "flag_fp", "flag_fn"))
            s["verifier_precision"] = round(tp / (tp + fp), 4) if tp + fp else None
            s["verifier_recall"] = round(tp / (tp + fn), 4) if tp + fn else None
        cap = caption_scores([r["text"] for r in ok if r["repeat"] == 0], [r["_refs"] for r in ok if r["repeat"] == 0])
        s["bleu"], s["rougeL"] = cap["bleu"], cap["rougeL"]
        # consistency across repeats: spread of value_recall per chart
        reps = {}
        for r in ok:
            if "value_recall" in r:
                reps.setdefault(r["chart"], []).append(r["value_recall"])
        spreads = [statistics.pstdev(v) for v in reps.values() if len(v) > 1]
        if spreads:
            s["recall_std_across_repeats"] = round(statistics.mean(spreads), 4)
        out.append(s)
    return out


def to_markdown(summary: list[dict]) -> str:
    cols = [c for c in ["system", "charts", "errors", "value_recall", "value_precision", "unsupported_number_rate",
                        "unsupported_sentence_rate", "verifier_precision", "verifier_recall", "bleu", "rougeL",
                        "judge_accuracy", "judge_completeness", "judge_clarity", "judge_trend",
                        "recall_std_across_repeats", "seconds", "input_tokens", "output_tokens", "cost_usd"]
            if any(c in s for s in summary)]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for s in summary:
        lines.append("| " + " | ".join("" if s.get(c) is None else str(s.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def main(argv=None):
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["vistext", "papers", "sample"], default="sample")
    ap.add_argument("--data-dir")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=None, help="number of charts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--systems", default="direct,full", help="comma list: direct,no_extraction,no_context,no_verify,full")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4, help="parallel API calls")
    ap.add_argument("--judge", action="store_true", help="add LLM-as-judge rubric scores (uses CHARTSIGHT_JUDGE_* or the main model)")
    ap.add_argument("--no-embeddings", action="store_true", help="BM25 only retrieval")
    ap.add_argument("--rating-sheet", action="store_true", help="also write a blank CSV for human rubric ratings")
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)

    items = load(args)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    model = model_from_env()
    judge_client = None
    if args.judge:
        jm = model_from_env("CHARTSIGHT_JUDGE")
        judge_client = LLMClient(jm if jm.provider != "mock" or model.provider == "mock" else model)

    name = args.name or f"{args.dataset}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(items)} charts x {len(systems)} systems x {args.repeats} repeats with {model.label} -> {out}", file=sys.stderr)

    jobs = [(it, s, r) for s in systems for r in range(args.repeats) for it in items]
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, row in enumerate(ex.map(lambda j: run_one(j[0], j[1], j[2], args, judge_client), jobs), 1):
            rows.append(row)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done", file=sys.stderr)

    summary = summarize(rows)
    keys = list(dict.fromkeys(k for r in rows for k in r if not k.startswith("_")))
    with open(out / "per_chart.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r.get(k) for k in keys} for r in rows])
    skeys = list(dict.fromkeys(k for s in summary for k in s))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=skeys)
        w.writeheader()
        w.writerows(summary)
    md = to_markdown(summary)
    (out / "summary.md").write_text(md + "\n")
    with open(out / "examples.md", "w") as f:  # failure cases for the error analysis
        for r in rows:
            if r.get("bad_examples") or r["error"]:
                f.write(f"## {r['system']} / {r['chart']}\n\n{r['error'] or ''}\n\n{r['bad_examples']}\n\n")
    (out / "config.json").write_text(json.dumps({"args": vars(args), "model": model.label}, indent=2))
    if args.rating_sheet:
        # blind: raters see an item id, not the system; the key file maps ids back
        import random
        rated = [r for r in rows if r["repeat"] == 0 and not r["error"]]
        random.Random(args.seed).shuffle(rated)
        with open(out / "human_rating_sheet.csv", "w", newline="") as f, open(out / "rating_key.csv", "w", newline="") as k:
            w, wk = csv.writer(f), csv.writer(k)
            w.writerow(["item", "chart", "text", "accuracy_1to5", "completeness_1to5", "clarity_1to5", "trend_1to5", "notes"])
            wk.writerow(["item", "system"])
            for i, r in enumerate(rated, 1):
                w.writerow([i, r["chart"], r["text"], "", "", "", "", ""])
                wk.writerow([i, r["system"]])
    print(md)
    return summary


if __name__ == "__main__":
    main()
