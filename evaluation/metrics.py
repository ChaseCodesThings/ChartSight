"""Metrics. Each one answers a question from the proposal's evaluation plan."""
from __future__ import annotations

from typing import Optional

from chartsight.numbers import Support, find_numbers
from chartsight.schema import AltText, ChartData, SentenceCheck
from chartsight.verify import check_attribution, check_extremes


def _match(a: float, b: float, rel: float, abs_tol: float) -> bool:
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), abs_tol)


def extraction_scores(pred: Optional[ChartData], truth: ChartData, rel: float = 0.05, axis_frac: float = 0.02) -> dict:
    """Does it read the chart correctly?

    value_recall: share of true y values that some extracted value matches (within 5% or 2% of the y range)
    value_precision: share of extracted values that match some true value
    point_count_error: |#extracted points - #true points|
    """
    true_ys = [p.y for _, p in truth.all_points()]
    if pred is None:
        return {"value_recall": 0.0, "value_precision": 0.0, "point_count_error": len(true_ys)}
    pred_ys = [p.y for _, p in pred.all_points()]
    span = (max(true_ys) - min(true_ys)) if len(true_ys) > 1 else (abs(true_ys[0]) if true_ys else 1.0)
    tol = axis_frac * span
    recall = sum(any(_match(t, p, rel, tol) for p in pred_ys) for t in true_ys) / len(true_ys) if true_ys else 0.0
    precision = sum(any(_match(p, t, rel, tol) for t in true_ys) for p in pred_ys) / len(pred_ys) if pred_ys else 0.0
    return {"value_recall": round(recall, 4), "value_precision": round(precision, 4),
            "point_count_error": abs(len(pred_ys) - len(true_ys))}


def hallucination_scores(alt: AltText, truth: ChartData, rel: float = 0.05, axis_tol: float = 0.02) -> dict:
    """Does it make things up? Checked against the TRUE data, never the model's own extraction."""
    sup = Support(truth, rel, axis_tol)
    n_nums = n_bad = n_sent = n_bad_sent = 0
    bad_examples = []
    for level, s in alt.sentences():
        n_sent += 1
        mentions = find_numbers(s)
        bad = [m.raw for m in mentions if sup.check(m, s) is None]
        wrong = check_extremes(s, truth) + check_attribution(s, sup)
        n_nums += len(mentions)
        n_bad += len(bad)
        if bad or wrong:
            n_bad_sent += 1
            bad_examples.append(f"{s}  [{'; '.join(([f'unsupported: {chr(44).join(bad)}'] if bad else []) + wrong)}]")
    return {"n_sentences": n_sent, "n_numbers": n_nums, "n_unsupported_numbers": n_bad,
            "unsupported_number_rate": round(n_bad / n_nums, 4) if n_nums else 0.0,
            "unsupported_sentence_rate": round(n_bad_sent / n_sent, 4) if n_sent else 0.0,
            "bad_examples": bad_examples}


def verifier_agreement(checks: list[SentenceCheck], truth: ChartData, rel: float = 0.05, axis_tol: float = 0.02) -> dict:
    """How well do the verifier's flags (made from EXTRACTED data) match unsupported-ness under TRUE data?
    Computed on the sentences as first generated (before any rewrite)."""
    sup = Support(truth, rel, axis_tol)
    tp = fp = fn = tn = 0
    for c in checks:
        text = c.original or c.text
        flagged = c.status in ("unsupported", "removed", "rewritten")
        truly_bad = any(sup.check(m, text) is None for m in find_numbers(text)) or \
            bool(check_extremes(text, truth) + check_attribution(text, sup))
        tp += flagged and truly_bad
        fp += flagged and not truly_bad
        fn += (not flagged) and truly_bad
        tn += (not flagged) and not truly_bad
    return {"flag_tp": tp, "flag_fp": fp, "flag_fn": fn, "flag_tn": tn}


def caption_scores(hyps: list[str], refs: list[list[str]]) -> dict:
    """Corpus BLEU (sacrebleu, multi-reference) and mean best-reference ROUGE-L F1, as in VisText.
    VisText reports BLEU on a 0-1 scale; sacrebleu reports 0-100, so we divide by 100."""
    import sacrebleu
    from rouge_score import rouge_scorer

    pairs = [(h, r) for h, r in zip(hyps, refs) if h and r]
    if not pairs:
        return {"bleu": None, "rougeL": None, "n": 0}
    hyps2 = [h for h, _ in pairs]
    max_refs = max(len(r) for _, r in pairs)
    ref_streams = [[r[i] if i < len(r) else r[0] for _, r in pairs] for i in range(max_refs)]
    bleu = sacrebleu.corpus_bleu(hyps2, ref_streams).score / 100
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl = [max(sc.score(ref, h)["rougeL"].fmeasure for ref in r) for h, r in pairs]
    return {"bleu": round(bleu, 4), "rougeL": round(sum(rl) / len(rl), 4), "n": len(pairs)}


def bertscore(hyps: list[str], refs: list[list[str]]) -> Optional[float]:
    """Optional: needs `pip install bert-score` (downloads a model on first use)."""
    try:
        from bert_score import score
    except ImportError:
        return None
    pairs = [(h, r) for h, r in zip(hyps, refs) if h and r]
    if not pairs:
        return None
    _, _, f = score([h for h, _ in pairs], [r for _, r in pairs], lang="en", verbose=False)
    return round(float(f.mean()), 4)
