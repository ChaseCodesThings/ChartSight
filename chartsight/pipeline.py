"""Chain the stages for one chart. The switches in PipelineConfig drive the ablation study."""
from __future__ import annotations

from typing import Optional

from .config import PipelineConfig
from .context import Retriever
from .extract import extract_chart
from .generate import direct_alt_text, generate_alt_text
from .llm import LLMClient
from .schema import ChartResult, SentenceCheck
from .verify import check_sentence, verify
from .numbers import Support


def describe_chart(figure_id: str, image: bytes, cfg: PipelineConfig, caption: Optional[str] = None,
                   retriever: Optional[Retriever] = None, client: Optional[LLMClient] = None) -> ChartResult:
    client = client or LLMClient(cfg.model)
    res = ChartResult(figure_id=figure_id, caption=caption)
    try:
        # 3. structured extraction. Also run when only verification is on, since checks need the data.
        if cfg.use_extraction or cfg.use_verification:
            res.data, u = extract_chart(client, image, caption)
            res.usage += u

        # 4. context retrieval
        if cfg.use_context and retriever is not None:
            res.context = retriever.for_figure(caption, k=cfg.context_k)

        # 5. generation
        if not cfg.use_extraction and not cfg.use_context:
            res.alt_text, u = direct_alt_text(client, image)
        else:
            res.alt_text, u = generate_alt_text(client, res.data if cfg.use_extraction else None, image,
                                                caption, res.context)
        res.usage += u

        # 6. verification
        if cfg.use_verification and res.data is not None:
            res.alt_text, res.checks, u = verify(res.alt_text, res.data, client, rounds=cfg.repair_rounds,
                                                 rel_tol=cfg.rel_tolerance, axis_tol=cfg.axis_tolerance)
            res.usage += u
        elif res.data is not None:
            # still record what WOULD have been flagged, without changing the text
            sup = Support(res.data, cfg.rel_tolerance, cfg.axis_tolerance)
            res.checks = [check_sentence(l, t, sup) for l, t in res.alt_text.sentences()]
    except Exception as e:  # keep the app and batch runs alive; the error is shown per figure
        res.error = f"{type(e).__name__}: {e}"
    return res


def flag_summary(checks: list[SentenceCheck]) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in checks:
        out[c.status] = out.get(c.status, 0) + 1
    return out
