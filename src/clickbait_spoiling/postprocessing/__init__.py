from typing import Literal

import numpy as np

from clickbait_spoiling.nlp.similarity import SpanSimilarityScorer
from clickbait_spoiling.postprocessing.enumeration_spoiler_generator import (
    generate_enumeration_spoiler,
)
from clickbait_spoiling.postprocessing.multi_spoiler_generator import (
    generate_multi_spoiler,
)
from clickbait_spoiling.postprocessing.span_selector import (
    ScoredSpan,
    get_n_best_spans,
    select_best_span,
)

__all__ = [
    "get_n_best_spans",
    "select_best_span",
    "ScoredSpan",
    "generate_enumeration_spoiler",
    "generate_multi_spoiler",
    "postprocess",
]


def postprocess(
    spoiler_type: Literal["phrase", "passage", "multi"],
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    offset_mapping: list,
    context: str,
    post_text: str,
    article_text: str,
    similarity_scorer: SpanSimilarityScorer,
) -> str:
    """Central postprocessing dispatch based on predicted or gold spoiler type."""
    if spoiler_type in ["phrase", "passage"]:
        candidates = get_n_best_spans(
            start_logits,
            end_logits,
            offset_mapping,
            context,
            n_best_size=20,
            max_answer_length=64,
        )
        if not candidates:
            return ""
        best = select_best_span(
            candidates, spoiler_type=spoiler_type, forbid_linebreak=True
        )
        # Apply fallback if no constraint-satisfactory span is resolved
        if best is None:
            best = candidates[0]
        return best.text

    elif spoiler_type == "multi":
        spoilers = generate_multi_spoiler(
            start_logits,
            end_logits,
            offset_mapping,
            context,
            post_text,
            article_text,
            similarity_scorer,
        )
        # Official validator format joins multiple answers into a single flat space-separated string
        return " ".join(spoilers)

    else:
        raise ValueError(
            f"Unknown spoiler type encountered during postprocessing: {spoiler_type}"
        )
