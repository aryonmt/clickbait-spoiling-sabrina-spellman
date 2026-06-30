import logging
from typing import List

import numpy as np

from clickbait_spoiling.constants import MAX_MULTI_ITERATIONS, MAX_MULTI_SPOILERS
from clickbait_spoiling.nlp.similarity import SpanSimilarityScorer
from clickbait_spoiling.postprocessing.enumeration_spoiler_generator import (
    generate_enumeration_spoiler,
)
from clickbait_spoiling.postprocessing.span_selector import get_n_best_spans

logger = logging.getLogger(__name__)


def generate_multi_spoiler(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    offset_mapping: list,
    context: str,
    post_text: str,
    article_text: str,
    similarity_scorer: SpanSimilarityScorer,
    similarity_threshold: float = 0.7,
    max_iterations: int = MAX_MULTI_ITERATIONS,
    max_spoilers: int = MAX_MULTI_SPOILERS,
) -> List[str]:
    """Continuous-span iterative generator with zero-out logits tracking and similarity deduplication."""
    # 1. Attempt sequence/listicle pattern-based extraction first
    enum_spoilers = generate_enumeration_spoiler(
        post_text, article_text, max_items=max_spoilers
    )
    if enum_spoilers:
        return enum_spoilers

    # Copy logits to safely modify token scores inside loop
    working_start_logits = start_logits.copy()
    working_end_logits = end_logits.copy()

    spoilers: List[str] = []

    for iteration in range(max_iterations):
        if len(spoilers) >= max_spoilers:
            break

        candidates = get_n_best_spans(
            working_start_logits,
            working_end_logits,
            offset_mapping,
            context,
            n_best_size=20,
            max_answer_length=64,
        )

        if not candidates:
            break

        # Select highest scoring unconstrained span candidate
        best_span = candidates[0]
        if not best_span.text:
            break

        # Deduplicate using text-similarity scoring
        is_duplicate = False
        for existing in spoilers:
            if (
                similarity_scorer.score(best_span.text, existing)
                >= similarity_threshold
            ):
                is_duplicate = True
                break

        if not is_duplicate:
            spoilers.append(best_span.text)

        # Zero out (Cover Your Tracks) overlaps
        for k, offset in enumerate(offset_mapping):
            if offset is None:
                continue
            t_start, t_end = offset[0], offset[1]
            if max(t_start, best_span.char_start) < min(t_end, best_span.char_end):
                working_start_logits[k] = -10000.0
                working_end_logits[k] = -10000.0

    return spoilers
