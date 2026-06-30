from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np

from clickbait_spoiling.constants import MAX_PHRASE_WORDS, MIN_PASSAGE_WORDS


@dataclass
class ScoredSpan:
    text: str
    score: float  # Combined start and end logit score
    char_start: int
    char_end: int


def get_n_best_spans(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    offset_mapping: List[Optional[Tuple[int, int]]],
    context: str,
    n_best_size: int = 20,
    max_answer_length: int = 64,
) -> List[ScoredSpan]:
    """Standard SQuAD-style n-best span extraction based on logit combinations."""
    start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
    end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

    candidates = []
    for start_idx in start_indexes:
        for end_idx in end_indexes:
            if start_idx >= len(offset_mapping) or end_idx >= len(offset_mapping):
                continue
            if offset_mapping[start_idx] is None or offset_mapping[end_idx] is None:
                continue
            if end_idx < start_idx:
                continue
            if end_idx - start_idx + 1 > max_answer_length:
                continue

            char_start = offset_mapping[start_idx][0]
            char_end = offset_mapping[end_idx][1]

            span_text = context[char_start:char_end].strip()
            score = float(start_logits[start_idx] + end_logits[end_idx])

            candidates.append(
                ScoredSpan(
                    text=span_text,
                    score=score,
                    char_start=char_start,
                    char_end=char_end,
                )
            )

    candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
    return candidates


def select_best_span(
    candidates: List[ScoredSpan],
    spoiler_type: Literal["phrase", "passage"],
    forbid_linebreak: bool = True,
) -> Optional[ScoredSpan]:
    """Walk candidates in score order and return the first one satisfying type constraints."""
    for cand in candidates:
        if not cand.text:
            continue
        if forbid_linebreak and "\n" in cand.text:
            continue

        words = cand.text.split()
        if spoiler_type == "phrase" and len(words) > MAX_PHRASE_WORDS:
            continue
        if spoiler_type == "passage" and len(words) < MIN_PASSAGE_WORDS:
            continue

        return cand
    return None
