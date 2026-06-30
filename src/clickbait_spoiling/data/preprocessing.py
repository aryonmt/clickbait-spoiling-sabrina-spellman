import logging
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz

from clickbait_spoiling.constants import LABEL2ID
from clickbait_spoiling.schema import ClickbaitPost

logger = logging.getLogger(__name__)


def join_paragraphs_with_offsets(
    paragraphs: List[str], sep: str = "\n"
) -> Tuple[str, List[int]]:
    """Join paragraphs with 'sep', return the joined string and the char offset of each
    paragraph's start within it.
    """
    paragraph_offsets = []
    current_offset = 0
    joined_paragraphs = []

    for p in paragraphs:
        paragraph_offsets.append(current_offset)
        joined_paragraphs.append(p)
        current_offset += len(p) + len(sep)

    context = sep.join(joined_paragraphs)
    return context, paragraph_offsets


def spoiler_positions_to_char_span(
    spoiler_position: List[List[int]], paragraph_offsets: List[int]
) -> Tuple[int, int]:
    """Convert official [[para_start, char_start], [para_end, char_end]] into absolute char spans."""
    if len(spoiler_position) != 2:
        raise ValueError(
            "Invalid spoiler position shape, expected [[p_start, c_start], [p_end, c_end]]"
        )

    start_pos, end_pos = spoiler_position[0], spoiler_position[1]
    p_start, c_start = start_pos[0], start_pos[1]
    p_end, c_end = end_pos[0], end_pos[1]

    if p_start >= len(paragraph_offsets) or p_end >= len(paragraph_offsets):
        raise IndexError("Paragraph index out of bounds based on provided offsets.")

    abs_start = paragraph_offsets[p_start] + c_start
    abs_end = paragraph_offsets[p_end] + c_end
    return abs_start, abs_end


def find_answer_span_by_string_match(
    context: str, answer_text: str
) -> Optional[Tuple[int, int]]:
    """Locate answer_text inside context using exact match or sliding window fuzzy match."""
    if not answer_text or not context:
        return None

    # 1. Try Exact match
    idx = context.find(answer_text)
    if idx != -1:
        return idx, idx + len(answer_text)

    # 2. Sliding window fuzzy matching (Robust fallback for minor tokenization/spacing mismatches)
    ans_len = len(answer_text)
    best_ratio = 0.0
    best_span = None

    # We scan with steps to make it fast
    step = max(1, ans_len // 10)
    for length in range(max(1, ans_len - 15), ans_len + 15):
        for start in range(0, len(context) - length + 1, step):
            window = context[start : start + length]
            ratio = fuzz.ratio(window, answer_text)
            if ratio > best_ratio:
                best_ratio = ratio
                best_span = (start, start + length)

    if best_ratio > 80.0 and best_span is not None:
        return best_span

    return None


def build_classification_example(post: ClickbaitPost) -> Dict:
    """Build a single Task 1 input: postText SEP targetTitle SEP context."""
    post_text = post.joined_post_text()
    context, _ = post.joined_context()

    label = LABEL2ID.get(post.gold_tag()) if post.gold_tag() else -1

    return {
        "uuid": post.uuid,
        "text": f"{post_text} [SEP] {post.target_title} [SEP] {context}",
        "label": label,
    }


def build_qa_example(post: ClickbaitPost) -> Optional[Dict]:
    """Build a single Task 2 training input mapping multiple answers if present."""
    context, paragraph_offsets = post.joined_context()
    question = post.joined_post_text()

    answers_spans = []
    answers_text = []

    # If gold answers are not present (inference on test set)
    if not post.spoiler or not post.spoiler_positions:
        return {
            "uuid": post.uuid,
            "question": question,
            "context": context,
            "answers": {"answer_start": [], "text": []},
        }

    for spoiler_text, spoiler_pos in zip(post.spoiler, post.spoiler_positions):
        try:
            start, end = spoiler_positions_to_char_span(spoiler_pos, paragraph_offsets)
            extracted_text = context[start:end]

            # If coordinates do not yield the correct string, apply fuzzy fallback
            if extracted_text.strip() != spoiler_text.strip():
                fallback_span = find_answer_span_by_string_match(context, spoiler_text)
                if fallback_span:
                    start, end = fallback_span
                    extracted_text = context[start:end]
                else:
                    continue  # Mismatched individual span skipped
            answers_spans.append(start)
            answers_text.append(extracted_text)
        except Exception as e:
            logger.debug(
                f"Span calculation failed for uuid {post.uuid}: {e}. Retrying with fuzzy string matching."
            )
            fallback_span = find_answer_span_by_string_match(context, spoiler_text)
            if fallback_span:
                start, end = fallback_span
                answers_spans.append(start)
                answers_text.append(context[start:end])

    if not answers_spans:
        logger.warning(
            f"Could not align any gold spoiler spans for post uuid: {post.uuid}. Skipping example."
        )
        return None

    return {
        "uuid": post.uuid,
        "question": question,
        "context": context,
        "answers": {"answer_start": answers_spans, "text": answers_text},
    }
