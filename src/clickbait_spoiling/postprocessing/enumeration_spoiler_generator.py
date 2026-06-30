import logging
from typing import List, Optional

from clickbait_spoiling.constants import MAX_MULTI_SPOILERS
from clickbait_spoiling.nlp.enumeration import (
    detect_enumeration_question,
    find_enumerations_in_text,
)

logger = logging.getLogger(__name__)


def generate_enumeration_spoiler(
    post_text: str,
    article_text: str,
    max_items: int = MAX_MULTI_SPOILERS,
) -> Optional[List[str]]:
    """Extract multi-spoilers from monotonic lists if the question follows an enumeration template."""
    cardinal_num = detect_enumeration_question(post_text)
    if not cardinal_num:
        return None

    items = find_enumerations_in_text(article_text)
    if len(items) < 2:
        return None

    # Pick items, capping at whichever value is smaller
    target_count = min(cardinal_num, len(items), max_items)
    spoilers = [it.text for it in items[:target_count]]

    logger.info(
        f"Enumeration heuristic triggered. Extracted {len(spoilers)} spoilers for: '{post_text[:30]}...'"
    )
    return spoilers
