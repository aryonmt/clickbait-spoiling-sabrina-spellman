import re
from dataclasses import dataclass
from typing import List, Optional

from clickbait_spoiling.nlp.spacy_loader import get_spacy_model


@dataclass
class EnumerationItem:
    cardinal_number: int
    char_start: int
    char_end: int
    text: str


def detect_enumeration_question(post_text: str) -> Optional[int]:
    """Detect whether a clickbait post follows the cloze enumeration pattern (e.g. '7 Things We Know...')."""
    # 1. Regex check for rapid detection
    match = re.search(r"^\s*(\d+)\s+(\w+)", post_text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # 2. Dependency parser check using spaCy
    try:
        nlp = get_spacy_model("en_core_web_sm")
        doc = nlp(post_text)
        for token in doc[:5]:  # Scanning only the beginning of the post
            if token.like_num:
                try:
                    return int(token.text)
                except ValueError:
                    continue
    except Exception:
        pass
    return None


def find_enumerations_in_text(article_text: str) -> List[EnumerationItem]:
    """Scan article_text for strictly monotonic ascending or descending list items."""
    # Matches patterns like "1. ", "2) ", " [3] " at line or sentence boundaries
    pattern = re.compile(
        r"(?:^|\n|\.\s+)(?P<num>\d+)\s*[\.\)\]\-]\s+(?P<content>[^\n]+)"
    )
    matches = list(pattern.finditer(article_text))

    if not matches:
        return []

    items = []
    for m in matches:
        num = int(m.group("num"))
        start = m.start("content")
        content = m.group("content").strip()

        # Stop at sentence boundary or a newline
        sentence_end = re.search(r"\.\s+|\n", content)
        end = start + (sentence_end.start() if sentence_end else len(content))
        text = article_text[start:end].strip()
        items.append(
            EnumerationItem(
                cardinal_number=num, char_start=start, char_end=end, text=text
            )
        )

    if len(items) < 2:
        return []

    # Check for strictly ascending order (1, 2, 3...)
    is_ascending = True
    for i in range(len(items) - 1):
        if items[i + 1].cardinal_number != items[i].cardinal_number + 1:
            is_ascending = False
            break

    if is_ascending:
        return items

    # Check for strictly descending order (10, 9, 8...)
    is_descending = True
    for i in range(len(items) - 1):
        if items[i + 1].cardinal_number != items[i].cardinal_number - 1:
            is_descending = False
            break

    if is_descending:
        return items

    return []
