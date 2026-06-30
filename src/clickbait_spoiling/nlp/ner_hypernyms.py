import logging
import re
from typing import Optional, Tuple

from clickbait_spoiling.nlp.spacy_loader import get_spacy_model

logger = logging.getLogger(__name__)

HYPERNYM_MAP: dict[str, str] = {
    "PERSON": "this person",
    "ORG": "this organization",
    "GPE": "this place",
    "PRODUCT": "this product",
    "EVENT": "this event",
    "WORK_OF_ART": "this work",
    "FAC": "this place",
}


def find_shared_named_entity(
    title: str, article_text: str
) -> Optional[Tuple[str, str]]:
    """Run NER on the title, check if any named entity also appears verbatim in article_text."""
    try:
        nlp = get_spacy_model("en_core_web_sm")
        doc = nlp(title)
        for ent in doc.ents:
            if ent.label_ in HYPERNYM_MAP:
                if ent.text.lower() in article_text.lower():
                    return ent.text, ent.label_
    except Exception as e:
        logger.debug(f"NER extraction failed during CSCP pipeline: {e}")
    return None


def replace_entity_with_hypernym(
    title: str, entity_text: str, entity_label: str
) -> Optional[str]:
    """Replace entity_text in title with predefined hypernym phrase from HYPERNYM_MAP."""
    hypernym = HYPERNYM_MAP.get(entity_label)
    if not hypernym:
        return None
    # Case-insensitive substitution
    pattern = re.compile(re.escape(entity_text), re.IGNORECASE)
    return pattern.sub(hypernym, title)
