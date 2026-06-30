from functools import lru_cache

import spacy


@lru_cache(maxsize=2)
def get_spacy_model(model_name: str = "en_core_web_sm"):
    """Cached loader for spaCy models to avoid redundant reload operations."""
    try:
        return spacy.load(model_name)
    except OSError:
        raise RuntimeError(
            f"Missing required spaCy model: '{model_name}'. Please run this command in your environment:\n"
            f"python -m spacy download {model_name}"
        )
