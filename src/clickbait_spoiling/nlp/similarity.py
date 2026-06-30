from typing import Literal

from rapidfuzz import fuzz

from clickbait_spoiling.nlp.spacy_loader import get_spacy_model
from clickbait_spoiling.utils.text_utils import strip_for_comparison


class SpanSimilarityScorer:
    """Computes similarity scores in [0, 1] between two spoiler strings to avoid duplicates."""

    def __init__(
        self,
        method: Literal["fuzzy", "jaccard", "cosine", "spacy", "combined"] = "combined",
    ):
        self.method = method

    def _jaccard_similarity(self, a: str, b: str) -> float:
        set_a = set(a.split())
        set_b = set(b.split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a.intersection(set_b)) / len(set_a.union(set_b))

    def _spacy_similarity(self, a: str, b: str) -> float:
        try:
            nlp = get_spacy_model("en_core_web_sm")
            doc_a = nlp(a)
            doc_b = nlp(b)
            if doc_a.vector_norm and doc_b.vector_norm:
                return float(doc_a.similarity(doc_b))
            return 0.0
        except Exception:
            return 0.0

    def score(self, a: str, b: str) -> float:
        """Returns the similarity score based on the selected method."""
        clean_a = strip_for_comparison(a)
        clean_b = strip_for_comparison(b)

        if not clean_a or not clean_b:
            return 0.0

        if self.method == "fuzzy":
            return float(fuzz.token_sort_ratio(clean_a, clean_b) / 100.0)
        elif self.method == "jaccard":
            return self._jaccard_similarity(clean_a, clean_b)
        elif self.method == "spacy":
            return self._spacy_similarity(clean_a, clean_b)
        elif self.method == "combined":
            f_score = float(fuzz.token_sort_ratio(clean_a, clean_b) / 100.0)
            j_score = self._jaccard_similarity(clean_a, clean_b)
            s_score = self._spacy_similarity(clean_a, clean_b)
            return max(f_score, j_score, s_score)
        else:
            raise ValueError(f"Unknown similarity method: {self.method}")
