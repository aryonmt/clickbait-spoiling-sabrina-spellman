import logging
from typing import Any, Dict, List

import bert_score
import nltk
import sacrebleu
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


# Lazy download helper for NLTK packages if not present in runtime environment
def _ensure_nltk_resources():
    try:
        nltk.data.find("corpora/wordnet")
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        logger.info("Downloading required NLTK assets (wordnet, punkt, omw-1.4)...")
        nltk.download("wordnet", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("omw-1.4", quiet=True)


def bleu4(reference: str, hypothesis: str) -> float:
    """Sentence-level BLEU-4 with exponential smoothing to prevent zero scores on short spans."""
    # sacrebleu expects list of reference lists and flat hypothesis string
    return (
        sacrebleu.sentence_bleu(hypothesis, [reference], smooth_method="exp").score
        / 100.0
    )


def meteor(reference: str, hypothesis: str) -> float:
    """Compute METEOR score on tokenized input sequences."""
    _ensure_nltk_resources()
    ref_tokens = nltk.word_tokenize(reference)
    hyp_tokens = nltk.word_tokenize(hypothesis)
    return meteor_score([ref_tokens], hyp_tokens)


def bertscore_batch(
    references: List[str], hypotheses: List[str], lang: str = "en"
) -> List[float]:
    """Compute batched BERTScore F1 metric for maximum execution efficiency."""
    if not references or not hypotheses:
        return []
    P, R, F1 = bert_score.score(hypotheses, references, lang=lang, verbose=False)
    return F1.tolist()


def classification_report_dict(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Calculate overall Task 1 classification report and raw confusion matrix metrics."""
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=["phrase", "passage", "multi"])
    return {"report": report, "confusion_matrix": cm.tolist()}
