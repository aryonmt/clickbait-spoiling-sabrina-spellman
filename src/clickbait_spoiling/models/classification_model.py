from transformers import AutoModelForSequenceClassification, AutoTokenizer

from clickbait_spoiling.models.model_factory import resolve_checkpoint


def build_classifier(model_name: str, num_labels: int = 3):
    """Returns (model, tokenizer) = (AutoModelForSequenceClassification, AutoTokenizer)
    for Task 1 classification.
    """
    checkpoint = resolve_checkpoint(model_name)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels
    )
    return model, tokenizer
