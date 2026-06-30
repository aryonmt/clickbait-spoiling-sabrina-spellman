from typing import Dict

MODEL_REGISTRY: Dict[str, str] = {
    "deberta-base": "microsoft/deberta-v3-base",
    "deberta-large": "microsoft/deberta-v3-large",
    "bert-large": "bert-large-uncased",
}


def resolve_checkpoint(name_or_alias: str) -> str:
    """Look up name_or_alias in MODEL_REGISTRY; if not found, assume it's already a valid
    Hugging Face Hub ID and return it unchanged.
    """
    return MODEL_REGISTRY.get(name_or_alias, name_or_alias)
