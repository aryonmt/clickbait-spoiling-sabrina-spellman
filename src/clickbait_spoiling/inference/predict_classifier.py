import torch
from typing import List, Dict
from tqdm import tqdm
from clickbait_spoiling.schema import ClickbaitPost
from clickbait_spoiling.constants import ID2LABEL
from clickbait_spoiling.data.preprocessing import build_classification_example


def predict_spoiler_types(
    model, tokenizer, posts: List[ClickbaitPost], device: str, batch_size: int = 32
) -> Dict[str, str]:
    """Perform batched sequence classification inference over posts to predict spoiler types.
    Safely handles potentially overflowing model_max_length to prevent Rust tokenizer errors.
    """
    model.to(device)
    model.eval()

    predictions = {}

    # Prepare inputs using standard template
    texts = []
    uuids = []
    for post in posts:
        example = build_classification_example(post)
        texts.append(example["text"])
        uuids.append(post.uuid)

    # Resolve model_max_length safely to prevent Rust-level integer overflow
    max_len = tokenizer.model_max_length
    if max_len is None or max_len > 1000000:
        max_len = 512

    for idx in tqdm(range(0, len(texts), batch_size), desc="Predicting spoiler types"):
        batch_texts = texts[idx : idx + batch_size]
        batch_uuids = uuids[idx : idx + batch_size]

        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_len,  # Using resolved safe maximum length limit
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

        for uuid, pred_idx in zip(batch_uuids, preds):
            predictions[uuid] = ID2LABEL[pred_idx]

    return predictions
