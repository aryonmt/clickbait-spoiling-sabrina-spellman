import torch
from typing import List, Dict
from tqdm import tqdm
from clickbait_spoiling.schema import ClickbaitPost
from clickbait_spoiling.constants import ID2LABEL
from clickbait_spoiling.data.preprocessing import build_classification_example


def predict_spoiler_types(
    model, tokenizer, posts: List[ClickbaitPost], device: str, batch_size: int = 32
) -> Dict[str, str]:
    """Perform batched sequence classification inference over posts to predict spoiler types."""
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

    for idx in tqdm(range(0, len(texts), batch_size), desc="Predicting spoiler types"):
        batch_texts = texts[idx : idx + batch_size]
        batch_uuids = uuids[idx : idx + batch_size]

        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=tokenizer.model_max_length,
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
