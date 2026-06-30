from clickbait_spoiling.inference.pipeline import run_pipeline
from clickbait_spoiling.inference.predict_classifier import predict_spoiler_types
from clickbait_spoiling.inference.predict_qa import predict_qa_logits

__all__ = [
    "predict_spoiler_types",
    "predict_qa_logits",
    "run_pipeline",
]
