import json
from unittest.mock import MagicMock

import numpy as np
import torch

from clickbait_spoiling.inference.pipeline import run_pipeline
from clickbait_spoiling.nlp.similarity import SpanSimilarityScorer
from clickbait_spoiling.postprocessing.multi_spoiler_generator import (
    generate_multi_spoiler,
)
from clickbait_spoiling.postprocessing.span_selector import ScoredSpan, select_best_span


class MockBatchEncoding(dict):
    """Perfect dictionary simulation of Hugging Face BatchEncoding to bypass tokenization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self["input_ids"] = [[101, 1000, 2000, 3000, 102]]
        self["attention_mask"] = [[1, 1, 1, 1, 1]]
        self["offset_mapping"] = [[(0, 0), (0, 4), (0, 5), (6, 12), (0, 0)]]
        self["overflow_to_sample_mapping"] = [0]
        self["token_type_ids"] = [[0, 0, 1, 1, 0]]

    def sequence_ids(self, i):
        return [None, 0, 1, 1, None]


def test_span_selector_constraints():
    candidates = [
        ScoredSpan(text="short", score=10.0, char_start=0, char_end=5),
        ScoredSpan(
            text="this is a very long text indeed", score=9.5, char_start=0, char_end=31
        ),
        ScoredSpan(text="invalid \n linebreak", score=12.0, char_start=0, char_end=18),
    ]

    best_phrase = select_best_span(
        candidates, spoiler_type="phrase", forbid_linebreak=True
    )
    assert best_phrase.text == "short"

    best_passage = select_best_span(
        candidates, spoiler_type="passage", forbid_linebreak=True
    )
    assert best_passage.text == "this is a very long text indeed"


def test_iterative_zero_out_generator():
    start_logits = np.array([0.0, 10.0, 0.0, 5.0, 0.0])
    end_logits = np.array([0.0, 10.0, 0.0, 5.0, 0.0])
    # Correction: 'orange' spans from char index 10 to 16 inside context
    offset_mapping = [None, (0, 5), None, (10, 16), None]
    context = "apple     orange"

    scorer = SpanSimilarityScorer(method="combined")
    spoilers = generate_multi_spoiler(
        start_logits=start_logits,
        end_logits=end_logits,
        offset_mapping=offset_mapping,
        context=context,
        post_text="Unrelated post text",
        article_text="Unrelated article text",
        similarity_scorer=scorer,
        max_iterations=2,
        max_spoilers=2,
    )
    assert len(spoilers) == 2
    assert "apple" in spoilers
    assert "orange" in spoilers


def test_pipeline_smoke_run(sample_post_phrase, tmp_path, monkeypatch):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    def tokenize_sim(questions, contexts, **kwargs):
        return MockBatchEncoding()

    mock_tokenizer.side_effect = tokenize_sim
    mock_tokenizer.model_max_length = 384

    monkeypatch.setattr(
        "clickbait_spoiling.inference.pipeline.build_qa_model",
        lambda checkpoint: (mock_model, mock_tokenizer),
    )

    mock_outputs = MagicMock()
    mock_outputs.start_logits = torch.zeros((1, 5))
    mock_outputs.end_logits = torch.zeros((1, 5))
    mock_model.return_value = mock_outputs

    output_path = tmp_path / "run.jsonl"

    run_pipeline(
        posts=[sample_post_phrase],
        classifier_dir=None,
        qa_model_dirs={"all": "mock-universal-checkpoint"},
        output_path=str(output_path),
        use_gold_tags=True,
        device="cpu",
    )

    assert output_path.exists()
    with open(output_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 1
    assert records[0]["uuid"] == "1111-2222"
    assert "spoiler" in records[0]
