from unittest.mock import MagicMock

import pytest

from clickbait_spoiling.schema import ClickbaitPost


@pytest.fixture
def sample_post_phrase():
    return ClickbaitPost(
        uuid="1111-2222",
        post_text=["This is a clickbait post!"],
        target_paragraphs=[
            "This is paragraph one.",
            "This is paragraph two, containing python spoiler.",
        ],
        target_title="Amazing Title",
        target_url="http://example.com",
        human_spoiler=["python spoiler"],
        spoiler=["python spoiler"],
        # "python spoiler" starts exactly at index 34 and ends at 48 in paragraph 1
        spoiler_positions=[[[1, 34], [1, 48]]],
        tags=["phrase"],
    )


@pytest.fixture
def sample_post_multi():
    return ClickbaitPost(
        uuid="3333-4444",
        post_text=["7 amazing tools!"],
        target_paragraphs=["We have lists:", "1. Git", "2. Python", "3. VS Code"],
        target_title="Tool listicle",
        target_url="http://example.com",
        human_spoiler=["Git, Python, VS Code"],
        spoiler=["Git", "Python", "VS Code"],
        spoiler_positions=[[[1, 3], [1, 6]], [[2, 3], [2, 9]], [[3, 3], [3, 10]]],
        tags=["multi"],
    )


@pytest.fixture
def mock_tokenizer():
    """Mock standard Hugging Face tokenizer encoding returns."""
    tokenizer = MagicMock()
    tokenizer.model_max_length = 384

    # Simple simulator function
    def tokenize_sim(questions, contexts, **kwargs):
        res = MagicMock()
        res.sequence_ids = lambda i: [None, 0, 1, 1, None]
        res.__getitem__ = lambda self, key: {
            "input_ids": [101, 1000, 2000, 3000, 102],
            "attention_mask": [1, 1, 1, 1, 1],
            "token_type_ids": [0, 0, 1, 1, 0],
            "offset_mapping": [(0, 0), (0, 4), (0, 5), (6, 12), (0, 0)],
        }.get(key)
        return res

    tokenizer.side_effect = tokenize_sim
    return tokenizer
