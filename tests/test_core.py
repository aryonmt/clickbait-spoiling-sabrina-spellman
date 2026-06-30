import os
import tempfile

from clickbait_spoiling.evaluation.metrics import bleu4, meteor
from clickbait_spoiling.utils.io_utils import directory_size_gb, read_jsonl, write_jsonl
from clickbait_spoiling.utils.text_utils import (
    normalize_whitespace,
    strip_for_comparison,
    word_count,
)


def test_io_utils_lifecycle():
    with tempfile.TemporaryDirectory() as tmp_dir:
        target_path = os.path.join(tmp_dir, "test.jsonl")
        data = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]

        # Test writing
        write_jsonl(data, target_path)
        assert os.path.exists(target_path)

        # Test reading
        loaded = read_jsonl(target_path)
        assert len(loaded) == 2
        assert loaded[0]["text"] == "hello"

        # Test directory size calculation
        size = directory_size_gb(tmp_dir)
        assert size >= 0.0


def test_schema_parsing(sample_post_phrase):
    assert sample_post_phrase.joined_post_text() == "This is a clickbait post!"
    context, offsets = sample_post_phrase.joined_context()
    assert "paragraph one" in context
    assert offsets[0] == 0
    assert offsets[1] > 0
    assert sample_post_phrase.gold_tag() == "phrase"


def test_text_utils():
    assert normalize_whitespace("hello   world \t text") == "hello world text"
    assert word_count("this is a test") == 4
    # Strip stopwords and casing
    assert strip_for_comparison("The Python program!") == "python program"


def test_evaluation_metrics():
    ref = "this is a very specific spoiler text"
    hyp = "this is a very specific spoiler text"
    # Exact matches must yield high values
    assert bleu4(ref, hyp) > 0.99
    assert meteor(ref, hyp) > 0.99
