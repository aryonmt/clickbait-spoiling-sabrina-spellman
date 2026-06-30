from clickbait_spoiling.data.preprocessing import (
    find_answer_span_by_string_match,
    spoiler_positions_to_char_span,
)
from clickbait_spoiling.nlp.enumeration import (
    detect_enumeration_question,
    find_enumerations_in_text,
)
from clickbait_spoiling.nlp.similarity import SpanSimilarityScorer


def test_spoiler_span_offsets(sample_post_phrase):
    context, paragraph_offsets = sample_post_phrase.joined_context()
    start, end = spoiler_positions_to_char_span(
        sample_post_phrase.spoiler_positions[0], paragraph_offsets
    )
    assert context[start:end] == "python spoiler"


def test_fuzzy_matching_fallback():
    context = "This is a sample article body containing python spoilers."
    ans = "python spoiler"
    span = find_answer_span_by_string_match(context, ans)
    assert span is not None
    assert "python spoiler" in context[span[0] : span[1]]


def test_enumeration_heuristics():
    post_txt = "7 Secrets We Finally Unveiled About AI"
    article_txt = (
        "We have major points:\n1. Neural nets\n2. Backpropagation\n3. Transformers"
    )

    assert detect_enumeration_question(post_txt) == 7
    assert detect_enumeration_question("Regular title here") is None

    items = find_enumerations_in_text(article_txt)
    assert len(items) == 3
    assert items[0].cardinal_number == 1
    assert items[0].text == "Neural nets"


def test_similarity_scorer():
    scorer = SpanSimilarityScorer(method="combined")
    score_high = scorer.score("Python framework", "The python frameworks!")
    score_low = scorer.score("Python framework", "Something entirely different")
    # Robust relative and absolute asserts
    assert score_high > 0.7
    assert score_low < 0.4
    assert score_high > score_low
