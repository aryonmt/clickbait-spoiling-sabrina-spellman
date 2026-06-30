def prepare_train_features(
    examples: dict, tokenizer, max_length: int, doc_stride: int
) -> dict:
    """Tokenize question+context with sliding window stride and calculate start/end token positions."""
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if not answers or len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
            continue

        # Cycle through multiple answers across overflowing windows
        ans_idx = i % len(answers["answer_start"])
        start_char = answers["answer_start"][ans_idx]
        end_char = start_char + len(answers["text"][ans_idx])

        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        token_start_index = 0
        while sequence_ids[token_start_index] != context_index:
            token_start_index += 1

        token_end_index = len(tokenized_examples["input_ids"][i]) - 1
        while sequence_ids[token_end_index] != context_index:
            token_end_index -= 1

        # Check if the answer is completely outside of this feature span
        if not (
            offsets[token_start_index][0] <= start_char
            and offsets[token_end_index][1] >= end_char
        ):
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
        else:
            # Map start character index to start token index
            curr_idx = token_start_index
            while curr_idx <= token_end_index and offsets[curr_idx][0] <= start_char:
                curr_idx += 1
            tokenized_examples["start_positions"].append(curr_idx - 1)

            # Map end character index to end token index
            curr_idx = token_end_index
            while curr_idx >= token_start_index and offsets[curr_idx][1] >= end_char:
                curr_idx -= 1
            tokenized_examples["end_positions"].append(curr_idx + 1)

    return tokenized_examples


def prepare_validation_features(
    examples: dict, tokenizer, max_length: int, doc_stride: int
) -> dict:
    """Same tokenization but keeps offset_mappings and example_ids for inference reconstruction."""
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["uuid"][sample_index])

        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # Mask non-context offsets to None so we don't extract tokens outside context
        offsets = tokenized_examples["offset_mapping"][i]
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(offsets)
        ]

    return tokenized_examples
