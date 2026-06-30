SPOILER_TYPES = ["phrase", "passage", "multi"]
LABEL2ID = {label: idx for idx, label in enumerate(SPOILER_TYPES)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

MAX_PHRASE_WORDS = 5  # phrase spoilers: <= 5 words
MIN_PASSAGE_WORDS = 6  # passage spoilers: >= 6 words
MAX_MULTI_SPOILERS = 5  # multi-spoiler generation hard cap
MAX_MULTI_ITERATIONS = 7  # iterative generator stop condition
DEFAULT_CYT_FACTOR = 0.9  # "Cover Your Tracks" factor for enumeration augmentation
DEFAULT_SIMILARITY_THRESHOLD = 0.7

RANDOM_SEED = 42
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
MAX_ANSWER_LENGTH = 64
N_BEST_SIZE = 20
