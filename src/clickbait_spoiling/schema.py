from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ClickbaitPost:
    uuid: str
    post_text: List[str]
    target_paragraphs: List[str]
    target_title: str
    target_url: Optional[str] = None
    human_spoiler: Optional[List[str]] = None
    spoiler: Optional[List[str]] = None
    spoiler_positions: Optional[List] = None
    tags: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, raw: dict) -> "ClickbaitPost":
        """Parse raw dictionary from JSONL file into a structured ClickbaitPost."""
        return cls(
            uuid=raw["uuid"],
            post_text=raw["postText"],
            target_paragraphs=raw["targetParagraphs"],
            target_title=raw["targetTitle"],
            target_url=raw.get("targetUrl"),
            human_spoiler=raw.get("humanSpoiler"),
            spoiler=raw.get("spoiler"),
            spoiler_positions=raw.get("spoilerPositions"),
            tags=raw.get("tags"),
        )

    def joined_post_text(self) -> str:
        """Join postText list into a single question string."""
        return " ".join(self.post_text)

    def joined_context(self, sep: str = "\n") -> Tuple[str, List[int]]:
        """Join targetParagraphs into one context string.
        Returns:
            context (str): The merged article body.
            paragraph_offsets (list): The starting character indices of each paragraph in context.
        """
        paragraph_offsets = []
        current_offset = 0
        joined_paragraphs = []

        for p in self.target_paragraphs:
            paragraph_offsets.append(current_offset)
            joined_paragraphs.append(p)
            current_offset += len(p) + len(sep)

        context = sep.join(joined_paragraphs)
        return context, paragraph_offsets

    def gold_tag(self) -> Optional[str]:
        """Return the first tag if present, representing the spoiler type (phrase/passage/multi)."""
        if self.tags and len(self.tags) > 0:
            return self.tags[0]
        return None
