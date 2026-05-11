"""Feature extraction for prompt complexity classification.

Features are intentionally cheap: no embeddings, no model calls. They run
in microseconds per prompt and produce a fixed-shape numeric vector.
"""

from __future__ import annotations

import re
from typing import List

# Programming-ish tokens that suggest the prompt is about code.
_CODE_TOKENS = (
    "def ",
    "class ",
    "function ",
    "return ",
    "import ",
    "const ",
    "var ",
    "public ",
    "private ",
    "if ",
    "else ",
    "for ",
    "while ",
    "lambda ",
    "async ",
    "await ",
    "=>",
    "println",
    "console.log",
    "print(",
)

# Words that suggest substantive reasoning / multi-step work.
_REASONING_TOKENS = (
    "analyze",
    "design",
    "compare",
    "trade-off",
    "tradeoff",
    "explain why",
    "prove",
    "derive",
    "outline",
    "step by step",
    "step-by-step",
    "architecture",
    "strategy",
    "implications",
    "implement",
    "refactor",
    "evaluate",
)

# Explicit feature order. The classifier depends on this being stable.
FEATURE_NAMES: List[str] = [
    "char_len",
    "word_count",
    "sentence_count",
    "question_count",
    "has_code_block",
    "has_inline_code",
    "code_token_count",
    "reasoning_token_count",
    "has_url",
    "has_math",
    "list_item_count",
    "avg_word_len",
    "uppercase_word_count",
]


def extract_features(prompt: str) -> List[float]:
    """Return a fixed-order feature vector for a prompt.

    The vector aligns with FEATURE_NAMES. All values are floats so the
    classifier sees a consistent dtype.
    """
    if prompt is None:
        prompt = ""
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")

    text = prompt
    lower = text.lower()
    char_len = float(len(text))
    words = re.findall(r"\b[\w']+\b", text)
    word_count = float(len(words))
    sentences = re.split(r"[.!?]+\s+", text.strip())
    sentence_count = float(sum(1 for s in sentences if s.strip()))
    question_count = float(text.count("?"))

    has_code_block = float(1 if "```" in text else 0)
    # Inline code: one or more backticks with text between them.
    has_inline_code = float(1 if re.search(r"`[^`\n]+`", text) else 0)
    code_token_count = float(sum(lower.count(tok) for tok in _CODE_TOKENS))
    reasoning_token_count = float(sum(lower.count(tok) for tok in _REASONING_TOKENS))
    has_url = float(1 if re.search(r"https?://", text) else 0)
    has_math = float(
        1
        if re.search(r"\$[^$]+\$", text)
        or re.search(r"\\\(", text)
        or re.search(r"\\frac\{", text)
        else 0
    )
    list_item_count = float(len(re.findall(r"(?:^|\n)\s*[-*]\s+", text)))
    avg_word_len = float(sum(len(w) for w in words) / len(words)) if words else 0.0
    uppercase_word_count = float(sum(1 for w in words if w.isupper() and len(w) > 1))

    return [
        char_len,
        word_count,
        sentence_count,
        question_count,
        has_code_block,
        has_inline_code,
        code_token_count,
        reasoning_token_count,
        has_url,
        has_math,
        list_item_count,
        avg_word_len,
        uppercase_word_count,
    ]
