"""
Chunking utilities:
- Split raw document text into retrieval-friendly chunks.
- Keeps paragraphs together and aims for ~400–800 tokens per chunk (roughly).
"""

from __future__ import annotations
import re
from typing import List

# Very rough heuristic: ~4 chars ≈ 1 token for English text.
CHARS_PER_TOKEN = 4


def _normalize(text: str) -> str:
    """Basic cleanup: unify newlines, strip trailing spaces, collapse excessive blanks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Trim spaces at line ends
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Collapse >2 blank lines to exactly 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(
    text: str,
    max_tokens: int = 700,
    min_tokens: int = 200,
) -> List[str]:
    """
    Split text into paragraph-aware chunks that target max_tokens (approx).
    We avoid splitting tables/code blocks mid-block by keeping fenced blocks intact.

    Args:
        text: full document text
        max_tokens: approximate soft cap per chunk
        min_tokens: if the last chunk is tiny, try to merge with previous

    Returns:
        List of chunk strings.
    """
    text = _normalize(text)

    # Keep fenced blocks (``` ... ```) intact by splitting around them first.
    # We tag them and treat each block as an atomic paragraph.
    blocks = []
    code_pat = re.compile(r"(```.*?```)", re.DOTALL)
    pos = 0
    for m in code_pat.finditer(text):
        if m.start() > pos:
            blocks.append(("text", text[pos:m.start()]))
        blocks.append(("fence", m.group(1)))
        pos = m.end()
    if pos < len(text):
        blocks.append(("text", text[pos:]))

    # From plain text blocks, split into paragraphs by blank lines.
    paragraphs: List[str] = []
    for kind, content in blocks:
        if kind == "fence":
            paragraphs.append(content.strip())
        else:
            paras = re.split(r"\n\s*\n", content.strip())
            paragraphs.extend([p.strip() for p in paras if p.strip()])

    # Now pack paragraphs into chunks by token estimate.
    chunks: List[str] = []
    cur: List[str] = []
    cur_chars = 0
    max_chars = max_tokens * CHARS_PER_TOKEN
    min_chars = min_tokens * CHARS_PER_TOKEN

    for p in paragraphs:
        p_chars = len(p)
        # If adding this paragraph would exceed max, flush the current chunk.
        if cur and (cur_chars + p_chars + 2) > max_chars:
            chunks.append("\n\n".join(cur).strip())
            cur, cur_chars = [], 0
        cur.append(p)
        cur_chars += p_chars + 2  # account for the blank line joiner

    # Flush remainder
    if cur:
        if chunks and (len(cur) * CHARS_PER_TOKEN) < min_chars:
            # Merge a too-small tail into the previous chunk
            chunks[-1] = (chunks[-1] + "\n\n" + "\n\n".join(cur)).strip()
        else:
            chunks.append("\n\n".join(cur).strip())

    # Final safety: drop any accidental empties
    chunks = [c for c in chunks if c.strip()]
    return chunks
