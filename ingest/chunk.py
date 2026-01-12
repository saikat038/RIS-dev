"""
Chunking utilities:
- Split raw document text into retrieval-friendly chunks.
- Keeps paragraphs together and enforces strict token limits.
"""

from __future__ import annotations
import re
from typing import List
import tiktoken

# Choose the SAME model you use for embeddings
ENCODING_MODEL = "text-embedding-3-large"
enc = tiktoken.encoding_for_model(ENCODING_MODEL)


def token_len(text: str) -> int:
    """Return real token count for the given text."""
    return len(enc.encode(text))


def hard_token_split(text: str, max_tokens: int) -> List[str]:
    """
    Force-split text by tokens if it exceeds max_tokens.
    This is the final safety net.
    """
    tokens = enc.encode(text)
    return [
        enc.decode(tokens[i : i + max_tokens])
        for i in range(0, len(tokens), max_tokens)
    ]


def _normalize(text: str) -> str:
    """Basic cleanup: unify newlines, strip trailing spaces, collapse excessive blanks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(
    text: str,
    max_tokens: int = 700,
    min_tokens: int = 200,
) -> List[str]:
    """
    Split text into paragraph-aware chunks with strict token enforcement.

    Args:
        text: full document text
        max_tokens: hard cap per chunk (real tokens)
        min_tokens: if the last chunk is tiny, try to merge safely

    Returns:
        List of chunk strings guaranteed <= max_tokens
    """
    text = _normalize(text)

    # --- Step 1: Keep fenced blocks intact ---
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

    # --- Step 2: Split into paragraphs ---
    paragraphs: List[str] = []
    for kind, content in blocks:
        if kind == "fence":
            paragraphs.append(content.strip())
        else:
            paras = re.split(r"\n\s*\n", content.strip())
            paragraphs.extend([p.strip() for p in paras if p.strip()])

    # --- Step 3: Pack paragraphs into chunks ---
    chunks: List[str] = []
    cur: List[str] = []
    cur_tokens = 0

    for p in paragraphs:
        p_tokens = token_len(p)

        # If a single paragraph is too large, split it directly
        if p_tokens > max_tokens:
            if cur:
                chunks.append("\n\n".join(cur))
                cur, cur_tokens = [], 0
            chunks.extend(hard_token_split(p, max_tokens))
            continue

        # If adding paragraph exceeds limit, flush current chunk
        if cur and (cur_tokens + p_tokens) > max_tokens:
            chunks.append("\n\n".join(cur))
            cur, cur_tokens = [], 0

        cur.append(p)
        cur_tokens += p_tokens

    # --- Step 4: Handle remainder safely ---
    if cur:
        tail_text = "\n\n".join(cur)
        tail_tokens = token_len(tail_text)

        if chunks and tail_tokens < min_tokens:
            merged = chunks[-1] + "\n\n" + tail_text
            if token_len(merged) <= max_tokens:
                chunks[-1] = merged
            else:
                chunks.append(tail_text)
        else:
            chunks.append(tail_text)

    # --- Step 5: Final safety pass ---
    final_chunks: List[str] = []
    for c in chunks:
        if token_len(c) <= max_tokens:
            final_chunks.append(c)
        else:
            final_chunks.extend(hard_token_split(c, max_tokens))

    return [c for c in final_chunks if c.strip()]