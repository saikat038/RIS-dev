"""
Chunking utilities:
- Split semantic normalized document blocks into retrieval-friendly chunks
- Paragraph-aware
- Table-aware
- Strict token enforcement
"""

from __future__ import annotations
from typing import List, Dict, Any
import tiktoken

# -------------------------------------------------
# Tokenizer setup (MUST match embedding model)
# -------------------------------------------------

ENCODING_MODEL = "text-embedding-3-large"
enc = tiktoken.encoding_for_model(ENCODING_MODEL)


def token_len(text: str) -> int:
    return len(enc.encode(text))


def hard_token_split(text: str, max_tokens: int) -> List[str]:
    tokens = enc.encode(text)
    return [
        enc.decode(tokens[i: i + max_tokens])
        for i in range(0, len(tokens), max_tokens)
    ]


# -------------------------------------------------
# Block text resolvers (schema-tolerant)
# -------------------------------------------------

def flatten_table(block: Dict[str, Any]) -> str:
    """
    Convert semantic table block into embedding-friendly plain text.
    """
    headers = block.get("headers", [])
    rows = block.get("rows", [])

    lines = []

    if headers:
        lines.append(" | ".join(h.strip() for h in headers if h.strip()))
        lines.append("-" * 40)

    for row in rows:
        if isinstance(row, list):
            lines.append(" | ".join(str(cell).strip() for cell in row))
        elif isinstance(row, dict):
            lines.append(" | ".join(str(v).strip() for v in row.values()))

    return "\n".join(lines).strip()


def resolve_block_text(block: Dict[str, Any]) -> str:
    """
    Safely extract text from any semantic block.
    """

    # --- TABLES ---
    if block.get("block_type") == "table":
        return flatten_table(block)

    # --- PARAGRAPHS ---
    if block.get("flattened_text"):
        return block["flattened_text"]

    if block.get("text"):
        return block["text"]

    if block.get("content"):
        return block["content"]

    if "lines" in block and isinstance(block["lines"], list):
        return " ".join(
            line.get("text", "") for line in block["lines"]
        ).strip()

    return ""


# -------------------------------------------------
# Main ICH chunker
# -------------------------------------------------
# -------------------------------------------------
# ICH chunker (RULE-ATOMIC, NO MERGING)
# -------------------------------------------------

def chunk_ich_units(
    normalized_doc: Dict[str, Any],
    max_tokens: int = 300,
) -> List[Dict[str, Any]]:
    """
    Convert ICH-normalized rule units into embedding-ready chunks.

    Rules:
    - 1 normalized rule = 1 chunk
    - NEVER merge rules
    - Hard-split only if a single rule exceeds max_tokens
    """

    chunks: List[Dict[str, Any]] = []

    for block in normalized_doc.get("blocks", []):
        # ICH normalizer should only emit rule-like blocks
        block_text = (
            block.get("content")
            or block.get("text")
            or block.get("flattened_text")
            or ""
        ).strip()

        if not block_text:
            continue

        metadata = {
            "guideline": block.get("guideline"),
            "section_path": block.get("section_path"),
            "rule_type": block.get("rule_type"),
            "page_numbers": [block.get("page_number")] if block.get("page_number") else [],
        }

        # Oversized single rule → hard split (rare but safe)
        if token_len(block_text) > max_tokens:
            for part in hard_token_split(block_text, max_tokens):
                chunks.append({
                    "chunk_type": "ich_rule",
                    "text": part,
                    "metadata": metadata
                })
        else:
            chunks.append({
                "chunk_type": "ich_rule",
                "text": block_text,
                "metadata": metadata
            })

    return chunks