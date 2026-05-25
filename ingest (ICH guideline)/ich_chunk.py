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

ENCODING_MODEL = "text-embedding-3-small"
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
    max_tokens: int = 800,
) -> List[Dict[str, Any]]:
    """
    Convert ICH-normalized document into embedding-ready chunks.

    Now properly:
    - Tracks section heading
    - Extracts section_path + section_title
    - Attaches section metadata to ALL chunks
    """

    import re

    SECTION_EXTRACT_RE = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+(.*)")

    chunks: List[Dict[str, Any]] = []
    seen_texts = set()

    current_heading = None
    current_section_path = None
    current_section_title = None

    guideline_name = normalized_doc.get("document_name") or normalized_doc.get("doc_id")

    # We iterate over normalized blocks ONLY
    for block in normalized_doc.get("blocks", []):

        block_type = block.get("block_type")
        text = (
            block.get("content")
            or block.get("text")
            or block.get("flattened_text")
            or ""
        ).strip()

        if not text:
            continue

        # ────────────────────────────────
        # 1️⃣ If heading → update context
        # ────────────────────────────────
        if block_type == "heading":
            current_heading = text

            match = SECTION_EXTRACT_RE.match(text)
            if match:
                current_section_path = match.group(1)
                current_section_title = match.group(2).strip()
            else:
                current_section_path = None
                current_section_title = None

            continue

        # ────────────────────────────────
        # 2️⃣ For paragraphs / rules
        # ────────────────────────────────
        if not current_section_path:
            # Skip anything not under a numbered section
            continue

        metadata = {
            "guideline": guideline_name,
            "section_heading": current_heading,
            "section_path": current_section_path,
            "section_title": current_section_title,
            "block_type": block_type or "paragraph",
            "rule_type": block.get("rule_type"),
            "page_number": block.get("page_number"),
        }

        if token_len(text) > max_tokens:
            for part in hard_token_split(text, max_tokens):
                if part in seen_texts:
                    continue

                chunks.append({
                    "chunk_type": block_type or "paragraph",
                    "text": part,
                    "metadata": metadata
                })
                seen_texts.add(part)
        else:
            if text in seen_texts:
                continue

            chunks.append({
                "chunk_type": block_type or "paragraph",
                "text": text,
                "metadata": metadata
            })
            seen_texts.add(text)

    print(f"Chunked {len(chunks)} total items (section-aware)")

    return chunks