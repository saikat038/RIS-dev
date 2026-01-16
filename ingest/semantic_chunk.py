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
# Main semantic chunker
# -------------------------------------------------

def chunk_semantic_blocks(
    normalized_doc: Dict[str, Any],
    max_tokens: int = 700,
) -> List[Dict[str, Any]]:
    """
    Convert semantic normalized blocks into embedding-ready chunks.
    """

    chunks: List[Dict[str, Any]] = []

    cur_text_parts: List[str] = []
    cur_tokens = 0
    cur_meta = {
        "heading_path": None,
        "page_numbers": set(),
        "source_block_ids": []
    }

    def flush_chunk():
        nonlocal cur_text_parts, cur_tokens, cur_meta

        if not cur_text_parts:
            return

        text = "\n\n".join(cur_text_parts).strip()
        if not text:
            return

        if token_len(text) <= max_tokens:
            chunks.append({
                "chunk_type": "paragraph",
                "text": text,
                "metadata": {
                    "heading_path": cur_meta["heading_path"],
                    "page_numbers": sorted(cur_meta["page_numbers"]),
                    "source_block_ids": cur_meta["source_block_ids"]
                }
            })
        else:
            for part in hard_token_split(text, max_tokens):
                chunks.append({
                    "chunk_type": "paragraph",
                    "text": part,
                    "metadata": {
                        "heading_path": cur_meta["heading_path"],
                        "page_numbers": sorted(cur_meta["page_numbers"]),
                        "source_block_ids": cur_meta["source_block_ids"]
                    }
                })

        cur_text_parts = []
        cur_tokens = 0
        cur_meta = {
            "heading_path": None,
            "page_numbers": set(),
            "source_block_ids": []
        }

    # -------------------------------------------------
    # Iterate semantic blocks
    # -------------------------------------------------

    for block in normalized_doc.get("blocks", []):
        block_type = block.get("block_type")

        # -------------------------------------------------
        # TABLES → ALWAYS standalone chunks
        # -------------------------------------------------
        if block_type == "table":
            flush_chunk()

            table_text = resolve_block_text(block)
            if not table_text:
                continue

            # Hard-split large tables
            if token_len(table_text) > max_tokens:
                for part in hard_token_split(table_text, max_tokens):
                    chunks.append({
                        "chunk_type": "table",
                        "text": part,
                        "metadata": {
                            "heading_path": block.get("heading_path"),
                            "page_numbers": [block.get("page_number")],
                            "source_block_ids": block.get("source_block_ids", [])
                        }
                    })
            else:
                chunks.append({
                    "chunk_type": "table",
                    "text": table_text,
                    "metadata": {
                        "heading_path": block.get("heading_path"),
                        "page_numbers": [block.get("page_number")],
                        "source_block_ids": block.get("source_block_ids", [])
                    }
                })

            continue

        # -------------------------------------------------
        # PARAGRAPHS ONLY
        # -------------------------------------------------
        if block_type != "paragraph":
            continue

        block_text = resolve_block_text(block).strip()
        if not block_text:
            continue

        block_tokens = token_len(block_text)

        # New heading → flush
        if cur_meta["heading_path"] != block.get("heading_path"):
            flush_chunk()
            cur_meta["heading_path"] = block.get("heading_path")

        # Oversized paragraph → standalone
        if block_tokens > max_tokens:
            flush_chunk()
            for part in hard_token_split(block_text, max_tokens):
                chunks.append({
                    "chunk_type": "paragraph",
                    "text": part,
                    "metadata": {
                        "heading_path": block.get("heading_path"),
                        "page_numbers": [block.get("page_number")],
                        "source_block_ids": block.get("source_block_ids", [])
                    }
                })
            continue

        # Would exceed → flush
        if cur_text_parts and (cur_tokens + block_tokens) > max_tokens:
            flush_chunk()
            cur_meta["heading_path"] = block.get("heading_path")

        cur_text_parts.append(block_text)
        cur_tokens += block_tokens
        cur_meta["page_numbers"].add(block.get("page_number"))
        cur_meta["source_block_ids"].extend(block.get("source_block_ids", []))

    flush_chunk()
    return chunks