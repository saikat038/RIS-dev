"""
Chunking utilities:
- Split raw document text into retrieval-friendly chunks.
- Keeps paragraphs together and enforces strict token limits.
"""

from __future__ import annotations
from typing import List, Dict
import tiktoken

# ----------------------------
# Tokenizer setup
# ----------------------------

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


# ----------------------------
# Schema-agnostic text resolver
# ----------------------------

def resolve_block_text(block: Dict) -> str:
    """
    Safely extract text from a normalized layout block.
    Works across schema versions and never raises KeyError.
    """

    if "flattened_text" in block and block["flattened_text"]:
        return block["flattened_text"]

    if "text" in block and block["text"]:
        return block["text"]

    if "content" in block and block["content"]:
        return block["content"]

    if "lines" in block and isinstance(block["lines"], list):
        return " ".join(
            line.get("text", "") for line in block["lines"]
        ).strip()

    return ""


# ----------------------------
# Main chunking logic
# ----------------------------

def chunk_semantic_blocks(
    normalized_doc: Dict,
    max_tokens: int = 700,
) -> List[Dict]:
    """
    Chunk semantic normalized blocks into embedding-ready chunks.
    """

    chunks: List[Dict] = []

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
            # Final safety split
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

    # ----------------------------
    # Iterate normalized blocks
    # ----------------------------

    for block in normalized_doc.get("blocks", []):
        block_type = block.get("block_type")

        # ---- TABLES: always standalone ----
        if block_type == "table":
            table_text = resolve_block_text(block)
            if not table_text:
                continue

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

        # ---- PARAGRAPHS ONLY ----
        if block_type != "paragraph":
            continue

        block_text = resolve_block_text(block).strip()
        if not block_text:
            continue

        block_tokens = token_len(block_text)

        # New section → flush
        if cur_meta["heading_path"] != block.get("heading_path"):
            flush_chunk()
            cur_text_parts = []
            cur_tokens = 0
            cur_meta = {
                "heading_path": block.get("heading_path"),
                "page_numbers": set(),
                "source_block_ids": []
            }

        # Oversized paragraph → standalone split
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

        # Would exceed limit → flush
        if cur_text_parts and (cur_tokens + block_tokens) > max_tokens:
            flush_chunk()
            cur_text_parts = []
            cur_tokens = 0
            cur_meta = {
                "heading_path": block.get("heading_path"),
                "page_numbers": set(),
                "source_block_ids": []
            }

        cur_text_parts.append(block_text)
        cur_tokens += block_tokens
        cur_meta["page_numbers"].add(block.get("page_number"))
        cur_meta["source_block_ids"].extend(block.get("source_block_ids", []))

    flush_chunk()

    return chunks