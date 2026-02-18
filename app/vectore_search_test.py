import os, sys, uuid
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import re
import json
from typing import List, Dict, Any

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from collections import defaultdict
from typing import List, Dict, Any

from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBED_MODEL,

    AZURE_SEARCH_SERVICE_ENDPOINT,
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_INDEX_NAME,          # SOURCE index
    AZURE_ICH_SEARCH_INDEX_NAME,      # ICH index
)

# ======================================================
# CLIENTS
# ======================================================

source_search_client = SearchClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
)

ich_search_client = SearchClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    index_name=AZURE_ICH_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
)

aoai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

VECTOR_FIELD = "vector"

# SOURCE index fields
SOURCE_SELECT_FIELDS = [
    "id",
    "doc_id",
    "text",
    "chunk_type",
    "heading_path",
    "page_numbers",
    "source_block_ids",

    # table-aware fields
    "table_context_heading",
    "table_context_text",
    "table_semantic_hint",
    "table_headers",
    "table_rows",
]


# ICH index fields (MATCHING YOUR SCHEMA)
ICH_SELECT_FIELDS = [
    "id",
    "doc_id",
    "source_type",
    "guideline",
    "block_type",
    "section_path",
    "section_title",
    "rule_type",
    "page_number",
    "text",
]

# ======================================================
# METADATA
# ======================================================

def get_authoring_metadata():
    return {
        "section": "Summary of Subject Demographics Safety Population - RP Patients in tabular",
        "synonyms": [
            "Table 14.1.3.1.1",
            "Subgroup: Mutation Subtype - Biallelic autosomal recessive NR2E3",
            "Subgroup: Mutation Subtype - Autosomal dominant NR2E3",
            "Subgroup: Mutation Subtype - Autosomal dominant RHO",
        ],
        "ich_refs": ["14.1 DEMOGRAPHIC DATA"],
        "allowed_sources": ["OCU401_CSR_Final_Tables.PDF"],
    }

# ======================================================
# EMBEDDING
# ======================================================

def embed_query(text: str) -> List[float]:
    resp = aoai_client.embeddings.create(
        model=AZURE_OPENAI_EMBED_MODEL,
        input=text,
    )
    return resp.data[0].embedding

# ======================================================
# VECTOR SEARCH – SOURCE (with min score filtering)
# ======================================================

def retrieve_source_chunks(
    metadata: Dict[str, Any],
    min_score: float = 0.60,
    k_nearest_neighbors: int = 100
) -> List[Dict[str, Any]]:
    """
    Retrieves clean source chunks with:
    - Removed unwanted Azure metadata fields
    - Cleaned table_rows (list of lists)
    - Only useful fields kept
    - Consistent field order (especially nice for tables)
    """
    section = metadata["section"]
    synonyms = metadata["synonyms"]
    allowed_sources = metadata["allowed_sources"]

    if not allowed_sources:
        return []

    doc_filter = " or ".join([f"doc_id eq '{doc}'" for doc in allowed_sources])

    queries = [section] + [s for s in synonyms if s != section]
    queries = [q.strip() for q in queries if q and q.strip()]

    if not queries:
        return []

    # Fields we actually want to keep
    WANTED_FIELDS = {
        "chunk_type", "text", "table_rows", "table_caption",
        "heading", "parent_heading", "section_title",
        "doc_id", "source_page", "@search.score",
        # Add any other fields you still want to keep
    }

    results = []

    for q in queries:
        try:
            vector = embed_query(q)
            vq = VectorizedQuery(vector=vector, fields=VECTOR_FIELD)
            vq.k = k_nearest_neighbors

            res = source_search_client.search(
                search_text=None,
                vector_queries=[vq],
                filter=doc_filter,
                select=SOURCE_SELECT_FIELDS,   # assuming this already contains what you need
                top=k_nearest_neighbors
            )

            for r in res:
                score = r.get("@search.score", 0)
                if score < min_score:
                    continue

                # Create clean dict with only wanted fields
                clean_chunk = {}

                # ─── Put table-related fields first when present ───
                if r.get("chunk_type") == "table":
                    if "table_caption" in r:
                        clean_chunk["table_caption"] = r["table_caption"]
                    if "heading" in r:
                        clean_chunk["heading"] = r["heading"]
                    if "table_rows" in r and isinstance(r["table_rows"], str):
                        clean_rows = _parse_table_rows(r["table_rows"])
                        if clean_rows:
                            clean_chunk["table_rows"] = clean_rows

                # Then other text/heading content
                for field in ["text", "heading", "parent_heading", "section_title"]:
                    if field in r and r[field]:
                        clean_chunk[field] = r[field]

                # Always include these at the end
                clean_chunk["@search.score"] = round(score, 4)   # nicer number
                clean_chunk["doc_id"] = r.get("doc_id")
                if "source_page" in r:
                    clean_chunk["source_page"] = r["source_page"]

                # Optional: keep chunk_type so we know what we're dealing with
                clean_chunk["chunk_type"] = r.get("chunk_type", "text")

                results.append(clean_chunk)

            print(f"Query '{q[:60]}...' → kept {len([r for r in res if r.get('@search.score', 0) >= min_score])} chunks (≥ {min_score})")

        except Exception as e:
            print(f"Search failed for query '{q}': {e}")
            continue

    # Deduplication by source_block_ids or id
    seen = set()
    deduplicated = []

    for chunk in results:
        # Use source_block_ids if available, otherwise fallback to id or doc_id+score
        key = chunk.get("source_block_ids", [None])[0] if "source_block_ids" in chunk else None
        if not key:
            key = f"{chunk.get('doc_id','')}_{chunk.get('@search.score',0):.4f}"
        if key not in seen:
            seen.add(key)
            deduplicated.append(chunk)

    print(f"Final clean chunks after deduplication: {len(deduplicated)}")

    # Optional: sort by score descending (best matches first)
    deduplicated.sort(key=lambda x: x.get("@search.score", 0), reverse=True)

    return deduplicated


def _parse_table_rows(rows_str: str) -> List[List[str]]:
    """Improved table rows string → list of lists parser"""
    if not rows_str or not isinstance(rows_str, str):
        return []

    try:
        rows = []
        for line in rows_str.split("\n"):
            line = line.strip()
            if not line or line == "[]":
                continue

            # Remove outer brackets if present
            if line.startswith("[") and line.endswith("]"):
                line = line[1:-1].strip()

            cells = []
            current = ""
            in_quotes = False
            i = 0

            while i < len(line):
                c = line[i]
                if c == '"' and (i == 0 or line[i-1] != '\\'):
                    in_quotes = not in_quotes
                    i += 1
                    continue
                if c == ',' and not in_quotes:
                    cells.append(current.strip().strip('"'))
                    current = ""
                    i += 1
                    continue
                current += c
                i += 1

            if current.strip():
                cells.append(current.strip().strip('"'))

            if cells:
                rows.append(cells)

        return rows

    except Exception as e:
        print(f"Table parse failed: {e}")
        return []

# ======================================================
# VECTOR SEARCH – ICH (with min score filtering)
# ======================================================

def retrieve_ich_chunks(
    metadata: Dict[str, Any],
    min_score: float = 0.60,              # 60% minimum similarity
    k_nearest_neighbors: int = 5
) -> List[Dict[str, Any]]:
    """
    Vector search on ICH index with minimum score filtering (≥ 60%).
    """
    section = metadata["section"]
    ich_refs = metadata.get("ich_refs", [])

    queries = [section] + ich_refs
    queries = [q.strip() for q in queries if q and q.strip()]

    if not queries:
        return []

    results = []

    for q in queries:
        try:
            vector = embed_query(q)
            vq = VectorizedQuery(vector=vector, fields=VECTOR_FIELD)
            vq.k = k_nearest_neighbors

            res = ich_search_client.search(
                search_text=None,
                vector_queries=[vq],
                select=ICH_SELECT_FIELDS,
                top=k_nearest_neighbors
            )

            # Filter by minimum score
            good_hits = [
                dict(r) for r in res
                if r.get("@search.score", 0) >= min_score
            ]

            print(f"ICH query '{q[:60]}...' returned {len(list(res))} hits → kept {len(good_hits)} (≥ {min_score})")

            results.extend(good_hits)

        except Exception as e:
            print(f"ICH vector search failed for query '{q}': {e}")
            continue

    # Deduplicate by id
    dedup = {r["id"]: r for r in results}
    print(f"Final ICH chunks after deduplication & score filtering: {len(dedup)}")
    return list(dedup.values())

# ======================================================
# GROUPING (unchanged)
# ======================================================

def group_ich_by_section(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)

    for c in chunks:
        section = c.get("section_path") or "UNKNOWN"
        grouped[section].append(c)

    assembled_sections = []

    for section_path, items in grouped.items():
        assembled_sections.append({
            "section_path": section_path,
            "guideline": items[0].get("guideline"),
            "rule_type": list({i.get("rule_type") for i in items}),
            "source_type": "ich",
            "clauses": [i["text"] for i in items if i.get("text")],
            "ids": [i["id"] for i in items],
        })

    return assembled_sections

# ======================================================
# SIMPLE CLASSIFICATION (FOR SOURCE ONLY)
# ======================================================

TABLE_RE = re.compile(r"\bTable\s+\d+(\.\d+)*", re.IGNORECASE)

def classify_source_chunk(chunk):
    if chunk.get("chunk_type"):
        return chunk

    # fallback ONLY if missing
    text = (chunk.get("text") or "").strip()
    if TABLE_RE.search(text):
        chunk["chunk_type"] = "TABLE"
    else:
        chunk["chunk_type"] = "PARAGRAPH"
    return chunk


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    metadata = get_authoring_metadata()

    source_chunks = retrieve_source_chunks(
        metadata,
        min_score=0.60,          # 60% threshold for source
        k_nearest_neighbors=50
    )

    raw_ich_chunks = retrieve_ich_chunks(
        metadata,
        min_score=0.60,          # 60% threshold for ICH
        k_nearest_neighbors=5
    )

    ich_sections = group_ich_by_section(raw_ich_chunks)

    enriched_source = [classify_source_chunk(c) for c in source_chunks]

    payload = {
        "section": metadata["section"],
        "source": {
            "total_chunks": len(enriched_source),
            "results": enriched_source,
        },
        "ich_guidelines": {
            "total_sections": len(ich_sections),
            "results": ich_sections,
        }
    }

    with open("vector_search_test_source_ich.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("✅ Vector search completed")
    print(f"📄 SOURCE chunks (≥60%): {len(enriched_source)}")
    print(f"📘 ICH chunks (≥60%): {len(ich_sections)}")