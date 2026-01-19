import re
import uuid
from typing import Dict, List, Any, Tuple

# ----------------------------
# CONFIG
# ----------------------------

SECTION_REGEX = re.compile(r"^\d+(\.\d+)*\s+.+$")
APPENDIX_REGEX = re.compile(r"^APPENDIX\s+[A-Z0-9]+", re.IGNORECASE)
FIGURE_REGEX = re.compile(r"^FIGURE\s+\d+", re.IGNORECASE)

Y_ROW_THRESHOLD = 0.04   # inches
MIN_TABLE_COLUMNS = 3

# ----------------------------
# GEOMETRY HELPERS
# ----------------------------

def get_bbox(block: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return block.get("bbox", (0, 0, 0, 0))


def y_overlap(b1, b2) -> bool:
    _, y1_min, _, y1_max = b1
    _, y2_min, _, y2_max = b2
    return not (y1_max < y2_min - Y_ROW_THRESHOLD or y2_max < y1_min - Y_ROW_THRESHOLD)

# ----------------------------
# TABLE INFERENCE (LAST RESORT)
# ----------------------------

def cluster_rows(blocks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    rows: List[List[Dict[str, Any]]] = []

    for block in sorted(blocks, key=lambda b: get_bbox(b)[1]):
        placed = False
        for row in rows:
            if y_overlap(get_bbox(row[0]), get_bbox(block)):
                row.append(block)
                placed = True
                break
        if not placed:
            rows.append([block])

    return rows


def infer_table_from_paragraphs(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = cluster_rows(blocks)

    table_rows = []
    source_ids = []

    for row in rows:
        row = sorted(row, key=lambda b: get_bbox(b)[0])
        table_rows.append([b["text"] for b in row])
        source_ids.extend(b.get("block_id") for b in row if "block_id" in b)

    headers = table_rows[0] if table_rows else []
    data_rows = table_rows[1:] if len(table_rows) > 1 else []

    return {
        "headers": headers,
        "rows": data_rows,
        "source_block_ids": source_ids
    }

# ----------------------------
# MAIN NORMALIZER
# ----------------------------

def normalize_layout_json(layout_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts layout JSON into semantic blocks with
    GENERIC container anchoring (section / appendix / figure / table).
    """

    normalized = {
        "doc_id": layout_json.get("document_name"),
        "blocks": []
    }

    # Active semantic container
    current_container = {
        "container_id": None,
        "container_type": None,
        "path": []
    }

    paragraph_buffer: List[str] = []
    buffer_ids: List[str] = []
    buffer_page = None

    def flush_paragraph():
        nonlocal paragraph_buffer, buffer_ids, buffer_page
        if not paragraph_buffer:
            return

        normalized["blocks"].append({
            "block_type": "paragraph",
            "text": " ".join(paragraph_buffer),
            "page_number": buffer_page,
            "container_id": current_container["container_id"],
            "container_type": current_container["container_type"],
            "container_path": current_container["path"].copy(),
            "source_block_ids": buffer_ids.copy()
        })

        paragraph_buffer.clear()
        buffer_ids.clear()

    # ----------------------------
    # PROCESS DOCUMENT
    # ----------------------------

    for page in layout_json["pages"]:
        page_number = page["page_number"]
        blocks = page["blocks"]
        i = 0

        while i < len(blocks):
            block = blocks[i]

            # =====================================================
            # 1️⃣ EXPLICIT TABLE (AUTHORITATIVE)
            # =====================================================
            if block["block_type"] == "table":
                flush_paragraph()

                table_id = str(uuid.uuid4())

                normalized["blocks"].append({
                    "block_type": "table",
                    "caption": block.get("caption"),
                    "headers": block.get("headers", []),
                    "rows": block.get("rows", []),
                    "page_number": page_number,
                    "container_id": table_id,
                    "container_type": "table_group",
                    "container_path": current_container["path"].copy(),
                    "source_block_ids": []
                })

                i += 1
                continue

            # =====================================================
            # 2️⃣ PARAGRAPH / HEADING
            # =====================================================
            if block["block_type"] == "paragraph":
                text = block["text"].strip()

                # -------- SECTION --------
                if SECTION_REGEX.match(text):
                    flush_paragraph()

                    cid = str(uuid.uuid4())
                    current_container = {
                        "container_id": cid,
                        "container_type": "section",
                        "path": [text]
                    }

                    normalized["blocks"].append({
                        "block_type": "heading",
                        "level": 1,
                        "text": text,
                        "page_number": page_number,
                        "container_id": cid,
                        "container_type": "section",
                        "container_path": current_container["path"].copy()
                    })
                    i += 1
                    continue

                # -------- APPENDIX --------
                if APPENDIX_REGEX.match(text):
                    flush_paragraph()

                    cid = str(uuid.uuid4())
                    current_container = {
                        "container_id": cid,
                        "container_type": "appendix",
                        "path": [text]
                    }

                    normalized["blocks"].append({
                        "block_type": "heading",
                        "level": 1,
                        "text": text,
                        "page_number": page_number,
                        "container_id": cid,
                        "container_type": "appendix",
                        "container_path": current_container["path"].copy()
                    })
                    i += 1
                    continue

                # -------- FIGURE --------
                if FIGURE_REGEX.match(text):
                    flush_paragraph()

                    cid = str(uuid.uuid4())
                    current_container = {
                        "container_id": cid,
                        "container_type": "figure_group",
                        "path": current_container["path"] + [text]
                    }

                    normalized["blocks"].append({
                        "block_type": "heading",
                        "level": 2,
                        "text": text,
                        "page_number": page_number,
                        "container_id": cid,
                        "container_type": "figure_group",
                        "container_path": current_container["path"].copy()
                    })
                    i += 1
                    continue

                # -------- TABLE INFERENCE (ONLY IF NO EXPLICIT TABLE) --------
                aligned = [block]
                bbox = get_bbox(block)
                j = i + 1

                while j < len(blocks):
                    nb = blocks[j]
                    if nb["block_type"] == "paragraph" and y_overlap(bbox, get_bbox(nb)):
                        aligned.append(nb)
                        j += 1
                    else:
                        break

                if len(aligned) >= MIN_TABLE_COLUMNS:
                    flush_paragraph()

                    inferred = infer_table_from_paragraphs(aligned)
                    table_id = str(uuid.uuid4())

                    normalized["blocks"].append({
                        "block_type": "table",
                        "headers": inferred["headers"],
                        "rows": inferred["rows"],
                        "page_number": page_number,
                        "container_id": table_id,
                        "container_type": "table_group",
                        "container_path": current_container["path"].copy(),
                        "source_block_ids": inferred["source_block_ids"]
                    })

                    i += len(aligned)
                    continue

                # -------- NORMAL PARAGRAPH --------
                if not paragraph_buffer:
                    buffer_page = page_number

                paragraph_buffer.append(text)
                buffer_ids.append(block.get("block_id"))
                i += 1
                continue

            i += 1

    flush_paragraph()
    return normalized