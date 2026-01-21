import re
import uuid
from typing import Dict, List, Any, Tuple

# ----------------------------
# CONFIG
# ----------------------------

SECTION_REGEX = re.compile(r"^\d+(\.\d+)*\s+.+$")
APPENDIX_REGEX = re.compile(r"^APPENDIX\s+[A-Z0-9]+", re.IGNORECASE)
FIGURE_REGEX = re.compile(r"^FIGURE\s+\d+", re.IGNORECASE)
TABLE_TITLE_REGEX = re.compile(r"^(table|figure)\s+\d+", re.IGNORECASE)

# ----------------------------
# GEOMETRY HELPERS
# ----------------------------

def get_bbox(block: Dict[str, Any]) -> Tuple[float, float, float, float]:
    bbox = block.get("bbox")
    if not bbox or len(bbox) != 4:
        return (0.0, 0.0, 0.0, 0.0)
    return (bbox[0], bbox[1], bbox[2], bbox[3])


def bbox_intersection_area(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def bbox_area(a) -> float:
    x1, y1, x2, y2 = a
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def is_duplicate_table_line(line_bbox, table_bbox, ratio_threshold=0.60) -> bool:
    la = bbox_area(line_bbox)
    if la == 0:
        return False
    inter = bbox_intersection_area(line_bbox, table_bbox)
    return (inter / la) >= ratio_threshold


# ----------------------------
# 🔑 NEW: PAGE HEADER / FOOTER GUARD
# ----------------------------

def is_page_header_footer(text: str) -> bool:
    t = text.strip()

    # Dates like "02 Dec 2025"
    if re.fullmatch(r"\d{1,2}\s+[A-Za-z]{3}\s+\d{4}", t):
        return True

    # Protocol identifiers
    if re.search(r"\bprotocol\b", t, re.IGNORECASE):
        return True

    # Short document codes like OCU200
    if re.fullmatch(r"[A-Z]{2,10}\d{0,4}", t):
        return True

    # Page footer
    if re.search(r"page\s+\d+", t, re.IGNORECASE):
        return True

    # Confidential footer
    if re.search(r"confidential", t, re.IGNORECASE):
        return True

    return False


# ----------------------------
# MAIN NORMALIZER
# ----------------------------

def normalize_layout_json(layout_json: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {
        "doc_id": layout_json.get("document_name"),
        "blocks": []
    }

    current_container = {
        "container_id": None,
        "container_type": None,
        "path": []
    }

    paragraph_buffer: List[str] = []
    buffer_ids: List[str] = []
    buffer_page = None

    # 🔑 Semantic memory
    last_heading_text = None
    last_heading_path = []
    last_paragraph_text = None

    def flush_paragraph():
        nonlocal paragraph_buffer, buffer_ids, buffer_page, last_paragraph_text
        if not paragraph_buffer:
            return

        text = " ".join(paragraph_buffer)

        normalized["blocks"].append({
            "block_type": "paragraph",
            "text": text,
            "page_number": buffer_page,
            "container_id": current_container["container_id"],
            "container_type": current_container["container_type"],
            "container_path": current_container["path"].copy(),
            "source_block_ids": buffer_ids.copy()
        })

        last_paragraph_text = text
        paragraph_buffer.clear()
        buffer_ids.clear()
        buffer_page = None

    # ----------------------------
    # PROCESS DOCUMENT
    # ----------------------------

    for page in layout_json.get("pages", []):
        page_number = page.get("page_number")
        blocks = page.get("blocks", [])

        table_bboxes = [
            get_bbox(b) for b in blocks
            if b.get("block_type") == "table" and b.get("bbox")
        ]

        blocks_sorted = sorted(
            blocks,
            key=lambda b: (get_bbox(b)[1], get_bbox(b)[0])
        )

        for block in blocks_sorted:
            btype = block.get("block_type")

            if btype != "paragraph":
                flush_paragraph()

            # ============================
            # TABLE
            # ============================
            if btype == "table":
                normalized["blocks"].append({
                    "block_type": "table",

                    "table_context_heading": last_heading_text,
                    "table_context_path": last_heading_path,
                    "table_context_text": last_paragraph_text,
                    "table_semantic_hint": (
                        f"This table contains structured data related to '{last_heading_text}'. "
                        f"Interpret the rows using the column headers."
                        if last_heading_text
                        else "This table contains structured data. Interpret the rows using the column headers."
                    ),

                    "caption": block.get("caption"),
                    "headers": block.get("headers", []),
                    "rows": block.get("rows", []),
                    "page_number": page_number,
                    "container_id": str(uuid.uuid4()),
                    "container_type": "table_group",
                    "container_path": current_container["path"].copy(),
                    "source_block_ids": block.get("source_block_ids", [])
                })
                continue

            # ============================
            # PARAGRAPH / HEADING
            # ============================
            if btype == "paragraph":
                text = (block.get("text") or "").strip()
                if not text:
                    continue

                line_bbox = get_bbox(block)
                if line_bbox != (0.0, 0.0, 0.0, 0.0):
                    if any(is_duplicate_table_line(line_bbox, tb) for tb in table_bboxes):
                        continue

                # -------- SECTION --------
                if SECTION_REGEX.match(text) and not is_page_header_footer(text):
                    flush_paragraph()
                    cid = str(uuid.uuid4())
                    current_container = {
                        "container_id": cid,
                        "container_type": "section",
                        "path": [text]
                    }
                    last_heading_text = text
                    last_heading_path = current_container["path"].copy()

                    normalized["blocks"].append({
                        "block_type": "heading",
                        "level": 1,
                        "text": text,
                        "page_number": page_number,
                        "container_id": cid,
                        "container_type": "section",
                        "container_path": current_container["path"].copy()
                    })
                    continue

                # -------- APPENDIX --------
                if APPENDIX_REGEX.match(text) and not is_page_header_footer(text):
                    flush_paragraph()
                    cid = str(uuid.uuid4())
                    current_container = {
                        "container_id": cid,
                        "container_type": "appendix",
                        "path": [text]
                    }
                    last_heading_text = text
                    last_heading_path = current_container["path"].copy()

                    normalized["blocks"].append({
                        "block_type": "heading",
                        "level": 1,
                        "text": text,
                        "page_number": page_number,
                        "container_id": cid,
                        "container_type": "appendix",
                        "container_path": current_container["path"].copy()
                    })
                    continue

                # -------- FIGURE --------
                if FIGURE_REGEX.match(text) and not is_page_header_footer(text):
                    flush_paragraph()
                    cid = str(uuid.uuid4())
                    current_container = {
                        "container_id": cid,
                        "container_type": "figure_group",
                        "path": current_container["path"] + [text]
                    }
                    last_heading_text = text
                    last_heading_path = current_container["path"].copy()

                    normalized["blocks"].append({
                        "block_type": "heading",
                        "level": 2,
                        "text": text,
                        "page_number": page_number,
                        "container_id": cid,
                        "container_type": "figure_group",
                        "container_path": current_container["path"].copy()
                    })
                    continue

                # -------- TABLE TITLES --------
                if TABLE_TITLE_REGEX.match(text) and not is_page_header_footer(text):
                    flush_paragraph()
                    last_heading_text = text
                    last_heading_path = current_container["path"].copy()

                    normalized["blocks"].append({
                        "block_type": "heading",
                        "level": 2,
                        "text": text,
                        "page_number": page_number,
                        "container_id": current_container["container_id"],
                        "container_type": "table_title",
                        "container_path": current_container["path"].copy()
                    })
                    continue

                # -------- COLON HEADINGS --------
                if text.endswith(":") and len(text.split()) <= 10 and not is_page_header_footer(text):
                    flush_paragraph()
                    cid = str(uuid.uuid4())
                    current_container = {
                        "container_id": cid,
                        "container_type": "section",
                        "path": [text]
                    }
                    last_heading_text = text
                    last_heading_path = current_container["path"].copy()

                    normalized["blocks"].append({
                        "block_type": "heading",
                        "level": 2,
                        "text": text,
                        "page_number": page_number,
                        "container_id": cid,
                        "container_type": "section",
                        "container_path": current_container["path"].copy()
                    })
                    continue

                # -------- NORMAL PARAGRAPH --------
                if not paragraph_buffer:
                    buffer_page = page_number

                paragraph_buffer.append(text)
                buffer_ids.append(block.get("block_id"))

        flush_paragraph()

    return normalized