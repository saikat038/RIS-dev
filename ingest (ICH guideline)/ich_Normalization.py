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


# =========================================================
# ICH NORMALIZER (RULE-ATOMIC)
# =========================================================

ICH_SECTION_REGEX = re.compile(r"^(\d+(\.\d+){0,4})\s+(.+)$")
ICH_RULE_REGEX = re.compile(r"\b(shall|must|should|may)\b", re.IGNORECASE)


def detect_rule_type(text: str) -> str:
    t = text.lower()
    if "shall" in t or "must" in t:
        return "mandatory"
    if "should" in t:
        return "recommended"
    if "may" in t:
        return "optional"
    return "informational"


def normalize_ich_layout_json(layout_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize layout JSON into structured blocks while preserving paragraph separation.
    - Keeps paragraphs as separate items (with newlines)
    - Detects headings / sections / tables
    - Avoids forcing everything into single-line strings
    """
    normalized = {
        "doc_id": layout_json.get("document_name"),
        "blocks": []
    }

    current_section = {
        "section_path": None,
        "section_title": None,
        "container_id": None,
    }

    paragraph_buffer = []
    buffer_page = None
    buffer_ids = []

    def flush_paragraph():
        nonlocal buffer_page

        if not paragraph_buffer:
            return

        # Join buffer with spaces inside paragraph, but keep paragraphs separate later
        paragraph_text = " ".join(paragraph_buffer).strip()

        if paragraph_text:
            normalized["blocks"].append({
                "block_type": "paragraph",
                "text": paragraph_text,
                "page_number": buffer_page,
                "section_path": current_section["section_path"],
                "section_title": current_section["section_title"],
                "source_block_ids": buffer_ids.copy()
            })

        paragraph_buffer.clear()
        buffer_ids.clear()
        buffer_page = None

    for page in layout_json.get("pages", []):
        page_number = page.get("page_number")
        blocks = page.get("blocks", [])

        # Sort blocks top-to-bottom, left-to-right
        blocks_sorted = sorted(
            blocks,
            key=lambda b: (get_bbox(b)[1], get_bbox(b)[0]) if get_bbox(b) else (0, 0)
        )

        for block in blocks_sorted:
            btype = block.get("block_type")
            text = (block.get("text") or "").strip()

            if not text:
                continue

            # Flush previous paragraph on structural change
            if btype != "paragraph":
                flush_paragraph()

            # ────────────────────────────────────────
            # HEADINGS / SECTIONS
            # ────────────────────────────────────────
            if SECTION_REGEX.match(text):
                flush_paragraph()
                current_section["section_path"] = text.split(" ", 1)[0]
                current_section["section_title"] = text.split(" ", 1)[1] if " " in text else text
                normalized["blocks"].append({
                    "block_type": "heading",
                    "level": 1,
                    "text": text,
                    "page_number": page_number,
                    "section_path": current_section["section_path"],
                    "section_title": current_section["section_title"]
                })
                continue

            # Table titles, figures, appendices (similar logic)
            if TABLE_TITLE_REGEX.match(text) or FIGURE_REGEX.match(text) or APPENDIX_REGEX.match(text):
                flush_paragraph()
                normalized["blocks"].append({
                    "block_type": "heading",
                    "level": 2,
                    "text": text,
                    "page_number": page_number,
                    "section_path": current_section["section_path"],
                    "section_title": current_section["section_title"]
                })
                continue

            # ────────────────────────────────────────
            # NORMAL PARAGRAPH
            # ────────────────────────────────────────
            if btype == "paragraph":
                if not paragraph_buffer:
                    buffer_page = page_number
                paragraph_buffer.append(text)
                buffer_ids.append(block.get("block_id", str(uuid.uuid4())))

        # Flush any remaining paragraph at end of page
        flush_paragraph()

    # Final flush
    flush_paragraph()

    return normalized
