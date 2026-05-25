import re
import uuid
from typing import Dict, List, Any, Tuple
from collections import Counter

# ----------------------------
# CONFIG
# ----------------------------
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


    paragraph_buffer = []
    buffer_page = None
    buffer_ids = []

    def flush_paragraph():
        nonlocal buffer_page

        if not paragraph_buffer:
            return

        # Join buffer with spaces inside paragraph, but keep paragraphs separate later
        paragraph_text = "\n".join(paragraph_buffer).strip()

        if paragraph_text:
            normalized["blocks"].append({
                "block_type": "paragraph",
                "text": paragraph_text,
                "page_number": buffer_page,
                "section_heading": current_heading or "",   # ← THIS is the link
                "source_block_ids": buffer_ids.copy()
            })

        paragraph_buffer.clear()
        buffer_ids.clear()
        buffer_page = None



    line_frequency = Counter()
    total_pages = len(layout_json.get("pages", []))



    # Count paragraph occurrences per page
    for page in layout_json.get("pages", []):
        seen_on_page = set()
        for block in page.get("blocks", []):
            if block.get("block_type") == "paragraph":
                text = (block.get("text") or "").strip()
                if text:
                    seen_on_page.add(text)
        for text in seen_on_page:
            line_frequency[text] += 1



    # ----------------------------
    # 🔑 NEW: PAGE HEADER / FOOTER GUARD
    # ----------------------------

    def is_repeated_header_footer(text: str) -> bool:
        t = text.strip()
        if not t:
            return False

        # Generic page patterns
        if re.fullmatch(r"\d+\s*/\s*\d+", t):
            return True

        if re.fullmatch(r"Page\s+\d+(\s+of\s+\d+)?", t, re.IGNORECASE):
            return True

        # Very short uppercase codes
        if len(t) <= 20 and re.fullmatch(r"[A-Z0-9\-]+", t):
            return True

        freq = line_frequency.get(t, 0)

        # Appears in >60% pages and short → header/footer
        if total_pages > 0 and freq >= total_pages * 0.6 and len(t) < 80:
            return True

        return False



    current_heading = None

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

            if is_repeated_header_footer(text):
                continue
            # Flush previous paragraph on structural change
            if btype != "paragraph":
                flush_paragraph()

            # ────────────────────────────────────────
            # HEADINGS / SECTIONS
            # ────────────────────────────────────────
            single_level_match = re.match(r"^(\d+)\.\s+(.+)", text)
            multi_level_match = re.match(r"^\d+(\.\d+)+\.?\s+", text)

            if multi_level_match:
                is_section = True

            elif single_level_match:
                title_part = single_level_match.group(2)

                # Single-level heading must be ALL CAPS
                if title_part.isupper():
                    is_section = True
                else:
                    is_section = False
            else:
                is_section = False


            if is_section and len(text.split()) <= 15:

                # 1️⃣ FIRST flush previous paragraph
                flush_paragraph()

                # 2️⃣ THEN update current heading
                current_heading = text

                # 3️⃣ THEN append heading block
                number_part = text.split()[0].rstrip(".")

                normalized["blocks"].append({
                    "block_type": "heading",
                    "level": number_part.count('.') + 1,
                    "text": text,
                    "page_number": page_number
                })

                continue

            # Table titles, figures, appendices (similar logic)
            if TABLE_TITLE_REGEX.match(text) or FIGURE_REGEX.match(text):
                flush_paragraph()

                current_heading = text  # treat as heading

                normalized["blocks"].append({
                    "block_type": "heading",
                    "level": 2,
                    "text": text,
                    "page_number": page_number
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

    # ---------------- DEBUG OUTPUT ----------------
    import os, json
    try:
        debug_path = os.path.join(f"ICH_layout_semantic.json")

        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(normalized, f, indent=2)

        print(f"🛠 Normalization debug saved to: {debug_path}")
    except Exception as e:
        print(f"⚠️ Debug save failed: {e}")
    # ------------------------------------------------

    return normalized
