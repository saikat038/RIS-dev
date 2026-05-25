import re
import uuid
from typing import Dict, List, Any, Tuple
from collections import Counter

# ----------------------------
# CONFIG
# ----------------------------

SECTION_REGEX = re.compile(
    r"^\d+(\.\d+)*\.?\s+[A-Z].+"
)

APPENDIX_REGEX = re.compile(r"^APPENDIX\s+[A-Z0-9]+", re.IGNORECASE)
FIGURE_REGEX = re.compile(r"^FIGURE\s+\d+", re.IGNORECASE)
TABLE_TITLE_REGEX = re.compile(
    r"^(?:Table|Listing|Figure|Graph|Output)\s+\d+(\.\d+)*[A-Za-z0-9\s\.\-\:\/]*$",
    re.IGNORECASE
)


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


# ----------------------------
# HEADER / FOOTER POSITION CHECK
# ----------------------------

def is_header_footer_region(block: Dict[str, Any], page_height: float = 11.0) -> bool:
    bbox = block.get("bbox")

    if not bbox or len(bbox) != 4:
        return False

    x1, y1, x2, y2 = bbox

    # Header → top 15% of page
    if y2 < page_height * 0.15:
        return True

    # Footer → bottom 15% of page
    if y1 > page_height * 0.85:
        return True

    return False


def is_duplicate_table_line(line_bbox, table_bbox, ratio_threshold=0.60) -> bool:
    la = bbox_area(line_bbox)
    if la == 0:
        return False
    inter = bbox_intersection_area(line_bbox, table_bbox)
    return (inter / la) >= ratio_threshold


# ----------------------------
# 🔑 NEW: PAGE HEADER / FOOTER GUARD
# ----------------------------

# def is_page_header_footer(text: str) -> bool:
#     t = text.strip()

#     # Dates like "02 Dec 2025"
#     if re.fullmatch(r"\d{1,2}\s+[A-Za-z]{3}\s+\d{4}", t):
#         return True

#     # Protocol identifiers
#     if re.search(r"\bprotocol\b", t, re.IGNORECASE):
#         return True

#     # Short document codes like OCU200
#     if re.fullmatch(r"[A-Z]{2,10}\d{0,4}", t):
#         return True

#     # Page footer
#     if re.search(r"page\s+\d+", t, re.IGNORECASE):
#         return True

#     # Confidential footer
#     if re.search(r"confidential", t, re.IGNORECASE):
#         return True

#     return False




# ----------------------------
# MAIN NORMALIZER
# ----------------------------

def normalize_layout_json(layout_json: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {
        "doc_id": layout_json.get("document_name"),
        "blocks": []
    }

    line_frequency = Counter()
    total_pages = len(layout_json.get("pages", []))

    # Count paragraph occurrences per page (not per block)
    for page in layout_json.get("pages", []):
        seen_on_page = set()

        for block in page.get("blocks", []):
            if block.get("block_type") == "paragraph":
                text = (block.get("text") or "").strip()
                if text:
                    seen_on_page.add(text)

        for text in seen_on_page:
            line_frequency[text] += 1


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
            return  # ← Nothing to flush → safe exit

        text = "\n\n".join(paragraph_buffer).strip()

        # Only add page_number if we have one
        page_info = buffer_page if buffer_page is not None else 0  # or None

        normalized["blocks"].append({
            "block_type": "paragraph",
            "text": text,
            "page_number": page_info,
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
    def is_repeated_header_footer(text: str) -> bool:
        t = text.strip()
        if not t:
            return False

        # Exclude structured table/listing titles
        if re.match(r"^(Table|Listing|Figure)\s+", t, re.IGNORECASE):
            return False

        # Generic page number patterns
        if re.fullmatch(r"\d+\s*/\s*\d+", t):
            return True

        if re.fullmatch(r"Page\s+\d+(\s+of\s+\d+)?", t, re.IGNORECASE):
            return True
        
        # Short uppercase alphanumeric code
        if len(t) <= 20 and re.fullmatch(r"[A-Z0-9\-]+", t):
            return True

        freq = line_frequency.get(t, 0)

        if total_pages > 0 and freq >= total_pages * 0.6:

            # Only short repeated content can be header/footer
            if len(t) < 60:

                # Must not contain scientific explanation patterns
                if not any(keyword in t.lower() for keyword in [
                    "note:", "% =", "defined as", "baseline", "="
                ]):
                    return True


        return False


    def looks_like_table_note(t: str) -> bool:
        s = t.strip().lower()
        if s.startswith("note:"): return True
        if s.startswith("notes:"): return True
        if s.startswith("abbreviations:"): return True
        if s.startswith("[1]") or s.startswith("(1)"): return True
        if " % =" in s: return True
        if "denominator" in s: return True
        if "mtd =" in s or " = " in s:  # abbreviations blocks
            # don’t overdo this, but it helps for your example
            return True
        return False



    
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

        page_header_table_candidate = None

        for block in blocks_sorted:
            if block.get("block_type") != "paragraph":
                continue

            text = (block.get("text") or "").strip()
            if not text:
                continue

            if TABLE_TITLE_REGEX.match(text):
                x1, y1, x2, y2 = get_bbox(block)
                if y1 < 1.0:
                    page_header_table_candidate = text
                break

        for block in blocks_sorted:
            btype = block.get("block_type")

            if btype != "paragraph":
                flush_paragraph()

            # ============================
            # TABLE
            # ============================
            if btype == "table":

                table_context_heading = ""
                table_context_text = ""

                # find closest paragraph above table on SAME page
                table_y1 = get_bbox(block)[1]  # your table bbox y1

                candidates = []
                for prev in reversed(normalized["blocks"][-200:]):
                    if prev.get("block_type") != "paragraph":
                        continue
                    if prev.get("page_number") != page_number:
                        continue
                    pb = prev.get("bbox")
                    if not pb:
                        continue
                    py2 = pb[3]
                    if py2 < table_y1 and not looks_like_table_note(prev.get("text","")):
                        candidates.append(prev)

                if candidates:
                    table_context_text = candidates[0]["text"]  # closest due to reverse scan


                # Prefer immediate previous paragraph if it looks like caption
                if normalized["blocks"]:
                    prev = normalized["blocks"][-1]
                    if prev["block_type"] == "paragraph":
                        candidate = (prev.get("text") or "").strip()
                        if "table" in candidate.lower():
                            table_context_heading = candidate

                # If no caption-like paragraph → inherit from nearest previous table or heading
                if not table_context_heading:
                    for prev in reversed(normalized["blocks"][-30:]):  # safer window
                        if prev["block_type"] == "table":
                            if prev.get("table_context_heading"):
                                table_context_heading = prev["table_context_heading"]
                            break

                        if prev["block_type"] == "heading":
                            candidate = prev.get("text", "").strip()
                            if "table" in candidate.lower():
                                table_context_heading = candidate
                                break
                            # optional: table_context_heading = candidate  # fallback to any heading

                        # Stop at unrelated content (don't inherit from very old tables)
                        if prev["block_type"] == "paragraph" and "table" not in prev.get("text", "").lower():
                            # Stop early on unrelated paragraph **once we already have a good context**
                            if table_context_heading:
                                break

                # Ultimate fallback
                if not table_context_heading:

                    if page_header_table_candidate:
                        table_context_heading = page_header_table_candidate

                    elif current_container["path"]:
                        table_context_heading = current_container["path"][-1]

                    else:
                        table_context_heading = ""



                # Prefer caption if available
                if block.get("caption") and "table" in block["caption"].lower():
                    table_context_heading = block["caption"]

                table_context_path = current_container["path"].copy()

                if not table_context_path and table_context_heading:
                    table_context_path = [table_context_heading]


                # Debug print (keep for 1-2 runs)
                # print(f"[TABLE DEBUG] Page {page_number} | rows={len(block.get('rows', []))} | heading={table_context_heading[:80]!r}")

                normalized["blocks"].append({
                    "block_type": "table",
                    "table_context_heading": table_context_heading,
                    "table_context_path": table_context_path,
                    "table_context_text": table_context_text,
                    "table_semantic_hint": (
                        f"This table contains structured data related to '{table_context_heading or 'the corresponding section'}'. "
                        f"Interpret the rows using the column headers."
                        if table_context_heading else
                        "This table contains structured data. Interpret the rows using the column headers."
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

                if is_repeated_header_footer(text) and is_header_footer_region(block):
                    continue

                line_bbox = get_bbox(block)
                if line_bbox != (0.0, 0.0, 0.0, 0.0):
                    if any(is_duplicate_table_line(line_bbox, tb) for tb in table_bboxes):
                        continue

                # flush_paragraph()

                # -------- SECTION --------

                single_level_match = re.match(r"^(\d+)\.\s+(.+)", text)
                multi_level_match = re.match(r"^\d+(\.\d+)+\.?\s+", text)

                if multi_level_match:
                    # Multi-level numbering → always heading
                    is_section = True

                elif single_level_match:
                    title_part = single_level_match.group(2)

                    # Single level must be ALL CAPS to be heading
                    if title_part.isupper():
                        is_section = True
                    else:
                        is_section = False
                else:
                    is_section = False


                if is_section and len(text.split()) <= 12:
                    flush_paragraph()
                    
                    level = text.split()[0].count('.')  # count dots in numbering
                    if text.split()[0].endswith('.'):
                        level -= 1

                    level = max(level, 1)

                    # Adjust path stack depth
                    while len(current_container["path"]) >= level:
                        current_container["path"].pop()

                    # Append new heading to hierarchy
                    current_container["path"].append(text)


                    cid = str(uuid.uuid4())
                    current_container = {
                        "container_id": cid,
                        "container_type": "section",
                        "path": current_container["path"].copy()
                    }
                    last_heading_text = text
                    last_heading_path = current_container["path"].copy()
                    number_part = text.split()[0].rstrip('.')

                    normalized["blocks"].append({
                        "block_type": "heading",
                        "level": number_part.count('.') + 1 ,  # dynamic level
                        "text": text,
                        "page_number": page_number,
                        "container_id": cid,
                        "container_type": "section",
                        "container_path": last_heading_path
                    })
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
                if FIGURE_REGEX.match(text):
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
                if TABLE_TITLE_REGEX.match(text):
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

                # # -------- COLON HEADINGS --------
                # if text.endswith(":") and len(text.split()) <= 10 and not is_page_header_footer(text):
                #     flush_paragraph()
                #     cid = str(uuid.uuid4())
                #     current_container = {
                #         "container_id": cid,
                #         "container_type": "section",
                #         "path": [text]
                #     }
                #     last_heading_text = text
                #     last_heading_path = current_container["path"].copy()

                #     normalized["blocks"].append({
                #         "block_type": "heading",
                #         "level": 2,
                #         "text": text,
                #         "page_number": page_number,
                #         "container_id": cid,
                #         "container_type": "section",
                #         "container_path": current_container["path"].copy()
                #     })
                #     continue

                # -------- NORMAL PARAGRAPH --------
                if not paragraph_buffer:
                    buffer_page = page_number

                paragraph_buffer.append(text)
                buffer_ids.append(block.get("block_id"))

        flush_paragraph()

    return normalized