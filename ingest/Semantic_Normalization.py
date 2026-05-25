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
FIGURE_REGEX = re.compile(r"^Figure\s+\d+(\.\d+)*.*:\s*$", re.IGNORECASE)
TABLE_TITLE_REGEX = re.compile(
    r"^(?:Table|Listing|Figure|Graph|Output)\s+\d+(\.\d+)*[A-Za-z0-9\s\.\,\-\:\/\(\)✕%]*$",
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

def is_potential_heading(text: str) -> bool:
    text = text.strip()

    if not text:
        return False


    # numbered headings ONLY if ending with ":"
    if re.match(r"^\d+(\.\d+)*\.?\s+", text) and text.strip().endswith(":"):
        return True

    # TABLE / FIGURE / APPENDIX
    if TABLE_TITLE_REGEX.match(text):
        return True
    if FIGURE_REGEX.match(text) and text.strip().endswith(":"):
        return True
    if APPENDIX_REGEX.match(text):
        return True

    # # colon headings (but NOT bullets)
    # if text.endswith(":") and not text.startswith(("·", "-", "o")):

    #     # ❌ Reject if sentence-like (has period)
    #     if "." in text:
    #         return False

    #     # ❌ Reject if too long (likely paragraph)
    #     if len(text.split()) > 12:
    #         return False

    #     return True

    return False

def is_numbered_heading(text: str) -> bool:
    return bool(re.match(r"^\d+(\.\d+)*\.?", text))

def normalize_ocr_text(text: str, collapse_newlines: bool = False) -> str:
    if not text:
        return text

    text = str(text).strip()

    if collapse_newlines:
        text = text.replace("\n", " ")

    text = text.replace(":selected: X", "✕")
    text = text.replace(":selected:", "✕")
    text = text.replace(":unselected:", "")
    text = text.replace("응", "%")
    text = text.replace("士", "±")
    text = text.replace("土", "±")

    if collapse_newlines:
        text = re.sub(r"\s{2,}", " ", text)
    else:
        text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def normalize_table_rows(rows):
    import ast

    if isinstance(rows, str):
        try:
            parsed_rows = []
            for line in rows.splitlines():
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    parsed_rows.append(ast.literal_eval(line))
            rows = parsed_rows if parsed_rows else None
        except Exception:
            return None

    if not isinstance(rows, list):
        return None

    cleaned_rows = []

    for row in rows:
        cleaned_row = []
        for cell in row:
            cleaned_row.append(normalize_ocr_text(cell, collapse_newlines=True))
        cleaned_rows.append(cleaned_row)

    return cleaned_rows


def is_figure_noise_line(line: str) -> bool:
    t = line.strip()

    if not t:
        return True

    # ❌ No sentence structure (no verb, no punctuation)
    if not re.search(r"[\.]", t):
        # short phrase → likely label
        if len(t.split()) <= 6:
            return True

    # ❌ gene / symbol heavy
    if re.fullmatch(r"[A-Za-z0-9\+\-/\. ]+", t) and len(t.split()) <= 5:
        return True

    # ❌ uppercase biomedical labels
    if t.isupper() and len(t) <= 20:
        return True

    # ❌ patterns like "Untreated/ Green opsin"
    if "/" in t and len(t.split()) <= 6:
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

    line_frequency = Counter()
    total_pages = len(layout_json.get("pages", []))

    # Count paragraph occurrences per page (not per block)
    for page in layout_json.get("pages", []):
        seen_on_page = set()

        for block in page.get("blocks", []):
            if block.get("block_type") == "paragraph":
                text = normalize_ocr_text(block.get("text") or "", collapse_newlines=False)
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

        # 🔥 APPLY ONLY FOR FIGURE CONTEXT (based on container_path)
        is_figure_context = (
            current_container.get("path") and
            "figure" in current_container["path"][-1].lower()
        )

        if is_figure_context:
            lines = text.split("\n")

            cleaned_lines = []
            for line in lines:
                line = line.strip()

                if not line:
                    continue

                # remove short noisy lines ONLY
                if len(line.split()) < 5:
                    continue

                cleaned_lines.append(line)

            text = "\n".join(cleaned_lines).strip()

        # 🔥 SPLIT INLINE "Heading: Value" patterns
        parts = re.split(r"\n(?=[A-Z][A-Za-z\s\-\/\(\)]+:)", text)
        
        if len(parts) > 1:
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                if ":" in part:
                    key, val = part.split(":", 1)

                    key = key.strip() + ":"
                    val = val.strip()

                    cid = str(uuid.uuid4())

                    base_path = (
                        current_container["path"].copy()
                        if current_container["path"]
                        else last_heading_path.copy()
                    )

                    # sibling handling for colon headings
                    if base_path and base_path[-1].endswith(":"):
                        base_path.pop()

                    new_path = base_path + [key]

                    normalized["blocks"].append({
                        "block_type": "heading",
                        "level": len(new_path),
                        "text": key,
                        "page_number": buffer_page,
                        "container_id": cid,
                        "container_type": "section",
                        "container_path": new_path.copy()
                    })

                    if val:
                        normalized["blocks"].append({
                            "block_type": "paragraph",
                            "text": val,
                            "page_number": buffer_page,
                            "container_id": cid,
                            "container_type": "section",
                            "container_path": new_path.copy(),
                            "source_block_ids": buffer_ids.copy()
                        })

                else:
                    # ✅ Preserve non-colon lines
                    normalized["blocks"].append({
                        "block_type": "paragraph",
                        "text": part,
                        "page_number": buffer_page,
                        "container_id": current_container["container_id"],
                        "container_type": current_container["container_type"],
                        "container_path": current_container["path"].copy(),
                        "source_block_ids": buffer_ids.copy()
                    })

            paragraph_buffer.clear()
            buffer_ids.clear()
            buffer_page = None
            return

            #         key = key.strip() + ":"
            #         val = val.strip()

            #         # 🔥 create heading
            #         cid = str(uuid.uuid4())

            #         base_path = [
            #             p for p in current_container["path"]
            #             if is_numbered_heading(p)
            #         ]

            #         new_path = base_path + [key]

            #         normalized["blocks"].append({
            #             "block_type": "heading",
            #             "level": len(new_path),
            #             "text": key,
            #             "page_number": buffer_page,
            #             "container_id": cid,
            #             "container_type": "section",
            #             "container_path": new_path.copy()
            #         })

            #         # 🔥 create paragraph
            #         if val:
            #             normalized["blocks"].append({
            #                 "block_type": "paragraph",
            #                 "text": val,
            #                 "page_number": buffer_page,
            #                 "container_id": cid,
            #                 "container_type": "section",
            #                 "container_path": new_path.copy(),
            #                 "source_block_ids": buffer_ids.copy()
            #             })

            # paragraph_buffer.clear()
            # buffer_ids.clear()
            # buffer_page = None
            # return
        text = normalize_ocr_text(text, collapse_newlines=False)
        text = re.sub(r"\n{3,}", "\n\n", text)

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


    def is_footer_line(text: str) -> bool:
        t = text.strip().lower()

        if re.fullmatch(r"\d{1,3}", t):
            return True

        if "confidential" in t:
            return True

        if re.search(r"\b(inc\.|ltd\.|corp\.)\b", t):
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

            text = normalize_ocr_text(block.get("text") or "", collapse_newlines=False)

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
                    table_context_text = normalize_ocr_text(candidates[0]["text"], collapse_newlines=True)  # closest due to reverse scan


                # Prefer immediate previous paragraph if it looks like caption
                # 🔥 table context should always come from the LAST heading
                if current_container["path"]:
                    table_context_heading = current_container["path"][-1]
                    table_context_path = current_container["path"].copy()
                else:
                    table_context_heading = ""
                    table_context_path = []

                # optional fallback only when there is no heading context at all
                if not table_context_heading and page_header_table_candidate:
                    table_context_heading = page_header_table_candidate
                    table_context_path = [page_header_table_candidate]


                # Debug print (keep for 1-2 runs)
                # print(f"[TABLE DEBUG] Page {page_number} | rows={len(block.get('rows', []))} | heading={table_context_heading[:80]!r}")

                # normalized["blocks"].append({
                #     "block_type": "table",
                #     "table_context_heading": table_context_heading,
                #     "table_context_path": table_context_path,
                #     "table_context_text": table_context_text,
                #     "table_semantic_hint": (
                #         f"This table contains structured data related to '{table_context_heading or 'the corresponding section'}'. "
                #         f"Interpret the rows using the column headers."
                #         if table_context_heading else
                #         "This table contains structured data. Interpret the rows using the column headers."
                #     ),
                #     "caption": block.get("caption"),
                #     "headers": block.get("headers", []),
                #     "rows": block.get("rows", []),
                #     "page_number": page_number,
                #     "container_id": str(uuid.uuid4()),
                #     "container_type": "table_group",
                #     "container_path": current_container["path"].copy(),
                #     "source_block_ids": block.get("source_block_ids", [])
                # })

                clean_rows = normalize_table_rows(block.get("rows"))

                # -------------------------------------------------
                # TABLE CONTINUATION LOGIC
                # -------------------------------------------------

                previous_block = normalized["blocks"][-1] if normalized["blocks"] else None

                if (
                    previous_block
                    and previous_block.get("block_type") == "table"
                ):

                    prev_headers = previous_block.get("headers", [])
                    curr_headers = block.get("headers", [])

                    # if consecutive tables have different headers
                    # treat current table as continuation
                    if curr_headers != prev_headers:

                        # move current headers into rows
                        if clean_rows is None:
                            clean_rows = []

                        if curr_headers:
                            clean_rows.insert(0, curr_headers)

                        # inherit previous headers
                        block["headers"] = prev_headers

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

                    "caption": normalize_ocr_text(block.get("caption"), collapse_newlines=True) if block.get("caption") else None,
                    "headers": block.get("headers", []),
                    "rows": clean_rows,
                    "table_markdown": (
                        "| " + " | ".join(block.get("headers", [])) + " |\n"
                        + "|" + "|".join(["---"] * len(block.get("headers", []))) + "|\n"
                        + "\n".join(
                            "| " + " | ".join(row) + " |"
                            for row in (clean_rows or [])[:50]
                        )
                    ) if clean_rows else None,
                    "page_number": page_number,
                    "container_id": str(uuid.uuid4()),
                    "container_type": "table_group",
                    "source_block_ids": block.get("source_block_ids", [])
                })

                if block.get("table_context_path"):

                    current_container = {
                        "container_id": str(uuid.uuid4()),
                        "container_type": "table_context",
                        "path": block.get("table_context_path").copy()
                    }

                    last_heading_path = block.get("table_context_path").copy()

                continue

            # ============================
            # PARAGRAPH / HEADING
            # ============================

            if btype == "paragraph":

                text = normalize_ocr_text(block.get("text") or "", collapse_newlines=False)

                # NEW
                azure_role = block.get("azure_role")
                azure_is_heading = azure_role == "sectionHeading"

                # drop page headers/footers from Azure DI
                if azure_role in {"pageHeader", "pageFooter"}:
                    continue


                # 🔥 Split merged numbered heading + table title in same raw paragraph block
                merged_table_match = re.match(
                    r"^(?P<section>\d+(?:\.\d+)+\.?\s+.+?)\s+(?P<table>(?:Table|Listing)\s+\d+(?:\.\d+)*.*)$",
                    text,
                    re.IGNORECASE,
                )

                if merged_table_match:
                    section_text = merged_table_match.group("section").strip()
                    table_text = merged_table_match.group("table").strip()

                    # ---- emit numbered section heading ----
                    number_part = section_text.split()[0].rstrip(".")
                    level = len(number_part.split("."))

                    numbered_path = [
                        p for p in current_container["path"]
                        if is_numbered_heading(p)
                    ]

                    while len(numbered_path) >= level:
                        numbered_path.pop()

                    section_path = numbered_path + [section_text]

                    cid = str(uuid.uuid4())
                    current_container = {
                        "container_id": cid,
                        "container_type": "section",
                        "path": section_path
                    }

                    last_heading_text = section_text
                    last_heading_path = section_path.copy()

                    normalized["blocks"].append({
                        "block_type": "heading",
                        "level": level,
                        "text": section_text,
                        "page_number": page_number,
                        "container_id": cid,
                        "container_type": "section",
                        "container_path": section_path.copy()
                    })

                    # ---- emit table title heading under current numbered hierarchy ----
                    table_cid = str(uuid.uuid4())
                    table_path = section_path + [table_text]

                    current_container = {
                        "container_id": table_cid,
                        "container_type": "table_title",
                        "path": table_path
                    }

                    last_heading_text = table_text
                    last_heading_path = table_path.copy()

                    normalized["blocks"].append({
                        "block_type": "heading",
                        "level": len(table_path),
                        "text": table_text,
                        "page_number": page_number,
                        "container_id": table_cid,
                        "container_type": "table_title",
                        "container_path": table_path.copy()
                    })

                    continue


                if not text:
                    continue

                # 🔥 Remove footer lines BEFORE anything else
                lines = text.split("\n")

                clean_lines = [
                    line for line in lines
                    if not is_footer_line(line)
                ]

                text = "\n".join(clean_lines).strip()

                # If everything removed → skip block
                if not text:
                    continue

                # Existing header/footer logic
                if is_repeated_header_footer(text) and is_header_footer_region(block):
                    continue

                line_bbox = get_bbox(block)
                if line_bbox != (0.0, 0.0, 0.0, 0.0):
                    if any(is_duplicate_table_line(line_bbox, tb) for tb in table_bboxes):
                        continue


                # 🔥 ONLY flush if current line is heading
                if is_potential_heading(text):
                    flush_paragraph()

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


                if is_section:
                    flush_paragraph()

                    number_part = text.split()[0].rstrip(".")
                    level = len(number_part.split("."))

                    numbered_path = [
                        p for p in current_container["path"]
                        if is_numbered_heading(p)
                    ]

                    while len(numbered_path) >= level:
                        numbered_path.pop()

                    new_path = numbered_path + [text]

                    cid = str(uuid.uuid4())
                    current_container = {
                        "container_id": cid,
                        "container_type": "section",
                        "path": new_path
                    }

                    last_heading_text = text
                    last_heading_path = new_path.copy()

                    normalized["blocks"].append({
                        "block_type": "heading",
                        "level": level,
                        "text": text,
                        "page_number": page_number,
                        "container_id": cid,
                        "container_type": "section",
                        "container_path": new_path.copy()
                    })
                    continue


                # -------- AZURE SEMANTIC HEADING --------
                if azure_is_heading:

                    flush_paragraph()

                    cid = str(uuid.uuid4())

                    # preserve only numbered hierarchy
                    base_path = []

                    for p in current_container["path"]:
                        if is_numbered_heading(p):
                            base_path.append(p)

                    new_path = base_path + [text]

                    current_container = {
                        "container_id": cid,
                        "container_type": "section",
                        "path": new_path
                    }

                    last_heading_text = text
                    last_heading_path = new_path.copy()

                    normalized["blocks"].append({
                        "block_type": "heading",
                        "level": len(new_path),
                        "text": text,
                        "page_number": page_number,
                        "container_id": cid,
                        "container_type": "section",
                        "container_path": new_path.copy()
                    })

                    continue



                # -------- APPENDIX --------
                if APPENDIX_REGEX.match(text):
                    flush_paragraph()
                    cid = str(uuid.uuid4())

                    if is_numbered_heading(text):
                        new_path = current_container["path"] + [text]
                    else:
                        new_path = [text]
                    current_container = {
                        "container_id": cid,
                        "container_type": "appendix",
                        "path": new_path
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
                if FIGURE_REGEX.match(text) and text.strip().endswith(":"):
                    flush_paragraph()
                    cid = str(uuid.uuid4())

                    numbered_path = [
                        p for p in current_container["path"]
                        if is_numbered_heading(p)
                    ]

                    if numbered_path:
                        new_path = numbered_path + [text]
                    else:
                        new_path = [text]

                    current_container = {
                        "container_id": cid,
                        "container_type": "figure_group",
                        "path": new_path
                    }

                    last_heading_text = text
                    last_heading_path = new_path.copy()

                    normalized["blocks"].append({
                        "block_type": "heading",
                        "level": len(new_path),
                        "text": text,
                        "page_number": page_number,
                        "container_id": cid,
                        "container_type": "figure_group",
                        "container_path": new_path.copy()
                    })
                    continue

                # -------- TABLE TITLES --------
                if TABLE_TITLE_REGEX.match(text):
                    flush_paragraph()
                    cid = str(uuid.uuid4())

                    numbered_path = [
                        p for p in current_container["path"]
                        if is_numbered_heading(p)
                    ]

                    if numbered_path:
                        new_path = numbered_path + [text]
                    else:
                        new_path = [text]

                    current_container = {
                        "container_id": cid,
                        "container_type": "table_title",
                        "path": new_path
                    }

                    last_heading_text = text
                    last_heading_path = new_path.copy()

                    normalized["blocks"].append({
                        "block_type": "heading",
                        "level": len(new_path),
                        "text": text,
                        "page_number": page_number,
                        "container_id": cid,
                        "container_type": "table_title",
                        "container_path": new_path.copy()
                    })
                    continue

                # -------- COLON HEADINGS --------
                # if text.endswith(":") and (text.startswith("·") or text.startswith("-")) and len(text.split()) <= 10:
                #     flush_paragraph()
                #     cid = str(uuid.uuid4())

                #     # keep higher hierarchy
                #     base_path = current_container["path"].copy()

                #     # if previous node is also bullet heading,
                #     # make this a sibling, not child
                #     if base_path and base_path[-1].startswith(("·", "-", "•")):
                #         base_path.pop()

                #     new_path = base_path + [text]

                #     current_container = {
                #         "container_id": cid,
                #         "container_type": "section",
                #         "path": new_path
                #     }

                #     normalized["blocks"].append({
                #         "block_type": "heading",
                #         "level": len(new_path),
                #         "text": text,
                #         "page_number": page_number,
                #         "container_id": cid,
                #         "container_type": "section",
                #         "container_path": new_path.copy()
                #     })

                #     continue

                

                # -------------------------------------------------
                # Preserve table hierarchy for table metadata blocks
                # -------------------------------------------------

                TABLE_META_HEADINGS = {
                    "Note:",
                    "Notes:",
                    "SDTM date:",
                    "Program name:",
                    "Analysis date:",
                }

                if (
                    text.strip() in TABLE_META_HEADINGS
                    and last_heading_path
                    and any(
                        p.startswith(("Table ", "Listing "))
                        for p in last_heading_path
                    )
                ):
                    current_container["path"] = last_heading_path.copy()
                            
                # -------- NORMAL PARAGRAPH --------
                # 🔥 Detect "Heading:" followed by next block value (separate blocks)

                if is_potential_heading(text) and text.endswith(":") and not text.startswith(("·", "-", "o")):

                    flush_paragraph()

                    cid = str(uuid.uuid4())

                    base_path = [
                        p for p in current_container["path"]
                        if is_numbered_heading(p)
                    ]

                    if current_container["path"] and current_container["path"][-1] == text:
                        new_path = current_container["path"]   # ✅ don't duplicate
                    else:
                        new_path = base_path + [text]

                    current_container = {
                        "container_id": cid,
                        "container_type": "section",
                        "path": new_path
                    }

                    normalized["blocks"].append({
                        "block_type": "heading",
                        "level": len(new_path),
                        "text": text,
                        "page_number": page_number,
                        "container_id": cid,
                        "container_type": "section",
                        "container_path": new_path.copy()
                    })

                    continue


                # -------- NORMAL PARAGRAPH --------
                if not paragraph_buffer:
                    buffer_page = page_number


                paragraph_buffer.append(text)
                buffer_ids.append(block.get("block_id"))

        flush_paragraph()

    return normalized











# GPT normalizer

import uuid
from openai import AzureOpenAI
import json
from tqdm import tqdm
from config.settings import (
    # Chat model (authoring)
    AZURE_OPENAI_CHAT_API_KEY,
    AZURE_OPENAI_CHAT_MODEL,
    AZURE_OPENAI_CHAT_ENDPOINT,
    AZURE_OPENAI_API_CHAT_VERSION,
)

client = AzureOpenAI(
    api_key=AZURE_OPENAI_CHAT_API_KEY,
    api_version=AZURE_OPENAI_API_CHAT_VERSION,
    azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT
)

MODEL = "gpt-4.1-mini"


def generate_uuid():
    return str(uuid.uuid4())


def normalize_chunk_with_gpt(chunk, prev_container_stack):
    """
    chunk: list of raw blocks (10–20 max)
    prev_container_stack: [
        {"level": 1, "text": "1. SYNOPSIS", "id": "..."}
    ]
    """

    system_prompt = """
You are a STRICT document structure normalizer.

RULES:
- NEVER hallucinate
- NEVER create new headings
- NEVER modify text
- NEVER merge unrelated content
- ONLY use given input

TASK:
Convert raw blocks into structured JSON.

HEADING DETECTION RULES:
- Numbered patterns → level 1 (e.g., "1. SYNOPSIS")
- Key-value ending with ":" → level 2
- ALL CAPS short text → heading

PARAGRAPH RULES:
- Everything else → paragraph
- Paragraph belongs to LAST heading

GROUPING:
- Consecutive lines under a heading → ONE paragraph
- Maintain source_block_ids

OUTPUT STRICT JSON ONLY:
{
  "blocks": [...],
  "updated_stack": [...]
}
"""

    user_prompt = {
        "input_blocks": chunk,
        "previous_stack": prev_container_stack
    }

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=10000,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(user_prompt)}
        ]
    )

    return response.choices[0].message.content




def normalize_layout_json_with_gpt(structured_doc, chunk_size=25):

    all_blocks = structured_doc["blocks"]
    normalized = []
    container_stack = []

    total_chunks = (len(all_blocks) + chunk_size - 1) // chunk_size

    for i in tqdm(range(0, len(all_blocks), chunk_size), total=total_chunks, desc="Normalizing chunks"):
        chunk = all_blocks[i:i + chunk_size]

        chunk = [b for b in chunk if b.get("block_type") != "table"]

        result = normalize_chunk_with_gpt(chunk, container_stack)

        parsed = json.loads(result)

        normalized.extend(parsed["blocks"])
        container_stack = parsed["updated_stack"]

    return {"blocks": normalized}


def flatten_document_blocks(structured_doc):
    all_blocks = []

    for page in structured_doc.get("pages", []):
        page_blocks = page.get("blocks", [])

        for block in page_blocks:
            # ensure page_number exists
            block["page_number"] = page.get("page_number")
            all_blocks.append(block)

    return {"blocks": all_blocks}