import re
from typing import Dict, List, Any, Tuple


HEADING_REGEX = re.compile(r"^\d+(\.\d+)*\s+[A-Z][A-Z\s&]+$")

Y_ALIGNMENT_THRESHOLD = 0.03  # inches (safe for DOCX/PDF)
MIN_TABLE_COLUMNS = 3


def is_heading(text: str) -> bool:
    return bool(HEADING_REGEX.match(text.strip()))


def get_bbox(block: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return block.get("bbox", (0, 0, 0, 0))


def y_overlap(b1: Tuple[float, float, float, float],
              b2: Tuple[float, float, float, float]) -> bool:
    _, y1_min, _, y1_max = b1
    _, y2_min, _, y2_max = b2
    return abs(y1_min - y2_min) <= Y_ALIGNMENT_THRESHOLD


def normalize_table_from_lines(
    line_blocks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Convert a set of horizontally aligned lines into a semantic table.
    """
    # Sort left → right
    line_blocks = sorted(line_blocks, key=lambda b: get_bbox(b)[0])

    headers = [b["text"] for b in line_blocks]

    return {
        "block_type": "table",
        "headers": headers,
        "rows": [],
        "source_block_ids": [b["block_id"] for b in line_blocks]
    }


def normalize_table(table_block: Dict[str, Any]) -> Dict[str, Any]:
    rows = {}
    max_col = 0

    for cell in table_block["cells"]:
        r = cell["row_index"]
        c = cell["column_index"]
        rows.setdefault(r, {})[c] = cell["text"]
        max_col = max(max_col, c)

    ordered_rows = [
        [rows[r].get(c, "") for c in range(max_col + 1)]
        for r in sorted(rows)
    ]

    headers = ordered_rows[0] if ordered_rows else []
    data_rows = ordered_rows[1:] if len(ordered_rows) > 1 else []

    return {
        "block_type": "table",
        "headers": headers,
        "rows": data_rows
    }


def normalize_layout_json(layout_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts Layout JSON into semantic, chunk-ready blocks
    with inferred table detection using Y-axis alignment.
    """
    normalized = {
        "doc_id": layout_json.get("document_name"),
        "blocks": []
    }

    current_heading_path: List[str] = []
    paragraph_buffer = []
    buffer_block_ids = []
    buffer_page = None

    def flush_paragraph():
        nonlocal paragraph_buffer, buffer_block_ids, buffer_page
        if paragraph_buffer:
            normalized["blocks"].append({
                "block_type": "paragraph",
                "text": " ".join(paragraph_buffer),
                "heading_path": current_heading_path.copy(),
                "page_number": buffer_page,
                "source_block_ids": buffer_block_ids.copy()
            })
            paragraph_buffer.clear()
            buffer_block_ids.clear()

    for page in layout_json["pages"]:
        page_number = page["page_number"]
        blocks = page["blocks"]
        i = 0

        while i < len(blocks):
            block = blocks[i]

            # --------------------------------------------------
            # 1️⃣ Explicit tables (Azure detected)
            # --------------------------------------------------
            if block["block_type"] == "table":
                flush_paragraph()

                table = normalize_table(block)
                table.update({
                    "page_number": page_number,
                    "heading_path": current_heading_path.copy()
                })

                normalized["blocks"].append(table)
                i += 1
                continue

            # --------------------------------------------------
            # 2️⃣ Inferred tables from aligned paragraphs
            # --------------------------------------------------
            if block["block_type"] == "paragraph":
                aligned = [block]
                base_bbox = get_bbox(block)

                j = i + 1
                while j < len(blocks):
                    next_block = blocks[j]
                    if (
                        next_block["block_type"] == "paragraph"
                        and y_overlap(base_bbox, get_bbox(next_block))
                    ):
                        aligned.append(next_block)
                        j += 1
                    else:
                        break

                if len(aligned) >= MIN_TABLE_COLUMNS:
                    flush_paragraph()

                    table = normalize_table_from_lines(aligned)
                    table.update({
                        "page_number": page_number,
                        "heading_path": current_heading_path.copy()
                    })

                    normalized["blocks"].append(table)
                    i += len(aligned)
                    continue

            # --------------------------------------------------
            # 3️⃣ Heading / paragraph logic (unchanged)
            # --------------------------------------------------
            if block["block_type"] == "paragraph":
                text = block["text"].strip()

                # Heading
                if is_heading(text):
                    flush_paragraph()
                    heading_text = text.split(" ", 1)[1].strip()
                    current_heading_path = [heading_text]

                    normalized["blocks"].append({
                        "block_type": "heading",
                        "level": 1,
                        "text": heading_text,
                        "page_number": page_number
                    })
                    i += 1
                    continue

                # Sub-heading
                if (text.isupper() or text.istitle()) and len(text.split()) <= 4:
                    flush_paragraph()
                    current_heading_path = current_heading_path[:1] + [text]

                    normalized["blocks"].append({
                        "block_type": "heading",
                        "level": 2,
                        "text": text,
                        "page_number": page_number
                    })
                    i += 1
                    continue

                # Paragraph continuation
                if not paragraph_buffer:
                    buffer_page = page_number

                paragraph_buffer.append(text)
                buffer_block_ids.append(block["block_id"])
                i += 1
                continue

            i += 1

    flush_paragraph()
    return normalized