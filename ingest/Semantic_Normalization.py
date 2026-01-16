import re
from typing import Dict, List, Any


HEADING_REGEX = re.compile(r"^\d+(\.\d+)*\s+[A-Z][A-Z\s&]+$")


def is_heading(text: str) -> bool:
    return bool(HEADING_REGEX.match(text.strip()))


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
    Converts Layout JSON into semantic, chunk-ready blocks.
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

        for block in page["blocks"]:
            if block["block_type"] == "paragraph":
                text = block["text"].strip()

                # --- Heading ---
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
                    continue

                # --- Sub-heading (role names etc.) ---
                if text.isupper() or text.istitle() and len(text.split()) <= 4:
                    flush_paragraph()
                    current_heading_path = current_heading_path[:1] + [text]

                    normalized["blocks"].append({
                        "block_type": "heading",
                        "level": 2,
                        "text": text,
                        "page_number": page_number
                    })
                    continue

                # --- Paragraph continuation ---
                if not paragraph_buffer:
                    buffer_page = page_number

                paragraph_buffer.append(text)
                buffer_block_ids.append(block["block_id"])

            # --- Tables ---
            elif block["block_type"] == "table":
                flush_paragraph()

                table = normalize_table(block)
                table.update({
                    "page_number": page_number,
                    "heading_path": current_heading_path.copy()
                })

                normalized["blocks"].append(table)

    flush_paragraph()
    return normalized