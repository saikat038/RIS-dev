import re
from typing import Dict, Any, List
from bs4 import BeautifulSoup

TABLE_TAGS = ("<table", "<tr", "<th", "<td", "</table", "</tr", "</th", "</td")


def parse_html_table(html_lines: List[str]):
    """
    Parse CU HTML-like table fragments into headers + rows.
    Works even when CU emits only <th>/<td> lines without full <table>/<tr>.
    """

    html = "\n".join(html_lines).strip()
    if not html:
        return [], []

    # Wrap safely if CU gave fragment only
    wrapped_html = f"<table>{html}</table>"
    soup = BeautifulSoup(wrapped_html, "html.parser")

    headers = []
    rows = []

    tr_tags = soup.find_all("tr")

    # -----------------------------------------
    # Case 1: Proper HTML rows exist
    # -----------------------------------------
    if tr_tags:
        for tr in tr_tags:
            ths = tr.find_all("th")
            tds = tr.find_all("td")

            if ths and not headers:
                headers = [th.get_text(" ", strip=True) for th in ths]
                continue

            cells = tr.find_all(["td", "th"])
            if cells:
                rows.append([c.get_text(" ", strip=True) for c in cells])

        return headers, rows

    # -----------------------------------------
    # Case 2: No <tr> tags, only loose <th>/<td>
    # -----------------------------------------
    ths = soup.find_all("th")
    tds = soup.find_all("td")

    if ths and not headers:
        headers = [th.get_text(" ", strip=True) for th in ths]

    if tds:
        # If we have headers, split td values into row chunks of header length
        if headers:
            row_width = len(headers)
            flat_cells = [td.get_text(" ", strip=True) for td in tds]

            for i in range(0, len(flat_cells), row_width):
                rows.append(flat_cells[i:i + row_width])
        else:
            rows = [[td.get_text(" ", strip=True)] for td in tds]

    return headers, rows


def flush_table_block(current_page, table_buffer, block_counter):
    headers, rows = parse_html_table(table_buffer)

    current_page["blocks"].append({
        "block_id": f"tbl_{block_counter}",
        "block_type": "table",
        "headers": headers,
        "rows": rows,
        "bbox": None
    })

    return block_counter + 1


def markdown_to_structured_layout(markdown: str, source_name: str) -> Dict[str, Any]:
    pages = []
    current_page = {"page_number": 1, "blocks": []}

    lines = markdown.split("\n")

    block_counter = 0
    table_buffer = []
    inside_table = False
    heading_stack = {}

    for raw_line in lines:
        line = raw_line.strip()

        # --------------------------
        # Page Break
        # --------------------------
        if "<!-- PageBreak -->" in line:
            if table_buffer:
                block_counter = flush_table_block(current_page, table_buffer, block_counter)
                table_buffer = []
                inside_table = False

            pages.append(current_page)
            current_page = {
                "page_number": len(pages) + 1,
                "blocks": []
            }
            continue

        # --------------------------
        # Skip headers/footers/comments
        # --------------------------
        if line.startswith("<!--"):
            continue

        # --------------------------
        # Blank line handling
        # IMPORTANT: do not end table just because of blank line
        # --------------------------
        if not line:
            if inside_table:
                continue
            else:
                continue

        # --------------------------
        # Table detection
        # --------------------------
        if line.startswith(TABLE_TAGS):
            inside_table = True
            table_buffer.append(line)
            continue

        # --------------------------
        # End of table if a normal non-table line appears
        # --------------------------
        if inside_table and not line.startswith(TABLE_TAGS):
            block_counter = flush_table_block(current_page, table_buffer, block_counter)
            table_buffer = []
            inside_table = False

        # --------------------------
        # Headings
        # --------------------------
        if line.startswith("#"):
            heading_text = re.sub(r"^#+", "", line).strip()

            current_page["blocks"].append({
                "block_id": f"h_{block_counter}",
                "block_type": "paragraph",
                "text": heading_text,
                "bbox": None
            })
            block_counter += 1
            continue

        # --------------------------
        # Images (optional skip)
        # --------------------------
        if line.startswith("!["):
            continue

        # --------------------------
        # Paragraph
        # --------------------------
        current_page["blocks"].append({
            "block_id": f"p_{block_counter}",
            "block_type": "paragraph",
            "text": re.sub(r"<[^>]+>", "", line).strip(),
            "bbox": None
        })
        block_counter += 1

    # --------------------------
    # Final flush at end of file
    # --------------------------
    if table_buffer:
        block_counter = flush_table_block(current_page, table_buffer, block_counter)

    pages.append(current_page)

    return {
        "document_name": source_name,
        "model": "content-understanding",
        "pages": pages
    }