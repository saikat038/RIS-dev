import os
import json
from typing import Dict, Any, List
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest


def get_polygon_bbox(polygon):
    if not polygon or len(polygon) != 8:
        return None
    xs = polygon[0::2]
    ys = polygon[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]



def extract_layout_to_structured_json(
    file_bytes: bytes,
    source_name: str
) -> Dict[str, Any]:
    """
    Extract document using Azure prebuilt-layout model
    and PRESERVE FULL TABLE SEMANTICS:
    - captions
    - headers
    - rows
    - cell geometry
    - table geometry
    """

    client = DocumentIntelligenceClient(
        endpoint=os.getenv("DOC_INTELLIGENCE_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("DOC_INTELLIGENCE_KEY"))
    )

    request = AnalyzeDocumentRequest(bytes_source=file_bytes)
    poller = client.begin_analyze_document("prebuilt-layout", body=request)
    result = poller.result()

    structured_doc = {
        "document_name": source_name,
        "model": "prebuilt-layout",
        "pages": []
    }

    # -------------------------------------------------
    # Pages & paragraph blocks
    # -------------------------------------------------
    for page in result.pages:
        page_data = {
            "page_number": page.page_number,
            "width": page.width,
            "height": page.height,
            "unit": page.unit,
            "blocks": []
        }

        for idx, line in enumerate(page.lines or []):
            page_data["blocks"].append({
                "block_id": f"p{page.page_number}_line_{idx}",
                "block_type": "paragraph",
                "text": line.content,
                "bbox": get_polygon_bbox(line.polygon)
            })

        structured_doc["pages"].append(page_data)

    # -------------------------------------------------
    # Tables (NO INFERENCE – PURE AZURE OUTPUT)
    # -------------------------------------------------
    previous_table_page = None
    previous_headers = None

    for t_idx, table in enumerate(result.tables or []):

        table_page = table.bounding_regions[0].page_number
        table_bbox = get_polygon_bbox(table.bounding_regions[0].polygon)

        is_continuation = (
            previous_table_page is not None
            and table_page == previous_table_page + 1
        )

        # Build grid
        grid = [[""] * table.column_count for _ in range(table.row_count)]

        for cell in table.cells:
            r = cell.row_index
            c = cell.column_index
            if grid[r][c] == "":
                grid[r][c] = cell.content or ""

        headers = []
        rows = []

        # ---------------- HEADER DETECTION ----------------
        header_row_indices = {
            cell.row_index
            for cell in table.cells
            if cell.kind == "columnHeader"
        }

        if header_row_indices and not is_continuation:
            header_row = min(header_row_indices)
            headers = grid[header_row]
            rows = [row for i, row in enumerate(grid) if i != header_row]
            previous_headers = headers

        elif is_continuation and previous_headers:
            headers = previous_headers
            rows = grid

        else:
            headers = []
            rows = grid

        # ---------------- REMOVE DUPLICATE HEADER ROW ----------------
        if headers:
            rows = [
                row for row in rows
                if any(cell.strip() != hdr.strip()
                    for cell, hdr in zip(row, headers))
            ]

        table_block = {
            "block_id": f"table_{t_idx}",
            "block_type": "table",
            "page_number": table_page,
            "bbox": table_bbox,
            "headers": headers,
            "rows": rows
        }

        for page in structured_doc["pages"]:
            if page["page_number"] == table_page:
                page["blocks"].append(table_block)
                break

        previous_table_page = table_page



    # -------------------------------------------------
    # Save raw structured layout
    # -------------------------------------------------
    with open("sample_layout_structured.json", "w", encoding="utf-8") as f:
        json.dump(structured_doc, f, ensure_ascii=False, indent=2)

    print("✅ Layout JSON saved to: sample_layout_structured.json")
    return structured_doc