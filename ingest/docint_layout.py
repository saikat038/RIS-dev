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
    # Tables (MINIMAL FIX APPLIED)
    # -------------------------------------------------
    for t_idx, table in enumerate(result.tables or []):

        table_page = (
            table.bounding_regions[0].page_number
            if table.bounding_regions else None
        )

        table_bbox = (
            get_polygon_bbox(table.bounding_regions[0].polygon)
            if table.bounding_regions else None
        )

        headers: Dict[int, Dict[str, Any]] = {}
        rows: Dict[int, Dict[int, Dict[str, Any]]] = {}
        max_col_index = -1

        for cell in table.cells:
            r = cell.row_index
            c = cell.column_index
            max_col_index = max(max_col_index, c)

            cell_obj = {
                "text": cell.content,
                "row_index": r,
                "column_index": c,
                "kind": cell.kind,
                "bbox": (
                    get_polygon_bbox(cell.bounding_regions[0].polygon)
                    if cell.bounding_regions else None
                )
            }

            rows.setdefault(r, {})[c] = cell_obj

            # Keep Azure-detected headers IF they exist
            if cell.kind == "columnHeader":
                headers[c] = cell_obj

        # ---- Build rectangular rows safely ----
        ordered_rows = [
            [
                rows[r].get(c, {}).get("text", "")
                for c in range(max_col_index + 1)
            ]
            for r in sorted(rows)
        ]

        # ---- Header fallback (ONLY if Azure headers missing) ----
        ordered_headers = (
            [headers[c]["text"] for c in sorted(headers)]
            if headers
            else (ordered_rows[0] if ordered_rows else [])
        )

        # If headers were inferred from first row, drop it from body
        if not headers and ordered_rows:
            ordered_rows = ordered_rows[1:]

        table_block = {
            "block_id": f"table_{t_idx}",
            "block_type": "table",
            "page_number": table_page,
            "bbox": table_bbox,
            "caption": (
                {
                    "text": table.caption.content,
                    "bbox": (
                        get_polygon_bbox(table.caption.bounding_regions[0].polygon)
                        if table.caption.bounding_regions else None
                    )
                }
                if table.caption else None
            ),
            "headers": ordered_headers,
            "rows": ordered_rows,
            "cells": {
                "headers": headers,
                "body": rows
            }
        }

        # Attach table to its page
        for page in structured_doc["pages"]:
            if page["page_number"] == table_page:
                page["blocks"].append(table_block)
                break

    # -------------------------------------------------
    # Save raw structured layout
    # -------------------------------------------------
    with open("sample_layout_structured.json", "w", encoding="utf-8") as f:
        json.dump(structured_doc, f, ensure_ascii=False, indent=2)

    print("✅ Layout JSON saved to: sample_layout_structured.json")
    return structured_doc
