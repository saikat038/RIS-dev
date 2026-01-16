import os
import json
from pathlib import Path
from typing import List, Dict, Any

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from Semantic_Normalization import normalize_layout_json


def get_polygon_bbox(polygon):
    if not polygon or len(polygon) != 8:
        return None
    xs = polygon[0::2]
    ys = polygon[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def extract_layout_to_structured_json(file_bytes: str, source_name: str):
    """
    Extracts document using Azure Layout model and stores
    a RAG-ready structured JSON representation.
    """

    client = DocumentIntelligenceClient(
        endpoint=os.getenv("DOC_INTELLIGENCE_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("DOC_INTELLIGENCE_KEY"))
    )

    # file_bytes = Path(file_path).read_bytes()
    request = AnalyzeDocumentRequest(bytes_source=file_bytes)

    poller = client.begin_analyze_document("prebuilt-layout", body=request)
    result = poller.result()

    structured_doc = {
        "document_name": source_name,
        "model": "prebuilt-layout",
        "pages": []
    }

    # --- Process pages ---
    for page in result.pages:
        page_data = {
            "page_number": page.page_number,
            "width": page.width,
            "height": page.height,
            "unit": page.unit,
            "blocks": []
        }

        # ---- Paragraphs (use lines) ----
        for idx, line in enumerate(page.lines or []):
            page_data["blocks"].append({
                "block_id": f"p{page.page_number}_line_{idx}",
                "block_type": "paragraph",
                "text": line.content,
                "bbox": get_polygon_bbox(line.polygon),
                "confidence": line.confidence if hasattr(line, "confidence") else None
            })

        structured_doc["pages"].append(page_data)

    # ---- Tables (global, mapped by page) ----
    if result.tables:
        for t_idx, table in enumerate(result.tables):
            table_block = {
                "block_id": f"table_{t_idx}",
                "block_type": "table",
                "page_number": table.bounding_regions[0].page_number if table.bounding_regions else None,
                "row_count": table.row_count,
                "column_count": table.column_count,
                "cells": []
            }

            for cell in table.cells:
                table_block["cells"].append({
                    "row_index": cell.row_index,
                    "column_index": cell.column_index,
                    "text": cell.content,
                    "bbox": (
                                get_polygon_bbox(cell.bounding_regions[0].polygon)
                                if cell.bounding_regions
                                else None
                            )
                })

            # Attach table to correct page
            for page in structured_doc["pages"]:
                if page["page_number"] == table_block["page_number"]:
                    page["blocks"].append(table_block)
                    break

    # ---- Save JSON ----
    output_json = "sample_layout_structured.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(structured_doc, f, ensure_ascii=False, indent=2)


    print(f"✅ Layout JSON saved to: {output_json}")
    return structured_doc


# if __name__ == "__main__":
#     input_pdf = "sample.pdf"
#     output_json = "sample_layout_structured.json"

#     # main_function
#     layout_json = extract_layout_to_structured_json(input_pdf, output_json)
    
#     # normalizing layout output
#     normalized = normalize_layout_json(layout_json)

#     with open("sample_chunk_ready.json", "w", encoding="utf-8") as f:
#         json.dump(normalized, f, indent=2, ensure_ascii=False)