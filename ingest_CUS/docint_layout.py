import os
import json
import os
import tempfile
from io import BytesIO
from docx import Document
from docx2pdf import convert
from typing import Dict, Any, List
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from content_understanding_layout import markdown_to_structured_layout
from content_understanding_client import AzureContentUnderstandingClient


def get_polygon_bbox(polygon):
    if not polygon or len(polygon) != 8:
        return None
    xs = polygon[0::2]
    ys = polygon[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]



import tempfile

def extract_layout_to_structured_json(file_bytes: bytes, source_name: str):

    client = AzureContentUnderstandingClient(
        endpoint="https://vishv-mhk5wnpr-eastus2.services.ai.azure.com/",
        api_version="2025-11-01",
        subscription_key="EecruWgqsnfExqvhutcZrXM1OUhiHNSA1n2ow7XvQng8HUXlwU27JQQJ99BKACHYHv6XJ3w3AAAAACOGKctF"
    )

    # ---------------------------------------
    # Save bytes to temporary file
    # ---------------------------------------
    suffix = os.path.splitext(source_name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # ---------------------------------------
    # Call Content Understanding
    # ---------------------------------------
    import time
    start_time = time.time()
    print("Submitting CU job...")

    response = client.begin_analyze_binary(
        analyzer_id="RIS_analyzer",
        file_location=tmp_path
    )

    print("CU job submitted. Operation ID:", response)
    print("Submit time:", time.time() - start_time, "seconds")

    print("Waiting for CU result...")
    poll_start = time.time()

    result = client.poll_result(response, timeout_seconds=300)
    print("CU finished in:", round(time.time() - poll_start, 2), "seconds")
    print("Total CU pipeline time:", round(time.time() - start_time, 2), "seconds")

    markdown = result["result"]["contents"][0]["markdown"]

    structured_doc = markdown_to_structured_layout(
        markdown,
        source_name
    )

    return structured_doc