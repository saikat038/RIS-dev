"""
A sample to demonstrate analyzing with Azure Content Understanding Python SDK.

Requirements:
    - Python 3.9 or later

Setup:
    Follow the steps in the url below to configure your Microsoft Foundry resource and model deployments:
    https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/contentunderstanding/azure-ai-contentunderstanding#configuring-microsoft-foundry-resource

Configuration:
    Before running, update the following variables in the script:
    - AZURE_CONTENT_UNDERSTANDING_ENDPOINT: The endpoint to your Content Understanding resource.
    - CONTENT_UNDERSTANDING_KEY: Your Content Understanding API key (optional if using DefaultAzureCredential).
    - FILE_URL: URL of the file to analyze.

Usage:
    1. Navigate to the directory containing this file:
       cd path/to/the/directory/containing/this/file    # In your terminal

    2. (Optional) Create and activate a virtual environment:
       python -m venv .venv         # One time setup
       source .venv/bin/activate      # On Linux/macOS
       .venv\\Scripts\\activate        # On Windows

    3. Install dependencies:
       python -m pip install azure-ai-contentunderstanding azure-identity

    4. Run the script:
       python sample.py
"""

# import sys
# import json

# from azure.ai.contentunderstanding import ContentUnderstandingClient
# from azure.ai.contentunderstanding.models import AnalysisInput, AnalysisResult
# from azure.core.credentials import AzureKeyCredential
# from azure.core.exceptions import AzureError
# from azure.identity import DefaultAzureCredential


# def main() -> None:
#     # Insert the following configurations.
#     # 1) AZURE_CONTENT_UNDERSTANDING_ENDPOINT - the endpoint to your Content Understanding resource.
#     endpoint = "https://vishv-mhk5wnpr-eastus2.services.ai.azure.com/"

#     # 2) CONTENT_UNDERSTANDING_KEY - your Content Understanding API key (optional if using DefaultAzureCredential).
#     key = "{{CONTENT_UNDERSTANDING_KEY}}"

#     # 3) FILE_URL - you can replace this with your own URL.
#     file_url = "{{FILE_URL}}"

#     # ANALYZER_ID - the ID of the analyzer to use.
#     analyzer_id = "prebuilt-layout"

#     # API_VERSION - the API version to use.
#     api_version = "2025-11-01"

#     # Set up Content Understanding client.
#     credential = AzureKeyCredential(key) if key and "{{CONTENT_UNDERSTANDING_KEY}}" not in key else DefaultAzureCredential()
#     client = ContentUnderstandingClient(endpoint=endpoint, credential=credential, api_version=api_version)

#     # [START analyze]
#     print(f"Analyzing with {analyzer_id} analyzer...")
#     print(f"  File URL: {file_url}\n")

#     try:
#         poller = client.begin_analyze(
#             analyzer_id=analyzer_id,
#             inputs=[AnalysisInput(url=file_url)],
#         )
#         result: AnalysisResult = poller.result()
#     except AzureError as err:
#         print(f"[Azure Error]: {err.message}")
#         sys.exit(1)
#     except Exception as ex:
#         print(f"[Unexpected Error]: {ex}")
#         sys.exit(1)
#     # [END analyze]

#     # [START output_result]
#     print("=" * 50)
#     print("Analysis result:")
#     print("=" * 50 + "\n")

#     max_display_lines = 50
#     result_str = json.dumps(result.as_dict(), indent=2)
#     ret_lines = result_str.splitlines()

#     if len(ret_lines) > max_display_lines:
#         print("\n".join(ret_lines[:max_display_lines]))
#         print(f"\n {len(ret_lines) - max_display_lines} more lines to be displayed...\n")
#     else:
#         print(result_str)
#     # [END output_result]


# if __name__ == "__main__":
#     main()




import sys
import os
import json
import re
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bs4 import BeautifulSoup

from azure.ai.contentunderstanding import ContentUnderstandingClient
from azure.ai.contentunderstanding.models import AnalyzeInput
from azure.core.credentials import AzureKeyCredential

AZURE_CONTENT_UNDERSTANDING_ENDPOINT = "https://ris-dev-resource-0974.services.ai.azure.com/"
AZURE_CONTENT_UNDERSTANDING_KEY = "3v5tHxuHYmbnYZvTEGFOJTaTskzn2riTzCfDYB058NrR1Xz19um5JQQJ99CBACYeBjFXJ3w3AAAAACOG2KhK"


ANALYZER_ID = "prebuilt-documentSearch"


TABLE_TAGS = ("<table", "<tr", "<th", "<td", "</table", "</tr", "</th", "</td")


def parse_html_table(html_lines: List[str]):
    html = "\n".join(html_lines).strip()
    if not html:
        return [], []

    wrapped_html = f"<table>{html}</table>"
    soup = BeautifulSoup(wrapped_html, "html.parser")

    headers = []
    rows = []

    tr_tags = soup.find_all("tr")

    if tr_tags:
        for tr in tr_tags:
            ths = tr.find_all("th")
            if ths and not headers:
                headers = [th.get_text(" ", strip=True) for th in ths]
                continue

            cells = tr.find_all(["td", "th"])
            if cells:
                rows.append([c.get_text(" ", strip=True) for c in cells])

        return headers, rows

    ths = soup.find_all("th")
    tds = soup.find_all("td")

    if ths and not headers:
        headers = [th.get_text(" ", strip=True) for th in ths]

    if tds:
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

    for raw_line in lines:
        line = raw_line.strip()

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

        if line.startswith("<!--"):
            continue

        if not line:
            continue

        if line.startswith(TABLE_TAGS):
            inside_table = True
            table_buffer.append(line)
            continue

        if inside_table and not line.startswith(TABLE_TAGS):
            block_counter = flush_table_block(current_page, table_buffer, block_counter)
            table_buffer = []
            inside_table = False

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

        if line.startswith("!["):
            continue

        current_page["blocks"].append({
            "block_id": f"p_{block_counter}",
            "block_type": "paragraph",
            "text": re.sub(r"<[^>]+>", "", line).strip(),
            "bbox": None
        })
        block_counter += 1

    if table_buffer:
        block_counter = flush_table_block(current_page, table_buffer, block_counter)

    pages.append(current_page)

    return {
        "document_name": source_name,
        "model": "content-understanding",
        "pages": pages
    }


def analyze_document_from_url(file_url: str, output_json_path: str) -> Dict[str, Any]:
    client = ContentUnderstandingClient(
        endpoint=AZURE_CONTENT_UNDERSTANDING_ENDPOINT,
        credential=AzureKeyCredential(AZURE_CONTENT_UNDERSTANDING_KEY),
    )

    print(f"\n🚀 Analyzing URL: {file_url}")

    poller = client.begin_analyze(
        analyzer_id=ANALYZER_ID,
        inputs=[AnalyzeInput(url=file_url)]
    )

    result = poller.result()

    raw_result = result.as_dict()

    markdown = ""
    if getattr(result, "contents", None):
        first_content = result.contents[0]
        markdown = getattr(first_content, "markdown", "") or ""

    structured_output = markdown_to_structured_layout(
        markdown=markdown,
        source_name=file_url
    )

    final_output = {
        "file_url": file_url,
        "analyzer_id": ANALYZER_ID,
        "markdown": markdown,
        "raw_response": raw_result,
        "structured_output": structured_output
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved output to: {output_json_path}")
    return final_output


if __name__ == "__main__":
    FILE_URL = "https://clinicalbase.blob.core.windows.net/clinicalbase/raw/OCU410-101_Protocol.pdf?sp=r&st=2026-04-06T11:52:44Z&se=2026-04-06T20:07:44Z&spr=https&sv=2024-11-04&sr=b&sig=flnkFzLCXcR0UM5BcuHLwL4V73R6%2FMO8XtzPvlg7CM8%3D"
    OUTPUT_JSON = "cu_output.json"

    try:
        output = analyze_document_from_url(FILE_URL, OUTPUT_JSON)

        print("\n" + "=" * 50)
        print("MARKDOWN PREVIEW")
        print("=" * 50)
        print((output["markdown"] or "")[:2000])

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)