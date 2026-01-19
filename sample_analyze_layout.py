import json
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

# ----------------------------------
# CONFIG
# ----------------------------------

ENDPOINT = "https://docint-ris.cognitiveservices.azure.com/"   # e.g. https://<resource-name>.cognitiveservices.azure.com/
KEY = "BPUWJhUrdXYFuGr54ddok7I95BS0INvRz8jmYch0duYYZDgJe3flJQQJ99BKACYeBjFXJ3w3AAALACOGaBcx"        # your key

INPUT_PDF_PATH = r"C:\Users\SaikatSome\Downloads\OCU200 -101\OCU200 -101\Ocugen_OCU200-101_Protocol_Amendment 9 edited.pdf"
OUTPUT_JSON_PATH = "layout_raw_output.json"

# ----------------------------------
# CLIENT
# ----------------------------------

client = DocumentIntelligenceClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(KEY)
)

# ----------------------------------
# READ LOCAL FILE
# ----------------------------------

file_bytes = Path(INPUT_PDF_PATH).read_bytes()

request = AnalyzeDocumentRequest(
    bytes_source=file_bytes
)

poller = client.begin_analyze_document(
    model_id="prebuilt-layout",
    body=request
)

result = poller.result()

# ----------------------------------
# SAVE RAW JSON OUTPUT
# ----------------------------------

with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(result.as_dict(), f, indent=2, ensure_ascii=False)

print(f"✅ Layout JSON saved to: {OUTPUT_JSON_PATH}")

# ----------------------------------
# OPTIONAL: DEBUG PRINTS
# ----------------------------------

for idx, style in enumerate(result.styles or []):
    print(
        f"Document contains {'handwritten' if style.is_handwritten else 'no handwritten'} content"
    )

for page in result.pages or []:
    for line_idx, line in enumerate(page.lines or []):
        print(f"...Line #{line_idx}: {line.content}")

for table_idx, table in enumerate(result.tables or []):
    print(
        f"Table #{table_idx} → rows={table.row_count}, cols={table.column_count}"
    )
    for cell in table.cells:
        print(
            f"  Cell[{cell.row_index}][{cell.column_index}]: {cell.content}"
        )

print("----------------------------------------")