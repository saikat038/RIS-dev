from content_understanding_client import AzureContentUnderstandingClient
import os

AZURE_AI_ENDPOINT = ""
AZURE_AI_API_KEY = ""
API_VERSION = "2025-11-01"

client = AzureContentUnderstandingClient(
    endpoint=AZURE_AI_ENDPOINT,
    api_version=API_VERSION,
    subscription_key=AZURE_AI_API_KEY
)

file_path = r"C:\Users\SaikatSome\Downloads\miriyala\OCU400-Protocol, SAP & Interim CSR\test files\cropped protocol.pdf"

response = client.begin_analyze_binary(
    analyzer_id="RIS_analyzer",
    file_location=file_path
)

result = client.poll_result(response, timeout_seconds=1200)

import json

output_path = "content_understanding_result.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"Result saved to {output_path}")