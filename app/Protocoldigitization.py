from io import BytesIO
from docxtpl import DocxTemplate
from azure.storage.blob import BlobServiceClient

from config.settings import (
    AZURE_BLOB_CONN_STRING,
    BLOB_CONTAINER,
    INDEX_PREFIX,
)

TEMPLATE_NAME = "CRS.docx"
OUTPUT_NAME = "CRS_filled.docx"


def render_crs_docx_in_memory(llm_text: str):
    # 1️⃣ Connect to Blob Storage using connection string
    blob_service = BlobServiceClient.from_connection_string(
        AZURE_BLOB_CONN_STRING
    )
    container = blob_service.get_container_client(BLOB_CONTAINER)

    # 2️⃣ Download CRS.docx template into RAM
    template_blob_path = f"{INDEX_PREFIX}/{TEMPLATE_NAME}"
    template_blob = container.get_blob_client(template_blob_path)

    template_bytes = template_blob.download_blob().readall()
    template_stream = BytesIO(template_bytes)

    # 3️⃣ Load template into docxtpl (from RAM)
    doc = DocxTemplate(template_stream)

    context = {
        "inclusion_criterion": llm_text
    }

    doc.render(context)

    # 4️⃣ Save rendered DOCX into RAM
    output_stream = BytesIO()
    doc.save(output_stream)
    output_stream.seek(0)

    # 5️⃣ Upload rendered DOCX back to Blob (raw/)
    output_blob_path = f"{INDEX_PREFIX}/{OUTPUT_NAME}"
    output_blob = container.get_blob_client(output_blob_path)

    output_blob.upload_blob(output_stream, overwrite=True)

    print(f"☁️ CRS rendered & uploaded → {output_blob_path}")