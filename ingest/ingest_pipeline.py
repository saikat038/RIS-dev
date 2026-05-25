import os
import sys
import uuid
import json
import re

# -----------------------------
# FIX IMPORT PATH
# -----------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# -----------------------------
# AZURE IMPORTS
# -----------------------------
from azure.storage.blob import ContainerClient, BlobClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# -----------------------------
# YOUR MODULES
# -----------------------------
from ingest.pdf_parser import process_pdf
from ingest.embed import batch_embed


# =============================
# CONFIG (ingest_pipeline.py)
# =============================
AZURE_BLOB_CONN_STRING = os.getenv("AZURE_BLOB_CONN_STRING")
BLOB_CONTAINER = os.getenv("BLOB_CONTAINER")
RAW_PREFIX = os.getenv("RAW_PREFIX")

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY")
SEARCH_INDEX = "csr-index"

TEMP_DIR = "temp_downloads"
os.makedirs(TEMP_DIR, exist_ok=True)


# =============================
# LIST BLOBS (ingest_pipeline.py)
# =============================
def list_blobs():
    cc = ContainerClient.from_connection_string(
        AZURE_BLOB_CONN_STRING, BLOB_CONTAINER
    )
    for blob in cc.list_blobs(name_starts_with=RAW_PREFIX):
        if blob.name.lower().endswith(".pdf"):
            yield blob.name


# =============================
# DOWNLOAD FILE (ingest_pipeline.py)
# =============================
def download_blob(blob_name: str) -> str:

    bc = BlobClient.from_connection_string(
        AZURE_BLOB_CONN_STRING,
        BLOB_CONTAINER,
        blob_name
    )

    local_path = os.path.join(TEMP_DIR, os.path.basename(blob_name))

    with open(local_path, "wb") as f:
        f.write(bc.download_blob().readall())

    return local_path


# =============================
# SECTION EXTRACTION (ingest_pipeline.py)
# =============================
def extract_section_info(text, file_name):
    """
    Extract section + subsection from:
    1. content
    2. fallback: filename
    """

    match = re.search(r"\d+(\.\d+)+", text)
    if match:
        subsection = match.group(0)
        section = subsection.split(".")[0]
        return section, subsection

    match = re.search(r"\d+(\.\d+)+", file_name)
    if match:
        subsection = match.group(0)
        section = subsection.split(".")[0]
        return section, subsection

    return "", ""


def extract_section_title(text):
    """
    Extract first meaningful line
    """

    for line in text.split("\n"):
        clean = line.strip()
        if len(clean) > 5 and len(clean) < 120:
            return clean

    return ""


# =============================
# KEYWORD EXTRACTION (ingest_pipeline.py)
# =============================
def extract_keywords(text):

    patterns = [
        r"\bMean\b", r"\bSD\b", r"\bMedian\b",
        r"\bBaseline\b", r"\bDay\s*\d+\b",
        r"\bMonth\s*\d+\b", r"\bCohort\s*\d+\b",
        r"\bTotal\b"
    ]

    keywords = []

    for p in patterns:
        keywords.extend(re.findall(p, text, re.IGNORECASE))

    return list(set([k.strip() for k in keywords]))


# =============================
# TIMEPOINT EXTRACTION (ingest_pipeline.py)
# =============================
def extract_timepoint(text):

    match = re.search(r"(Baseline|Day\s*\d+|Month\s*\d+)", text, re.IGNORECASE)
    return match.group(0) if match else ""


# =============================
# SPLIT TABLE LOGICALLY (ingest_pipeline.py)
# =============================
def split_table_blocks(block):

    text = block["content"]

    parts = re.split(r"(Baseline|Day\s*\d+|Month\s*\d+)", text)

    results = []

    for i in range(1, len(parts), 2):
        tp = parts[i]
        content = parts[i] + " " + (parts[i+1] if i+1 < len(parts) else "")

        results.append({
            "content": content.strip(),
            "page": block["page"],
            "type": "table",
            "timepoint": tp
        })

    return results if results else [block]


# =============================
# CONVERT PARSER OUTPUT (ingest_pipeline.py)
# =============================
def convert_parser_to_blocks(parsed_json):

    blocks = []

    # TEXT BLOCKS
    for page in parsed_json.get("text_pages", []):
        text = page.get("text", "")
        if text.strip():
            blocks.append({
                "content": text.strip(),
                "page": page.get("page_number", 0),
                "type": "text"
            })

    # TABLE BLOCKS (CLEAN FORMAT)
    for table in parsed_json.get("tables", []):
        rows = table.get("data", [])

        table_text = "\n".join(
            [
                " | ".join([str(cell).strip() if cell else "" for cell in row])
                for row in rows if any(row)
            ]
        )

        if table_text.strip():
            blocks.append({
                "content": table_text,
                "page": table.get("page_number", 0),
                "type": "table",
                "table": rows
            })

    return blocks


# =============================
# BUILD CHUNKS + ENRICHMENT (ingest_pipeline.py)
# =============================
def build_chunks(blocks, file_name):

    final_chunks = []

    for block in blocks:

        if block["type"] == "table":
            table_chunks = split_table_blocks(block)
        else:
            table_chunks = [block]

        for chunk in table_chunks:

            text = chunk["content"]

            section, subsection = extract_section_info(text, file_name)
            title = extract_section_title(text)

            enriched = {
                **chunk,
                "section": section,
                "subsection": subsection,
                "section_title": title,
                "keywords": extract_keywords(text),
                "timepoint": extract_timepoint(text)
            }

            final_chunks.append(enriched)

    return final_chunks


# =============================
# BUILD AZURE DOCS (ingest_pipeline.py)
# =============================
def build_search_documents(chunks, embeddings, file_name):

    docs = []

    for i, chunk in enumerate(chunks):

        doc = {
            "chunk_id": str(uuid.uuid4()),
            "document_id": file_name,
            "file_name": file_name,
            "page_number": chunk.get("page", 0),

            "section": chunk.get("section", ""),
            "subsection": chunk.get("subsection", ""),
            "section_title": chunk.get("section_title", ""),

            "chunk_text": chunk["content"],
            "keywords": chunk.get("keywords", []),
            "chunk_type": chunk.get("type", "text"),

            "table_id": "",
            "table_data": json.dumps(chunk.get("table", "")) if chunk.get("table") else "",

            "vector": embeddings[i]
        }

        docs.append(doc)

    return docs


# =============================
# MAIN INGESTION (ingest_pipeline.py)
# =============================
def ingest():

    print("🚀 Starting FULL ingestion (with structure extraction)")

    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX,
        credential=AzureKeyCredential(SEARCH_KEY),
    )

    all_docs = []

    for blob_name in list_blobs():

        print(f"\n📄 Processing: {blob_name}")

        try:
            pdf_path = download_blob(blob_name)

            parsed = process_pdf(pdf_path)

            if not parsed:
                print("⚠️ Empty parser output")
                continue

            blocks = convert_parser_to_blocks(parsed)
            print("Blocks:", len(blocks))

            if not blocks:
                continue

            chunks = build_chunks(blocks, blob_name)
            print("Chunks:", len(chunks))

            texts = [c["content"] for c in chunks]

            embeddings = batch_embed(texts)

            docs = build_search_documents(chunks, embeddings, blob_name)
            print("Docs:", len(docs))

            all_docs.extend(docs)

        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    # -----------------------------
    # UPLOAD
    # -----------------------------
    print(f"\n⬆️ Uploading {len(all_docs)} docs...")

    if not all_docs:
        print("❌ No docs to upload")
        return

    batch_size = 500

    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i:i+batch_size]

        results = search_client.upload_documents(documents=batch)
        success = sum(r.succeeded for r in results)

        print(f"✅ Uploaded {success}/{len(batch)}")

    print("\n🎯 INGESTION COMPLETE ✅")


# =============================
# ENTRY (ingest_pipeline.py)
# =============================
if __name__ == "__main__":
    ingest()