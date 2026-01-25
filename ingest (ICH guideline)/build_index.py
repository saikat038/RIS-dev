# # ingest/build_index.py
# """
# Builds a FAISS vector index from text or markdown documents
# stored in Azure Blob Storage. Each document in 'raw/' is:
#   1. Downloaded
#   2. Split into chunks
#   3. Embedded using Azure OpenAI
#   4. Saved locally as FAISS + meta.json
#   5. Uploaded back to Blob under 'index/'
# """

# import os, sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import json
# import faiss
# import numpy as np
# from azure.storage.blob import ContainerClient, BlobClient
# from config.settings import (
#     AZURE_BLOB_CONN_STRING,
#     BLOB_CONTAINER,
#     RAW_PREFIX,
#     INDEX_PREFIX,
#     LOCAL_INDEX_DIR,
# )
# from ingest.chunk import split_into_chunks
# from ingest.embed import batch_embed


# def list_raw_text_blobs():
#     """
#     Lists all text or markdown blobs in the 'raw/' folder of the Azure container.
#     Yields blob names that match extensions like .txt or .pdf.
#     """
#     cc = ContainerClient.from_connection_string(AZURE_BLOB_CONN_STRING, BLOB_CONTAINER)
#     for b in cc.list_blobs(name_starts_with=RAW_PREFIX):
#         if b.name.lower().endswith((".txt", ".pdf")):
#             yield b.name


# def download_text(blob_name: str) -> str:
#     """
#     Downloads the full content of a blob (text file) as a string.
#     Args:
#         blob_name (str): the name/path of the blob inside the container.
#     Returns:
#         str: text content of the blob.
#     """
#     bc = BlobClient.from_connection_string(
#         AZURE_BLOB_CONN_STRING, BLOB_CONTAINER, blob_name
#     )
#     return bc.download_blob().content_as_text()


# def save_index(vectors: np.ndarray, metas: list[dict]):
#     """
#     Saves a FAISS index and its associated metadata locally.
#     Args:
#         vectors: numpy array of embeddings.
#         metas: list of metadata dictionaries aligned with each vector.
#     """
#     # Get embedding dimension
#     dim = vectors.shape[1]

#     # Normalize vectors (required for cosine similarity)
#     faiss.normalize_L2(vectors)

#     # Create FAISS index for inner-product search (cosine)
#     index = faiss.IndexFlatIP(dim)
#     index.add(vectors)

#     # Ensure the local index directory exists
#     os.makedirs(LOCAL_INDEX_DIR, exist_ok=True)

#     # Save FAISS index to file
#     faiss.write_index(index, os.path.join(LOCAL_INDEX_DIR, "faiss.index"))

#     # Save metadata (chunk text, doc_id, etc.) as JSON
#     with open(
#         os.path.join(LOCAL_INDEX_DIR, "meta.json"), "w", encoding="utf-8"
#     ) as f:
#         json.dump(metas, f, ensure_ascii=False, indent=2)


# def upload_index():
#     """
#     Uploads the generated FAISS index and meta.json to Azure Blob
#     under the 'index/' prefix of your container.
#     """
#     for fname in ["faiss.index", "meta.json"]:
#         path = os.path.join(LOCAL_INDEX_DIR, fname)
#         bc = BlobClient.from_connection_string(
#             AZURE_BLOB_CONN_STRING, BLOB_CONTAINER, f"{INDEX_PREFIX}{fname}"
#         )
#         with open(path, "rb") as f:
#             bc.upload_blob(f, overwrite=True)
#     print("✅ Uploaded FAISS index and meta.json to Azure Blob.")




# def main():
#     """
#     Main orchestration function.
#     - Lists text blobs in 'raw/'
#     - Downloads them and splits into chunks
#     - Generates embeddings
#     - Builds FAISS index and uploads to Blob
#     """
#     print("🔍 Building FAISS index from Azure Blob files...")
#     metas, texts = [], []

#     # Iterate through all raw files
#     for blob_name in list_raw_text_blobs():
#         doc_id = os.path.basename(blob_name)
#         print(f"📄 Processing: {doc_id}")

#         # Step 1: download content
#         text = download_text(blob_name)

#         # Step 2: split into smaller chunks
#         chunks = split_into_chunks(text)

#         # Step 3: add each chunk’s text and metadata
#         for i, ch in enumerate(chunks):
#             metas.append({"doc_id": doc_id, "page": i + 1, "text": ch})
#             texts.append(ch)

#     # Stop if no text files were found
#     if not texts:
#         print("⚠️ No text files found in raw/. Please upload some first.")
#         return

#     # Step 4: embed chunks using Azure OpenAI
#     print(f"🧠 Creating embeddings for {len(texts)} chunks...")
#     vecs = np.array(batch_embed(texts), dtype=np.float32)

#     # Step 5: save index locally
#     save_index(vecs, metas)

#     # Step 6: upload index back to Azure Blob
#     upload_index()

#     print("🎯 Index build complete!")


# # Run only if executed directly (not imported)
# if __name__ == "__main__":
#     main()













#########################################################################################

# # ingest/build_index.py
# """
# Builds a FAISS vector index from text or markdown/PDF documents
# stored in Azure Blob Storage. Each document in 'raw/' is:
#   1. Downloaded
#   2. Converted to text (PDFs are parsed)
#   3. Split into chunks
#   4. Embedded using Azure OpenAI
#   5. Saved locally as FAISS + meta.json
#   6. Uploaded back to Blob under 'index/'
# """

# import os, sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import json
# import faiss
# import numpy as np
# from io import BytesIO
# from typing import List

# from azure.storage.blob import ContainerClient, BlobClient
# from PyPDF2 import PdfReader  # NEW: for PDF text extraction

# from config.settings import (
#     AZURE_BLOB_CONN_STRING,
#     BLOB_CONTAINER,
#     RAW_PREFIX,
#     INDEX_PREFIX,
#     LOCAL_INDEX_DIR,
# )
# from ingest.chunk import split_into_chunks
# from ingest.embed import batch_embed


# def list_raw_text_blobs():
#     """
#     Lists all text or PDF blobs in the 'raw/' folder of the Azure container.
#     Yields blob names that match extensions like .txt or .pdf.
#     """
#     cc = ContainerClient.from_connection_string(AZURE_BLOB_CONN_STRING, BLOB_CONTAINER)
#     for b in cc.list_blobs(name_starts_with=RAW_PREFIX):
#         name = b.name.lower()
#         if name.endswith(".txt") or name.endswith(".pdf"):
#             yield b.name


# def _download_blob_bytes(blob_name: str) -> bytes:
#     """
#     Downloads a blob as raw bytes.
#     """
#     bc = BlobClient.from_connection_string(
#         AZURE_BLOB_CONN_STRING, BLOB_CONTAINER, blob_name
#     )
#     downloader = bc.download_blob()
#     return downloader.readall()


# def download_and_extract_text(blob_name: str) -> str:
#     """
#     Downloads the content of a blob and returns its extracted text.

#     - For .txt files: reads as UTF-8 text.
#     - For .pdf files: reads as bytes and extracts text with PyPDF2.

#     Args:
#         blob_name (str): the name/path of the blob inside the container.
#     Returns:
#         str: text content of the blob (possibly empty if parsing fails).
#     """
#     ext = os.path.splitext(blob_name)[1].lower()

#     # Text files: simple content_as_text
#     if ext == ".txt":
#         bc = BlobClient.from_connection_string(
#             AZURE_BLOB_CONN_STRING, BLOB_CONTAINER, blob_name
#         )
#         # You can change encoding if needed
#         return bc.download_blob().content_as_text(encoding="utf-8")

#     # PDF files: download as bytes and parse pages
#     if ext == ".pdf":
#         try:
#             raw_bytes = _download_blob_bytes(blob_name)
#             reader = PdfReader(BytesIO(raw_bytes))

#             pages_text: List[str] = []
#             for page in reader.pages:
#                 # extract_text() can return None
#                 page_text = page.extract_text() or ""
#                 pages_text.append(page_text)

#             full_text = "\n\n".join(pages_text).strip()
#             if not full_text:
#                 print(f"⚠️ No text extracted from PDF: {blob_name}")
#             return full_text
#         except Exception as e:
#             print(f"❌ Failed to parse PDF {blob_name}: {e}")
#             return ""

#     # Fallback: unsupported type
#     print(f"⚠️ Unsupported file type for {blob_name}, skipping.")
#     return ""


# def save_index(vectors: np.ndarray, metas: list[dict]):
#     """
#     Saves a FAISS index and its associated metadata locally.
#     Args:
#         vectors: numpy array of embeddings.
#         metas: list of metadata dictionaries aligned with each vector.
#     """
#     # Get embedding dimension
#     dim = vectors.shape[1]

#     # Normalize vectors (required for cosine similarity)
#     faiss.normalize_L2(vectors)

#     # Create FAISS index for inner-product search (cosine)
#     index = faiss.IndexFlatIP(dim)
#     index.add(vectors)

#     # Ensure the local index directory exists
#     os.makedirs(LOCAL_INDEX_DIR, exist_ok=True)

#     # Save FAISS index to file
#     faiss.write_index(index, os.path.join(LOCAL_INDEX_DIR, "faiss.index"))

#     # Save metadata (chunk text, doc_id, etc.) as JSON
#     with open(
#         os.path.join(LOCAL_INDEX_DIR, "meta.json"), "w", encoding="utf-8"
#     ) as f:
#         json.dump(metas, f, ensure_ascii=False, indent=2)


# def upload_index():
#     """
#     Uploads the generated FAISS index and meta.json to Azure Blob
#     under the 'index/' prefix of your container.
#     """
#     for fname in ["faiss.index", "meta.json"]:
#         path = os.path.join(LOCAL_INDEX_DIR, fname)
#         bc = BlobClient.from_connection_string(
#             AZURE_BLOB_CONN_STRING, BLOB_CONTAINER, f"{INDEX_PREFIX}{fname}"
#         )
#         with open(path, "rb") as f:
#             bc.upload_blob(f, overwrite=True)
#     print("✅ Uploaded FAISS index and meta.json to Azure Blob.")


# def main():
#     """
#     Main orchestration function.
#     - Lists .txt and .pdf blobs in 'raw/'
#     - Downloads them and converts to text
#     - Splits into chunks
#     - Generates embeddings
#     - Builds FAISS index and uploads to Blob
#     """
#     print("🔍 Building FAISS index from Azure Blob files...")
#     metas, texts = [], []

#     # Iterate through all raw files
#     for blob_name in list_raw_text_blobs():
#         doc_id = os.path.basename(blob_name)
#         print(f"📄 Processing: {doc_id}")

#         # Step 1: download & extract content (handles .txt and .pdf)
#         text = download_and_extract_text(blob_name)
#         if not text.strip():
#             print(f"⚠️ Skipping {doc_id} because extracted text is empty.")
#             continue

#         # Step 2: split into smaller chunks
#         chunks = split_into_chunks(text)

#         # Step 3: add each chunk’s text and metadata
#         for i, ch in enumerate(chunks):
#             metas.append({"doc_id": doc_id, "page": i + 1, "text": ch})
#             texts.append(ch)

#     # Stop if no files were processed
#     if not texts:
#         print("⚠️ No usable text found in raw/. Please upload some .txt or .pdf files.")
#         return

#     # Step 4: embed chunks using Azure OpenAI
#     print(f"🧠 Creating embeddings for {len(texts)} chunks...")
#     vecs = np.array(batch_embed(texts), dtype=np.float32)

#     # Step 5: save index locally
#     save_index(vecs, metas)

#     # Step 6: upload index back to Azure Blob
#     upload_index()

#     print("🎯 Index build complete!")


# # Run only if executed directly (not imported)
# if __name__ == "__main__":
#     main()















# # ingest/build_index.py
# """
# Builds a FAISS vector index from text, PDF, or DOCX documents
# stored in Azure Blob Storage. Each document in 'raw/' is:
#   1. Downloaded
#   2. Converted to text (PDFs/DOCX are parsed)
#   3. Split into chunks
#   4. Embedded using Azure OpenAI
#   5. Saved locally as FAISS + meta.json
#   6. Uploaded back to Blob under 'index/'
# """

# import os, sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import json
# import faiss
# import numpy as np
# from io import BytesIO
# from typing import List
# from azure.ai.formrecognizer import DocumentAnalysisClient
# from azure.core.credentials import AzureKeyCredential
# from azure.storage.blob import ContainerClient, BlobClient
# from PyPDF2 import PdfReader          # For PDF text extraction
# from docx import Document             # NEW: For DOCX text extraction

# from config.settings import (
#     AZURE_BLOB_CONN_STRING,
#     BLOB_CONTAINER,
#     RAW_PREFIX,
#     INDEX_PREFIX,
#     LOCAL_INDEX_DIR,
# )
# from ingest.chunk import split_into_chunks
# from ingest.embed import batch_embed


# def list_raw_text_blobs():
#     """
#     Lists all text, PDF, or DOCX blobs in the 'raw/' folder of the Azure container.
#     Yields blob names that match extensions like .txt, .pdf, or .docx.
#     """
#     cc = ContainerClient.from_connection_string(AZURE_BLOB_CONN_STRING, BLOB_CONTAINER)
#     for b in cc.list_blobs(name_starts_with=RAW_PREFIX):
#         name = b.name.lower()
#         if name.endswith(".txt") or name.endswith(".pdf") or name.endswith(".docx"):
#             yield b.name


# def _download_blob_bytes(blob_name: str) -> bytes:
#     """
#     Downloads a blob as raw bytes.
#     """
#     bc = BlobClient.from_connection_string(
#         AZURE_BLOB_CONN_STRING, BLOB_CONTAINER, blob_name
#     )
#     downloader = bc.download_blob()
#     return downloader.readall()


# def download_and_extract_text(blob_name: str) -> str:
#     """
#     Downloads the content of a blob and returns its extracted text.

#     - For .txt files: reads as UTF-8 text.
#     - For .pdf files: reads as bytes and extracts text with PyPDF2.
#     - For .docx files: reads as bytes and extracts paragraphs with python-docx.

#     Args:
#         blob_name (str): the name/path of the blob inside the container.
#     Returns:
#         str: text content of the blob (possibly empty if parsing fails).
#     """
#     ext = os.path.splitext(blob_name)[1].lower()

#     # TEXT FILES
#     if ext == ".txt":
#         bc = BlobClient.from_connection_string(
#             AZURE_BLOB_CONN_STRING, BLOB_CONTAINER, blob_name
#         )
#         return bc.download_blob().content_as_text(encoding="utf-8")

#     # PDF FILES
#     if ext == ".pdf":
#         try:
#             raw_bytes = _download_blob_bytes(blob_name)
#             reader = PdfReader(BytesIO(raw_bytes))

#             pages_text: List[str] = []
#             for page in reader.pages:
#                 page_text = page.extract_text() or ""
#                 pages_text.append(page_text)

#             full_text = "\n\n".join(pages_text).strip()
#             if not full_text:
#                 print(f"⚠️ No text extracted from PDF: {blob_name}")
#             return full_text
#         except Exception as e:
#             print(f"❌ Failed to parse PDF {blob_name}: {e}")
#             return ""

#     # DOCX FILES
#     if ext == ".docx":
#         try:
#             raw_bytes = _download_blob_bytes(blob_name)
#             doc = Document(BytesIO(raw_bytes))
#             paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
#             full_text = "\n".join(paragraphs).strip()
#             if not full_text:
#                 print(f"⚠️ No text extracted from DOCX: {blob_name}")
#             return full_text
#         except Exception as e:
#             print(f"❌ Failed to parse DOCX {blob_name}: {e}")
#             return ""

#     # UNSUPPORTED
#     print(f"⚠️ Unsupported file type for {blob_name}, skipping.")
#     return ""


# def save_index(vectors: np.ndarray, metas: list[dict]):
#     """
#     Saves a FAISS index and its associated metadata locally.
#     Args:
#         vectors: numpy array of embeddings.
#         metas: list of metadata dictionaries aligned with each vector.
#     """
#     # Get embedding dimension
#     dim = vectors.shape[1]

#     # Normalize vectors (required for cosine similarity)
#     faiss.normalize_L2(vectors)

#     # Create FAISS index for inner-product search (cosine)
#     index = faiss.IndexFlatIP(dim)
#     index.add(vectors)

#     # Ensure the local index directory exists
#     os.makedirs(LOCAL_INDEX_DIR, exist_ok=True)

#     # Save FAISS index to file
#     faiss.write_index(index, os.path.join(LOCAL_INDEX_DIR, "faiss.index"))

#     # Save metadata (chunk text, doc_id, etc.) as JSON
#     with open(
#         os.path.join(LOCAL_INDEX_DIR, "meta.json"), "w", encoding="utf-8"
#     ) as f:
#         json.dump(metas, f, ensure_ascii=False, indent=2)


# def upload_index():
#     """
#     Uploads the generated FAISS index and meta.json to Azure Blob
#     under the 'index/' prefix of your container.
#     """
#     for fname in ["faiss.index", "meta.json"]:
#         path = os.path.join(LOCAL_INDEX_DIR, fname)
#         bc = BlobClient.from_connection_string(
#             AZURE_BLOB_CONN_STRING, BLOB_CONTAINER, f"{INDEX_PREFIX}{fname}"
#         )
#         with open(path, "rb") as f:
#             bc.upload_blob(f, overwrite=True)
#     print("✅ Uploaded FAISS index and meta.json to Azure Blob.")


# def main():
#     """
#     Main orchestration function.
#     - Lists .txt, .pdf, and .docx blobs in 'raw/'
#     - Downloads them and converts to text
#     - Splits into chunks
#     - Generates embeddings
#     - Builds FAISS index and uploads to Blob
#     """
#     print("🔍 Building FAISS index from Azure Blob files...")
#     metas, texts = [], []

#     # Iterate through all raw files
#     for blob_name in list_raw_text_blobs():
#         doc_id = os.path.basename(blob_name)
#         print(f"📄 Processing: {doc_id}")

#         # Step 1: download & extract content
#         text = download_and_extract_text(blob_name)
#         if not text.strip():
#             print(f"⚠️ Skipping {doc_id} because extracted text is empty.")
#             continue

#         # Step 2: split into smaller chunks
#         chunks = split_into_chunks(text)

#         # Step 3: add each chunk’s text and metadata
#         for i, ch in enumerate(chunks):
#             metas.append({"doc_id": doc_id, "page": i + 1, "text": ch})
#             texts.append(ch)

#     # Stop if no files were processed
#     if not texts:
#         print("⚠️ No usable text found in raw/. Please upload some .txt, .pdf, or .docx files.")
#         return

#     # Step 4: embed chunks using Azure OpenAI
#     print(f"🧠 Creating embeddings for {len(texts)} chunks...")
#     vecs = np.array(batch_embed(texts), dtype=np.float32)

#     # Step 5: save index locally
#     save_index(vecs, metas)

#     # Step 6: upload index back to Azure Blob
#     upload_index()

#     print("🎯 Index build complete!")


# # Run only if executed directly (not imported)
# if __name__ == "__main__":
#     main()



















######################################################################################################################
"""
Builds an Azure AI Search vector index from text, PDF, or DOCX documents
stored in Azure Blob Storage. Each document in 'raw/' is:
  1. Downloaded
  2. Converted to text (PDFs/DOCX are parsed)
  3. Split into chunks
  4. Embedded using Azure OpenAI
  5. Uploaded directly to Azure AI Search as searchable vector documents
"""

# import os, sys, uuid
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if ROOT_DIR not in sys.path:
#     sys.path.insert(0, ROOT_DIR)



import os
import sys
import uuid
from typing import List
import json

# ------------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ------------------------------------------------------------------
# AZURE SDK IMPORTS
# ------------------------------------------------------------------
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import ContainerClient, BlobClient

# ------------------------------------------------------------------
# PROJECT IMPORTS
# ------------------------------------------------------------------
from docint_layout import extract_layout_to_structured_json
from ingest.embed import batch_embed

# ICH PIPELINE ONLY
from ich_Normalization import normalize_ich_layout_json
from ich_chunk import chunk_ich_units

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
from config.settings import (
    AZURE_BLOB_CONN_STRING,
    BLOB_CONTAINER,
    ICH_RAW_PREFIX,
    AZURE_SEARCH_SERVICE_ENDPOINT,
    AZURE_SEARCH_API_KEY,
    AZURE_ICH_SEARCH_INDEX_NAME,
)

INDEX_NAME = AZURE_ICH_SEARCH_INDEX_NAME

print("📌 Running ICH guideline ingestion")
print(f"📌 Target index: {INDEX_NAME}")

# ------------------------------------------------------------------
# BLOB HELPERS
# ------------------------------------------------------------------
def list_ich_blobs():
    cc = ContainerClient.from_connection_string(
        AZURE_BLOB_CONN_STRING, BLOB_CONTAINER
    )
    for b in cc.list_blobs(name_starts_with=ICH_RAW_PREFIX):
        if b.name.lower().endswith((".txt", ".pdf", ".docx")):
            yield b.name


def download_blob_bytes(blob_name: str) -> bytes:
    bc = BlobClient.from_connection_string(
        AZURE_BLOB_CONN_STRING, BLOB_CONTAINER, blob_name
    )
    return bc.download_blob().readall()

# ------------------------------------------------------------------
# DOCUMENT EXTRACTION
# ------------------------------------------------------------------
def download_and_extract_document(blob_name: str) -> dict:
    ext = os.path.splitext(blob_name)[1].lower()
    source_name = os.path.basename(blob_name)

    if ext == ".txt":
        bc = BlobClient.from_connection_string(
            AZURE_BLOB_CONN_STRING, BLOB_CONTAINER, blob_name
        )
        text = bc.download_blob().content_as_text(encoding="utf-8")

        return {
            "document_name": source_name,
            "model": "plain-text",
            "pages": [
                {
                    "page_number": 1,
                    "blocks": [
                        {
                            "block_id": "p1_line_0",
                            "block_type": "paragraph",
                            "text": text,
                            "bbox": None,
                        }
                    ],
                }
            ],
        }

    if ext in [".pdf", ".docx"]:
        raw_bytes = download_blob_bytes(blob_name)
        return extract_layout_to_structured_json(raw_bytes, source_name)

    raise ValueError(f"Unsupported file type: {ext}")

# ------------------------------------------------------------------
# MAIN INGESTION
# ------------------------------------------------------------------
def main():
    print("🔍 Starting ICH ingestion...")

    search_client = SearchClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
    )

    documents_to_upload: List[dict] = []

    for blob_name in list_ich_blobs():
        doc_id = os.path.basename(blob_name)
        print(f"\n📄 Processing ICH file: {doc_id}")

        structured_doc = download_and_extract_document(blob_name)

        if not structured_doc or not structured_doc.get("pages"):
            print("⚠️ No pages found. Skipping.")
            continue

        # ----------------------------------------------------------
        # ICH NORMALIZATION (RULE-ATOMIC)
        # ----------------------------------------------------------
        normalized = normalize_ich_layout_json(structured_doc)

        if not normalized.get("blocks"):
            print("⚠️ No ICH rules detected. Skipping.")
            continue

        output_json = "ICH_layout_semantic.json"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)

        # ----------------------------------------------------------
        # ICH CHUNKING (1 RULE = 1 CHUNK)
        # ----------------------------------------------------------
        chunks = chunk_ich_units(normalized)

        if not chunks:
            print("⚠️ No chunks produced. Skipping.")
            continue

        print(f"🧠 Embedding {len(chunks)} ICH rules...")

        texts = [c["text"] for c in chunks]
        embeddings = batch_embed(texts)

        # ----------------------------------------------------------
        # BUILD SEARCH DOCUMENTS
        # ----------------------------------------------------------
        for chunk, vector in zip(chunks, embeddings):
            doc = {
                "id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "source_type": "ich",
                "guideline": chunk["metadata"].get("guideline"),
                "section_path": chunk["metadata"].get("section_path"),
                "rule_type": chunk["metadata"].get("rule_type"),
                "text": chunk["text"],
                "vector": vector,
            }

            documents_to_upload.append(doc)

    # --------------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------------
    if documents_to_upload:
        print(f"\n🚀 Uploading {len(documents_to_upload)} ICH rules...")
        result = search_client.upload_documents(documents_to_upload)
        print("✅ ICH ingestion complete")
        print(result)
    else:
        print("⚠️ No ICH documents to upload")

# ------------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------------
if __name__ == "__main__":
    main()