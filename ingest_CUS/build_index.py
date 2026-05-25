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

import os, sys, uuid
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)



import json
import numpy as np
from io import BytesIO
from typing import List

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import ContainerClient, BlobClient
from PyPDF2 import PdfReader
from docx import Document
from docint_layout import *
from semantic_chunk import *
from Semantic_Normalization import *

from config.settings import (
    AZURE_BLOB_CONN_STRING,
    BLOB_CONTAINER,
    RAW_PREFIX,
    AZURE_SEARCH_SERVICE_ENDPOINT,
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_INDEX_NAME,
)
from ingest.embed import batch_embed



# ----------------------------
# LIST RAW FILES IN BLOB
# ----------------------------
def list_raw_text_blobs():
    cc = ContainerClient.from_connection_string(AZURE_BLOB_CONN_STRING, BLOB_CONTAINER)
    for b in cc.list_blobs(name_starts_with=RAW_PREFIX):
        name = b.name.lower()
        if name.endswith(".txt") or name.endswith(".pdf") or name.endswith(".docx"):
            yield b.name




# ----------------------------
# DOWNLOAD BYTES FROM BLOB
# ----------------------------
def _download_blob_bytes(blob_name: str) -> bytes:
    bc = BlobClient.from_connection_string(
        AZURE_BLOB_CONN_STRING, BLOB_CONTAINER, blob_name
    )
    return bc.download_blob().readall()




# ----------------------------
# MAIN TEXT EXTRACTION LOGIC
# ----------------------------
def download_and_extract_document(blob_name: str) -> dict:
    ext = os.path.splitext(blob_name)[1].lower()
    source_name = os.path.basename(blob_name)

    # -------- TXT (wrap as structured doc) --------
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
                    "width": None,
                    "height": None,
                    "unit": None,
                    "blocks": [
                        {
                            "block_id": "p1_line_0",
                            "block_type": "paragraph",
                            "text": text,
                            "bbox": None,
                            "confidence": None
                        }
                    ]
                }
            ]
        }

    # -------- PDF / DOCX (Doc Intelligence) --------
    if ext in [".pdf", ".docx"]:
        raw_bytes = _download_blob_bytes(blob_name)

        structured_doc = extract_layout_to_structured_json(raw_bytes, source_name)

        return structured_doc

    # -------- Unsupported --------
    raise ValueError(f"Unsupported file type: {ext}")






# ----------------------------
# MAIN INGESTION SCRIPT
# ----------------------------
def main():
    print("🔍 Building Azure AI Search Vector Index from documents...")

    search_client = SearchClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
    )


    documents_to_upload = []

    for blob_name in list_raw_text_blobs():
        doc_id = os.path.basename(blob_name)
        print(f"📄 Processing: {doc_id}")

        # Step 1: Download + OCR (STRUCTURED)
        structured_doc = download_and_extract_document(blob_name)

        if not structured_doc or not structured_doc.get("pages"):
            print(f"⚠️ Skipping {doc_id}: No content found.")
            continue

        # Step 2: Semantic normalization
        normalized_doc = normalize_layout_json(structured_doc)

        if not normalized_doc.get("blocks"):
            print(f"⚠️ Skipping {doc_id}: No semantic blocks.")
            continue

        output_json = "sample_layout_semantic.json"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(normalized_doc, f, ensure_ascii=False, indent=2)

        # Step 3: Semantic chunking
        chunks = chunk_semantic_blocks(normalized_doc)

        if not chunks:
            print(f"⚠️ Skipping {doc_id}: No chunks produced.")
            continue

        print(f"🧠 Embedding {len(chunks)} chunks for {doc_id}...")

        # Step 4: Embed chunk texts
        texts_to_embed = [c["text"] for c in chunks]
        embeddings = batch_embed(texts_to_embed)

        # Step 5: Build Azure Search documents
        for chunk, vector in zip(chunks, embeddings):

            # ---- normalize page numbers ----
            page_numbers = chunk["metadata"].get("page_numbers") or []
            if isinstance(page_numbers, int):
                page_numbers = [page_numbers]

            # ---- normalize heading_path (MUST be string) ----
            heading_path = chunk["metadata"].get("heading_path")
            if isinstance(heading_path, list):
                heading_path = " > ".join(heading_path)
            elif heading_path is None:
                heading_path = ""

            doc = {
                "id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "text": str(chunk.get("text") or ""),
                "chunk_type": str(chunk.get("chunk_type") or ""),
                "heading_path": heading_path,
                "page_numbers": page_numbers,
                "source_block_ids": chunk["metadata"].get("source_block_ids") or [],
                "vector": vector,
            }

            # ============================
            # TABLE-SPECIFIC FIELDS
            # ============================
            if chunk["chunk_type"] == "table":
                rows = chunk["metadata"].get("rows") or []

                # Azure index expects table_rows as STRING
                table_rows_str = "\n".join(
                    json.dumps(row, ensure_ascii=False)
                    for row in rows
                    if isinstance(row, (list, dict)) and row
                )

                doc.update({
                    "table_context_heading": str(chunk["metadata"].get("table_context_heading") or ""),
                    "table_context_text": str(chunk["metadata"].get("table_context_text") or ""),
                    "table_semantic_hint": str(chunk["metadata"].get("table_semantic_hint") or ""),
                    "table_headers": [
                        str(h) for h in (chunk["metadata"].get("headers") or [])
                    ],
                    "table_rows": table_rows_str,   # ✅ STRING ONLY
                })

            documents_to_upload.append(doc)





    # Final upload with retry
    import time
    if documents_to_upload:
        print(f"🚀 Uploading {len(documents_to_upload)} chunks...")
        
        batch_size = 100
        total_uploaded = 0
        
        for i in range(0, len(documents_to_upload), batch_size):
            batch = documents_to_upload[i:i + batch_size]
            for attempt in range(3):  # retry up to 3 times
                try:
                    result = search_client.upload_documents(batch)
                    total_uploaded += len(batch)
                    print(f"  Batch {i//batch_size + 1} uploaded ({len(batch)} chunks) — total: {total_uploaded}")
                    break
                except Exception as e:
                    if attempt < 2:
                        sleep = 2 ** attempt
                        print(f"  Retry {attempt+1}/3 after {sleep}s: {e}")
                        time.sleep(sleep)
                    else:
                        print(f"  Failed batch {i//batch_size + 1} ({len(batch)} chunks): {e}")
                        # Decide: raise or continue
                        raise

# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    main()