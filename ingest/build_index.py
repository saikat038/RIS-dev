# ingest/build_index.py
"""
Builds a FAISS vector index from text or markdown documents
stored in Azure Blob Storage. Each document in 'raw/' is:
  1. Downloaded
  2. Split into chunks
  3. Embedded using Azure OpenAI
  4. Saved locally as FAISS + meta.json
  5. Uploaded back to Blob under 'index/'
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import faiss
import numpy as np
from azure.storage.blob import ContainerClient, BlobClient
from config.settings import (
    AZURE_BLOB_CONN_STRING,
    BLOB_CONTAINER,
    RAW_PREFIX,
    INDEX_PREFIX,
    LOCAL_INDEX_DIR,
)
from ingest.chunk import split_into_chunks
from ingest.embed import batch_embed


def list_raw_text_blobs():
    """
    Lists all text or markdown blobs in the 'raw/' folder of the Azure container.
    Yields blob names that match extensions like .txt or .md.
    """
    cc = ContainerClient.from_connection_string(AZURE_BLOB_CONN_STRING, BLOB_CONTAINER)
    for b in cc.list_blobs(name_starts_with=RAW_PREFIX):
        if b.name.lower().endswith((".txt", ".md")):
            yield b.name


def download_text(blob_name: str) -> str:
    """
    Downloads the full content of a blob (text file) as a string.
    Args:
        blob_name (str): the name/path of the blob inside the container.
    Returns:
        str: text content of the blob.
    """
    bc = BlobClient.from_connection_string(
        AZURE_BLOB_CONN_STRING, BLOB_CONTAINER, blob_name
    )
    return bc.download_blob().content_as_text()


def save_index(vectors: np.ndarray, metas: list[dict]):
    """
    Saves a FAISS index and its associated metadata locally.
    Args:
        vectors: numpy array of embeddings.
        metas: list of metadata dictionaries aligned with each vector.
    """
    # Get embedding dimension
    dim = vectors.shape[1]

    # Normalize vectors (required for cosine similarity)
    faiss.normalize_L2(vectors)

    # Create FAISS index for inner-product search (cosine)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Ensure the local index directory exists
    os.makedirs(LOCAL_INDEX_DIR, exist_ok=True)

    # Save FAISS index to file
    faiss.write_index(index, os.path.join(LOCAL_INDEX_DIR, "faiss.index"))

    # Save metadata (chunk text, doc_id, etc.) as JSON
    with open(
        os.path.join(LOCAL_INDEX_DIR, "meta.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)


def upload_index():
    """
    Uploads the generated FAISS index and meta.json to Azure Blob
    under the 'index/' prefix of your container.
    """
    for fname in ["faiss.index", "meta.json"]:
        path = os.path.join(LOCAL_INDEX_DIR, fname)
        bc = BlobClient.from_connection_string(
            AZURE_BLOB_CONN_STRING, BLOB_CONTAINER, f"{INDEX_PREFIX}{fname}"
        )
        with open(path, "rb") as f:
            bc.upload_blob(f, overwrite=True)
    print("✅ Uploaded FAISS index and meta.json to Azure Blob.")




def main():
    """
    Main orchestration function.
    - Lists text blobs in 'raw/'
    - Downloads them and splits into chunks
    - Generates embeddings
    - Builds FAISS index and uploads to Blob
    """
    print("🔍 Building FAISS index from Azure Blob files...")
    metas, texts = [], []

    # Iterate through all raw files
    for blob_name in list_raw_text_blobs():
        doc_id = os.path.basename(blob_name)
        print(f"📄 Processing: {doc_id}")

        # Step 1: download content
        text = download_text(blob_name)

        # Step 2: split into smaller chunks
        chunks = split_into_chunks(text)

        # Step 3: add each chunk’s text and metadata
        for i, ch in enumerate(chunks):
            metas.append({"doc_id": doc_id, "page": i + 1, "text": ch})
            texts.append(ch)

    # Stop if no text files were found
    if not texts:
        print("⚠️ No text files found in raw/. Please upload some first.")
        return

    # Step 4: embed chunks using Azure OpenAI
    print(f"🧠 Creating embeddings for {len(texts)} chunks...")
    vecs = np.array(batch_embed(texts), dtype=np.float32)

    # Step 5: save index locally
    save_index(vecs, metas)

    # Step 6: upload index back to Azure Blob
    upload_index()

    print("🎯 Index build complete!")


# Run only if executed directly (not imported)
if __name__ == "__main__":
    main()
