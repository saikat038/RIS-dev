# app/vectorstore.py
import os
import json
import faiss
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from config.settings import (
    AZURE_SEARCH_SERVICE_ENDPOINT,
    AZURE_SEARCH_INDEX_NAME,
    AZURE_SEARCH_API_KEY,
)


def load_vectorstore():
  search_client = SearchClient(
      endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
      index_name=AZURE_SEARCH_INDEX_NAME,
      credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
      )

  return search_client



# def download_index():
#     """Download faiss.index and meta.json from Azure Blob to LOCAL_INDEX_DIR."""
#     os.makedirs(LOCAL_INDEX_DIR, exist_ok=True)

#     blob_service = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN_STRING)
#     container = blob_service.get_container_client(BLOB_CONTAINER)

#     for fname in ["faiss.index", "meta.json"]:
#         blob_name = f"{INDEX_PREFIX}{fname}"
#         local_path = os.path.join(LOCAL_INDEX_DIR, fname)

#         with open(local_path, "wb") as f:
#             data = container.download_blob(blob_name).readall()
#             f.write(data)


# def load_vectorstore():
#     """
#     Load FAISS index and metadata.

#     meta.json structure (from build_index.py) is:
#       [
#         {"doc_id": "...", "page": 1, "text": "..."},
#         {"doc_id": "...", "page": 2, "text": "..."},
#         ...
#       ]
#     We only need the 'text' field for RAG.
#     """
#     download_index()

#     index_path = os.path.join(LOCAL_INDEX_DIR, "faiss.index")
#     meta_path = os.path.join(LOCAL_INDEX_DIR, "meta.json")

#     index = faiss.read_index(index_path)

#     with open(meta_path, "r", encoding="utf-8") as f:
#         metas = json.load(f)          # this is a list of dicts

#     chunks = [m["text"] for m in metas]

#     # we return metas too in case you want doc_id/page later
#     return index, chunks, metas
