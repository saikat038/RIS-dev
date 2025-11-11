# upload_sample.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from azure.storage.blob import BlobServiceClient
from config.settings import AZURE_BLOB_CONN_STRING, BLOB_CONTAINER, RAW_PREFIX

svc = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN_STRING)
cc = svc.get_container_client(BLOB_CONTAINER)

name = RAW_PREFIX + "hello_kb.txt"   # e.g., raw/hello_kb.txt
data = b"""This is a tiny knowledge-base note.
It says: Azure Blob works and KB chat will answer from here."""

cc.upload_blob(name=name, data=data, overwrite=True)
print(f"✅ Uploaded: {name}")

print("Now in raw/:")
for b in cc.list_blobs(name_starts_with=RAW_PREFIX):
    print(" -", b.name)
