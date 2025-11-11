# test_blob_connect.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from azure.storage.blob import BlobServiceClient
from config.settings import AZURE_BLOB_CONN_STRING, BLOB_CONTAINER

# Connect
svc = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN_STRING)
print("✅ Connected to storage account")

# List containers
containers = [c.name for c in svc.list_containers()]
print("Available containers:", containers)

# Ensure the configured one exists
if BLOB_CONTAINER in containers:
    print(f"✅ Container '{BLOB_CONTAINER}' is accessible.")
else:
    print(f"⚠️ Container '{BLOB_CONTAINER}' not found.")
