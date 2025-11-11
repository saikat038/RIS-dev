import os
from dotenv import load_dotenv

# Load .env from same folder
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# openAI
AZURE_OPENAI_ENDPOINT = os.getenv("OPEN_AI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("OPEN_AI_DEPLOYMENT_VERSION")
AZURE_OPENAI_EMBED_MODEL = os.getenv("OPEN_AI_DEPLOYMENT_NAME_GPT4O")


# Local folder to cache FAISS index and metadata before upload
LOCAL_INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "index")
AZURE_BLOB_CONN_STRING = os.getenv("AZURE_BLOB_CONN_STRING")
BLOB_CONTAINER = os.getenv("BLOB_CONTAINER", "clinicalbase")
RAW_PREFIX = os.getenv("RAW_PREFIX", "raw/")
INDEX_PREFIX = os.getenv("INDEX_PREFIX", "index/")
