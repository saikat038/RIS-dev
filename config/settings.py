import os
from dotenv import load_dotenv
import streamlit as st

# Load .env from same folder
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))



# # openAI rag model
# AZURE_OPENAI_CHAT_ENDPOINT = os.getenv("OPEN_AI_ENDPOINT")
# AZURE_OPENAI_CHAT_API_KEY = os.getenv("OPEN_AI_KEY")
# AZURE_OPENAI_API_CHAT_VERSION = os.getenv("OPEN_AI_DEPLOYMENT_VERSION")
# AZURE_OPENAI_CHAT_MODEL = os.getenv("OPEN_AI_DEPLOYMENT_NAME")


# # openAI embedding model
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
# AZURE_OPENAI_EMBED_MODEL = os.getenv("AZURE_OPENAI_EMBED_MODEL")


# # Local folder to cache FAISS index and metadata before upload
# LOCAL_INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "index")
# AZURE_BLOB_CONN_STRING = os.getenv("AZURE_BLOB_CONN_STRING")
# BLOB_CONTAINER = os.getenv("BLOB_CONTAINER", "clinicalbase")
# RAW_PREFIX = os.getenv("RAW_PREFIX", "raw/")
# INDEX_PREFIX = os.getenv("INDEX_PREFIX", "index/")


# # Docint
# AZURE_DOCINT_ENDPOINT = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
# AZURE_DOCINT_KEY = os.getenv("DOC_INTELLIGENCE_KEY")
# AZURE_DOCINT_MODEL_ID = "prebuilt-layout"


# # azure AI search
# AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
# AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
# AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")


# openAI rag model
AZURE_OPENAI_CHAT_ENDPOINT = st.secrets["OPEN_AI_ENDPOINT"]
AZURE_OPENAI_CHAT_API_KEY = st.secrets["OPEN_AI_KEY"]
AZURE_OPENAI_API_CHAT_VERSION = st.secrets["OPEN_AI_DEPLOYMENT_VERSION"]
AZURE_OPENAI_CHAT_MODEL = st.secrets["OPEN_AI_DEPLOYMENT_NAME"]


# openAI embedding model
AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]
AZURE_OPENAI_EMBED_MODEL = st.secrets["AZURE_OPENAI_EMBED_MODEL"]


# Local folder to cache FAISS index and metadata before upload
LOCAL_INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "index")
AZURE_BLOB_CONN_STRING = st.secrets["AZURE_BLOB_CONN_STRING"]
BLOB_CONTAINER = st.secrets["BLOB_CONTAINER"]
RAW_PREFIX = st.secrets["RAW_PREFIX", "raw/"]
INDEX_PREFIX = st.secrets["INDEX_PREFIX", "index/"]


# Docint
AZURE_DOCINT_ENDPOINT = st.secrets["DOC_INTELLIGENCE_ENDPOINT"]
AZURE_DOCINT_KEY = st.secrets["DOC_INTELLIGENCE_KEY"]
AZURE_DOCINT_MODEL_ID = "prebuilt-layout"


# azure AI search
AZURE_SEARCH_SERVICE_ENDPOINT = st.secrets["AZURE_SEARCH_SERVICE_ENDPOINT"]
AZURE_SEARCH_INDEX_NAME = st.secrets["AZURE_SEARCH_INDEX_NAME"]
AZURE_SEARCH_API_KEY = st.secrets["AZURE_SEARCH_API_KEY"]