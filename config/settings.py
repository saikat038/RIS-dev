import os
from dotenv import load_dotenv
import streamlit as st
from typing import Optional

# Load .env from same folder
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))


def get_secret(key: str, default: Optional[str] = None) -> str:
    """
    Retrieves a secret, prioritizing st.secrets (Streamlit Cloud) 
    and falling back to os.getenv() (Local .env).
    
    Raises EnvironmentError if a required key is missing and no default is provided.
    """
    # 1. Try Streamlit Secrets (for cloud deployment via secrets.toml)
    if key in st.secrets:
        return st.secrets[key]
    
    # 2. Fallback to standard environment variable (for local .env access)
    value = os.getenv(key)
    
    if value is None and default is None:
        # Halt execution and log the error if a critical key is missing
        error_msg = f"FATAL: Missing required configuration key: {key}"
        st.error(error_msg)
        raise EnvironmentError(error_msg)
        
    return value if value is not None else default


# --- Application Configuration Variables ---

# ===============================================
# 1. Azure OpenAI RAG Model (Chat/Completions)
# ===============================================

# Keys: OPEN_AI_ENDPOINT, OPEN_AI_KEY, OPEN_AI_DEPLOYMENT_VERSION, OPEN_AI_DEPLOYMENT_NAME
AZURE_OPENAI_CHAT_ENDPOINT = get_secret("OPEN_AI_ENDPOINT")
AZURE_OPENAI_CHAT_API_KEY = get_secret("OPEN_AI_KEY")
AZURE_OPENAI_CHAT_API_VERSION = get_secret("OPEN_AI_DEPLOYMENT_VERSION")
AZURE_OPENAI_CHAT_MODEL = get_secret("OPEN_AI_DEPLOYMENT_NAME")


# ===============================================
# 2. Azure OpenAI Embedding Model
# ===============================================

# Keys: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_EMBED_MODEL
AZURE_OPENAI_ENDPOINT = get_secret("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = get_secret("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = get_secret("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBED_MODEL = get_secret("AZURE_OPENAI_EMBED_MODEL")


# ===============================================
# 3. Azure Blob Storage
# ===============================================

# Keys: AZURE_BLOB_CONN_STRING, BLOB_CONTAINER, RAW_PREFIX, INDEX_PREFIX
AZURE_BLOB_CONN_STRING = get_secret("AZURE_BLOB_CONN_STRING")

# Using get_secret with a default value (like os.getenv with a default)
BLOB_CONTAINER = get_secret("BLOB_CONTAINER", default="clinicalbase") 
RAW_PREFIX = get_secret("RAW_PREFIX", default="raw/")
INDEX_PREFIX = get_secret("INDEX_PREFIX", default="index/")


# ===============================================
# 4. Azure Document Intelligence (Docint)
# ===============================================

# Keys: AZURE_DOCINT_ENDPOINT, AZURE_DOCINT_KEY, AZURE_DOCINT_MODEL_ID
AZURE_DOCINT_ENDPOINT = get_secret("AZURE_DOCINT_ENDPOINT", default="https://docint-ris.cognitiveservices.azure.com/")
AZURE_DOCINT_KEY = get_secret("AZURE_DOCINT_KEY", default="BPUWJhUrdXYFuGr54ddok7I95BS0INvRz8jmYch0duYYZDgJe3flJQQJ99BKACYeBjFXJ3w3AAALACOGaBcx")
AZURE_DOCINT_MODEL_ID = get_secret("AZURE_DOCINT_MODEL_ID", default="prebuilt-layout")

##########################################################################################################################################
# openAI rag model
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
# AZURE_DOCINT_ENDPOINT = "https://docint-ris.cognitiveservices.azure.com/"
# AZURE_DOCINT_KEY = "BPUWJhUrdXYFuGr54ddok7I95BS0INvRz8jmYch0duYYZDgJe3flJQQJ99BKACYeBjFXJ3w3AAALACOGaBcx"
# AZURE_DOCINT_MODEL_ID = "prebuilt-layout"