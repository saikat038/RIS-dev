import numpy as np
from openai import AzureOpenAI
from app.vectorstore import load_vectorstore
from config.settings import (
    AZURE_OPENAI_CHAT_API_KEY,
    AZURE_OPENAI_CHAT_MODEL,          # <-- deployment name
    AZURE_OPENAI_CHAT_ENDPOINT,       # <-- your endpoint, e.g. https://ocugen-aoai.openai.azure.com/
    AZURE_OPENAI_API_CHAT_VERSION,     # <-- api version
    AZURE_OPENAI_API_KEY,          # <-- api key
    AZURE_OPENAI_ENDPOINT,       # <-- your endpoint, e.g. https://ocugen-aoai.openai.azure.com/
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBED_MODEL
)

# ----------------------------
# AZURE OPENAI CLIENT
# ----------------------------
client = AzureOpenAI(
    api_key=AZURE_OPENAI_CHAT_API_KEY,
    azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
    api_version=AZURE_OPENAI_API_CHAT_VERSION
)


client1 = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# Load FAISS index into memory
index, vectors, chunks = load_vectorstore()

# -------------------------------------------------------------
# EMBEDDING FUNCTION
# -------------------------------------------------------------
def embed_query(text: str):
    """Embed the query using Azure OpenAI."""
    resp = client1.embeddings.create(
        model=AZURE_OPENAI_EMBED_MODEL,      # deployment name in Azure
        input=text
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)

# -------------------------------------------------------------
# VECTOR SEARCH USING FAISS
# -------------------------------------------------------------
def search(query: str, k: int = 3):
    q_vec = embed_query(query).reshape(1, -1)
    scores, indices = index.search(q_vec, k)

    # Each match: (chunk_obj, score)
    matches = [(chunks[i], float(scores[0][j])) for j, i in enumerate(indices[0])]
    return matches


def _extract_text_from_chunk(chunk):
    """Try to extract text from a chunk that may be str or dict-like."""
    if isinstance(chunk, str):
        return chunk

    if isinstance(chunk, dict):
        for key in ("text", "content", "page_content", "chunk"):
            if key in chunk and isinstance(chunk[key], str):
                return chunk[key]
        # fallback: string representation
        return str(chunk)

    # any other type (e.g. list/tuple), fallback to str
    return str(chunk)


def answer(query: str) -> str:
    docs = search(query, k=3)

    # docs is a list of (chunk_obj, score)
    context_pieces = []
    for chunk_obj, score in docs:
        text = _extract_text_from_chunk(chunk_obj)
        if text:
            context_pieces.append(text)

    context = "\n\n".join(context_pieces) if context_pieces else "No relevant context found."

    prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer.
If something is not in context, say "Not in knowledge base."

Context:
{context}

Question: {query}
Answer:
    """

    response = client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    # new OpenAI SDK: message is an object, not dict
    return response.choices[0].message.content
