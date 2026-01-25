"""
Embedding utilities using Azure OpenAI via the AzureOpenAI SDK.
Reads credentials and deployment names from config.settings.
"""

from __future__ import annotations
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
import time

from openai import AzureOpenAI  # ✅ CORRECT CLIENT FOR AZURE

from config.settings import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBED_MODEL,   # e.g., "text-embedding-3-small"
)

# -----------------------------------------------------------
# ✅ Correct Azure OpenAI client
# -----------------------------------------------------------
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,     # MUST end with /cognitiveservices.azure.com/
    api_version=AZURE_OPENAI_API_VERSION,     # "2024-02-15-preview"
)

# Deployment name (NOT model name)
EMBED_MODEL = AZURE_OPENAI_EMBED_MODEL


# -----------------------------------------------------------
# INTERNAL: Call Azure embeddings for a batch
# -----------------------------------------------------------
def _embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Sends one embedding request for a batch of texts.
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,    # MUST MATCH DEPLOYMENT NAME
        input=texts,
    )

    # Returned in same order as input
    return [item.embedding for item in response.data]


# -----------------------------------------------------------
# PUBLIC: Batch embedding with retries
# -----------------------------------------------------------
def batch_embed(
    texts: List[str],
    batch_size: int = 64,
    max_retries: int = 3,
    backoff_sec: float = 2.0,
) -> List[List[float]]:
    """
    Embed many texts with batching + retries.
    """

    if not texts:
        return []

    results: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        attempt = 0
        delay = backoff_sec

        while True:
            try:
                embeddings = _embed_batch(batch)
                results.extend(embeddings)
                break  # success — exit retry loop

            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(
                        f"Embedding failed after {max_retries} retries: {e}"
                    ) from e

                time.sleep(delay)
                delay *= 2  # exponential backoff

    return results
