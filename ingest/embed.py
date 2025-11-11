"""
Embedding utilities using Azure OpenAI via the OpenAI Python SDK.
Reads credentials and deployment names from config.settings.
"""
from __future__ import annotations
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
import time

from openai import OpenAI
from config.settings import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBED_MODEL,
)

# Azure OpenAI client configuration
client = OpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    base_url=f"{AZURE_OPENAI_ENDPOINT}openai/deployments/",
    default_query={"api-version": AZURE_OPENAI_API_VERSION},
)
EMBED_MODEL = AZURE_OPENAI_EMBED_MODEL



def _embed_batch(texts: List[str]) -> List[List[float]]:
    """Call the embeddings endpoint once for a batch."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    # Returned in same order
    return [d.embedding for d in resp.data]


def batch_embed(
    texts: List[str],
    batch_size: int = 64,
    max_retries: int = 3,
    backoff_sec: float = 2.0,
) -> List[List[float]]:
    """
    Embed many texts with simple batching + retry.
    Args:
        texts: strings to embed
        batch_size: number of inputs per API call
        max_retries: times to retry a failing batch
        backoff_sec: initial backoff between retries
    Returns:
        List of embeddings (one per input), in order.
    """
    if not texts:
        return []

    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # Retry loop per batch
        attempt, delay = 0, backoff_sec
        while True:
            try:
                out.extend(_embed_batch(batch))
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(f"Embedding failed after {max_retries} retries: {e}") from e
                time.sleep(delay)
                delay *= 2  # exponential backoff

    return out
