# import numpy as np
# from openai import AzureOpenAI
# from app.vectorstore import load_vectorstore
# from config.settings import (
#     AZURE_OPENAI_CHAT_API_KEY,
#     AZURE_OPENAI_CHAT_MODEL,          # <-- deployment name
#     AZURE_OPENAI_CHAT_ENDPOINT,       # <-- your endpoint, e.g. https://ocugen-aoai.openai.azure.com/
#     AZURE_OPENAI_API_CHAT_VERSION,     # <-- api version
#     AZURE_OPENAI_API_KEY,          # <-- api key
#     AZURE_OPENAI_ENDPOINT,       # <-- your endpoint, e.g. https://ocugen-aoai.openai.azure.com/
#     AZURE_OPENAI_API_VERSION,
#     AZURE_OPENAI_EMBED_MODEL
# )

# # ----------------------------
# # AZURE OPENAI CLIENT
# # ----------------------------
# client = AzureOpenAI(
#     api_key=AZURE_OPENAI_CHAT_API_KEY,
#     azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
#     api_version=AZURE_OPENAI_API_CHAT_VERSION
# )


# client1 = AzureOpenAI(
#     api_key=AZURE_OPENAI_API_KEY,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_version=AZURE_OPENAI_API_VERSION
# )

# # Load FAISS index into memory
# index, vectors, chunks = load_vectorstore()

# # -------------------------------------------------------------
# # EMBEDDING FUNCTION
# # -------------------------------------------------------------
# def embed_query(text: str):
#     """Embed the query using Azure OpenAI."""
#     resp = client1.embeddings.create(
#         model=AZURE_OPENAI_EMBED_MODEL,      # deployment name in Azure
#         input=text
#     )
#     return np.array(resp.data[0].embedding, dtype=np.float32)

# # -------------------------------------------------------------
# # VECTOR SEARCH USING FAISS
# # -------------------------------------------------------------
# def search(query: str, k: int = 3):
#     q_vec = embed_query(query).reshape(1, -1)
#     scores, indices = index.search(q_vec, k)

#     # Each match: (chunk_obj, score)
#     matches = [(chunks[i], float(scores[0][j])) for j, i in enumerate(indices[0])]
#     return matches


# def _extract_text_from_chunk(chunk):
#     """Try to extract text from a chunk that may be str or dict-like."""
#     if isinstance(chunk, str):
#         return chunk

#     if isinstance(chunk, dict):
#         for key in ("text", "content", "page_content", "chunk"):
#             if key in chunk and isinstance(chunk[key], str):
#                 return chunk[key]
#         # fallback: string representation
#         return str(chunk)

#     # any other type (e.g. list/tuple), fallback to str
#     return str(chunk)





# def answer(query: str) -> str:
#     docs = search(query, k=3)

#     # docs is a list of (chunk_obj, score)
#     context_pieces = []
#     for chunk_obj, score in docs:
#         text = _extract_text_from_chunk(chunk_obj)
#         if text:
#             context_pieces.append(text)

#     context = "\n\n".join(context_pieces) if context_pieces else "No relevant context found."

#     prompt = f"""
# You are an advanced analytical assistant specialized in document understanding.

# Your job is to:
# 1.Accurately interpret unstructured text, structured text, tables, bullet points, forms, and mixed-format documents.
# 2.Extract, compare, filter, and reason over data, including data found inside tables.
# 3.Perform analytical operations such as:
#     - filtering rows
#     - finding matching entries
#     - extracting key-value fields
#     - performing calculations if possible
#     - comparing relationships in the document
# 4.Use BOTH provided context and your own reasoning, but:
#     - Prioritize provided context first
#     - If the answer is not directly in context but can be logically inferred, infer it
#     - If it cannot be inferred, say "Not in knowledge base."

    
# RULES
# 1.Never hallucinate facts that are not in the document or cannot be logically deduced.
# 2.When answering questions about tables:
#     - Convert the table to structured form internally
#     - Perform filtering, searching, and comparison
#     - Give the exact rows/columns matched

# 3.Always explain how you arrived at the answer (short reasoning).
# 4.If user requests anything impossible from the given context, answer:
# “Not in knowledge base.”
# 5.You are allowed to use numeric reasoning and multi-step reasoning.
# 6.If the context is empty or incomplete, say so.
# 7.Never say “As an AI model…” or break character.


# OUTPUT FORMAT GUIDELINES
# When responding:
# - Use clear bullet points or tables when needed
# - If extracting information, show exact snippet or row
# - If applying filters (example: “give me rows where status=Active and amount>500”), respond with a filtered table
# - If the result is empty, return:
# “No matching records found based on your filters.”


# YOU MUST ALWAYS:
# - Prioritize context
# - Use reasoning
# - Avoid hallucinations
# - Return “Not in knowledge base” when applicable
# - Understand and process table data with accuracy
# - Be consistent across all answers


# Example Behavior
# User asks:
# “What is the total amount for rows where Category=‘Lab’ and Date after 2023-01-01?”

# Assistant should:
# - Parse the table
# - Filter rows
# - Sum numeric values
# - Return clean structured output

# Context:
# {context}

# Question: {query}
# Answer:
#     """

#     response = client.chat.completions.create(
#         model=AZURE_OPENAI_CHAT_MODEL,
#         messages=[{"role": "user", "content": prompt}]
#     )

#     # new OpenAI SDK: message is an object, not dict
#     return response.choices[0].message.content










####################################################################################
# import numpy as np
# from typing import List, Dict
# from openai import AzureOpenAI
# from app.vectorstore import load_vectorstore
# from config.settings import (
#     AZURE_OPENAI_CHAT_API_KEY,
#     AZURE_OPENAI_CHAT_MODEL,          # chat deployment name
#     AZURE_OPENAI_CHAT_ENDPOINT,       # e.g. https://ocugen-aoai.openai.azure.com/
#     AZURE_OPENAI_API_CHAT_VERSION,    # chat api version

#     AZURE_OPENAI_API_KEY,             # embedding api key
#     AZURE_OPENAI_ENDPOINT,            # embedding endpoint
#     AZURE_OPENAI_API_VERSION,         # embedding api version
#     AZURE_OPENAI_EMBED_MODEL,         # embedding deployment name
# )

# # ----------------------------
# # AZURE OPENAI CLIENTS
# # ----------------------------

# # Chat client
# client = AzureOpenAI(
#     api_key=AZURE_OPENAI_CHAT_API_KEY,
#     azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
#     api_version=AZURE_OPENAI_API_CHAT_VERSION,
# )

# # Embedding client
# client1 = AzureOpenAI(
#     api_key=AZURE_OPENAI_API_KEY,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_version=AZURE_OPENAI_API_VERSION,
# )


# # ----------------------------
# # VECTORSTORE ACCESS
# # ----------------------------

# def get_index_and_chunks():
#     """
#     Always load the latest FAISS index + metadata.
#     Ensures newly built indexes from uploads are picked up.
#     """
#     index, vectors, chunks = load_vectorstore()
#     return index, chunks


# # -------------------------------------------------------------
# # EMBEDDING FUNCTION
# # -------------------------------------------------------------
# def embed_query(text: str) -> np.ndarray:
#     """Embed the query using Azure OpenAI."""
#     resp = client1.embeddings.create(
#         model=AZURE_OPENAI_EMBED_MODEL,  # embedding deployment name in Azure
#         input=text,
#     )
#     return np.array(resp.data[0].embedding, dtype=np.float32)


# # -------------------------------------------------------------
# # VECTOR SEARCH USING FAISS
# # -------------------------------------------------------------
# def search(query: str, k: int = 3):
#     index, chunks = get_index_and_chunks()
#     q_vec = embed_query(query).reshape(1, -1)
#     scores, indices = index.search(q_vec, k)

#     # Each match: (chunk_obj, score)
#     matches = [(chunks[i], float(scores[0][j])) for j, i in enumerate(indices[0])]
#     return matches


# def _extract_text_from_chunk(chunk):
#     """Try to extract text from a chunk that may be str or dict-like."""
#     if isinstance(chunk, str):
#         return chunk

#     if isinstance(chunk, dict):
#         for key in ("text", "content", "page_content", "chunk"):
#             if key in chunk and isinstance(chunk[key], str):
#                 return chunk[key]
#         # fallback: string representation
#         return str(chunk)

#     # any other type (e.g. list/tuple), fallback to str
#     return str(chunk)


# # -------------------------------------------------------------
# # HISTORY FORMATTING
# # -------------------------------------------------------------
# def format_history(history: List[Dict], max_turns: int = 5) -> str:
#     """
#     Convert st.session_state.messages (list of {"role", "content"}) into
#     a compact text history. Keeps only the last few turns.
#     """
#     if not history:
#         return ""

#     trimmed = history[-(max_turns * 2):]  # rough cap

#     lines = []
#     for msg in trimmed:
#         role = msg.get("role", "")
#         content = msg.get("content", "")
#         if not content:
#             continue
#         if role == "user":
#             lines.append(f"User: {content}")
#         elif role == "assistant":
#             lines.append(f"Assistant: {content}")
#     return "\n".join(lines)


# # -------------------------------------------------------------
# # MAIN ANSWER FUNCTION (STATEFUL)
# # -------------------------------------------------------------
# def answer(query: str, history: List[Dict]) -> str:
#     """
#     Generate final RAG answer using:
#     - KB context from FAISS
#     - Recent chat history (within this Streamlit session)
#     """
#     docs = search(query, k=7)

#     # docs is a list of (chunk_obj, score)
#     context_pieces = []
#     for chunk_obj, score in docs:
#         text = _extract_text_from_chunk(chunk_obj)
#         if text:
#             context_pieces.append(text)

#     context = "\n\n".join(context_pieces) if context_pieces else "No relevant context found."
#     conv_history = format_history(history)

#     # Your original detailed behavior + rules, kept intact
#     instructions = """
# You are an excelent focused assistant specialized in understanding scientific and regulatory documents,
# including tables and structured data.

# Your priorities:
# 1. Use the provided context as the primary source of truth.
# 2. You are allowed and expected to analyze, transform, and compute over the context
#    (for example: counting table columns or rows, summing values, identifying patterns,
#    filtering by conditions, or comparing entries).
# 3. Only if the answer is clearly not in the context AND cannot be logically derived
#    from the context (including such computations), reply exactly with:
#    Not in knowledge base.

# Answering style:
# - Start with a direct, natural-language answer.
# - Do NOT repeat the user's question.
# - Do NOT add headings like "Reasoning:" or "Analysis:" unless the user explicitly asks for them.
# - Use plain paragraphs by default.
# - Use bullet points or tables only when they clearly make the answer easier to read or the user asks for them.
# - Do NOT describe your internal thought process step-by-step. Just give the conclusion and any minimal explanation needed.

# Tables:
# - You can interpret table-like text from the context.
# - You may reconstruct tables internally to:
#   - count columns or rows,
#   - extract specific cells,
#   - filter rows based on conditions (e.g., by exon, category, date, status),
#   - compute aggregates (e.g., totals, averages).
#   - If the user asks for filtering (e.g., "rows where exon = 13" or "amount > 500"), apply that logically.
#   - If no rows match the requested filters, reply:
#   "No matching records found based on your filters."

# Important:
# - **Do not invent data** that is not supported by or logically derivable from the context.
# """.strip()


#     user_content = f"""
# [KB Context]
# {context}

# [Conversation So Far]
# {conv_history if conv_history else "(no previous turns)"}

# [Current Question]
# {query}
#     """.strip()

#     response = client.chat.completions.create(
#         model=AZURE_OPENAI_CHAT_MODEL,
#         messages=[
#             {"role": "system", "content": instructions},
#             {"role": "user", "content": user_content},
#         ],
#         temperature=0.0,        # low randomness, consistent answers
#         max_tokens=4500,         # adjust based on how long your answers should be
#     )

#     # new OpenAI SDK: message is an object, not a dict
#     return response.choices[0].message.content








################################################################################################


# Langgraph

import numpy as np
from typing import List, Dict, TypedDict

from openai import AzureOpenAI
from langgraph.graph import StateGraph, END
from azure.search.documents.models import VectorizedQuery
from ingest.embed import batch_embed

from app.vectorstore import load_vectorstore
from config.settings import (
    AZURE_OPENAI_CHAT_API_KEY,
    AZURE_OPENAI_CHAT_MODEL,          # chat deployment name
    AZURE_OPENAI_CHAT_ENDPOINT,       # e.g. https://ocugen-aoai.openai.azure.com/
    AZURE_OPENAI_API_CHAT_VERSION,    # chat api version

    AZURE_OPENAI_API_KEY,             # embedding api key
    AZURE_OPENAI_ENDPOINT,            # embedding endpoint
    AZURE_OPENAI_API_VERSION,         # embedding api version
    AZURE_OPENAI_EMBED_MODEL,         # embedding deployment name
)

# ============================
# AZURE OPENAI CLIENTS
# ============================

# Chat client (for answering)
client = AzureOpenAI(
    api_key=AZURE_OPENAI_CHAT_API_KEY,
    azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
    api_version=AZURE_OPENAI_API_CHAT_VERSION,
)

# Embedding client (for vector search)
client1 = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)


# ============================
# EMBEDDING FUNCTION
# ============================

def embed_query(text: str) -> np.ndarray:
    """
    Create an embedding for the query text using Azure OpenAI.
    """
    resp = client1.embeddings.create(model=AZURE_OPENAI_EMBED_MODEL, input=text)

    embedding = resp.data[0].embedding

    # Convert to numpy array of floats
    return list(embedding)



# ============================
# VECTOR SEARCH (Azure AI Search)
# ============================

def search(query: str, k: int = 3):
    q_vec = batch_embed([query])  # embedding

    search_client = load_vectorstore()

    # NEW CORRECT VECTOR QUERY for 11.7.0b2
    vector_query = VectorizedQuery(
        vector=q_vec,
        k=k,
        fields="vector"
    )

    # MUST WRAP IN LIST: vector_queries=[...]
    results = search_client.search(
        search_text=query,              
        vector_queries=[vector_query],
        select=[
            "text",
            "doc_id",
            "page_numbers",
            "chunk_type",
            "heading_path"
        ],
        top=k
    )

    output = []
    for r in results:
        output.append((r["text"], r["@search.score"]))

    return output


# ============================
# CHUNK TEXT EXTRACTION
# ============================

def _extract_text_from_chunk(chunk):
    """
    Extract text from a chunk object.

    The chunk can be:
    - a plain string
    - a dict with keys like "text", "content", "page_content", or "chunk"
    - any other type (fall back to str(chunk))
    """
    if isinstance(chunk, str):
        return chunk

    if isinstance(chunk, dict):
        for key in ("text", "content", "page_content", "chunk"):
            if key in chunk and isinstance(chunk[key], str):
                return chunk[key]
        # Fallback: string representation of the dict
        return str(chunk)

    # Any other type: fallback to text representation
    return str(chunk)


# ============================
# CHAT HISTORY FORMATTING
# ============================

def format_history(history: List[Dict], max_turns: int = 5) -> str:
    """
    Convert a list of messages ({"role": "user"/"assistant", "content": str})
    into a compact text representation.

    Keeps only the last few turns for brevity.
    """
    if not history:
        return ""

    # Rough cap on length: last N turns
    trimmed = history[-(max_turns * 2):]

    lines = []
    for msg in trimmed:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue

        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")

    return "\n".join(lines)


# ============================
# LANGGRAPH STATE DEFINITION
# ============================

class RAGState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.

    - query:        User's current question.
    - history:      Previous chat messages in this session.
    - context:      Retrieved KB context (combined chunks).
    - conv_history: Formatted history text.
    - llm_input:    Final prompt text passed to the LLM.
    - answer:       Final answer generated by the LLM.
    """
    query: str
    history: List[Dict]
    context: str
    conv_history: str
    llm_input: str
    answer: str


# ============================
# LANGGRAPH NODES
# ============================

def retrieve_context_node(state: RAGState) -> RAGState:
    """
    Node 1: Retrieve relevant context from the KB using FAISS.

    - Uses the current query from state["query"]
    - Calls `search(...)`
    - Combines the top chunks into a single context string
    - Stores it in state["context"]
    """
    query = state.get("query", "")

    # Search top-k documents for this query
    docs = search(query, k=12)

    # Extract text from each chunk
    context_pieces = []
    for chunk_obj, score in docs:
        text = _extract_text_from_chunk(chunk_obj)
        if text:
            context_pieces.append(text)

    # Merge all context into one string
    context = (
        "\n\n".join(context_pieces)
        if context_pieces
        else "No relevant context found."
    )

    # Return updated state
    new_state = dict(state)
    new_state["context"] = context
    return new_state


def build_prompt_node(state: RAGState) -> RAGState:
    """
    Node 2: Build the final prompt that will be sent to the LLM.

    - Takes state["context"] and state["history"]
    - Formats history into a string
    - Creates a single combined user message with:
      [KB Context], [Conversation So Far], [Current Question]
    - Stores this in state["llm_input"]
    """
    context = state.get("context", "No relevant context found.")
    history = state.get("history", [])
    query = state.get("query", "")

    # Convert previous messages into nice text
    conv_history = format_history(history)

    # Build the user message content
    user_content = f"""
[Knowledge Base Context]
{context}

[Conversation So Far]
{conv_history if conv_history else "(no previous turns)"}

[Current Question]
{query}
    """.strip()

    new_state = dict(state)
    new_state["conv_history"] = conv_history
    new_state["llm_input"] = user_content
    return new_state


def generate_answer_node(state: RAGState) -> RAGState:
    """
    Node 3: Call the Azure OpenAI chat model to get the final answer.

    - Uses a system prompt with your rules
    - Uses state["llm_input"] as user content
    - Writes the result into state["answer"]
    """
    llm_input = state.get("llm_input", "")

    instructions = """
You are an excelent focused assistant specialized in understanding scientific and regulatory documents,
including tables and structured data.

Your priorities:
1. Use the provided context as the primary source of truth.
2. You are allowed and expected to analyze, transform, and compute over the context
   (for example: counting table columns or rows, summing values, identifying patterns,
   filtering by conditions, or comparing entries).
3. Only if the answer is clearly not in the context AND cannot be logically derived
   from the context (including such computations), reply exactly with:
   Not in knowledge base.

Answering style:
- Start with a direct, natural-language answer.
- Do NOT repeat the user's question.
- Do NOT add headings like "Reasoning:" or "Analysis:" unless the user explicitly asks for them.
- Use plain paragraphs by default.
- Use bullet points or tables only when they clearly make the answer easier to read or the user asks for them.
- Do NOT describe your internal thought process step-by-step. Just give the conclusion and any minimal explanation needed.

Tables:
- You can interpret table-like text from the context.
- You may reconstruct tables internally to:
  - count columns or rows,
  - extract specific cells,
  - filter rows based on conditions (e.g., by exon, category, date, status),
  - compute aggregates (e.g., totals, averages).
- If the user asks for filtering (e.g., "rows where exon = 13" or "amount > 500"), apply that logically.
- If no rows match the requested filters, reply:
  "No matching records found based on your filters."
- Return the result as a proper markdown table.

Critical instruction:
- The "Guideline" describes HOW to answer, not WHAT the answer is.
- The guideline must NOT be treated as factual content.
- You must derive the answer ONLY from the provided knowledge base context.
- If the knowledge base does not support the answer, reply exactly:
  Not in knowledge base.

Important:
- **Do not invent data** that is not supported by or logically derivable from the context.
""".strip()

    # instructions = """
    # You are a senior Regulatory Medical Writer and Subject Matter Expert (SME)
    # with experience authoring clinical trial protocols, CSR sections, and
    # regulatory submission documents (ICH-GCP compliant).

    # You can operate in TWO complementary roles:
    # 1. Regulatory Author (narrative, protocol-style writing)
    # 2. Analytical SME (counting, comparing, structuring, summarizing)

    # ────────────────────────────────
    # ANALYTICAL AUTHORIZATION (CRITICAL)
    # ────────────────────────────────
    # You are explicitly allowed to perform analytical operations on the provided
    # content, even if the document represents a single version only.

    # Allowed analytical operations include:
    # - counting explicitly described changes, updates, revisions, or modifications,
    # - identifying and enumerating phrases such as "updated", "revised", "modified",
    # "added", "removed", "clarified", or "amended",
    # - generating tables or lists derived directly from the document,
    # - summarizing amendment scope based on explicit statements in the text.

    # If a baseline or prior version is NOT provided:
    # - Do NOT assume or invent changes.
    # - Do NOT infer differences paragraph-by-paragraph.
    # - You MAY state analytical limitations clearly and professionally.

    # ────────────────────────────────
    # SOURCE RULES
    # ────────────────────────────────
    # 1. Use the provided content as the ONLY source of factual information.
    # 2. You may derive logical conclusions and analytical summaries strictly
    # from what is explicitly stated in the document.
    # 3. If an exact numerical answer cannot be determined, you MUST:
    # - explain why in regulatory-safe language,
    # - state what CAN be determined from the content.
    # 4. Reply "Not in knowledge base" ONLY when:
    # - no analytical conclusion,
    # - no scoped explanation,
    # - and no limitation statement can be reasonably produced.

    # ────────────────────────────────
    # AUTHORING STYLE (WHEN NARRATIVE IS REQUIRED)
    # ────────────────────────────────
    # - Use formal regulatory / protocol language.
    # - Use complete, structured paragraphs.
    # - Maintain neutral, objective tone.
    # - Avoid conversational phrasing.
    # - Do NOT repeat the user's question.
    # - Do NOT describe internal reasoning steps.
    # - Do NOT mention "knowledge base" or "context".

    # ────────────────────────────────
    # STRUCTURED OUTPUT RULES
    # ────────────────────────────────
    # - If the user asks for a count, comparison, list, or table:
    # - perform the analysis if possible,
    # - otherwise provide a limitation statement instead of refusing.
    # - If a table is requested and derivable, return a markdown table.

    # ────────────────────────────────
    # REGULATORY SAFETY
    # ────────────────────────────────
    # - Do NOT invent data.
    # - Do NOT assume unstated baselines.
    # - Do NOT soften uncertainty with speculative language.
    # - Regulatory accuracy and audit defensibility take priority.

    # Use "Not in knowledge base" ONLY as a last resort.""".strip()

    # Call Azure OpenAI chat completion
    response = client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": llm_input},
        ],
        temperature=0.0,
        max_tokens=4500,
    )

    answer_text = response.choices[0].message.content

    new_state = dict(state)
    new_state["answer"] = answer_text
    return new_state


# ============================
# BUILD LANGGRAPH
# ============================

def build_rag_graph():
    """
    Build and compile the LangGraph graph.

    Flow:
        retrieve_context  ->  build_prompt  ->  generate_answer  ->  END
    """
    graph_builder = StateGraph(RAGState)

    # Register nodes
    graph_builder.add_node("retrieve_context", retrieve_context_node)
    graph_builder.add_node("build_prompt", build_prompt_node)
    graph_builder.add_node("generate_answer", generate_answer_node)

    # Set entry point
    graph_builder.set_entry_point("retrieve_context")

    # Connect nodes
    graph_builder.add_edge("retrieve_context", "build_prompt")
    graph_builder.add_edge("build_prompt", "generate_answer")
    graph_builder.add_edge("generate_answer", END)

    # Compile into a runnable graph
    return graph_builder.compile()


# Create a single graph instance to reuse
rag_graph = build_rag_graph()


# ============================
# PUBLIC ANSWER FUNCTION
# ============================

def answer(query: str, history: List[Dict]) -> str:
    """
    Public function to answer a question using the RAG LangGraph pipeline.

    - Takes a query and chat history
    - Runs the LangGraph
    - Returns the final answer string
    """
    # Initial state given to the graph
    initial_state: RAGState = {
        "query": query,
        "history": history,
    }

    # Run the graph synchronously
    final_state: RAGState = rag_graph.invoke(initial_state)

    # Return the answer from the final state
    return final_state.get("answer", "")












##################################################################################

# import numpy as np
# from typing import List, Dict, TypedDict

# from openai import AzureOpenAI
# from langgraph.graph import StateGraph, END
# from azure.search.documents.models import VectorizedQuery

# from ingest.embed import batch_embed
# from app.vectorstore import load_vectorstore
# from config.settings import (
#     AZURE_OPENAI_CHAT_API_KEY,
#     AZURE_OPENAI_CHAT_MODEL,
#     AZURE_OPENAI_CHAT_ENDPOINT,
#     AZURE_OPENAI_API_CHAT_VERSION,

#     AZURE_OPENAI_API_KEY,
#     AZURE_OPENAI_ENDPOINT,
#     AZURE_OPENAI_API_VERSION,
#     AZURE_OPENAI_EMBED_MODEL,
# )

# # ============================
# # AZURE OPENAI CLIENTS
# # ============================

# chat_client = AzureOpenAI(
#     api_key=AZURE_OPENAI_CHAT_API_KEY,
#     azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
#     api_version=AZURE_OPENAI_API_CHAT_VERSION,
# )

# embed_client = AzureOpenAI(
#     api_key=AZURE_OPENAI_API_KEY,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_version=AZURE_OPENAI_API_VERSION,
# )


# # # ============================
# # # EMBEDDING FUNCTION
# # # ============================

# def embed_query(text: str) -> List[float]:
#     """
#     Create an embedding for a query string using Azure OpenAI.
#     This MUST match the embedding model used during ingestion.
#     """
#     response = embed_client.embeddings.create(
#         model=AZURE_OPENAI_EMBED_MODEL,
#         input=text
#     )
#     return response.data[0].embedding


# # ============================
# # VECTOR SEARCH (HYBRID)
# # ============================

# def search(query: str, k: int = 12):
#     """
#     Hybrid Azure AI Search:
#     - BM25 keyword matching
#     - Vector similarity
#     - Returns structured chunks
#     """

#     query_vector = batch_embed([query])[0]
#     search_client = load_vectorstore()

#     vector_query = VectorizedQuery(
#         vector=query_vector,
#         k=k,
#         fields="vector",
#     )

#     results = search_client.search(
#         search_text=query,                     # 🔑 CRITICAL: hybrid search
#         vector_queries=[vector_query],
#         top=k,
#         select=[
#             "text",
#             "doc_id",
#             "page_numbers",
#             "chunk_type",
#             "heading_path",
#         ],
#     )

#     output = []
#     for r in results:
#         output.append({
#             "text": r.get("text", ""),
#             "doc_id": r.get("doc_id"),
#             "page_numbers": r.get("page_numbers", []),
#             "chunk_type": r.get("chunk_type"),
#             "heading_path": r.get("heading_path"),
#             "score": r.get("@search.score"),
#         })

#     return output

# # ============================
# # LANGGRAPH STATE
# # ============================

# class RAGState(TypedDict, total=False):
#     query: str
#     history: List[Dict]
#     context: str
#     conv_history: str
#     llm_input: str
#     answer: str

# # ============================
# # NODE 1: RETRIEVE CONTEXT
# # ============================

# def retrieve_context_node(state: RAGState) -> RAGState:
#     query = state.get("query", "")
#     docs = search(query, k=12)

#     # Order chunks by page number (restore document flow)
#     docs_sorted = sorted(
#         docs,
#         key=lambda d: min(d["page_numbers"]) if d["page_numbers"] else 9999
#     )

#     context_blocks = []

#     for d in docs_sorted:
#         text = d["text"]
#         if not text.strip():
#             continue

#         pages = ", ".join(map(str, d["page_numbers"])) if d["page_numbers"] else "N/A"
#         heading = d.get("heading_path") or "Unknown section"
#         doc_id = d.get("doc_id") or "Unknown document"

#         context_blocks.append(
#             f"[Document: {doc_id} | Pages: {pages} | Section: {heading}]\n{text}"
#         )

#     context = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."

#     new_state = dict(state)
#     new_state["context"] = context
#     return new_state

# # ============================
# # NODE 2: BUILD PROMPT
# # ============================

# def format_history(history: List[Dict], max_turns: int = 5) -> str:
#     if not history:
#         return ""

#     trimmed = history[-(max_turns * 2):]
#     lines = []

#     for msg in trimmed:
#         role = msg.get("role")
#         content = msg.get("content")
#         if not content:
#             continue

#         prefix = "User" if role == "user" else "Assistant"
#         lines.append(f"{prefix}: {content}")

#     return "\n".join(lines)


# def build_prompt_node(state: RAGState) -> RAGState:
#     context = state.get("context", "")
#     history = state.get("history", [])
#     query = state.get("query", "")

#     conv_history = format_history(history)

#     user_content = f"""
# [Knowledge Base Context]
# {context}

# [Conversation So Far]
# {conv_history if conv_history else "(no previous conversation)"}

# [Current Question]
# {query}
# """.strip()

#     new_state = dict(state)
#     new_state["conv_history"] = conv_history
#     new_state["llm_input"] = user_content
#     return new_state

# # ============================
# # NODE 3: GENERATE ANSWER
# # ============================

# def generate_answer_node(state: RAGState) -> RAGState:
#     llm_input = state.get("llm_input", "")

#     system_prompt ="""
# You are an excelent focused assistant specialized in understanding scientific and regulatory documents,
# including tables and structured data.

# Your priorities:
# 1. Use the provided context as the primary source of truth.
# 2. You are allowed and expected to analyze, transform, and compute over the context
#    (for example: counting table columns or rows, summing values, identifying patterns,
#    filtering by conditions, or comparing entries).
# 3. Only if the answer is clearly not in the context AND cannot be logically derived
#    from the context (including such computations), reply exactly with:
#    Not in knowledge base.

# Answering style:
# - Start with a direct, natural-language answer.
# - Do NOT repeat the user's question.
# - Do NOT add headings like "Reasoning:" or "Analysis:" unless the user explicitly asks for them.
# - Use plain paragraphs by default.
# - Use bullet points or tables only when they clearly make the answer easier to read or the user asks for them.
# - Do NOT describe your internal thought process step-by-step. Just give the conclusion and any minimal explanation needed.

# Tables:
# - You can interpret table-like text from the context.
# - You may reconstruct tables internally to:
#   - count columns or rows,
#   - extract specific cells,
#   - filter rows based on conditions (e.g., by exon, category, date, status),
#   - compute aggregates (e.g., totals, averages).
# - If the user asks for filtering (e.g., "rows where exon = 13" or "amount > 500"), apply that logically.
# - If no rows match the requested filters, reply:
#   "No matching records found based on your filters."
# - Return the result as a proper markdown table.

# Critical instruction:
# - The "Guideline" describes HOW to answer, not WHAT the answer is.
# - The guideline must NOT be treated as factual content.
# - You must derive the answer ONLY from the provided knowledge base context.
# - If the knowledge base does not support the answer, reply exactly:
#   Not in knowledge base.

# Important:
# - **Do not invent data** that is not supported by or logically derivable from the context.
# """.strip()

#     response = chat_client.chat.completions.create(
#         model=AZURE_OPENAI_CHAT_MODEL,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": llm_input},
#         ],
#         temperature=0.0,
#         max_tokens=4500,
#     )

#     new_state = dict(state)
#     new_state["answer"] = response.choices[0].message.content
#     return new_state

# # ============================
# # BUILD LANGGRAPH
# # ============================

# def build_rag_graph():
#     graph = StateGraph(RAGState)

#     graph.add_node("retrieve_context", retrieve_context_node)
#     graph.add_node("build_prompt", build_prompt_node)
#     graph.add_node("generate_answer", generate_answer_node)

#     graph.set_entry_point("retrieve_context")
#     graph.add_edge("retrieve_context", "build_prompt")
#     graph.add_edge("build_prompt", "generate_answer")
#     graph.add_edge("generate_answer", END)

#     return graph.compile()

# rag_graph = build_rag_graph()

# # ============================
# # PUBLIC ENTRY
# # ============================

# def answer(query: str, history: List[Dict]) -> str:
#     initial_state: RAGState = {
#         "query": query,
#         "history": history,
#     }

#     final_state = rag_graph.invoke(initial_state)
#     return final_state.get("answer", "")
