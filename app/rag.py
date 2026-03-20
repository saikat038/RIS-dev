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


# # Langgraph

# import numpy as np
# import json
# from typing import List, Dict, TypedDict

# from openai import AzureOpenAI
# from langgraph.graph import StateGraph, END
# from azure.search.documents.models import VectorizedQuery
# from ingest.embed import batch_embed

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

# # ============================
# # AZURE OPENAI CLIENTS
# # ============================

# # Chat client (for answering)
# client = AzureOpenAI(
#     api_key=AZURE_OPENAI_CHAT_API_KEY,
#     azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
#     api_version=AZURE_OPENAI_API_CHAT_VERSION,
# )

# # Embedding client (for vector search)
# client1 = AzureOpenAI(
#     api_key=AZURE_OPENAI_API_KEY,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_version=AZURE_OPENAI_API_VERSION,
# )


# # ============================
# # EMBEDDING FUNCTION
# # ============================

# def embed_query(text: str) -> np.ndarray:
#     """
#     Create an embedding for the query text using Azure OpenAI.
#     """
#     resp = client1.embeddings.create(model=AZURE_OPENAI_EMBED_MODEL, input=text)

#     embedding = resp.data[0].embedding

#     # Convert to numpy array of floats
#     return list(embedding)



# # ============================
# # VECTOR SEARCH (Azure AI Search)
# # ============================

# def search(query: str, k: int = 3):
#     q_vec = batch_embed([query])[0]  # embedding

#     search_client = load_vectorstore()

#     # NEW CORRECT VECTOR QUERY for 11.7.0b2
#     vector_query = VectorizedQuery(
#         vector=q_vec,
#         k=k,
#         fields="vector"
#     )

#     # MUST WRAP IN LIST: vector_queries=[...]
#     results = search_client.search(
#         search_text="",      # required
#         vector_queries=[vector_query],
#         select=["text", "doc_id", "page_numbers"]
#     )

#     output = []
#     for r in results:
#         output.append((r["text"], r["@search.score"]))

#     return output


# # ============================
# # CHUNK TEXT EXTRACTION
# # ============================

# def _extract_text_from_chunk(chunk):
#     """
#     Extract text from a chunk object.

#     The chunk can be:
#     - a plain string
#     - a dict with keys like "text", "content", "page_content", or "chunk"
#     - any other type (fall back to str(chunk))
#     """
#     if isinstance(chunk, str):
#         return chunk

#     if isinstance(chunk, dict):
#         for key in ("text", "content", "page_content", "chunk"):
#             if key in chunk and isinstance(chunk[key], str):
#                 return chunk[key]
#         # Fallback: string representation of the dict
#         return str(chunk)

#     # Any other type: fallback to text representation
#     return str(chunk)


# # ============================
# # CHAT HISTORY FORMATTING
# # ============================

# def format_history(history: List[Dict], max_turns: int = 5) -> str:
#     """
#     Convert a list of messages ({"role": "user"/"assistant", "content": str})
#     into a compact text representation.

#     Keeps only the last few turns for brevity.
#     """
#     if not history:
#         return ""

#     # Rough cap on length: last N turns
#     trimmed = history[-(max_turns * 2):]

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


# # ============================
# # LANGGRAPH STATE DEFINITION
# # ============================

# class RAGState(TypedDict, total=False):
#     """
#     Shared state passed between LangGraph nodes.

#     - query:        User's current question.
#     - history:      Previous chat messages in this session.
#     - context:      Retrieved KB context (combined chunks).
#     - conv_history: Formatted history text.
#     - llm_input:    Final prompt text passed to the LLM.
#     - answer:       Final answer generated by the LLM.
#     """
#     query: str
#     history: List[Dict]
#     context: str
#     conv_history: str
#     llm_input: str
#     answer: str


# # ============================
# # LANGGRAPH NODES
# # ============================

# def retrieve_context_node(state: RAGState) -> RAGState:
#     """
#     Node 1: Retrieve relevant context from the KB using FAISS.

#     - Uses the current query from state["query"]
#     - Calls `search(...)`
#     - Combines the top chunks into a single context string
#     - Stores it in state["context"]
#     """

#     query = state.get("query", "").lower()

#     TABLE_QUERY_HINTS = [
#         "table", "comparison", "changes", "row", "column", "section",
#         "applicable section", "actual", "new (proposed)", "summary of changes"
#     ]

#     is_table_query = any(hint in query for hint in TABLE_QUERY_HINTS)

#     query = state.get("query", "")

#     # Search top-k documents for this query
#     docs = search(query, k=5)

#     # Extract text from each chunk
#     context_pieces = []
#     for chunk_obj, score in docs:
#         # 🚨 Skip non-table chunks if this is a table query
#         if is_table_query and isinstance(chunk_obj, dict):
#             if chunk_obj.get("block_type") != "table":
#                 continue

#         # chunk_obj may be text or dict
#         if isinstance(chunk_obj, dict):
#             # 🚨 TABLE PRESERVATION
#             if chunk_obj.get("block_type") == "table":
#                 context_pieces.append(
#                     json.dumps(
#                         {
#                             "type": "table",
#                             "headers": chunk_obj.get("headers", []),
#                             "rows": chunk_obj.get("rows", [])
#                         },
#                         indent=2
#                     )
#                 )
#             else:
#                 text = _extract_text_from_chunk(chunk_obj)
#                 if text:
#                     context_pieces.append(text)
#         else:
#             text = _extract_text_from_chunk(chunk_obj)
#             if text:
#                 context_pieces.append(text)

#     # Merge all context into one string
#     context = (
#         "\n\n".join(context_pieces)
#         if context_pieces
#         else "No relevant context found."
#     )

#     # Return updated state
#     new_state = dict(state)
#     new_state["context"] = context
#     return new_state


# def build_prompt_node(state: RAGState) -> RAGState:
#     """
#     Node 2: Build the final prompt that will be sent to the LLM.

#     - Takes state["context"] and state["history"]
#     - Formats history into a string
#     - Creates a single combined user message with:
#       [KB Context], [Conversation So Far], [Current Question]
#     - Stores this in state["llm_input"]
#     """
#     context = state.get("context", "No relevant context found.")
#     history = state.get("history", [])
#     query = state.get("query", "")

#     # Convert previous messages into nice text
#     conv_history = format_history(history)

#     # Build the user message content
#     user_content = f"""
# [Knowledge Base Context]
# {context}

# [Conversation So Far]
# {conv_history if conv_history else "(no previous turns)"}

# [Current Question]
# {query}
#     """.strip()

#     new_state = dict(state)
#     new_state["conv_history"] = conv_history
#     new_state["llm_input"] = user_content
#     return new_state


# def generate_answer_node(state: RAGState) -> RAGState:
#     """
#     Node 3: Call the Azure OpenAI chat model to get the final answer.

#     - Uses a system prompt with your rules
#     - Uses state["llm_input"] as user content
#     - Writes the result into state["answer"]
#     """
#     llm_input = state.get("llm_input", "")

# #     instructions = """
# # You are an excelent focused assistant specialized in understanding scientific and regulatory documents,
# # including tables and structured data.

# # Your priorities:
# # 1. Use the provided context as the primary source of truth.
# # 2. You are allowed and expected to analyze, transform, and compute over the context
# #    (for example: counting table columns or rows, summing values, identifying patterns,
# #    filtering by conditions, or comparing entries).
# # 3. Only if the answer is clearly not in the context AND cannot be logically derived
# #    from the context (including such computations), reply exactly with:
# #    Not in knowledge base.

# # Answering style:
# # - Start with a direct, natural-language answer.
# # - Do NOT repeat the user's question.
# # - Do NOT add headings like "Reasoning:" or "Analysis:" unless the user explicitly asks for them.
# # - Use plain paragraphs by default.
# # - Use bullet points or tables only when they clearly make the answer easier to read or the user asks for them.
# # - Do NOT describe your internal thought process step-by-step. Just give the conclusion and any minimal explanation needed.

# # Tables:
# # - You can interpret table-like text from the context.
# # - You may reconstruct tables internally to:
# #   - count columns or rows,
# #   - extract specific cells,
# #   - filter rows based on conditions (e.g., by exon, category, date, status),
# #   - compute aggregates (e.g., totals, averages).
# # - If the user asks for filtering (e.g., "rows where exon = 13" or "amount > 500"), apply that logically.
# # - If no rows match the requested filters, reply:
# #   "No matching records found based on your filters."
# # - Return the result as a proper markdown table.

# # Critical instruction:
# # - The "Guideline" describes HOW to answer, not WHAT the answer is.
# # - The guideline must NOT be treated as factual content.
# # - You must derive the answer ONLY from the provided knowledge base context.
# # - If the knowledge base does not support the answer, reply exactly:
# #   Not in knowledge base.

# # STRICT TABLE RULE (MANDATORY):
# # - When answering from a table, you MUST:
# #   1. Identify the exact row(s) used
# #   2. Ensure ALL relevant columns for that row are present
# # - If ANY required column or cell is missing, reply exactly:
# #   Not in knowledge base.
# # - NEVER infer, assume, merge, or reconstruct missing table cells.


# # Important:
# # - **Do not invent data** that is not supported by or logically derivable from the context.
# # """.strip()
#     instructions = """
# You are an excellent focused assistant specialized in understanding scientific and regulatory documents,
# including tables and structured data.

# You operate in TWO complementary roles:
# 1. Analytical Expert – for counting, filtering, comparing, and extracting structured facts.
# 2. Senior Regulatory Author / SME – for interpreting explicitly stated regulatory changes in a precise,
#    audit-defensible manner.

# ────────────────────────
# CORE PRIORITIES
# ────────────────────────
# 1. Use the provided context as the primary and authoritative source of truth.
# 2. You are explicitly allowed and expected to perform analytical operations over the context, including:
#    - counting items,
#    - enumerating changes,
#    - decomposing compound statements into distinct change items,
#    - interpreting table rows as structured records,
#    - interpreting bullet-style or sentence-separated changes inside a single table cell.
# 3. Only if the answer is clearly NOT present in the context AND cannot be logically derived
#    from explicitly stated information, reply exactly with:
#    Not in knowledge base.

# ────────────────────────
# ANSWERING STYLE
# ────────────────────────
# - Start with a direct, natural-language answer.
# - Do NOT repeat the user's question.
# - Do NOT add headings like "Reasoning:" or "Analysis:" unless explicitly asked.
# - Use plain paragraphs by default.
# - Use bullet points or tables ONLY when they improve clarity.
# - Do NOT describe internal chain-of-thought.
# - Provide short, professional justification only when necessary.

# ────────────────────────
# TABLE INTERPRETATION RULES
# ────────────────────────
# You can interpret table-like data from the context.

# You are authorized to:
# - Treat each table row as a single structured record.
# - Treat each row as one semantic unit when the table represents changes, comparisons, or updates.
# - Decompose a single cell (e.g., “Summary of changes”) into multiple distinct changes
#   IF they are explicitly stated as separate actions (e.g., “Added…”, “Updated…”, “Removed…”).
# - Count the number of changes based on explicit statements, bullet points, or sentence-level actions.

# You may reconstruct tables internally to:
# - count rows,
# - extract specific cells,
# - enumerate changes per row,
# - aggregate counts (e.g., total number of changes).

# ────────────────────────
# STRICT TABLE SAFETY RULE (MANDATORY)
# ────────────────────────
# - When answering from a table, you MUST:
#   1. Identify the exact row(s) used.
#   2. Use only explicitly stated content from the table cells.
# - You MUST NOT invent missing values or assume unstated facts.
# - However, breaking a long cell into multiple explicit change statements
#   DOES NOT count as inference if each change is explicitly written in the cell.

# If a required column or cell is entirely absent, reply exactly:
# Not in knowledge base.

# ────────────────────────
# CRITICAL INSTRUCTION
# ────────────────────────
# - The "Guideline" describes HOW to answer, not WHAT the answer is.
# - Guidelines must NEVER be treated as factual content.
# - Answers must be derived ONLY from the provided knowledge base context.
# - If the knowledge base does not support the answer, reply exactly:
#   Not in knowledge base.

# ────────────────────────
# IMPORTANT
# ────────────────────────
# - Do NOT invent data.
# - Do NOT assume unstated baselines.
# - Regulatory accuracy and audit defensibility take priority.
# """.strip()

#     # instructions = """
#     # You are a senior Regulatory Medical Writer and Subject Matter Expert (SME)
#     # with experience authoring clinical trial protocols, CSR sections, and
#     # regulatory submission documents (ICH-GCP compliant).

#     # You can operate in TWO complementary roles:
#     # 1. Regulatory Author (narrative, protocol-style writing)
#     # 2. Analytical SME (counting, comparing, structuring, summarizing)

#     # ────────────────────────────────
#     # ANALYTICAL AUTHORIZATION (CRITICAL)
#     # ────────────────────────────────
#     # You are explicitly allowed to perform analytical operations on the provided
#     # content, even if the document represents a single version only.

#     # Allowed analytical operations include:
#     # - counting explicitly described changes, updates, revisions, or modifications,
#     # - identifying and enumerating phrases such as "updated", "revised", "modified",
#     # "added", "removed", "clarified", or "amended",
#     # - generating tables or lists derived directly from the document,
#     # - summarizing amendment scope based on explicit statements in the text.

#     # If a baseline or prior version is NOT provided:
#     # - Do NOT assume or invent changes.
#     # - Do NOT infer differences paragraph-by-paragraph.
#     # - You MAY state analytical limitations clearly and professionally.

#     # ────────────────────────────────
#     # SOURCE RULES
#     # ────────────────────────────────
#     # 1. Use the provided content as the ONLY source of factual information.
#     # 2. You may derive logical conclusions and analytical summaries strictly
#     # from what is explicitly stated in the document.
#     # 3. If an exact numerical answer cannot be determined, you MUST:
#     # - explain why in regulatory-safe language,
#     # - state what CAN be determined from the content.
#     # 4. Reply "Not in knowledge base" ONLY when:
#     # - no analytical conclusion,
#     # - no scoped explanation,
#     # - and no limitation statement can be reasonably produced.

#     # ────────────────────────────────
#     # AUTHORING STYLE (WHEN NARRATIVE IS REQUIRED)
#     # ────────────────────────────────
#     # - Use formal regulatory / protocol language.
#     # - Use complete, structured paragraphs.
#     # - Maintain neutral, objective tone.
#     # - Avoid conversational phrasing.
#     # - Do NOT repeat the user's question.
#     # - Do NOT describe internal reasoning steps.
#     # - Do NOT mention "knowledge base" or "context".

#     # ────────────────────────────────
#     # STRUCTURED OUTPUT RULES
#     # ────────────────────────────────
#     # - If the user asks for a count, comparison, list, or table:
#     # - perform the analysis if possible,
#     # - otherwise provide a limitation statement instead of refusing.
#     # - If a table is requested and derivable, return a markdown table.

#     # ────────────────────────────────
#     # REGULATORY SAFETY
#     # ────────────────────────────────
#     # - Do NOT invent data.
#     # - Do NOT assume unstated baselines.
#     # - Do NOT soften uncertainty with speculative language.
#     # - Regulatory accuracy and audit defensibility take priority.

#     # Use "Not in knowledge base" ONLY as a last resort.""".strip()

#     # Call Azure OpenAI chat completion
#     response = client.chat.completions.create(
#         model=AZURE_OPENAI_CHAT_MODEL,
#         messages=[
#             {"role": "system", "content": instructions},
#             {"role": "user", "content": llm_input},
#         ],
#         temperature=0.0,
#         max_tokens=4500,
#     )

#     answer_text = response.choices[0].message.content

#     new_state = dict(state)
#     new_state["answer"] = answer_text
#     return new_state


# # ============================
# # BUILD LANGGRAPH
# # ============================

# def build_rag_graph():
#     """
#     Build and compile the LangGraph graph.

#     Flow:
#         retrieve_context  ->  build_prompt  ->  generate_answer  ->  END
#     """
#     graph_builder = StateGraph(RAGState)

#     # Register nodes
#     graph_builder.add_node("retrieve_context", retrieve_context_node)
#     graph_builder.add_node("build_prompt", build_prompt_node)
#     graph_builder.add_node("generate_answer", generate_answer_node)

#     # Set entry point
#     graph_builder.set_entry_point("retrieve_context")

#     # Connect nodes
#     graph_builder.add_edge("retrieve_context", "build_prompt")
#     graph_builder.add_edge("build_prompt", "generate_answer")
#     graph_builder.add_edge("generate_answer", END)

#     # Compile into a runnable graph
#     return graph_builder.compile()


# # Create a single graph instance to reuse
# rag_graph = build_rag_graph()


# # ============================
# # PUBLIC ANSWER FUNCTION
# # ============================

# def answer(query: str, history: List[Dict]) -> str:
#     """
#     Public function to answer a question using the RAG LangGraph pipeline.

#     - Takes a query and chat history
#     - Runs the LangGraph
#     - Returns the final answer string
#     """
#     # Initial state given to the graph
#     initial_state: RAGState = {
#         "query": query,
#         "history": history,
#     }

#     # Run the graph synchronously
#     final_state: RAGState = rag_graph.invoke(initial_state)

#     # Return the answer from the final state
#     return final_state.get("answer", "")












################################################################################################
# Langgraph – AUTHORING PIPELINE (ICH → SOURCE → AUTHORING)

'''
User intent
   ↓
Load Authoring Control JSON (static)
   ↓
Vector search → ICH index (rules)
   ↓
Vector search → Source index (evidence)
   ↓
Merge context
   ↓
Your EXISTING authoring prompt
'''

import os, sys, uuid
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import re
import json
from openai import AzureOpenAI
from typing import List, Dict, Any
from Protocoldigitization import *
from collections import defaultdict
from ingest.embed import batch_embed
from typing import List, Dict, TypedDict
from azure.storage.blob import BlobClient
from langgraph.graph import StateGraph, END
from azure.search.documents import SearchClient
from azure.core.exceptions import HttpResponseError
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.models import VectorizedQuery
from config.settings import (
    # Chat model (authoring)
    AZURE_OPENAI_CHAT_API_KEY,
    AZURE_OPENAI_CHAT_MODEL,
    AZURE_OPENAI_CHAT_ENDPOINT,
    AZURE_OPENAI_API_CHAT_VERSION,

    # Embedding model
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBED_MODEL,

    # Azure Search
    AZURE_SEARCH_SERVICE_ENDPOINT,
    AZURE_SEARCH_API_KEY,
    AZURE_SEARCH_INDEX_NAME,          # SOURCE index
    AZURE_ICH_SEARCH_INDEX_NAME,      # ICH index

    # azure blob
    AUTHOR_SCHEMA_PREFIX,
    AZURE_BLOB_CONN_STRING,
    BLOB_CONTAINER,
)

# print("ICH index name: ",AZURE_ICH_SEARCH_INDEX_NAME)
# ============================================================
# LOAD AUTHORING SCHEMA FROM BLOB STORAGE
# ============================================================
def load_authoring_schema_from_blob(schema_name: str) -> dict:
    """
    Load authoring control schema JSON directly from Azure Blob Storage
    into memory (RAM) without downloading to disk.
    """
    # print("name of the shcema: ",schema_name)
    blob_path = f"{AUTHOR_SCHEMA_PREFIX}{schema_name}"

    blob_client = BlobClient.from_connection_string(
        AZURE_BLOB_CONN_STRING,
        BLOB_CONTAINER,
        blob_path
    )

    raw_bytes = blob_client.download_blob().readall()
    return json.loads(raw_bytes.decode("utf-8"))


# ============================================================
# validate schema
# ============================================================

def validate_authoring_control_schema(schema: dict) -> None:
    """
    Minimal safety validation so the pipeline fails early with a clear error.
    """
    if not isinstance(schema, dict):
        raise ValueError("Authoring control schema must be a JSON object (dict).")

    # You can adjust this depending on your master schema structure
    if "sections" not in schema:
        raise ValueError("Authoring control schema missing required key: 'sections'.")

    if not isinstance(schema["sections"], list):
        raise ValueError("Authoring control schema 'sections' must be a list.")
    


# ============================================================
# IF SECTION IS MISSING IN AUTHORING CONTROL SCHEMA
# ============================================================

def build_missing_section_message(authoring_control: dict) -> str:
    sections = authoring_control.get("sections", [])
    section_names = [s.get("section") for s in sections if s.get("section")]

    formatted = "\n".join([f"- {name}" for name in section_names])

    return f"""I cannot author this section yet.

Available sections are:
{formatted}

Please add this section to the authoring control schema."""


# ============================================================
# RELEVANT SECTION MATCHING
# ============================================================

def pick_active_control(authoring_control: dict, user_query: str) -> dict:
    """
    Pick the section with the highest token overlap score.
    Rules:
      1) Exact match = score 1.0 (always wins)
      2) Otherwise choose section with highest token overlap
      3) Must be >= 0.75 overlap
      4) If none qualify → fallback to first section
    """

    def normalize(text: str) -> set[str]:
        return {
            t for t in text.lower().replace("-", " ").split()
            if t.isalnum()
        }

    query = (user_query or "").lower().strip()
    q_tokens = normalize(query)

    sections = authoring_control.get("sections", [])
    if not sections:
        return {}

    best_section = None
    best_score = 0.0

    for sec in sections:
        name = (sec.get("section") or "").lower().strip()
        sec_tokens = normalize(name)

        if not sec_tokens:
            continue

        # 1️⃣ Exact match wins immediately
        if name == query:
            return sec

        # 2️⃣ Token overlap score
        overlap = len(sec_tokens & q_tokens)
        score = overlap / len(sec_tokens)

        if score > best_score:
            best_score = score
            best_section = sec

    # 3️⃣ Accept only if >= 75%
    if best_section and best_score >= 0.75:
        return best_section

    # 4️⃣ Fallback
    return sections[0]


    

# ============================================================
# LOAD AUTHORING CONTROL SCHEMA (STATIC, SYSTEM-OWNED)
# ============================================================

_AUTHORING_CONTROL_CACHE = None

def get_authoring_control() -> dict:
    """
    Lazy-load the schema once per process to avoid repeated blob reads on reload.
    """
    global _AUTHORING_CONTROL_CACHE
    if _AUTHORING_CONTROL_CACHE is None:
        _AUTHORING_CONTROL_CACHE = load_authoring_schema_from_blob("master_schema.json")
        validate_authoring_control_schema(_AUTHORING_CONTROL_CACHE)
    return _AUTHORING_CONTROL_CACHE

AUTHORING_CONTROL = get_authoring_control()




# ============================================================
# AZURE OPENAI CLIENTS
# ============================================================

# Chat client (for AUTHORING – unchanged)
client = AzureOpenAI(
    api_key=AZURE_OPENAI_CHAT_API_KEY,
    azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
    api_version=AZURE_OPENAI_API_CHAT_VERSION,
)

# Embedding client (for VECTOR SEARCH – unchanged)
client1 = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)





# ============================================================
# EMBEDDING FUNCTION
# ============================================================

# def embed_query(text: str) -> np.ndarray:
#     """
#     Create an embedding vector for a given text using Azure OpenAI embeddings.
#     This is a pure transformation step (NO reasoning).
#     """
#     resp = client1.embeddings.create(
#         model=AZURE_OPENAI_EMBED_MODEL,
#         input=text
#     )
#     return list(resp.data[0].embedding)

# ============================================================
# AZURE SEARCH CLIENTS
# ============================================================

def load_source_search_client() -> SearchClient:
    """
    Returns Azure Search client for SOURCE documents.
    """

    print("[SOURCE INDEX]")
    print("  AZURE_SEARCH_INDEX_NAME:", AZURE_SEARCH_INDEX_NAME)
    print("  endpoint:", AZURE_SEARCH_SERVICE_ENDPOINT)
    return SearchClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
    )

def load_ich_search_client() -> SearchClient:
    """
    Returns Azure Search client for ICH GUIDELINES.
    """
    return SearchClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        index_name=AZURE_ICH_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
    )



# ============================================================
# HELPER
# ============================================================
def format_chunk_for_context(chunk: Dict) -> str:
    """
    Format SOURCE chunks for LLM context.
    Content first, metadata second.
    Improved table rendering to help LLM understand structure.
    """
    if not isinstance(chunk, dict):
        return str(chunk)

    text = (chunk.get("text") or "").strip()
    if not text:
        return ""

    chunk_type = chunk.get("chunk_type")
    heading = chunk.get("heading_path")
    pages = chunk.get("page_numbers")

    # Optional: suppress heading-only chunks
    if chunk_type == "heading" and len(text.split()) < 12:
        return ""

    # ── Special handling for tables ────────────────────────────────────────
    if chunk_type == "table" or any(k.startswith("table_") for k in chunk):
        lines = []

        # Table title / context heading
        table_title = (
            chunk.get("table_context_heading")
            or heading
            or "Table (no caption)"
        ).strip()
        lines.append(f"**{table_title}**")
        lines.append("")

        rows = chunk.get("table_rows")

        if rows:
            # FIX 1: Proper parsing of newline-separated row lists
            if isinstance(rows, str):
                try:
                    import ast
                    parsed_rows = []
                    for line in rows.splitlines():
                        line = line.strip()
                        if line.startswith("[") and line.endswith("]"):
                            parsed_rows.append(ast.literal_eval(line))
                    rows = parsed_rows if parsed_rows else None
                except Exception:
                    rows = None

            if isinstance(rows, list) and rows and isinstance(rows[0], list):

                # USE TRUE HEADERS FROM INDEX
                headers = chunk.get("table_headers")

                if headers:
                    header = [str(h).strip() for h in headers]
                    data_rows = rows
                else:
                    # fallback if headers missing
                    header = rows[0]
                    data_rows = rows[1:]

                lines.append("| " + " | ".join(header) + " |")
                lines.append("|" + "|".join(["---"] * len(header)) + "|")

                MAX_ROWS = 50

                for row in data_rows[:MAX_ROWS]:
                    cleaned_row = []

                    for cell in row:
                        cell = str(cell).strip().replace("\n", " ")

                        # normalize OCR checkbox artifacts
                        cell = cell.replace(":selected: X", "✕")
                        cell = cell.replace(":selected:", "✕")

                        # normalize OCR symbol errors
                        cell = cell.replace("士", "±")
                        cell = cell.replace("土", "±")

                        cleaned_row.append(cell)

                    lines.append("| " + " | ".join(cleaned_row) + " |")

                # Metadata
                meta = [f"type=table", f"pages={pages if pages else '?'}"]
                if heading:
                    meta.append(f"section={heading}")

                lines.append(f"[{', '.join(meta)}]")
                return "\n".join(lines).strip()

        # Fallback: use raw text if no usable table_rows
        lines.append(text)

    else:
        # Original non-table logic (unchanged)
        meta = []
        if chunk_type:
            meta.append(f"type={chunk_type}")
        if heading:
            meta.append(f"section={heading}")
        if pages:
            if isinstance(pages, list):
                pages = ",".join(map(str, pages))
            meta.append(f"pages={pages}")

        meta_line = f"[{', '.join(meta)}]" if meta else ""
        return f"{text}\n{meta_line}".strip()

    # Final metadata for table case (fallback path)
    meta = [f"type={chunk_type or 'table'}"]
    if heading:
        meta.append(f"section={heading}")
    if pages:
        if isinstance(pages, list):
            pages = ",".join(map(str, pages))
        meta.append(f"pages={pages}")

    meta_line = f"[{', '.join(meta)}]" if meta else ""

    result = "\n".join(lines)
    if meta_line:
        result += "\n" + meta_line
    return result.strip()



import re
import json

def normalize_section_numbering(answer_text: str, context) -> str:
    """
    1. Remove top-level uppercase headers like '6. INVESTIGATIONAL PLAN'
    2. Extract ICH reference number (e.g. 9.3.1)
    3. Renumber headings sequentially:
       9.3.1.1
       9.3.1.2
       9.3.1.3
    """

    # -----------------------------
    # Extract ich_refs from context
    # -----------------------------
    def extract_ich_refs(ctx):

        if isinstance(ctx, dict):
            return ctx.get("ich_refs", [])

        if isinstance(ctx, list):
            for item in ctx:
                if isinstance(item, dict) and "ich_refs" in item:
                    return item.get("ich_refs", [])

        if isinstance(ctx, str):
            match = re.search(r"\{.*?\}", ctx, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    return data.get("ich_refs", [])
                except:
                    pass

        return []

    ich_refs = extract_ich_refs(context)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>ICH refs:", ich_refs)

    if not ich_refs:
        return answer_text

    ich_ref = ich_refs[0]

    # -----------------------------
    # Extract numeric prefix
    # -----------------------------
    match = re.match(r"(\d+(?:\.\d+)*)", ich_ref)

    if not match:
        return answer_text

    ich_number = match.group(1)  # e.g. "9.3.1"

    # -----------------------------
    # Remove top-level headers
    # -----------------------------
    lines = answer_text.splitlines()
    output_lines = []
    heading_counter = 1

    for line in lines:

        stripped = line.strip()

        # Remove top level section like:
        # 6. INVESTIGATIONAL PLAN
        if re.match(r"^\*?\*?\d+\.\s+[A-Z][A-Z\s\-/(),:]+\*?\*?$", stripped):
            continue

        # Detect bold heading
        bold_heading_match = re.match(
            r"^\*\*(\d+(?:\.\d+)+)\.?\s+(.*?)\*\*$",
            stripped
        )

        if bold_heading_match:

            title = bold_heading_match.group(2).strip()

            new_heading = f"**{ich_number}.{heading_counter} {title}**"

            output_lines.append(new_heading)

            heading_counter += 1
            continue

        output_lines.append(line)

    text = "\n".join(output_lines)

    # -----------------------------
    # Renumber headings
    # -----------------------------
    heading_pattern = re.compile(
    r"^(\d+(?:\.\d+)+)\.?\s+(.+)$",
    flags=re.MULTILINE
    )

    counter = 1

    def replace_heading(match):
        nonlocal counter

        title = match.group(2)

        new_number = f"{ich_number}.{counter}"
        counter += 1

        return f"{new_number} {title}"

    text = heading_pattern.sub(replace_heading, text)

    return text.strip()



def build_generic_query(payload: dict) -> str:
    section = payload.get("section", "").strip()
    synonyms = payload.get("synonyms", [])

    lines = []

    if section:
        lines.append(f"{section} ")

    if len(synonyms)>1:
        lines.append("Also look for content related to the following terms:")
        for term in synonyms:
            if term != section:
                lines.append(f"- {term}")
    else:
        lines = [f"{lines[0]}complete information from study", ""]

    return "\n".join(lines)



def split_section(text: str):
    match = re.match(r"^\s*([\d\.]+)\s+(.*)$", text)
    if not match:
        return None, text.strip()

    section_number = match.group(1)
    section_text = match.group(2)

    return section_number, section_text



# ============================================================
# VECTOR SEARCH (GENERIC, REUSED)
# ============================================================

def vector_search_ich(
    search_client,
    section: str,
    synonyms: List[str],
    ich_refs: List[str],
    k_nearest_neighbors: int = 15,
    min_score: float = 0.62,
) -> List[Dict[str, Any]]:

    section_number = None

    for ref in ich_refs:
        match = re.match(r'^(\d+(\.\d+)*)', ref.strip())
        if match:
            section_number = match.group(1)
            break

    if not section_number:
        print("No valid section number found in ich_refs")
        return []

    print(f"Fetching paragraph chunks under section_path: {section_number}")

    filter_query = (
        f"section_path eq '{section_number}' "
        "and block_type eq 'paragraph'"
    )

    try:
        results = search_client.search(
            search_text=None,
            filter=filter_query,
            select=[
                "id", "doc_id", "source_type", "guideline",
                "section_path", "section_title",
                "block_type", "rule_type",
                "page_number", "text"
            ],
            order_by="page_number asc",
            top=100
        )

        chunks = [
            dict(r)
            for r in results
            if (r.get("text") or "").strip()
        ]

        print(f"Fetched {len(chunks)} paragraph chunks")
        return chunks

    except Exception as e:
        print(f"ICH paragraph retrieval failed: {e}")
        return []

##################################################################################################################################################################

from azure.search.documents.models import VectorizedQuery

TABLE_RE = re.compile(r"\bTable\s+\d+(\.\d+)*", re.IGNORECASE)

def classify_source_chunk(chunk: Dict) -> Dict:
    """Fallback classification if chunk_type is missing"""
    if chunk.get("chunk_type"):
        return chunk

    text = (chunk.get("text") or "").strip()
    if TABLE_RE.search(text):
        chunk["chunk_type"] = "TABLE"
    else:
        chunk["chunk_type"] = "PARAGRAPH"
    return chunk


def vector_search_source(
    search_client,
    section: str,
    synonyms: List[str],
    allowed_sources: List[str],
    k_nearest_neighbors: int = 50,
    min_score: float = 0.60,           # ← NEW: default 60% threshold
) -> List[Dict[str, Any]]:
    """
    Multi-query vector search on source index with minimum score filtering.
    Deduplicates by source_block_ids[0] or id.
    """
    if not allowed_sources:
        return []
    
    print("min score >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",min_score)

    # Build OR filter for allowed documents
    doc_filter = f"search.in(doc_id, '{','.join(allowed_sources)}', ',')"

    queries = [section] + [s for s in synonyms if s and s != section]
    queries = [q.strip() for q in queries if q and q.strip()]

    if not queries:
        return []

    results = []

    for q in queries:
        try:
            vector = batch_embed([q])[0]
            vq = VectorizedQuery(
                vector=vector,
                fields="vector",
            )
            vq.k = k_nearest_neighbors   # ← correct way (avoids warning)

            print("Vector field being queried:", vq.fields)
            print("Vector dimensions of query:", len(vector))

            res = search_client.search(
                search_text=q,
                vector_queries=[vq],
                filter=doc_filter,
                select=[
                    "id", "doc_id", "text", "chunk_type", "heading_path",
                    "page_numbers", "source_block_ids",
                    "table_context_heading", "table_context_text",
                    "table_semantic_hint", "table_headers", "table_rows"
                ],
                top=k_nearest_neighbors
            )

            # Filter by minimum score
            res_list = list(res)

            good_hits = []

            for r in res_list:
                d = dict(r)

                ordered = {
                    "doc_id": d.get("doc_id"),
                    "table_context_heading": d.get("table_context_heading"),
                    "table_headers": d.get("table_headers"),
                    "table_rows": d.get("table_rows"),
                    "text": d.get("text"),
                    "source_block_ids": d.get("source_block_ids"),
                    "page_numbers": d.get("page_numbers"),
                    "heading_path": d.get("heading_path"),
                    "chunk_type": d.get("chunk_type"),
                    "id": d.get("id"),
                    "table_context_text": d.get("table_context_text"),
                    "table_semantic_hint": d.get("table_semantic_hint"),
                    "@search.score": d.get("@search.score")
                }

                good_hits.append(ordered)

            print(f"Query '{q[:60]}...' returned {len(res_list)} hits → kept {len(good_hits)}")

            results.extend(good_hits)

        except Exception as e:
            print(f"Source vector search failed for query '{q}': {e}")
            continue

    # Deduplicate by source_block_ids[0] or id
    seen = set()
    deduplicated = []

    for r in results:
        key = (r.get("source_block_ids") or [r.get("id", "")])[0]
        if key and key not in seen:
            seen.add(key)
            deduplicated.append(r)

    print(f"Final source chunks after deduplication & score filtering: {len(deduplicated)}")


    print("[SOURCE SEARCH]")
    print("  queries used:", queries)
    print("  number of queries:", len(queries))
    print("  doc_filter:", doc_filter)
    print("  k_nearest_neighbors:", k_nearest_neighbors)
    print("  min_score:", min_score)
    print("  raw results before dedup:", len(results))
    return deduplicated



# ============================================================
# save vector search as json
# ============================================================
import json
import os
from datetime import datetime

def save_vector_search_results(
    section_name: str,
    source_chunks: List[Dict[str, Any]],
    ich_sections: List[Dict[str, Any]],
    debug_dir: str = "RIS-dev"
) -> None:
    """
    Saves vector search results (source chunks + grouped ICH sections) 
    to a timestamped JSON file in the specified debug directory.
    
    Args:
        section_name: The name of the section being processed
        source_chunks: List of raw source document chunks
        ich_sections: List of grouped ICH guideline sections
        debug_dir: Directory where debug files should be saved (default: "RIS-dev")
    """
    if not section_name:
        section_name = "unknown_section"

    # Clean section name for safe filename
    safe_section = re.sub(r'[^a-zA-Z0-9_-]', '_', section_name.lower().strip())

    # Always add timestamp to avoid overwriting files
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Normalize debug directory
    debug_dir = debug_dir.strip()
    if not debug_dir:
        debug_dir = "RIS-dev"  # fallback if empty string passed

    # Create directory if it doesn't exist
    os.makedirs(debug_dir, exist_ok=True)

    # Build full filename
    filename = os.path.join(debug_dir, f"vector_search_{safe_section}_{ts}.json")

    # Build payload (matches test file structure)
    payload = {
        "section": section_name,
        "timestamp": datetime.now().isoformat(),
        "source": {
            "total_chunks": len(source_chunks),
            "results": source_chunks,
        },
        "ich_guidelines": {
            "total_sections": len(ich_sections),
            "results": ich_sections,
        }
    }

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        
        print(f"🔍 Debug file saved successfully:")
        print(f"   Path: {filename}")
        print(f"   Source chunks: {len(source_chunks):,}")
        print(f"   ICH sections:  {len(ich_sections):,}")
    
    except PermissionError:
        print(f"Permission denied: Cannot write to {filename}")
    except OSError as e:
        print(f"OS error while saving debug file {filename}: {e}")
    except Exception as e:
        print(f"Unexpected error while saving debug file {filename}: {e}")


def group_ich_by_section(chunks: list[dict], ich_refs: list[str] = None) -> dict:
    """
    ONLY keeps the section whose section_path matches a number in ich_refs.
    Drops all others.
    Returns clean structure with only requested fields.
    """
    # Extract numeric section numbers from ich_refs (e.g. "14.1")
    allowed_paths = set()
    if ich_refs:
        for ref in ich_refs:
            match = re.search(r'\b(\d+(\.\d+)*)\b', ref.strip())
            if match:
                allowed_paths.add(match.group(1))

    print(f"Allowed paths from ich_refs: {allowed_paths or 'None provided'}")

    if not allowed_paths:
        print("No numeric section refs found → returning empty")
        return {"total_sections": 0, "results": []}

    # Group chunks
    grouped = defaultdict(list)
    for chunk in chunks:
        path = chunk.get("section_path")
        if path:
            grouped[path].append(chunk)

    results = []

    # Only process matching sections
    for section_path, items in grouped.items():
        if section_path not in allowed_paths:
            print(f"Skipped {section_path} (not matching ich_refs)")
            continue

        items.sort(key=lambda x: x.get("page_number") or 0)

        headings = []
        guideline_parts = []

        for item in items:
            text = (item.get("text") or "").strip()
            if not text:
                continue

            block_type = item.get("block_type", "").lower()

            if block_type == "heading":
                headings.append(text)
                guideline_parts.append(text)
            else:
                guideline_parts.append(text)

        guideline_text = " ".join(guideline_parts).strip() or "(no content)"

        results.append({
            "section_path": section_path,
            "source_type": "ich",
            "guideline_text": guideline_text,
            "headings": headings
        })

    # Sort (though usually only 1)
    results.sort(key=lambda x: x["section_path"])

    print(f"Kept {len(results)} matching sections")
    print("ich_refs before grouping:", ich_refs)

    return {
        "total_sections": len(results),
        "results": results
    }

# ============================================================
# CHAT HISTORY FORMATTING (UNCHANGED)
# ============================================================

def format_history(history: List[Dict], max_turns: int = 5) -> str:
    """
    Converts recent chat history into compact text.
    """
    if not history:
        return ""

    trimmed = history[-(max_turns * 2):]
    lines = []

    for msg in trimmed:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        lines.append(f"{role.capitalize()}: {content}")

    return "\n".join(lines)

# ============================================================
# LANGGRAPH STATE DEFINITION
# ============================================================

class RAGState(TypedDict, total=False):
    """
    Shared state across LangGraph nodes.
    """
    query: str
    history: List[Dict]
    context: str
    conv_history: str
    llm_input: str
    answer: str
    section_name: str



# ============================================================
# NODE 1 — RETRIEVE CONTEXT (ICH FIRST, SOURCE SECOND)
# ============================================================

def retrieve_context_node(state: RAGState) -> RAGState:
    """
    Retrieves context for AUTHORING.

    Order is STRICT:
    1. Authoring Control (section-level)
    2. ICH Guidelines (authoritative rules)
    3. Source Evidence (facts, INCLUDING TABLES)
    """

    query = state.get("query", "")

    # -------------------------------------------------
    # PICK ACTIVE AUTHORING CONTROL
    # -------------------------------------------------
    active_control = pick_active_control(AUTHORING_CONTROL, query)

    print("[ACTIVE CONTROL USED]")
    print("section:", active_control.get("section"))
    print("synonyms:", active_control.get("synonyms", []))
    print("len(synonyms):", len(active_control.get("synonyms", [])))
    print("allowed_sources:", active_control.get("allowed_sources", []))
    print("len(allowed_sources):", len(active_control.get("allowed_sources", [])))
    # print(active_control.get("section", ""))

    if len(active_control.get("section", "")) == 0:
        active_control = {
      "section": query,
      "synonyms": [""],
      "ich_refs": [""],
      "allowed_sources": ["ocu400-101-protocol.PDF", "OCU401_CSR_Final_Tables.PDF"],
      "detail_level": "high",
      "output_style": "verbatim",
      "forbidden_content": ["operational procedures"]
    }

    # print("active control",active_control)

    if not active_control:
        new_state = dict(state)
        new_state["answer"] = build_missing_section_message(AUTHORING_CONTROL)
        new_state["context"] = ""
        new_state["section_name"] = None
        return new_state


    # -------------------------------------------------
    # ICH RETRIEVAL (multi-query + deduplication)
    # -------------------------------------------------
    ich_client = load_ich_search_client()

    section_name = active_control.get("section", "")
    synonyms = active_control.get("synonyms", [])
    ich_refs = active_control.get("ich_refs", [])

    print("section name >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", section_name)

    ich_chunks = vector_search_ich(
        search_client=ich_client,
        section=section_name,
        synonyms=synonyms,
        ich_refs=ich_refs,
        k_nearest_neighbors=5,          # you can tune this
        min_score=0.60                   # start here, adjust 0.58–0.68 based on results
        )

    # -------------------------------------------------
    # GROUP ICH CHUNKS INTO STRUCTURED SECTIONS
    # -------------------------------------------------
    ich_sections = group_ich_by_section(ich_chunks, ich_refs=ich_refs)

    # -------------------------------------------------
    # BUILD ICH CONTEXT FOR LLM (text version)
    # -------------------------------------------------

    ich_context_blocks = []

    for section in ich_sections.get("results", []):
        if not isinstance(section, dict):
            print(f"Skipping invalid section item: {section}")
            continue

        guideline_text = section.get("guideline_text") or ""
        paragraphs = section.get("paragraphs", [])

        content = guideline_text or "".join(paragraphs)

        if not content.strip():
            continue

        # Format exactly as requested
        block = f"\n{content}"
        ich_context_blocks.append(block)

    # Join with double newline between blocks (if multiple, though you likely have only one)
    ich_context = "".join(ich_context_blocks) if ich_context_blocks else "No ICH guidance found."

    # print(f"ICH sections found: {len(ich_sections)}")


    # -------------------------------------------------
    # SOURCE RETRIEVAL (multi-query + deduplication + classification)
    # -------------------------------------------------
    source_client = load_source_search_client()

    allowed_sources = active_control.get("allowed_sources", [])
    source_chunks = vector_search_source(
                        search_client=source_client,
                        section=section_name,
                        synonyms=synonyms,
                        allowed_sources=allowed_sources,
                        k_nearest_neighbors=30,
                        min_score=0.50          # ← 60% threshold
                    )
    
    # After getting source_chunks
    # with open("source_chunks_raw.txt", "w", encoding="utf-8") as f:
    #     f.write(str(source_chunks))

    source_context_pieces = []

    for chunk in source_chunks:
        formatted = format_chunk_for_context(chunk)
        if formatted.strip():
            source_context_pieces.append(formatted)

    source_context = "\n\n".join(source_context_pieces) if source_context_pieces else "No source evidence found."

    # After building source_context and ich_context

    print("[DEBUG RETRIEVAL STATS]")
    print(f"  • source_chunks count: {len(source_chunks)}")
    print(f"  • ich_chunks count:    {len(ich_chunks)}")
    print(f"  • source_context len:  {len(source_context):,} chars")
    print(f"  • ich_context len:     {len(ich_context):,} chars")
    print(f"  • total context len:   {len(source_context) + len(ich_context):,} chars")
    print(f"  • active_control json len: {len(json.dumps(active_control)):,} chars")


    # ────────────────────────────────────────────────
    #  SAVE RAW VECTOR SEARCH RESULTS FOR DEBUGGING
    # ────────────────────────────────────────────────
    # Comment out or remove in production if not needed
    # ────────────────────────────────────────────────
    #  SAVE STRUCTURED VECTOR SEARCH RESULTS FOR DEBUGGING
    # ────────────────────────────────────────────────

    print("=== Debug: ICH retrieval status ===")
    print(f"Raw ich_chunks count: {len(ich_chunks)}")
    # if ich_chunks:
    #     print("First raw chunk keys:", list(ich_chunks[0].keys()))
    #     print("First raw chunk section_path:", ich_chunks[0].get("section_path"))
    #     print("First raw chunk text preview:", ich_chunks[0].get("text", "")[:100])

    # print(f"Grouped ich_sections count: {len(ich_sections)}")
    # if ich_sections:
    #     print("First grouped section:", ich_sections[0])
    # else:
    #     print("→ No ICH sections after grouping ←")

    print("Saving debug file now...")
    save_vector_search_results(
        section_name=section_name,
        source_chunks=source_chunks,
        ich_sections=ich_sections,
        debug_dir="RIS-dev"
    )
    
    if len(active_control.get("synonyms", "")[0])>1:
        # -------------------------------------------------
        # FINAL MERGED CONTEXT (meta data driven)
        # -------------------------------------------------
        print("inside meta driven mode >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        final_context = f"""
        [AUTHORING CONTROL]
        {json.dumps(active_control, indent=2)}

        [ICH GUIDELINES]
        {ich_context}

        [SOURCE EVIDENCE]
        {source_context}
        """.strip()

        new_state = dict(state)
        new_state["context"] = final_context
        new_state["section_name"] = section_name


    else:
        print("inside Q&A mode ?????????????????????????????????????")
        # -------------------------------------------------
        # FINAL MERGED CONTEXT (Q&A)
        # -------------------------------------------------
        final_context = f"""
        [AUTHORING CONTROL]
        {json.dumps(active_control, indent=2)}

        [SOURCE EVIDENCE]
        {source_context}
        """.strip()

        new_state = dict(state)
        new_state["context"] = final_context
        new_state["section_name"] = section_name

    
    # print("\n\nnew state:\n",new_state)
    with open("last_final_context.txt", "w", encoding="utf-8") as f:
        f.write(final_context)



    return new_state


# ============================================================
# NODE 2 — BUILD PROMPT (UNCHANGED STRUCTURE)
# ============================================================

def build_prompt_node(state: RAGState) -> RAGState:
    """
    Builds the final user message passed to the LLM.
    """
    if state.get("answer"):
        return state
    
    if state.get("section_name") is None:
        return state

    context = state.get("context", "")
    history = state.get("history", [])
    query = state.get("query", "")

    conv_history = format_history(history)

    user_content = f"""
[Knowledge Base Context]
{context}

[Conversation So Far]
{conv_history if conv_history else "(no previous turns)"}

[Current Authoring Request]
{query}
""".strip()
    
    print("[BUILD PROMPT DEBUG]")
    print(f"  • context length: {len(context):,} chars")
    print(f"  • conv_history length: {len(conv_history):,} chars")
    print(f"  • query length: {len(query):,} chars")
    print(f"  • final llm_input length: {len(user_content):,} chars")
    print(f"  • history messages count: {len(history)}")
    if conv_history:
        print("  • conv_history preview (first 200):")
        print(conv_history[:200])
        print("  • conv_history preview (last 200):")
        print(conv_history[-200:])

    new_state = dict(state)
    new_state["conv_history"] = conv_history
    new_state["llm_input"] = user_content
    return new_state

# ============================================================
# NODE 3 — GENERATE ANSWER (AUTHORING PROMPT UNTOUCHED)
# ============================================================

def generate_answer_node(state: RAGState) -> RAGState:
    """
    Calls Azure OpenAI Chat Completion to generate AUTHORING output.
    AUTHORING PROMPT IS KEPT EXACTLY AS-IS.
    """
    # If answer already exists, skip GPT call
    if state.get("answer"):
        return state
    
    if state.get("section_name") is None:
        return state

    llm_input = state.get("llm_input", "")


##################################################################################last working prompt#####################################################################################
    instructions = """
You are an expert regulatory authoring engine specialized in scientific and regulatory documents,
including structured text and tables.

You operate in TWO complementary modes simultaneously:
1. Analytical Expert — authorized to perform explicit analytical operations strictly on provided content.
2. Senior Regulatory Author / SME — authorized to author compliant regulatory text when explicitly allowed.

────────────────────────
GLOBAL AUTHORITY & SCOPE (NON-NEGOTIABLE)
────────────────────────
- You may ONLY use the content provided in the CONTEXT blocks.
- You MUST NOT use prior knowledge, training data, or assumptions.
- You MUST NOT infer or invent missing information.
- If required information is absent or incomplete, output exactly:
  Not in knowledge base.

────────────────────────
CONTEXT HIERARCHY (MANDATORY)
────────────────────────
1. SOURCE_CONTEXT
   - The ONLY authoritative source for factual content.
   - All factual assertions MUST be traceable to explicit statements in SOURCE_CONTEXT.

2. ICH_CONTEXT
   - MAY be used ONLY when Output Style = regulatory author.
   - MAY be used for structural guidance and regulatory phrasing alignment ONLY.
   - MUST NOT introduce new facts, criteria, thresholds, or content.
   - MUST NOT override SOURCE_CONTEXT.

   IF Output Style = verbatim:
   - ICH_CONTEXT MUST be completely ignored.
   - No restructuring, regulatory harmonization, or terminology normalization is permitted.

────────────────────────
AUTHORIZED ANALYTICAL OPERATIONS
────────────────────────
You are explicitly authorized to perform analytical operations ONLY on explicitly stated content,
including:

- Counting items, rows, criteria, or conditions
- Filtering records based on explicit conditions
- Sorting lists or table rows by explicit values
- Comparing values across rows or sections
- Decomposing compound statements into discrete actions
- Interpreting tables as structured records
- Performing mathematical operations using explicit numeric values
  (e.g., sums, differences, thresholds)

STRICT RULE:
- Analytical operations MUST NOT introduce assumptions or inferred values.
- If an operation cannot be performed using explicit data, output:
  Not in knowledge base.

────────────────────────
STRICT TABLE SAFETY RULE
────────────────────────
When using table data:
1. 1. Identify the exact row(s) used ONLY when performing analytical operations. In verbatim mode the full table must be rendered.
2. Use ONLY explicit cell content.
3. A single cell may be decomposed into multiple items ONLY if explicitly written.
4. Do NOT infer missing cells, relationships, or intent.

────────────────────────
MANDATORY TABLE RENDERING RULE
────────────────────────
IF Output Style = verbatim:

- Tables MUST be rendered as tables.
- All rows, columns, and cell values MUST be preserved exactly.
- Tables MUST NOT be flattened into paragraphs, bullets, or narrative text.
- When Output Style = verbatim and a table is present in SOURCE_CONTEXT, the entire table including all subgroup rows must be rendered; row selection or omission is not permitted.

IF Output Style = regulatory author:

- Table data MAY be transformed into narrative form when necessary for regulatory clarity.
- All factual values from the table MUST remain unchanged.
- No rows, columns, or values may be omitted unless they are structurally irrelevant to the section being authored.
- No new information may be introduced.
- Narrative text MUST remain fully traceable to the original table cells in SOURCE_CONTEXT.

────────────────────────
SECTION AUTHORING CONTROL
────────────────────────
You will be provided with SECTION_METADATA containing:
- Section Name
- Allowed Sources
- Output Style (verbatim | regulatory author)
- Detail Level
- Forbidden Content

You MUST obey all SECTION_METADATA constraints.

────────────────────────
ABSOLUTE SECTION BOUNDARY RULE (HIGHEST PRIORITY)
────────────────────────
You are authorized to use content if ANY synonym appears either:

- as a heading, OR
- within table content (including any row or cell)

If a synonym appears anywhere inside a table, the ENTIRE table MUST be extracted and rendered.

This requirement does NOT exclude other relevant content. Paragraphs and headings that match the section MUST ALSO be included.

This means:

- The content MUST include the full structural unit (e.g., complete table or section) in which the synonym appears, even if the synonym appears between table blocks or within table rows.
- Content outside the matched structural unit MUST be ignored, EXCEPT when the structural unit (e.g., a table) spans multiple contiguous blocks or pages, in which case all such blocks MUST be included to preserve the complete structure.

If the synonym includes a numbered heading (e.g., "4.2.4. Selection of the starting dose:"),
only the content under that exact structural heading is permitted.

Occurrences of the phrase elsewhere in the document
(e.g., cross-references, citations, summaries, tables, narrative mentions)
MUST NOT be used.

────────────────────────
STRUCTURAL EXTRACTION & RENDERING (HIGHEST PRIORITY)
────────────────────────
Before writing any content:

- Scan SOURCE_CONTEXT line-by-line.
- Identify structural elements under the authorized section.
- You MUST include ALL relevant structural elements present in SOURCE_CONTEXT. Do NOT omit paragraphs when tables are present, or vice versa.

IF Output Style = verbatim:
  • ALL structural elements MUST be rendered.
  • Headings and paragraph text are IMMUTABLE TOKENS.
  • You MUST NOT omit, merge, flatten, or downgrade structure.
  • Preserve hierarchy exactly as written.

IF Output Style = regulatory author:
  • Structural elements MAY be reorganized for alignment with ICH Guideline structure.
  • Structural elements MUST NOT be fabricated.
  • Reorganization is allowed ONLY if supported by SOURCE_CONTEXT.
  • Headings may be harmonized to ICH terminology if ICH_CONTEXT supports it.

────────────────────────
OUTPUT STYLE RULES
────────────────────────

IF Output Style = verbatim:
- Preserve wording EXACTLY as written in SOURCE_CONTEXT.
- Preserve structure, hierarchy, and ordering.
- Preserve headings and sub-headings exactly.
- Preserve numeric prefixes (e.g., 7.1, 1., 1.1).
- Do NOT paraphrase, summarize, normalize, or interpret.

IF Output Style = regulatory author:
- Author using formal regulatory language consistent with ICH E3.

Rephrasing, sentence restructuring, grammatical harmonization, and consolidation
ARE permitted provided:
- No new factual content is introduced.
- No criteria, thresholds, or procedures are added.
- All information remains directly traceable to SOURCE_CONTEXT.

Structural refinement for regulatory clarity is authorized.


────────────────────────
FORMAT & STRUCTURE ENFORCEMENT
────────────────────────
- Begin directly with the section content.
- Do NOT add introductions or framing statements.
- Do NOT restate or rename the section.
- Headings and sub-headings MUST:
  • Appear on their own line
  • Be formatted in **bold markdown**
  • Preserve original wording EXACTLY as written, including numeric prefixes.
- Content MUST appear immediately under its heading.
- Use plain paragraphs by default for non-tabular content.
- If SOURCE_CONTEXT contains tabular data, it MUST be rendered as a table. Bullets may be used only if present in SOURCE_CONTEXT.

────────────────────────
HALLUCINATION PREVENTION (NON-NEGOTIABLE)
────────────────────────
- Every factual assertion MUST be directly supported by SOURCE_CONTEXT. Linguistic restructuring, sentence consolidation, and formalization are permitted provided factual meaning is unchanged.
- If a sentence cannot be traced, it MUST be omitted.
- Do NOT generalize beyond explicit statements.
- Do NOT add rationale, examples, assumptions, or clarifications.

────────────────────────
FAIL-SAFE BEHAVIOR
────────────────────────
If the section cannot be authored using SOURCE_CONTEXT alone,
output exactly:
Not in knowledge base.

────────────────────────
FINAL VALIDATION (MANDATORY)
────────────────────────
Before outputting:

IF Output Style = verbatim:
- Verify ALL headings and sub-headings from SOURCE_CONTEXT are present.
- Verify wording matches SOURCE_CONTEXT exactly.

IF Output Style = regulatory author:
- Verify all factual assertions are traceable to SOURCE_CONTEXT.
- Verify no factual content has been added, removed, or altered.
- Verify structural reorganization (if any) remains supported by SOURCE_CONTEXT.

In all modes:
- Verify analytical operations use explicit values only.
- Verify forbidden content is excluded.
- Verify formatting rules are satisfied.

Output ONLY the final authored section content.
""".strip()

##################################################################################last working prompt#####################################################################################



    # Add this before calling client.chat.completions.create
    print("\n=== DEBUG: LLM INPUT SIZE ===")
    print(f"Length of instructions: {len(instructions)} chars (~{len(instructions)//4} tokens)")
    print(f"Length of llm_input: {len(llm_input)} chars (~{len(llm_input)//4} tokens)")
    print(f"Estimated total tokens: ~{(len(instructions) + len(llm_input)) // 4}")
    print(f"Model: {AZURE_OPENAI_CHAT_MODEL}")
    # print(f"Max context (typical): {'128k for gpt-4o' if 'gpt-4o' in AZURE_OPENAI_CHAT_MODEL else '16k for gpt-3.5-turbo'}")
    # print("First 200 chars of llm_input:")
    # print(llm_input[:200])
    # print("Last 200 chars of llm_input:")
    # print(llm_input[-200:])

    response = client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": llm_input},
        ],
        temperature=0.0,
        max_tokens=20000
    )

    new_state = dict(state)
    new_state["answer"] = response.choices[0].message.content
    return new_state

# ============================================================
# BUILD LANGGRAPH (UNCHANGED)
# ============================================================

def build_rag_graph():
    graph_builder = StateGraph(RAGState)

    graph_builder.add_node("retrieve_context", retrieve_context_node)
    graph_builder.add_node("build_prompt", build_prompt_node)
    graph_builder.add_node("generate_answer", generate_answer_node)

    graph_builder.set_entry_point("retrieve_context")
    graph_builder.add_edge("retrieve_context", "build_prompt")
    graph_builder.add_edge("build_prompt", "generate_answer")
    graph_builder.add_edge("generate_answer", END)

    return graph_builder.compile()

rag_graph = build_rag_graph()

# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def answer(query: str, history: List[Dict]) -> str:
    """
    Entry point for AUTHORING requests.
    """
    print(f"[DEBUG] answer() received history of length: {len(history)}")
    if history:
        print(f"Last message: {history[-1]['content'][:80]}...")
    initial_state: RAGState = {
        "query": query,
        "history": history,
    }

    final_state = rag_graph.invoke(initial_state)

    section_name = final_state.get("section_name")
    print("Section name: ",section_name)

    context = final_state.get("context", "")

    processed_answer = normalize_section_numbering(
        final_state.get("answer", ""),
        context
    )

    store_temp_llm_output(
        section_name=section_name,
        llm_text=processed_answer
    )

    return processed_answer


# answer("Summary of Baseline and Clinical Characteristics Safety Population", [])
# answer("DEMOGRAPHIC AND OTHER BASELINE CHARACTERISTICS", [])
# answer("inclusion criteria", [])