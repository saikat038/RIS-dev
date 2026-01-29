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
from typing import List, Dict, TypedDict
from azure.core.exceptions import HttpResponseError

from openai import AzureOpenAI
from langgraph.graph import StateGraph, END
from azure.search.documents.models import VectorizedQuery
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

from ingest.embed import batch_embed
from azure.storage.blob import BlobClient
from Protocoldigitization import *

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

print("ICH index name: ",AZURE_ICH_SEARCH_INDEX_NAME)
# ============================================================
# LOAD AUTHORING SCHEMA FROM BLOB STORAGE
# ============================================================
def load_authoring_schema_from_blob(schema_name: str) -> dict:
    """
    Load authoring control schema JSON directly from Azure Blob Storage
    into memory (RAM) without downloading to disk.
    """
    print("name of the shcema: ",schema_name)
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
    Pick the most relevant section control from the master schema based on the user's request.
    This is a deterministic, non-LLM matcher (fast and auditable).
    """

    q = (user_query or "").lower()
    sections = authoring_control.get("sections", [])

    # 1) direct keyword match on 'section'
    for sec in sections:
        name = (sec.get("section") or "").lower()
        if name and name in q:
            return sec

    # 2) match on optional synonyms if present
    for sec in sections:
        synonyms = sec.get("synonyms", [])
        if isinstance(synonyms, list):
            for s in synonyms:
                if isinstance(s, str) and s.lower() in q:
                    return sec

    # 3) fallback (you can make this stricter)
    # If you prefer strict behavior, raise error instead.
    return sections[0] if sections else {}
    # return {}


    

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
    Format SOURCE chunks.
    Table chunks are preserved semantically (not structurally).
    """
    if not isinstance(chunk, dict):
        return str(chunk)

    text = (chunk.get("text") or "").strip()
    if not text:
        return ""

    chunk_type = chunk.get("chunk_type")
    heading = chunk.get("heading_path")
    pages = chunk.get("page_numbers")

    meta = []
    if chunk_type:
        meta.append(f"type={chunk_type}")
    if heading:
        meta.append(f"section={heading}")
    if pages:
        meta.append(f"pages={pages}")

    meta_line = f"[{', '.join(meta)}]" if meta else ""

    return f"{meta_line}\n{text}".strip()


def build_generic_query(payload: dict) -> str:
    section = payload.get("section", "").strip()
    synonyms = payload.get("synonyms", [])

    lines = []

    if section:
        lines.append(f"{section}.")

    if synonyms:
        lines.append("Also look for content related to the following terms:")
        for term in synonyms:
            if term != section:
                lines.append(f"- {term}")

    return "\n".join(lines)


import re

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
    query,
    k_nearest_neighbors=100,
    filter_expr=None,
):
    
    min_results = 3
    # ---------- Always-safe guards ----------
    if not query or not isinstance(query, str):
        return []   # or log and return empty safely

    try:
        q_vec = batch_embed([query])[0]
    except Exception:
        # Embedding failure → safest fallback
        return []

    vector_query = VectorizedQuery(vector=q_vec, fields="vector")

    # ---------- Tier 1: vector + filter ----------
    try:
        filtered_results = list(
            search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                filter=filter_expr,
                top=k_nearest_neighbors,
                select=["text", "section_path", "rule_type"]
            )
        )

        if len(filtered_results) >= min_results:
            return [dict(r) for r in filtered_results]

    except HttpResponseError:
        # Filter/schema issues → fallback
        pass

    except Exception:
        # Any unexpected Azure/runtime issue → fallback
        pass

    # ---------- Tier 2: vector-only fallback ----------
    try:
        fallback_results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=k_nearest_neighbors,
            select=["text", "section_path", "rule_type"]
        )

        return [dict(r) for r in fallback_results]

    except Exception:
        # Absolute worst-case safety net
        return []



def vector_search_source(search_client, query, k_nearest_neighbors=100, filter_expr=None):
    q_vec = batch_embed([query])[0]
    vector_query = VectorizedQuery(vector=q_vec, fields="vector")

    results = search_client.search(
        search_text="",
        vector_queries=[vector_query],
        filter=filter_expr,
        top=k_nearest_neighbors,
        select=[
            "text",
            "chunk_type",
            "heading_path",
            "page_numbers",
            "doc_id",
        ]
    )
    return [dict(r) for r in results]


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

    query = query = state.get("query", "")

    # -------------------------------------------------
    # PICK ACTIVE AUTHORING CONTROL
    # -------------------------------------------------
    active_control = pick_active_control(AUTHORING_CONTROL, query)

    if not active_control:
        new_state = dict(state)
        new_state["answer"] = build_missing_section_message(AUTHORING_CONTROL)
        new_state["context"] = ""
        new_state["section_name"] = None
        return new_state


    # -------------------------------------------------
    # ICH RETRIEVAL (MANDATORY)
    # -------------------------------------------------
    ich_client = load_ich_search_client()

    section_name = active_control.get("section", "")
    ich_refs = active_control.get("ich_refs", [])

    section_number,section_text  =split_section(ich_refs[0])
    
    filter_expr = (
        f"section_path eq '{section_number}' "
        f"and section_title eq '{section_text}'"
        
    )
    ich_query_parts = build_generic_query({k: active_control[k] for k in ('section', 'synonyms')})
    print("part ich query:", ich_query_parts)

    # optional boost terms if your schema has them
    # if active_control.get("output_style"):
    #     ich_query_parts.append(active_control["output_style"])
    # if active_control.get("detail_level"):
    #     ich_query_parts.append(active_control["detail_level"])

    # ich_query = " ".join([p for p in ich_query_parts if p])
    ich_query = ich_query_parts

    print("Final ich query:", ich_query)

    query = build_generic_query({k: active_control[k] for k in ("section", "synonyms")})
    print("matched shema: ", active_control)
    print("This is the final query:",type(query))

    ich_chunks = vector_search_ich(ich_client, ich_query, k_nearest_neighbors=100, filter_expr=filter_expr)

    ich_context_pieces = [
        (chunk.get("text") or "").strip()
        for chunk in ich_chunks
        if isinstance(chunk, dict) and chunk.get("text")
    ]


    ich_context = (
        "\n\n".join(ich_context_pieces)
        if ich_context_pieces
        else "No ICH guidance found."
    )


    # -------------------------------------------------
    # SOURCE RETRIEVAL (EVIDENCE + TABLES)
    # -------------------------------------------------
    source_client = load_source_search_client()

    filter_expr = None  # source index does not support doc_type filtering


    # If filter is not supported in your index, just call without filter
    source_chunks = vector_search_source(
        source_client,
        query,
        k_nearest_neighbors=100
    )



    source_context_pieces = []
    for chunk in source_chunks:
        formatted = format_chunk_for_context(chunk)
        if formatted:
            source_context_pieces.append(formatted)

    source_context = (
        "\n\n".join(source_context_pieces)
        if source_context_pieces
        else "No source evidence found."
    )


    # -------------------------------------------------
    # FINAL MERGED CONTEXT
    # -------------------------------------------------
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

    # 🚨 AUTHORING PROMPT — UNCHANGED
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

# STRICT TABLE RULE (MANDATORY):
# - When answering from a table, you MUST:
#   1. Identify the exact row(s) used
#   2. Ensure ALL relevant columns for that row are present
# - If ANY required column or cell is missing, reply exactly:
#   Not in knowledge base.
# - NEVER infer, assume, merge, or reconstruct missing table cells.


# Important:
# - **Do not invent data** that is not supported by or logically derivable from the context.
# """.strip()
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
   - All authored sentences MUST be traceable to explicit statements here.

2. ICH_CONTEXT
   - Provides regulatory structure and terminology ONLY.
   - MUST NOT introduce new facts, criteria, thresholds, or content.

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
1. Identify the exact row(s) used.
2. Use ONLY explicit cell content.
3. A single cell may be decomposed into multiple items ONLY if explicitly written.
4. Do NOT infer missing cells, relationships, or intent.

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
STRUCTURAL EXTRACTION & RENDERING (HIGHEST PRIORITY)
────────────────────────
Before writing any content:

- Scan SOURCE_CONTEXT line-by-line.
- Identify ALL structural elements in order:
  • Headings
  • Sub-headings
  • Group labels

STRUCTURAL RULES:
- Structural elements are IMMUTABLE TOKENS.
- ALL identified headings and sub-headings MUST be rendered.
- Structural rendering takes precedence over content completeness checks.
- Structural elements MUST be rendered EVEN IF associated content is minimal or empty.
- You MUST NOT omit, merge, flatten, or downgrade structure.

If a line qualifies as a heading or sub-heading, it MUST be rendered.

────────────────────────
OUTPUT STYLE RULES
────────────────────────

IF Output Style = verbatim:
- Preserve wording EXACTLY as written in SOURCE_CONTEXT.
- Preserve structure, hierarchy, and ordering.
- Preserve headings and sub-headings exactly.
- Remove numeric prefixes (e.g., 7.1, 1., 1.1).
- Do NOT paraphrase, summarize, normalize, or interpret.

IF Output Style = regulatory author:
- Author using formal regulatory language consistent with ICH E3.
- Reorganize or consolidate ONLY when explicitly supported by SOURCE_CONTEXT.
- Do NOT introduce new criteria, rationale, interpretation, or procedures.
- Do NOT operationalize content.

────────────────────────
FORMAT & STRUCTURE ENFORCEMENT
────────────────────────
- Begin directly with the section content.
- Do NOT add introductions or framing statements.
- Do NOT restate or rename the section.
- Headings and sub-headings MUST:
  • Appear on their own line
  • Be formatted in **bold markdown**
  • Preserve original wording (numbering removed)
- Content MUST appear immediately under its heading.
- Use plain paragraphs by default.
- Use bullets or tables ONLY if present in SOURCE_CONTEXT or required for clarity.

────────────────────────
HALLUCINATION PREVENTION (NON-NEGOTIABLE)
────────────────────────
- Every sentence MUST be directly supported by SOURCE_CONTEXT.
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
- Verify ALL headings and sub-headings from SOURCE_CONTEXT are present.
- Verify every sentence is traceable to SOURCE_CONTEXT.
- Verify analytical operations use explicit values only.
- Verify forbidden content is excluded.
- Verify formatting rules are satisfied.

Output ONLY the final authored section content.
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

    response = client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": llm_input},
        ],
        temperature=0.0,
        max_tokens=4500,
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
    initial_state: RAGState = {
        "query": query,
        "history": history,
    }

    final_state = rag_graph.invoke(initial_state)

    section_name = final_state.get("section_name")
    print("Section name: ",section_name)

    store_temp_llm_output(
        section_name=section_name,
        llm_text=final_state["answer"]
    )

    return final_state.get("answer", "")


answer("Subject Disposition Screening Population - RP Patients", [])