import streamlit as st
import os, sys
import base64

# --------------------------------------------------
# SAFE PATH SETUP
# --------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --------------------------------------------------
# CACHED HEAVY IMPORTS (CRITICAL)
# --------------------------------------------------
@st.cache_resource
def load_protocol_module():
    import Protocoldigitization
    return Protocoldigitization

proto = load_protocol_module()

from app.rag import answer


# ========================
# CONFIG
# ========================
st.set_page_config(
    page_title="Regulatory Authoring Intelligence System",
    layout="wide"
)

# ========================
# LOGO PATH
# ========================
LOGO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "assets",
    "ocugen.png"
)

def get_base64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ========================
# HEADER
# ========================
if os.path.exists(LOGO_PATH):
    logo_b64 = get_base64_image(LOGO_PATH)
    st.markdown(
        f"""
        <div style="text-align:center; margin-bottom: 1rem;">
            <img src="data:image/png;base64,{logo_b64}" style="width:120px;" />
            <h1 style="margin-top:0.5rem;">Regulatory Authoring Intelligence System</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<h1 style='text-align:center; margin-top: 0.5rem;'>Regulatory Authoring Intelligence System</h1>",
        unsafe_allow_html=True,
    )

st.write("")

# ========================
# SESSION STATE
# ========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = None


# Display chat history (this should always run)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])



# ────────────────────────────────────────────────
if prompt := st.chat_input(...):

    st.write("DEBUG: new prompt received →", repr(prompt[:80]))

    if prompt == st.session_state.get("last_prompt"):
        st.write("DEBUG: duplicate prompt detected — stopping")
        st.stop()

    st.session_state.last_prompt = prompt

    # ─── very important line ───
    st.write(f"DEBUG: current message count before append = {len(st.session_state.messages)}")

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show what we're actually sending
    MAX_TURNS = 6
    safe_history = st.session_state.messages[-MAX_TURNS:]
    st.write(f"DEBUG: sending {len(safe_history)} messages to LLM")

    # Optional: show rough token estimate
    rough_tokens = sum(len(m["content"]) // 4 + 10 for m in safe_history)
    st.write(f"DEBUG: rough token estimate of history = ~{rough_tokens}")

    # ─── now call your function ───
    with st.spinner("Thinking..."):
        result = answer(prompt, safe_history)
        #                           ^^^^^^^^^^^^  ← must be this truncated version


# ────────────────────────────────────────────────
#           Only process when there's NEW input
# ────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about regulations, guidance, policies, IND, etc..."):

    # ─── Prevent duplicate run on same prompt ───
    if prompt == st.session_state.get("last_prompt"):
        st.stop()           # ← safe, but usually not even needed anymore

    st.session_state.last_prompt = prompt


    # Add user message to history & display it
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)


    prompt_clean = prompt.strip().lower()

    # ─── Commands ───
    if prompt_clean in ("add", "remove", "populate"):

        if prompt_clean == "add":
            proto.add_last_section_to_final()
            response = "✅ Section added to final CSR buffer."

        elif prompt_clean == "remove":
            proto.remove_last_added_section()
            response = "🗑️ Section removed from final CSR buffer."

        elif prompt_clean == "populate":
            proto.render_all_sections()
            response = "📄 Population completed successfully!"

    # ─── Normal LLM query ───
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Hard limit history (good practice)
                MAX_TURNS = 6
                safe_history = st.session_state.messages[-MAX_TURNS:]
                
                result = answer(prompt, safe_history)
                st.markdown(result)

        response = result


    # Add assistant response
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})