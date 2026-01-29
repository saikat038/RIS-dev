import streamlit as st
import os, sys
import base64
from Protocoldigitization import *

# So we can import from app/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.rag import answer

# ========================
# CONFIG
# ========================

st.set_page_config(
    page_title="Regulatory Authoring Intelligence System",
    layout="wide"
)

# # Path to your logo
# LOGO_PATH = "C:/Users/SaikatSome/OneDrive - Ocugen OpCo Inc/Desktop/RIS-dev/assets/ocugen.png"

# Path relative to this file
LOGO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "assets", 
    "ocugen.png"
)



def get_base64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ========================
# HEADER: CENTERED LOGO + TITLE
# ========================

if os.path.exists(LOGO_PATH):
    logo_b64 = get_base64_image(LOGO_PATH)
    st.markdown(
        f"""
        <div style="text-align:center; margin-bottom: 1rem;">
            <img src="data:image/png;base64,{logo_b64}" style="width:120px;" />
            <h1 style="margin-top:0.5rem;">Regulatory Intelligence System</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Fallback if logo path is wrong
    st.markdown(
        "<h1 style='text-align:center; margin-top: 0.5rem;'>Regulatory Intelligence System</h1>",
        unsafe_allow_html=True,
    )

st.write("")  # small spacing

# ========================
# SESSION STATE FOR CHAT
# ========================

if "messages" not in st.session_state:
    st.session_state.messages = []
    

# ========================
# DISPLAY CHAT HISTORY (TOP → DOWN)
# ========================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ========================
# CHAT INPUT (BOTTOM, CHATGPT-LIKE)
# ========================
history = st.session_state.messages
prompt = st.chat_input("Ask anything about regulations, guidance, policies, IND, etc...")




# ⛔ Stop execution if no new input was submitted
if prompt is None:
    st.stop()

prompt_clean = prompt.strip().lower()

# ========================
# COMMANDS
# ========================

if prompt_clean == "add":
    add_last_section_to_final()

    with st.chat_message("assistant"):
        st.markdown("✅ Section added to final CSR buffer.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": "✅ Section added to final CSR buffer."
    })

elif prompt_clean == "remove":
    remove_last_added_section()

    with st.chat_message("assistant"):
        st.markdown("🗑️ Section removed from final CSR buffer.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": "🗑️ Section removed from final CSR buffer."
    })

elif prompt_clean == "populate":
    render_all_sections()

    with st.chat_message("assistant"):
        st.markdown("📄 Population completed successfully!")

    st.session_state.messages.append({
        "role": "assistant",
        "content": "📄 Population completed successfully!"
    })

# ========================
# NORMAL QUERY → LLM
# ========================
else:
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer(prompt, st.session_state.messages)
            st.markdown(result)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result
    })

