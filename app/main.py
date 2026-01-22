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
    page_title="Regulatory Intelligence System",
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

if prompt:
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer(prompt, history)
            st.markdown(result)
            render_docx(result)

    st.session_state.messages.append({"role": "assistant", "content": result})
