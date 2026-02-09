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
            <h1 style="margin-top:0.5rem;">Regulatory Authoring Intelligence System</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Fallback if logo path is wrong
    st.markdown(
        "<h1 style='text-align:center; margin-top: 0.5rem;'>Regulatory Authoring Intelligence System</h1>",
        unsafe_allow_html=True,
    )

st.write("")  # small spacing

# ========================
# SESSION STATE INITIALIZATION
# ========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "buffer" not in st.session_state:
    st.session_state.buffer = []   # will store selected assistant responses


# ========================
# DISPLAY CHAT HISTORY
# ========================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ========================
# CHAT INPUT + COMMAND HANDLING
# ========================

prompt = st.chat_input("Ask anything about regulations, guidance, policies, IND, etc... or type 'add', 'remove', 'populate'")

if prompt:
    prompt_clean = prompt.strip().lower()

    # ────────────────────────────────────────────────
    # SPECIAL COMMANDS
    # ────────────────────────────────────────────────
    if prompt_clean in ["add", "remove", "populate"]:
        if prompt_clean == "add":
            # Add the LAST assistant message to buffer
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                last_answer = st.session_state.messages[-1]["content"]
                st.session_state.buffer.append(last_answer)
                with st.chat_message("assistant"):
                    st.success(f"Added to buffer ({len(st.session_state.buffer)} items now)")
            else:
                with st.chat_message("assistant"):
                    st.warning("No assistant message available to add")

        elif prompt_clean == "remove":
            if st.session_state.buffer:
                removed = st.session_state.buffer.pop()
                with st.chat_message("assistant"):
                    st.success(f"Removed last item from buffer ({len(st.session_state.buffer)} items left)")
            else:
                with st.chat_message("assistant"):
                    st.warning("Buffer is empty — nothing to remove")

        elif prompt_clean == "populate":
            if st.session_state.buffer:
                with st.chat_message("assistant"):
                    st.markdown("### Buffered Sections (ready for template):")
                    for i, content in enumerate(st.session_state.buffer, 1):
                        st.markdown(f"**Section {i}:**")
                        st.markdown(content)
                        st.markdown("---")
            else:
                with st.chat_message("assistant"):
                    st.info("Buffer is empty — nothing to populate")

        # Add command to history (optional — looks nice in chat)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": st.session_state.messages[-1]["content"]})

    # ────────────────────────────────────────────────
    # NORMAL QUESTION → SEND TO LLM
    # ────────────────────────────────────────────────
    else:
        # Prevent duplicate processing of same prompt
        if "last_processed_prompt" not in st.session_state or st.session_state.last_processed_prompt != prompt:
            st.session_state.last_processed_prompt = prompt

            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Show user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = answer(prompt, st.session_state.messages)
                    st.markdown(result)

            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": result})

# Always re-display full chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
