import streamlit as st
import os, sys
import base64

params = st.experimental_get_query_params()
is_dev = params.get("dev", ["false"])[0].lower() == "true"
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

st.markdown("""
<style>
/* Hide left-side cloud/app icon */
header div[data-testid="stToolbar"] > div:first-child {
    display: none !important;
}

/* Hide red crown / deploy badge */
header button[title*="Deploy"],
header div[title*="Deploy"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

if not is_dev:
    st.markdown("""
    <style>
    /* Hide full Streamlit header */
    header {visibility: hidden;}

    /* Hide 3 dots menu */
    #MainMenu {visibility: hidden;}

    /* Hide footer */
    footer {visibility: hidden;}

    /* Fix spacing after removing header */
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

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

        try:

            if prompt_clean == "add":
                proto.add_last_section_to_final()
                response = "✅ Section added to final CSR buffer."

            elif prompt_clean == "remove":
                proto.remove_last_added_section()
                response = "🗑️ Section removed from final CSR buffer."

            elif prompt_clean == "populate":
                proto.render_all_sections()
                response = "📄 Population completed successfully!"

        except ValueError as e:
            response = str(e)

        # Show command response
        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    # ─── Normal LLM query ───
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                MAX_TURNS = 6
                safe_history = st.session_state.messages[-MAX_TURNS:]
                
                result = answer(prompt, safe_history)
                st.markdown(result)

        st.session_state.messages.append({"role": "assistant", "content": result})