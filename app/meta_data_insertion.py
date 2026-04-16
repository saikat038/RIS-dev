
import os, sys, uuid
import base64
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import streamlit as st
import json
from azure.storage.blob import BlobClient
params = st.experimental_get_query_params()
is_dev = params.get("dev", ["false"])[0].lower() == "true"

# ================= CONFIG =================
from config.settings import (

    # azure blob
    AUTHOR_SCHEMA_PREFIX,
    AZURE_BLOB_CONN_STRING,
    BLOB_CONTAINER,
)
SCHEMA_FILE = "master_schema.json"


# ========================
# CONFIG
# ========================
st.set_page_config(
    page_title="Regulatory Authoring Intelligence System",
    layout="wide",
    initial_sidebar_state="expanded"
)

col1, col2 = st.columns([1, 10])

with col1:
    st.markdown("""
    <div style="margin-top:10px;">
        <a href="https://ris-dev-rvx5qbbut4mydxxnkzn5fz.streamlit.app/" target="_blank"
        class="rais-link"
        style="
            text-decoration:none;
            color:black;
            font-weight:600;
            font-size:14px;
            display:inline-flex;
            align-items:center;
            gap:6px;
        ">
            <span style="color:#4A90E2;">⬅️</span>
            RAIS
        </a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
/* ✅ Force proper sidebar width */
section[data-testid="stSidebar"] {
    width: 260px !important;
    min-width: 260px !important;
    max-width: 260px !important;
}

/* ✅ Fix inner content so text doesn’t wrap vertically */
section[data-testid="stSidebar"] * {
    white-space: normal !important;
}

/* ✅ Prevent collapse behavior */
section[data-testid="stSidebar"] {
    transform: translateX(0px) !important;
}

/* ❌ Hide collapse button */
[data-testid="collapsedControl"] {
    display: none !important;
}

button[aria-label="Collapse sidebar"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


if not is_dev:
    st.markdown("""
    <style>
    /* Hide header */
    header {visibility: hidden !important;}
    #MainMenu {visibility: hidden !important;}

    /* Disable ALL links EXCEPT RAIS */
    a:not(.rais-link) {
        pointer-events: none !important;
        opacity: 0.4;
        cursor: default !important;
    }

    /* Ensure RAIS looks active */
    a.rais-link {
        pointer-events: auto !important;
        opacity: 1 !important;
        color: black !important;
        cursor: pointer !important;
    }

    /* Fix spacing */
    .block-container {
        padding-top: 1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ================= BLOB FUNCTIONS =================

def get_blob_client():
    blob_path = f"{AUTHOR_SCHEMA_PREFIX}{SCHEMA_FILE}"
    return BlobClient.from_connection_string(
        AZURE_BLOB_CONN_STRING,
        BLOB_CONTAINER,
        blob_path
    )


def load_schema():
    blob_client = get_blob_client()
    raw_bytes = blob_client.download_blob().readall()
    return json.loads(raw_bytes.decode("utf-8"))


def save_schema(data):
    blob_client = get_blob_client()
    blob_client.upload_blob(
        json.dumps(data, indent=2),
        overwrite=True
    )


# ================= CRUD LOGIC =================

def add_section(data, new_entry):
    data["sections"].append(new_entry)
    return data


def update_section(data, section_name, updated_entry):
    for i, sec in enumerate(data["sections"]):
        if sec["section"] == section_name:
            data["sections"][i] = updated_entry
            return data
    return None


def delete_section(data, section_name):
    data["sections"] = [
        sec for sec in data["sections"]
        if sec["section"] != section_name
    ]
    return data


# ================= UI =================

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
            <h1 style="margin-top:0.5rem;">📄 Authoring Schema Manager</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<h1 style='text-align:center; margin-top: 0.5rem;'>Authoring Schema Manager</h1>",
        unsafe_allow_html=True,
    )

st.write("")

data = load_schema()

sections = [sec["section"] for sec in data["sections"]]

operation = st.sidebar.selectbox(
    "Select Operation",
    ["View", "Add", "Update", "Delete"]
)


# ================= VIEW =================
if operation == "View":
    st.subheader("All Sections")

    with st.container(height=550):   # ✅ scrollable window
        st.json(data)               # ✅ original colors preserved


# ================= ADD =================
elif operation == "Add":
    st.subheader("Add New Section")

    # -------- FORM INPUTS --------
    section = st.text_input(
        "Section",
        placeholder="Inclusion Criteria"
    )

    synonyms = st.text_area(
        "Synonyms (comma separated)",
        placeholder="8.1.1. Investigational Product"
    )

    ich_refs = st.text_area(
        "ICH References (comma separated)",
        placeholder="9.4.1 Treatments Administered"
    )

    allowed_sources = st.text_area(
        "Allowed Sources (comma separated)",
        placeholder="ocu400-101-protocol.pdf"
    )

    detail_level = st.selectbox(
        "Detail Level",
        ["high", "medium", "low"]
    )

    output_style = st.selectbox(
        "Output Style",
        ["verbatim", "regulatory author"]
    )

    # -------- NON-EDITABLE FIELD --------
    st.text_input(
        "Forbidden Content",
        value="operational procedures",
        disabled=True
    )

    # -------- ADD BUTTON --------
    if st.button("Add Section"):
        try:
            if not section.strip():
                st.error("Section name is required")
            else:
                new_entry = {
                    "section": section.strip(),
                    "synonyms": [s.strip() for s in synonyms.split(",") if s.strip()],
                    "ich_refs": [i.strip() for i in ich_refs.split(",") if i.strip()],
                    "allowed_sources": [a.strip() for a in allowed_sources.split(",") if a.strip()],
                    "detail_level": detail_level,
                    "output_style": output_style,
                    "forbidden_content": ["operational procedures"]   # 🔒 FIXED
                }

                # duplicate check
                if any(sec["section"] == section for sec in data["sections"]):
                    st.warning("⚠️ Section already exists")
                else:
                    data = add_section(data, new_entry)
                    save_schema(data)
                    st.success("✅ Section added successfully")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# ================= UPDATE =================
elif operation == "Update":
    st.subheader("Update Section")

    selected_section = st.selectbox("Select Section", sections)

    existing_data = next(
        sec for sec in data["sections"]
        if sec["section"] == selected_section
    )

    user_input = st.text_area(
        "Edit JSON",
        value=json.dumps(existing_data, indent=2),
        height=500
    )

    if st.button("Update"):
        try:
            updated_entry = json.loads(user_input)

            updated_data = update_section(
                data,
                selected_section,
                updated_entry
            )

            if not updated_data:
                st.warning("📄 No section found in template")
            else:
                save_schema(updated_data)
                st.success("✅ Section updated successfully")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# ================= DELETE =================
elif operation == "Delete":
    st.subheader("Delete Section")

    selected_section = st.selectbox("Select Section", sections)

    # 🔴 Trigger popup
    if st.button("Delete"):
        st.session_state.show_delete_popup = True

    # 🔥 Popup-like UI
    if st.session_state.get("show_delete_popup", False):
        with st.container(border=True):
            st.warning("⚠️ Confirm Deletion")

            password = st.text_input("Enter Admin Password", type="password")

            col1, col2, col3 = st.columns([1, 1, 6])  # 🔥 third column pushes left

            with col1:
                if st.button("Confirm Delete"):
                    if password == "Admin123":
                        data = delete_section(data, selected_section)
                        save_schema(data)

                        st.success("🗑️ Section deleted successfully")

                        # close popup
                        st.session_state.show_delete_popup = False
                    else:
                        st.error("❌ Incorrect password")

            with col2:
                if st.button("Cancel"):
                    st.session_state.show_delete_popup = False