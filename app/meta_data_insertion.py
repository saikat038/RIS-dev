
import os, sys, uuid
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

st.markdown("""
<style>
.rais-btn {
    position: fixed;
    top: 12px;
    left: 270px;   /* 🔥 move right of sidebar */
    z-index: 99999;

    text-decoration: none !important;
    color: black !important;
    font-weight: 600;
    font-size: 14px;

    display: flex;
    align-items: center;
    gap: 6px;

    pointer-events: auto !important;
    opacity: 1 !important;
}

/* override global anchor disabling */
a.rais-btn {
    pointer-events: auto !important;
    opacity: 1 !important;
}
</style>

<a href="https://ris-dev-rvx5qbbut4mydxxnkzn5fz.streamlit.app/" target="_self" class="rais-btn">
    <span style="color:#4A90E2;">⬅️</span>
    RAIS
</a>
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

    /* Disable ALL links (this catches bottom icons) */
    a {
        pointer-events: none !important;
        cursor: default !important;
    }

    /* Optional: make them look disabled */
    a {
        opacity: 0.4;
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

st.title("📄 Authoring Schema Manager")

data = load_schema()

sections = [sec["section"] for sec in data["sections"]]

operation = st.sidebar.selectbox(
    "Select Operation",
    ["View", "Add", "Update", "Delete"]
)


# ================= VIEW =================
if operation == "View":
    st.subheader("All Sections")
    st.json(data)


# ================= ADD =================
elif operation == "Add":
    st.subheader("Add New Section")

    # -------- FORM INPUTS --------
    section = st.text_input("Section")

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
        ["low", "medium", "high"]
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

    if st.button("Delete"):
        data = delete_section(data, selected_section)
        save_schema(data)
        st.success("🗑️ Section deleted successfully")