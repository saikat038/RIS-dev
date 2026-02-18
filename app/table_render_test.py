from docxtpl import DocxTemplate
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.part import Part
from io import BytesIO
import uuid



SECTION_TO_TEMPLATE_VAR = {
    "Subject Information and Consent": "subject_information_and_consent"
}


def generate_random_table_altchunk() -> str:
    """
    Returns a raw Word altChunk containing a table.
    Formatting is intentionally untouched.
    """
    return """
<w:altChunk xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
            r:id="table_chunk_1"/>
"""


def attach_table_as_altchunk(doc: DocxTemplate, table_docx_bytes: bytes) -> str:
    """
    Attaches a DOCX containing a table as an altChunk.
    Returns the relationship ID.
    """
    partname = f"/word/altChunk-{uuid.uuid4()}.docx"
    content_type = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    part = Part(
        partname=partname,
        content_type=content_type,
        blob=table_docx_bytes,
        package=doc.docx.package
    )

    r_id = doc.docx.part.relate_to(part, RT.ALT_CHUNK)

    return r_id


def render_subject_information_and_consent(
    template_path: str,
    output_path: str,
    table_docx_bytes: bytes
):
    doc = DocxTemplate(template_path)

    # Attach the table chunk
    r_id = attach_table_as_altchunk(doc, table_docx_bytes)

    # Build altChunk XML
    altchunk_xml = f"""
<w:altChunk xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
            r:id="{r_id}"/>
"""

    context = {
        SECTION_TO_TEMPLATE_VAR["Subject Information and Consent"]: altchunk_xml
    }

    doc.render(context)
    doc.save(output_path)
