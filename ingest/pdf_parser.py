import fitz  # PyMuPDF
import pdfplumber

def extract_text_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text")

        pages.append({
            "page_number": i + 1,
            "text": text
        })

    doc.close()
    return pages


def extract_tables(pdf_path):
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()

            for t in page_tables:
                if t:
                    tables.append({
                        "page_number": i + 1,
                        "data": t
                    })

    return tables


def process_pdf(pdf_path):

    print(f"\nProcessing: {pdf_path}")

    text_pages = extract_text_pymupdf(pdf_path)
    tables = extract_tables(pdf_path)

    return {
        "text_pages": text_pages,
        "tables": tables
    }