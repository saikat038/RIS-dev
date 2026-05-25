from pypdf import PdfReader, PdfWriter


def crop_first_50_pages(input_path: str, output_path: str, max_pages: int = 50):
    """
    Crops the first `max_pages` pages from a PDF and saves them as a new PDF.

    :param input_path: Path to input PDF
    :param output_path: Path to save cropped PDF
    :param max_pages: Number of pages to keep (default = 50)
    """

    reader = PdfReader(input_path)
    writer = PdfWriter()

    total_pages = len(reader.pages)
    pages_to_keep = min(max_pages, total_pages)

    print(f"Total pages in input PDF: {total_pages}")
    print(f"Cropping first {pages_to_keep} pages...")

    for page_num in range(8,pages_to_keep):
        writer.add_page(reader.pages[page_num])

    with open(output_path, "wb") as output_file:
        writer.write(output_file)

    print(f"New PDF saved at: {output_path}")


if __name__ == "__main__":
    input_pdf = r"C:\Users\SaikatSome\OneDrive - Ocugen OpCo Inc\India BU - Artificial Intelligence\RAIS\Blob Storage Replica\E3_Guideline-CSR_N.pdf"
    output_pdf = r"C:\Users\SaikatSome\Downloads\miriyala\OCU400-Protocol, SAP & Interim CSR\E3_Guideline-CSR_N.pdf"

    crop_first_50_pages(input_pdf, output_pdf, 51)