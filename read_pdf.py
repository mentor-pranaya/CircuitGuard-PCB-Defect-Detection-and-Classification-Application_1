import sys
import os

pdf_path = "AI - CircuitGuard.pdf"
output_path = "pdf_content.txt"

try:
    import pypdf
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Using pypdf\n")
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            f.write(text + "\n")
    print("Done writing to " + output_path)
except ImportError:
    print("pypdf not installed")
    try:
        import PyPDF2
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Using PyPDF2\n")
            with open(pdf_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfFileReader(pdf_file)
                for page in range(reader.numPages):
                    text = reader.getPage(page).extractText()
                    f.write(text + "\n")
        print("Done writing to " + output_path)
    except ImportError:
        print("PyPDF2 not installed")
