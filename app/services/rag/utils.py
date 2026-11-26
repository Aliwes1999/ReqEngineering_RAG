from pathlib import Path
from pypdf import PdfReader
from docx import Document
import pandas as pd


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []

    for page in reader.pages:
        texts.append(page.extract_text() or "")

    return "\n".join(texts)


def read_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


def read_xlsx(path: str) -> str:
    excel = pd.ExcelFile(path)
    parts = []

    for sheet in excel.sheet_names:
        df = excel.parse(sheet)
        parts.append(f"=== Sheet: {sheet} ===")
        parts.append(df.fillna("").astype(str).to_string(index=False))

    return "\n".join(parts)


def read_file(path: str) -> str:
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        return read_pdf(path)
    elif ext == ".docx":
        return read_docx(path)
    elif ext == ".xlsx":
        return read_xlsx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150):
    """
    Split text into overlapping chunks.

    chunk_size=800 is safer for local embedding models than 8000.
    overlap=150 keeps some context between chunks.
    """
    chunks = []
    start = 0
    text = text.strip()

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks