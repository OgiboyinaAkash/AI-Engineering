from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
import tempfile
import os


def load_pdf(file_bytes: bytes, filename: str) -> list:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = filename
        return docs
    finally:
        os.unlink(tmp_path)


def load_text(file_bytes: bytes, filename: str) -> list:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        loader = TextLoader(tmp_path, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = filename
        return docs
    finally:
        os.unlink(tmp_path)


def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def process_uploaded_file(file_bytes: bytes, filename: str) -> list:
    if filename.lower().endswith(".pdf"):
        docs = load_pdf(file_bytes, filename)
    elif filename.lower().endswith(".txt"):
        docs = load_text(file_bytes, filename)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    return split_documents(docs)


def process_url(url: str) -> list:
    loader = WebBaseLoader(url)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = url
    return split_documents(docs)
