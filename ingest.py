import os
import uuid
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import (
    DOCUMENTS_DIR,
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)


def load_recipe_documents(doc_dir: str):
    """
    Load recipe documents from PDFs and .txt files.
    You should place at least ~10 recipe docs into the data/ folder.
    """
    base = Path(doc_dir)
    pdf_paths = list(base.glob("*.pdf"))
    txt_paths = list(base.glob("*.txt"))

    if len(pdf_paths) + len(txt_paths) == 0:
        print(f"No documents found in {doc_dir}. Add some recipe PDFs or .txt files.")
        return []

    docs = []

    # Load PDFs
    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = pdf_path.name
                page.metadata["type"] = "pdf"
            docs.extend(pages)
        except Exception as e:
            print(f"Error loading PDF {pdf_path}: {e}")

    # Load text files (one recipe per file)
    for txt_path in txt_paths:
        try:
            loader = TextLoader(str(txt_path), encoding="utf-8")
            texts = loader.load()
            for t in texts:
                t.metadata["source"] = txt_path.name
                t.metadata["type"] = "text"
            docs.extend(texts)
        except Exception as e:
            print(f"Error loading text file {txt_path}: {e}")

    if len(docs) < 10:
        print(f"Warning: only {len(docs)} documents loaded. "
              f"Assignment recommends at least 10.")

    return docs


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(docs)


def main():
    print(f"Loading documents from: {DOCUMENTS_DIR}")
    docs = load_recipe_documents(DOCUMENTS_DIR)
    if not docs:
        print("No documents to index. Exiting.")
        return

    chunks = split_docs(docs)
    print(f"Total chunks: {len(chunks)}")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=Settings(allow_reset=True),
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts: List[str] = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]

    print("Computing embeddings (this may take a bit)...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True).tolist()

    print("Upserting into Chroma...")
    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print("Ingestion complete! Chroma DB is ready.")


if __name__ == "__main__":
    main()

