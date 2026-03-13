python
import os
import numpy as np
from models.embeddings import embed_texts, embed_query
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS

_index = None
_chunks = []
_embeddings = None

def load_documents(folder: str = "documents") -> list:
    """Load and chunk all documents from the folder."""
    try:
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        print("Install langchain: pip install langchain langchain-community pypdf")
        return []

    docs = []
    if not os.path.exists(folder):
        os.makedirs(folder)
        return []

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                docs.extend(loader.load())
            elif filename.endswith(".txt"):
                loader = TextLoader(filepath)
                docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    return [chunk.page_content for chunk in chunks]

def build_index():
    """Build vector index from documents using numpy (no FAISS needed)."""
    global _index, _chunks, _embeddings
    _chunks = load_documents()
    if not _chunks:
        return False
    _embeddings = embed_texts(_chunks)
    if not _embeddings:
        return False
    _embeddings = np.array(_embeddings).astype("float32")
    _index = True
    return True

def search_documents(query: str) -> str:
    """Retrieve relevant chunks for a query using cosine similarity."""
    global _index, _chunks, _embeddings
    if _index is None:
        if not build_index():
            return ""

    query_vec = embed_query(query)
    if query_vec is None:
        return ""

    query_vec = np.array(query_vec).astype("float32")

    # Cosine similarity
    similarities = np.dot(_embeddings, query_vec) / (
        np.linalg.norm(_embeddings, axis=1) * np.linalg.norm(query_vec)
    )
    top_indices = np.argsort(similarities)[-TOP_K_RESULTS:][::-1]
    results = [_chunks[i] for i in top_indices]
    return "\n\n---\n\n".join(results)
