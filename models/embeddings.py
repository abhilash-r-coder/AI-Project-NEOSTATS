python
import google.generativeai as genai
from config.config import GEMINI_API_KEY

def embed_texts(texts: list) -> list:
    """Generate embeddings using Gemini's free embedding model."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        return embeddings
    except Exception as e:
        print(f"Embedding error: {e}")
        return []

def embed_query(query: str):
    """Generate embedding for a single query."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        result = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        print(f"Query embedding error: {e}")
        return None