python
import google.generativeai as genai
from config.config import GEMINI_API_KEY

def search_web(query: str) -> str:
    """Use Gemini itself as a web-knowledge fallback (no extra API needed)."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            f"Search the web and provide the latest information about: {query}\n"
            f"Provide factual, up-to-date information with sources if possible."
        )
        return response.text
    except Exception as e:
        return f"Web search error: {str(e)}"