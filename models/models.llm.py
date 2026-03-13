python
import google.generativeai as genai
from config.config import GEMINI_API_KEY

def get_llm_response(prompt: str, mode: str = "concise") -> str:
    """Get response from Gemini (free API)."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)

        system_prompt = (
            "You are a helpful assistant. Be brief and to the point."
            if mode == "concise"
            else "You are a helpful assistant. Provide thorough, detailed explanations with examples."
        )

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"{system_prompt}\n\n{prompt}")
        return response.text

    except Exception as e:
        return f"Gemini Error: {str(e)}"