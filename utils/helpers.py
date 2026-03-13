python
def build_rag_prompt(query: str, context: str, web_results: str = "") -> str:
    """Build augmented prompt with RAG context and optional web results."""
    prompt = "Answer the following question using the provided context.\n\n"
    if context.strip():
        prompt += f"### Context from documents:\n{context}\n\n"
    if web_results.strip():
        prompt += f"### Additional info from web:\n{web_results}\n\n"
    prompt += f"### Question:\n{query}"
    return prompt
