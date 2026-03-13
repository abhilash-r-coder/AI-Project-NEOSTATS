python
import streamlit as st
from models.llm import get_llm_response
from utils.rag import search_documents, build_index
from utils.web_search import search_web
from utils.helpers import build_rag_prompt

st.set_page_config(page_title="AI Chatbot - Gemini", page_icon=" ", layout="wide")
st.title("Intelligent RAG Chatbot (Gemini)")
st.caption("Powered by Google Gemini Free API")

# Sidebar
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Response Mode", ["concise", "detailed"])
    use_rag = st.checkbox("Use Document RAG", value=True)
    use_web = st.checkbox("Web Search Fallback", value=True)

    st.divider()
    if st.button("Rebuild Document Index"):
        with st.spinner("Indexing documents..."):
            if build_index():
                st.success("Index built!")
            else:
                st.warning("No documents found. Add PDFs/txt to /documents folder.")

    st.divider()
    st.markdown("### How to use RAG")
    st.markdown("1. Put PDF or TXT files in the `documents/` folder\n2. Click **Rebuild Document Index**\n3. Ask questions about your documents!")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = ""
            web_results = ""

            # Step 1: RAG
            if use_rag:
                context = search_documents(user_input)
                if context.strip():
                    st.info("Found relevant documents!")

            # Step 2: Web fallback
            if use_web and not context.strip():
                st.info("Searching web...")
                web_results = search_web(user_input)

            # Step 3: Build prompt & respond
            if context or web_results:
                prompt = build_rag_prompt(user_input, context, web_results)
            else:
                prompt = user_input

            response = get_llm_response(prompt, mode=mode)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})