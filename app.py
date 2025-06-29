# app.py (XtremeFit UI Version)

import streamlit as st
import os
import asyncio
import time
from dotenv import load_dotenv

# Core RAG and Assistant components
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import AsyncInferenceClient

# --- 1. CONFIGURATION AND STYLING ---
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")
st.set_page_config(page_title="XtremeFit Mentor", page_icon="💪", layout="centered")

# CSS to be injected for the XtremeFit dark theme
page_style = """
<style>
/* General App Styling */
body {
    background-color: #0e1117; /* Dark background */
    color: #fafafa;
}

/* Main chat container */
[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
}

/* Chat Input Box */
[data-testid="stChatInput"] {
    background-color: #1f2937; /* Darker input background */
    border-top: 1px solid #374151;
}
[data-testid="stChatInput"] textarea {
    background-color: #374151; /* Input field color */
    color: #fafafa;
}

/* Chat Bubbles */
[data-testid="stChatMessage"] {
    background-color: #1f2937; /* Darker bubbles */
    border: none;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    color: #fafafa;
}

/* User chat bubble specific styling */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #374151; /* Slightly different color for user */
}

/* Conversation Starter Buttons */
.stButton>button {
    background-color: #374151;
    border: 1px solid #4b5563;
    color: #d1d5db;
    padding: 12px 16px;
    border-radius: 8px;
    transition: all 0.2s ease-in-out;
    width: 100%;
    font-weight: 500;
}

.stButton>button:hover {
    background-color: #4b5563;
    border-color: #6b7280;
    color: #ffffff;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)


# --- 2. RAG PIPELINE SETUP (RETRIEVER) ---
@st.cache_resource
def setup_retriever():
    if not os.path.exists("knowledge.txt") or os.path.getsize("knowledge.txt") == 0:
        st.error("`knowledge.txt` file is missing or empty. Please add fitness knowledge to it.")
        return None
    with st.spinner("Loading fitness knowledge base..."):
        try:
            loader = TextLoader("knowledge.txt")
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = FAISS.from_documents(docs, embeddings)
            return db.as_retriever(search_kwargs={"k": 3})
        except Exception as e:
            st.error(f"Error setting up the knowledge base: {e}")
            return None

# --- 3. HUGGING FACE ASSISTANT INTERACTION ---
async def get_assistant_response_stream(user_prompt: str, retriever):
    relevant_docs = retriever.invoke(user_prompt) 
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # This prompt is a wrapper that tells the assistant how to use the context
    # Your main, detailed prompt is already set in the Hugging Face UI
    augmented_prompt = f"""
    Based on the context from the knowledge base provided below, answer the user's question. The context is your primary source of information.

    --- CONTEXT FROM KNOWLEDGE BASE ---
    {context}
    -----------------------------------
    
    USER'S QUESTION:
    {user_prompt}
    """
    
    client = AsyncInferenceClient(token=HF_TOKEN)
    try:
        async for token in await client.chat_completion(messages=[{"role": "user", "content": augmented_prompt}], model=ASSISTANT_ID, max_tokens=2048, stream=True, temperature=0.1):
            if token.choices and token.choices[0].delta.content:
                yield token.choices[0].delta.content
    except Exception as e:
        yield f"An error occurred while connecting to the assistant: {e}"


# --- 4. STREAMLIT APPLICATION UI ---

# Header Section
st.image("https://em-content.zobj.net/source/apple/354/flexed-biceps_1f4aa.png", width=80)
st.title("XtremeFit Mentor")
st.markdown("""
Welcome to the future of fitness coaching. XtremeFit Mentor harnesses the power of advanced AI to be your ultimate wellness partner. Go beyond simple advice: understand complex exercises through curated knowledge and get personalized guidance.
""")

# Main logic
if not HF_TOKEN or not ASSISTANT_ID:
    st.error("Missing Hugging Face Token or Assistant ID. Please check your .env file.", icon="🔒")
else:
    retriever = setup_retriever()
    if retriever:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Function to handle sending a prompt
        def handle_prompt(prompt):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    stream = get_assistant_response_stream(prompt, retriever)
                    for chunk in asyncio.run(stream):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.01)
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"An error occurred: {e}"
                    message_placeholder.error(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()

        # Display Conversation Starters only if the chat is empty
        if not st.session_state.messages:
            st.markdown("---")
            st.markdown("**Examples**")
            cols = st.columns(2)
            starters = [
                "How many calories should I eat to lose weight?",
                "\"I'm new to the gym and feel overwhelmed. Can you help?",
                "What supplements are often discussed for muscle growth?",
                "Explain progressive overload."
            ]
            for i, starter in enumerate(starters):
                if cols[i % 2].button(starter, use_container_width=True, key=f"starter_{i}"):
                    handle_prompt(starter)
                    
        # Main chat input at the bottom
        if prompt := st.chat_input("Ask anything"):
            handle_prompt(prompt)