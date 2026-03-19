import streamlit as st
import os
import json
import uuid
from datetime import datetime
from rag_backend import get_rag_chain

# --- 1. Configuration & Storage Setup ---
st.set_page_config(page_title="AI Mentor RAG", layout="wide")
CHAT_DIR = "./chat_sessions"
DB_DIR = "./chroma_db_storage"
os.makedirs(CHAT_DIR, exist_ok=True) 

# Bulletproof check: Does the folder exist AND does it actually have files inside it?
db_is_ready = os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0

# --- 2. Helper Functions (LLMOps Logging) ---
def save_chat(chat_id, title, messages):
    file_path = os.path.join(CHAT_DIR, f"{chat_id}.json")
    with open(file_path, "w") as f:
        json.dump({"title": title, "messages": messages, "updated_at": str(datetime.now())}, f)

def load_chat(chat_id):
    file_path = os.path.join(CHAT_DIR, f"{chat_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {"title": "New Chat", "messages": []}

def get_all_chats():
    chats = []
    for filename in os.listdir(CHAT_DIR):
        if filename.endswith(".json"):
            chat_id = filename.replace(".json", "")
            data = load_chat(chat_id)
            chats.append({"id": chat_id, "title": data.get("title", "Untitled"), "updated_at": data.get("updated_at", "")})
    return sorted(chats, key=lambda x: x["updated_at"], reverse=True)

# --- 3. Session State Initialization ---
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. Sidebar UI (Chat History & File Upload) ---
with st.sidebar:
    st.title("💬 Chat History")
    
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_chat_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun() 
        
    st.divider()
    
    for chat in get_all_chats():
        if st.button(chat["title"], key=chat["id"], use_container_width=True):
            st.session_state.current_chat_id = chat["id"]
            st.session_state.messages = load_chat(chat["id"])["messages"]
            st.rerun()
            
    st.divider()
    st.caption("Admin Controls")
    
    # Document Uploader Logic
    if not db_is_ready:
        st.warning("No database found. Please upload a PDF.")
        uploaded_file = st.file_uploader("Upload Knowledge Base (PDF)", type="pdf")
        if uploaded_file:
            with st.spinner("Processing & Embedding..."):
                with open("temp_doc.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.rag_chain = get_rag_chain("temp_doc.pdf")
                st.success("Database Built! Ready to chat.")
                st.rerun() # Refresh the UI to hide the uploader!
    else:
        st.success("✅ Vector Database Active")
        # ONLY initialize the chain if the DB is actually ready!
        if "rag_chain" not in st.session_state:
            st.session_state.rag_chain = get_rag_chain()

# --- 5. Main Chat Interface ---
st.title("📄 Persistent AI Mentor")

# Prevent the user from chatting if the database isn't built yet
if not db_is_ready:
    st.info("👈 Please upload a document in the sidebar to begin.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke(user_query)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        chat_title = "New Chat"
        if len(st.session_state.messages) <= 2:
            chat_title = user_query[:25] + "..."
        else:
            chat_title = load_chat(st.session_state.current_chat_id).get("title", "Chat")

        save_chat(st.session_state.current_chat_id, chat_title, st.session_state.messages)