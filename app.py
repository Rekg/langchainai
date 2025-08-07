import streamlit as st
import os
from dotenv import load_dotenv

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import json

# Load environment variables
load_dotenv()

# --- Functions for core logic ---

def load_documents_and_create_vectorstore(uploaded_file):
    """Loads a PDF, splits it into chunks, and creates a Chroma vector store."""
    with open("temp_doc.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader("temp_doc.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Use Hugging Face embeddings instead of OpenAI
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs, embeddings)

    os.remove("temp_doc.pdf")  # Clean up temporary file
    return vector_store

def create_conversation_chain(vector_store):
    """Sets up the LLM, memory, and conversational retrieval chain."""
    llm = ChatOpenAI(temperature=0.7)  # You can replace this with another local model if needed
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def handle_user_input(user_question):
    """Processes user input and gets a response from the conversation chain."""
    if st.session_state.conversation is None:
        st.warning("Please upload a document and click 'Process' before asking questions.")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

# --- Optional: Save Chat History ---

def save_chat_history(session_id, history):
    if not os.path.exists('chat_history'):
        os.makedirs('chat_history')
    with open(f"chat_history/{session_id}.json", "w") as f:
        json.dump(history, f)

# --- Streamlit UI and Main Loop ---

def main():
    st.set_page_config(page_title="Chat with your Document", page_icon=":books:")
    st.header("Chat with your Document :books:")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your document here and click 'Process'", type=["pdf"])
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    # Get vector store
                    vector_store = load_documents_and_create_vectorstore(pdf_docs)

                    # Create conversation chain
                    st.session_state.conversation = create_conversation_chain(vector_store)
                    st.success("Document processed!")

    # Main chat interface
    user_question = st.text_input(
        "Ask a question about your document:",
        disabled=st.session_state.conversation is None
    )
    if user_question:
        handle_user_input(user_question)

    # Display chat history
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(f"**You:** {message.content}")
            else:
                st.write(f"**Assistant:** {message.content}")

    # Optional: Save chat history
    if st.session_state.chat_history and st.button("Save Chat"):
        session_id = "some_unique_id"  # You may replace with actual session ID logic
        save_chat_history(session_id, [msg.content for msg in st.session_state.chat_history])
        st.success("Chat history saved!")

if __name__ == '__main__':
    main()
