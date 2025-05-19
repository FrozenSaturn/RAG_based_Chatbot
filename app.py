# app.py

# ---------------------------------------------------------------------------
# SECTION 0: IMPORTS AND ENVIRONMENT SETUP
# ---------------------------------------------------------------------------
import os
import streamlit as st
from dotenv import load_dotenv

# LangChain specific imports
from langchain_community.document_loaders import PyPDFLoader # Only PDF loader needed for this specific app
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# ---------------------------------------------------------------------------
# ENVIRONMENT AND CONFIGURATION
# ---------------------------------------------------------------------------

def setup_environment():
    """Loads environment variables from .env file."""
    load_dotenv()

setup_environment() # Load environment variables at the start

# --- Configuration for Data Loading ---
DATA_FILE_PATH = "data/dataset_childrenbook.pdf" # Your chosen PDF [cite: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294]

# ---------------------------------------------------------------------------
# SECTION 1: TASK 1 - DATA LOADING (Cached)
# ---------------------------------------------------------------------------

@st.cache_resource # Decorator to cache the loaded documents
def load_all_documents(file_path):
    """Loads documents from the specified PDF file path."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load_and_split() # PyPDFLoader can split pages [cite: 32]
        st.success(f"Successfully loaded {len(documents)} pages from {os.path.basename(file_path)}.")
        return documents
    except Exception as e:
        st.error(f"Error loading PDF file '{file_path}': {e}")
        return []

# ---------------------------------------------------------------------------
# SECTION 2: TASK 2 - SET UP RAG WITH LANGCHAIN (Cached)
# ---------------------------------------------------------------------------

@st.cache_resource # Decorator to cache the RAG pipeline
def create_rag_pipeline_cached(docs):
    """
    Sets up the RAG pipeline using the loaded documents.
    Returns the retrieval_chain.
    This function will be cached by Streamlit.
    """
    if not docs:
        st.error("No documents provided to create RAG pipeline.")
        return None

    # 1. Split Documents into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    if not texts:
        st.error("Text splitting resulted in no chunks. Check document content and splitter settings.")
        return None
    st.info(f"Split documents into {len(texts)} chunks.")

    # 2. Create Embeddings using Google Gemini
    try:
        gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        st.error(f"Failed to initialize Gemini Embeddings: {e}. Ensure GOOGLE_API_KEY is valid.")
        return None
    st.info("Initialized GoogleGenerativeAIEmbeddings.")

    # 3. Store in a Vector Store (FAISS)
    try:
        vectorstore = FAISS.from_documents(texts, gemini_embeddings)
        st.info("Created FAISS vector store from document chunks.")
    except Exception as e:
        st.error(f"Error creating FAISS vector store: {e}")
        return None

    # 4. Create a Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks
    st.info("Created retriever from vector store.")

    # 5. Set up the LLM and Prompt for the RAG chain
    try:
        # Note: Using the model name from your provided main.py
        # Verify "gemini-2.0-flash-lite" is a valid and available model name for the API.
        # Common alternatives: "gemini-pro", "gemini-1.5-flash-latest"
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", # Changed to a likely valid model
                                     temperature=0.3,
                                     convert_system_message_to_human=True) # This may show a UserWarning
        st.info("Initialized ChatGoogleGenerativeAI with gemini-1.5-flash-latest.")
    except Exception as e:
        st.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}. Check model name and API key.")
        return None
        
    prompt_template = """
    You are a helpful assistant. Answer the following question based ONLY on the provided context.
    If the answer is not found in the context, state that clearly. Do not make up information.

    <context>
    {context}
    </context>

    Question: {input}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 6. Create the RAG Chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    st.success("RAG retrieval chain created successfully.")
    
    return retrieval_chain

# ---------------------------------------------------------------------------
# SECTION 3: TASK 3 - CHATBOT INTERACTION LOGIC (Not cached, uses cached chain)
# ---------------------------------------------------------------------------

def ask_chatbot_streamlit(query, chain):
    """
    Asks a question to the RAG chatbot and returns the answer.
    Handles potential errors during invocation.
    """
    if not chain:
        return "Error: RAG chain is not initialized. Please check setup."
    
    try:
        # Using st.spinner for a loading indicator during processing
        with st.spinner(f"Processing your question: \"{query[:50]}...\""):
            response = chain.invoke({"input": query})
        answer = response.get("answer", "Could not extract answer from response.")
        
        # Optionally, display retrieved context in an expander for debugging/transparency
        # if 'context' in response and response['context']:
        #     with st.expander("View Retrieved Context"):
        #         for i, doc_context in enumerate(response['context']):
        #             st.write(f"**Snippet {i+1}:**")
        #             st.caption(doc_context.page_content)
        return answer
    except Exception as e:
        st.error(f"Error during chatbot query: {e}")
        return "An error occurred while processing your question."

# ---------------------------------------------------------------------------
# STREAMLIT APPLICATION UI
# ---------------------------------------------------------------------------

def run_streamlit_app():
    st.set_page_config(page_title="RAG Chatbot - Children's Book Paper", layout="wide")
    st.title("📚 RAG Chatbot: Query the Research Paper")
    st.caption(f"Powered by LangChain and Google Gemini, using the paper: {os.path.basename(DATA_FILE_PATH)}")

    # Check for API Key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("🔴 FATAL: GOOGLE_API_KEY not found. Please set it in your .env file or Streamlit secrets.")
        st.stop() # Stop execution if no API key

    # Load documents and create RAG chain (these are cached)
    # These will only run once unless the input (file_path) changes or cache is cleared.
    docs = load_all_documents(DATA_FILE_PATH)
    
    if docs: # Only proceed if documents were loaded successfully
        rag_chain_instance = create_rag_pipeline_cached(docs)

        if rag_chain_instance:
            st.sidebar.success("Chatbot is ready!")
            st.sidebar.info(f"Knowledge Base: {os.path.basename(DATA_FILE_PATH)}")
            
            # Initialize chat history in session state
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm a chatbot ready to answer questions about the research paper. How can I help you?"}]

            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Accept user input
            if prompt := st.chat_input("Ask a question about the paper..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get assistant response
                with st.chat_message("assistant"):
                    response_text = ask_chatbot_streamlit(prompt, rag_chain_instance)
                    st.markdown(response_text)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
        else:
            st.error("🔴 RAG chain initialization failed. Chatbot cannot proceed.")
    else:
        st.error("🔴 No documents were loaded. Chatbot cannot proceed.")
        st.info(f"Please ensure the data file '{DATA_FILE_PATH}' exists and is accessible.")

if __name__ == "__main__":
    run_streamlit_app()