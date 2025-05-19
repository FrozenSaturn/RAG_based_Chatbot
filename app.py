# app.py

# ---------------------------------------------------------------------------
# SECTION 0: IMPORTS AND ENVIRONMENT SETUP
# ---------------------------------------------------------------------------
import os
import streamlit as st
from dotenv import load_dotenv
import io 
import tempfile

# LangChain specific imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
# MODIFIED: Removed create_stuff_documents_chain and create_retrieval_chain as ConversationalRetrievalChain handles this
from langchain_core.prompts import PromptTemplate # Changed from ChatPromptTemplate for combine_docs_chain
from langchain.chains import ConversationalRetrievalChain # MODIFIED: Added
from langchain.memory import ConversationBufferMemory # MODIFIED: Added
from langchain_core.messages import HumanMessage, AIMessage # For memory if constructing manually, but buffer memory handles this

# ---------------------------------------------------------------------------
# ENVIRONMENT AND CONFIGURATION
# ---------------------------------------------------------------------------

def setup_environment():
    load_dotenv()
setup_environment()

# ---------------------------------------------------------------------------
# SECTION 1: DATA LOADING (Dynamic based on Upload)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Processing PDF...")
def load_documents_from_bytes(pdf_bytes, filename="uploaded_pdf"):
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            temp_file_path = tmp_file.name
        if temp_file_path:
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load_and_split()
            # Add filename to metadata for clarity if needed later, though PyPDFLoader adds 'source'
            for doc in documents:
                doc.metadata["source_filename"] = filename
            st.success(f"Successfully processed '{filename}' into {len(documents)} pages/documents.")
            return documents
        else:
            st.error("Could not create a temporary file for PDF processing.")
            return []
    except Exception as e:
        st.error(f"Error processing PDF file '{filename}': {e}")
        return []
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                st.warning(f"Could not delete temporary file {temp_file_path}: {e}")

# ---------------------------------------------------------------------------
# SECTION 2: SET UP RAG WITH LANGCHAIN (Cached Resource)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Building RAG pipeline with memory...")
def create_rag_pipeline_for_docs(_docs, k_value):
    """
    Sets up the ConversationalRetrievalChain using the loaded documents and specified k_value.
    Returns the conversational chain.
    """
    if not _docs:
        st.error("No documents provided to create RAG pipeline.")
        return None
    
    st.info(f"Building conversational RAG pipeline with k={k_value}...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(_docs)
    if not texts:
        st.error("Text splitting resulted in no chunks.")
        return None
    st.info(f"Split documents into {len(texts)} chunks for vectorization.")

    try:
        gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        st.error(f"Failed to initialize Gemini Embeddings: {e}.")
        return None
    st.info("Initialized GoogleGenerativeAIEmbeddings.")

    try:
        vectorstore = FAISS.from_documents(texts, gemini_embeddings)
        st.info("Created FAISS vector store.")
    except Exception as e:
        st.error(f"Error creating FAISS vector store: {e}")
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": k_value}) 
    st.info(f"Created retriever (k={k_value}).")

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",
                                     temperature=0.3,
                                     convert_system_message_to_human=True) # This flag is for older versions, might not be needed or could be deprecated
        st.info("Initialized ChatGoogleGenerativeAI for answering.")
    except Exception as e:
        st.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}.")
        return None
    
    # MODIFIED: Setup memory for the conversation
    # The memory object will be part of the returned chain, and thus part of the cached resource.
    # Each new chain (due to new PDF or k change) will get its own fresh memory.
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        output_key='answer' # Important: ensures the LLM's answer is stored correctly in history
    )

    # Define the prompt for the document combining step (question answering)
    # Note: ConversationalRetrievalChain might use slightly different variable names internally
    # for context and question than a basic 'stuff' chain.
    # We'll use a more generic prompt template suitable for it.
    # The default prompt for ConversationalRetrievalChain is usually quite good.
    # However, if we want to enforce "answer ONLY from context", we can customize.
    
    # This prompt is for the combine_docs_chain part of ConversationalRetrievalChain
    prompt_template_str = """You are a helpful assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.
Keep the answer concise.

Context:
{context}

Question: {question}

Helpful Answer:"""
    QA_PROMPT = PromptTemplate(template=prompt_template_str, input_variables=["context", "question"])

    # MODIFIED: Create ConversationalRetrievalChain
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True, # We want to see the source documents
        # condense_question_llm=llm, # Optionally use a specific (or same) LLM for condensing the question
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}, # Pass our custom prompt
        verbose=False # Set to True for debugging chain behavior
    )
    st.success("Conversational RAG chain created successfully.")
    return conversational_chain

# ---------------------------------------------------------------------------
# SECTION 3: CHATBOT INTERACTION LOGIC
# ---------------------------------------------------------------------------
def ask_chatbot_streamlit(query, chain):
    if not chain:
        return "Error: RAG chain is not initialized. Please upload and process a PDF.", []
    
    try:
        with st.spinner(f"Thinking... (using conversation history)"):
            # The chain internally uses its memory object for chat_history
            result = chain.invoke({"question": query}) 
            answer = result.get("answer", "Could not extract answer from response.")
            source_documents = result.get("source_documents", [])
        
        if source_documents:
            with st.expander("View Retrieved Context Snippets"):
                for i, doc_context in enumerate(source_documents):
                    page_label = doc_context.metadata.get('page', 'N/A')
                    source_filename = doc_context.metadata.get('source_filename', os.path.basename(doc_context.metadata.get('source', 'N/A')))
                    st.write(f"**Snippet {i+1} (Source: {source_filename}, Page {page_label}):**")
                    st.caption(doc_context.page_content)
        return answer
    except Exception as e:
        st.error(f"Error during chatbot query: {e}")
        return "An error occurred while processing your question."

# ---------------------------------------------------------------------------
# STREAMLIT APPLICATION UI 
# (The run_streamlit_app function remains largely the same as the previous version,
#  as the core logic changes are within create_rag_pipeline_for_docs and how the chain is called.
#  The session state management for rag_chain, messages, current_pdf_name, processed_docs, 
#  and current_k_value should still work correctly.)
# ---------------------------------------------------------------------------
def run_streamlit_app():
    st.set_page_config(page_title="Conversational RAG Chatbot", layout="wide")
    st.title("💬 Conversational RAG Chatbot")
    st.caption("Upload a PDF, configure retrieval, and have a conversation about its content.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state: # This will store our ConversationalRetrievalChain
        st.session_state.rag_chain = None
    if "current_pdf_name" not in st.session_state:
        st.session_state.current_pdf_name = None
    if "processed_docs" not in st.session_state: 
        st.session_state.processed_docs = None
    if "current_k_value" not in st.session_state: 
        st.session_state.current_k_value = 3 

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("🔴 FATAL: GOOGLE_API_KEY not found. Please set it in your .env file or Streamlit secrets.")
        st.sidebar.error("API Key Missing!")
        st.stop()

    with st.sidebar:
        st.header("⚙️ Configuration")
        
        k_value_input = st.number_input(
            "Number of chunks (k) to retrieve:", 
            min_value=1, max_value=10, value=st.session_state.current_k_value, step=1,
            help="Number of relevant text chunks retrieved from the PDF to answer your question."
        )

        st.header("📄 PDF Upload")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        # Determine if context for RAG pipeline needs to be reset/rebuilt
        rebuild_pipeline = False
        if uploaded_file is not None and uploaded_file.name != st.session_state.current_pdf_name:
            st.info(f"New PDF detected: '{uploaded_file.name}'.")
            st.session_state.current_pdf_name = uploaded_file.name
            st.session_state.processed_docs = None # Clear old docs
            st.session_state.messages = [] # Reset chat for new PDF
            rebuild_pipeline = True
        
        if k_value_input != st.session_state.current_k_value:
            st.info(f"Retriever 'k' value changed to {k_value_input}.")
            st.session_state.current_k_value = k_value_input
            rebuild_pipeline = True # Rebuild if k changes

        if rebuild_pipeline:
            create_rag_pipeline_for_docs.clear() # Clear cache for the RAG pipeline function
            st.session_state.rag_chain = None # Ensure old chain is cleared from session

            if uploaded_file: # If new file caused rebuild, re-process it
                 pdf_bytes = uploaded_file.getvalue()
                 st.session_state.processed_docs = load_documents_from_bytes(pdf_bytes, uploaded_file.name)
            
            # If only k changed, processed_docs should still be in session_state from previous load

        # Build/Rebuild RAG chain if we have documents and (it's not built OR needs rebuild)
        if st.session_state.processed_docs:
            if st.session_state.rag_chain is None or rebuild_pipeline:
                st.session_state.rag_chain = create_rag_pipeline_for_docs(
                    st.session_state.processed_docs, 
                    st.session_state.current_k_value
                )
        else: # No processed docs, so no chain
            st.session_state.rag_chain = None
        
        if st.session_state.rag_chain:
            st.success(f"Ready: '{st.session_state.current_pdf_name}' (k={st.session_state.current_k_value})")
        elif st.session_state.current_pdf_name:
            st.warning(f"Processed '{st.session_state.current_pdf_name}', but RAG setup incomplete/failed.")
        
    # Main chat interface
    if not st.session_state.rag_chain:
        st.info("Please upload a PDF and ensure settings are correct to begin chatting.")
    else:
        if not st.session_state.messages: # Initial message if chat is empty
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Hi! I'm ready for a conversation about '{st.session_state.current_pdf_name}' (k={st.session_state.current_k_value}). How can I help?"}
            )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input(f"Ask about '{st.session_state.current_pdf_name}'..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                # The ConversationalRetrievalChain handles history internally via its memory object
                response_text = ask_chatbot_streamlit(prompt, st.session_state.rag_chain)
                st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
        if st.session_state.messages and len(st.session_state.messages) > 1:
            if st.button("Clear Chat History for this PDF"):
                # When clearing history, we also effectively reset the chain's internal memory
                # by creating a new chain instance.
                st.info("Clearing chat history and resetting conversation memory...")
                create_rag_pipeline_for_docs.clear() # Clear the resource cache
                # Rebuild the chain with fresh memory
                if st.session_state.processed_docs:
                    st.session_state.rag_chain = create_rag_pipeline_for_docs(
                        st.session_state.processed_docs, 
                        st.session_state.current_k_value
                    )
                
                initial_message = {"role": "assistant", "content": f"Chat history cleared for '{st.session_state.current_pdf_name}' (k={st.session_state.current_k_value}). Let's start a new conversation!"}
                st.session_state.messages = [initial_message] if st.session_state.rag_chain else []
                st.rerun()

if __name__ == "__main__":
    run_streamlit_app()