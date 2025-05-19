# app.py

# ---------------------------------------------------------------------------
# SECTION 0: IMPORTS AND ENVIRONMENT SETUP
# ---------------------------------------------------------------------------
import os
import streamlit as st
from dotenv import load_dotenv
import io 
import tempfile
import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ---------------------------------------------------------------------------
# ENVIRONMENT AND CONFIGURATION
# ---------------------------------------------------------------------------

def setup_environment():
    load_dotenv()
setup_environment()

# ---------------------------------------------------------------------------
# SECTION 1: DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="⏳ Processing PDF...")
def load_documents_from_bytes(pdf_bytes, filename="uploaded_pdf"):
    """Loads and splits documents from PDF bytes using a temporary file."""
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            temp_file_path = tmp_file.name
        if temp_file_path:
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load_and_split()
            for doc in documents: 
                doc.metadata["source_filename"] = filename
            st.success(f"✅ Successfully processed '{filename}'.")
            return documents
        else:
            st.error("⚠️ Could not create a temporary file for PDF processing.")
            return []
    except Exception as e:
        st.error(f"⚠️ Error processing PDF file '{filename}': {e}")
        return []
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                st.warning(f"⚠️ Could not delete temporary file {temp_file_path}: {e}")

# ---------------------------------------------------------------------------
# SECTION 2: SET UP RAG WITH LANGCHAIN
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="🛠️ Building RAG pipeline with memory...")
def create_rag_pipeline_for_docs(_docs, k_value):
    """
    Sets up the ConversationalRetrievalChain using the loaded documents and specified k_value.
    """
    if not _docs:
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(_docs)
    if not texts:
        st.error("⚠️ Text splitting resulted in no usable chunks.")
        return None

    try:
        gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        st.error(f"⚠️ Failed to initialize Gemini Embeddings: {e}. Check API key.")
        return None

    try:
        vectorstore = FAISS.from_documents(texts, gemini_embeddings)
    except Exception as e:
        st.error(f"⚠️ Error creating FAISS vector store: {e}")
        return None

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k_value, "fetch_k": 20, "lambda_mult": 0.5}
    ) 

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",
                                     temperature=0.3,
                                     convert_system_message_to_human=True)
    except Exception as e:
        st.error(f"⚠️ Failed to initialize ChatGoogleGenerativeAI: {e}. Check model name.")
        return None
    
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        output_key='answer'
    )

    prompt_template_str = """You are a helpful assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.
Keep the answer concise and helpful.

Context:
{context}

Question: {question}

Helpful Answer:"""
    QA_PROMPT = PromptTemplate(template=prompt_template_str, input_variables=["context", "question"])

    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_generated_question=True,
        verbose=False 
    )
    return conversational_chain

# ---------------------------------------------------------------------------
# SECTION 3: CHATBOT INTERACTION LOGIC
# ---------------------------------------------------------------------------
def ask_chatbot_streamlit(query, chain):
    """Asks a question to the RAG chatbot and returns answer and generated_question."""
    if not chain:
        return "⚠️ Error: RAG chain is not initialized. Please upload and process a PDF.", None
    
    try:
        with st.spinner(f"🧠 Thinking..."):
            result = chain.invoke({"question": query}) 
            answer = result.get("answer", "⚠️ Could not extract answer from response.")
            source_documents = result.get("source_documents", []) 
            generated_question = result.get("generated_question")
        
        if source_documents:
            with st.expander("🔍 View Retrieved Context Snippets"):
                for i, doc_context in enumerate(source_documents):
                    page_label = doc_context.metadata.get('page', 'N/A')
                    source_filename = doc_context.metadata.get('source_filename', os.path.basename(doc_context.metadata.get('source', 'N/A')))
                    st.write(f"**Snippet {i+1} (Source: {source_filename}, Page {page_label}):**")
                    st.caption(doc_context.page_content)
        
        return answer, generated_question
    
    except Exception as e:
        st.error(f"⚠️ Error during chatbot query: {e}")
        return "⚠️ An error occurred while processing your question.", None

# ---------------------------------------------------------------------------
# SECTION 4: UTILITY FUNCTIONS
# ---------------------------------------------------------------------------
def format_chat_history_for_download(messages):
    """Formats chat history into a readable string for download."""
    history_str = f"Chat History ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n"
    history_str += "=" * 40 + "\n\n"
    for message in messages:
        history_str += f"{message['role'].capitalize()}: {message['content']}\n\n"
    return history_str

# ---------------------------------------------------------------------------
# SECTION 5: STREAMLIT APPLICATION UI 
# ---------------------------------------------------------------------------
def run_streamlit_app():
    st.set_page_config(page_title="PDF Chatbot Pro", layout="wide", initial_sidebar_state="expanded")
    
    st.markdown("""
        <style>
            .stButton>button {
                border-radius: 0.5rem;
            }
            .stExpander {
                border: 1px solid #e0e0e0;
                border-radius: 0.5rem;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("✨ PDF Chatbot Pro")
    st.caption("Upload a PDF, configure retrieval, and have an intelligent conversation about its content.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "current_pdf_name" not in st.session_state:
        st.session_state.current_pdf_name = None
    if "processed_docs" not in st.session_state: 
        st.session_state.processed_docs = None
    if "current_k_value" not in st.session_state: 
        st.session_state.current_k_value = 3 
    if "copy_text_content" not in st.session_state:
        st.session_state.copy_text_content = None
    if "show_copy_expander" not in st.session_state:
        st.session_state.show_copy_expander = False

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("🔴 **FATAL: GOOGLE_API_KEY not found.** Please set it in your `.env` file or Streamlit secrets.")
        st.sidebar.error("API Key Missing!")
        st.stop()

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")
        
        k_value_input = st.number_input(
            "Chunks to retrieve (k):", 
            min_value=1, max_value=10, value=st.session_state.current_k_value, step=1,
            help="Number of relevant text chunks from the PDF used to answer your question."
        )
        st.markdown("---")
        st.header("📄 PDF Upload")
        uploaded_file = st.file_uploader("Choose a PDF file to chat with", type="pdf", label_visibility="collapsed")

        rebuild_pipeline = False
        if uploaded_file is not None and uploaded_file.name != st.session_state.current_pdf_name:
            st.info(f"⏳ New PDF detected: '{uploaded_file.name}'. Processing...")
            st.session_state.current_pdf_name = uploaded_file.name
            st.session_state.processed_docs = None 
            st.session_state.messages = [] 
            rebuild_pipeline = True
        
        if k_value_input != st.session_state.current_k_value:
            st.info(f"⚙️ Retriever 'k' value changed to {k_value_input}.")
            st.session_state.current_k_value = k_value_input
            if st.session_state.processed_docs: 
                rebuild_pipeline = True 

        if rebuild_pipeline:
            create_rag_pipeline_for_docs.clear() 
            st.session_state.rag_chain = None 

            if uploaded_file and st.session_state.current_pdf_name == uploaded_file.name and st.session_state.processed_docs is None:
                 pdf_bytes = uploaded_file.getvalue()
                 st.session_state.processed_docs = load_documents_from_bytes(pdf_bytes, uploaded_file.name)
            
        if st.session_state.processed_docs:
            if st.session_state.rag_chain is None or rebuild_pipeline: 
                st.session_state.rag_chain = create_rag_pipeline_for_docs(
                    st.session_state.processed_docs, 
                    st.session_state.current_k_value
                )
        else: 
            st.session_state.rag_chain = None
        
        if st.session_state.rag_chain:
            st.success(f"✅ Ready: '{st.session_state.current_pdf_name}' (k={st.session_state.current_k_value})")
        elif st.session_state.current_pdf_name: 
            st.warning(f"⚠️ File '{st.session_state.current_pdf_name}' selected. RAG setup may have failed.")
        
        st.markdown("---")
        if st.session_state.messages and len(st.session_state.messages) > 1 and st.session_state.rag_chain : 
            if st.button("🧹 Clear Chat History", use_container_width=True):
                st.info("Clearing chat history and resetting conversation memory...")
                create_rag_pipeline_for_docs.clear() 
                if st.session_state.processed_docs: 
                    st.session_state.rag_chain = create_rag_pipeline_for_docs(
                        st.session_state.processed_docs, 
                        st.session_state.current_k_value
                    )
                
                initial_message = {"role": "assistant", "content": f"Chat history cleared for '{st.session_state.current_pdf_name}' (k={st.session_state.current_k_value}). Let's start a new conversation!"}
                st.session_state.messages = [initial_message] if st.session_state.rag_chain else []
                st.rerun()
        
    # Main chat interface
    chat_container = st.container()

    with chat_container:
        if not st.session_state.rag_chain:
            st.info("👋 Welcome! Please upload a PDF using the sidebar to begin chatting.")
        else:
            if not st.session_state.messages:
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Hi! I'm ready for a conversation about '{st.session_state.current_pdf_name}' (using k={st.session_state.current_k_value}). How can I help?"}
                )

            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    col1, col2 = st.columns([0.93, 0.07]) 
                    with col1:
                        st.markdown(message["content"])
                    
                    if message["role"] == "assistant":
                        with col2:
                            if st.button("📄", key=f"copy_{i}", help="Copy this answer", use_container_width=True):
                                st.session_state.copy_text_content = message["content"]
                                st.session_state.show_copy_expander = True
            
            if st.session_state.show_copy_expander and st.session_state.copy_text_content:
                with st.expander("📝 Copy Answer Text", expanded=True):
                    st.text_area("Select and copy this text:", value=st.session_state.copy_text_content, height=120, key="text_area_to_copy")
                    if st.button("✅ Done Copying", key="done_copying_expander"):
                        st.session_state.show_copy_expander = False
                        st.session_state.copy_text_content = None
                        st.rerun() 

    if st.session_state.rag_chain:
        user_prompt = st.chat_input(f"Ask a question about '{st.session_state.current_pdf_name}'...")
        
        if user_prompt:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with chat_container: 
                with st.chat_message("user"):
                    st.markdown(user_prompt)
            
            with chat_container: 
                with st.chat_message("assistant"):
                    answer_text, generated_question_text = ask_chatbot_streamlit(user_prompt, st.session_state.rag_chain)
                    
                    if generated_question_text and generated_question_text.lower().strip() != user_prompt.lower().strip():
                        st.caption(f"Rephrased for retrieval: *{generated_question_text}*")
                    
                    st.markdown(answer_text)

                st.session_state.messages.append({"role": "assistant", "content": answer_text})
            st.rerun() 

        if st.session_state.messages and len(st.session_state.messages) > 1:
            chat_history_str = format_chat_history_for_download(st.session_state.messages)
            st.download_button(
                label="📥 Download Chat History",
                data=chat_history_str,
                file_name=f"chat_with_{st.session_state.current_pdf_name or 'session'}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                key="download_chat_main" # Added a key for uniqueness
            )

if __name__ == "__main__":
    run_streamlit_app()
