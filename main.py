# main.py

# ---------------------------------------------------------------------------
# SECTION 0: IMPORTS AND ENVIRONMENT SETUP
# ---------------------------------------------------------------------------

import os
from dotenv import load_dotenv

# LangChain specific imports
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Other useful libraries (if needed for specific data loaders)
import pandas as pd

def setup_environment():
    """Loads environment variables from .env file."""
    load_dotenv()
    # Optionally, check if the key is loaded, but avoid printing it
    # if not os.getenv("GOOGLE_API_KEY"):
    #     print("Warning: GOOGLE_API_KEY not found in environment.")

# Call environment setup at the beginning
setup_environment()

# ---------------------------------------------------------------------------
# SECTION 1: TASK 1 - DATA LOADING
# ---------------------------------------------------------------------------

def load_documents(file_path, file_type):
    """
    Loads documents from the specified file path and type.
    Supported file_types: "txt", "pdf", "csv".
    """
    documents = []
    try:
        if file_type == "txt":
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
        elif file_type == "pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load_and_split() # PyPDFLoader can split pages
        elif file_type == "csv":
            # For CSVs, you might want more control with pandas,
            # but CSVLoader is a simpler start.
            # This example uses CSVLoader; adjust if pandas method is preferred.
            loader = CSVLoader(file_path=file_path, encoding="utf-8")
            documents = loader.load()
            # If using pandas:
            # df = pd.read_csv(file_path, encoding="utf-8")
            # documents = []
            # for index, row in df.iterrows():
            #     content = ". ".join(str(x) for x in row if pd.notna(x))
            #     metadata = {"row": index, "source": os.path.basename(file_path)}
            #     documents.append(Document(page_content=content, metadata=metadata))
        else:
            print(f"Unsupported file type: {file_type}")
            return []

        print(f"Loaded {len(documents)} document(s) from {file_path}.")
    except Exception as e:
        print(f"Error loading {file_type} file '{file_path}': {e}")
        return []
    return documents
DATA_FILE_PATH = "data/dataset_childrenbook.pdf" # Using the provided PDF
DATA_FILE_TYPE = "pdf"

documents = load_documents(DATA_FILE_PATH, DATA_FILE_TYPE)

if not documents:
    print("No documents loaded. Please ensure 'dataset_childrenbook.pdf' is in the 'data' directory.")
    # Exit if no documents, or handle appropriately
    # exit() # Consider exiting if documents are critical for the rest of the script
else:
    print(f"Successfully loaded {len(documents)} pages/documents from the PDF.")
    # You can inspect a document's content if needed:
    # print(f"Sample content from first page: {documents[0].page_content[:500]}")

# ---------------------------------------------------------------------------
# SECTION 2: TASK 2 - SET UP RAG WITH LANGCHAIN (Using Gemini)
# ---------------------------------------------------------------------------

def create_rag_pipeline(docs):
    """
    Sets up the RAG pipeline using the loaded documents.
    Returns the retrieval_chain.
    """
    if not docs:
        print("No documents provided to create RAG pipeline.")
        return None

    # 1. Split Documents into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    if not texts:
        print("Text splitting resulted in no chunks. Check document content and splitter settings.")
        return None
    print(f"Split documents into {len(texts)} chunks.")

    # 2. Create Embeddings using Google Gemini
    # Ensure GOOGLE_API_KEY is set in your .env file
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("Initialized GoogleGenerativeAIEmbeddings.")

    # 3. Store in a Vector Store (FAISS)
    try:
        vectorstore = FAISS.from_documents(texts, gemini_embeddings)
        print("Created FAISS vector store from document chunks.")
    except Exception as e:
        print(f"Error creating FAISS vector store: {e}")
        return None

    # 4. Create a Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks
    print("Created retriever from vector store.")

    # 5. Set up the LLM and Prompt for the RAG chain
    # Using Gemini Pro model via ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.3, convert_system_message_to_human=True)
    print("Initialized ChatGoogleGenerativeAI with gemini-pro.")

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
    # This chain will take an input question, retrieve relevant documents,
    # stuff them into the prompt, and pass to the LLM.
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    print("Created RAG retrieval chain.")

    return retrieval_chain

# Create the RAG pipeline if documents are loaded
rag_chain = None
if documents:
    rag_chain = create_rag_pipeline(documents)
else:
    print("Skipping RAG pipeline creation as no documents were loaded.")

# ---------------------------------------------------------------------------
# SECTION 3: TASK 3 - BUILD THE CHATBOT
# ---------------------------------------------------------------------------

def ask_chatbot(query, chain):
    """
    Asks a question to the RAG chatbot and returns the answer.
    """
    if not chain:
        return "RAG chain is not initialized. Please ensure data is loaded and pipeline is set up."
    
    print(f"\nProcessing query: {query}")
    try:
        response = chain.invoke({"input": query})
        # The response object structure might vary slightly based on LangChain versions
        # but 'answer' is typical for chains made with create_retrieval_chain.
        answer = response.get("answer", "Could not extract answer from response.")
        
        # You can also inspect the retrieved documents if needed for debugging
        # if 'context' in response and response['context']:
        #     print("\n--- Retrieved Context Snippets ---")
        #     for i, doc_context in enumerate(response['context'][:2]): # Show first 2 retrieved docs
        #         print(f"Snippet {i+1}: {doc_context.page_content[:200]}...")
        #     print("--- End of Snippets ---")

        return answer
    except Exception as e:
        print(f"Error during chatbot query: {e}")
        return "An error occurred while processing your question."

# ---------------------------------------------------------------------------
# MAIN EXECUTION / EXAMPLE USAGE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # First, ensure environment is set up and API key is available
    setup_environment()
    if not os.getenv("GOOGLE_API_KEY"):
        print("🔴 FATAL: GOOGLE_API_KEY not found. Please set it in your .env file.")
    else:
        print("🟢 GOOGLE_API_KEY found.")

        # Load data (this part is now above, but ensure it runs)
        if not documents:
            print("🔴 No documents were loaded. Chatbot cannot proceed.")
            print("   Please check the DATA_FILE_PATH and ensure the file exists in the 'data' folder.")
        else:
            # Create RAG chain (this part is now above, but ensure it runs)
            if not rag_chain:
                print("🔴 RAG chain initialization failed. Chatbot cannot proceed.")
            else:
                print("\n✅ Chatbot initialized successfully. You can now ask questions.")
                print("-----------------------------------------------------------")

                # Example Questions (Modify these based on your PDF's content)
                sample_questions = [
                    "What is the Goldilocks Principle mentioned in the paper?", # [cite: 1]
                    "What is the Children's Book Test (CBT)?", # [cite: 10]
                    "According to the paper, how do Memory Networks compare to RNNs for predicting nouns?", # [cite: 15, 18]
                    "What dataset was used to build the CBT?" # [cite: 29]
                ]

                # Store questions and answers for the deliverable file
                qa_pairs = []

                for question in sample_questions:
                    answer = ask_chatbot(question, rag_chain)
                    print(f"\nUser: {question}")
                    print(f"Chatbot: {answer}")
                    qa_pairs.append({"question": question, "answer": answer})
                
                print("-----------------------------------------------------------")
                print("\nSample Q&A session finished.")

                # --- Saving Sample Questions and Responses (Deliverable) ---
                # You'll need to save 'qa_pairs' to a .txt, .pdf, or .xlsx file.
                # Here's a simple way to save to a .txt file:
                try:
                    with open("sample_qa_responses.txt", "w", encoding="utf-8") as f:
                        f.write("Sample Questions and Chatbot Responses\n")
                        f.write("=======================================\n\n")
                        for pair in qa_pairs:
                            f.write(f"Question: {pair['question']}\n")
                            f.write(f"Answer: {pair['answer']}\n\n")
                        f.write("---------------------------------------\n")
                        f.write("Context Document: dataset_childrenbook.pdf\n")
                    print("\n✅ Sample Q&A saved to sample_qa_responses.txt")
                except Exception as e:
                    print(f"Error saving sample Q&A: {e}")
