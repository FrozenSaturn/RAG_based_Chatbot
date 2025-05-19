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

# --- User Configuration for Data Loading ---
# <strong>IMPORTANT: Modify these lines for your specific dataset</strong>
# Example:
# DATA_FILE_PATH = "data/your_document_name.pdf" # Replace with your data file path
# DATA_FILE_TYPE = "pdf" # "txt", "pdf", or "csv"

# For the script to run, you MUST define these:
# documents = load_documents(DATA_FILE_PATH, DATA_FILE_TYPE)
# For now, I will initialize 'documents' as an empty list.
# You need to uncomment and set the DATA_FILE_PATH and DATA_FILE_TYPE above,
# and then uncomment the line below to load your actual data.
documents = []
# Example (uncomment and edit after placing your file in data/):
# DATA_FILE_PATH = "data/my_knowledge_base.txt"
# DATA_FILE_TYPE = "txt"
# documents = load_documents(DATA_FILE_PATH, DATA_FILE_TYPE)

# if not documents:
# print("No documents loaded. Please check DATA_FILE_PATH and DATA_FILE_TYPE.")
# else:
# print(f"Successfully loaded {len(documents)} documents.")