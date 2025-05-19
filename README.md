# Interactive PDF RAG Chatbot

## Objective

This project implements an interactive Retrieval-Augmented Generation (RAG) chatbot. Users can upload PDF documents and engage in contextual conversations, receiving answers based solely on the document's content. The application is built with Python, LangChain, Google Gemini, and Streamlit.

## Key Features

* **Dynamic PDF Upload**: Users can upload their own PDF documents to serve as the knowledge base for the chatbot.
* **Conversational Q&A with Memory**: Engages in multi-turn conversations, remembering previous interactions to understand follow-up questions.
* **Configurable Retrieval (`k` value)**: Allows users to adjust the number of relevant text chunks retrieved from the PDF to generate answers.
* **Maximal Marginal Relevance (MMR) Retrieval**: Employs MMR to fetch diverse yet relevant context, enhancing answer quality.
* **Rephrased Question Display**: Shows the rephrased (standalone) question generated from conversational context, offering transparency into the retrieval process.
* **View Retrieved Context**: Users can inspect the source text snippets from the PDF that were used to formulate the answer.
* **Copy Answer**: Easily copy the chatbot's answers.
* **Download Chat History**: Users can download the current conversation transcript.
* **Streamlit Web Interface**: Provides an intuitive and interactive user interface.

## Tech Stack

* Python
* Streamlit
* LangChain
* Google Gemini API (via `langchain-google-genai`)
* FAISS (for vector storage)
* PyPDF (for PDF processing)

## Setup and Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/FrozenSaturn/RAG_based_Chatbot.git
    cd RAG_based_Chatbot.git
    ```

2.  **Create and Activate Virtual Environment**:
    ```bash
    python -m venv .venv
    # On Windows: .venv\Scripts\activate
    # On macOS/Linux: source .venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**:
    * Create a `.env` file in the project root.
    * Add your Google API key:
        ```env
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
        ```
    * An `.env.example` file is included as a template.

## Running the Application

1.  Ensure your virtual environment is activated.
2.  Execute the Streamlit application:
    ```bash
    streamlit run app.py
    ```
3.  Access the app via the local URL provided (typically `http://localhost:8501`).

## Project Structure

* `app.py`: Main Streamlit application script.
* `requirements.txt`: Python dependencies.
* `.env.example`: Template for environment variables.
* `.gitignore`: Specifies intentionally untracked files.
* `sample_qa_responses.txt`: (To be created by user) A file showcasing sample interactions with the chatbot, as per assignment guidelines.

## Code Documentation

The Python code (`app.py`) includes docstrings for functions and comments for clarity, adhering to standard coding practices.