# RAG-Based PDF Chatbot

## Objective

This project implements a Retrieval-Augmented Generation (RAG) based chatbot. [cite: 299, 300] The chatbot allows users to upload PDF documents and ask questions about their content, providing accurate and contextually relevant answers based on the uploaded document. [cite: 300] The application is built using Python, LangChain, Google Gemini models, and Streamlit. [cite: 299]

## Features

* **PDF Upload**: Users can upload their own PDF documents to serve as the knowledge base.
* **Conversational Q&A**: Ask questions about the content of the uploaded PDF and receive answers.
* **Conversational Memory**: The chatbot remembers previous turns in the conversation for follow-up questions.
* **Configurable Retrieval**: Users can adjust the number of context chunks (`k`) retrieved to answer questions.
* **View Context**: Option to view the source text snippets retrieved from the PDF that were used to generate the answer.

## Tech Stack

* Python
* LangChain
* Google Gemini (via `langchain-google-genai`)
* FAISS (for vector storage)
* Streamlit (for the web interface)
* PyPDF (for PDF processing)

## Setup and Installation

1.  **Clone the Repository**:
    ```bash
    git clone <your_repository_link>
    cd <repository_name>
    ```

2.  **Create and Activate Virtual Environment**:
    ```bash
    python -m venv .venv
    # On Windows
    # .venv\Scripts\activate
    # On macOS/Linux
    # source .venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to generate `requirements.txt` using `pip freeze > requirements.txt` from your activated virtual environment after installing all packages.)*

4.  **Set Up Environment Variables**:
    * Create a `.env` file in the project root directory.
    * Add your Google API key to the `.env` file:
        ```
        GOOGLE_API_KEY="your_google_api_key_here"
        ```
    * An `.env.example` file can be provided as a template.

## Running the Application

1.  Ensure your virtual environment is activated.
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
3.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Project Structure

* `app.py`: The main Streamlit application code.
* `requirements.txt`: Lists all Python dependencies.
* `.env`: (Not committed) Stores API keys. Use `.env.example` as a template.
* `sample_qa_responses.txt`: Contains a sample set of questions and chatbot responses, as per the assignment requirements. [cite: 309, 313]
* (Optional) `data/`: If you choose to include a default PDF. (Currently, the app relies on user uploads).

## Code Documentation

All classes and functions within the Python code are documented with docstrings and comments where necessary, adhering to Python coding standards. [cite: 308]