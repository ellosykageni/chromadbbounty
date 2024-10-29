## Medical Literature Search Application Documentation

### Project Overview
This project is a medical literature search application designed to assist healthcare professionals, researchers, and academics in quickly retrieving relevant medical papers based on specific queries. The application leverages embeddings to represent the semantic content of medical documents and employs a vector database (ChromaDB) to efficiently search and retrieve relevant documents. A Gradio-based web interface allows users to interactively input search queries and view results, with options to filter by publication year, journal, and keywords.

### Application Flow
### Document Ingestion:

The application ingests medical documents into ChromaDB by embedding each document (using the SentenceTransformer model fine-tuned for medical literature) and storing the document text, metadata (title, journal, publication date, DOI, keywords), and embedding.
This enables efficient similarity-based retrieval when a query is made.

### Query Processing and Search:
When a user inputs a search query (e.g., "COVID-19 treatment options") in the Gradio interface, the application generates an embedding for the query.
The system then compares this query embedding to the stored document embeddings in ChromaDB, retrieving the most similar documents.
Additional filters (publication year, journal, keywords) narrow down the results to match specific user needs.

### Result Presentation:
The search results are displayed on the Gradio interface, showing each document's title, journal, DOI, keywords, and a truncated abstract.
The interface also provides search statistics, such as the time taken to complete the search.

### Installation Guide
Follow these steps to set up and run the application locally.

### Prerequisites
Python 3.8 or later: Ensure Python is installed. You can download it from Python's official website.
pip: Python's package installer should be installed (it comes pre-installed with Python).
### 1. Clone or Download the Project
Open your terminal or command prompt.
Clone the project repository using:
  `git clone https://github.com/ellosykageni/chromadbbounty.git`
Navigate to the project directory:
`cd chromadbbounty`

### 2. Create a Virtual Environment (Optional but Recommended)
Creating a virtual environment helps to manage dependencies independently of other projects.
  `python -m venv venv`

Activate the virtual environment:
Windows: `venv\Scripts\activate`
Mac/Linux: `source venv/bin/activate`

### 3. Install Required Dependencies
Install the dependencies listed in requirements.txt. Run:
  `pip install -r requirements.txt`
  
The primary dependencies are:
transformers: To load the pre-trained embedding model.
sentence-transformers: Specifically for handling and encoding sentences using SentenceTransformer.
chromadb: A vector database to store and retrieve embeddings.
gradio: For creating the web interface.

### 4. Model Setup (Medical Embedding Model)
The application uses SentenceTransformer fine-tuned on medical literature (pritamdeka/S-PubMedBert-MS-MARCO). This model is automatically downloaded by sentence-transformers during runtime. Ensure you have an internet connection for the first run.

### 5. Running the Application
Once everything is set up, you can start the application by running:
  `python app.py`
This will launch a local server, and a link will appear in the terminal, which you can open in your browser to access the application interface.

Gradio Shareable Link: By default, Gradio provides a shareable link (share=True in the code). This is useful if you want others to access your local server (e.g., for testing).
Application Usage
Ingesting Documents:

The ingest_documents function prepares and adds document embeddings to ChromaDB. Sample documents are included in the code for testing purposes.
You can add more documents by expanding the sample_documents list with your data. Each document should contain:
title
abstract
keywords (as a list of strings)
publication_date
doi
journal

### Running a Search:

In the Gradio interface, enter a query in the "Search Query" field.
Optionally, specify filters:
Publication Year: e.g., 2023
Journal: e.g., Diabetes Care
Keywords: e.g., treatment
Set the desired number of results using the slider.
Click "Submit" to retrieve the most relevant documents. The search time will be displayed alongside the formatted results.

This application provides a robust solution for fast, similarity-based document retrieval in the medical field, which can be extended to larger datasets and more complex filtering as needed.
