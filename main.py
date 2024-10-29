import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import chromadb
import gradio as gr
import time
import numpy as np

# Initialize ChromaDB
client = chromadb.Client()
collection = client.create_collection("medical_documents")

# Initialize the model for medical text embeddings
model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')

def ingest_documents(documents):
    """
    Ingest medical documents into ChromaDB
    """
    start_time = time.time()
    
    embeddings = []
    metadatas = []
    texts = []
    ids = []
    
    for idx, doc in enumerate(documents):
        # Combine title and abstract for embedding
        combined_text = f"{doc['title']} {doc['abstract']}"
        texts.append(combined_text)
        
        # Generate embedding
        embedding = model.encode(combined_text)
        embeddings.append(embedding.tolist())
        
        # Prepare metadata
        metadatas.append({
            'title': doc['title'],
            'publication_date': doc['publication_date'],
            'journal': doc['journal'],
            'doi': doc['doi'],
            'keywords': ','.join(doc['keywords'])
        })
        
        ids.append(f"doc_{idx}")
    
    # Add to ChromaDB
    collection.add(
        embeddings=embeddings,
        metadatas=metadatas,
        documents=texts,
        ids=ids
    )
    
    ingestion_time = time.time() - start_time
    return {
        'documents_processed': len(documents),
        'ingestion_time': ingestion_time
    }

def search_documents(
    query: str,
    publication_year: str = "",
    journal: str = "",
    keywords: str = "",
    n_results: int = 5
):
    """
    Search for medical documents based on query and filters
    """
    start_time = time.time()
    
    # Generate embedding for query
    query_embedding = model.encode(query).tolist()
    
    # Prepare filter conditions
    filter_conditions = {}
    if publication_year:
        filter_conditions['publication_date'] = {"$contains": publication_year}
    if journal:
        filter_conditions['journal'] = {"$contains": journal}
    if keywords:
        filter_conditions['keywords'] = {"$contains": keywords}
    
    # Perform search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    
    # Format results
    formatted_results = "Search Results:\n\n"
    for idx in range(len(results['ids'][0])):
        formatted_results += f"Result {idx + 1}:\n"
        formatted_results += f"Title: {results['metadatas'][0][idx]['title']}\n"
        formatted_results += f"Journal: {results['metadatas'][0][idx]['journal']}\n"
        formatted_results += f"DOI: {results['metadatas'][0][idx]['doi']}\n"
        formatted_results += f"Keywords: {results['metadatas'][0][idx]['keywords']}\n"
        formatted_results += f"Abstract: {results['documents'][0][idx][:300]}...\n\n"
    
    search_time = time.time() - start_time
    stats = f"Search completed in {search_time:.2f} seconds"
    
    return formatted_results, stats

# Sample medical documents
sample_documents = [
    {
        'title': 'Recent Advances in COVID-19 Treatment',
        'abstract': 'A comprehensive review of emerging therapeutic approaches for COVID-19 management.',
        'keywords': ['COVID-19', 'treatment', 'clinical trials'],
        'publication_date': '2023',
        'doi': 'https://doi.org/10.1016/j.biopha.2021.112107',
        'journal': 'Journal of Infectious Diseases'
    },
    {
        'title': 'Diabetes Management in the Elderly',
        'abstract': 'Analysis of optimal treatment strategies for elderly patients with type 2 diabetes.',
        'keywords': ['diabetes', 'elderly care', 'treatment'],
        'publication_date': '2023',
        'doi': '10.2337/ds18-0033',
        'journal': 'Diabetes Care'
    },
    {
        'title': 'Hypertension Treatment Guidelines',
        'abstract': 'Updated guidelines for the management of hypertension in adults.',
        'keywords': ['hypertension', 'guidelines', 'treatment'],
        'publication_date': '2023',
        'doi': '10.1234/cardio.2023',
        'journal': 'Cardiology Journal'
    },
    {
        'title': 'Diabetes mellitus and its treatment',
        'abstract': 'Diabetes mellitus (DM) is a metabolic disorder resulting from a defect in insulin secretion, insulin action, or both. Insulin deficiency in turn leads to chronic hyperglycaemia with disturbances of carbohydrate, fat and protein metabolism.',
        'keywords': ['Diabetes mellitus', 'treatment', 'insulin', 'oral hypoglycaemic agents'],
        'publication_date': '2023',
        'doi': 'https://doi.org/10.1159/000497580',
        'journal': 'International Journal of Diabetes and Metabolism'
    }
]

# Ingest sample documents
print("Ingesting sample documents...")
ingest_result = ingest_documents(sample_documents)
print(f"Processed {ingest_result['documents_processed']} documents in {ingest_result['ingestion_time']:.2f} seconds")

# Create Gradio interface
interface = gr.Interface(
    fn=search_documents,
    inputs=[
        gr.Textbox(label="Search Query", placeholder="Enter patient symptoms or medical condition..."),
        gr.Textbox(label="Publication Year (optional)", placeholder="e.g., 2023"),
        gr.Textbox(label="Journal (optional)", placeholder="e.g., Journal of Medicine"),
        gr.Textbox(label="Keywords (optional)", placeholder="e.g., clinical trial"),
        gr.Slider(minimum=1, maximum=20, value=5, label="Number of Results")
    ],
    outputs=[
        gr.Textbox(label="Results", lines=20),
        gr.Textbox(label="Statistics")
    ],
    title="Medical Literature Search",
    description="Search for relevant medical papers based on patient symptoms or diagnoses.",
    examples=[
        ["COVID-19 treatment options", "2023", "", "clinical trial"],
        ["diabetes management elderly patients", "", "Diabetes Care", ""],
        ["hypertension guidelines", "2023", "", "treatment"]
    ]
)

if __name__ == "__main__":
    interface.launch(share=True)