# Import necessary libraries
import random
import numpy as np
import torch
import json
import re
from tqdm import tqdm
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from ragatouille import RAGPretrainedModel

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load documents
print("Loading documents...")
documents = SimpleDirectoryReader("data/rel18").load_data()

# Initialize BM25 retriever
splitter = SentenceSplitter(chunk_size=150, chunk_overlap=20)
nodes = splitter.get_nodes_from_documents(documents)
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=20)

# Initialize ColBERT RAG retriever
RAG = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/ITU RAG 150", verbose=0)

# Load questions
with open("data/366qs.txt", "r") as file1:
    questions = json.load(file1)

# Dictionary to store results for each k combination
results_by_k_combination = {}

# Process each question
for q_id, q_data in tqdm(questions.items(), desc="Processing questions"):
    question_text = re.sub(r"\s*\[.*?\]\s*$", "", q_data["question"])  # Clean question text
    correct_answer = q_data["answer"]

    # Extract options, ensuring only non-null options are included
    options = [
        (k, v) for k, v in q_data.items() if k.startswith("option") and v is not None
    ]

    # Retrieve results from ColBERT and BM25
    colbert_results = RAG.search(query=question_text, k=20)
    bm25_results = bm25_retriever.retrieve(question_text)
    
    # Iterate through odd values of kColbert (1, 3, 5, ..., 19) and corresponding kBM25
    for k_colbert in range(1, 21, 2):  # odd numbers 1, 3, 5, ..., 19
        k_bm25 = 20 - k_colbert  # Ensure that the sum of kColbert + kBM25 is 20
        
        top_colbert = colbert_results[:k_colbert]
        top_bm25 = bm25_results[:k_bm25]
        
        # Combine results from ColBERT and BM25
        chunks = []
        for result in top_colbert:
            chunks.append({
                "doc_text": result["content"]
            })
        
        for node in top_bm25:
            chunks.append({
                "doc_text": node.text
            })

        # Create the key for this specific k combination
        k_combination_key = f"kColbert_{k_colbert}_kBM25_{k_bm25}"
        
        # Initialize the entry for this k combination if it doesn't exist
        if k_combination_key not in results_by_k_combination:
            results_by_k_combination[k_combination_key] = {}
        
        # Store the question, correct_answer, and retrieved_chunks under this combination
        results_by_k_combination[k_combination_key][q_id] = {
            "question": question_text,
            "options": options,
            "correct_answer": correct_answer,
            "retrieved_chunks": chunks
        }

# Save the results for each k combination in separate JSON files
for k_combination_key, result_data in results_by_k_combination.items():
    output_filename = f"data/ensemble_results/results_{k_combination_key}_search.json"
    with open(output_filename, "w") as file2:
        json.dump(result_data, file2, indent=4)
    print(f"Saved results for {k_combination_key} to {output_filename}")
