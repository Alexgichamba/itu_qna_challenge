# bm25.py

import random
import numpy as np
import torch
import json
import re
import csv
from tqdm import tqdm
from ragatouille import RAGPretrainedModel
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
import sys

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load documents
print("Loading documents...")
documents = SimpleDirectoryReader("data/rel18").load_data()
docs_str = [doc.text for doc in documents]

# Initialize node parser
splitter = SentenceSplitter(chunk_size=150, chunk_overlap=20)
nodes = splitter.get_nodes_from_documents(documents)

print("Initializing BM25Retriever...")
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=20
)

# Read questions
with open("data/366qs.txt", "r") as file1:
    questions = json.load(file1)

responses = {}

# Find the top k documents for each question
for q_id, q_data in tqdm(questions.items(), desc="Processing questions"):
    question_text = re.sub(r"\s*\[.*?\]\s*$", "", q_data["question"])  # Clean question text
    answers = q_data["answer"]
    retrieved_nodes = bm25_retriever.retrieve(question_text)

    # Extract options, ensuring only non-null options are included
    options = [
        (k, v) for k, v in q_data.items() if k.startswith("option") and v is not None
    ]
    
    # Prepare results for each question
    chunks = []
    for node in retrieved_nodes:
        chunk = {
            "doc_id": node.node_id,
            "doc_text": node.text,
            "score": node.score
        }
        chunks.append(chunk)
    
    # Store the question, answers, and results in the responses dict
    responses[q_id] = {
        "question": question_text,
        "options": options,
        "correct_answer": answers,
        "retrieved_chunks": chunks
    }

# Save the responses
with open("data/results_bm25_search.json", "w") as file2:
    json.dump(responses, file2, indent=4)