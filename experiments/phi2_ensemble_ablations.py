import random
import numpy as np
import torch
import json
import re
import csv
import argparse
from tqdm import tqdm
from ragatouille import RAGPretrainedModel
from llama_index.core import SimpleDirectoryReader
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.prepare_docs import find_appearing_abbreviations
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter
import Stemmer
import itertools
import time

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load the configuration for the fine-tuned model
config = PeftConfig.from_pretrained(
    "alexgichamba/phi-2-finetuned-qa-lora-r32-a16_longcontext"
)

# Load the base model (phi-2)
base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2").to("cuda")

# Apply the LoRA adapter
model = PeftModel.from_pretrained(
    base_model, "alexgichamba/phi-2-finetuned-qa-lora-r32-a16_longcontext"
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

# Load documents
print("Loading documents...")
print("Takes about 5 minutes...")
documents = SimpleDirectoryReader("data/rel18").load_data()
docs_str = [doc.text for doc in documents]

# Initialize RAG
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Index the documents using the RAG model
print("Indexing documents...")
print("Takes about 20 minutes...")
RAG.index(
    collection=docs_str,
    index_name="ITU RAG 150",
    max_document_length=150,
    split_documents=True,
)

# Initialize node parser
print("Indexing documents BM25...")
splitter = SentenceSplitter(chunk_size=150, chunk_overlap=30)
nodes = splitter.get_nodes_from_documents(documents)

# Initializing BM25Retriever with a high k value
max_bm25_k = 15  
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=max_bm25_k)

# Read questions from the text files
with open("data/TeleQnA_testing1.txt", "r") as file1:
    questions = json.load(file1)

def create_prompt(question, options, context, abbreviations):
    options_text = "\n".join(
        [f"Option {i+1}: {opt[1]}" for i, opt in enumerate(options)]
    )
    abbreviations_text = "\n".join(
        [
            f"{list(abbrev.keys())[0]}: {list(abbrev.values())[0]}"
            for abbrev in abbreviations
        ]
    )

    prompt = (
        f"Instruct: You will answer each question correctly using the context below\n"
        f"Context: {context}\n\n"
        f"Abbreviations:\n{abbreviations_text}\n\n"
        f"Question: {question}\n"
        f"{options_text}\n"
        f"Answer: Option"
    )
    return prompt

def generate_answer(question, options, context, abbreviations, model, tokenizer):
    prompt = create_prompt(question, options, context, abbreviations)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to("cuda")

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=10,
        pad_token_id=tokenizer.eos_token_id,
        num_beams=1,
        early_stopping=True,
    )
    answer = tokenizer.decode(
        outputs[0][input_ids.shape[1]:], skip_special_tokens=True
    )
    return answer

def parse_answer(response):
    match = re.search(r"Answer:\s*Option\s*(\d+)", response, re.IGNORECASE)
    if match:
        answer = f"Option {match.group(1)}"
    else:
        match = re.search(r"(\d+)", response, re.IGNORECASE)
        if match:
            answer = f"Option {match.group(1)}"
        else:
            answer = "Error"
    return answer

# Define ranges for k values
colbert_k_values = range(2, 12) 
bm25_k_values = range(1, 12)  

# Create all combinations of k values
k_combinations = list(itertools.product(colbert_k_values, bm25_k_values))

# Main processing loop
total_time = 0
for colbert_k, bm25_k in tqdm(k_combinations, desc="Processing k-value combinations"):
    start_time = time.time()
    responses = []
    
    for q_id, q_data in tqdm(questions.items(), desc=f"Processing questions (ColBERT k={colbert_k}, BM25 k={bm25_k})"):
        q_id_number = q_id.split()[1]
        question_text = re.sub(r"\s*\[.*?\]\s*$", "", q_data["question"])
        options = [(k, v) for k, v in q_data.items() if k.startswith("option") and v is not None]

        # Retrieve using ColBERT
        colbert_results = RAG.search(query=question_text, k=colbert_k)
        colbert_context = " ".join([result['content'] for result in colbert_results])

        # Retrieve using BM25
        bm25_results = bm25_retriever.retrieve(question_text)
        bm25_context = " ".join([node.text for node in bm25_results[:bm25_k]])  # Use only the first bm25_k results

        # Combine contexts
        combined_context = f"{colbert_context} {bm25_context}"

        abbreviations = find_appearing_abbreviations(q_data)

        response = generate_answer(question_text, options, combined_context, abbreviations, model, tokenizer)
        answer = parse_answer(response)

        match = re.search(r"Option (\d+)", answer)
        if match:
            try:
                answer_id = int(match.group(1))
                responses.append([q_id_number, answer_id, "Phi-2"])
            except (KeyError, IndexError, ValueError) as e:
                responses.append([q_id_number, "Error", "Phi-2"])
                print(f"Error processing question {q_id}: {answer}")
        else:
            responses.append([q_id_number, "Error", "Phi-2"])
            print(f"Error processing question {q_id_number}: {answer}")

    # Save the responses to a CSV file
    file_name = f"output_results_colbert{colbert_k}_bm25{bm25_k}.csv"
    with open(file_name, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Question_ID", "Answer_ID", "Task"])
        csvwriter.writerows(responses)

    end_time = time.time()
    combination_time = end_time - start_time
    total_time += combination_time

    print(f"Responses saved to '{file_name}'")
    print(f"Time taken for this combination: {combination_time:.2f} seconds")
    print(f"Total time so far: {total_time:.2f} seconds")

print("All combinations processed. Check the output files for results.")
print(f"Total time taken: {total_time:.2f} seconds")