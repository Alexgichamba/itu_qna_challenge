"""
Script Overview:
----------------
This script processes a set of multiple-choice questions by leveraging a pre-trained language
model to generate answers based on provided context. 
It integrates a retrieval-augmented generation (RAG) model for context retrieval and a
fine-tuned language model for generating answers.

Key Components:
---------------
1. **Document Indexing:** Uses the RAG model to index documents and facilitate efficient retrieval.
2. **Question Processing:** Reads questions from text files, formats them, and generates answers using the pre-trained language model.
3. **Abbreviation Handling:** Incorporates abbreviation expansions into the prompt to enrich the context.
4. **Answer Parsing:** Extracts and formats the model's response into a usable answer format.
5. **Results Storage:** Saves the generated answers to a CSV file for further analysis.

Environment:
-------------
- **Development and Testing:** The script was developed and tested in a Python 3.10 environment.

System Requirements and Environment:
-------------------------------------
**Instance Details:**
- **Instance Type:** g6.2xlarge (AWS)
- **Operating System:** Ubuntu with Deep Learning Image

**GPU Information:**
- **GPU Model:** NVIDIA L4
- **NVIDIA-SMI Version:** 535.183.01
- **CUDA Version:** 12.2
- **Total GPU Memory:** 23034 MiB
-------------------------------------
OR
-------------------------------------
**Instance Details:**
- **Instance Type:** g5.2xlarge (AWS)
- **Operating System:** Ubuntu with Deep Learning Image

**GPU Information:**
- **GPU Model:** NVIDIA A10G
- **NVIDIA-SMI Version:** 535.183.01
- **CUDA Version:** 12.2
- **Total GPU Memory:** 23034 MiB
-------------------------------------

Usage:
------
1. Run the script to process questions and generate answers.
2. Check the `output_results.csv` for the results.

Runtime:
This script takes roughly 2 hours to complete running
"""

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
docs_str = []
for doc in documents:
    docs_str.append(doc.text)

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


# Read questions from the 2 text files and merge them
with open("data/TeleQnA_testing1.txt", "r") as file1:
    with open("data/questions_new.txt", "r") as file2:
        questions = json.load(file1)
        # questions.update(json.load(file2))


def create_prompt(question, options, context, abbreviations):
    """
    Create a formatted prompt string for the language model based on the provided question, options, context, and abbreviations.

    Parameters:
    - question (str): The question to be answered by the model.
    - options (list of tuples): A list of possible answer options, where each tuple contains:
        - The key (str) of the option (e.g., 'option1')
        - The value (str) of the option (e.g., 'Option text')
    - context (str): Contextual information that provides background relevant to the question.
    - abbreviations (list of dicts): A list of dictionaries where each dictionary contains:
        - A single key-value pair representing an abbreviation and its full form extracted from the rel18 documents (e.g.,[{'DL': 'Down Link'}, {'gNB': 'next Generation Node B'}])
    Returns:
    - str: The formatted prompt string for the language model.


    """
    # Format the list of options into a text block for the prompt
    # Each option is represented as 'Option <number>: <option_text>'
    options_text = "\n".join(
        [f"Option {i+1}: {opt[1]}" for i, opt in enumerate(options)]
    )
    # Format the list of abbreviations into a text block for the prompt
    # Each abbreviation is represented as '<abbreviation>: <full_form>'
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
    """
    Generate an answer for the given question using the specified model and tokenizer.

    Parameters:
    - question (str): The question to be answered.
    - options (list): List of possible answer options.
    - context (str): Contextual information to aid in answering the question.
    - abbreviations (list): List of abbreviations and their full forms.
    - model: The pre-trained language model.
    - tokenizer: The tokenizer associated with the model.

    Returns:
    - str: The generated answer.
    """
    # Create the prompt string for the model
    prompt = create_prompt(question, options, context, abbreviations)

    # Tokenize the prompt and move it to GPU for faster processing
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    # Ensure that the pad_token_id is set for handling padding
    # If the tokenizer does not have a pad_token_id, use eos_token_id as a fallback
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    attention_mask = (
        input_ids.ne(tokenizer.pad_token_id).long().to("cuda")
    )  # Set attention mask

    # Generate the answer using the model
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=10,  # Limit the number of new tokens generated
        pad_token_id=tokenizer.eos_token_id,
        num_beams=1,
        early_stopping=True,
    )
    # Decode the generated output tokens to get the final answer
    answer = tokenizer.decode(
        outputs[0][input_ids.shape[1] :], skip_special_tokens=True
    )
    return answer


# Function to parse the answer from the generated response
def parse_answer(response):
    """
    Parse the generated response to extract the answer option.

    Parameters:
    - response (str): The generated response from the model.

    Returns:
    - str: The extracted answer option or 'Error' if no valid option is found.
    """
    # First, search for the full pattern 'Answer: Option <number>'
    match = re.search(r"Answer:\s*Option\s*(\d+)", response, re.IGNORECASE)
    if match:
        # If a match is found, format the answer as 'Option <number>'
        answer = f"Option {match.group(1)}"
    else:
        # If the full pattern is not found, try to find any standalone number
        match = re.search(r"(\d+)", response, re.IGNORECASE)
        if match:
            # If a number is found, format it as 'Option <number>'
            answer = f"Option {match.group(1)}"
        else:
            # If no valid option is found, return 'Error'
            answer = "Error"
    return answer



# make a list of values of k from 1 to 13
k_values = list(range(14, 17))
for k in k_values:
    responses = []
    # Loop through each question and get the response
    for q_id, q_data in tqdm(questions.items(), desc="Processing questions"):
        # Extract the question ID number
        q_id_number = q_id.split()[1]

        # Get the question text and remove the tag eg [3GPP Rel 18]
        question_text = q_data["question"]
        question_text = re.sub(r"\s*\[.*?\]\s*$", "", question_text)

        # Extract options, ensuring only non-null options are included
        options = [
            (k, v) for k, v in q_data.items() if k.startswith("option") and v is not None
        ]

        #  Retrieve the top 13 relevant search results for the question using ColBERT
        results = RAG.search(query=question_text, k=k)
        context = " ".join([result["content"] for result in results])

        # Find abbreviations appearing in the question data
        abbreviations = find_appearing_abbreviations(q_data)

        # Generate the response using the loaded model
        response = generate_answer(
            question_text, options, context, abbreviations, model, tokenizer
        )

        # Parse the generated response to extract the answer option
        answer = parse_answer(response)

        # Extract the answer ID from the response
        match = re.search(r"Option (\d+)", answer)
        if match:
            try:
                # Convert the answer ID to an integer and append it to the responses list
                answer_id = int(match.group(1))
                print(f"Answer ID: {answer_id}")
                responses.append([q_id_number, answer_id, "Phi-2"])
            except (KeyError, IndexError, ValueError) as e:
                # Handle any errors that occur during the conversion and append 'Error' to the responses list
                responses.append([q_id_number, "Error", "Phi-2"])
                print(f"Error processing question {q_id}: {answer}")
        else:
            # If no valid answer ID is found, append 'Error' to the responses list
            responses.append([q_id_number, "Error", "Phi-2"])
            print(f"Error processing question {q_id_number}: {answer}")    

    # Save the responses to a CSV file
    file_name = f"output_results_{k}.csv"
    with open(file_name, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Question_ID", "Answer_ID", "Task"])
        # Write the list of responses to the CSV file
        # Each response is a list where:
        # - The first element is the question ID
        # - The second element is the answer ID (or "Error" if there was an issue)
        # - The third element is a constant string "Phi-2" indicating the model used
        csvwriter.writerows(responses)

print("Processing complete. Responses saved to 'output_results.csv'.")
