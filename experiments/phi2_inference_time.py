import random
import numpy as np
import torch
import json
import re
import csv
import time  # Import the time module for timing
from tqdm import tqdm
from ragatouille import RAGPretrainedModel
from llama_index.core import SimpleDirectoryReader
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
documents = SimpleDirectoryReader("data/rel18").load_data()
docs_str = [doc.text for doc in documents]

# Initialize RAG
# RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# # Index the documents using the RAG model
# print("Indexing documents...")
# RAG.index(
#     collection=docs_str,
#     index_name="ITU RAG 150",
#     max_document_length=150,
#     split_documents=True,
# )

RAG = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/ITU RAG 150",verbose=0)

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

# Timing mechanism: function to measure inference time
def measure_inference_time(function):
    """
    Measures the time taken to execute a function.
    """
    start_time = time.time()
    result = function()
    end_time = time.time()
    return result, end_time - start_time


# Loop through each question and get the response, measuring time
responses = []
for q_id, q_data in tqdm(questions.items(), desc="Processing questions"):
    q_id_number = q_id.split()[1]

    # Get the question text and remove the tag e.g., [3GPP Rel 18]
    question_text = re.sub(r"\s*\[.*?\]\s*$", "", q_data["question"])

    # Extract options, ensuring only non-null options are included
    options = [
        (k, v) for k, v in q_data.items() if k.startswith("option") and v is not None
    ]

    # Measure time for inference without retrieval
    _, time_without_retrieval = measure_inference_time(
        lambda: generate_answer(question_text, options, "", [], model, tokenizer)
    )

    # Measure time for inference with retrieval
    results = RAG.search(query=question_text, k=13)
    context = " ".join([result["content"] for result in results])
    abbreviations = find_appearing_abbreviations(q_data)

    _, time_with_retrieval = measure_inference_time(
        lambda: generate_answer(
            question_text, options, context, abbreviations, model, tokenizer
        )
    )

    # Append the times to the responses list
    responses.append([q_id_number, time_with_retrieval, time_without_retrieval])

# Save the responses to a CSV file
with open("inference_times.csv", "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Question_ID", "Time_with_Retrieval", "Time_without_Retrieval"])
    csvwriter.writerows(responses)

print("Inference times saved to 'inference_times.csv'.")
