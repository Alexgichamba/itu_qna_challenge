"""
Script Overview:
----------------
This script processes a set of multiple-choice questions related to telecommunications by 
leveraging a pre-trained language model to generate answers based on provided context. 
It integrates a retrieval-augmented generation (RAG) model for context retrieval and
a fine-tuned language model for generating answers.

Key Components:
---------------
1. **Document Indexing:** Uses the ColBERT RAG model to index documents from the
    "data/rel18" directory for efficient retrieval.
2. **Question Processing:**  Reads questions from JSON files, formats them, and generates
    answers using the Falcon 7B Instruct model.
3. **Abbreviation Handling:** Incorporates telecommunication-specific abbreviation expansions
    into the prompt to enrich the context.
4. **Answer Generation:** Uses the Falcon 7B model to generate responses based on
    the question, context, and abbreviations.
5. **Answer Scoring:** Employs word overlap and cosine similarity techniques to
    score generated answers against multiple-choice options.
6. **Results Storage:** Saves the generated answers and scores to JSON and CSV files for further analysis.

Environment:
-------------
- **Development and Testing:** The script is designed to run in a Python environment 3.10 with GPU support

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

Usage:
------
1. Ensure all required libraries are installed.
2. Place the document files in the data/rel18 directory.
3. Prepare the question files data/TeleQnA_testing1.txt and data/questions_new.txt.
4. Run the script to process questions, generate answers, and score responses.
5. Check responses.json for detailed responses and output_results.csv for final results.


"""

# Import necessary libraries
import json
import random
import re
import csv
import numpy as np
import torch
from llama_index.core import SimpleDirectoryReader
from ragatouille import RAGPretrainedModel
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from sentence_transformers.util import cos_sim

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# Define Helper Functions
def create_prompt(question, context, abbrevs):
    """
    Constructs a prompt string for a language model based on the given question, context, and abbreviations.

    Args:
        question (str): The question to be included in the prompt.
        context (str): The context or domain information relevant to the question.
        abbrevs (list of dict): A list of dictionaries containing abbreviations and their full forms.

    Returns:
        str: A formatted prompt string ready for input to a language model.
    """
    # Strip any leading or trailing whitespace from the context
    context = context.strip()
    # Generate a formatted string of abbreviations
    abbreviations_text = "\n".join(
        [f"{list(abbrev.keys())[0]}: {list(abbrev.values())[0]}" for abbrev in abbrevs]
    )
    # Construct the prompt string with the given context, abbreviations, and question
    prompt = (
        # special tag required by Falcon Language Model
        f">>DOMAIN<<\n{context}\n"
        # Add the abbreviations under the Abbreviations section
        f"Abbreviations:\n{abbreviations_text}\n"
        # Add the question under the QUESTION section
        f">>QUESTION<<{question}\n\n"
        f">>ANSWER<<"  # Placeholder for the answer
    )
    # Return the constructed prompt
    return prompt


def find_appearing_abbreviations(question):
    """
    Identifies abbreviations that appear in a given question and its options.

    This function reads a list of abbreviations from a file, checks which of these
    abbreviations appear in the provided question and its options, and returns a list
    of dictionaries containing the abbreviations and their full forms.

    Args:
        question (dict): A dictionary containing the question and its options. The question
                         text should be under the key 'question', and options should be under
                         keys starting with 'option'.

    Returns:
        list of dict: A list of dictionaries where each dictionary has an abbreviation as the key
                      and its full form as the value.
    """
    # Read abbreviations from the file
    with open("data/abbreviations.txt", "r") as f_abbrevs:
        abbreviations = {}
        for line in f_abbrevs:
            # Split each line into abbreviation and full form
            abbreviation, full_form = line.split(": ", 1)
            # Store in the dictionary
            abbreviations[abbreviation.strip()] = full_form.strip()
    # Sort abbreviations by length in descending order to handle cases where one abbreviation is a substring of another
    sorted_abbrevs = sorted(
        abbreviations.items(), key=lambda x: len(x[0]), reverse=True
    )
    # Ensure the question is a dictionary
    assert isinstance(question, dict)
    # Use a set to store unique abbreviations that appear in the question or options
    appearing_abbreviations = set()
    # Check for each abbreviation in the question and its options
    for abbreviation, full_form in sorted_abbrevs:
        # Create a regex pattern to find the abbreviation as a whole word
        pattern = r"\b" + re.escape(abbreviation) + r"\b"
        # Check if the abbreviation appears in the question text
        if re.search(pattern, (question["question"].split("?")[0])):
            appearing_abbreviations.add(abbreviation)
        # Check if the abbreviation appears in any of the options
        for key in question:
            if key.startswith("option") and question[key] is not None:
                if re.search(pattern, question[key]):
                    appearing_abbreviations.add(abbreviation)
    # Return a list of dictionaries with the abbreviation and its full form
    returned_abbreviations = [
        {abbrev: abbreviations[abbrev]} for abbrev in appearing_abbreviations
    ]
    return returned_abbreviations


def preprocess_text(text):
    """
    Preprocesses the input text by performing several cleaning steps:
    1. Converts the text to lowercase.
    2. Removes all characters except word characters, whitespace, and hyphens.
    3. Replaces hyphens with spaces.
    4. Removes extra whitespace.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The cleaned and preprocessed text.
    """
    # Convert the text to lowercase
    text = text.lower()
    # Remove all characters except word characters, whitespace, and hyphens
    text = re.sub(r"[^\w\s-]", "", text)
    # Replace hyphens with spaces
    text = text.replace("-", " ")
    # Remove extra whitespace by splitting and rejoining the text
    text = " ".join(text.split())
    # Return the cleaned and preprocessed text
    return text


def compute_word_overlap(options, response):
    """
    Computes the word overlap scores between a response and a list of options using TF-IDF.

    This function preprocesses the response and options, computes their TF-IDF representations,
    and calculates the overlap scores based on the TF-IDF values.

    Args:
        options (list of str): A list of option strings to compare against the response.
        response (str): The response string to compare with the options.

    Returns:
        dict: A dictionary where the keys are the indices of the options and the values are the overlap scores.
    """
    # Preprocess the response and options
    processed_response = preprocess_text(response)
    processed_options = [preprocess_text(option) for option in options]
    # Create a TfidfVectorizer instance
    vectorizer = TfidfVectorizer()
    # Fit and transform the response and options into a TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform([processed_response] + processed_options)
    # Get the feature names (words) from the TF-IDF vectorizer
    words = vectorizer.get_feature_names_out()
    # Extract the TF-IDF scores for the response and options
    response_tfidf = tfidf_matrix[0].toarray()[0]
    options_tfidf = tfidf_matrix[1:].toarray()
    # Create a dictionary to map words to their TF-IDF scores for the response
    response_tfidf_dict = dict(zip(words, response_tfidf))
    # Create a list of dictionaries to map words to their TF-IDF scores for each option
    options_tfidf_dicts = [
        dict(zip(words, option_tfidf)) for option_tfidf in options_tfidf
    ]
    # Initialize a dictionary to store the overlap scores for each option
    scores_dict = {}
    # Calculate the overlap score for each option
    for i, option_tfidf_dict in enumerate(options_tfidf_dicts):
        overlap_score = 0.0
        # Sum the product of TF-IDF scores for words that appear in both the response and the option
        for word, score in option_tfidf_dict.items():
            if word in response_tfidf_dict:
                overlap_score += score * response_tfidf_dict[word]
        # Store the overlap score in the dictionary with the option index as the key
        scores_dict[i] = overlap_score
    # Return the dictionary of overlap scores
    return scores_dict


def compute_cosine_similarity(options, response, embedding_model):
    """
    Computes the cosine similarity scores between a response and a list of options using a given embedding model.

    This function preprocesses the response and options, computes their embeddings using the provided embedding model,
    and calculates the cosine similarity scores between the response embedding and each option embedding.

    Args:
        options (list of str): A list of option strings to compare against the response.
        response (str): The response string to compare with the options.
        embedding_model: The embedding model used to encode the options and response into embeddings.

    Returns:
        dict: A dictionary where the keys are the indices of the options and the values are the cosine similarity scores.
    """
    # Preprocess the options and response
    options = [preprocess_text(option) for option in options]
    response = preprocess_text(response)
    # Encode the options and response into embeddings using the embedding model
    options_embeddings = embedding_model.encode(options)
    response_embedding = embedding_model.encode([response])[0]
    # Initialize a dictionary to store the cosine similarity scores for each option
    scores_dict = {}
    # Calculate the cosine similarity score for each option
    for i, option in enumerate(options):
        option_embedding = options_embeddings[i]
        scores_dict[i] = cos_sim(response_embedding, option_embedding)
    # Convert the scores from tensor to float
    for key, value in scores_dict.items():
        scores_dict[key] = float(value)
    # Return the dictionary of cosine similarity scores
    return scores_dict


def generate_response(question, context, abbreviations, tokenizer):
    """
    Generates a response to a given question using a language model.

    This function constructs a prompt using the provided question, context, and abbreviations,
    then uses a language model pipeline to generate a response based on the prompt.

    Args:
        question (str): The question to be answered.
        context (str): The context or domain information relevant to the question.
        abbreviations (list of dict): A list of dictionaries containing abbreviations and their meanings.
        tokenizer: The tokenizer associated with the language model, used for handling special tokens.

    Returns:
        str: The generated response text.
    """
    # Create the prompt using the provided question, context, and abbreviations
    prompt = create_prompt(question, context, abbreviations)
    # Generate sequences using the language model pipeline
    sequences = pipeline(
        f"{prompt}",  # The prompt to be used for generation
        max_new_tokens=100,  # Maximum number of new tokens to generate
        do_sample=False,  # Disable sampling to generate deterministic output
        num_return_sequences=1,  # Number of sequences to return
        truncation=True,  # Enable truncation of the input prompt
        eos_token_id=tokenizer.eos_token_id,  # End-of-sequence token ID
        pad_token_id=tokenizer.eos_token_id,  # Padding token ID
    )
    # Initialize an empty string to store the response
    response = ""
    # Join the generated sequences into a single response string
    for seq in sequences:
        response += seq["generated_text"]
    # Return the generated response
    return response


# Load documents
print("Loading documents...")
print("Takes about 5 minutes...")
# Load documents from the specified directory
documents = SimpleDirectoryReader("data/rel18").load_data()
docs_str = [doc.text for doc in documents]  # Extract text from each document

# Initialize RAG model and Index Documents
# Load the pre-trained RAG model ragatouille with  colbert embeddings
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
print("Indexing documents...")
print("Takes about 20 minutes...")
# Index the documents using the RAG model
RAG.index(
    collection=docs_str,  # The collection of document texts to index
    index_name="ITU RAG 150",  # The name of the index to be created
    max_document_length=150,  # Maximum length of each document
    split_documents=True,  # Whether to split documents into smaller chunks
    use_faiss=True,  # Whether to use FAISS for efficient similarity search
)

# Load questions from JSON files
with open("data/TeleQnA_testing1.txt", "r") as file1, open(
        "data/questions_new.txt", "r"
) as file2:
    questions = json.load(file1)  # Load questions from the first JSON file
    # Load and merge questions from the second JSON file
    questions.update(json.load(file2))

# Loading Language Model, Tokenizer  and embedding model
# Tag for the embedding model
embedding_model_tag = "alexgichamba/gte-large-en-v1.5-triplet-finetuned-for-telco-qa"
# Initialize the embedding model
embedding_model = SentenceTransformer(embedding_model_tag, trust_remote_code=True)
# Define language model to be loaded
model_name = "tiiuae/falcon-7b-instruct"
# Initialize the tokenizer for the language model
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Initialize the text generation pipeline with the specified model and tokenizer
pipeline = pipeline(
    "text-generation",  # Task type for the pipeline
    model=model_name,  # The pre-trained model to use
    tokenizer=tokenizer,  # The tokenizer to use
    torch_dtype=torch.bfloat16,  # Data type for the model's tensors
    trust_remote_code=True,  # Whether to trust remote code
    device_map="auto",  # Automatically map the model to available devices
)

# Generate responses for each question
responses = {}  # Initialize an empty dictionary to store responses
# Iterate over each question in the questions dictionary
for q_id, q_data in tqdm(questions.items(), desc="Processing questions"):
    q_id_number = q_id.split()[1]  # Extract the question ID number
    # Remove any text within square brackets from the question
    question_text = re.sub(r"\s*\[.*?\]\s*$", "", q_data["question"])
    # Extract options from the question data
    options = [
        (k, v) for k, v in q_data.items() if k.startswith("option") and v is not None
    ]
    # Perform a search using the RAG model to get the top 3 relevant documents
    results = RAG.search(query=question_text, k=3)
    # Concatenate the content of the top 3 search results to form the context
    context = " ".join([result["content"] for result in results])
    # Find abbreviations that appear in the question data
    abbreviations = find_appearing_abbreviations(q_data)
    # Generate a response using the question text, context, and abbreviations
    response = generate_response(question_text, context, abbreviations, tokenizer)
    # Extract the actual response text from the generated response
    response = response.split(">>ANSWER<<")[1].strip()
    # Print the question and the generated response
    print(f"\nQuestion: {question_text}\nResponse: {response}")
    # Store the question and response in the responses dictionary
    responses[q_id] = {"question": question_text, "response": response}

# Save responses to JSON file
with open("responses.json", "w") as f:
    json.dump(responses, f)  # Write the responses dictionary to a JSON file
# Load responses for scoring
with open("responses.json", "r") as f:
    responses = json.load(f)  # Load the responses from the JSON file

# Compute scores and save to CSV
answers = []  # Initialize an empty list to store answers
# Iterate over each question in the questions dictionary
for q_id, q_data in tqdm(questions.items(), desc="Scoring responses"):
    q_id_number = q_id.split()[1]  # Extract the question ID number
    # Remove any text within square brackets from the question
    question_text = re.sub(r"\s*\[.*?\]\s*$", "", q_data["question"])
    # Extract options from the question data
    options = [v for k, v in q_data.items() if k.startswith("option") and v is not None]
    # Retrieve the generated response for the current question
    response = responses[q_id]["response"]
    # Compute word overlap scores between the response and each option
    word_overlap_scores = compute_word_overlap(options, response)
    # Compute cosine similarity scores between the response and each option
    cosine_similarity_scores = compute_cosine_similarity(
        options, response, embedding_model
    )
    # Print the question text
    print(f"\nQuestion: {question_text}")
    # Print word overlap scores for each option
    print("Word overlap scores:")
    for i, score in word_overlap_scores.items():
        print(f"Option {i + 1}: {score}")
    # Print cosine similarity scores for each option
    print("Cosine similarity scores:")
    for i, score in cosine_similarity_scores.items():
        print(f"Option {i + 1}: {score}")
    # Define weights for word overlap and cosine similarity scores
    overlap_weight = 0.2
    similarity_weight = 0.8
    # Compute combined scores for each option using the defined weights
    scores = {
        i: word_overlap_scores[i] * overlap_weight
           + cosine_similarity_scores[i] * similarity_weight
        for i in range(len(options))
    }
    # Determine the option with the highest combined score
    # Add 1 to convert zero-based index to one-based index
    answer_id = max(scores, key=scores.get) + 1
    # Print the selected answer ID
    print(f"Answer ID: {answer_id}")
    # Append the question ID number, answer ID, and task name to the answers list
    answers.append([q_id_number, answer_id, "Falcon 7.5B"])

# Write the answers to a CSV file
with open("output_results.csv", "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Question_ID", "Answer_ID", "Task"])  # Write the header row
    csvwriter.writerows(answers)  # Write the answer rows

# Inform the user that processing is complete and responses have been saved
print("Processing complete. Responses saved to 'output_results.csv'.")
