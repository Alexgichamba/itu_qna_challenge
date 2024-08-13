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


def create_prompt(question, context):
    """
    Constructs a prompt string for a language model based on the given question and context.

    Args:
        question (str): The question to be included in the prompt.
        context (str): The context or domain information relevant to the question.
    Returns:
        str: A formatted prompt string ready for input to a language model.
    """
    # Strip any leading or trailing whitespace from the context
    context = context.strip()
    # Construct the prompt string with the given context and question
    prompt = (
        f">>DOMAIN<<\n{context}\n"
        f">>QUESTION<<{question}\n\n"
        f"Summarize the answer to just one sentence\n"
        f">>ANSWER<<"
    )
    return prompt

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
    for key, value in scores_dict.items():
        scores_dict[key] = float(value)
    return scores_dict


def generate_response(question, context, tokenizer):
    """
    Generates a response to a given question using a language model.

    This function constructs a prompt using the provided question, context,
    then uses a language model pipeline to generate a response based on the prompt.

    Args:
        question (str): The question to be answered.
        context (str): The context or domain information relevant to the question.
        tokenizer: The tokenizer associated with the language model, used for handling special tokens.

    Returns:
        str: The generated response text.
    """
    # Create the prompt using the provided question, and context
    prompt = create_prompt(question, context)
    sequences = pipeline(
        f"{prompt}",
        max_new_tokens=100,  # Maximum number of new tokens to generate
        do_sample=False,  # Disable sampling to generate deterministic output
        num_return_sequences=1,
        truncation=True,
        eos_token_id=tokenizer.eos_token_id,  # End-of-sequence token ID
        pad_token_id=tokenizer.eos_token_id,  # Padding token ID
    )
    response = ""
    for seq in sequences:
        response += seq["generated_text"]
    print(prompt)
    print(response)
    return response


# Load documents
print("Loading documents...")
# Load documents from the specified directory
documents = SimpleDirectoryReader("data/books").load_data()
docs_str = [doc.text for doc in documents]  # Extract text from each document

# Initialize RAG model and Index Documents
# RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
# print("Takes about 20 minutes...")
# # Index the documents using the RAG model
# RAG.index(
#     collection=docs_str,
#     index_name="MedMCQA",
#     max_document_length=150,
#     split_documents=True,
#     use_faiss=True,  # Whether to use FAISS for efficient similarity search
# )
RAG = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/MedMCQA")

# Load questions from JSON files
with open("data/validation_medmcqa_Pharmacology.txt", "r") as file1:
    questions = json.load(file1)

# Tag for the embedding model
embedding_model_tag = "alexgichamba/gte-large-en-v1.5-triplet-finetuned-for-telco-qa"
embedding_model = SentenceTransformer(embedding_model_tag, trust_remote_code=True)
# Define language model and tokenizer to be loaded
model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Initialize the text generation pipeline with the specified model and tokenizer
pipeline = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

# Generate responses for each question
responses = {}
# Iterate over each question in the questions dictionary
for q_id, q_data in tqdm(questions.items(), desc="Processing questions"):
    q_id_number = q_id.split()[1]  # Extract the question ID number
    # Remove the tag specifying the release version
    question_text = re.sub(r"\s*\[.*?\]\s*$", "", q_data["question"])
    options = [
        (k, v) for k, v in q_data.items() if k.startswith("option") and v is not None
    ]
    # Perform a search using the RAG model to get the top 3 relevant documents
    results = RAG.search(query=question_text, k=3)
    context = " ".join([result["content"] for result in results])
    response = generate_response(question_text, context, tokenizer)
    response = response.split(">>ANSWER<<")[1].strip()
    # print(f"\nQuestion: {question_text}\nResponse: {response}")
    responses[q_id] = {"question": question_text, "response": response}

# Save responses to JSON file
with open("responses.json", "w") as f:
    json.dump(responses, f)
# Load responses for scoring
with open("responses.json", "r") as f:
    responses = json.load(f)

# Compute scores and save to CSV
answers = []
# Iterate over each question in the questions dictionary
for q_id, q_data in tqdm(questions.items(), desc="Scoring responses"):
    q_id_number = q_id.split()[1]
    question_text = re.sub(r"\s*\[.*?\]\s*$", "", q_data["question"])
    options = [v for k, v in q_data.items() if k.startswith("option") and v is not None]
    response = responses[q_id]["response"]
    # Compute word overlap scores between the response and each option
    word_overlap_scores = compute_word_overlap(options, response)
    # Compute cosine similarity scores between the response and each option
    cosine_similarity_scores = compute_cosine_similarity(
        options, response, embedding_model
    )
    print(f"\nQuestion: {question_text}")
    print("Word overlap scores:")
    for i, score in word_overlap_scores.items():
        print(f"Option {i + 1}: {score}")
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
    print(f"Answer ID: {answer_id}")
    answers.append([q_id_number, answer_id, "Falcon 7.5B"])

# Write the answers to a CSV file
with open("output_results.csv", "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Question_ID", "Answer_ID", "Task"])  # Write the header row
    csvwriter.writerows(answers)  # Write the answer rows

print("Processing complete. Responses saved to 'output_results.csv'.")
