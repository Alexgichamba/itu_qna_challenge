
from llama_index.core import SimpleDirectoryReader

print("Loading documents...")
print("Takes about 5 minutes...")
documents = SimpleDirectoryReader("data/rel18").load_data()


docs_str = []
for doc in documents:
  docs_str.append(doc.text)


len(docs_str)


from ragatouille import RAGPretrainedModel
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

print("Indexing documents...")
print("Takes about 20 minutes...")
RAG.index(
    collection=docs_str,
    index_name="ITU RAG 150",
    max_document_length=150,
    split_documents=True,
    use_faiss=True
)


results = RAG.search(query="What does the UE provide to the AS for slice aware cell reselection?", k=7)
results


# Read questions from the JSON file
import json
with open('data/TeleQnA_testing1.txt', 'r') as file1:
  with open('data/questions_new.txt', 'r') as file2:
    questions = json.load(file1)
    questions.update(json.load(file2))


from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

print("Loading model...")
model = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)


# set random seeds for reproducibility
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def create_prompt(question, context, abbrevs):
    context = context.strip()
    # options_text = "\n".join([f"Option {i+1}: {opt}" for i, opt in enumerate(options)])
    abbreviations_text = "\n".join([f"{list(abbrev.keys())[0]}: {list(abbrev.values())[0]}" for abbrev in abbrevs])
    prompt = (
        f">>DOMAIN<<\n{context}\n"
        f"Abbreviations:\n{abbreviations_text}\n"
        f">>QUESTION<<{question}\n\n"
        f">>ANSWER<<"
    )
    return prompt


def find_appearing_abbreviations(question):
    # read abbreviations
    with open('data/abbreviations.txt', 'r') as f_abbrevs:
        abbreviations = {}
        for line in f_abbrevs:
            abbreviation, full_form = line.split(': ', 1)
            abbreviations[abbreviation.strip()] = full_form.strip()
    # sort abbreviations by length in descending order to handle cases
    #  where one abbreviation is a substring of another
    sorted_abbrevs = sorted(abbreviations.items(), key=lambda x: len(x[0]), reverse=True)
    assert isinstance(question, dict)
    appearing_abbreviations = set()  # Use a set to store unique abbreviations

    for abbreviation, full_form in sorted_abbrevs:
        # find the abbreviation in the text
        pattern = r'\b' + re.escape(abbreviation) + r'\b'
        # if the abbreviation is found:
        if re.search(pattern, (question['question'].split('?')[0])):
            appearing_abbreviations.add(abbreviation)
        for key in question:
            if key.startswith('option') and question[key] is not None:
                if re.search(pattern, question[key]):
                    appearing_abbreviations.add(abbreviation)

    # return a list of dicts with the abbreviation and its full form
    returned_abbreviations = [{abbrev: abbreviations[abbrev]} for abbrev in appearing_abbreviations]
    return returned_abbreviations


import re
q_id = list(questions.keys())[3]
q_data = questions[q_id]
question_text = q_data["question"]
abbrevs = find_appearing_abbreviations(q_data)
question_text = re.sub(r'\s*\[.*?\]\s*$', '', question_text) # remove tags
options = [v for k, v in q_data.items() if k.startswith("option")]
context = RAG.search(query=question_text, k=4)
context = " ".join([result['content'] for result in context])
prompt = create_prompt(question_text, context, abbrevs)
sequences = pipeline(
   f"{prompt}",
    max_new_tokens=35,
    do_sample=False,
    num_return_sequences=1,
    truncation=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)
for seq in sequences:
    print(f"{seq['generated_text']}")


def generate_response(question, context, abbreviations, model, tokenizer):
    prompt = create_prompt(question_text, context, abbrevs)
    sequences = pipeline(
      f"{prompt}",
        max_new_tokens=100,
        do_sample=False,
        num_return_sequences=1,
        truncation=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    # join returned sequences
    response = ""
    for seq in sequences:
        response += seq['generated_text']
    return response


from tqdm import tqdm

responses = {}

# Loop through each question and get the response
for q_id, q_data in tqdm(questions.items(), desc="Processing questions"):
    q_id_number = q_id.split()[1]
    question_text = q_data["question"]
    question_text = re.sub(r'\s*\[.*?\]\s*$', '', question_text)
    options = [(k, v) for k, v in q_data.items() if k.startswith("option") and v is not None]

    # Retrieve context using ColBERT search
    results = RAG.search(query=question_text, k=3)
    context = " ".join([result['content'] for result in results])

    abbreviations = find_appearing_abbreviations(q_data)
    # Generate the answer using the loaded model
    response = generate_response(question_text, context, abbreviations, model, tokenizer)


    response = response.split(">>ANSWER<<")[1].strip()
    print(f"\nQuestion: {question_text}\nResponse: {response}")
    # Append the response to the list
    responses[q_id] = {
    "question": question_text,
    "response": response}

# Save responses to a json file
with open('responses.json', 'w') as f:
    json.dump(responses, f)


import re
from sklearn.feature_extraction.text import TfidfVectorizer

# compute word overlap between options and the response
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation except for hyphens
    text = re.sub(r'[^\w\s-]', '', text)
    # Split on hyphens and rejoin with spaces
    text = text.replace('-', ' ')
    # Split and rejoin to handle multiple spaces
    text = ' '.join(text.split())
    return text

def compute_word_overlap(options, response):
    # Preprocess the response and options
    processed_response = preprocess_text(response)
    processed_options = [preprocess_text(option) for option in options]

    # Create a TfidfVectorizer instance
    vectorizer = TfidfVectorizer()

    # Fit and transform the response and options
    tfidf_matrix = vectorizer.fit_transform([processed_response] + processed_options)

    # Get the feature names (words)
    words = vectorizer.get_feature_names_out()

    # Get the TF-IDF scores
    response_tfidf = tfidf_matrix[0].toarray()[0]
    options_tfidf = tfidf_matrix[1:].toarray()

    # Create dictionaries to map words to their TF-IDF scores
    response_tfidf_dict = dict(zip(words, response_tfidf))
    options_tfidf_dicts = [dict(zip(words, option_tfidf)) for option_tfidf in options_tfidf]

    scores_dict = {}

    for i, option_tfidf_dict in enumerate(options_tfidf_dicts):
        overlap_score = 0.0

        for word, score in option_tfidf_dict.items():
            if word in response_tfidf_dict:
                overlap_score += score * response_tfidf_dict[word]

        scores_dict[i] = overlap_score

    return scores_dict


# setup embedding model
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# we use a finetuned version of "Alibaba-NLP/gte-large-en-v1.5"
# we finetuned it on triplets of:
# Explanation (anchor), Correct option (positive), A randomly selected wrong option (negative)
embedding_model_tag = "alexgichamba/gte-large-en-v1.5-triplet-finetuned-for-telco-qa"
embedding_model = SentenceTransformer(embedding_model_tag, trust_remote_code=True)


# compute cosine similarity
def compute_cosine_similarity(options, response):
    # first preprocess
    options = [preprocess_text(option) for option in options]
    response = preprocess_text(response)
    options_embeddings = embedding_model.encode(options)
    response_embedding = embedding_model.encode([response])[0]
    scores_dict = {}
    for i, option in enumerate(options):
        option_embedding = options_embeddings[i]
        scores_dict[i] = cos_sim(response_embedding, option_embedding)

    # change scores to numbers, not tensor
    for key, value in scores_dict.items():
        scores_dict[key] = float(value)
    return scores_dict


# open responses file
answers = []
import csv
import json
from tqdm import tqdm
import re
with open('responses.json', 'r') as f:
    responses = json.load(f)

for q_id, q_data in tqdm(questions.items(), desc="Scoring responses"):
    q_id_number = q_id.split()[1]
    question_text = q_data["question"]
    question_text = re.sub(r'\s*\[.*?\]\s*$', '', question_text)
    options = [v for k, v in q_data.items() if k.startswith("option") and v is not None]

    # find matching response
    response = responses[q_id]["response"]

    # compute word overlap and cosine similarity scores
    word_overlap_scores = compute_word_overlap(options, response)
    cosine_similarity_scores = compute_cosine_similarity(options, response)

    # print scores
    print(f"\nQuestion: {question_text}")
    print("Word overlap scores:")
    for i, score in word_overlap_scores.items():
        print(f"Option {i+1}: {score}")
    print("Cosine similarity scores:")
    for i, score in cosine_similarity_scores.items():
        print(f"Option {i+1}: {score}")

    # static weights
    overlap_weight = 0.2
    similarity_weight = 0.8

    # compute final scores
    scores = {}
    for i, option in enumerate(options):
        scores[i] = word_overlap_scores[i] * overlap_weight + cosine_similarity_scores[i] * similarity_weight

    # find the answer id (key) for which the score is highest
    answer_id = max(scores, key=scores.get)
    answer_id = answer_id + 1
    print(f"Answer ID: {answer_id}")
    # append the answer id to the list
    answers.append([q_id_number, answer_id, "Falcon 7.5B"])

# Save answera to a CSV file
with open('output_results.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Question_ID", "Answer_ID", "Task"])
    csvwriter.writerows(answers)

print("Processing complete. Responses saved to 'output_results.csv'.")



