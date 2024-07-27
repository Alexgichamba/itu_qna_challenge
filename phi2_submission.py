# phi2 submission script
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


from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
print("Loading model...")
config = PeftConfig.from_pretrained("alexgichamba/phi-2-finetuned-qa-lora-r32-a16_longcontext")
base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2").to('cuda')
model = PeftModel.from_pretrained(base_model, "alexgichamba/phi-2-finetuned-qa-lora-r32-a16_longcontext").to('cuda')
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")


# set random seeds for reproducibility
import random
import numpy as np
import torch

seed = 20
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

import re

def create_prompt(question, options, context, abbreviations):
    options_text = "\n".join([f"Option {i+1}: {opt[1]}" for i, opt in enumerate(options)])
    # abbreviations is a list of dictionaries of form {"abbreviation": "full form"}
    abbreviations_text = "\n".join([f"{list(abbrev.keys())[0]}: {list(abbrev.values())[0]}" for abbrev in abbreviations])
    prompt = (
        f"Instruct: You will answer each question correctly using the context below\n"
        # f"The output should be in the format: Option <Option id>\n"
        # f"Provide the answer to the following multiple choice question in the specified format.\n\n"
        f"Context: {context}\n\n"
        f"Abbreviations:\n{abbreviations_text}\n\n"
        f"Question: {question}\n"
        f"{options_text}\n"
        f"Answer: Option"
    )
    return prompt


def generate_answer(question, options, context, abbreviations, model, tokenizer):
    prompt = create_prompt(question, options, context, abbreviations)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to('cuda')

    # Ensure the pad token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to('cuda')  # Set attention mask

    # Generate the answer with appropriate parameters
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=10,  # Limit the number of new tokens generated
        pad_token_id=tokenizer.eos_token_id,  # Handle padding correctly
        num_beams=1,  # Use beam search to improve quality of generated answers
        early_stopping=True  # Stop early when enough beams have reached EOS
    )
    answer = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated answer: {answer}")
    return answer


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


# First search for the full pattern
def parse_answer(response):
  match = re.search(r'Answer:\s*Option\s*(\d+)', response, re.IGNORECASE)
  if match:
      answer = f"Option {match.group(1)}"
  else:
      # Try another pattern if the first one fails
      match = re.search(r'(\d+)', response, re.IGNORECASE)
      if match:
          answer = f"Option {match.group(1)}"
      else:
          answer = "Error"
  return answer


import csv
from tqdm import tqdm

responses = []

# Loop through each question and get the response
for q_id, q_data in tqdm(questions.items(), desc="Processing questions"):
    q_id_number = q_id.split()[1]
    question_text = q_data["question"]
    question_text = re.sub(r'\s*\[.*?\]\s*$', '', question_text)
    options = [(k, v) for k, v in q_data.items() if k.startswith("option") and v is not None]

    # Retrieve context using ColBERT search
    results = RAG.search(query=question_text, k=13)
    context = " ".join([result['content'] for result in results])

    abbreviations = find_appearing_abbreviations(q_data)
    # Generate the answer using the loaded model
    response = generate_answer(question_text, options, context, abbreviations, model, tokenizer)

    answer = parse_answer(response)

    # Extract the answer ID from the response
    match = re.search(r'Option (\d+)', answer)
    if match:
        try:
            answer_id = int(match.group(1))
            print(f"Answer ID: {answer_id}")
            responses.append([q_id_number, answer_id, "Phi-2"])
        except (KeyError, IndexError, ValueError) as e:
            responses.append([q_id_number, "Error", "Phi-2"])
            print(f"Error processing question {q_id}: {answer}")
    else:
        responses.append([q_id_number, "Error", "Phi-2"])
        print(f"Error processing question {q_id_number}: {answer}")

# Save responses to a CSV file
with open('output_results.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Question_ID", "Answer_ID", "Task"])
    csvwriter.writerows(responses)

print("Processing complete. Responses saved to 'output_results.csv'.")