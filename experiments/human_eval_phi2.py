import json
import os
import re
from tqdm import tqdm

# Function to load the question data from the 366qs.txt file and map subdomains and releases
def load_questions_metadata(questions_filepath):
    if not os.path.exists(questions_filepath):
        print(f"Error: {questions_filepath} not found.")
        return {}
    
    with open(questions_filepath, 'r') as f:
        questions_data = json.load(f)
    
    metadata = {}
    release_pattern = re.compile(r"\[3GPP Release (\d+)\]")  # Pattern to capture the release number
    
    for question_id, question in questions_data.items():
        subdomain = question.get('category', 'Unknown')
        
        # Extract the release number from the question text using regex
        release_match = release_pattern.search(question['question'])        
        release = release_match.group(1) if release_match else "Unknown"
        if release == "Unknown":
            print(f"Release not found for question: {question_id}")
        
        correct_option_pattern = re.compile(r"option (\d+):")  # Pattern to capture the correct option number

        # Extract the correct option number using regex
        answer_match = correct_option_pattern.search(question.get('answer', ''))
        correct_option = answer_match.group(1) if answer_match else "Unknown"
        if correct_option == "Unknown":
            print(f"Correct option not found for question: {question_id}")
        
        # Store subdomain and release for each question
        metadata[question_id] = {
            'subdomain': subdomain,
            'release': release,
            'correct_option': correct_option
        }
    
    return metadata

# Function to prompt evaluator and restrict input to possible options
def get_valid_input(prompt_message, valid_options):
    print(f"{prompt_message} (Choose from: {', '.join(valid_options)})")
    user_input = input("> ")
    while user_input not in valid_options:
        print(f"Invalid input. Please choose from: {', '.join(valid_options)}")
        user_input = input("> ")
    return user_input

# Function to prompt evaluator and collect input
def collect_input_for_question(question_data, subdomain, release, correct_position):
    print(f"Question: {question_data['question']}")
    
    # Print abbreviations if available
    if 'abbreviations' in question_data:
        abbrevs = question_data['abbreviations']
        print("Abbreviations:")
        for abbrev in abbrevs:
            print(f"{list(abbrev.keys())[0]}: {list(abbrev.values())[0]}")
    
    # Print context (retrieved chunks)
    if 'retrieved_chunks' in question_data:
        print("Context:")
        for chunk in question_data['retrieved_chunks']:
            print(f"- {chunk['content']}")
    
    # Print options
    print(f"Question: {question_data['question']}")
    print("Options:")
    for option in question_data['options']:
        print(f"{option[0]}: {option[1]}")
    
    # Print correct answer
    print(f"Correct Answer: {question_data['correct_answer']}")
    
    # print selected answer
    print(f"Selected Answer: {question_data['response']}")
    
    # Define sets of possible options
    phrases = ["why", "when", "what", "how", "who", "which", "where"]
    # question_types = ["factoid", "list", "yesno", "relationship", "definition","other"]
    answer_types = ["entity", "place", "number", "date", "explanation", "method", "document", "thing", "list", "other"]
    involved_options = ["standalone", "both", "all", "none"]
    # correct_positions = ["1", "2", "3", "4", "5"]
    wrong_positions = ["1", "2", "3", "4", "5"]
    document_recall = ["0", "1"]

    # Prompt for evaluator's input, constrained to valid options
    print("\nPlease provide the following details:")
    phrase = get_valid_input("Question Phrase", phrases)
    # question_type = get_valid_input("Question Type", question_types)

    # Prompt for answer details, constrained to valid options
    involved_option = get_valid_input("Options involved: ", involved_options)
    answer_type = get_valid_input("Answer Type", answer_types)
    # correct_position = get_valid_input("Correct position: ", correct_positions)
    wrong_position = get_valid_input("Wrong position: ", wrong_positions)

    # Prompt for retrieval
    document_recall = get_valid_input("Document recall: ", document_recall)

    # Construct new fields
    return {
        'humaneval_question': {
            'phrase': phrase,
            'subdomain': subdomain,  # Automatically populated from 366qs.txt
            'release': release,      # Automatically populated from 366qs.txt
            # 'type': question_type
        },
        'humaneval_answer': {
            'options_involved': involved_option,
            'answer_type': answer_type,
            'correct_position': correct_position, # Automatically populated from 366qs.txt
            'wrong_position': wrong_position
        },
        'humaneval_retrieval': {
            'document_recall': document_recall
        }
    }

# Load the dataset
def load_errors(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# Save the updated dataset
def save_updated_errors(filepath, errors_data):
    with open(filepath, 'w') as f:
        json.dump(errors_data, f, indent=4)

# Main function to run the evaluation
def run_evaluation():
    evaluator_name = input("Please enter your name: ")

    # Load errors dataset
    errors_filepath = "data/errors_phi2.json"
    if not os.path.exists(errors_filepath):
        print(f"Error: {errors_filepath} not found.")
        return
    
    # Load the subdomains and releases from data/366qs.txt
    questions_filepath = "data/366qs.txt"
    metadata_map = load_questions_metadata(questions_filepath)

    if not metadata_map:
        print("No metadata (subdomains/releases) found. Please check the dataset.")
        return

    errors_data = load_errors(errors_filepath)

    # Collect input for each question in the dataset
    for question_id, question_data in tqdm(errors_data.items(), desc="Processing errors"):
        print(f"\n--- Evaluating {question_id} ---")
        
        # Fetch the corresponding subdomain and release from 366qs.txt
        metadata = metadata_map.get(question_id, {'subdomain': 'Unknown', 'release': 'Unknown', 'correct_option': 'Unknown'})
        subdomain = metadata['subdomain']
        release = metadata['release']
        correct_option = metadata['correct_option']
        print(f"Subdomain: {subdomain}, Release: {release} (Correct Option: {correct_option})")

        question_evaluation = collect_input_for_question(question_data, subdomain, release, correct_option)

        # Add evaluator name and collected input to the question data
        question_data['evaluator'] = evaluator_name
        question_data.update(question_evaluation)
    
    # Save the updated dataset
    output_filepath = f"data/{evaluator_name}_eval_errors_phi2.json"
    save_updated_errors(output_filepath, errors_data)
    print("Evaluation complete and saved.")

if __name__ == "__main__":
    run_evaluation()
