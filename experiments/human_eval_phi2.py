import json
import os
import re
from tqdm import tqdm
import platform

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

# Function to clear the console output
def clear_console():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')

def get_first_word_as_phrase(question):
    # Extract the first word from the question
    first_word = question.split()[0].lower() if question else ""
    
    # Predefined list of phrases
    phrases = ["why", "when", "what", "how", "who", "which", "where"]
    
    # Automatically assign the first word if it's in the list
    if first_word in phrases:
        return first_word
    else:
        # If first word doesn't match any phrase, return None to trigger manual input
        return None

# Function to prompt evaluator and restrict input to possible options
def get_valid_input(prompt_message, valid_options):
    print(f"{prompt_message} (Choose from: {', '.join(valid_options)})")
    user_input = input("> ")
    while user_input not in valid_options:
        print(f"Invalid input. Please choose from: {', '.join(valid_options)}")
        user_input = input("> ")
    return user_input

# Function to prompt evaluator and collect input
def collect_input_for_question(question_data, subdomain, release, correct_position, wrong_position):
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
    
    # Print selected answer
    print(f"Selected Answer: {question_data['response']}")
    
    # Automatically determine the question phrase from the first word
    automatic_phrase = get_first_word_as_phrase(question_data['question'])
    
    # Prompt for evaluator's input if automatic_phrase is not found
    if automatic_phrase:
        print(f"Automatically detected phrase: {automatic_phrase}")
    else:
        print("\nCould not detect a phrase from the question. Please provide the following details:")
        phrases = ["why", "when", "what", "how", "who", "which", "where"]
        automatic_phrase = get_valid_input("Question Phrase", phrases)

    # Define sets of possible options
    answer_types = ["numerical", "entity", "specification document", "functional role", "definition", "explanation", "other"]
    involved_options = ["standalone", "both", "all", "none"]
    document_recall = ["0", "1"]

    # Prompt for answer details, constrained to valid options
    involved_option = get_valid_input("Options involved: ", involved_options)
    answer_type = get_valid_input("Answer Type", answer_types)
    # Prompt for retrieval
    document_recall = get_valid_input("Document recall: ", document_recall)

    # Construct new fields
    return {
        'humaneval_question': {
            'phrase': automatic_phrase,
            'subdomain': subdomain,  # Automatically populated from 366qs.txt
            'release': release,      # Automatically populated from 366qs.txt
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
    evaluator_name = input("Please enter your name (alex or brian): ").lower()
    
    if evaluator_name not in ["alex", "brian"]:
        print("Invalid evaluator name. Please enter 'alex' or 'brian'.")
        return

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
    total_questions = len(errors_data)
    half_point = total_questions // 2

    # Split the questions for Alex and Brian based on the name
    if evaluator_name == "alex":
        questions_subset = list(errors_data.items())[:half_point]
    elif evaluator_name == "brian":
        questions_subset = list(errors_data.items())[half_point:]

    # Evaluate the assigned questions for the user
    output_filepath = f"data/phi2_error_analysis_human.json"
    
    for question_id, question_data in tqdm(questions_subset, desc=f"Processing errors for {evaluator_name}"):
        # Skip the question if it has already been evaluated
        if 'evaluator' in question_data:
            print(f"Skipping question {question_id} as it has already been evaluated.")
            continue
        
        # Clear the console at the start of each question
        clear_console()
        print(f"\n--- Evaluating {question_id} ---")
        
        # Fetch the corresponding subdomain and release from 366qs.txt
        metadata = metadata_map.get(question_id, {'subdomain': 'Unknown', 'release': 'Unknown', 'correct_option': 'Unknown'})
        subdomain = metadata['subdomain']
        release = metadata['release']
        correct_option = metadata['correct_option']
        wrong_option = question_data['response'].split(" ")[1]
        print(f"Subdomain: {subdomain}, Release: {release}, Correct Option: {correct_option}, Wrong Option: {wrong_option}")

        question_evaluation = collect_input_for_question(question_data, subdomain, release, correct_option, wrong_option)

        # Add evaluator name and collected input to the question data
        question_data['evaluator'] = evaluator_name
        question_data.update(question_evaluation)
    
        # Save the updated dataset after each question
        save_updated_errors(output_filepath, errors_data)
    
    print("Evaluation complete and saved.")

if __name__ == "__main__":
    run_evaluation()
