# human_eval_bm25.py

import random
import json
import os
import platform

# Set a fixed seed for deterministic behavior
random.seed(42)

# Function to shuffle and divide questions based on the evaluator's name
def assign_questions(questions, name):
    question_list = list(questions.items())  # Convert dictionary to list of (key, value) pairs
    random.shuffle(question_list)
    
    n = len(question_list) // 3
    if name == "alex":
        assigned_questions = question_list[:n]
    else:
        print("Invalid name. Only 'alex' is allowed.")
        return []
    
    return assigned_questions

# Function to evaluate binary recall
def evaluate_recall():
    recall = input("Please enter the binary recall value (0 or 1): ")
    while recall not in ["0", "1"]:
        print("Invalid input. Please enter 0 or 1.")
        recall = input("Please enter the binary recall value (0 or 1): ")
    return int(recall)

# Function to save results to a JSON file
def save_results(evaluator_name, results):
    output_filepath = f"data/{evaluator_name}_bm25_eval.json"
    
    # Check if the file already exists and load the data if it does
    if os.path.exists(output_filepath):
        with open(output_filepath, 'r') as f:
            existing_results = json.load(f)
    else:
        existing_results = {}

    # Update the results
    existing_results.update(results)

    # Save the updated results to the file
    with open(output_filepath, 'w') as f:
        json.dump(existing_results, f, indent=4)

    print(f"Results saved to {output_filepath}")

# Function to clear the console output
def clear_console():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')

# Main function to run the evaluation
def run_evaluation():
    # Get evaluator's name
    evaluator_name = input("Please enter your name (alex): ").lower()
    while evaluator_name not in ["alex"]:
        print("Invalid name. Please enter 'alex'.")
        evaluator_name = input("Please enter your name (alex): ").lower()

    # Load the retrieval results
    with open("data/results_bm25_search.json", "r") as file:
        questions = json.load(file)

    # Load additional data from 366qs.txt
    with open("data/366qs.txt", "r") as file:
        full_questions = json.load(file)

    # Assign questions based on the name
    assigned_questions = assign_questions(questions, evaluator_name)
    if not assigned_questions:
        return

    # Store results
    results = {}

    # Evaluate each assigned question
    for question_id, question_data in assigned_questions:
        # Clear the console at the start of each question
        clear_console()

        print(f"\n----------------- Evaluating {question_id} -----------------")
        print(f"\nQuestion: {question_data['question']}")

        # Get additional information from 366qs.txt
        full_question_data = full_questions.get(question_id)
        if full_question_data is None:
            print(f"No additional data found for question ID: {question_id}")
            continue

        # Dynamically extract all options
        options = {key: value for key, value in full_question_data.items() if key.startswith("option")}
        correct_answer = full_question_data.get("answer")

        print("\nOptions:")
        for key, option in options.items():
            print(f"{key}: {option}")

        print(f"\nCorrect Answer: {correct_answer}")

        print("\nContext:")
        for i, context in enumerate(question_data['retrieved_chunks']):
            print(f"\n-----Chunk {i+1}-----")
            print(f"{context['doc_text']}\n")
            if i == 12:
                break

        recall = evaluate_recall()
        results[question_id] = {
            "binary_recall": recall
        }

    # Save the results to a JSON file
    save_results(evaluator_name, results)

if __name__ == "__main__":
    run_evaluation()

