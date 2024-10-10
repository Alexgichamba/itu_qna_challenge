import json
from collections import defaultdict
import re

# Load the dataset from a file
def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# Extract the phrase from the question text
def extract_phrase(question_text):
    possible_phrases = ["why", "when", "what", "how", "who", "which", "where"]
    first_word = question_text.split()[0].lower()
    return first_word if first_word in possible_phrases else "none"

# Extract correct position from the answer field
def extract_correct_position(answer_text):
    match = re.search(r'option (\d+):', answer_text)
    return match.group(1) if match else answer_text

# Determine options_involved based on the answer text
def determine_options_involved(answer_text):
    if "All of the above" in answer_text:
        return "all"
    elif "None of the above" in answer_text:
        return "none"
    elif "Both option" in answer_text:
        return "both"
    else:
        return "standalone"

# Compute frequencies from the dataset
def compute_frequencies(dataset, is_error_set=False):
    frequencies = defaultdict(lambda: defaultdict(int))

    for question_id, question_data in dataset.items():
        if is_error_set:
            question_text = question_data['humaneval_question']['phrase']
            subdomain = question_data['humaneval_question']['subdomain']
            release = question_data['humaneval_question']['release']
            answer_text = question_data['humaneval_answer']['correct_position']
            options_involved = question_data['humaneval_answer']['options_involved']
            # Adding binary recall from the error set
            binary_recall = question_data['humaneval_retrieval'].get('document_recall', 'unknown')
        else:
            question_text = question_data['question']
            subdomain = question_data['category']
            answer_text = question_data['answer']
            release_match = re.search(r'\[3GPP Release (\d+)\]', question_text)
            release = release_match.group(1) if release_match else "unknown"
            options_involved = determine_options_involved(answer_text)

            # Hardcoded binary recall values for the dataset
            binary_recall = '1'

        # Extract phrase
        phrase = extract_phrase(question_text)
        frequencies['phrase'][phrase] += 1

        # Subdomain
        frequencies['subdomain'][subdomain] += 1

        # Release
        frequencies['release'][release] += 1

        # Correct position
        correct_position = extract_correct_position(answer_text)
        frequencies['correct_position'][correct_position] += 1

        # Options involved
        frequencies['options_involved'][options_involved] += 1

        # Binary recall (hardcoded)
        frequencies['binary_recall'][binary_recall] += 1

    return frequencies

# Hardcoded binary recall values for the dataset
def hardcode_binary_recall(frequencies):
    frequencies['binary_recall']['0'] = 24
    frequencies['binary_recall']['1'] = 98
    return frequencies

# Compute total counts for each field
def compute_totals(frequencies):
    totals = {field: sum(outcome_counts.values()) for field, outcome_counts in frequencies.items()}
    return totals

# Pretty print the comparison between dataset-level and error set-level frequencies with percentages and relative changes
def print_comparison(dataset_frequencies, error_set_frequencies, dataset_totals, error_set_totals):
    print("\n=== Dataset vs Error Set Frequencies ===\n")

    for field in dataset_frequencies.keys():
        print(f"Frequencies for {field}:")
        print(f"{'Outcome':<25}{'Dataset Count':<15}{'Error Set Count':<15}{'Dataset %':<12}{'Error Set %':<12}{'Relative Change (%)':<18}")
        print("-" * 95)
        dataset_outcomes = dataset_frequencies[field]
        error_set_outcomes = error_set_frequencies[field]

        all_outcomes = set(dataset_outcomes.keys()).union(set(error_set_outcomes.keys()))
        for outcome in sorted(all_outcomes):
            dataset_count = dataset_outcomes.get(outcome, 0)
            error_set_count = error_set_outcomes.get(outcome, 0)

            # Calculate percentages
            dataset_percentage = (dataset_count / dataset_totals[field]) * 100 if dataset_totals[field] > 0 else 0
            error_set_percentage = (error_set_count / error_set_totals[field]) * 100 if error_set_totals[field] > 0 else 0

            # Calculate the relative change
            if dataset_percentage == 0:
                relative_change = "N/A"  # Cannot compute relative change with a zero dataset percentage
                print(f"{outcome:<25}{dataset_count:<15}{error_set_count:<15}{dataset_percentage:<12.2f}{error_set_percentage:<12.2f}{relative_change:<18}")
            else:
                relative_change = ((error_set_percentage - dataset_percentage) / dataset_percentage) * 100
                print(f"{outcome:<25}{dataset_count:<15}{error_set_count:<15}{dataset_percentage:<12.2f}{error_set_percentage:<12.2f}{relative_change:<18.2f}")
        print()

# Main function to analyze both the dataset and the error set
def analyze_frequencies_side_by_side():
    # Load dataset and error set
    dataset_filepath = "data/366qs.txt"
    error_set_filepath = "data/phi2_human_evaluations.json"

    dataset = load_json(dataset_filepath)
    error_set = load_json(error_set_filepath)

    # Compute frequencies for dataset and error set
    dataset_frequencies = compute_frequencies(dataset, is_error_set=False)
    error_set_frequencies = compute_frequencies(error_set, is_error_set=True)

    # Hardcode binary recall values for the dataset
    dataset_frequencies = hardcode_binary_recall(dataset_frequencies)

    # Compute total counts for dataset and error set
    dataset_totals = compute_totals(dataset_frequencies)
    error_set_totals = compute_totals(error_set_frequencies)

    # Print side-by-side comparison with percentages and relative changes
    print_comparison(dataset_frequencies, error_set_frequencies, dataset_totals, error_set_totals)

if __name__ == "__main__":
    analyze_frequencies_side_by_side()
