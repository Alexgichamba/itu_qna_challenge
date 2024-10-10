# gpt4_bin_recall.py

import json

def calculate_recall_stats(data, top_n=13):
    total_recall = 0
    num_ones = 0
    num_zeros = 0

    for question_id, details in data.items():
        retrieved_chunks = details['retrieved_chunks'][:top_n]  # Only consider top N chunks

        # Binary recall: 1 if there are relevant chunks, otherwise 0
        recall = 1 if any(chunk['relevance'] == "1" for chunk in retrieved_chunks) else 0
        
        if recall == 1:
            num_ones += 1
        else:
            num_zeros += 1

    total_recall = num_ones + num_zeros
    percentage_ones = (num_ones / total_recall) * 100 if total_recall > 0 else 0

    return num_ones, num_zeros, percentage_ones

# Load your data (replace 'your_data.json' with the actual file path)
with open('data/gpt-4o-question-topk-analysis.json', 'r') as f:
    data = json.load(f)

# Get the statistics
num_ones, num_zeros, percentage_ones = calculate_recall_stats(data, top_n=13)

# Print the results
print(f"Number of 1s: {num_ones}")
print(f"Number of 0s: {num_zeros}")
print(f"Percentage of 1s: {percentage_ones:.2f}%")

