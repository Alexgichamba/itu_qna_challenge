""""Count the number of occurrences of binary recall values in a JSON file."""

import json
import argparse

def count_binary_recall(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    recall_count = {0: 0, 1: 0}
    
    for question, details in data.items():
        binary_recall = details.get('binary_recall')
        if binary_recall in recall_count:
            recall_count[binary_recall] += 1
    
    return recall_count

def main():
    parser = argparse.ArgumentParser(description='Count occurrences of binary recall values in a JSON file.')
    parser.add_argument('file_path', type=str, help='Path to the JSON file')
    args = parser.parse_args()
    
    recall_count = count_binary_recall(args.file_path)
    print(f"Occurrences of binary_recall 0: {recall_count[0]}")
    print(f"Occurrences of binary_recall 1: {recall_count[1]}")

if __name__ == '__main__':
    main()

#python count-binary-recall.py data/brian_colbert_eval.json