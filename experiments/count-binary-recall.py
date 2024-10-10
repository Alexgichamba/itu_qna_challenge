import json

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def count_binary_recall(data):
    recall_count = {0: 0, 1: 0}
    
    for question, details in data.items():
        binary_recall = details.get('binary_recall')
        if binary_recall in recall_count:
            recall_count[binary_recall] += 1
    
    return recall_count

def compare_colbert_bm25(colbert_data, bm25_data):
    colbert_got_not_bm25 = []
    bm25_got_not_colbert = []
    
    for question, colbert_details in colbert_data.items():
        bm25_details = bm25_data.get(question)
        
        if colbert_details['binary_recall'] == 1 and bm25_details['binary_recall'] == 0:
            colbert_got_not_bm25.append(question)
        elif colbert_details['binary_recall'] == 0 and bm25_details['binary_recall'] == 1:
            bm25_got_not_colbert.append(question)
    
    return colbert_got_not_bm25, bm25_got_not_colbert

def main():
    # Hard-coded file paths
    colbert_files = ['data/brian_colbert_eval.json', 'data/teddy_colbert_eval.json']
    bm25_file = 'data/alex_bm25_eval.json'
    
    # Load and combine ColBERT results
    colbert_data_combined = {}
    for file in colbert_files:
        colbert_data = load_data(file)
        colbert_data_combined.update(colbert_data)
    
    # Load BM25 results
    bm25_data = load_data(bm25_file)
    
    # Count binary recall for ColBERT
    recall_count_colbert = count_binary_recall(colbert_data_combined)
    total_colbert = sum(recall_count_colbert.values())
    print(f"ColBERT Results:")
    print(f"Occurrences of binary_recall 0: {recall_count_colbert[0]}, percentage: {recall_count_colbert[0]/total_colbert*100:.2f}%")
    print(f"Occurrences of binary_recall 1: {recall_count_colbert[1]}, percentage: {recall_count_colbert[1]/total_colbert*100:.2f}%")

    # Count binary recall for BM25
    recall_count_bm25 = count_binary_recall(bm25_data)
    total_bm25 = sum(recall_count_bm25.values())
    print(f"\nBM25 Results:")
    print(f"Occurrences of binary_recall 0: {recall_count_bm25[0]}, percentage: {recall_count_bm25[0]/total_bm25*100:.2f}%")
    print(f"Occurrences of binary_recall 1: {recall_count_bm25[1]}, percentage: {recall_count_bm25[1]/total_bm25*100:.2f}%")

    # Compare results
    colbert_got_not_bm25, bm25_got_not_colbert = compare_colbert_bm25(colbert_data_combined, bm25_data)
    
    # Report results
    print(f"\nQuestions ColBERT got right but BM25 didn't (Total: {len(colbert_got_not_bm25)}):")
    for question in colbert_got_not_bm25:
        print(f"- {question}")
    
    print(f"\nQuestions BM25 got right but ColBERT didn't (Total: {len(bm25_got_not_colbert)}):")
    for question in bm25_got_not_colbert:
        print(f"- {question}")

if __name__ == '__main__':
    main()
