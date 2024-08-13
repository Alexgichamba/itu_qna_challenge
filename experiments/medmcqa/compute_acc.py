import csv
import json
import os

# Load the questions dataset from the JSON file
with open('data/validation_medmcqa_Pharmacology.txt', 'r') as rubric:
    qs_w_ans = json.load(rubric)

# Find all the output_results files
output_file = "output_results.csv"

responses = []
with open(output_file, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        responses.append(row)

# Initialize score
correct_answers = 0
total_questions = len(responses)
print(f"Total questions: {total_questions}")
# Track question_ids for failed questions
failed_questions = []

# Compare the responses with the correct answers for the first 366 questions
for response in responses:
    question_id = response['Question_ID']
    answer_id = response['Answer_ID']
    task = response['Task']
    
    # Find the corresponding question in the JSON data
    question_key = f"question {question_id}"
    if question_key in qs_w_ans:
        correct_answer = qs_w_ans[question_key]['answer']
        # Extract the correct option number from the correct answer string
        correct_option_number = correct_answer.split()[1].replace(":", "")
        
        # Check if the given answer matches the correct answer
        if answer_id == correct_option_number:
            correct_answers += 1
        else:
            # Append question_id and answer_id to failed questions
            failed_questions.append((question_id, answer_id))

# Calculate the score
if total_questions > 0:
    score = (correct_answers / total_questions) * 100

print(f"{score:.1f}%")
