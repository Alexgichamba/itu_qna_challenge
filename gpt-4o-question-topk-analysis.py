import json
import re
from openai import OpenAI
from tqdm.auto import tqdm
import time

openai_api = "YOUR_API_KEY"
client = OpenAI(api_key=openai_api)
context_file = '../data/Retrieval Results.json'
questions_file = '../data/366qs.txt'

def load_questions(file):
    with open(file, 'r') as f:
        questions = json.load(f)
    return questions

def load_context(file):
    with open(file, 'r') as f:
        context = json.load(f)
    return context



def find_appearing_abbreviations(question):
    """
    Identifies abbreviations that appear in a given question and its options.

    This function reads a list of abbreviations from a file, checks which of these
    abbreviations appear in the provided question and its options, and returns a list
    of dictionaries containing the abbreviations and their full forms.

    Args:
        question (dict): A dictionary containing the question and its options. The question
                         text should be under the key 'question', and options should be under
                         keys starting with 'option'.

    Returns:
        list of dict: A list of dictionaries where each dictionary has an abbreviation as the key
                      and its full form as the value.
    """
    with open("../data/abbreviations.txt", "r") as f_abbrevs:
        abbreviations = {}
        for line in f_abbrevs:
            # Split each line into abbreviation and full form
            abbreviation, full_form = line.split(": ", 1)
            # Store in the dictionary
            abbreviations[abbreviation.strip()] = full_form.strip()
    # Sort abbreviations by length in descending order to handle cases where one abbreviation is a substring of another
    sorted_abbrevs = sorted(
        abbreviations.items(), key=lambda x: len(x[0]), reverse=True
    )
    assert isinstance(question, dict)
    appearing_abbreviations = set()
    # Check for each abbreviation in the question and its options
    for abbreviation, full_form in sorted_abbrevs:
        pattern = r"\b" + re.escape(abbreviation) + r"\b"
        # Check if the abbreviation appears in the question text
        if re.search(pattern, (question["question"].split("?")[0])):
            appearing_abbreviations.add(abbreviation)
        # Check if the abbreviation appears in any of the options
        for key in question:
            if key.startswith("option") and question[key] is not None:
                if re.search(pattern, question[key]):
                    appearing_abbreviations.add(abbreviation)
    # Return a list of dictionaries with the abbreviation and its full form
    returned_abbreviations = [
        {abbrev: abbreviations[abbrev]} for abbrev in appearing_abbreviations
    ]
    return returned_abbreviations


def analyze_context(question, options, answer, explanation, chunk, abbreviations):
    #options_text = "\n".join([f"{i+1}. {opt[1]}" for i, opt in enumerate(options)])
    abbreviations_text = "\n".join(
        [
            f"{list(abbrev.keys())[0]}: {list(abbrev.values())[0]}"
            for abbrev in abbreviations
        ]
    )
    
    prompt = f"""
Question: {question}
Correct Answer: {answer}
Explanation: {explanation}
Abbreviations Full Forms: {abbreviations_text}

Context:
{chunk['doc_text']}

Task: Determine if the given context helps answer the question correctly.
Instructions:
1. Consider the question, correct answer, and explanation.
2. Analyze whether the context provides information that supports the correct answer.
3. Respond with only 1 if the context helps, or 0 if it doesn't help.

Your response should be a single digit: 0 or 1.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant analyzing whether given context helps answer a multiple-choice question correctly."},
            {"role": "user", "content": prompt}
        ]
    )
    
    result = response.choices[0].message.content.strip()
    return result


def process_questions(questions, context,output_file,question_per_batch=10,sleep_time=120):
    results = {}
    idx = 0
    for q_id, q_data in tqdm(questions.items(),desc="Processing questions"):
        # Sleep for 30 seconds after every 20 questions to avoid rate limits
        if idx % 20 == 0:
            time.sleep(30)
            print('\nSleeping for 30 seconds\n')

        # Clean the question text    
        question = re.sub(r"\s*\[.*?\]\s*$", "", q_data["question"])
        options = [(k, v) for k, v in q_data.items() if k.startswith("option") and v is not None]
        abbreviations = find_appearing_abbreviations(q_data)
        answer = q_data['answer']
        explanation = q_data['explanation']
        # Check if context is available for the question
        if q_id not in context:
            print(f"Warning: No context found for {q_id}")
            continue
        
        retrieved_chunks = context[q_id]['retrieved_chunks']
        chunk_results = []
         # Analyze each chunk of context
        for chunk in retrieved_chunks:
            relevance_score = analyze_context(question, options, answer, explanation, chunk,abbreviations)
            chunk_results.append({
                'relevance': relevance_score,
                'rank': chunk['rank'],
                'doc_text': chunk['doc_text']
            })
        results[q_id] = {
            'question': question,
            'correct_answer': answer,
            'retrieved_chunks': chunk_results

        }

        # Save results to a JSON file after each question
        with open(output_file, 'w') as outfile:
            json.dump(results, outfile, indent=4)      

        idx += 1
        if idx % question_per_batch == 0:
            print(f"Processed {idx} questions.... Sleeping for {sleep_time} seconds")


    
    
    return results


def main():
        # load questions and context
    qtns = load_questions(questions_file)
    context = load_context(context_file)

    # output_file
    output_file = 'gpt-4o-answer-context-analysis.json'

    # process questions
    results = process_questions(qtns, context, output_file)

if __name__ == '__main__':
    main()