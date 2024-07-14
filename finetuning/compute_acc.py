import json
import re
import torch
from tqdm import tqdm
import sys
sys.path.append('data/')
from prepare_docs import find_appearing_abbreviations

def remove_tags(question):
    return (question.split('?')[0] + '?')

def compute_accuracy(model, tokenizer, dev_file):
    model.eval()
    correct = 0
    total = 0
    
    with open(dev_file, 'r') as f:
        dev_data = json.load(f)
    
    # Ensure the pad token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        for question_id, question_data in tqdm(dev_data.items(), desc="Computing accuracy"):
            question = remove_tags(question_data['question'])
            option_keys = sorted(key for key in question_data if key.startswith("option "))
            options = [question_data[key] for key in option_keys]
            correct_answer = question_data['answer']
            context = question_data['context']
            abbrevs_list = find_appearing_abbreviations(question_data)

            # Prepare input
            # prompt = f"Instruct: You will answer each question correctly by giving only the Option ID, the number that follows each Option.\n"
            # prompt += f"The output should be in the format: Option <Option id>\n"
            prompt = f"Instruct: Use the context below to correctly answer the following multiple choice question.\n\n"
            prompt += f"Context: {context}\n\n"
            abbreviations_text = "\n".join([f"{list(abbrev.keys())[0]}: {list(abbrev.values())[0]}" for abbrev in abbrevs_list])
            f"Abbreviations:\n{abbreviations_text}\n\n"
            prompt += f"Question: {question}\n"
            for i, option in enumerate(options, 1):
                prompt += f"Option {i}: {option}\n"
            prompt += "Answer: "
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
            # attention_mask = inputs['attention_mask']
            # Generate answer
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                # attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
            )
            
            generated_answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            print(prompt)
            print(generated_answer)
            # Extract the option number from the generated answer
            match = re.search(r'option (\d)', generated_answer.lower())
            if match:
                generated_option = f"option {match.group(1)}"
                if generated_option in correct_answer.lower():
                    correct += 1
            # if not, find the first number in the generated answer
            elif match := re.search(r'\d', generated_answer):
                generated_option = f"option {match.group()}"
                if generated_option in correct_answer.lower():
                    correct += 1
            else:
                print(f"ERROR: Could not extract option number from the generated answer for question {question_id}")
            
            total += 1
    
    accuracy = correct / total
    return accuracy