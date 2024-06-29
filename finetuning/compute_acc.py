import json
import re
import torch

def compute_accuracy(model, tokenizer, dev_file):
    model.eval()
    correct = 0
    total = 0
    
    with open(dev_file, 'r') as f:
        dev_data = json.load(f)
    
    tokenizer.pad_token = tokenizer.eos_token
    with torch.no_grad():
        for question_id, question_data in dev_data.items():
            question = question_data['question']
            option_keys = sorted(key for key in question_data if key.startswith("option "))
            options = [question_data[key] for key in option_keys]
            correct_answer = question_data['answer']
            
            # Prepare input
            input_text = f"Instruct: {question}\n"
            for i, option in enumerate(options, 1):
                input_text += f"Option {i}: {option}\n"
            input_text += "Output: "
            
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            
            # Generate answer
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
            )
            
            generated_answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            print(input_text)
            print(generated_answer)
            # Extract the option number from the generated answer
            match = re.search(r'option (\d)', generated_answer.lower())
            if match:
                generated_option = f"option {match.group(1)}"
                if generated_option in correct_answer.lower():
                    correct += 1
            else:
                print(f"ERROR: Could not extract option number from the generated answer for question {question_id}")
            
            total += 1
    
    accuracy = correct / total
    return accuracy