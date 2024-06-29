# phi2_inference.py

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, PeftModel
from huggingface_hub import HfApi
import argparse
import csv
from tqdm import tqdm

def load_model(model_name, local_checkpoint=None, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              padding_side="left",
                                              add_eos_token=True,
                                              add_bos_token=True,
                                              use_fast=False
                                              )
    tokenizer.pad_token = tokenizer.eos_token

    if local_checkpoint:
        model = AutoModelForCausalLM.from_pretrained(local_checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    if adapter_path:
        # Load the PEFT model directly
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        # If no adapter, apply the LoRA config
        peft_config = LoraConfig(
                                r=32,
                                lora_alpha=64,
                                target_modules=[
                                                "Wqkv",
                                                "fc1",
                                                "fc2",],
                                bias="none",
                                lora_dropout=0.05,
                                task_type="CAUSAL_LM",
                                )
        model = get_peft_model(model, peft_config)

    return model, tokenizer

def generate_answer(question, options, model, tokenizer, max_length=512, max_new_tokens=20):
    input_text = f"Instruct: {question}\n"
    for i, option in enumerate(options, 1):
        input_text += f"Option {i}: {option}\n"
    input_text += "Output: "

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
        )

    generated_answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    print(input_text)
    print(generated_answer)

    return generated_answer

def process_file(input_file, model, tokenizer):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    results = []
    
    for question_id, question_data in tqdm(data.items(), desc="Processing questions"):
        question_id = question_id.split(" ")[1]
        question = question_data['question']
        option_keys = sorted(key for key in question_data if key.startswith("option "))
        options = [question_data[key] for key in option_keys]
        
        generated_answer = generate_answer(question, options, model, tokenizer)
        
        # Extract the option number from the generated answer
        match = re.search(r'option (\d)', generated_answer.lower())
        if match:
            answer_id = int(match.group(1))
            print(f"Answer ID: {answer_id}")
        else:
            print(f"ERROR: Could not extract option number from the generated answer for question {question_id}")
            answer_id = 0  # Indicates failure to extract an answer
        
        results.append((question_id, answer_id, "Phi-2"))
    
    return results

def save_results(results, output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Question_ID", "Answer_ID", "Task"])
        for row in results:
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description="Inference script for question answering.")
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2", help="Hugging Face model name")
    parser.add_argument("--local_checkpoint", type=str, help="Path to local checkpoint if available")
    parser.add_argument("--adapter_path", type=str, help="Path to the adapter checkpoint")
    parser.add_argument("--input_file", type=str, default="data/qs_eval.txt", help="Path to the input questions file")
    parser.add_argument("--output_file", type=str, default="preds.csv", help="Path to save the predictions")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name, args.local_checkpoint, args.adapter_path)
    model.eval()

    results = process_file(args.input_file, model, tokenizer)
    save_results(results, args.output_file)

if __name__ == "__main__":
    main()
