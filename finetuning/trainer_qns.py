# train_qns.py
# This script trains the model on the training questions.
# The training questions are expected to be in the data/qs_train.txt file.
# The development questions are expected to be in the data/qs_dev.txt file.

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    default_data_collator,
)
from peft import get_peft_model, LoraConfig, TaskType
import wandb
from huggingface_hub import HfApi

import argparse

from compute_acc import compute_accuracy

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with optional resume and epoch settings.")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from the last checkpoint.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training.")
    return parser.parse_args()

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

# Initialize wandb
wandb.init(project="phi-2-finetuning")

@dataclass
class QuestionAnsweringExample:
    question: str
    options: List[str]
    answer: str
    explanation: str

class QuestionAnsweringDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self.load_examples(file_path)

    def load_examples(self, file_path: str) -> List[QuestionAnsweringExample]:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        examples = []
        for item in data.values():
            option_keys = sorted(key for key in item if key.startswith("option "))
            options = [item[key] for key in option_keys]
            examples.append(QuestionAnsweringExample(
                question=item["question"],
                options=options,
                answer=item["answer"],
                explanation=item["explanation"]
            ))
        return examples

    def __len__(self):
        return len(self.examples)

    def formatting_func(self, example):
        prompt = f"Instruct: {example.question}\n"
        for i, option in enumerate(example.options, 1):
            prompt += f"Option {i}: {option}\n"
        prompt += "Output: "
        
        target = f"{example.answer}\nExplanation: {example.explanation}"
        
        return prompt + target

    def __getitem__(self, idx):
        example = self.examples[idx]
        full_text = self.formatting_func(example)
        
        result = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = result["input_ids"].squeeze(0)
        attention_mask = result["attention_mask"].squeeze(0)
        
        # Find the position where the answer starts
        prompt = full_text.split("Answer:")[0] + "Answer:"
        answer_start = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        
        labels = input_ids.clone()
        labels[:answer_start] = -100  # We don't want to predict the prompt, only the answer and explanation
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

class CustomTrainer(Trainer):
    def __init__(self, dev_file=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_file = dev_file

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        accuracy = compute_accuracy(self.model, self.tokenizer, self.dev_file)
        self.log({"eval_accuracy": accuracy})
        logging.info(f"Validation Accuracy: {accuracy}")
        metrics = super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        metrics["eval_accuracy"] = accuracy
        
        return metrics

def main():
    args = parse_args()
    model_name = "microsoft/phi-2"
    train_file = "data/qs_train.txt"
    dev_file = "data/qs_dev.txt"
    output_dir = "./save_phi2_ft_lora"
    hf_repo_name = "alexgichamba/phi-2-finetuned-qa-lora-r32-a16"

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              padding_side="left",
                                              add_eos_token=True,
                                              add_bos_token=True,
                                              use_fast=False
                                              )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print(model)

    # Configure LoRA
    peft_config = LoraConfig(
                            r=32,
                            lora_alpha=64,
                            target_modules=[
                                            "Wqkv",
                                            "fc1",
                                            "fc2",],
                            bias="none",
                            lora_dropout=0.05,  # Conventional
                            task_type="CAUSAL_LM",
                            )
    model = get_peft_model(model, peft_config)

    train_dataset = QuestionAnsweringDataset(train_file, tokenizer)
    dev_dataset = QuestionAnsweringDataset(dev_file, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=20,
        load_best_model_at_end=True,
        report_to="wandb",
    )

    # Use DataCollatorForLanguageModeling for dynamic padding
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = CustomTrainer(
        dev_file=dev_file, # Custom argument
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save the final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Upload to Hugging Face
    api = HfApi()
    api.create_repo(repo_id=hf_repo_name, exist_ok=True)
    api.upload_folder(
        folder_path=output_dir,
        repo_id=hf_repo_name,
        repo_type="model",
    )

    logging.info(f"Model uploaded to Hugging Face: https://huggingface.co/{hf_repo_name}")

    wandb.finish()

if __name__ == "__main__":
    main()