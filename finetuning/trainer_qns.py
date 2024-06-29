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
        prompt = f"Question: {example.question}\n"
        for i, option in enumerate(example.options, 1):
            prompt += f"Option {i}: {option}\n"
        prompt += "Answer: "
        
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

def main():
    args = parse_args()
    model_name = "microsoft/phi-2"
    train_file = "data/qs_train.txt"
    dev_file = "data/qs_dev.txt"
    output_dir = "./save_phi2_ft_lora"
    hf_repo_name = "alexgichamba/phi-2-finetuned-qa-lora-r8"

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              padding_side="left",
                                              add_eos_token=True,
                                              add_bos_token=True,
                                              use_fast=False
                                              )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)

    train_dataset = QuestionAnsweringDataset(train_file, tokenizer)
    dev_dataset = QuestionAnsweringDataset(dev_file, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1400,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="wandb",
    )

    # Use DataCollatorForLanguageModeling for dynamic padding
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=default_data_collator,
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