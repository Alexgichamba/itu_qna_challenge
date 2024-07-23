import json
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, losses, InputExample, LoggingHandler

import random
from huggingface_hub import HfApi
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Load triplet data
with open('data/triplets_train.txt', 'r') as f_train_in:
    train_qns = json.load(f_train_in)

triplets = list(train_qns.values())

# Load pre-trained model
model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)

# Create custom dataset class
class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        item = self.triplets[idx]
        anchor = item['explanation']
        positive = item['correct'][0]
        # randomly select the incorrect answer
        incorrect_idx = random.randint(0, len(item['incorrect']) - 1)
        negative = item['incorrect'][incorrect_idx]
        return InputExample(texts=[anchor, positive, negative])
    
# Create DataLoader
train_dataset = TripletDataset(triplets)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)

# Define triplet loss
train_loss = losses.TripletLoss(model=model)

# Finetune model
num_epochs = 5
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) # 10% of train data
checpoint_save_steps = int(len(train_dataloader) * 0.2) # Save checkpoint every 20% of train data
output_path = "output/gte-large-en-v1.5-triplet-finetuned-for-telco-qa"

class PrintLossCallback:
    def __init__(self, steps=50):
        self.steps = steps

    def on_training_step(self, score, epoch, steps):
        if steps % self.steps == 0:
            logging.info(f"Epoch: {epoch}, Step: {steps}, Loss: {score:.4f}")

# In your model.fit() call, add this callback:
callback = PrintLossCallback(steps=50)  # Print every 100 steps

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=output_path,
    use_amp=True,
    evaluator=None,
    evaluation_steps=0,
    scheduler='WarmupLinear',
    optimizer_params={'lr': 2e-5},
    weight_decay=0.01,
    checkpoint_path=None,
    checkpoint_save_steps=checpoint_save_steps,
    checkpoint_save_total_limit=0,
    callback=callback  # Add this line
)

# Save and evaluate model
model = SentenceTransformer(output_path, trust_remote_code=True)


# upload to huggingface
model_id = "alexgichamba/gte-large-en-v1.5-triplet-finetuned-for-telco-qa"
api = HfApi()
# Create the repository
try:
    api.create_repo(repo_id=model_id)
    print(f"Repository {model_id} created successfully.")
except Exception as e:
    print(f"Repository creation failed or already exists: {e}")

# Upload the folder
try:
    api.upload_folder(
        folder_path=output_path,
        repo_id=model_id,
        ignore_patterns=["*.lock"]
    )
    print("Folder uploaded successfully.")
except Exception as e:
    print(f"Folder upload failed: {e}")
