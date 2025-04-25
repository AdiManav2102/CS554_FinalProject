# fine_tune_bert.py
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForTokenClassification

class InstructionDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tag2id = {"B-ACTION": 0, "B-OBJECT": 1, "I-OBJECT": 2, "O": 3}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instruction = self.data.iloc[idx]["instruction"]
        bio_tags = self.data.iloc[idx]["bio_tags"].split()
        encoding = self.tokenizer(instruction, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
        labels = [self.tag2id[tag] for tag in bio_tags] + [self.tag2id["O"]] * (32 - len(bio_tags))
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def fine_tune_bert():
    dataset = InstructionDataset("snli_instructions_train.csv")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    for epoch in range(3):
        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} completed")

    model.save_pretrained("bert_finetuned")

if __name__ == "__main__":
    fine_tune_bert()