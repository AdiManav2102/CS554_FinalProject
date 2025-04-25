# evaluate.py
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForTokenClassification
from sklearn.metrics import precision_recall_fscore_support
from environment import CubeTableEnv
from nlp_processor import NLPProcessor
from vision_processor import VisionProcessor
from vla_model import VLAModel
import numpy as np

class InstructionDataset:
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
            "labels": torch.tensor(labels, dtype=torch.long),
            "instruction": instruction,
            "action": self.data.iloc[idx]["action"],
            "object": self.data.iloc[idx]["object"]
        }

def evaluate_nlp():
    dataset = InstructionDataset("snli_instructions_test.csv")
    dataloader = DataLoader(dataset, batch_size=8)
    model = BertForTokenClassification.from_pretrained("bert_finetuned", num_labels=4)
    model.eval()

    all_preds = []
    all_labels = []
    id2tag = {0: "B-ACTION", 1: "B-OBJECT", 2: "I-OBJECT", 3: "O"}

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            preds = torch.argmax(outputs.logits, dim=2)
            for pred, label in zip(preds, batch["labels"]):
                all_preds.extend(pred.tolist())
                all_labels.extend(label.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1, 2, 3])
    macro_f1 = precision_recall_fscore_support(all_labels, all_preds, average="macro")[2]
    
    print("NLP Evaluation (BERT Sequence Labeling):")
    for i, tag in id2tag.items():
        print(f"{tag} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

def evaluate_vla():
    env = CubeTableEnv()
    nlp_processor = NLPProcessor(model_path="bert_finetuned")
    vision_processor = VisionProcessor()
    vla_model = VLAModel()
    
    test_data = pd.read_csv("snli_instructions_test.csv")
    success_count = 0
    action_correct_count = 0
    total = len(test_data)

    for idx in range(total):
        instruction = test_data.iloc[idx]["instruction"]
        expected_action = test_data.iloc[idx]["action"]
        expected_obj = test_data.iloc[idx]["object"]

        action, obj = nlp_processor.process_instruction(instruction)
        if not action or not obj:
            continue

        obs, robot_state = env.get_observation()
        object_position, image_embedding = vision_processor.identify_object(obs, obj, env)
        text_embedding = nlp_processor.get_text_embedding(instruction)

        action_pred = vla_model.predict_action(text_embedding, image_embedding, robot_state, action, object_position)

        # Check action correctness
        expected_trajectory = vla_model.action_map.get(expected_action, lambda pos: [0]*8)(object_position)
        action_correct = np.allclose(action_pred.numpy(), expected_trajectory, atol=0.1)
        if action_correct:
            action_correct_count += 1

        # Execute action and check task success
        env.step(action_pred.numpy())
        if env.check_task_success(action, obj, expected_action, expected_obj):
            success_count += 1
        
        env.reset() 

    task_success_rate = success_count / total
    action_accuracy = action_correct_count / total
    print("\nVLA Evaluation:")
    print(f"Task Success Rate: {task_success_rate:.4f} ({success_count}/{total})")
    print(f"Action Accuracy: {action_accuracy:.4f} ({action_correct_count}/{total})")
    env.close()

if __name__ == "__main__":
    evaluate_nlp()
    evaluate_vla()