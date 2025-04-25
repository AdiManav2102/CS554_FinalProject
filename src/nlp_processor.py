
from transformers import BertTokenizer, BertForTokenClassification
import torch

class NLPProcessor:
    def __init__(self, model_path="bert_finetuned"):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForTokenClassification.from_pretrained(model_path, num_labels=4)  # B-ACTION, B-OBJECT, I-OBJECT, O
        self.tag2id = {"B-ACTION": 0, "B-OBJECT": 1, "I-OBJECT": 2, "O": 3}
        self.id2tag = {v: k for k, v in self.tag2id.items()}

    def process_instruction(self, instruction):
        inputs = self.tokenizer(instruction, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
        with torch.no_grad():
            outputs = self.model(**inputs).logits
        predictions = torch.argmax(outputs, dim=2)[0]
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        action = None
        obj = []
        for token, pred, offset in zip(tokens, predictions, inputs["offset_mapping"][0]):
            tag = self.id2tag[pred.item()]
            if tag == "B-ACTION":
                action = token
            elif tag == "B-OBJECT":
                obj = [token]
            elif tag == "I-OBJECT" and obj:
                obj.append(token)
        
        action = action.replace("##", "") if action else None
        obj = " ".join(obj).replace("##", "") if obj else None
        return action, obj

    def get_text_embedding(self, instruction):
        inputs = self.tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model.bert(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.squeeze()