# preprocess_snli.py
import pandas as pd
import spacy
from datasets import load_dataset
from sklearn.model_selection import train_test_split

nlp = spacy.load("en_core_web_sm")

def extract_action_object(sentence):
    doc = nlp(sentence)
    action = None
    obj = None
    for token in doc:
        if token.lemma_ in ["push", "move", "shift"]:
            action = "push"
        elif token.lemma_ in ["place", "put", "set"]:
            action = "place"
        elif token.lemma_ in ["lift", "raise"]:
            action = "lift"
        if token.text.lower() in ["cube", "pyramid", "box"]:
            obj = token.text.lower()
            for child in token.children:
                if child.dep_ == "amod" and child.text.lower() in ["red", "blue", "green", "orange", "yellow"]:
                    obj = f"{child.text.lower()} {obj}"
    return action, obj

def generate_bio_tags(instruction, action, obj):
    tokens = instruction.split()
    tags = ["O"] * len(tokens)
    if action:
        for i, token in enumerate(tokens):
            if token.lower() == action.lower():
                tags[i] = "B-ACTION"
                break
    if obj:
        obj_tokens = obj.split()
        for i in range(len(tokens) - len(obj_tokens) + 1):
            if " ".join(tokens[i:i+len(obj_tokens)]).lower() == obj.lower():
                tags[i] = "B-OBJECT"
                for j in range(1, len(obj_tokens)):
                    tags[i+j] = "I-OBJECT"
                break
    return tags

def process_snli():
    dataset = load_dataset("snli")["train"]
    instructions = []
    
    for example in dataset:
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        if any(keyword in premise.lower() + hypothesis.lower() for keyword in ["cube", "pyramid", "box", "red", "blue", "green", "orange", "yellow"]):
            for sentence in [premise, hypothesis]:
                action, obj = extract_action_object(sentence)
                if action and obj:
                    instruction = f"{action.capitalize()} the {obj}"
                    bio_tags = generate_bio_tags(instruction, action, obj)
                    instructions.append({
                        "instruction": instruction,
                        "action": action,
                        "object": obj,
                        "bio_tags": " ".join(bio_tags)
                    })
    
    # Fallback templated instructions
    fallback_instructions = [
        {"instruction": "Push the red cube", "action": "push", "object": "red cube", "bio_tags": "B-ACTION O B-OBJECT I-OBJECT"},
        {"instruction": "Place the orange pyramid", "action": "place", "object": "orange pyramid", "bio_tags": "B-ACTION O B-OBJECT I-OBJECT"},
        {"instruction": "Lift the blue cube", "action": "lift", "object": "blue cube", "bio_tags": "B-ACTION O B-OBJECT I-OBJECT"},
        {"instruction": "Move the yellow pyramid", "action": "push", "object": "yellow pyramid", "bio_tags": "B-ACTION O B-OBJECT I-OBJECT"},
        {"instruction": "Place in the box", "action": "place", "object": "box", "bio_tags": "B-ACTION O O B-OBJECT"}
    ]
    instructions.extend(fallback_instructions)
    
    # Split into train/test
    df = pd.DataFrame(instructions)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv("snli_instructions_train.csv", index=False)
    test_df.to_csv("snli_instructions_test.csv", index=False)
    print(f"Generated {len(train_df)} training and {len(test_df)} test instructions")

if __name__ == "__main__":
    process_snli()