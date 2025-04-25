# main.py
from environment import CubeTableEnv
from nlp_processor import NLPProcessor
from vision_processor import VisionProcessor
from vla_model import VLAModel
import torch
import numpy as np

def main():
    
    env = CubeTableEnv()
    nlp_processor = NLPProcessor(model_path="bert_finetuned")
    vision_processor = VisionProcessor()
    vla_model = VLAModel()

    print("Enter natural language instructions (type 'exit' to quit).")
    print("Examples: 'Push the red cube', 'Place the orange pyramid', 'Lift the blue cube'")

    while True:
        instruction = input("Enter instruction: ").strip()
        if instruction.lower() == "exit":
            print("Exiting program.")
            break

        print(f"Processing instruction: {instruction}")
        action, obj = nlp_processor.process_instruction(instruction)
        if not action or not obj:
            print(f"Failed to parse instruction: {instruction}")
            print("Please try a different command (e.g., 'Push the red cube').")
            continue

        obs, robot_state = env.get_observation()
        object_position, image_embedding = vision_processor.identify_object(obs, obj, env)
        text_embedding = nlp_processor.get_text_embedding(instruction)
        action_pred = vla_model.predict_action(text_embedding, image_embedding, robot_state, action, object_position)
        env.step(action_pred.numpy())
        
        success = env.check_task_success(action, obj, action, obj)
        print(f"Task executed: {action} {obj} (Success: {success})")
        env.reset()
        print("Environment reset. Ready for next instruction.")

    env.close()

if __name__ == "__main__":
    main()