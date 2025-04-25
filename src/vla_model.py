
import torch
from torch import nn

class VLAModel:
    def __init__(self):
        self.action_net = nn.Sequential(
            nn.Linear(768 + 512 + 7, 256),  # BERT embedding + CLIP embedding + robot state
            nn.ReLU(),
            nn.Linear(256, 8)  # 7 joints + gripper
        )
        self.flow_steps = 10
        self.delta = 0.1
        self.action_map = {
            "push": lambda pos: [pos[0], pos[1], pos[2], 0, 0, 0, 0, 0],  # Lateral motion
            "place": lambda pos: [pos[0], pos[1], pos[2]+0.1, 0, 0, 0, 0, 1],  # Grasp and release
            "lift": lambda pos: [pos[0], pos[1], pos[2]+0.2, 0, 0, 0, 0, 1],  # Raise object
        }

    def flow_matching(self, input_tensor):
        actions = []
        noise = torch.randn(8)
        current = noise
        for _ in range(self.flow_steps):
            pred = self.action_net(input_tensor)
            current = current + self.delta * pred
            actions.append(current)
        return actions[-1]

    def predict_action(self, text_embedding, image_embedding, robot_state, action, object_position):
        combined_input = torch.cat([
            text_embedding,
            image_embedding,
            torch.tensor(robot_state, dtype=torch.float32)
        ])
        base_action = self.flow_matching(combined_input)
        mapped_action = self.action_map.get(action, lambda pos: base_action.numpy())(object_position)
        return torch.tensor(mapped_action, dtype=torch.float32)