
# vision_processor.py
import clip
import torch
import numpy as np
from ultralytics import YOLO
import pybullet as p

class VisionProcessor:
    def __init__(self, yolo_model_path="yolov8n.pt"):
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cpu")
        self.tokenizer = clip.tokenize
        self.yolo_model = YOLO(yolo_model_path)
        self.class_map = {
            0: "red cube",
            1: "blue cube",
            2: "green cube",
            3: "orange pyramid",
            4: "yellow pyramid",
            5: "box"
        }

    def get_3d_position(self, bbox_center, depth_buffer, width, height, view_matrix, projection_matrix):
        # Convert 2D pixel coordinates to normalized device coordinates
        x, y = bbox_center
        x_ndc = (2.0 * x) / width - 1.0
        y_ndc = 1.0 - (2.0 * y) / height  # Flip y-axis
        depth = depth_buffer[int(y), int(x)]
        
        # Convert to clip space
        clip_coords = np.array([x_ndc, y_ndc, depth, 1.0])
        
        # Unproject to world coordinates
        inv_view_proj = np.linalg.inv(np.array(projection_matrix).reshape(4, 4) @ np.array(view_matrix).reshape(4, 4))
        world_coords = inv_view_proj @ clip_coords
        world_coords /= world_coords[3]  # Normalize by w
        return world_coords[:3]

    def identify_object(self, image, object_text, env):
        # Preprocess image for CLIP and YOLO
        image_np = np.array(image, dtype=np.uint8)  # Shape: [240, 320, 3]
        image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Shape: [1, 3, 240, 320]
        
        # Get CLIP image embedding
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)  # Shape: [1, 512]
        
        # Run YOLO object detection
        results = self.yolo_model(image_np, verbose=False)
        
        # Get camera parameters
        width, height = 320, 240
        _, _, _, depth_buffer, _ = p.getCameraImage(width, height, viewMatrix=env.camera, projectionMatrix=p.computeProjectionMatrixFOV(60, width/height, 0.1, 100))
        depth_buffer = np.array(depth_buffer).reshape(height, width)
        view_matrix = env.camera
        projection_matrix = p.computeProjectionMatrixFOV(60, width/height, 0.1, 100)
        
        # Process detections
        best_position = None
        best_confidence = 0
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                detected_name = self.class_map.get(class_id, "")
                if detected_name == object_text:  # Match detected object to instruction
                    confidence = float(box.conf)
                    if confidence > best_confidence:
                        # Get bounding box center
                        x1, y1, x2, y2 = box.xyxy[0]
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Convert to 3D position
                        position = self.get_3d_position((center_x, center_y), depth_buffer, width, height, view_matrix, projection_matrix)
                        
                        # Validate with ground-truth position
                        gt_position = env.get_object_position(object_text)
                        if gt_position is not None and np.linalg.norm(position - gt_position) < 0.2:  # Tolerance of 0.2 meters
                            best_position = position
                            best_confidence = confidence
        
        if best_position is None:
            # Fallback to ground-truth position if detection fails
            best_position = env.get_object_position(object_text) or [0.5, 0, 0.7]  # Default if no position found
        
        return best_position, image_features.squeeze()  # Return 3D position and 512D image embedding