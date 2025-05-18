import os
import numpy as np
import cv2
from ultralytics import YOLO
import torch

class YOLODetector:
    """
    YOLOv8 object detector class
    """
    def __init__(self, model_path=None, conf_threshold=0.5, classes=None):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_path: Path to YOLOv8 model file (.pt)
            conf_threshold: Confidence threshold for detections
            classes: List of class IDs to detect (None for all classes)
        """
        if model_path is None:
            model_path = os.path.join("app", "models", "yolo", "yolov8n.pt")
        
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.classes = classes
        
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load YOLOv8 model
        try:
            # Temporarily patch ultralytics torch_safe_load to use weights_only=False
            from ultralytics.nn.tasks import torch_safe_load
            
            # Save the original function
            original_torch_safe_load = torch_safe_load
            
            # Create a patched version
            def patched_torch_safe_load(file):
                return torch.load(file, map_location='cpu', weights_only=False), file
            
            # Apply the patch
            from ultralytics.nn import tasks
            tasks.torch_safe_load = patched_torch_safe_load
            
            # Load the model
            self.model = YOLO(self.model_path)
            print(f"YOLOv8 model loaded from {self.model_path}")
            
            # Restore the original function
            tasks.torch_safe_load = original_torch_safe_load
            
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            raise
    
    def detect(self, frame):
        """
        Detect objects in a frame
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            List of detections in format: [class_id, confidence, x1, y1, x2, y2]
        """
        try:
            # Run inference
            results = self.model(frame, verbose=False)[0]
            
            # Process results
            detections = []
            
            for result in results.boxes.data.cpu().numpy():
                x1, y1, x2, y2, confidence, class_id = result
                
                # Apply confidence threshold
                if confidence < self.conf_threshold:
                    continue
                
                # Filter classes if specified
                if self.classes is not None and int(class_id) not in self.classes:
                    continue
                
                # Add detection
                detections.append({
                    'class_id': int(class_id),
                    'confidence': float(confidence),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
            
            return detections
        
        except Exception as e:
            print(f"Error in YOLOv8 detection: {e}")
            return []
    
    def detect_with_classes(self, frame):
        """
        Detect objects with class names
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            List of detections with class names
        """
        detections = self.detect(frame)
        
        # Get class names from the model
        class_names = self.model.names
        
        for detection in detections:
            class_id = detection['class_id']
            detection['class_name'] = class_names.get(class_id, f"Class {class_id}")
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw detections on a frame
        
        Args:
            frame: OpenCV BGR image
            detections: List of detections from detect() method
            
        Returns:
            Frame with detections drawn
        """
        # Make a copy of the frame
        output_frame = frame.copy()
        
        # Get class names
        class_names = self.model.names
        
        # Draw each detection
        for detection in detections:
            class_id = detection['class_id']
            confidence = detection['confidence']
            x1, y1, x2, y2 = map(int, detection['bbox'])
            
            # Get class name
            class_name = class_names.get(class_id, f"Class {class_id}")
            
            # Create label
            label = f"{class_name}: {confidence:.2f}"
            
            # Generate a color based on class_id
            color = (int(hash(class_name) & 0xFF), 
                     int(hash(class_name + '1') & 0xFF), 
                     int(hash(class_name + '2') & 0xFF))
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(output_frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(output_frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return output_frame 