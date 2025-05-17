import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from collections import deque

# Note: Deep SORT implementation simplified for clarity
# In a production system, you would use a full Deep SORT implementation
# This is a simplified version for demonstration purposes

class DeepSORTTracker:
    """
    DeepSORT tracker for multi-object tracking
    """
    def __init__(self, model_path=None, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize DeepSORT tracker
        
        Args:
            model_path: Path to DeepSORT model file
            max_age: Maximum number of frames to keep track
            min_hits: Minimum number of hits to start tracking
            iou_threshold: IOU threshold for track association
        """
        if model_path is None:
            model_path = os.path.join("app", "models", "deep_sort", "ckpt.t7")
        
        self.model_path = model_path
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        # Initialize track ID counter
        self.next_id = 0
        
        # Dictionary to store tracks
        # track = {
        #     'id': track_id,
        #     'bbox': bbox,  # [x1, y1, x2, y2]
        #     'class_id': class_id,
        #     'hits': hits,
        #     'age': age,
        #     'confidence': confidence,
        #     'time_since_update': time_since_update
        # }
        self.tracks = []
        
        # Load feature extractor model (simplified)
        self.load_model()
    
    def load_model(self):
        """
        Load DeepSORT feature extractor model
        """
        try:
            # In a real implementation, you would load the actual DeepSORT model
            # For simplicity, we're skipping the actual model loading
            print(f"DeepSORT model would be loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading DeepSORT model: {e}")
            # Continue without model - will use IoU matching only
    
    def calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union between two bounding boxes
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            float: IoU value
        """
        # Determine the coordinates of the intersection rectangle
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        # Calculate area of intersection rectangle
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of both bounding boxes
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # Calculate Union area
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def update(self, frame, detections):
        """
        Update tracker with new detections
        
        Args:
            frame: Current frame
            detections: List of detections from detector
            
        Returns:
            List of tracks: [(id, bbox, class_id, confidence), ...]
        """
        # Increment age of all tracks
        for track in self.tracks:
            track['age'] += 1
            track['time_since_update'] += 1
        
        # Create association matrix between tracks and detections
        if not detections:
            # No detections, return updated tracks
            updated_tracks = []
            for track in self.tracks:
                if track['time_since_update'] <= self.max_age:
                    updated_tracks.append((
                        track['id'],
                        track['bbox'],
                        track['class_id'],
                        track['confidence']
                    ))
            return updated_tracks
        
        # Calculate IoU between each track and detection
        association_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                bbox1 = track['bbox']
                bbox2 = detection['bbox']
                
                # Calculate IoU
                iou = self.calculate_iou(bbox1, bbox2)
                
                # Only consider matches with the same class (person or object)
                # For simplicity, we consider class 0 as person and all others as objects
                is_same_class_type = (track['class_id'] == 0 and detection['class_id'] == 0) or \
                                     (track['class_id'] > 0 and detection['class_id'] > 0)
                
                if is_same_class_type:
                    association_matrix[i, j] = iou
        
        # Find matches with IoU above threshold
        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))
        
        # Simple greedy matching (in a real implementation, this would use Hungarian algorithm)
        while association_matrix.size > 0 and association_matrix.max() >= self.iou_threshold:
            # Get indices of max value
            a, b = np.unravel_index(association_matrix.argmax(), association_matrix.shape)
            
            # Add match
            matches.append((a, b))
            
            # Remove matched indices
            unmatched_tracks.remove(a)
            unmatched_detections.remove(b)
            
            # Set matched values to zero to find next max
            association_matrix[a, :] = 0
            association_matrix[:, b] = 0
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx]['bbox'] = detections[det_idx]['bbox']
            self.tracks[track_idx]['confidence'] = detections[det_idx]['confidence']
            self.tracks[track_idx]['hits'] += 1
            self.tracks[track_idx]['time_since_update'] = 0
            self.tracks[track_idx]['class_id'] = detections[det_idx]['class_id']
        
        # Initialize new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self.add_track(
                bbox=detections[det_idx]['bbox'],
                class_id=detections[det_idx]['class_id'],
                confidence=detections[det_idx]['confidence']
            )
        
        # Remove dead tracks
        self.tracks = [track for track in self.tracks if track['time_since_update'] <= self.max_age]
        
        # Return active tracks
        active_tracks = []
        for track in self.tracks:
            if track['hits'] >= self.min_hits or track['time_since_update'] == 0:
                active_tracks.append((
                    track['id'],
                    track['bbox'],
                    track['class_id'],
                    track['confidence']
                ))
        
        return active_tracks
    
    def add_track(self, bbox, class_id, confidence):
        """
        Add a new track
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            class_id: Class ID of the object
            confidence: Detection confidence
        """
        self.tracks.append({
            'id': self.next_id,
            'bbox': bbox,
            'class_id': class_id,
            'hits': 1,
            'age': 1,
            'confidence': confidence,
            'time_since_update': 0
        })
        
        self.next_id += 1
    
    def draw_tracks(self, frame, tracks):
        """
        Draw tracks on a frame
        
        Args:
            frame: OpenCV BGR image
            tracks: List of tracks from update() method
            
        Returns:
            Frame with tracks drawn
        """
        # Make a copy of the frame
        output_frame = frame.copy()
        
        # Draw each track
        for track_id, bbox, class_id, confidence in tracks:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Generate color based on track ID
            color = ((track_id * 57) % 256, (track_id * 91) % 256, (track_id * 37) % 256)
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            cv2.putText(output_frame, f"ID: {track_id}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output_frame 