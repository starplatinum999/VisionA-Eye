from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

class BaseDetector:
    """
    Base class for all detector implementations.
    
    This provides the common interface and functionality that all
    specialized detectors should implement.
    """
    
    def __init__(self, name="BaseDetector"):
        """Initialize the detector with a name."""
        self.name = name
        self.roi_areas = {}
        self.roi_colors = {}
        self.enabled = True
    
    def set_roi_areas(self, roi_areas: Dict, roi_colors: Dict):
        """Set the Regions of Interest for detection."""
        self.roi_areas = roi_areas
        self.roi_colors = roi_colors
    
    def process_frame(self, frame, tracks, timestamp, frame_count):
        """
        Process a video frame and generate events.
        
        Args:
            frame: The video frame to process
            tracks: List of detected object tracks [(track_id, bbox, class_id, confidence), ...]
            timestamp: Current frame timestamp
            frame_count: Current frame count
            
        Returns:
            List of detected events
        """
        # Base implementation returns no events
        return []
    
    def annotate_frame(self, frame):
        """
        Add detector-specific annotations to the frame.
        
        Args:
            frame: The video frame to annotate
            
        Returns:
            Annotated frame
        """
        # Base implementation returns original frame
        return frame
    
    def enable(self):
        """Enable the detector."""
        self.enabled = True
    
    def disable(self):
        """Disable the detector."""
        self.enabled = False
    
    def is_enabled(self):
        """Check if the detector is enabled."""
        return self.enabled 