import threading
import queue
import time
from typing import Dict, List, Callable, Any
from datetime import datetime

class EventDetectorManager:
    """Manager class for various event detectors."""
    
    def __init__(self):
        # Detectors registry
        self.detectors = {}
        
        # Currently enabled detector ID (only one at a time)
        self.active_detector_id = None
        
        # ROI data
        self.roi_areas = {}
        self.roi_colors = {}
        
        # Event callback
        self.event_callback = None
        
        # Async processing setup
        self.processing_queue = queue.Queue()
        self.async_thread = None
        self.running = False
    
    @property
    def enabled_detectors(self):
        """Get list of enabled detector IDs (currently just the active one)."""
        if self.active_detector_id:
            return [self.active_detector_id]
        return []
    
    def register_detector(self, detector_id: str, detector):
        """Register a detector with the manager."""
        self.detectors[detector_id] = detector
        
        # If this is the first detector registered, set it as active
        if self.active_detector_id is None:
            self.active_detector_id = detector_id
        
        # Initialize detector with ROI data if available
        if self.roi_areas and hasattr(detector, 'set_roi_areas'):
            detector.set_roi_areas(self.roi_areas, self.roi_colors)
    
    def unregister_detector(self, detector_id: str):
        """Unregister a detector from the manager."""
        if detector_id in self.detectors:
            del self.detectors[detector_id]
            
            # If we removed the active detector, set a new one if possible
            if detector_id == self.active_detector_id and self.detectors:
                self.active_detector_id = next(iter(self.detectors.keys()))
            elif not self.detectors:
                self.active_detector_id = None
    
    def set_active_detector(self, detector_id: str):
        """Set which detector should be active (only one at a time)."""
        if detector_id in self.detectors:
            self.active_detector_id = detector_id
            return True
        return False
    
    def get_active_detector(self):
        """Get the currently active detector instance."""
        if self.active_detector_id and self.active_detector_id in self.detectors:
            return self.detectors[self.active_detector_id]
        return None
    
    def set_roi_areas(self, roi_areas: Dict, roi_colors: Dict):
        """Update ROI areas for all detectors."""
        self.roi_areas = roi_areas
        self.roi_colors = roi_colors
        
        # Update all detectors
        for detector in self.detectors.values():
            if hasattr(detector, 'set_roi_areas'):
                detector.set_roi_areas(roi_areas, roi_colors)
    
    def register_event_callback(self, callback: Callable):
        """Register a callback function to handle detected events."""
        self.event_callback = callback
    
    def process_frame(self, frame, tracks, timestamp, frame_count):
        """Process a video frame with the active detector."""
        events = []
        
        # Only use the active detector
        active_detector = self.get_active_detector()
        if active_detector and hasattr(active_detector, 'process_frame'):
            detector_events = active_detector.process_frame(frame, tracks, timestamp, frame_count)
            events.extend(detector_events)
        
        # If async processing is enabled, add events to the queue
        if self.running and events and self.event_callback:
            for event in events:
                self.processing_queue.put(event)
        
        return events
    
    def start_async_processing(self):
        """Start asynchronous event processing thread."""
        if self.async_thread is not None and self.running:
            return  # Already running
        
        self.running = True
        self.async_thread = threading.Thread(target=self._process_events_async)
        self.async_thread.daemon = True
        self.async_thread.start()
    
    def stop_async_processing(self):
        """Stop asynchronous event processing thread."""
        self.running = False
        if self.async_thread is not None:
            self.async_thread.join(timeout=1.0)
            self.async_thread = None
    
    def _process_events_async(self):
        """Worker function for asynchronous event processing."""
        while self.running:
            try:
                # Get an event with timeout to allow checking self.running
                try:
                    event = self.processing_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process the event
                if self.event_callback:
                    self.event_callback(event)
                
                # Mark task as done
                self.processing_queue.task_done()
            
            except Exception as e:
                print(f"Error in event processing thread: {e}")
                time.sleep(0.1)  # Prevent tight loop in case of repeated errors
    
    def get_available_detector_types(self):
        """Get list of available detector types."""
        return [
            {"id": "footfall_detector", "name": "Footfall Detector", "description": "Track people movement, entries, and exits"},
            {"id": "theft_detector", "name": "Theft Detector", "description": "Detect potential theft and suspicious activities"},
            {"id": "shelf_monitor_detector", "name": "Shelf Monitor", "description": "Monitor product placement and stock levels"},
            {"id": "unauthorized_area_detector", "name": "Unauthorized Area Detector", "description": "Detect people in restricted areas"}
        ] 