import cv2
import numpy as np
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from app.components.detection.detectors.base_detector import BaseDetector
from app.utils.theft_detector import TheftDetector as TheftDetectorUtil
from app.utils.video_utils import get_frame_thumbnail

class TheftDetector(BaseDetector):
    """
    Detector for theft-related events.
    
    This detector integrates with the TheftDetector utility to identify:
    - Potential theft events
    - Suspicious movements
    - Unauthorized access to restricted areas
    - Quick grabs of items
    """
    
    def __init__(self):
        super().__init__(name="TheftDetector")
        self.detector = None
        self.initialized = False
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Event type mapping with more detailed event types
        self.event_type_mapping = {
            "quick_grab": "Suspicious Item Grab",
            "unauthorized_zone": "Unauthorized Access",
            "unusual_dwell_time": "Long Dwell Time",
            "suspicious_movement": "Suspicious Movement",
            "object_taken": "Item Taken",
            "suspicious_object_movement": "Potential Theft",
            "abandoned_object": "Unattended Item",
            "movement_pattern": "Suspicious Movement",
            "potential_theft": "Potential Theft",
            "theft_in_progress": "Theft In Progress"
        }
        
        # Confidence threshold for generating events
        self.confidence_threshold = 0.65
        
        # Track already detected events to prevent duplicates
        self.recent_events = {}  # Format: {event_key: timestamp}
        self.event_cooldown = 5.0  # Seconds between similar events
        
        # Cache of recent object positions for better trend analysis
        self.object_position_cache = {}  # {object_id: [(position, timestamp), ...]}
        self.max_cache_size = 30
    
    def set_roi_areas(self, roi_areas: Dict, roi_colors: Dict):
        """Set the ROI areas and initialize the theft detector."""
        super().set_roi_areas(roi_areas, roi_colors)
        
        # Define regions of interest for the theft detector
        cash_counter_roi = None
        entry_zone = None
        exit_zone = None
        customer_zone = None
        cashier_zone = None
        
        # Look for specific ROI types in the roi_areas
        for roi_name, (x1, y1, x2, y2) in roi_areas.items():
            roi_name_lower = roi_name.lower()
            h, w = 1080, 1920  # Assuming standard resolution, will be adjusted in processing
            
            # Convert to normalized coordinates (0-1)
            x1_norm, y1_norm = x1/w, y1/h
            x2_norm, y2_norm = x2/w, y2/h
            
            # Create polygon for the ROI
            roi_polygon = np.array([
                [x1_norm, y1_norm],
                [x2_norm, y1_norm],
                [x2_norm, y2_norm],
                [x1_norm, y2_norm]
            ])
            
            # Assign to appropriate zone based on name (more comprehensive matching)
            if any(keyword in roi_name_lower for keyword in ["cash", "counter", "register", "checkout", "payment"]):
                cash_counter_roi = roi_polygon
                self.logger.info(f"Detected cash counter ROI: {roi_name}")
            elif any(keyword in roi_name_lower for keyword in ["entry", "entrance", "door", "in"]):
                entry_zone = roi_polygon
                self.logger.info(f"Detected entry zone: {roi_name}")
            elif any(keyword in roi_name_lower for keyword in ["exit", "out"]):
                exit_zone = roi_polygon
                self.logger.info(f"Detected exit zone: {roi_name}")
            elif any(keyword in roi_name_lower for keyword in ["customer", "shopping", "public"]):
                customer_zone = roi_polygon
                self.logger.info(f"Detected customer zone: {roi_name}")
            elif any(keyword in roi_name_lower for keyword in ["cashier", "employee", "staff", "private", "restricted"]):
                cashier_zone = roi_polygon
                self.logger.info(f"Detected cashier/restricted zone: {roi_name}")
        
        # Use first ROI as fallback if no cash counter ROI is found
        if cash_counter_roi is None and roi_areas:
            first_roi_name = next(iter(roi_areas))
            x1, y1, x2, y2 = roi_areas[first_roi_name]
            x1_norm, y1_norm = x1/w, y1/h
            x2_norm, y2_norm = x2/w, y2/h
            cash_counter_roi = np.array([
                [x1_norm, y1_norm],
                [x2_norm, y1_norm],
                [x2_norm, y2_norm],
                [x1_norm, y2_norm]
            ])
            self.logger.info(f"Using {first_roi_name} as default cash counter ROI (no specific counter ROI found)")
        
        # Initialize theft detector with optimized parameters
        try:
            self.detector = TheftDetectorUtil(
                cash_counter_roi=cash_counter_roi,
                dwell_time_threshold=12.0,  # Reduced from 15s to 12s for faster alert
                quick_grab_threshold=1.2,   # Reduced from 1.5s for more sensitive detection
                suspicious_movement_threshold=0.55,  # More sensitive detection threshold
                data_dir="data/theft",
                entry_zone=entry_zone,
                exit_zone=exit_zone,
                customer_zone=customer_zone,
                cashier_zone=cashier_zone
            )
            self.initialized = True
            self.logger.info("Theft detector initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize theft detector: {str(e)}")
            self.initialized = False
    
    def process_frame(self, frame, tracks, timestamp, frame_count) -> List[Dict[str, Any]]:
        """Process a frame for theft-related events."""
        if not self.initialized or self.detector is None:
            return []
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        events = []
        timestamp_float = time.time()  # Current time in seconds
        
        # Process each tracked object
        for track_id, bbox, class_id, confidence in tracks:
            x1, y1, x2, y2 = map(int, bbox)
            center_x, center_y = (x1 + x2) / (2 * w), (y1 + y2) / (2 * h)
            
            # Add to position cache
            if class_id > 0:  # Only cache non-person objects
                obj_key = f"class{class_id}_{track_id}"
                if obj_key not in self.object_position_cache:
                    self.object_position_cache[obj_key] = []
                
                # Add to cache
                self.object_position_cache[obj_key].append(((center_x, center_y), timestamp_float))
                
                # Trim cache if needed
                if len(self.object_position_cache[obj_key]) > self.max_cache_size:
                    self.object_position_cache[obj_key].pop(0)
            
            # Process person detections
            if class_id == 0:  # Person class
                alert = self.detector.update_person_position(
                    track_id, 
                    (center_x, center_y), 
                    timestamp_float
                )
                
                if alert:
                    # Check if we should create an event (based on confidence and cooldown)
                    event_key = f"person_{track_id}_{alert.get('alert_type', 'unknown')}"
                    last_event_time = self.recent_events.get(event_key, 0)
                    
                    if (timestamp_float - last_event_time > self.event_cooldown and 
                            alert.get('confidence', 0) >= self.confidence_threshold):
                        
                        # Create event from alert
                        event_data = self._create_event_from_alert(
                            alert, track_id, frame, timestamp, frame_count
                        )
                        events.append(event_data)
                        
                        # Update recent events tracker
                        self.recent_events[event_key] = timestamp_float
            
            # Process object detections (items that can be stolen)
            elif class_id > 0:
                # Map class IDs to names
                object_names = {
                    1: "cart",
                    2: "bag", 
                    3: "product",
                    4: "phone",
                    5: "wallet",
                    6: "item",
                    # Add more mappings based on your model
                }
                
                object_class = object_names.get(class_id, f"item-{class_id}")
                
                alert = self.detector.update_object_position(
                    track_id,
                    object_class,
                    (center_x, center_y),
                    timestamp_float
                )
                
                if alert:
                    # Check confidence and cooldown for this object event
                    event_key = f"object_{track_id}_{alert.get('alert_type', 'unknown')}"
                    last_event_time = self.recent_events.get(event_key, 0)
                    
                    if (timestamp_float - last_event_time > self.event_cooldown and 
                            alert.get('confidence', 0) >= self.confidence_threshold):
                        
                        # Create event from alert
                        event_data = self._create_event_from_alert(
                            alert, track_id, frame, timestamp, frame_count
                        )
                        # Add object class information
                        event_data['object_class'] = object_class
                        events.append(event_data)
                        
                        # Update recent events tracker
                        self.recent_events[event_key] = timestamp_float
        
        # Check for unusually fast object movement (potential grabbing)
        for obj_key, positions in self.object_position_cache.items():
            if len(positions) < 5:  # Need enough history
                continue
                
            # Calculate recent velocity
            prev_pos, prev_time = positions[-5]
            curr_pos, curr_time = positions[-1]
            
            if curr_time - prev_time <= 0:
                continue
                
            distance = ((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)**0.5
            velocity = distance / (curr_time - prev_time)
            
            # Check for unusually fast movement
            if velocity > 0.5:  # Threshold for unusually quick movement
                # Extract object info from key
                try:
                    _, track_id_str = obj_key.split('_')
                    track_id = int(track_id_str)
                    
                    # Create potential theft event
                    event_key = f"fast_object_{track_id}"
                    last_event_time = self.recent_events.get(event_key, 0)
                    
                    if timestamp_float - last_event_time > self.event_cooldown:
                        # Find closest person
                        closest_person = None
                        for person_track in tracks:
                            if person_track[2] == 0:  # Person class
                                person_id = person_track[0]
                                person_bbox = person_track[1]
                                person_x = (person_bbox[0] + person_bbox[2]) / (2 * w)
                                person_y = (person_bbox[1] + person_bbox[3]) / (2 * h)
                                
                                # Check if person is close to object
                                if ((person_x - curr_pos[0])**2 + (person_y - curr_pos[1])**2)**0.5 < 0.2:
                                    closest_person = person_id
                                    break
                        
                        if closest_person is not None:
                            event_data = {
                                'type': "Potential Theft",
                                'timestamp': timestamp.strftime('%H:%M:%S'),
                                'description': f"Fast movement of object {track_id} near person {closest_person}",
                                'frame_idx': frame_count,
                                'track_id': closest_person,
                                'object_id': track_id,
                                'confidence': 0.7,
                                'alert_id': len(events) + 1,
                                'needs_attention': True,
                                'thumbnail': get_frame_thumbnail(frame)
                            }
                            events.append(event_data)
                            self.recent_events[event_key] = timestamp_float
                except (ValueError, IndexError):
                    pass  # Skip if can't parse object key
        
        # Clean up stale objects periodically
        if frame_count % 100 == 0:  # Every 100 frames
            self.detector.cleanup_stale_objects()
            
            # Clean up stale event records
            current_time = time.time()
            stale_keys = [k for k, v in self.recent_events.items() if current_time - v > 60.0]
            for key in stale_keys:
                del self.recent_events[key]
            
        return events
    
    def _create_event_from_alert(self, alert, track_id, frame, timestamp, frame_count):
        """Create an event dictionary from a theft detector alert."""
        alert_type = alert.get('alert_type', 'potential_theft')
        mapped_type = self.event_type_mapping.get(alert_type, "Potential Theft")
        
        # Create descriptive message based on alert type
        base_message = alert.get('message', "")
        
        # Create detailed description based on alert type
        if alert_type == "quick_grab":
            description = f"Person {track_id} grabbed an item unusually quickly"
            if len(base_message) > 0:
                description += f" - {base_message}"
        elif alert_type == "unauthorized_zone":
            description = f"Person {track_id} entered unauthorized/restricted area"
            if len(base_message) > 0:
                description += f" - {base_message}"
        elif alert_type == "unusual_dwell_time":
            description = f"Person {track_id} is lingering suspiciously in one area"
            if len(base_message) > 0:
                description += f" - {base_message}"
        elif alert_type == "suspicious_movement":
            description = f"Person {track_id} displaying suspicious movement patterns"
            if len(base_message) > 0:
                description += f" - {base_message}"
        elif alert_type == "object_taken":
            description = f"Attention: Item taken by person {track_id}"
            if len(base_message) > 0:
                description += f" - {base_message}"
        elif alert_type == "suspicious_object_movement":
            description = f"Unusual movement of object near person {track_id}"
            if len(base_message) > 0:
                description += f" - {base_message}"
        else:
            description = base_message if len(base_message) > 0 else f"Suspicious activity detected with ID {track_id}"
        
        event_data = {
            'type': mapped_type,
            'timestamp': timestamp.strftime('%H:%M:%S'),
            'description': description,
            'frame_idx': frame_count,
            'track_id': track_id,
            'confidence': alert.get('confidence', 0.8),
            'alert_id': alert.get('alert_id', 0),
            'needs_attention': True,
            'thumbnail': get_frame_thumbnail(frame)
        }
        
        return event_data
    
    def annotate_frame(self, frame):
        """Add theft detector annotations to the frame."""
        if not self.initialized or self.detector is None:
            return frame
        
        # Get original frame with zone annotations
        annotated_frame = self.detector.draw_zones(frame.copy())
        
        # Draw recent alerts if available
        recent_alerts = self.detector.get_recent_alerts(1)
        if recent_alerts:
            # Get most recent alert
            alert = recent_alerts[0]
            
            # Add the alert overlay regardless of timestamp
            # This ensures alerts are always visible when they exist
            annotated_frame = self.detector.display_alert_overlay(annotated_frame, alert)
            
            # Also generate an event if we don't have one for this alert yet
            # This helps ensure events are logged even if not captured during process_frame
            alert_id = alert.get('alert_id', 0)
            alert_key = f"alert_{alert_id}"
            
            if alert_key not in self.recent_events:
                # Create event from alert to be captured by event system
                timestamp = datetime.now()
                event_data = {
                    'type': self.event_type_mapping.get(alert.get('alert_type', 'unknown'), "Potential Theft"),
                    'timestamp': timestamp.strftime('%H:%M:%S'),
                    'description': alert.get('message', "Suspicious activity detected"),
                    'frame_idx': 0,  # Frame count not available here
                    'track_id': alert.get('track_id', -1),
                    'confidence': alert.get('confidence', 0.8),
                    'alert_id': alert.get('alert_id', 0),
                    'needs_attention': True,
                    'thumbnail': get_frame_thumbnail(frame)
                }
                
                # Record that we've processed this alert
                self.recent_events[alert_key] = time.time()
        
        return annotated_frame
    
    def get_recent_alerts(self, count=10):
        """Get the most recent theft alerts."""
        if not self.initialized or self.detector is None:
            return []
            
        return self.detector.get_recent_alerts(count)
    
    def get_confirmed_thefts(self, count=10):
        """Get confirmed theft events."""
        if not self.initialized or self.detector is None:
            return []
            
        return self.detector.get_confirmed_thefts(count) 