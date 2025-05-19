import cv2
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from app.components.detection.detectors.base_detector import BaseDetector
from app.utils.video_utils import get_frame_thumbnail

class UnauthorizedAreaDetector(BaseDetector):
    """
    Detector for identifying people in unauthorized or restricted areas.
    
    This detector monitors defined restricted zones and generates alerts
    when people enter these areas or perform suspicious activities.
    """
    
    def __init__(self):
        super().__init__(name="UnauthorizedAreaDetector")
        self.initialized = False
        
        # Event type mapping
        self.event_type_mapping = {
            "unauthorized_entry": "Unauthorized Access",
            "restricted_area": "Restricted Area Access",
            "staff_only": "Staff-Only Area Access",
            "loitering": "Loitering in Restricted Area"
        }
        
        # Zone data
        self.restricted_areas = {}  # name -> polygon
        self.authorized_tracks = set()  # IDs of tracks that are authorized
        
        # Track people in restricted areas
        self.people_in_restricted = {}  # track_id -> {'area': area_name, 'entry_time': timestamp, 'last_position': (x,y)}
        
        # Confidence threshold for generating events
        self.confidence_threshold = 0.75
        
        # Cooldown between similar events
        self.recent_events = {}  # Format: {event_key: timestamp}
        self.event_cooldown = 5.0  # Seconds between similar events
        
        # Define dwell time for loitering
        self.loitering_threshold = 10.0  # Seconds
    
    def set_roi_areas(self, roi_areas: Dict, roi_colors: Dict):
        """Set the ROI areas for unauthorized area detection."""
        super().set_roi_areas(roi_areas, roi_colors)
        
        # Reset restricted areas
        self.restricted_areas = {}
        
        # Identify restricted areas from ROIs
        for roi_name, (x1, y1, x2, y2) in roi_areas.items():
            roi_name_lower = roi_name.lower()
            
            # Create polygon for the ROI
            roi_polygon = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], np.int32)
            
            # Check if this is a restricted area based on name
            if ("restricted" in roi_name_lower or 
                "secure" in roi_name_lower or 
                "staff" in roi_name_lower or
                "private" in roi_name_lower or
                "authorized" in roi_name_lower):
                self.restricted_areas[roi_name] = roi_polygon
        
        # If no restricted areas were defined, create a default one
        if not self.restricted_areas:
            h, w = 1080, 1920  # Default resolution, will be adjusted in processing
            
            # Default restricted area in top right corner (20% of the frame)
            restricted_polygon = np.array([
                [int(0.8 * w), 0],
                [w, 0],
                [w, int(0.2 * h)],
                [int(0.8 * w), int(0.2 * h)]
            ], np.int32)
            
            self.restricted_areas["Default Restricted Area"] = restricted_polygon
        
        self.initialized = True
    
    def process_frame(self, frame, tracks, timestamp, frame_count):
        """Process the frame and detect unauthorized access events."""
        if not self.initialized:
            return []
            
        events = []
        h, w = frame.shape[:2]
        timestamp_float = time.mktime(timestamp.timetuple()) + timestamp.microsecond / 1000000.0
        
        # Check for people in the frame
        for track_id, bbox, class_id, confidence in tracks:
            if class_id != 0:  # We only care about people
                continue
                
            # Get center of the bounding box
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            point = (int(center_x), int(center_y))
            
            # Check if person is in any restricted area
            in_restricted_area = False
            current_area = None
            
            for area_name, area_polygon in self.restricted_areas.items():
                if cv2.pointPolygonTest(area_polygon, point, False) >= 0:
                    in_restricted_area = True
                    current_area = area_name
                    break
            
            # Handle person in restricted area
            if in_restricted_area:
                # Check if already tracking this person in this area
                if track_id in self.people_in_restricted:
                    entry_time = self.people_in_restricted[track_id]['entry_time']
                    area_name = self.people_in_restricted[track_id]['area']
                    
                    # Update position
                    self.people_in_restricted[track_id]['last_position'] = point
                    
                    # Check for loitering
                    dwell_time = timestamp_float - entry_time
                    if dwell_time > self.loitering_threshold:
                        # Generate loitering event
                        event_key = f"loitering_{track_id}_{area_name}"
                        if timestamp_float - self.recent_events.get(event_key, 0) > self.event_cooldown:
                            event = {
                                'type': 'Loitering in Restricted Area',
                                'timestamp': timestamp.strftime('%H:%M:%S'),
                                'description': f"Person {track_id} has been in {area_name} for {int(dwell_time)} seconds",
                                'frame_idx': frame_count,
                                'track_id': track_id,
                                'confidence': 0.85,
                                'thumbnail': get_frame_thumbnail(frame),
                                'needs_attention': True
                            }
                            events.append(event)
                            self.recent_events[event_key] = timestamp_float
                
                else:
                    # First time seeing this person in the restricted area
                    self.people_in_restricted[track_id] = {
                        'area': current_area,
                        'entry_time': timestamp_float,
                        'last_position': point
                    }
                    
                    # Generate unauthorized entry event
                    event_key = f"unauthorized_{track_id}_{current_area}"
                    if timestamp_float - self.recent_events.get(event_key, 0) > self.event_cooldown:
                        event = {
                            'type': 'Unauthorized Access',
                            'timestamp': timestamp.strftime('%H:%M:%S'),
                            'description': f"Person {track_id} entered restricted area: {current_area}",
                            'frame_idx': frame_count,
                            'track_id': track_id,
                            'confidence': 0.9,
                            'thumbnail': get_frame_thumbnail(frame),
                            'needs_attention': True
                        }
                        events.append(event)
                        self.recent_events[event_key] = timestamp_float
            
            else:
                # Person not in restricted area - check if they were previously there
                if track_id in self.people_in_restricted:
                    area_name = self.people_in_restricted[track_id]['area']
                    time_spent = timestamp_float - self.people_in_restricted[track_id]['entry_time']
                    
                    # Generate exit event
                    event_key = f"exit_restricted_{track_id}_{area_name}"
                    if timestamp_float - self.recent_events.get(event_key, 0) > self.event_cooldown:
                        event = {
                            'type': 'Exited Restricted Area',
                            'timestamp': timestamp.strftime('%H:%M:%S'),
                            'description': f"Person {track_id} exited {area_name} after {int(time_spent)} seconds",
                            'frame_idx': frame_count,
                            'track_id': track_id,
                            'confidence': 0.8,
                            'thumbnail': get_frame_thumbnail(frame)
                        }
                        events.append(event)
                        self.recent_events[event_key] = timestamp_float
                    
                    # Remove from tracking
                    del self.people_in_restricted[track_id]
        
        # Clean up stale tracks
        stale_time = 5.0  # 5 seconds
        stale_tracks = []
        
        for track_id, track_data in self.people_in_restricted.items():
            # Check if this track is still being detected
            track_still_active = False
            for t_id, _, _, _ in tracks:
                if t_id == track_id:
                    track_still_active = True
                    break
            
            if not track_still_active:
                stale_tracks.append(track_id)
        
        for track_id in stale_tracks:
            del self.people_in_restricted[track_id]
        
        return events
    
    def annotate_frame(self, frame):
        """Add restricted area annotations to the frame."""
        if not self.initialized:
            return frame
            
        # Draw restricted areas
        for area_name, area_polygon in self.restricted_areas.items():
            # Draw with red color
            cv2.polylines(frame, [area_polygon], True, (0, 0, 255), 2)
            
            # Find center of area for label
            center_x = int(np.mean(area_polygon[:, 0]))
            center_y = int(np.mean(area_polygon[:, 1]))
            
            # Add area name
            cv2.putText(frame, area_name, (center_x - 50, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw people in restricted areas
        for track_id, data in self.people_in_restricted.items():
            position = data['last_position']
            entry_time = data['entry_time']
            area_name = data['area']
            
            # Calculate time spent
            current_time = time.time()
            time_spent = int(current_time - entry_time)
            
            # Draw circle at person's position
            cv2.circle(frame, position, 7, (0, 0, 255), -1)
            
            # Add track ID and time spent
            cv2.putText(frame, f"ID: {track_id}", (position[0] + 10, position[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f"Time: {time_spent}s", (position[0] + 10, position[1] + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add count of people in restricted areas
        if self.people_in_restricted:
            h, w = frame.shape[:2]
            
            # Create overlay for alert text
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, h - 60), (400, h - 10), (0, 0, 0), -1)
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Add alert text
            count = len(self.people_in_restricted)
            alert_text = f"⚠️ ALERT: {count} {'person' if count == 1 else 'people'} in restricted areas"
            cv2.putText(frame, alert_text, (20, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return frame 