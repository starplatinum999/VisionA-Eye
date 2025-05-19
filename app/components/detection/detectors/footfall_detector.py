import cv2
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from app.components.detection.detectors.base_detector import BaseDetector
from app.utils.video_utils import get_frame_thumbnail

class FootfallDetector(BaseDetector):
    """
    Detector for tracking people movement, entries, and exits.
    
    This detector counts people entering and exiting defined areas
    and tracks dwell time and movement patterns.
    """
    
    def __init__(self):
        super().__init__(name="FootfallDetector")
        self.initialized = False
        
        # Event type mapping
        self.event_type_mapping = {
            "entry": "Area Entry",
            "exit": "Area Exit",
            "dwell": "Long Dwell Time",
            "crossing": "Boundary Crossing",
        }
        
        # Tracking data
        self.tracked_people = {}  # track_id -> last known position and status
        self.entries = 0
        self.exits = 0
        self.current_visitors = 0
        
        # Customer classification counters
        self.potential_customers = 0
        self.window_shoppers = 0
        self.staff_members = 0
        
        # ROI information
        self.entry_zone = None
        self.exit_zone = None
        self.shopping_zone = None
        self.cashier_zone = None  # Added for staff identification
        
        # Confidence threshold for generating events
        self.confidence_threshold = 0.7
        
        # Cooldown between similar events (reduce to ensure more events are generated)
        self.recent_events = {}  # Format: {event_key: timestamp}
        self.event_cooldown = 2.0  # Seconds between similar events (reduced from 3.0)
        
        # Debug mode to print extra information
        self.debug = True
        
        # Distance threshold for "near cashier" detection (in pixels)
        self.proximity_threshold = 100
    
    def set_roi_areas(self, roi_areas: Dict, roi_colors: Dict):
        """Set the ROI areas for footfall detection."""
        super().set_roi_areas(roi_areas, roi_colors)
        
        # Reset zones
        self.entry_zone = None
        self.exit_zone = None
        self.shopping_zone = None
        self.cashier_zone = None
        
        # Extract frame dimensions from first ROI to normalize coordinates
        if roi_areas:
            first_roi = list(roi_areas.values())[0]
            x1, y1, x2, y2 = first_roi
            # Estimate frame dimensions from ROIs
            self.frame_width = max(1920, x2 * 2)  # Ensure reasonable minimum size
            self.frame_height = max(1080, y2 * 2)
        else:
            # Default dimensions if no ROIs provided
            self.frame_width = 1920
            self.frame_height = 1080
        
        # Look for specific ROI types in the roi_areas
        for roi_name, (x1, y1, x2, y2) in roi_areas.items():
            roi_name_lower = roi_name.lower()
            
            # Create polygon for the ROI with pixel coordinates (not normalized)
            roi_polygon = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], np.int32)
            
            # Assign to appropriate zone based on name (more comprehensive matching)
            if any(keyword in roi_name_lower for keyword in ["entry", "entrance", "door", "in"]):
                self.entry_zone = roi_polygon
                if self.debug:
                    print(f"FootfallDetector: Detected entry zone: {roi_name} at {roi_polygon}")
            elif any(keyword in roi_name_lower for keyword in ["exit", "out"]):
                self.exit_zone = roi_polygon
                if self.debug:
                    print(f"FootfallDetector: Detected exit zone: {roi_name} at {roi_polygon}")
            elif any(keyword in roi_name_lower for keyword in ["shop", "browse", "aisle", "shelf"]):
                self.shopping_zone = roi_polygon
                if self.debug:
                    print(f"FootfallDetector: Detected shopping zone: {roi_name} at {roi_polygon}")
            elif any(keyword in roi_name_lower for keyword in ["cashier", "checkout", "cash", "register", "counter"]):
                self.cashier_zone = roi_polygon
                if self.debug:
                    print(f"FootfallDetector: Detected cashier zone: {roi_name} at {roi_polygon}")
        
        # Initialize with default zones if not specified, using frame dimensions
        w, h = self.frame_width, self.frame_height
        
        if self.entry_zone is None:
            self.entry_zone = np.array([
                [0, 0],
                [int(0.3 * w), 0],
                [int(0.3 * w), h],
                [0, h]
            ], np.int32)
            if self.debug:
                print(f"FootfallDetector: Created default entry zone at {self.entry_zone}")
        
        if self.exit_zone is None:
            self.exit_zone = np.array([
                [int(0.7 * w), 0],
                [w, 0],
                [w, h],
                [int(0.7 * w), h]
            ], np.int32)
            if self.debug:
                print(f"FootfallDetector: Created default exit zone at {self.exit_zone}")
        
        if self.shopping_zone is None:
            self.shopping_zone = np.array([
                [int(0.3 * w), 0],
                [int(0.7 * w), 0],
                [int(0.7 * w), h],
                [int(0.3 * w), h]
            ], np.int32)
            if self.debug:
                print(f"FootfallDetector: Created default shopping zone at {self.shopping_zone}")
        
        if self.cashier_zone is None:
            # Default cashier zone near the exit
            self.cashier_zone = np.array([
                [int(0.6 * w), int(0.7 * h)],
                [int(0.9 * w), int(0.7 * h)],
                [int(0.9 * w), int(0.9 * h)],
                [int(0.6 * w), int(0.9 * h)]
            ], np.int32)
            if self.debug:
                print(f"FootfallDetector: Created default cashier zone at {self.cashier_zone}")
        
        # Set proximity threshold as a fraction of the frame dimensions
        self.proximity_threshold = int(min(w, h) * 0.1)  # 10% of min dimension
        if self.debug:
            print(f"FootfallDetector: Proximity threshold set to {self.proximity_threshold} pixels")
            
        self.initialized = True
        print("FootfallDetector: Initialization complete")
    
    def is_point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon."""
        return cv2.pointPolygonTest(polygon, point, False) >= 0
    
    def distance_to_polygon(self, point, polygon):
        """Calculate distance from a point to a polygon."""
        # Convert polygon to the format OpenCV expects
        polygon_arr = np.array(polygon, dtype=np.int32)
        
        # Check if person is inside polygon first
        is_inside = self.is_point_in_polygon(point, polygon_arr)
        if is_inside:
            return 0  # No distance if inside
        
        # Calculate minimum distance to any edge of the polygon
        min_dist = float('inf')
        
        # Iterate through each edge of the polygon
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i+1) % len(polygon)]
            
            # Calculate distance to the line segment
            line_len_sq = (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2
            if line_len_sq == 0:  # Handle zero-length edge
                dist = np.sqrt((point[0] - p1[0])**2 + (point[1] - p1[1])**2)
            else:
                # Calculate projection onto line segment
                t = max(0, min(1, ((point[0] - p1[0]) * (p2[0] - p1[0]) + 
                               (point[1] - p1[1]) * (p2[1] - p1[1])) / line_len_sq))
                proj_x = p1[0] + t * (p2[0] - p1[0])
                proj_y = p1[1] + t * (p2[1] - p1[1])
                dist = np.sqrt((point[0] - proj_x)**2 + (point[1] - proj_y)**2)
            
            min_dist = min(min_dist, dist)
            
        return min_dist
    
    def classify_person(self, position, in_cashier, in_shopping, in_entry, in_exit, track_data):
        """Classify a person as staff, potential customer, or window shopper."""
        # Calculate distance to cashier zone if not already in it
        near_cashier = False
        if not in_cashier and self.cashier_zone is not None:
            distance = self.distance_to_polygon(position, self.cashier_zone)
            near_cashier = distance <= self.proximity_threshold
        
        # Calculate zone ratios
        frames_total = track_data.get('total_frames', 1)
        frames_in_cashier = track_data.get('frames_in_cashier', 0)
        frames_in_shopping = track_data.get('frames_in_shopping', 0)
        
        cashier_ratio = frames_in_cashier / max(1, frames_total) if frames_total > 0 else 0
        shopping_ratio = frames_in_shopping / max(1, frames_total) if frames_total > 0 else 0
        
        # CLASSIFICATION LOGIC
        # 1. Staff: People who spend significant time near cashier area
        if (in_cashier or near_cashier) and (cashier_ratio > 0.3 or frames_in_cashier > 15):
            return "staff"
            
        # 2. Potential customers: People near cashier or who've visited cashier
        elif in_cashier or near_cashier or track_data.get('ever_at_cashier', False):
            return "potential"
            
        # 3. Window shoppers: Everyone else in the shopping area
        elif in_shopping:
            return "window"
            
        # 4. Default classification for others
        else:
            return "visitor"
    
    def process_frame(self, frame, tracks, timestamp, frame_count):
        """Process the frame and detect footfall events."""
        if not self.initialized:
            print("FootfallDetector: Not initialized yet")
            return []
        
        # Update frame dimensions based on actual frame
        h, w = frame.shape[:2]
        self.frame_width = w
        self.frame_height = h
            
        events = []
        timestamp_float = time.mktime(timestamp.timetuple()) + timestamp.microsecond / 1000000.0
        
        # Reset person type counters for each frame
        self.potential_customers = 0
        self.window_shoppers = 0
        self.staff_members = 0
        
        # Debug info
        person_count = sum(1 for _, _, class_id, _ in tracks if class_id == 0)
        if self.debug and frame_count % 100 == 0:
            print(f"FootfallDetector: Processing frame {frame_count} with {person_count} people")
        
        # Check for people in the frame
        for track_id, bbox, class_id, confidence in tracks:
            if class_id != 0:  # We only care about people
                continue
                
            # Get center of the bounding box
            x1, y1, x2, y2 = bbox
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Create a point from the center
            point = (center_x, center_y)
            
            # Check person's position relative to zones
            in_entry = self.is_point_in_polygon(point, self.entry_zone)
            in_exit = self.is_point_in_polygon(point, self.exit_zone)
            in_shopping = self.shopping_zone is not None and self.is_point_in_polygon(point, self.shopping_zone)
            in_cashier = self.cashier_zone is not None and self.is_point_in_polygon(point, self.cashier_zone)
            
            # Get previous state
            prev_state = self.tracked_people.get(track_id, {
                'in_entry': False,
                'in_exit': False,
                'in_shopping': False,
                'in_cashier': False,
                'last_seen': timestamp_float - 10,  # Make sure first detection triggers events
                'entry_time': None,
                'exit_time': None,
                'shopping_entry_time': None,
                'cashier_entry_time': None,
                'position': point,
                'type': 'visitor',  # Default type
                'total_frames': 0,
                'frames_in_shopping': 0,
                'frames_in_cashier': 0,
                'ever_at_cashier': False
            })
            
            # Check for zone transitions
            current_time = timestamp_float
            
            # Entry zone events
            if in_entry and not prev_state['in_entry']:
                # Person entered the entry zone
                event_key = f"entry_{track_id}"
                if current_time - self.recent_events.get(event_key, 0) > self.event_cooldown:
                    event = {
                        'type': 'Area Entry',
                        'timestamp': timestamp.strftime('%H:%M:%S'),
                        'description': f"Person {track_id} entered through entrance area",
                        'frame_idx': frame_count,
                        'track_id': track_id,
                        'confidence': 0.9,
                        'thumbnail': get_frame_thumbnail(frame),
                        'roi_name': 'Entry Zone'
                    }
                    events.append(event)
                    self.recent_events[event_key] = current_time
                    self.entries += 1
                    self.current_visitors += 1
                    
                    if self.debug:
                        print(f"FootfallDetector: Generated 'Area Entry' event for person {track_id}")
            
            # Exit zone events
            if in_exit and not prev_state['in_exit']:
                # Person entered the exit zone
                event_key = f"exit_{track_id}"
                if current_time - self.recent_events.get(event_key, 0) > self.event_cooldown:
                    event = {
                        'type': 'Area Exit',
                        'timestamp': timestamp.strftime('%H:%M:%S'),
                        'description': f"Person {track_id} exited through exit area",
                        'frame_idx': frame_count,
                        'track_id': track_id,
                        'confidence': 0.9,
                        'thumbnail': get_frame_thumbnail(frame),
                        'roi_name': 'Exit Zone'
                    }
                    events.append(event)
                    self.recent_events[event_key] = current_time
                    self.exits += 1
                    self.current_visitors = max(0, self.current_visitors - 1)
                    
                    if self.debug:
                        print(f"FootfallDetector: Generated 'Area Exit' event for person {track_id}")
            
            # Shopping zone events
            if in_shopping and not prev_state['in_shopping']:
                # Person entered shopping zone
                event_key = f"shopping_{track_id}"
                if current_time - self.recent_events.get(event_key, 0) > self.event_cooldown:
                    event = {
                        'type': 'Boundary Crossing',
                        'timestamp': timestamp.strftime('%H:%M:%S'),
                        'description': f"Person {track_id} entered shopping area",
                        'frame_idx': frame_count,
                        'track_id': track_id,
                        'confidence': 0.8,
                        'thumbnail': get_frame_thumbnail(frame),
                        'roi_name': 'Shopping Zone'
                    }
                    events.append(event)
                    self.recent_events[event_key] = current_time
                    
                    # Start tracking dwell time
                    prev_state['shopping_entry_time'] = current_time
                    if self.debug:
                        print(f"FootfallDetector: Generated 'Boundary Crossing' event for person {track_id}")
            
            # Cashier zone events - track when someone enters cashier area
            if in_cashier and not prev_state['in_cashier']:
                event_key = f"cashier_{track_id}"
                if current_time - self.recent_events.get(event_key, 0) > self.event_cooldown:
                    event = {
                        'type': 'Boundary Crossing',
                        'timestamp': timestamp.strftime('%H:%M:%S'),
                        'description': f"Person {track_id} entered cashier area",
                        'frame_idx': frame_count,
                        'track_id': track_id,
                        'confidence': 0.85,
                        'thumbnail': get_frame_thumbnail(frame),
                        'roi_name': 'Cashier Zone'
                    }
                    events.append(event)
                    self.recent_events[event_key] = current_time
                    
                    # Mark that this person has been at the cashier
                    prev_state['ever_at_cashier'] = True
                    prev_state['cashier_entry_time'] = current_time
                    if self.debug:
                        print(f"FootfallDetector: Generated 'Boundary Crossing' event for person {track_id} (cashier)")
            
            # Check for long dwell time in shopping zone
            dwell_time_threshold = 5.0  # Reduced to 5 seconds for testing (from 20)
            if prev_state['in_shopping'] and in_shopping and prev_state['shopping_entry_time'] is not None:
                dwell_time = current_time - prev_state['shopping_entry_time']
                
                if dwell_time > dwell_time_threshold:
                    event_key = f"dwell_{track_id}_{int(dwell_time)}"  # Include dwell time in key to create multiple events
                    if current_time - self.recent_events.get(event_key, 0) > self.event_cooldown:
                        event = {
                            'type': 'Long Dwell Time',
                            'timestamp': timestamp.strftime('%H:%M:%S'),
                            'description': f"Person {track_id} dwelling in shopping area for {int(dwell_time)} seconds",
                            'frame_idx': frame_count,
                            'track_id': track_id,
                            'confidence': 0.75,
                            'thumbnail': get_frame_thumbnail(frame),
                            'roi_name': 'Shopping Zone',
                            'dwell_time': dwell_time
                        }
                        events.append(event)
                        self.recent_events[event_key] = current_time
                        
                        if self.debug:
                            print(f"FootfallDetector: Generated 'Long Dwell Time' event for person {track_id} - {dwell_time}s")
            
            # Update tracked person data
            # Update zone counters
            frames_in_shopping = prev_state.get('frames_in_shopping', 0) + (1 if in_shopping else 0)
            frames_in_cashier = prev_state.get('frames_in_cashier', 0) + (1 if in_cashier else 0)
            total_frames = prev_state.get('total_frames', 0) + 1
            
            # Classify person
            person_type = self.classify_person(
                point, in_cashier, in_shopping, in_entry, in_exit, 
                {
                    'frames_in_shopping': frames_in_shopping,
                    'frames_in_cashier': frames_in_cashier,
                    'total_frames': total_frames,
                    'ever_at_cashier': prev_state.get('ever_at_cashier', False) or in_cashier
                }
            )
            
            # Update tracking data
            self.tracked_people[track_id] = {
                'in_entry': in_entry,
                'in_exit': in_exit,
                'in_shopping': in_shopping,
                'in_cashier': in_cashier,
                'last_seen': current_time,
                'position': point,
                'entry_time': current_time if in_entry and not prev_state['in_entry'] else prev_state['entry_time'],
                'exit_time': current_time if in_exit and not prev_state['in_exit'] else prev_state['exit_time'],
                'shopping_entry_time': prev_state['shopping_entry_time'],
                'cashier_entry_time': current_time if in_cashier and not prev_state['in_cashier'] else prev_state.get('cashier_entry_time'),
                'type': person_type,
                'total_frames': total_frames,
                'frames_in_shopping': frames_in_shopping,
                'frames_in_cashier': frames_in_cashier,
                'ever_at_cashier': prev_state.get('ever_at_cashier', False) or in_cashier
            }
            
            # Update person type counters
            if person_type == "staff":
                self.staff_members += 1
            elif person_type == "potential":
                self.potential_customers += 1
            elif person_type == "window":
                self.window_shoppers += 1
        
        # Clean up stale tracks
        stale_time = 5.0  # Reduced from 10 seconds to 5 seconds
        stale_tracks = []
        
        for track_id, track_data in self.tracked_people.items():
            if timestamp_float - track_data['last_seen'] > stale_time:
                stale_tracks.append(track_id)
        
        for track_id in stale_tracks:
            del self.tracked_people[track_id]
        
        # Clean up stale events
        if frame_count % 100 == 0:
            stale_events = []
            for event_key, event_time in self.recent_events.items():
                if timestamp_float - event_time > 30.0:  # Clean events older than 30 seconds
                    stale_events.append(event_key)
            
            for key in stale_events:
                del self.recent_events[key]
        
        # Debug output
        if events and self.debug:
            print(f"FootfallDetector: Generated {len(events)} events")
            
        return events
    
    def annotate_frame(self, frame):
        """Add annotations to the frame showing zones and metrics."""
        if not self.initialized:
            return frame
            
        # Make a copy to avoid modifying the original
        annotated_frame = frame.copy()
        h, w = annotated_frame.shape[:2]
        
        # Draw zones with transparency and labels
        # Entry zone - Green
        overlay = annotated_frame.copy()
        cv2.polylines(overlay, [self.entry_zone], True, (0, 255, 0), 2)
        cv2.fillPoly(overlay, [self.entry_zone], (0, 255, 0, 64))
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
        
        # Add label for entry zone
        entry_x, entry_y = self.entry_zone[0]
        cv2.putText(annotated_frame, "Entry Zone", (entry_x + 10, entry_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Exit zone - Red
        overlay = annotated_frame.copy()
        cv2.polylines(overlay, [self.exit_zone], True, (0, 0, 255), 2)
        cv2.fillPoly(overlay, [self.exit_zone], (0, 0, 255, 64))
        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
        
        # Add label for exit zone
        exit_x, exit_y = self.exit_zone[0]
        cv2.putText(annotated_frame, "Exit Zone", (exit_x + 10, exit_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Shopping zone - Blue
        if self.shopping_zone is not None:
            overlay = annotated_frame.copy()
            cv2.polylines(overlay, [self.shopping_zone], True, (255, 0, 0), 2)
            cv2.fillPoly(overlay, [self.shopping_zone], (255, 0, 0, 64))
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
            
            # Add label for shopping zone
            shop_x, shop_y = self.shopping_zone[0]
            cv2.putText(annotated_frame, "Shopping Zone", (shop_x + 10, shop_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Cashier zone - Purple
        if self.cashier_zone is not None:
            overlay = annotated_frame.copy()
            cv2.polylines(overlay, [self.cashier_zone], True, (255, 0, 255), 2)
            cv2.fillPoly(overlay, [self.cashier_zone], (255, 0, 255, 64))
            cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
            
            # Add label for cashier zone
            cash_x, cash_y = self.cashier_zone[0]
            cv2.putText(annotated_frame, "Cashier Zone", (cash_x + 10, cash_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Draw information box in the top right
        info_panel_width = 300
        info_panel_height = 250
        info_start_x = w - info_panel_width - 10
        info_start_y = 10
        
        # Draw panel with transparency
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, 
                     (info_start_x, info_start_y), 
                     (info_start_x + info_panel_width, info_start_y + info_panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        
        # Add title
        cv2.putText(annotated_frame, "Footfall Analysis", 
                   (info_start_x + 10, info_start_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add metrics
        line_height = 30
        cv2.putText(annotated_frame, f"Total Entries: {self.entries}", 
                   (info_start_x + 15, info_start_y + line_height + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                   
        cv2.putText(annotated_frame, f"Total Exits: {self.exits}", 
                   (info_start_x + 15, info_start_y + 2*line_height + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                   
        cv2.putText(annotated_frame, f"Current Visitors: {self.current_visitors}", 
                   (info_start_x + 15, info_start_y + 3*line_height + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add customer type counts
        cv2.putText(annotated_frame, f"Staff Members: {self.staff_members}", 
                   (info_start_x + 15, info_start_y + 4*line_height + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                   
        cv2.putText(annotated_frame, f"Potential Customers: {self.potential_customers}", 
                   (info_start_x + 15, info_start_y + 5*line_height + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                   
        cv2.putText(annotated_frame, f"Window Shoppers: {self.window_shoppers}", 
                   (info_start_x + 15, info_start_y + 6*line_height + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                   
        # Draw proximity threshold for visualization
        if self.cashier_zone is not None:
            # Draw proximity outline around cashier with dashed line
            expanded_points = []
            for point in self.cashier_zone:
                # Calculate unit normal vector
                next_idx = (np.where(np.all(self.cashier_zone == point, axis=1))[0][0] + 1) % len(self.cashier_zone)
                next_point = self.cashier_zone[next_idx]
                dx = next_point[0] - point[0]
                dy = next_point[1] - point[1]
                length = np.sqrt(dx*dx + dy*dy)
                if length > 0:
                    nx = -dy / length
                    ny = dx / length
                else:
                    nx, ny = 0, 0
                
                # Add expanded point
                expanded_points.append((
                    int(point[0] + self.proximity_threshold * nx),
                    int(point[1] + self.proximity_threshold * ny)
                ))
            
            # Draw dashed proximity outline
            prev_point = expanded_points[-1]
            for i, curr_point in enumerate(expanded_points):
                # Draw dashed line
                if i % 2 == 0:  # Only draw every other segment for dashed effect
                    cv2.line(annotated_frame, prev_point, curr_point, (255, 255, 0), 1, cv2.LINE_AA)
                prev_point = curr_point
        
        # Draw tracked people positions with type indicators
        for track_id, person_data in self.tracked_people.items():
            position = person_data.get('position')
            if position:
                # Determine dot color and label based on person type
                color = (255, 255, 255)  # Default white
                label = f"#{track_id}"
                
                person_type = person_data.get('type', 'visitor')
                if person_type == "staff":
                    color = (255, 0, 255)  # Purple for staff
                    label = f"STAFF #{track_id}"
                elif person_type == "potential":
                    color = (0, 255, 255)  # Cyan for potential customers
                    label = f"POTENTIAL #{track_id}"
                elif person_type == "window":
                    color = (255, 165, 0)  # Orange for window shoppers
                    label = f"WINDOW #{track_id}"
                
                # Draw circle at position
                cv2.circle(annotated_frame, position, 5, color, -1)
                
                # Add bounding box with label
                label_width = len(label) * 8  # Approximate width of text
                label_height = 20
                
                # Label background
                cv2.rectangle(annotated_frame, 
                             (position[0] - 5, position[1] - label_height - 5),
                             (position[0] + label_width, position[1] - 5),
                             color, -1)
                
                # Label text (black on colored background)
                cv2.putText(annotated_frame, label, 
                           (position[0], position[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated_frame 