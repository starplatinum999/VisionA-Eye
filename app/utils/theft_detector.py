import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional

class TheftDetector:
    """
    Utility class for detecting theft-related events.
    
    This class analyzes movement patterns, dwell times, and spatial relationships
    to detect potential theft or suspicious behaviors, including more realistic scenarios such as concealment and exit without checkout.
    """
    
    def __init__(self, 
                 cash_counter_roi=None, 
                 dwell_time_threshold=20.0, 
                 quick_grab_threshold=2.0,
                 data_dir=None,
                 suspicious_movement_threshold=0.5,
                 entry_zone=None,
                 exit_zone=None,
                 customer_zone=None,
                 cashier_zone=None):
        """
        Initialize theft detector.
        
        Args:
            cash_counter_roi: Region of interest for cash counter
            dwell_time_threshold: Time threshold for suspicious dwell time (seconds)
            quick_grab_threshold: Time threshold for quick grab detection (seconds)
            data_dir: Directory to save theft event data
            suspicious_movement_threshold: Threshold for suspicious movement speed
            entry_zone: Entry zone region of interest
            exit_zone: Exit zone region of interest
            customer_zone: Customer zone region of interest
            cashier_zone: Cashier/restricted zone region of interest
        """
        self.cash_counter_roi = cash_counter_roi
        self.dwell_time_threshold = dwell_time_threshold
        self.quick_grab_threshold = quick_grab_threshold
        self.data_dir = data_dir
        self.suspicious_movement_threshold = suspicious_movement_threshold or 0.5
        self.entry_zone = entry_zone
        self.exit_zone = exit_zone
        self.customer_zone = customer_zone
        self.cashier_zone = cashier_zone
        
        # Tracking dictionaries
        self.tracked_people = {}  # track_id -> position history
        self.tracked_objects = {}  # track_id -> object data
        
        # Alert history
        self.recent_alerts = []
        self.max_alerts = 50
        
        self.person_item_map = {}  # person_id -> set of item_ids
        self.item_last_seen = {}   # item_id -> (frame, position, last_person_id)
        self.person_bag_map = {}   # person_id -> bool (has bag)
        self.person_roi_history = {}  # person_id -> list of (roi_name, frame_idx)
        self.item_roi_history = {}    # item_id -> list of (roi_name, frame_idx)
        self.exit_roi_name = 'Exit'
        self.cashier_roi_name = 'Cashier'
        self.shelf_roi_names = ['Shelf1', 'Shelf2', 'Shelf3', 'Electronics', 'Jewelry', 'Clothing']
        self.bag_class_ids = [24, 26, 27, 31]  # COCO: backpack, handbag, suitcase, etc.
        self.person_class_id = 0  # COCO: person
        self.item_class_ids = list(range(1, 24)) + list(range(25, 80))  # All non-person, non-bag
    
    def update(self, detections, frame_idx, rois):
        """
        Main update function for each frame.
        
        Args:
            detections: list of dicts with keys: class_id, bbox, track_id
            frame_idx: current frame number
            rois: dict of ROI names to polygons
        """
        # 1. Associate items with persons (proximity)
        for det in detections:
            if det['class_id'] == self.person_class_id:
                person_id = det['track_id']
                person_bbox = det['bbox']
                # Check for bag
                self.person_bag_map[person_id] = self._has_bag(person_id, detections)
                # Track ROI history
                roi_name = self._get_current_roi(person_bbox, rois)
                self.person_roi_history.setdefault(person_id, []).append((roi_name, frame_idx))
                # Check for nearby items
                for item_det in detections:
                    if item_det['class_id'] in self.item_class_ids:
                        item_id = item_det['track_id']
                        item_bbox = item_det['bbox']
                        if self._is_near(person_bbox, item_bbox):
                            self.person_item_map.setdefault(person_id, set()).add(item_id)
                            self.item_last_seen[item_id] = (frame_idx, item_bbox, person_id)
                            self.item_roi_history.setdefault(item_id, []).append((roi_name, frame_idx))
        # 2. Detect concealment (item disappears inside bag)
        for person_id, item_ids in self.person_item_map.items():
            if self.person_bag_map.get(person_id, False):
                for item_id in list(item_ids):
                    if not self._item_visible(item_id, detections):
                        # If item was last seen overlapping with bag, flag as concealment
                        if self._was_item_concealed(item_id, person_id, detections):
                            self._add_alert({
                                'alert_type': 'concealment',
                                'message': f"Person #{person_id} may have concealed item #{item_id} in a bag.",
                                'track_id': person_id,
                                'item_id': item_id,
                                'confidence': 0.9
                            })
        # 3. Detect exit without checkout
        for person_id, item_ids in self.person_item_map.items():
            if self._person_exited(person_id, rois) and not self._person_visited_cashier(person_id, rois):
                self._add_alert({
                    'alert_type': 'exit_without_checkout',
                    'message': f"Person #{person_id} exited with items {list(item_ids)} without visiting cashier.",
                    'track_id': person_id,
                    'item_ids': list(item_ids),
                    'confidence': 0.95
                })
    
    def update_person_position(self, track_id, position, timestamp):
        """
        Update a person's position and check for suspicious activity.
        
        Args:
            track_id: Tracking ID
            position: (x, y) normalized coordinates
            timestamp: Current timestamp
            
        Returns:
            Dict with alert information or None
        """
        x, y = position
        
        # Initialize tracking if new person
        if track_id not in self.tracked_people:
            self.tracked_people[track_id] = {
                'positions': [(x, y)],
                'timestamps': [timestamp],
                'in_cash_counter': False,
                'cash_counter_entry_time': None,
                'last_position': (x, y),
                'last_seen': timestamp
            }
            return None
        
        # Get tracking data
        person_data = self.tracked_people[track_id]
        person_data['positions'].append((x, y))
        person_data['timestamps'].append(timestamp)
        person_data['last_position'] = (x, y)
        person_data['last_seen'] = timestamp
        
        # Limit history size
        if len(person_data['positions']) > 50:
            person_data['positions'].pop(0)
            person_data['timestamps'].pop(0)
        
        # Check if person is in cash counter ROI
        in_cash_counter = False
        if self.cash_counter_roi is not None:
            in_cash_counter = self._point_in_polygon((x, y), self.cash_counter_roi)
        
        # Check for cash counter entry
        if in_cash_counter and not person_data['in_cash_counter']:
            person_data['in_cash_counter'] = True
            person_data['cash_counter_entry_time'] = timestamp
        
        # Check for cash counter exit
        elif not in_cash_counter and person_data['in_cash_counter']:
            person_data['in_cash_counter'] = False
            
            # Check dwell time in cash counter
            if person_data['cash_counter_entry_time'] is not None:
                dwell_time = timestamp - person_data['cash_counter_entry_time']
                
                # Reset entry time
                person_data['cash_counter_entry_time'] = None
                
                # Check for quick grab (suspiciously short time)
                if dwell_time < self.quick_grab_threshold:
                    alert = {
                        'alert_type': 'quick_grab',
                        'message': f"Person #{track_id} had suspiciously quick interaction at cash counter ({dwell_time:.1f}s)",
                        'track_id': track_id,
                        'position': (x, y),
                        'frame_time': timestamp,
                        'confidence': 0.75
                    }
                    self._add_alert(alert)
                    return alert
        
        # Check for unusual dwell time within cash counter
        if in_cash_counter and person_data['cash_counter_entry_time'] is not None:
            current_dwell_time = timestamp - person_data['cash_counter_entry_time']
            if current_dwell_time > self.dwell_time_threshold:
                alert = {
                    'alert_type': 'unusual_dwell_time',
                    'message': f"Person #{track_id} dwelling at cash counter for {current_dwell_time:.1f}s",
                    'track_id': track_id,
                    'position': (x, y),
                    'frame_time': timestamp,
                    'confidence': 0.65 + min(0.25, (current_dwell_time - self.dwell_time_threshold) / 60.0)
                }
                self._add_alert(alert)
                return alert
        
        # Check for suspicious movement patterns
        if len(person_data['positions']) > 5:
            # Analyze movement (fast back-and-forth movements, unusual paths)
            positions = person_data['positions']
            timestamps = person_data['timestamps']
            
            # Calculate recent movement speed (distance / time)
            recent_distance = self._calculate_distance(positions[-5], positions[-1])
            time_diff = timestamps[-1] - timestamps[-5]
            
            if time_diff > 0:
                speed = recent_distance / time_diff
                
                # Check for unusually fast movement
                if speed > self.suspicious_movement_threshold:
                    alert = {
                        'alert_type': 'suspicious_movement',
                        'message': f"Person #{track_id} moving unusually quickly ({speed:.2f} units/s)",
                        'track_id': track_id,
                        'position': (x, y),
                        'frame_time': timestamp,
                        'confidence': min(0.9, 0.5 + speed / 2.0)
                    }
                    self._add_alert(alert)
                    return alert
        
        return None
    
    def update_object_position(self, track_id, object_class, position, timestamp):
        """
        Update an object's position and check for suspicious movement.
        
        Args:
            track_id: Tracking ID
            object_class: Object class name
            position: (x, y) normalized coordinates
            timestamp: Current timestamp
            
        Returns:
            Dict with alert information or None
        """
        x, y = position
        
        # Initialize tracking if new object
        if track_id not in self.tracked_objects:
            self.tracked_objects[track_id] = {
                'class': object_class,
                'positions': [(x, y)],
                'timestamps': [timestamp],
                'last_position': (x, y),
                'last_seen': timestamp,
                'near_person': None
            }
            return None
        
        # Get tracking data
        object_data = self.tracked_objects[track_id]
        object_data['positions'].append((x, y))
        object_data['timestamps'].append(timestamp)
        object_data['last_position'] = (x, y)
        object_data['last_seen'] = timestamp
        
        # Limit history size
        if len(object_data['positions']) > 50:
            object_data['positions'].pop(0)
            object_data['timestamps'].pop(0)
        
        # Check if object is moving quickly (potential theft)
        if len(object_data['positions']) > 5:
            positions = object_data['positions']
            timestamps = object_data['timestamps']
            
            # Calculate recent movement speed
            recent_distance = self._calculate_distance(positions[-5], positions[-1])
            time_diff = timestamps[-1] - timestamps[-5]
            
            if time_diff > 0:
                speed = recent_distance / time_diff
                
                # Check for unusually fast object movement
                if speed > 0.4:  # Threshold for fast object movement
                    alert = {
                        'alert_type': 'suspicious_object_movement',
                        'message': f"{object_class.title()} #{track_id} moving unusually quickly ({speed:.2f} units/s)",
                        'track_id': track_id,
                        'position': (x, y),
                        'frame_time': timestamp,
                        'object_class': object_class,
                        'confidence': min(0.9, 0.5 + speed / 2.0)
                    }
                    self._add_alert(alert)
                    return alert
        
        return None
    
    def cleanup_stale_objects(self, max_age=10.0):
        """
        Remove stale objects and people from tracking.
        
        Args:
            max_age: Maximum time since last update (seconds)
        """
        current_time = time.time()
        
        # Clean up people
        stale_people = []
        for track_id, person_data in self.tracked_people.items():
            if current_time - person_data['last_seen'] > max_age:
                stale_people.append(track_id)
        
        for track_id in stale_people:
            del self.tracked_people[track_id]
        
        # Clean up objects
        stale_objects = []
        for track_id, object_data in self.tracked_objects.items():
            if current_time - object_data['last_seen'] > max_age:
                stale_objects.append(track_id)
        
        for track_id in stale_objects:
            del self.tracked_objects[track_id]
    
    def get_recent_alerts(self, count=None):
        """Get list of recent alerts.
        
        Args:
            count: Maximum number of alerts to return (default: all)
        
        Returns:
            List of recent alert dictionaries
        """
        if count is None:
            return self.recent_alerts
        else:
            return self.recent_alerts[-count:] if count > 0 else []
    
    def get_confirmed_thefts(self, count=None):
        """Get list of confirmed theft events.
        
        Args:
            count: Maximum number of events to return (default: all)
            
        Returns:
            List of confirmed theft event dictionaries
        """
        # Filter alerts to only include high-confidence theft events
        theft_events = [alert for alert in self.recent_alerts 
                       if alert.get('alert_type') in ['theft_in_progress', 'suspicious_object_movement', 'quick_grab']
                       and alert.get('confidence', 0) > 0.8]
        
        if count is None:
            return theft_events
        else:
            return theft_events[-count:] if count > 0 else []
    
    def draw_zones(self, frame):
        """
        Draw detection zones on the frame.
        
        Args:
            frame: Video frame to annotate
            
        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]
        
        # Draw cash counter area if defined
        if self.cash_counter_roi is not None:
            # Convert normalized coordinates to pixel coordinates
            points = []
            for px, py in self.cash_counter_roi:
                x, y = int(px * w), int(py * h)
                points.append((x, y))
            
            # Draw polygon
            points = np.array(points, np.int32)
            cv2.polylines(frame, [points], True, (0, 0, 255), 2)
            
            # Add label
            cx = int(np.mean([p[0] for p in points]))
            cy = int(np.mean([p[1] for p in points]))
            cv2.putText(frame, "Cash Counter", (cx - 60, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def display_alert_overlay(self, frame, alert):
        """
        Add alert overlay to the frame.
        
        Args:
            frame: Video frame to annotate
            alert: Alert data
            
        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]
        
        # Create alert box
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 100), (w - 10, h - 20), (0, 0, 150), -1)
        
        # Set transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Format timestamp for display if needed
        timestamp_display = alert.get('timestamp', '')
        
        # Handle different timestamp formats
        if isinstance(timestamp_display, float):
            # Convert float timestamp to formatted string
            timestamp_display = time.strftime("%H:%M:%S", time.localtime(timestamp_display))
        elif isinstance(timestamp_display, str) and len(timestamp_display) > 0:
            # String timestamp (already formatted)
            pass
        else:
            # If no timestamp or invalid, use current time
            timestamp_display = time.strftime("%H:%M:%S", time.localtime(time.time()))
        
        # Add alert text
        alert_type = alert.get('alert_type', 'ALERT').upper()
        cv2.putText(frame, f"⚠️ ALERT: {alert_type}", (20, h - 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, alert.get('message', f"Alert detected at {timestamp_display}"), (20, h - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _add_alert(self, alert):
        """Add an alert to the recent alerts list."""
        # Add timestamp if not present
        if 'timestamp' not in alert:
            alert['timestamp'] = time.time()  # Store as float timestamp instead of string
        
        # Add to recent alerts
        self.recent_alerts.append(alert)
        
        # Limit size of recent alerts
        if len(self.recent_alerts) > self.max_alerts:
            self.recent_alerts.pop(0)
    
    def _point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon."""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        x1, y1 = point1
        x2, y2 = point2
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 
    
    def _is_near(self, bbox1, bbox2, threshold=80):
        # Compute center points and check distance
        x1 = (bbox1[0] + bbox1[2]) / 2
        y1 = (bbox1[1] + bbox1[3]) / 2
        x2 = (bbox2[0] + bbox2[2]) / 2
        y2 = (bbox2[1] + bbox2[3]) / 2
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return dist < threshold
    
    def _has_bag(self, person_id, detections):
        # Check if a bag/backpack is detected near the person
        person_bbox = None
        for det in detections:
            if det['class_id'] == self.person_class_id and det['track_id'] == person_id:
                person_bbox = det['bbox']
                break
        if person_bbox is None:
            return False
        for det in detections:
            if det['class_id'] in self.bag_class_ids:
                if self._is_near(person_bbox, det['bbox'], threshold=100):
                    return True
        return False
    
    def _item_visible(self, item_id, detections):
        # Check if item_id is present in current detections
        for det in detections:
            if det['track_id'] == item_id:
                return True
        return False
    
    def _was_item_concealed(self, item_id, person_id, detections):
        # Check if last seen position of item overlapped with person's bag
        if item_id not in self.item_last_seen:
            return False
        _, item_bbox, _ = self.item_last_seen[item_id]
        for det in detections:
            if det['class_id'] in self.bag_class_ids and self._is_near(item_bbox, det['bbox'], threshold=60):
                return True
        return False
    
    def _get_current_roi(self, bbox, rois):
        # Returns the name of the ROI the bbox is currently in, or None
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        for roi_name, polygon in rois.items():
            if self._point_in_polygon((x, y), polygon):
                return roi_name
        return None
    
    def _person_exited(self, person_id, rois):
        # Check if person's last ROI is exit
        if person_id not in self.person_roi_history:
            return False
        for roi_name, _ in reversed(self.person_roi_history[person_id]):
            if roi_name == self.exit_roi_name:
                return True
            elif roi_name is not None:
                break  # Last ROI was not exit
        return False
    
    def _person_visited_cashier(self, person_id, rois):
        # Check if person's ROI history includes cashier
        if person_id not in self.person_roi_history:
            return False
        for roi_name, _ in self.person_roi_history[person_id]:
            if roi_name == self.cashier_roi_name:
                return True
        return False