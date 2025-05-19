import cv2
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from app.components.detection.detectors.base_detector import BaseDetector
from app.utils.video_utils import get_frame_thumbnail

class ShelfMonitorDetector(BaseDetector):
    """
    Detector for monitoring shelf occupancy and product status.
    
    This detector tracks product placement, movement, and stock levels on shelves.
    """
    
    def __init__(self):
        super().__init__(name="ShelfMonitorDetector")
        self.initialized = False
        
        # Event type mapping
        self.event_type_mapping = {
            "low_stock": "Low Stock",
            "empty_shelf": "Empty Shelf",
            "product_placement": "Product Placement",
            "product_removal": "Product Removal",
            "shelf_full": "Shelf Stocked"
        }
        
        # Shelf data
        self.shelf_areas = {}  # name -> polygon
        self.shelf_occupancy = {}  # name -> current product count
        self.shelf_capacities = {}  # name -> max observed count
        self.product_positions = {}  # product_id -> {'shelf': shelf_name, 'position': (x, y), 'last_seen': timestamp}
        
        # Product class IDs of interest (adapted from COCO classes)
        self.product_class_ids = [39, 40, 41, 42, 43, 44, 45, 46, 47, 73, 74, 75, 76, 77]  # Common retail items
        
        # Confidence threshold for generating events
        self.confidence_threshold = 0.7
        
        # Cooldown between similar events
        self.recent_events = {}  # Format: {event_key: timestamp}
        self.event_cooldown = 10.0  # Seconds between similar events
        
        # Shelf detection parameters
        self.shelf_detection_active = True
        self.shelf_layout = None
    
    def set_roi_areas(self, roi_areas: Dict, roi_colors: Dict):
        """Set the ROI areas for shelf monitoring."""
        super().set_roi_areas(roi_areas, roi_colors)
        
        # Reset shelf areas
        self.shelf_areas = {}
        
        # Identify shelf areas from ROIs
        for roi_name, (x1, y1, x2, y2) in roi_areas.items():
            roi_name_lower = roi_name.lower()
            
            # Create polygon for the ROI
            roi_polygon = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], np.int32)
            
            if "shelf" in roi_name_lower or "product" in roi_name_lower:
                self.shelf_areas[roi_name] = roi_polygon
                # Initialize occupancy and capacity
                self.shelf_occupancy[roi_name] = 0
                self.shelf_capacities[roi_name] = 0
        
        # If no shelves were defined, create default shelf layout
        if not self.shelf_areas:
            h, w = 1080, 1920  # Default resolution, will be adjusted in processing
            
            # Create 4 default shelves
            shelf_height = h // 4
            for i in range(4):
                shelf_name = f"Shelf {i+1}"
                y1 = i * shelf_height
                y2 = (i + 1) * shelf_height
                
                shelf_polygon = np.array([
                    [0, y1],
                    [w, y1],
                    [w, y2],
                    [0, y2]
                ], np.int32)
                
                self.shelf_areas[shelf_name] = shelf_polygon
                self.shelf_occupancy[shelf_name] = 0
                self.shelf_capacities[shelf_name] = 0
        
        self.initialized = True
    
    def get_shelf_status(self, shelf_name, current_count, max_capacity):
        """Get shelf status based on product count and capacity."""
        if current_count == 0:
            return "Empty Shelf", (0, 0, 255)  # Red for empty
        
        # If capacity is too low, assume a minimum
        if max_capacity < 3:
            max_capacity = 3
            
        # Calculate occupancy percentage
        occupancy_ratio = current_count / max_capacity
        
        if occupancy_ratio <= 0.33:
            return "Low Stock", (0, 165, 255)  # Orange for low stock
        elif occupancy_ratio <= 0.66:
            return "Medium Stock", (0, 255, 255)  # Yellow for medium stock
        else:
            return "Shelf Stocked", (0, 255, 0)  # Green for well stocked
    
    def process_frame(self, frame, tracks, timestamp, frame_count):
        """Process the frame and detect shelf events."""
        if not self.initialized:
            return []
            
        events = []
        h, w = frame.shape[:2]
        timestamp_float = time.mktime(timestamp.timetuple()) + timestamp.microsecond / 1000000.0
        
        # Reset product counts for this frame
        current_shelf_products = {shelf_name: 0 for shelf_name in self.shelf_areas.keys()}
        
        # Track products in shelves
        for track_id, bbox, class_id, confidence in tracks:
            # Skip non-product classes
            if class_id not in self.product_class_ids and class_id != 0:  # Allow people detection too
                continue
                
            # Get bounding box center
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            point = (int(center_x), int(center_y))
            
            # Check which shelf this product is in
            current_shelf = None
            for shelf_name, shelf_polygon in self.shelf_areas.items():
                if cv2.pointPolygonTest(shelf_polygon, point, False) >= 0:
                    current_shelf = shelf_name
                    if class_id in self.product_class_ids:  # Only count actual products
                        current_shelf_products[shelf_name] += 1
                    break
            
            # If it's a product, track its position
            if class_id in self.product_class_ids:
                product_key = f"product_{track_id}"
                
                # Check if product was previously tracked
                prev_data = self.product_positions.get(product_key, {
                    'shelf': None,
                    'position': point,
                    'last_seen': timestamp_float,
                    'class_id': class_id
                })
                
                # Check for product movement between shelves
                if prev_data['shelf'] != current_shelf and prev_data['shelf'] is not None and current_shelf is not None:
                    # Product moved from one shelf to another
                    event_key = f"product_move_{track_id}_{timestamp_float}"
                    if timestamp_float - self.recent_events.get(event_key, 0) > self.event_cooldown:
                        event = {
                            'type': 'Product Movement',
                            'timestamp': timestamp.strftime('%H:%M:%S'),
                            'description': f"Product moved from {prev_data['shelf']} to {current_shelf}",
                            'frame_idx': frame_count,
                            'track_id': track_id,
                            'confidence': 0.8,
                            'thumbnail': get_frame_thumbnail(frame)
                        }
                        events.append(event)
                        self.recent_events[event_key] = timestamp_float
                
                # Update product position data
                self.product_positions[product_key] = {
                    'shelf': current_shelf,
                    'position': point,
                    'last_seen': timestamp_float,
                    'class_id': class_id
                }
            
            # Track people interacting with shelves
            if class_id == 0:  # Person
                person_key = f"person_{track_id}"
                
                # Check if person is near any shelf
                for shelf_name, shelf_polygon in self.shelf_areas.items():
                    # Check if the person is inside or very close to the shelf
                    distance = cv2.pointPolygonTest(shelf_polygon, point, True)
                    if distance >= -50:  # Within 50 pixels of shelf
                        # Person is interacting with shelf
                        event_key = f"person_shelf_{track_id}_{shelf_name}"
                        if timestamp_float - self.recent_events.get(event_key, 0) > self.event_cooldown:
                            event = {
                                'type': 'Person at Shelf',
                                'timestamp': timestamp.strftime('%H:%M:%S'),
                                'description': f"Person {track_id} interacting with {shelf_name}",
                                'frame_idx': frame_count,
                                'track_id': track_id,
                                'confidence': 0.75,
                                'thumbnail': get_frame_thumbnail(frame)
                            }
                            events.append(event)
                            self.recent_events[event_key] = timestamp_float
        
        # Update shelf statistics and generate events for stock changes
        for shelf_name, product_count in current_shelf_products.items():
            # Update max capacity if current count is higher
            if product_count > self.shelf_capacities.get(shelf_name, 0):
                self.shelf_capacities[shelf_name] = product_count
            
            # Check for stock status changes
            prev_count = self.shelf_occupancy.get(shelf_name, 0)
            if product_count != prev_count:
                # Calculate status
                status, _ = self.get_shelf_status(
                    shelf_name, 
                    product_count, 
                    self.shelf_capacities[shelf_name]
                )
                
                # Generate event for empty or low stock
                if status in ["Empty Shelf", "Low Stock"]:
                    event_key = f"{status}_{shelf_name}"
                    if timestamp_float - self.recent_events.get(event_key, 0) > self.event_cooldown:
                        event = {
                            'type': status,
                            'timestamp': timestamp.strftime('%H:%M:%S'),
                            'description': f"{shelf_name} is {status.lower()} - {product_count} items detected",
                            'frame_idx': frame_count,
                            'track_id': -1,  # No specific track ID for shelf events
                            'confidence': 0.9,
                            'thumbnail': get_frame_thumbnail(frame)
                        }
                        events.append(event)
                        self.recent_events[event_key] = timestamp_float
                
                # Generate events for significant stock changes
                if product_count > prev_count and product_count - prev_count >= 3:
                    # Products were added (restocking)
                    event_key = f"restock_{shelf_name}"
                    if timestamp_float - self.recent_events.get(event_key, 0) > self.event_cooldown:
                        event = {
                            'type': 'Shelf Stocked',
                            'timestamp': timestamp.strftime('%H:%M:%S'),
                            'description': f"{shelf_name} was restocked with {product_count - prev_count} items",
                            'frame_idx': frame_count,
                            'track_id': -1,
                            'confidence': 0.85,
                            'thumbnail': get_frame_thumbnail(frame)
                        }
                        events.append(event)
                        self.recent_events[event_key] = timestamp_float
                elif prev_count > product_count and prev_count - product_count >= 3:
                    # Products were removed (potential sales)
                    event_key = f"stock_decrease_{shelf_name}"
                    if timestamp_float - self.recent_events.get(event_key, 0) > self.event_cooldown:
                        event = {
                            'type': 'Product Removal',
                            'timestamp': timestamp.strftime('%H:%M:%S'),
                            'description': f"{prev_count - product_count} items removed from {shelf_name}",
                            'frame_idx': frame_count,
                            'track_id': -1,
                            'confidence': 0.8,
                            'thumbnail': get_frame_thumbnail(frame)
                        }
                        events.append(event)
                        self.recent_events[event_key] = timestamp_float
            
            # Update shelf occupancy
            self.shelf_occupancy[shelf_name] = product_count
        
        # Clean up stale product positions
        stale_time = 30.0  # 30 seconds
        stale_products = []
        for product_key, product_data in self.product_positions.items():
            if timestamp_float - product_data['last_seen'] > stale_time:
                stale_products.append(product_key)
        
        for product_key in stale_products:
            del self.product_positions[product_key]
        
        return events
    
    def annotate_frame(self, frame):
        """Add shelf status and annotations to the frame."""
        if not self.initialized:
            return frame
        
        # Draw shelf areas with status
        for shelf_name, shelf_polygon in self.shelf_areas.items():
            product_count = self.shelf_occupancy.get(shelf_name, 0)
            max_capacity = self.shelf_capacities.get(shelf_name, 0)
            
            # Get status and color
            status, color = self.get_shelf_status(shelf_name, product_count, max_capacity)
            
            # Draw shelf outline
            cv2.polylines(frame, [shelf_polygon], True, color, 2)
            
            # Calculate center for text
            x_center = int(np.mean(shelf_polygon[:, 0]))
            y_center = int(np.mean(shelf_polygon[:, 1]))
            
            # Draw shelf name and status
            cv2.putText(frame, shelf_name, (x_center - 50, y_center - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"{status} ({product_count}/{max_capacity})", 
                       (x_center - 50, y_center + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw tracked products
        for product_key, product_data in self.product_positions.items():
            if product_data['shelf'] is not None:  # Only show products on shelves
                position = product_data['position']
                class_id = product_data['class_id']
                
                # Draw circle at product position
                cv2.circle(frame, position, 3, (0, 255, 255), -1)
                
                # Extract track ID from product key
                try:
                    track_id = int(product_key.split('_')[1])
                    cv2.putText(frame, f"P-{track_id}", (position[0] + 5, position[1] + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                except:
                    pass
        
        # Add statistics overlay
        h, w = frame.shape[:2]
        overlay = frame.copy()
        overlay_width = 240
        overlay_height = 120
        x_pos = 10
        y_pos = 10
        
        # Draw transparent background
        cv2.rectangle(overlay, (x_pos, y_pos), (x_pos + overlay_width, y_pos + overlay_height), (0, 0, 0), -1)
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw stats
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.5
        font_thickness = 1
        line_height = 25
        
        text_x = x_pos + 10
        text_y = y_pos + 25
        
        # Count total products and empty shelves
        total_products = sum(self.shelf_occupancy.values())
        empty_shelves = sum(1 for count in self.shelf_occupancy.values() if count == 0)
        low_stock_shelves = sum(1 for shelf, count in self.shelf_occupancy.items() 
                              if 0 < count <= 0.33 * self.shelf_capacities.get(shelf, 3))
        
        cv2.putText(frame, f"Total Products: {total_products}", (text_x, text_y), 
                   font, font_size, (255, 255, 255), font_thickness)
        cv2.putText(frame, f"Empty Shelves: {empty_shelves}", (text_x, text_y + line_height), 
                   font, font_size, (0, 0, 255), font_thickness)
        cv2.putText(frame, f"Low Stock Shelves: {low_stock_shelves}", (text_x, text_y + 2*line_height), 
                   font, font_size, (0, 165, 255), font_thickness)
        
        return frame 