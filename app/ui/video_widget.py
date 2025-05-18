import cv2
import numpy as np
from datetime import datetime, timedelta
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
    QFrame, QProgressBar, QSizePolicy, QSlider
)
from PySide6.QtCore import QTimer, Qt, Signal, Slot, QSize
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QPalette

from app.components.detection.detector import YOLODetector
from app.components.tracking.tracker import DeepSORTTracker
from app.utils.video_utils import get_frame_thumbnail

class VideoWidget(QWidget):
    """Widget for displaying and processing video frames with YOLO detection and DeepSORT tracking."""
    
    # Signal to emit when events are detected
    event_detected = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Main layout
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(10)
        self.setLayout(self.layout)
        
        # Video display container with border
        video_container = QFrame()
        video_container.setStyleSheet("""
            QFrame {
                border: 2px solid #E5E7EB;
                border-radius: 12px;
                background-color: #111827;
                padding: 5px;
            }
        """)
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        # Video display label
        self.video_label = QLabel("Video feed will appear here")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("""
            color: #9CA3AF;
            font-size: 18px;
            font-weight: bold;
            background-color: #1F2937;
        """)
        video_layout.addWidget(self.video_label)
        
        self.layout.addWidget(video_container)
        
        # Video information panel
        info_panel = QFrame()
        info_panel.setStyleSheet("""
            QFrame {
                background-color: #F3F4F6;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        info_layout = QHBoxLayout(info_panel)
        
        # Frame counter
        self.frame_counter = QLabel("Frame: 0 / 0")
        self.frame_counter.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #4B5563;
        """)
        info_layout.addWidget(self.frame_counter)
        
        # Timestamp
        self.timestamp_label = QLabel("Time: 00:00:00")
        self.timestamp_label.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #4B5563;
        """)
        info_layout.addWidget(self.timestamp_label)
        
        # Detection counter
        self.detection_counter = QLabel("Detections: 0")
        self.detection_counter.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #4B5563;
        """)
        info_layout.addWidget(self.detection_counter)
        
        # Add spacer to push info items left
        info_layout.addStretch()
        
        self.layout.addWidget(info_panel)
        
        # Progress bar for video playback
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                background-color: #F9FAFB;
                height: 12px;
                text-align: center;
                color: #4B5563;
                font-size: 12px;
            }
            QProgressBar::chunk {
                background-color: #3B82F6;
                border-radius: 7px;
            }
        """)
        self.layout.addWidget(self.progress_bar)
        
        # Controls with modern styling
        controls_container = QFrame()
        controls_container.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 10px;
                border: 1px solid #E5E7EB;
                padding: 10px;
            }
        """)
        
        self.controls_layout = QHBoxLayout(controls_container)
        self.controls_layout.setContentsMargins(10, 10, 10, 10)
        self.controls_layout.setSpacing(15)
        
        # Play/pause button with gradient styling
        self.play_button = QPushButton("Start Processing")
        self.play_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                         stop:0 #4F46E5, stop:1 #4338CA);
                color: white;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 14px;
                min-width: 180px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                         stop:0 #6366F1, stop:1 #4F46E5);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                         stop:0 #4338CA, stop:1 #3730A3);
            }
        """)
        self.play_button.clicked.connect(self.toggle_play)
        self.controls_layout.addWidget(self.play_button)
        
        # Stop button
        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #F87171;
                color: white;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #EF4444;
            }
            QPushButton:pressed {
                background-color: #DC2626;
            }
            QPushButton:disabled {
                background-color: #D1D5DB;
                color: #9CA3AF;
            }
        """)
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setEnabled(False)
        self.controls_layout.addWidget(self.stop_button)
        
        # Snapshot button
        self.snapshot_button = QPushButton("Take Snapshot")
        self.snapshot_button.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: white;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:pressed {
                background-color: #047857;
            }
        """)
        self.snapshot_button.clicked.connect(self.take_snapshot)
        self.controls_layout.addWidget(self.snapshot_button)
        
        # Add controls to main layout
        self.layout.addWidget(controls_container)
        
        # Initialize video processing components
        self.detector = YOLODetector()
        self.tracker = DeepSORTTracker()
        self.roi_areas = {}
        self.roi_colors = {}
        
        # Video state variables
        self.video_path = None
        self.is_camera = False
        self.cap = None
        self.frame_count = 0
        self.total_frames = 0
        self.fps = 0
        self.is_playing = False
        self.current_frame = None
        self.base_time = datetime.now()
        self.detection_count = 0
        
        # Tracking state
        self.tracks = {}
        
        # Video playback timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
    
    def take_snapshot(self):
        """Save the current frame as an image file."""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            # Show confirmation
            self.timestamp_label.setText(f"Snapshot saved: {filename}")
            self.timestamp_label.setStyleSheet("""
                font-size: 14px;
                font-weight: bold;
                color: #10B981;
            """)
            # Reset style after 2 seconds
            QTimer.singleShot(2000, self.reset_timestamp_style)
    
    def reset_timestamp_style(self):
        """Reset the timestamp label style after snapshot notification."""
        self.timestamp_label.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #4B5563;
        """)
        
    def load_video(self, video_path, is_camera=False):
        """Load a video file or camera feed."""
        self.video_path = video_path
        self.is_camera = is_camera
        
        # Print diagnostic info
        print(f"Attempting to open video source: {video_path}")
        print(f"Is camera/stream: {is_camera}")
        
        # Open video capture
        if is_camera and video_path.isdigit():
            # For numeric camera indices (0, 1, 2, etc.)
            self.cap = cv2.VideoCapture(int(video_path))
        else:
            # For video files and RTSP URLs
            # Set RTSP transport protocol for better reliability
            if video_path.startswith('rtsp://'):
                print(f"Connecting to RTSP stream: {video_path}")
                # Configure RTSP transport protocol
                cv2.setUseOptimized(True)
                
                # Attempt with TCP transport first (more reliable)
                self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1024)  # Increase buffer size
                
                # Check if connection was successful
                if not self.cap.isOpened():
                    print("Failed to connect with default settings, trying alternative settings...")
                    # Try with UDP transport as fallback
                    self.cap.release()
                    self.cap = cv2.VideoCapture(video_path)
            else:
                self.cap = cv2.VideoCapture(video_path)
            
        if not self.cap.isOpened():
            error_msg = f"Cannot open video source: {video_path}"
            print(error_msg)
            if video_path.startswith('rtsp://'):
                print("RTSP connection troubleshooting:")
                print("1. Verify the URL is correct and the stream is active")
                print("2. Check network connectivity")
                print("3. Verify if authentication is required")
                print("4. Make sure required codecs are installed")
            raise ValueError(error_msg)
        
        print(f"Successfully opened video source: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"Video FPS: {self.fps}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {self.total_frames}")
        
        # Read first frame
        print("Attempting to read first frame...")
        ret, frame = self.cap.read()
        if ret:
            print("First frame read successfully")
            self.current_frame = frame
            self.display_frame(frame)
            
            # Update frame counter
            self.frame_counter.setText(f"Frame: 1 / {self.total_frames if self.total_frames > 0 else 'Live'}")
            
            # Update progress bar for video files
            if self.total_frames > 0:
                self.progress_bar.setValue(0)
            else:
                # Hide progress bar for live feeds
                self.progress_bar.setVisible(False)
        else:
            print("Failed to read first frame")
            raise ValueError("Failed to read first frame from video source")
        
        # Reset state
        self.frame_count = 0
        self.tracks = {}
        self.base_time = datetime.now()
        self.detection_count = 0
        
        # Set timer interval based on FPS
        self.timer.setInterval(int(1000 / self.fps))
        
        # Update UI
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.is_playing = False
        
        return True
    
    def set_roi_areas(self, roi_areas, roi_colors):
        """Set the ROI areas for processing."""
        self.roi_areas = roi_areas
        self.roi_colors = roi_colors
        
        # Update display if we have a current frame
        if self.current_frame is not None:
            self.display_frame_with_annotations(self.current_frame)
    
    def toggle_play(self):
        """Toggle video playback and processing."""
        if not self.is_playing:
            if not self.cap or not self.cap.isOpened():
                if not self.load_video(self.video_path, self.is_camera):
                    return
            
            # Start processing
            self.is_playing = True
            self.play_button.setText("Pause")
            self.stop_button.setEnabled(True)
            self.timer.start()
        else:
            # Pause processing
            self.is_playing = False
            self.play_button.setText("Resume")
            self.timer.stop()
    
    def stop_video(self):
        """Stop video playback and processing."""
        self.timer.stop()
        self.is_playing = False
        self.play_button.setText("Start Processing")
        self.stop_button.setEnabled(False)
        
        # Release video capture
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
    
    def get_current_frame(self):
        """Get the current frame for use by other widgets."""
        return self.current_frame
    
    def process_frame(self):
        """Process the next video frame."""
        if not self.cap or not self.cap.isOpened():
            self.stop_video()
            return
        
        ret, frame = self.cap.read()
        if not ret:
            # End of video file
            self.stop_video()
            return
        
        # Update frame count
        self.frame_count += 1
        
        # Generate timestamp based on frame count and FPS
        seconds = self.frame_count / self.fps
        timestamp = self.base_time + timedelta(seconds=seconds)
        
        # Process frame with detector and tracker
        try:
            # Run object detection
            detections = self.detector.detect(frame)
            
            # Update tracker
            tracks_updated = self.tracker.update(frame, detections)
            
            # Process tracks
            events = self.process_tracks(tracks_updated, frame, timestamp)
            
            # Update detection count
            self.detection_count = len(detections)
            self.detection_counter.setText(f"Detections: {self.detection_count}")
            
            # Emit events if any
            for event in events:
                self.event_detected.emit(event)
            
            # Draw annotations
            processed_frame = self.draw_annotations(frame.copy(), tracks_updated, timestamp)
            
            # Display frame
            self.display_frame_with_annotations(processed_frame)
            
            # Update progress bar for video files
            if self.total_frames > 0:
                progress = int((self.frame_count / self.total_frames) * 100)
                self.progress_bar.setValue(progress)
            
            # Update frame counter
            self.frame_counter.setText(f"Frame: {self.frame_count} / {self.total_frames if self.total_frames > 0 else 'Live'}")
            
            # Update timestamp
            self.timestamp_label.setText(f"Time: {timestamp.strftime('%H:%M:%S')}")
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()
        
        # Store current frame
        self.current_frame = frame
    
    def process_tracks(self, tracks_updated, frame, timestamp):
        """Process tracked objects to detect events."""
        frame_events = []
        
        for track_id, bbox, class_id, confidence in tracks_updated:
            x1, y1, x2, y2 = map(int, bbox)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Get or create track data
            track = self.tracks.setdefault(track_id, {
                'frames': [],
                'locations': [],
                'class_id': class_id,
                'roi_visits': {},
                'last_roi': None,
                'items_picked': []
            })
            
            track['frames'].append(self.frame_count)
            track['locations'].append((cx, cy))
            
            # Check ROI interactions
            for roi_name, (rx1, ry1, rx2, ry2) in self.roi_areas.items():
                in_roi = rx1 <= cx <= rx2 and ry1 <= cy <= ry2
                
                if in_roi:
                    if roi_name not in track['roi_visits']:
                        track['roi_visits'][roi_name] = {'enter_frame': self.frame_count, 'exit_frame': None}
                        
                        # Generate ROI transition event
                        if track['last_roi'] != roi_name:
                            event_data = {
                                'type': 'ROI Transition',
                                'timestamp': timestamp.strftime('%H:%M:%S'),
                                'description': f"Object ID {track_id} entered {roi_name}",
                                'thumbnail': get_frame_thumbnail(frame),
                                'frame_idx': self.frame_count,
                                'track_id': track_id,
                                'from_roi': track['last_roi'],
                                'to_roi': roi_name,
                                'needs_reasoning': False
                            }
                            
                            # Specialized events based on context
                            if class_id == 0 and 'shelf' in roi_name.lower():
                                event_data['type'] = 'Person at Shelf'
                                event_data['description'] = f"Person ID {track_id} is browsing at {roi_name}"
                            
                            if class_id == 0 and track['last_roi'] and 'shelf' in track['last_roi'].lower() and 'exit' in roi_name.lower():
                                event_data['type'] = 'Potential Theft'
                                event_data['description'] = f"Person ID {track_id} moved from {track['last_roi']} directly to {roi_name}"
                            
                            # Item pickup events - check if a product is near a person
                            if class_id > 0 and 'shelf' in roi_name.lower():
                                for pid, pdata in self.tracks.items():
                                    if pdata['class_id'] == 0 and self.frame_count in pdata['frames']:
                                        pidx = pdata['frames'].index(self.frame_count)
                                        px, py = pdata['locations'][pidx]
                                        dist = np.hypot(cx - px, cy - py)
                                        if dist < 100:
                                            pickup_event = {
                                                'type': 'Item Pickup',
                                                'timestamp': timestamp.strftime('%H:%M:%S'),
                                                'description': f"Item ID {track_id} picked up by Person ID {pid}",
                                                'thumbnail': get_frame_thumbnail(frame),
                                                'frame_idx': self.frame_count,
                                                'item_id': track_id,
                                                'person_id': pid,
                                                'needs_reasoning': False
                                            }
                                            frame_events.append(pickup_event)
                                            pdata['items_picked'].append(track_id)
                                            break
                            
                            frame_events.append(event_data)
                    
                    track['last_roi'] = roi_name
                    break
                elif roi_name in track['roi_visits'] and track['roi_visits'][roi_name]['exit_frame'] is None:
                    track['roi_visits'][roi_name]['exit_frame'] = self.frame_count
                    
                    # Generate ROI exit event
                    exit_event = {
                        'type': 'ROI Exit',
                        'timestamp': timestamp.strftime('%H:%M:%S'),
                        'description': f"Object ID {track_id} exited {roi_name}",
                        'thumbnail': get_frame_thumbnail(frame),
                        'frame_idx': self.frame_count,
                        'track_id': track_id,
                        'roi_name': roi_name,
                        'needs_reasoning': False
                    }
                    frame_events.append(exit_event)
        
        return frame_events
    
    def draw_annotations(self, frame, tracks_updated, timestamp):
        """Draw bounding boxes, labels, and ROI areas on the frame."""
        # Draw ROI areas with labels
        for roi_name, (rx1, ry1, rx2, ry2) in self.roi_areas.items():
            color = self.roi_colors.get(roi_name, (255, 0, 0))
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 2)
            cv2.putText(frame, roi_name, (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw tracked objects
        for track_id, bbox, class_id, confidence in tracks_updated:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Object class information
            class_names = {0: "Person", 1: "Cart", 2: "Bag", 3: "Product"}
            class_name = class_names.get(class_id, f"Class-{class_id}")
            label = f"{class_name} #{track_id} ({confidence:.2f})"
            
            # Colors for different classes
            colors = {
                0: (0, 255, 0),    # Person: Green
                1: (255, 0, 0),    # Cart: Blue
                2: (0, 0, 255),    # Bag: Red
                3: (255, 255, 0)   # Product: Cyan
            }
            color = colors.get(class_id, (200, 200, 200))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display frame number and timestamp
        cv2.putText(frame, f"Frame: {self.frame_count} | Time: {timestamp.strftime('%H:%M:%S')}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show event count
        cv2.putText(frame, f"Events: {len(self.tracks)}", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def display_frame(self, frame):
        """Display the frame on the label."""
        # Convert OpenCV BGR format to RGB for Qt
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        
        # Convert to Qt format
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.width(), self.video_label.height(), 
                                               Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def display_frame_with_annotations(self, frame):
        """Display frame with ROI annotations but without detection/tracking."""
        # Make a copy of the frame
        annotated_frame = frame.copy()
        
        # Draw ROI areas
        for roi_name, (rx1, ry1, rx2, ry2) in self.roi_areas.items():
            color = self.roi_colors.get(roi_name, (255, 0, 0))
            cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), color, 2)
            cv2.putText(annotated_frame, roi_name, (rx1, ry1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display the annotated frame
        self.display_frame(annotated_frame) 