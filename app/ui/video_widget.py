import cv2
import numpy as np
from datetime import datetime, timedelta
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, 
    QFrame, QProgressBar, QSizePolicy, QSlider, QComboBox
)
from PySide6.QtCore import QTimer, Qt, Signal, Slot, QSize
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QPalette

from app.components.detection.detector import YOLODetector
from app.components.tracking.tracker import DeepSORTTracker
from app.components.detection.event_detector_manager import EventDetectorManager
from app.components.detection.detector_factory import DetectorFactory
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
        
        # Video seek slider
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 100)
        self.seek_slider.setValue(0)
        self.seek_slider.setEnabled(False)
        self.seek_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #E5E7EB;
                height: 8px;
                background: #F9FAFB;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3B82F6;
                border: 1px solid #2563EB;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider::add-page:horizontal {
                background: #E5E7EB;
                border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background: #93C5FD;
                border-radius: 4px;
            }
        """)
        self.seek_slider.valueChanged.connect(self.seek_video)
        self.layout.addWidget(self.seek_slider)
        
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
        
        # Rewind button
        self.rewind_button = QPushButton("⏪ Rewind")
        self.rewind_button.setStyleSheet("""
            QPushButton {
                background-color: #6B7280;
                color: white;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #4B5563;
            }
            QPushButton:pressed {
                background-color: #374151;
            }
            QPushButton:disabled {
                background-color: #D1D5DB;
                color: #9CA3AF;
            }
        """)
        self.rewind_button.clicked.connect(self.rewind_video)
        self.rewind_button.setEnabled(False)
        self.controls_layout.addWidget(self.rewind_button)
        
        # Step back button
        self.step_back_button = QPushButton("⏮️ Step Back")
        self.step_back_button.setStyleSheet("""
            QPushButton {
                background-color: #6366F1;
                color: white;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: 600;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #4F46E5;
            }
            QPushButton:pressed {
                background-color: #4338CA;
            }
            QPushButton:disabled {
                background-color: #D1D5DB;
                color: #9CA3AF;
            }
        """)
        self.step_back_button.clicked.connect(self.step_back)
        self.step_back_button.setEnabled(False)
        self.controls_layout.addWidget(self.step_back_button)
        
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
        
        # Add event type selection dropdown
        event_selector_container = QFrame()
        event_selector_container.setStyleSheet("""
            QFrame {
                background-color: #F3F4F6;
                border-radius: 8px;
                padding: 8px;
                margin-top: 5px;
            }
        """)
        
        event_selector_layout = QHBoxLayout(event_selector_container)
        event_selector_layout.setContentsMargins(5, 5, 5, 5)
        
        event_type_label = QLabel("Event Detection Filter:")
        event_type_label.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #4B5563;
        """)
        event_selector_layout.addWidget(event_type_label)
        
        self.event_type_dropdown = QComboBox()
        self.event_type_dropdown.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 1px solid #D1D5DB;
                border-radius: 4px;
                padding: 5px;
                min-width: 200px;
                font-size: 14px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: right;
                width: 20px;
                border-left: 1px solid #D1D5DB;
            }
        """)
        
        # Add event type options
        self.event_type_dropdown.addItem("All Events", "all")
        self.event_type_dropdown.addItem("Theft Detection", "theft_detector")
        self.event_type_dropdown.addItem("Suspicious Movement", "suspicious_movement_detector")
        self.event_type_dropdown.addItem("Entrance/Exit", "entrance_exit_detector")
        
        self.event_type_dropdown.currentIndexChanged.connect(self.on_event_type_changed)
        event_selector_layout.addWidget(self.event_type_dropdown)
        
        self.layout.addWidget(event_selector_container)
        
        self.layout.addWidget(controls_container)
        
        # Initialize variables
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.setInterval(30)  # ~30fps
        
        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.is_camera = False
        self.current_frame = None
        self.frame_count = 0
        self.total_frames = 0
        self.fps = 25.0
        self.base_time = datetime.now()
        self.detection_count = 0
        
        # For backward functionality
        self.frame_buffer = []
        self.buffer_size = 30  # Buffer size for backward playback
        self.buffering_active = False
        self.buffered_frames = {}  # Dictionary to store buffered frames by position
        
        # ROI data
        self.roi_areas = {}
        self.roi_colors = {}
        
        # Track data
        self.tracks = {}
        
        # Initialize detector and tracker
        self.detector = YOLODetector()
        self.tracker = DeepSORTTracker()
        
        # Initialize event detector manager
        self.event_detector_manager = EventDetectorManager()
        
        # Register the active detector directly
        self.current_detector_type = "theft_detector"  # Default detector
        detector = DetectorFactory.create_detector(self.current_detector_type)
        if detector:
            self.event_detector_manager.register_detector(self.current_detector_type, detector)
            self.event_detector_manager.set_active_detector(self.current_detector_type)
        
        # Start async event processing
        self.event_detector_manager.register_event_callback(self.handle_detected_event)
        self.event_detector_manager.start_async_processing()
    
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
        """Load video file or camera stream."""
        self.stop_video()
        
        self.video_path = video_path
        self.is_camera = is_camera
        
        try:
            self.cap = cv2.VideoCapture(video_path)
            
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video source: {video_path}")
            
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 25.0  # Default FPS if not available
            
            if not is_camera:
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # Enable seek slider for videos
                self.seek_slider.setRange(0, self.total_frames)
                self.seek_slider.setEnabled(True)
                self.rewind_button.setEnabled(True)
                self.step_back_button.setEnabled(True)
                
                # Start frame buffering for backward playback
                self.buffering_active = True
                self.buffered_frames = {}
            else:
                self.total_frames = 0
                self.seek_slider.setEnabled(False)
                self.rewind_button.setEnabled(False)
                self.step_back_button.setEnabled(False)
                self.buffering_active = False
            
            # Reset frame count
            self.frame_count = 0
            
            # Reset base time
            self.base_time = datetime.now()
            
            # Get first frame
            ret, frame = self.cap.read()
            if not ret:
                raise ValueError("Could not read initial frame")
            
            # Display first frame
            self.display_frame(frame)
            
            # Store current frame
            self.current_frame = frame
            
            # Clear track data
            self.tracks = {}
            
            # Reset progress
            self.progress_bar.setValue(0)
            
            # Update frame counter
            self.frame_counter.setText(f"Frame: {self.frame_count} / {self.total_frames if self.total_frames > 0 else 'Live'}")
            
            # Update timestamp
            self.timestamp_label.setText(f"Time: {self.base_time.strftime('%H:%M:%S')}")
        
        except Exception as e:
            print(f"Error loading video: {e}")
            return False
        
        # Update UI
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.is_playing = False
        
        return True
    
    def set_roi_areas(self, roi_areas, roi_colors):
        """Set the ROI areas for processing."""
        self.roi_areas = roi_areas
        self.roi_colors = roi_colors
        
        # Update event detector manager with ROI information
        self.event_detector_manager.set_roi_areas(roi_areas, roi_colors)
        
        # Update display if we have a current frame
        if self.current_frame is not None:
            self.display_frame_with_annotations(self.current_frame)
    
    def handle_detected_event(self, event):
        """Handle events from the event detector manager."""
        # Make sure the event has the required fields
        required_fields = {'type', 'timestamp', 'description'}
        for field in required_fields:
            if field not in event:
                print(f"Warning: Event missing required field: {field}")
                if field == 'type':
                    event['type'] = 'Unknown Event'
                elif field == 'timestamp':
                    event['timestamp'] = datetime.now().strftime('%H:%M:%S')
                elif field == 'description':
                    event['description'] = f"Event detected from {event.get('detector', 'unknown source')}"
        
        # Add frame count if missing
        if 'frame_idx' not in event:
            event['frame_idx'] = self.frame_count
            
        # Add thumbnail if missing
        if 'thumbnail' not in event and self.current_frame is not None:
            event['thumbnail'] = get_frame_thumbnail(self.current_frame)
            
        # Print event info for debugging
        print(f"Detected event: {event['type']} - {event['description']}")
            
        # Emit the event to be captured by external components
        self.event_detected.emit(event)
            
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
            
        # Stop async event processing
        if hasattr(self, 'event_detector_manager'):
            self.event_detector_manager.stop_async_processing()
    
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
        
        # Buffer this frame for backward playback if enabled
        if self.buffering_active and not self.is_camera:
            self.buffered_frames[self.frame_count] = frame.copy()
            
            # Keep buffer size manageable by removing old frames
            keys = sorted(list(self.buffered_frames.keys()))
            if len(keys) > self.buffer_size:
                oldest_key = keys[0]
                del self.buffered_frames[oldest_key]
        
        # Generate timestamp based on frame count and FPS
        seconds = self.frame_count / self.fps
        timestamp = self.base_time + timedelta(seconds=seconds)
        
        # Process frame with detector and tracker
        try:
            # Run object detection
            detections = self.detector.detect(frame)
            
            # Update tracker
            tracks_updated = self.tracker.update(frame, detections)
            
            # Process events using the event detector manager
            core_events = self.event_detector_manager.process_frame(
                frame.copy(), tracks_updated, timestamp, self.frame_count
            )
            
            # Log events generated directly from process_frame
            for event in core_events:
                self.handle_detected_event(event)
            
            # Update detection count
            self.detection_count = len(detections)
            self.detection_counter.setText(f"Detections: {self.detection_count}")
            
            # Draw annotations
            processed_frame = frame.copy()
            
            # Draw tracked objects
            processed_frame = self.draw_annotations(processed_frame, tracks_updated, timestamp)
            
            # Display frame
            self.display_frame_with_annotations(processed_frame)
            
            # Update progress bar for video files
            if self.total_frames > 0:
                progress = int((self.frame_count / self.total_frames) * 100)
                self.progress_bar.setValue(progress)
                
                # Update slider position (without triggering valueChanged)
                self.seek_slider.blockSignals(True)
                self.seek_slider.setValue(self.frame_count)
                self.seek_slider.blockSignals(False)
            
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
        
        # Apply annotations from event detectors
        for detector_id in self.event_detector_manager.enabled_detectors:
            detector = self.event_detector_manager.detectors.get(detector_id)
            if detector and hasattr(detector, 'annotate_frame'):
                try:
                    frame = detector.annotate_frame(frame)
                except Exception as e:
                    print(f"Error annotating frame with detector {detector_id}: {e}")
        
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
    
    def step_back(self):
        """Step back one frame in the video."""
        if self.is_camera or not self.cap or not self.cap.isOpened():
            return
        
        # Stop playback
        was_playing = self.is_playing
        if self.is_playing:
            self.toggle_play()
        
        # Calculate target frame (current - 2 because we just processed current - 1)
        target_frame = max(0, self.frame_count - 2)
        
        # Check if we have this frame in buffer
        if target_frame in self.buffered_frames:
            # Use buffered frame directly
            frame = self.buffered_frames[target_frame]
            self.frame_count = target_frame
            
            # Generate timestamp
            seconds = self.frame_count / self.fps
            timestamp = self.base_time + timedelta(seconds=seconds)
            
            # Process and display frame
            detections = self.detector.detect(frame)
            tracks_updated = self.tracker.update(frame, detections)
            processed_frame = self.draw_annotations(frame.copy(), tracks_updated, timestamp)
            self.display_frame_with_annotations(processed_frame)
            
            # Update counters and slider
            self.detection_count = len(detections)
            self.detection_counter.setText(f"Detections: {self.detection_count}")
            self.frame_counter.setText(f"Frame: {self.frame_count} / {self.total_frames}")
            self.timestamp_label.setText(f"Time: {timestamp.strftime('%H:%M:%S')}")
            
            # Update slider
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(self.frame_count)
            self.seek_slider.blockSignals(False)
            
            # Store current frame
            self.current_frame = frame
        else:
            # Need to seek in the video
            self.seek_video(target_frame)
            
        # Resume playback if it was playing
        if was_playing:
            self.toggle_play()

    def rewind_video(self):
        """Rewind video to beginning."""
        if self.is_camera or not self.cap or not self.cap.isOpened():
            return
        
        # Stop playback
        was_playing = self.is_playing
        if self.is_playing:
            self.toggle_play()
        
        # Seek to the beginning of the video
        self.seek_video(0)
        
        # Resume playback if it was playing
        if was_playing:
            self.toggle_play()

    def seek_video(self, position):
        """Seek to a specific position in the video."""
        if self.is_camera or not self.cap or not self.cap.isOpened():
            return
        
        # Ensure position is an integer
        position = int(position)
        
        # Clamp position to valid range
        position = max(0, min(position, self.total_frames))
        
        # Set position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        
        # Update frame count
        self.frame_count = position
        
        # Read and display the frame at the seeked position
        ret, frame = self.cap.read()
        if ret:
            # Generate timestamp
            seconds = self.frame_count / self.fps
            timestamp = self.base_time + timedelta(seconds=seconds)
            
            # Process and display frame
            detections = self.detector.detect(frame)
            tracks_updated = self.tracker.update(frame, detections)
            processed_frame = self.draw_annotations(frame.copy(), tracks_updated, timestamp)
            self.display_frame_with_annotations(processed_frame)
            
            # Update counters
            self.detection_count = len(detections)
            self.detection_counter.setText(f"Detections: {self.detection_count}")
            self.frame_counter.setText(f"Frame: {self.frame_count} / {self.total_frames}")
            self.timestamp_label.setText(f"Time: {timestamp.strftime('%H:%M:%S')}")
            
            # Update progress bar
            progress = int((position / self.total_frames) * 100)
            self.progress_bar.setValue(progress)
            
            # Store current frame
            self.current_frame = frame
        else:
            print("Error seeking to frame")

    def on_event_type_changed(self, index):
        """Handle changes to the event type dropdown."""
        selected_data = self.event_type_dropdown.currentData()
        print(f"Selected event type: {selected_data}")
        
        # Show user feedback about the change
        self.timestamp_label.setText(f"Event filter: {self.event_type_dropdown.currentText()}")
        self.timestamp_label.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #3B82F6;
        """)
        # Reset style after 2 seconds
        QTimer.singleShot(2000, self.reset_timestamp_style)

    def set_detector_type(self, detector_type):
        """Set the active detector type."""
        # Skip if it's already the current detector
        if detector_type == self.current_detector_type:
            return True
            
        # Create the new detector
        detector = DetectorFactory.create_detector(detector_type)
        if not detector:
            return False
            
        # Register the new detector
        self.event_detector_manager.register_detector(detector_type, detector)
        
        # Set it as active
        success = self.event_detector_manager.set_active_detector(detector_type)
        if success:
            self.current_detector_type = detector_type
            
            # Set ROI areas for new detector
            if self.roi_areas:
                detector.set_roi_areas(self.roi_areas, self.roi_colors)
                
            # Update UI to show the current detector
            self.update_detector_ui()
            
        return success
    
    def update_detector_ui(self):
        """Update UI to reflect the current detector type."""
        # Get detector info
        detector_info = DetectorFactory.get_detector_info(self.current_detector_type)
        
        # Update the event dropdown to show events for this detector
        self.event_type_dropdown.blockSignals(True)
        self.event_type_dropdown.clear()
        
        # Add "All Events" option first
        self.event_type_dropdown.addItem("All Events", "all")
        
        # Add specific event types for this detector
        for event_type in detector_info.get("events", []):
            self.event_type_dropdown.addItem(event_type, event_type)
            
        self.event_type_dropdown.blockSignals(False)
        
        # Show current detector in UI
        self.timestamp_label.setText(f"Active detector: {detector_info.get('name', 'Unknown')}")
        self.timestamp_label.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #3B82F6;
        """)
        # Reset style after 2 seconds
        QTimer.singleShot(2000, self.reset_timestamp_style) 