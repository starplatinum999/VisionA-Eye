import cv2
import numpy as np
import uuid
import json
import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QListWidget, QListWidgetItem, QInputDialog, QColorDialog, 
    QMessageBox, QFileDialog, QComboBox, QFrame, QGroupBox,
    QScrollArea, QSizePolicy, QDialog, QFormLayout, QLineEdit
)
from PySide6.QtCore import Qt, Signal, QPoint, QRect, QSize, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent, QFont, QIcon

class ROIWidget(QWidget):
    """Widget for defining and managing Regions of Interest (ROIs)."""
    
    # Signal to notify that ROI areas have been updated
    roi_updated = Signal(dict, dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # ROI data
        self.roi_areas = {}  # {name: (x1, y1, x2, y2)}
        self.roi_colors = {}  # {name: (r, g, b)}
        
        # Current ROI drawing state
        self.drawing = False
        self.roi_start = None
        self.roi_end = None
        self.current_roi_name = None
        
        # For quadrilateral ROI creation
        self.quad_mode = False
        self.quad_points = []
        self.roi_points = {}  # {name: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]}
        
        # Frame data
        self.frame = None
        self.display_frame = None
        
        # Drawing performance optimization
        self.draw_timer = QTimer()
        self.draw_timer.setSingleShot(True)
        self.draw_timer.timeout.connect(self.update_display)
        
        # Status message
        self.status_message = "No video loaded"
        
        # UI setup
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        self.setLayout(main_layout)
        
        # Left side - Frame display and drawing area
        left_container = QFrame()
        left_container.setProperty("class", "card")
        left_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(15, 15, 15, 15)
        left_layout.setSpacing(15)
        
        # Header for ROI area
        header = QLabel("Define Regions of Interest")
        header.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            color: #1F2937;
            margin-bottom: 10px;
        """)
        left_layout.addWidget(header)
        
        # Drawing mode selection
        drawing_mode_layout = QHBoxLayout()
        drawing_mode_layout.setSpacing(10)
        
        self.rect_mode_button = QPushButton("Drag image")
        self.rect_mode_button.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border-radius: 6px;
                padding: 8px 15px;
                font-weight: 600;
            }
            QPushButton:checked {
                background-color: #1D4ED8;
            }
        """)
        self.rect_mode_button.setCheckable(True)
        self.rect_mode_button.setChecked(True)
        self.rect_mode_button.clicked.connect(lambda: self.set_drawing_mode(False))
        drawing_mode_layout.addWidget(self.rect_mode_button)
        
        self.quad_mode_button = QPushButton("From points")
        self.quad_mode_button.setStyleSheet("""
            QPushButton {
                background-color: #8B5CF6;
                color: white;
                border-radius: 6px;
                padding: 8px 15px;
                font-weight: 600;
            }
            QPushButton:checked {
                background-color: #6D28D9;
            }
        """)
        self.quad_mode_button.setCheckable(True)
        self.quad_mode_button.clicked.connect(lambda: self.set_drawing_mode(True))
        drawing_mode_layout.addWidget(self.quad_mode_button)
        
        left_layout.addLayout(drawing_mode_layout)
        
        # Frame display label with a nice border
        frame_container = QFrame()
        frame_container.setStyleSheet("""
            border: 2px solid #E5E7EB;
            border-radius: 8px;
            background-color: #F9FAFB;
            padding: 5px;
        """)
        frame_layout = QVBoxLayout(frame_container)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        
        self.frame_label = QLabel("No video loaded")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setMinimumSize(640, 480)
        self.frame_label.setStyleSheet("""
            background-color: #111827;
            color: #9CA3AF;
            font-size: 16px;
            border-radius: 6px;
        """)
        self.frame_label.mousePressEvent = self.on_mouse_press
        self.frame_label.mouseMoveEvent = self.on_mouse_move
        self.frame_label.mouseReleaseEvent = self.on_mouse_release
        
        frame_layout.addWidget(self.frame_label)
        left_layout.addWidget(frame_container)
        
        # Drawing instructions with better styling
        instructions_box = QFrame()
        instructions_box.setStyleSheet("""
            background-color: #EFF6FF;
            border: 1px solid #BFDBFE;
            border-radius: 8px;
            padding: 10px;
        """)
        instructions_layout = QHBoxLayout(instructions_box)
        
        # Info icon (placeholder)
        info_icon = QLabel("ℹ️")
        info_icon.setStyleSheet("font-size: 24px; min-width: 30px;")
        instructions_layout.addWidget(info_icon)
        
        instructions = QLabel(
            "<b>How to create ROIs:</b> Click the 'Add ROI' button and draw by clicking and dragging. "
            "Name your region when prompted. Right-click to cancel drawing. "
            "Use the ROI list to manage your regions."
        )
        instructions.setStyleSheet("color: #1E40AF; font-size: 14px;")
        instructions.setWordWrap(True)
        instructions_layout.addWidget(instructions)
        
        left_layout.addWidget(instructions_box)
        
        # Drawing controls with modern buttons
        drawing_controls = QHBoxLayout()
        drawing_controls.setSpacing(10)
        
        self.add_roi_button = QPushButton("Add ROI")
        self.add_roi_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4F46E5, stop:1 #4338CA);
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6366F1, stop:1 #4F46E5);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4338CA, stop:1 #3730A3);
            }
        """)
        self.add_roi_button.clicked.connect(self.start_roi_drawing)
        drawing_controls.addWidget(self.add_roi_button)
        
        self.cancel_roi_button = QPushButton("Cancel Drawing")
        self.cancel_roi_button.setStyleSheet("""
            QPushButton {
                background-color: #F87171;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 600;
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
        self.cancel_roi_button.clicked.connect(self.cancel_roi_drawing)
        self.cancel_roi_button.setEnabled(False)
        drawing_controls.addWidget(self.cancel_roi_button)
        
        self.finish_button = QPushButton("Finish Configuration")
        self.finish_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #059669, stop:1 #047857);
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #10B981, stop:1 #059669);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #047857, stop:1 #065F46);
            }
        """)
        self.finish_button.clicked.connect(self.finish_roi_configuration)
        drawing_controls.addWidget(self.finish_button)
        
        left_layout.addLayout(drawing_controls)
        
        # Status bar for user guidance
        self.status_bar = QLabel(self.status_message)
        self.status_bar.setStyleSheet("""
            background-color: #F3F4F6;
            color: #4B5563;
            font-size: 14px;
            font-weight: 500;
            padding: 10px;
            border-radius: 6px;
            border-left: 4px solid #3B82F6;
        """)
        self.status_bar.setWordWrap(True)
        left_layout.addWidget(self.status_bar)
        
        # Right side - ROI list and management
        right_container = QFrame()
        right_container.setProperty("class", "card")
        right_container.setMaximumWidth(350)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(15, 15, 15, 15)
        right_layout.setSpacing(15)
        
        # ROI list section
        roi_list_group = QGroupBox("Defined Regions")
        roi_list_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                margin-top: 16px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #4B5563;
            }
        """)
        roi_list_layout = QVBoxLayout(roi_list_group)
        
        # Enhanced ROI list widget
        self.roi_list = QListWidget()
        self.roi_list.setMinimumHeight(200)
        self.roi_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                background-color: #F9FAFB;
                padding: 5px;
                font-size: 14px;
            }
            QListWidget::item {
                border-bottom: 1px solid #F3F4F6;
                padding: 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #EFF6FF;
                color: #2563EB;
                border-left: 3px solid #2563EB;
            }
            QListWidget::item:hover:!selected {
                background-color: #F3F4F6;
            }
        """)
        self.roi_list.itemClicked.connect(self.on_roi_selected)
        roi_list_layout.addWidget(self.roi_list)
        
        # ROI management buttons
        roi_buttons = QHBoxLayout()
        roi_buttons.setSpacing(10)
        
        self.edit_roi_button = QPushButton("Edit ROI")
        self.edit_roi_button.setStyleSheet("""
            QPushButton {
                background-color: #60A5FA;
                color: white;
                border-radius: 6px;
                padding: 8px 15px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3B82F6;
            }
            QPushButton:pressed {
                background-color: #2563EB;
            }
            QPushButton:disabled {
                background-color: #D1D5DB;
                color: #9CA3AF;
            }
        """)
        self.edit_roi_button.clicked.connect(self.edit_selected_roi)
        self.edit_roi_button.setEnabled(False)
        roi_buttons.addWidget(self.edit_roi_button)
        
        # Add Properties button for editing name and color
        self.properties_button = QPushButton("Properties")
        self.properties_button.setStyleSheet("""
            QPushButton {
                background-color: #A855F7;
                color: white;
                border-radius: 6px;
                padding: 8px 15px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #9333EA;
            }
            QPushButton:pressed {
                background-color: #7E22CE;
            }
            QPushButton:disabled {
                background-color: #D1D5DB;
                color: #9CA3AF;
            }
        """)
        self.properties_button.clicked.connect(self.edit_roi_properties)
        self.properties_button.setEnabled(False)
        roi_buttons.addWidget(self.properties_button)
        
        self.delete_roi_button = QPushButton("Delete ROI")
        self.delete_roi_button.setStyleSheet("""
            QPushButton {
                background-color: #F87171;
                color: white;
                border-radius: 8px;
                padding: 8px 15px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #EF4444;
            }
            QPushButton:pressed {
                background-color: #DC2626;
            }
        """)
        self.delete_roi_button.clicked.connect(self.delete_selected_roi)
        roi_buttons.addWidget(self.delete_roi_button)
        
        roi_list_layout.addLayout(roi_buttons)
        right_layout.addWidget(roi_list_group)
        
        # ROI file management section
        file_group = QGroupBox("Save & Load")
        file_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                margin-top: 16px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #4B5563;
            }
        """)
        file_layout = QVBoxLayout(file_group)
        
        # Enhanced file buttons
        roi_file_buttons = QHBoxLayout()
        roi_file_buttons.setSpacing(10)
        
        self.save_roi_button = QPushButton("Save ROIs")
        self.save_roi_button.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: white;
                border-radius: 8px;
                padding: 8px 15px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:pressed {
                background-color: #047857;
            }
        """)
        self.save_roi_button.clicked.connect(self.save_roi_config)
        roi_file_buttons.addWidget(self.save_roi_button)
        
        self.load_roi_button = QPushButton("Load ROIs")
        self.load_roi_button.setStyleSheet("""
            QPushButton {
                background-color: #8B5CF6;
                color: white;
                border-radius: 8px;
                padding: 8px 15px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #7C3AED;
            }
            QPushButton:pressed {
                background-color: #6D28D9;
            }
        """)
        self.load_roi_button.clicked.connect(self.load_roi_config)
        roi_file_buttons.addWidget(self.load_roi_button)
        
        file_layout.addLayout(roi_file_buttons)
        right_layout.addWidget(file_group)
        
        # ROI preset section with improved styling
        presets_group = QGroupBox("Quick Presets")
        presets_group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                margin-top: 16px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #4B5563;
            }
        """)
        presets_layout = QVBoxLayout(presets_group)
        
        preset_label = QLabel("Select a predefined ROI layout:")
        preset_label.setStyleSheet("color: #6B7280; font-size: 14px;")
        presets_layout.addWidget(preset_label)
        
        self.preset_combo = QComboBox()
        self.preset_combo.setStyleSheet("""
            QComboBox {
                background-color: #F9FAFB;
                border: 1px solid #D1D5DB;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }
            QComboBox:focus {
                border-color: #3B82F6;
                border-width: 2px;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
        """)
        self.preset_combo.addItems([
            "Retail Store Layout",
            "Checkout Counters",
            "Entrance/Exit",
            "Shelf Areas"
        ])
        presets_layout.addWidget(self.preset_combo)
        
        apply_preset_button = QPushButton("Apply Preset")
        apply_preset_button.setStyleSheet("""
            QPushButton {
                background-color: #F59E0B;
                color: white;
                border-radius: 8px;
                padding: 8px 15px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #D97706;
            }
            QPushButton:pressed {
                background-color: #B45309;
            }
        """)
        apply_preset_button.clicked.connect(self.apply_roi_preset)
        presets_layout.addWidget(apply_preset_button)
        
        right_layout.addWidget(presets_group)
        right_layout.addStretch()
        
        # Add containers to main layout
        main_layout.addWidget(left_container, 3)  # 3:1 ratio
        main_layout.addWidget(right_container, 1)
        
        # Update button states
        self.update_button_states()
    
    def set_status(self, message, is_error=False):
        """Update the status message with visual indication of severity."""
        self.status_message = message
        
        # Set style based on the type of message
        if is_error:
            self.status_bar.setStyleSheet("""
                background-color: #FEF2F2;
                color: #B91C1C;
                font-size: 14px;
                font-weight: 500;
                padding: 10px;
                border-radius: 6px;
                border-left: 4px solid #EF4444;
            """)
        else:
            self.status_bar.setStyleSheet("""
                background-color: #F3F4F6;
                color: #4B5563;
                font-size: 14px;
                font-weight: 500;
                padding: 10px;
                border-radius: 6px;
                border-left: 4px solid #3B82F6;
            """)
            
        # Update label text
        self.status_bar.setText(message)
    
    def load_frame(self, frame):
        """Load a frame for ROI definition."""
        if frame is None:
            self.set_status("Error: No frame provided", is_error=True)
            return
        
        # Store a deep copy of the frame to avoid reference issues
        self.frame = frame.copy()
        
        # Update the label text to empty since we now have a frame
        self.frame_label.setText("")
        
        # Update the display immediately
        self.update_display()
        
        # Update button states now that we have a frame
        self.update_button_states()
        
        # Update status
        self.set_status("Frame loaded. Click 'Add ROI' to begin drawing regions.")
        
        # Show a notification to inform the user
        QMessageBox.information(self, "ROI Configuration Ready", 
                             "Video frame loaded. You can now define Regions of Interest (ROIs).\n\n"
                             "1. Click 'Add ROI' to start drawing\n"
                             "2. Name your region and select a color\n"
                             "3. Click and drag on the image to define the area")
    
    def update_display(self):
        """Update the displayed frame with ROI overlays."""
        if self.frame is None:
            return
        
        # Make a copy of the original frame
        self.display_frame = self.frame.copy()
        
        # Get currently selected ROI name
        selected_roi = None
        if self.roi_list.currentItem():
            selected_roi = self.roi_list.currentItem().text()
        
        # Draw existing ROIs
        for name, (x1, y1, x2, y2) in self.roi_areas.items():
            # Determine color and thickness based on selection status
            color = self.roi_colors.get(name, (255, 0, 0))  # Default blue
            thickness = 3 if name == selected_roi else 2
            
            if name in self.roi_points:
                # Draw the quadrilateral shape using stored points
                points = np.array(self.roi_points[name], np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.polylines(self.display_frame, [points], True, color, thickness)
                
                # Add a filled semi-transparent polygon
                overlay = self.display_frame.copy()
                cv2.fillPoly(overlay, [points], color)
                alpha = 0.2  # Transparency factor
                cv2.addWeighted(overlay, alpha, self.display_frame, 1 - alpha, 0, self.display_frame)
                
                # Use first point as the label position
                label_x, label_y = self.roi_points[name][0]
            else:
                # Draw rectangle and label for rectangular ROI
                cv2.rectangle(self.display_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Add a filled rectangle behind the text for better readability
                text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(
                    self.display_frame, 
                    (x1, y1 - text_size[1] - 10), 
                    (x1 + text_size[0] + 10, y1), 
                    color, 
                    -1  # Filled rectangle
                )
                
                label_x, label_y = x1, y1
            
            # Add text with white color for contrast
            cv2.putText(
                self.display_frame, 
                name, 
                (label_x + 5, label_y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255),  # White text
                2
            )
            
            # Add highlight border for selected ROI
            if name == selected_roi:
                # Draw a double border with a different color
                highlight_color = (0, 255, 255)  # Yellow
                if name in self.roi_points:
                    # Highlight the quadrilateral
                    points = np.array(self.roi_points[name], np.int32)
                    points = points.reshape((-1, 1, 2))
                    cv2.polylines(self.display_frame, [points], True, highlight_color, 1)
                else:
                    # Highlight the rectangle
                    cv2.rectangle(
                        self.display_frame, 
                        (x1 - 2, y1 - 2), 
                        (x2 + 2, y2 + 2), 
                        highlight_color, 
                        1
                    )
        
        # Draw currently being defined ROI with improved visual feedback
        if self.drawing:
            if self.quad_mode and self.quad_points:
                # Draw the points collected so far
                for i, (x, y) in enumerate(self.quad_points):
                    # Draw the point
                    cv2.circle(self.display_frame, (x, y), 5, (255, 0, 0), -1)
                    cv2.putText(self.display_frame, f"P{i+1}", (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Draw lines between consecutive points
                for i in range(len(self.quad_points)):
                    if i > 0:  # Draw line from previous point to current point
                        cv2.line(
                            self.display_frame, 
                            self.quad_points[i-1], 
                            self.quad_points[i], 
                            self.roi_colors.get(self.current_roi_name, (0, 255, 0)), 
                            2
                        )
                
                # If we have at least 3 points, close the polygon to the first point
                if len(self.quad_points) >= 3:
                    cv2.line(
                        self.display_frame, 
                        self.quad_points[-1], 
                        self.quad_points[0], 
                        self.roi_colors.get(self.current_roi_name, (0, 255, 0)), 
                        2, 
                        cv2.LINE_AA  # Use anti-aliased line instead of dashed line
                    )
            elif not self.quad_mode and self.roi_start and self.roi_end:
                # Draw the rectangular ROI being created
                x1, y1 = self.roi_start.x(), self.roi_start.y()
                x2, y2 = self.roi_end.x(), self.roi_end.y()
                
                # Normalize coordinates
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                    
                # Draw with a prominent style for better visibility
                # Draw a semi-transparent fill for better visibility
                overlay = self.display_frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)  # Filled rectangle
                cv2.addWeighted(overlay, 0.3, self.display_frame, 0.7, 0, self.display_frame)
                
                # Draw border
                cv2.rectangle(self.display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Show dimensions in pixels for better user feedback
                width = x2 - x1
                height = y2 - y1
                dim_text = f"{width}x{height}px"
                
                # Add dimension text at the top right corner of the rectangle
                cv2.putText(
                    self.display_frame,
                    dim_text,
                    (x2 - 10, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
                
                if self.current_roi_name:
                    # Add label with background
                    text_size = cv2.getTextSize(self.current_roi_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(
                        self.display_frame, 
                        (x1, y1 - text_size[1] - 10), 
                        (x1 + text_size[0] + 10, y1), 
                        (0, 255, 0), 
                        -1
                    )
                    
                    cv2.putText(
                        self.display_frame, 
                        self.current_roi_name, 
                        (x1 + 5, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (255, 255, 255), 
                        2
                    )
                    
                # Draw corner markers for better visibility
                corner_size = 5
                cv2.line(self.display_frame, (x1, y1), (x1 + corner_size, y1), (0, 255, 255), 2)
                cv2.line(self.display_frame, (x1, y1), (x1, y1 + corner_size), (0, 255, 255), 2)
                
                cv2.line(self.display_frame, (x2, y1), (x2 - corner_size, y1), (0, 255, 255), 2)
                cv2.line(self.display_frame, (x2, y1), (x2, y1 + corner_size), (0, 255, 255), 2)
                
                cv2.line(self.display_frame, (x1, y2), (x1 + corner_size, y2), (0, 255, 255), 2)
                cv2.line(self.display_frame, (x1, y2), (x1, y2 - corner_size), (0, 255, 255), 2)
                
                cv2.line(self.display_frame, (x2, y2), (x2 - corner_size, y2), (0, 255, 255), 2)
                cv2.line(self.display_frame, (x2, y2), (x2, y2 - corner_size), (0, 255, 255), 2)
        
        # Convert to Qt format and display
        self.show_cv_image(self.display_frame)
    
    def show_cv_image(self, img):
        """Display an OpenCV image on the QLabel."""
        if img is None:
            return
        
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale pixmap to fit label while preserving aspect ratio
        self.frame_label.setPixmap(pixmap.scaled(
            self.frame_label.width(), self.frame_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
    
    def set_drawing_mode(self, quad_mode):
        """Set the drawing mode between rectangular and quadrilateral ROIs."""
        self.quad_mode = quad_mode
        
        # Update button states
        self.rect_mode_button.setChecked(not quad_mode)
        self.quad_mode_button.setChecked(quad_mode)
        
        # Reset any ongoing drawing
        if self.drawing:
            self.cancel_roi_drawing()
        
        if quad_mode:
            self.set_status("Quadrilateral ROI mode: Select 4 corners to create a custom shape")
        else:
            self.set_status("Rectangle ROI mode: Click and drag to create a rectangular region")
    
    def start_roi_drawing(self):
        """Start the ROI drawing process."""
        if self.frame is None:
            self.set_status("No video loaded. Please load a video first.", is_error=True)
            QMessageBox.warning(self, "No Frame", "Please load a video first.")
            return
        
        # Get ROI name from user with a more helpful dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Create New Region of Interest")
        dialog.setMinimumWidth(400)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
                border-radius: 8px;
            }
            QLabel {
                color: #1F2937;
                font-size: 14px;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton {
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton#confirmButton {
                background-color: #3B82F6;
                color: white;
            }
            QPushButton#cancelButton {
                background-color: #F3F4F6;
                color: #4B5563;
                border: 1px solid #D1D5DB;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Help text
        help_label = QLabel("Create a new region to monitor specific areas in your video.")
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        
        # Name input
        form_layout = QFormLayout()
        form_layout.setSpacing(12)
        
        name_input = QLineEdit()
        name_input.setPlaceholderText("e.g., Entrance, Checkout, Shelf-A")
        form_layout.addRow("Region Name:", name_input)
        layout.addLayout(form_layout)
        
        # Information about next steps
        if self.quad_mode:
            next_label = QLabel("Next, you'll select a color and then click on 4 points in the image to create a quadrilateral region.")
        else:
            next_label = QLabel("Next, you'll select a color and draw the region by clicking and dragging on the video frame.")
        
        next_label.setWordWrap(True)
        next_label.setStyleSheet("color: #6B7280; font-size: 13px; margin-top: 4px;")
        layout.addWidget(next_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.setObjectName("cancelButton")
        cancel_button.clicked.connect(dialog.reject)
        
        confirm_button = QPushButton("Next: Choose Color")
        confirm_button.setObjectName("confirmButton")
        confirm_button.clicked.connect(dialog.accept)
        
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(confirm_button)
        layout.addLayout(button_layout)
        
        # Execute dialog
        if dialog.exec_() != QDialog.Accepted:
            return
        
        name = name_input.text().strip()
        if not name:
            self.set_status("ROI creation canceled: No name provided", is_error=True)
            return
        
        # Check for duplicate names
        if name in self.roi_areas:
            self.set_status(f"An ROI named '{name}' already exists. Please use a different name.", is_error=True)
            QMessageBox.warning(self, "Duplicate Name", f"An ROI named '{name}' already exists.")
            return
        
        # Get ROI color from user with a custom dialog that shows color presets
        color_dialog = QColorDialog(self)
        color_dialog.setWindowTitle("Select Color for " + name)
        color_dialog.setOption(QColorDialog.ShowAlphaChannel, False)
        
        # Set some standard presets for common region types
        if "entrance" in name.lower() or "entry" in name.lower():
            color_dialog.setCurrentColor(QColor(0, 255, 0))  # Green for entrances
        elif "exit" in name.lower():
            color_dialog.setCurrentColor(QColor(255, 0, 0))  # Red for exits
        elif "checkout" in name.lower() or "cash" in name.lower():
            color_dialog.setCurrentColor(QColor(255, 165, 0))  # Orange for checkout
        elif "shelf" in name.lower() or "aisle" in name.lower():
            color_dialog.setCurrentColor(QColor(0, 0, 255))  # Blue for shelves
        else:
            color_dialog.setCurrentColor(QColor(127, 0, 255))  # Purple default
            
        if color_dialog.exec_():
            qcolor = color_dialog.selectedColor()
            color = (qcolor.blue(), qcolor.green(), qcolor.red())  # BGR for OpenCV
        else:
            # User canceled
            self.set_status("ROI creation canceled: No color selected", is_error=True)
            return
        
        # Set up drawing state
        self.roi_colors[name] = color
        self.current_roi_name = name
        self.drawing = True
        
        if self.quad_mode:
            # Clear any existing points for quadrilateral mode
            self.quad_points = []
            self.set_status(f"Drawing '{name}': Click on 4 points to create a quadrilateral (0/4 points)")
        else:
            # For rectangle mode
            self.roi_start = None
            self.roi_end = None
            self.set_status(f"Drawing ROI '{name}': Click and drag on the image to define the area")
        
        # Update UI
        self.update_button_states()
        self.setCursor(Qt.CrossCursor)  # Set crosshair cursor for drawing
    
    def on_mouse_press(self, event: QMouseEvent):
        """Handle mouse press events for ROI drawing."""
        if not self.drawing or self.frame is None:
            return
        
        # Get position relative to the image
        pos = self.get_image_position(event.position())
        if not pos:
            print("Debug - on_mouse_press: Invalid position (outside image bounds)")
            # Show a warning or just return
            self.set_status("Click position outside image bounds. Please click within the image area.", is_error=True)
            return
            
        if self.quad_mode:
            # In quad mode, each click adds a point to the quadrilateral
            if len(self.quad_points) < 4:
                # Log the exact coordinates for debugging
                print(f"Debug - Adding point {len(self.quad_points)+1}: ({pos.x()}, {pos.y()})")
                
                # Store the point
                self.quad_points.append((pos.x(), pos.y()))
                
                # Update status with point information
                points_str = ", ".join([f"({x}, {y})" for x, y in self.quad_points])
                self.set_status(f"Drawing '{self.current_roi_name}': Point {len(self.quad_points)}/4 added at ({pos.x()}, {pos.y()}). Points: [{points_str}]")
                
                # If we have 4 points, complete the ROI
                if len(self.quad_points) == 4:
                    self.complete_quad_roi()
                
                self.update_display()
        else:
            # In rectangle mode, start dragging
            self.roi_start = pos
            self.roi_end = pos
            self.set_status(f"Drawing ROI: {self.current_roi_name} - Click and drag to define area")
            self.update_display()
    
    def on_mouse_move(self, event: QMouseEvent):
        """Handle mouse move events for ROI drawing."""
        if not self.drawing or self.frame is None:
            return
            
        # Get position relative to the image
        pos = self.get_image_position(event.position())
        if not pos:
            return
            
        # Only update in rectangle mode during dragging
        if not self.quad_mode and self.roi_start:
            self.roi_end = pos
            
            # Throttle updates for better performance
            if not self.draw_timer.isActive():
                self.draw_timer.start(30)  # Update at most every 30ms (about 33fps)
    
    def on_mouse_release(self, event: QMouseEvent):
        """Handle mouse release events for ROI drawing."""
        if not self.drawing or not self.roi_start or not self.roi_end or self.frame is None:
            return
        
        # Get position relative to the image
        pos = self.get_image_position(event.position())
        if not pos:
            return
        
        self.roi_end = pos
        
        # Normalize coordinates (ensure x1,y1 is top-left and x2,y2 is bottom-right)
        x1 = min(self.roi_start.x(), self.roi_end.x())
        y1 = min(self.roi_start.y(), self.roi_end.y())
        x2 = max(self.roi_start.x(), self.roi_end.x())
        y2 = max(self.roi_start.y(), self.roi_end.y())
        
        # Ensure coordinates are valid
        x1, y1, x2, y2 = self.validate_roi_coordinates(x1, y1, x2, y2)
        
        # Ensure minimum size
        if x2 - x1 < 20 or y2 - y1 < 20:
            self.set_status("ROI too small. Please draw a larger area (at least 20x20 pixels).", is_error=True)
            QMessageBox.warning(self, "ROI Too Small", 
                              "The ROI is too small. Please draw a larger area (at least 20x20 pixels).")
            return
        
        # Add the ROI
        self.roi_areas[self.current_roi_name] = (x1, y1, x2, y2)
        
        # Update list and display
        self.update_roi_list()
        
        # Reset drawing state
        self.drawing = False
        self.roi_start = None
        self.roi_end = None
        self.current_roi_name = None
        
        # Reset cursor
        self.unsetCursor()
        
        # Update UI and status
        self.update_display()
        self.update_button_states()
        self.set_status(f"ROI added successfully! ({len(self.roi_areas)} regions defined)")
        
        # Emit signal with updated ROIs
        self.roi_updated.emit(self.roi_areas, self.roi_colors)
        
        # Show a success animation or highlight
        self.highlight_success()
    
    def highlight_success(self):
        """Show a brief visual success indication."""
        # Flash the status bar green briefly to indicate success
        original_style = self.status_bar.styleSheet()
        
        # Set success style
        self.status_bar.setStyleSheet("""
            background-color: #ECFDF5;
            color: #047857;
            font-size: 14px;
            font-weight: 500;
            padding: 10px;
            border-radius: 6px;
            border-left: 4px solid #10B981;
        """)
        
        # Reset style after 1.5 seconds
        QTimer.singleShot(1500, lambda: self.status_bar.setStyleSheet(original_style))
    
    def cancel_roi_drawing(self):
        """Cancel the current ROI drawing."""
        self.drawing = False
        self.roi_start = None
        self.roi_end = None
        
        # Remove provisional color if we cancel before completing
        if self.current_roi_name and self.current_roi_name not in self.roi_areas:
            if self.current_roi_name in self.roi_colors:
                del self.roi_colors[self.current_roi_name]
        
        self.current_roi_name = None
        
        # Reset cursor
        self.unsetCursor()
        
        # Update display and button states
        self.update_display()
        self.update_button_states()
        self.set_status("ROI drawing canceled")
    
    def on_roi_selected(self, item):
        """Handle ROI selection from the list."""
        # Highlight the selected ROI on the display
        self.update_display()
        
        # Update button states
        self.update_button_states()
        
        # Debug information
        roi_name = item.text() if item else "None"
        print(f"ROI selected: {roi_name}")
    
    def edit_selected_roi(self):
        """Edit the currently selected ROI."""
        if self.drawing:
            return
            
        item = self.roi_list.currentItem()
        if not item:
            return
        
        roi_name = item.text()
        if roi_name not in self.roi_areas:
            return
        
        # Show instructions to the user
        QMessageBox.information(self, "Edit ROI", 
                              f"You are editing the ROI '{roi_name}'.\n\n"
                              "1. Click and drag on the image to redefine the area\n"
                              "2. Right-click to cancel editing")
        
        # Set up editing mode
        self.drawing = True
        self.current_roi_name = roi_name
        
        # Set initial points
        self.roi_start = QPoint(self.roi_areas[roi_name][0], self.roi_areas[roi_name][1])
        self.roi_end = QPoint(self.roi_areas[roi_name][2], self.roi_areas[roi_name][3])
        
        # Update UI
        self.add_roi_button.setEnabled(False)
        self.cancel_roi_button.setEnabled(True)
        self.edit_roi_button.setEnabled(False)
        self.properties_button.setEnabled(False)
        self.delete_roi_button.setEnabled(False)
        
        # Update display to show the ROI being edited
        self.update_display()
        
        print(f"Editing ROI: {roi_name} at {self.roi_areas[roi_name]}")
    
    def delete_selected_roi(self):
        """Delete the selected ROI."""
        if not self.roi_list.currentItem():
            return
        
        name = self.roi_list.currentItem().text()
        
        # Confirm deletion
        reply = QMessageBox.question(
            self, "Confirm Deletion", 
            f"Are you sure you want to delete the ROI '{name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Remove the ROI
            del self.roi_areas[name]
            if name in self.roi_colors:
                del self.roi_colors[name]
            
            # Update display
            self.update_roi_list()
            self.update_display()
            
            # Emit signal with updated ROIs
            self.roi_updated.emit(self.roi_areas, self.roi_colors)
    
    def save_roi_config(self):
        """Save the ROI configuration to a file."""
        if not self.roi_areas:
            QMessageBox.warning(self, "No ROIs", "There are no ROIs to save.")
            return
        
        # Get file path from user
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save ROI Configuration", "", "JSON Files (*.json)"
        )
        
        if not file_path:
            return
        
        # Ensure the file has .json extension
        if not file_path.lower().endswith('.json'):
            file_path += '.json'
        
        # Prepare data for saving
        # Convert BGR color tuples to lists for JSON serialization
        roi_colors_serializable = {name: list(color) for name, color in self.roi_colors.items()}
        
        # Create a serializable version of roi_points for quadrilateral ROIs
        roi_points_serializable = {}
        for name, points in self.roi_points.items():
            roi_points_serializable[name] = [list(point) for point in points]
        
        data = {
            'roi_areas': self.roi_areas,
            'roi_colors': roi_colors_serializable,
            'roi_points': roi_points_serializable
        }
        
        try:
            import json
            with open(file_path, 'w') as f:
                json.dump(data, f)
            QMessageBox.information(self, "Save Successful", 
                                  f"ROI configuration saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", 
                               f"Failed to save ROI configuration: {str(e)}")
    
    def load_roi_config(self):
        """Load ROI configuration from a file."""
        # Get file path from user
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load ROI Configuration", "", "JSON Files (*.json)"
        )
        
        if not file_path:
            return False
        
        try:
            import json
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if 'roi_areas' in data and 'roi_colors' in data:
                # Convert loaded data to appropriate types
                self.roi_areas = {name: tuple(coords) for name, coords in data['roi_areas'].items()}
                self.roi_colors = {name: tuple(color) for name, color in data['roi_colors'].items()}
                
                # Load quadrilateral points if available
                if 'roi_points' in data:
                    self.roi_points = {}
                    for name, points in data['roi_points'].items():
                        self.roi_points[name] = [tuple(point) for point in points]
                
                # Update display
                self.update_roi_list()
                self.update_display()
                
                # Emit signal with updated ROIs
                self.roi_updated.emit(self.roi_areas, self.roi_colors)
                
                QMessageBox.information(self, "Load Successful", 
                                      f"ROI configuration loaded from {file_path}")
                return True
            else:
                QMessageBox.warning(self, "Invalid File", 
                                  "The selected file does not contain valid ROI configuration.")
                return False
        except Exception as e:
            QMessageBox.critical(self, "Load Error", 
                               f"Failed to load ROI configuration: {str(e)}")
            return False
    
    def apply_roi_preset(self):
        """Apply a predefined ROI preset based on the selected option."""
        if self.frame is None:
            QMessageBox.warning(self, "No Frame", "Please load a video first.")
            return
        
        preset = self.preset_combo.currentText()
        
        # Confirm before applying preset
        reply = QMessageBox.question(
            self, "Apply Preset", 
            f"Applying the preset '{preset}' will replace any existing ROIs. Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Clear existing ROIs
        self.roi_areas.clear()
        self.roi_colors.clear()
        
        # Get frame dimensions
        h, w = self.frame.shape[:2]
        
        # Define presets based on frame size
        if preset == "Retail Store Layout":
            # Define entrance, exit, shelves, checkout
            self.roi_areas["Entrance"] = (0, h//2 - h//8, w//8, h//2 + h//8)
            self.roi_colors["Entrance"] = (0, 255, 0)  # Green
            
            self.roi_areas["Exit"] = (w - w//8, h//2 - h//8, w, h//2 + h//8)
            self.roi_colors["Exit"] = (0, 0, 255)  # Red
            
            self.roi_areas["Checkout"] = (w//2 - w//6, h - h//4, w//2 + w//6, h)
            self.roi_colors["Checkout"] = (255, 0, 0)  # Blue
            
            self.roi_areas["Shelf-A"] = (w//4, h//4, w//4 + w//8, 3*h//4)
            self.roi_colors["Shelf-A"] = (255, 255, 0)  # Cyan
            
            self.roi_areas["Shelf-B"] = (w//2, h//4, w//2 + w//8, 3*h//4)
            self.roi_colors["Shelf-B"] = (255, 0, 255)  # Magenta
            
        elif preset == "Checkout Counters":
            # Define multiple checkout counters
            counter_width = w // 6
            counter_height = h // 5
            spacing = w // 20
            
            y_pos = h - counter_height - h//10
            
            for i in range(3):
                x_pos = w//10 + i * (counter_width + spacing)
                name = f"Checkout-{i+1}"
                self.roi_areas[name] = (x_pos, y_pos, x_pos + counter_width, y_pos + counter_height)
                self.roi_colors[name] = (255, 128 * (i % 2), 128 * ((i+1) % 2))
            
        elif preset == "Entrance/Exit":
            # Define entrance and exit areas
            self.roi_areas["Main-Entrance"] = (0, h//3, w//6, 2*h//3)
            self.roi_colors["Main-Entrance"] = (0, 255, 0)  # Green
            
            self.roi_areas["Side-Entrance"] = (w//3, 0, 2*w//3, h//6)
            self.roi_colors["Side-Entrance"] = (128, 255, 0)  # Light green
            
            self.roi_areas["Main-Exit"] = (w - w//6, h//3, w, 2*h//3)
            self.roi_colors["Main-Exit"] = (0, 0, 255)  # Red
            
            self.roi_areas["Emergency-Exit"] = (w//3, h - h//6, 2*w//3, h)
            self.roi_colors["Emergency-Exit"] = (0, 0, 128)  # Dark red
            
        elif preset == "Shelf Areas":
            # Define multiple shelf areas
            shelf_width = w // 10
            shelf_height = h // 2
            
            # Left-side shelves
            for i in range(3):
                y_pos = h//8 + i * (shelf_height // 3)
                name = f"Left-Shelf-{i+1}"
                self.roi_areas[name] = (w//8, y_pos, w//8 + shelf_width, y_pos + shelf_height//4)
                self.roi_colors[name] = (128, 128, 128 + 40*i)
            
            # Right-side shelves
            for i in range(3):
                y_pos = h//8 + i * (shelf_height // 3)
                name = f"Right-Shelf-{i+1}"
                self.roi_areas[name] = (w - w//8 - shelf_width, y_pos, w - w//8, y_pos + shelf_height//4)
                self.roi_colors[name] = (128, 128 + 40*i, 128)
            
            # Center display
            self.roi_areas["Center-Display"] = (w//2 - shelf_width, h//2 - shelf_height//4, w//2 + shelf_width, h//2 + shelf_height//4)
            self.roi_colors["Center-Display"] = (128, 0, 128)
        
        # Update display
        self.update_roi_list()
        self.update_display()
        
        # Emit signal with updated ROIs
        self.roi_updated.emit(self.roi_areas, self.roi_colors)
        
        QMessageBox.information(self, "Preset Applied", 
                              f"Applied the '{preset}' preset with {len(self.roi_areas)} ROIs.")
    
    def finish_roi_configuration(self):
        """Finish ROI configuration and move to the next step."""
        if not self.roi_areas:
            reply = QMessageBox.question(
                self, "No ROIs Defined", 
                "You haven't defined any ROIs. Are you sure you want to continue without ROIs?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
        
        # Emit signal with the final ROI configuration
        self.roi_updated.emit(self.roi_areas, self.roi_colors)
    
    def update_button_states(self):
        """Update the enabled state of buttons based on current conditions."""
        has_frame = self.frame is not None
        has_rois = len(self.roi_areas) > 0
        is_roi_selected = self.roi_list.currentItem() is not None
        
        # Button states
        self.add_roi_button.setEnabled(has_frame and not self.drawing)
        self.cancel_roi_button.setEnabled(self.drawing)
        self.edit_roi_button.setEnabled(is_roi_selected and not self.drawing)
        self.properties_button.setEnabled(is_roi_selected and not self.drawing)
        self.delete_roi_button.setEnabled(is_roi_selected and not self.drawing)
        self.save_roi_button.setEnabled(has_rois and not self.drawing)
        self.load_roi_button.setEnabled(has_frame and not self.drawing)
        self.finish_button.setEnabled(has_frame and not self.drawing)
        
        # Debug information
        print(f"ROI Button States: Add={self.add_roi_button.isEnabled()}, Edit={self.edit_roi_button.isEnabled()}, "
              f"Properties={self.properties_button.isEnabled()}, Delete={self.delete_roi_button.isEnabled()}, "
              f"Save={self.save_roi_button.isEnabled()}, Load={self.load_roi_button.isEnabled()}, "
              f"Finish={self.finish_button.isEnabled()}")
    
    def edit_roi_properties(self):
        """Edit the name or color of the selected ROI."""
        if not self.roi_list.currentItem():
            return
            
        current_name = self.roi_list.currentItem().text()
        
        # Edit name
        new_name, ok = QInputDialog.getText(
            self, "Edit ROI Name", "Enter a new name for this ROI:", 
            text=current_name
        )
        
        if not ok or not new_name:
            return
        
        # Check for duplicate unless it's the same name
        if new_name != current_name and new_name in self.roi_areas:
            QMessageBox.warning(self, "Duplicate Name", f"An ROI named '{new_name}' already exists.")
            return
        
        # Edit color
        current_color = self.roi_colors.get(current_name, (255, 0, 0))
        color_dialog = QColorDialog(self)
        color_dialog.setCurrentColor(QColor(current_color[2], current_color[1], current_color[0]))
        
        if color_dialog.exec_():
            qcolor = color_dialog.selectedColor()
            new_color = (qcolor.blue(), qcolor.green(), qcolor.red())  # BGR for OpenCV
        else:
            new_color = current_color
        
        # Update ROI data
        coords = self.roi_areas[current_name]
        if new_name != current_name:
            self.roi_areas[new_name] = coords
            self.roi_colors[new_name] = new_color
            del self.roi_areas[current_name]
            del self.roi_colors[current_name]
        else:
            self.roi_colors[current_name] = new_color
        
        # Update display
        self.update_roi_list()
        self.update_display()
        
        # Emit signal with updated ROIs
        self.roi_updated.emit(self.roi_areas, self.roi_colors)
    
    def validate_roi_coordinates(self, x1, y1, x2, y2):
        """Validate and adjust ROI coordinates to ensure they are within frame bounds."""
        if self.frame is None:
            return x1, y1, x2, y2
        
        height, width = self.frame.shape[:2]
        
        # Clamp coordinates to frame boundaries
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))
        
        return x1, y1, x2, y2
    
    def get_image_position(self, qt_pos):
        """Convert Qt position to coordinates on the actual image."""
        if self.frame is None:
            return None
        
        # Get the dimensions of the original frame (the true image size)
        frame_height, frame_width = self.frame.shape[:2]
        
        # Get the dimensions of the displayed pixmap
        pixmap = self.frame_label.pixmap()
        if not pixmap:
            return None
            
        # Get label geometry
        label_width = self.frame_label.width()
        label_height = self.frame_label.height()
        
        # Print debug information
        print(f"Debug - Click Position: ({qt_pos.x()}, {qt_pos.y()})")
        print(f"Debug - Label size: {label_width}x{label_height}")
        print(f"Debug - Original frame size: {frame_width}x{frame_height}")
        
        # For consistent coordinate mapping, we'll use the actual frame dimensions 
        # rather than the pixmap, which can change between calls
        
        # Calculate scaling factors based on the frame dimensions
        width_ratio = label_width / frame_width
        height_ratio = label_height / frame_height
        
        # Use the smaller ratio to ensure the image fits completely
        scale_factor = min(width_ratio, height_ratio)
        
        # Calculate the displayed image dimensions
        display_width = frame_width * scale_factor
        display_height = frame_height * scale_factor
        
        # Calculate the offset for centering the image in the label
        offset_x = (label_width - display_width) / 2
        offset_y = (label_height - display_height) / 2
        
        print(f"Debug - Display size: {display_width}x{display_height}")
        print(f"Debug - Offset: ({offset_x}, {offset_y})")
        print(f"Debug - Scale factor: {scale_factor}")
        
        # Get the position relative to the displayed image
        image_x = (qt_pos.x() - offset_x) / scale_factor
        image_y = (qt_pos.y() - offset_y) / scale_factor
        
        print(f"Debug - Calculated image position: ({image_x}, {image_y})")
        
        # Check if the point is within the image bounds
        if (0 <= image_x < frame_width and 0 <= image_y < frame_height):
            # Convert to integer coordinates for image processing
            return QPoint(int(image_x), int(image_y))
        
        # If point is outside image bounds, log it
        print(f"Debug - Point outside image bounds: ({image_x}, {image_y})")
        print(f"Debug - Image bounds: (0, 0) to ({frame_width}, {frame_height})")
        
        # For debugging, let's try to clamp the coordinates to the image bounds
        clamped_x = max(0, min(int(image_x), frame_width - 1))
        clamped_y = max(0, min(int(image_y), frame_height - 1))
        
        if clamped_x != int(image_x) or clamped_y != int(image_y):
            print(f"Debug - Clamped point to: ({clamped_x}, {clamped_y})")
            return QPoint(clamped_x, clamped_y)
        
        return None
    
    def update_roi_list(self):
        """Update the ROI list widget with current ROIs."""
        self.roi_list.clear()
        for name in self.roi_areas:
            item = QListWidgetItem(name)
            # Set a small square of color as the icon
            color = self.roi_colors.get(name, (255, 0, 0))
            # Convert BGR to RGB for Qt
            qcolor = QColor(color[2], color[1], color[0])
            item.setBackground(qcolor)
            item.setForeground(QColor(255, 255, 255) if sum(color) < 380 else QColor(0, 0, 0))
            self.roi_list.addItem(item)
    
    def complete_quad_roi(self):
        """Complete the creation of a quadrilateral ROI after 4 points have been selected."""
        if not self.quad_mode or len(self.quad_points) != 4:
            return
            
        # Calculate bounding box for compatibility
        x_coords = [p[0] for p in self.quad_points]
        y_coords = [p[1] for p in self.quad_points]
        
        x1 = min(x_coords)
        y1 = min(y_coords)
        x2 = max(x_coords)
        y2 = max(y_coords)
        
        # Store the rectangular bounding box
        self.roi_areas[self.current_roi_name] = (x1, y1, x2, y2)
        
        # Store the quadrilateral points
        self.roi_points[self.current_roi_name] = self.quad_points
        
        # Update list
        self.update_roi_list()
        
        # Reset drawing state
        self.drawing = False
        self.quad_points = []
        
        # Reset cursor
        self.unsetCursor()
        
        # Update UI
        self.update_display()
        self.update_button_states()
        self.set_status(f"Quadrilateral ROI added successfully! ({len(self.roi_areas)} regions defined)")
        
        # Emit signal with updated ROIs
        self.roi_updated.emit(self.roi_areas, self.roi_colors)
        
        # Show success animation
        self.highlight_success()
        
        # Clear current name
        self.current_roi_name = None 