import cv2
import numpy as np
import uuid
import json
import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QListWidget, QListWidgetItem, QInputDialog, QColorDialog, 
    QMessageBox, QFileDialog, QComboBox, QFrame
)
from PySide6.QtCore import Qt, Signal, QPoint, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent

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
        
        # Frame data
        self.frame = None
        self.display_frame = None
        
        # UI setup
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        
        # Left side - Frame display and drawing area
        self.left_layout = QVBoxLayout()
        
        # Frame display label
        self.frame_label = QLabel("No video loaded")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setMinimumSize(640, 480)
        self.frame_label.mousePressEvent = self.on_mouse_press
        self.frame_label.mouseMoveEvent = self.on_mouse_move
        self.frame_label.mouseReleaseEvent = self.on_mouse_release
        
        self.left_layout.addWidget(self.frame_label)
        
        # Drawing instructions
        instructions = QLabel(
            "Click and drag to define ROI. Right-click to cancel. "
            "When you're done, click 'Finish ROI Configuration'."
        )
        instructions.setWordWrap(True)
        self.left_layout.addWidget(instructions)
        
        # Drawing controls
        drawing_controls = QHBoxLayout()
        
        self.add_roi_button = QPushButton("Add ROI")
        self.add_roi_button.clicked.connect(self.start_roi_drawing)
        drawing_controls.addWidget(self.add_roi_button)
        
        self.cancel_roi_button = QPushButton("Cancel Drawing")
        self.cancel_roi_button.clicked.connect(self.cancel_roi_drawing)
        self.cancel_roi_button.setEnabled(False)
        drawing_controls.addWidget(self.cancel_roi_button)
        
        self.finish_button = QPushButton("Finish ROI Configuration")
        self.finish_button.clicked.connect(self.finish_roi_configuration)
        drawing_controls.addWidget(self.finish_button)
        
        self.left_layout.addLayout(drawing_controls)
        
        # Right side - ROI list and management
        self.right_layout = QVBoxLayout()
        
        # ROI list title
        roi_list_label = QLabel("Defined ROIs")
        roi_list_label.setAlignment(Qt.AlignCenter)
        self.right_layout.addWidget(roi_list_label)
        
        # ROI list widget
        self.roi_list = QListWidget()
        self.roi_list.setMinimumWidth(200)
        self.roi_list.itemClicked.connect(self.on_roi_selected)
        self.right_layout.addWidget(self.roi_list)
        
        # ROI management buttons
        roi_buttons = QHBoxLayout()
        
        self.edit_roi_button = QPushButton("Edit ROI")
        self.edit_roi_button.clicked.connect(self.edit_selected_roi)
        roi_buttons.addWidget(self.edit_roi_button)
        
        self.delete_roi_button = QPushButton("Delete ROI")
        self.delete_roi_button.clicked.connect(self.delete_selected_roi)
        roi_buttons.addWidget(self.delete_roi_button)
        
        self.right_layout.addLayout(roi_buttons)
        
        # ROI save/load buttons
        roi_file_buttons = QHBoxLayout()
        
        self.save_roi_button = QPushButton("Save ROIs")
        self.save_roi_button.clicked.connect(self.save_roi_config)
        roi_file_buttons.addWidget(self.save_roi_button)
        
        self.load_roi_button = QPushButton("Load ROIs")
        self.load_roi_button.clicked.connect(self.load_roi_config)
        roi_file_buttons.addWidget(self.load_roi_button)
        
        self.right_layout.addLayout(roi_file_buttons)
        
        # ROI quick presets
        presets_frame = QFrame()
        presets_frame.setFrameShape(QFrame.StyledPanel)
        presets_layout = QVBoxLayout()
        presets_frame.setLayout(presets_layout)
        
        presets_label = QLabel("Quick ROI Presets")
        presets_label.setAlignment(Qt.AlignCenter)
        presets_layout.addWidget(presets_label)
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Retail Store Layout",
            "Checkout Counters",
            "Entrance/Exit",
            "Shelf Areas"
        ])
        presets_layout.addWidget(self.preset_combo)
        
        apply_preset_button = QPushButton("Apply Preset")
        apply_preset_button.clicked.connect(self.apply_roi_preset)
        presets_layout.addWidget(apply_preset_button)
        
        self.right_layout.addWidget(presets_frame)
        self.right_layout.addStretch()
        
        # Add layouts to main layout
        self.layout.addLayout(self.left_layout, 3)  # 3:1 ratio
        self.layout.addLayout(self.right_layout, 1)
        
        # Update button states
        self.update_button_states()
    
    def load_frame(self, frame):
        """Load a frame for ROI definition."""
        if frame is None:
            return
        
        self.frame = frame.copy()
        self.update_display()
    
    def update_display(self):
        """Update the displayed frame with ROI overlays."""
        if self.frame is None:
            return
        
        # Make a copy of the original frame
        self.display_frame = self.frame.copy()
        
        # Draw existing ROIs
        for name, (x1, y1, x2, y2) in self.roi_areas.items():
            color = self.roi_colors.get(name, (255, 0, 0))
            cv2.rectangle(self.display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(self.display_frame, name, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw currently being defined ROI
        if self.drawing and self.roi_start and self.roi_end:
            x1, y1 = self.roi_start.x(), self.roi_start.y()
            x2, y2 = self.roi_end.x(), self.roi_end.y()
            cv2.rectangle(self.display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if self.current_roi_name:
                cv2.putText(self.display_frame, self.current_roi_name, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
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
    
    def start_roi_drawing(self):
        """Start the ROI drawing process."""
        if self.frame is None:
            QMessageBox.warning(self, "No Frame", "Please load a video first.")
            return
        
        # Get ROI name from user
        name, ok = QInputDialog.getText(self, "ROI Name", "Enter a name for this ROI:")
        if not ok or not name:
            return
        
        # Check for duplicate names
        if name in self.roi_areas:
            QMessageBox.warning(self, "Duplicate Name", f"An ROI named '{name}' already exists.")
            return
        
        # Get ROI color from user
        color_dialog = QColorDialog(self)
        if color_dialog.exec_():
            qcolor = color_dialog.selectedColor()
            color = (qcolor.blue(), qcolor.green(), qcolor.red())  # BGR for OpenCV
        else:
            color = (255, 0, 0)  # Default to blue
        
        self.roi_colors[name] = color
        self.current_roi_name = name
        self.drawing = True
        self.roi_start = None
        self.roi_end = None
        
        # Update button states
        self.add_roi_button.setEnabled(False)
        self.cancel_roi_button.setEnabled(True)
        
        # Instruction for user
        QMessageBox.information(self, "Draw ROI", 
                              "Click and drag on the image to define the ROI area.")
    
    def cancel_roi_drawing(self):
        """Cancel the current ROI drawing."""
        self.drawing = False
        self.roi_start = None
        self.roi_end = None
        self.current_roi_name = None
        
        # Update display and button states
        self.update_display()
        self.update_button_states()
    
    def on_mouse_press(self, event: QMouseEvent):
        """Handle mouse press events for ROI drawing."""
        if not self.drawing or self.frame is None:
            return
        
        # Get position relative to the image
        pos = self.get_image_position(event.position())
        if pos:
            self.roi_start = pos
            self.roi_end = pos
            self.update_display()
    
    def on_mouse_move(self, event: QMouseEvent):
        """Handle mouse move events for ROI drawing."""
        if not self.drawing or not self.roi_start or self.frame is None:
            return
        
        # Get position relative to the image
        pos = self.get_image_position(event.position())
        if pos:
            self.roi_end = pos
            self.update_display()
    
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
        
        # Ensure minimum size
        if x2 - x1 < 20 or y2 - y1 < 20:
            QMessageBox.warning(self, "ROI Too Small", 
                              "The ROI is too small. Please draw a larger area.")
            return
        
        # Add the ROI
        self.roi_areas[self.current_roi_name] = (x1, y1, x2, y2)
        
        # Add to list widget
        self.update_roi_list()
        
        # Reset drawing state
        self.drawing = False
        self.roi_start = None
        self.roi_end = None
        self.current_roi_name = None
        
        # Update display and button states
        self.update_display()
        self.update_button_states()
        
        # Emit signal with updated ROIs
        self.roi_updated.emit(self.roi_areas, self.roi_colors)
    
    def get_image_position(self, qt_pos):
        """Convert Qt position to image position, accounting for scaling."""
        if self.frame is None or not self.frame_label.pixmap():
            return None
        
        # Get label and pixmap geometries
        label_rect = self.frame_label.rect()
        pixmap = self.frame_label.pixmap()
        pixmap_rect = pixmap.rect()
        
        # Calculate scaling and position
        scale_x = pixmap_rect.width() / self.frame.shape[1]
        scale_y = pixmap_rect.height() / self.frame.shape[0]
        scale = min(scale_x, scale_y)
        
        # Calculate offset (to center the image in the label)
        offset_x = (label_rect.width() - self.frame.shape[1] * scale) / 2
        offset_y = (label_rect.height() - self.frame.shape[0] * scale) / 2
        
        # Convert to image coordinates
        image_x = int((qt_pos.x() - offset_x) / scale)
        image_y = int((qt_pos.y() - offset_y) / scale)
        
        # Ensure within bounds
        if 0 <= image_x < self.frame.shape[1] and 0 <= image_y < self.frame.shape[0]:
            return QPoint(image_x, image_y)
        
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
    
    def on_roi_selected(self, item):
        """Handle ROI selection in the list."""
        # Highlight the selected ROI on the display
        self.update_display()
        self.update_button_states()
    
    def edit_selected_roi(self):
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
        
        data = {
            'roi_areas': self.roi_areas,
            'roi_colors': roi_colors_serializable
        }
        
        try:
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
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if 'roi_areas' in data and 'roi_colors' in data:
                # Convert loaded data to appropriate types
                self.roi_areas = {name: tuple(coords) for name, coords in data['roi_areas'].items()}
                self.roi_colors = {name: tuple(color) for name, color in data['roi_colors'].items()}
                
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
        
        self.add_roi_button.setEnabled(has_frame and not self.drawing)
        self.cancel_roi_button.setEnabled(self.drawing)
        self.finish_button.setEnabled(has_frame)
        self.edit_roi_button.setEnabled(is_roi_selected)
        self.delete_roi_button.setEnabled(is_roi_selected)
        self.save_roi_button.setEnabled(has_rois)
        self.load_roi_button.setEnabled(has_frame) 