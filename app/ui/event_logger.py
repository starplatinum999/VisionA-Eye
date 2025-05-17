from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QLabel, 
    QListWidget, QListWidgetItem, QHBoxLayout, 
    QFrame, QPushButton, QSplitter
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImage, QPixmap, QFont, QColor

import cv2
import numpy as np

class EventListItem(QWidget):
    """Custom widget for displaying an individual event with thumbnail."""
    
    def __init__(self, event, parent=None):
        super().__init__(parent)
        
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(self.layout)
        
        # Set item background color based on event type
        bg_color = self.get_event_color(event['type'])
        self.setStyleSheet(f"background-color: {bg_color}; border-radius: 5px; margin: 2px;")
        
        # Thumbnail (if available)
        if 'thumbnail' in event and event['thumbnail'] is not None:
            try:
                thumbnail = event['thumbnail']
                h, w = thumbnail.shape[:2]
                
                # Convert OpenCV image to Qt format
                rgb_image = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                qimage = QImage(rgb_image.data, w, h, w * 3, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                
                thumbnail_label = QLabel()
                thumbnail_label.setPixmap(pixmap.scaled(120, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                thumbnail_label.setFixedSize(120, 90)
                thumbnail_label.setStyleSheet("background-color: #FFFFFF; padding: 2px; border: 1px solid #D1D5DB;")
                self.layout.addWidget(thumbnail_label)
            except Exception as e:
                print(f"Error displaying thumbnail: {e}")
        
        # Event information
        info_layout = QVBoxLayout()
        
        # Event type with bold font
        type_label = QLabel(event['type'])
        type_font = QFont()
        type_font.setBold(True)
        type_font.setPointSize(10)
        type_label.setFont(type_font)
        type_label.setStyleSheet("color: #1F2937; background-color: rgba(0,0,0,0);")
        info_layout.addWidget(type_label)
        
        # Event timestamp
        timestamp_label = QLabel(f"Time: {event['timestamp']}")
        timestamp_label.setStyleSheet("color: #6B7280; background-color: rgba(0,0,0,0);")
        info_layout.addWidget(timestamp_label)
        
        # Event description
        description_label = QLabel(event['description'])
        description_label.setWordWrap(True)
        description_label.setStyleSheet("color: #1F2937; background-color: rgba(0,0,0,0);")
        info_layout.addWidget(description_label)
        
        self.layout.addLayout(info_layout)
    
    def get_event_color(self, event_type):
        """Return a color based on event type for visual differentiation."""
        color_map = {
            'ROI Transition': "#E4E7EB",  # Light Gray
            'ROI Exit': "#F5F7FA",        # Very Light Gray
            'Person at Shelf': "#3B82F6",  # Soft Blue accent
            'Item Pickup': "#93C5FD",     # Lighter Blue
            'Potential Theft': "#EF4444", # Error Red
            'Long Dwell Time': "#FACC15", # Warning Yellow
        }
        
        return color_map.get(event_type, "#E4E7EB")

class EventLogger(QWidget):
    """Widget for displaying and logging events detected by the video processing."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.setStyleSheet("background-color: #F5F7FA;")
        
        # Header
        header = QLabel("Event Log")
        header.setAlignment(Qt.AlignCenter)
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setStyleSheet("color: #1F2937; background-color: #E4E7EB; padding: 8px; border-radius: 5px;")
        self.layout.addWidget(header)
        
        # Event count label
        self.event_count_label = QLabel("Events: 0")
        self.event_count_label.setStyleSheet("color: #6B7280; padding: 5px;")
        self.layout.addWidget(self.event_count_label)
        
        # Scroll area for events
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll_area.setStyleSheet("background-color: #E4E7EB; border: none;")
        
        # Event list widget
        self.event_list = QListWidget()
        self.event_list.setSpacing(8)
        self.event_list.setStyleSheet("background-color: #E4E7EB; border: none;")
        
        self.scroll_area.setWidget(self.event_list)
        self.layout.addWidget(self.scroll_area)
        
        # Button container with horizontal layout
        button_container = QWidget()
        button_layout = QHBoxLayout()
        button_container.setLayout(button_layout)
        
        # Clear button
        self.clear_button = QPushButton("Clear Events")
        self.clear_button.clicked.connect(self.clear_events)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #EF4444;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #DC2626;
            }
            QPushButton:pressed {
                background-color: #B91C1C;
            }
        """)
        button_layout.addWidget(self.clear_button)
        
        # Export button
        self.export_button = QPushButton("Export Events")
        self.export_button.clicked.connect(self.export_events)
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)
        button_layout.addWidget(self.export_button)
        
        self.layout.addWidget(button_container)
        
        # Initialize event list
        self.events = []
    
    def add_event(self, event):
        """Add a new event to the log."""
        self.events.append(event)
        
        # Create a custom list item
        item = QListWidgetItem()
        event_widget = EventListItem(event)
        
        # Set appropriate size for the item
        item.setSizeHint(QSize(self.event_list.width() - 30, 110))
        
        # Add to list
        self.event_list.addItem(item)
        self.event_list.setItemWidget(item, event_widget)
        
        # Scroll to bottom to show newest event
        self.event_list.scrollToBottom()
        
        # Update event count
        self.event_count_label.setText(f"Events: {len(self.events)}")
    
    def clear_events(self):
        """Clear all events from the log."""
        self.events.clear()
        self.event_list.clear()
        self.event_count_label.setText("Events: 0")
    
    def export_events(self):
        """Export event data to CSV or JSON file."""
        if not self.events:
            return
            
        from PySide6.QtWidgets import QFileDialog
        import csv
        import json
        import os
        from datetime import datetime
        
        # Get file path from user
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Events",
            f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "CSV Files (*.csv);;JSON Files (*.json)"
        )
        
        if not file_path:
            return
            
        try:
            if selected_filter == "CSV Files (*.csv)":
                # Ensure file has .csv extension
                if not file_path.lower().endswith('.csv'):
                    file_path += '.csv'
                    
                with open(file_path, 'w', newline='') as csvfile:
                    fieldnames = ['type', 'timestamp', 'description', 'frame_idx', 'track_id']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for event in self.events:
                        # Extract only serializable fields
                        event_data = {
                            'type': event.get('type', ''),
                            'timestamp': event.get('timestamp', ''),
                            'description': event.get('description', ''),
                            'frame_idx': event.get('frame_idx', 0),
                            'track_id': event.get('track_id', '')
                        }
                        writer.writerow(event_data)
            else:
                # Ensure file has .json extension
                if not file_path.lower().endswith('.json'):
                    file_path += '.json'
                    
                # Create serializable version of events (without thumbnails)
                serializable_events = []
                for event in self.events:
                    event_copy = event.copy()
                    if 'thumbnail' in event_copy:
                        del event_copy['thumbnail']
                    serializable_events.append(event_copy)
                    
                with open(file_path, 'w') as jsonfile:
                    json.dump(serializable_events, jsonfile, indent=2)
                    
            # Show success message
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Export Successful", f"Events exported to {file_path}")
                
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Export Error", f"Failed to export events: {str(e)}")
    
    def get_events(self):
        """Return the current list of events."""
        return self.events 