import sys
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QPushButton, QFileDialog, 
    QStatusBar, QMessageBox, QApplication, QLineEdit,
    QDialog, QFormLayout, QTextEdit, QFrame, QSplashScreen
)
from PySide6.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QIcon, QFont, QPalette, QColor, QPixmap, QLinearGradient, QBrush, QPainter, QFontDatabase
import os

from app.ui.video_widget import VideoWidget
from app.ui.roi_widget import ROIWidget
from app.ui.event_logger import EventLogger
from app.ui.analytics_dashboard import AnalyticsDashboard

class MainWindow(QMainWindow):
    """Main window for the Vision AI desktop application."""
    
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.setWindowTitle("Vision AI - Smart Surveillance")
        self.setMinimumSize(1280, 900)
        
        # Load custom fonts if available
        self.load_fonts()
        
        # Set modern theme for entire application
        self.apply_modern_theme()
        
        # Shared data between tabs
        self.video_path = None
        self.roi_areas = {}
        self.roi_colors = {}
        
        # Create tab widget (similar to Streamlit tabs)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Style the tabs with modern look
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #E4E7EB;
                background-color: #FFFFFF;
                border-radius: 12px;
                padding: 10px;
            }
            QTabBar::tab {
                background-color: #F5F7FA;
                color: #4B5563;
                border: 1px solid #E4E7EB;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 10px 20px;
                margin-right: 4px;
                font-weight: 600;
                min-width: 120px;
                font-size: 14px;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #FFFFFF, stop:1 #F0F5FF);
                color: #2563EB;
                border-bottom: 3px solid #2563EB;
            }
            QTabBar::tab:hover:!selected {
                background-color: #EFF6FF;
                color: #1D4ED8;
            }
        """)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #F5F7FA;
                color: #4B5563;
                border-top: 1px solid #E4E7EB;
                padding: 5px;
                font-size: 13px;
            }
        """)
        
        # Create tabs for each section
        self.setup_home_tab()
        self.setup_roi_tab()
        self.setup_surveillance_tab()
        self.setup_analytics_tab()
    
    def load_fonts(self):
        """Load custom fonts if available in the system."""
        # Preferred fonts for modern UI
        preferred_fonts = ["SF Pro Display", "Segoe UI", "Roboto", "Inter", "Helvetica Neue"]
        
        # Check if fonts are available
        available_fonts = QFontDatabase().families()
        
        # Set application font to first available preferred font
        for font_name in preferred_fonts:
            if any(font_name.lower() in f.lower() for f in available_fonts):
                app_font = QFont(font_name)
                QApplication.setFont(app_font)
                break
    
    def apply_modern_theme(self):
        """Apply a modern theme to the entire application."""
        app = QApplication.instance()
        
        # Set fusion style for a more modern look
        app.setStyle("Fusion")
        
        # Create a light palette with a professional blue accent color scheme
        palette = QPalette()
        
        # Base colors
        palette.setColor(QPalette.Window, QColor("#FFFFFF"))         # White
        palette.setColor(QPalette.WindowText, QColor("#1F2937"))     # Charcoal text
        palette.setColor(QPalette.Base, QColor("#F5F7FA"))          # Very light gray
        palette.setColor(QPalette.AlternateBase, QColor("#E4E7EB"))  # Light gray
        palette.setColor(QPalette.ToolTipBase, QColor("#F5F7FA"))
        palette.setColor(QPalette.ToolTipText, QColor("#1F2937"))
        
        # Text and button colors
        palette.setColor(QPalette.Text, QColor("#1F2937"))          # Charcoal text
        palette.setColor(QPalette.Button, QColor("#F9FAFB"))        # Lighter gray for buttons
        palette.setColor(QPalette.ButtonText, QColor("#1F2937"))    # Charcoal text
        palette.setColor(QPalette.BrightText, QColor("#EF4444"))    # Error Red
        
        # Highlight and link colors
        palette.setColor(QPalette.Link, QColor("#2563EB"))          # Blue links
        palette.setColor(QPalette.Highlight, QColor("#3B82F6"))     # Blue highlight
        palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF")) # White on highlight
        
        # Set the palette
        app.setPalette(palette)
        
        # Set the default style sheet with modern controls
        app.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                color: #1F2937;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "SF Pro Display", Roboto, Inter, Helvetica, Arial, sans-serif;
                font-size: 14px;
            }
            QLabel {
                background: transparent;
                border: none;
                padding: 0;
                color: #1F2937;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #4F86F7, stop:1 #3B72D9);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 600;
                font-size: 14px;
                min-height: 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #5D93FF, stop:1 #4B82E9);
                color: white;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #3A67BA, stop:1 #2B57A9);
            }
            QPushButton:disabled {
                background: #D1D5DB;
                color: #9CA3AF;
            }
            QLineEdit, QComboBox, QTextEdit {
                background-color: #F9FAFB;
                color: #1F2937;
                border: 1px solid #D1D5DB;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                selection-background-color: #93C5FD;
            }
            QLineEdit:focus, QComboBox:focus, QTextEdit:focus {
                border-color: #3B82F6;
                border-width: 2px;
                background-color: #FFFFFF;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox::down-arrow {
                color: #6B7280;
                font-size: 12px;
            }
            QStatusBar {
                background-color: #F5F7FA;
                color: #4B5563;
                border-top: 1px solid #E4E7EB;
            }
            QScrollBar:vertical {
                background-color: #F5F7FA;
                width: 14px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #D1D5DB;
                min-height: 30px;
                border-radius: 7px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #9CA3AF;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background-color: #F5F7FA;
                height: 14px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background-color: #D1D5DB;
                min-width: 30px;
                border-radius: 7px;
                margin: 2px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #9CA3AF;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QFrame.card {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #E4E7EB;
                padding: 15px;
            }
            QFrame.card:hover {
                border-color: #93C5FD;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            }
            QLabel.card-title {
                color: #1F2937;
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
            }
            QLabel.card-subtitle {
                color: #4B5563;
                font-size: 14px;
                margin-bottom: 15px;
            }
        """)
    
    def setup_home_tab(self):
        """Set up the home tab with welcome screen and video selection."""
        # Create the home tab
        home_tab = QWidget()
        self.tabs.addTab(home_tab, "Home")
        
        # Create layout for home tab
        layout = QVBoxLayout(home_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Cards container
        cards_widget = QWidget()
        cards_layout = QHBoxLayout(cards_widget)
        cards_layout.setSpacing(20)
        
        # Video File Card
        video_card = QFrame()
        video_card.setObjectName("videoCard")
        video_card.setProperty("class", "card")
        video_card.setMinimumHeight(300)
        video_card_layout = QVBoxLayout(video_card)
        video_card_layout.setContentsMargins(30, 30, 30, 30)
        video_card_layout.setAlignment(Qt.AlignTop)
        
        # Card icon
        video_icon_label = QLabel()
        video_icon_label.setAlignment(Qt.AlignCenter)
        video_icon_label.setFixedSize(60, 60)
        video_icon_label.setStyleSheet("""
            background-color: #EFF6FF;
            border-radius: 30px;
            padding: 15px;
            margin-bottom: 15px;
        """)
        # You can replace this with an actual icon later
        video_icon = QLabel("⬆️")
        video_icon.setStyleSheet("font-size: 32px; color: #3B82F6;")
        video_icon_layout = QVBoxLayout(video_icon_label)
        video_icon_layout.setContentsMargins(0, 0, 0, 0)
        video_icon_layout.addWidget(video_icon)
        
        # Card content
        video_title = QLabel("Upload Video File")
        video_title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #1F2937;
            margin-top: 10px;
        """)
        
        video_desc = QLabel("Import a pre-recorded video file for analysis")
        video_desc.setStyleSheet("""
            font-size: 14px;
            color: #6B7280;
            margin-top: 5px;
            margin-bottom: 20px;
        """)
        video_desc.setWordWrap(True)
        
        video_button = QPushButton("Select Video File")
        video_button.setStyleSheet("""
            background-color: #3B82F6;
            color: white;
            border-radius: 8px;
            padding: 10px 15px;
            font-weight: 600;
        """)
        video_button.setMinimumWidth(180)
        video_button.clicked.connect(self.select_video_file)
        
        video_card_layout.addWidget(video_icon_label, 0, Qt.AlignLeft)
        video_card_layout.addWidget(video_title)
        video_card_layout.addWidget(video_desc)
        video_card_layout.addWidget(video_button)
        video_card_layout.addStretch()
        
        # Camera Card
        camera_card = QFrame()
        camera_card.setProperty("class", "card")
        camera_card.setMinimumHeight(300)
        camera_card_layout = QVBoxLayout(camera_card)
        camera_card_layout.setContentsMargins(30, 30, 30, 30)
        camera_card_layout.setAlignment(Qt.AlignTop)
        
        # Card icon
        camera_icon_label = QLabel()
        camera_icon_label.setAlignment(Qt.AlignCenter)
        camera_icon_label.setFixedSize(60, 60)
        camera_icon_label.setStyleSheet("""
            background-color: #ECFDF5;
            border-radius: 30px;
            padding: 15px;
            margin-bottom: 15px;
        """)
        # You can replace this with an actual icon later
        camera_icon = QLabel("📷")
        camera_icon.setStyleSheet("font-size: 32px; color: #10B981;")
        camera_icon_layout = QVBoxLayout(camera_icon_label)
        camera_icon_layout.setContentsMargins(0, 0, 0, 0)
        camera_icon_layout.addWidget(camera_icon)
        
        # Card content
        camera_title = QLabel("Connect to Camera")
        camera_title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #1F2937;
            margin-top: 10px;
        """)
        
        camera_desc = QLabel("Connect to a webcam or IP camera for live monitoring")
        camera_desc.setStyleSheet("""
            font-size: 14px;
            color: #6B7280;
            margin-top: 5px;
            margin-bottom: 20px;
        """)
        camera_desc.setWordWrap(True)
        
        camera_button = QPushButton("Connect")
        camera_button.setStyleSheet("""
            background-color: #10B981;
            color: white;
            border-radius: 8px;
            padding: 10px 15px;
            font-weight: 600;
        """)
        camera_button.setMinimumWidth(120)
        camera_button.clicked.connect(self.connect_to_camera)
        
        camera_card_layout.addWidget(camera_icon_label, 0, Qt.AlignLeft)
        camera_card_layout.addWidget(camera_title)
        camera_card_layout.addWidget(camera_desc)
        camera_card_layout.addWidget(camera_button)
        camera_card_layout.addStretch()
        
        # RTSP stream card
        rtsp_card = QFrame()
        rtsp_card.setProperty("class", "card")
        rtsp_card.setMinimumHeight(300)
        rtsp_card_layout = QVBoxLayout(rtsp_card)
        rtsp_card_layout.setContentsMargins(30, 30, 30, 30)
        rtsp_card_layout.setAlignment(Qt.AlignTop)
        
        # Card icon
        rtsp_icon_label = QLabel()
        rtsp_icon_label.setAlignment(Qt.AlignCenter)
        rtsp_icon_label.setFixedSize(60, 60)
        rtsp_icon_label.setStyleSheet("""
            background-color: #EEE7FF;
            border-radius: 30px;
            padding: 15px;
            margin-bottom: 15px;
        """)
        # You can replace this with an actual icon later
        rtsp_icon = QLabel("📡")
        rtsp_icon.setStyleSheet("font-size: 32px; color: #8B5CF6;")
        rtsp_icon_layout = QVBoxLayout(rtsp_icon_label)
        rtsp_icon_layout.setContentsMargins(0, 0, 0, 0)
        rtsp_icon_layout.addWidget(rtsp_icon)
        
        # Card content
        rtsp_title = QLabel("RTSP Stream")
        rtsp_title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #1F2937;
            margin-top: 10px;
        """)
        
        rtsp_desc = QLabel("Connect to an RTSP stream from a network camera")
        rtsp_desc.setStyleSheet("""
            font-size: 14px;
            color: #6B7280;
            margin-top: 5px;
            margin-bottom: 20px;
        """)
        rtsp_desc.setWordWrap(True)
        
        rtsp_button = QPushButton("Connect")
        rtsp_button.setStyleSheet("""
            background-color: #8B5CF6;
            color: white;
            border-radius: 8px;
            padding: 10px 15px;
            font-weight: 600;
        """)
        rtsp_button.setMinimumWidth(120)
        rtsp_button.clicked.connect(self.connect_to_rtsp)
        
        rtsp_card_layout.addWidget(rtsp_icon_label, 0, Qt.AlignLeft)
        rtsp_card_layout.addWidget(rtsp_title)
        rtsp_card_layout.addWidget(rtsp_desc)
        rtsp_card_layout.addWidget(rtsp_button)
        rtsp_card_layout.addStretch()
        
        # Add cards to layout
        cards_layout.addWidget(video_card)
        cards_layout.addWidget(camera_card)
        cards_layout.addWidget(rtsp_card)
        
        layout.addWidget(cards_widget)
        
        # Add feature highlights section
        features_frame = QFrame()
        features_frame.setProperty("class", "card")
        features_layout = QVBoxLayout(features_frame)
        
        features_title = QLabel("Key Features")
        features_title.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 15px;
        """)
        
        features_grid = QHBoxLayout()
        
        # Feature 1
        feature1 = QVBoxLayout()
        feature1_icon_container = QLabel()
        feature1_icon_container.setFixedSize(60, 60)
        feature1_icon_container.setStyleSheet("""
            background-color: #FEF3C7;
            border-radius: 30px;
            padding: 15px;
        """)
        feature1_icon = QLabel("🔍")
        feature1_icon.setStyleSheet("font-size: 24px; color: #D97706;")
        feature1_icon_layout = QVBoxLayout(feature1_icon_container)
        feature1_icon_layout.setContentsMargins(0, 0, 0, 0)
        feature1_icon_layout.addWidget(feature1_icon)
        
        feature1_title = QLabel("Object Detection")
        feature1_title.setStyleSheet("font-weight: bold; font-size: 16px; margin-top: 10px;")
        feature1_desc = QLabel("Real-time detection of people, objects, and activities with high accuracy")
        feature1_desc.setWordWrap(True)
        feature1_desc.setStyleSheet("color: #6B7280; font-size: 14px;")
        feature1.addWidget(feature1_icon_container, 0, Qt.AlignLeft)
        feature1.addWidget(feature1_title)
        feature1.addWidget(feature1_desc)
        feature1.addStretch()
        
        # Feature 2
        feature2 = QVBoxLayout()
        feature2_icon_container = QLabel()
        feature2_icon_container.setFixedSize(60, 60)
        feature2_icon_container.setStyleSheet("""
            background-color: #FEE2E2;
            border-radius: 30px;
            padding: 15px;
        """)
        feature2_icon = QLabel("🎯")
        feature2_icon.setStyleSheet("font-size: 24px; color: #EF4444;")
        feature2_icon_layout = QVBoxLayout(feature2_icon_container)
        feature2_icon_layout.setContentsMargins(0, 0, 0, 0)
        feature2_icon_layout.addWidget(feature2_icon)
        
        feature2_title = QLabel("Region Tracking")
        feature2_title.setStyleSheet("font-weight: bold; font-size: 16px; margin-top: 10px;")
        feature2_desc = QLabel("Define custom areas of interest for targeted monitoring and alerts")
        feature2_desc.setWordWrap(True)
        feature2_desc.setStyleSheet("color: #6B7280; font-size: 14px;")
        feature2.addWidget(feature2_icon_container, 0, Qt.AlignLeft)
        feature2.addWidget(feature2_title)
        feature2.addWidget(feature2_desc)
        feature2.addStretch()
        
        # Feature 3
        feature3 = QVBoxLayout()
        feature3_icon_container = QLabel()
        feature3_icon_container.setFixedSize(60, 60)
        feature3_icon_container.setStyleSheet("""
            background-color: #DBEAFE;
            border-radius: 30px;
            padding: 15px;
        """)
        feature3_icon = QLabel("🧠")
        feature3_icon.setStyleSheet("font-size: 24px; color: #3B82F6;")
        feature3_icon_layout = QVBoxLayout(feature3_icon_container)
        feature3_icon_layout.setContentsMargins(0, 0, 0, 0)
        feature3_icon_layout.addWidget(feature3_icon)
        
        feature3_title = QLabel("AI Reasoning")
        feature3_title.setStyleSheet("font-weight: bold; font-size: 16px; margin-top: 10px;")
        feature3_desc = QLabel("Smart event detection and anomaly identification using advanced algorithms")
        feature3_desc.setWordWrap(True)
        feature3_desc.setStyleSheet("color: #6B7280; font-size: 14px;")
        feature3.addWidget(feature3_icon_container, 0, Qt.AlignLeft)
        feature3.addWidget(feature3_title)
        feature3.addWidget(feature3_desc)
        feature3.addStretch()
        
        # Feature 4
        feature4 = QVBoxLayout()
        feature4_icon_container = QLabel()
        feature4_icon_container.setFixedSize(60, 60)
        feature4_icon_container.setStyleSheet("""
            background-color: #D1FAE5;
            border-radius: 30px;
            padding: 15px;
        """)
        feature4_icon = QLabel("📊")
        feature4_icon.setStyleSheet("font-size: 24px; color: #10B981;")
        feature4_icon_layout = QVBoxLayout(feature4_icon_container)
        feature4_icon_layout.setContentsMargins(0, 0, 0, 0)
        feature4_icon_layout.addWidget(feature4_icon)
        
        feature4_title = QLabel("Analytics")
        feature4_title.setStyleSheet("font-weight: bold; font-size: 16px; margin-top: 10px;")
        feature4_desc = QLabel("Comprehensive data visualization and reporting for actionable insights")
        feature4_desc.setWordWrap(True)
        feature4_desc.setStyleSheet("color: #6B7280; font-size: 14px;")
        feature4.addWidget(feature4_icon_container, 0, Qt.AlignLeft)
        feature4.addWidget(feature4_title)
        feature4.addWidget(feature4_desc)
        feature4.addStretch()
        
        features_grid.addLayout(feature1)
        features_grid.addLayout(feature2)
        features_grid.addLayout(feature3)
        features_grid.addLayout(feature4)
        
        features_layout.addWidget(features_title)
        features_layout.addLayout(features_grid)
        
        layout.addWidget(features_frame)
        layout.addStretch()
    
    def setup_roi_tab(self):
        """Setup the ROI configuration tab."""
        self.roi_widget = ROIWidget()
        
        # Connect signals
        self.roi_widget.roi_updated.connect(self.update_roi_data)
        
        self.tabs.addTab(self.roi_widget, "ROI Configuration")
    
    def setup_surveillance_tab(self):
        """Setup the surveillance tab with video display and event log."""
        surveillance_widget = QWidget()
        layout = QHBoxLayout()
        
        # Left side - Video display
        self.video_widget = VideoWidget()
        self.video_widget.event_detected.connect(self.handle_event_detection)
        layout.addWidget(self.video_widget, 3)  # 3:1 ratio
        
        # Right side - Event log
        self.event_logger = EventLogger()
        layout.addWidget(self.event_logger, 1)  # 3:1 ratio
        
        surveillance_widget.setLayout(layout)
        self.tabs.addTab(surveillance_widget, "Surveillance View")
    
    def setup_analytics_tab(self):
        """Setup the analytics dashboard tab."""
        # Create analytics dashboard instance
        self.analytics_dashboard = AnalyticsDashboard()
        
        # Add to tabs
        self.tabs.addTab(self.analytics_dashboard, "Analytics Dashboard")
    
    def select_video_file(self):
        """Open file dialog to select a video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        
        if file_path:
            self.video_path = file_path
            self.status_bar.showMessage(f"Video loaded: {file_path}")
            
            # Update video widget with the frame
            self.video_widget.load_video(file_path)
            
            # Update ROI widget with the first frame
            self.roi_widget.load_frame(self.video_widget.get_current_frame())
            
            # Switch to ROI tab
            self.tabs.setCurrentIndex(1)
    
    def connect_to_camera(self):
        """Connect to camera device."""
        # Placeholder for camera connection
        # In a real implementation, you would show a dialog to select camera index
        try:
            # For now, just use camera index 0
            camera_index = 0
            self.video_path = camera_index
            
            # Test camera connection
            self.video_widget.load_video(camera_index, is_camera=True)
            
            self.status_bar.showMessage(f"Camera connected: {camera_index}")
            
            # Update ROI widget with the first frame
            self.roi_widget.load_frame(self.video_widget.get_current_frame())
            
            # Switch to ROI tab
            self.tabs.setCurrentIndex(1)
        except Exception as e:
            QMessageBox.critical(self, "Camera Error", f"Failed to connect to camera: {str(e)}")
    
    def connect_to_rtsp(self):
        """Connect to RTSP stream."""
        # Create a dialog to get RTSP URL
        dialog = QDialog(self)
        dialog.setWindowTitle("Connect to RTSP Stream")
        dialog.setMinimumWidth(500)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
                color: #1F2937;
            }
            QLabel {
                color: #1F2937;
                font-size: 14px;
                margin-bottom: 5px;
            }
            QLineEdit {
                background-color: #F9FAFB;
                color: #1F2937;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
            QTextEdit {
                background-color: #F9FAFB;
                color: #1F2937;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                padding: 8px;
                font-size: 13px;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Instructions
        instruction_label = QLabel("Enter the RTSP stream URL:")
        instruction_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(instruction_label)
        
        # URL input field with label
        url_layout = QHBoxLayout()
        url_label = QLabel("URL:")
        url_layout.addWidget(url_label)
        
        url_input = QLineEdit()
        url_input.setPlaceholderText("rtsp://...")
        url_layout.addWidget(url_input, 1)
        layout.addLayout(url_layout)
        
        # Example
        example_label = QLabel("Example formats:")
        example_label.setStyleSheet("color: #B0B3C0; font-size: 13px; margin-top: 5px;")
        layout.addWidget(example_label)
        
        examples = QTextEdit()
        examples.setReadOnly(True)
        examples.setFixedHeight(100)
        examples.setHtml("""
            <ul>
                <li>rtsp://username:password@ip_address:port/path</li>
                <li>rtsp://192.168.1.100:554/stream1</li>
                <li>rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4</li>
            </ul>
        """)
        layout.addWidget(examples)
        
        # Help information
        help_label = QLabel("Troubleshooting:")
        help_label.setStyleSheet("color: #B0B3C0; font-size: 13px; margin-top: 5px;")
        layout.addWidget(help_label)
        
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setFixedHeight(100)
        help_text.setHtml("""
            <ul>
                <li>Ensure the RTSP stream is active and accessible from your network</li>
                <li>Include username and password if authentication is required</li>
                <li>Verify the port number (554 is standard for RTSP)</li>
                <li>For testing, try a public RTSP stream like the Wowza example</li>
            </ul>
        """)
        layout.addWidget(help_text)
        
        # Connect button
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 10, 0, 0)
        
        connect_button = QPushButton("Connect")
        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet("background-color: #6B7280;")
        
        button_layout.addStretch()
        button_layout.addWidget(connect_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        # Connect signals
        connect_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        
        # Validate URL function
        def validate_url():
            url = url_input.text().strip()
            if url and url.startswith("rtsp://"):
                connect_button.setEnabled(True)
            else:
                connect_button.setEnabled(False)
        
        # Set initial state and connect textChanged signal
        connect_button.setEnabled(False)
        url_input.textChanged.connect(validate_url)
        
        # Show dialog
        if dialog.exec() == QDialog.Accepted:
            rtsp_url = url_input.text().strip()
            if rtsp_url:
                try:
                    # Update status before attempting connection
                    self.status_bar.showMessage(f"Connecting to RTSP stream: {rtsp_url}")
                    
                    # Process events to update UI
                    QApplication.processEvents()
                    
                    # Load the RTSP stream
                    self.video_path = rtsp_url
                    
                    # Test connection to the stream
                    self.video_widget.load_video(rtsp_url, is_camera=True)
                    
                    # Update UI
                    self.status_bar.showMessage(f"RTSP stream connected: {rtsp_url}")
                    
                    # Update ROI widget with the first frame
                    self.roi_widget.load_frame(self.video_widget.get_current_frame())
                    
                    # Switch to ROI tab
                    self.tabs.setCurrentIndex(1)
                except Exception as e:
                    error_message = f"Failed to connect to RTSP stream: {str(e)}"
                    print(error_message)
                    QMessageBox.critical(self, "Stream Error", error_message)
                    
                    # Reset status
                    self.status_bar.showMessage("RTSP connection failed")
            else:
                QMessageBox.warning(self, "Input Error", "Please enter a valid RTSP URL.")
    
    def update_roi_data(self, roi_areas, roi_colors):
        """Receive updated ROI data from the ROI widget."""
        self.roi_areas = roi_areas
        self.roi_colors = roi_colors
        
        # Update video widget with ROI areas
        self.video_widget.set_roi_areas(roi_areas, roi_colors)
        
        # If ROIs are defined, enable the surveillance tab
        if roi_areas:
            self.tabs.setTabEnabled(2, True)
            self.status_bar.showMessage(f"ROI configuration completed with {len(roi_areas)} areas")
            
            # Show prompt to move to surveillance view
            reply = QMessageBox.question(
                self, 
                "ROI Configuration Complete",
                "Region of Interest configuration is complete. Would you like to proceed to Surveillance view?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.tabs.setCurrentIndex(2)
        else:
            self.tabs.setTabEnabled(2, False)
    
    def handle_event_detection(self, event):
        """Handle events detected by the video widget."""
        # Add event to the event logger
        self.event_logger.add_event(event)
        
        # Show status message
        self.status_bar.showMessage(f"Event detected: {event['type']} - {event['description']}", 3000)
        
        # Also update analytics dashboard with current events list
        self.analytics_dashboard.set_events(self.event_logger.get_events()) 