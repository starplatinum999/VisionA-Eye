import sys
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QPushButton, QFileDialog, 
    QStatusBar, QMessageBox, QApplication, QLineEdit,
    QDialog, QFormLayout, QTextEdit, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QFont, QPalette, QColor
from PySide6.QtWidgets import QStyle

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
        
        # Set light theme for entire application
        self.apply_light_theme()
        
        # Shared data between tabs
        self.video_path = None
        self.roi_areas = {}
        self.roi_colors = {}
        
        # Create tab widget (similar to Streamlit tabs)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Style the tabs
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #E4E7EB;
                background-color: #FFFFFF;
                border-radius: 0 0 8px 8px;
            }
            QTabBar::tab {
                background-color: #F5F7FA;
                color: #4B5563;
                border: 1px solid #E4E7EB;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 16px;
                margin-right: 4px;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background-color: #FFFFFF;
                color: #2563EB;
                border-bottom: 2px solid #2563EB;
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
            }
        """)
        
        # Create tabs for each section
        self.setup_home_tab()
        self.setup_roi_tab()
        self.setup_surveillance_tab()
        self.setup_analytics_tab()
    
    def apply_light_theme(self):
        """Apply a light theme to the entire application."""
        app = QApplication.instance()
        
        # Set fusion style for a more modern look
        app.setStyle("Fusion")
        
        # Create a light palette with a professional color scheme
        light_palette = QPalette()
        
        # Base colors
        light_palette.setColor(QPalette.Window, QColor("#FFFFFF"))         # White
        light_palette.setColor(QPalette.WindowText, QColor("#1F2937"))     # Charcoal text
        light_palette.setColor(QPalette.Base, QColor("#F5F7FA"))          # Very light gray
        light_palette.setColor(QPalette.AlternateBase, QColor("#E4E7EB"))  # Light gray
        light_palette.setColor(QPalette.ToolTipBase, QColor("#F5F7FA"))
        light_palette.setColor(QPalette.ToolTipText, QColor("#1F2937"))
        
        # Text and button colors
        light_palette.setColor(QPalette.Text, QColor("#1F2937"))          # Charcoal text
        light_palette.setColor(QPalette.Button, QColor("#F9FAFB"))        # Lighter gray for buttons
        light_palette.setColor(QPalette.ButtonText, QColor("#1F2937"))    # Charcoal text
        light_palette.setColor(QPalette.BrightText, QColor("#EF4444"))    # Error Red
        
        # Highlight and link colors
        light_palette.setColor(QPalette.Link, QColor("#2563EB"))          # Blue links
        light_palette.setColor(QPalette.Highlight, QColor("#3B82F6"))     # Blue highlight
        light_palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF")) # White on highlight
        
        # Set the palette
        app.setPalette(light_palette)
        
        # Set the default style sheet
        app.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
                color: #1F2937;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
            QLabel {
                background: transparent;
                border: none;
                padding: 0;
                color: #1F2937;
            }
            QPushButton {
                background-color: #F9FAFB;
                color: #1F2937;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 500;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #EFF6FF;
                border-color: #93C5FD;
                color: #1D4ED8;
            }
            QPushButton:pressed {
                background-color: #DBEAFE;
                border-color: #60A5FA;
            }
            QLineEdit, QComboBox {
                background-color: #F9FAFB;
                color: #1F2937;
                border: 1px solid #D1D5DB;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #3B82F6;
                border-width: 2px;
            }
            QStatusBar {
                background-color: #F5F7FA;
                color: #4B5563;
                border-top: 1px solid #E4E7EB;
            }
            QScrollBar:vertical {
                background-color: #F5F7FA;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #D1D5DB;
                min-height: 30px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #9CA3AF;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background-color: #F5F7FA;
                height: 12px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background-color: #D1D5DB;
                min-width: 30px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #9CA3AF;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
    
    def setup_home_tab(self):
        """Setup the home tab with file/camera selection."""
        home_widget = QWidget()
        main_layout = QVBoxLayout(home_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setAlignment(Qt.AlignCenter)
        
        # Header section
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setAlignment(Qt.AlignCenter)
        header_layout.setSpacing(10)
        
        # Title
        title_label = QLabel("Welcome to Vision AI")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setStyleSheet("color: #3B82F6;")
        title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Smart Surveillance System")
        subtitle_label.setFont(QFont("Arial", 16))
        subtitle_label.setStyleSheet("color: #111827; margin-bottom: 5px;")
        subtitle_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(subtitle_label)
        
        # Description
        description = QLabel("Upload a video or connect to a stream for AI-powered analysis")
        description.setStyleSheet("color: #4B5563; font-size: 14px; margin-bottom: 20px;")
        description.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(description)
        
        main_layout.addWidget(header)
        
        # Content container (centered and width-limited)
        content_container = QWidget()
        content_container.setMaximumWidth(800)
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 20, 0, 0)
        content_layout.setSpacing(20)
        
        # File selection section
        file_section = QFrame()
        file_section.setFrameShape(QFrame.StyledPanel)
        file_section.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 1px solid #E4E7EB;
                border-radius: 8px;
            }
        """)
        
        file_layout = QVBoxLayout(file_section)
        file_layout.setContentsMargins(20, 20, 20, 20)
        file_layout.setSpacing(15)
        
        # Section title
        file_title = QLabel("Video File")
        file_title.setFont(QFont("Arial", 16, QFont.Bold))
        file_title.setStyleSheet("color: #111827;")
        file_layout.addWidget(file_title)
        
        # Description
        file_desc = QLabel("Select a video file for analysis")
        file_desc.setStyleSheet("color: #6B7280;")
        file_layout.addWidget(file_desc)
        
        # Button and status
        file_controls = QHBoxLayout()
        file_controls.setSpacing(15)
        
        upload_button = QPushButton("Select Video File")
        upload_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        upload_button.clicked.connect(self.select_video_file)
        upload_button.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                min-width: 180px;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)
        
        file_controls.addWidget(upload_button)
        
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("color: #6B7280;")
        file_controls.addWidget(self.file_path_label, 1)
        
        file_layout.addLayout(file_controls)
        content_layout.addWidget(file_section)
        
        # RTSP stream section
        rtsp_section = QFrame()
        rtsp_section.setFrameShape(QFrame.StyledPanel)
        rtsp_section.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border: 1px solid #E4E7EB;
                border-radius: 8px;
            }
        """)
        
        rtsp_layout = QVBoxLayout(rtsp_section)
        rtsp_layout.setContentsMargins(20, 20, 20, 20)
        rtsp_layout.setSpacing(15)
        
        # Section title
        rtsp_title = QLabel("Camera Stream")
        rtsp_title.setFont(QFont("Arial", 16, QFont.Bold))
        rtsp_title.setStyleSheet("color: #111827;")
        rtsp_layout.addWidget(rtsp_title)
        
        # Description
        rtsp_desc = QLabel("Connect to an RTSP camera stream")
        rtsp_desc.setStyleSheet("color: #6B7280;")
        rtsp_layout.addWidget(rtsp_desc)
        
        # Button and status
        rtsp_controls = QHBoxLayout()
        rtsp_controls.setSpacing(15)
        
        rtsp_button = QPushButton("Connect to RTSP Stream")
        rtsp_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_DriveNetIcon))
        rtsp_button.clicked.connect(self.connect_to_rtsp)
        rtsp_button.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
                min-width: 180px;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:pressed {
                background-color: #1D4ED8;
            }
        """)
        
        rtsp_controls.addWidget(rtsp_button)
        
        self.camera_status_label = QLabel("No camera connected")
        self.camera_status_label.setStyleSheet("color: #6B7280;")
        rtsp_controls.addWidget(self.camera_status_label, 1)
        
        rtsp_layout.addLayout(rtsp_controls)
        content_layout.addWidget(rtsp_section)
        
        # Status section
        status_section = QFrame()
        status_section.setFrameShape(QFrame.StyledPanel)
        status_section.setStyleSheet("""
            QFrame {
                background-color: #F9FAFB;
                border: 1px solid #E4E7EB;
                border-radius: 8px;
            }
        """)
        
        status_layout = QHBoxLayout(status_section)
        status_layout.setContentsMargins(20, 15, 20, 15)
        
        # Model status
        model_widget = QWidget()
        model_layout = QVBoxLayout(model_widget)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(3)
        
        model_label = QLabel("AI Model")
        model_label.setStyleSheet("color: #6B7280; font-size: 12px;")
        model_layout.addWidget(model_label)
        
        model_status = QLabel("Loaded")
        model_status.setStyleSheet("color: #10B981; font-weight: bold;")
        model_layout.addWidget(model_status)
        
        status_layout.addWidget(model_widget, 1)
        
        # System status
        system_widget = QWidget()
        system_layout = QVBoxLayout(system_widget)
        system_layout.setContentsMargins(0, 0, 0, 0)
        system_layout.setSpacing(3)
        
        system_label = QLabel("System")
        system_label.setStyleSheet("color: #6B7280; font-size: 12px;")
        system_layout.addWidget(system_label)
        
        system_status = QLabel("Ready")
        system_status.setStyleSheet("color: #10B981; font-weight: bold;")
        system_layout.addWidget(system_status)
        
        status_layout.addWidget(system_widget, 1)
        
        # Last activity
        activity_widget = QWidget()
        activity_layout = QVBoxLayout(activity_widget)
        activity_layout.setContentsMargins(0, 0, 0, 0)
        activity_layout.setSpacing(3)
        
        activity_label = QLabel("Last Activity")
        activity_label.setStyleSheet("color: #6B7280; font-size: 12px;")
        activity_layout.addWidget(activity_label)
        
        activity_status = QLabel("None")
        activity_status.setStyleSheet("color: #6B7280; font-weight: bold;")
        activity_layout.addWidget(activity_status)
        
        status_layout.addWidget(activity_widget, 1)
        
        content_layout.addWidget(status_section)
        
        # Add content to main layout (centered)
        container_layout = QHBoxLayout()
        container_layout.addStretch()
        container_layout.addWidget(content_container)
        container_layout.addStretch()
        main_layout.addLayout(container_layout)
        
        # Footer
        footer = QLabel("Vision AI - Real-time object detection and tracking")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #9CA3AF; margin-top: 20px;")
        main_layout.addWidget(footer)
        
        self.tabs.addTab(home_widget, "Home")
    
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
            self.file_path_label.setText(file_path)
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
            
            self.camera_status_label.setText(f"Connected to camera {camera_index}")
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
                    self.camera_status_label.setText(f"Connecting to RTSP stream...")
                    self.status_bar.showMessage(f"Connecting to RTSP stream: {rtsp_url}")
                    
                    # Process events to update UI
                    QApplication.processEvents()
                    
                    # Load the RTSP stream
                    self.video_path = rtsp_url
                    
                    # Test connection to the stream
                    self.video_widget.load_video(rtsp_url, is_camera=True)
                    
                    # Update UI
                    self.camera_status_label.setText(f"Connected to RTSP stream")
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
                    self.camera_status_label.setText("No camera connected")
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