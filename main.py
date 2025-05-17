#!/usr/bin/env python3
"""
Vision AI - Smart Surveillance System
Desktop Application Main Entry Point

This application provides real-time object detection, tracking,
and event detection for surveillance video.
"""

import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication, Qt
from app.ui.main_window import MainWindow

# Add application information
QCoreApplication.setApplicationName("Vision AI")
QCoreApplication.setOrganizationName("VisionA-Eye")
QCoreApplication.setApplicationVersion("1.0.0")

def main():
    """Main application entry point."""
    # Create application
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 