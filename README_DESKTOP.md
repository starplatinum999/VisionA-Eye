# Vision AI Desktop Application

This desktop application provides real-time video surveillance with object detection, tracking, and event detection capabilities.

## Features

- **Real-time processing**: No pre-processing step, see results as the video plays
- **YOLO object detection**: Uses YOLOv8 for real-time object detection
- **DeepSORT tracking**: Tracks objects across frames to maintain identity
- **Region of Interest (ROI) configuration**: Define and customize regions for event detection
- **Event logging**: Automatically detects and logs events in real-time
- **ROI presets**: Quick setup with predefined ROI layouts
- **Camera support**: Works with video files or live camera feeds

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/VisionA-Eye.git
   cd VisionA-Eye
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python main.py
   ```

## Usage

### 1. Load Video or Connect Camera

- On the Home tab, click "Select Video File" to choose a video for analysis
- Or click "Connect to Camera" to use a webcam or IP camera feed

### 2. Configure Regions of Interest (ROIs)

- Use the ROI Configuration tab to define areas of interest
- Click "Add ROI" and draw on the video frame
- Use presets for common layouts like "Retail Store Layout" or "Checkout Counters"
- Save ROI configurations for future use

### 3. Start Surveillance

- On the Surveillance View tab, click "Start Processing"
- The video will play with real-time detection and tracking
- Events are automatically logged in the right panel

### 4. Review Events

- The Event Log displays all detected events with thumbnails
- Events include:
  - ROI Transitions
  - Person at Shelf
  - Item Pickup
  - Potential Theft
  - Dwell Time Alerts

## Customization

- The application uses the existing YOLO and DeepSORT components
- Models can be updated or changed in the app/components directory
- ROI detection logic can be customized in the video_widget.py file

## Requirements

- Python 3.8+
- PySide6 (Qt for Python)
- OpenCV
- PyTorch
- YOLO v8 (ultralytics)
- DeepSORT

## Converting from Streamlit

This application is a desktop version of the original Streamlit web app. The main differences are:

1. Uses Qt/PySide6 for the UI instead of Streamlit
2. Processes video in real-time instead of pre-processing
3. Maintains the same detection and tracking components
4. Has a more responsive and native interface

## License

See the LICENSE file for details.
