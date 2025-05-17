# Vision AI - Smart Surveillance System

An AI-powered surveillance system designed for medium-sized businesses like supermarkets, jewelry stores, and retail outlets. The system uses computer vision and symbolic reasoning to monitor, detect, and analyze activity from CCTV footage.

## Core Functionalities

- **Theft Detection**: Identify suspicious object removals and unauthorized exits
- **ROI Marking**: Define zones (entry, exit, shelf, cashier) manually or automatically
- **Anomalous Behavior Analysis**: Detect shoplifting, lingering, unusual movement
- **Footfall Tracking**: Monitor movement density and dwell time across areas
- **Inventory Interaction**: Track items picked, moved, or left behind

## Technology Stack

- **Computer Vision**: YOLOv8 (object detection) + Deep SORT (object tracking)
- **Symbolic Reasoning**: Local LLM (DeepSeek-Coder via llama.cpp)
- **Interface**: Streamlit-based UI running locally

## Setup & Installation

1. Clone this repository:

```
git clone https://github.com/yourusername/vision-ai.git
cd vision-ai
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Download the required models:

```
python -m app.utils.download_models
```

4. Start the application:

```
streamlit run app.py
```

## Usage Guide

1. **Upload Video**: Upload a video file or provide a camera URL on the Home screen
2. **Define ROIs**: Mark regions of interest on the first frame (Entry, Exit, Shelves, etc.)
3. **Analyze**: View real-time detection with event logging in the Surveillance View
4. **Review**: Access the Analytics Dashboard for detailed insights and potential security threats

## Project Structure

```
vision-ai/
├── app/
│   ├── components/
│   │   ├── analytics/       # Analytics visualizations
│   │   ├── detection/       # YOLO object detection
│   │   ├── roi/             # Region of Interest management
│   │   ├── symbolic_reasoning/ # LLM-based reasoning
│   │   └── tracking/        # DeepSORT object tracking
│   ├── data/                # Uploaded videos storage
│   ├── models/              # Model weights storage
│   └── utils/               # Utility functions
├── app.py                   # Main Streamlit application
└── requirements.txt         # Project dependencies
```

## Requirements

- Python 3.8+
- MacBook M1 or equivalent for local processing
- Approximately 2GB disk space for models

## Troubleshooting

If you encounter issues running the application, use the included fix script:

```
./fix_environment.sh
```

### Common Issues

1. **Streamlit command not found**

   - The fix script will install streamlit and add it to your PATH

2. **Python version mismatch**

   - The fix script can create a dedicated virtual environment with the correct dependencies

3. **Missing packages**

   - If you see import errors like `No module named 'streamlit_drawable_canvas'`, run the fix script

4. **Manual PATH setup**
   - If streamlit is installed but not in PATH, add pip's bin directory to your PATH:
   ```
   export PATH="$(python -c "import site; print(site.USER_BASE + '/bin')"):$PATH"
   ```
   - Add this line to your `~/.bashrc` or `~/.zshrc` for persistence
