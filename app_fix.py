import sys
import os

# Get the absolute path to the current directory
base_dir = os.path.abspath(os.path.dirname(__file__))

# Add the base directory to the Python path if not already there
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
    print(f"Added {base_dir} to Python path")

# Import check
try:
    from app.components.detection.detector import YOLODetector
    from app.components.tracking.tracker import DeepSORTTracker
    from app.components.roi.roi_manager import ROIManager
    from app.components.analytics.analytics import AnalyticsManager
    from app.components.symbolic_reasoning.llm_reasoning import SymbolicReasoner
    from app.utils.video_utils import get_video_frame, process_video
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

print("Python path:")
for path in sys.path:
    print(f"  {path}")
