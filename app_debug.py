import streamlit as st
import os
import traceback
from app.components.detection.detector import YOLODetector
from app.components.tracking.tracker import DeepSORTTracker
from app.components.roi.roi_manager import ROIManager
from app.components.analytics.analytics import AnalyticsManager
from app.utils.video_utils import get_video_frame, process_video
from app.components.symbolic_reasoning.llm_reasoning import SymbolicReasoner

st.set_page_config(
    page_title="Vision AI - Smart Surveillance (Debug Mode)",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'roi_defined' not in st.session_state:
    st.session_state.roi_defined = False
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'first_frame' not in st.session_state:
    st.session_state.first_frame = None
if 'roi_areas' not in st.session_state:
    st.session_state.roi_areas = {}
if 'events' not in st.session_state:
    st.session_state.events = []
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Home"
if 'error_messages' not in st.session_state:
    st.session_state.error_messages = []

# Debug info section in sidebar
with st.sidebar:
    st.title("Vision AI 👁️ (Debug Mode)")
    st.subheader("Smart Surveillance System")
    
    # Navigation
    selected_tab = st.radio(
        "Navigation",
        ["Home", "ROI Configuration", "Surveillance View", "Analytics Dashboard", "Debug Info"]
    )
    st.session_state.current_tab = selected_tab
    
    # App information
    with st.expander("About"):
        st.write("""
        Vision AI is a surveillance system designed for retail businesses.
        It uses computer vision and AI to detect theft, track footfall,
        and monitor inventory interactions.
        """)
    
    # Debug info
    with st.expander("Debug Information", expanded=True):
        st.write("**Session State**")
        st.write(f"- Video Path: {st.session_state.video_path}")
        st.write(f"- ROI Defined: {st.session_state.roi_defined}")
        st.write(f"- ROI Areas Count: {len(st.session_state.roi_areas)}")
        st.write(f"- First Frame: {'Loaded' if st.session_state.first_frame is not None else 'None'}")
        st.write(f"- Processed Video: {'Processed' if st.session_state.processed_video is not None else 'None'}")
        st.write(f"- Events Count: {len(st.session_state.events)}")
        
        # Error messages
        if st.session_state.error_messages:
            st.warning("### Errors Detected")
            for error in st.session_state.error_messages:
                st.error(error)

# Main content based on selected tab
if st.session_state.current_tab == "Home":
    st.title("Welcome to Vision AI")
    st.write("Upload a video or provide a camera URL to get started.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        if uploaded_file:
            try:
                # Save uploaded file
                if not os.path.exists("app/data"):
                    os.makedirs("app/data")
                    
                with open(os.path.join("app/data", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.video_path = os.path.join("app/data", uploaded_file.name)
                st.success(f"Video uploaded: {uploaded_file.name}")
                
                # Extract first frame for ROI marking
                try:
                    st.session_state.first_frame = get_video_frame(st.session_state.video_path, frame_number=0)
                    
                    # Navigate to ROI configuration
                    st.session_state.current_tab = "ROI Configuration"
                    st.rerun()
                except Exception as e:
                    error_msg = f"Error extracting first frame: {str(e)}"
                    st.session_state.error_messages.append(error_msg)
                    st.error(error_msg)
            except Exception as e:
                error_msg = f"Error processing video upload: {str(e)}"
                st.session_state.error_messages.append(error_msg)
                st.error(error_msg)
    
    with col2:
        camera_url = st.text_input("Or provide a camera URL:")
        if camera_url and st.button("Connect to Camera"):
            try:
                # For URL, we'll use 0 as placeholder for now
                st.session_state.video_path = camera_url
                st.session_state.first_frame = get_video_frame(0, is_camera=True)
                
                # Navigate to ROI configuration
                st.session_state.current_tab = "ROI Configuration"
                st.rerun()
            except Exception as e:
                error_msg = f"Error connecting to camera: {str(e)}"
                st.session_state.error_messages.append(error_msg)
                st.error(error_msg)

elif st.session_state.current_tab == "ROI Configuration":
    st.title("Configure Regions of Interest")
    
    if st.session_state.first_frame is not None:
        try:
            roi_manager = ROIManager()
            st.session_state.roi_areas = roi_manager.define_roi(st.session_state.first_frame)
            
            if st.session_state.roi_areas and st.button("Start Surveillance"):
                st.session_state.roi_defined = True
                st.session_state.current_tab = "Surveillance View"
                st.rerun()
        except Exception as e:
            error_msg = f"Error in ROI configuration: {str(e)}\n{traceback.format_exc()}"
            st.session_state.error_messages.append(error_msg)
            st.error(error_msg)
    else:
        st.error("No video loaded. Please go back to the Home tab.")

elif st.session_state.current_tab == "Surveillance View":
    st.title("Live Surveillance View")
    
    if not st.session_state.roi_defined:
        st.warning("Please define ROIs first.")
        if st.button("Go to ROI Configuration"):
            st.session_state.current_tab = "ROI Configuration"
            st.rerun()
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Initialize detection components with error handling
            try:
                # Show debugging info
                st.info("**Debug Info**: Initializing detection components")
                
                # Initialize detector
                detector = YOLODetector()
                st.success("YOLODetector initialized")
                
                # Initialize tracker
                tracker = DeepSORTTracker()
                st.success("DeepSORTTracker initialized")
                
                # Initialize reasoner with fallback
                try:
                    reasoner = SymbolicReasoner()
                    st.success("SymbolicReasoner initialized")
                except Exception as e:
                    error_msg = f"SymbolicReasoner init error (using fallback): {str(e)}"
                    st.session_state.error_messages.append(error_msg)
                    st.warning(error_msg)
                    # Use a fallback reasoner that doesn't require llama-cpp
                    class FallbackReasoner:
                        def analyze_event(self, event_data, track_data):
                            return "Reasoning unavailable (fallback mode)"
                    reasoner = FallbackReasoner()
                
                if st.session_state.processed_video is None:
                    with st.spinner("Processing video..."):
                        try:
                            # Debug ROI areas
                            st.info(f"**Debug Info**: Processing video with ROI areas: {st.session_state.roi_areas}")
                            
                            # Process the video with detection, tracking, and ROI analysis
                            st.session_state.processed_video, st.session_state.events = process_video(
                                st.session_state.video_path,
                                detector,
                                tracker,
                                st.session_state.roi_areas,
                                reasoner
                            )
                            st.success("Video processed successfully")
                        except Exception as e:
                            error_msg = f"Error processing video: {str(e)}\n{traceback.format_exc()}"
                            st.session_state.error_messages.append(error_msg)
                            st.error(error_msg)
                
                # Display processed video
                if st.session_state.processed_video:
                    st.video(st.session_state.processed_video)
                else:
                    st.error("No processed video available. Check debug info for errors.")
            
            except Exception as e:
                error_msg = f"Error in Surveillance View: {str(e)}\n{traceback.format_exc()}"
                st.session_state.error_messages.append(error_msg)
                st.error(error_msg)
        
        with col2:
            st.subheader("Event Log")
            if st.session_state.events:
                for event in st.session_state.events:
                    with st.expander(f"{event['type']} at {event['timestamp']}"):
                        st.write(f"Description: {event['description']}")
                        if 'reasoning' in event:
                            st.write(f"AI Analysis: {event['reasoning']}")
                        if 'thumbnail' in event:
                            st.image(event.get('thumbnail', None), caption="Event Thumbnail")
            else:
                st.info("No events detected yet.")

elif st.session_state.current_tab == "Analytics Dashboard":
    st.title("Analytics Dashboard")
    
    if not st.session_state.events:
        st.warning("No events captured yet. Run surveillance first.")
        if st.button("Go to Surveillance View"):
            st.session_state.current_tab = "Surveillance View"
            st.rerun()
    else:
        try:
            analytics = AnalyticsManager(st.session_state.events, st.session_state.roi_areas)
            analytics.display_dashboard()
        except Exception as e:
            error_msg = f"Error in Analytics Dashboard: {str(e)}\n{traceback.format_exc()}"
            st.session_state.error_messages.append(error_msg)
            st.error(error_msg)

elif st.session_state.current_tab == "Debug Info":
    st.title("Debug Information")
    
    # Display detailed debug information
    st.subheader("System Info")
    st.code(f"""
Python Path: {os.environ.get('PYTHONPATH', 'Not set')}
Current Directory: {os.getcwd()}
Video Path: {st.session_state.video_path}
    """)
    
    # Check module imports
    st.subheader("Module Import Check")
    
    import_checks = {
        "streamlit": "import streamlit",
        "opencv-python": "import cv2",
        "numpy": "import numpy",
        "pandas": "import pandas",
        "torch": "import torch",
        "ultralytics": "from ultralytics import YOLO",
        "llama-cpp-python": "from llama_cpp import Llama",
        "plotly": "import plotly.express",
        "PIL": "from PIL import Image",
        "streamlit-drawable-canvas": "from streamlit_drawable_canvas import st_canvas"
    }
    
    results = {}
    for module_name, import_statement in import_checks.items():
        try:
            exec(import_statement)
            results[module_name] = "✅ Installed"
        except ImportError as e:
            results[module_name] = f"❌ Not installed: {str(e)}"
        except Exception as e:
            results[module_name] = f"⚠️ Error: {str(e)}"
    
    # Display results in a table
    st.table({"Module": list(results.keys()), "Status": list(results.values())})
    
    # Display all error messages
    st.subheader("Error Log")
    if st.session_state.error_messages:
        for i, error in enumerate(st.session_state.error_messages):
            st.error(f"Error {i+1}:\n{error}")
    else:
        st.success("No errors recorded.")
    
    # Clear error log button
    if st.button("Clear Error Log"):
        st.session_state.error_messages = []
        st.success("Error log cleared!")
        st.rerun()

# Footer
st.markdown("---")
st.markdown("Vision AI © 2023 | All Rights Reserved | Debug Mode") 