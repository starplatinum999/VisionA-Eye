import streamlit as st
import os
import time
from app.components.detection.detector import YOLODetector
from app.components.tracking.tracker import DeepSORTTracker
from app.components.roi.roi_manager import ROIManager
from app.components.analytics.analytics import AnalyticsManager
from app.utils.video_utils import get_video_frame, process_video
from app.components.symbolic_reasoning.llm_reasoning import SymbolicReasoner

st.set_page_config(
    page_title="Vision AI - Smart Surveillance",
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
if 'roi_colors' not in st.session_state:
    st.session_state.roi_colors = {}
if 'events' not in st.session_state:
    st.session_state.events = []
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Home"
# Progress tracking state
if 'progress_percent' not in st.session_state:
    st.session_state.progress_percent = 0
if 'progress_frame' not in st.session_state:
    st.session_state.progress_frame = 0
if 'progress_total' not in st.session_state:
    st.session_state.progress_total = 0
if 'progress_time' not in st.session_state:
    st.session_state.progress_time = 0

# Define callback functions to ensure session state updates happen
def go_to_surveillance():
    st.session_state.roi_defined = True
    st.session_state.current_tab = "Surveillance View"

# If we have defined ROI areas, make sure roi_defined is set to True
if len(st.session_state.roi_areas) > 0:
    st.session_state.roi_defined = True

# Create a sidebar for navigation
with st.sidebar:
    st.title("Vision AI 👁️")
    st.subheader("Smart Surveillance System")
    
    # Navigation
    selected_tab = st.radio(
        "Navigation",
        ["Home", "ROI Configuration", "Surveillance View", "Analytics Dashboard"]
    )
    st.session_state.current_tab = selected_tab
    
    # Debug info (added temporarily)
    st.write("**Debug State:**")
    st.write(f"ROI Defined: {st.session_state.roi_defined}")
    st.write(f"ROI Areas: {len(st.session_state.roi_areas)}")
    if st.session_state.roi_areas:
        with st.expander("View ROI Details"):
            for name, coords in st.session_state.roi_areas.items():
                st.write(f"{name}: {coords}")
    
    # App information
    with st.expander("About"):
        st.write("""
        Vision AI is a surveillance system designed for retail businesses.
        It uses computer vision and AI to detect theft, track footfall,
        and monitor inventory interactions.
        """)

# Main content based on selected tab
if st.session_state.current_tab == "Home":
    st.title("Welcome to Vision AI")
    st.write("Upload a video or provide a camera URL to get started.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        if uploaded_file:
            # Create data directory if it doesn't exist
            os.makedirs("app/data", exist_ok=True)
            
            # Save uploaded file
            with open(os.path.join("app/data", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.video_path = os.path.join("app/data", uploaded_file.name)
            st.success(f"Video uploaded: {uploaded_file.name}")
            
            # Extract first frame for ROI marking
            st.session_state.first_frame = get_video_frame(st.session_state.video_path, frame_number=0)
            
            # Navigate to ROI configuration
            st.session_state.current_tab = "ROI Configuration"
            st.rerun()
    
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
                st.error(f"Error connecting to camera: {e}")

elif st.session_state.current_tab == "ROI Configuration":
    st.title("Configure Regions of Interest")
    
    if st.session_state.first_frame is not None:
        # Create ROI manager and define ROIs
        roi_manager = ROIManager()
        
        # If we have existing ROIs in session state, load them into the manager
        if st.session_state.roi_areas and st.session_state.roi_colors:
            roi_manager.roi_areas = st.session_state.roi_areas
            roi_manager.roi_colors = st.session_state.roi_colors
        
        defined_roi_areas = roi_manager.define_roi(st.session_state.first_frame)
        
        # Store defined ROI areas and colors in session state
        if defined_roi_areas:
            st.session_state.roi_areas = roi_manager.roi_areas
            st.session_state.roi_colors = roi_manager.roi_colors
            
            # Save ROI configuration to file for persistence
            roi_config_path = os.path.join("app/data", "roi_config.json")
            roi_manager.save_roi_config(roi_config_path)
            st.success(f"ROI configuration saved to {roi_config_path}")
        
        # Add a separator for clarity
        st.markdown("---")
        
        # Display ROI summary and navigation button separately
        if st.session_state.roi_areas:
            st.success(f"✅ {len(st.session_state.roi_areas)} Regions of Interest defined")
            
            # Debug info
            st.write("ROI Areas in session state:", st.session_state.roi_areas)
            
            # Use the callback function for the button
            st.button("Start Surveillance", key="start_surveillance", on_click=go_to_surveillance)
    else:
        st.error("No video loaded. Please go back to the Home tab.")

elif st.session_state.current_tab == "Surveillance View":
    st.title("Live Surveillance View")
    
    # Check for ROI config file and load if needed
    roi_config_path = os.path.join("app/data", "roi_config.json")
    if not st.session_state.roi_areas and os.path.exists(roi_config_path):
        roi_manager = ROIManager()
        if roi_manager.load_roi_config(roi_config_path):
            st.session_state.roi_areas = roi_manager.roi_areas
            st.session_state.roi_colors = roi_manager.roi_colors
            st.session_state.roi_defined = True
            st.success("Loaded ROI configuration from file")
    
    # Display current state for debugging
    st.write(f"Debug - ROI Defined: {st.session_state.roi_defined}")
    st.write(f"Debug - ROI Areas: {st.session_state.roi_areas}")
    
    # Force ROI defined if we have ROI areas (failsafe)
    if not st.session_state.roi_defined and len(st.session_state.roi_areas) > 0:
        st.session_state.roi_defined = True
    
    if not st.session_state.roi_defined or not st.session_state.roi_areas:
        st.warning("Please define ROIs first.")
        if st.button("Go to ROI Configuration"):
            st.session_state.current_tab = "ROI Configuration"
            st.rerun()
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Initialize detector and tracker
            try:
                detector = YOLODetector()
                tracker = DeepSORTTracker()
                
                # Use a try-except for the reasoning component
                try:
                    reasoner = SymbolicReasoner()
                except Exception as e:
                    st.warning(f"Symbolic reasoner error: {e}")
                    # Create a simple fallback reasoner that doesn't use llama-cpp
                    class FallbackReasoner:
                        def analyze_event(self, event_data, track_data):
                            return "Analysis not available (fallback mode)"
                    reasoner = FallbackReasoner()
                
                if st.session_state.processed_video is None:
                    # Add debug information before processing
                    st.info("Starting video processing with the following configuration:")
                    st.write(f"- Video path: {st.session_state.video_path}")
                    st.write(f"- ROI areas: {st.session_state.roi_areas}")
                    
                    # Add timeout and processing options
                    with st.expander("Processing Options", expanded=True):
                        timeout = st.slider("Processing Timeout (seconds)", 
                                           min_value=30, max_value=600, value=120, 
                                           help="Maximum time to spend processing the video")
                        
                        process_every_n_frames = st.checkbox("Process every other frame", value=True,
                                                          help="Skip frames to speed up processing")
                        
                        use_detector = st.checkbox("Enable object detection", value=True,
                                                help="Disable to use placeholder detection")
                        
                        use_tracker = st.checkbox("Enable object tracking", value=True,
                                               help="Disable to use placeholder tracking")
                        
                        use_reasoner = st.checkbox("Enable AI reasoning", value=False,
                                                help="Keep disabled for real-time processing, use in Analytics instead")
                        
                        process_mode = st.radio(
                            "Processing Mode",
                            ["Process and Save", "Real-time Processing"],
                            help="Process and Save creates a video file with processing, Real-time processes the video as it plays"
                        )
                    
                    # Add a bypass option with a more prominent display
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("🚀 Start Processing", 
                                   help="Start video processing with the selected options"):
                            should_process_video = True
                    
                    with col2:
                        if st.button("⏭️ Skip Processing (Use Original Video)", 
                                   help="Skip processing and use the original video"):
                            st.session_state.events = []
                            st.session_state.processed_video = st.session_state.video_path
                            st.rerun()
                    
                    # Only run processing if the Start Processing button was clicked
                    if 'should_process_video' in locals() and should_process_video:
                        try:
                            with st.spinner("Processing video..."):
                                # Confirm ROI areas again
                                if not st.session_state.roi_areas:
                                    st.error("ROI areas are empty. Please go back to ROI Configuration.")
                                    if st.button("Return to ROI Configuration"):
                                        st.session_state.current_tab = "ROI Configuration"
                                        st.rerun()
                                
                                # Debug progress updates
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Add a status display that will show progress
                                progress_status = st.empty()
                                
                                # Setup components based on selected options
                                if use_detector:
                                    status_text.text("Initializing detector...")
                                    progress_bar.progress(10)
                                else:
                                    # Create a simple mock detector if disabled
                                    class MockDetector:
                                        def detect(self, frame):
                                            return []
                                    detector = MockDetector()
                                
                                if use_tracker:
                                    status_text.text("Initializing tracker...")
                                    progress_bar.progress(20)
                                else:
                                    # Create a simple mock tracker if disabled
                                    class MockTracker:
                                        def update(self, frame, detections):
                                            return []
                                    tracker = MockTracker()
                                
                                if use_reasoner:
                                    status_text.text("Initializing AI reasoner...")
                                    progress_bar.progress(30)
                                else:
                                    # Create a simple mock reasoner if disabled
                                    class MockReasoner:
                                        def analyze_event(self, event_data, track_data):
                                            return "AI reasoning disabled"
                                        def analyze_dwell_time(self, event_data, track_data):
                                            return "AI reasoning disabled"
                                    reasoner = MockReasoner()
                                
                                # Run video processing with timeout and progress updates
                                status_text.text("Processing video frames...")
                                progress_bar.progress(40)
                                
                                # Create a safe progress callback 
                                # This will be called from the main thread periodically
                                def safe_progress_update(percent, frame_idx, total_frames, elapsed_time):
                                    # Store progress values in session state to avoid threading issues
                                    st.session_state.progress_percent = percent
                                    st.session_state.progress_frame = frame_idx
                                    st.session_state.progress_total = total_frames
                                    st.session_state.progress_time = elapsed_time
                                    
                                    # Update progress displays (this is safe because it's called from the main thread)
                                    progress_bar.progress(int(percent))
                                    status_text.text(f"Processing video... {percent:.1f}%")
                                    
                                    # Display detailed status
                                    progress_info = f"""
                                    **Processing Status:**
                                    - Frame: {frame_idx}/{total_frames}
                                    - Progress: {percent:.1f}%
                                    - Elapsed Time: {elapsed_time:.1f} seconds
                                    - Speed: {frame_idx/max(elapsed_time, 0.1):.1f} frames/second
                                    """
                                    progress_status.markdown(progress_info)
                                
                                try:
                                    # Process options for frame skipping
                                    processing_options = {
                                        "timeout": timeout,
                                        "skip_frames": process_every_n_frames
                                    }
                                    
                                    # Run the video processing with the safe progress callback
                                    if process_mode == "Process and Save":
                                        st.session_state.processed_video, st.session_state.events = process_video(
                                            st.session_state.video_path,
                                            detector,
                                            tracker,
                                            st.session_state.roi_areas,
                                            reasoner if use_reasoner else None,
                                            timeout=timeout,
                                            skip_frames=process_every_n_frames,
                                            progress_callback=safe_progress_update,
                                            show_live=True
                                        )
                                        
                                        # Display final progress
                                        progress_bar.progress(100)
                                        status_text.text(f"Completed processing {st.session_state.progress_total} frames in {st.session_state.progress_time:.1f} seconds")
                                        st.success("Video processed successfully!")
                                        st.rerun()
                                    else:
                                        # Real-time processing mode
                                        st.info("Starting real-time video processing. The video will play with detection and tracking directly.")
                                        
                                        # Play the video directly with real-time processing
                                        video_container = st.empty()
                                        events_container = st.empty()
                                        
                                        # Process video in real-time
                                        import cv2
                                        from app.utils.video_utils import get_frame_thumbnail
                                        
                                        # Open video
                                        cap = cv2.VideoCapture(st.session_state.video_path)
                                        if not cap.isOpened():
                                            st.error(f"Error opening video: {st.session_state.video_path}")
                                        else:
                                            # Get video properties
                                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                            fps = cap.get(cv2.CAP_PROP_FPS) or 25
                                            frame_delay = int(1000 / fps)
                                            
                                            # Track events and objects
                                            live_events = []
                                            live_tracks = {}
                                            frame_idx = 0
                                            
                                            # Process frames in a loop
                                            with st.spinner("Processing video in real-time..."):
                                                while cap.isOpened():
                                                    ret, frame = cap.read()
                                                    if not ret:
                                                        break
                                                    
                                                    # Skip frames if needed
                                                    if process_every_n_frames and frame_idx % 2 == 1:
                                                        frame_idx += 1
                                                        continue
                                                    
                                                    # Detect objects
                                                    detections = detector.detect(frame) if use_detector else []
                                                    
                                                    # Track objects
                                                    tracks_updated = tracker.update(frame, detections) if use_tracker else []
                                                    
                                                    # Process tracked objects (simplified)
                                                    for track_id, bbox, class_id, confidence in tracks_updated:
                                                        x1, y1, x2, y2 = map(int, bbox)
                                                        
                                                        # Draw bounding box with class color
                                                        class_names = {0: "Person", 1: "Cart", 2: "Bag", 3: "Product"}
                                                        class_name = class_names.get(class_id, f"Class-{class_id}")
                                                        label = f"{class_name} #{track_id}"
                                                        
                                                        # Colors for different classes
                                                        colors = {
                                                            0: (0, 255, 0),    # Person: Green
                                                            1: (255, 0, 0),    # Cart: Blue
                                                            2: (0, 0, 255),    # Bag: Red
                                                            3: (255, 255, 0)   # Product: Cyan
                                                        }
                                                        color = colors.get(class_id, (200, 200, 200))
                                                        
                                                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                                        cv2.putText(frame, label, (x1, y1 - 10), 
                                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                    
                                                    # Draw ROI areas
                                                    for roi_name, (rx1, ry1, rx2, ry2) in st.session_state.roi_areas.items():
                                                        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
                                                        cv2.putText(frame, roi_name, (rx1, ry1 - 10), 
                                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                                                    
                                                    # Add frame counter
                                                    cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                                    
                                                    # Display frame
                                                    video_container.image(frame, channels="BGR", caption=f"Live Processing", use_column_width=True)
                                                    
                                                    # Collect events (simplified for demo)
                                                    if frame_idx % 30 == 0:  # Show events every second
                                                        live_events_text = "\n".join([f"Event: {e['type']} - {e['description']}" 
                                                                                     for e in live_events[-5:]])
                                                        events_container.text(live_events_text)
                                                    
                                                    # Update frame index
                                                    frame_idx += 1
                                                    
                                                    # Update progress callback
                                                    if frame_idx % 10 == 0 and safe_progress_update:
                                                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                                        progress = (frame_idx / max(total_frames, 1)) * 100
                                                        elapsed_time = time.time() - start_time
                                                        safe_progress_update(progress, frame_idx, total_frames, elapsed_time)
                                            
                                            # Store events for analytics
                                            st.session_state.events = live_events
                                            st.session_state.processed_video = st.session_state.video_path  # Use original
                                            st.success(f"Completed real-time processing of {frame_idx} frames.")
                                            cap.release()
                                except Exception as e:
                                    import traceback
                                    error_details = traceback.format_exc()
                                    st.error(f"**Processing Error:** {str(e)}")
                                    with st.expander("View detailed error"):
                                        st.code(error_details)
                                    
                                    # Offer fallback options
                                    st.warning("Choose a fallback option to proceed:")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        if st.button("Use Original Video"):
                                            st.session_state.processed_video = st.session_state.video_path
                                            st.session_state.events = []
                                            st.rerun()
                                    
                                    with col2:
                                        if st.button("Try Again with Simplified Processing"):
                                            # This will reload the page with the processing options visible
                                            st.rerun()
                        except Exception as outer_e:
                            st.error(f"Unexpected error: {outer_e}")
                            
                            # Fallback to original video
            except Exception as e:
                import traceback
                st.error(f"Error in video processing: {str(e)}")
                with st.expander("View full error details"):
                    st.code(traceback.format_exc())
        
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

        # Display processed video
        if st.session_state.processed_video:
            try:
                st.video(st.session_state.processed_video)
            except Exception as video_e:
                st.error(f"Error displaying video: {video_e}")
                st.write("Try playing the video manually at:")
                st.code(st.session_state.processed_video)

elif st.session_state.current_tab == "Analytics Dashboard":
    st.title("Analytics Dashboard")
    
    if not st.session_state.events:
        st.warning("No events captured yet. Run surveillance first.")
        if st.button("Go to Surveillance View"):
            st.session_state.current_tab = "Surveillance View"
            st.rerun()
    else:
        # Create a reasoner for analytics (not used in real-time processing)
        try:
            from app.components.symbolic_reasoning.llm_reasoning import SymbolicReasoner
            analytics_reasoner = SymbolicReasoner()
            st.success("🧠 LLM reasoning engine loaded for analytics")
        except Exception as e:
            st.warning(f"⚠️ Could not load LLM reasoning: {e}")
            analytics_reasoner = None
        
        analytics = AnalyticsManager(st.session_state.events, st.session_state.roi_areas, reasoner=analytics_reasoner)
        analytics.display_dashboard()

# Footer
st.markdown("---")
st.markdown("Vision AI © 2023 | All Rights Reserved") 