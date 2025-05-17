import cv2
import numpy as np
import os
import tempfile
from datetime import datetime, timedelta
import time
import streamlit as st  # Comment this out if not using Streamlit
import json

def get_frame_thumbnail(frame, max_width=600):
    h, w = frame.shape[:2]
    scale = min(max_width / w, 1.0)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))

def get_video_frame(video_path, frame_number=0, is_camera=False):
    cap = cv2.VideoCapture(video_path if not is_camera else int(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {video_path}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Cannot read frame {frame_number} from video.")
    
    return frame

def process_video(video_path, detector, tracker, roi_areas, reasoner=None, timeout=300, skip_frames=False, progress_callback=None, show_live=True):
    """
    Process video with detection, tracking, and ROI analysis
    
    Args:
        video_path: Path to video file or camera URL
        detector: YOLODetector instance
        tracker: DeepSORTTracker instance
        roi_areas: Dictionary of ROI areas
        reasoner: SymbolicReasoner instance (optional, can be None for faster processing)
        timeout: Maximum processing time in seconds (default: 300s)
        skip_frames: If True, process every other frame for faster processing
        progress_callback: Optional callback function to report progress updates
        show_live: Show live preview in Streamlit
        
    Returns:
        tuple: (processed_video_path, events)
    """
    start_time = time.time()
    print(f"Starting video processing: {video_path}")

    cap = cv2.VideoCapture(video_path if not str(video_path).isdigit() else int(video_path))
    if not cap.isOpened():
        raise Exception(f"Error opening video source: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")

    # Create output video with H.264 codec for better compatibility
    output_path = os.path.join(tempfile.gettempdir(), f"processed_{int(time.time())}.mp4")
    
    # Try different codec options based on platform support
    try:
        # Try H.264 codec first for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise Exception("Failed to open VideoWriter with avc1 codec")
    except Exception as e:
        print(f"Failed to use H.264 codec: {e}, falling back to MJPG")
        try:
            # Try MJPG as fallback
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            temp_output = os.path.join(tempfile.gettempdir(), f"processed_{int(time.time())}.avi")
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            if not out.isOpened():
                raise Exception("Failed to open VideoWriter with MJPG codec")
            output_path = temp_output
        except Exception as e2:
            print(f"Failed to use MJPG codec: {e2}, falling back to mp4v")
            # Last resort: mp4v
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = os.path.join(tempfile.gettempdir(), f"processed_{int(time.time())}.mp4")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    events = []
    tracks = {}
    frame_idx = 0
    base_time = datetime.now()

    # Initialize placeholders for live display
    image_placeholder = None
    log_placeholder = None
    
    if show_live and 'streamlit' in globals():
        image_placeholder = st.empty()
        log_placeholder = st.empty()
        
        # Add explanation text
        st.info("⚠️ Processing video... This may take a while. If the video doesn't display when complete, check the 'Skip Processing' option for better performance.")

    # Flag for delayed reasoning to avoid slowing down real-time processing
    events_needing_reasoning = []

    while cap.isOpened():
        if time.time() - start_time > timeout:
            print(f"Timeout: {timeout} seconds reached.")
            break

        ret, frame = cap.read()
        if not ret:
            print(f"End of video at frame {frame_idx}")
            break

        timestamp = base_time + timedelta(seconds=frame_idx / fps)

        try:
            detections = detector.detect(frame)
        except Exception as e:
            print(f"Detection error @ frame {frame_idx}: {e}")
            detections = []

        try:
            tracks_updated = tracker.update(frame, detections)
        except Exception as e:
            print(f"Tracking error @ frame {frame_idx}: {e}")
            tracks_updated = []

        frame_events = []

        for track_id, bbox, class_id, confidence in tracks_updated:
            x1, y1, x2, y2 = map(int, bbox)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            track = tracks.setdefault(track_id, {
                'frames': [],
                'locations': [],
                'class_id': class_id,
                'roi_visits': {},
                'last_roi': None,
                'items_picked': []
            })

            track['frames'].append(frame_idx)
            track['locations'].append((cx, cy))

            for roi_name, (rx1, ry1, rx2, ry2) in roi_areas.items():
                in_roi = rx1 <= cx <= rx2 and ry1 <= cy <= ry2

                if in_roi:
                    if roi_name not in track['roi_visits']:
                        track['roi_visits'][roi_name] = {'enter_frame': frame_idx, 'exit_frame': None}

                        if track['last_roi'] != roi_name:
                            event_data = {
                                'type': 'ROI Transition',
                                'timestamp': timestamp.strftime('%H:%M:%S'),
                                'description': f"Object ID {track_id} entered {roi_name}",
                                'thumbnail': get_frame_thumbnail(frame),
                                'frame_idx': frame_idx,
                                'track_id': track_id,
                                'from_roi': track['last_roi'],
                                'to_roi': roi_name,
                                'needs_reasoning': False
                            }

                            if class_id == 0 and 'shelf' in roi_name.lower():
                                event_data['type'] = 'Person at Shelf'
                                event_data['description'] = f"Person ID {track_id} is browsing at {roi_name}"

                            if class_id == 0 and track['last_roi'] and 'shelf' in track['last_roi'].lower() and 'exit' in roi_name.lower():
                                event_data['type'] = 'Potential Theft'
                                event_data['description'] = f"Person ID {track_id} moved from {track['last_roi']} directly to {roi_name}"
                                # Mark for later reasoning rather than doing it now
                                event_data['needs_reasoning'] = True
                                events_needing_reasoning.append((len(events) + len(frame_events), event_data, track))

                            if class_id > 0 and 'shelf' in roi_name.lower():
                                for pid, pdata in tracks.items():
                                    if pdata['class_id'] == 0 and frame_idx in pdata['frames']:
                                        pidx = pdata['frames'].index(frame_idx)
                                        px, py = pdata['locations'][pidx]
                                        dist = np.hypot(cx - px, cy - py)
                                        if dist < 100:
                                            event_data = {
                                                'type': 'Item Pickup',
                                                'timestamp': timestamp.strftime('%H:%M:%S'),
                                                'description': f"Item ID {track_id} picked up by Person ID {pid}",
                                                'thumbnail': get_frame_thumbnail(frame),
                                                'frame_idx': frame_idx,
                                                'item_id': track_id,
                                                'person_id': pid,
                                                'needs_reasoning': False
                                            }
                                            frame_events.append(event_data)
                                            pdata['items_picked'].append(track_id)
                                            break

                            frame_events.append(event_data)

                    track['last_roi'] = roi_name
                    break
                elif roi_name in track['roi_visits'] and track['roi_visits'][roi_name]['exit_frame'] is None:
                    track['roi_visits'][roi_name]['exit_frame'] = frame_idx

            # Draw class name, track ID and confidence
            class_names = {0: "Person", 1: "Cart", 2: "Bag", 3: "Product"}
            class_name = class_names.get(class_id, f"Class-{class_id}")
            label = f"{class_name} #{track_id} ({confidence:.2f})"
            
            # Use different colors for different object classes
            colors = {
                0: (0, 255, 0),    # Person: Green
                1: (255, 0, 0),    # Cart: Blue
                2: (0, 0, 255),    # Bag: Red
                3: (255, 255, 0)   # Product: Cyan
            }
            color = colors.get(class_id, (200, 200, 200))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw ROI areas with labels
        for roi_name, (rx1, ry1, rx2, ry2) in roi_areas.items():
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
            cv2.putText(frame, roi_name, (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display frame number and timestamp
        cv2.putText(frame, f"Frame: {frame_idx} | Time: {timestamp.strftime('%H:%M:%S')}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Show event count
        cv2.putText(frame, f"Events: {len(events) + len(frame_events)}", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)

        if show_live and image_placeholder:
            try:
                # Only update the display every few frames to reduce overhead
                if frame_idx % 5 == 0:
                    image_placeholder.image(frame, channels="BGR", caption=f"Frame {frame_idx}", use_column_width=True)
                    if frame_events and log_placeholder:
                        event_text = "\n".join([f"Event: {e['type']} - {e['description']}" for e in frame_events])
                        log_placeholder.text(event_text)
            except Exception as e:
                print(f"Streamlit image display error: {e}")

        events.extend(frame_events)
        frame_idx += 1

        if skip_frames:
            cap.grab()  # Skip one frame

        if frame_idx % 10 == 0 and progress_callback:
            try:
                progress = (frame_idx / max(total_frames, 1)) * 100
                progress_callback(progress, frame_idx, total_frames, time.time() - start_time)
            except Exception as e:
                print(f"Progress callback error: {e}")

    print(f"Processed {frame_idx} frames in {time.time() - start_time:.1f} seconds")

    # Add dwell time events (without reasoning for speed)
    for track_id, data in tracks.items():
        if data['class_id'] == 0:
            for roi_name, visit in data['roi_visits'].items():
                if visit['exit_frame']:
                    dwell = (visit['exit_frame'] - visit['enter_frame']) / fps
                    if dwell > 30 and any(x in roi_name.lower() for x in ['jewelry', 'electronics']):
                        timestamp = base_time + timedelta(seconds=visit['enter_frame'] / fps)
                        event = {
                            'type': 'Long Dwell Time',
                            'timestamp': timestamp.strftime('%H:%M:%S'),
                            'description': f"Person ID {track_id} spent {dwell:.1f}s in {roi_name}",
                            'thumbnail': None,
                            'frame_idx': visit['enter_frame'],
                            'track_id': track_id,
                            'roi_name': roi_name,
                            'dwell_time': dwell,
                            'needs_reasoning': True
                        }
                        events.append(event)
                        events_needing_reasoning.append((len(events) - 1, event, data))

    cap.release()
    out.release()
    
    print(f"Video processing completed. Output saved to: {output_path}")
    
    # Store events that need reasoning for later analysis in the dashboard
    if events_needing_reasoning:
        reasoning_data_path = os.path.join(tempfile.gettempdir(), "events_needing_reasoning.json")
        try:
            with open(reasoning_data_path, 'w') as f:
                # We can't directly serialize the data, so just store the indices
                json.dump([idx for idx, _, _ in events_needing_reasoning], f)
        except Exception as e:
            print(f"Error saving reasoning data: {e}")
    
    return output_path, events
