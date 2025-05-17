import cv2
import numpy as np
import os
import tempfile
from datetime import datetime, timedelta
import time
import streamlit as st  # Comment this out if not using Streamlit

def get_frame_thumbnail(frame, max_width=300):
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

def process_video(video_path, detector, tracker, roi_areas, reasoner, timeout=300, skip_frames=False, progress_callback=None, show_live=True):
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

    output_path = os.path.join(tempfile.gettempdir(), f"processed_{int(time.time())}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
                                'to_roi': roi_name
                            }

                            if class_id == 0 and 'shelf' in roi_name.lower():
                                event_data['type'] = 'Person at Shelf'

                            if class_id == 0 and track['last_roi'] and 'shelf' in track['last_roi'].lower() and 'exit' in roi_name.lower():
                                event_data['type'] = 'Potential Theft'
                                try:
                                    event_data['reasoning'] = reasoner.analyze_event(event_data, track)
                                except Exception as e:
                                    event_data['reasoning'] = f"Error: {e}"

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
                                                'person_id': pid
                                            }
                                            frame_events.append(event_data)
                                            pdata['items_picked'].append(track_id)
                                            break

                            frame_events.append(event_data)

                    track['last_roi'] = roi_name
                    break
                elif roi_name in track['roi_visits'] and track['roi_visits'][roi_name]['exit_frame'] is None:
                    track['roi_visits'][roi_name]['exit_frame'] = frame_idx

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for roi_name, (rx1, ry1, rx2, ry2) in roi_areas.items():
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)
            cv2.putText(frame, roi_name, (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.putText(frame, f"Frame: {frame_idx} | Time: {timestamp.strftime('%H:%M:%S')}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(frame)

        if show_live and image_placeholder:
            try:
                image_placeholder.image(frame, channels="BGR", caption=f"Frame {frame_idx}", use_column_width=True)
            except Exception as e:
                print(f"Streamlit image display error: {e}")

        events.extend(frame_events)
        frame_idx += 1

        if skip_frames:
            cap.grab()

        if frame_idx % 10 == 0 and progress_callback:
            try:
                progress = (frame_idx / max(total_frames, 1)) * 100
                progress_callback(progress, frame_idx, total_frames, time.time() - start_time)
            except Exception as e:
                print(f"Progress callback error: {e}")

    print(f"Processed {frame_idx} frames in {time.time() - start_time:.1f} seconds")

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
                            'dwell_time': dwell
                        }
                        try:
                            event['reasoning'] = reasoner.analyze_dwell_time(event, data)
                        except Exception as e:
                            event['reasoning'] = f"Error: {e}"
                        events.append(event)

    cap.release()
    out.release()

    print(f"Video processing completed. Output saved to: {output_path}")
    return output_path, events
