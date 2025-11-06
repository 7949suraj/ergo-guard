# app.py - Ergo-Guard: Real-time Posture Detection System
# Author: Your Name
# Description: Monitors sitting posture using webcam and MediaPipe Pose detection

import os
import warnings
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque
from datetime import datetime

# Suppress TensorFlow and protobuf warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype\\(\\) is deprecated.*")

# ======================== PAGE CONFIGURATION ========================
st.set_page_config(
    layout="wide",
    page_title="Ergo-Guard - Posture Detection",
    page_icon="💺",
    initial_sidebar_state="expanded"
)

# ======================== MEDIAPIPE SETUP ========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize pose detector
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ======================== CONSTANTS & FEEDBACK ========================
FEEDBACK_MESSAGES = {
    "GOOD": "✅ Excellent posture! Keep it up!",
    "NECK_WARN": "⚠️ Slight forward head tilt detected. Adjust your monitor height.",
    "NECK_BAD": "❌ Severe tech neck! Sit back and raise your screen.",
    "TORSO_WARN": "⚠️ Slight slouching detected. Sit upright and engage core.",
    "TORSO_BAD": "❌ Significant slouching! Adjust your chair or monitor.",
    "PROXIMITY_BAD": "❌ Too close to screen! Move back at least 20 inches.",
    "NO_DETECTION": "👤 No person detected. Please sit in frame."
}

# Default threshold values (can be adjusted in sidebar)
DEFAULT_THRESHOLDS = {
    "neck_good_max": 15.0,      # Below this = good neck posture
    "neck_warn": 30.0,          # Above this = warning
    "neck_bad": 45.0,           # Above this = bad
    "torso_good_max": 12.0,     # Below this = good torso posture
    "torso_warn": 22.0,         # Above this = warning
    "torso_bad": 35.0,          # Above this = bad
    "proximity_bad": 55.0,      # Below this = too close (percentage)
    "smoothing_window": 8,      # Number of frames to average
    "confirmation_frames": 4    # Frames needed to confirm state change
}

# Color codes (BGR format for OpenCV)
COLORS = {
    "GOOD": (0, 255, 0),        # Green
    "WARN": (0, 200, 255),      # Orange
    "BAD": (0, 0, 255),         # Red
    "NEUTRAL": (128, 128, 128)  # Gray
}

# ======================== HELPER FUNCTIONS ========================

def calculate_angle(point_a, point_b, point_c):
    """
    Calculate angle at point_b formed by points a-b-c
    Returns angle in degrees (0-180)
    """
    a = np.array(point_a)
    b = np.array(point_b)
    c = np.array(point_c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def angle_from_vertical(point_top, point_bottom):
    """
    Calculate angle deviation from vertical axis
    Returns angle in degrees (0 = perfectly vertical)
    """
    vector = np.array(point_top) - np.array(point_bottom)
    vertical = np.array([0, -1])  # Upward direction in image coordinates
    
    norm_vector = np.linalg.norm(vector)
    if norm_vector < 1e-6:
        return None
    
    cosine_angle = np.dot(vector, vertical) / norm_vector
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def extract_landmarks(landmarks, frame_width, frame_height):
    """
    Extract key body landmarks from MediaPipe results
    Returns dictionary of landmark coordinates
    """
    def get_coords(landmark_id):
        lm = landmarks.landmark[landmark_id]
        return (int(lm.x * frame_width), int(lm.y * frame_height))
    
    return {
        'nose': get_coords(mp_pose.PoseLandmark.NOSE.value),
        'left_eye': get_coords(mp_pose.PoseLandmark.LEFT_EYE.value),
        'right_eye': get_coords(mp_pose.PoseLandmark.RIGHT_EYE.value),
        'left_ear': get_coords(mp_pose.PoseLandmark.LEFT_EAR.value),
        'right_ear': get_coords(mp_pose.PoseLandmark.RIGHT_EAR.value),
        'left_shoulder': get_coords(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
        'right_shoulder': get_coords(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
        'left_hip': get_coords(mp_pose.PoseLandmark.LEFT_HIP.value),
        'right_hip': get_coords(mp_pose.PoseLandmark.RIGHT_HIP.value)
    }

def calculate_posture_metrics(points):
    """
    Calculate key posture metrics from body landmarks
    Returns dict with neck_angle, torso_angle, and proximity_ratio
    """
    metrics = {
        'neck_angle': None,
        'torso_angle': None,
        'proximity_ratio': None
    }
    
    # Calculate neck angle (ear to shoulder deviation from vertical)
    ear = points['left_ear']
    shoulder = points['left_shoulder']
    neck_angle = angle_from_vertical(ear, shoulder)
    metrics['neck_angle'] = neck_angle
    
    # Calculate torso angle (shoulder to hip deviation from vertical)
    hip = points['left_hip']
    torso_angle = angle_from_vertical(shoulder, hip)
    metrics['torso_angle'] = torso_angle
    
    # Calculate proximity ratio (head-to-shoulder distance vs shoulder width)
    shoulder_width = np.linalg.norm(
        np.array(points['left_shoulder']) - np.array(points['right_shoulder'])
    )
    
    mid_shoulder = (
        (points['left_shoulder'][0] + points['right_shoulder'][0]) / 2,
        (points['left_shoulder'][1] + points['right_shoulder'][1]) / 2
    )
    
    head_to_shoulder_dist = np.linalg.norm(
        np.array(points['nose']) - np.array(mid_shoulder)
    )
    
    if shoulder_width > 0:
        proximity_ratio = (head_to_shoulder_dist / shoulder_width) * 100
        metrics['proximity_ratio'] = proximity_ratio
    
    return metrics

def evaluate_posture(metrics, thresholds):
    """
    Evaluate posture based on metrics and thresholds
    Returns (status_level, primary_issue, color)
    """
    neck = metrics.get('neck_angle')
    torso = metrics.get('torso_angle')
    proximity = metrics.get('proximity_ratio')
    
    bad_issues = []
    warn_issues = []
    
    # Check neck posture
    if neck is not None:
        if neck > thresholds['neck_bad']:
            bad_issues.append('NECK_BAD')
        elif neck > thresholds['neck_warn']:
            warn_issues.append('NECK_WARN')
    
    # Check torso posture
    if torso is not None:
        if torso > thresholds['torso_bad']:
            bad_issues.append('TORSO_BAD')
        elif torso > thresholds['torso_warn']:
            warn_issues.append('TORSO_WARN')
    
    # Check proximity (closer = smaller value)
    if proximity is not None:
        if proximity < thresholds['proximity_bad']:
            bad_issues.append('PROXIMITY_BAD')
    
    # Determine overall status
    if bad_issues:
        return 'BAD', bad_issues[0], COLORS['BAD']
    elif warn_issues:
        return 'WARN', warn_issues[0], COLORS['WARN']
    else:
        # Check if both neck and torso are in good range
        neck_good = neck is not None and neck <= thresholds['neck_good_max']
        torso_good = torso is not None and torso <= thresholds['torso_good_max']
        
        if neck_good and torso_good:
            return 'GOOD', 'GOOD', COLORS['GOOD']
        else:
            return 'GOOD', 'GOOD', COLORS['GOOD']

def smooth_metrics(buffer):
    """
    Apply moving average smoothing to metrics buffer
    Returns smoothed metrics dict
    """
    smoothed = {'neck_angle': None, 'torso_angle': None, 'proximity_ratio': None}
    
    for key in smoothed.keys():
        values = [m[key] for m in buffer if m.get(key) is not None]
        if values:
            smoothed[key] = sum(values) / len(values)
    
    return smoothed

# ======================== SESSION STATE INITIALIZATION ========================
if 'metrics_buffer' not in st.session_state:
    st.session_state.metrics_buffer = deque(maxlen=DEFAULT_THRESHOLDS['smoothing_window'])

if 'state_counters' not in st.session_state:
    st.session_state.state_counters = {'GOOD': 0, 'WARN': 0, 'BAD': 0}

if 'current_status' not in st.session_state:
    st.session_state.current_status = 'NEUTRAL'
    st.session_state.current_issue = 'NO_DETECTION'
    st.session_state.current_color = COLORS['NEUTRAL']

if 'session_stats' not in st.session_state:
    st.session_state.session_stats = {
        'good_frames': 0,
        'warn_frames': 0,
        'bad_frames': 0,
        'start_time': datetime.now()
    }

# ======================== SIDEBAR CONTROLS ========================
with st.sidebar:
    st.title("⚙️ Settings")
    
    st.markdown("### Posture Thresholds")
    st.caption("Adjust sensitivity (smaller angle = stricter)")
    
    # Neck thresholds
    st.markdown("**🦴 Neck Posture**")
    neck_good = st.slider(
        "Good (below °)",
        5.0, 30.0,
        DEFAULT_THRESHOLDS['neck_good_max'],
        1.0,
        help="Maximum angle for good neck posture"
    )
    neck_warn = st.slider(
        "Warning (above °)",
        15.0, 50.0,
        DEFAULT_THRESHOLDS['neck_warn'],
        1.0,
        help="Angle threshold for neck warning"
    )
    neck_bad = st.slider(
        "Bad (above °)",
        25.0, 70.0,
        DEFAULT_THRESHOLDS['neck_bad'],
        1.0,
        help="Angle threshold for bad neck posture"
    )
    
    st.markdown("---")
    
    # Torso thresholds
    st.markdown("**🧍 Torso Posture**")
    torso_good = st.slider(
        "Good (below °)",
        5.0, 25.0,
        DEFAULT_THRESHOLDS['torso_good_max'],
        1.0,
        help="Maximum angle for good torso posture"
    )
    torso_warn = st.slider(
        "Warning (above °)",
        10.0, 40.0,
        DEFAULT_THRESHOLDS['torso_warn'],
        1.0,
        help="Angle threshold for torso warning"
    )
    torso_bad = st.slider(
        "Bad (above °)",
        20.0, 60.0,
        DEFAULT_THRESHOLDS['torso_bad'],
        1.0,
        help="Angle threshold for bad torso posture"
    )
    
    st.markdown("---")
    
    # Proximity threshold
    st.markdown("**📏 Screen Distance**")
    proximity_bad = st.slider(
        "Too close (below %)",
        30.0, 80.0,
        DEFAULT_THRESHOLDS['proximity_bad'],
        1.0,
        help="Proximity threshold (lower = closer to screen)"
    )
    
    st.markdown("---")
    
    # Smoothing settings
    st.markdown("**🎯 Detection Settings**")
    smoothing_frames = st.slider(
        "Smoothing window",
        1, 15,
        DEFAULT_THRESHOLDS['smoothing_window'],
        1,
        help="Number of frames to average (higher = smoother)"
    )
    confirmation_frames = st.slider(
        "Confirmation frames",
        1, 8,
        DEFAULT_THRESHOLDS['confirmation_frames'],
        1,
        help="Frames needed to confirm status change"
    )
    
    st.markdown("---")
    
    # Reset button
    if st.button("🔄 Reset Statistics", use_container_width=True):
        st.session_state.session_stats = {
            'good_frames': 0,
            'warn_frames': 0,
            'bad_frames': 0,
            'start_time': datetime.now()
        }
        st.rerun()

# Update thresholds dict
THRESHOLDS = {
    'neck_good_max': neck_good,
    'neck_warn': neck_warn,
    'neck_bad': neck_bad,
    'torso_good_max': torso_good,
    'torso_warn': torso_warn,
    'torso_bad': torso_bad,
    'proximity_bad': proximity_bad,
    'smoothing_window': smoothing_frames,
    'confirmation_frames': confirmation_frames
}

# Update buffer size if changed
if len(st.session_state.metrics_buffer) != smoothing_frames:
    old_data = list(st.session_state.metrics_buffer)
    st.session_state.metrics_buffer = deque(maxlen=smoothing_frames)
    for item in old_data[-smoothing_frames:]:
        st.session_state.metrics_buffer.append(item)

# ======================== MAIN UI LAYOUT ========================
st.title("💺 Ergo-Guard - Real-Time Posture Detection")
st.markdown("Monitor your sitting posture in real-time to prevent back pain and improve health.")

# Create layout columns
col_video, col_stats = st.columns([2.5, 1])

with col_stats:
    st.markdown("### 📊 Current Status")
    status_container = st.container()
    
    st.markdown("---")
    st.markdown("### 📈 Live Metrics")
    metrics_container = st.container()
    
    st.markdown("---")
    st.markdown("### 📉 Session Statistics")
    stats_container = st.container()

with col_video:
    st.markdown("### 📹 Live Feed")
    video_frame = st.empty()
    
    st.markdown("---")
    control_col1, control_col2 = st.columns(2)
    with control_col1:
        st.info("💡 **Tip:** Position camera at eye level, 18-24 inches away")
    with control_col2:
        st.info("✅ **Good posture:** Ears aligned over shoulders")

# ======================== VIDEO PROCESSING LOOP ========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("❌ Cannot access webcam. Please check:")
    st.markdown("""
    - Camera permissions are enabled
    - No other application is using the camera
    - Camera is properly connected
    """)
else:
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read from webcam")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            # Initialize metrics for this frame
            current_metrics = {
                'neck_angle': None,
                'torso_angle': None,
                'proximity_ratio': None
            }
            
            if results.pose_landmarks:
                # Extract landmarks
                points = extract_landmarks(results.pose_landmarks, w, h)
                
                # Calculate posture metrics
                current_metrics = calculate_posture_metrics(points)
                
                # Draw skeleton on frame
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Draw key points
                for point_name, (x, y) in points.items():
                    if point_name in ['left_ear', 'left_shoulder', 'left_hip']:
                        cv2.circle(frame, (x, y), 6, (255, 255, 0), -1)
                        cv2.circle(frame, (x, y), 8, (0, 0, 0), 2)
            
            # Add metrics to buffer
            st.session_state.metrics_buffer.append(current_metrics)
            
            # Smooth metrics
            smoothed_metrics = smooth_metrics(st.session_state.metrics_buffer)
            
            # Evaluate posture
            if results.pose_landmarks:
                status_level, issue, color = evaluate_posture(smoothed_metrics, THRESHOLDS)
            else:
                status_level, issue, color = 'NEUTRAL', 'NO_DETECTION', COLORS['NEUTRAL']
            
            # Update state counters for confirmation
            for key in st.session_state.state_counters:
                if key == status_level:
                    st.session_state.state_counters[key] += 1
                else:
                    st.session_state.state_counters[key] = 0
            
            # Confirm status change
            if st.session_state.state_counters[status_level] >= THRESHOLDS['confirmation_frames']:
                st.session_state.current_status = status_level
                st.session_state.current_issue = issue
                st.session_state.current_color = color
                
                # Update session statistics
                if status_level == 'GOOD':
                    st.session_state.session_stats['good_frames'] += 1
                elif status_level == 'WARN':
                    st.session_state.session_stats['warn_frames'] += 1
                elif status_level == 'BAD':
                    st.session_state.session_stats['bad_frames'] += 1
            
            # Draw status overlay on video
            overlay_height = 70
            cv2.rectangle(frame, (0, 0), (w, overlay_height), 
                         st.session_state.current_color, -1)
            
            status_text = {
                'GOOD': 'GOOD POSTURE',
                'WARN': 'WARNING',
                'BAD': 'BAD POSTURE',
                'NEUTRAL': 'DETECTING...'
            }.get(st.session_state.current_status, 'UNKNOWN')
            
            cv2.putText(frame, status_text, (20, 45),
                       cv2.FONT_HERSHEY_BOLD, 1.2, (255, 255, 255), 3)
            
            # Draw metrics on frame
            metrics_y = h - 20
            if smoothed_metrics['neck_angle'] is not None:
                neck_text = f"Neck: {smoothed_metrics['neck_angle']:.1f}°"
                cv2.putText(frame, neck_text, (20, metrics_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if smoothed_metrics['torso_angle'] is not None:
                torso_text = f"Torso: {smoothed_metrics['torso_angle']:.1f}°"
                cv2.putText(frame, torso_text, (200, metrics_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if smoothed_metrics['proximity_ratio'] is not None:
                prox_text = f"Distance: {smoothed_metrics['proximity_ratio']:.0f}%"
                cv2.putText(frame, prox_text, (380, metrics_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display video frame
            rgb_output = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame.image(rgb_output, channels='RGB', use_container_width=True)
            
            # Update status display
            with status_container:
                status_emoji = {
                    'GOOD': '✅',
                    'WARN': '⚠️',
                    'BAD': '❌',
                    'NEUTRAL': '👤'
                }.get(st.session_state.current_status, '❓')
                
                st.markdown(f"## {status_emoji} {status_text}")
                st.markdown(f"**{FEEDBACK_MESSAGES.get(st.session_state.current_issue, 'Initializing...')}**")
            
            # Update metrics display
            with metrics_container:
                if smoothed_metrics['neck_angle'] is not None:
                    st.metric("Neck Angle", f"{smoothed_metrics['neck_angle']:.1f}°",
                             delta=f"{'Good' if smoothed_metrics['neck_angle'] <= neck_good else 'Check'}")
                else:
                    st.metric("Neck Angle", "—")
                
                if smoothed_metrics['torso_angle'] is not None:
                    st.metric("Torso Angle", f"{smoothed_metrics['torso_angle']:.1f}°",
                             delta=f"{'Good' if smoothed_metrics['torso_angle'] <= torso_good else 'Check'}")
                else:
                    st.metric("Torso Angle", "—")
                
                if smoothed_metrics['proximity_ratio'] is not None:
                    st.metric("Screen Distance", f"{smoothed_metrics['proximity_ratio']:.0f}%",
                             delta=f"{'Good' if smoothed_metrics['proximity_ratio'] >= proximity_bad else 'Too close'}")
                else:
                    st.metric("Screen Distance", "—")
            
            # Update session statistics
            with stats_container:
                total_frames = sum(st.session_state.session_stats[k] 
                                 for k in ['good_frames', 'warn_frames', 'bad_frames'])
                
                if total_frames > 0:
                    good_pct = (st.session_state.session_stats['good_frames'] / total_frames) * 100
                    warn_pct = (st.session_state.session_stats['warn_frames'] / total_frames) * 100
                    bad_pct = (st.session_state.session_stats['bad_frames'] / total_frames) * 100
                    
                    st.metric("Good Posture", f"{good_pct:.1f}%", 
                             delta=f"{st.session_state.session_stats['good_frames']} frames")
                    st.metric("Warnings", f"{warn_pct:.1f}%",
                             delta=f"{st.session_state.session_stats['warn_frames']} frames")
                    st.metric("Bad Posture", f"{bad_pct:.1f}%",
                             delta=f"{st.session_state.session_stats['bad_frames']} frames")
                    
                    elapsed = datetime.now() - st.session_state.session_stats['start_time']
                    st.caption(f"⏱️ Session: {elapsed.seconds // 60}m {elapsed.seconds % 60}s")
                else:
                    st.info("Statistics will appear once tracking begins")
            
            # Control frame rate
            time.sleep(0.03)
            
    except KeyboardInterrupt:
        st.info("Stopped by user")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
    finally:
        cap.release()
        pose.close()
