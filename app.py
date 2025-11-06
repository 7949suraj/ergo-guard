import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import time

# --- 1. CONFIGURATION AND UTILITY FUNCTIONS ---

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Designer-Focused Feedback Messages (The Innovation - 20% Weightage)
DESIGNER_FEEDBACK = {
    "GOOD": "Posture is excellent! Keep up the good design work.",
    "NECK_WARN": "Minor forward head tilt. Remember to look up from your drawing tablet/screen every 15 mins.",
    "NECK_BAD": "Severe forward head tilt! Risk of 'Tech Neck'. Adjust monitor height to eye level.",
    "SHOULDER_WARN": "Slight shoulder tension detected. Ensure your chair armrests support your elbows at 90 degrees.",
    "SHOULDER_BAD": "Shoulder hunched! Relax and drop your shoulder away from your ear. Check mouse/keyboard height.",
    "PROXIMITY_BAD": "Too close to the screen! Stop leaning in for detail work. Increase monitor distance to reduce eye strain."
}

# Calibration/Thresholds (Tune these based on testing)
# These are ABSOLUTE thresholds for 'Bad' state (angles are in degrees, ratio is normalized)
NECK_ABS_BAD = 160     # Angle between Ear-Shoulder-Vertical line (lower is worse)
SHOULDER_ABS_BAD = 150 # Angle between Shoulder-Hip-Vertical line (lower is worse)
PROXIMITY_BAD_THRESHOLD = 50 # Normalized ratio (lower means leaning in too much)

# Function to calculate angle between three points (A, B, C where B is the vertex)
def calculate_angle(a, b, c):
    a = np.array(a) # First point
    b = np.array(b) # Mid point/Vertex
    c = np.array(c) # End point
    
    # Calculate vectors BA and BC
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/math.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

# --- 2. CORE POSTURE PROCESSING FUNCTION ---

def process_posture(frame):
    # Convert the BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    current_status = "GOOD"
    feedback_key = "GOOD"
    color = (0, 255, 0)  # Green

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Extract Key Body Landmarks (for simplicity, using Left side)
        left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1],
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0])
        left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0])
        left_ear = (landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0])
        nose = (landmarks[mp_pose.PoseLandmark.NOSE.value].x * frame.shape[1],
                landmarks[mp_pose.PoseLandmark.NOSE.value].y * frame.shape[0])
        
        # For shoulder width/midpoint, use both shoulders
        right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0])

        # --- Metric 1: Neck Angle (Ear-Shoulder-Vertical) ---
        # Vertical point is directly above the shoulder
        vertical_point = (int(left_shoulder[0]), 0) 
        neck_angle = calculate_angle(left_ear, left_shoulder, vertical_point)

        # --- Metric 2: Shoulder Angle (Shoulder-Hip-Vertical) ---
        # Vertical point is directly below the hip
        vertical_hip = (int(left_hip[0]), frame.shape[0])
        shoulder_angle = calculate_angle(left_shoulder, left_hip, vertical_hip)

        # --- Metric 3: Monitor Proximity Ratio (Nose-Midpoint normalized by Shoulder Width) ---
        midpoint = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2) 
        shoulder_width = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
        nose_to_shoulder_mid_dist = np.linalg.norm(np.array(nose) - np.array(midpoint))
        proximity_ratio = (nose_to_shoulder_mid_dist / shoulder_width) * 100 
        
        # --- Three-Tier CLASSIFICATION and Feedback Logic ---
        
        # Check for BAD Posture (Red - needs immediate correction)
        if neck_angle < NECK_ABS_BAD:
            current_status = "BAD POSTURE"
            color = (0, 0, 255) # Red
            feedback_key = "NECK_BAD"
        elif shoulder_angle > 110: # Example: Shoulder too far back or too forward (adjust this)
            current_status = "BAD POSTURE"
            color = (0, 0, 255)
            feedback_key = "SHOULDER_BAD"
        elif proximity_ratio < PROXIMITY_BAD_THRESHOLD:
            current_status = "BAD POSTURE"
            color = (0, 0, 255)
            feedback_key = "PROXIMITY_BAD"
            
        # Check for WARNING Posture (Yellow - needs mindfulness)
        elif neck_angle < NECK_ABS_BAD + 10:
            current_status = "WARNING"
            color = (0, 255, 255) # Yellow
            feedback_key = "NECK_WARN"
        elif shoulder_angle < 150: # Example: slightly slumped (adjust this)
            current_status = "WARNING"
            color = (0, 255, 255)
            feedback_key = "SHOULDER_WARN"
            
        # Draw Skeleton and Angles on the Frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Display current angles/ratio (for debugging/visualization)
        cv2.putText(frame, f"Neck: {neck_angle:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Shoulder: {shoulder_angle:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Proximity: {proximity_ratio:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        
        # Store status and feedback in session state for Streamlit columns
        st.session_state.posture_status = current_status
        st.session_state.posture_advice = DESIGNER_FEEDBACK[feedback_key]
        st.session_state.posture_color = color # Store the color for the UI
        
    else:
        # No person detected
        st.session_state.posture_status = "NO USER DETECTED"
        st.session_state.posture_advice = "Please position yourself fully in front of the webcam."
        st.session_state.posture_color = (128, 128, 128) # Grey
        
    return frame

# --- 3. STREAMLIT APPLICATION LAYOUT ---

def main():
    st.set_page_config(layout="wide")
    st.title("💡 Ergo-Guard: AI Posture Correction for Designers")

    # Initialize session state variables if they don't exist
    if 'posture_status' not in st.session_state:
        st.session_state.posture_status = "Starting..."
        st.session_state.posture_advice = "Initializing camera and detection."
        st.session_state.posture_color = (128, 128, 128)
        
    # Layout columns
    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Real-Time Ergonomic Feedback")
        
        # Display Status with Color
        st.markdown(
            f"**STATUS:** <span style='font-size: 24px; color: rgb({st.session_state.posture_color[2]}, {st.session_state.posture_color[1]}, {st.session_state.posture_color[0]})'>**{st.session_state.posture_status}**</span>", 
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.subheader("Designer's Advice")
        st.info(st.session_state.posture_advice)
        
        # Display Key Metrics
        st.markdown("---")
        st.markdown(f"**Neck Angle (Criteria):** < {NECK_ABS_BAD+10:.1f}°")
        st.markdown(f"**Proximity Ratio (Bad):** < {PROXIMITY_BAD_THRESHOLD:.1f}")


    with col1:
        st.subheader("Live Webcam Feed")
        frame_placeholder = st.empty()
        
    # Start Webcam Capture
    cap = cv2.VideoCapture(0)

    # Loop for processing video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not access webcam.")
            break
            
        # Flip the frame for a more natural mirror effect
        frame = cv2.flip(frame, 1)

        # Process the frame for posture
        processed_frame = process_posture(frame)
        
        # Convert BGR (OpenCV) to RGB (Streamlit)
        processed_rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Update the video placeholder
        frame_placeholder.image(processed_rgb_frame, channels="RGB")
        
        # Add a short delay to manage CPU usage
        time.sleep(0.01)

    cap.release()

if __name__ == "__main__":
    main()
    