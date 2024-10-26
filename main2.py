import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize MediaPipe drawing utility
mp_drawing = mp.solutions.drawing_utils

# Thresholds for detecting a jab, hook, cross, and uppercut
jab_threshold = 0.1
hook_threshold = 0.1
cross_threshold = 0.1
uppercut_threshold = 0.1

# Function to calculate the Euclidean distance between two points


def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Function to display text with timer


def display_text(frame, text, position, timer, duration=30):
    if timer > 0:
        cv2.putText(frame, text, position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return timer - 1
    return timer


# Capture from webcam
cap = cv2.VideoCapture(0)

# Variables to store previous hand positions and counters
previous_left_hand = None
previous_right_hand = None
jab_count = 0
hook_count = 0
cross_count = 0
uppercut_count = 0

# Timers for displaying detection texts
jab_timer = 0
hook_timer = 0
cross_timer = 0
uppercut_timer = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror-like view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get pose estimation
    results = pose.process(rgb_frame)

    # Draw pose annotations on the image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract keypoints for detection
        landmarks = results.pose_landmarks.landmark
        left_hand = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
        right_hand = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]

        # Detect left hand jab
        if previous_left_hand:
            distance_moved_left_hand = calculate_distance(
                left_hand, previous_left_hand)
            if distance_moved_left_hand > jab_threshold and left_hand[0] < left_shoulder[0]:
                jab_count += 1
                jab_timer = 30  # Set timer to keep the text on screen longer

        # Detect right hand jab
        if previous_right_hand:
            distance_moved_right_hand = calculate_distance(
                right_hand, previous_right_hand)
            if distance_moved_right_hand > jab_threshold and right_hand[0] > right_shoulder[0]:
                jab_count += 1
                jab_timer = 30  # Set timer to keep the text on screen longer

        # Detect hooks (curve motion around the head)
        if previous_left_hand:
            if distance_moved_left_hand > hook_threshold and left_hand[0] > left_shoulder[0]:
                hook_count += 1
                hook_timer = 30

        if previous_right_hand:
            if distance_moved_right_hand > hook_threshold and right_hand[0] < right_shoulder[0]:
                hook_count += 1
                hook_timer = 30

        # Detect crosses (straight punch opposite to the jab hand)
        if previous_left_hand and distance_moved_right_hand > cross_threshold and right_hand[0] > right_shoulder[0]:
            cross_count += 1
            cross_timer = 30

        # Detect uppercuts (vertical punch motion)
        if previous_left_hand and left_hand[1] < left_shoulder[1] - uppercut_threshold:
            uppercut_count += 1
            uppercut_timer = 30

        if previous_right_hand and right_hand[1] < right_shoulder[1] - uppercut_threshold:
            uppercut_count += 1
            uppercut_timer = 30

        # Update previous hand positions
        previous_left_hand = left_hand
        previous_right_hand = right_hand

    # Display jab, hook, cross, and uppercut detections
    jab_timer = display_text(frame, f"Jab Detected", (50, 50), jab_timer)
    hook_timer = display_text(frame, f"Hook Detected", (50, 100), hook_timer)
    cross_timer = display_text(
        frame, f"Cross Detected", (50, 150), cross_timer)
    uppercut_timer = display_text(
        frame, f"Uppercut Detected", (50, 200), uppercut_timer)

    # Display the counters for each action
    cv2.putText(frame, f"Jabs: {jab_count}", (400, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Hooks: {hook_count}", (400, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Crosses: {cross_count}", (400, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Uppercuts: {
                uppercut_count}", (400, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Punch Detection', frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
