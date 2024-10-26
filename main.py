import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize MediaPipe drawing utility
mp_drawing = mp.solutions.drawing_utils

# Threshold for detecting a jab (movement of hand relative to shoulder)
jab_threshold = 0.1

# Function to calculate the Euclidean distance between two points


def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# Capture from webcam
cap = cv2.VideoCapture(0)

previous_left_hand = None
previous_right_hand = None

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

        # Extract keypoints for jab detection (left hand, right hand, and shoulders)
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
            distance_left_hand_shoulder = calculate_distance(
                left_hand, left_shoulder)
            if distance_moved_left_hand > jab_threshold and left_hand[0] < left_shoulder[0]:
                cv2.putText(frame, "Left Jab Detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Detect right hand jab
        if previous_right_hand:
            distance_moved_right_hand = calculate_distance(
                right_hand, previous_right_hand)
            distance_right_hand_shoulder = calculate_distance(
                right_hand, right_shoulder)
            if distance_moved_right_hand > jab_threshold and right_hand[0] > right_shoulder[0]:
                cv2.putText(frame, "Right Jab Detected", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Update previous hand positions
        previous_left_hand = left_hand
        previous_right_hand = right_hand

    # Display the frame
    cv2.imshow('Jab Detection', frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
