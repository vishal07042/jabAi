import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean

# Initialize Mediapipe Pose and OpenCV
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define the core area coordinates (adjust based on darker area in your plot)
core_x_min, core_x_max = 300, 500
core_y_min, core_y_max = 50, 150

# Define a threshold for counting a jab cycle
# The distance the hand should extend from the core area to count as a jab
jab_extension_threshold = 100

cap = cv2.VideoCapture(0)
right_jab_count = 0
left_jab_count = 0
right_x_positions, right_y_positions = [], []
left_x_positions, left_y_positions = [], []
right_in_core_area = False  # Flag for detecting right hand entry into core area
left_in_core_area = False  # Flag for detecting left hand entry into core area

print("Start jabbing with both hands...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        height, width, _ = frame.shape

        # Get right wrist coordinates
        right_wrist = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        right_wrist_x = int(right_wrist.x * width)
        right_wrist_y = int(right_wrist.y * height)

        # Get left wrist coordinates
        left_wrist = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        left_wrist_x = int(left_wrist.x * width)
        left_wrist_y = int(left_wrist.y * height)

        # Collect wrist positions
        right_x_positions.append(right_wrist_x)
        # Invert y for graph orientation
        right_y_positions.append(height - right_wrist_y)
        left_x_positions.append(left_wrist_x)
        left_y_positions.append(height - left_wrist_y)

        # Draw circles on the wrists
        cv2.circle(frame, (right_wrist_x, right_wrist_y),
                   8, (0, 255, 0), -1)  # Right hand (Green)
        cv2.circle(frame, (left_wrist_x, left_wrist_y),
                   8, (255, 0, 0), -1)  # Left hand (Blue)

        # Right Hand Jab Detection
        if core_x_min <= right_wrist_x <= core_x_max and core_y_min <= (height - right_wrist_y) <= core_y_max:
            if not right_in_core_area:  # First entry into the core area
                right_in_core_area = True
                # Save the start of the jab
                right_start_x, right_start_y = right_wrist_x, right_wrist_y
        else:
            # Check if the hand extended far enough to count as a jab
            if right_in_core_area and euclidean([right_start_x, right_start_y], [right_wrist_x, right_wrist_y]) > jab_extension_threshold:
                right_jab_count += 1
                print(f"Right Jab detected! Total Right Jabs: {
                      right_jab_count}")
                right_in_core_area = False  # Reset core area flag

            # Reset path after counting a jab
            right_x_positions, right_y_positions = [], []

        # Left Hand Jab Detection
        if core_x_min <= left_wrist_x <= core_x_max and core_y_min <= (height - left_wrist_y) <= core_y_max:
            if not left_in_core_area:  # First entry into the core area
                left_in_core_area = True
                left_start_x, left_start_y = left_wrist_x, left_wrist_y  # Save the start of the jab
        else:
            # Check if the hand extended far enough to count as a jab
            if left_in_core_area and euclidean([left_start_x, left_start_y], [left_wrist_x, left_wrist_y]) > jab_extension_threshold:
                left_jab_count += 1
                print(f"Left Jab detected! Total Left Jabs: {left_jab_count}")
                left_in_core_area = False  # Reset core area flag

            # Reset path after counting a jab
            left_x_positions, left_y_positions = [], []

        # Plot the current paths for both hands
        plt.clf()
        plt.plot(right_x_positions, right_y_positions, color='green', marker='o',
                 markersize=3, linestyle='-', label=f'Right Hand (Jabs: {right_jab_count})')
        plt.plot(left_x_positions, left_y_positions, color='blue', marker='o',
                 markersize=3, linestyle='-', label=f'Left Hand (Jabs: {left_jab_count})')
        plt.title(
            f"Jab Motion Path - Right Jabs: {right_jab_count}, Left Jabs: {left_jab_count}")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend(loc='upper right')
        plt.gca().invert_yaxis()
        plt.pause(0.01)  # Allows the plot to update in real-time

    # Display the video feed
    cv2.imshow('Jab Motion Capture', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.show()
