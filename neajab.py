import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean

# Initialize Mediapipe Pose and OpenCV
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define the core area coordinates (adjust based on the darker area in your plot)
core_x_min, core_x_max = 300, 500
core_y_min, core_y_max = 50, 150

# Define a threshold for counting a jab cycle
# The distance the hand should extend from the core area to count as a jab
jab_extension_threshold = 100

cap = cv2.VideoCapture(0)
jab_count = 0
x_positions, y_positions = [], []
in_core_area = False  # Flag to detect entry into the core area

print("Start jabbing...")

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
        # Get the coordinates of the right hand's wrist (landmark #16)
        wrist = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        height, width, _ = frame.shape
        wrist_x = int(wrist.x * width)
        wrist_y = int(wrist.y * height)

        # Collect wrist positions
        x_positions.append(wrist_x)
        y_positions.append(height - wrist_y)  # Invert y for graph orientation

        # Draw circle on the wrist
        cv2.circle(frame, (wrist_x, wrist_y), 8, (0, 255, 0), -1)

        # Check if wrist is in the core area
        if core_x_min <= wrist_x <= core_x_max and core_y_min <= (height - wrist_y) <= core_y_max:
            if not in_core_area:  # First entry into the core area
                in_core_area = True
                path_start_x, path_start_y = wrist_x, wrist_y  # Save the start of the jab
        else:
            # Check if the hand extended far enough to count as a jab
            if in_core_area and euclidean([path_start_x, path_start_y], [wrist_x, wrist_y]) > jab_extension_threshold:
                jab_count += 1
                print(f"Jab detected! Total jabs: {jab_count}")
                in_core_area = False  # Reset core area flag

            # Reset path after counting a jab
            x_positions, y_positions = [], []

        # Plot the current path
        plt.clf()
        plt.plot(x_positions, y_positions, color='blue',
                 marker='o', markersize=3, linestyle='-')
        plt.title(f"Jab Motion Path - Total Jabs: {jab_count}")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
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
# this is perfect jab
