import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean

# Initialize Mediapipe Pose and OpenCV
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Placeholder coordinates for the reference jab path.
# Replace these with actual coordinates from the uploaded image.
perfect_jab_path = np.array([
    [100, 200], [150, 180], [200, 160], [250, 120], [300, 80], [350, 40]
])

# Define a threshold for similarity
similarity_threshold = 50  # You may need to adjust this based on test results

cap = cv2.VideoCapture(0)
jab_count = 0
x_positions, y_positions = [], []


def calculate_similarity(path1, path2):
    min_len = min(len(path1), len(path2))
    total_distance = sum(euclidean(path1[i], path2[i]) for i in range(min_len))
    return total_distance / min_len


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

        # Check if enough points are collected
        if len(x_positions) > 5:
            # Convert the tracked path to an array
            jab_path = np.column_stack((x_positions, y_positions))

            # Calculate similarity with the reference jab
            similarity = calculate_similarity(jab_path, perfect_jab_path)

            if similarity < similarity_threshold:
                jab_count += 1
                print(f"Jab detected! Total jabs: {jab_count}")
                x_positions, y_positions = [], []  # Reset positions after a detected jab

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
