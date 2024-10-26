import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

# Initialize Mediapipe Pose and OpenCV
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# Lists to store hand positions
x_positions = []
y_positions = []

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

        # Convert normalized coordinates to pixel values
        height, width, _ = frame.shape
        wrist_x = int(wrist.x * width)
        wrist_y = int(wrist.y * height)

        # Store positions in lists
        x_positions.append(wrist_x)
        # Reverse y to match graph orientation
        y_positions.append(height - wrist_y)

        # Draw circle on the wrist
        cv2.circle(frame, (wrist_x, wrist_y), 8, (0, 255, 0), -1)

    # Display the video feed
    cv2.imshow('Jab Motion Capture', frame)

    # Press 'q' to stop capturing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Plot the hand movement path
plt.figure(figsize=(10, 6))
plt.plot(x_positions, y_positions, color='blue',
         marker='o', markersize=3, linestyle='-')
plt.title("Jab Motion Path")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.gca().invert_yaxis()  # Optional: Invert y-axis for visual clarity
plt.show()
