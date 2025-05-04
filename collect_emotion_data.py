"""
Emotion Data Collection Script

This script is used to collect facial and hand landmark data for emotion recognition.
It's based on the original data_collection.py but configured specifically for emotion labeling.

Instructions:
1. Run this script
2. Enter the name of the emotion you want to capture (e.g., 'happy', 'sad', 'angry')
3. Position your face in the camera and express that emotion
4. The script will capture 100 frames and save the data as '[emotion].npy'
5. Repeat for different emotions

Press ESC to stop recording early
"""

import mediapipe as mp
import numpy as np
import cv2
import os

# Set the path to save data
SAVE_PATH = r"C:\Users\srini\Downloads"

# Initialize video capture
cap = cv2.VideoCapture(0)

# Get emotion name from user
print("=== Emotion Data Collection ===")
print("Available emotions: happy, sad, angry, neutral, surprised")
name = input("Enter the emotion to capture: ")

# Initialize MediaPipe components
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Data collection list
X = []
data_size = 0

print(f"Recording emotion: {name}")
print("Position your face in the camera and express the emotion")
print("Press ESC to stop recording early")

while True:
    lst = []

    # Read frame from camera
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)  # Mirror image

    # Process the frame with MediaPipe
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # Extract facial landmarks
    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        # Extract left hand landmarks
        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        # Extract right hand landmarks
        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        # Save landmarks to data list
        X.append(lst)
        data_size = data_size + 1

    # Draw landmarks on frame
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    # Display frame count
    cv2.putText(frm, f"Frames: {data_size}/100", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frm, f"Recording: {name}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frm, "Press ESC to stop", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display frame
    cv2.imshow("Emotion Data Collection", frm)

    # Check for exit conditions
    key = cv2.waitKey(1)
    if key == 27 or data_size >= 100:  # ESC key or enough data collected
        break

# Save data
save_file = os.path.join(SAVE_PATH, f"{name}.npy")
np.save(save_file, np.array(X))
print(f"Data saved to {save_file}")
print(f"Data shape: {np.array(X).shape}")

# Clean up
cv2.destroyAllWindows()
cap.release()
