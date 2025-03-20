# Computer Vision - OpenCV

import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Model - MediaPipe for Hand Detection (best for Hand Landmark than other models)
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(static_image_mode=False,
                               max_num_hands=2,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)
drawing_tool = mp.solutions.drawing_utils

# Colour for each finger
FINGER_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

# Finger names and landmark labels
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
LANDMARK_LABELS = [
    "Wrist", "Thumb CMC", "Thumb MCP", "Thumb IP", "Thumb TIP",
    "Index MCP", "Index PIP", "Index DIP", "Index TIP",
    "Middle MCP", "Middle PIP", "Middle DIP", "Middle TIP",
    "Ring MCP", "Ring PIP", "Ring DIP", "Ring TIP",
    "Pinky MCP", "Pinky PIP", "Pinky DIP", "Pinky TIP"
]

# Smoothing buffer for moving average (buffer size = 5)
landmark_smoothing = [{i: deque(maxlen=5) for i in range(21)} for _ in range(2)]

# Gesture thresholds
HAND_OPEN_THRESHOLD = 0.7
FINGER_EXTENDED_THRESHOLD = 0.3

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Utility Functions
def calculate_distance(coord1, coord2):
    """Euclidean distance"""
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def is_hand_open(landmarks):
    thumb_tip = landmarks[4]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]

    hand_width = calculate_distance(thumb_tip, pinky_tip)
    hand_length = calculate_distance(wrist, pinky_tip)

    hand_openness = hand_width / hand_length
    return hand_openness > HAND_OPEN_THRESHOLD

def is_finger_extended(landmarks, tip_index, base_index):
    tip = landmarks[tip_index]
    base = landmarks[base_index]

    # Distance between fingertip and base
    tip_base_distance = calculate_distance(tip, base)

    # Normalize distance by hand length
    hand_length = calculate_distance(landmarks[0], landmarks[9])
    normalized_distance = tip_base_distance / hand_length

    return normalized_distance > FINGER_EXTENDED_THRESHOLD

# Main Loop
while webcam.isOpened():
    ret, video_frame = webcam.read()
    if not ret:
        break

    video_frame = cv2.flip(video_frame, 1)

    # Convert frame to RGB
    video_frame_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)

    results = hand_detector.process(video_frame_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_idx, (hand_landmarks, hand_type) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            hand_label = hand_type.classification[0].label  

            processed_landmarks = []
            for index, landmark in enumerate(hand_landmarks.landmark):
                frame_height, frame_width, _ = video_frame.shape
                x_coord, y_coord = int(landmark.x * frame_width), int(landmark.y * frame_height)

                # Smooth landmark positions
                landmark_smoothing[hand_idx][index].append((x_coord, y_coord))
                avg_x = int(np.mean([coord[0] for coord in landmark_smoothing[hand_idx][index]]))
                avg_y = int(np.mean([coord[1] for coord in landmark_smoothing[hand_idx][index]]))

                processed_landmarks.append((avg_x, avg_y))

                # Draw a circle on each finger landmark
                color = FINGER_COLORS[index % 5]
                cv2.circle(video_frame, (avg_x, avg_y), 6, color, -1)

                # Display the landmark label
                cv2.putText(video_frame, f"{index}: {LANDMARK_LABELS[index]}", (avg_x + 10, avg_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            drawing_tool.draw_landmarks(video_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect whether the hand is open or closed
            if is_hand_open(processed_landmarks):
                cv2.putText(video_frame, f"{hand_label} Hand: Open",
                            (10, 30 + hand_idx * 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
            else:
                cv2.putText(video_frame, f"{hand_label} Hand: Closed",
                            (10, 30 + hand_idx * 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)

                fingertip_indices = [4, 8, 12, 16, 20]
                finger_base_indices = [2, 5, 9, 13, 17]

                for i in range(5):
                    extended = is_finger_extended(processed_landmarks, fingertip_indices[i], finger_base_indices[i])
                    if extended:
                        tip_idx = fingertip_indices[i]
                        base_idx = finger_base_indices[i]

                        # Highlight extended fingers
                        cv2.circle(video_frame, processed_landmarks[tip_idx], 8, FINGER_COLORS[i], -1)
                        cv2.circle(video_frame, processed_landmarks[base_idx], 8, FINGER_COLORS[i], -1)

                        cv2.line(video_frame, processed_landmarks[base_idx], processed_landmarks[tip_idx], FINGER_COLORS[i], 2)

                        # Display finger name near the fingertip
                        cv2.putText(video_frame, FINGER_NAMES[i], (processed_landmarks[tip_idx][0] + 10, processed_landmarks[tip_idx][1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, FINGER_COLORS[i], 1)
    else:
        cv2.putText(video_frame, "No Hand Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    cv2.imshow("Hand Tracking with Finger Detection", video_frame)

    # Exit 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
webcam.release()
cv2.destroyAllWindows()
