"""
Monkey Face Detection Mediapipe Program - ZodiacMonkeyCV

A open CV progeram detecting popular monkey reaction faces from Tiktok 

A project to represent my Chinese Zodiac Animal :)

"""
import mediapipe as mp
import cv2
import os
import numpy as np 
from collections import deque

expression_buffer = deque(maxlen=7)
last_expression = "sincere"
cooldown_frames = 0
COOLDOWN = 8

# Initializing mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Pose Initialization
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

#Hands Initalization 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=3,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

# Open webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: webcam not working")
    exit()
cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
cv2.namedWindow("Monkey Image", cv2.WINDOW_NORMAL)


# Image standardization + loading
# Image standardization + loading
MONKEY_SIZE = (640, 480)
def load_monkey(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Missing image: {path}")
        return None
    return cv2.resize(img, MONKEY_SIZE)
monkeys_expressions = {
    "excited": load_monkey("assets/monkey_excited.jpg"),
    "thinking": load_monkey("assets/monkey_thinking.jpg"),
    "smirk": load_monkey("assets/monkey_smirking.jpg"),
    "suprised": load_monkey("assets/monkey_suprised.jpg"),
    "sincere": load_monkey("assets/monkey_sincere.jpg"),
    "proud": load_monkey("assets/monkey_proud.jpg"),
    "peace": load_monkey("assets/monkey_peacesign.jpg"),
}


# Helpers
def vdist(lms, i, j):
    return abs(lms[i].y - lms[j].y)

def count_fingers(hand_landmarks):
    fingers = []

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    fingers.append(thumb_tip.x > thumb_ip.x)

    tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]
    pips = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP,
    ]

    for tip, pip in zip(tips, pips):
        fingers.append(
            hand_landmarks.landmark[tip].y <
            hand_landmarks.landmark[pip].y
        )

    return sum(fingers)

# Expression logic
def monkey_excited(face_landmark_points):
    mouth_open = vdist(face_landmark_points, 13, 14)
    return mouth_open > 0.04

def monkey_thinking(face_landmark_points):
    left_eye = vdist(face_landmark_points, 159, 145)
    right_eye = vdist(face_landmark_points, 386, 374)
    return abs(left_eye - right_eye) > 0.01

def monkey_smirk(face_landmark_points):
    eye_open = (
        vdist(face_landmark_points, 159, 145) +
        vdist(face_landmark_points, 386, 374)
    ) / 2
    return eye_open < 0.017

def monkey_suprised(face_landmark_points):
    eye_open = (
        vdist(face_landmark_points, 159, 145) +
        vdist(face_landmark_points, 386, 374)
    ) / 2
    mouth_open = vdist(face_landmark_points, 13, 14)
    return eye_open > 0.028 and mouth_open > 0.02

def monkey_proud(pose_landmarks):
    if pose_landmarks is None:
        return False
    lms = pose_landmarks.landmark
    thumb = lms[mp_pose.PoseLandmark.RIGHT_THUMB]
    index = lms[mp_pose.PoseLandmark.RIGHT_INDEX]
    wrist = lms[mp_pose.PoseLandmark.RIGHT_WRIST]
    return thumb.y < index.y and thumb.y < wrist.y

def monkey_peace(face_landmark_points):
    mouth_open = vdist(face_landmark_points, 13, 14)
    eye_open = (
        vdist(face_landmark_points, 159, 145) +
        vdist(face_landmark_points, 386, 374)
    ) / 2
    return mouth_open > 0.045 and eye_open < 0.028

def monkey_sincere(face_landmark_points):
    eye_open = (
        vdist(face_landmark_points, 159, 145) +
        vdist(face_landmark_points, 386, 374)
    ) / 2

    mouth_open = vdist(face_landmark_points, 13, 14)

    return (
        eye_open > 0.02 and eye_open < 0.028 and
        mouth_open < 0.03
    )

#Running Face with Hands Logic program
while True:
    ret, frame = cam.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    pose_results = pose.process(rgb)
    hands_results = hands.process(rgb)


    expression = "sincere"
    if results.multi_face_landmarks:
        face_landmark_points = results.multi_face_landmarks[0].landmark
        if monkey_suprised(face_landmark_points):
            expression = "suprised"
        elif monkey_proud(pose_results.pose_landmarks):
            expression = "proud"
        elif monkey_peace(face_landmark_points):
            expression = "peace"
        elif monkey_excited(face_landmark_points):
            expression = "excited"
        elif monkey_smirk(face_landmark_points):
            expression = "smirk"

    #Hands Logic 
    if hands_results.multi_hand_landmarks:
        hand_lms = hands_results.multi_hand_landmarks[0]
        finger_count = count_fingers(hand_lms)
        if finger_count == 2:
            expression = "peace"
        elif finger_count == 1:
            expression = "excited"
        elif finger_count == 5:
            expression = "proud"

    #Smoothing
    expression_buffer.append(expression)
    stable_expression = max(set(expression_buffer), key=expression_buffer.count)
    if cooldown_frames == 0 and stable_expression != last_expression:
        last_expression = stable_expression
        cooldown_frames = COOLDOWN
    else:
        cooldown_frames = max(0, cooldown_frames - 1)
    expression = last_expression

    # Showebcan
    cv2.imshow("Webcam", frame)
    # Show any monkey
    monkey_img = monkeys_expressions.get(expression)
    if monkey_img is None:
        monkey_img = np.zeros((MONKEY_SIZE[1], MONKEY_SIZE[0], 3), dtype=np.uint8)
        cv2.putText(
            monkey_img,
            f"Missing: {expression}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
    cv2.imshow("Monkey Image", monkey_img)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
