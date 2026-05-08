# evans_faceid_pi.py
# Raspberry Pi 3 optimized version of EvansFaceID script
# ---------------------------------------------------
# Author: Evan Hoeger & GPT-5
# Description: Lightweight facial recognition system designed for Raspberry Pi 3
# with optional onboard training and configurable distance threshold.
# ---------------------------------------------------

import os
import cv2
import numpy as np
import face_recognition
import argparse
import datetime

# ---------------------------------------------------
# Configuration and Argument Parsing
# ---------------------------------------------------
parser = argparse.ArgumentParser(description="Evans Face ID - Raspberry Pi Edition")
parser.add_argument("--train", action="store_true", help="Enable training mode (saves new faces)")
parser.add_argument("--threshold", type=float, default=0.45, help="Recognition distance threshold (lower = stricter)")
parser.add_argument("--save_log", action="store_true", help="Save recognition logs to file")
args = parser.parse_args()

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
DATA_DIR = os.path.join(os.getcwd(), "face_data")
LOG_FILE = os.path.join(os.getcwd(), "recognition_log.txt")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def load_known_faces():
    known_encodings = []
    known_names = []
    for person_name in os.listdir(DATA_DIR):
        person_folder = os.path.join(DATA_DIR, person_name)
        for file in os.listdir(person_folder):
            if file.endswith('.npy'):
                encoding = np.load(os.path.join(person_folder, file))
                known_encodings.append(encoding)
                known_names.append(person_name)
    return known_encodings, known_names


def log_recognition(name, distance):
    if args.save_log:
        with open(LOG_FILE, "a") as f:
            f.write(f"{datetime.datetime.now()} - Recognized {name} (distance={distance:.3f})\n")


def train_new_face(name, frame):
    person_dir = os.path.join(DATA_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_frame)
    if encodings:
        np.save(os.path.join(person_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"), encodings[0])
        print(f"[INFO] Saved new encoding for {name}.")
    else:
        print("[WARN] No face detected. Try again.")

# ---------------------------------------------------
# Initialize Camera
# ---------------------------------------------------
print("[INFO] Initializing camera...")
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Failed to open camera. Ensure the Pi Camera module is enabled (sudo raspi-config).")

# ---------------------------------------------------
# Load Known Faces
# ---------------------------------------------------
print("[INFO] Loading known faces...")
known_encodings, known_names = load_known_faces()

print(f"[INFO] Loaded {len(known_encodings)} face encodings.")

# ---------------------------------------------------
# Main Loop
# ---------------------------------------------------
print("[INFO] Starting recognition. Press 'q' to quit.")
while True:
    ret, frame = camera.read()
    if not ret:
        continue

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        distances = face_recognition.face_distance(known_encodings, encoding)
        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            if distances[best_match_index] < args.threshold:
                name = known_names[best_match_index]
                color = (0, 255, 0)
                log_recognition(name, distances[best_match_index])
            else:
                name = "Unknown"
                color = (0, 0, 255)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        top *= 4; right *= 4; bottom *= 4; left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow('EvansFaceID Pi', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif args.train and key == ord('t'):
        user_name = input("Enter name for new face: ")
        train_new_face(user_name, frame)

# ---------------------------------------------------
# Cleanup
# ---------------------------------------------------
camera.release()
cv2.destroyAllWindows()
print("[INFO] Program terminated.")