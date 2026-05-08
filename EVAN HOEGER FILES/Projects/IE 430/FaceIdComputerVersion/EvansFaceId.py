import face_recognition
import cv2
import os
import numpy as np
from collections import defaultdict
import time

class EvansFaceId:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.last_logged_names = {}
        self.last_log_time = {}
        self.last_compare_time = 0
        self.compare_interval = 0.5  # seconds between comparisons
        self.log_buffer_time = 2.0   # seconds between logs
        self.log_file = "face_log.txt"
        self.FaceTolerance = 0.5  # tolerance for face comparison

    def loadKnownEncodings(self):
        known_faces_dir = 'known_faces'
        known_encodings = []
        known_names = []

        for person_name in os.listdir(known_faces_dir):
            person_dir = os.path.join(known_faces_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            for filename in os.listdir(person_dir):
                if filename.endswith('.npy'):
                    encoding_path = os.path.join(person_dir, filename)
                    encoding = np.load(encoding_path)
                    known_encodings.append(encoding)
                    known_names.append(person_name)

        self.known_encodings = known_encodings
        self.known_names = known_names
        print(f"Loaded {len(known_encodings)} encodings for {len(set(known_names))} people.")

    def CompareFaces(self, unknown_encodings):
        if not unknown_encodings or not self.known_encodings:
            return ["Unknown"] * len(unknown_encodings)

        names = []
        for face_encoding in unknown_encodings:
            distances = face_recognition.face_distance(self.known_encodings, face_encoding)
            person_min_dist = {}

            # Get minimum distance per person
            for i, distance in enumerate(distances):
                name = self.known_names[i]
                if name not in person_min_dist or distance < person_min_dist[name]:
                    person_min_dist[name] = distance

            if person_min_dist:
                best_name, best_distance = min(person_min_dist.items(), key=lambda x: x[1])
                if best_distance <= self.FaceTolerance:
                    names.append(best_name)
                else:
                    names.append("Unknown")
            else:
                names.append("Unknown")

        return names

    def logDetection(self, names):
        """Logs all detected names if changed or time buffer passed."""
        current_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))

        for name in names:
            last_time = self.last_log_time.get(name, 0)
            last_name_logged = self.last_logged_names.get(name, None)

            if (name != last_name_logged or last_time == 0) and (current_time - last_time > self.log_buffer_time):
                with open(self.log_file, "a") as f:
                    f.write(f"{timestamp}  {name}\n")
                print(f"[LOGGED] {timestamp} - {name}")
                self.last_logged_names[name] = name
                self.last_log_time[name] = current_time

    def runDetection(self):
        self.loadKnownEncodings()

        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Press 'q' to quit.")
        last_names = []

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            now = time.time()

            if now - self.last_compare_time >= self.compare_interval:
                # Resize frame to 1/4 size for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Detect faces and compute encodings
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                # Compare faces using minimum-distance method
                last_names = self.CompareFaces(face_encodings)
                self.last_compare_time = now

                if last_names:
                    self.logDetection(last_names)

                # Scale face locations back to original frame size
                face_locations = [(top*4, right*4, bottom*4, left*4) for (top, right, bottom, left) in face_locations]

            # Draw boxes and labels
            for (top, right, bottom, left), name in zip(face_locations, last_names):
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imshow('Evans Face ID', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    face_id = EvansFaceId()
    face_id.compare_interval = 0.5  # seconds between comparisons
    face_id.log_buffer_time = 2.0
    face_id.FaceTolerance = 0.5
    face_id.runDetection()
