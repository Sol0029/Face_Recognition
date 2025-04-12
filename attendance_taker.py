from ultralytics import YOLO
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime
from deepface import DeepFace


class FaceRecognizerYOLO:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')  # Default YOLOv8 nano model
        self.font = cv2.FONT_ITALIC
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        self.face_features_known_list = []
        self.face_name_known_list = []

        self.get_face_database()
        self.conn = sqlite3.connect("attendance.db")
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                name TEXT, time TEXT, date DATE, UNIQUE(name, date)
            )
        """)
        self.conn.commit()

    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            csv_rd = pd.read_csv("data/features_all.csv", header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = [csv_rd.iloc[i][j] if csv_rd.iloc[i][j] != '' else '0' for j in range(1, 129)]
                name = csv_rd.iloc[i][0].replace("person_", "")  # Strip 'person_' prefix
                self.face_name_known_list.append(name)
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in database: %d", len(self.face_features_known_list))
        else:
            logging.warning("features_all.csv not found!")

    def return_euclidean_distance(self, feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        return np.sqrt(np.sum(np.square(feature_1 - feature_2)))

    def attendance(self, name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        self.cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, current_date))
        if not self.cursor.fetchone():
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            self.cursor.execute("INSERT INTO attendance (name, time, date) VALUES (?, ?, ?)", (name, current_time, current_date))
            self.conn.commit()
            print(f"{name} marked as present for {current_date} at {current_time}")

    def update_fps(self):
        now = time.time()
        self.fps = 1.0 / (now - self.start_time)
        self.start_time = now

    def recognize_face(self, face_img):
        try:
            face_img = cv2.resize(face_img, (160, 160))  # Faster model input
            embedding = DeepFace.represent(
                face_img,
                model_name='SFace',  # <- MUCH faster than Facenet
                enforce_detection=False,
                detector_backend='skip'
            )[0]['embedding']
            return np.array(embedding)
        except Exception as e:
            print("Embedding failed:", e)
            return np.zeros(128)

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)[0]
            boxes = results.boxes
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            classes = boxes.cls.cpu().numpy().astype(int)

            person_boxes = [box for box, cls in zip(xyxy, classes) if cls == 0]
            logged_names = set()  # Prevent duplicate logging in the same frame

            best_overall_name = "unknown"
            best_overall_distance = float("inf")
            best_overall_box = None

            for box in person_boxes:
                x1, y1, x2, y2 = box
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue

                features = self.recognize_face(face_crop)
                print("Live embedding sample:", features[:5], "norm=", np.linalg.norm(features))
                distances = [self.return_euclidean_distance(features, f) for f in self.face_features_known_list]

                for i, d in enumerate(distances):
                    print(f"Compared with: {self.face_name_known_list[i]}, distance = {d:.2f}")

                if distances:
                    min_distance = min(distances)
                    best_match = self.face_name_known_list[distances.index(min_distance)]

                    if min_distance < 3.5 and best_match not in logged_names:
                        if min_distance < best_overall_distance:
                            best_overall_name = best_match
                            best_overall_distance = min_distance
                            best_overall_box = (x1, y1, x2, y2)

            # Log attendance and draw box only for best match in this frame
            if best_overall_name != "unknown":
                self.attendance(best_overall_name)
                logged_names.add(best_overall_name)
                x1, y1, x2, y2 = best_overall_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, f"{best_overall_name} ({best_overall_distance:.2f})", (x1, y1 - 10), self.font, 0.6, (0, 255, 255), 1)

            self.update_fps()
            cv2.putText(frame, f"FPS: {self.fps:.2f}", (20, 30), self.font, 0.6, (0, 255, 0), 2)
            cv2.imshow("YOLO Attendance", frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        self.conn.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app = FaceRecognizerYOLO()
    app.run()
