import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import logging

# YOLO model
model = YOLO('yolov8n.pt')

# Input directory
path_images_from_camera = "data/data_faces_from_camera/"
output_csv_path = "data/features_all.csv"

# Extract 128D embedding using DeepFace (SFace)
def extract_embedding(face_img):
    try:
        face_img = cv2.resize(face_img, (160, 160))
        embedding = DeepFace.represent(
            face_img,
            model_name='SFace',
            enforce_detection=False,
            detector_backend='skip'  # Skip detection, YOLO already cropped it
        )[0]['embedding']
        return np.array(embedding)
    except Exception as e:
        print("Embedding error:", e)
        return np.zeros(128)

def return_features_mean_personX(path_face_personX):
    features_list_personX = []
    photos_list = os.listdir(path_face_personX)
    if photos_list:
        for photo_name in photos_list:
            img_path = os.path.join(path_face_personX, photo_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            results = model(img, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            if len(boxes) == 0:
                logging.warning("No face detected in %s", img_path)
                continue
            x1, y1, x2, y2 = boxes[0]
            face_crop = img[y1:y2, x1:x2]
            feature = extract_embedding(face_crop)
            features_list_personX.append(feature)

    if features_list_personX:
        return np.mean(features_list_personX, axis=0)
    else:
        return np.zeros(128)

def main():
    logging.basicConfig(level=logging.INFO)
    person_list = os.listdir(path_images_from_camera)
    person_list.sort()

    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for person in person_list:
            logging.info("Processing: %s", person)
            person_path = os.path.join(path_images_from_camera, person)
            features_mean = return_features_mean_personX(person_path)
            print(f"{person} embedding: {features_mean[:5]}... norm={np.linalg.norm(features_mean):.2f}")
            features_mean = np.insert(features_mean.astype(object), 0, person)
            writer.writerow(features_mean)
            logging.info("Saved features for %s", person)

    logging.info("All features saved to %s", output_csv_path)

if __name__ == '__main__':
    main()
