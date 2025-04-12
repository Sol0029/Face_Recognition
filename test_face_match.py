import cv2
import numpy as np
from deepface import DeepFace

# Load saved face image
img1 = cv2.imread("data/data_faces_from_camera/person_Moreno/img_face_23.jpg")
img1 = cv2.resize(img1, (160, 160))
emb1 = DeepFace.represent(img1, model_name="SFace", enforce_detection=False, detector_backend="skip")[0]['embedding']

# Capture from webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
frame = cv2.resize(frame, (160, 160))
emb2 = DeepFace.represent(frame, model_name="SFace", enforce_detection=False, detector_backend="skip")[0]['embedding']

# Compare
distance = np.linalg.norm(np.array(emb1) - np.array(emb2))
print(f"Distance from stored image to live webcam: {distance}")
