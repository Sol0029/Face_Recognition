import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import font as tkFont
from PIL import Image, ImageTk
import shutil
import time
import logging
import random
from ultralytics import YOLO

def apply_simple_augmentation(image, mode=0):
    aug = image.copy()

    if mode in [0, 1]:  # Brightness/contrast adjustment
        alpha = np.random.uniform(0.9, 1.1)
        beta = np.random.randint(-15, 15)
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    elif mode in [2, 3]:  # Slight blur
        k = np.random.choice([3, 5])
        aug = cv2.GaussianBlur(aug, (k, k), 0)

    return aug

class FaceRegisterYOLO:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')  # Use YOLOv8 nano model

        self.face_count = 0
        self.ss_cnt = 0
        self.current_name = ""

        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.current_face_dir = ""
        self.face_folder_created_flag = False

        self.cap = cv2.VideoCapture(0)
        self.current_frame = None

        # UI
        self.win = tk.Tk()
        self.win.title("Face Register")
        self.win.geometry("1000x500")

        self.frame_left = tk.Frame(self.win)
        self.frame_left.pack(side=tk.LEFT)
        self.label_img = tk.Label(self.frame_left)
        self.label_img.pack()

        self.frame_right = tk.Frame(self.win)
        self.font_title = tkFont.Font(family='Helvetica', size=20, weight='bold')
        self.font_step_title = tkFont.Font(family='Helvetica', size=15, weight='bold')

        self.label_title = tk.Label(self.frame_right, text="Face register", font=self.font_title)
        self.label_title.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        self.label_fps = tk.Label(self.frame_right, text="FPS: ")
        self.label_fps.grid(row=1, column=0, sticky=tk.W)

        self.label_faces = tk.Label(self.frame_right, text="Faces in current frame: ")
        self.label_faces.grid(row=2, column=0, sticky=tk.W)

        # Step 1
        tk.Label(self.frame_right, text="Step 1: Clear face photos", font=self.font_step_title).grid(row=3, column=0, sticky=tk.W, pady=(20, 5))
        tk.Button(self.frame_right, text="Clear", command=self.clear_data).grid(row=4, column=0, sticky=tk.W)

        # Step 2
        tk.Label(self.frame_right, text="Step 2: Input name", font=self.font_step_title).grid(row=5, column=0, sticky=tk.W, pady=(20, 5))
        tk.Label(self.frame_right, text="Name:").grid(row=6, column=0, sticky=tk.W)
        self.entry_name = tk.Entry(self.frame_right)
        self.entry_name.grid(row=6, column=1)
        tk.Button(self.frame_right, text="Input", command=self.get_name).grid(row=6, column=2)

        self.log_label = tk.Label(self.frame_right, text="")
        self.log_label.grid(row=9, column=0, columnspan=3, sticky=tk.W, pady=10)

        self.frame_right.pack(side=tk.RIGHT)

        self.start_time = time.time()
        self.last_save_time = 0
        self.fps = 0

    def clear_data(self):
        if os.path.exists(self.path_photos_from_camera):
            shutil.rmtree(self.path_photos_from_camera)
        os.makedirs(self.path_photos_from_camera)
        self.log_label.config(text="All face images cleared.")

    def get_name(self):
        name = self.entry_name.get().strip()
        if name:
            self.current_name = name
            self.face_folder_created_flag = True
            self.ss_cnt = 0
            folder_name = f"person_{name}"
            self.current_face_dir = os.path.join(self.path_photos_from_camera, folder_name)
            os.makedirs(self.current_face_dir, exist_ok=True)
            self.log_label.config(text=f"Directory created: {self.current_face_dir}")

    def auto_save_face(self):
        current_time = time.time()

        # Save every 0.5 seconds
        if self.face_folder_created_flag and self.current_frame is not None and self.ss_cnt < 30:
            if current_time - self.last_save_time >= 0.5:
                small_frame = cv2.resize(self.current_frame, (640, 480))  # Resize input
                results = self.model(small_frame, verbose=False)[0]
                boxes = results.boxes
                scale_x = self.current_frame.shape[1] / 640
                scale_y = self.current_frame.shape[0] / 480
                xyxy = (boxes.xyxy.cpu().numpy() * [scale_x, scale_y, scale_x, scale_y]).astype(int)
                classes = boxes.cls.cpu().numpy().astype(int)
                person_boxes = [box for box, cls in zip(xyxy, classes) if cls == 0]

                if person_boxes:
                    largest_box = max(person_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
                    x1, y1, x2, y2 = largest_box
                    face_crop = self.current_frame[y1:y2, x1:x2]
                    self.ss_cnt += 1

                    # Save original first
                    dup_count = (self.ss_cnt - 1) * 5 + 1
                    img_path = os.path.join(self.current_face_dir, f"img_face_{dup_count}.jpg")
                    cv2.imwrite(img_path, face_crop)

                    # Save 4 augmented images
                    for i in range(4):
                        aug_img = apply_simple_augmentation(face_crop, mode=i)
                        dup_count = (self.ss_cnt - 1) * 5 + i + 2
                        aug_path = os.path.join(self.current_face_dir, f"img_face_{dup_count}.jpg")
                        cv2.imwrite(aug_path, aug_img)

                    self.last_save_time = current_time
                    self.log_label.config(
                        text=f"Captured set {self.ss_cnt}/30 → Saved 5 images (total: {self.ss_cnt * 5}/150)"
                    )
                    self.log_label.config(fg='green')
                    self.win.after(200, lambda: self.log_label.config(fg='black'))

    def update_fps(self):
        now = time.time()
        self.fps = 1.0 / (now - self.start_time)
        self.start_time = now
        self.label_fps.config(text=f"FPS: {self.fps:.2f}")

    def update_camera(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.current_frame = frame.copy()

        self.frame_counter = getattr(self, 'frame_counter', 0) + 1
        if self.frame_counter % 3 == 0:
            small_frame = cv2.resize(frame, (640, 480))
            results = self.model(small_frame, verbose=False)[0]
            boxes = results.boxes
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 480
            xyxy = (boxes.xyxy.cpu().numpy() * [scale_x, scale_y, scale_x, scale_y]).astype(int)
            classes = boxes.cls.cpu().numpy().astype(int)
            self.person_boxes_cache = [box for box, cls in zip(xyxy, classes) if cls == 0]

        for (x1, y1, x2, y2) in getattr(self, 'person_boxes_cache', []):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.label_faces.config(text=f"Faces in current frame: {len(getattr(self, 'person_boxes_cache', []))}")
        self.update_fps()

        if self.face_folder_created_flag and self.ss_cnt < 30:
            self.auto_save_face()

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        self.label_img.imgtk = img_pil
        self.label_img.configure(image=img_pil)

        self.win.after(100, self.update_camera)  # Run every 100ms (~10 FPS)

    def run(self):
        self.update_camera()
        self.win.mainloop()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app = FaceRegisterYOLO()
    app.run()
