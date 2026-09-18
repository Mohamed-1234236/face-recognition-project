import cv2
import os
import urllib.request
import tkinter as tk
from tkinter import simpledialog

# 1. Get the person's name via a GUI popup box (so terminal isn't needed)
root = tk.Tk()
root.withdraw()  # Hide main Tkinter window
name = simpledialog.askstring("Input", "Enter person's name:")

if not name or not name.strip():
    exit()  # Exit if user cancels or inputs empty text

name = name.strip()

# 2. Guarantee the Haar Cascade XML exists locally
cascade_filename = "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_filename):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_filename)

face_detector = cv2.CascadeClassifier(cascade_filename)

# 3. Setup output folder
dataset_path = os.path.join("dataset", name)
os.makedirs(dataset_path, exist_ok=True)

# 4. Initialize Camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    exit()

count = 0
max_samples = 30

while True:
    success, frame = camera.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        count += 1

        # Save cropped grayscale face
        face = gray[y:y + h, x:x + w]
        filename = os.path.join(dataset_path, f"{count}.jpg")
        cv2.imwrite(filename, face)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw text label
        text_y = y - 10 if y - 10 > 10 else y + 10
        cv2.putText(
            frame,
            f"Images: {count}/{max_samples}",
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        if count >= max_samples:
            break

    cv2.imshow("Face Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or count >= max_samples or cv2.getWindowProperty("Face Data Collector", cv2.WND_PROP_VISIBLE) < 1:
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()
cv2.waitKey(1)