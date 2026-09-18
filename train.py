import cv2
import os
import numpy as np

# Verify opencv-contrib-python is installed
if not hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
    raise ImportError("Please install opencv-contrib-python: pip install opencv-contrib-python")

dataset_path = "dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

faces = []
ids = []
names = {}
current_id = 0

print("Loading dataset...")
for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_dir):
        continue

    names[current_id] = person_name

    for image_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, image_name)

        # FIX: Read image from Unicode/Arabic path using numpy instead of cv2.imread
        try:
            # Read image file data as a raw byte array
            img_array = np.fromfile(img_path, np.uint8)
            # Decode the raw bytes into a grayscale image matrix
            gray_img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        except Exception:
            continue

        if gray_img is None:
            continue

        faces.append(gray_img)
        ids.append(current_id)

    current_id += 1

# Check if we actually found any faces before training
if len(faces) == 0:
    raise ValueError("No images were loaded. Ensure your dataset folder has valid images.")

print("Training model...")
recognizer.train(faces, np.array(ids))

# Save trained model and name mappings
recognizer.write("trainer.yml")
np.save("names.npy", names)

print("Training complete! 'trainer.yml' saved.")
