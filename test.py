import cv2
import os
import numpy as np

# Load trained model and names
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
names = np.load("names.npy", allow_pickle=True).item()

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)

print("Starting video recognition. Press ESC to quit.")

while True:
    success, frame = camera.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]

        # Predict face ID (LBPH distance: lower confidence score = higher match accuracy)
        label_id, confidence = recognizer.predict(face)

        # Convert distance score to an estimated match percentage (0% to 100%)
        accuracy = max(0, min(100, round(100 - confidence)))

        # Condition: Known face AND at least 60% confidence match
        if accuracy >= 75:
            person_name = names.get(label_id, "Unknown")
            display_text = f"{person_name} {accuracy}%"
            color = (0, 255, 0)  # Green (BGR format)
        else:
            person_name = "Unknown"
            display_text = f"Unknown ({accuracy}%)" if accuracy > 0 else "Unknown"
            color = (0, 0, 255)  # Red (BGR format)

        # Draw bounding box (Green for known >=75%, Red for unknown/low confidence)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Display name directly above bounding box
        text_y = y - 10 if y - 10 > 10 else y + 15

        cv2.putText(
            frame,
            display_text,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow("Face Recognition", frame)

    # Detect ESC key (ASCII 27) or clicking window close button ('X')
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
        break

# Clean up resources
camera.release()
cv2.destroyAllWindows()
cv2.waitKey(1)