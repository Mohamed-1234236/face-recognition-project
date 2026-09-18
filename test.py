import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

# Load trained model and names
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
names = np.load("names.npy", allow_pickle=True).item()

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)

# Load a Windows font that natively supports Arabic characters
font_path = "C:\\Windows\\Fonts\\arial.ttf"  # Default on Windows
font_size = 24
try:
    font = ImageFont.truetype(font_path, font_size)
except IOError:
    font = ImageFont.load_default()  # Fallback


def draw_arabic_text(img, text, position, text_color):
    """Reshapes, reverses, and draws Arabic text onto an OpenCV frame using PIL."""
    # 1. Reshape the letters (hooks them together correctly)
    reshaped_text = arabic_reshaper.reshape(text)
    # 2. Fix the direction (right-to-left alignment)
    bidi_text = get_display(reshaped_text)

    # 3. Convert OpenCV BGR image to PIL Image
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 4. Draw the text (PIL uses RGB, so we swap colors if needed. White is (255,255,255))
    # converting color layout from BGR to RGB for PIL drawing
    rgb_color = (text_color[2], text_color[1], text_color[0])
    draw.text(position, bidi_text, font=font, fill=rgb_color)

    # 5. Convert back to OpenCV frame matrix
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


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

        # Predict face ID
        label_id, confidence = recognizer.predict(face)

        # Convert distance score to match percentage
        accuracy = max(0, min(100, round(100 - confidence)))

        # Condition: Known face AND at least 75% confidence match
        if accuracy >= 75:
            person_name = names.get(label_id, "Unknown")
            display_text = f"{person_name} {accuracy}%"
            color = (0, 255, 0)  # Green
        else:
            person_name = "Unknown"
            display_text = f"Unknown ({accuracy}%)" if accuracy > 0 else "Unknown"
            color = (0, 0, 255)  # Red

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Position text safely above the box
        text_y = y - 35 if y - 35 > 10 else y + h + 10

        # FIX: Draw text using our custom Arabic function instead of cv2.putText
        frame = draw_arabic_text(frame, display_text, (x, text_y), color)

    cv2.imshow("Face Recognition", frame)

    # Detect ESC key or window closing
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
        break

# Clean up resources
camera.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
