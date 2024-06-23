import cv2
import numpy as np
import os
from gtts import gTTS
import pyautogui
from pygame import mixer

# Initialize Pygame mixer for audio playback
mixer.init()

# YOLOv4 Model Files
model_cfg = "yolov4-tiny.cfg"
model_weights = "yolov4-tiny.weights"
classes_file = "classes.txt"

# Load YOLOv4 Model
net = cv2.dnn.readNet(model_weights, model_cfg)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Load class names
with open(classes_file, "r") as f:
    classes = f.read().strip().split("\n")

# Define colors for bounding boxes (one for each class)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Initialize camera
cam = cv2.VideoCapture(0)  # Change to appropriate camera index if necessary
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Global flag to track button state
button_detect = False

# Function to handle mouse events for detecting button clicks
def handle_mouse_event(event, x, y, flags, params):
    global button_detect
    if event == cv2.EVENT_LBUTTONDOWN:
        terminate_region = np.array([(20, 80), (200, 80), (200, 130), (20, 130)])
        if cv2.pointPolygonTest(terminate_region, (x, y), False) > 0:
            print("Terminating program.")
            pyautogui.press('q')
            exit()

# Set button_detect to True directly without any conditional checks
button_detect = True

# Set up the window and mouse callback
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('frame', handle_mouse_event)

# Initialize a dictionary to track announced objects and scores
announced_objects = {}

# Main loop
while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to retrieve frame from camera.")
        break

    # Object detection
    class_ids, scores, bboxes = model.detect(frame)
    for (class_id, score, bbox) in zip(class_ids, scores, bboxes):
        x, y, w, h = bbox
        class_name = classes[class_id]

        # Draw bounding box and label
        color = colors[class_id]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{class_name}: {score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Announce object if the button is on and score is above threshold
        if button_detect and score > 0.8:
            # Avoid repetitive announcements
            if class_name not in announced_objects or announced_objects[class_name] < score:
                # Create announcement
                announcement_text = f"I see a {class_name}."
                announcement_file = f"{class_name}.mp3"

                # Check if the announcement file already exists
                if not os.path.exists(announcement_file):
                    tts = gTTS(text=announcement_text, lang='en')
                    tts.save(announcement_file)

                # Play the announcement if it is not already being played
                if not mixer.music.get_busy():
                    mixer.music.load(announcement_file)
                    mixer.music.play()

                # Update announced objects dictionary
                announced_objects[class_name] = score

    # Create detect button
    cv2.rectangle(frame, (20,80), (200, 130), (0, 0, 0) , -1)
    cv2.putText(frame, "Terminate", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
