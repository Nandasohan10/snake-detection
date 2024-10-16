import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import cvzone
import cv2
import math
import threading
import time
import pygame

cap = None
processing = False

def detect_from_file():
    global cap
    filepath = filedialog.askopenfilename()
    if filepath:
        if cap is not None:
            cap.release()  # Release any previously opened video capture
        cap = cv2.VideoCapture(filepath)
        threading.Thread(target=run_detection).start()  # Start detection in a separate thread

def detect_from_webcam():
    global cap
    if cap is not None:
        cap.release()  # Release any previously opened video capture
    cap = cv2.VideoCapture(0)
    threading.Thread(target=run_detection).start()  # Start detection in a separate thread

def stop_processing():
    global processing
    processing = False  # Set processing flag to False to stop the detection loop

def play_fire_alarm():
    pygame.mixer.init()  # Initialize pygame mixer
    pygame.mixer.music.load(r'.\sounds\alaramiing_sound.mp3')
    pygame.mixer.music.play()
    time.sleep(1)  # Play for 1 second
    pygame.mixer.music.stop()

def run_detection():
    global cap, processing
    model = YOLO(r'.\snake_detetion\best.pt')
    classnames = ['Snake']
    processing = True
    
    while processing:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            result = model(frame, stream=True)
            fire_detected = False

            for info in result:
                boxes = info.boxes
                for box in boxes:
                    confidence = box.conf[0]
                    confidence = math.ceil(confidence * 100)
                    Class = int(box.cls[0])
                    if confidence > 50:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                           scale=1.5, thickness=2)
                        fire_detected = True

            cv2.imshow('frame', frame)
            if fire_detected:
                threading.Thread(target=play_fire_alarm).start()  # Play sound in a separate thread

            if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to stop processing
                break

    cv2.destroyAllWindows()  # Close all OpenCV windows
    if cap is not None:
        cap.release()  # Release video capture

root = tk.Tk()
root.title("Fire Detection")
root.geometry("400x200")

browse_button = tk.Button(root, text="Browse File", command=detect_from_file)
browse_button.pack()

webcam_button = tk.Button(root, text="Use Webcam", command=detect_from_webcam)
webcam_button.pack()

stop_button = tk.Button(root, text="Stop", command=stop_processing)
stop_button.pack()

pygame.mixer.init()  # Initialize pygame mixer

root.mainloop()  # Start the Tkinter event loop
