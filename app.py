import os
os.environ["TF_USE_LEGACY_KERAS"] = "1" 

import tkinter as tk
from tkinter import messagebox
import tensorflow as tf
import numpy as np
import cv2

# --- 1. The Recognition Function ---
def start_recognition():
    try:
        model = tf.keras.models.load_model('keras_model.h5', compile=False)
        facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
# Change 0 to 1 to force the external USB webcam!
        cap = cv2.VideoCapture(1) 
        cap.set(3, 640)
        cap.set(4, 480)
        font = cv2.FONT_HERSHEY_COMPLEX

        def get_className(classNo):
            if classNo == 0:
                return "Me"     # Your placeholder
            elif classNo == 1:
                return "Lights"  # Your placeholder

        while True:
            success, imgOrignal = cap.read()
            if not success:
                break

            # MASSIVE SPEED BOOST: Convert to grayscale just for the face detector
            gray = cv2.cvtColor(imgOrignal, cv2.COLOR_BGR2GRAY)
            
            # Pass 'gray' to the detector instead of the color image
            faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(150, 150))
            
            for x, y, w, h in faces:
                offset = 20
                y1 = max(0, y - offset)
                y2 = min(imgOrignal.shape[0], y + h + offset)
                x1 = max(0, x - offset)
                x2 = min(imgOrignal.shape[1], x + w + offset)
                
                crop_img = imgOrignal[y1:y2, x1:x2]
                
                if crop_img.size == 0:
                    continue

                img = cv2.resize(crop_img, (224, 224))
                img = img.reshape(1, 224, 224, 3)
                
                prediction = model.predict(img, verbose=0)
                classIndex = np.argmax(prediction) 
                probabilityValue = np.amax(prediction)
                
                cv2.rectangle(imgOrignal, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.rectangle(imgOrignal, (x, y-40), (x+w, y), (0, 255, 0), -2)
                cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(imgOrignal, str(round(probabilityValue*100, 2)) + "%", (x, y + h + 30), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow("Facial Recognition Active (Press 'q' to exit)", imgOrignal)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        messagebox.showerror("Error", f"Could not start camera or load model: {e}")

# --- 2. The User Interface ---
root = tk.Tk()
root.title("IT310 Final Project - Face Recognition")
root.geometry("400x250")
root.configure(bg="#2c3e50") # Dark sleek background

# App Title Label
title_label = tk.Label(root, text="Facial Recognition System", font=("Helvetica", 16, "bold"), fg="white", bg="#2c3e50")
title_label.pack(pady=30)

# The Single Labeled Button required by the rubric
start_btn = tk.Button(root, text="Start Camera & Recognition", font=("Helvetica", 12), bg="#27ae60", fg="white", padx=20, pady=10, command=start_recognition)
start_btn.pack()

root.mainloop()