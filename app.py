import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tkinter as tk
from tkinter import messagebox
import tensorflow as tf
import numpy as np
import cv2


def start_recognition():
    try:
        model = tf.keras.models.load_model('keras_model.h5', compile=False)
        facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        
        cap = cv2.VideoCapture(1) 
        cap.set(3, 320)
        cap.set(4, 240)
        font = cv2.FONT_HERSHEY_COMPLEX

        def get_className(classNo):
            if classNo == 0: return "RJ"
            elif classNo == 1: return "DM"
            elif classNo == 2: return "Russel"
            elif classNo == 3: return "Jomarc"
            elif classNo == 4: return "AJ"
            elif classNo == 5: return "Quimay"
            else: return "Unknown"

        while True:
            success, imgOrignal = cap.read()
            if not success:
                break

            
            gray = cv2.cvtColor(imgOrignal, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))
            
            for x, y, w, h in faces:
                
                offset_x = int(w * 0.7)  
                offset_y = int(h * 0.7)  
                
                y1 = max(0, y - offset_y)
                y2 = min(imgOrignal.shape[0], y + h + offset_y)
                x1 = max(0, x - offset_x)
                x2 = min(imgOrignal.shape[1], x + w + offset_x)
                
                crop_img = imgOrignal[y1:y2, x1:x2]
                if crop_img.size == 0:
                    continue

                
                color_corrected = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(color_corrected, (224, 224))
                img_array = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
                img_normalized = (img_array / 127.5) - 1.0

                
                prediction = model.predict(img_normalized, verbose=0)
                classIndex = np.argmax(prediction) 
                probabilityValue = np.amax(prediction)

                
                if probabilityValue > 0.70:  
                    display_name = str(get_className(classIndex))
                    display_prob = str(round(probabilityValue*100, 2)) + "%"
                    box_color = (0, 255, 0) 
                else:
                    display_name = "Unknown"
                    display_prob = ""
                    box_color = (0, 0, 255) 

                
                cv2.rectangle(imgOrignal, (x, y), (x+w, y+h), box_color, 2)
                cv2.rectangle(imgOrignal, (x, y-40), (x+w, y), box_color, -2)
                cv2.putText(imgOrignal, display_name, (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
                
                if display_prob != "":
                    cv2.putText(imgOrignal, display_prob, (x, y + h + 30), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                
            cv2.imshow("Facial Recognition Active (Press 'q' to exit)", imgOrignal)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        messagebox.showerror("Error", f"Could not start camera or load model: {e}")


root = tk.Tk()
root.title("IT310 Final Project - Face Recognition")
root.geometry("400x250")
root.configure(bg="#2c3e50") 

title_label = tk.Label(root, text="Facial Recognition System", font=("Helvetica", 16, "bold"), fg="white", bg="#2c3e50")
title_label.pack(pady=30)

start_btn = tk.Button(root, text="Start Camera & Recognition", font=("Helvetica", 12), bg="#27ae60", fg="white", padx=20, pady=10, command=start_recognition)
start_btn.pack()

root.mainloop()