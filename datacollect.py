import cv2
import os

video = cv2.VideoCapture(0) 
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0
nameID = input("Enter your Name: ").lower()

# Saved inside a 'Data' folder to make training easier
path = 'Data/' + nameID
    
while os.path.exists(path):
    print("Name already exists, try another name")
    nameID = input("Enter your name again: ").lower()
    path = 'Data/' + nameID

os.makedirs(path)

while True:
    ret, frame = video.read()
    if not ret or frame is None:
        continue
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 8)

    for (x, y, w, h) in faces:
        count += 1
        
        # Add a 50 pixel offset so we capture the hair and chin!
        offset = 50
        y1 = max(0, y - offset)
        y2 = min(frame.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(frame.shape[1], x + w + offset)
        
        # FIX: Cut the face out of the COLOR frame, not the gray one!
        color_face = frame[y1:y2, x1:x2]
        
        # FIX: Resize it so the AI gets perfectly uniform data
        final_face = cv2.resize(color_face, (224, 224))

        filename = path + '/' + str(count) + '.jpg'
        cv2.imwrite(filename, final_face)
        print("Creating image:", filename)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Data Collection", frame)

    if count >= 500:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()