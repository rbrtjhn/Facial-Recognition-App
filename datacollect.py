import cv2
import os

# Change 0 to 1 if your external USB webcam doesn't turn on!
video = cv2.VideoCapture(1) 
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0
nameID = input("Enter your Name: ").lower()
path = 'images/' + nameID
    
while os.path.exists(path):
    print("Name already exists, try another name")
    nameID = input("Enter your name again: ").lower()
    path = 'images/' + nameID

os.makedirs(path)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 8)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        filename = path + '/' + str(count) + '.jpg'
        cv2.imwrite(filename, face)
        print("Creating image:", filename)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Frame", frame)

    if count >= 500:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()