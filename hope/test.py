from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import pyttsx3


def speak(str1):
    engine = pyttsx3.init()
    engine.say(str1)
    engine.runAndWait()


with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)


facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(FACES, LABELS)


video = cv2.VideoCapture(0)


if not os.path.exists('Attendance'):
    os.makedirs('Attendance')


while True:
    ret, frame = video.read()
    if not ret:
        break 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist = os.path.isfile(f"Attendance/Attendance_{date}.csv")
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        attendance = [str(output[0]), str(timestamp)]

    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)
    if k == ord('o'):
        print("Detected 'o' key press - Attempting to log attendance.")
        file_path = f"Attendance/Attendance_{date}.csv"
        with open(file_path, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(['NAME', 'TIME'])
            writer.writerow(attendance)
            print(f"Attendance for {attendance[0]} logged at {attendance[1]}")
        speak("Attendance logged.")
    elif k == ord('q'):
        print("Quitting...")
        break

video.release()
cv2.destroyAllWindows()
