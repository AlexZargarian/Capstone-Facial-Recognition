import cv2
import pickle
import numpy as np
import os
import time

# Open the video capture
video = cv2.VideoCapture(0)
# video = cv2.VideoCapture(1) for external camera

# Load the Haar cascade classifier for face detection
facedetect = cv2.CascadeClassifier('/Users/alex/Desktop/Face/haarcascade_frontalface_default.xml')

# Initialize variables
faces_data = []
i = 0
name = input("Enter your name: ")
capture_interval = 10  # Interval for capturing frames
frame_rate = 30  # Frame rate of the camera (assuming 30 fps)

# Start time measurement
start_time = time.time()

# Capture frames
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_images = frame[y:y+h, x:x+w, :]
        resize_images = cv2.resize(crop_images, (50, 50))
        if len(faces_data) < 50 and i % capture_interval == 0:
            faces_data.append(resize_images)
        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 225), 3)  # Thickness 3 for fun
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 50:
        break

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()

# Calculate the elapsed time
elapsed_time = time.time() - start_time
print("Time taken to capture frames:", elapsed_time, "seconds")

# Convert faces_data to numpy array
faces_data = np.array(faces_data)

# Save names and faces_data
if 'names.pkl' not in os.listdir('Face/'):
    names = [name] * 100
else:
    with open('Face/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names += [name] * 100

with open('Face/names.pkl', 'wb') as f:
    pickle.dump(names, f)

if 'faces_data.pkl' not in os.listdir('Face/'):
    with open('Face/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('Face/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open('Face/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
