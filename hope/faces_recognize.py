import cv2
import pickle
import numpy as np
import os

facedetect = cv2.CascadeClassifier('/Users/alex/Desktop/hope/data/haarcascade_frontalface_default.xml')


images_directory = '/Users/alex/Desktop/Faces with names'

faces_data = []
names = []


if not os.path.exists('data/'):
    os.makedirs('data/')


for file_name in os.listdir(images_directory):

    if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue


    file_path = os.path.join(images_directory, file_name)
    print(f"Processing file: {file_path}") 


    frame = cv2.imread(file_path)

 
    if frame is None:
        print(f"Warning: Could not read image {file_name}. Skipping.")
        continue  

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)


    for (x, y, w, h) in faces:
      
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))
        
        faces_data.append(resized_img)
        names.append(os.path.splitext(file_name)[0]) 


faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(len(faces_data), -1)
names = np.asarray(names)


with open('data/faces_data.pkl', 'wb') as f:
    pickle.dump(faces_data, f)
with open('data/names.pkl', 'wb') as f:
    pickle.dump(names, f)

print("Processing complete. Data saved.")
