
import os
import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('C:/Users/gg/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

people=[]
for i in os.listdir(r'C:/Users/gg/OneDrive/Desktop/face recognition project/Faces/train'):
    people.append(i)
# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:/Users/gg/OneDrive/Desktop/face recognition project/face_trained.yml')

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Unable to open camera")
    exit()
while True:
   
    ret, frame = cap.read()

   
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break


    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

   
    for (x,y,w,h) in faces:
        faces_roi = gray[y:y+h,x:x+w]

        label, confidence = face_recognizer.predict(faces_roi)
        # print(f'Label = {people[label]} with a confidence of {confidence}')

        cv.putText(frame, str(people[label]), (x,y+h+25), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)

    
    cv.imshow('frame', frame)

    
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
