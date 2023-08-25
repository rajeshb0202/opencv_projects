
#detecting face using haar cascade classifier and use the detectMultiScale method to detect faces

import cv2
import numpy as np
import os

path = os.getcwd() + '/haarcascade_frontalface_default.xml'

#face cascade
face_cascade = cv2.CascadeClassifier(path)

#start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()

    #convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #draw rectangle around faces
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    
    #display image
    img = cv2.flip(img, 1)
    cv2.imshow('img', img)

    #press esc to exit
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

#release webcam
cap.release()
cv2.destroyAllWindows()

