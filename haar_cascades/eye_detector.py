
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier( 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]

    # Angles through which the image will be rotated
    angles = [0, 30, -30, 45, -45]
    found_face = False

    for angle in angles:
        if found_face:
            break

        # Rotate the image by the specified angle
        rotated_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        rotated_gray = cv2.warpAffine(gray, rotated_matrix, (width, height))

        faces = face_cascade.detectMultiScale(rotated_gray, 1.3, 5)
        for (x, y, w, h) in faces:
            found_face = True
            # Continue with eye detection and other processing on rotated_gray

            roi_gray = rotated_gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

            break

    cv2.imshow('Eye Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
