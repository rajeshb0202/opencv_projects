import cv2
import numpy as np

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]

    # Angles through which the image will be rotated
    angles = [0, 30, -30, 45, -45, -60, 60]
    found_face = False

    for angle in angles:
        if found_face:
            break

        # Rotate the image by the specified angle
        rotated_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        rotated_gray = cv2.warpAffine(gray, rotated_matrix, (width, height))

        # Detect faces in the rotated image
        faces = face_cascade.detectMultiScale(rotated_gray, 1.3, 5)
        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            found_face = True
            break

    # Display the frame with the detected face
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
