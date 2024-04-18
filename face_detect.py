# import

import cv2
import os

def face_detection():
    
    # Load the cascade
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


    cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascPath)

    # To capture video from webcam. 
    cap = cv2.VideoCapture(0)
    # To use a video file as input 
    # cap = cv2.VideoCapture('sample_video.mp4')
    while True:
        # Read the frame
        _, img = cap.read()
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, 
            flags=cv2.CASCADE_SCALE_IMAGE)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Display
        cv2.imshow('Face Detector   |    Press Esc or q key to close', img)
        
        # Stop if q key is pressed
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break   
        # Stop if escape key is pressed
        k = cv2.waitKey(1) & 0xff
        if k==27:
            break
    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()
