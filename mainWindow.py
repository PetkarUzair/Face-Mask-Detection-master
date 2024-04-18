# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:50:55 2022

@author: Uzair
"""
import sys
import tkinter as tk
from PIL import ImageTk, Image
from datetime import datetime

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os   

from face_detect import *
from video_detection import *
 

# about page 
def about():
    new_window = tk.Toplevel(root)
    new_window.geometry('600x600')
    new_window.title("About Us")
    new_window.resizable(False, False)
    
    about_label= tk.Label(new_window,text="About Us", height=1, bg="#03A9F4",  fg="#ffffff")

    about_label.pack()

    #creating tuple 
    Font_tuples = ("Comic Sans MS", 20 ,"bold")

    about_label.configure(font= Font_tuples)

    
    
    frame= tk.Frame(new_window, width=500, height=400)
    frame.pack()
    frame.place(anchor='center',relx=0.5, rely=0.5)
    


    label_mem = tk.Label(frame, text="GROUP MEMBERS:\n1. Uzair Petkar\n2.  Fatima Dhanse \n3.  Ghanem Parker \n\nCLASS:\nTY Bsc ITR\n\nCOLLEGE:\nICS KHED\n\nNAME OF GUIDE:\nKHUSBHOO SODI", font="Times 15" )
    label_mem.pack(padx=10,pady=10)
      
    label_ver = tk.Label(frame, text="Version 1.0", font="Times 12" )
    label_ver.pack(padx=10,pady=10)

    frame.pack()
    frame.place(anchor='center',relx=0.5, rely=0.5)
    


# Face detection
def face_detect():

    face_window= tk.Toplevel(root)
    face_window.title ("Face Detection")

    face_window.wm_iconbitmap('resources/face_detect.ICO')

    # CREATE WINDOW - RESIZE FALSE - SIZE - TITLE CARD
    face_window.resizable(False, False)
    face_window.geometry('800x500')
    face_window.configure(bg='#c7d5e0')

    # DRAW TOP BLUE BAR - DRAW TITLE - DRAW DATETIME
    bg_head= tk.Canvas(face_window, width=800, height=60, bg='#1b2838', highlightthickness=0 ).place(x=0, y=0)
    tk.Label(bg_head, font='Montserrat 20', bg='#1b2838', fg='white').place(x = 15, y = 3)
    

    # Dashboard Button for back in dashboard window
    tk.Button (face_window, text="Back to Dashboard",command=face_window.destroy, height=3, width=20,  bg="green", fg='#fff').place(x=600, y=3)


    # Creating a photoimage object to use image
    img_web = tk.PhotoImage(file = "resources/webcam_face_detection.png")

    # here, image option is used to
    # set image on button
    tk.Button(face_window, text = 'WebCam Face Detect', command= face_detection ,  image = img_web).place(x=80, y=150)
    tk.Label(face_window, text="WEB-CAM FACE DETECTION", bg= "#FFFFFF", fg= "#f0f").place(x = 160, y = 150)

    # Creating a photoimage object to use image
    img_video = tk.PhotoImage(file = "resources/face_image.png")

    # here, image option is used to
    # set image on button
    tk.Button(face_window, text = 'Video Face Detect', command= video_face_detection ,  image = img_video).place(x=420, y=150)
    tk.Label(face_window, text="Image FACE DETECTION", bg= "#FFFFFF", fg= "#f0f").place(x = 460, y = 150)


    face_window.mainloop()


# detect_mask_video.py 
def faceMaskDetect():

    
    def detect_and_predict_mask(frame, faceNet, maskNet):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
            (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()
        print(detections.shape)

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)

    # load our serialized face detector model from disk
    prototxtPath = r"face_detector/deploy.prototxt"
    weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    maskNet = load_model("mask_detector.model")

    # initialize the video stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Face Mask Detector   |   q or Esc key to close", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # Stop loop if escape key is pressed
        k = cv2.waitKey(1) & 0xff
        if k==27:
            break
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


root= tk.Tk()
root.title ("Face Detection System")

root.wm_iconbitmap('resources/face_detect.ICO')


# CREATE WINDOW - RESIZE FALSE - SIZE - TITLE CARD
root.resizable(False, False)
root.geometry('1000x600')
root.configure(bg='#c7d5e0')


# DRAW TOP BLUE BAR - DRAW TITLE - DRAW DATETIME
top_bg = tk.Canvas(root, width=1000, height=60, bg='#1b2838', highlightthickness=0).place(x=0, y=0)
tk.Label(top_bg, text='Dashboard', font='Montserrat 25', bg='#1b2838', fg='white').place(x=15, y=3)
tk.Label(top_bg, text=datetime.now().strftime('%A, %d %B %Y'), font='Montserrat 20', bg='#1b2838', fg='white').place(
    x=600, y=8)

toolbar_box = tk.Canvas(root, width=960, height=450, bg='#2a475e', highlightthickness=0).place(x=20, y=100)
toolbar_box_top = tk.Canvas(root, width=960, height=20, bg='#1b2838', highlightthickness=0).place(x=20, y=80)
tk.Label(toolbar_box_top, text='Face Detection Toolbar', font='Montserrat 7 bold', bg='#1b2838',
         fg='#FFFFFF').place(x=25, y=80)

 
# Creating a photoimage object to use image
img_mask = tk.PhotoImage(file = "resources/bg.png")
  
# here, image option is used to
# set image on button
tk.Button(root, text = 'Face Mask Detect',  image = img_mask,command= faceMaskDetect).place(x=100, y=120)
tk.Label(root, text="FACE MASK DETECTION", bg= "#FFFFFF", fg= "#f0f").place(x = 230, y = 120)

# Creating a photoimage object to use image
photo = tk.PhotoImage(file = "resources/img_face.png")
  
# here, image option is used to
# set image on button
tk.Button(root, text = 'Face Detect', image = photo, command= face_detect).place(x=520, y=120)
tk.Label(root, text="FACE DETECTION", bg= "#FFFFFF", fg= "#f00").place(x = 670, y = 120)


btn_about = tk.Button (root, text="About US", command= about,height=3,width=20,  bg="#006C62", fg='#fff').place(x=230, y=450)

btn_exit = tk.Button (root, text="Exit",command=root.destroy, height=3, width=20,  bg="red", fg='#fff').place(x=640, y=450)

root.mainloop() 
