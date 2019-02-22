#!/usr/bin/env python

import numpy as np
import cv2

#randcolor = np.random.randint(0,115, size=3)

cap = cv2.VideoCapture(0)

while(True):

    # captured video feed
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # displays normal black and white video feed
    cv2.imshow('Original Image',gray)
    
    fast = cv2.FastFeatureDetector_create(25, True)

    # calls FAST algorithm using OpenCV
    kp = fast.detect(frame, None)

    # draws the points that FAST finds on the image
    corners = cv2.drawKeypoints(gray, kp, None, color=(80, 0, 200))

    # displays image with keypoints mapped
    cv2.imshow('Image with Corners', corners)

    # staggers window positions so they don't cover each other
    cv2.moveWindow('Image with Corners', 620, 0)

    # kill if escape or exit button are pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    if cv2.getWindowProperty('Image with Corners', cv2.WND_PROP_VISIBLE) < 1:
        break  
    if cv2.getWindowProperty('Original Image', cv2.WND_PROP_VISIBLE) < 1:
        break 
cap.release()
cv2.destroyAllWindows()
