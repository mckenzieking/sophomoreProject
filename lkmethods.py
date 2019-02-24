#!/usr/bin/env python

import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 3 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

good_new = []
good_old = []
img = []
p1 = []

def calc():
    global p1
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    global good_new 
    good_new = p1[st==1]
    global good_old 
    good_old = p0[st==1]

def draw(mask, frame):
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
    global img
    img = cv.add(frame,mask)


def update(frame_gray):
    # Now update the previous frame and previous points
    global old_gray
    old_gray = frame_gray.copy()
    global p0
    p0 = good_new.reshape(-1,1,2)

while(1):
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    calc()

    draw(mask, frame)

    update(frame_gray)

    cv.imshow('frame',img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cv.destroyAllWindows()
cap.release()
