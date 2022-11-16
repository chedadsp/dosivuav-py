import numpy as np
import cv2 as cv

cap = cv.VideoCapture('vid/vtest.avi')
#cap = cv.VideoCapture(0)

fgbgMOG = cv.bgsegm.createBackgroundSubtractorMOG()
fgbgMOG2 = cv.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret: break
    cv.imshow("Original", frame)
    fgmask = fgbgMOG.apply(frame)
    cv.imshow('MOG',fgmask)
    fgmask = fgbgMOG2.apply(frame)
    cv.imshow('MOG2',fgmask)
    
    if (cv.waitKey(30) & 0xff) == 27: break

cap.release()
cv.destroyAllWindows()