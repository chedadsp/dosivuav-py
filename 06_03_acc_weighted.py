import numpy as np
import cv2 as cv

cap = cv.VideoCapture('vid/vtest.avi')
#cap = cv.VideoCapture(0)

ret, frame = cap.read()

average = np.float32(frame)

while True:
    ret, frame = cap.read()
    if not ret: break

    cv.accumulateWeighted(frame, average, 0.01)
    
    background = cv.convertScaleAbs(average)

    cv.imshow('Input', frame)
    cv.imshow('Background', background)

    if (cv.waitKey(30) & 0xff) == 27: break

cap.release()
cv.destroyAllWindows()