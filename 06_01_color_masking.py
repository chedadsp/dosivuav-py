import numpy as np
import cv2 as cv

cap = cv.VideoCapture(2)

lower_color = np.array([90, 90, 90])
upper_color = np.array([120, 255, 255])

while True:
    ret, frame = cap.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_color, upper_color)
    res = cv.bitwise_and(frame, frame, mask=mask)
    
    cv.imshow('Input', frame)
    cv.imshow('Mask', mask)
    cv.imshow('Filtered', res)

    if (cv.waitKey(30) & 0xff) == 27: break

cap.release()
cv.destroyAllWindows()