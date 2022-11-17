import numpy as np
import cv2 as cv

cap = cv.VideoCapture('vid/slow_traffic_small.mp4')
#cap = cv.VideoCapture(0)

ret, frame = cap.read()

x, y, w, h = 300, 200, 100, 50
track_window = (x, y, w, h)

roi = frame[y:y+h, x:x+w]

hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

lower_color = np.array([0., 60., 32.])
upper_color = np.array([180., 255., 255.])
mask = cv.inRange(hsv_roi, lower_color, upper_color)

roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])

cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 1, 10)

while True:
    ret, frame = cap.read()
    if not ret: break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    
    ret, track_window = cv.meanShift(dst, track_window, term_crit)

    x, y, w, h = track_window
    cv.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
    
    cv.imshow('Meanshift', frame)

    if (cv.waitKey(30) & 0xff) == 27: break

cv.destroyAllWindows()
cap.release()
