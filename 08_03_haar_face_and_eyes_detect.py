import cv2 as cv
from cv2 import COLOR_BGR2GRAY
import numpy as np

img = cv.imread('img/tesla.jpg')

gray = cv.cvtColor(img, COLOR_BGR2GRAY)

haar_face_classifier = cv.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
heer_eyes_classifier = cv.CascadeClassifier('haar/haarcascade_eye.xml')

faces_rect = haar_face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), 255, 3)
    # we want to wind eyes only on the face - ROI
    roi_img = img[y:y+h, x:x+w]
    roi_gray = gray[y:y+h, x:x+w]
    eyes = heer_eyes_classifier.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_img, (ex, ey), (ex+ew, ey+eh), 150, 3)

cv.imshow('Face & eyes', img)
cv.waitKey(0)
