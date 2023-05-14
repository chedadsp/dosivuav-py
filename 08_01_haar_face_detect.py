import cv2 as cv
from cv2 import COLOR_BGR2GRAY
import numpy as np

img = cv.imread('img/tesla.jpg')

gray = cv.cvtColor(img, COLOR_BGR2GRAY)

haar_cascade = cv.CascadeClassifier('haar/haarcascade_frontalface_default.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), 255, 1)

cv.imshow('Face', img)

cv.waitKey(0)
