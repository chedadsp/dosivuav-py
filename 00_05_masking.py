import cv2 as cv
import numpy as np

img = cv.imread('img/car_and_road.jpg')
cv.imshow('Original image', img)

blank = np.zeros(img.shape[:2], dtype='uint8')
mask = cv.circle(blank, (450, 320), 100, 255, -1)

masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Masked', masked)

cv.waitKey(0)
