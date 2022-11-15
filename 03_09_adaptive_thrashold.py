import cv2 as cv
import numpy as np

img = cv.imread('img/boat.jpg', 0) # Load as grayscale

cv.imshow('Original', img)

ret, thr = cv.threshold(img, 200, 255, cv.THRESH_BINARY)
cv.imshow('1. Threshold Binary', thr)

gauss = cv.GaussianBlur(img, (5, 5), 0)

thr = cv.adaptiveThreshold(gauss, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 5)
cv.imshow('2. Adaptive Mean Thresholding', thr)

ret, thr = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) # threshold is not used set to zero
cv.imshow('3. Threshold Otsu', thr)

ret, thr = cv.threshold(gauss, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) # threshold is not used set to zero
cv.imshow('4. Threshold Otsu after Blur', thr)

cv.waitKey(0)

cv.destroyAllWindows()