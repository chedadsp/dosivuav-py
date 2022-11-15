import cv2 as cv
import numpy as np

img = cv.imread('img/gradient.jpg', 0) # Load as grayscale
#img = cv.imread('img/hgradient.jpg', 0) # Load as grayscale
#img = cv.imread('img/cgradient.jpg', 0) # Load as grayscale

cv.imshow('Original', img)

ret, thr = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
cv.imshow('1. Threshold Binary', thr)

ret, thr = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
cv.imshow('2. Threshold Binary Inverse', thr)

ret, thr = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
cv.imshow('3. Threshold Truncate', thr)

ret, thr = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
cv.imshow('4. Threshold To Zero', thr)

ret, thr = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
cv.imshow('5. Threshold To Zero Inverse', thr)

cv.waitKey(0)

cv.destroyAllWindows()