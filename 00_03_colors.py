import cv2 as cv
import numpy as np

img = cv.imread("img/car_and_road.jpg")
cv.imshow("Original image", img)


blank = np.zeros(img.shape[:2], dtype='uint8')

# Split color channels

b, g, r = cv.split(img)

blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

cv.imshow('Blue', blue)
cv.imshow('Red', red)
cv.imshow('Green', green)

merged = cv.merge([b, g, r])
cv.imshow('Merged', merged)

cv.waitKey(0)
