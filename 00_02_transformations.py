from configparser import Interpolation
from fnmatch import translate

import cv2 as cv
import numpy as np

img = cv.imread("img/car_and_road.jpg")

cv.imshow("Original image", img)

# Transfpormation
def translate(img, x, y):
    transMat = np.float32([[1, 0, x],[0,1,y]])
    dimenstions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimenstions)

translated = translate(img, 100, 200)

cv.imshow('Translated', translated)

# Rotation

def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if(rotPoint is None):
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, 1)
for _ in range(360):
    rotated = rotate(rotated, 1)
cv.imshow('Rotated', rotated)


# Resizing

resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# Flipping

flip = cv.flip(img, 0)
cv.imshow('Flipped', flip)

cv.waitKey(0)
