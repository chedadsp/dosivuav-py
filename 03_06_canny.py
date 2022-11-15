import cv2 as cv
import numpy as np

img = cv.imread('img/road.jpg')
#img = cv.imread('img/road_shadow.jpg')

cv.imshow('Road', img)

blank = np.zeros(img.shape, dtype='uint8')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)

ret, thresh = cv.threshold(blur, 125, 225, cv.THRESH_BINARY)
cv.imshow('Thrash', thresh)

canny = cv.Canny(thresh, 25, 175)
cv.imshow('Canny Edged', canny)

cv.waitKey(0)