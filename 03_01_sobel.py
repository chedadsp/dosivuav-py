import numpy as np
import cv2 as cv

img = cv.imread("img/shapes.jpg")
cv.imshow("Original image", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sobelX = cv.Sobel(gray, ddepth=cv.CV_32F, dx=1, dy=0, ksize=5)
absX = cv.convertScaleAbs(sobelX)
cv.imshow("Sobel X", absX )

sobelY = cv.Sobel(gray, ddepth=cv.CV_32F, dx=0, dy=1, ksize=5)
absY = cv.convertScaleAbs(sobelY)
cv.imshow("Sobel Y", absY )

grad = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

cv.imshow("Gradient XY", grad )

cv.waitKey(0)
cv.destroyAllWindows()