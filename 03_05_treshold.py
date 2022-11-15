import numpy as np
import cv2 as cv

img = cv.imread("img/road.jpg")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)   
laplace = cv.Laplacian(gray, ddepth=-1)
cv.imshow("Laplace", laplace )

ret, threshold1 = cv.threshold(laplace, 50, 255, cv.THRESH_BINARY)
cv.imshow("Threshold 50", threshold1 )
cv.waitKey(0)

ret, threshold2 = cv.threshold(laplace, 200, 255, cv.THRESH_BINARY)
cv.imshow("Threshold 200", threshold2 )
cv.waitKey(0)

cv.destroyAllWindows()