import numpy as np
import cv2 as cv

img = cv.imread("img/shapes.jpg")
cv.imshow("Original image", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

laplace = cv.Laplacian(gray, ddepth=-1, ksize=5)

cv.imshow("Laplace", laplace )

cv.waitKey(0)
cv.destroyAllWindows()