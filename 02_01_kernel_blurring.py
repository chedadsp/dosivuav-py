import numpy as np
import cv2 as cv

image = cv.imread("img/car_and_road.jpg")

cv.imshow("Original image", image)
cv.waitKey(0)

kernel_3x3 = np.ones((3, 3), np.float32) / 9
blurred = cv.filter2D(image, -1, kernel_3x3)
cv.imshow("3x3 Blur", blurred)
cv.waitKey(0)

kernel_7x7 = np.ones((7, 7), np.float32) / 49
blurred = cv.filter2D(image, -1, kernel_7x7)
cv.imshow("7x7 Blur", blurred)
cv.waitKey(0)

cv.destroyAllWindows()