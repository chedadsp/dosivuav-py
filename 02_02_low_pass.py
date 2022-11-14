import numpy as np
import cv2 as cv

image = cv.imread("img/car_and_road.jpg")

cv.imshow("Original image", image)

averaged = cv.blur(image, (3, 3))
cv.imshow("Average blur", averaged)
cv.waitKey(0)

gaussian = cv.GaussianBlur(image, (7, 7), 0)
cv.imshow("Gussian blur", gaussian)
cv.waitKey(0)

median = cv.medianBlur(image, 7)
cv.imshow("Median blur", median)
cv.waitKey(0)

bilateral = cv.bilateralFilter(image, 9, 75, 75)
cv.imshow("Bilateral blur", bilateral)
cv.waitKey(0)

dst = cv.fastNlMeansDenoisingColored(image, 6, 6, 7, 21)
cv.imshow("Fast Means Denoising", dst)
cv.waitKey(0)

cv.destroyAllWindows()