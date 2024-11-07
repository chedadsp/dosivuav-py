import numpy as np
import cv2 as cv

img = cv.imread("img/pattern.png")
#img = cv.imread("img/my_photo01.jpg")
#img = cv.imread("img/car_and_road.jpg")

gray = np.float32(cv.cvtColor(img, cv.COLOR_BGR2GRAY))

corners = cv.cornerHarris(gray, blockSize=3, ksize=3, k=0.05)

kernel = np.ones((7,7), np.uint8)
corners = cv.dilate(corners, kernel, iterations= 2)

img[corners > 0.025 * corners.max()] = [255, 127, 127]

cv.imshow("Harris Corners", img)

cv.waitKey(0)
cv.destroyAllWindows()